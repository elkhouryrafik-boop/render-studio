"""Uploading reference images and masks, and serving them back."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from api import schemas
from api.deps import ActorDep, ConnDep, SettingsDep, UserDep
from app import db, ratelimit
from app.models import AssetKind
from app.services import assets as asset_service
from app.services import depth
from app.services import render as render_service

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/assets", tags=["assets"])

# What this endpoint will hand back. A CAD_UPLOAD row's `local_path` points at
# the raw .dxf/.ifc/.skp the architect uploaded, not at an image — serving it
# here turned an image route into a download link for the office's CAD source
# files. Everything else in this set is genuinely image bytes this app wrote or
# validated.
SERVABLE_KINDS = {
    AssetKind.UPLOAD,
    AssetKind.RENDER_OUTPUT,
    AssetKind.MASK,
    AssetKind.CONTROL,
}


# The kinds a depth map can be estimated from: a view the architect uploaded,
# or a finished render they want to take further. Masks, CAD sources and
# control images are not views.
DEPTH_SOURCE_KINDS = {AssetKind.UPLOAD, AssetKind.RENDER_OUTPUT}

DEPTH_MODEL_MISSING = (
    "No depth model is installed. Run scripts/fetch_depth_model.py on the server."
)


async def _read_capped(file: UploadFile) -> bytes:
    """Read an upload, but never more than the cap plus one byte.

    A plain `await file.read()` pulls the whole body into memory *before* the
    size check in `ingest()` runs, so a 2 GB POST from anywhere on the LAN
    would balloon the process first and be refused second. Reading cap+1 keeps
    the refusal (ingest sees len > cap) while bounding what a request can cost.
    """
    return await file.read(asset_service.MAX_UPLOAD_BYTES + 1)


@router.post("", response_model=schemas.AssetOut)
async def upload(
    conn: ConnDep,
    settings: SettingsDep,
    actor: ActorDep,
    file: Annotated[UploadFile, File()],
    project_id: Annotated[int | None, Form()] = None,
) -> schemas.AssetOut:
    """Store an image and return the asset the render request will refer to.

    Content-addressed: attaching the same massing view again returns the asset
    that already exists, so it is neither stored twice nor uploaded to fal.ai
    twice. Format is validated from the bytes, not the filename.
    """
    data = await _read_capped(file)
    try:
        # Off the event loop. `ingest` is a SHA-256, a PIL probe and a disk
        # write of up to 40 MB, all synchronous — run inline in an async
        # handler, one large upload stalls every other request in the process,
        # the background poller included.
        row = await run_in_threadpool(
            asset_service.ingest,
            conn, settings,
            data=data,
            filename=file.filename or "upload.png",
            project_id=project_id,
            uploaded_by=actor.user_id,
            kind=AssetKind.UPLOAD,
        )
    except asset_service.AssetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.asset_out(row)


@router.post("/mask", response_model=schemas.AssetOut)
async def upload_mask(
    conn: ConnDep,
    settings: SettingsDep,
    actor: ActorDep,
    file: Annotated[UploadFile, File()],
    source_asset_id: Annotated[int, Form()],
    project_id: Annotated[int | None, Form()] = None,
) -> schemas.AssetOut:
    """Store a painted mask, refusing one that does not match its source.

    `flux-pro/v1/fill` requires the mask and the image to have identical
    dimensions. Checking here means the mismatch is reported while the architect
    is still looking at the mask they drew.
    """
    data = await _read_capped(file)
    try:
        row = await run_in_threadpool(
            asset_service.ingest_mask,
            conn, settings,
            data=data,
            source_asset_id=source_asset_id,
            project_id=project_id,
            uploaded_by=actor.user_id,
        )
    except asset_service.AssetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.asset_out(row)


@router.get("/{asset_id}/image")
def get_image(asset_id: int, conn: ConnDep, _user: UserDep) -> FileResponse:
    row = db.get_asset(conn, asset_id)
    if row is None or not row["local_path"] or str(row["kind"]) not in SERVABLE_KINDS:
        raise HTTPException(status_code=404, detail="No such image.")
    path = Path(str(row["local_path"]))
    if not path.exists():
        raise HTTPException(status_code=404, detail="That image is no longer on disk.")
    return FileResponse(path)


@router.get("", response_model=list[schemas.AssetOut])
def list_project_assets(
    project_id: int, conn: ConnDep, _user: UserDep
) -> list[schemas.AssetOut]:
    return [
        schemas.asset_out(r)
        for r in db.list_assets(conn, project_id)
        # Masks are working files, a CAD upload is not itself an image fal.ai
        # can read (its converted PNG shows up separately, as an ordinary
        # UPLOAD), and a control image is a conditioning input, not a
        # reference. None of them belong in a reference gallery.
        if str(r["kind"]) not in (AssetKind.MASK, AssetKind.CAD_UPLOAD, AssetKind.CONTROL)
    ]


@router.post("/{asset_id}/depth", response_model=schemas.DepthPreviewOut)
def depth_preview(
    asset_id: int,
    conn: ConnDep,
    settings: SettingsDep,
    actor: ActorDep,
) -> schemas.DepthPreviewOut:
    """Estimate depth for an uploaded view now, so the architect can see it before rendering.

    The same estimator, cache and ingest the submit path uses, so the map shown
    here is the one a render with Hold or Exact would send. The result is a
    control asset whose `url` serves the 8-bit depth PNG. `seconds` is 0 when the
    map was already cached.

    Owner or administrator only: the view belongs to whoever uploaded it.
    """
    row = db.get_asset(conn, asset_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such image.")
    owner = row["uploaded_by"]
    if not actor.is_admin and (owner is None or int(owner) != actor.user_id):
        raise HTTPException(
            status_code=403,
            detail="Only the person who uploaded this image, or an administrator, can estimate its depth.",
        )
    if str(row["kind"]) not in DEPTH_SOURCE_KINDS:
        raise HTTPException(
            status_code=422,
            detail="Depth can only be estimated from an uploaded view or a finished render.",
        )
    if not depth.is_available(settings):
        raise HTTPException(status_code=503, detail=DEPTH_MODEL_MISSING)

    key = str(actor.user_id) if actor.user_id is not None else actor.username
    ratelimit.limiter.check("depth_preview", key, ratelimit.DEPTH_PREVIEW)
    ratelimit.limiter.hit("depth_preview", key, ratelimit.DEPTH_PREVIEW)

    try:
        control = render_service.depth_control_for_asset(
            conn, settings, row,
            project_id=int(row["project_id"]) if row["project_id"] is not None else None,
            uploaded_by=actor.user_id,
        )
    except render_service.DepthSourceMissing as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Depth preview failed for asset %s", asset_id)
        raise HTTPException(
            status_code=500,
            detail="The depth estimate failed. Try again, or send the render without a depth map.",
        ) from exc

    control_row = db.get_asset(conn, control.asset_id)
    return schemas.DepthPreviewOut(
        asset=schemas.asset_out(control_row),
        seconds=round(control.seconds, 2),
    )
