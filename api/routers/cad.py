"""DXF/SKP uploads: layer manifest, preview, and the confirm step that turns
a CAD file into a normal render input.

fal.ai reads pixels only, so a CAD file is never itself a reference image —
it is stored, previewed, adjusted, and only turned into a PNG (a plain
`upload` asset, indistinguishable afterwards from a photo an architect
dragged in) once the architect confirms it. Nothing here spends a fal.ai
credit; that only happens later, when the confirmed PNG is attached to a
render like any other reference.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from api import schemas
from api.deps import ActorDep, ConnDep, SettingsDep
from app import db, ratelimit
from app.config import Settings
from app.models import AssetKind
from app.services import assets as asset_service
from app.services import cad_convert

router = APIRouter(prefix="/api/cad", tags=["cad"])


async def _read_capped(file: UploadFile) -> bytes:
    """Same reasoning as the image upload endpoint: bound what a POST can cost
    before `ingest_cad`'s own size check gets a chance to run."""
    return await file.read(asset_service.MAX_CAD_UPLOAD_BYTES + 1)


def _load_cad(conn: sqlite3.Connection, asset_id: int) -> tuple[sqlite3.Row, Path]:
    """The CAD asset row and its file on disk, or a 404 that says which failed.

    A plain image asset id here is a caller mistake, not a missing file — kept
    as a 404 anyway rather than a 400, since from the caller's side both mean
    "there is nothing to convert at that id".
    """
    row = db.get_asset(conn, asset_id)
    if row is None or str(row["kind"]) != AssetKind.CAD_UPLOAD:
        raise HTTPException(status_code=404, detail="No CAD upload with that id.")
    if not row["local_path"]:
        raise HTTPException(status_code=404, detail="That upload has no file on disk.")
    path = Path(str(row["local_path"]))
    if not path.exists():
        raise HTTPException(status_code=404, detail="That file is no longer on disk.")
    return row, path


def _limit(actor, scope: str, limit: ratelimit.Limit) -> None:
    """Keyed on the account, because these routes cost the office time.

    A CAD upload is 100 MB hashed and written; a preview or a confirm is
    seconds of local rasterisation on a request thread. Neither spends a fal.ai
    credit, so the budget guard never sees them.
    """
    key = str(actor.user_id) if actor.user_id is not None else actor.username
    ratelimit.limiter.check(scope, key, limit)
    ratelimit.limiter.hit(scope, key, limit)


def _converter(path: Path, settings: Settings) -> cad_convert.CadConverter:
    try:
        return cad_convert.converter_for(path.suffix, settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _run(fn, *args, **kwargs):
    """Every converter call funnelled through one place to translate its
    failure modes into the status codes a caller can act on: a bad or empty
    file is the architect's to fix (400), a format that is not built yet is
    the office's to wait for (501), and a background bake still running is
    not a failure at all (202) — the caller polls the same endpoint again."""
    try:
        return fn(*args, **kwargs)
    except cad_convert.CadConversionPending as exc:
        raise HTTPException(
            status_code=202,
            detail={
                "pending": True,
                "stage": exc.stage,
                "elapsed_seconds": round(exc.elapsed_seconds, 1),
                "estimated_seconds": round(exc.estimated_seconds, 1),
                "message": str(exc),
            },
        ) from exc
    except cad_convert.CadConversionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.post("", response_model=schemas.AssetOut)
async def upload(
    conn: ConnDep,
    settings: SettingsDep,
    actor: ActorDep,
    file: Annotated[UploadFile, File()],
    project_id: Annotated[int | None, Form()] = None,
) -> schemas.AssetOut:
    """Store a DXF/SKP file. Nothing is converted yet — that is `/confirm`."""
    _limit(actor, "cad_upload", ratelimit.CAD_UPLOAD)
    data = await _read_capped(file)
    try:
        # Off the event loop: `ingest_cad` hashes and writes up to 100 MB
        # synchronously, and an async handler doing that blocks every other
        # request and the background poller for the duration.
        row = await run_in_threadpool(
            asset_service.ingest_cad,
            conn, settings,
            data=data,
            filename=file.filename or "upload.dxf",
            project_id=project_id,
            uploaded_by=actor.user_id,
        )
    except asset_service.AssetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.asset_out(row)


@router.get("/{asset_id}/bake-plan", response_model=schemas.CadBakePlan)
def bake_plan(
    asset_id: int, conn: ConnDep, settings: SettingsDep, actor: ActorDep
) -> schemas.CadBakePlan:
    """What this file will cost in TIME before anyone commits to converting it.

    A SketchUp model is parsed and baked once, and on a ~27 MB test
    model that first look takes 16 minutes. Sixteen minutes with no number on
    screen reads as a hang, and the honest thing to say is not just "this is
    slow" but "this is slow ONCE" — the bake is cached by content hash, so
    every later conversion of the same file is seconds.

    DXF and IFC are converted inline in about a second and report no wait.
    """
    _limit(actor, "cad_render", ratelimit.CAD_RENDER)
    _row, path = _load_cad(conn, asset_id)

    if path.suffix.lower() != ".skp":
        return schemas.CadBakePlan(
            cached=True, estimated_seconds=0.0,
            message="This format converts in a moment.",
        )

    if cad_convert.is_bake_cached(path, settings):
        return schemas.CadBakePlan(
            cached=True, estimated_seconds=0.0,
            message=(
                "This file has been converted before, so it is already parsed "
                "and cached. Converting it again takes a few seconds."
            ),
        )

    seconds = cad_convert.estimated_bake_seconds(path)
    return schemas.CadBakePlan(
        cached=False,
        estimated_seconds=seconds,
        message=(
            f"First conversion of this file will take about "
            f"{round(seconds / 60)} minutes. That happens only once — the "
            "result is cached, so every conversion of the same file after "
            "this one takes seconds. You can leave the page; it keeps going."
        ),
    )


@router.get("/{asset_id}/layers", response_model=list[schemas.CadLayerOut])
def layers(
    asset_id: int, conn: ConnDep, settings: SettingsDep, actor: ActorDep
) -> list[schemas.CadLayerOut]:
    _limit(actor, "cad_render", ratelimit.CAD_RENDER)
    _row, path = _load_cad(conn, asset_id)
    converter = _converter(path, settings)
    result = _run(converter.layers, path)
    return [
        schemas.CadLayerOut(name=layer.name, color=layer.color, visible=layer.visible)
        for layer in result
    ]


@router.get("/{asset_id}/preview")
def preview(
    asset_id: int,
    conn: ConnDep,
    settings: SettingsDep,
    actor: ActorDep,
    layers: Annotated[str | None, Query(description="Comma-separated layer names")] = None,
    camera: Annotated[str, Query(description="Camera preset — 3D formats only")] = "iso",
) -> Response:
    """A preview — SVG for 2D formats, PNG for 3D ones — free to regenerate on
    every layer toggle or camera change since nothing here is billed (SKP is
    "free" only in the fal.ai sense — a few seconds of local render once its
    background bake has completed; see `/confirm`'s 202 contract for that
    wait). The full rasterization is `/confirm` only. Content type is sniffed
    from the bytes because which format comes back is the converter's
    business, not this route's."""
    _limit(actor, "cad_render", ratelimit.CAD_RENDER)
    _row, path = _load_cad(conn, asset_id)
    converter = _converter(path, settings)
    wanted = [name for name in layers.split(",") if name] if layers else None
    body = _run(converter.preview, path, layers=wanted, camera=camera)
    media_type = "image/png" if body.startswith(b"\x89PNG") else "image/svg+xml"
    return Response(content=body, media_type=media_type)


@router.post("/{asset_id}/confirm", response_model=schemas.CadConfirmResponse)
def confirm(
    asset_id: int,
    body: schemas.CadConfirmRequest,
    conn: ConnDep,
    settings: SettingsDep,
    actor: ActorDep,
) -> schemas.CadConfirmResponse:
    """Rasterize the final PNG and hand it into the render pipeline as a
    normal `upload` asset — the only step in this router that is not free,
    in the sense that it is the last one before a fal.ai credit can be spent
    on whatever gets attached next."""
    _limit(actor, "cad_render", ratelimit.CAD_RENDER)
    cad_row, path = _load_cad(conn, asset_id)
    converter = _converter(path, settings)
    result = _run(
        converter.convert, path,
        layers=body.layers, line_weight_scale=body.line_weight_scale,
        camera=body.camera,
    )
    project_id = body.project_id if body.project_id is not None else cad_row["project_id"]
    stem = Path(str(cad_row["local_path"])).stem
    try:
        row = asset_service.ingest(
            conn, settings,
            data=result.view_png,
            filename=f"{stem}.png",
            project_id=project_id,
            uploaded_by=actor.user_id,
            kind=AssetKind.UPLOAD,
        )
    except asset_service.AssetError as exc:
        # A project_id from the request body that no longer exists lands here
        # rather than escaping as an IntegrityError and a 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # The true depth pass, and (SKP only) the albedo/change pair, ride along
    # as their own CONTROL assets so a render can condition on them
    # separately (control_assets={"depth": id, "change_map": id}). Content-
    # addressed like everything else: re-confirming the same view re-uses them.
    def _control_asset(data: bytes | None, suffix: str) -> schemas.AssetOut | None:
        if data is None:
            return None
        try:
            control_row = asset_service.ingest(
                conn, settings,
                data=data,
                filename=f"{stem}_{suffix}.png",
                project_id=project_id,
                uploaded_by=actor.user_id,
                kind=AssetKind.CONTROL,
            )
        except asset_service.AssetError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return schemas.asset_out(control_row)

    depth_out = _control_asset(result.depth_png, "depth")
    albedo_out = _control_asset(result.albedo_png, "albedo")
    change_out = _control_asset(result.change_png, "change")

    return schemas.CadConfirmResponse(
        asset=schemas.asset_out(row),
        depth_asset=depth_out,
        albedo_asset=albedo_out,
        change_asset=change_out,
        camera=result.camera.preset if result.camera else None,
        warnings=result.warnings,
        materials=[
            schemas.CadMaterialOut(
                material_class=m.material_class,
                confidence=m.confidence,
                albedo=m.albedo,
                share_of_visible_pixels=m.share_of_visible_pixels,
                source=m.source,
                tile_width_mm=m.tile_width_mm,
            )
            for m in result.materials
        ],
    )
