"""Submitting a render, and continuing from one.

This is the seam. The HTTP routers call it, and the rules about who may spend
what live here rather than in the router.

Ordering is inherited from the gateway and is the point: resolve, price,
authorise, reserve, submit. Nothing reaches fal.ai before the budget has
approved it, and the reservation is written before the call so a crash cannot
lose track of money already committed.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import budget, db, fal_gateway
from app.config import Settings
from app.models import AssetKind, ImageRole, UserStatus
from app.routing import RoutingTable
from app.services import assets, depth
from app.services.intents import intent_meta_from_params

log = logging.getLogger(__name__)


class RenderRefused(Exception):
    """The render was not submitted and nothing was spent."""

    def __init__(self, message: str, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class Actor:
    """Who is rendering, and what they are allowed to spend.

    Carries `username` as well as `user_id` because the ledger predates
    accounts: a person's pre-accounts spend still counts against their monthly
    allowance, and matching on both is what makes that true.
    """

    username: str
    user_id: int | None = None
    per_user_cap_usd: float | None = None
    is_admin: bool = False

    @classmethod
    def from_user(cls, row: sqlite3.Row) -> Actor:
        if str(row["status"]) != UserStatus.ACTIVE:
            raise RenderRefused(
                "This account is not active yet. An administrator has to approve "
                "it before you can render.",
                reason="not_active",
            )
        return cls(
            username=str(row["email"]),
            user_id=int(row["id"]),
            per_user_cap_usd=(
                float(row["budget_monthly_usd"])
                if row["budget_monthly_usd"] is not None
                else None
            ),
            is_admin=bool(row["is_admin"]),
        )


@dataclass(frozen=True)
class Reference:
    """One attached image and the job it is doing in this render."""

    asset_id: int
    role: ImageRole = ImageRole.GEOMETRY


def reference_brief(
    conn: sqlite3.Connection, references: Sequence[Reference]
) -> str:
    """The sentences that tell the model what each attached image is for.

    Handing a model several pictures and no explanation is the single most
    common way multi-reference editing fails — it averages them, or copies the
    wrong one's composition. Numbering matches the order the images are actually
    sent, which `fal_gateway.order_references` fixes.

    Returned rather than silently prepended. The architect sees the exact text
    that will be submitted, in an editable box, and can change it.
    """
    if len(references) < 2:
        return ""
    ordered = fal_gateway.order_references(
        [(r.role, str(r.asset_id)) for r in references]
    )
    lines = [
        f"Image {i}: {ImageRole(role).instruction}."
        for i, (role, _) in enumerate(ordered, start=1)
    ]
    return "\n".join(lines)


def compose_prompt(
    conn: sqlite3.Connection, prompt: str, references: Sequence[Reference]
) -> str:
    """The architect's words plus the reference key, as one editable block."""
    brief = reference_brief(conn, references)
    if not brief:
        return prompt.strip()
    return f"{prompt.strip()}\n\n{brief}"


# The geometry hold above which a real depth map is needed. 50 is "Hold": the
# scales shipped before the hold existed, which are safe on edges alone.
EDGES_ONLY_CEILING = 50
EDGES_ONLY_NOTE = "Edges only: no depth model is installed, so Exact is capped at Hold."
HOLD_PERSPECTIVE_ONLY_NOTE = (
    "Geometry hold applies to perspective views only; this drawing keeps its tuned scales."
)

# Image modes whose conditioning reads a `depth` control image.
_DEPTH_IMAGE_MODES = {"easycontrols", "differential_diffusion"}


class DepthSourceMissing(Exception):
    """The image to estimate depth from is no longer on disk."""


@dataclass(frozen=True)
class DepthControl:
    """A depth map ingested as a control asset. `cached` is True when no inference ran."""

    asset_id: int
    seconds: float
    cached: bool


_digest_locks: dict[str, threading.Lock] = {}
_digest_locks_guard = threading.Lock()


def _digest_lock(digest: str) -> threading.Lock:
    """One lock per screenshot digest, so two submits of one image estimate it once."""
    with _digest_locks_guard:
        return _digest_locks.setdefault(digest, threading.Lock())


def depth_control_for_asset(
    conn: sqlite3.Connection,
    settings: Settings,
    row: sqlite3.Row,
    *,
    project_id: int | None,
    uploaded_by: int | None,
) -> DepthControl:
    """Estimate depth for one image asset and ingest it as a control asset.

    The one place depth is estimated from an uploaded image. Both the submit
    path (through `_estimated_depth_asset`, which must never fail a submit) and
    the preview route (`POST /api/assets/{id}/depth`, which reports failures)
    call it, so they share the cache and the ingest.

    Raises `DepthSourceMissing` when the image is not on disk; estimator errors
    propagate for the caller to decide what they mean.

    The PNG is cached under data/depth_cache/ by sha256(source bytes) + model
    name, so the same screenshot submitted again does no inference. It is
    written 8-bit (what the CAD writer writes, and what survives a PIL
    `convert("RGB")` on fal's side - see app/services/depth.py).
    """
    local = Path(str(row["local_path"])) if row["local_path"] else None
    if local is None or not local.is_file():
        raise DepthSourceMissing("That image is no longer on disk.")
    digest = assets.sha256_bytes(local.read_bytes())
    cached = settings.depth_cache_dir() / f"{digest}_{depth.MODEL_NAME}.png"
    seconds = 0.0
    # Under the digest lock a second submit of the same image waits here, then
    # finds the PNG the first one wrote and does no inference.
    with _digest_lock(digest):
        was_cached = cached.is_file()
        if not was_cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_name(f"{cached.stem}.{uuid.uuid4().hex}.tmp.png")
            try:
                result = depth.estimate_depth(local, tmp, settings, bits=8)
                os.replace(tmp, cached)
                seconds = result.seconds
                log.info("Estimated depth for asset %s in %.2fs", row["id"], result.seconds)
            finally:
                tmp.unlink(missing_ok=True)
        asset = assets.ingest(
            conn, settings,
            data=cached.read_bytes(), filename=cached.name,
            project_id=project_id, uploaded_by=uploaded_by, kind=AssetKind.CONTROL,
        )
    return DepthControl(asset_id=int(asset["id"]), seconds=seconds, cached=was_cached)


def _estimated_depth_asset(
    conn: sqlite3.Connection,
    settings: Settings,
    references: Sequence[Reference],
    *,
    project_id: int | None,
    uploaded_by: int | None,
) -> int | None:
    """Estimate depth for the geometry reference and ingest it; None if not possible.

    Never raises: a missing model, an unreadable image or an inference error is
    logged and the render goes ahead exactly as it did before this existed.
    """
    if not references or not depth.is_available(settings):
        return None
    # A paused office is refused at the gate anyway; do not spend CPU before that.
    if budget.kill_switch(conn, settings):
        return None
    try:
        ordered = fal_gateway.order_references([(r.role, str(r.asset_id)) for r in references])
        row = db.get_asset(conn, int(ordered[0][1]))
        if row is None:
            return None
        control = depth_control_for_asset(
            conn, settings, row, project_id=project_id, uploaded_by=uploaded_by,
        )
        return control.asset_id
    except DepthSourceMissing:
        return None
    except Exception:
        log.warning("Depth estimation failed; submitting without it", exc_info=True)
        return None


def submit_render(
    conn: sqlite3.Connection,
    settings: Settings,
    table: RoutingTable,
    *,
    actor: Actor,
    task: str,
    prompt: str,
    engine: str | None = None,
    references: Sequence[Reference] = (),
    width: int = 1280,
    height: int = 960,
    num_images: int = 1,
    project_id: int | None = None,
    parent_job_id: int | None = None,
    mask_asset_id: int | None = None,
    control_assets: dict[str, int] | None = None,
    geometry_hold: int | None = None,
    intent_meta: dict[str, Any] | None = None,
    retry_of_job_id: int | None = None,
    retry_salt: str | None = None,
) -> fal_gateway.SubmitResult:
    """Resolve every reference to a URL, then submit.

    `intent_meta` (intent, view, style, line, geometry_hold) is recorded in the
    job's params_json next to the request, and is never sent to fal.ai. The
    stored `geometry_hold` is the value after the edges-only clamp.

    Reference URLs are resolved *before* the budget is spent but *after* the
    route is known, so an engine that cannot carry this many images refuses
    before anything is uploaded.

    `control_assets` is the per-entry control image enabler: `{"depth": id}`
    sends that asset as the `depth` conditioning entry instead of fanning the
    geometry reference across every entry — a true depth pass from a CAD
    conversion, conditioned separately from the colour view. Absent, nothing
    about conditioning changes from before this parameter existed.

    Depth: when the route conditions on `depth` and the caller supplied no
    depth asset, a depth map is estimated from the geometry reference
    (app/services/depth.py) and sent in that slot. `depth_source` on the job and
    in the result says which it was: "cad" (supplied), "estimated", or "none"
    (the colour image still fills the slot, as before).

    `geometry_hold` (0-100) overrides the depth and canny scales through
    routing.yaml's geometry_hold snaps. Above 50 it needs a real depth source;
    with none it is capped at 50 and the result carries a note saying so.
    """
    if not prompt.strip():
        raise RenderRefused("A render needs a prompt.", reason="no_prompt")

    route = table.resolve(task, engine)
    spec = table.payload_spec(route.model_id)

    max_images = int(spec.get("max_images", 1))
    if len(references) > max_images:
        raise RenderRefused(
            f"{route.engine_label} takes {max_images} reference image"
            f"{'' if max_images == 1 else 's'}, and {len(references)} are attached. "
            "Remove some, or switch to an engine that reads several references.",
            reason="too_many_references",
        )

    resolved: list[tuple[ImageRole, str]] = []
    for ref in references:
        try:
            resolved.append((ref.role, assets.ensure_fal_url(conn, settings, ref.asset_id)))
        except assets.AssetError as exc:
            raise RenderRefused(str(exc), reason="asset") from exc

    mask_url: str | None = None
    if mask_asset_id is not None:
        try:
            mask_url = assets.ensure_fal_url(conn, settings, mask_asset_id)
        except assets.AssetError as exc:
            raise RenderRefused(str(exc), reason="asset") from exc

    control_assets = dict(control_assets or {})
    if "depth" in control_assets:
        depth_source = "cad"
    else:
        depth_source = "none"
        wants_depth = any(c.get("type") == "depth" for c in route.conditioning)
        if wants_depth and spec.get("image") in _DEPTH_IMAGE_MODES:
            estimated = _estimated_depth_asset(
                conn, settings, references,
                project_id=project_id, uploaded_by=actor.user_id,
            )
            if estimated is not None:
                control_assets["depth"] = estimated
                depth_source = "estimated"

    notes: list[str] = []
    conditioning_scales: dict[str, float] | None = None
    meta: dict[str, Any] = dict(intent_meta or {})
    if geometry_hold is not None and not table.takes_geometry_hold(task):
        notes.append(HOLD_PERSPECTIVE_ONLY_NOTE)
        geometry_hold = None
        meta.pop("geometry_hold", None)
    if geometry_hold is not None:
        hold = max(0, min(100, int(geometry_hold)))
        if hold > EDGES_ONLY_CEILING and depth_source == "none":
            hold = EDGES_ONLY_CEILING
            notes.append(EDGES_ONLY_NOTE)
        conditioning_scales = table.scales_for_hold(hold)
        meta["geometry_hold"] = hold

    control_images: dict[str, str] = {}
    for control_type, asset_id in control_assets.items():
        try:
            control_images[control_type] = assets.ensure_fal_url(conn, settings, asset_id)
        except assets.AssetError as exc:
            raise RenderRefused(str(exc), reason="asset") from exc

    try:
        result = fal_gateway.submit(
            conn, settings, table,
            username=actor.username,
            task=task,
            prompt=prompt.strip(),
            engine=engine,
            width=width,
            height=height,
            num_images=num_images,
            references=resolved,
            mask_url=mask_url,
            mask_asset_id=mask_asset_id,
            control_images=control_images or None,
            user_id=actor.user_id,
            project_id=project_id,
            parent_job_id=parent_job_id,
            per_user_cap_usd=actor.per_user_cap_usd,
            conditioning_scales=conditioning_scales,
            retry_of_job_id=retry_of_job_id,
            retry_salt=retry_salt,
            meta=meta or None,
        )
    except fal_gateway.SubmitRefused as exc:
        raise RenderRefused(str(exc), reason=exc.reason) from exc
    except db.ForeignKeyViolation as exc:
        # A project_id that does not exist reached the browser as a 500: the
        # gateway's reserve returned None (the duplicate-key path), and the
        # duplicate branch then dereferenced a job that was never inserted.
        # It is a bad request and says which record is missing.
        raise RenderRefused(str(exc), reason="missing_reference") from exc

    # Record what went in, in the order it went in — so the version tree can
    # show it and a later render can be reproduced. Skipped for a duplicate,
    # whose inputs are already recorded against the original job.
    #
    # One transaction: a job whose inputs are half-written describes a render
    # nobody submitted, and the position numbers are what the prompt text means
    # by "image 2". The reservation itself is outside it — that INSERT belongs
    # to the gateway, which has already committed by the time we get here.
    if not result.duplicate:
        ordered = fal_gateway.order_references([(r.role, str(r.asset_id)) for r in references])
        with db.transaction(conn):
            for position, (role, asset_id) in enumerate(ordered, start=1):
                db.add_job_input(
                    conn, job_id=result.job_id, asset_id=int(asset_id),
                    role=role, position=position,
                )
            if project_id is not None:
                db.touch_project(conn, project_id)
        db.set_job_depth_source(conn, result.job_id, depth_source)

    return dataclasses.replace(result, depth_source=depth_source, notes=tuple(notes))


# --- retry ------------------------------------------------------------------

_STATUS_WORDS = {
    "completed": "complete",
    "failed": "failed",
    "refused": "refused",
    "cancelled": "cancelled",
    "reserved": "still being sent",
    "queued": "still waiting for fal.ai",
    "in_progress": "still rendering",
}


def _status_words(status: str) -> str:
    return _STATUS_WORDS.get(status, status)


def _control_assets_for_retry(
    conn: sqlite3.Connection, job: sqlite3.Row, params: dict
) -> dict[str, int]:
    """Find the control images the failed render was sent with, by their URLs.

    A CAD depth pass and an SKP change map are passed to fal.ai as control
    images, which the job does not record as inputs. Their fal URLs do appear in
    the stored payload, so the asset rows are matched on those. Only what the
    original render used is recovered: a depth map that was estimated is simply
    estimated again on the retry, and a render with no control image gets none.

    The geometry references and the mask are excluded first, because their URLs
    are in the payload too and would otherwise be taken for control images.
    """
    change_url = params.get("change_map_image_url")
    # NULL is a job from before depth_source was recorded: its depth pass, if it
    # had one, is in the payload like the change map, so look for it there.
    source = job["depth_source"]
    wants_depth = source is None or str(source) == "cad"
    if not change_url and not wants_depth:
        return {}

    text = str(job["params_json"] or "")
    excluded = {int(r["id"]) for r in db.job_inputs(conn, int(job["id"]))}
    if job["mask_asset_id"] is not None:
        excluded.add(int(job["mask_asset_id"]))
    rows = conn.execute(
        "SELECT id, fal_url, kind FROM assets WHERE fal_url IS NOT NULL"
    ).fetchall()
    candidates = [
        r for r in rows
        if int(r["id"]) not in excluded and r["fal_url"] and str(r["fal_url"]) in text
    ]

    found: dict[str, int] = {}
    if change_url:
        for r in candidates:
            if str(r["fal_url"]) == str(change_url):
                found["change_map"] = int(r["id"])
                break
    if wants_depth:
        for r in candidates:
            if str(r["kind"]) == str(AssetKind.CONTROL) and str(r["fal_url"]) != str(change_url):
                found["depth"] = int(r["id"])
                break
    return found


def retry_job(
    conn: sqlite3.Connection,
    settings: Settings,
    table: RoutingTable,
    *,
    actor: Actor,
    job_id: int,
) -> fal_gateway.SubmitResult:
    """Send a failed render again, as a new job with the same inputs.

    The new job goes through `submit_render`, so it is priced, budget-checked
    and reserved exactly like a fresh render. It differs in two ways. It carries
    `retry_of_job_id`, and its idempotency key is salted with the attempt number:
    the arguments are identical to the failed job's, and without the salt the
    retry would collapse onto the job it is meant to replace.

    Refusals carry a `reason` the router maps to a status code:
    `no_job` (404), `not_owner` (403), `not_failed` and `already_retried` (409).

    The spend is charged to whoever presses Retry. An administrator retrying
    another person's render is therefore charged to the administrator.
    """
    job = db.get_job(conn, job_id)
    if job is None:
        raise RenderRefused(f"No render {job_id} to retry.", reason="no_job")
    if not actor.is_admin and job["user_id"] != actor.user_id:
        raise RenderRefused(
            "Only the person who started this render, or an administrator, can retry it.",
            reason="not_owner",
        )
    if job["status"] != "failed":
        raise RenderRefused(
            f"Render #{job_id} is {_status_words(str(job['status']))}. "
            "Only a failed render can be retried.",
            reason="not_failed",
        )

    attempts = db.retries_of(conn, job_id)
    live = [r for r in attempts if r["status"] != "failed"]
    if live:
        newest = live[-1]
        raise RenderRefused(
            f"Render #{job_id} was already sent again as #{newest['id']}, which is "
            f"{_status_words(str(newest['status']))}. Only a failed render can be retried.",
            reason="already_retried",
        )

    try:
        params = json.loads(job["params_json"] or "{}")
    except ValueError:
        params = {}
    prompt = str(params.get("prompt") or "").strip()
    if not prompt:
        raise RenderRefused(
            "This render has no prompt stored, so it cannot be sent again. Start a new render instead.",
            reason="no_prompt",
        )
    size = params.get("image_size")
    if isinstance(size, dict):
        width, height = int(size.get("width", 1280)), int(size.get("height", 960))
    else:
        width, height = 1280, 960
    num_images = int(params.get("num_images", 1) or 1)

    references = [
        Reference(asset_id=int(r["id"]), role=ImageRole(str(r["role"])))
        for r in db.job_inputs(conn, job_id)
    ]
    control_assets = _control_assets_for_retry(conn, job, params)
    hold = params.get("geometry_hold")

    return submit_render(
        conn, settings, table,
        actor=actor,
        task=str(job["task"]),
        prompt=prompt,
        engine=str(job["engine"]) if job["engine"] else None,
        references=references,
        width=width,
        height=height,
        num_images=num_images,
        project_id=int(job["project_id"]) if job["project_id"] is not None else None,
        mask_asset_id=int(job["mask_asset_id"]) if job["mask_asset_id"] is not None else None,
        control_assets=control_assets or None,
        geometry_hold=int(hold) if hold is not None else None,
        intent_meta=intent_meta_from_params(params),
        retry_of_job_id=job_id,
        retry_salt=f"retry:{job_id}:{len(attempts) + 1}",
    )


def continue_from(
    conn: sqlite3.Connection,
    settings: Settings,
    table: RoutingTable,
    *,
    actor: Actor,
    job_id: int,
    task: str,
    prompt: str,
    engine: str | None = None,
    extra_references: Sequence[Reference] = (),
    width: int = 1280,
    height: int = 960,
    num_images: int = 1,
    mask_asset_id: int | None = None,
    output_index: int = 0,
    geometry_hold: int | None = None,
    intent_meta: dict[str, Any] | None = None,
) -> fal_gateway.SubmitResult:
    """Render again, starting from a finished render.

    `output_index` chooses which of the parent's images becomes the geometry.

    The parent enters as GEOMETRY: in an edit, the existing image *is* the
    structure that must survive. Anything else the architect attaches — a
    material sample, an approved hero view to match — comes in alongside it in
    its own role.

    The child records `parent_job_id`, so the version tree keeps its shape no
    matter how many times someone branches off the same render.
    """
    parent = db.get_job(conn, job_id)
    if parent is None:
        raise RenderRefused(f"No render {job_id} to continue from.", reason="no_parent")

    try:
        source = assets.from_job_output(
            conn, settings, job_id,
            project_id=parent["project_id"],
            uploaded_by=actor.user_id,
            output_index=output_index,
        )
    except assets.AssetError as exc:
        raise RenderRefused(str(exc), reason="asset") from exc

    references = [Reference(asset_id=int(source["id"]), role=ImageRole.GEOMETRY)]
    references.extend(extra_references)

    return submit_render(
        conn, settings, table,
        actor=actor,
        task=task,
        prompt=prompt,
        engine=engine,
        references=references,
        width=width,
        height=height,
        num_images=num_images,
        project_id=parent["project_id"],
        parent_job_id=job_id,
        mask_asset_id=mask_asset_id,
        geometry_hold=geometry_hold,
        intent_meta=intent_meta,
    )


# One poll at a time, whoever asked for it. The background poller and the
# manual refresh run the same `poll_all`, and every open tab calls the manual
# one every six seconds while anything is in flight — so without this, two
# threads routinely polled the same job at once and both wrote the same
# `data/outputs/jobNNNNNN_0.png` with a plain open(...,"wb"). The result is a
# torn image: two writers interleaving into one file.
#
# The guard belongs here rather than in either caller, because there are two
# callers and they must share it.
_poll_lock = threading.Lock()
_last_poll: dict[int, str] = {}
_last_poll_at: float = 0.0

# Six tabs polling every six seconds is a poll per second against fal.ai for
# nothing. Inside this window the last result is returned unchanged; the
# background poller's own ten-second cadence is what actually advances jobs.
MANUAL_REFRESH_MIN_SECONDS = 3.0


def refresh(
    conn: sqlite3.Connection, settings: Settings, *, min_interval: float = 0.0
) -> dict[int, str]:
    """Advance every in-flight job. Safe to call on any page load.

    Never runs twice at once. A caller that arrives while a poll is in progress,
    or inside `min_interval` of the last one, gets the most recent result rather
    than a second concurrent poll — which is the honest answer, since the poll
    it would have started is already running.
    """
    global _last_poll_at

    now = time.monotonic()
    if min_interval > 0 and (now - _last_poll_at) < min_interval:
        return dict(_last_poll)

    if not _poll_lock.acquire(blocking=False):
        return dict(_last_poll)
    try:
        result = fal_gateway.poll_all(conn, settings)
        _last_poll.clear()
        _last_poll.update(result)
        _last_poll_at = time.monotonic()
        return result
    finally:
        _poll_lock.release()
