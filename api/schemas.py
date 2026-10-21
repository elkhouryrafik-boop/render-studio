"""Request and response shapes.

Every response model is built by an explicit function below rather than by
dumping a database row. A `sqlite3.Row` serialised wholesale is how a
`password_hash` column ends up in a JSON payload the day someone adds a join.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app import db
from app.models import ImageRole

# --- Auth -------------------------------------------------------------------


class SignupRequest(BaseModel):
    access_code: str
    email: str
    name: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AccessCodeRequest(BaseModel):
    access_code: str


class ApprovalOut(BaseModel):
    """Deliberately one field. Nothing about the account but its standing."""

    status: Literal["pending", "active", "unknown"]


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class UserOut(BaseModel):
    id: int
    email: str
    name: str
    status: str
    is_admin: bool
    budget_monthly_usd: float | None = None
    created_at: str | None = None
    approved_at: str | None = None


def user_out(row: sqlite3.Row) -> UserOut:
    return UserOut(
        id=int(row["id"]),
        email=str(row["email"]),
        name=str(row["name"]),
        status=str(row["status"]),
        is_admin=bool(row["is_admin"]),
        budget_monthly_usd=(
            float(row["budget_monthly_usd"])
            if row["budget_monthly_usd"] is not None
            else None
        ),
        created_at=str(row["created_at"]) if row["created_at"] else None,
        approved_at=str(row["approved_at"]) if row["approved_at"] else None,
    )


class MeOut(UserOut):
    """The signed-in user, plus the engine this server runs (fake renders are placeholders)."""

    engine: Literal["fal", "fake"]


class KillSwitchRequest(BaseModel):
    on: bool


class ControlsOut(BaseModel):
    """The runtime controls in force now. Both come from the database override
    when an admin has set one, and from `.env` otherwise."""

    kill_switch: bool
    monthly_cap_usd: float


class SignupResponse(BaseModel):
    needs_approval: bool
    message: str


# --- Assets -----------------------------------------------------------------


class AssetOut(BaseModel):
    id: int
    kind: str
    width: int | None
    height: int | None
    created_at: str
    # Deliberately not the fal.ai URL. The browser fetches images back through
    # this app, so nothing has to reason about how long a CDN link stays valid
    # and no external URL ends up pasted into a client email.
    url: str


def asset_out(row: sqlite3.Row) -> AssetOut:
    return AssetOut(
        id=int(row["id"]),
        kind=str(row["kind"]),
        width=int(row["width"]) if row["width"] is not None else None,
        height=int(row["height"]) if row["height"] is not None else None,
        created_at=str(row["created_at"]),
        url=f"/api/assets/{int(row['id'])}/image",
    )


# --- CAD --------------------------------------------------------------------


class CadLayerOut(BaseModel):
    name: str
    color: str | None
    visible: bool


class CadConfirmRequest(BaseModel):
    layers: list[str] | None = None
    line_weight_scale: float = 1.0
    # Camera preset for 3D formats (IFC, later SKP). Ignored for DXF — a 2D
    # drawing has no camera to choose.
    camera: str = "iso"
    project_id: int | None = None


class CadMaterialOut(BaseModel):
    """One row of the materials panel — SKP only, empty for DXF/IFC."""

    material_class: str
    confidence: str
    albedo: tuple[float, float, float]
    share_of_visible_pixels: float
    source: str
    tile_width_mm: float | None = None


class CadConfirmResponse(BaseModel):
    asset: AssetOut
    # The true depth pass, when the converter produced one (IFC, SKP). Attach
    # it as control_assets={"depth": id} on the render for separate depth
    # conditioning; a DXF confirm has none.
    depth_asset: AssetOut | None = None
    # SKP only: the pre-textured, unlit material base and the per-pixel
    # confidence map differential diffusion reads as
    # control_assets={"change_map": id} on the skp_material_render task.
    albedo_asset: AssetOut | None = None
    change_asset: AssetOut | None = None
    camera: str | None = None
    warnings: list[str]
    materials: list[CadMaterialOut] = []


# --- Renders ----------------------------------------------------------------


class ReferenceIn(BaseModel):
    asset_id: int
    role: ImageRole = ImageRole.GEOMETRY


class RenderRequest(BaseModel):
    """A render, either by intent (the server resolves task and engine) or by task.

    With `intent`, `task`, `engine` and `prompt` are ignored and the server
    composes the prompt from the view, style and line. Without it, `task` and
    `prompt` are required, as they always were.
    """

    task: str | None = None
    prompt: str | None = None
    engine: str | None = None
    intent: str | None = None
    view: str | None = None
    style: str | None = None
    line: str | None = None
    references: list[ReferenceIn] = Field(default_factory=list)
    width: int = 1280
    height: int = 960
    # None means "the intent's default" (4) when an intent is given, and 1 otherwise.
    num_images: int | None = None
    project_id: int | None = None
    mask_asset_id: int | None = None
    # Per-entry conditioning overrides for flux-general: {"depth": asset_id}
    # sends that asset on the depth easycontrols entry while every other entry
    # still reads the geometry reference. Comes from a CAD confirm's
    # depth_asset. Harmless when absent — behaviour is exactly pre-existing.
    control_assets: dict[str, int] | None = None
    # 0 = loose, 50 = hold (the scales shipped before this existed), 100 = exact.
    # Maps to depth/canny conditioning scales via routing.yaml geometry_hold.
    geometry_hold: int | None = Field(default=None, ge=0, le=100)


class ContinueRequest(BaseModel):
    task: str = "variation"
    prompt: str | None = None
    engine: str | None = None
    # change, region, enlarge or make_real. The parent's output is the base.
    intent: str | None = None
    view: str | None = None
    style: str | None = None
    line: str | None = None
    geometry_hold: int | None = Field(default=None, ge=0, le=100)
    extra_references: list[ReferenceIn] = Field(default_factory=list)
    width: int = 1280
    height: int = 960
    num_images: int | None = None
    mask_asset_id: int | None = None
    # Which of the parent render's images to continue from. 0 is the first.
    output_index: int = Field(default=0, ge=0)


class JobOutputOut(BaseModel):
    """One image a render produced.

    `url` is null when the image was never saved; the index still stands, so the
    list keeps its shape and the UI can say which image is missing.
    """

    index: int
    url: str | None
    seed: int | None


class JobOut(BaseModel):
    id: int
    task: str
    model_id: str
    engine: str | None
    status: str
    est_usd: float
    username: str
    project_id: int | None
    parent_job_id: int | None
    root_job_id: int | None
    created_at: str
    error: str | None
    output_url: str | None
    output_seed: int | None
    # "cad" | "estimated" | "none"; null for jobs that predate the column.
    depth_source: str | None = None
    # Every image the render produced, in index order. Empty while in flight.
    outputs: list[JobOutputOut] = Field(default_factory=list)
    # The job this one re-sends, when it is a retry of a failed render.
    retry_of_job_id: int | None = None
    # What the architect asked for, read back from params_json. Null for a
    # render that was not made from an intent, or that predates the fields.
    intent: str | None = None
    view: str | None = None
    style: str | None = None
    line: str | None = None
    geometry_hold: int | None = None
    # How many images were asked for (1 when the job does not say) and the size,
    # so the page can show the right number of pending slots. Size is null when
    # the engine takes no image_size.
    num_images: int = 1
    width: int | None = None
    height: int | None = None


def _params(row: sqlite3.Row) -> dict[str, Any]:
    try:
        loaded = json.loads(row["params_json"] or "{}")
    except (ValueError, TypeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def job_output_out(job_id: int, row: sqlite3.Row) -> JobOutputOut:
    # No stat here: a page of renders has up to four outputs each. A row whose
    # download failed has no path and no url; a file that has since vanished
    # keeps its url and the image route answers 404 for it.
    idx = int(row["idx"])
    return JobOutputOut(
        index=idx,
        url=f"/api/renders/{job_id}/image?i={idx}" if row["path"] else None,
        seed=db.seed_from_db(row["seed"]),
    )


def job_out(row: sqlite3.Row, outputs: list[sqlite3.Row] | None = None) -> JobOut:
    """One job as the API shows it. `outputs` are its job_outputs rows, if the
    caller has them; the list routes fetch them for a whole page in one query."""
    job_id = int(row["id"])
    # What `GET /api/renders/{id}/image` can actually serve, which is a local
    # file and nothing else. `output_path OR output_url` was a broader claim
    # than that endpoint honours: a completed job whose download failed keeps
    # its fal.ai URL, so the gallery showed a thumbnail slot that 404s. A stat
    # per row is cheap next to being wrong about it.
    path = row["output_path"]
    has_image = bool(path) and Path(str(path)).exists()
    return JobOut(
        id=int(row["id"]),
        task=str(row["task"]),
        model_id=str(row["model_id"]),
        engine=str(row["engine"]) if row["engine"] else None,
        status=str(row["status"]),
        est_usd=float(row["est_usd"]),
        username=str(row["username"]),
        project_id=int(row["project_id"]) if row["project_id"] is not None else None,
        parent_job_id=(
            int(row["parent_job_id"]) if row["parent_job_id"] is not None else None
        ),
        root_job_id=int(row["root_job_id"]) if row["root_job_id"] is not None else None,
        created_at=str(row["created_at"]),
        error=str(row["error"]) if row["error"] else None,
        output_url=f"/api/renders/{int(row['id'])}/image" if has_image else None,
        output_seed=db.seed_from_db(row["output_seed"]),
        depth_source=(
            str(row["depth_source"])
            if "depth_source" in row.keys() and row["depth_source"]  # noqa: SIM118
            else None
        ),
        outputs=[job_output_out(job_id, r) for r in (outputs or [])],
        retry_of_job_id=(
            int(row["retry_of_job_id"])
            if "retry_of_job_id" in row.keys() and row["retry_of_job_id"] is not None  # noqa: SIM118
            else None
        ),
        **_intent_fields(_params(row)),
        **_request_fields(_params(row)),
    )


class ActivityJobOut(JobOut):
    """A job as the administrator's Activity view shows it: who made it, what it
    finally cost, and when it last changed."""

    user_id: int | None
    user_name: str | None
    actual_usd: float | None
    updated_at: str
    timed_out_at: str | None = None


def activity_job_out(
    row: sqlite3.Row, outputs: list[sqlite3.Row] | None = None
) -> ActivityJobOut:
    """job_out plus the account fields. `row` comes from db.activity_jobs, which
    joins the users table; a job from before accounts has no user_id or name."""
    keys = row.keys()
    return ActivityJobOut(
        **job_out(row, outputs).model_dump(),
        user_id=int(row["user_id"]) if row["user_id"] is not None else None,
        user_name=str(row["user_name"]) if "user_name" in keys and row["user_name"] else None,
        actual_usd=float(row["actual_usd"]) if row["actual_usd"] is not None else None,
        updated_at=str(row["updated_at"]),
        timed_out_at=(
            str(row["timed_out_at"])
            if "timed_out_at" in keys and row["timed_out_at"]
            else None
        ),
    )


class ActivityUserOut(BaseModel):
    """One person's totals. user_id is null for the old jobs that were recorded
    against a free-text username before accounts existed."""

    user_id: int | None
    email: str
    name: str
    status: str
    jobs_total: int
    jobs_month: int
    spent_month_usd: float
    last_active: str | None


class ActivityOut(BaseModel):
    jobs: list[ActivityJobOut]
    users: list[ActivityUserOut]
    # The id to send as before_id for the next page; null when this was the last.
    next_before_id: int | None
    month_total_usd: float


def _request_fields(params: dict[str, Any]) -> dict[str, Any]:
    try:
        count = max(1, int(params.get("num_images") or 1))
    except (TypeError, ValueError):
        count = 1
    size = params.get("image_size")
    width = height = None
    if isinstance(size, dict):
        try:
            width, height = int(size["width"]), int(size["height"])
        except (KeyError, TypeError, ValueError):
            width = height = None
    return {"num_images": count, "width": width, "height": height}


def _intent_fields(params: dict[str, Any]) -> dict[str, Any]:
    hold = params.get("geometry_hold")
    return {
        "intent": str(params["intent"]) if params.get("intent") else None,
        "view": str(params["view"]) if params.get("view") else None,
        "style": str(params["style"]) if params.get("style") else None,
        "line": str(params["line"]) if params.get("line") else None,
        "geometry_hold": int(hold) if hold is not None else None,
    }


class SubmitResponse(BaseModel):
    job_id: int
    est_usd: float
    duplicate: bool
    message: str
    depth_source: str = "none"
    notes: list[str] = []


# --- Tasks and engines ------------------------------------------------------


class EngineOut(BaseModel):
    key: str
    label: str
    note: str
    model_id: str
    max_images: int
    supports_num_images: bool
    needs_mask: bool


class CadBakePlan(BaseModel):
    """How long this CAD file will take to convert, and whether it is a one-off."""

    cached: bool
    estimated_seconds: float
    message: str


class TaskOut(BaseModel):
    key: str
    label: str
    engines: list[EngineOut]
    default_engine: str
    # Listed but not offered. Non-empty means the UI must show the task greyed
    # with this sentence, and the server will refuse it anyway.
    in_development: str = ""


class EstimateRequest(BaseModel):
    task: str | None = None
    engine: str | None = None
    intent: str | None = None
    view: str | None = None
    width: int = 1280
    height: int = 960
    # None means the intent's default when an intent is given, and 1 otherwise.
    num_images: int | None = None
    geometry_hold: int | None = Field(default=None, ge=0, le=100)


class EstimateResponse(BaseModel):
    est_usd: float
    over_per_job_cap: bool
    model_id: str


# --- Intents ----------------------------------------------------------------


class IntentViewOut(BaseModel):
    key: str
    label: str


class IntentOut(BaseModel):
    key: str
    label: str
    hint: str
    card: bool
    slider: bool
    needs_mask: bool
    needs_line: bool
    max_references: int
    requires: str | None
    views: list[IntentViewOut]
    default_view: str | None


class StyleOut(BaseModel):
    key: str
    label: str


class HoldSnapOut(BaseModel):
    key: str
    value: int
    label: str
    hint: str


class GeometryHoldOut(BaseModel):
    default: int
    snaps: list[HoldSnapOut]


class SizeOut(BaseModel):
    key: str
    label: str
    width: int
    height: int


class IntentsOut(BaseModel):
    """What the one-screen Studio offers. Read once, with the page."""

    intents: list[IntentOut]
    styles: list[StyleOut]
    default_style: str
    geometry_hold: GeometryHoldOut
    sizes: list[SizeOut]
    depth_available: bool
    num_images_default: int
    num_images_max: int


class DepthPreviewOut(BaseModel):
    """The depth map estimated from an uploaded view, as the control asset a render would use."""

    asset: AssetOut
    depth_source: Literal["estimated"] = "estimated"
    seconds: float


class BudgetOut(BaseModel):
    month_total_usd: float
    month_user_usd: float
    monthly_cap_usd: float
    per_user_cap_usd: float
    kill_switch: bool
    warning: str | None


# --- Admin ------------------------------------------------------------------


class BudgetUpdate(BaseModel):
    budget_monthly_usd: float | None = None


def error(detail: str, reason: str | None = None) -> dict[str, Any]:
    return {"detail": detail, "reason": reason}


# "degraded" means the app is answering but something it needs is not working —
# a read-only database, a missing FAL_KEY, a poller that has died. A health
# check whose only possible answer is "ok" reports that it is running, which is
# the one thing the caller could already tell.
HealthState = Literal["ok", "degraded"]


class HealthOut(BaseModel):
    status: HealthState
    schema_version: int
    tasks: int
    # Which engine renders: "fal" spends money, "fake" does not. Not a secret.
    engine: Literal["fal", "fake"] = "fal"
    # One line per check, keyed by name: "ok", or what is wrong. Deliberately
    # coarse — this route is unauthenticated, so it says that the key is missing
    # and never anything about what the key is.
    checks: dict[str, str] = {}
