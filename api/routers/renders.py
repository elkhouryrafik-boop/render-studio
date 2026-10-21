"""Tasks, estimates, submitting, and continuing from a render.

Nothing here decides anything. It converts JSON to service calls and service
refusals to status codes — the rules about what may be rendered, by whom, at
what cost, stay in app/services where both front-ends reach them.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse

from api import schemas
from api.deps import ActorDep, ConnDep, SettingsDep, TableDep, UserDep
from app import budget, db, ratelimit
from app.models import ImageRole
from app.routing import IntentError, RoutingError, RoutingTable
from app.services import assets as asset_service
from app.services import depth
from app.services import intents as intent_service
from app.services import render as render_service
from app.services.intents import IntentPlan

router = APIRouter(prefix="/api", tags=["renders"])

# Budget refusals are reworded for the person in front of the screen. The
# service keeps its explicit amounts, because that message is what reaches the
# logs and the ledger.
REFUSAL_COPY: dict[str, str] = {
    "kill_switch": "Rendering is paused for the whole office. Ask whoever "
                   "administers the account to lift it.",
    "per_job": "This render is larger than a single job is allowed to be. "
               "Reduce the size or the number of variants.",
    "per_user": "You have reached your allowance for this month. It resets on "
                "the first.",
    "monthly": "The office has reached its allowance for this month. It resets "
               "on the first.",
}


def _spend_key(actor) -> str:
    """Rate-limit bucket for a spending route: the account, not the address.

    Keyed on the account so opening a second tab, or moving to another machine,
    does not hand someone a second allowance.
    """
    return str(actor.user_id) if actor.user_id is not None else actor.username


def _job_outs(conn, rows) -> list[schemas.JobOut]:
    """Render a page of jobs with their outputs, in one outputs query."""
    outputs = db.job_outputs_for(conn, [int(r["id"]) for r in rows])
    return [schemas.job_out(r, outputs.get(int(r["id"]), [])) for r in rows]


def _missing(field: str) -> RequestValidationError:
    """The same 422 a missing required field gets, so the plain-sentence handler words it."""
    return RequestValidationError(
        [{"type": "missing", "loc": ["body", field], "msg": "Field required", "input": None}]
    )


def _plan(table: RoutingTable, **kwargs) -> IntentPlan:
    """An intent resolved to a plan, or a 422 with the sentence the architect needs."""
    try:
        return intent_service.plan_intent(table, **kwargs)
    except IntentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _refused(exc: render_service.RenderRefused) -> HTTPException:
    detail = REFUSAL_COPY.get(exc.reason or "", str(exc))
    code = (
        status.HTTP_402_PAYMENT_REQUIRED
        if exc.reason in {"per_job", "per_user", "monthly", "kill_switch"}
        else status.HTTP_400_BAD_REQUEST
    )
    return HTTPException(status_code=code, detail=detail)


@router.get("/intents", response_model=schemas.IntentsOut)
def list_intents(table: TableDep, settings: SettingsDep, _user: UserDep) -> schemas.IntentsOut:
    """The four intents, the styles, the geometry-hold snaps and the sizes.

    Everything the one-screen Studio shows comes from here, so the browser
    holds no list of its own that could drift from routing.yaml.
    """
    return schemas.IntentsOut(
        **table.intents_out(depth_available=depth.is_available(settings))
    )


@router.get("/tasks", response_model=list[schemas.TaskOut])
def list_tasks(table: TableDep, _user: UserDep) -> list[schemas.TaskOut]:
    """What the office can do, and what each engine gives up.

    The notes come straight from routing.yaml. The architect chooses the engine,
    so the sentence explaining the trade-off has to travel with it.
    """
    out: list[schemas.TaskOut] = []
    for key, label in table.available_tasks():
        engines = []
        for engine in table.engines(key):
            spec = table.payload_spec(engine.model_id)
            engines.append(
                schemas.EngineOut(
                    key=engine.key,
                    label=engine.label,
                    note=engine.note,
                    model_id=engine.model_id,
                    max_images=int(spec.get("max_images", 1)),
                    supports_num_images=bool(spec.get("num_images", True)),
                    needs_mask=spec.get("mask") == "required",
                )
            )
        out.append(
            schemas.TaskOut(
                key=key, label=label, engines=engines,
                default_engine=table.default_engine(key),
                in_development=table.in_development(key),
            )
        )
    return out


@router.post("/estimate", response_model=schemas.EstimateResponse)
def estimate(
    body: schemas.EstimateRequest,
    table: TableDep,
    settings: SettingsDep,
    _user: UserDep,
) -> schemas.EstimateResponse:
    try:
        if body.intent is not None:
            task, engine = table.resolve_intent(body.intent, body.view)
            count = intent_service.intent_count(table, body.intent, body.view, body.num_images)
        else:
            if not body.task:
                raise HTTPException(
                    status_code=422, detail="Say which task to estimate, or send an intent."
                )
            task, engine = body.task, body.engine
            count = body.num_images if body.num_images is not None else 1
        route = table.resolve(task, engine)
        spec = table.payload_spec(route.model_id)
        num_images = count if spec.get("num_images", True) else 1
        est = table.estimate_usd(route.model_id, body.width, body.height, num_images)
    except IntentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RoutingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.EstimateResponse(
        est_usd=est,
        over_per_job_cap=est > settings.budget_per_job_usd,
        model_id=route.model_id,
    )


@router.get("/budget", response_model=schemas.BudgetOut)
def budget_status(
    conn: ConnDep, settings: SettingsDep, actor: ActorDep
) -> schemas.BudgetOut:
    st = budget.status(
        conn, settings, actor.username,
        user_id=actor.user_id, per_user_cap_usd=actor.per_user_cap_usd,
    )
    return schemas.BudgetOut(
        month_total_usd=st.month_total_usd,
        month_user_usd=st.month_user_usd,
        monthly_cap_usd=st.monthly_cap_usd,
        per_user_cap_usd=st.per_user_cap_usd,
        kill_switch=st.kill_switch,
        warning=budget.warning_text(st),
    )


@router.post("/renders", response_model=schemas.SubmitResponse)
def submit(
    body: schemas.RenderRequest,
    conn: ConnDep,
    settings: SettingsDep,
    table: TableDep,
    actor: ActorDep,
) -> schemas.SubmitResponse:
    # The budget caps how much can be spent, not how fast. A script inside a
    # valid session can commit the whole month in a few seconds, and the caps
    # only notice once it is gone.
    key = _spend_key(actor)
    ratelimit.limiter.check("render_submit", key, ratelimit.RENDER_SUBMIT)
    ratelimit.limiter.hit("render_submit", key, ratelimit.RENDER_SUBMIT)

    references = [render_service.Reference(r.asset_id, r.role) for r in body.references]
    try:
        if body.intent is not None:
            plan = _plan(
                table,
                intent=body.intent, view=body.view, style=body.style, line=body.line,
                num_images=body.num_images, geometry_hold=body.geometry_hold,
                roles=[r.role for r in references],
                has_mask=body.mask_asset_id is not None,
            )
            task, engine, prompt = plan.task, plan.engine, plan.prompt
            count, hold, intent_meta = plan.num_images, plan.geometry_hold, plan.meta
        else:
            if not body.task:
                raise HTTPException(
                    status_code=422, detail="Say which task to run, or send an intent."
                )
            if body.prompt is None:
                raise _missing("prompt")
            task, engine, prompt = body.task, body.engine, body.prompt
            count = body.num_images if body.num_images is not None else 1
            hold, intent_meta = body.geometry_hold, {}

        result = render_service.submit_render(
            conn, settings, table,
            actor=actor,
            task=task,
            prompt=prompt,
            engine=engine,
            references=references,
            width=body.width,
            height=body.height,
            num_images=count,
            project_id=body.project_id,
            mask_asset_id=body.mask_asset_id,
            control_assets=body.control_assets,
            geometry_hold=hold,
            intent_meta=intent_meta,
        )
    except render_service.RenderRefused as exc:
        raise _refused(exc) from exc
    except IntentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RoutingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return schemas.SubmitResponse(
        job_id=result.job_id,
        est_usd=result.est_usd,
        duplicate=result.duplicate,
        message=(
            f"An identical render is already running as #{result.job_id}. "
            "Reusing it rather than rendering twice."
            if result.duplicate
            else f"Render #{result.job_id} queued."
        ),
        depth_source=result.depth_source,
        notes=list(result.notes),
    )


@router.post("/renders/{job_id}/continue", response_model=schemas.SubmitResponse)
def continue_render(
    job_id: int,
    body: schemas.ContinueRequest,
    conn: ConnDep,
    settings: SettingsDep,
    table: TableDep,
    actor: ActorDep,
) -> schemas.SubmitResponse:
    """Branch a new render off a finished one.

    Costs no upload — the parent's output URL is already hosted by fal.ai, so
    this is the cheap path by construction rather than by optimisation.
    """
    key = _spend_key(actor)
    ratelimit.limiter.check("render_submit", key, ratelimit.RENDER_SUBMIT)
    ratelimit.limiter.hit("render_submit", key, ratelimit.RENDER_SUBMIT)

    extra = [render_service.Reference(r.asset_id, r.role) for r in body.extra_references]
    try:
        if body.intent is not None:
            # The parent's output is the base, so it is always the geometry picture.
            plan = _plan(
                table,
                intent=body.intent, view=body.view, style=body.style, line=body.line,
                num_images=body.num_images, geometry_hold=body.geometry_hold,
                roles=[ImageRole.GEOMETRY, *(r.role for r in extra)],
                has_mask=body.mask_asset_id is not None,
            )
            task, engine, prompt = plan.task, plan.engine, plan.prompt
            count, hold, intent_meta = plan.num_images, plan.geometry_hold, plan.meta
        else:
            if body.prompt is None:
                raise _missing("prompt")
            task, engine, prompt = body.task, body.engine, body.prompt
            count = body.num_images if body.num_images is not None else 1
            hold, intent_meta = body.geometry_hold, {}

        result = render_service.continue_from(
            conn, settings, table,
            actor=actor,
            job_id=job_id,
            task=task,
            prompt=prompt,
            engine=engine,
            extra_references=extra,
            width=body.width,
            height=body.height,
            num_images=count,
            mask_asset_id=body.mask_asset_id,
            output_index=body.output_index,
            geometry_hold=hold,
            intent_meta=intent_meta,
        )
    except render_service.RenderRefused as exc:
        raise _refused(exc) from exc
    except IntentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RoutingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return schemas.SubmitResponse(
        job_id=result.job_id,
        est_usd=result.est_usd,
        duplicate=result.duplicate,
        message=f"Render #{result.job_id} queued, continuing from #{job_id}.",
    )


def _retry_refused(exc: render_service.RenderRefused) -> HTTPException:
    """Retry refusals name their own status; anything else is a submit refusal."""
    if exc.reason == "no_job":
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if exc.reason == "not_owner":
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    if exc.reason in {"not_failed", "already_retried"}:
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return _refused(exc)


@router.post("/renders/{job_id}/retry", response_model=schemas.SubmitResponse)
def retry_render(
    job_id: int,
    conn: ConnDep,
    settings: SettingsDep,
    table: TableDep,
    actor: ActorDep,
) -> schemas.SubmitResponse:
    """Send a failed render again. Same task, prompt, size, count and inputs.

    Only the person who started the render, or an administrator, may retry it,
    and only while it is failed. The new render is priced and budget-checked as
    a fresh one would be.
    """
    key = _spend_key(actor)
    ratelimit.limiter.check("render_submit", key, ratelimit.RENDER_SUBMIT)
    ratelimit.limiter.hit("render_submit", key, ratelimit.RENDER_SUBMIT)

    try:
        result = render_service.retry_job(
            conn, settings, table, actor=actor, job_id=job_id,
        )
    except render_service.RenderRefused as exc:
        raise _retry_refused(exc) from exc
    except RoutingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return schemas.SubmitResponse(
        job_id=result.job_id,
        est_usd=result.est_usd,
        duplicate=result.duplicate,
        message=(
            f"This retry matches a render already queued as #{result.job_id}, "
            "so nothing extra was charged."
            if result.duplicate
            else f"Render #{result.job_id} queued, retrying #{job_id}."
        ),
        depth_source=result.depth_source,
        notes=list(result.notes),
    )


@router.get("/renders", response_model=list[schemas.JobOut])
def list_renders(
    conn: ConnDep,
    _user: UserDep,
    project_id: int | None = None,
    # ge=1 is not decoration: without it `?limit=-1` becomes `LIMIT -1`, which
    # SQLite reads as "no limit" and hands back every render the office has ever
    # made in one response.
    limit: int = Query(default=50, ge=1, le=200),
) -> list[schemas.JobOut]:
    """Everything the office has rendered, newest first.

    Not scoped to the signed-in user. An office of architects working on shared
    projects needs to see each other's work; that is the point of it being a
    shared tool rather than five copies of a script.
    """
    rows = db.recent_jobs(conn, limit, project_id=project_id)
    return _job_outs(conn, rows)


@router.get("/renders/{job_id}", response_model=schemas.JobOut)
def get_render(job_id: int, conn: ConnDep, _user: UserDep) -> schemas.JobOut:
    row = db.get_job(conn, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such render.")
    return schemas.job_out(row, db.list_job_outputs(conn, job_id))


@router.get("/renders/{job_id}/lineage", response_model=list[schemas.JobOut])
def get_lineage(job_id: int, conn: ConnDep, _user: UserDep) -> list[schemas.JobOut]:
    """Every render descended from the same original.

    Flat, oldest first. `parent_job_id` on each row is what the front-end
    assembles into the version tree.
    """
    rows = db.lineage(conn, job_id)
    if not rows:
        raise HTTPException(status_code=404, detail="No such render.")
    return _job_outs(conn, rows)


@router.get("/renders/{job_id}/image")
def get_render_image(
    job_id: int,
    conn: ConnDep,
    _user: UserDep,
    i: int = Query(default=0, ge=0, description="Which image of the render; 0 is the first."),
) -> FileResponse:
    """Serve one image of a render from local storage.

    Deliberately not a redirect to fal.ai's CDN: retention there is documented
    ambiguously, and an image the office might show a client should not depend
    on a third party's cache policy. It also keeps the URL behind the session
    cookie, so a link pasted into an email does not work for a stranger.
    """
    if db.get_job(conn, job_id) is None:
        raise HTTPException(status_code=404, detail="No such render.")
    output = db.get_job_output(conn, job_id, i)
    if output is None or not output["path"]:
        raise HTTPException(status_code=404, detail=f"No image {i} for that render.")
    path = Path(str(output["path"]))
    if not path.exists():
        raise HTTPException(status_code=404, detail="That image is no longer on disk.")
    return FileResponse(path)


@router.post("/renders/refresh", response_model=dict[int, str])
def refresh(conn: ConnDep, settings: SettingsDep, actor: ActorDep) -> dict[int, str]:
    """Advance every in-flight job now, rather than waiting for the poller.

    Every open tab calls this on a six-second interval, and it runs the same
    `poll_all` as the background poller. Two polls of the same job at once both
    write the same output file, so the service serialises them and returns the
    most recent result to whoever arrives during one — see
    `render_service.refresh`. The limiter below is the backstop for a client
    that has stopped honouring its own interval.
    """
    key = _spend_key(actor)
    ratelimit.limiter.check("render_refresh", key, ratelimit.RENDER_REFRESH)
    ratelimit.limiter.hit("render_refresh", key, ratelimit.RENDER_REFRESH)
    return render_service.refresh(
        conn, settings, min_interval=render_service.MANUAL_REFRESH_MIN_SECONDS
    )


@router.get("/renders/{job_id}/inputs", response_model=list[schemas.AssetOut])
def render_inputs(job_id: int, conn: ConnDep, _user: UserDep) -> list[schemas.AssetOut]:
    return [schemas.asset_out(r) for r in db.job_inputs(conn, job_id)]


@router.post("/renders/{job_id}/as-reference", response_model=schemas.AssetOut)
def use_as_reference(
    job_id: int,
    conn: ConnDep,
    settings: SettingsDep,
    actor: ActorDep,
    output_index: int = Query(default=0, ge=0),
) -> schemas.AssetOut:
    """Turn one image of a finished render into an attachable reference.

    This is what lets an approved image anchor the next render. `output_index`
    picks which of the render's images it is; 0 is the first.
    """
    try:
        row = asset_service.from_job_output(
            conn, settings, job_id, uploaded_by=actor.user_id, output_index=output_index,
        )
    except asset_service.AssetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.asset_out(row)
