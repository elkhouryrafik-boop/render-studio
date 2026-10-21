"""The application. One process on one office PC.

    uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

`--host 0.0.0.0` is what makes it reachable from other machines on the office
LAN rather than only from the PC it runs on. It does **not** expose anything to
the internet: 192.168.x.x is a private address and nothing outside the building
can route to it unless someone deliberately configures port-forwarding.

FastAPI serves both the JSON API and the built React bundle, so there is no
second process, no Node at runtime, and one port to remember.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextvars import ContextVar
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from api import deps, schemas
from api.routers import admin, assets, auth, cad, dev, renders
from app import db, migrations, ratelimit
from app.config import Settings, get_settings
from app.routing import RoutingError
from app.services import render as render_service

log = logging.getLogger(__name__)

WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"

# How often in-flight jobs are advanced without anyone clicking anything.
# fal.ai's queue is the job queue, so this is a poll, not a worker.
POLL_SECONDS = 10

# Expired sessions were never deleted, so the table only ever grew. Once an hour
# is often enough for a row that is already useless the moment it expires.
SESSION_PURGE_EVERY = int(3600 / POLL_SECONDS)

# The running poller, so /api/health can tell whether it is still alive. A dead
# poller is otherwise invisible: renders simply stop advancing and nothing says
# why. None means the lifespan never ran, which is the case under TestClient.
_poller_task: asyncio.Task[None] | None = None

# One id per request, on the response and in whatever the error handler logs, so
# a person reporting "it broke" and the line in the log can be matched up.
_request_id: ContextVar[str] = ContextVar("request_id", default="-")

REQUEST_ID_HEADER = "X-Request-ID"


def configure_logging() -> None:
    """Set up logging only if nothing else has.

    `logging.basicConfig` at import time fights uvicorn's `--log-config`: the
    module is imported after uvicorn has configured its handlers, and basicConfig
    then either does nothing or adds a second one that double-prints. Calling it
    from the lifespan, and only when the root logger is bare, means running under
    uvicorn keeps uvicorn's format and running the app any other way still logs.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


def _refresh_once(settings: Settings) -> dict[int, str]:
    with deps.connection() as conn:
        return render_service.refresh(conn, settings)


def _purge_sessions_once() -> int:
    with deps.connection() as conn:
        return db.purge_expired_sessions(conn)


async def _poller() -> None:
    """Advance in-flight renders in the background.

    Runs in a thread because every call underneath is blocking SQLite and
    httpx-sync. Any failure is logged and the loop continues: a poller that dies
    on one bad response would leave every later render stuck at "queued" with
    nothing to say why.
    """
    settings = get_settings()
    tick = 0
    while True:
        try:
            # Each tick opens its own connection inside the worker thread and
            # closes it when the tick is done.
            await asyncio.to_thread(_refresh_once, settings)
            tick += 1
            if tick % SESSION_PURGE_EVERY == 0:
                removed = await asyncio.to_thread(_purge_sessions_once)
                if removed:
                    log.info("Purged %s expired session(s)", removed)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Background poll failed; continuing")
        await asyncio.sleep(POLL_SECONDS)


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _poller_task

    configure_logging()
    settings = get_settings()
    with deps.connection() as conn:
        schema_version = migrations.current_version(conn)
        # Sessions expire but were never deleted. Once at startup, then hourly
        # from the poller.
        removed = db.purge_expired_sessions(conn)
    log.info(
        "Starting. engine=%s schema=v%s tasks=%s key=%s access_code=%s",
        settings.render_engine,
        schema_version,
        len(deps.get_table().available_tasks()),
        "SET" if settings.fal_key.get_secret_value() else "MISSING",
        "SET" if settings.office_access_code.get_secret_value() else "MISSING",
    )
    if settings.render_engine == "fake":
        log.warning("RENDER_ENGINE=fake: renders are placeholders and nothing is sent to fal.ai")
    if not settings.office_access_code.get_secret_value():
        log.warning("OFFICE_ACCESS_CODE is not set — signup is disabled.")
    if not settings.admin_email:
        log.warning(
            "ADMIN_EMAIL is not set. Nobody will be able to approve the first "
            "account. Set it before anyone signs up."
        )

    if removed:
        log.info("Purged %s expired session(s) at startup", removed)

    _poller_task = asyncio.create_task(_poller())
    try:
        yield
    finally:
        _poller_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _poller_task
        _poller_task = None


app = FastAPI(
    title="Render Studio",
    version="2.0.0",
    lifespan=lifespan,
    # No public docs. This is an internal tool and the schema is not a feature
    # anyone on the LAN needs to browse.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.middleware("http")
async def _request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Give every request an id and echo it back.

    An inbound id is honoured so a value from a log or a bug report can be
    followed straight through; nothing trusts it for anything but correlation,
    and it is truncated so a client cannot write an arbitrarily long header into
    the log.

    The id is written onto the request state as well as the context variable.
    The context variable alone is not enough: it is reset when this middleware
    unwinds, and an unhandled exception is caught *outside* it, so the error
    handler read the default and logged `request_id=-` — which is precisely the
    case the id exists for. Request state is backed by the ASGI scope, so it
    survives out to the handler.
    """
    incoming = request.headers.get(REQUEST_ID_HEADER, "").strip()[:64]
    rid = incoming or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    token = _request_id.set(rid)
    try:
        response = await call_next(request)
    finally:
        _request_id.reset(token)
    response.headers[REQUEST_ID_HEADER] = rid
    return response


def request_id_of(request: Request) -> str:
    """This request's id, from the scope, whatever else has unwound."""
    state = request.scope.get("state") or {}
    return str(state.get("request_id") or _request_id.get())


@app.exception_handler(ratelimit.RateLimited)
async def _rate_limited(_request: Request, exc: ratelimit.RateLimited) -> JSONResponse:
    """A limiter that escapes as a 500 tells an attacker they found something."""
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc)},
        headers={"Retry-After": str(exc.retry_after)},
    )


@app.exception_handler(RoutingError)
async def _routing_error(_request: Request, exc: RoutingError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# Field names as the architect would say them. Anything not listed is the
# snake_case name with its underscores turned to spaces.
_FIELD_LABELS = {
    "num_images": "Number of images",
    "mask_asset_id": "Mask",
    "project_id": "Project",
    "control_assets": "Control images",
    "geometry_hold": "Geometry hold",
    "output_index": "Output number",
    "extra_references": "Extra references",
    "i": "Image number",
}
_LOCATION_WORDS = {"body", "query", "path", "header", "cookie"}


def _field_label(loc: tuple) -> str:
    """The name of the field that failed, from the last string in its location."""
    names = [str(part) for part in loc if isinstance(part, str)]
    if not names:
        return "The request"
    if names[-1] == "body":
        return "Request body"
    if names[-1] in _LOCATION_WORDS:
        return "The request"
    raw = names[-1]
    if raw in _FIELD_LABELS:
        return _FIELD_LABELS[raw]
    text = raw.replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def plain_sentence(error: dict) -> str:
    """One pydantic error as one sentence an office member can act on."""
    field = _field_label(tuple(error.get("loc", ())))
    msg = str(error.get("msg", "")).strip()
    kind = str(error.get("type", ""))

    if kind == "missing" or msg == "Field required":
        return f"{field} is required."
    if kind == "int_parsing":
        return f"{field} must be a whole number."
    if kind in {"float_parsing", "decimal_parsing"}:
        return f"{field} must be a number."
    if kind in {"enum", "literal_error"}:
        choices = re.findall(r"'([^']*)'", msg)
        if choices:
            return f"{field} must be one of {', '.join(choices)}."
    bound = re.match(r"Input should be (less|greater) than or equal to (.+)$", msg)
    if bound:
        word = "at most" if bound.group(1) == "less" else "at least"
        return f"{field} must be {word} {bound.group(2)}."
    strict = re.match(r"Input should be (less|greater) than (.+)$", msg)
    if strict:
        return f"{field} must be {strict.group(1)} than {strict.group(2)}."
    return f"{field}: {msg.rstrip('.')}."


@app.exception_handler(RequestValidationError)
async def _invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """A 422 whose `detail` is a sentence, with the full pydantic list beside it.

    FastAPI's default puts a list of error objects in `detail`, which the front
    end could only show as "[object Object]". The first error decides the
    sentence; `errors` keeps every one for anything that wants the detail.
    """
    errors = jsonable_encoder(exc.errors())
    first = errors[0] if errors else {}
    return JSONResponse(
        status_code=422,
        content={
            "detail": plain_sentence(first) if first else "The request was not valid.",
            "errors": errors,
        },
    )


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Log the traceback with the request id, answer with JSON.

    Starlette's default returns a plain-text "Internal Server Error" and puts
    the traceback on stdout with nothing tying the two together. The front-end
    parses every response as JSON, so a bare text body became `null` and the
    error the architect saw was "something went wrong" with no way to find it in
    the log afterwards.
    """
    rid = request_id_of(request)
    log.exception(
        "Unhandled error on %s %s [request_id=%s]",
        request.method, request.url.path, rid,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Something went wrong on the server. The error has been logged.",
            "request_id": rid,
        },
        headers={REQUEST_ID_HEADER: rid},
    )


app.include_router(auth.router)
app.include_router(assets.router)
app.include_router(cad.router)
app.include_router(renders.router)
app.include_router(admin.router)
app.include_router(dev.router)


@app.get("/api/health", response_model=schemas.HealthOut)
def health(
    response: Response,
    conn: deps.ConnDep,
    table: deps.TableDep,
    settings: deps.SettingsDep,
) -> schemas.HealthOut:
    """Unauthenticated on purpose — it reports nothing an attacker can use.

    It used to be typed `Literal["ok"]`, which made it a liveness probe wearing
    a health check's name: it could report that the process was answering, and
    nothing else. It now checks the three things that fail silently — a database
    that has gone read-only, a missing key, and a poller that has died and left
    every render stuck at "queued" — and answers 503 when one of them has.

    Injected rather than reaching for the module-level connection, so a test can
    point it at a temporary database instead of the office's real one.
    """
    checks: dict[str, str] = {}

    # Writable, not merely readable. A full disk or a read-only mount answers
    # SELECT perfectly well and fails the moment a render is reserved.
    try:
        with db.transaction(conn):
            pass
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001 — the reason is the answer
        checks["database"] = f"not writable: {type(exc).__name__}"

    # The fake engine needs no key. Say so, rather than reporting a key problem
    # that the configured engine cannot have.
    fake = settings.render_engine == "fake"
    key_present = bool(settings.fal_key.get_secret_value())
    if key_present:
        checks["fal_key"] = "ok"
    elif fake:
        checks["fal_key"] = "not needed (fake engine)"
    else:
        checks["fal_key"] = "missing"

    if _poller_task is None:
        # Under TestClient the lifespan never runs, and a poller that was never
        # started is not a poller that died. Reported, not counted against.
        checks["poller"] = "not started"
    elif _poller_task.done():
        checks["poller"] = "stopped"
    else:
        checks["poller"] = "ok"

    healthy = (
        checks["database"] == "ok"
        and (key_present or fake)
        and checks["poller"] in {"ok", "not started"}
    )
    if not healthy:
        response.status_code = 503

    return schemas.HealthOut(
        status="ok" if healthy else "degraded",
        schema_version=migrations.current_version(conn),
        tasks=len(table.available_tasks()),
        engine=settings.render_engine,
        checks=checks,
    )


@app.api_route(
    "/api/{path:path}",
    include_in_schema=False,
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
def api_not_found(path: str) -> None:
    """Unknown API paths are 404 JSON, not the SPA.

    Registered after every router and before the catch-all below, which would
    otherwise answer `/api/does-not-exist` with 200 and index.html — and the
    front-end, which parses every response as JSON, turned that HTML into
    `null` and carried on as though the call had succeeded.
    """
    raise HTTPException(status_code=404, detail=f"No API route at /api/{path}.")


# --- The front-end ----------------------------------------------------------


class _BundleFiles(StaticFiles):
    """Static files with cache headers that match how Vite names them.

    Everything under /assets carries a content hash in its filename, so the
    bytes at a given URL never change and a year is a safe max-age. Without it
    the browser revalidates the whole bundle on every page load, which on a LAN
    is not slow but is entirely pointless.
    """

    def file_response(self, *args: object, **kwargs: object) -> Response:
        response = super().file_response(*args, **kwargs)  # type: ignore[arg-type]
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


def bundle_ready(dist: Path) -> bool:
    """Whether the built front-end can actually be served.

    `web/dist` existing does not mean it is usable: a half-finished or
    interrupted `npm run build` can leave the directory without `assets/`, and
    `StaticFiles(directory=...)` raises from its constructor — at import — so
    the app would not boot at all, API included. Both parts are checked so a
    broken build degrades to the 503 below, which at least says what to do
    about it.
    """
    return (dist / "index.html").is_file() and (dist / "assets").is_dir()


if bundle_ready(WEB_DIST):
    app.mount(
        "/assets", _BundleFiles(directory=WEB_DIST / "assets"), name="static-assets"
    )

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        """Serve the built React app, and let it own its own routing.

        Any path that is not a file falls through to index.html so a deep link
        or a page refresh lands in the app rather than on a 404. API routes are
        registered above this one, so they always win.
        """
        candidate = (WEB_DIST / path).resolve()
        # Containment check: a crafted path must not escape the bundle.
        if path and WEB_DIST in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        # index.html names the hashed bundle, so it is the one file that must
        # never be cached — a stale copy points at assets that no longer exist.
        return FileResponse(
            WEB_DIST / "index.html", headers={"Cache-Control": "no-cache"}
        )

else:

    @app.get("/", include_in_schema=False)
    def no_frontend() -> JSONResponse:
        detail = (
            "The API is running but the front-end has not been built. "
            "Run `npm install && npm run build` in web/."
        )
        if WEB_DIST.is_dir():
            detail = (
                "The API is running but web/dist is incomplete — index.html or "
                "assets/ is missing. Re-run `npm run build` in web/."
            )
        return JSONResponse(status_code=503, content={"detail": detail})
