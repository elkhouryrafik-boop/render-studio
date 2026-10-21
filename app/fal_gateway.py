"""The only place that talks to fal.ai.

Verified against fal-client 1.0.0 by introspection on 2026-08-03:
    submit(application, arguments, *, webhook_url=...) -> SyncRequestHandle(.request_id)
    status(application, request_id, *, with_logs=False) -> Queued | InProgress | Completed
    result(application, request_id) -> dict
    upload_file(path) -> str (url)
    cancel(application, request_id) -> None

Three things this module is responsible for, all of them safety rather than features:

  * The key. fal-client reads FAL_KEY from the process environment. This module is
    the one place it gets materialised, and it is never logged, echoed, or returned.
  * Idempotency. fal.ai's API has no idempotency key, so we build one. A job row is
    reserved under a content hash BEFORE submitting; a duplicate key means the work
    was already paid for and we return the existing job instead of spending twice.
  * Bounded retries. fal.ai's queue already auto-retries 10x on 5xx internally. Our
    retries are only for transport failures on OUR side of the call, capped at 2,
    and never for a request that may already have been accepted.
  * No stranded reservations. A row that is reserved and then never submitted
    holds budget forever — it counts as spend, nothing polls it, nothing reaps
    it. Every path out of `submit()` after the reservation therefore leaves the
    row either carrying a request_id or marked `failed`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fal_client
import httpx

from app import budget, db
from app.config import Settings
from app.fake_engine import DEFAULT_SIZE, FakeFalClient, placeholder_png
from app.models import ROLE_ORDER, ImageRole
from app.routing import Route, RoutingError, RoutingTable

log = logging.getLogger(__name__)

# Our own retries. fal.ai's queue handles 5xx internally so this
# only covers a connection dying before the request was accepted.
MAX_TRANSPORT_RETRIES = 2
RETRY_BACKOFF_SECONDS = (1.0, 3.0)


class SubmitRefused(Exception):
    """The job was not submitted and nothing was spent.

    `reason` is set when the refusal came from a budget gate, so a caller can
    choose its own wording instead of parsing the message.
    """

    def __init__(self, message: str, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class SubmitResult:
    job_id: int
    request_id: str | None
    est_usd: float
    duplicate: bool = False
    # Filled in by services.render.submit_render, which knows where the depth
    # control image came from and whether the geometry hold was capped.
    depth_source: str = "none"
    notes: tuple[str, ...] = ()


_fake_client: FakeFalClient | None = None
_fake_lock = threading.Lock()


def _client(settings: Settings) -> Any:
    """The one switch between the real fal client and the zero-spend fake.

    Every call this module makes to fal goes through here. With
    RENDER_ENGINE=fal it is the `fal_client` module itself, so tests that
    monkeypatch `fal_gateway.fal_client.submit` keep working unchanged. With
    RENDER_ENGINE=fake it is a process singleton, rebuilt only if the configured
    delay changes.
    """
    global _fake_client
    if settings.render_engine != "fake":
        return fal_client
    with _fake_lock:
        if _fake_client is None or _fake_client.delay_s != settings.fake_engine_delay_s:
            _fake_client = FakeFalClient(delay_s=settings.fake_engine_delay_s)
        return _fake_client


def _install_key(settings: Settings) -> None:
    """Materialise the key into the environment for fal-client.

    Deliberately not stored on a module global and never included in log output.
    The fake engine has no key to install, so this returns at once in that mode.
    """
    if settings.render_engine == "fake":
        return
    os.environ["FAL_KEY"] = settings.require_key()


def _idempotency_key(
    username: str,
    task: str,
    model_id: str,
    arguments: dict[str, Any],
    salt: str | None = None,
) -> str:
    """Stable hash of what is actually being asked for.

    Two identical submissions from the same user collapse to one paid job. Change
    any argument — prompt, seed, resolution — and it is legitimately a new job.

    `salt` is for a retry. A retry sends the same arguments as the failed job, so
    without a salt it would hash to the failed job's key and collapse onto it. The
    salt names the retry attempt; two clicks on Retry for the same attempt still
    collapse to one job.
    """
    body: dict[str, Any] = {"u": username, "t": task, "m": model_id, "a": arguments}
    if salt is not None:
        body["s"] = salt
    payload = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def order_references(
    references: Sequence[tuple[ImageRole, str]],
) -> list[tuple[ImageRole, str]]:
    """Sort references into the order the endpoint receives them.

    The prompt refers to images by position — "image 1 is the geometry to obey".
    That only holds if the ordering is fixed and shared by the prompt builder and
    the payload builder, so it lives in one function that both call.

    Geometry first: it is the shape the render must not drift from, and on a
    conditioning engine it is the only reference that can be honoured at all.
    Ties keep the order the architect arranged them in.
    """
    return sorted(references, key=lambda r: ROLE_ORDER.index(r[0]) if r[0] in ROLE_ORDER else 99)


def build_arguments(route: Route, spec: dict[str, Any], *, prompt: str,
                    width: int, height: int, num_images: int,
                    image_url: str | None = None,
                    references: Sequence[tuple[ImageRole, str]] | None = None,
                    mask_url: str | None = None,
                    control_images: dict[str, str] | None = None,
                    extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the fal.ai payload for a route.

    `spec` is the endpoint's request contract from routing.yaml. These endpoints
    genuinely disagree about their shape — one wants `image_url`, another
    `image_urls`, flux-general wants the source image inside its conditioning
    entries, and flux-2-pro/edit has no `num_images` field at all. Guessing here
    is what produced the first round of 422s, so nothing is assumed: the shape
    comes from config, and an unknown value raises rather than falling through.

    `references` carries several images with their roles. `image_url` is the
    single-image shorthand and means the same as one GEOMETRY reference — most
    tasks have exactly one input and should not have to say so twice.

    `control_images` overrides individual conditioning entries by type —
    `{"depth": url}` sends a true depth pass on the `depth` entry while every
    other entry still reads the geometry reference. Absent or missing keys fall
    back to that single reference, which is why a task with no true depth pass
    to offer behaves exactly as it did before this parameter existed.
    """
    if image_url and references:
        raise ValueError(
            "Pass either image_url or references, not both — otherwise the "
            "position each image occupies in the prompt is ambiguous."
        )
    refs = order_references(list(references or []))
    if image_url:
        refs = [(ImageRole.GEOMETRY, image_url)]

    args: dict[str, Any] = dict(route.defaults)
    args["prompt"] = prompt

    if spec.get("size", "image_size") == "image_size":
        args["image_size"] = {"width": width, "height": height}

    if spec.get("num_images", True):
        args["num_images"] = num_images

    image_mode = spec.get("image", "image_url")

    # How many references this endpoint can actually carry. Declared per
    # endpoint in routing.yaml because the schemas genuinely differ: two take an
    # unbounded array, the rest take exactly one. Going over is refused rather
    # than truncated — silently dropping the material reference an architect
    # attached produces a render that is wrong for a reason nobody can see.
    max_images = int(spec.get("max_images", 1))
    if len(refs) > max_images:
        raise ValueError(
            f"{route.engine_label} accepts {max_images} reference image"
            f"{'' if max_images == 1 else 's'}, but {len(refs)} were attached "
            f"({', '.join(r.value for r, _ in refs)}). Remove some, or choose an "
            "engine that reads several references."
        )

    if refs:
        urls = [url for _, url in refs]
        if image_mode == "image_url":
            args["image_url"] = urls[0]
        elif image_mode == "image_urls":
            args["image_urls"] = urls
        elif image_mode == "easycontrols":
            # flux-general. It offers two conditioning transports and they are
            # MUTUALLY EXCLUSIVE — a request carrying both fails to load the
            # pipeline at all. Verified with paid probes on 2026-08-27:
            # easycontrols[canny] + controlnet_unions[depth] at 512x512 returns
            #   "The size of tensor a (1537) must match the size of tensor b
            #    (2561) at non-singleton dimension 1"
            # while either transport alone works. The arithmetic says why:
            # 1537 = (512/16)^2 + 512 text tokens + 1 is the normal sequence
            # length, and 2561 = 1537 + 1024 is that sequence after EasyControl
            # appends its own fixed ~512px conditioning tokens ((512/16)^2 =
            # 1024). The ControlNet residual is computed against the
            # un-appended length, so it cannot be added to the longer sequence.
            # At our default 1280x960 the same clash reads 5313 vs 6337. This
            # is structural, not a scale or a resolution we can tune around.
            #
            # So: if the endpoint declares a `controlnet_union` block, ALL
            # conditioning goes through it and no easycontrols entry is emitted.
            # That is also the better half of the pair on fidelity — measured
            # 2026-08-27 at 1280x960 on a ~28 MB test drawing: easycontrols canny
            # at 0.75/0.82 hallucinated content that was not in the source, while
            # union canny at the same scale kept the drawn linework and labels in
            # their original positions.
            #
            # Only when there is no union block do we fall back to easycontrols,
            # where control_method_url takes a plain method name ("canny",
            # "hedsketch", ...) and image_control_type must be spatial for
            # architectural geometry — subject would transfer the object, not
            # the layout.
            #
            # Either way, conditioning applies to the geometry reference by
            # default: every entry reads the same image, which is why
            # max_images still pins this endpoint to one reference.
            # `control_images` is the opt-in exception — an entry whose type has
            # an override in that mapping reads its own image instead (a true
            # depth pass from the CAD pipeline alongside the colour view), and
            # everything else keeps fanning out as before.
            control_url = urls[0]
            overrides = control_images or {}
            union = spec.get("controlnet_union")
            if union:
                modes = union.get("modes") or {}
                union_controls = []
                for c in route.conditioning:
                    ctype = c.get("type")
                    mode = modes.get(ctype)
                    if mode is None:
                        # Never silently drop the entry and never fall back to
                        # emitting both lists — one loses conditioning nobody
                        # can see missing, the other produces the tensor clash
                        # above on every single render.
                        raise ValueError(
                            f"Task {route.task!r} conditions on {ctype!r}, which "
                            f"has no controlnet_union control_mode on "
                            f"{route.engine_label}. This engine cannot carry "
                            "easycontrols and controlnet_unions in the same "
                            "request, so this type cannot be combined with union "
                            "conditioning. Map it in routing.yaml under "
                            "the engine's controlnet_union modes, "
                            "or route this task to an engine that does not use "
                            "union conditioning."
                        )
                    union_controls.append({
                        "control_image_url": overrides.get(ctype, control_url),
                        "control_mode": mode,
                        "conditioning_scale": c.get("scale", 0.5),
                    })
                if union_controls:
                    args["controlnet_unions"] = [{
                        "path": union["path"],
                        "controls": union_controls,
                    }]
            else:
                easy_entries = [
                    {
                        "control_method_url": c.get("type"),
                        "image_url": overrides.get(c.get("type"), control_url),
                        "image_control_type": "spatial",
                        "scale": c.get("scale", 0.5),
                    }
                    for c in route.conditioning
                ]
                if easy_entries:
                    args["easycontrols"] = easy_entries
        elif image_mode == "differential_diffusion":
            # fal-ai/flux-general/differential-diffusion. image_url carries
            # the pre-textured, unlit albedo base — not a colour photo, so
            # this is never fanned out the way easycontrols entries are.
            # change_map_image_url is a genuine per-pixel strength multiplier
            # (arXiv 2306.00950) built from our own confidence tiers, and it
            # is not a reference image — it arrives only through
            # control_images, exactly like a true depth pass does for the
            # base model. Depth still conditions geometry via
            # controlnet_unions. E4 (2026-08-07) validated this exact
            # three-part combination in one call: 2 renders, $0.15, 0
            # failures.
            args["image_url"] = urls[0]
            overrides = control_images or {}
            change_url = overrides.get("change_map")
            if not change_url:
                raise ValueError(
                    f"{route.engine_label} needs a change map. Pass "
                    "control_assets={'change_map': <asset id>} from the SKP "
                    "confirm response."
                )
            args["change_map_image_url"] = change_url
            union = spec.get("controlnet_union")
            depth_url = overrides.get("depth")
            depth_conditioning = next(
                (c for c in route.conditioning if c.get("type") == "depth"), None
            )
            if union and depth_conditioning and depth_url:
                args["controlnet_unions"] = [{
                    "path": union["path"],
                    "controls": [{
                        "control_image_url": depth_url,
                        "control_mode": (union.get("modes") or {}).get("depth", "depth"),
                        "conditioning_scale": depth_conditioning.get("scale", 0.6),
                    }],
                }]
        elif image_mode != "none":
            raise ValueError(
                f"Unknown image mode {image_mode!r} for {route.engine_label} "
                "in routing.yaml."
            )

    # Masked endpoints declare `mask: required` in routing.yaml. flux-pro/v1/fill
    # lists mask_url as required and rejects the request without it, so refusing
    # here turns a 422 that reads like a bug into a sentence that says what is
    # missing. The mask must match the source image's dimensions exactly — that
    # is the endpoint's rule, enforced wherever the mask is produced.
    if spec.get("mask") == "required":
        if not mask_url:
            raise ValueError(
                f"{route.engine_label} needs a mask. Mark a region first, or "
                "use an engine that takes a written instruction instead."
            )
        args["mask_url"] = mask_url
    elif mask_url:
        raise ValueError(
            f"{route.engine_label} takes no mask, but one was supplied. "
            "Route this to a masked engine or drop the mask."
        )

    if route.lora_url:
        args["loras"] = [{"path": route.lora_url, "scale": 1.0}]

    # Parameters this endpoint rejects outright.
    for key in spec.get("drop") or []:
        args.pop(key, None)

    if extra:
        args.update(extra)
    return args


def upload_input(settings: Settings, path: Path) -> str:
    """Upload a local file to fal.ai storage and return its URL."""
    _install_key(settings)
    return _client(settings).upload_file(path)


def submit(
    conn: sqlite3.Connection,
    settings: Settings,
    table: RoutingTable,
    *,
    username: str,
    task: str,
    prompt: str,
    engine: str | None = None,
    width: int = 1280,
    height: int = 960,
    num_images: int = 1,
    input_path: Path | None = None,
    references: Sequence[tuple[ImageRole, str]] | None = None,
    mask_url: str | None = None,
    mask_asset_id: int | None = None,
    control_images: dict[str, str] | None = None,
    user_id: int | None = None,
    project_id: int | None = None,
    parent_job_id: int | None = None,
    per_user_cap_usd: float | None = None,
    extra: dict[str, Any] | None = None,
    conditioning_scales: dict[str, float] | None = None,
    retry_of_job_id: int | None = None,
    retry_salt: str | None = None,
    meta: dict[str, Any] | None = None,
) -> SubmitResult:
    """Resolve, price, authorise, reserve, then submit. In that order.

    `meta` is recorded alongside the request arguments in params_json. It is
    not part of what fal.ai receives, and not part of the idempotency key.

    The order is the point. Nothing reaches fal.ai until the cost is known and
    the budget has approved it, and the reservation is written before the call so
    a crash mid-submit cannot lose track of money already committed.
    """
    # 0. A fake:// URL names a file only the fake engine can read. Sent to fal.ai
    #    it is a rejected request that the reservation would still pay for, so it
    #    is refused before anything is priced or spent.
    if settings.render_engine == "fal":
        carried = [url for _, url in (references or ())]
        carried += [mask_url or "", *(control_images or {}).values()]
        if any(url.startswith("fake://") for url in carried):
            raise SubmitRefused(
                "This image was made by the fake engine and cannot be sent to fal.ai. "
                "Attach the original image again, then render.",
                reason="fake_url",
            )

    # 1. Resolve — refuses disabled tasks and non-allowlisted models.
    #    engine=None takes the task's default. The chosen engine reaches the
    #    idempotency key through route.model_id, so the same prompt run on the
    #    other engine is correctly treated as a different job.
    route = table.resolve(task, engine)
    spec = table.payload_spec(route.model_id)
    if conditioning_scales:
        # Geometry hold: replace the scale of the entries named (depth, canny),
        # leave every other entry as routing.yaml has it.
        route = dataclasses.replace(route, conditioning=[
            {**c, "scale": conditioning_scales[c["type"]]}
            if c.get("type") in conditioning_scales else c
            for c in route.conditioning
        ])

    # 2. Price. An unpriced model raises rather than defaulting to zero.
    #    Endpoints that return one image per call are billed for one, whatever
    #    the caller asked for, so the estimate has to agree with the request.
    if not spec.get("num_images", True):
        num_images = 1
    est_usd = table.estimate_usd(route.model_id, width, height, num_images)

    # 3. Pre-flight. Advisory only: this is the same set of gates as step 5 but
    #    read outside a transaction, so it can be raced. It runs here anyway
    #    because step 4 uploads the reference images, and refusing a
    #    fat-fingered resolution *after* pushing 40MB to fal.ai is a poor trade.
    #    The gate that actually decides is the atomic one below.
    try:
        budget.check(
            conn, settings, username=username, est_usd=est_usd,
            user_id=user_id, per_user_cap_usd=per_user_cap_usd,
        )
    except budget.BudgetExceeded as exc:
        raise SubmitRefused(str(exc), reason=exc.reason) from exc

    # 4. Upload the input first — its URL is part of the idempotency hash.
    #    Callers holding already-hosted images (a parent render's output, a
    #    cached upload) pass `references` and skip this entirely.
    image_url: str | None = None
    if input_path is not None:
        image_url = upload_input(settings, input_path)

    # A shape this endpoint cannot accept — a missing mask, an unknown image
    # mode — is a refusal, not a crash. Nothing has been submitted at this point,
    # so it reaches the caller through the same channel as a budget refusal.
    try:
        arguments = build_arguments(
            route, spec, prompt=prompt, width=width, height=height,
            num_images=num_images, image_url=image_url, references=references,
            mask_url=mask_url, control_images=control_images, extra=extra,
        )
    except ValueError as exc:
        raise SubmitRefused(str(exc), reason="payload") from exc

    # 5. Authorise and reserve, as ONE write. This is the gate that decides.
    #
    #    Step 3 reads a running total and this writes a row, and with nothing
    #    transactional between them N concurrent submissions all measured
    #    themselves against the same pre-spend number and all passed. Measured
    #    2026-08-27 against a temp database, 40 threads, a $1 cap and $0.10 a
    #    job: the split path committed $4.00 and reserved 40 jobs. This one
    #    commits exactly $1.00 and reserves 10.
    #
    #    A duplicate key means we already paid for this exact request.
    key = _idempotency_key(username, task, route.model_id, arguments, salt=retry_salt)
    try:
        reservation = budget.reserve_within_budget(
            conn,
            settings,
            username=username,
            est_usd=est_usd,
            user_id=user_id,
            per_user_cap_usd=per_user_cap_usd,
            idempotency_key=key,
            task=task,
            model_id=route.model_id,
            params_json=json.dumps({**arguments, **(meta or {})}, default=str),
            input_path=str(input_path) if input_path else None,
            project_id=project_id,
            parent_job_id=parent_job_id,
            engine=route.engine,
            mask_asset_id=mask_asset_id,
            retry_of_job_id=retry_of_job_id,
        )
    except budget.BudgetExceeded as exc:
        raise SubmitRefused(str(exc), reason=exc.reason) from exc

    if reservation.job_id is None:
        existing = (
            db.get_job(conn, reservation.duplicate_of)
            if reservation.duplicate_of is not None
            else db.find_by_key(conn, key)
        )
        if existing is None:
            raise SubmitRefused(
                "That render could not be reserved. Try again.", reason="reserve"
            )
        log.info("Duplicate submission collapsed to job %s", existing["id"])
        return SubmitResult(
            job_id=int(existing["id"]),
            request_id=existing["request_id"],
            est_usd=float(existing["est_usd"]),
            duplicate=True,
        )
    job_id = reservation.job_id

    # 6. Submit, with bounded transport retries.
    #
    # Everything from here is guarded, because a row left `reserved` is a leak
    # that never heals: db.spend_this_month counts it forever (only `refused`
    # and `failed` are excluded), db.pending_jobs never polls it (it filters on
    # request_id IS NOT NULL), and nothing reaps it. The reservation above is
    # already holding budget, so any failure before a request_id is recorded
    # has to close the row on its way out. Reproduced 2026-08-27 with an empty
    # FAL_KEY: _install_key raised out of this function, POST /api/renders
    # returned 500, and the reserved row held its share of the ceiling
    # permanently — that failure is now the `except BaseException` below.
    request_id: str | None = None
    try:
        _install_key(settings)
        last_exc: Exception | None = None
        for attempt in range(MAX_TRANSPORT_RETRIES + 1):
            try:
                handle = _client(settings).submit(route.model_id, arguments=arguments)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                # Connection never established — safe to retry, nothing was accepted.
                last_exc = exc
                if attempt < MAX_TRANSPORT_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                    continue
                break
            except Exception as exc:  # noqa: BLE001 — record and stop, never blind-retry
                # Anything else may have been accepted server-side. Retrying could
                # double-bill, so we stop and surface it.
                last_exc = exc
                break

            # Accepted, and therefore paid for. From this line on the row must
            # never be closed as failed, whatever else goes wrong — that would
            # under-count real spend and lose the only handle we have on the work.
            request_id = str(handle.request_id)
            db.update_job(conn, job_id, request_id=request_id, status="queued")
            log.info("Submitted job %s to %s (est $%.4f)", job_id, route.model_id, est_usd)
            return SubmitResult(job_id=job_id, request_id=request_id, est_usd=est_usd)

        _close_reservation(conn, job_id, f"Submit failed: {last_exc}")
        raise SubmitRefused(f"Submit failed, nothing charged: {last_exc}")
    except SubmitRefused:
        raise  # the row above is already closed
    except BaseException as exc:
        if request_id is None:
            _close_reservation(conn, job_id, f"{type(exc).__name__}: {exc}")
        else:
            log.error(
                "Job %s was accepted by fal.ai as %s but its row could not be "
                "updated — it is paid for and must NOT be marked failed: %s",
                job_id, request_id, exc,
            )
        raise


def _close_reservation(conn: sqlite3.Connection, job_id: int, error: str) -> None:
    """Mark a reserved row failed so it stops holding budget.

    Best-effort on purpose: this runs while another exception is already on its
    way out, and masking that one with a database error would trade a stranded
    row for a lost diagnosis. `failed` is the right terminal state rather than
    `refused` — the request did reach the point of being submitted, it just
    never got there.
    """
    try:
        db.update_job(conn, job_id, status="failed", error=str(error)[:500])
    except Exception:
        # Never mask the original failure with a database one.
        log.exception("Could not release the reservation on job %s", job_id)


def poll(conn: sqlite3.Connection, settings: Settings, job_id: int) -> str:
    """Check one in-flight job. Returns its status string.

    Uses fal.ai's own durable queue as the job queue — no Redis, no worker
    process. The request_id in SQLite is the handle.
    """
    row = db.get_job(conn, job_id)
    if row is None:
        raise ValueError(f"No job {job_id}")
    if row["status"] in db.TERMINAL:
        return str(row["status"])
    if not row["request_id"]:
        return str(row["status"])

    _install_key(settings)
    model_id = str(row["model_id"])
    request_id = str(row["request_id"])

    client = _client(settings)
    try:
        st = client.status(model_id, request_id)
    except Exception as exc:
        # fal.ai does not keep a request forever. Once it 404s there is nothing
        # left to wait for, and a job left `in_progress` is polled on every UI
        # refresh for the rest of the install's life while holding its estimate
        # against the monthly cap. This is the third variant of the same leak:
        # `reserved` rows that never reached fal, rows that reached fal but lost
        # their request_id, and now rows whose request fal has forgotten. Found
        # 2026-08-31 on a job submitted 2026-08-04 that was still logging
        # `{"status":"NOT_FOUND"}` on every poll cycle.
        #
        # Only NOT_FOUND is terminal. A network blip or a 5xx must stay
        # in-flight and be retried, so anything else is re-raised for poll_all
        # to log — losing a render the office paid for because the office wifi
        # dropped would be a far worse bug than the one being fixed.
        if "NOT_FOUND" not in str(exc) and "404" not in str(exc):
            raise
        db.update_job(
            conn, job_id, status="failed",
            error="fal.ai no longer has this request. It expired from their "
                  "queue before the result was collected, most likely because "
                  "the server was off for an extended period. Re-submit to "
                  "render it again.",
        )
        return "failed"

    if isinstance(st, fal_client.Queued):
        db.update_job(conn, job_id, status="queued")
        return "queued"
    if isinstance(st, fal_client.InProgress):
        db.update_job(conn, job_id, status="in_progress")
        return "in_progress"

    # Completed — fetch the payload and persist the image locally.
    try:
        payload = client.result(model_id, request_id)
    except Exception as exc:  # noqa: BLE001
        db.update_job(conn, job_id, status="failed", error=str(exc)[:500])
        return "failed"

    items = _output_items(payload)
    urls = [item.url for item in items]
    if not urls:
        db.update_job(
            conn, job_id, status="failed",
            error="The render came back without an image. Retry sends it again.",
        )
        return "failed"

    seed = _output_seed(payload)
    # An image saved on an earlier tick is kept; only the gaps are fetched again.
    # The job stays in flight until every image is on disk, and the timeout bounds
    # the retries.
    on_disk = {int(r["idx"]): r["path"] for r in db.list_job_outputs(conn, job_id)}
    gaps = [i for i in range(len(urls)) if not on_disk.get(i)]
    fetched = dict(zip(gaps, _save_images(
        settings, job_id, [urls[i] for i in gaps],
        sizes=[(items[i].width, items[i].height) for i in gaps], indexes=gaps,
    ), strict=True)) if gaps else {}
    saved: list[Path | None] = [
        Path(on_disk[i]) if on_disk.get(i) else fetched.get(i) for i in range(len(urls))
    ]
    complete = all(path is not None for path in saved)
    if not complete:
        log.warning("Job %s: %d image(s) not saved yet; the next tick retries them", job_id,
                    sum(path is None for path in saved))
    # One row per image, so a four-image job keeps four. Index 0 is also written
    # to the job row: it is what every older reader still looks at.
    with db.transaction(conn):
        for idx, (url, path) in enumerate(zip(urls, saved, strict=True)):
            db.add_job_output(
                conn, job_id=job_id, idx=idx,
                path=str(path) if path else None, url=url, seed=seed,
            )
        db.update_job(
            conn, job_id,
            status="completed" if complete else "in_progress",
            output_path=str(saved[0]) if saved[0] else None,
            # The URL is what makes editing cheap: continuing from this render hands
            # fal.ai an image it is already hosting, so there is no download and no
            # re-upload. Retention is undocumented, hence the local copy as well.
            output_url=urls[0],
            output_seed=seed,
        )
    return "completed" if complete else "in_progress"


@dataclass(frozen=True)
class _OutputImage:
    url: str
    width: int | None
    height: int | None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _output_items(payload: Any) -> list[_OutputImage]:
    """Image URLs with the size the endpoint reported for each, where it did.

    Endpoints disagree here too: most return `images: [{url, width, height}]`,
    some a single `image: {url}`. Both shapes are read rather than assumed.
    """
    items: list[_OutputImage] = []
    if isinstance(payload, dict):
        for entry in payload.get("images", []) or []:
            if isinstance(entry, dict) and entry.get("url"):
                items.append(_OutputImage(
                    str(entry["url"]),
                    _int_or_none(entry.get("width")),
                    _int_or_none(entry.get("height")),
                ))
        if not items and isinstance(payload.get("image"), dict):
            entry = payload["image"]
            if entry.get("url"):
                items.append(_OutputImage(
                    str(entry["url"]),
                    _int_or_none(entry.get("width")),
                    _int_or_none(entry.get("height")),
                ))
    return items


_SQLITE_INT_MAX = 2**63 - 1


def _output_seed(payload: Any) -> int | None:
    """The seed fal.ai actually used, where it reports one.

    Worth keeping: re-rendering with the same seed and one changed word is the
    cheapest controlled comparison available, and it is impossible after the
    fact if the seed was never recorded.
    """
    if isinstance(payload, dict):
        seed = payload.get("seed")
        if isinstance(seed, str) and seed.isdigit():
            seed = int(seed)
        if isinstance(seed, int) and not isinstance(seed, bool):
            # fal reports unsigned 64-bit seeds. SQLite INTEGER is signed
            # 64-bit, and sqlite3 raises OverflowError past 2**63-1, which
            # made every poll of such a job fail until the timeout sweep
            # threw the paid result away (seen 2026-10-10, job 2 of the paid
            # smoke). Anything out of range is stored as hex text ("0x..."): the
            # column has INTEGER affinity, which turns decimal text that does
            # not fit into a lossy REAL but leaves hex text alone;
            # db.seed_from_db decodes it back to the exact integer.
            if 0 <= seed <= _SQLITE_INT_MAX:
                return seed
            return hex(seed)  # type: ignore[return-value]
    return None


def _save_images(
    settings: Settings,
    job_id: int,
    urls: list[str],
    sizes: Sequence[tuple[int | None, int | None]] | None = None,
    indexes: Sequence[int] | None = None,
) -> list[Path | None]:
    """Pull generated images down to local storage.

    fal.ai's CDN retention is ambiguous in their docs (7 days vs 'forever'), so we do not rely on it. Everything
    a user might show a client is stored locally.

    A `fake://` URL comes from the fake engine: nothing is downloaded, and a
    labelled placeholder is written at the size the request asked for.
    """
    # Aligned with `urls`: entry i is the file for urls[i], or None if that one
    # could not be saved. Index and file must agree, or output 3 would be served
    # as output 2 whenever output 2 failed to download.
    # `indexes` names the output each url belongs to, so a retry of one image still
    # writes its own file name. Defaults to 0..n-1.
    numbers = list(indexes) if indexes is not None else list(range(len(urls)))
    out: list[Path | None] = []
    outputs = settings.outputs_dir()
    outputs.mkdir(parents=True, exist_ok=True)
    for pos, url in enumerate(urls):
        idx = numbers[pos]
        dest = outputs / f"job{job_id:06d}_{idx}.png"
        try:
            if url.startswith("fake://"):
                width, height = sizes[pos] if sizes and pos < len(sizes) else (None, None)
                # fake://result/<request_id>/<i>
                parts = url.split("/")
                request_id = parts[3] if len(parts) > 3 else "unknown"
                dest.write_bytes(placeholder_png(
                    width=width or DEFAULT_SIZE[0],
                    height=height or DEFAULT_SIZE[1],
                    request_id=request_id,
                    index=idx,
                ))
                out.append(dest)
                continue
            with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    fh.writelines(r.iter_bytes())
            out.append(dest)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not download output %s for job %s: %s", idx, job_id, exc)
            out.append(None)
    return out


# In-flight states the timeout applies to. `reserved` is included: a row that
# never got a request_id would otherwise hold its budget share indefinitely.
_IN_FLIGHT = ("reserved", "queued", "in_progress")


def _now() -> datetime:
    """The clock the timeout reads. Tests replace this rather than sleep."""
    return datetime.now(UTC)


def _timeout_message(job_timeout_s: int) -> str:
    minutes = round(job_timeout_s / 60)
    if minutes < 1:
        return f"Timed out after {job_timeout_s} seconds. Retry sends it again."
    unit = "minute" if minutes == 1 else "minutes"
    return f"Timed out after {minutes} {unit}. Retry sends it again."


def _cancel_upstream(settings: Settings, application: str, request_id: str) -> None:
    """Ask fal.ai to stop a request we are about to give up on. Best-effort.

    fal.ai bills for work that has run, so a render we fail is cancelled rather
    than left to finish unseen. Nothing depends on the answer: the job is failed
    either way, and a request that already completed simply refuses the cancel.
    """
    _install_key(settings)
    client = _client(settings)
    if not hasattr(client, "cancel"):
        return
    try:
        client.cancel(application, request_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not cancel fal request %s: %s", request_id, exc)


def expire_stale_jobs(conn: sqlite3.Connection, settings: Settings) -> list[int]:
    """Fail every in-flight job older than `settings.job_timeout_s`.

    A render that fal.ai never finishes would otherwise sit `in_progress` for
    as long as the install runs, holding its estimate against the monthly cap.
    The reservation is released through the same path as a submit failure, and
    `timed_out_at` says it was a timeout rather than a report from fal.ai.

    Returns the ids it expired. A job that fal.ai completes after this point is
    not collected: it is terminal here, and the output is lost to the office.
    That is the trade the timeout makes; Retry sends the render again.
    """
    cutoff = _now().timestamp() - settings.job_timeout_s
    placeholders = ", ".join("?" for _ in _IN_FLIGHT)
    rows = conn.execute(
        f"SELECT id, created_at, model_id, request_id FROM jobs WHERE status IN ({placeholders})",
        _IN_FLIGHT,
    ).fetchall()
    expired: list[int] = []
    message = _timeout_message(settings.job_timeout_s)
    for row in rows:
        created = datetime.fromisoformat(str(row["created_at"]))
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        if created.timestamp() >= cutoff:
            continue
        job_id = int(row["id"])
        if row["request_id"]:
            _cancel_upstream(settings, str(row["model_id"]), str(row["request_id"]))
        # One transaction, so a job is never `failed` without its timeout mark.
        # A failure here leaves this job in flight for the next tick and does not
        # stop the other expiries.
        try:
            with db.transaction(conn):
                db.update_job(conn, job_id, status="failed", error=message)
                db.update_job(conn, job_id, timed_out_at=db.utcnow())
        except Exception:
            log.exception("Could not expire job %s; the next tick will retry", job_id)
            continue
        log.warning("Job %s timed out after %ss and was failed", job_id, settings.job_timeout_s)
        expired.append(job_id)
    return expired


def poll_all(conn: sqlite3.Connection, settings: Settings) -> dict[int, str]:
    """Advance every in-flight job. Cheap enough to call on each UI refresh.

    Every job is polled before any is expired. A render that fal finished between
    two ticks is then saved, not failed, and only a job still in flight after its
    poll can time out.
    """
    results: dict[int, str] = {}
    for row in db.pending_jobs(conn):
        jid = int(row["id"])
        try:
            results[jid] = poll(conn, settings, jid)
        except Exception as exc:  # noqa: BLE001
            log.warning("Poll failed for job %s: %s", jid, exc)
            results[jid] = str(row["status"])
    for jid in expire_stale_jobs(conn, settings):
        results[jid] = "failed"
    return results


__all__ = [
    "RoutingError",
    "SubmitRefused",
    "SubmitResult",
    "build_arguments",
    "poll",
    "poll_all",
    "submit",
    "upload_input",
]
