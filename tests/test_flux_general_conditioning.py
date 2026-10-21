"""flux-general's two conditioning transports, and the reservation leak.

Two things are locked here, both reproduced against the live endpoint or the
live app on 2026-08-27 and both capable of costing real money if they regress.

  1. flux-general cannot carry `easycontrols` and `controlnet_unions` in the
     same request. It does not degrade — the pipeline fails to load, so every
     render on the proven engine of massing, interior and aerial returned a
     paid failure. The gateway now sends one transport or the other, never
     both, and refuses loudly when a conditioning type has no union equivalent.
  2. A job row reserved but never submitted holds budget forever. It counts as
     spend, is never polled, and nothing reaps it.

No network, no spend. Every fal.ai call is monkeypatched.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app import db, fal_gateway
from app.config import Settings
from app.models import ImageRole
from app.routing import Route, RoutingTable

ROUTING = Path(__file__).resolve().parents[1] / "routing.yaml"

UNION_PATH = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"
VIEW = "https://fal.test/view.png"


@pytest.fixture
def table() -> RoutingTable:
    return RoutingTable.load(ROUTING)


@pytest.fixture
def conn(tmp_path: Path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        fal_key="test-key-not-real",
        budget_monthly_usd=10.0,
        budget_per_user_usd=5.0,
        budget_per_job_usd=0.50,
        data_dir=tmp_path,
        db_path=tmp_path / "t.db",
    )


def _args(table: RoutingTable, task: str, **kw):
    route = table.resolve(task, "proven")
    spec = table.payload_spec(route.model_id)
    return fal_gateway.build_arguments(
        route, spec, prompt="a courtyard", width=1280, height=960, num_images=1,
        references=[(ImageRole.GEOMETRY, VIEW)], **kw,
    )


# --- Mutual exclusivity -----------------------------------------------------


def test_depth_and_canny_both_ride_the_union_and_easycontrols_is_absent(
    table: RoutingTable,
):
    """The bug this file exists for.

    Sending both lists returned "The size of tensor a (1537) must match the
    size of tensor b (2561)" at 512x512 — EasyControl appends its own 1024
    conditioning tokens to a 1537-token sequence and the ControlNet residual is
    computed for the shorter one. Two conditioning types therefore have to
    arrive as two controls inside ONE controlnet_unions entry.
    """
    args = _args(table, "massing_to_render")

    assert "easycontrols" not in args, (
        "easycontrols alongside controlnet_unions fails to load the pipeline"
    )
    union = args["controlnet_unions"]
    assert len(union) == 1
    assert union[0]["path"] == UNION_PATH

    controls = union[0]["controls"]
    assert [c["control_mode"] for c in controls] == ["depth", "canny"]
    assert all(c["control_image_url"] == VIEW for c in controls)
    assert [c["conditioning_scale"] for c in controls] == [0.35, 0.20]


def test_a_single_conditioning_type_also_goes_through_the_union(table: RoutingTable):
    """Not just the multi-control case.

    A canny-only task has no tensor clash to avoid, but easycontrols is also
    much the weaker control: at the same 0.82 it hallucinated straight through
    the master plan that the union reproduced line for line. Once the spec
    declares a union block, nothing on this endpoint uses easycontrols.
    """
    args = _args(table, "masterplan_prettify")

    assert "easycontrols" not in args
    controls = args["controlnet_unions"][0]["controls"]
    assert len(controls) == 1
    assert controls[0]["control_mode"] == "canny"
    assert controls[0]["conditioning_scale"] == 0.82


def test_a_sketch_is_conditioned_as_linework(table: RoutingTable):
    """`hedsketch` is an EasyControl preset with no union equivalent.

    It maps to the union's canny rather than being dropped — a hand sketch is
    linework — and the tolerance the soft-edge control used to provide now
    comes from the scale instead.
    """
    args = _args(table, "sketch_to_render")

    assert "easycontrols" not in args
    controls = args["controlnet_unions"][0]["controls"]
    assert [c["control_mode"] for c in controls] == ["canny"]
    assert controls[0]["conditioning_scale"] == 0.28


def test_an_unmappable_conditioning_type_is_refused_by_name(table: RoutingTable):
    """`seg` has no Union-Pro control_mode.

    The two silent options are both wrong: dropping the entry loses
    conditioning nobody can see missing, and falling back to emitting both
    lists reintroduces the tensor clash on every render. So it raises, naming
    the task, the type, and why the two cannot be combined.
    """
    spec = table.payload_spec("fal-ai/flux-general")
    route = Route(
        task="segmented_thing",
        label="l",
        model_id="fal-ai/flux-general",
        conditioning=[{"type": "seg", "scale": 0.5}],
    )
    with pytest.raises(ValueError) as excinfo:
        fal_gateway.build_arguments(
            route, spec, prompt="p", width=1280, height=960, num_images=1,
            image_url=VIEW,
        )

    message = str(excinfo.value)
    assert "segmented_thing" in message
    assert "seg" in message
    assert "easycontrols" in message and "controlnet_unions" in message


def test_seg_is_deliberately_unmapped(table: RoutingTable):
    """Guards the refusal above against someone inventing a mode for seg.

    Union-Pro's modes are canny, tile, depth, blur, pose, gray and low_quality.
    An ADE20K segmentation map is none of them.
    """
    modes = table.payload_spec("fal-ai/flux-general")["controlnet_union"]["modes"]
    assert "seg" not in modes
    assert set(modes.values()) <= {"canny", "tile", "depth", "blur", "pose",
                                   "gray", "low_quality"}


# --- Per-entry control images ----------------------------------------------


def test_a_true_depth_pass_overrides_only_the_depth_control(table: RoutingTable):
    """The CAD pipeline's whole point, preserved across the transport change.

    A true depth pass exported from Revit/SketchUp conditions the depth control
    while canny keeps reading the colour view — one union entry, two controls,
    two different images.
    """
    args = _args(table, "massing_to_render",
                 control_images={"depth": "https://fal.test/depth.png"})

    controls = args["controlnet_unions"][0]["controls"]
    by_mode = {c["control_mode"]: c["control_image_url"] for c in controls}
    assert by_mode == {"depth": "https://fal.test/depth.png", "canny": VIEW}


def test_an_override_for_a_type_the_route_does_not_use_changes_nothing(
    table: RoutingTable,
):
    """masterplan has no depth entry, so a depth pass has nowhere to land."""
    args = _args(table, "masterplan_prettify",
                 control_images={"depth": "https://fal.test/depth.png"})

    controls = args["controlnet_unions"][0]["controls"]
    assert [c["control_image_url"] for c in controls] == [VIEW]


# --- The differential-diffusion variant -------------------------------------


def test_differential_diffusion_reads_the_same_union_block(table: RoutingTable):
    """That endpoint has always sent unions only, so the exclusivity bug never
    reached it — but it reads the same renamed spec key, and a typo there would
    silently drop depth conditioning instead of raising. Nothing else covers
    this branch, so it is locked here.
    """
    route = table.resolve("skp_material_render", "proven")
    spec = table.payload_spec(route.model_id)
    args = fal_gateway.build_arguments(
        route, spec, prompt="p", width=1280, height=960, num_images=1,
        image_url="https://fal.test/albedo.png",
        control_images={
            "change_map": "https://fal.test/change.png",
            "depth": "https://fal.test/depth.png",
        },
    )

    assert args["image_url"] == "https://fal.test/albedo.png"
    assert args["change_map_image_url"] == "https://fal.test/change.png"
    assert "easycontrols" not in args
    union = args["controlnet_unions"][0]
    assert union["path"] == UNION_PATH
    # The scale is read from routing.yaml rather than repeated here: this test
    # locks the payload SHAPE (depth reaches the union block, and does not get
    # silently dropped by a renamed spec key), and pinning a tuned number here
    # only means a scale change looks like a structural regression. It was 0.6,
    # is 0.35 as of the 2026-08-31 measurement, and will move again.
    depth_scale = next(
        c["scale"] for c in route.conditioning if c["type"] == "depth"
    )
    assert union["controls"] == [{
        "control_image_url": "https://fal.test/depth.png",
        "control_mode": "depth",
        "conditioning_scale": depth_scale,
    }]


def test_differential_diffusion_refuses_without_a_change_map(table: RoutingTable):
    route = table.resolve("skp_material_render", "proven")
    spec = table.payload_spec(route.model_id)
    with pytest.raises(ValueError, match="needs a change map"):
        fal_gateway.build_arguments(
            route, spec, prompt="p", width=1280, height=960, num_images=1,
            image_url="https://fal.test/albedo.png",
        )


# --- The reservation leak ---------------------------------------------------


def _submit(conn, settings, table, **kw):
    return fal_gateway.submit(
        conn, settings, table,
        username="alice", task="massing_to_render", engine="proven",
        prompt="warm dusk light", references=[(ImageRole.GEOMETRY, VIEW)],
        num_images=1, **kw,
    )


def test_a_failure_between_reserving_and_submitting_closes_the_row(
    conn, settings: Settings, table: RoutingTable, monkeypatch
):
    """The P0. Anything raising after the reservation must mark the row failed.

    Reproduced 2026-08-27 with FAL_KEY="": _install_key raised out of submit(),
    POST /api/renders returned 500, and the row stayed `reserved` — counted as
    spend by db.spend_this_month, skipped by db.pending_jobs because it has no
    request_id, and reaped by nothing. Every 500 permanently ate a slice of the
    monthly ceiling.
    """
    def no_key(_settings):
        raise RuntimeError("FAL_KEY is not set.")

    monkeypatch.setattr(fal_gateway, "_install_key", no_key)
    monkeypatch.setattr(
        fal_gateway.fal_client, "submit",
        lambda *a, **k: pytest.fail("nothing should reach fal.ai without a key"),
    )

    with pytest.raises(RuntimeError, match="FAL_KEY"):
        _submit(conn, settings, table)

    rows = db.recent_jobs(conn, limit=10)
    assert len(rows) == 1
    assert rows[0]["status"] == "failed", "a reserved row would hold budget forever"
    assert rows[0]["request_id"] is None
    assert "FAL_KEY" in (rows[0]["error"] or "")

    # The point of marking it failed rather than leaving it: spend is released.
    assert db.spend_this_month(conn, "alice") == 0.0


def test_the_original_exception_is_not_swallowed(
    conn, settings: Settings, table: RoutingTable, monkeypatch
):
    """Closing the row must not turn a crash into a silent success or a
    different error — whoever is debugging needs the real one."""
    boom = ZeroDivisionError("disk went away mid-submit")

    def explode(_settings):
        raise boom

    monkeypatch.setattr(fal_gateway, "_install_key", explode)

    with pytest.raises(ZeroDivisionError) as excinfo:
        _submit(conn, settings, table)
    assert excinfo.value is boom
    assert db.recent_jobs(conn, limit=1)[0]["status"] == "failed"


def test_a_refused_submit_still_reports_nothing_charged(
    conn, settings: Settings, table: RoutingTable, monkeypatch
):
    """The pre-existing path — fal.ai itself rejecting the call — is unchanged:
    one `failed` row, one SubmitRefused, no double-marking."""
    monkeypatch.setattr(fal_gateway, "_install_key", lambda _s: None)

    def reject(*_a, **_kw):
        raise RuntimeError("422 unprocessable")

    monkeypatch.setattr(fal_gateway.fal_client, "submit", reject)

    with pytest.raises(fal_gateway.SubmitRefused, match="nothing charged"):
        _submit(conn, settings, table)

    row = db.recent_jobs(conn, limit=1)[0]
    assert row["status"] == "failed"
    assert "422 unprocessable" in (row["error"] or "")


def test_a_successful_submit_is_never_marked_failed(
    conn, settings: Settings, table: RoutingTable, monkeypatch
):
    """The other half of the guard. Once fal.ai has accepted the work it is
    paid for, so a later error must not close the row — that would under-count
    real spend and lose the only handle on the render."""
    monkeypatch.setattr(fal_gateway, "_install_key", lambda _s: None)

    class Handle:
        request_id = "req-accepted-1"

    monkeypatch.setattr(fal_gateway.fal_client, "submit", lambda *a, **k: Handle())

    real_update = db.update_job

    def update_then_break(c, job_id, **fields):
        real_update(c, job_id, **fields)
        if fields.get("status") == "queued":
            raise sqlite3.OperationalError("connection lost after the write")

    monkeypatch.setattr(fal_gateway.db, "update_job", update_then_break)

    with pytest.raises(sqlite3.OperationalError, match="connection lost"):
        _submit(conn, settings, table)

    row = db.recent_jobs(conn, limit=1)[0]
    assert row["status"] == "queued"
    assert row["request_id"] == "req-accepted-1"


# --- The third reservation leak: a request fal.ai has forgotten --------------


def _in_flight(conn, settings, table, monkeypatch, request_id: str) -> int:
    """Submit one job and leave it in flight with a known request_id."""
    monkeypatch.setattr(fal_gateway, "_install_key", lambda _s: None)
    monkeypatch.setattr(
        fal_gateway.fal_client, "submit",
        lambda *a, **k: type("H", (), {"request_id": request_id})(),
    )
    monkeypatch.setattr(fal_gateway, "upload_input", lambda *a, **k: VIEW)
    _submit(conn, settings, table)
    row = db.recent_jobs(conn, limit=1)[0]
    db.update_job(conn, int(row["id"]), status="in_progress")
    return int(row["id"])


def test_a_request_fal_no_longer_has_is_marked_failed(
    conn, settings: Settings, table: RoutingTable, monkeypatch
):
    """A 404 from fal's queue is terminal: nothing more is coming.

    Left `in_progress`, such a job is polled on every UI refresh forever and
    holds its estimate against the monthly cap. Found 2026-08-31 on a real row
    submitted 27 days earlier, still logging `{"status": "NOT_FOUND"}` on every
    poll cycle. Third variant of the same leak this file already covers twice.
    """
    job_id = _in_flight(conn, settings, table, monkeypatch, "gone-from-fal")

    def not_found(_model_id, _request_id):
        raise RuntimeError('{"status": "NOT_FOUND"}')

    monkeypatch.setattr(fal_gateway.fal_client, "status", not_found)

    assert fal_gateway.poll(conn, settings, job_id) == "failed"
    assert db.get_job(conn, job_id)["status"] == "failed"
    assert db.spend_this_month(conn, "alice") == 0.0, "budget must be released"


def test_a_transient_poll_error_leaves_the_job_in_flight(
    conn, settings: Settings, table: RoutingTable, monkeypatch
):
    """The other half, and the more expensive one to get wrong.

    Binning a render on a network blip would throw away an image the office has
    already paid for, so only NOT_FOUND is terminal — everything else re-raises
    and stays in flight for the next poll.
    """
    job_id = _in_flight(conn, settings, table, monkeypatch, "still-there")

    def blip(_model_id, _request_id):
        raise RuntimeError("Connection reset by peer")

    monkeypatch.setattr(fal_gateway.fal_client, "status", blip)

    with pytest.raises(RuntimeError, match="Connection reset"):
        fal_gateway.poll(conn, settings, job_id)
    assert db.get_job(conn, job_id)["status"] == "in_progress"
