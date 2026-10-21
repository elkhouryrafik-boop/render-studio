"""Guardrail tests. These never touch fal.ai and never spend money.

They exist because the guardrails are the part of this app that must not silently
regress: fal.ai has no spend cap of its own, so if these break, nothing else stops
a runaway bill.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import budget, db, fal_gateway
from app.config import Settings
from app.routing import Route, RoutingError, RoutingTable

ROUTING = Path(__file__).resolve().parents[1] / "routing.yaml"


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
        kill_switch=False,
        data_dir=tmp_path,
        db_path=tmp_path / "t.db",
    )


# --- Routing ----------------------------------------------------------------


def test_routing_table_loads_and_self_validates(table: RoutingTable):
    assert table.available_tasks(), "no enabled tasks"


def test_workhorse_task_routes_to_controlnet_endpoint(table: RoutingTable):
    route = table.resolve("massing_to_render")
    assert route.model_id == "fal-ai/flux-general"
    assert any(c["type"] == "depth" for c in route.conditioning)


def test_masterplan_holds_the_hardest_linework(table: RoutingTable):
    """Site plans are read at board scale and scaled off. If someone tunes the
    conditioning down below the floorplan task's, plans stop being measurable
    and nothing else in the app will say so."""
    masterplan = table.resolve("masterplan_prettify")
    floorplan = table.resolve("floorplan_prettify")
    mp_canny = next(c["scale"] for c in masterplan.conditioning if c["type"] == "canny")
    fp_canny = next(c["scale"] for c in floorplan.conditioning if c["type"] == "canny")
    assert mp_canny >= fp_canny


def test_perspective_tasks_condition_far_below_orthographic_ones(table: RoutingTable):
    """The scale split that 2026-08-27's measurements established.

    This replaces "interiors are conditioned softer than facades", which was a
    judgement about EasyControl's grip and did not survive the move to
    controlnet_unions — interior and massing now sit at the same numbers. The
    real axis turned out to be what the OUTPUT has to be, not what the input
    is: a perspective view becoming a photograph needs the conditioning to get
    out of the way (0.7/0.4 came back as an untextured clay model), while an
    orthographic drawing being restyled has no photorealism to lose and wants
    everything the control can give (canny 0.45 lost the plan, 0.75 held it).
    """
    def highest(task: str) -> float:
        return max(c["scale"] for c in table.resolve(task).conditioning)

    perspective = ["massing_to_render", "interior_to_render",
                   "aerial_view_render", "sketch_to_render"]
    orthographic = ["floorplan_prettify", "masterplan_prettify",
                    "elevation_section_render"]

    assert max(highest(t) for t in perspective) < min(highest(t) for t in orthographic)


def test_unknown_task_raises_rather_than_defaulting(table: RoutingTable):
    with pytest.raises(RoutingError):
        table.resolve("make_it_pretty")


def test_every_enabled_task_routes_to_an_allowlisted_model(table: RoutingTable):
    for name, _ in table.available_tasks():
        if table.in_development(name):
            # Listed so the office can see it exists, refused by resolve() so
            # nothing can spend on it. Its engines are still checked for
            # allowlisting and pricing by the test below, which reads the spec
            # directly — so releasing it later cannot surface a broken route.
            continue
        assert table.is_allowed(table.resolve(name).model_id)


def test_every_engine_of_every_task_is_allowlisted_and_priced(table: RoutingTable):
    for name, _ in table.available_tasks():
        for engine in table.engines(name):
            assert table.is_allowed(engine.model_id), f"{name}/{engine.key}"
            # Raises RoutingError if the model has no cost entry.
            table.estimate_usd(engine.model_id, 1024, 1024)


def test_every_task_offers_a_choice_and_explains_it(table: RoutingTable):
    # Both single-engine exceptions are for the same reason: there is only
    # one model that does this specific thing, so a second choice would be
    # a placeholder, not an alternative.
    single_engine_tasks = {
        "skp_material_render",  # nothing else reads an albedo+confidence-map pair
    }
    for name, _ in table.available_tasks():
        engines = table.engines(name)
        if name in single_engine_tasks:
            continue
        assert len(engines) == 2, f"{name} should offer two engines"
        for engine in engines:
            assert engine.label, f"{name}/{engine.key} has no label"
            assert len(engine.note) > 30, f"{name}/{engine.key} has no explanation"


def test_geometry_critical_tasks_default_to_the_conditioned_pipeline(
    table: RoutingTable,
):
    """The default must hold the building's shape.

    The newer engines produce a better picture but treat the input as a
    reference rather than tracing it. Whichever way someone edits routing.yaml
    later, the untouched default on these tasks stays the one that constrains
    geometry.
    """
    for name in (
        "massing_to_render",
        "sketch_to_render",
        "floorplan_prettify",
        "interior_to_render",
        "aerial_view_render",
        "masterplan_prettify",
        "elevation_section_render",
    ):
        route = table.resolve(name)
        assert route.model_id == "fal-ai/flux-general", name
        assert route.conditioning, f"{name} default carries no conditioning"


def test_choosing_an_engine_changes_the_model(table: RoutingTable):
    proven = table.resolve("massing_to_render", "proven")
    latest = table.resolve("massing_to_render", "latest")
    assert proven.model_id != latest.model_id
    # The engine reaches the idempotency key via model_id, so the same prompt
    # on the other engine is a different job rather than a deduplicated one.
    assert latest.model_id == "fal-ai/flux-2-pro/edit"


def test_unknown_engine_raises_rather_than_falling_back(table: RoutingTable):
    with pytest.raises(RoutingError):
        table.resolve("massing_to_render", "whatever_is_newest")


def test_unpriced_model_refuses_to_estimate(table: RoutingTable):
    with pytest.raises(RoutingError):
        table.estimate_usd("fal-ai/not-a-real-model", 1024, 1024)


# --- Request shapes ---------------------------------------------------------
# These endpoints do not share a request contract, and the first live calls
# failed with 422s because the app assumed they did. The shapes below were read
# off each endpoint's own schema page. Lock them so a refactor cannot quietly
# reintroduce the guess.


def _args(table: RoutingTable, task: str, engine: str, **kw):
    route = table.resolve(task, engine)
    spec = table.payload_spec(route.model_id)
    return fal_gateway.build_arguments(
        route, spec, prompt="a courtyard", width=1280, height=960,
        num_images=kw.get("num_images", 2),
        image_url=kw.get("image_url", "https://example.test/plan.png"),
        mask_url=kw.get("mask_url"),
    )


def test_unpriced_or_unshaped_model_refuses_to_build(table: RoutingTable):
    with pytest.raises(RoutingError):
        table.payload_spec("fal-ai/not-a-real-model")


def test_flux2_edit_sends_a_list_and_omits_num_images(table: RoutingTable):
    """flux-2-pro/edit takes image_urls (a list) and has no num_images field."""
    args = _args(table, "massing_to_render", "latest")
    assert args["image_urls"] == ["https://example.test/plan.png"]
    assert "image_url" not in args
    assert "num_images" not in args
    # Also absent from that endpoint's schema, so it must not be sent.
    assert "guidance_scale" not in args


def test_flux_general_conditions_via_controlnet_unions_only(table: RoutingTable):
    """The source image belongs inside the conditioning entries, not top level.

    The earlier shape sent controlnets entries carrying a `type` key and no
    `path`, which the endpoint rejected. It was then split across easycontrols
    and controlnet_unions, which turned out to be a shape flux-general cannot
    load at all — the two are mutually exclusive (2026-08-27). Everything now
    rides one controlnet_unions entry; tests/test_flux_general_conditioning.py
    carries the mechanism and the arithmetic.
    """
    args = _args(table, "massing_to_render", "proven")
    assert "controlnets" not in args
    assert "easycontrols" not in args
    assert "image_url" not in args
    assert args["num_images"] == 2

    union = args["controlnet_unions"]
    assert len(union) == 1
    assert union[0]["path"] == "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"
    controls = union[0]["controls"]
    assert [c["control_mode"] for c in controls] == ["depth", "canny"]
    for control in controls:
        assert control["control_image_url"] == "https://example.test/plan.png"


def test_nano_banana_omits_image_size(table: RoutingTable):
    """It sizes by aspect_ratio and resolution, and rejects image_size."""
    args = _args(table, "variation", "latest")
    assert args["image_urls"] == ["https://example.test/plan.png"]
    assert "image_size" not in args
    assert args["num_images"] == 2


def test_single_image_endpoints_are_declared_as_such(table: RoutingTable):
    assert table.supports_num_images("fal-ai/flux-general") is True
    assert table.supports_num_images("fal-ai/flux-2-pro/edit") is False


def test_unknown_image_mode_raises_rather_than_sending_a_bad_request():
    route = Route(task="t", label="l", model_id="fal-ai/whatever")
    with pytest.raises(ValueError):
        fal_gateway.build_arguments(
            route, {"image": "telepathy"}, prompt="p", width=1, height=1,
            num_images=1, image_url="https://example.test/x.png",
        )


# --- Contracts read from the endpoints' own OpenAPI schemas, 2026-08-04 -------
# Four entries in routing.yaml disagreed with the schema they described. Each of
# these locks one of those corrections so it cannot quietly come back.


def test_kontext_omits_image_size(table: RoutingTable):
    """kontext has no image_size field. It sizes by aspect_ratio.

    The old entry declared `size: image_size`, so every edit on the proven
    engine sent a field the endpoint does not have.
    """
    args = _args(table, "variation", "proven")
    assert args["image_url"] == "https://example.test/plan.png"
    assert "image_size" not in args
    assert "num_inference_steps" not in args
    assert args["guidance_scale"] == 3.5     # this one it does accept, 1-20


def test_fill_refuses_without_a_mask(table: RoutingTable):
    """mask_url is required by the schema. Nothing ever sent one.

    Refusing here turns a 422 that reads like a bug into a sentence naming what
    is missing, before anything is submitted or reserved.
    """
    with pytest.raises(ValueError, match="needs a mask"):
        _args(table, "inpaint", "proven")


def test_fill_carries_the_mask_when_it_has_one(table: RoutingTable):
    args = _args(table, "inpaint", "proven", mask_url="https://example.test/m.png")
    assert args["mask_url"] == "https://example.test/m.png"
    assert args["image_url"] == "https://example.test/plan.png"
    assert "image_size" not in args          # absent from this schema too


def test_mask_is_refused_by_endpoints_that_take_none(table: RoutingTable):
    """A mask reaching a maskless endpoint is a routing mistake, not a no-op."""
    with pytest.raises(ValueError, match="takes no mask"):
        _args(table, "variation", "latest", mask_url="https://example.test/m.png")


def test_seedvr_drops_the_prompt_it_has_no_field_for(table: RoutingTable):
    """seedvr/upscale/image declares no prompt. build_arguments always sets one."""
    args = _args(table, "upscale", "latest")
    assert "prompt" not in args
    assert args["image_url"] == "https://example.test/plan.png"


def test_clarity_keeps_its_prompt(table: RoutingTable):
    """The other upscaler does take a prompt — the drop must not be blanket."""
    args = _args(table, "upscale", "proven")
    assert args["prompt"] == "a courtyard"


def test_sketches_are_conditioned_on_their_linework(table: RoutingTable):
    """The task still declares `hedsketch`, the control meant for hand-drawn
    input, but that preset only exists on EasyControl — which flux-general can
    no longer be sent (2026-08-27). routing.yaml maps it to the union's canny,
    because a hand sketch is linework, and the low scale does the tolerating
    the soft-edge control used to.
    """
    route = table.resolve("sketch_to_render", "proven")
    assert [c["type"] for c in route.conditioning] == ["hedsketch"]

    args = _args(table, "sketch_to_render", "proven")
    assert "easycontrols" not in args
    controls = args["controlnet_unions"][0]["controls"]
    assert [c["control_mode"] for c in controls] == ["canny"]


def test_nano_banana_renders_above_its_1k_default(table: RoutingTable):
    """Left alone it returns 1 megapixel, which is too small to show a client."""
    args = _args(table, "variation", "latest")
    assert args["resolution"] == "2K"


def test_verified_endpoints_declare_how_many_images_they_accept(table: RoutingTable):
    """max_images is what makes a refusal possible instead of a silent truncation.

    Two endpoints take an unbounded array and the rest take exactly one. Without
    this declared per endpoint, attaching a second reference to a single-image
    model would be dropped without telling anyone.
    """
    for name, _ in table.available_tasks():
        for engine in table.engines(name):
            spec = table.payload_spec(engine.model_id)
            if spec.get("verified") == "UNVERIFIED":
                continue
            assert spec.get("max_images", 0) >= 1, engine.model_id
    assert table.payload_spec("fal-ai/flux-pro/kontext")["max_images"] == 1
    assert table.payload_spec("fal-ai/nano-banana-pro/edit")["max_images"] > 1


def test_every_routed_endpoint_has_had_its_schema_read(table: RoutingTable):
    """An UNVERIFIED shape on a task someone can click is a 422 waiting to happen."""
    for name, _ in table.available_tasks():
        for engine in table.engines(name):
            assert table.payload_verified(engine.model_id) != "UNVERIFIED", (
                f"{name}/{engine.key} routes to {engine.model_id}, whose request "
                "shape has never been checked against its schema page."
            )


def test_controlnet_path_is_priced_3x_the_plain_model(table: RoutingTable):
    """The cost trap from the research: arch-viz needs flux-general, not flux/dev."""
    general = table.estimate_usd("fal-ai/flux-general", 1000, 1000)
    dev = table.estimate_usd("fal-ai/flux/dev", 1000, 1000)
    assert general == pytest.approx(dev * 3.0)


# --- Budget -----------------------------------------------------------------


def test_kill_switch_blocks_everything(conn, settings: Settings):
    settings.kill_switch = True
    with pytest.raises(budget.BudgetExceeded, match="kill switch"):
        budget.check(conn, settings, username="alice", est_usd=0.01)


def test_per_job_cap_catches_oversized_request(conn, settings: Settings):
    with pytest.raises(budget.BudgetExceeded, match="per-job limit"):
        budget.check(conn, settings, username="alice", est_usd=0.99)


def test_per_user_cap_stops_one_person_draining_the_budget(conn, settings: Settings):
    for i in range(11):
        jid = db.reserve_job(
            conn, idempotency_key=f"k{i}", username="alice", task="t",
            model_id="fal-ai/flux/dev", est_usd=0.45, params_json="{}", input_path=None,
        )
        db.update_job(conn, jid, status="completed")
    with pytest.raises(budget.BudgetExceeded, match="monthly allowance"):
        budget.check(conn, settings, username="alice", est_usd=0.40)


def test_global_cap_holds_across_different_users(conn, settings: Settings):
    for i in range(23):
        jid = db.reserve_job(
            conn, idempotency_key=f"g{i}", username=f"user{i}", task="t",
            model_id="fal-ai/flux/dev", est_usd=0.45, params_json="{}", input_path=None,
        )
        db.update_job(conn, jid, status="completed")
    with pytest.raises(budget.BudgetExceeded, match="monthly cap"):
        budget.check(conn, settings, username="fresh_user", est_usd=0.40)


def test_refused_and_failed_jobs_do_not_count_as_spend(conn, settings: Settings):
    for i, status in enumerate(["refused", "failed"]):
        jid = db.reserve_job(
            conn, idempotency_key=f"r{i}", username="alice", task="t",
            model_id="fal-ai/flux/dev", est_usd=99.0, params_json="{}", input_path=None,
        )
        db.update_job(conn, jid, status=status)
    assert db.spend_this_month(conn) == 0.0
    budget.check(conn, settings, username="alice", est_usd=0.10)


# --- Idempotency ------------------------------------------------------------


def test_duplicate_key_cannot_reserve_twice(conn):
    first = db.reserve_job(
        conn, idempotency_key="same", username="alice", task="t",
        model_id="fal-ai/flux/dev", est_usd=0.05, params_json="{}", input_path=None,
    )
    second = db.reserve_job(
        conn, idempotency_key="same", username="alice", task="t",
        model_id="fal-ai/flux/dev", est_usd=0.05, params_json="{}", input_path=None,
    )
    assert first is not None
    assert second is None, "duplicate submission would have been billed twice"


def test_identical_requests_hash_to_the_same_key():
    from app.fal_gateway import _idempotency_key
    a = _idempotency_key("alice", "sketch_to_render", "m", {"prompt": "x", "seed": 1})
    b = _idempotency_key("alice", "sketch_to_render", "m", {"seed": 1, "prompt": "x"})
    assert a == b, "key must not depend on dict ordering"


def test_different_prompt_is_a_different_job():
    from app.fal_gateway import _idempotency_key
    a = _idempotency_key("alice", "sketch_to_render", "m", {"prompt": "dusk"})
    b = _idempotency_key("alice", "sketch_to_render", "m", {"prompt": "noon"})
    assert a != b


# --- Secret hygiene ---------------------------------------------------------


def test_settings_never_expose_the_key_in_dumps(settings: Settings):
    dumped = settings.safe_dump()
    assert dumped["fal_key"] == "SET"
    assert "test-key-not-real" not in repr(settings)
    assert "test-key-not-real" not in str(dumped)


def test_missing_key_fails_loudly(tmp_path: Path):
    s = Settings(fal_key="", data_dir=tmp_path, db_path=tmp_path / "x.db")
    with pytest.raises(RuntimeError, match="FAL_KEY is not set"):
        s.require_key()


def test_update_job_refuses_unknown_columns(conn):
    jid = db.reserve_job(
        conn, idempotency_key="col", username="alice", task="t",
        model_id="fal-ai/flux/dev", est_usd=0.01, params_json="{}", input_path=None,
    )
    with pytest.raises(ValueError):
        db.update_job(conn, jid, username="attacker")


# --- The atomic reservation -------------------------------------------------
#
# `budget.check()` reads a SUM() and returns; the row that changes that SUM is
# written afterwards. Between the two, N concurrent submissions all read the
# same pre-spend total and all pass, so the office can overshoot its cap by the
# per-job limit times the concurrency. `budget.reserve_within_budget()` does the
# read and the write inside one BEGIN IMMEDIATE transaction, which is what these
# tests hold to.


def _reserve_kwargs(key: str) -> dict:
    return {
        "idempotency_key": key,
        "task": "massing_to_render",
        "model_id": "fal-ai/flux/dev",
        "params_json": "{}",
        "input_path": None,
    }


def test_concurrent_reservations_cannot_overshoot_the_office_cap(tmp_path: Path):
    """Forty threads, a $1 cap and ten cents a job: exactly ten get through.

    Each thread opens its own connection, which is the shape that matters —
    BEGIN IMMEDIATE is what makes the check honest between two writers, and a
    single shared handle would prove only that a mutex works.
    """
    import threading

    db_path = tmp_path / "race.db"
    db.connect(db_path).close()  # migrate once, before the threads start

    settings = Settings(
        fal_key="k", budget_monthly_usd=1.0, budget_per_user_usd=100.0,
        budget_per_job_usd=0.50, data_dir=tmp_path, db_path=db_path,
    )

    attempts = 40
    est = 0.10
    start = threading.Barrier(attempts)
    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt(n: int) -> None:
        conn = db.connect(db_path)
        try:
            start.wait(timeout=30)
            try:
                budget.reserve_within_budget(
                    conn, settings, username="alice", est_usd=est,
                    **_reserve_kwargs(f"race-{n}"),
                )
                result = "reserved"
            except budget.BudgetExceeded as exc:
                result = exc.reason
            with lock:
                outcomes.append(result)
        finally:
            conn.close()

    threads = [threading.Thread(target=attempt, args=(n,)) for n in range(attempts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "a reservation thread hung"

    conn = db.connect(db_path)
    try:
        committed = db.spend_this_month(conn)
        rows = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
    finally:
        conn.close()

    assert outcomes.count("reserved") == 10, outcomes
    assert rows == 10
    assert committed <= 1.0 + 1e-9, f"office cap overshot: ${committed:.2f} of $1.00"
    assert all(o in {"reserved", "monthly"} for o in outcomes)


def test_reserve_within_budget_refuses_over_the_per_user_cap(conn, settings: Settings):
    db.reserve_job(
        conn, username="alice", est_usd=4.9, user_id=None,
        **_reserve_kwargs("spent-already"),
    )
    with pytest.raises(budget.BudgetExceeded) as exc_info:
        budget.reserve_within_budget(
            conn, settings, username="alice", est_usd=0.4,
            **_reserve_kwargs("over-the-line"),
        )
    assert exc_info.value.reason == "per_user"
    assert db.find_by_key(conn, "over-the-line") is None, "a refused job was still written"


def test_reserve_within_budget_collapses_a_duplicate_instead_of_spending_twice(
    conn, settings: Settings
):
    first = budget.reserve_within_budget(
        conn, settings, username="alice", est_usd=0.10, **_reserve_kwargs("same-request"),
    )
    second = budget.reserve_within_budget(
        conn, settings, username="alice", est_usd=0.10, **_reserve_kwargs("same-request"),
    )
    assert first.job_id is not None
    assert second.duplicate and second.duplicate_of == first.job_id
    assert db.spend_this_month(conn) == pytest.approx(0.10)


def test_reserve_within_budget_still_honours_the_cheap_gates(conn, settings: Settings):
    with pytest.raises(budget.BudgetExceeded, match="per-job limit"):
        budget.reserve_within_budget(
            conn, settings, username="alice", est_usd=0.99, **_reserve_kwargs("too-big"),
        )
    settings.kill_switch = True
    with pytest.raises(budget.BudgetExceeded, match="kill switch"):
        budget.reserve_within_budget(
            conn, settings, username="alice", est_usd=0.01, **_reserve_kwargs("paused"),
        )


def test_a_dangling_project_id_is_a_named_refusal_not_a_crash(conn):
    """It used to be swallowed as a duplicate idempotency key, which returned
    None to a caller that then dereferenced a job it had never inserted."""
    with pytest.raises(db.ForeignKeyViolation, match="No project 424242"):
        db.reserve_job(
            conn, username="alice", est_usd=0.01, project_id=424242,
            **_reserve_kwargs("bad-project"),
        )
    with pytest.raises(db.ForeignKeyViolation, match="No project 424242"):
        db.create_asset(conn, sha256="abc", project_id=424242)


def test_a_genuine_duplicate_key_is_still_a_duplicate(conn):
    """The FK branch must not swallow the case it was split out of."""
    first = db.reserve_job(
        conn, username="alice", est_usd=0.01, **_reserve_kwargs("twice"),
    )
    assert first is not None
    assert db.reserve_job(
        conn, username="alice", est_usd=0.01, **_reserve_kwargs("twice"),
    ) is None


# --- Runtime controls -------------------------------------------------------


def test_the_kill_switch_can_be_turned_on_without_a_restart(conn, settings: Settings):
    """`.env` is read once at boot through an lru_cache, so it used to be the
    only way to stop spending — and the only way to apply it was to kill the
    process."""
    assert budget.kill_switch(conn, settings) is False
    budget.set_kill_switch(conn, enabled=True, actor="boss@example.test")
    assert budget.kill_switch(conn, settings) is True
    with pytest.raises(budget.BudgetExceeded, match="kill switch"):
        budget.check(conn, settings, username="alice", est_usd=0.01)

    budget.set_kill_switch(conn, enabled=False, actor="boss@example.test")
    assert budget.kill_switch(conn, settings) is False


def test_the_office_cap_can_be_lowered_without_a_restart(conn, settings: Settings):
    assert budget.monthly_cap_usd(conn, settings) == 10.0
    budget.set_monthly_cap(conn, cap_usd=0.05, actor="boss@example.test")
    assert budget.monthly_cap_usd(conn, settings) == 0.05
    with pytest.raises(budget.BudgetExceeded) as exc_info:
        budget.check(conn, settings, username="alice", est_usd=0.10)
    assert exc_info.value.reason == "monthly"

    # null restores the .env default rather than meaning "zero".
    budget.set_monthly_cap(conn, cap_usd=None, actor="boss@example.test")
    assert budget.monthly_cap_usd(conn, settings) == 10.0


def test_an_unparseable_stored_cap_falls_back_to_env(conn, settings: Settings):
    db.set_app_setting(conn, budget.MONTHLY_CAP_KEY, "not a number", updated_by=None)
    assert budget.monthly_cap_usd(conn, settings) == 10.0
