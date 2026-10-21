"""Tests for the v2 layer: migrations, accounts, assets, multi-reference.

Same rule as the guardrail tests — no network, no spend. Every fal.ai call is
monkeypatched, and any test that reaches one is a bug in the test.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app import db, fal_gateway, migrations
from app.config import Settings
from app.models import AssetKind, ImageRole, UserStatus
from app.routing import RoutingTable
from app.services import assets, render

ROUTING = Path(__file__).resolve().parents[1] / "routing.yaml"


@pytest.fixture
def table() -> RoutingTable:
    return RoutingTable.load(ROUTING)


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


@pytest.fixture
def conn(tmp_path: Path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def png(width: int = 64, height: int = 48, colour: str = "white") -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="PNG")
    return buf.getvalue()


# --- Migrations -------------------------------------------------------------


def test_fresh_database_is_at_the_current_version(conn):
    assert migrations.current_version(conn) == migrations.SCHEMA_VERSION


def test_migrating_twice_changes_nothing(tmp_path: Path):
    path = tmp_path / "twice.db"
    first = db.connect(path)
    jid = db.reserve_job(
        first, idempotency_key="k", username="alice", task="t",
        model_id="fal-ai/flux/dev", est_usd=0.01, params_json="{}", input_path=None,
    )
    first.close()

    second = db.connect(path)
    assert migrations.current_version(second) == migrations.SCHEMA_VERSION
    assert db.get_job(second, jid) is not None
    second.close()


def test_a_v1_database_keeps_its_jobs_and_its_ledger(tmp_path: Path):
    """The live database has real jobs and real spend in it.

    Migrating must not lose a row or move a number, and every pre-existing job
    must come out as the root of its own lineage.
    """
    path = tmp_path / "v1.db"
    raw = sqlite3.connect(path)
    raw.executescript(
        """
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idempotency_key TEXT NOT NULL UNIQUE, username TEXT NOT NULL,
            task TEXT NOT NULL, model_id TEXT NOT NULL, request_id TEXT,
            status TEXT NOT NULL, est_usd REAL NOT NULL DEFAULT 0, actual_usd REAL,
            params_json TEXT, input_path TEXT, output_path TEXT, error TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        INSERT INTO jobs (idempotency_key, username, task, model_id, status,
                          est_usd, created_at, updated_at)
        VALUES ('old1', 'alice', 'massing_to_render', 'fal-ai/flux-general',
                'completed', 0.18, '2099-01-01T00:00:00+00:00',
                '2099-01-01T00:00:00+00:00');
        """
    )
    raw.commit()
    raw.close()

    conn = db.connect(path)
    row = db.get_job(conn, 1)
    assert row is not None
    assert row["username"] == "alice"
    assert float(row["est_usd"]) == 0.18
    assert row["user_id"] is None          # predates accounts, and that is fine
    assert row["root_job_id"] == 1         # its own lineage root
    conn.close()


def test_a_newer_database_is_refused_rather_than_guessed_at(tmp_path: Path):
    path = tmp_path / "future.db"
    conn = db.connect(path)
    conn.execute(f"PRAGMA user_version = {migrations.SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="newer version"):
        migrations.migrate(conn)
    conn.close()


# --- Accounts ---------------------------------------------------------------


def test_new_accounts_cannot_do_anything_until_approved(conn):
    uid = db.create_user(conn, email="A@example.test", name="A", password_hash="x")
    row = db.get_user(conn, uid)
    assert row["status"] == UserStatus.PENDING
    assert row["email"] == "a@example.test", "email must be normalised for lookup"


def test_a_second_account_on_the_same_email_is_refused(conn):
    assert db.create_user(conn, email="a@example.test", name="A", password_hash="x")
    assert db.create_user(conn, email="A@example.test", name="A again", password_hash="y") is None


def test_a_session_resolves_only_while_the_account_is_active(conn):
    uid = db.create_user(
        conn, email="a@example.test", name="A", password_hash="x",
        status=UserStatus.ACTIVE,
    )
    token = db.create_session(conn, uid)
    assert db.session_user(conn, token)["id"] == uid

    # Suspension takes effect on the next request, not whenever the cookie
    # happens to expire.
    db.set_user_status(conn, uid, UserStatus.SUSPENDED)
    assert db.session_user(conn, token) is None


def test_the_raw_session_token_is_never_stored(conn):
    uid = db.create_user(
        conn, email="a@example.test", name="A", password_hash="x",
        status=UserStatus.ACTIVE,
    )
    token = db.create_session(conn, uid)
    stored = conn.execute("SELECT token_hash FROM sessions").fetchone()["token_hash"]
    assert token not in stored
    assert len(token) > 20


def test_revoked_and_expired_sessions_stop_resolving(conn):
    uid = db.create_user(
        conn, email="a@example.test", name="A", password_hash="x",
        status=UserStatus.ACTIVE,
    )
    revoked = db.create_session(conn, uid)
    db.revoke_session(conn, revoked)
    assert db.session_user(conn, revoked) is None

    stale = db.create_session(conn, uid)
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    conn.execute("UPDATE sessions SET expires_at = ? WHERE revoked_at IS NULL", (past,))
    assert db.session_user(conn, stale) is None


# --- Per-user budget, now that it can be enforced ---------------------------


def test_spend_from_before_accounts_still_counts_against_the_person(conn):
    """Activating accounts must not hand everyone a fresh budget mid-month.

    A v1 row carries a username and no user_id. If the per-user query only
    matched on user_id, everyone's monthly spend would reset to zero the day
    accounts shipped, and the office could spend its cap twice in one month.
    """
    uid = db.create_user(
        conn, email="alice@example.test", name="Alice", password_hash="x",
        status=UserStatus.ACTIVE,
    )
    legacy = db.reserve_job(
        conn, idempotency_key="legacy", username="alice@example.test", task="t",
        model_id="fal-ai/flux/dev", est_usd=0.40, params_json="{}", input_path=None,
    )
    db.update_job(conn, legacy, status="completed")
    modern = db.reserve_job(
        conn, idempotency_key="modern", username="alice@example.test", task="t",
        model_id="fal-ai/flux/dev", est_usd=0.60, params_json="{}", input_path=None,
        user_id=uid,
    )
    db.update_job(conn, modern, status="completed")

    both = db.spend_this_month(conn, username="alice@example.test", user_id=uid)
    assert both == pytest.approx(1.00)


def test_an_admin_can_give_one_person_a_different_allowance(conn, settings: Settings):
    from app import budget

    uid = db.create_user(
        conn, email="a@example.test", name="A", password_hash="x",
        status=UserStatus.ACTIVE,
    )
    db.set_user_budget(conn, uid, 1.00)
    actor = render.Actor.from_user(db.get_user(conn, uid))
    assert actor.per_user_cap_usd == 1.00

    jid = db.reserve_job(
        conn, idempotency_key="j", username=actor.username, task="t",
        model_id="fal-ai/flux/dev", est_usd=0.90, params_json="{}",
        input_path=None, user_id=uid,
    )
    db.update_job(conn, jid, status="completed")

    with pytest.raises(budget.BudgetExceeded, match="monthly allowance"):
        budget.check(
            conn, settings, username=actor.username, est_usd=0.40,
            user_id=uid, per_user_cap_usd=actor.per_user_cap_usd,
        )


def test_a_pending_account_cannot_become_an_actor(conn):
    uid = db.create_user(conn, email="a@example.test", name="A", password_hash="x")
    with pytest.raises(render.RenderRefused, match="approve"):
        render.Actor.from_user(db.get_user(conn, uid))


# --- Assets -----------------------------------------------------------------


def test_the_same_image_is_stored_once(conn, settings: Settings):
    data = png()
    first = assets.ingest(conn, settings, data=data, filename="massing.png")
    second = assets.ingest(conn, settings, data=data, filename="massing-copy.png")
    assert first["id"] == second["id"]
    assert conn.execute("SELECT COUNT(*) AS n FROM assets").fetchone()["n"] == 1


def test_a_file_that_is_not_an_image_is_refused_locally(conn, settings: Settings):
    with pytest.raises(assets.AssetError, match="not an image"):
        assets.ingest(conn, settings, data=b"%PDF-1.7 not really", filename="plan.png")


def test_an_unsupported_extension_is_refused(conn, settings: Settings):
    with pytest.raises(assets.AssetError, match="not supported"):
        assets.ingest(conn, settings, data=png(), filename="plan.tiff")


def test_an_upload_url_is_reused_across_projects(conn, settings: Settings, monkeypatch):
    """Identical bytes must not be uploaded to fal.ai twice.

    Two projects legitimately hold their own asset row for the same site photo,
    but there is no reason to pay for the transfer again.
    """
    calls: list[Path] = []

    def fake_upload(_settings, path: Path) -> str:
        calls.append(path)
        return f"https://fal.test/{path.name}"

    monkeypatch.setattr(fal_gateway, "upload_input", fake_upload)

    p1 = db.create_project(conn, owner_id=None, name="One")
    p2 = db.create_project(conn, owner_id=None, name="Two")
    data = png()
    a1 = assets.ingest(conn, settings, data=data, filename="site.png", project_id=p1)
    a2 = assets.ingest(conn, settings, data=data, filename="site.png", project_id=p2)
    assert a1["id"] != a2["id"], "each project keeps its own asset row"

    url1 = assets.ensure_fal_url(conn, settings, int(a1["id"]))
    url2 = assets.ensure_fal_url(conn, settings, int(a2["id"]))
    assert url1 == url2
    assert len(calls) == 1, "the second project re-uploaded the same bytes"


def test_continuing_from_a_render_costs_no_upload(conn, settings: Settings, monkeypatch):
    """The whole point of caching output_url: an edit transfers nothing."""

    def explode(*_args, **_kw):
        raise AssertionError("continuing from a render must not upload anything")

    monkeypatch.setattr(fal_gateway, "upload_input", explode)

    jid = db.reserve_job(
        conn, idempotency_key="done", username="alice", task="t",
        model_id="fal-ai/flux-general", est_usd=0.1, params_json="{}", input_path=None,
    )
    db.update_job(
        conn, jid, status="completed", output_url="https://fal.test/render.png"
    )
    asset = assets.from_job_output(conn, settings, jid)
    assert asset["kind"] == AssetKind.RENDER_OUTPUT
    assert assets.ensure_fal_url(conn, settings, int(asset["id"])) == (
        "https://fal.test/render.png"
    )


def test_an_unfinished_render_cannot_be_used_as_a_reference(conn, settings: Settings):
    jid = db.reserve_job(
        conn, idempotency_key="wip", username="alice", task="t",
        model_id="fal-ai/flux-general", est_usd=0.1, params_json="{}", input_path=None,
    )
    db.update_job(conn, jid, status="in_progress")
    with pytest.raises(assets.AssetError, match="Only a finished render"):
        assets.from_job_output(conn, settings, jid)


def test_a_mask_that_does_not_match_its_image_is_refused(conn, settings: Settings):
    """flux-pro/v1/fill requires identical dimensions and fails otherwise."""
    source = assets.ingest(conn, settings, data=png(64, 48), filename="render.png")
    with pytest.raises(assets.AssetError, match="match exactly"):
        assets.ingest_mask(
            conn, settings, data=png(32, 32), source_asset_id=int(source["id"])
        )
    ok = assets.ingest_mask(
        conn, settings, data=png(64, 48, "black"), source_asset_id=int(source["id"])
    )
    assert ok["kind"] == AssetKind.MASK


# --- Multi-reference --------------------------------------------------------


def test_references_are_ordered_with_geometry_first(table: RoutingTable):
    """The prompt says "image 1 is the geometry", so image 1 must be."""
    ordered = fal_gateway.order_references(
        [
            (ImageRole.MATERIAL, "m.png"),
            (ImageRole.STYLE, "s.png"),
            (ImageRole.GEOMETRY, "g.png"),
        ]
    )
    assert [url for _, url in ordered] == ["g.png", "s.png", "m.png"]


def test_a_multi_reference_endpoint_sends_every_image(table: RoutingTable):
    route = table.resolve("variation", "latest")
    spec = table.payload_spec(route.model_id)
    args = fal_gateway.build_arguments(
        route, spec, prompt="clad it in stone", width=1280, height=960, num_images=1,
        references=[
            (ImageRole.STYLE, "https://fal.test/hero.png"),
            (ImageRole.GEOMETRY, "https://fal.test/massing.png"),
        ],
    )
    assert args["image_urls"] == [
        "https://fal.test/massing.png",
        "https://fal.test/hero.png",
    ]


def test_a_single_image_endpoint_refuses_extra_references(table: RoutingTable):
    """Refused, not truncated.

    Silently dropping the material sample an architect attached produces a
    render that is wrong for a reason nobody can see from the output.
    """
    route = table.resolve("variation", "proven")     # kontext, image_url singular
    spec = table.payload_spec(route.model_id)
    with pytest.raises(ValueError, match="accepts 1 reference image"):
        fal_gateway.build_arguments(
            route, spec, prompt="dusk", width=1280, height=960, num_images=1,
            references=[
                (ImageRole.GEOMETRY, "https://fal.test/a.png"),
                (ImageRole.MATERIAL, "https://fal.test/b.png"),
            ],
        )


def test_image_url_and_references_together_are_ambiguous(table: RoutingTable):
    route = table.resolve("variation", "latest")
    spec = table.payload_spec(route.model_id)
    with pytest.raises(ValueError, match="not both"):
        fal_gateway.build_arguments(
            route, spec, prompt="p", width=64, height=64, num_images=1,
            image_url="https://fal.test/a.png",
            references=[(ImageRole.STYLE, "https://fal.test/b.png")],
        )


def test_the_service_refuses_too_many_references_before_uploading(
    conn, settings: Settings, table: RoutingTable, monkeypatch
):
    def explode(*_args, **_kw):
        raise AssertionError("nothing should be uploaded for a refused render")

    monkeypatch.setattr(fal_gateway, "upload_input", explode)

    a = assets.ingest(conn, settings, data=png(), filename="a.png")
    b = assets.ingest(conn, settings, data=png(32, 32), filename="b.png")
    with pytest.raises(render.RenderRefused, match="reference image"):
        render.submit_render(
            conn, settings, table,
            actor=render.Actor(username="alice"),
            task="variation", engine="proven", prompt="dusk",
            references=[
                render.Reference(int(a["id"]), ImageRole.GEOMETRY),
                render.Reference(int(b["id"]), ImageRole.MATERIAL),
            ],
        )


# --- Per-entry control images (the CAD true-depth enabler) ------------------


def test_the_geometry_image_fans_out_across_every_control_by_default(
    table: RoutingTable,
):
    """Unchanged behaviour: with no override, every conditioning entry reads
    the same reference — this is the path every task used before
    `control_images` existed, and it must still be the default.

    All conditioning now lands in controlnet_unions rather than easycontrols
    (2026-08-27), which does not change the fan-out."""
    route = table.resolve("massing_to_render", "proven")
    spec = table.payload_spec(route.model_id)
    args = fal_gateway.build_arguments(
        route, spec, prompt="p", width=1280, height=960, num_images=1,
        references=[(ImageRole.GEOMETRY, "https://fal.test/view.png")],
    )
    assert "easycontrols" not in args
    controls = args["controlnet_unions"][0]["controls"]
    urls = {c["control_mode"]: c["control_image_url"] for c in controls}
    assert urls == {
        "depth": "https://fal.test/view.png",
        "canny": "https://fal.test/view.png",
    }


def test_control_images_overrides_one_entry_by_type(table: RoutingTable):
    """A true depth pass conditions the `depth` entry on its own image while
    `canny` keeps reading the colour view — the enabler CAD depth output is
    built for."""
    route = table.resolve("massing_to_render", "proven")
    spec = table.payload_spec(route.model_id)
    args = fal_gateway.build_arguments(
        route, spec, prompt="p", width=1280, height=960, num_images=1,
        references=[(ImageRole.GEOMETRY, "https://fal.test/view.png")],
        control_images={"depth": "https://fal.test/depth.png"},
    )
    controls = args["controlnet_unions"][0]["controls"]
    urls = {c["control_mode"]: c["control_image_url"] for c in controls}
    assert urls == {
        "depth": "https://fal.test/depth.png",
        "canny": "https://fal.test/view.png",
    }


def test_the_render_service_resolves_control_assets_to_urls(
    conn, settings: Settings, table: RoutingTable, monkeypatch
):
    """`submit_render(control_assets=...)` is the caller-facing form: asset
    ids in, resolved fal URLs out, landing in the same union control the two
    tests above build directly."""
    view = assets.ingest(conn, settings, data=png(), filename="view.png")
    depth = assets.ingest(conn, settings, data=png(64, 48, "black"), filename="depth.png")

    def fake_upload(_settings, path):
        name = Path(path).stem
        return f"https://fal.test/{name}.png"

    monkeypatch.setattr(fal_gateway, "upload_input", fake_upload)

    submitted: list[tuple[str, dict]] = []

    class Handle:
        request_id = "req-1"

    def fake_submit(application, arguments=None, **_kw):
        submitted.append((application, arguments or {}))
        return Handle()

    monkeypatch.setattr(fal_gateway.fal_client, "submit", fake_submit)

    render.submit_render(
        conn, settings, table,
        actor=render.Actor(username="alice"),
        task="massing_to_render", engine="proven", prompt="warm dusk light",
        references=[render.Reference(int(view["id"]), ImageRole.GEOMETRY)],
        control_assets={"depth": int(depth["id"])},
    )

    _application, arguments = submitted[0]
    assert "easycontrols" not in arguments
    controls = arguments["controlnet_unions"][0]["controls"]
    urls = {c["control_mode"]: c["control_image_url"] for c in controls}
    assert urls["depth"] != urls["canny"]
    assert urls["depth"] == assets.ensure_fal_url(conn, settings, int(depth["id"]))
    assert urls["canny"] == assets.ensure_fal_url(conn, settings, int(view["id"]))


def test_the_reference_key_numbers_images_the_way_they_are_sent(conn):
    brief = render.reference_brief(
        conn,
        [
            render.Reference(2, ImageRole.STYLE),
            render.Reference(1, ImageRole.GEOMETRY),
        ],
    )
    assert brief.startswith("Image 1: the geometry to obey")
    assert "Image 2: the approved appearance" in brief


def test_a_single_reference_needs_no_key(conn):
    assert render.reference_brief(conn, [render.Reference(1, ImageRole.GEOMETRY)]) == ""


def test_composed_prompt_keeps_the_architects_words_first(conn):
    composed = render.compose_prompt(
        conn, "warm dusk light",
        [
            render.Reference(1, ImageRole.GEOMETRY),
            render.Reference(2, ImageRole.MATERIAL),
        ],
    )
    assert composed.startswith("warm dusk light")
    assert "Image 2: a material sample" in composed


# --- Lineage ----------------------------------------------------------------


def test_a_branch_keeps_the_root_of_its_lineage(conn):
    root = db.reserve_job(
        conn, idempotency_key="r", username="alice", task="t",
        model_id="fal-ai/flux-general", est_usd=0.1, params_json="{}", input_path=None,
    )
    child = db.reserve_job(
        conn, idempotency_key="c", username="alice", task="t",
        model_id="fal-ai/flux-pro/kontext", est_usd=0.1, params_json="{}",
        input_path=None, parent_job_id=root,
    )
    grandchild = db.reserve_job(
        conn, idempotency_key="g", username="alice", task="t",
        model_id="fal-ai/flux-pro/kontext", est_usd=0.1, params_json="{}",
        input_path=None, parent_job_id=child,
    )
    assert db.get_job(conn, grandchild)["root_job_id"] == root
    assert [r["id"] for r in db.lineage(conn, child)] == [root, child, grandchild]


def test_continuing_from_a_job_that_does_not_exist_is_refused(conn):
    with pytest.raises(ValueError, match="no such job"):
        db.reserve_job(
            conn, idempotency_key="orphan", username="alice", task="t",
            model_id="fal-ai/flux/dev", est_usd=0.1, params_json="{}",
            input_path=None, parent_job_id=9999,
        )
