"""HTTP-level tests.

Every dependency is overridden onto a temporary database. Nothing here may
touch the office's real `data/render_studio.db`, and `TestClient` is constructed
without entering its context manager so the lifespan — and with it the
background poller — never starts.

fal.ai is monkeypatched at the gateway. A test that reaches the network is a
bug in the test.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import deps
from api.main import app
from app import db, fal_gateway, ratelimit
from app.config import Settings
from app.models import AssetKind, UserStatus
from app.routing import RoutingTable

ROUTING = Path(__file__).resolve().parents[1] / "routing.yaml"
CODE = "office-code-123"
PASSWORD = "correct-horse-battery"


def png(width: int = 64, height: int = 48) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def clean_limiter():
    ratelimit.limiter.clear()
    yield
    ratelimit.limiter.clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        fal_key="test-key-not-real",
        office_access_code=CODE,
        admin_email="boss@example.test",
        budget_monthly_usd=100.0,
        budget_per_user_usd=50.0,
        budget_per_job_usd=1.0,
        data_dir=tmp_path,
        db_path=tmp_path / "api.db",
        routing_path=ROUTING,
    )


@pytest.fixture
def conn(settings: Settings):
    c = db.connect(settings.db_path)
    yield c
    c.close()


@pytest.fixture
def client(conn, settings: Settings):
    table = RoutingTable.load(ROUTING)
    app.dependency_overrides[deps.get_conn] = lambda: conn
    app.dependency_overrides[deps.get_config] = lambda: settings
    app.dependency_overrides[deps.get_table] = lambda: table
    # No `with` — the lifespan and its background poller must not start.
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def no_network(monkeypatch):
    """fal.ai, replaced. Records what would have been submitted."""
    submitted: list[tuple[str, dict]] = []

    class Handle:
        request_id = "req-test-1"

    def fake_submit(application, arguments=None, **_kw):
        submitted.append((application, arguments or {}))
        return Handle()

    monkeypatch.setattr(fal_gateway.fal_client, "submit", fake_submit)
    monkeypatch.setattr(
        fal_gateway, "upload_input", lambda _s, path: f"https://fal.test/{Path(path).name}"
    )
    return submitted


# --- Helpers ----------------------------------------------------------------


def _signup(client, email="a@example.test", name="A"):
    return client.post(
        "/api/auth/signup",
        json={"access_code": CODE, "email": email, "name": name, "password": PASSWORD},
    )


def _admin_client(client, conn):
    """Sign up the bootstrap admin and return the signed-in client."""
    r = _signup(client, email="boss@example.test", name="Boss")
    assert r.status_code == 200, r.text
    assert r.json()["needs_approval"] is False
    return client


def _approved_user(client, conn, email="a@example.test"):
    _signup(client, email=email)
    row = db.get_user_by_email(conn, email)
    db.set_user_status(conn, int(row["id"]), UserStatus.ACTIVE)
    return int(row["id"])


# --- Health and the front-end ----------------------------------------------


def test_health_is_open_and_reports_the_schema_version(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["schema_version"] >= 5
    assert body["tasks"] > 0


def test_no_api_schema_is_published(client):
    """An internal tool does not need to hand the LAN a schema browser.

    Asserted on the *content*, not the status code: once web/dist is built the
    SPA catch-all answers unknown paths with index.html, so /docs returns 200
    either way. What matters is that neither route ever serves Swagger or an
    OpenAPI document.
    """
    for path in ("/docs", "/redoc", "/openapi.json"):
        body = client.get(path).text.lower()
        assert "swagger" not in body, f"{path} served a schema browser"
        assert '"openapi"' not in body, f"{path} served an OpenAPI document"


# --- Everything needs a session --------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/tasks"),
        ("get", "/api/renders"),
        ("get", "/api/budget"),
        ("get", "/api/auth/me"),
        ("get", "/api/admin/users"),
        ("post", "/api/renders"),
    ],
)
def test_signed_out_requests_are_refused(client, method, path):
    kwargs = {"json": {"task": "massing_to_render", "prompt": "x"}} if method == "post" else {}
    r = client.request(method.upper(), path, **kwargs)
    assert r.status_code == 401, f"{path} answered {r.status_code} to a stranger"


def test_a_forged_cookie_does_not_sign_anyone_in(client, conn):
    _approved_user(client, conn)
    client.cookies.set(deps.SESSION_COOKIE, "not-a-real-token")
    assert client.get("/api/auth/me").status_code == 401


# --- Signup and approval ----------------------------------------------------


def test_signup_needs_the_access_code(client):
    r = client.post(
        "/api/auth/signup",
        json={"access_code": "wrong", "email": "a@example.test", "name": "A",
              "password": PASSWORD},
    )
    assert r.status_code == 403


def test_a_new_account_waits_and_cannot_sign_in(client, conn):
    r = _signup(client)
    assert r.status_code == 200
    assert r.json()["needs_approval"] is True
    assert deps.SESSION_COOKIE not in r.cookies

    login = client.post(
        "/api/auth/login", json={"email": "a@example.test", "password": PASSWORD}
    )
    assert login.status_code == 403
    assert "approve" in login.json()["detail"]


def test_the_bootstrap_admin_is_signed_in_immediately(client, conn):
    r = _signup(client, email="boss@example.test", name="Boss")
    assert r.json()["needs_approval"] is False
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["is_admin"] is True


def test_an_admin_approves_and_the_person_can_then_work(client, conn):
    _admin_client(client, conn)
    _signup(client, email="new@example.test", name="New")

    pending = client.get("/api/admin/users/pending").json()
    assert [p["email"] for p in pending] == ["new@example.test"]

    approved = client.post(f"/api/admin/users/{pending[0]['id']}/approve")
    assert approved.status_code == 200
    assert approved.json()["status"] == "active"

    client.post("/api/auth/logout")
    login = client.post(
        "/api/auth/login", json={"email": "new@example.test", "password": PASSWORD}
    )
    assert login.status_code == 200
    assert client.get("/api/tasks").status_code == 200


def test_an_ordinary_user_cannot_reach_the_admin_routes(client, conn):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    assert client.get("/api/admin/users").status_code == 403
    assert client.get("/api/admin/spend").status_code == 403


def test_suspending_someone_takes_effect_on_their_next_request(client, conn, settings):
    """Not whenever their cookie happens to expire, up to thirty days later."""
    victim_id = _approved_user(client, conn, email="v@example.test")
    victim = TestClient(app)
    r = victim.post("/api/auth/login", json={"email": "v@example.test", "password": PASSWORD})
    assert r.status_code == 200
    assert victim.get("/api/auth/me").status_code == 200

    _admin_client(client, conn)
    assert client.post(f"/api/admin/users/{victim_id}/suspend").status_code == 200

    assert victim.get("/api/auth/me").status_code == 401


def test_logging_out_clears_the_session(client, conn):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    assert client.get("/api/auth/me").status_code == 200
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401


def test_repeated_bad_logins_are_rate_limited(client, conn):
    _approved_user(client, conn)
    for _ in range(ratelimit.LOGIN.attempts):
        client.post("/api/auth/login", json={"email": "a@example.test", "password": "no"})
    r = client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    assert r.status_code == 429
    assert r.headers.get("Retry-After")


# --- Tasks ------------------------------------------------------------------


def test_tasks_carry_the_sentence_that_explains_each_engine(client, conn):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    tasks = client.get("/api/tasks").json()
    assert tasks
    for task in tasks:
        assert task["engines"]
        for engine in task["engines"]:
            assert engine["max_images"] >= 1
            assert len(engine["note"]) > 30, f"{task['key']}/{engine['key']}"


def test_the_engine_that_needs_a_mask_says_so(client, conn):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    tasks = {t["key"]: t for t in client.get("/api/tasks").json()}
    fill = next(
        e for e in tasks["inpaint"]["engines"] if e["model_id"] == "fal-ai/flux-pro/v1/fill"
    )
    assert fill["needs_mask"] is True



# --- Experimental tasks -------------------------------------------------------


def _experimental_submit(client, conn, task):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    return client.post(
        "/api/renders",
        json={"task": task, "prompt": "people on the plaza", "references": [], "num_images": 1},
    )


def test_experimental_tasks_hidden_and_refused_without_the_flag(
    client, conn, no_network, monkeypatch
):
    monkeypatch.delenv("ENABLE_EXPERIMENTAL_TASKS", raising=False)
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    keys = {t["key"] for t in client.get("/api/tasks").json()}
    assert not keys & {"entourage", "skp_material_render"}
    r = _experimental_submit(client, conn, "entourage")
    assert r.status_code >= 400
    assert "ENABLE_EXPERIMENTAL_TASKS" in r.text
    assert no_network == []


def test_experimental_tasks_listed_and_accepted_with_the_flag(client, conn, monkeypatch):
    monkeypatch.setenv("ENABLE_EXPERIMENTAL_TASKS", "1")
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    keys = {t["key"] for t in client.get("/api/tasks").json()}
    assert {"entourage", "skp_material_render"} <= keys


# --- Rendering --------------------------------------------------------------


def test_a_render_goes_through_and_records_its_inputs(client, conn, no_network):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})

    project_id = db.create_project(conn, owner_id=None, name="Marina", client="ACME")

    upload = client.post(
        "/api/assets",
        files={"file": ("massing.png", png(), "image/png")},
        data={"project_id": str(project_id)},
    )
    assert upload.status_code == 200, upload.text
    asset_id = upload.json()["id"]

    r = client.post(
        "/api/renders",
        json={
            "task": "massing_to_render",
            "engine": "proven",
            "prompt": "warm late afternoon light, board-formed concrete",
            "references": [{"asset_id": asset_id, "role": "geometry"}],
            "project_id": project_id,
            "num_images": 1,
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    application, arguments = no_network[0]
    assert application == "fal-ai/flux-general"
    # All conditioning rides controlnet_unions since 2026-08-27 — this endpoint
    # cannot carry easycontrols in the same request, see
    # tests/test_flux_general_conditioning.py.
    assert "controlnet_unions" in arguments
    assert "easycontrols" not in arguments

    inputs = client.get(f"/api/renders/{job_id}/inputs").json()
    assert [i["id"] for i in inputs] == [asset_id]


def test_an_identical_render_is_not_paid_for_twice(client, conn, no_network):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    asset_id = client.post(
        "/api/assets", files={"file": ("m.png", png(), "image/png")}
    ).json()["id"]

    body = {
        "task": "massing_to_render",
        "engine": "proven",
        "prompt": "dusk",
        "references": [{"asset_id": asset_id, "role": "geometry"}],
    }
    first = client.post("/api/renders", json=body).json()
    second = client.post("/api/renders", json=body).json()
    assert second["duplicate"] is True
    assert second["job_id"] == first["job_id"]
    assert len(no_network) == 1, "the duplicate reached fal.ai"


def test_too_many_references_are_refused_with_a_reason(client, conn, no_network):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    a = client.post("/api/assets", files={"file": ("a.png", png(), "image/png")}).json()
    b = client.post(
        "/api/assets", files={"file": ("b.png", png(32, 32), "image/png")}
    ).json()

    r = client.post(
        "/api/renders",
        json={
            "task": "variation",
            "engine": "proven",  # kontext — one image only
            "prompt": "stone",
            "references": [
                {"asset_id": a["id"], "role": "geometry"},
                {"asset_id": b["id"], "role": "material"},
            ],
        },
    )
    assert r.status_code == 400
    assert "reference image" in r.json()["detail"]
    assert not no_network


def test_a_render_over_the_per_job_cap_is_refused_before_submitting(
    client, conn, settings, no_network
):
    settings.budget_per_job_usd = 0.001
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    asset = client.post("/api/assets", files={"file": ("a.png", png(), "image/png")}).json()

    r = client.post(
        "/api/renders",
        json={
            "task": "massing_to_render", "engine": "proven", "prompt": "dusk",
            "references": [{"asset_id": asset["id"], "role": "geometry"}],
        },
    )
    assert r.status_code == 402
    assert not no_network, "money was committed past the cap"


def test_the_kill_switch_stops_everyone(client, conn, settings, no_network):
    settings.kill_switch = True
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    asset = client.post("/api/assets", files={"file": ("a.png", png(), "image/png")}).json()
    r = client.post(
        "/api/renders",
        json={
            "task": "massing_to_render", "engine": "proven", "prompt": "dusk",
            "references": [{"asset_id": asset["id"], "role": "geometry"}],
        },
    )
    assert r.status_code == 402
    assert not no_network


def test_continuing_from_a_render_records_the_lineage(client, conn, no_network):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    asset = client.post("/api/assets", files={"file": ("a.png", png(), "image/png")}).json()
    first = client.post(
        "/api/renders",
        json={
            "task": "massing_to_render", "engine": "proven", "prompt": "noon",
            "references": [{"asset_id": asset["id"], "role": "geometry"}],
        },
    ).json()

    # Pretend it finished, with fal.ai hosting the output.
    db.update_job(
        conn, first["job_id"], status="completed",
        output_url="https://fal.test/out.png",
    )

    child = client.post(
        f"/api/renders/{first['job_id']}/continue",
        json={"task": "variation", "engine": "proven", "prompt": "now at dusk"},
    )
    assert child.status_code == 200, child.text
    child_id = child.json()["job_id"]

    lineage = client.get(f"/api/renders/{child_id}/lineage").json()
    assert [j["id"] for j in lineage] == [first["job_id"], child_id]
    assert lineage[1]["parent_job_id"] == first["job_id"]

    # The parent's hosted URL was reused — the edit uploaded nothing.
    _application, arguments = no_network[-1]
    assert arguments["image_url"] == "https://fal.test/out.png"


def test_an_unfinished_render_cannot_be_continued(client, conn, no_network):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    asset = client.post("/api/assets", files={"file": ("a.png", png(), "image/png")}).json()
    first = client.post(
        "/api/renders",
        json={
            "task": "massing_to_render", "engine": "proven", "prompt": "noon",
            "references": [{"asset_id": asset["id"], "role": "geometry"}],
        },
    ).json()
    r = client.post(
        f"/api/renders/{first['job_id']}/continue",
        json={"task": "variation", "prompt": "dusk"},
    )
    assert r.status_code == 400
    assert "finished render" in r.json()["detail"]


# --- Uploads ----------------------------------------------------------------


def test_a_file_that_is_not_an_image_is_refused(client, conn):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    r = client.post(
        "/api/assets", files={"file": ("plan.png", b"%PDF-1.7 nope", "image/png")}
    )
    assert r.status_code == 400
    assert "not an image" in r.json()["detail"]


def test_a_mask_must_match_its_source(client, conn):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    source = client.post(
        "/api/assets", files={"file": ("r.png", png(64, 48), "image/png")}
    ).json()

    bad = client.post(
        "/api/assets/mask",
        files={"file": ("m.png", png(32, 32), "image/png")},
        data={"source_asset_id": str(source["id"])},
    )
    assert bad.status_code == 400
    assert "match exactly" in bad.json()["detail"]

    good = client.post(
        "/api/assets/mask",
        files={"file": ("m.png", png(64, 48), "image/png")},
        data={"source_asset_id": str(source["id"])},
    )
    assert good.status_code == 200


def test_uploading_the_same_image_twice_reuses_the_asset(client, conn):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    data = png()
    a = client.post("/api/assets", files={"file": ("x.png", data, "image/png")}).json()
    b = client.post("/api/assets", files={"file": ("y.png", data, "image/png")}).json()
    assert a["id"] == b["id"]


# --- Money ------------------------------------------------------------------


def test_the_budget_route_reports_the_caps_that_apply(client, conn, settings):
    uid = _approved_user(client, conn)
    db.set_user_budget(conn, uid, 3.0)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    body = client.get("/api/budget").json()
    assert body["per_user_cap_usd"] == 3.0
    assert body["monthly_cap_usd"] == settings.budget_monthly_usd


def test_an_estimate_never_guesses_at_an_unknown_task(client, conn):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    r = client.post("/api/estimate", json={"task": "make_it_pretty"})
    assert r.status_code == 400


# --- Hardening --------------------------------------------------------------


def _user_client(client, conn, email="a@example.test"):
    _approved_user(client, conn, email=email)
    client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    return client


def test_a_render_for_a_project_that_does_not_exist_is_a_400(client, conn, no_network):
    """A dangling project_id used to hit the FK, be swallowed as a duplicate
    idempotency key, and blow up several frames later as a 500."""
    _user_client(client, conn)
    asset = client.post("/api/assets", files={"file": ("a.png", png(), "image/png")}).json()
    r = client.post(
        "/api/renders",
        json={
            "task": "massing_to_render", "engine": "proven", "prompt": "dusk",
            "references": [{"asset_id": asset["id"], "role": "geometry"}],
            "project_id": 424242,
        },
    )
    assert r.status_code == 400, r.text
    assert "424242" in r.json()["detail"]
    assert not no_network, "a refused render must not have been submitted"


def test_an_upload_for_a_project_that_does_not_exist_is_a_400(client, conn):
    _user_client(client, conn)
    r = client.post(
        "/api/assets",
        files={"file": ("a.png", png(), "image/png")},
        data={"project_id": "424242"},
    )
    assert r.status_code == 400, r.text
    assert "424242" in r.json()["detail"]


def test_unknown_api_paths_are_json_404_not_the_spa(client, conn):
    """The SPA catch-all answered `/api/does-not-exist` with 200 and
    index.html, and the front-end JSON-parsed that HTML into `null`."""
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    assert "detail" in r.json()

    posted = client.post("/api/also/not/here", json={})
    assert posted.status_code == 404


def test_health_reports_a_missing_key_as_degraded(client, conn, settings):
    """It was typed Literal["ok"], so it could not report anything else."""
    ok = client.get("/api/health")
    assert ok.status_code == 200
    assert ok.json()["status"] == "ok"
    assert ok.json()["checks"]["database"] == "ok"

    settings.fal_key = type(settings.fal_key)("")
    bad = client.get("/api/health")
    assert bad.status_code == 503
    body = bad.json()
    assert body["status"] == "degraded"
    assert body["checks"]["fal_key"] == "missing"
    # It says the key is absent and nothing whatever about the key itself.
    assert "test-key-not-real" not in bad.text


def test_health_notices_a_dead_poller(client, conn, monkeypatch):
    """A poller that has died leaves every render stuck at 'queued' with
    nothing anywhere to say why."""
    from api import main

    class _Dead:
        @staticmethod
        def done() -> bool:
            return True

    monkeypatch.setattr(main, "_poller_task", _Dead())
    r = client.get("/api/health")
    assert r.status_code == 503
    assert r.json()["checks"]["poller"] == "stopped"


def test_logout_actually_clears_the_cookie(client, conn):
    """Measured before the fix: 204 with no `set-cookie` at all, because the
    handler returned a NEW Response and FastAPI dropped the injected one."""
    _user_client(client, conn)
    r = client.post("/api/auth/logout")
    assert r.status_code == 204
    assert "set-cookie" in {k.lower() for k in r.headers}
    assert deps.SESSION_COOKIE in r.headers["set-cookie"]


def test_changing_a_password_clears_the_cookie_too(client, conn):
    _user_client(client, conn)
    r = client.post(
        "/api/auth/password",
        json={"current_password": PASSWORD, "new_password": "another-long-one"},
    )
    assert r.status_code == 204
    assert deps.SESSION_COOKIE in r.headers.get("set-cookie", "")


def test_guessing_the_current_password_is_rate_limited(client, conn):
    """A borrowed session was otherwise an oracle for the password it already
    had access to."""
    _user_client(client, conn)
    for _ in range(ratelimit.PASSWORD_CHANGE.attempts):
        client.post(
            "/api/auth/password",
            json={"current_password": "wrong", "new_password": "another-long-one"},
        )
    r = client.post(
        "/api/auth/password",
        json={"current_password": "wrong", "new_password": "another-long-one"},
    )
    assert r.status_code == 429
    assert r.headers.get("Retry-After")


def test_the_render_list_limit_has_a_floor(client, conn):
    """`?limit=-1` became SQL `LIMIT -1`, which SQLite reads as no limit."""
    _user_client(client, conn)
    assert client.get("/api/renders?limit=-1").status_code == 422
    assert client.get("/api/renders?limit=0").status_code == 422
    assert client.get("/api/renders?limit=1").status_code == 200


def test_the_image_route_will_not_serve_a_cad_source_file(client, conn, tmp_path):
    """A CAD_UPLOAD's local_path points at the raw .dxf, so this endpoint was
    handing out the office's CAD source files."""
    _user_client(client, conn)
    asset_id = db.create_asset(
        conn, sha256="deadbeef", local_path=str(tmp_path / "plan.dxf"),
        kind=AssetKind.CAD_UPLOAD,
    )
    (tmp_path / "plan.dxf").write_bytes(b"0\nSECTION\n2\nHEADER\n")
    assert client.get(f"/api/assets/{asset_id}/image").status_code == 404


def test_has_image_tells_the_truth_about_what_can_be_served(client, conn, tmp_path):
    """`output_path OR output_url` claimed more than the image route honours:
    a completed job whose download failed kept its fal.ai URL, and the gallery
    showed a thumbnail slot that 404s."""
    _user_client(client, conn)
    job_id = db.reserve_job(
        conn, idempotency_key="no-file", username="a@example.test", task="t",
        model_id="fal-ai/flux/dev", est_usd=0.01, params_json="{}", input_path=None,
    )
    db.update_job(
        conn, job_id, status="completed", output_url="https://fal.test/gone.png",
    )
    listed = {j["id"]: j for j in client.get("/api/renders").json()}
    assert listed[job_id]["output_url"] is None
    assert client.get(f"/api/renders/{job_id}/image").status_code == 404

    real = tmp_path / "out.png"
    real.write_bytes(png())
    db.update_job(conn, job_id, output_path=str(real))
    db.add_job_output(conn, job_id=job_id, idx=0, path=str(real), url=None, seed=None)
    listed = {j["id"]: j for j in client.get("/api/renders").json()}
    assert listed[job_id]["output_url"] == f"/api/renders/{job_id}/image"
    assert client.get(f"/api/renders/{job_id}/image").status_code == 200


def test_submitting_renders_is_rate_limited(client, conn, no_network):
    """The budget caps how much can be spent, not how fast."""
    _user_client(client, conn)
    asset = client.post("/api/assets", files={"file": ("a.png", png(), "image/png")}).json()

    def submit(n: int):
        return client.post(
            "/api/renders",
            json={
                "task": "massing_to_render", "engine": "proven", "prompt": f"dusk {n}",
                "references": [{"asset_id": asset["id"], "role": "geometry"}],
            },
        )

    for n in range(ratelimit.RENDER_SUBMIT.attempts):
        assert submit(n).status_code == 200
    blocked = submit(999)
    assert blocked.status_code == 429
    assert len(no_network) == ratelimit.RENDER_SUBMIT.attempts


def test_refresh_does_not_poll_twice_in_a_row(client, conn, monkeypatch):
    """Two threads polling the same job both write the same output file. The
    manual refresh and the background poller share one guard."""
    from app.services import render as render_service

    calls: list[int] = []
    monkeypatch.setattr(
        render_service.fal_gateway, "poll_all", lambda *_a, **_k: calls.append(1) or {}
    )
    monkeypatch.setattr(render_service, "_last_poll_at", 0.0)
    _user_client(client, conn)

    assert client.post("/api/renders/refresh").status_code == 200
    assert client.post("/api/renders/refresh").status_code == 200
    assert len(calls) == 1, "the second refresh inside the window polled again"


def test_every_response_carries_a_request_id(client):
    r = client.get("/api/health")
    assert r.headers.get("X-Request-ID")
    mine = client.get("/api/health", headers={"X-Request-ID": "from-the-log"})
    assert mine.headers["X-Request-ID"] == "from-the-log"


def test_an_unhandled_error_answers_json_with_its_request_id(client, conn, monkeypatch):
    """Starlette's default is a plain-text body, which the front-end
    JSON-parsed into `null`, and a traceback on stdout with nothing tying the
    two together."""
    def explode() -> None:
        raise RuntimeError("something the code did not expect")

    app.dependency_overrides[deps.get_table] = explode
    try:
        # A client that lets the handler run rather than re-raising into the
        # test, which is the only way to see what a browser would get.
        loud = TestClient(app, raise_server_exceptions=False)
        r = loud.get("/api/health", headers={"X-Request-ID": "traceme"})
    finally:
        app.dependency_overrides.pop(deps.get_table, None)

    assert r.status_code == 500
    body = r.json()
    # The id in the body, the header and the logged line have to be the same
    # one, or it correlates nothing. The context variable alone is reset before
    # the handler runs and this read back "-".
    assert body["request_id"] == "traceme"
    assert r.headers["X-Request-ID"] == "traceme"
    assert "something the code did not expect" not in r.text, "internals must not leak"


def test_uploading_does_not_run_on_the_event_loop(client, conn, monkeypatch):
    """`ingest` is a SHA-256, a PIL probe and a write of up to 40 MB, all
    synchronous. Run inline in an async handler it stalls every other request
    and the background poller for the duration, so it belongs in a worker
    thread. Asserted on the thread it actually ran on, because "the handler is
    async" is exactly the shape that made this a bug.
    """
    import asyncio
    import threading

    from app.services import assets as asset_service

    loop_thread = threading.current_thread().name
    ran_on: list[str] = []
    real_ingest = asset_service.ingest

    def watched(*args, **kwargs):
        ran_on.append(threading.current_thread().name)
        try:
            asyncio.get_running_loop()
            ran_on.append("EVENT LOOP")
        except RuntimeError:
            pass  # no running loop on this thread — which is the point
        return real_ingest(*args, **kwargs)

    monkeypatch.setattr(asset_service, "ingest", watched)
    _user_client(client, conn)

    r = client.post("/api/assets", files={"file": ("a.png", png(), "image/png")})
    assert r.status_code == 200, r.text
    assert ran_on, "ingest was never called"
    assert "EVENT LOOP" not in ran_on
    assert ran_on[0] != loop_thread


def test_cad_upload_does_not_run_on_the_event_loop(client, conn, monkeypatch):
    """Same shape, 100 MB instead of 40."""
    import asyncio

    from app.services import assets as asset_service

    ran_off_loop: list[bool] = []
    real = asset_service.ingest_cad

    def watched(*args, **kwargs):
        try:
            asyncio.get_running_loop()
            ran_off_loop.append(False)
        except RuntimeError:
            ran_off_loop.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(asset_service, "ingest_cad", watched)
    _user_client(client, conn)

    r = client.post(
        "/api/cad",
        files={"file": ("plan.dxf", b"  0\nSECTION\n  2\nHEADER\n", "application/dxf")},
    )
    assert r.status_code == 200, r.text
    assert ran_off_loop == [True]


def test_an_incomplete_web_build_does_not_stop_the_app_booting(tmp_path):
    """`StaticFiles(directory=...)` raises from its constructor, at import, so
    a `web/dist` left without `assets/` by an interrupted build took the whole
    API down with it rather than degrading to the 503 that says what to do."""
    from api.main import bundle_ready

    assert bundle_ready(tmp_path) is False

    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
    assert bundle_ready(tmp_path) is False, "index.html alone is not a usable build"

    (tmp_path / "assets").mkdir()
    assert bundle_ready(tmp_path) is True


def test_the_hashed_bundle_is_cached_and_index_is_not(client):
    """Everything under /assets carries a content hash in its filename, so its
    bytes never change; index.html names those files and must never be stale."""
    from api.main import WEB_DIST, bundle_ready

    if not bundle_ready(WEB_DIST):
        pytest.skip("web/dist is not built")

    hashed = next(p for p in (WEB_DIST / "assets").iterdir() if p.is_file())
    r = client.get(f"/assets/{hashed.name}")
    assert r.status_code == 200
    assert "immutable" in r.headers.get("cache-control", "")

    index = client.get("/")
    assert index.headers.get("cache-control") == "no-cache"


# --- The approval lookup ----------------------------------------------------


def test_the_approval_lookup_answers_with_a_status_and_nothing_else(client, conn):
    _signup(client, email="waiting@example.test", name="Wait Ing")
    r = client.get("/api/auth/approval", params={"email": "waiting@example.test"})
    assert r.status_code == 200
    assert r.json() == {"status": "pending"}


def test_the_approval_lookup_reports_an_approved_account_as_active(client, conn):
    _approved_user(client, conn, email="yes@example.test")
    r = client.get("/api/auth/approval", params={"email": "yes@example.test"})
    assert r.json() == {"status": "active"}


def test_the_approval_lookup_reports_suspended_and_unknown_identically(client, conn):
    _approved_user(client, conn, email="sus@example.test")
    db.set_user_status(conn, int(db.get_user_by_email(conn, "sus@example.test")["id"]),
                       UserStatus.SUSPENDED)
    suspended = client.get("/api/auth/approval", params={"email": "sus@example.test"})
    unknown = client.get("/api/auth/approval", params={"email": "nobody@example.test"})
    assert suspended.json() == unknown.json() == {"status": "unknown"}


def test_unknown_lookups_are_rate_limited_after_the_access_code_allowance(client, conn):
    """Same allowance as /access-code: that many failed guesses, then a wall."""
    assert ratelimit.APPROVAL_LOOKUP.attempts == ratelimit.ACCESS_CODE.attempts
    for n in range(ratelimit.ACCESS_CODE.attempts):
        r = client.get("/api/auth/approval", params={"email": f"guess{n}@example.test"})
        assert r.status_code == 200
    r = client.get("/api/auth/approval", params={"email": "one-more@example.test"})
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_polling_a_pending_account_is_free_up_to_the_overall_allowance(client, conn):
    """A waiting newcomer polls every 15 s: about 60 lookups in 15 minutes, well inside 120."""
    _signup(client, email="poll@example.test")
    for _ in range(ratelimit.APPROVAL_LOOKUP_ANY.attempts):
        r = client.get("/api/auth/approval", params={"email": "poll@example.test"})
        assert r.status_code == 200
    # The overall bucket counts every lookup, pending ones included, so a script
    # that polls a known address without end still meets a wall.
    r = client.get("/api/auth/approval", params={"email": "poll@example.test"})
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_the_controls_report_the_kill_switch_and_the_cap(client, conn, settings):
    _admin_client(client, conn)
    r = client.get("/api/admin/controls")
    assert r.status_code == 200, r.text
    assert r.json() == {"kill_switch": False, "monthly_cap_usd": settings.budget_monthly_usd}


def test_an_admin_can_stop_and_resume_spending_without_a_restart(client, conn):
    _admin_client(client, conn)
    r = client.put("/api/admin/controls/kill-switch", json={"on": True})
    assert r.status_code == 200, r.text
    assert r.json()["kill_switch"] is True
    assert client.get("/api/admin/controls").json()["kill_switch"] is True

    r = client.put("/api/admin/controls/kill-switch", json={"on": False})
    assert r.json()["kill_switch"] is False
    assert client.get("/api/admin/controls").json()["kill_switch"] is False


def test_an_ordinary_user_cannot_reach_the_controls(client, conn):
    _approved_user(client, conn)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    assert client.get("/api/admin/controls").status_code == 403
    assert client.put("/api/admin/controls/kill-switch", json={"on": True}).status_code == 403

