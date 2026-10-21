"""A 422 says what is wrong in one sentence, and the other refusals say so too."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import deps
from api.main import app, plain_sentence
from app import db, ratelimit
from app.config import Settings
from app.routing import RoutingTable

ROUTING = Path(__file__).resolve().parents[1] / "routing.yaml"
CODE = "office-code-123"
PASSWORD = "correct-horse-battery"


def _png() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (64, 48), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _fresh_state():
    ratelimit.limiter.clear()
    yield
    ratelimit.limiter.clear()
    app.dependency_overrides.clear()


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        fal_key="",
        office_access_code=CODE,
        admin_email="boss@example.test",
        budget_monthly_usd=100.0,
        budget_per_user_usd=50.0,
        budget_per_job_usd=1.0,
        data_dir=tmp_path,
        db_path=tmp_path / "api.db",
        routing_path=ROUTING,
        render_engine="fake",
        fake_engine_delay_s=0.0,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return _settings(tmp_path)


@pytest.fixture
def conn(settings: Settings):
    c = db.connect(settings.db_path)
    yield c
    c.close()


def _wire(conn, settings: Settings) -> TestClient:
    table = RoutingTable.load(ROUTING)
    app.dependency_overrides[deps.get_conn] = lambda: conn
    app.dependency_overrides[deps.get_config] = lambda: settings
    app.dependency_overrides[deps.get_table] = lambda: table
    return TestClient(app)


def _signed_in_admin(client) -> None:
    r = client.post(
        "/api/auth/signup",
        json={"access_code": CODE, "email": "boss@example.test", "name": "Boss", "password": PASSWORD},
    )
    assert r.status_code == 200, r.text


def _reference(client, conn) -> int:
    project_id = db.create_project(conn, owner_id=None, name="Marina", client="ACME")
    up = client.post(
        "/api/assets",
        files={"file": ("massing.png", _png(), "image/png")},
        data={"project_id": str(project_id)},
    )
    assert up.status_code == 200, up.text
    return int(up.json()["id"])


# --- the sentence -----------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            {"loc": ["body", "prompt"], "msg": "Field required", "type": "missing"},
            "Prompt is required.",
        ),
        (
            {"loc": ["body", "num_images"], "msg": "Input should be greater than or equal to 1",
             "type": "greater_than_equal"},
            "Number of images must be at least 1.",
        ),
        (
            {"loc": ["body", "width"], "msg": "Input should be less than or equal to 2048",
             "type": "less_than_equal"},
            "Width must be at most 2048.",
        ),
        (
            {"loc": ["body", "references", 0, "role"], "msg": "Something odd", "type": "custom"},
            "Role: Something odd.",
        ),
        (
            {"loc": ["body", "width"], "msg": "Input should be a valid integer, unable to parse string as an integer",
             "type": "int_parsing"},
            "Width must be a whole number.",
        ),
        (
            {"loc": ["body"], "msg": "Field required", "type": "missing"},
            "Request body is required.",
        ),
        (
            {"loc": ["body", "references", 0, "role"],
             "msg": "Input should be 'geometry', 'style', 'material', 'context', 'entourage' or 'mask'",
             "type": "enum"},
            "Role must be one of geometry, style, material, context, entourage, mask.",
        ),
        (
            {"loc": ["body", "view"], "msg": "Input should be 'model' or 'plan'",
             "type": "literal_error"},
            "View must be one of model, plan.",
        ),
    ],
)
def test_each_kind_of_pydantic_error_becomes_one_plain_sentence(error, expected):
    assert plain_sentence(error) == expected


# --- over HTTP --------------------------------------------------------------


def test_a_render_with_no_prompt_is_a_422_with_a_sentence_and_the_full_list(conn, settings):
    client = _wire(conn, settings)
    _signed_in_admin(client)
    asset = _reference(client, conn)

    r = client.post(
        "/api/renders",
        json={
            "task": "massing_to_render",
            "engine": "proven",
            "references": [{"asset_id": asset, "role": "geometry"}],
        },
    )

    assert r.status_code == 422, r.text
    body = r.json()
    assert body["detail"] == "Prompt is required."
    assert isinstance(body["errors"], list) and body["errors"]
    assert body["errors"][0]["loc"] == ["body", "prompt"]


def test_an_out_of_range_field_is_named_in_the_sentence(conn, settings):
    client = _wire(conn, settings)
    _signed_in_admin(client)
    asset = _reference(client, conn)

    r = client.post(
        "/api/renders",
        json={
            "task": "massing_to_render",
            "engine": "proven",
            "prompt": "dusk",
            "references": [{"asset_id": asset, "role": "geometry"}],
            "geometry_hold": 150,
        },
    )

    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "Geometry hold must be at most 100."
    assert r.json()["errors"][0]["loc"] == ["body", "geometry_hold"]


def test_a_budget_refusal_still_answers_with_a_string_detail(tmp_path, conn):
    settings = _settings(tmp_path, kill_switch=True)
    client = _wire(conn, settings)
    _signed_in_admin(client)
    asset = _reference(client, conn)

    r = client.post(
        "/api/renders",
        json={
            "task": "massing_to_render",
            "engine": "proven",
            "prompt": "dusk",
            "references": [{"asset_id": asset, "role": "geometry"}],
        },
    )

    assert r.status_code == 402, r.text
    assert isinstance(r.json()["detail"], str)
    assert "paused" in r.json()["detail"]


def test_a_bad_reference_role_names_the_roles_it_takes(conn, settings):
    client = _wire(conn, settings)
    _signed_in_admin(client)
    asset = _reference(client, conn)

    r = client.post(
        "/api/renders",
        json={
            "task": "massing_to_render",
            "engine": "proven",
            "prompt": "dusk",
            "references": [{"asset_id": asset, "role": "bogus"}],
        },
    )

    assert r.status_code == 422, r.text
    assert r.json()["detail"] == (
        "Role must be one of geometry, style, material, context, entourage, mask."
    )



def test_a_missing_mask_sentence_names_no_model_id(tmp_path):
    """Refusals are read by architects: no `fal-ai/...` ids, no slashes."""
    from io import BytesIO
    from pathlib import Path

    from fastapi.testclient import TestClient
    from PIL import Image

    from api import deps
    from api.main import app
    from app import db, fal_gateway, ratelimit
    from app.config import Settings
    from app.routing import RoutingTable

    routing = Path(__file__).resolve().parents[1] / "routing.yaml"
    settings = Settings(
        fal_key="", office_access_code="office-code-123", admin_email="boss@example.test",
        budget_monthly_usd=100.0, budget_per_user_usd=50.0, budget_per_job_usd=1.0,
        data_dir=tmp_path, db_path=tmp_path / "p.db", routing_path=routing,
        render_engine="fake", fake_engine_delay_s=0.0,
    )
    conn = db.connect(settings.db_path)
    table = RoutingTable.load(routing)
    ratelimit.limiter.clear()
    fal_gateway._fake_client = None
    app.dependency_overrides[deps.get_conn] = lambda: conn
    app.dependency_overrides[deps.get_config] = lambda: settings
    app.dependency_overrides[deps.get_table] = lambda: table
    try:
        client = TestClient(app)
        assert client.post("/api/auth/signup", json={
            "access_code": "office-code-123", "email": "boss@example.test",
            "name": "boss", "password": "correct-horse-battery"}).status_code == 200
        buf = BytesIO()
        Image.new("RGB", (64, 48), "white").save(buf, format="PNG")
        asset = client.post("/api/assets", files={"file": ("a.png", buf.getvalue(), "image/png")}).json()
        r = client.post("/api/renders", json={
            "task": "inpaint", "prompt": "repaint the window",
            "references": [{"asset_id": asset["id"], "role": "geometry"}],
            "width": 64, "height": 48})
        assert r.status_code >= 400, r.text
        detail = r.json()["detail"]
        assert "fal-ai/" not in detail and "/" not in detail, detail
        assert "mask" in detail.lower()
    finally:
        app.dependency_overrides.clear()
        fal_gateway._fake_client = None
        conn.close()
