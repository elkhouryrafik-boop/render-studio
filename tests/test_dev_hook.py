"""POST /api/dev/fail-next: a way to see the failure path with the fake engine."""

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
from app.routing import RoutingTable

ROUTING = Path(__file__).resolve().parents[1] / "routing.yaml"
CODE = "office-code-123"


def _settings(tmp_path: Path, engine: str) -> Settings:
    return Settings(
        fal_key="", office_access_code=CODE, admin_email="boss@example.test",
        budget_monthly_usd=100.0, budget_per_user_usd=50.0, budget_per_job_usd=1.0,
        data_dir=tmp_path, db_path=tmp_path / "d.db", routing_path=ROUTING,
        render_engine=engine, fake_engine_delay_s=0.0,
    )


@pytest.fixture(autouse=True)
def _fresh():
    ratelimit.limiter.clear()
    fal_gateway._fake_client = None
    yield
    app.dependency_overrides.clear()
    fal_gateway._fake_client = None
    ratelimit.limiter.clear()


def _client(tmp_path: Path, engine: str):
    settings = _settings(tmp_path, engine)
    conn = db.connect(settings.db_path)
    table = RoutingTable.load(ROUTING)
    app.dependency_overrides[deps.get_conn] = lambda: conn
    app.dependency_overrides[deps.get_config] = lambda: settings
    app.dependency_overrides[deps.get_table] = lambda: table
    client = TestClient(app)
    r = client.post("/api/auth/signup", json={
        "access_code": CODE, "email": "boss@example.test", "name": "boss",
        "password": "correct-horse-battery"})
    assert r.status_code == 200, r.text
    return client, conn, settings


def test_under_the_real_engine_the_hook_is_not_there(tmp_path):
    client, conn, _ = _client(tmp_path, "fal")
    r = client.post("/api/dev/fail-next")
    assert r.status_code == 404
    assert r.json()["detail"] == "Not available."
    conn.close()


def test_under_the_fake_engine_the_next_job_fails_with_the_plain_error(tmp_path):
    client, conn, settings = _client(tmp_path, "fake")
    buf = BytesIO()
    Image.new("RGB", (64, 48), "white").save(buf, format="PNG")
    asset = client.post("/api/assets", files={"file": ("a.png", buf.getvalue(), "image/png")}).json()

    assert client.post("/api/dev/fail-next").status_code == 204

    sub = client.post("/api/renders", json={
        "task": "massing_to_render", "engine": "proven", "prompt": "warm light",
        "references": [{"asset_id": asset["id"], "role": "geometry"}],
        "width": 800, "height": 600, "num_images": 1})
    assert sub.status_code == 200, sub.text
    job_id = int(sub.json()["job_id"])
    assert fal_gateway.poll(conn, settings, job_id) == "failed"
    job = client.get(f"/api/renders/{job_id}").json()
    assert job["status"] == "failed"
    assert job["error"]
    assert "fal-ai/" not in job["error"]
    conn.close()


def test_the_hook_is_for_administrators_only(tmp_path):
    client, conn, _ = _client(tmp_path, "fake")
    client.post("/api/auth/logout")
    assert client.post("/api/dev/fail-next").status_code == 401
    conn.close()
