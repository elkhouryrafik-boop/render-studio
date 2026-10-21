"""Parallel requests must not share a SQLite connection.

The Studio loads four image tiles at once. With one connection handed to every
threadpool thread, that raised `sqlite3.InterfaceError: bad parameter or other
API misuse` and produced spurious 401s and 404s. Here the real `deps.get_conn`
is used (only settings and the routing table are overridden), against a temp
database, with the fake engine.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
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


def _png() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (64, 48), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        fal_key="", office_access_code=CODE, admin_email="boss@example.test",
        budget_monthly_usd=100.0, budget_per_user_usd=50.0, budget_per_job_usd=1.0,
        data_dir=tmp_path, db_path=tmp_path / "api.db", routing_path=ROUTING,
        render_engine="fake", fake_engine_delay_s=0.0,
    )


@pytest.fixture
def client(settings, monkeypatch):
    ratelimit.limiter.clear()
    fal_gateway._fake_client = None
    table = RoutingTable.load(ROUTING)
    # get_conn reads the process settings; point those at the temp database.
    monkeypatch.setattr(deps, "get_settings", lambda: settings)
    app.dependency_overrides[deps.get_config] = lambda: settings
    app.dependency_overrides[deps.get_table] = lambda: table
    yield TestClient(app)
    app.dependency_overrides.clear()
    fal_gateway._fake_client = None
    ratelimit.limiter.clear()


def test_thirty_two_parallel_requests_from_eight_threads_all_succeed(client, settings, caplog):
    r = client.post("/api/auth/signup", json={
        "access_code": CODE, "email": "boss@example.test", "name": "boss",
        "password": "correct-horse-battery"})
    assert r.status_code == 200, r.text
    project = client.post("/api/projects", json={"name": "Marina", "client": "ACME"})
    pid = project.json()["id"] if project.status_code == 200 else None
    upload = client.post(
        "/api/assets", files={"file": ("m.png", _png(), "image/png")},
        data={"project_id": str(pid)} if pid else {},
    )
    assert upload.status_code == 200, upload.text
    sub = client.post("/api/renders", json={
        "task": "massing_to_render", "engine": "proven", "prompt": "warm light",
        "references": [{"asset_id": upload.json()["id"], "role": "geometry"}],
        "project_id": pid, "width": 800, "height": 600, "num_images": 4,
    })
    assert sub.status_code == 200, sub.text
    job_id = int(sub.json()["job_id"])

    other = db.connect(settings.db_path)
    try:
        assert fal_gateway.poll(other, settings, job_id) == "completed"
    finally:
        other.close()

    urls = [
        "/api/renders", f"/api/renders/{job_id}", "/api/auth/me", "/api/budget",
        *[f"/api/renders/{job_id}/image?i={i}" for i in range(4)],
    ]
    cookies = client.cookies
    barrier = threading.Barrier(8)

    def worker(n: int) -> list[tuple[str, int]]:
        local = TestClient(app, cookies=cookies)
        barrier.wait()
        return [(u, local.get(u).status_code)
                for u in (urls[(n * 4 + k) % len(urls)] for k in range(4))]

    with caplog.at_level(logging.DEBUG), ThreadPoolExecutor(max_workers=8) as pool:
        results = [x for chunk in pool.map(worker, range(8)) for x in chunk]

    assert len(results) == 32
    bad = [(u, s) for u, s in results if s != 200]
    assert not bad, bad
    assert "InterfaceError" not in caplog.text


def test_each_request_opens_its_own_connection_and_closes_it_after(client, monkeypatch):
    import sqlite3

    opened: list[sqlite3.Connection] = []
    real_connect = deps.db.connect

    def spy(path):
        conn = real_connect(path)
        opened.append(conn)
        return conn

    monkeypatch.setattr(deps.db, "connect", spy)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health").status_code == 200
    assert len(opened) >= 2
    assert opened[-1] is not opened[-2]
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
