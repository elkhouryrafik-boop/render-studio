"""The administrator's Activity view: every user's renders, paged and totalled.

Same setup as test_api.py: a temporary database, no lifespan, no network. Jobs
are written with the real db helpers rather than by running the engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import deps
from api.main import app
from app import db, ratelimit
from app.config import Settings
from app.models import UserStatus
from app.routing import RoutingTable

ROUTING = Path(__file__).resolve().parents[1] / "routing.yaml"
CODE = "office-code-123"
PASSWORD = "correct-horse-battery"


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
        db_path=tmp_path / "activity.db",
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
    yield TestClient(app)
    app.dependency_overrides.clear()


def _signup(client, email, name):
    return client.post(
        "/api/auth/signup",
        json={"access_code": CODE, "email": email, "name": name, "password": PASSWORD},
    )


def _admin(client):
    assert _signup(client, "boss@example.test", "Boss").status_code == 200
    return client


def _make_user(client, conn, email, name) -> int:
    _signup(client, email, name)
    row = db.get_user_by_email(conn, email)
    db.set_user_status(conn, int(row["id"]), UserStatus.ACTIVE)
    return int(row["id"])


_counter = 0


def _job(conn, user_id, email, *, est=0.1, status="completed", actual=None, outputs=0):
    global _counter
    _counter += 1
    params = {"line": "a warm lobby", "view": "lobby", "style": "warm", "geometry_hold": 50}
    job_id = db.reserve_job(
        conn,
        idempotency_key=f"act-{_counter}",
        username=email,
        task="img2img",
        model_id="m",
        est_usd=est,
        params_json=json.dumps(params),
        input_path=None,
        user_id=user_id,
    )
    assert job_id is not None
    conn.execute(
        "UPDATE jobs SET status = ?, actual_usd = ? WHERE id = ?", (status, actual, job_id)
    )
    for i in range(outputs):
        db.add_job_output(conn, job_id=job_id, idx=i, path=f"x{job_id}_{i}.png", url=None, seed=i)
    return job_id


def test_an_administrator_gets_every_users_jobs_newest_first_with_names_and_outputs(client, conn):
    ana = _make_user(client, conn, "ana@example.test", "Ana")
    ben = _make_user(client, conn, "ben@example.test", "Ben")
    first = _job(conn, ana, "ana@example.test", outputs=2)
    second = _job(conn, ben, "ben@example.test")
    _admin(client)

    r = client.get("/api/admin/activity")
    assert r.status_code == 200, r.text
    body = r.json()
    assert [j["id"] for j in body["jobs"]] == [second, first]
    assert body["jobs"][0]["user_name"] == "Ben"
    assert body["jobs"][1]["user_name"] == "Ana"
    assert body["jobs"][1]["user_id"] == ana
    assert [o["index"] for o in body["jobs"][1]["outputs"]] == [0, 1]
    assert body["jobs"][1]["outputs"][0]["url"] == f"/api/renders/{first}/image?i=0"
    assert body["jobs"][1]["line"] == "a warm lobby"
    assert body["next_before_id"] is None


def test_an_ordinary_user_gets_403_on_the_activity_route(client, conn):
    _make_user(client, conn, "a@example.test", "A")
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    assert client.get("/api/admin/activity").status_code == 403


def test_a_signed_out_request_gets_401_on_the_activity_route(client):
    assert client.get("/api/admin/activity").status_code == 401


def test_before_id_pages_without_overlap_or_gap(client, conn):
    ana = _make_user(client, conn, "ana@example.test", "Ana")
    ids = [_job(conn, ana, "ana@example.test") for _ in range(5)]
    _admin(client)

    p1 = client.get("/api/admin/activity", params={"limit": 2}).json()
    assert [j["id"] for j in p1["jobs"]] == [ids[4], ids[3]]
    assert p1["next_before_id"] == ids[3]

    p2 = client.get(
        "/api/admin/activity", params={"limit": 2, "before_id": p1["next_before_id"]}
    ).json()
    assert [j["id"] for j in p2["jobs"]] == [ids[2], ids[1]]
    assert p2["next_before_id"] == ids[1]

    p3 = client.get(
        "/api/admin/activity", params={"limit": 2, "before_id": p2["next_before_id"]}
    ).json()
    assert [j["id"] for j in p3["jobs"]] == [ids[0]]
    assert p3["next_before_id"] is None


def test_user_id_filters_to_that_users_jobs_only(client, conn):
    ana = _make_user(client, conn, "ana@example.test", "Ana")
    ben = _make_user(client, conn, "ben@example.test", "Ben")
    _job(conn, ana, "ana@example.test")
    b1 = _job(conn, ben, "ben@example.test")
    b2 = _job(conn, ben, "ben@example.test")
    _admin(client)

    body = client.get("/api/admin/activity", params={"user_id": ben}).json()
    assert [j["id"] for j in body["jobs"]] == [b2, b1]
    assert {j["user_id"] for j in body["jobs"]} == {ben}


def test_status_filters_and_an_unknown_status_gets_a_422_naming_the_choices(client, conn):
    ana = _make_user(client, conn, "ana@example.test", "Ana")
    _job(conn, ana, "ana@example.test", status="completed")
    bad = _job(conn, ana, "ana@example.test", status="failed")
    _admin(client)

    body = client.get("/api/admin/activity", params={"status": "failed"}).json()
    assert [j["id"] for j in body["jobs"]] == [bad]

    r = client.get("/api/admin/activity", params={"status": "exploded"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, str)
    assert "completed" in detail and "failed" in detail and "queued" in detail


def test_totals_count_jobs_and_spend_per_user(client, conn):
    ana = _make_user(client, conn, "ana@example.test", "Ana")
    ben = _make_user(client, conn, "ben@example.test", "Ben")
    _job(conn, ana, "ana@example.test", est=0.10)
    _job(conn, ana, "ana@example.test", est=0.10, actual=0.25)
    _job(conn, ana, "ana@example.test", est=0.50, status="failed")  # costs nothing
    old = _job(conn, ana, "ana@example.test", est=0.10)
    conn.execute("UPDATE jobs SET created_at = '2020-01-01T00:00:00+00:00' WHERE id = ?", (old,))
    _job(conn, ben, "ben@example.test", est=0.30)
    _admin(client)

    body = client.get("/api/admin/activity").json()
    users = {u["email"]: u for u in body["users"]}
    assert users["ana@example.test"]["jobs_total"] == 4
    assert users["ana@example.test"]["jobs_month"] == 3
    assert users["ana@example.test"]["spent_month_usd"] == pytest.approx(0.35)
    assert users["ana@example.test"]["name"] == "Ana"
    assert users["ben@example.test"]["jobs_total"] == 1
    assert users["ben@example.test"]["jobs_month"] == 1
    assert users["ben@example.test"]["spent_month_usd"] == pytest.approx(0.30)
    assert users["boss@example.test"]["jobs_total"] == 0
    assert body["month_total_usd"] == pytest.approx(0.65)


def test_pre_account_jobs_are_counted_once_and_a_legacy_name_has_its_own_line(client, conn):
    ana = _make_user(client, conn, "ana@example.test", "Ana")
    _job(conn, ana, "ana@example.test", est=0.10)
    old = _job(conn, None, "ana@example.test", est=0.20)  # before accounts existed
    ghost = _job(conn, None, "ghost@example.test", est=0.40)
    _admin(client)

    body = client.get("/api/admin/activity").json()
    users = {u["email"]: u for u in body["users"]}
    assert users["ana@example.test"]["jobs_total"] == 2
    assert users["ana@example.test"]["jobs_month"] == 2
    assert users["ana@example.test"]["spent_month_usd"] == pytest.approx(0.30)
    assert [u["email"] for u in body["users"]].count("ana@example.test") == 1
    assert users["ghost@example.test"]["user_id"] is None
    assert users["ghost@example.test"]["name"] == ""
    assert users["ghost@example.test"]["jobs_total"] == 1
    assert users["ghost@example.test"]["spent_month_usd"] == pytest.approx(0.40)

    mine = client.get("/api/admin/activity", params={"user_id": ana}).json()
    assert old in [j["id"] for j in mine["jobs"]]
    assert ghost not in [j["id"] for j in mine["jobs"]]


def test_next_before_id_is_set_on_a_full_page_and_null_on_the_empty_one_after(client, conn):
    ana = _make_user(client, conn, "ana@example.test", "Ana")
    ids = [_job(conn, ana, "ana@example.test") for _ in range(2)]
    _admin(client)

    p1 = client.get("/api/admin/activity", params={"limit": 2}).json()
    assert p1["next_before_id"] == ids[0]
    p2 = client.get(
        "/api/admin/activity", params={"limit": 2, "before_id": p1["next_before_id"]}
    ).json()
    assert p2["jobs"] == []
    assert p2["next_before_id"] is None


def test_a_job_inserted_between_page_fetches_is_not_shown_twice_and_nothing_is_skipped(
    client, conn
):
    ana = _make_user(client, conn, "ana@example.test", "Ana")
    ids = [_job(conn, ana, "ana@example.test") for _ in range(4)]
    _admin(client)

    p1 = client.get("/api/admin/activity", params={"limit": 2}).json()
    newest = _job(conn, ana, "ana@example.test")
    p2 = client.get(
        "/api/admin/activity", params={"limit": 2, "before_id": p1["next_before_id"]}
    ).json()

    seen = [j["id"] for j in p1["jobs"]] + [j["id"] for j in p2["jobs"]]
    assert seen == [ids[3], ids[2], ids[1], ids[0]]
    assert newest not in seen


@pytest.mark.parametrize("limit", [0, 501])
def test_a_limit_outside_1_to_500_gets_a_422(client, limit):
    _admin(client)
    assert client.get("/api/admin/activity", params={"limit": limit}).status_code == 422
