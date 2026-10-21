"""A render that never finishes is failed and its budget released.

The clock is injected by replacing `fal_gateway._now`, so nothing sleeps.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app import db, fal_gateway, ratelimit
from app.config import Settings
from app.models import ImageRole
from app.routing import RoutingTable

ROUTING = Path(__file__).resolve().parents[1] / "routing.yaml"
REF_URL = "https://fal.test/geometry.png"


@pytest.fixture(autouse=True)
def _fresh_state():
    ratelimit.limiter.clear()
    fal_gateway._fake_client = None
    yield
    fal_gateway._fake_client = None


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        fal_key="",
        office_access_code="office-code-123",
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


@pytest.fixture
def conn(settings: Settings):
    c = db.connect(settings.db_path)
    yield c
    c.close()


@pytest.fixture
def table() -> RoutingTable:
    return RoutingTable.load(ROUTING)


def _submit(conn, settings, table, prompt: str = "board-formed concrete, late light") -> int:
    result = fal_gateway.submit(
        conn, settings, table,
        username="a@example.test",
        task="massing_to_render",
        engine="proven",
        prompt=prompt,
        width=1280,
        height=960,
        num_images=1,
        references=[(ImageRole.GEOMETRY, REF_URL)],
        user_id=None,
    )
    return result.job_id


def _slow(settings: Settings) -> Settings:
    """Settings whose fake render never finishes within a test: it stays in flight."""
    return settings.model_copy(update={"fake_engine_delay_s": 3600.0})


def _age(conn, job_id: int, minutes: float) -> datetime:
    """Set a job's created_at to `minutes` before now and return it."""
    created = datetime.now(UTC) - timedelta(minutes=minutes)
    conn.execute("UPDATE jobs SET created_at = ? WHERE id = ?", (created.isoformat(), job_id))
    return created


def _clock_at(monkeypatch, when: datetime) -> None:
    monkeypatch.setattr(fal_gateway, "_now", lambda: when)


def test_a_job_nine_minutes_old_is_failed_and_its_reservation_released(
    conn, settings, table, monkeypatch
):
    settings = _slow(settings)
    job_id = _submit(conn, settings, table)
    created = _age(conn, job_id, minutes=9)
    spend_before = db.committed_spend_this_month(conn)
    assert spend_before > 0  # the reservation is holding budget

    _clock_at(monkeypatch, created + timedelta(minutes=9))
    results = fal_gateway.poll_all(conn, settings)

    assert results[job_id] == "failed"
    job = db.get_job(conn, job_id)
    assert job["status"] == "failed"
    assert job["error"] == "Timed out after 8 minutes. Retry sends it again."
    assert job["timed_out_at"] is not None
    assert db.committed_spend_this_month(conn) < spend_before


def test_a_render_that_finishes_before_the_timeout_is_saved_not_failed(
    conn, settings, table, monkeypatch
):
    # Fake delay 0: fal has the result by the time this tick polls, although the
    # job is past its timeout. The poll comes first, so the output is kept.
    job_id = _submit(conn, settings, table)
    created = _age(conn, job_id, minutes=9)

    _clock_at(monkeypatch, created + timedelta(minutes=9))
    results = fal_gateway.poll_all(conn, settings)

    assert results[job_id] == "completed"
    job = db.get_job(conn, job_id)
    assert job["status"] == "completed"
    assert job["timed_out_at"] is None
    assert db.get_job_output(conn, job_id, 0)["path"] is not None


def test_an_expired_render_is_cancelled_at_fal(conn, settings, table, monkeypatch):
    slow = _slow(settings)
    job_id = _submit(conn, slow, table)
    created = _age(conn, job_id, minutes=9)

    _clock_at(monkeypatch, created + timedelta(minutes=9))
    fal_gateway.poll_all(conn, slow)

    request_id = db.get_job(conn, job_id)["request_id"]
    assert request_id in fal_gateway._client(slow).cancelled


def test_one_failed_expiry_does_not_stop_the_others(conn, settings, table, monkeypatch):
    slow = _slow(settings)
    first = _submit(conn, slow, table)
    second = _submit(conn, slow, table, prompt="a second render, same site")
    created = _age(conn, first, minutes=9)
    _age(conn, second, minutes=9)
    real_update = db.update_job

    def flaky_update(c, job_id, **fields):
        if job_id == first and "timed_out_at" in fields:
            raise RuntimeError("disk I/O error")
        return real_update(c, job_id, **fields)

    monkeypatch.setattr(db, "update_job", flaky_update)
    _clock_at(monkeypatch, created + timedelta(minutes=9))
    results = fal_gateway.poll_all(conn, slow)

    assert results[second] == "failed"
    assert db.get_job(conn, second)["timed_out_at"] is not None
    # The failed write rolled back with its transaction: no `failed` without a mark.
    assert db.get_job(conn, first)["status"] in ("queued", "in_progress")
    assert db.get_job(conn, first)["timed_out_at"] is None


def test_a_job_one_minute_old_is_not_timed_out(conn, settings, table, monkeypatch):
    job_id = _submit(conn, settings, table)
    created = _age(conn, job_id, minutes=1)

    _clock_at(monkeypatch, created + timedelta(minutes=1))
    results = fal_gateway.poll_all(conn, settings)

    job = db.get_job(conn, job_id)
    assert job["timed_out_at"] is None
    assert job["status"] != "failed"
    assert results[job_id] == "completed"  # fake engine, delay 0: it finishes normally


def test_a_reserved_job_with_no_request_is_also_expired(conn, settings, monkeypatch):
    """A row stuck in `reserved` holds budget and is never polled, so it must age out too."""
    job_id = db.reserve_job(
        conn, idempotency_key="stuck", username="a@example.test", task="t",
        model_id="fal-ai/flux-general", est_usd=0.25, params_json="{}", input_path=None,
    )
    created = _age(conn, job_id, minutes=9)

    _clock_at(monkeypatch, created + timedelta(minutes=9))
    assert fal_gateway.poll_all(conn, settings)[job_id] == "failed"
    assert db.get_job(conn, job_id)["status"] == "failed"


def test_the_timeout_length_comes_from_settings(conn, settings, table, monkeypatch):
    settings = _slow(settings)
    job_id = _submit(conn, settings, table)
    created = _age(conn, job_id, minutes=2)
    short = settings.model_copy(update={"job_timeout_s": 60})

    _clock_at(monkeypatch, created + timedelta(minutes=2))
    fal_gateway.poll_all(conn, short)

    assert db.get_job(conn, job_id)["error"] == "Timed out after 1 minute. Retry sends it again."


def test_terminal_jobs_are_never_touched_by_the_timeout(conn, settings, table, monkeypatch):
    job_id = _submit(conn, settings, table)
    fal_gateway.poll(conn, settings, job_id)  # completes
    created = _age(conn, job_id, minutes=30)

    _clock_at(monkeypatch, created + timedelta(minutes=30))
    fal_gateway.poll_all(conn, settings)

    job = db.get_job(conn, job_id)
    assert job["status"] == "completed"
    assert job["timed_out_at"] is None
