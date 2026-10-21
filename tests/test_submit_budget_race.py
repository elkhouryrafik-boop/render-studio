"""The monthly cap must hold when several architects submit at once.

`budget.check()` reads a running SUM and `db.reserve_job()` writes the row that
changes it. With nothing transactional between them, N concurrent submissions
all measured themselves against the same pre-spend total and all passed, so the
overshoot was roughly the per-job cap times the concurrency.

`app/budget.py` grew an atomic path for this, but the gateway was still calling
the split one, so the fix existed and was not live. These tests exercise
`fal_gateway.submit()` itself, which is the only thing that proves the wiring.

No network, no spend: every fal.ai call is monkeypatched.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app import db, fal_gateway
from app.config import Settings
from app.models import ImageRole
from app.routing import RoutingTable

ROUTING = Path(__file__).resolve().parents[1] / "routing.yaml"
VIEW = "https://fal.test/view.png"

# massing_to_render on `latest` is priced per megapixel; 1280x960 lands each
# job at a known figure, which is what makes the arithmetic below checkable.
TASK = "massing_to_render"
ENGINE = "latest"


@pytest.fixture
def table() -> RoutingTable:
    return RoutingTable.load(ROUTING)


class _Handle:
    def __init__(self, n: int) -> None:
        self.request_id = f"req-{n}"


def _make_settings(tmp_path: Path, *, monthly: float, per_user: float) -> Settings:
    return Settings(
        fal_key="test-key-not-real",
        budget_monthly_usd=monthly,
        budget_per_user_usd=per_user,
        budget_per_job_usd=0.50,
        data_dir=tmp_path,
        db_path=tmp_path / "race.db",
    )


def _submit(conn, settings, table, *, username: str, prompt: str):
    return fal_gateway.submit(
        conn,
        settings,
        table,
        username=username,
        task=TASK,
        engine=ENGINE,
        prompt=prompt,
        references=[(ImageRole.GEOMETRY, VIEW)],
        num_images=1,
    )


def _run_concurrently(db_path: Path, settings, table, *, threads: int, username: str):
    """Each thread gets its own connection, as each request does in the app."""
    results: list[object] = []
    errors: list[BaseException] = []
    lock = threading.Lock()
    ready = threading.Barrier(threads)

    def worker(n: int) -> None:
        conn = db.connect(db_path)
        try:
            ready.wait(timeout=10)
            out = _submit(conn, settings, table, username=username, prompt=f"dusk {n}")
        except BaseException as exc:  # noqa: BLE001 - the refusals are the data
            with lock:
                errors.append(exc)
        else:
            with lock:
                results.append(out)
        finally:
            conn.close()

    workers = [threading.Thread(target=worker, args=(n,)) for n in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=60)
    return results, errors


def test_the_monthly_cap_holds_under_concurrent_submits(tmp_path, table, monkeypatch):
    counter = iter(range(1000))
    monkeypatch.setattr(fal_gateway, "_install_key", lambda _s: None)
    monkeypatch.setattr(
        fal_gateway.fal_client, "submit", lambda *a, **k: _Handle(next(counter))
    )

    settings = _make_settings(tmp_path, monthly=1.00, per_user=1.00)
    conn = db.connect(settings.db_path)
    est = table.estimate_usd(
        table.resolve(TASK, ENGINE).model_id, 1280, 960, 1
    )
    conn.close()
    affordable = int(1.00 // est)
    assert affordable >= 2, "pick a cap that fits more than one job, or this proves nothing"

    results, errors = _run_concurrently(
        settings.db_path, settings, table, threads=affordable * 4, username="alice"
    )

    conn = db.connect(settings.db_path)
    try:
        committed = db.spend_this_month(conn)
    finally:
        conn.close()

    assert committed <= 1.00 + 1e-9, (
        f"committed ${committed:.4f} against a $1.00 cap from {len(results)} "
        f"accepted submissions of ${est:.4f} each"
    )
    assert len(results) == affordable
    assert all(isinstance(e, fal_gateway.SubmitRefused) for e in errors), (
        f"every rejection should be a refusal, got {[type(e).__name__ for e in errors]}"
    )
    assert all(e.reason in {"monthly", "per_user"} for e in errors)


def test_the_per_user_cap_holds_while_the_office_cap_has_room(
    tmp_path, table, monkeypatch
):
    """One architect must not be able to outrun their own allowance."""
    counter = iter(range(1000))
    monkeypatch.setattr(fal_gateway, "_install_key", lambda _s: None)
    monkeypatch.setattr(
        fal_gateway.fal_client, "submit", lambda *a, **k: _Handle(next(counter))
    )

    settings = _make_settings(tmp_path, monthly=100.0, per_user=0.50)
    conn = db.connect(settings.db_path)
    est = table.estimate_usd(table.resolve(TASK, ENGINE).model_id, 1280, 960, 1)
    conn.close()
    affordable = int(0.50 // est)

    results, errors = _run_concurrently(
        settings.db_path, settings, table, threads=affordable * 4, username="alice"
    )

    conn = db.connect(settings.db_path)
    try:
        spent = db.spend_this_month(conn, username="alice")
    finally:
        conn.close()

    assert spent <= 0.50 + 1e-9, f"one user committed ${spent:.4f} against a $0.50 allowance"
    assert len(results) == affordable
    assert errors


def test_identical_concurrent_submissions_collapse_to_one_job(
    tmp_path, table, monkeypatch
):
    """The idempotency key has to survive the same race it now runs inside."""
    counter = iter(range(1000))
    monkeypatch.setattr(fal_gateway, "_install_key", lambda _s: None)
    monkeypatch.setattr(
        fal_gateway.fal_client, "submit", lambda *a, **k: _Handle(next(counter))
    )

    settings = _make_settings(tmp_path, monthly=100.0, per_user=100.0)
    threads = 8
    results: list[object] = []
    errors: list[BaseException] = []
    lock = threading.Lock()
    ready = threading.Barrier(threads)

    def worker() -> None:
        conn = db.connect(settings.db_path)
        try:
            ready.wait(timeout=10)
            out = _submit(conn, settings, table, username="alice", prompt="identical")
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)
        else:
            with lock:
                results.append(out)
        finally:
            conn.close()

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=60)

    assert not errors, [repr(e) for e in errors]
    job_ids = {r.job_id for r in results}  # type: ignore[attr-defined]
    assert len(job_ids) == 1, f"the same request became {len(job_ids)} jobs: {job_ids}"
    assert sum(1 for r in results if not r.duplicate) == 1  # type: ignore[attr-defined]

    conn = db.connect(settings.db_path)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    finally:
        conn.close()
    assert rows == 1, f"{rows} rows on the ledger for one request"


def test_a_refused_submission_leaves_nothing_on_the_ledger(tmp_path, table, monkeypatch):
    """A refusal must not leave a `reserved` row holding budget."""
    monkeypatch.setattr(fal_gateway, "_install_key", lambda _s: None)
    monkeypatch.setattr(
        fal_gateway.fal_client,
        "submit",
        lambda *a, **k: pytest.fail("a refused job must never reach fal.ai"),
    )

    settings = _make_settings(tmp_path, monthly=0.001, per_user=0.001)
    conn = db.connect(settings.db_path)
    try:
        with pytest.raises(fal_gateway.SubmitRefused):
            _submit(conn, settings, table, username="alice", prompt="over the cap")
        held = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = 'reserved'"
        ).fetchone()[0]
        assert held == 0
        assert db.spend_this_month(conn) == pytest.approx(0.0)
    finally:
        conn.close()
