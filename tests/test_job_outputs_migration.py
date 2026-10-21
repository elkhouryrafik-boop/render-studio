"""Schema version 8: job_outputs, retry_of_job_id, timed_out_at.

The v7 database is built by running the shipped steps by hand, so this test
exercises the real upgrade path rather than a fresh schema.
"""

from __future__ import annotations

import sqlite3

from app import db, migrations


def _v7_database(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "v7.db", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    for number, _name, step in migrations.MIGRATIONS[:7]:
        step(conn)
    conn.execute("PRAGMA user_version = 7")
    return conn


def _insert_job(conn, key: str, *, output_path=None, output_url=None, output_seed=None) -> int:
    cur = conn.execute(
        """
        INSERT INTO jobs (idempotency_key, username, task, model_id, status, est_usd,
                          created_at, updated_at, output_path, output_url, output_seed)
        VALUES (?, 'a@office.test', 'massing_to_render', 'fal-ai/flux-general',
                'completed', 0.5, '2026-10-01T10:00:00+00:00',
                '2026-10-01T10:05:00+00:00', ?, ?, ?)
        """,
        (key, output_path, output_url, output_seed),
    )
    return int(cur.lastrowid)


def test_migrating_a_v7_database_backfills_index_zero_for_completed_jobs(tmp_path):
    conn = _v7_database(tmp_path)
    done = _insert_job(
        conn, "k-done",
        output_path="/data/outputs/job000001_0.png",
        output_url="https://fal.test/one.png",
        output_seed=77,
    )
    _insert_job(conn, "k-failed")  # no output_path: must not get a row

    assert migrations.migrate(conn) == migrations.SCHEMA_VERSION == 9

    rows = conn.execute(
        "SELECT job_id, idx, path, url, seed, created_at FROM job_outputs ORDER BY job_id, idx"
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["job_id"] == done
    assert row["idx"] == 0
    assert row["path"] == "/data/outputs/job000001_0.png"
    assert row["url"] == "https://fal.test/one.png"
    assert row["seed"] == 77
    assert row["created_at"]  # NOT NULL, filled from the job's timestamp

    job_cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    assert {"retry_of_job_id", "timed_out_at"} <= job_cols


def test_rerunning_the_migrations_changes_nothing(tmp_path):
    conn = _v7_database(tmp_path)
    _insert_job(conn, "k-done", output_path="/x.png", output_url="u", output_seed=None)
    migrations.migrate(conn)
    migrations.migrate(conn)
    count = conn.execute("SELECT COUNT(*) FROM job_outputs").fetchone()[0]
    assert count == 1


def test_the_unique_index_is_one_row_per_job_and_index(tmp_path):
    conn = _v7_database(tmp_path)
    migrations.migrate(conn)
    job = _insert_job(conn, "k-new", output_path=None)
    conn.execute(
        "INSERT INTO job_outputs (job_id, idx, path, created_at) VALUES (?, 0, 'a', 't')", (job,)
    )
    try:
        conn.execute(
            "INSERT INTO job_outputs (job_id, idx, path, created_at) VALUES (?, 0, 'b', 't')", (job,)
        )
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("a second row for (job_id, idx) must be refused")


def test_retry_lookups_have_an_index_after_migrating(tmp_path):
    conn = db.connect(tmp_path / "idx.db")
    plan = conn.execute(
        "EXPLAIN QUERY PLAN SELECT id FROM jobs WHERE retry_of_job_id = ?", (1,)
    ).fetchall()
    assert any("idx_jobs_retry_of" in str(tuple(row)) for row in plan)

