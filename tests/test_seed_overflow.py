"""A seed past SQLite's signed 64-bit range must not sink a paid render.

Seen on 2026-10-10 in the paid smoke: fal.ai reported a seed above 2**63-1,
sqlite3 raised OverflowError on every poll, the job sat in_progress until the
timeout sweep failed it, and the image fal had already billed was thrown away.
"""

from __future__ import annotations

from app import db, fal_gateway
from app.fake_engine import FakeFalClient
from tests.test_job_outputs import _submit, client, conn, fake_settings  # noqa: F401

HUGE_SEED = 2**64 - 1


def test_a_seed_past_sqlite_range_is_kept_and_served(client, conn, fake_settings, monkeypatch):
    original = FakeFalClient.result

    def result_with_huge_seed(self, application, request_id):
        payload = original(self, application, request_id)
        payload["seed"] = HUGE_SEED
        return payload

    monkeypatch.setattr(FakeFalClient, "result", result_with_huge_seed)

    job_id = _submit(client, conn, num_images=2)
    assert fal_gateway.poll(conn, fake_settings, job_id) == "completed"

    rows = db.list_job_outputs(conn, job_id)
    assert [r["idx"] for r in rows] == [0, 1]
    assert all(db.seed_from_db(r["seed"]) == HUGE_SEED for r in rows)
    assert db.seed_from_db(db.get_job(conn, job_id)["output_seed"]) == HUGE_SEED

    body = client.get(f"/api/renders/{job_id}").json()
    assert body["status"] == "completed"
    assert body["output_seed"] == HUGE_SEED
    assert [o["seed"] for o in body["outputs"]] == [HUGE_SEED, HUGE_SEED]


def test_output_seed_keeps_small_seeds_as_integers_and_rejects_bools():
    assert fal_gateway._output_seed({"seed": 1234}) == 1234
    assert fal_gateway._output_seed({"seed": "99"}) == 99
    assert fal_gateway._output_seed({"seed": True}) is None
    assert fal_gateway._output_seed({"seed": HUGE_SEED}) == hex(HUGE_SEED)
    assert fal_gateway._output_seed({}) is None
