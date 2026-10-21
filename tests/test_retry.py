"""Sending a failed render again.

Runs the real lifecycle with RENDER_ENGINE=fake. A failure is produced with the
fake engine's `fail_next`, the same switch the gateway tests use.
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
from app.models import UserStatus
from app.routing import RoutingTable

ROUTING = Path(__file__).resolve().parents[1] / "routing.yaml"
CODE = "office-code-123"
PASSWORD = "correct-horse-battery"


def _png(width: int = 64, height: int = 48) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def _settings(tmp_path: Path) -> Settings:
    return Settings(
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


@pytest.fixture(autouse=True)
def _fresh_state():
    ratelimit.limiter.clear()
    fal_gateway._fake_client = None
    yield
    fal_gateway._fake_client = None
    ratelimit.limiter.clear()
    app.dependency_overrides.clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return _settings(tmp_path)


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


def _signup(client, email: str) -> None:
    r = client.post(
        "/api/auth/signup",
        json={"access_code": CODE, "email": email, "name": email.split("@")[0], "password": PASSWORD},
    )
    assert r.status_code == 200, r.text


def _admin(client, conn) -> None:
    _signup(client, "boss@example.test")


def _submit(client, conn, *, num_images: int = 2) -> int:
    project_id = db.create_project(conn, owner_id=None, name="Marina", client="ACME")
    upload = client.post(
        "/api/assets",
        files={"file": ("massing.png", _png(), "image/png")},
        data={"project_id": str(project_id)},
    )
    assert upload.status_code == 200, upload.text
    r = client.post(
        "/api/renders",
        json={
            "task": "massing_to_render",
            "engine": "proven",
            "prompt": "warm late afternoon light, board-formed concrete",
            "references": [{"asset_id": upload.json()["id"], "role": "geometry"}],
            "project_id": project_id,
            "width": 800,
            "height": 600,
            "num_images": num_images,
        },
    )
    assert r.status_code == 200, r.text
    return int(r.json()["job_id"])


def _fail(conn, settings, job_id: int) -> None:
    """Make the next completion of this render come back with no image."""
    fal_gateway._client(settings).fail_next = True
    assert fal_gateway.poll(conn, settings, job_id) == "failed"


def test_a_failed_render_is_sent_again_and_completes_with_outputs(client, conn, settings):
    _admin(client, conn)
    first = _submit(client, conn, num_images=2)
    _fail(conn, settings, first)
    assert db.get_job(conn, first)["status"] == "failed"

    r = client.post(f"/api/renders/{first}/retry")
    assert r.status_code == 200, r.text
    body = r.json()
    second = int(body["job_id"])
    assert second != first
    assert body["duplicate"] is False
    assert body["message"] == f"Render #{second} queued, retrying #{first}."

    assert fal_gateway.poll(conn, settings, second) == "completed"
    listed = client.get(f"/api/renders/{second}").json()
    assert listed["retry_of_job_id"] == first
    assert listed["status"] == "completed"
    assert [o["index"] for o in listed["outputs"]] == [0, 1]
    assert all(o["url"] for o in listed["outputs"])


def test_the_retry_sends_the_same_task_prompt_size_count_and_inputs(client, conn, settings):
    _admin(client, conn)
    first = _submit(client, conn, num_images=2)
    before = db.get_job(conn, first)
    before_inputs = [(r["id"], r["role"], r["position"]) for r in db.job_inputs(conn, first)]
    _fail(conn, settings, first)

    second = int(client.post(f"/api/renders/{first}/retry").json()["job_id"])

    after = db.get_job(conn, second)
    assert after["task"] == before["task"]
    assert after["engine"] == before["engine"]
    assert after["model_id"] == before["model_id"]
    assert after["mask_asset_id"] is None and before["mask_asset_id"] is None
    assert after["project_id"] == before["project_id"]
    assert after["est_usd"] == pytest.approx(before["est_usd"])
    assert [(r["id"], r["role"], r["position"]) for r in db.job_inputs(conn, second)] == before_inputs

    import json as _json

    old_args = _json.loads(before["params_json"])
    new_args = _json.loads(after["params_json"])
    assert new_args == old_args  # same prompt, size, count, references, scales


def test_retrying_a_completed_render_is_refused_with_a_sentence(client, conn, settings):
    _admin(client, conn)
    job_id = _submit(client, conn, num_images=1)
    assert fal_gateway.poll(conn, settings, job_id) == "completed"

    r = client.post(f"/api/renders/{job_id}/retry")
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == (
        f"Render #{job_id} is complete. Only a failed render can be retried."
    )


def test_a_render_still_in_flight_cannot_be_retried(client, conn, settings):
    _admin(client, conn)
    job_id = _submit(client, conn, num_images=1)

    r = client.post(f"/api/renders/{job_id}/retry")
    assert r.status_code == 409
    assert "still" in r.json()["detail"]
    assert "Only a failed render can be retried." in r.json()["detail"]


def test_a_failed_retry_can_itself_be_retried_and_gets_a_new_job(client, conn, settings):
    _admin(client, conn)
    first = _submit(client, conn, num_images=1)
    _fail(conn, settings, first)
    second = int(client.post(f"/api/renders/{first}/retry").json()["job_id"])
    _fail(conn, settings, second)

    third = int(client.post(f"/api/renders/{first}/retry").json()["job_id"])
    assert third not in (first, second)
    assert db.get_job(conn, third)["retry_of_job_id"] == first


def test_a_second_retry_while_the_first_is_live_is_refused(client, conn, settings):
    """A double click on Retry must not spend twice."""
    _admin(client, conn)
    first = _submit(client, conn, num_images=1)
    _fail(conn, settings, first)
    second = int(client.post(f"/api/renders/{first}/retry").json()["job_id"])

    r = client.post(f"/api/renders/{first}/retry")
    assert r.status_code == 409
    assert r.json()["detail"] == (
        f"Render #{first} was already sent again as #{second}, which is still "
        "waiting for fal.ai. Only a failed render can be retried."
    )


def test_retrying_an_unknown_render_is_a_404(client, conn, settings):
    _admin(client, conn)
    r = client.post("/api/renders/999999/retry")
    assert r.status_code == 404
    assert r.json()["detail"] == "No render 999999 to retry."


def test_another_user_cannot_retry_someone_elses_render(client, conn, settings):
    _admin(client, conn)
    job_id = _submit(client, conn, num_images=1)
    _fail(conn, settings, job_id)

    # A second, active account that did not start the render.
    _signup(client, "other@example.test")
    other = db.get_user_by_email(conn, "other@example.test")
    db.set_user_status(conn, int(other["id"]), UserStatus.ACTIVE)
    client.post("/api/auth/logout")
    client.post("/api/auth/login", json={"email": "other@example.test", "password": PASSWORD})

    r = client.post(f"/api/renders/{job_id}/retry")
    assert r.status_code == 403
    assert r.json()["detail"] == (
        "Only the person who started this render, or an administrator, can retry it."
    )
    assert db.get_job(conn, job_id)["retry_of_job_id"] is None


def test_a_retry_sends_the_cad_depth_control_again(client, conn, settings):
    """A depth pass from a CAD confirm is not a job input, so it is found again by
    its fal URL in the stored payload, and the retry is still a CAD-depth render."""
    from app.models import AssetKind
    from app.services import assets

    _admin(client, conn)
    project_id = db.create_project(conn, owner_id=None, name="Marina", client="ACME")
    geometry = client.post(
        "/api/assets",
        files={"file": ("massing.png", _png(), "image/png")},
        data={"project_id": str(project_id)},
    ).json()
    depth = assets.ingest(
        conn, settings, data=_png(40, 30), filename="depth.png",
        project_id=project_id, uploaded_by=None, kind=AssetKind.CONTROL,
    )
    r = client.post(
        "/api/renders",
        json={
            "task": "massing_to_render",
            "engine": "proven",
            "prompt": "warm late afternoon light",
            "references": [{"asset_id": geometry["id"], "role": "geometry"}],
            "project_id": project_id,
            "width": 800,
            "height": 600,
            "num_images": 1,
            "control_assets": {"depth": int(depth["id"])},
        },
    )
    assert r.status_code == 200, r.text
    first = int(r.json()["job_id"])
    assert db.get_job(conn, first)["depth_source"] == "cad"
    _fail(conn, settings, first)

    second = int(client.post(f"/api/renders/{first}/retry").json()["job_id"])

    job = db.get_job(conn, second)
    assert job["depth_source"] == "cad"
    depth_row = db.get_asset(conn, int(depth["id"]))
    assert depth_row["fal_url"], "the depth asset must have been hosted for the first render"
    assert depth_row["fal_url"] in job["params_json"]


def test_a_retry_of_an_old_job_without_a_depth_source_still_finds_its_depth_control(
    client, conn, settings
):
    """Jobs from before depth_source was recorded have it NULL. The depth pass is
    still in the stored payload, so the retry finds it there, as it does for a
    change map."""
    from app.models import AssetKind
    from app.services import assets

    _admin(client, conn)
    project_id = db.create_project(conn, owner_id=None, name="Marina", client="ACME")
    geometry = client.post(
        "/api/assets",
        files={"file": ("massing.png", _png(), "image/png")},
        data={"project_id": str(project_id)},
    ).json()
    depth = assets.ingest(
        conn, settings, data=_png(40, 30), filename="depth.png",
        project_id=project_id, uploaded_by=None, kind=AssetKind.CONTROL,
    )
    r = client.post(
        "/api/renders",
        json={
            "task": "massing_to_render",
            "engine": "proven",
            "prompt": "warm late afternoon light",
            "references": [{"asset_id": geometry["id"], "role": "geometry"}],
            "project_id": project_id,
            "width": 800,
            "height": 600,
            "num_images": 1,
            "control_assets": {"depth": int(depth["id"])},
        },
    )
    assert r.status_code == 200, r.text
    first = int(r.json()["job_id"])
    conn.execute("UPDATE jobs SET depth_source = NULL WHERE id = ?", (first,))
    _fail(conn, settings, first)

    second = int(client.post(f"/api/renders/{first}/retry").json()["job_id"])

    job = db.get_job(conn, second)
    assert job["depth_source"] == "cad"
    assert db.get_asset(conn, int(depth["id"]))["fal_url"] in job["params_json"]



def test_a_failed_region_render_is_retried_with_its_mask(client, conn, settings):
    """The mask is part of the job, so the retry carries it and does not fail
    with "cannot run without a mask"."""
    import json as _json

    _admin(client, conn)
    project_id = db.create_project(conn, owner_id=None, name="Marina", client="ACME")
    source = client.post(
        "/api/assets",
        files={"file": ("room.png", _png(64, 48), "image/png")},
        data={"project_id": str(project_id)},
    ).json()
    mask = client.post(
        "/api/assets/mask",
        files={"file": ("mask.png", _png(64, 48), "image/png")},
        data={"source_asset_id": str(source["id"])},
    )
    assert mask.status_code == 200, mask.text
    mask_id = int(mask.json()["id"])

    r = client.post(
        "/api/renders",
        json={
            "intent": "region",
            "line": "replace the sofa with a walnut bench",
            "references": [{"asset_id": source["id"], "role": "geometry"}],
            "mask_asset_id": mask_id,
            "project_id": project_id,
            "width": 64,
            "height": 48,
        },
    )
    assert r.status_code == 200, r.text
    first = int(r.json()["job_id"])
    assert db.get_job(conn, first)["mask_asset_id"] == mask_id
    _fail(conn, settings, first)

    again = client.post(f"/api/renders/{first}/retry")
    assert again.status_code == 200, again.text
    second = int(again.json()["job_id"])
    assert db.get_job(conn, second)["mask_asset_id"] == mask_id

    assert fal_gateway.poll(conn, settings, second) == "completed"
    stored = _json.loads(db.get_job(conn, second)["params_json"])
    assert stored["mask_url"]
