"""Every image a job produces is kept, served and reusable.

Runs the real job lifecycle with RENDER_ENGINE=fake: reserve, submit, poll,
save, serve. No fal_client function is monkeypatched.
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
from app.routing import RoutingTable

ROUTING = Path(__file__).resolve().parents[1] / "routing.yaml"
CODE = "office-code-123"
PASSWORD = "correct-horse-battery"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _png(width: int = 64, height: int = 48) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


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


@pytest.fixture(autouse=True)
def _fresh_state():
    ratelimit.limiter.clear()
    fal_gateway._fake_client = None
    yield
    fal_gateway._fake_client = None
    ratelimit.limiter.clear()
    app.dependency_overrides.clear()


@pytest.fixture
def fake_settings(tmp_path: Path) -> Settings:
    return _settings(tmp_path)


@pytest.fixture
def conn(fake_settings: Settings):
    c = db.connect(fake_settings.db_path)
    yield c
    c.close()


@pytest.fixture
def client(conn, fake_settings: Settings):
    table = RoutingTable.load(ROUTING)
    app.dependency_overrides[deps.get_conn] = lambda: conn
    app.dependency_overrides[deps.get_config] = lambda: fake_settings
    app.dependency_overrides[deps.get_table] = lambda: table
    yield TestClient(app)


def _signed_in_admin(client) -> None:
    r = client.post(
        "/api/auth/signup",
        json={"access_code": CODE, "email": "boss@example.test", "name": "Boss", "password": PASSWORD},
    )
    assert r.status_code == 200, r.text


def _submit(client, conn, *, num_images: int, width: int = 800, height: int = 600, signup: bool = True) -> int:
    if signup:
        _signed_in_admin(client)
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
            "width": width,
            "height": height,
            "num_images": num_images,
        },
    )
    assert r.status_code == 200, r.text
    return int(r.json()["job_id"])


def test_four_images_persist_four_output_rows_and_index_zero_on_the_job(client, conn, fake_settings):
    job_id = _submit(client, conn, num_images=4)
    assert fal_gateway.poll(conn, fake_settings, job_id) == "completed"

    rows = db.list_job_outputs(conn, job_id)
    assert [r["idx"] for r in rows] == [0, 1, 2, 3]
    assert all(r["seed"] == 1234 for r in rows)
    assert [r["url"] for r in rows] == [
        f"fake://result/{db.get_job(conn, job_id)['request_id']}/{i}" for i in range(4)
    ]
    job = db.get_job(conn, job_id)
    assert job["output_path"] == rows[0]["path"]
    assert job["output_url"] == rows[0]["url"]


def test_the_render_lists_all_four_outputs_with_served_urls(client, conn, fake_settings):
    job_id = _submit(client, conn, num_images=4)
    fal_gateway.poll(conn, fake_settings, job_id)

    body = client.get(f"/api/renders/{job_id}").json()
    assert [o["index"] for o in body["outputs"]] == [0, 1, 2, 3]
    assert [o["url"] for o in body["outputs"]] == [
        f"/api/renders/{job_id}/image?i={i}" for i in range(4)
    ]
    assert all(o["seed"] == 1234 for o in body["outputs"])
    assert body["output_url"] == f"/api/renders/{job_id}/image"


def test_each_index_is_served_as_its_own_png(client, conn, fake_settings):
    job_id = _submit(client, conn, num_images=4)
    fal_gateway.poll(conn, fake_settings, job_id)

    sizes = []
    for i in range(4):
        r = client.get(f"/api/renders/{job_id}/image", params={"i": i})
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "image/png"
        assert r.content.startswith(PNG_MAGIC)
        with Image.open(BytesIO(r.content)) as img:
            sizes.append(img.size)
    assert sizes == [(800, 600)] * 4

    # Default is index 0, and the files are distinct, not one copy served four times.
    default = client.get(f"/api/renders/{job_id}/image")
    assert default.content == client.get(f"/api/renders/{job_id}/image", params={"i": 0}).content
    assert client.get(f"/api/renders/{job_id}/image", params={"i": 3}).content != default.content


def test_a_missing_index_is_a_404_with_a_sentence(client, conn, fake_settings):
    job_id = _submit(client, conn, num_images=2)
    fal_gateway.poll(conn, fake_settings, job_id)

    r = client.get(f"/api/renders/{job_id}/image", params={"i": 2})
    assert r.status_code == 404
    assert r.json()["detail"] == "No image 2 for that render."


def test_an_output_whose_file_is_gone_is_listed_and_its_image_is_a_404(client, conn, fake_settings):
    job_id = _submit(client, conn, num_images=2)
    fal_gateway.poll(conn, fake_settings, job_id)
    Path(db.list_job_outputs(conn, job_id)[1]["path"]).unlink()

    body = client.get(f"/api/renders/{job_id}").json()
    # The list does not stat each file; the image route is the one that decides.
    assert [o["url"] is not None for o in body["outputs"]] == [True, True]
    assert client.get(f"/api/renders/{job_id}/image", params={"i": 1}).status_code == 404


def test_an_output_that_was_never_saved_has_no_url(client, conn, fake_settings):
    job_id = _submit(client, conn, num_images=2)
    fal_gateway.poll(conn, fake_settings, job_id)
    db.add_job_output(conn, job_id=job_id, idx=1, path=None, url="fake://result/x/1", seed=None)

    body = client.get(f"/api/renders/{job_id}").json()
    assert body["outputs"][1]["url"] is None


def test_a_failed_download_keeps_its_index_so_later_images_are_not_shifted(
    client, conn, fake_settings, monkeypatch
):
    """Output 1 failing to download must not make output 2 answer as index 1."""
    job_id = _submit(client, conn, num_images=3)
    real = fal_gateway._save_images

    def flaky(settings, jid, urls, sizes=None, indexes=None):
        saved = real(settings, jid, urls, sizes=sizes, indexes=indexes)
        saved[1] = None  # as if the download for index 1 had failed
        return saved

    monkeypatch.setattr(fal_gateway, "_save_images", flaky)
    fal_gateway.poll(conn, fake_settings, job_id)

    rows = db.list_job_outputs(conn, job_id)
    assert [r["path"] is not None for r in rows] == [True, False, True]
    assert rows[2]["path"].endswith(f"job{job_id:06d}_2.png")
    assert client.get(f"/api/renders/{job_id}/image", params={"i": 1}).status_code == 404
    assert client.get(f"/api/renders/{job_id}/image", params={"i": 2}).status_code == 200


@pytest.mark.parametrize("failing", [0, 2])
def test_a_download_that_fails_once_is_retried_on_the_next_tick(
    client, conn, fake_settings, monkeypatch, failing
):
    """Only the missing image is fetched again, and the job stays in flight until it lands."""
    job_id = _submit(client, conn, num_images=3)
    real = fal_gateway._save_images
    calls: list[list[int]] = []

    def fail_once(settings, jid, urls, sizes=None, indexes=None):
        numbers = list(indexes) if indexes is not None else list(range(len(urls)))
        calls.append(numbers)
        saved = real(settings, jid, urls, sizes=sizes, indexes=indexes)
        if len(calls) == 1:
            saved[numbers.index(failing)] = None
        return saved

    monkeypatch.setattr(fal_gateway, "_save_images", fail_once)
    assert fal_gateway.poll(conn, fake_settings, job_id) == "in_progress"
    assert db.get_job(conn, job_id)["status"] == "in_progress"
    rows = db.list_job_outputs(conn, job_id)
    assert [r["path"] is None for r in rows] == [i == failing for i in range(3)]

    assert fal_gateway.poll(conn, fake_settings, job_id) == "completed"
    assert calls == [[0, 1, 2], [failing]]
    assert db.get_job(conn, job_id)["status"] == "completed"
    rows = db.list_job_outputs(conn, job_id)
    assert all(r["path"].endswith(f"job{job_id:06d}_{r['idx']}.png") for r in rows)


def test_continue_from_a_chosen_output_uses_that_file(client, conn, fake_settings):
    job_id = _submit(client, conn, num_images=4)
    fal_gateway.poll(conn, fake_settings, job_id)

    r = client.post(
        f"/api/renders/{job_id}/continue",
        json={
            "task": "massing_to_render",
            "engine": "proven",
            "prompt": "same building, dusk",
            "output_index": 2,
        },
    )
    assert r.status_code == 200, r.text
    child = int(r.json()["job_id"])

    inputs = db.job_inputs(conn, child)
    assert len(inputs) == 1
    assert inputs[0]["local_path"].endswith(f"job{job_id:06d}_2.png")


def test_as_reference_takes_the_chosen_output(client, conn, fake_settings):
    job_id = _submit(client, conn, num_images=4)
    fal_gateway.poll(conn, fake_settings, job_id)

    r = client.post(f"/api/renders/{job_id}/as-reference", params={"output_index": 3})
    assert r.status_code == 200, r.text
    asset = db.get_asset(conn, int(r.json()["id"]))
    assert asset["local_path"].endswith(f"job{job_id:06d}_3.png")
    assert asset["fal_url"].endswith(f"/{job_id}/3") or asset["fal_url"].endswith("/3")


def test_list_returns_outputs_for_every_job_on_the_page(client, conn, fake_settings):
    first = _submit(client, conn, num_images=2)
    fal_gateway.poll(conn, fake_settings, first)
    second = _submit(client, conn, num_images=1, width=640, height=480, signup=False)
    fal_gateway.poll(conn, fake_settings, second)

    listed = {r["id"]: r for r in client.get("/api/renders").json()}
    assert len(listed[first]["outputs"]) == 2
    assert len(listed[second]["outputs"]) == 1
    assert listed[first]["outputs"][1]["url"] == f"/api/renders/{first}/image?i=1"


def test_a_job_says_how_many_images_and_what_size_it_asked_for(client, conn, fake_settings):
    _signed_in_admin(client)
    job_id = _submit(client, conn, num_images=3, width=800, height=600, signup=False)
    body = client.get(f"/api/renders/{job_id}").json()
    assert body["num_images"] == 3
    assert (body["width"], body["height"]) == (800, 600)


def test_a_job_with_no_stored_request_counts_one_image(conn):
    from api import schemas

    jid = db.reserve_job(
        conn, idempotency_key="k-legacy", username="x", task="massing_to_render",
        model_id="m", est_usd=0.1, params_json="{}", input_path=None,
    )
    out = schemas.job_out(db.get_job(conn, jid))
    assert out.num_images == 1
    assert out.width is None and out.height is None
