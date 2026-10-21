"""The fake engine: a zero-spend stand-in for the fal.ai client.

No network and no key. The fake client is driven by an injected clock, so the
status walk (queued, in progress, completed) is exact rather than sleepy.
"""

from __future__ import annotations

import fal_client
import pytest

from app.fake_engine import FakeFalClient
from app.models import ImageRole

MODEL = "fal-ai/flux-general"


class Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def test_status_walks_queued_then_in_progress_then_completed():
    clock = Clock()
    fake = FakeFalClient(delay_s=4.0, clock=clock)
    handle = fake.submit(MODEL, arguments={"num_images": 1})
    assert handle.request_id == "fake-1"

    # Queued for the first half of the delay (0 .. 2.0 s).
    assert isinstance(fake.status(MODEL, handle.request_id), fal_client.Queued)
    clock.now += 1.9
    assert isinstance(fake.status(MODEL, handle.request_id), fal_client.Queued)

    # In progress from 2.0 s until the delay (4.0 s).
    clock.now += 0.2  # 2.1 s
    assert isinstance(fake.status(MODEL, handle.request_id), fal_client.InProgress)
    clock.now += 1.8  # 3.9 s
    assert isinstance(fake.status(MODEL, handle.request_id), fal_client.InProgress)

    # Completed from the delay onwards.
    clock.now += 0.1  # 4.0 s
    assert isinstance(fake.status(MODEL, handle.request_id), fal_client.Completed)


def test_zero_delay_completes_on_the_first_status_call():
    fake = FakeFalClient(delay_s=0.0, clock=Clock())
    handle = fake.submit(MODEL, arguments={})
    assert isinstance(fake.status(MODEL, handle.request_id), fal_client.Completed)


def test_each_submission_gets_its_own_request_id_and_remembers_its_arguments():
    fake = FakeFalClient(delay_s=0.0, clock=Clock())
    a = fake.submit(MODEL, arguments={"prompt": "a"})
    b = fake.submit("fal-ai/other", arguments={"prompt": "b"})
    assert a.request_id != b.request_id
    assert fake._requests[a.request_id].application == MODEL
    assert fake._requests[b.request_id].arguments == {"prompt": "b"}


def test_four_images_at_a_preset_size():
    fake = FakeFalClient(delay_s=0.0, clock=Clock())
    rid = fake.submit(MODEL, arguments={"num_images": 4, "image_size": "square_hd"}).request_id
    payload = fake.result(MODEL, rid)

    assert payload["seed"] == 1234
    images = payload["images"]
    assert [img["url"] for img in images] == [f"fake://result/{rid}/{i}" for i in range(4)]
    assert all((img["width"], img["height"]) == (1024, 1024) for img in images)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"image_size": "landscape_4_3"}, (1280, 960)),
        ({"image_size": "portrait_4_3"}, (960, 1280)),
        ({"image_size": "something_unknown"}, (1024, 768)),
        ({"image_size": {"width": 640, "height": 480}}, (640, 480)),
        ({"width": 800, "height": 600}, (800, 600)),
        ({}, (1024, 768)),
    ],
)
def test_the_requested_size_is_read_from_every_shape_the_gateway_sends(arguments, expected):
    fake = FakeFalClient(delay_s=0.0, clock=Clock())
    rid = fake.submit(MODEL, arguments=arguments).request_id
    image = fake.result(MODEL, rid)["images"][0]
    assert (image["width"], image["height"]) == expected


def test_a_single_image_is_the_default():
    fake = FakeFalClient(delay_s=0.0, clock=Clock())
    rid = fake.submit(MODEL, arguments={}).request_id
    assert len(fake.result(MODEL, rid)["images"]) == 1


def test_upload_is_a_content_addressed_fake_url(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(b"not really a png, but stable bytes")
    fake = FakeFalClient(delay_s=0.0, clock=Clock())
    url = fake.upload_file(path)
    assert url.startswith("fake://upload/")
    assert url == fake.upload_file(path)  # same bytes, same URL


def test_an_unknown_request_says_not_found_as_the_gateway_expects():
    fake = FakeFalClient(delay_s=0.0, clock=Clock())
    with pytest.raises(Exception, match="NOT_FOUND"):
        fake.status(MODEL, "fake-999")
    with pytest.raises(Exception, match="NOT_FOUND"):
        fake.result(MODEL, "fake-999")


def test_fail_next_completes_the_next_status_and_returns_no_images():
    clock = Clock()
    fake = FakeFalClient(delay_s=100.0, clock=clock)
    rid = fake.submit(MODEL, arguments={}).request_id
    fake.fail_next = True

    # Completed even though the delay has not passed, and the flag is spent.
    assert isinstance(fake.status(MODEL, rid), fal_client.Completed)
    assert fake.fail_next is False
    assert fake.result(MODEL, rid) == {"images": []}
    # The failure is spent: a later result for the same request is normal again.
    assert len(fake.result(MODEL, rid)["images"]) == 1


# --- Through the gateway and the API ----------------------------------------
#
# These drive the real job lifecycle (reserve, submit, poll, save, serve) with
# RENDER_ENGINE=fake. No fal_client function is monkeypatched in the fake
# tests: if one reached the network, the test would fail loudly.

from io import BytesIO
from pathlib import Path

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
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(autouse=True)
def _fresh_state():
    """The fake client is a process singleton; each test starts with none."""
    ratelimit.limiter.clear()
    fal_gateway._fake_client = None
    yield
    fal_gateway._fake_client = None
    ratelimit.limiter.clear()
    app.dependency_overrides.clear()


@pytest.fixture
def fake_settings(tmp_path: Path) -> Settings:
    return _settings(tmp_path, render_engine="fake", fake_engine_delay_s=0.0)


@pytest.fixture
def conn(fake_settings: Settings):
    c = db.connect(fake_settings.db_path)
    yield c
    c.close()


def _wire(conn, settings: Settings) -> None:
    table = RoutingTable.load(ROUTING)
    app.dependency_overrides[deps.get_conn] = lambda: conn
    app.dependency_overrides[deps.get_config] = lambda: settings
    app.dependency_overrides[deps.get_table] = lambda: table


@pytest.fixture
def client(conn, fake_settings: Settings):
    _wire(conn, fake_settings)
    yield TestClient(app)


def _signed_in_admin(client) -> None:
    r = client.post(
        "/api/auth/signup",
        json={"access_code": CODE, "email": "boss@example.test", "name": "Boss", "password": PASSWORD},
    )
    assert r.status_code == 200, r.text


def _submit_massing(client, conn, *, width: int, height: int, num_images: int) -> int:
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


def test_a_four_image_job_completes_with_four_pngs_at_the_requested_size(client, conn, fake_settings):
    job_id = _submit_massing(client, conn, width=800, height=600, num_images=4)

    assert fal_gateway.poll(conn, fake_settings, job_id) == "completed"

    job = db.get_job(conn, job_id)
    assert job["status"] == "completed"
    assert job["output_url"] == f"fake://result/{job['request_id']}/0"

    files = sorted(fake_settings.outputs_dir().glob(f"job{job_id:06d}_*.png"))
    assert len(files) == 4
    for path in files:
        with Image.open(path) as img:
            assert img.format == "PNG"
            assert img.size == (800, 600)


def test_a_fake_failure_ends_the_job_failed_with_the_plain_no_image_error(client, conn, fake_settings):
    job_id = _submit_massing(client, conn, width=800, height=600, num_images=1)
    fal_gateway._client(fake_settings).fail_next = True

    assert fal_gateway.poll(conn, fake_settings, job_id) == "failed"

    job = db.get_job(conn, job_id)
    assert job["status"] == "failed"
    assert job["error"] == "The render came back without an image. Retry sends it again."
    assert list(fake_settings.outputs_dir().glob(f"job{job_id:06d}_*.png")) == []


def test_health_in_fake_mode_needs_no_key_and_names_the_engine(client, conn, fake_settings):
    assert fake_settings.fal_key.get_secret_value() == ""
    r = client.get("/api/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"] == "fake"


def test_the_fal_engine_still_submits_through_the_fal_client(conn, tmp_path, monkeypatch):
    fal_settings = _settings(tmp_path, render_engine="fal", fal_key="test-key-not-real")
    _wire(conn, fal_settings)
    submitted: list[tuple[str, dict]] = []

    class Handle:
        request_id = "req-real-path"

    def fake_submit(application, arguments=None, **_kw):
        submitted.append((application, arguments or {}))
        return Handle()

    monkeypatch.setattr(fal_gateway.fal_client, "submit", fake_submit)
    monkeypatch.setattr(
        fal_gateway, "upload_input", lambda _s, path: f"https://fal.test/{Path(path).name}"
    )
    client = TestClient(app)
    job_id = _submit_massing(client, conn, width=800, height=600, num_images=1)

    assert [app_id for app_id, _ in submitted] == ["fal-ai/flux-general"]
    assert db.get_job(conn, job_id)["request_id"] == "req-real-path"
    assert fal_gateway._fake_client is None  # the fake was never constructed


# --- fake URLs never reach fal.ai ---------------------------------------------


def _ingest_png(conn, settings: Settings, name: str = "a.png") -> int:
    from app.services import assets

    row = assets.ingest(conn, settings, data=_png(), filename=name)
    return int(row["id"])


def test_a_stored_fake_url_is_uploaded_again_under_the_fal_engine(conn, tmp_path, monkeypatch):
    from app.services import assets

    fal_settings = _settings(tmp_path, render_engine="fal", fal_key="test-key-not-real")
    asset_id = _ingest_png(conn, fal_settings)
    db.set_asset_fal_url(conn, asset_id, "fake://upload/from-a-dev-run")
    monkeypatch.setattr(fal_gateway, "upload_input", lambda _s, path: "https://fal.test/real.png")

    assert assets.ensure_fal_url(conn, fal_settings, asset_id) == "https://fal.test/real.png"
    assert db.get_asset(conn, asset_id)["fal_url"] == "https://fal.test/real.png"


def test_a_same_image_with_only_a_fake_url_elsewhere_is_uploaded_under_fal(conn, tmp_path, monkeypatch):
    from app.services import assets

    fal_settings = _settings(tmp_path, render_engine="fal", fal_key="test-key-not-real")
    first = _ingest_png(conn, fal_settings, "first.png")
    db.set_asset_fal_url(conn, first, "fake://upload/shared-bytes")
    second = _ingest_png(conn, fal_settings, "second.png")  # same bytes, no URL of its own
    monkeypatch.setattr(fal_gateway, "upload_input", lambda _s, path: "https://fal.test/real.png")

    assert assets.ensure_fal_url(conn, fal_settings, second) == "https://fal.test/real.png"


def test_a_fake_url_is_still_reused_under_the_fake_engine(conn, fake_settings):
    from app.services import assets

    asset_id = _ingest_png(conn, fake_settings)
    db.set_asset_fal_url(conn, asset_id, "fake://upload/kept")
    assert assets.ensure_fal_url(conn, fake_settings, asset_id) == "fake://upload/kept"


def test_fal_refuses_a_fake_reference_before_anything_is_spent(conn, tmp_path, monkeypatch):
    fal_settings = _settings(tmp_path, render_engine="fal", fal_key="test-key-not-real")
    _wire(conn, fal_settings)
    submitted: list[str] = []
    monkeypatch.setattr(fal_gateway.fal_client, "submit", lambda app, **_kw: submitted.append(app))
    table = RoutingTable.load(ROUTING)

    with pytest.raises(fal_gateway.SubmitRefused, match="fake engine"):
        fal_gateway.submit(
            conn, fal_settings, table,
            username="a@example.test", task="massing_to_render", prompt="a render",
            references=[(ImageRole.GEOMETRY, "fake://upload/x")], user_id=None,
        )
    assert submitted == []
    assert db.committed_spend_this_month(conn) == 0


def test_me_says_which_engine_is_running(conn, fake_settings, tmp_path):
    _wire(conn, fake_settings)
    client = TestClient(app)
    _signed_in_admin(client)
    assert client.get("/api/auth/me").json()["engine"] == "fake"
    assert "engine" not in client.get("/api/admin/users").json()[0]
