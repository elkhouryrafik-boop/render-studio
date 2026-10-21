"""Depth estimation, geometry-hold mapping, and the submit path that uses them.

The ONNX session is replaced by a fake in everything except the opt-in
`RUN_DEPTH_MODEL=1` test, so CI never needs the 95 MB model.
"""

from __future__ import annotations

import json
import os
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import budget
from app.config import Settings
from app.models import AssetKind
from app.services import assets, depth
from app.services import render as render_service


class FakeSession:
    """Stands in for onnxruntime.InferenceSession: inverse depth rising left to right."""

    def __init__(self) -> None:
        self.seen_shape: tuple[int, ...] | None = None
        self.seen_input: np.ndarray | None = None

    def get_inputs(self):
        class _In:
            name = "pixel_values"

        return [_In()]

    def run(self, _outputs, feeds):
        x = feeds["pixel_values"]
        self.seen_shape = x.shape
        self.seen_input = x
        _, _, h, w = x.shape
        ramp = np.tile(np.linspace(2.0, 10.0, w, dtype=np.float32), (h, 1))
        return [ramp[None]]  # (1, H, W), like predicted_depth


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, db_path=tmp_path / "t.db")


def _write_png(path: Path, size=(300, 200), colour="white") -> Path:
    Image.new("RGB", size, colour).save(path)
    return path


# --- (b) the estimator, with the session mocked -----------------------------


def test_estimate_writes_16_bit_png_at_source_size_near_is_bright(tmp_path, monkeypatch):
    fake = FakeSession()
    monkeypatch.setattr(depth, "_get_session", lambda settings: fake)
    src = _write_png(tmp_path / "in.png", size=(300, 200))

    res = depth.estimate_depth(src, tmp_path / "out.png", _settings(tmp_path))

    assert res.source == "estimated"
    assert res.model == "depth-anything-v2-small"
    assert res.path == tmp_path / "out.png"
    assert res.seconds >= 0

    img = Image.open(res.path)
    assert img.mode in ("I;16", "I")  # 16-bit greyscale
    assert img.size == (300, 200)
    a = np.array(img).astype(np.int64)
    assert a.max() == 65535 and a.min() == 0
    # Larger model output (nearer) on the right -> brighter on the right.
    assert a[:, -1].mean() > a[:, 0].mean()
    assert a[:, 0].mean() < 2000 and a[:, -1].mean() > 63000


def test_preprocess_shorter_side_518_multiple_of_14_imagenet_normalised(tmp_path, monkeypatch):
    fake = FakeSession()
    monkeypatch.setattr(depth, "_get_session", lambda settings: fake)
    src = _write_png(tmp_path / "in.png", size=(1280, 960), colour=(255, 255, 255))

    depth.estimate_depth(src, tmp_path / "o.png", _settings(tmp_path))

    n, c, h, w = fake.seen_shape
    assert (n, c) == (1, 3)
    assert min(h, w) == 518
    assert h % 14 == 0 and w % 14 == 0
    assert abs(w / h - 1280 / 960) < 0.03
    # White through ImageNet mean/std: (1 - 0.485) / 0.229 on the red channel.
    assert fake.seen_input.dtype == np.float32
    assert fake.seen_input[0, 0, 0, 0] == pytest.approx((1 - 0.485) / 0.229, abs=1e-3)


def test_eight_bit_output_for_the_control_image(tmp_path, monkeypatch):
    monkeypatch.setattr(depth, "_get_session", lambda settings: FakeSession())
    src = _write_png(tmp_path / "in.png")
    res = depth.estimate_depth(src, tmp_path / "o8.png", _settings(tmp_path), bits=8)
    img = Image.open(res.path)
    assert img.mode == "L"
    a = np.array(img)
    assert a.max() == 255 and a.min() == 0 and a[:, -1].mean() > a[:, 0].mean()


def test_flat_prediction_does_not_divide_by_zero(tmp_path, monkeypatch):
    class Flat(FakeSession):
        def run(self, _o, feeds):
            _, _, h, w = feeds["pixel_values"].shape
            return [np.full((1, h, w), 3.0, dtype=np.float32)]

    monkeypatch.setattr(depth, "_get_session", lambda settings: Flat())
    res = depth.estimate_depth(_write_png(tmp_path / "i.png"), tmp_path / "o.png", _settings(tmp_path))
    assert np.array(Image.open(res.path)).max() == 0


def test_is_available_needs_the_model_file(tmp_path):
    s = _settings(tmp_path)
    assert depth.is_available(s) is False
    model_dir = s.depth_model_dir()
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"x")
    assert depth.is_available(s) is True


# --- (d) opt-in: the real model on a synthetic scene -------------------------


@pytest.mark.skipif(os.environ.get("RUN_DEPTH_MODEL") != "1", reason="set RUN_DEPTH_MODEL=1")
def test_real_model_puts_a_near_rectangle_at_the_near_end(tmp_path):
    settings = Settings(db_path=tmp_path / "t.db")  # real DATA_DIR: where the model lives
    if not depth.is_available(settings):
        pytest.skip("depth model not fetched: run scripts/fetch_depth_model.py")
    w, h = 640, 480
    yy = np.linspace(0, 1, h)[:, None]
    # Far background: a vertical gradient like a ground plane receding to a horizon.
    bg = np.stack([np.tile(60 + 120 * yy, (1, w))] * 3, axis=-1).astype(np.uint8)
    scene = Image.fromarray(bg)
    from PIL import ImageDraw

    d = ImageDraw.Draw(scene)
    d.rectangle([180, 140, 460, 440], fill=(170, 90, 60), outline=(20, 10, 10), width=6)
    src = tmp_path / "scene.png"
    scene.save(src)

    res = depth.estimate_depth(src, tmp_path / "depth.png", settings)
    a = np.array(Image.open(res.path)).astype(np.float64)
    inside = a[200:400, 230:410].mean()
    outside = np.concatenate([a[:100, :].ravel(), a[:, :100].ravel()]).mean()
    print(f"real model: {res.seconds:.2f}s inside={inside:.0f} outside={outside:.0f}")
    assert inside > outside, "the near rectangle must sit at the near (bright) end"


# --- (a) geometry hold -> scales --------------------------------------------

from app.routing import RoutingTable

ROUTING = Path(__file__).resolve().parents[1] / "routing.yaml"


@pytest.fixture(scope="module")
def table() -> RoutingTable:
    return RoutingTable.load(ROUTING)


@pytest.mark.parametrize(
    ("value", "depth", "canny"),
    [
        (0, 0.20, 0.10),
        (25, 0.275, 0.15),
        (50, 0.35, 0.20),  # today's shipped scales
        (75, 0.475, 0.275),
        (100, 0.60, 0.35),
        (-10, 0.20, 0.10),  # clamped
        (250, 0.60, 0.35),  # clamped
    ],
)
def test_scales_for_hold_interpolates_and_clamps(table, value, depth, canny):
    s = table.scales_for_hold(value)
    assert s["depth"] == pytest.approx(depth)
    assert s["canny"] == pytest.approx(canny)
    assert set(s) == {"depth", "canny"}


def test_hold_snaps_are_sorted_loose_to_exact(table):
    assert table.scales_for_hold(0)["depth"] < table.scales_for_hold(50)["depth"] < table.scales_for_hold(100)["depth"]


# --- (c)(e) the submit path ---------------------------------------------------
# Fixture pattern copied from tests/test_api.py: temp DB, dependency overrides,
# fal.ai replaced at the gateway. Nothing here reaches the network.

from fastapi.testclient import TestClient

from api import deps
from api.main import app
from app import db, fal_gateway, ratelimit
from app.models import UserStatus

CODE = "office-code-123"
PASSWORD = "correct-horse-battery"
NOTE = "Edges only: no depth model is installed, so Exact is capped at Hold."


def _png_bytes(width: int = 64, height: int = 48) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), (200, 120, 80)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _clean_limiter():
    ratelimit.limiter.clear()
    yield
    ratelimit.limiter.clear()


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    return Settings(
        fal_key="test-key-not-real",
        office_access_code=CODE,
        admin_email="boss@example.test",
        budget_monthly_usd=100.0,
        budget_per_user_usd=50.0,
        budget_per_job_usd=1.0,
        data_dir=tmp_path,
        db_path=tmp_path / "api.db",
        routing_path=ROUTING,
    )


@pytest.fixture
def conn(api_settings: Settings):
    c = db.connect(api_settings.db_path)
    yield c
    c.close()


@pytest.fixture
def client(conn, api_settings, table):
    app.dependency_overrides[deps.get_conn] = lambda: conn
    app.dependency_overrides[deps.get_config] = lambda: api_settings
    app.dependency_overrides[deps.get_table] = lambda: table
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def submitted(monkeypatch):
    sent: list[tuple[str, dict]] = []

    class Handle:
        request_id = "req-test-1"

    def fake_submit(application, arguments=None, **_kw):
        sent.append((application, arguments or {}))
        return Handle()

    monkeypatch.setattr(fal_gateway.fal_client, "submit", fake_submit)
    monkeypatch.setattr(
        fal_gateway, "upload_input", lambda _s, path: f"https://fal.test/{Path(path).name}"
    )
    return sent


@pytest.fixture
def depth_model(monkeypatch):
    """Pretend the model is installed; count how many times inference runs."""
    fake = FakeSession()
    runs = {"n": 0}
    real_run = fake.run

    def counting_run(o, f):
        runs["n"] += 1
        return real_run(o, f)

    fake.run = counting_run  # type: ignore[method-assign]
    monkeypatch.setattr(depth, "is_available", lambda s: True)
    monkeypatch.setattr(depth, "_get_session", lambda s: fake)
    return runs


def _login(client, conn):
    client.post(
        "/api/auth/signup",
        json={"access_code": CODE, "email": "a@example.test", "name": "A", "password": PASSWORD},
    )
    row = db.get_user_by_email(conn, "a@example.test")
    db.set_user_status(conn, int(row["id"]), UserStatus.ACTIVE)
    client.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD})
    return client.post(
        "/api/assets", files={"file": ("massing.png", _png_bytes(), "image/png")}
    ).json()["id"]


def _render(client, asset_id, **extra):
    body = {
        "task": "massing_to_render",
        "engine": "proven",
        "prompt": "late afternoon light",
        "references": [{"asset_id": asset_id, "role": "geometry"}],
        **extra,
    }
    return client.post("/api/renders", json=body)


def _controls(arguments: dict) -> dict[str, dict]:
    return {c["control_mode"]: c for c in arguments["controlnet_unions"][0]["controls"]}


def test_an_estimated_depth_replaces_the_colour_image_in_the_depth_slot(
    client, conn, submitted, depth_model
):
    asset_id = _login(client, conn)
    r = _render(client, asset_id)
    assert r.status_code == 200, r.text
    assert r.json()["depth_source"] == "estimated"
    assert r.json()["notes"] == []

    controls = _controls(submitted[0][1])
    assert controls["depth"]["control_image_url"] != controls["canny"]["control_image_url"]
    assert db.get_job(conn, r.json()["job_id"])["depth_source"] == "estimated"
    # The estimated PNG is an ordinary CONTROL asset.
    kinds = [row["kind"] for row in conn.execute("SELECT kind FROM assets")]
    assert "control" in kinds
    # And it is reported on the job as the browser reads it.
    jobs = client.get("/api/renders").json()
    assert jobs[0]["depth_source"] == "estimated"


def test_a_repeat_upload_reuses_the_cached_depth(client, conn, submitted, depth_model, api_settings):
    asset_id = _login(client, conn)
    _render(client, asset_id, prompt="first")
    _render(client, asset_id, prompt="second")
    assert depth_model["n"] == 1
    cached = list(api_settings.depth_cache_dir().glob("*_depth-anything-v2-small.png"))
    assert len(cached) == 1


def test_a_caller_supplied_depth_is_cad_and_no_estimate_runs(client, conn, submitted, depth_model):
    asset_id = _login(client, conn)
    buf = BytesIO()
    Image.new("L", (64, 48), 128).save(buf, format="PNG")
    depth_asset = client.post(
        "/api/assets", files={"file": ("cad_depth.png", buf.getvalue(), "image/png")}
    ).json()["id"]
    r = _render(client, asset_id, control_assets={"depth": depth_asset})
    assert r.json()["depth_source"] == "cad"
    assert depth_model["n"] == 0


def test_without_a_model_the_job_still_submits_with_depth_source_none(client, conn, submitted):
    asset_id = _login(client, conn)
    r = _render(client, asset_id)  # tmp data_dir has no model
    assert r.status_code == 200, r.text
    assert r.json()["depth_source"] == "none"
    assert db.get_job(conn, r.json()["job_id"])["depth_source"] == "none"
    controls = _controls(submitted[0][1])
    assert controls["depth"]["control_image_url"] == controls["canny"]["control_image_url"]


def test_a_failing_estimator_never_fails_the_submit(client, conn, submitted, monkeypatch):
    class Boom(FakeSession):
        def run(self, _o, _f):
            raise RuntimeError("onnx exploded")

    monkeypatch.setattr(depth, "is_available", lambda s: True)
    monkeypatch.setattr(depth, "_get_session", lambda s: Boom())
    asset_id = _login(client, conn)
    r = _render(client, asset_id)
    assert r.status_code == 200, r.text
    assert r.json()["depth_source"] == "none"


# --- geometry hold on the wire ------------------------------------------------


def test_geometry_hold_overrides_depth_and_canny_scales(client, conn, submitted, depth_model):
    asset_id = _login(client, conn)
    r = _render(client, asset_id, geometry_hold=100)
    assert r.status_code == 200, r.text
    assert r.json()["notes"] == []
    controls = _controls(submitted[0][1])
    assert controls["depth"]["conditioning_scale"] == pytest.approx(0.60)
    assert controls["canny"]["conditioning_scale"] == pytest.approx(0.35)


def test_without_geometry_hold_the_shipped_scales_are_untouched(client, conn, submitted, depth_model):
    asset_id = _login(client, conn)
    _render(client, asset_id)
    controls = _controls(submitted[0][1])
    assert controls["depth"]["conditioning_scale"] == pytest.approx(0.35)
    assert controls["canny"]["conditioning_scale"] == pytest.approx(0.20)


HOLD_NOTE = "Geometry hold applies to perspective views only; this drawing keeps its tuned scales."


def test_a_hold_on_a_drawing_route_is_ignored_and_says_so(client, conn, submitted):
    asset_id = _login(client, conn)
    r = _render(client, asset_id, task="floorplan_prettify", geometry_hold=50)
    assert r.status_code == 200, r.text
    assert r.json()["notes"] == [HOLD_NOTE]
    controls = _controls(submitted[0][1])
    assert controls["canny"]["conditioning_scale"] == pytest.approx(0.75)
    params = json.loads(db.get_job(conn, r.json()["job_id"])["params_json"])
    assert "geometry_hold" not in params


def test_a_hold_on_a_perspective_route_still_moves_the_scales(client, conn, submitted):
    asset_id = _login(client, conn)
    r = _render(client, asset_id, geometry_hold=0)
    assert r.status_code == 200, r.text
    assert r.json()["notes"] == []
    controls = _controls(submitted[0][1])
    assert controls["depth"]["conditioning_scale"] == pytest.approx(0.20)
    assert controls["canny"]["conditioning_scale"] == pytest.approx(0.10)


def test_exact_is_capped_at_hold_when_there_is_no_depth_and_says_so(client, conn, submitted):
    asset_id = _login(client, conn)
    r = _render(client, asset_id, geometry_hold=80)
    assert r.status_code == 200, r.text
    assert r.json()["depth_source"] == "none"
    assert r.json()["notes"] == [NOTE]
    controls = _controls(submitted[0][1])
    assert controls["depth"]["conditioning_scale"] == pytest.approx(0.35)
    assert controls["canny"]["conditioning_scale"] == pytest.approx(0.20)


def test_a_hold_at_or_below_50_gets_no_note_without_depth(client, conn, submitted):
    asset_id = _login(client, conn)
    r = _render(client, asset_id, geometry_hold=20)
    assert r.json()["notes"] == []
    assert _controls(submitted[0][1])["depth"]["conditioning_scale"] == pytest.approx(0.26)


@pytest.mark.parametrize("bad", [-1, 101])
def test_geometry_hold_outside_0_100_is_a_validation_error(client, conn, submitted, bad):
    asset_id = _login(client, conn)
    assert _render(client, asset_id, geometry_hold=bad).status_code == 422


def test_estimate_accepts_geometry_hold(client, conn):
    _login(client, conn)
    ok = client.post("/api/estimate", json={"task": "massing_to_render", "geometry_hold": 50})
    assert ok.status_code == 200, ok.text
    assert client.post(
        "/api/estimate", json={"task": "massing_to_render", "geometry_hold": 101}
    ).status_code == 422


def test_migration_adds_depth_source_to_jobs(conn):
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(jobs)")]
    assert "depth_source" in cols


# --- concurrent and gated estimation ------------------------------------------


def test_two_concurrent_submits_of_one_screenshot_estimate_once(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    conn = db.connect(settings.db_path)
    asset_id = int(
        assets.ingest(conn, settings, data=_write_png(tmp_path / "s.png").read_bytes(),
                      filename="s.png", kind=AssetKind.UPLOAD)["id"]
    )
    conn.close()

    calls = {"n": 0}
    count_lock = threading.Lock()

    def slow_estimate(src, dst, settings, *, bits=16):
        with count_lock:
            calls["n"] += 1
        time.sleep(0.3)  # long enough that the second thread arrives mid-estimate
        Image.new("L", (64, 48), 128).save(dst, format="PNG")
        return depth.DepthResult(path=dst, seconds=0.3)

    monkeypatch.setattr(depth, "estimate_depth", slow_estimate)
    results: list[object] = []
    errors: list[BaseException] = []

    def submit_once():
        try:
            c = db.connect(settings.db_path)
            try:
                row = db.get_asset(c, asset_id)
                results.append(render_service.depth_control_for_asset(
                    c, settings, row, project_id=None, uploaded_by=None,
                ))
            finally:
                c.close()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=submit_once) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert calls["n"] == 1
    assert sorted(r.cached for r in results) == [False, True]


def test_a_kill_switched_office_does_not_run_inference_before_refusing(
    client, conn, submitted, depth_model
):
    asset_id = _login(client, conn)
    budget.set_kill_switch(conn, enabled=True, actor="test")
    r = _render(client, asset_id)
    assert r.status_code in (402, 403), r.text
    assert depth_model["n"] == 0
    assert submitted == []
