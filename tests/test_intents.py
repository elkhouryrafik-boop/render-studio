"""Intents, styles and the always-four rule, as the contract states them.

Routing-level tests here need no database. The API-level tests at the bottom
drive the real submit path with RENDER_ENGINE=fake, so nothing reaches fal.ai.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from app.routing import IntentError, RoutingError, RoutingTable

ROUTING = Path(__file__).resolve().parents[1] / "routing.yaml"


@pytest.fixture(scope="module")
def table() -> RoutingTable:
    return RoutingTable.load(ROUTING)


@pytest.fixture(scope="module")
def raw() -> dict:
    with open(ROUTING, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --- GET /api/intents payload, against the contract -------------------------

CONTRACT_VIEWS = [
    {"key": "model", "label": "Model view"},
    {"key": "sketch", "label": "Sketch"},
    {"key": "plan", "label": "Plan"},
    {"key": "masterplan", "label": "Master plan"},
    {"key": "elevation", "label": "Elevation"},
    {"key": "interior", "label": "Interior"},
    {"key": "aerial", "label": "Aerial"},
]

CONTRACT_INTENTS = [
    {"key": "make_real", "label": "Make it real",
     "hint": "Turns the view into a render and holds the drawn geometry.",
     "card": True, "slider": True, "needs_mask": False, "needs_line": False,
     "max_references": 0, "requires": None, "views": CONTRACT_VIEWS, "default_view": "model"},
    {"key": "change", "label": "Change something",
     "hint": "Same building, different light, materials or season. The picture stays as drawn.",
     "card": True, "slider": False, "needs_mask": False, "needs_line": True,
     "max_references": 7, "requires": None, "views": [], "default_view": None},
    {"key": "region", "label": "Change a region",
     "hint": "Mark the part that is wrong and say what it becomes.",
     "card": True, "slider": False, "needs_mask": True, "needs_line": True,
     "max_references": 0, "requires": None, "views": [], "default_view": None},
    {"key": "material", "label": "Material render (SKP)",
     "hint": "Renders with the materials assigned in SketchUp.",
     "card": True, "slider": True, "needs_mask": False, "needs_line": False,
     "max_references": 0, "requires": "cad_materials", "views": [], "default_view": None},
    {"key": "enlarge", "label": "Enlarge",
     "hint": "Upscales a finished sheet for presentation.",
     "card": False, "slider": False, "needs_mask": False, "needs_line": False,
     "max_references": 0, "requires": None, "views": [], "default_view": None},
]

CONTRACT_STYLES = [
    {"key": "daylight", "label": "Daylight"}, {"key": "overcast", "label": "Overcast"},
    {"key": "golden", "label": "Golden hour"}, {"key": "night", "label": "Night"},
    {"key": "clay", "label": "Clay"},
]

CONTRACT_HOLD = {
    "default": 50,
    "snaps": [
        {"key": "loose", "value": 0, "label": "Loose",
         "hint": "Follows the view loosely; proportions may move."},
        {"key": "hold", "value": 50, "label": "Hold",
         "hint": "Holds your drawing with a depth and edge pass."},
        {"key": "exact", "value": 100, "label": "Exact",
         "hint": "Traces the depth as drawn; materials may come back flatter."},
    ],
}

CONTRACT_SIZES = [
    {"key": "1280x960", "label": "1280 × 960", "width": 1280, "height": 960},
    {"key": "1920x1080", "label": "1920 × 1080", "width": 1920, "height": 1080},
    {"key": "1024x1024", "label": "1024 × 1024", "width": 1024, "height": 1024},
    {"key": "960x1280", "label": "960 × 1280", "width": 960, "height": 1280},
]


def test_intents_payload_matches_the_contract(table):
    payload = table.intents_out(depth_available=True)
    assert set(payload) == {
        "intents", "styles", "default_style", "geometry_hold", "sizes",
        "depth_available", "num_images_default", "num_images_max",
    }
    assert payload["intents"] == CONTRACT_INTENTS
    assert payload["styles"] == CONTRACT_STYLES
    assert payload["default_style"] == "daylight"
    assert payload["geometry_hold"] == CONTRACT_HOLD
    assert payload["sizes"] == CONTRACT_SIZES
    assert payload["depth_available"] is True
    assert payload["num_images_default"] == 4
    assert payload["num_images_max"] == 4


def test_depth_available_is_the_callers_answer(table):
    assert table.intents_out()["depth_available"] is False
    assert table.intents_out(depth_available=True)["depth_available"] is True


def test_there_are_five_intents_and_five_styles(table):
    payload = table.intents_out()
    assert [i["key"] for i in payload["intents"]] == [
        "make_real", "change", "region", "material", "enlarge",
    ]
    assert [s["key"] for s in payload["styles"]] == ["daylight", "overcast", "golden", "night", "clay"]


def test_only_enlarge_is_not_a_card(table):
    cards = {i["key"]: i["card"] for i in table.intents_out()["intents"]}
    assert cards == {"make_real": True, "change": True, "region": True, "material": True, "enlarge": False}


# --- resolve_intent ---------------------------------------------------------

MAKE_REAL_TASKS = {
    "model": "massing_to_render",
    "sketch": "sketch_to_render",
    "plan": "floorplan_prettify",
    "masterplan": "masterplan_prettify",
    "elevation": "elevation_section_render",
    "interior": "interior_to_render",
    "aerial": "aerial_view_render",
}


@pytest.mark.parametrize(("view", "task"), sorted(MAKE_REAL_TASKS.items()))
def test_make_real_resolves_every_view_to_its_task_on_the_proven_engine(table, view, task):
    assert table.resolve_intent("make_real", view) == (task, "proven")


def test_make_real_without_a_view_takes_the_default_view(table):
    assert table.resolve_intent("make_real") == ("massing_to_render", "proven")


@pytest.mark.parametrize(
    ("intent", "task", "engine"),
    [
        ("change", "variation", "latest"),
        ("region", "inpaint", "proven"),
        ("material", "skp_material_render", "proven"),
        ("enlarge", "upscale", "latest"),
    ],
)
def test_the_other_intents_resolve_to_their_task_and_engine(table, intent, task, engine):
    assert table.resolve_intent(intent) == (task, engine)


def test_an_unknown_intent_is_refused_with_a_sentence(table):
    with pytest.raises(IntentError) as exc:
        table.resolve_intent("repaint_everything")
    assert "Unknown intent 'repaint_everything'" in str(exc.value)


def test_an_unknown_view_is_refused_and_names_the_views(table):
    with pytest.raises(IntentError) as exc:
        table.resolve_intent("make_real", "section")
    message = str(exc.value)
    assert "'section' view" in message
    assert "model" in message and "aerial" in message


def test_a_view_on_an_intent_without_views_is_refused(table):
    with pytest.raises(IntentError) as exc:
        table.resolve_intent("change", "model")
    assert "takes no view" in str(exc.value)


def test_intent_errors_are_routing_errors_too(table):
    # The API maps IntentError to 422 and every other RoutingError to 400; both
    # must still be catchable as the older base class.
    assert issubclass(IntentError, RoutingError)


# --- always four -------------------------------------------------------------

@pytest.mark.parametrize("intent", ["make_real", "change", "region", "material"])
def test_intents_ask_for_four_on_engines_that_return_four(table, intent):
    assert table.intent_num_images(intent) == 4


def test_enlarge_asks_for_one_because_its_engine_returns_one(table):
    assert table.intent_num_images("enlarge") == 1


def test_every_make_real_view_asks_for_four(table):
    for view in MAKE_REAL_TASKS:
        assert table.intent_num_images("make_real", view) == 4


# --- compose_prompt ---------------------------------------------------------

def test_compose_joins_scaffold_style_and_line(table):
    assert table.compose_prompt("massing_to_render", "daylight", "brick, oak frames") == (
        "A photoreal exterior render of the modelled building, as seen in the view. "
        "photoreal architectural photograph, clear daylight, soft shadows, true-to-material colours "
        "brick, oak frames"
    )


def test_compose_without_a_line_ends_at_the_style(table):
    text = table.compose_prompt("massing_to_render", "night", None)
    assert text.endswith("dusk, interior lights on, cool sky, warm windows")
    assert text.startswith("A photoreal exterior render")


def test_compose_without_a_scaffold_is_just_style_and_line(table):
    # relight has no prompt_scaffold in routing.yaml.
    assert table.compose_prompt("relight", "clay", "pale stone") == (
        "white clay model, matte white surfaces, soft studio light, no textures pale stone"
    )


def test_compose_with_no_style_and_no_line_is_the_scaffold_alone(table):
    assert table.compose_prompt("upscale", None, None) == (
        "A sharper, larger copy of the image, with the same content and no added detail."
    )


def test_compose_collapses_whitespace_in_the_line(table):
    text = table.compose_prompt("variation", "overcast", "  warped \n  window  frame ")
    assert "  " not in text
    assert text.endswith("muted saturation warped window frame")


def test_compose_refuses_an_unknown_style(table):
    with pytest.raises(IntentError):
        table.compose_prompt("massing_to_render", "sepia", None)


def test_every_style_prompt_is_in_the_house_voice(table):
    # No marketing words in what users read.
    banned = ("generate", "ai-powered", "unleash", "seamless", "effortless")
    for key in table.intents_out()["styles"]:
        prompt = table.compose_prompt("massing_to_render", key["key"], None).lower()
        assert not any(word in prompt for word in banned), key


# --- snaps and sizes ---------------------------------------------------------

def test_geometry_hold_still_interpolates_with_presentation_keys_present(table):
    assert table.scales_for_hold(0) == {"depth": 0.20, "canny": 0.10}
    assert table.scales_for_hold(50) == {"depth": 0.35, "canny": 0.20}
    assert table.scales_for_hold(100) == {"depth": 0.60, "canny": 0.35}


def test_hold_default_comes_from_routing_yaml(table):
    assert table.hold_default == 50


# --- the table refuses to load a broken intents block --------------------------

def _load(raw_dict: dict) -> RoutingTable:
    return RoutingTable(copy.deepcopy(raw_dict))


def test_an_intent_naming_an_unknown_task_fails_loudly(raw):
    broken = copy.deepcopy(raw)
    broken["intents"]["change"]["task"] = "variaton"
    with pytest.raises(RoutingError, match="variaton"):
        _load(broken)


def test_an_intent_naming_an_engine_its_task_does_not_offer_fails_loudly(raw):
    broken = copy.deepcopy(raw)
    broken["intents"]["region"]["engine"] = "latest"  # inpaint no longer offers latest, so the intent must be refused
    broken["tasks"]["inpaint"]["engines"].pop("latest")
    with pytest.raises(RoutingError, match="does not offer"):
        _load(broken)


def test_a_view_naming_an_unknown_task_fails_loudly(raw):
    broken = copy.deepcopy(raw)
    broken["intents"]["make_real"]["views"]["plan"]["task"] = "floorplan_prettfy"
    with pytest.raises(RoutingError, match="floorplan_prettfy"):
        _load(broken)


def test_a_default_style_that_does_not_exist_fails_loudly(raw):
    broken = copy.deepcopy(raw)
    broken["intent_defaults"]["style"] = "sepia"
    with pytest.raises(RoutingError, match="sepia"):
        _load(broken)


def test_a_style_without_a_prompt_fails_loudly(raw):
    broken = copy.deepcopy(raw)
    broken["styles"]["clay"]["prompt"] = "   "
    with pytest.raises(RoutingError, match="clay"):
        _load(broken)


# --- Through the API, with the fake engine (zero spend) ----------------------

import json as _json
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api import deps
from api.main import app
from app import db, fal_gateway, ratelimit
from app.config import Settings
from app.fake_engine import FakeFalClient

CODE = "office-code-123"
PASSWORD = "correct-horse-battery"


def _png(width: int = 64, height: int = 48) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _fresh_fake_state():
    ratelimit.limiter.clear()
    fal_gateway._fake_client = None
    yield
    fal_gateway._fake_client = None
    ratelimit.limiter.clear()
    app.dependency_overrides.clear()


@pytest.fixture
def fake_settings(tmp_path: Path) -> Settings:
    # render_engine=fake: no FAL_KEY, no network. The submit path is the real one.
    return Settings(
        fal_key="",
        render_engine="fake",
        fake_engine_delay_s=0.0,
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
    c = TestClient(app)
    r = c.post(
        "/api/auth/signup",
        json={"access_code": CODE, "email": "boss@example.test", "name": "Boss", "password": PASSWORD},
    )
    assert r.status_code == 200, r.text
    return c


def _upload(client, name: str = "view.png", **kw) -> int:
    r = client.post("/api/assets", files={"file": (name, _png(**kw), "image/png")})
    assert r.status_code == 200, r.text
    return int(r.json()["id"])


def _mask_for(client, source_id: int) -> int:
    # Its own bytes: an identical PNG would be content-addressed to the source asset.
    buf = BytesIO()
    Image.new("L", (64, 48), 255).save(buf, format="PNG")
    r = client.post(
        "/api/assets/mask",
        files={"file": ("mask.png", buf.getvalue(), "image/png")},
        data={"source_asset_id": str(source_id)},
    )
    assert r.status_code == 200, r.text
    return int(r.json()["id"])


def _params(conn, job_id: int) -> dict:
    return _json.loads(db.get_job(conn, job_id)["params_json"])


def _submit(client, **body):
    r = client.post("/api/renders", json=body)
    return r, (r.json() if r.status_code == 200 else None)


def test_get_intents_matches_the_contract_over_http(client):
    r = client.get("/api/intents")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["intents"] == CONTRACT_INTENTS
    assert body["styles"] == CONTRACT_STYLES
    assert body["geometry_hold"] == CONTRACT_HOLD
    assert body["sizes"] == CONTRACT_SIZES
    assert body["num_images_default"] == 4
    # No model file in the test settings, so the depth route is not offered.
    assert body["depth_available"] is False


def test_make_real_submits_flux_general_with_four_results_and_records_the_intent(client, conn, fake_settings):
    asset = _upload(client)
    r, body = _submit(
        client,
        intent="make_real", view="model", style="daylight", line="brick, oak frames",
        references=[{"asset_id": asset, "role": "geometry"}],
    )
    assert r.status_code == 200, r.text
    job_id = body["job_id"]

    job = db.get_job(conn, job_id)
    assert job["task"] == "massing_to_render"
    assert job["model_id"] == "fal-ai/flux-general"

    params = _params(conn, job_id)
    assert params["num_images"] == 4
    assert params["intent"] == "make_real"
    assert params["view"] == "model"
    assert params["style"] == "daylight"
    assert params["line"] == "brick, oak frames"
    assert params["geometry_hold"] == 50
    assert "photoreal architectural photograph" in params["prompt"]
    assert params["prompt"].endswith("brick, oak frames")

    assert fal_gateway.poll(conn, fake_settings, job_id) == "completed"
    assert len(db.list_job_outputs(conn, job_id)) == 4


def test_job_out_returns_the_intent_fields(client, conn):
    asset = _upload(client)
    _, body = _submit(
        client, intent="make_real", view="plan", style="clay",
        references=[{"asset_id": asset, "role": "geometry"}],
    )
    out = client.get(f"/api/renders/{body['job_id']}").json()
    assert out["intent"] == "make_real"
    assert out["view"] == "plan"
    assert out["style"] == "clay"
    assert out["line"] is None
    # A plan is a drawing: the hold does not apply, so none is recorded.
    assert out["geometry_hold"] is None


def test_a_legacy_render_has_null_intent_fields(client, conn):
    asset = _upload(client)
    _, body = _submit(
        client, task="massing_to_render", engine="proven", prompt="dusk",
        references=[{"asset_id": asset, "role": "geometry"}],
    )
    out = client.get(f"/api/renders/{body['job_id']}").json()
    assert out["intent"] is None and out["view"] is None and out["style"] is None
    assert out["line"] is None and out["geometry_hold"] is None


def test_the_view_defaults_to_the_intents_default_view(client, conn):
    asset = _upload(client)
    _, body = _submit(client, intent="make_real", references=[{"asset_id": asset, "role": "geometry"}])
    assert db.get_job(conn, body["job_id"])["task"] == "massing_to_render"
    assert _params(conn, body["job_id"])["view"] == "model"


def test_a_prompt_sent_with_an_intent_is_ignored_and_the_server_composes_one(client, conn):
    asset = _upload(client)
    _, body = _submit(
        client, intent="make_real", prompt="user typed this", style="night",
        references=[{"asset_id": asset, "role": "geometry"}],
    )
    prompt = _params(conn, body["job_id"])["prompt"]
    assert "user typed this" not in prompt
    assert prompt.endswith("dusk, interior lights on, cool sky, warm windows")


def test_change_takes_the_base_and_two_style_references(client, conn):
    base = _upload(client, "base.png")
    ref_a = _upload(client, "ref_a.png", width=80)
    ref_b = _upload(client, "ref_b.png", width=96)
    r, body = _submit(
        client, intent="change", style="golden", line="warmer, dusk",
        references=[
            {"asset_id": base, "role": "geometry"},
            {"asset_id": ref_a, "role": "style"},
            {"asset_id": ref_b, "role": "style"},
        ],
    )
    assert r.status_code == 200, r.text
    job = db.get_job(conn, body["job_id"])
    assert job["task"] == "variation"
    assert job["model_id"] == "fal-ai/nano-banana-pro/edit"
    params = _params(conn, body["job_id"])
    assert params["intent"] == "change"
    assert params["geometry_hold"] is None  # change has no slider: the hold is dropped
    assert params["num_images"] == 4
    assert len(db.job_inputs(conn, body["job_id"])) == 3


def test_change_refuses_an_extra_picture_that_is_not_a_style_reference(client):
    base = _upload(client, "base.png")
    other = _upload(client, "other.png", width=80)
    r = client.post("/api/renders", json={
        "intent": "change", "line": "warmer",
        "references": [
            {"asset_id": base, "role": "geometry"},
            {"asset_id": other, "role": "context"},
        ],
    })
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "The extra pictures for change something are style references."


def test_region_without_a_mask_is_refused_with_a_sentence(client):
    base = _upload(client)
    r = client.post("/api/renders", json={
        "intent": "region", "line": "a plain window",
        "references": [{"asset_id": base, "role": "geometry"}],
    })
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "Mark a region first."


def test_region_without_a_line_is_refused_with_a_sentence(client):
    base = _upload(client)
    r = client.post("/api/renders", json={
        "intent": "region",
        "references": [{"asset_id": base, "role": "geometry"}],
        "mask_asset_id": _mask_for(client, base),
    })
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "Say what the marked part becomes."


def test_region_with_a_mask_and_a_line_submits_on_the_masked_repair_engine(client, conn):
    base = _upload(client)
    r, body = _submit(
        client, intent="region", line="a plain sash window",
        references=[{"asset_id": base, "role": "geometry"}],
        mask_asset_id=_mask_for(client, base),
    )
    assert r.status_code == 200, r.text
    job = db.get_job(conn, body["job_id"])
    assert job["task"] == "inpaint"
    assert job["model_id"] == "fal-ai/flux-pro/v1/fill"


def test_a_mask_on_an_intent_that_takes_none_is_refused(client):
    base = _upload(client)
    r = client.post("/api/renders", json={
        "intent": "make_real",
        "references": [{"asset_id": base, "role": "geometry"}],
        "mask_asset_id": _mask_for(client, base),
    })
    assert r.status_code == 422
    assert "takes no mask" in r.json()["detail"]


def test_make_real_refuses_a_second_picture(client):
    base = _upload(client, "a.png")
    other = _upload(client, "b.png", width=80)
    r = client.post("/api/renders", json={
        "intent": "make_real",
        "references": [
            {"asset_id": base, "role": "geometry"},
            {"asset_id": other, "role": "style"},
        ],
    })
    assert r.status_code == 422
    assert r.json()["detail"] == "Make it real takes one picture. Remove the extra ones."


def test_an_unknown_intent_over_http_is_a_422_sentence(client):
    base = _upload(client)
    r = client.post("/api/renders", json={
        "intent": "sepia_all", "references": [{"asset_id": base, "role": "geometry"}],
    })
    assert r.status_code == 422
    assert "Unknown intent 'sepia_all'" in r.json()["detail"]


def test_an_unknown_view_over_http_is_a_422_sentence(client):
    base = _upload(client)
    r = client.post("/api/renders", json={
        "intent": "make_real", "view": "section",
        "references": [{"asset_id": base, "role": "geometry"}],
    })
    assert r.status_code == 422
    assert "'section' view" in r.json()["detail"]


def test_an_unknown_style_over_http_is_a_422_sentence(client):
    base = _upload(client)
    r = client.post("/api/renders", json={
        "intent": "make_real", "style": "sepia",
        "references": [{"asset_id": base, "role": "geometry"}],
    })
    assert r.status_code == 422
    assert "Unknown style 'sepia'" in r.json()["detail"]


def test_a_legacy_request_still_needs_a_prompt(client):
    base = _upload(client)
    r = client.post("/api/renders", json={
        "task": "massing_to_render", "references": [{"asset_id": base, "role": "geometry"}],
    })
    assert r.status_code == 422
    assert r.json()["detail"] == "Prompt is required."


def test_geometry_hold_is_clamped_stored_and_says_so_without_a_depth_model(client, conn):
    asset = _upload(client)
    _, body = _submit(
        client, intent="make_real", geometry_hold=100,
        references=[{"asset_id": asset, "role": "geometry"}],
    )
    # No depth model in the test settings, so Exact is capped at Hold and says so.
    assert _params(conn, body["job_id"])["geometry_hold"] == 50
    assert body["notes"] == [
        "Edges only: no depth model is installed, so Exact is capped at Hold."
    ]


def test_enlarge_via_continue_runs_upscale_with_one_result(client, conn, fake_settings):
    asset = _upload(client)
    _, first = _submit(
        client, intent="make_real", references=[{"asset_id": asset, "role": "geometry"}],
    )
    assert fal_gateway.poll(conn, fake_settings, first["job_id"]) == "completed"

    r = client.post(
        f"/api/renders/{first['job_id']}/continue",
        json={"intent": "enlarge", "num_images": 4, "style": "night", "line": "ignored"},
    )
    assert r.status_code == 200, r.text
    child = r.json()["job_id"]
    job = db.get_job(conn, child)
    assert job["task"] == "upscale"
    assert job["model_id"] == "fal-ai/seedvr/upscale/image"
    # seedvr has no num_images field, so the request carries none and the job is
    # priced for the one image it returns.
    assert job["est_usd"] == pytest.approx(0.001 * 1.2288)
    params = _params(conn, child)
    assert "num_images" not in params
    assert params["style"] is None and params["line"] is None
    assert params["intent"] == "enlarge"


def test_estimate_with_an_intent_defaults_to_four_images(client):
    r = client.post("/api/estimate", json={"intent": "make_real", "width": 1280, "height": 960})
    assert r.status_code == 200, r.text
    # flux-general at $0.075 per megapixel: 1280 x 960 is 1.2288 MP, times 4 results.
    assert r.json()["est_usd"] == pytest.approx(0.075 * 1.2288 * 4)
    assert r.json()["model_id"] == "fal-ai/flux-general"


def test_estimate_for_enlarge_is_one_result(client):
    r = client.post("/api/estimate", json={"intent": "enlarge", "width": 1280, "height": 960})
    assert r.status_code == 200, r.text
    assert r.json()["est_usd"] == pytest.approx(0.001 * 1.2288)


def test_estimate_refuses_a_bad_view_with_a_sentence(client):
    r = client.post("/api/estimate", json={"intent": "make_real", "view": "section"})
    assert r.status_code == 422
    assert "'section' view" in r.json()["detail"]


def test_retry_restores_the_intent_and_the_geometry_hold(client, conn, fake_settings):
    asset = _upload(client)
    _, first = _submit(
        client, intent="make_real", view="model", style="overcast", line="timber",
        geometry_hold=0, references=[{"asset_id": asset, "role": "geometry"}],
    )
    job_id = first["job_id"]
    assert _params(conn, job_id)["geometry_hold"] == 0

    # The next completion fails; the failed job is then retried.
    fal_gateway._fake_client = FakeFalClient(delay_s=0.0)
    fal_gateway._fake_client.fail_next = True
    assert fal_gateway.poll(conn, fake_settings, job_id) == "failed"

    r = client.post(f"/api/renders/{job_id}/retry")
    assert r.status_code == 200, r.text
    retry_id = r.json()["job_id"]
    params = _params(conn, retry_id)
    assert params["geometry_hold"] == 0
    assert params["intent"] == "make_real"
    assert params["view"] == "model"
    assert params["style"] == "overcast"
    assert params["line"] == "timber"
    assert params["num_images"] == 4
    assert db.get_job(conn, retry_id)["task"] == "massing_to_render"


def test_a_legacy_retry_stays_a_legacy_render(client, conn, fake_settings):
    asset = _upload(client)
    _, first = _submit(
        client, task="massing_to_render", engine="proven", prompt="dusk",
        references=[{"asset_id": asset, "role": "geometry"}],
    )
    fal_gateway._fake_client = FakeFalClient(delay_s=0.0)
    fal_gateway._fake_client.fail_next = True
    assert fal_gateway.poll(conn, fake_settings, first["job_id"]) == "failed"
    r = client.post(f"/api/renders/{first['job_id']}/retry")
    assert r.status_code == 200, r.text
    assert "intent" not in _params(conn, r.json()["job_id"])


# --- POST /api/assets/{id}/depth ---------------------------------------------

from app.models import UserStatus
from app.services import depth

DEPTH_MISSING = "No depth model is installed. Run scripts/fetch_depth_model.py on the server."


def _estimator_that_counts(monkeypatch, calls: list[Path], seconds: float = 1.5):
    """A stand-in estimator: writes a flat 8-bit PNG at the source size, counts its calls."""

    def fake_estimate(src, dst, settings, *, bits=16):
        calls.append(Path(src))
        with Image.open(src) as im:
            size = im.size
        Image.new("L", size, 180).save(dst, format="PNG")
        return depth.DepthResult(path=Path(dst), seconds=seconds)

    monkeypatch.setattr(depth, "is_available", lambda _settings: True)
    monkeypatch.setattr(depth, "estimate_depth", fake_estimate)


def test_depth_preview_is_503_with_the_sentence_when_no_model_is_installed(client):
    asset = _upload(client)
    r = client.post(f"/api/assets/{asset}/depth")
    assert r.status_code == 503, r.text
    assert r.json()["detail"] == DEPTH_MISSING


def test_depth_preview_returns_a_control_asset_whose_url_serves_the_depth_png(client, monkeypatch):
    calls: list[Path] = []
    _estimator_that_counts(monkeypatch, calls)
    asset = _upload(client, width=64, height=48)

    r = client.post(f"/api/assets/{asset}/depth")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["depth_source"] == "estimated"
    assert body["seconds"] == 1.5
    assert body["asset"]["kind"] == "control"
    assert (body["asset"]["width"], body["asset"]["height"]) == (64, 48)

    served = client.get(body["asset"]["url"])
    assert served.status_code == 200, served.text
    assert served.headers["content-type"] == "image/png"
    with Image.open(BytesIO(served.content)) as im:
        assert im.mode == "L"  # 8-bit, as the submit path writes it
        assert im.size == (64, 48)
    assert len(calls) == 1


def test_depth_preview_reuses_the_cache_and_the_same_control_asset(client, monkeypatch):
    calls: list[Path] = []
    _estimator_that_counts(monkeypatch, calls)
    asset = _upload(client)

    first = client.post(f"/api/assets/{asset}/depth").json()
    second = client.post(f"/api/assets/{asset}/depth")
    assert second.status_code == 200, second.text
    assert second.json()["asset"]["id"] == first["asset"]["id"]
    assert second.json()["seconds"] == 0.0  # cached: no inference ran
    assert len(calls) == 1


def test_a_preview_and_a_submit_share_one_depth_map(client, conn, monkeypatch, fake_settings):
    calls: list[Path] = []
    _estimator_that_counts(monkeypatch, calls)
    asset = _upload(client)
    previewed = client.post(f"/api/assets/{asset}/depth").json()["asset"]["id"]

    # The submit path has no way to ask for the preview's asset; it reuses the cache.
    from app.services import render as render_service

    row = db.get_asset(conn, asset)
    control = render_service.depth_control_for_asset(
        conn, fake_settings, row, project_id=None, uploaded_by=None,
    )
    assert control.cached is True
    assert control.asset_id == previewed
    assert len(calls) == 1


def test_depth_preview_refuses_a_mask_with_a_sentence(client, monkeypatch):
    _estimator_that_counts(monkeypatch, [])
    base = _upload(client)
    mask = _mask_for(client, base)
    r = client.post(f"/api/assets/{mask}/depth")
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "Depth can only be estimated from an uploaded view or a finished render."


def test_depth_preview_for_a_missing_asset_is_404(client, monkeypatch):
    _estimator_that_counts(monkeypatch, [])
    assert client.post("/api/assets/999999/depth").status_code == 404


def test_depth_preview_is_owner_or_admin_only(client, conn, fake_settings, monkeypatch):
    _estimator_that_counts(monkeypatch, [])
    asset = _upload(client)  # uploaded by the admin

    other = TestClient(app)
    r = other.post(
        "/api/auth/signup",
        json={"access_code": CODE, "email": "a@example.test", "name": "A", "password": PASSWORD},
    )
    assert r.status_code == 200, r.text
    db.set_user_status(conn, int(db.get_user_by_email(conn, "a@example.test")["id"]), UserStatus.ACTIVE)
    assert other.post("/api/auth/login", json={"email": "a@example.test", "password": PASSWORD}).status_code == 200

    r = other.post(f"/api/assets/{asset}/depth")
    assert r.status_code == 403, r.text
    assert "Only the person who uploaded this image" in r.json()["detail"]


def test_an_estimator_failure_is_a_500_sentence_not_a_crash(client, monkeypatch):
    def boom(src, dst, settings, *, bits=16):
        raise RuntimeError("onnx exploded")

    monkeypatch.setattr(depth, "is_available", lambda _settings: True)
    monkeypatch.setattr(depth, "estimate_depth", boom)
    asset = _upload(client)
    r = client.post(f"/api/assets/{asset}/depth")
    assert r.status_code == 500, r.text
    assert r.json()["detail"].startswith("The depth estimate failed.")


def test_depth_preview_is_rate_limited(client, monkeypatch):
    _estimator_that_counts(monkeypatch, [])
    asset = _upload(client)
    statuses = [client.post(f"/api/assets/{asset}/depth").status_code for _ in range(31)]
    assert statuses[:30] == [200] * 30
    assert statuses[30] == 429


# --- The old task listing keeps working until the web rewrite lands ----------


def test_get_tasks_still_lists_the_tasks_with_their_engines(client):
    r = client.get("/api/tasks")
    assert r.status_code == 200, r.text
    tasks = {t["key"]: t for t in r.json()}
    assert len(tasks) == 14
    massing = tasks["massing_to_render"]
    assert massing["default_engine"] == "proven"
    assert [e["key"] for e in massing["engines"]] == ["proven", "latest"]
    assert massing["engines"][0]["model_id"] == "fal-ai/flux-general"
    assert set(massing["engines"][0]) >= {"max_images", "supports_num_images", "needs_mask"}


def test_region_prompt_puts_the_words_inside_the_sentence_and_skips_the_style(table):
    text = table.compose_prompt("inpaint", "daylight", "a timber-slat facade")
    assert text.startswith("Built into this same building, a timber-slat facade, in the same light and perspective")
    assert "photograph," not in text and "true-to-material" not in text
    assert table.compose_prompt("inpaint", None, "") == table.compose_prompt("inpaint", "clay", None)
    # A bad style key is still refused even where the style text is not used.
    import pytest

    from app.routing import IntentError

    with pytest.raises(IntentError):
        table.compose_prompt("inpaint", "neon", "x")
