"""CAD conversion: the DXF pipeline, and the upload/layers/preview/confirm
routes that put it in front of an architect before any fal.ai credit is spent.

Same rule as the rest of the suite — no network, no spend. Confirming a CAD
file only ever produces a local `upload` asset; nothing here reaches fal.ai.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import ezdxf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import deps
from api.main import app
from app import db
from app.config import Settings
from app.routing import RoutingTable
from app.services import cad_convert

ROUTING = Path(__file__).resolve().parents[1] / "routing.yaml"
CODE = "office-code-123"
PASSWORD = "correct-horse-battery"

# What a SketchUp 2013+ file starts with: a UTF-16LE "SketchUp Model" banner.
# `ingest_cad` checks it, so any test that uploads a .skp has to carry it.
SKP_BANNER = b"\xff\xfe\xff\x0e" + "SketchUp Model".encode("utf-16-le")


def _write_test_dxf(path: Path) -> None:
    """A simple floorplan: a walled room, a door on its own layer, dimensions.

    Deliberately not a fixture file checked into the repo — generating it with
    ezdxf means the test proves ezdxf can round-trip what it wrote, which is
    the same thing a real architect's export needs to survive.
    """
    doc = ezdxf.new("R2010")
    doc.layers.add("WALLS", color=7)
    doc.layers.add("DOORS", color=1)
    doc.layers.add("DIMS", color=3)
    msp = doc.modelspace()
    msp.add_lwpolyline(
        [(0, 0), (6000, 0), (6000, 4000), (0, 4000), (0, 0)],
        dxfattribs={"layer": "WALLS"},
    )
    msp.add_line((2000, 0), (2000, 900), dxfattribs={"layer": "DOORS"})
    msp.add_arc((2000, 0), radius=900, start_angle=0, end_angle=90, dxfattribs={"layer": "DOORS"})
    dim = msp.add_linear_dim(base=(0, -300), p1=(0, 0), p2=(6000, 0), dxfattribs={"layer": "DIMS"})
    dim.render()
    doc.saveas(str(path))


@pytest.fixture
def dxf_path(tmp_path: Path) -> Path:
    path = tmp_path / "floorplan.dxf"
    _write_test_dxf(path)
    return path


# --- The converter, directly --------------------------------------------------


def test_layers_are_read_without_rasterizing_anything(dxf_path: Path):
    conv = cad_convert.DxfConverter()
    names = {layer.name for layer in conv.layers(dxf_path)}
    assert {"WALLS", "DOORS", "DIMS"} <= names


def test_a_closed_room_layer_is_flagged_visible_with_its_colour(dxf_path: Path):
    conv = cad_convert.DxfConverter()
    by_name = {layer.name: layer for layer in conv.layers(dxf_path)}
    assert by_name["DOORS"].visible is True
    assert by_name["DOORS"].color == "#ff0000"


def test_convert_rasterizes_a_png_with_the_layer_manifest_attached(dxf_path: Path):
    conv = cad_convert.DxfConverter()
    result = conv.convert(dxf_path)
    assert result.view_png[:8] == b"\x89PNG\r\n\x1a\n"
    assert {layer.name for layer in result.layers} >= {"WALLS", "DOORS", "DIMS"}
    assert result.depth_png is None
    assert result.semantic_png is None
    assert result.camera is None
    assert result.warnings == []


def test_filtering_to_one_layer_produces_a_smaller_image(dxf_path: Path):
    conv = cad_convert.DxfConverter()
    full = conv.convert(dxf_path)
    walls_only = conv.convert(dxf_path, layers=["WALLS"])
    assert len(walls_only.view_png) != len(full.view_png)


def test_a_layer_name_that_does_not_exist_warns_but_still_converts(dxf_path: Path):
    conv = cad_convert.DxfConverter()
    result = conv.convert(dxf_path, layers=["WALLS", "NOPE"])
    assert "NOPE" in result.warnings[0]
    assert result.view_png


def test_selecting_only_layers_with_nothing_on_them_is_refused(dxf_path: Path):
    """Refused before rasterizing — an empty selection has no content extents
    for the page to auto-size against, which crashes deep inside PyMuPDF if
    it is not caught first."""
    conv = cad_convert.DxfConverter()
    with pytest.raises(cad_convert.CadConversionError, match="anything on them"):
        conv.convert(dxf_path, layers=["NOPE"])


def test_dxf_preview_is_svg_and_regenerates_per_layer_selection(dxf_path: Path):
    conv = cad_convert.DxfConverter()
    full = conv.preview(dxf_path)
    walls_only = conv.preview(dxf_path, layers=["WALLS"])
    assert full.startswith(b"<?xml")
    assert len(walls_only) != len(full)


def test_a_file_that_is_not_dxf_at_all_is_refused_locally(tmp_path: Path):
    garbage = tmp_path / "not-a-drawing.dxf"
    garbage.write_bytes(b"this is not a DXF file")
    conv = cad_convert.DxfConverter()
    with pytest.raises(cad_convert.CadConversionError, match="could not be read"):
        conv.layers(garbage)


def test_conversion_is_deterministic(dxf_path: Path):
    """Same file, same layers in, byte-identical PNG out.

    This is what makes re-confirming the same CAD upload content-addressed
    rather than a fresh asset each time — see the API-level idempotency test.
    """
    conv = cad_convert.DxfConverter()
    first = conv.convert(dxf_path)
    second = conv.convert(dxf_path)
    assert first.view_png == second.view_png


def _skp_settings(tmp_path: Path) -> Settings:
    return Settings(
        fal_key="test-key-not-real",
        office_access_code=CODE,
        admin_email="boss@example.test",
        data_dir=tmp_path,
        db_path=tmp_path / "skp.db",
        routing_path=ROUTING,
    )


def test_skp_bakes_in_the_background_and_reports_a_parse_failure(tmp_path: Path):
    """First call starts a background bake and returns `CadConversionPending`
    rather than blocking for however long a real parse takes; once that bake
    finishes, a file openskp cannot parse surfaces as `CadConversionError`
    instead of the caller waiting forever."""
    settings = _skp_settings(tmp_path)
    # `path.stem` stands in for the content hash `ingest_cad` derives — the
    # converter trusts that storage convention rather than re-hashing the file.
    stub_path = tmp_path / ("deadbeef" * 4 + ".skp")
    stub_path.write_bytes(b"not a real skp container")
    conv = cad_convert.SkpConverter(settings)

    with pytest.raises(cad_convert.CadConversionPending) as exc_info:
        conv.layers(stub_path)
    assert exc_info.value.stage
    assert exc_info.value.elapsed_seconds >= 0

    pkl_path, _status_path = cad_convert._skp_cache_paths(
        settings.skp_cache_dir(), cad_convert._skp_cache_key(stub_path)
    )
    thread = cad_convert._SKP_BAKE_THREADS.get(str(pkl_path))
    assert thread is not None, "expected a background bake thread to have started"
    thread.join(timeout=15)
    assert not thread.is_alive(), "bake did not finish in time"

    with pytest.raises(cad_convert.CadConversionError, match="could not be parsed"):
        conv.layers(stub_path)
    # And it stays failed rather than silently retrying forever.
    with pytest.raises(cad_convert.CadConversionError, match="could not be parsed"):
        conv.preview(stub_path)


def test_converter_for_refuses_an_unknown_suffix():
    with pytest.raises(ValueError, match="No CAD converter"):
        cad_convert.converter_for(".rvt")


# --- The HTTP routes ---------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_limiter():
    from app import ratelimit

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
        db_path=tmp_path / "api.db",
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


def _signed_in(client: TestClient) -> TestClient:
    r = client.post(
        "/api/auth/signup",
        json={
            "access_code": CODE, "email": "boss@example.test",
            "name": "Boss", "password": PASSWORD,
        },
    )
    assert r.status_code == 200, r.text
    return client


def test_uploading_a_dxf_creates_a_cad_upload_asset(client, dxf_path: Path):
    _signed_in(client)
    with open(dxf_path, "rb") as fh:
        r = client.post("/api/cad", files={"file": ("plan.dxf", fh, "application/dxf")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "cad_upload"
    assert body["width"] is None, "a CAD file is not an image, so it has no pixel dimensions"


def test_a_non_cad_extension_is_refused_at_upload(client):
    _signed_in(client)
    r = client.post("/api/cad", files={"file": ("plan.pdf", b"%PDF-1.7 nope", "application/pdf")})
    assert r.status_code == 400
    assert "CAD format" in r.json()["detail"]


def test_the_cad_gallery_excludes_the_raw_upload(client, conn, dxf_path: Path):
    """The original DXF is not itself a reference image — only what /confirm
    produces is."""
    _signed_in(client)
    project_id = db.create_project(conn, owner_id=None, name="Marina")
    with open(dxf_path, "rb") as fh:
        client.post(
            "/api/cad",
            files={"file": ("plan.dxf", fh, "application/dxf")},
            data={"project_id": str(project_id)},
        )
    gallery = client.get(f"/api/assets?project_id={project_id}").json()
    assert gallery == []


def test_layers_and_preview_round_trip_through_http(client, dxf_path: Path):
    _signed_in(client)
    with open(dxf_path, "rb") as fh:
        asset_id = client.post(
            "/api/cad", files={"file": ("plan.dxf", fh, "application/dxf")}
        ).json()["id"]

    layers = client.get(f"/api/cad/{asset_id}/layers")
    assert layers.status_code == 200
    names = {layer["name"] for layer in layers.json()}
    assert {"WALLS", "DOORS", "DIMS"} <= names

    preview = client.get(f"/api/cad/{asset_id}/preview")
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("image/svg+xml")

    filtered = client.get(f"/api/cad/{asset_id}/preview", params={"layers": "WALLS"})
    assert filtered.status_code == 200
    assert filtered.content != preview.content


def test_confirm_produces_a_normal_upload_asset(client, dxf_path: Path):
    _signed_in(client)
    with open(dxf_path, "rb") as fh:
        asset_id = client.post(
            "/api/cad", files={"file": ("plan.dxf", fh, "application/dxf")}
        ).json()["id"]

    r = client.post(f"/api/cad/{asset_id}/confirm", json={"layers": ["WALLS", "DOORS"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["asset"]["kind"] == "upload"
    assert body["asset"]["width"] is not None, "the confirmed PNG is a real image"
    assert body["warnings"] == []

    # The PNG is fetchable back through the normal asset image route, same as
    # any other upload — the CAD origin is invisible to the render pipeline
    # from here on.
    image = client.get(body["asset"]["url"])
    assert image.status_code == 200
    assert image.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_confirming_the_same_selection_twice_reuses_the_asset(client, dxf_path: Path):
    """Deterministic rasterization plus content-addressed ingest means
    confirming twice with the same layer selection is free the second time."""
    _signed_in(client)
    with open(dxf_path, "rb") as fh:
        asset_id = client.post(
            "/api/cad", files={"file": ("plan.dxf", fh, "application/dxf")}
        ).json()["id"]

    first = client.post(f"/api/cad/{asset_id}/confirm", json={}).json()
    second = client.post(f"/api/cad/{asset_id}/confirm", json={}).json()
    assert first["asset"]["id"] == second["asset"]["id"]


def test_confirming_with_an_empty_layer_selection_is_refused(client, dxf_path: Path):
    _signed_in(client)
    with open(dxf_path, "rb") as fh:
        asset_id = client.post(
            "/api/cad", files={"file": ("plan.dxf", fh, "application/dxf")}
        ).json()["id"]

    r = client.post(f"/api/cad/{asset_id}/confirm", json={"layers": ["NOPE"]})
    assert r.status_code == 400
    assert "anything on them" in r.json()["detail"]


def test_cad_routes_404_for_an_ordinary_image_asset(client):
    """An image asset id at a CAD route is a caller mistake, not a missing
    file — both read the same to the caller, so both are a 404."""
    _signed_in(client)
    buf = BytesIO()
    Image.new("RGB", (16, 16), "white").save(buf, format="PNG")
    asset_id = client.post(
        "/api/assets", files={"file": ("a.png", buf.getvalue(), "image/png")}
    ).json()["id"]

    assert client.get(f"/api/cad/{asset_id}/layers").status_code == 404
    assert client.post(f"/api/cad/{asset_id}/confirm", json={}).status_code == 404


def test_skp_upload_bakes_in_the_background_over_http(client, conn, settings: Settings):
    """Upload is accepted immediately; the first `/layers` call kicks off the
    background bake and answers 202 rather than blocking on it; once that
    bake finishes (here, fails — the upload is not a real SKP container),
    the same route answers 400 instead of hanging or retrying forever."""
    _signed_in(client)
    # A real SKP banner over nonsense: the point of this test is the bake
    # lifecycle, and the header check at ingest is a separate gate with its own
    # test below. Without the banner the upload is refused before a bake ever
    # starts, which would test the wrong thing.
    skp_bytes = SKP_BANNER + b"not a real skp container, but a valid upload candidate"
    r = client.post("/api/cad", files={"file": ("model.skp", skp_bytes, "application/octet-stream")})
    assert r.status_code == 200, r.text
    asset_id = r.json()["id"]
    assert r.json()["kind"] == "cad_upload"

    pending = client.get(f"/api/cad/{asset_id}/layers")
    assert pending.status_code == 202
    body = pending.json()["detail"]
    assert body["pending"] is True
    assert body["stage"]

    asset_row = db.get_asset(conn, asset_id)
    pkl_path, _status_path = cad_convert._skp_cache_paths(
        settings.skp_cache_dir(), cad_convert._skp_cache_key(Path(str(asset_row["local_path"])))
    )
    thread = cad_convert._SKP_BAKE_THREADS.get(str(pkl_path))
    assert thread is not None
    thread.join(timeout=15)
    assert not thread.is_alive(), "bake did not finish in time"

    failed = client.get(f"/api/cad/{asset_id}/layers")
    assert failed.status_code == 400
    assert "could not be parsed" in failed.json()["detail"]

    confirm = client.post(f"/api/cad/{asset_id}/confirm", json={})
    assert confirm.status_code == 400
    assert "could not be parsed" in confirm.json()["detail"]


# --- IFC (the Revit path) -----------------------------------------------------


def _write_test_ifc(path: Path) -> None:
    """Two storeys, one wall each — the smallest model that exercises storey
    filtering, and generated with ifcopenshell itself so the test proves the
    library round-trips what it wrote."""
    import ifcopenshell.api.aggregate
    import ifcopenshell.api.context
    import ifcopenshell.api.geometry
    import ifcopenshell.api.project
    import ifcopenshell.api.root
    import ifcopenshell.api.spatial
    import ifcopenshell.api.unit
    import numpy as np

    model = ifcopenshell.api.project.create_file(version="IFC4")
    project = ifcopenshell.api.root.create_entity(model, ifc_class="IfcProject", name="T")
    ifcopenshell.api.unit.assign_unit(model)
    ctx = ifcopenshell.api.context.add_context(model, context_type="Model")
    body = ifcopenshell.api.context.add_context(
        model, context_type="Model", context_identifier="Body",
        target_view="MODEL_VIEW", parent=ctx,
    )
    site = ifcopenshell.api.root.create_entity(model, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.root.create_entity(model, ifc_class="IfcBuilding", name="B")
    ground = ifcopenshell.api.root.create_entity(
        model, ifc_class="IfcBuildingStorey", name="Ground"
    )
    level1 = ifcopenshell.api.root.create_entity(
        model, ifc_class="IfcBuildingStorey", name="Level 1"
    )
    ifcopenshell.api.aggregate.assign_object(model, relating_object=project, products=[site])
    ifcopenshell.api.aggregate.assign_object(model, relating_object=site, products=[building])
    ifcopenshell.api.aggregate.assign_object(
        model, relating_object=building, products=[ground, level1]
    )

    for storey, length, z in ((ground, 6.0, 0.0), (level1, 4.0, 3.0)):
        wall = ifcopenshell.api.root.create_entity(model, ifc_class="IfcWall")
        rep = ifcopenshell.api.geometry.add_wall_representation(
            model, context=body, length=length, height=3.0, thickness=0.3
        )
        ifcopenshell.api.geometry.assign_representation(model, product=wall, representation=rep)
        matrix = np.eye(4)
        matrix[2][3] = z
        ifcopenshell.api.geometry.edit_object_placement(model, product=wall, matrix=matrix)
        ifcopenshell.api.spatial.assign_container(model, relating_structure=storey, products=[wall])

    model.write(str(path))


@pytest.fixture
def ifc_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.ifc"
    _write_test_ifc(path)
    return path


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_ifc_layers_are_the_building_storeys(ifc_path: Path):
    conv = cad_convert.IfcConverter()
    assert [layer.name for layer in conv.layers(ifc_path)] == ["Ground", "Level 1"]


def test_ifc_preview_is_a_png(ifc_path: Path):
    conv = cad_convert.IfcConverter()
    assert conv.preview(ifc_path).startswith(PNG_MAGIC)


def test_ifc_convert_produces_view_and_true_depth(ifc_path: Path):
    conv = cad_convert.IfcConverter()
    result = conv.convert(ifc_path, camera="iso")
    assert result.view_png.startswith(PNG_MAGIC)
    assert result.depth_png is not None and result.depth_png.startswith(PNG_MAGIC)
    assert result.camera is not None and result.camera.preset == "iso"
    assert result.warnings == []


def test_ifc_depth_follows_the_controlnet_convention(ifc_path: Path):
    """8-bit, near = bright, empty background = 0 — what flux-general's depth
    control reads. Wrong convention here silently inverts the conditioning."""
    from io import BytesIO as _BytesIO

    import numpy as np

    conv = cad_convert.IfcConverter()
    result = conv.convert(ifc_path)
    depth = np.array(Image.open(_BytesIO(result.depth_png)))
    assert depth.dtype == np.uint8
    assert depth[0, 0] == 0, "background must be far (black)"
    assert depth.max() == 255, "the nearest surface must use the full range"


def test_ifc_storey_filter_changes_the_image(ifc_path: Path):
    conv = cad_convert.IfcConverter()
    everything = conv.convert(ifc_path)
    ground_only = conv.convert(ifc_path, layers=["Ground"])
    assert ground_only.view_png != everything.view_png


def test_ifc_camera_presets_produce_different_views(ifc_path: Path):
    conv = cad_convert.IfcConverter()
    iso = conv.convert(ifc_path, camera="iso")
    top = conv.convert(ifc_path, camera="top")
    assert iso.view_png != top.view_png


def test_ifc_unknown_storey_warns_but_still_converts(ifc_path: Path):
    conv = cad_convert.IfcConverter()
    result = conv.convert(ifc_path, layers=["Ground", "Penthouse"])
    assert any("Penthouse" in w for w in result.warnings)
    assert result.view_png


def test_ifc_empty_storey_selection_is_refused(ifc_path: Path):
    conv = cad_convert.IfcConverter()
    with pytest.raises(cad_convert.CadConversionError, match="anything on them"):
        conv.convert(ifc_path, layers=["Penthouse"])


def test_ifc_unknown_camera_is_refused(ifc_path: Path):
    conv = cad_convert.IfcConverter()
    with pytest.raises(cad_convert.CadConversionError, match="camera preset"):
        conv.convert(ifc_path, camera="drone")


def test_ifc_garbage_is_refused_locally(tmp_path: Path):
    garbage = tmp_path / "not-a-model.ifc"
    garbage.write_bytes(b"this is not an IFC STEP file")
    conv = cad_convert.IfcConverter()
    with pytest.raises(cad_convert.CadConversionError, match="could not be read"):
        conv.layers(garbage)


def test_ifc_conversion_is_deterministic(ifc_path: Path):
    """Same file, same storeys, same camera → byte-identical PNGs, which is
    what makes re-confirming content-addressed instead of a fresh asset."""
    conv = cad_convert.IfcConverter()
    first = conv.convert(ifc_path)
    second = conv.convert(ifc_path)
    assert first.view_png == second.view_png
    assert first.depth_png == second.depth_png


# --- IFC over HTTP ------------------------------------------------------------


def test_ifc_round_trip_through_http(client, ifc_path: Path):
    _signed_in(client)
    with open(ifc_path, "rb") as fh:
        r = client.post("/api/cad", files={"file": ("model.ifc", fh, "application/octet-stream")})
    assert r.status_code == 200, r.text
    asset_id = r.json()["id"]
    assert r.json()["kind"] == "cad_upload"

    layers = client.get(f"/api/cad/{asset_id}/layers")
    assert [x["name"] for x in layers.json()] == ["Ground", "Level 1"]

    preview = client.get(f"/api/cad/{asset_id}/preview", params={"camera": "front"})
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("image/png")

    bad_camera = client.get(f"/api/cad/{asset_id}/preview", params={"camera": "drone"})
    assert bad_camera.status_code == 400

    r = client.post(f"/api/cad/{asset_id}/confirm", json={"camera": "iso"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["asset"]["kind"] == "upload"
    assert body["depth_asset"]["kind"] == "control"
    assert body["camera"] == "iso"

    # Both images are fetchable back through the normal asset route.
    assert client.get(body["asset"]["url"]).status_code == 200
    assert client.get(body["depth_asset"]["url"]).status_code == 200


def test_the_depth_asset_stays_out_of_the_gallery(client, conn, ifc_path: Path):
    _signed_in(client)
    project_id = db.create_project(conn, owner_id=None, name="IFC")
    with open(ifc_path, "rb") as fh:
        asset_id = client.post(
            "/api/cad",
            files={"file": ("model.ifc", fh, "application/octet-stream")},
            data={"project_id": str(project_id)},
        ).json()["id"]
    body = client.post(f"/api/cad/{asset_id}/confirm", json={"project_id": project_id}).json()

    gallery = client.get(f"/api/assets?project_id={project_id}").json()
    kinds = {a["id"]: a["kind"] for a in gallery}
    assert body["asset"]["id"] in kinds, "the confirmed view belongs in the gallery"
    assert body["depth_asset"]["id"] not in kinds, "the depth pass does not"


def test_reconfirming_the_same_ifc_view_reuses_both_assets(client, ifc_path: Path):
    _signed_in(client)
    with open(ifc_path, "rb") as fh:
        asset_id = client.post(
            "/api/cad", files={"file": ("model.ifc", fh, "application/octet-stream")}
        ).json()["id"]
    first = client.post(f"/api/cad/{asset_id}/confirm", json={"camera": "iso"}).json()
    second = client.post(f"/api/cad/{asset_id}/confirm", json={"camera": "iso"}).json()
    assert first["asset"]["id"] == second["asset"]["id"]
    assert first["depth_asset"]["id"] == second["depth_asset"]["id"]


def test_control_assets_reach_the_depth_entry_end_to_end(
    client, conn, ifc_path: Path, monkeypatch
):
    """The whole point of the feature: an IFC confirm's depth pass, sent as
    control_assets, must land on flux-general's depth control while canny keeps
    reading the colour view. Both are controls inside the single
    controlnet_unions entry (2026-08-27 — this endpoint cannot carry
    easycontrols alongside it)."""
    from app import fal_gateway

    submitted: list[tuple[str, dict]] = []

    class Handle:
        request_id = "req-ifc-1"

    def fake_submit(application, arguments=None, **_kw):
        submitted.append((application, arguments or {}))
        return Handle()

    monkeypatch.setattr(fal_gateway.fal_client, "submit", fake_submit)
    monkeypatch.setattr(
        fal_gateway, "upload_input", lambda _s, path: f"https://fal.test/{Path(path).name}"
    )

    _signed_in(client)
    with open(ifc_path, "rb") as fh:
        cad_id = client.post(
            "/api/cad", files={"file": ("model.ifc", fh, "application/octet-stream")}
        ).json()["id"]
    body = client.post(f"/api/cad/{cad_id}/confirm", json={"camera": "iso"}).json()

    r = client.post(
        "/api/renders",
        json={
            "task": "massing_to_render",
            "engine": "proven",
            "prompt": "warm evening light",
            "references": [{"asset_id": body["asset"]["id"], "role": "geometry"}],
            "control_assets": {"depth": body["depth_asset"]["id"]},
            "num_images": 1,
        },
    )
    assert r.status_code == 200, r.text

    _application, arguments = submitted[0]
    assert "easycontrols" not in arguments
    controls = arguments["controlnet_unions"][0]["controls"]
    urls = {c["control_mode"]: c["control_image_url"] for c in controls}
    assert set(urls) == {"depth", "canny"}
    assert urls["depth"] != urls["canny"], "depth must condition on its own image"


# --- What a CAD upload is allowed to be --------------------------------------
#
# The extension alone used to decide, and the file then went straight to ezdxf,
# ifcopenshell or openskp — C parsers that were never meant to be the first
# thing a hostile file meets. A header check does not make them safe; it means
# they are only asked to parse files that begin like the format they handle.


def test_a_dxf_that_is_not_a_dxf_is_refused_before_the_parser_sees_it(client, conn):
    _signed_in(client)
    r = client.post(
        "/api/cad",
        files={"file": ("plan.dxf", b"MZ\x90\x00" + b"\x00" * 500, "application/dxf")},
    )
    assert r.status_code == 400
    assert "DXF header" in r.json()["detail"]


def test_an_ifc_that_is_not_an_ifc_is_refused(client, conn):
    _signed_in(client)
    r = client.post(
        "/api/cad",
        files={"file": ("model.ifc", b"just some text", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "ISO-10303-21" in r.json()["detail"]


def test_an_skp_that_is_not_an_skp_is_refused(client, conn):
    _signed_in(client)
    r = client.post(
        "/api/cad",
        files={"file": ("model.skp", b"PK\x03\x04nope", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "SketchUp" in r.json()["detail"]


def test_a_real_dxf_still_uploads(client, conn, tmp_path: Path):
    """The check has to admit what ezdxf writes, or it is just an outage."""
    _signed_in(client)
    path = tmp_path / "real.dxf"
    _write_test_dxf(path)
    with path.open("rb") as fh:
        r = client.post("/api/cad", files={"file": ("real.dxf", fh, "application/dxf")})
    assert r.status_code == 200, r.text


def test_an_image_named_dxf_is_refused_by_its_bytes(client, conn):
    _signed_in(client)
    buf = BytesIO()
    Image.new("RGB", (16, 16), "white").save(buf, format="PNG")
    r = client.post(
        "/api/cad", files={"file": ("plan.dxf", buf.getvalue(), "application/dxf")}
    )
    assert r.status_code == 400


# --- The SKP bake cache ------------------------------------------------------


def test_the_bake_cache_is_not_a_pickle(tmp_path: Path):
    """`pickle.load` on a file in DATA_DIR is code execution in the server
    process for anyone who can write to that directory. The replacement holds
    arrays and one JSON blob, and refuses pickles on the way back in."""
    import numpy as np

    payload = {
        "primitives": [
            {
                "positions": np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
                "indices": np.array([0, 1, 2], dtype=np.int64),
                "material_index": 3,
                "geom_name": "mesh_WALLS",
            }
        ],
        "gltf_materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.5, 0.5, 1.0]}}],
        "layer_by_geom": {"mesh_WALLS": "WALLS"},
        "names_by_color": {(128, 128, 128): ["Concrete", "Layer_WALLS"]},
        "texture_by_name": {"Concrete": ("concrete.jpg", 610.0)},
        "layers": ["WALLS", "Layer0"],
    }

    cache_path, _status = cad_convert._skp_cache_paths(tmp_path, "abc123")
    cad_convert._skp_write_payload(payload, cache_path)
    assert cache_path.suffix == ".npz"
    # npz is a zip of .npy members. A pickle would start with a protocol
    # opcode, and np.load(allow_pickle=False) is what refuses one.
    assert cache_path.read_bytes()[:2] == b"PK"

    back = cad_convert._skp_read_payload(cache_path)
    assert back["layers"] == ["WALLS", "Layer0"]
    assert back["layer_by_geom"] == {"mesh_WALLS": "WALLS"}
    # The two shapes JSON cannot hold directly have to survive exactly: an RGB
    # tuple key is what the material join looks a colour up by, and the texture
    # pair is unpacked into two values by its caller.
    assert back["names_by_color"][(128, 128, 128)] == ["Concrete", "Layer_WALLS"]
    assert back["texture_by_name"]["Concrete"] == ("concrete.jpg", 610.0)

    prim = back["primitives"][0]
    assert prim["material_index"] == 3
    assert prim["geom_name"] == "mesh_WALLS"
    np.testing.assert_array_equal(prim["positions"], payload["primitives"][0]["positions"])
    np.testing.assert_array_equal(prim["indices"], payload["primitives"][0]["indices"])
