"""The synthetic sample inputs load through the asset and CAD APIs.

`examples/make_sample.py` is run into a temp dir (not the committed
`examples/inputs`), so a change to the generator is what gets tested.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

# Fixtures (settings, conn, client) are shared with the CAD tests.
from tests.test_cad import _signed_in, client, conn, settings  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNGS = ["massing-aerial.png", "massing-street.png", "sketch-street.png", "elevation-sample.png"]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "make_sample", ROOT / "examples" / "make_sample.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def samples(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("samples")
    assert _load_generator().main(["--out", str(out)]) == 0
    return out


def test_generator_writes_every_sample(samples: Path):
    for name in [*PNGS, "plan-sample.dxf", "model-sample.ifc"]:
        assert (samples / name).stat().st_size > 1000, name


@pytest.mark.parametrize("name", PNGS)
def test_pngs_are_1536x1024_and_carry_no_metadata(samples: Path, name: str):
    with Image.open(samples / name) as im:
        assert im.size == (1536, 1024)
        assert im.format == "PNG"
        assert not im.getexif()
        assert set(im.info) <= {"dpi", "gamma", "transparency"}


def test_pngs_are_deterministic(samples: Path, tmp_path: Path):
    _load_generator().main(["--out", str(tmp_path)])
    for name in [*PNGS, "plan-sample.dxf"][:4]:
        assert (samples / name).read_bytes() == (tmp_path / name).read_bytes(), name


@pytest.mark.parametrize("name", PNGS)
def test_png_samples_upload_as_assets(client, samples: Path, name: str):
    _signed_in(client)
    with open(samples / name, "rb") as fh:
        r = client.post("/api/assets", files={"file": (name, fh, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["width"], body["height"]) == (1536, 1024)
    image = client.get(body["url"])
    assert image.status_code == 200
    assert image.content[:8] == PNG_MAGIC


def test_dxf_sample_has_the_documented_layers_and_converts(client, samples: Path):
    _signed_in(client)
    with open(samples / "plan-sample.dxf", "rb") as fh:
        asset_id = client.post(
            "/api/cad", files={"file": ("plan-sample.dxf", fh, "application/dxf")}
        ).json()["id"]
    names = {layer["name"] for layer in client.get(f"/api/cad/{asset_id}/layers").json()}
    assert {"WALLS", "DOORS", "WINDOWS", "TEXT", "DIMS"} <= names
    assert client.get(f"/api/cad/{asset_id}/preview").status_code == 200
    r = client.post(f"/api/cad/{asset_id}/confirm", json={})
    assert r.status_code == 200, r.text
    assert r.json()["asset"]["width"] is not None


def test_dxf_sample_contains_the_synthetic_title_and_no_header_names(samples: Path):
    import ezdxf

    doc = ezdxf.readfile(samples / "plan-sample.dxf")
    texts = [e.dxf.text for e in doc.modelspace().query("TEXT")]
    assert "SAMPLE PLAN — SYNTHETIC" in texts
    for var in ("$PROJECTNAME", "$TITLE", "$AUTHOR", "$COMPANY"):
        assert not doc.header.get(var)


def test_ifc_sample_is_a_single_storey_and_converts(client, samples: Path):
    import ifcopenshell

    model = ifcopenshell.open(str(samples / "model-sample.ifc"))
    assert model.by_type("IfcProject")[0].Name == "Synthetic sample"
    assert len(model.by_type("IfcBuildingStorey")) == 1
    assert len(model.by_type("IfcWall")) == 4
    assert len(model.by_type("IfcSlab")) == 1

    _signed_in(client)
    with open(samples / "model-sample.ifc", "rb") as fh:
        asset_id = client.post(
            "/api/cad", files={"file": ("model-sample.ifc", fh, "application/octet-stream")}
        ).json()["id"]
    layers = client.get(f"/api/cad/{asset_id}/layers")
    assert layers.status_code == 200
    assert [layer["name"] for layer in layers.json()] == ["Ground"]
    r = client.post(f"/api/cad/{asset_id}/confirm", json={"camera": "iso"})
    assert r.status_code == 200, r.text
