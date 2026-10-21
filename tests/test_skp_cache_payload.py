"""The baked-SKP cache: exact round trip, and not pathologically large.

Replacing `pickle` with `.npz` closed a code-execution hole (anyone able to
write to DATA_DIR could have run code in the server) but shipped a serious
performance regression, because the first layout wrote one zip member per
array. A .npz is a zip, and every member costs a local header, a central
directory entry and a 128-byte .npy header.

Measured on a ~27MB SketchUp test model, which bakes to 52,017 primitives:

    pickle (what it replaced)          28.8 MB    read       ?
    npz, one member per array          98.1 MB    read  101.0 s
    npz, concatenated + offsets        65.8 MB    read    0.2 s
    npz, concatenated + compressed      4.5 MB    read    0.3 s

The last row is what is in the tree. It is 6x smaller than the pickle and the
cache load stopped being the bottleneck: one `convert()` on that model went
from 76.6s to 6.4s.

These tests keep all three properties: the round trip is exact, the container
does not grow a member per primitive, and a payload written today is readable.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from app.services import cad_convert


def _payload(n_primitives: int = 400) -> dict:
    rng = np.random.default_rng(7)
    primitives = []
    for i in range(n_primitives):
        verts = rng.random((rng.integers(3, 30), 3), dtype=np.float32)
        primitives.append(
            {
                "positions": verts,
                "indices": rng.integers(0, len(verts), size=(len(verts) // 3) * 3).astype(
                    np.int32
                ),
                "material_index": int(i % 11),
                "geom_name": f"group_{i}#instance",
            }
        )
    return {
        "primitives": primitives,
        "gltf_materials": [
            {"name": "Blacktop Old 01", "baseColorFactor": [0.52, 0.53, 0.55, 1.0]},
            {"name": "Water Pool Light", "baseColorFactor": [0.25, 0.58, 0.65, 1.0]},
        ],
        "layer_by_geom": {f"group_{i}#instance": "ASPHALT" for i in range(n_primitives)},
        # An RGB tuple key and a 2-tuple value: the two shapes JSON cannot hold
        # directly, and therefore the two most likely to break on a round trip.
        "names_by_color": {(12, 200, 255): ["Water Pool Light"], (0, 0, 0): ["Blacktop New"]},
        "texture_by_name": {"Blacktop Old 01": ("blacktop.jpg", 610.0)},
        "layers": ["ASPHALT", "CONCRET", "POOL UP", "0 BLOCK-A GLASS"],
    }


def test_the_round_trip_is_exact(tmp_path):
    original = _payload()
    path = tmp_path / "scene.npz"
    cad_convert._skp_write_payload(original, path)
    restored = cad_convert._skp_read_payload(path)

    assert len(restored["primitives"]) == len(original["primitives"])
    for before, after in zip(original["primitives"], restored["primitives"]):
        assert np.array_equal(before["positions"], after["positions"])
        assert np.array_equal(before["indices"], after["indices"])
        assert before["material_index"] == after["material_index"]
        assert before["geom_name"] == after["geom_name"]

    for key in ("gltf_materials", "layer_by_geom", "names_by_color", "texture_by_name", "layers"):
        assert restored[key] == original[key], key


def test_non_latin_layer_names_survive(tmp_path):
    """Layer names may be non-Latin."""
    payload = _payload(4)
    payload["layers"] = ["0 زجاج", "0 BLOCK-A STONE", "0 BRICK"]
    path = tmp_path / "arabic.npz"
    cad_convert._skp_write_payload(payload, path)
    assert cad_convert._skp_read_payload(path)["layers"] == payload["layers"]


def test_the_container_does_not_grow_a_member_per_primitive(tmp_path):
    """The regression that made the cache 98MB and took 101s to read."""
    path = tmp_path / "scene.npz"
    cad_convert._skp_write_payload(_payload(400), path)
    with zipfile.ZipFile(path) as bundle:
        members = bundle.namelist()
    assert len(members) <= 8, (
        f"{len(members)} zip members for 400 primitives; the layout is per-primitive again"
    )


def test_the_payload_is_not_a_pickle(tmp_path):
    """`allow_pickle=False` is the whole point of the format."""
    path = tmp_path / "scene.npz"
    cad_convert._skp_write_payload(_payload(20), path)
    with np.load(path, allow_pickle=False) as bundle:
        assert set(bundle.files) >= {"positions", "indices", "meta"}
        assert json.loads(str(bundle["meta"]))["layout"] == 2


def test_indices_stay_narrow(tmp_path):
    """int32, not int64. They address vertices inside one primitive."""
    path = tmp_path / "scene.npz"
    cad_convert._skp_write_payload(_payload(20), path)
    with np.load(path, allow_pickle=False) as bundle:
        assert bundle["indices"].dtype == np.int32
        assert bundle["positions"].dtype == np.float32


def test_an_empty_scene_round_trips(tmp_path):
    payload = _payload(0)
    path = tmp_path / "empty.npz"
    cad_convert._skp_write_payload(payload, path)
    assert cad_convert._skp_read_payload(path)["primitives"] == []


def test_the_write_is_atomic(tmp_path):
    """A poller reading mid-write must never see a truncated cache."""
    path = tmp_path / "scene.npz"
    cad_convert._skp_write_payload(_payload(20), path)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp")), "a temporary file was left behind"


@pytest.mark.parametrize("count", [1, 3, 250])
def test_offsets_slice_every_primitive_back_out(tmp_path, count):
    original = _payload(count)
    path = tmp_path / f"n{count}.npz"
    cad_convert._skp_write_payload(original, path)
    restored = cad_convert._skp_read_payload(path)
    assert [len(p["positions"]) for p in restored["primitives"]] == [
        len(p["positions"]) for p in original["primitives"]
    ]
    assert [len(p["indices"]) for p in restored["primitives"]] == [
        len(p["indices"]) for p in original["primitives"]
    ]
