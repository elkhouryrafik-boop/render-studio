"""The depth model fetch pins a Hub revision and refuses weights that do not match it."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_depth_model.py"


def _load():
    spec = importlib.util.spec_from_file_location("fetch_depth_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_fetch_is_pinned_to_a_commit_and_a_model_hash():
    fetch = _load()
    assert len(fetch.REVISION) == 40
    assert fetch.REVISION in fetch.RESOLVE
    assert fetch.REVISION in fetch.API
    assert len(fetch.MODEL_SHA256["model.onnx"]) == 64


def test_a_model_with_the_pinned_hash_is_kept(tmp_path):
    fetch = _load()
    model = tmp_path / "model.onnx"
    model.write_bytes(b"tiny model")

    fetch.verify_sha256(model, hashlib.sha256(b"tiny model").hexdigest())

    assert model.read_bytes() == b"tiny model"


def test_a_model_with_another_hash_is_removed_and_the_run_stops(tmp_path):
    fetch = _load()
    model = tmp_path / "model.onnx"
    model.write_bytes(b"tampered model")

    with pytest.raises(SystemExit, match="does not match the pinned"):
        fetch.verify_sha256(model, hashlib.sha256(b"tiny model").hexdigest())

    assert not model.exists()
