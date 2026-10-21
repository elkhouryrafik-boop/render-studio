"""Acceptance metric M1 (scripts/acceptance_m1.py): the scoring maths and the CLI.

The pure-function tests use synthetic depth maps, so they run with no model.
The CLI smoke test builds a temporary database with the real migrations and
monkeypatches the depth estimator, so it also needs no model.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import acceptance_m1 as m1

from app.db import connect
from app.migrations import migrate
from app.services import depth


def _scene(h: int = 64, w: int = 80) -> np.ndarray:
    """Background plus two rectangles: clear depth steps, so there are real edges."""
    d = np.full((h, w), 0.2)
    d[10:30, 10:40] = 0.8
    d[35:60, 45:75] = 0.5
    return d


# --- scoring maths ------------------------------------------------------------


def test_identical_maps_score_zero_rmse_and_perfect_edges():
    d = _scene()
    valid = np.ones(d.shape, dtype=bool)
    out = m1.score_pair(d, d, valid)
    assert out["rmse_aligned"] == pytest.approx(0.0, abs=1e-9)
    assert out["edge_f1"] == 1.0
    assert out["scale"] == pytest.approx(1.0)
    assert out["shift"] == pytest.approx(0.0, abs=1e-9)


def test_fit_recovers_known_scale_and_shift():
    d = _scene()
    s, t = m1.fit_scale_shift(0.5 * d + 0.2, d, np.ones(d.shape, dtype=bool))
    assert s == pytest.approx(2.0)
    assert t == pytest.approx(-0.4)


def test_scaled_and_shifted_copy_scores_zero_after_alignment():
    d = _scene()
    valid = np.ones(d.shape, dtype=bool)
    # Depth from a monocular model is only known up to scale and shift; the
    # metric must not punish that.
    out = m1.score_pair(d, 0.5 * d + 0.2, valid)
    assert out["rmse_aligned"] == pytest.approx(0.0, abs=1e-9)
    assert out["edge_f1"] == 1.0


def test_shuffled_map_scores_high_rmse_and_low_edge_f1():
    d = _scene()
    valid = np.ones(d.shape, dtype=bool)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(d.ravel()).reshape(d.shape)
    out = m1.score_pair(d, shuffled, valid)
    assert out["rmse_aligned"] > 0.2
    # Random edges are dense, so recall stays moderate even though precision is
    # poor; the F1 lands well under the 1.0 of an identical pair. Bound is 0.5.
    assert out["edge_f1"] < 0.5


def test_invalid_pixels_are_excluded_from_the_fit_and_the_error():
    d = _scene()
    cad = d.copy()
    cad[0:8, :] = 0.0  # CAD background: zero means "no surface"
    valid = cad > 0
    out_map = 0.5 * d + 0.2
    out_map[0:8, :] = 9.9  # garbage where the input has no depth

    masked = m1.score_pair(cad, out_map, valid)
    assert masked["rmse_aligned"] == pytest.approx(0.0, abs=1e-9)
    assert masked["n_valid"] == int(valid.sum())

    # The same pair without the mask is visibly wrong, so the mask is doing work.
    unmasked = m1.score_pair(cad, out_map, np.ones(d.shape, dtype=bool))
    assert unmasked["rmse_aligned"] > 0.1


def test_edge_f1_tolerates_two_pixels_but_not_three():
    ref = np.zeros((40, 40), dtype=bool)
    ref[:, 20] = True
    two = np.zeros_like(ref)
    two[:, 22] = True
    three = np.zeros_like(ref)
    three[:, 23] = True
    assert m1.edge_f1(two, ref, tolerance=2) == 1.0
    assert m1.edge_f1(three, ref, tolerance=2) == 0.0


def test_edge_f1_both_empty_is_one_and_one_empty_is_zero():
    empty = np.zeros((10, 10), dtype=bool)
    full = np.zeros((10, 10), dtype=bool)
    full[5, 5] = True
    assert m1.edge_f1(empty, empty) == 1.0
    assert m1.edge_f1(empty, full) == 0.0


def test_pair_is_resized_to_the_smaller_common_size():
    a = np.full((100, 120), 0.4)
    b = np.full((80, 150), 0.6)
    d_in, valid, d_out = m1.prepare_pair(a, np.ones(a.shape, dtype=bool), b)
    assert d_in.shape == (80, 120)
    assert d_out.shape == (80, 120)
    assert valid.shape == (80, 120)


def test_stored_relative_paths_resolve_against_the_repo_root(tmp_path):
    data = tmp_path / "data"
    (data / "outputs").mkdir(parents=True)
    target = data / "outputs" / "job000001_0.png"
    target.write_bytes(b"x")
    # The office DB stores Windows-relative paths such as data\outputs\...
    assert m1.resolve_stored_path("data\\outputs\\job000001_0.png", data) == target
    assert m1.resolve_stored_path(str(target), data) == target
    assert m1.resolve_stored_path("data\\outputs\\missing.png", data) is None
    assert m1.resolve_stored_path(None, data) is None


def test_open_readonly_refuses_writes(tmp_path):
    db = tmp_path / "ro.db"
    seed = connect(db)
    migrate(seed)
    seed.close()
    conn = m1.open_readonly(db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("CREATE TABLE should_not_exist (x INTEGER)")
    conn.close()


# --- CLI smoke test -------------------------------------------------------------


def _png(path: Path, arr: np.ndarray, mode: str = "L") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode=mode).save(path)
    return path


def _seed_db(db: Path, data: Path) -> None:
    conn = connect(db)
    migrate(conn)
    scene = _scene(120, 160)
    rgb = np.stack([scene * 255] * 3, axis=-1)
    geometry = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    geo_path = data / "inputs" / "geo.png"
    geo_path.parent.mkdir(parents=True, exist_ok=True)
    geometry.save(geo_path)
    cad_path = _png(data / "inputs" / "cad_depth.png", scene * 255)
    out_path = _png(data / "outputs" / "job000001_0.png", (scene[::-1] * 255) * 0.5 + 60)
    out_path_2 = _png(data / "outputs" / "job000002_0.png", (scene * 255))

    conn.execute(
        "INSERT INTO assets (id, sha256, local_path, fal_url, kind, created_at) "
        "VALUES (1, 'a', ?, NULL, 'upload', '2026-10-01T00:00:00Z')",
        (str(geo_path),),
    )
    conn.execute(
        "INSERT INTO assets (id, sha256, local_path, fal_url, kind, created_at) "
        "VALUES (2, 'b', ?, 'https://example.test/cad_depth.png', 'control', "
        "'2026-10-01T00:00:00Z')",
        (str(cad_path),),
    )

    def job(job_id, model, status, out, params, depth_source):
        conn.execute(
            "INSERT INTO jobs (id, idempotency_key, username, task, model_id, status, "
            "est_usd, params_json, output_path, created_at, updated_at, engine, depth_source) "
            "VALUES (?, ?, 'u', 'massing_to_render', ?, ?, 0, ?, ?, "
            "'2026-10-02T00:00:00Z', '2026-10-02T00:00:00Z', 'flux-general', ?)",
            (job_id, f"k{job_id}", model, status, json.dumps(params), out, depth_source),
        )
        conn.execute(
            "INSERT INTO job_inputs (job_id, asset_id, role, position) VALUES (?, 1, 'geometry', 1)",
            (job_id,),
        )

    # Estimated depth: no controlnet URL, the column says estimated.
    job(1, "fal-ai/flux-general", "completed", str(out_path), {"num_images": 1}, "estimated")
    # CAD depth: the control URL matches the CAD asset's fal_url.
    job(
        2,
        "fal-ai/flux-general/differential-diffusion",
        "completed",
        str(out_path_2),
        {
            "num_images": 1,
            "controlnet_unions": [
                {"path": "x", "controls": [{"control_image_url": "https://example.test/cad_depth.png"}]}
            ],
        },
        "cad",
    )
    # Not completed: must not be selected at all.
    job(3, "fal-ai/flux-general", "failed", str(out_path), {}, None)
    # Completed but the output file is gone: must be listed as skipped.
    job(4, "fal-ai/flux-general", "completed", str(data / "outputs" / "gone.png"), {}, None)
    conn.close()


def _fake_estimate(src, dst, settings, *, bits=16):
    """Stands in for Depth Anything: a deterministic 8-bit map, no model needed."""
    with Image.open(src) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    norm = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    Image.fromarray(np.rint(norm * 255).astype(np.uint8), mode="L").save(dst, format="PNG")
    return depth.DepthResult(path=Path(dst))


def test_cli_smoke_on_a_temp_database(tmp_path, monkeypatch):
    data = tmp_path / "data"
    db = data / "render_studio.db"
    _seed_db(db, data)
    monkeypatch.setattr(depth, "is_available", lambda settings: True)
    monkeypatch.setattr(depth, "estimate_depth", _fake_estimate)
    out_dir = tmp_path / "acceptance"

    rc = m1.main([
        "--db", str(db),
        "--data-dir", str(data),
        "--out-dir", str(out_dir),
        "--date", "2026-10-10",
    ])
    assert rc == 0

    with (out_dir / "2026-10-10-m1.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    scored = {int(r["job_id"]): r for r in rows}
    assert set(scored) == {1, 2}
    assert scored[1]["depth_source_in"] == "estimated"
    assert scored[2]["depth_source_in"] == "cad"
    for r in rows:
        assert 0.0 <= float(r["rmse_aligned"]) <= 2.0
        assert 0.0 <= float(r["edge_f1"]) <= 1.0

    md = (out_dir / "2026-10-10-baseline.md").read_text(encoding="utf-8")
    assert "## Summary by model" in md
    assert "fal-ai/flux-general/differential-diffusion" in md
    assert "## Skipped" in md
    assert "gone.png" in md or "output file not on disk" in md
    assert "Lower RMSE" in md


def test_cli_skips_cleanly_without_the_depth_model(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    db = data / "render_studio.db"
    _seed_db(db, data)
    monkeypatch.setattr(depth, "is_available", lambda settings: False)
    out_dir = tmp_path / "acceptance"

    rc = m1.main(["--db", str(db), "--data-dir", str(data), "--out-dir", str(out_dir)])
    assert rc == 0
    assert "SKIP" in capsys.readouterr().out
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_depth_source_none_is_estimated_not_read_as_cad(tmp_path, monkeypatch):
    data = tmp_path / "data"
    db = data / "render_studio.db"
    _seed_db(db, data)
    # Job 5 sent the colour image itself as the control (depth_source "none"), so
    # its control URL resolves to a stored asset that is NOT a depth map.
    conn = connect(db)
    conn.execute("UPDATE assets SET fal_url = 'https://example.test/geo.png' WHERE id = 1")
    out_path = _png(data / "outputs" / "job000005_0.png", _scene(120, 160) * 255 * 0.5 + 40)
    conn.execute(
        "INSERT INTO jobs (id, idempotency_key, username, task, model_id, status, "
        "est_usd, params_json, output_path, created_at, updated_at, engine, depth_source) "
        "VALUES (5, 'k5', 'u', 'massing_to_render', 'fal-ai/flux-general', 'completed', 0, ?, ?, "
        "'2026-10-02T00:00:00Z', '2026-10-02T00:00:00Z', 'flux-general', 'none')",
        (
            json.dumps({
                "num_images": 1,
                "controlnet_unions": [
                    {"path": "x", "controls": [{"control_image_url": "https://example.test/geo.png"}]}
                ],
            }),
            str(out_path),
        ),
    )
    conn.execute(
        "INSERT INTO job_inputs (job_id, asset_id, role, position) VALUES (5, 1, 'geometry', 1)"
    )
    conn.close()
    monkeypatch.setattr(depth, "is_available", lambda settings: True)
    monkeypatch.setattr(depth, "estimate_depth", _fake_estimate)
    out_dir = tmp_path / "acceptance"

    rc = m1.main([
        "--db", str(db), "--data-dir", str(data), "--out-dir", str(out_dir), "--date", "2026-10-10",
    ])
    assert rc == 0
    with (out_dir / "2026-10-10-m1.csv").open(encoding="utf-8") as fh:
        scored = {int(r["job_id"]): r for r in csv.DictReader(fh)}
    assert scored[5]["depth_source_in"] == "estimated (none sent)"
