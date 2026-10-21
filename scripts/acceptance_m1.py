"""Acceptance metric M1: did each completed render keep its input's depth structure?

Zero spend. Reads the office database, never writes to it, and scores every
completed job that has an output file and a geometry input on disk:

  * input depth: the CAD depth pass if the job used one (resolved through the
    job's controlnet_unions control_image_url to an assets.fal_url), otherwise
    depth estimated from the geometry input (a depth_source of "none" is always
    estimated from the geometry input, since the colour image itself was sent as the control);
  * output depth: estimated from the output image;
  * both resized to the smaller common size; invalid input pixels (zero in a
    CAD depth map, which means "no surface") are masked out;
  * output depth is aligned to input depth by least squares, s and t in
    argmin ||s * d_out + t - d_in||, so the score is scale- and shift-invariant
    (the monocular-depth convention of Li et al. 2024, arXiv 2404.07987);
  * rmse_aligned: RMSE of the aligned output against the input, 0..1 scale;
  * edge_f1: F-score between Canny-like edge maps of the two depth maps, a
    2-pixel tolerance on each side. Numpy and Pillow only.

Estimated depths go through the same cache the render path uses
(<DATA_DIR>/depth_cache/<sha256 of file bytes>_<model>.png, 8-bit), so a rerun
is instant and an input already estimated for a render is free.

Usage (from the repo root):
    uv run python scripts/acceptance_m1.py --out-dir examples/out/acceptance
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sqlite3
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.services import depth

log = logging.getLogger("acceptance_m1")

EDGE_TOLERANCE_PX = 2
EDGE_PERCENTILE = 90.0
MIN_VALID_PIXELS = 1000

CSV_COLUMNS = [
    "job_id",
    "task",
    "engine",
    "model_id",
    "created_at",
    "num_images",
    "geometry_hold",
    "depth_source_in",
    "geometry_asset_id",
    "rmse_aligned",
    "edge_f1",
    "scale",
    "shift",
    "n_valid_px",
]

INTERPRETATION = (
    "Lower RMSE means the output's depth matches the input's depth structure more "
    "closely once the best scale and shift are applied, so the render kept the "
    "geometry it was given. The expectation was that the geometry-held flux-general "
    "outputs to drift less than the edit-model outputs, and this table is the first "
    "place to check that."
)


# --- depth maps and the metric (pure functions) -------------------------------


def load_depth_map(path: Path) -> np.ndarray:
    """A depth PNG as float64 in 0..1, near = bright (the app's convention)."""
    with Image.open(path) as im:
        if im.mode in ("I", "I;16", "I;16B", "I;16L"):
            return np.asarray(im, dtype=np.float64) / 65535.0
        return np.asarray(im.convert("L"), dtype=np.float64) / 255.0


def _resize(arr: np.ndarray, size_hw: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    h, w = size_hw
    if arr.shape == (h, w):
        return arr.astype(np.float64)
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    im = Image.fromarray(arr.astype(np.float32))
    return np.asarray(im.resize((w, h), resample), dtype=np.float64)


def prepare_pair(
    depth_in: np.ndarray, valid_in: np.ndarray, depth_out: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Both maps at the smaller common size; the validity mask follows the input."""
    h = min(depth_in.shape[0], depth_out.shape[0])
    w = min(depth_in.shape[1], depth_out.shape[1])
    d_in = _resize(depth_in, (h, w))
    d_out = _resize(depth_out, (h, w))
    valid = _resize(valid_in.astype(np.float64), (h, w), nearest=True) >= 0.5
    return d_in, valid & np.isfinite(d_in), d_out


def fit_scale_shift(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """(s, t) minimising ||s * pred + t - target|| over the valid pixels."""
    x = pred[valid]
    y = target[valid]
    if x.size < 2:
        raise ValueError("need at least two valid pixels to align depth")
    design = np.stack([x, np.ones_like(x)], axis=1)
    (s, t), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(s), float(t)


def rmse(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    diff = pred[valid] - target[valid]
    return float(np.sqrt(np.mean(diff * diff)))


def _shifted(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """out[y, x] = mask[y + dy, x + dx], zero outside the image."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    out[max(0, -dy) : min(h, h - dy), max(0, -dx) : min(w, w - dx)] = mask[
        max(0, dy) : min(h, h + dy), max(0, dx) : min(w, w + dx)
    ]
    return out


def _offsets(radius: int, *, square: bool) -> list[tuple[int, int]]:
    return [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if square or dy * dy + dx * dx <= radius * radius
    ]


def dilate(mask: np.ndarray, radius: int, *, square: bool = False) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for dy, dx in _offsets(radius, square=square):
        out |= _shifted(mask, dy, dx).astype(bool)
    return out


def erode(mask: np.ndarray, radius: int, *, square: bool = True) -> np.ndarray:
    """Pixels whose whole neighbourhood is inside the mask (outside counts as out)."""
    out = np.ones_like(mask, dtype=bool)
    for dy, dx in _offsets(radius, square=square):
        out &= _shifted(mask, dy, dx).astype(bool)
    return out


def _gaussian(a: np.ndarray, sigma: float = 1.0, radius: int = 2) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    h, w = a.shape
    padded = np.pad(a, radius, mode="edge")
    tmp = np.zeros((h + 2 * radius, w))
    for i, kv in enumerate(k):
        tmp += kv * padded[:, i : i + w]
    out = np.zeros((h, w))
    for i, kv in enumerate(k):
        out += kv * tmp[i : i + h, :]
    return out


def _sobel(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.pad(a, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return gx, gy


def edge_map(depth_map: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Canny-like edges: Gaussian, Sobel, non-maximum suppression, one percentile threshold.

    Hysteresis is left out on purpose: a single threshold at the 90th percentile
    of the gradient magnitude is scale-free, so the same structure gives the
    same edges whatever range the model happened to produce. Edges within three
    pixels of the invalid region are dropped, since blurring there is meaningless.
    """
    filled = np.where(valid, depth_map, 0.0)
    gx, gy = _sobel(_gaussian(filled))
    mag = np.hypot(gx, gy)
    core = erode(valid, 3, square=True)
    if not core.any():
        return np.zeros_like(valid, dtype=bool)
    threshold = np.percentile(mag[core], EDGE_PERCENTILE)

    # Quantise the gradient direction to four neighbour pairs.
    angle = np.degrees(np.arctan2(gy, gx)) % 180.0
    quadrant = np.floor((angle + 22.5) / 45.0).astype(int) % 4
    neighbours = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}
    keep = np.zeros_like(valid, dtype=bool)
    for q, (dy, dx) in neighbours.items():
        here = quadrant == q
        ahead = _shifted(mag, dy, dx)
        behind = _shifted(mag, -dy, -dx)
        keep |= here & (mag >= ahead) & (mag >= behind)
    return keep & (mag >= threshold) & (mag > 0) & core


def edge_f1(pred_edges: np.ndarray, ref_edges: np.ndarray, tolerance: int = EDGE_TOLERANCE_PX) -> float:
    """F1 of edge pixels, each side allowed to be `tolerance` pixels off."""
    pred = pred_edges.astype(bool)
    ref = ref_edges.astype(bool)
    n_pred = int(pred.sum())
    n_ref = int(ref.sum())
    if n_pred == 0 and n_ref == 0:
        return 1.0
    if n_pred == 0 or n_ref == 0:
        return 0.0
    precision = float((pred & dilate(ref, tolerance)).sum()) / n_pred
    recall = float((ref & dilate(pred, tolerance)).sum()) / n_ref
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def score_pair(depth_in: np.ndarray, depth_out: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    """Score output depth against input depth, both already at a common size."""
    s, t = fit_scale_shift(depth_out, depth_in, valid)
    aligned = s * depth_out + t
    return {
        "rmse_aligned": rmse(aligned, depth_in, valid),
        "edge_f1": edge_f1(edge_map(aligned, valid), edge_map(depth_in, valid)),
        "scale": s,
        "shift": t,
        "n_valid": int(valid.sum()),
    }


# --- paths, database and cache ----------------------------------------------


def resolve_stored_path(raw: str | None, data_dir: Path, fallback_dir: Path | None = None) -> Path | None:
    """A path as the app stored it, or None if it is not on disk.

    The office database holds Windows-relative paths such as data\\outputs\\x.png,
    relative to the repository root (the parent of the data directory), and
    some absolute paths from upload time. Try each interpretation in turn.
    """
    if not raw:
        return None
    text = str(raw).replace("\\", "/")
    p = Path(text)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(data_dir.parent / p)
        candidates.append(Path.cwd() / p)
        if text.startswith("data/"):
            candidates.append(data_dir / text[len("data/") :])
        if fallback_dir is not None:
            candidates.append(fallback_dir / p.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def open_readonly(db_path: Path) -> sqlite3.Connection:
    """The office database, opened read-only. Never call app.db.connect here: it sets WAL."""
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    assert "mode=ro" in uri
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def cached_estimate(src: Path, settings: Settings) -> Path:
    """Estimated depth for `src` through the shared cache the render path uses."""
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    cache_dir = settings.depth_cache_dir()
    cached = cache_dir / f"{digest}_{depth.MODEL_NAME}.png"
    if not cached.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_name(f"{cached.stem}.{os.getpid()}.tmp.png")
        try:
            depth.estimate_depth(src, tmp, settings, bits=8)
            os.replace(tmp, cached)
        finally:
            tmp.unlink(missing_ok=True)
    return cached


def control_urls(params: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for union in params.get("controlnet_unions") or []:
        for control in union.get("controls") or []:
            url = control.get("control_image_url")
            if url:
                urls.append(str(url))
    return urls


def find_cad_depth_asset(conn: sqlite3.Connection, params: dict[str, Any]) -> sqlite3.Row | None:
    for url in control_urls(params):
        row = conn.execute(
            "SELECT * FROM assets WHERE fal_url = ? ORDER BY (kind = 'control') DESC, id LIMIT 1",
            (url,),
        ).fetchone()
        if row is not None:
            return row
    return None


def _params(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


# --- per job ----------------------------------------------------------------


def score_job(
    conn: sqlite3.Connection,
    settings: Settings,
    job: sqlite3.Row,
    outputs_dir: Path | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """(scored row, None) or (None, reason it was skipped)."""
    data_dir = settings.data_dir
    output = resolve_stored_path(job["output_path"], data_dir, outputs_dir)
    if output is None:
        return None, f"output file not on disk ({job['output_path'] or 'no output_path'})"

    geo = conn.execute(
        "SELECT a.id AS asset_id, a.local_path AS local_path FROM job_inputs ji "
        "JOIN assets a ON a.id = ji.asset_id WHERE ji.job_id = ? AND ji.role = 'geometry' "
        "ORDER BY ji.position LIMIT 1",
        (job["id"],),
    ).fetchone()
    if geo is None:
        return None, "no geometry input"
    geo_path = resolve_stored_path(geo["local_path"], data_dir)
    if geo_path is None:
        return None, f"geometry input not on disk ({geo['local_path']})"

    params = _params(job["params_json"])
    job_columns = set(job.keys())
    column = job["depth_source"] if "depth_source" in job_columns else None
    cad_path: Path | None = None
    if column is None or column == "cad":
        cad_row = find_cad_depth_asset(conn, params)
        if cad_row is not None:
            cad_path = resolve_stored_path(cad_row["local_path"], data_dir)
    if cad_path is not None:
        depth_source_in = "cad"
        depth_in_full = load_depth_map(cad_path)
        valid_in = depth_in_full > 0.0
    else:
        unresolved = column == "cad" or (column is None and bool(control_urls(params)))
        depth_source_in = "estimated (cad unresolved)" if unresolved else "estimated"
        if column == "none":
            depth_source_in = "estimated (none sent)"
        depth_in_full = load_depth_map(cached_estimate(geo_path, settings))
        valid_in = np.ones(depth_in_full.shape, dtype=bool)

    depth_out_full = load_depth_map(cached_estimate(output, settings))
    d_in, valid, d_out = prepare_pair(depth_in_full, valid_in, depth_out_full)
    if int(valid.sum()) < MIN_VALID_PIXELS:
        return None, f"too few valid input depth pixels ({int(valid.sum())})"

    scores = score_pair(d_in, d_out, valid)
    hold = params.get("geometry_hold")
    return {
        "job_id": int(job["id"]),
        "task": job["task"],
        "engine": job["engine"] if "engine" in set(job.keys()) else "",
        "model_id": job["model_id"],
        "created_at": job["created_at"],
        "num_images": params.get("num_images", 1),
        "geometry_hold": "" if hold is None else hold,
        "depth_source_in": depth_source_in,
        "geometry_asset_id": int(geo["asset_id"]),
        "rmse_aligned": round(scores["rmse_aligned"], 4),
        "edge_f1": round(scores["edge_f1"], 4),
        "scale": round(scores["scale"], 4),
        "shift": round(scores["shift"], 4),
        "n_valid_px": scores["n_valid"],
    }, None


# --- reports -----------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _median(values: list[float]) -> str:
    return f"{statistics.median(values):.4f}" if values else "n/a"


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    skipped: list[tuple[int, str, str]],
    *,
    run_date: str,
    db_path: Path,
    total_completed: int,
) -> None:
    lines: list[str] = [
        f"# Acceptance M1 baseline, {run_date}",
        "",
        (f"Source: `{db_path.name}`, opened read-only. Completed jobs: {total_completed}. "
         f"Scored: {len(rows)}. Skipped: {len(skipped)}."),
        "",
        ("Metric: depth re-estimated from each output, compared with the input's depth "
         "(CAD depth where the job had one, otherwise estimated from the input image), "
         "after a least-squares scale and shift. `rmse_aligned` is on the 0..1 scale; "
         "`edge_f1` uses a 2-pixel tolerance."),
        "",
        "## How to read it",
        "",
        INTERPRETATION,
        "",
        "## Summary by model",
        "",
        "| model_id | jobs | median rmse_aligned | median edge_f1 |",
        "|---|---:|---:|---:|",
    ]
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model_id"], []).append(row)
    for model in sorted(by_model):
        group = by_model[model]
        lines.append(
            f"| {model} | {len(group)} | "
            f"{_median([r['rmse_aligned'] for r in group])} | "
            f"{_median([r['edge_f1'] for r in group])} |"
        )
    if not by_model:
        lines.append("| (none scored) | 0 | n/a | n/a |")

    lines += [
        "",
        "## Per job",
        "",
        "| job | task | engine | model_id | created | n | hold | depth in | rmse_aligned | edge_f1 |",
        "|---:|---|---|---|---|---:|---|---|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: r["job_id"]):
        lines.append(
            f"| {r['job_id']} | {r['task']} | {r['engine']} | {r['model_id']} | "
            f"{str(r['created_at'])[:16]} | {r['num_images']} | {r['geometry_hold']} | "
            f"{r['depth_source_in']} | {r['rmse_aligned']:.4f} | {r['edge_f1']:.4f} |"
        )

    lines += ["", "## Skipped", ""]
    if skipped:
        for job_id, task, reason in skipped:
            lines.append(f"- job {job_id} ({task}): {reason}")
    else:
        lines.append("None.")
    lines += [
        "",
        "## Caveats",
        "",
        ("- For databases before schema 7 there is no `depth_source` column; a job "
         "counts as CAD only when its controlnet control URL resolves to a stored asset."),
        "- Estimated input depth is only as good as Depth Anything V2 Small on that image.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    defaults = Settings()
    parser = argparse.ArgumentParser(description="Acceptance metric M1 (depth structure kept).")
    parser.add_argument("--db", type=Path, default=defaults.db_path)
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--outputs-dir", type=Path, default=None,
                        help="Fallback for output files (default: <data-dir>/outputs).")
    parser.add_argument("--out-dir", type=Path, default=Path("examples/out/acceptance"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many completed jobs.")
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args(argv)

    settings = Settings(data_dir=args.data_dir, db_path=args.db)
    if not depth.is_available(settings):
        print(f"SKIP: depth model not installed in {settings.depth_model_dir()}; "
              "run scripts/fetch_depth_model.py first.")
        return 0

    outputs_dir = args.outputs_dir or (args.data_dir / "outputs")
    try:
        conn = open_readonly(args.db)
    except sqlite3.OperationalError as exc:
        print(f"ERROR: cannot open {args.db} read-only: {exc}")
        return 2

    try:
        sql = "SELECT * FROM jobs WHERE status = 'completed' ORDER BY id"
        params: tuple[Any, ...] = ()
        if args.limit:
            sql += " LIMIT ?"
            params = (int(args.limit),)
        jobs = conn.execute(sql, params).fetchall()
        rows: list[dict[str, Any]] = []
        skipped: list[tuple[int, str, str]] = []
        for job in jobs:
            try:
                row, reason = score_job(conn, settings, job, outputs_dir)
            except Exception as exc:
                log.warning("job %s failed to score", job["id"], exc_info=True)
                row, reason = None, f"error: {exc}"
            if row is not None:
                rows.append(row)
            else:
                skipped.append((int(job["id"]), job["task"], reason or "unknown"))
    finally:
        conn.close()

    out_dir = args.out_dir
    write_csv(out_dir / f"{args.date}-m1.csv", rows)
    write_markdown(
        out_dir / f"{args.date}-baseline.md",
        rows,
        skipped,
        run_date=args.date,
        db_path=args.db,
        total_completed=len(jobs),
    )
    print(f"Scored {len(rows)} of {len(jobs)} completed jobs; {len(skipped)} skipped. "
          f"Wrote {out_dir / (args.date + '-m1.csv')} and {out_dir / (args.date + '-baseline.md')}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
