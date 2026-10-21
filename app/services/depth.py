"""Local monocular depth for screenshots (Depth Anything V2 Small, ONNX, CPU).

Why this exists: for any uploaded screenshot the depth ControlNet entry on
flux-general was handed the *colour image itself* as its control image
(fal_gateway.build_arguments fans the geometry reference across every
conditioning entry). Only CAD conversions supplied a true depth pass. This
module estimates one so the depth slot receives depth.

POLARITY (read off the code, not assumed): the CAD pipeline writes depth in the
"ControlNet convention (8-bit, near = bright)" - app/services/cad_convert.py,
`_ifc_render` and the SKP pass both compute `255 * (1 - (z - zmin)/(zmax - zmin))`
over the z-buffer, so the nearest surface is 255 and the far plane / background
is 0 (see the comment at the "ControlNet depth convention" lines and
docs/ARCHITECTURE.md). The fal depth control (Shakker-Labs FLUX.1-dev
ControlNet-Union-Pro, mode `depth`) was validated with those images
(tests/test_flux_general_conditioning.py only asserts routing of the URL, the
convention is that of the CAD writer). Depth Anything V2 outputs RELATIVE
INVERSE depth (disparity): larger = nearer. That is already near = bright, so
the prediction is normalised per image and written as-is, NOT inverted.

Normalisation is per image (min..max -> 0..max_value): the model gives
affine-invariant disparity, so absolute values mean nothing across images.

Output depth: `bits=16` writes the 0..65535 greyscale PNG; `bits=8` writes the
0..255 form the CAD pipeline writes. The render path asks for 8: Pillow
saturates an "I;16" image to white on `.convert("RGB")`, so a 16-bit control
image would reach a PIL-based loader as a solid white rectangle.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import Settings

log = logging.getLogger(__name__)

MODEL_NAME = "depth-anything-v2-small"
_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)
_DEFAULT_SHORT_SIDE = 518
_MULTIPLE = 14
# Cap on the longer side so a panorama does not ask the CPU for a 4000 px input.
_MAX_LONG_SIDE = 1568


@dataclass(frozen=True)
class DepthResult:
    path: Path
    source: str = "estimated"
    model: str = MODEL_NAME
    seconds: float = 0.0


_session_lock = threading.Lock()
# Inference is CPU-bound and every session uses all cores. Two at once keeps a
# burst of distinct screenshots from starving the office's other work.
_inference_slots = threading.BoundedSemaphore(2)
_session = None
_session_key: str | None = None


def _model_file(settings: Settings) -> Path | None:
    d = settings.depth_model_dir()
    for name in ("model.onnx", "model_fp16.onnx"):
        if (d / name).is_file():
            return d / name
    return None


def is_available(settings: Settings) -> bool:
    """True when a model file is on disk and onnxruntime can be imported."""
    if _model_file(settings) is None:
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _get_session(settings: Settings):
    """The shared InferenceSession, built on first use (CPU provider)."""
    global _session, _session_key
    model = _model_file(settings)
    if model is None:
        raise FileNotFoundError(
            f"No depth model in {settings.depth_model_dir()}. "
            "Run: uv run python scripts/fetch_depth_model.py"
        )
    with _session_lock:
        if _session is None or _session_key != str(model):
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = os.cpu_count() or 1
            _session = ort.InferenceSession(
                str(model), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            _session_key = str(model)
        return _session


def _preprocessor(settings: Settings) -> tuple[tuple[float, ...], tuple[float, ...], int, int]:
    """(mean, std, short side, multiple) from preprocessor_config.json if present."""
    mean, std, short, mult = _DEFAULT_MEAN, _DEFAULT_STD, _DEFAULT_SHORT_SIDE, _MULTIPLE
    cfg = settings.depth_model_dir() / "preprocessor_config.json"
    try:
        raw = json.loads(cfg.read_text(encoding="utf-8"))
        mean = tuple(raw.get("image_mean", mean))
        std = tuple(raw.get("image_std", std))
        size = raw.get("size") or {}
        short = int(min(size.get("height", short), size.get("width", short)))
        mult = int(raw.get("ensure_multiple_of", mult))
    except (OSError, ValueError, TypeError):
        pass
    return mean, std, short, mult


def _target_size(w: int, h: int, short: int, mult: int) -> tuple[int, int]:
    """Shorter side -> `short`, both sides rounded to a multiple, aspect kept."""
    scale = short / min(w, h)
    if max(w, h) * scale > _MAX_LONG_SIDE:
        scale = _MAX_LONG_SIDE / max(w, h)
    nw = max(mult, round(w * scale / mult) * mult)
    nh = max(mult, round(h * scale / mult) * mult)
    return nw, nh


def estimate_depth(src: Path, dst: Path, settings: Settings, *, bits: int = 16) -> DepthResult:
    """Estimate depth for `src`, write a greyscale PNG (near = bright) to `dst`.

    The PNG has the source's pixel size. Raises if the model is missing or
    inference fails; callers that must not fail (`submit_render`) catch it.
    """
    if bits not in (8, 16):
        raise ValueError("bits must be 8 or 16")
    t0 = time.perf_counter()
    mean, std, short, mult = _preprocessor(settings)

    with Image.open(src) as im:
        rgb = im.convert("RGB")
    w, h = rgb.size
    nw, nh = _target_size(w, h, short, mult)
    arr = np.asarray(rgb.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    x = np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)

    session = _get_session(settings)
    name = session.get_inputs()[0].name
    with _inference_slots:
        raw = session.run(None, {name: x})[0]
    pred = np.asarray(raw, dtype=np.float32)
    pred = pred.reshape(pred.shape[-2], pred.shape[-1])

    # Resize the raw prediction first and normalise after, so the written image
    # spans the full range at the source size (bilinear resize pulls extremes in).
    full = np.asarray(Image.fromarray(pred).resize((w, h), Image.BILINEAR))
    lo, hi = float(full.min()), float(full.max())
    norm = (full - lo) / (hi - lo) if hi > lo else np.zeros_like(full)

    if bits == 16:
        out = Image.fromarray(np.rint(norm * 65535.0).astype(np.uint16))
    else:
        out = Image.fromarray(np.rint(norm * 255.0).astype(np.uint8))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, format="PNG")
    return DepthResult(path=dst, seconds=time.perf_counter() - t0)
