"""Download the Depth Anything V2 Small ONNX model into DATA_DIR/models.

    uv run python scripts/fetch_depth_model.py

Lands in `<DATA_DIR>/models/depth-anything-v2-small/`, where
`app/services/depth.py` looks (DATA_DIR comes from the same Settings the server
reads, so the two agree). Idempotent: a file already present at the size the
Hub reports is left alone. Downloads stream to `<name>.part` and are renamed
into place only when complete; an interrupted run resumes from the part file.

Model: onnx-community/depth-anything-v2-small, Apache-2.0
(base model depth-anything/Depth-Anything-V2-Small).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import httpx

# Allow `python scripts/fetch_depth_model.py` from the repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings

REPO = "onnx-community/depth-anything-v2-small"
# Pinned 2026-10-10 to the Hub commit that `main` pointed at that day (the `sha`
# field of the model API). Moving the pin is a deliberate change: take the new
# commit, read the new hash of onnx/model.onnx from the Hub's LFS listing for it,
# and update both constants together.
REVISION = "4472b7362082ad9968fee890ca0f1e5aca36b93d"
# sha256 of onnx/model.onnx at REVISION (the Hub's LFS oid). Checked after download.
MODEL_SHA256 = {"model.onnx": "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c"}
API = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}"
RESOLVE = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
LICENCE = "Licence: Apache-2.0 (onnx-community/depth-anything-v2-small, base model Depth-Anything-V2-Small)"

# Preference order for the weights. The first one the repo lists wins; it is
# stored locally as model.onnx or model_fp16.onnx under its own name.
MODEL_CANDIDATES = ("onnx/model.onnx", "onnx/model_fp16.onnx")
OPTIONAL = ("config.json", "preprocessor_config.json")
CHUNK = 1 << 20


def choose_files(siblings: list[str]) -> list[str]:
    """Which repo files to fetch, given the Hub's file list."""
    model = next((f for f in MODEL_CANDIDATES if f in siblings), None)
    if model is None:
        raise SystemExit(
            f"{REPO} lists none of {MODEL_CANDIDATES}. Files present: {sorted(siblings)}"
        )
    return [model, *[f for f in OPTIONAL if f in siblings]]


def verify_sha256(model: Path, expected: str) -> None:
    """Keep the model only if its bytes are the pinned ones; otherwise remove it and stop."""
    digest = hashlib.sha256()
    with model.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        model.unlink()
        raise SystemExit(
            f"{model.name}: sha256 {digest.hexdigest()} does not match the pinned "
            f"{expected} for revision {REVISION[:12]}. The file was removed. Fetch again; "
            "if the upstream file really changed, pin the new revision and hash in this script."
        )


def _remote_size(client: httpx.Client, url: str) -> int | None:
    r = client.head(url, follow_redirects=True)
    if r.status_code != 200:
        return None
    # The Hub reports the real size in x-linked-size for LFS files.
    size = r.headers.get("x-linked-size") or r.headers.get("content-length")
    return int(size) if size and size.isdigit() else None


def download(client: httpx.Client, url: str, dest: Path) -> int:
    """Stream `url` to `dest`, resuming a `.part` file. Returns the final size."""
    expected = _remote_size(client, url)
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        return dest.stat().st_size

    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    if expected is not None and start > expected:
        part.unlink()
        start = 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with client.stream("GET", url, headers=headers, follow_redirects=True) as r:
        if r.status_code == 416 and expected is not None and start == expected:
            pass  # part file is already complete
        else:
            r.raise_for_status()
            if start and r.status_code != 206:
                start = 0  # server ignored Range; start over
            with open(part, "ab" if start else "wb") as fh:
                fh.writelines(r.iter_bytes(CHUNK))
    if expected is not None and part.stat().st_size != expected:
        raise SystemExit(
            f"{dest.name}: got {part.stat().st_size} bytes, expected {expected}. "
            "Run again to resume."
        )
    part.replace(dest)
    return dest.stat().st_size


def main() -> int:
    settings = Settings()
    target = settings.depth_model_dir()
    target.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=httpx.Timeout(30.0, read=120.0)) as client:
        meta = client.get(API, follow_redirects=True)
        meta.raise_for_status()
        siblings = [s["rfilename"] for s in meta.json().get("siblings", [])]
        total = 0
        for name in choose_files(siblings):
            dest = target / Path(name).name
            size = download(client, f"{RESOLVE}/{name}", dest)
            if dest.name in MODEL_SHA256:
                verify_sha256(dest, MODEL_SHA256[dest.name])
            elif dest.suffix == ".onnx":
                print(f"{dest.name}: no pinned hash, not verified")
            total += size
            print(f"{dest.name:<28} {size / 1_048_576:8.1f} MB")
    print(f"{'total':<28} {total / 1_048_576:8.1f} MB  ->  {target}")
    print(LICENCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
