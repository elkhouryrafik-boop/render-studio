"""Shared fixture for the browser E2E suite: a fake-engine server on a free port.

The suite is opt-in. Every test under tests/e2e carries the `e2e` marker and is
skipped unless RUN_E2E=1, so `pytest tests` stays fast and browserless. The
server runs `api.main:app` under uvicorn in a subprocess with the zero-spend fake
engine, a temporary database and data directory, and no FAL_KEY, so nothing here
can reach fal.ai or touch the real data directory. The built web app
(web/dist) must already exist; `scripts/e2e.sh` builds it.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parents[2]
E2E_DIR = Path(__file__).resolve().parent
WEB_INDEX = REPO / "web" / "dist" / "index.html"
DEPTH_MODEL = REPO / "data" / "models" / "depth-anything-v2-small"
PORT_RANGE = range(8800, 8900)
ACCESS_CODE = "e2e-code"
ADMIN_EMAIL = "e2e@example.test"
ADMIN_PASSWORD = "e2e-password-123"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark every test under tests/e2e as `e2e`; skip them unless RUN_E2E=1."""
    run_e2e = os.environ.get("RUN_E2E") == "1"
    for item in items:
        if not Path(str(item.path)).resolve().is_relative_to(E2E_DIR):
            continue
        item.add_marker(pytest.mark.e2e)
        if not run_e2e:
            item.add_marker(pytest.mark.skip(reason="set RUN_E2E=1 to run the browser suite"))


def _free_port() -> int:
    for port in PORT_RANGE:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free port in 8800-8899 for the E2E server.")


def _wait_healthy(base: str, proc: subprocess.Popen, log: Path, timeout_s: float = 60) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"E2E server exited early; see {log}:\n{log.read_text(errors='replace')[-2000:]}"
            )
        try:
            if httpx.get(f"{base}/api/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"E2E server did not answer /api/health within {timeout_s} s; see {log}")


@pytest.fixture(scope="session")
def e2e_base_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    if not WEB_INDEX.exists():
        pytest.skip(
            "web/dist/index.html is missing: run `npm run build` in web/ (or scripts/e2e.sh)."
        )

    tmp = tmp_path_factory.mktemp("e2e")
    data_dir = tmp / "data"
    if DEPTH_MODEL.exists():
        shutil.copytree(DEPTH_MODEL, data_dir / "models" / DEPTH_MODEL.name)

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    log = tmp / "server.log"
    env = {
        **os.environ,
        "RENDER_ENGINE": "fake",
        "FAL_KEY": "",
        "FAKE_ENGINE_DELAY_S": "1",
        "DB_PATH": str(tmp / "e2e.db"),
        "DATA_DIR": str(data_dir),
        "OFFICE_ACCESS_CODE": ACCESS_CODE,
        "ADMIN_EMAIL": ADMIN_EMAIL,
    }
    with log.open("w") as out:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=REPO,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    try:
        _wait_healthy(base, proc, log)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
