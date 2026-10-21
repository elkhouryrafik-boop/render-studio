#!/usr/bin/env bash
# Build the web app, then run the browser E2E suite against a fake-engine server.
# Zero spend: the suite starts its own server with RENDER_ENGINE=fake and no FAL_KEY.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

(cd "$ROOT/web" && npm run build)

cd "$ROOT"
RUN_E2E=1 uv run python -m pytest -q tests/e2e -p no:cacheprovider "$@"
