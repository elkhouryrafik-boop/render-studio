#!/usr/bin/env bash
# Zero-spend smoke test against a running fake-engine stack:
#   RENDER_ENGINE=fake docker compose up -d && bash scripts/smoke_fake.sh
# Env: BASE_URL (default http://localhost:8000), OFFICE_ACCESS_CODE (default change-me),
#      ADMIN_EMAIL (default admin@example.test).
set -euo pipefail
BASE="${BASE_URL:-http://localhost:8000}"
CODE="${OFFICE_ACCESS_CODE:-change-me}"
EMAIL="${ADMIN_EMAIL:-admin@example.test}"
PASS="smoke-test-password-1"
JAR="$(mktemp)"; PNG="$(mktemp)"
trap 'rm -f "$JAR" "$PNG"' EXIT

curl -fsS "$BASE/api/health" >/dev/null && echo "health ok"

# A tiny valid PNG (64x64 grey), embedded so no external files or tools are needed.
echo "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAAAAACPAi4CAAAAKklEQVR4nO3MMQ0AAAwDoPq/Krki9i0ggPQoAoFAIBAIBAKBQCAQCL4HA1lcAIghJl+dAAAAAElFTkSuQmCC" | base64 -d > "$PNG"
# Native Windows curl (Git Bash) needs a Windows path for -F @file; a no-op elsewhere.
PNG_ARG="$(cygpath -m "$PNG" 2>/dev/null || echo "$PNG")"

SIGNUP_STATUS=$(curl -sS -o /dev/null -w '%{http_code}' -c "$JAR" -X POST "$BASE/api/auth/signup" \
  -H 'Content-Type: application/json' \
  -d "{\"access_code\":\"$CODE\",\"email\":\"$EMAIL\",\"name\":\"Smoke Test\",\"password\":\"$PASS\"}")
if [ "$SIGNUP_STATUS" != "200" ]; then
  # Account may exist from an earlier run on the same volume: sign in instead.
  curl -fsS -c "$JAR" -X POST "$BASE/api/auth/login" -H 'Content-Type: application/json' \
    -d "{\"email\":\"$EMAIL\",\"password\":\"$PASS\"}" >/dev/null
fi
echo "signed in"

ASSET=$(curl -fsS -b "$JAR" -F "file=@$PNG_ARG;filename=smoke.png" "$BASE/api/assets" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "asset $ASSET"

JOB=$(curl -fsS -b "$JAR" -X POST "$BASE/api/renders" -H 'Content-Type: application/json' \
  -d "{\"task\":\"sketch_to_render\",\"prompt\":\"smoke test\",\"engine\":\"proven\",\"references\":[{\"asset_id\":$ASSET,\"role\":\"geometry\"}],\"width\":512,\"height\":512}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
echo "job $JOB queued"

for _ in $(seq 1 60); do
  STATE=$(curl -fsS -b "$JAR" "$BASE/api/renders/$JOB" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["status"], d.get("output_url") or "-")')
  case "$STATE" in
    succeeded*|done*|completed*) echo "render finished: $STATE"; exit 0;;
    failed*|refused*|cancelled*|error*) echo "render failed: $STATE" >&2; exit 1;;
  esac
  sleep 2
done
echo "timed out waiting for job $JOB (last: $STATE)" >&2; exit 1
