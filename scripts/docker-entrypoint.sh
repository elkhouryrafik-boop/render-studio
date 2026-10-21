#!/bin/sh
# Start the app at once; fetch the depth model into the data volume in the background
# if it is missing. Soft-fail: offline, the app still runs and the Exact slider is
# capped at Hold. The model is picked up on the next depth request after it lands.
set -u
DATA="${DATA_DIR:-/app/data}"
MODEL_DIR="$DATA/models/depth-anything-v2-small"
mkdir -p "$DATA"
if ! ls "$MODEL_DIR"/*.onnx >/dev/null 2>&1; then
    if [ "${SKIP_DEPTH_MODEL_FETCH:-0}" = "1" ]; then
        echo "entrypoint: depth model fetch skipped (SKIP_DEPTH_MODEL_FETCH=1)"
    else
        echo "entrypoint: fetching depth model into $MODEL_DIR in the background"
        ( timeout 900 python /app/scripts/fetch_depth_model.py \
            || echo "entrypoint: depth model unavailable; Exact slider will be capped at Hold" ) &
    fi
fi
exec "$@"
