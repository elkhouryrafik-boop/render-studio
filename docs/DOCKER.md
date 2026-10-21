# Docker

```bash
cp .env.example .env            # then set OFFICE_ACCESS_CODE and, for real renders, FAL_KEY
RENDER_ENGINE=fake docker compose up -d --build   # zero-key demo; omit RENDER_ENGINE for fal.ai
curl -f http://localhost:8000/api/health
bash scripts/smoke_fake.sh      # sign up + one fake render through the API
docker compose down -v          # stop and delete the data volume
```

Open <http://localhost:8000>. Sign up with the `OFFICE_ACCESS_CODE` from `.env`; the first
account created with `ADMIN_EMAIL` is the administrator.

## How it is built

- **Stage 1** (`node:20-bookworm`): `npm ci && npm run build` in `web/`.
- **Stage 2** (`python:3.12-slim-bookworm`): `libgl1 libegl1 libosmesa6 curl`, `uv sync --frozen --no-dev`,
  then `api/`, `app/`, `routing.yaml`, the depth-model fetch script and `web/dist`. Runs as a non-root user.
- `PYRENDER_GL_BACKEND=osmesa` so IFC/SKP clay previews can render headless. DXF import needs no GL.
  IFC/SKP clay rendering inside the container is best effort; the test suite runs on the host.
- `.env` is never copied into the image (`.dockerignore`); compose reads it at run time
  (`env_file`, optional). The image holds no `data/` content: `/app/data` is an empty mount point for the
  `render-data` volume (SQLite DB, uploads, outputs, depth model).

## Depth model

On start the entrypoint fetches the Depth Anything V2 Small ONNX model (Apache-2.0) into
`/app/data/models/` **in the background** if it is missing, so the server is up immediately. If the
download fails (offline) the app keeps running and the "Exact" geometry-hold slider is capped at "Hold".
The model is used from the next depth request once it has landed. Set `SKIP_DEPTH_MODEL_FETCH=1` to skip
the download (CI, air-gapped hosts). The volume keeps the model across restarts.

## Defaults and notes

- `RENDER_ENGINE` defaults to `fal` in compose. Without `FAL_KEY` the app still starts and refuses jobs.
- `COOKIE_SECURE=0` is required on plain `http://`; set it to `1` only behind https.
- Port 8000 is published on all interfaces; bind it to `127.0.0.1:8000:8000` if the host is shared.
- On Windows Git Bash, `smoke_fake.sh` converts its temp-file path for native curl automatically.
