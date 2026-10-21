# syntax=docker/dockerfile:1
# Stage 1: build the web bundle.
FROM node:20-bookworm AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# Stage 2: the API, serving the built bundle.
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    PYRENDER_GL_BACKEND=osmesa \
    DATA_DIR=/app/data \
    DB_PATH=/app/data/render_studio.db \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libegl1 libosmesa6 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY api ./api
COPY app ./app
COPY routing.yaml ./
COPY scripts/fetch_depth_model.py ./scripts/fetch_depth_model.py
COPY scripts/docker-entrypoint.sh ./scripts/docker-entrypoint.sh
COPY --from=web /web/dist ./web/dist

# The data dir is a volume; it is created here only so a non-root user owns it.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data \
    && chown -R app:app /app/data \
    && chmod +x /app/scripts/docker-entrypoint.sh
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/api/health || exit 1

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
