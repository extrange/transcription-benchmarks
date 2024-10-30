FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

# Inherited by all stages and persisted in the final image
ENV APP_DIR=/app
ENV TZ=Asia/Singapore
ENV PYTHONPATH=$APP_DIR/src

# Speed up container start time https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
ENV UV_COMPILE_BYTECODE=1

# Ensure that the `python` command uses the venv
ENV PATH="$APP_DIR/.venv/bin:$PATH"

# uv: Don't link packages to the mounted cache on the host (which will not be present in the final image). Instead, copy packages to the container
ENV UV_LINK_MODE=copy

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg curl vim && \
    rm -rf /var/lib/apt/lists/*
