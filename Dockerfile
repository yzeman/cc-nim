FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:/usr/local/bin:$PATH" \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git bash \
    && rm -rf /var/lib/apt/lists/*

# Install uv and the Claude Code CLI (native Linux installer).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN curl -fsSL https://claude.ai/install.sh | bash

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY api ./api
COPY cli ./cli
COPY config ./config
COPY core ./core
COPY messaging ./messaging
COPY providers ./providers
COPY server.py ./
COPY .env.example ./

RUN uv sync --frozen --no-dev

ENV HOST=0.0.0.0
EXPOSE 10000

CMD ["sh", "-c", "uv run --no-sync uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}"]
