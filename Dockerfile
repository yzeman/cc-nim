FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git bash \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

# Install uv system-wide, then run the application and Claude Code as an
# unprivileged user so Claude Code can safely use --dangerously-skip-permissions.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY --chown=app:app pyproject.toml uv.lock README.md ./
COPY --chown=app:app api ./api
COPY --chown=app:app cli ./cli
COPY --chown=app:app config ./config
COPY --chown=app:app core ./core
COPY --chown=app:app messaging ./messaging
COPY --chown=app:app providers ./providers
COPY --chown=app:app server.py ./
COPY --chown=app:app .env.example ./
RUN chown app:app /app

USER app
ENV HOME=/home/app \
    PATH="/home/app/.local/bin:/usr/local/bin:/bin:$PATH" \
    HOST=0.0.0.0

RUN curl -fsSL https://claude.ai/install.sh | bash \
    && uv sync --frozen --no-dev

EXPOSE 10000

CMD ["sh", "-c", "uv run --no-sync uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}"]
