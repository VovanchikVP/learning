# ---- сборка зависимостей ----
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# слой с зависимостями кэшируется отдельно от кода;
# torch под linux ставится из индекса pytorch-cpu (см. [tool.uv.sources]),
# поэтому колёс nvidia-* в образе нет
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# ---- рантайм ----
FROM python:3.13-slim-bookworm

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # пакет лежит в src/ без промежуточной директории: import api, import config
    PYTHONPATH=/app/src \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    # кэш картинок, чекпоинты и state.json — на том же пути монтируется PVC
    CL_WORKDIR=/var/lib/continuous-learning \
    # в кластере обучение идёт на CPU: mps нет, cuda в образе нет
    CL_DEVICE=cpu \
    # корень только на чтение, torch и matplotlib пишут кэши в $HOME
    HOME=/tmp

WORKDIR /app

# libgomp нужен OpenMP-ядрам torch
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY src ./src

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p "$CL_WORKDIR" \
    && chown -R app:app /app "$CL_WORKDIR"
USER app

EXPOSE 8000

# host/port берутся из UVICORN_HOST/UVICORN_PORT — явные флаги здесь перебили бы env.
# --factory: приложение собирается create_app(), фоновый цикл обучения стартует в lifespan
CMD ["uvicorn", "--factory", "api:create_app"]
