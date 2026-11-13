from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import Settings
from constants.api import API_VERSION, BACKGROUND_TASK_NAME
from templates.api import API_TITLE
from utils import setup_logging

from .container import AppContainer
from .routes import router

log = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    container: AppContainer | None = None,
    run_watcher: bool = True,
) -> FastAPI:
    """Собирает приложение.

    `container` позволяет подставить готовые зависимости (используется в тестах),
    `run_watcher=False` выключает фоновый цикл обучения.
    """
    settings = settings or Settings()
    # под uvicorn логирование никто не настраивает, иначе логов сервиса не видно
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = container or AppContainer.build(settings)
        task = None
        if run_watcher:
            task = asyncio.create_task(
                app.state.container.training.run(), name=BACKGROUND_TASK_NAME
            )
        try:
            yield
        finally:
            if task is not None:
                app.state.container.training.stop()
                await task

    app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)
    app.include_router(router)
    return app
