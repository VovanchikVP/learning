from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import Text, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from config import Settings
from constants.db import (
    COLUMN_ALIAS_CURSOR,
    COLUMN_ALIAS_ID,
    POOL_MAX_OVERFLOW,
    POOL_SIZE,
)
from models import build_labeled_image_model
from schemas import ImageRow
from templates import errors, logs

log = logging.getLogger(__name__)


class ImageRepository:
    """Доступ к таблице с размеченными картинками.

    Первичный ключ — uuid, поэтому «новые строки» нельзя искать по возрастанию
    id. Поддерживаются два режима:

    * **keyset по cursor_column** (например ``created_at``): постраничный обход
      по паре ``(cursor_column, id)``; условие ``cursor >= :value`` вынесено
      отдельно, чтобы работал индекс;
    * **сверка идентификаторов** (если такой колонки нет): из базы вычитывается
      список id, из него вычитается содержимое локального кэша, дочитываются
      только недостающие строки.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cursor_column = settings.cursor_column
        self.model = build_labeled_image_model(settings)
        self._id_text = cast(self.model.id, Text)
        self._engine: AsyncEngine | None = None
        self._session: async_sessionmaker | None = None

    # ------------------------------------------------------------ соединение

    async def connect(self) -> None:
        if self._engine is not None:
            return
        self._engine = create_async_engine(
            self.settings.sqlalchemy_dsn, pool_size=POOL_SIZE, max_overflow=POOL_MAX_OVERFLOW
        )
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)
        mode = logs.DB_MODE_KEYSET % self.cursor_column if self.cursor_column else logs.DB_MODE_DIFF
        log.info(logs.DB_CONNECTED, mode)

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session = None

    def _sessionmaker(self) -> async_sessionmaker:
        if self._session is None:
            raise RuntimeError(errors.REPOSITORY_NOT_CONNECTED)
        return self._session

    # --------------------------------------------------------------- запросы

    def _base_select(self):
        model = self.model
        columns = [self._id_text.label(COLUMN_ALIAS_ID), model.image, model.answer]
        if self.cursor_column:
            columns.append(model.cursor.label(COLUMN_ALIAS_CURSOR))
        return select(*columns).where(model.answer.is_not(None))

    def keyset_statement(self, cursor: tuple[Any, str] | None, limit: int):
        """Запрос следующей страницы keyset-обхода (вынесен ради тестов)."""
        if not self.cursor_column:
            raise RuntimeError(errors.CURSOR_COLUMN_NOT_SET)
        model = self.model
        stmt = self._base_select()
        if cursor is not None:
            value, last_id = cursor
            stmt = stmt.where(
                model.cursor >= value,
                or_(model.cursor > value, self._id_text > last_id),
            )
        return stmt.order_by(model.cursor.asc(), self._id_text.asc()).limit(limit)

    async def fetch_after(self, cursor: tuple[Any, str] | None, limit: int) -> list[ImageRow]:
        async with self._sessionmaker()() as session:
            rows = (await session.execute(self.keyset_statement(cursor, limit))).mappings().all()
        return [ImageRow.model_validate(dict(r)) for r in rows]

    async def fetch_ids(self) -> list[str]:
        """Все id таблицы — для режима сверки с локальным кэшем."""
        stmt = select(self._id_text.label(COLUMN_ALIAS_ID)).where(self.model.answer.is_not(None))
        async with self._sessionmaker()() as session:
            return list((await session.execute(stmt)).scalars().all())

    async def fetch_by_ids(self, ids: list[str]) -> list[ImageRow]:
        stmt = self._base_select().where(self._id_text.in_(ids))
        async with self._sessionmaker()() as session:
            rows = (await session.execute(stmt)).mappings().all()
        return [ImageRow.model_validate(dict(r)) for r in rows]

    async def count(self) -> int:
        async with self._sessionmaker()() as session:
            return int(await session.scalar(select(func.count()).select_from(self.model)))

    async def count_labeled(self) -> int:
        """Строк с заполненным answer — только они идут в обучение."""
        stmt = select(func.count()).select_from(self.model).where(self.model.answer.is_not(None))
        async with self._sessionmaker()() as session:
            return int(await session.scalar(stmt))
