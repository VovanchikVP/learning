"""Проверка репозитория на настоящем PostgreSQL.

Запускается, только если задан CL_TEST_PG_DSN, например:

    docker run -d --name cl-test -e POSTGRES_PASSWORD=test -p 55433:5432 postgres:17-alpine
    CL_TEST_PG_DSN=postgresql://postgres:test@127.0.0.1:55433/postgres \
        uv run pytest tests/integration
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from config import Settings
from services import ImageRepository
from tests.conftest import image_base64

DSN = os.environ.get("CL_TEST_PG_DSN")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="не задан CL_TEST_PG_DSN"),
]

TABLE = "cl_test_images"


@pytest.fixture
async def table():
    settings = Settings(pg_dsn=DSN, table=TABLE)
    engine = create_async_engine(settings.sqlalchemy_dsn)
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
        await conn.execute(
            text(
                f"CREATE TABLE {TABLE} ("
                "id uuid PRIMARY KEY DEFAULT gen_random_uuid(), "
                "image text NOT NULL, answer text NOT NULL, "
                "created_at timestamptz NOT NULL DEFAULT now())"
            )
        )
        await conn.execute(
            text(
                f"INSERT INTO {TABLE} (image, answer, created_at) "
                "VALUES (:image, :answer, now() + make_interval(secs => :n))"
            ),
            [{"image": image_base64(str(i)), "answer": str(i), "n": i} for i in range(5)],
        )
    yield settings
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
    await engine.dispose()


async def test_keyset_pagination_walks_every_row(table):
    settings = table.model_copy(update={"cursor_column": "created_at"})
    repo = ImageRepository(settings)
    await repo.connect()
    try:
        seen, cursor = [], None
        while True:
            page = await repo.fetch_after(cursor, 2)
            if not page:
                break
            seen.extend(row.id for row in page)
            cursor = (page[-1].cursor, page[-1].id)
        assert len(seen) == 5
        assert len(set(seen)) == 5  # без дублей на границах страниц
        assert all(uuid.UUID(row_id) for row_id in seen)
    finally:
        await repo.close()


async def test_fetch_ids_and_by_ids(table):
    repo = ImageRepository(table)
    await repo.connect()
    try:
        ids = await repo.fetch_ids()
        assert len(ids) == 5
        rows = await repo.fetch_by_ids(ids[:2])
        assert {row.id for row in rows} == set(ids[:2])
        assert all(row.image and row.answer for row in rows)
        assert await repo.count() == 5
    finally:
        await repo.close()
