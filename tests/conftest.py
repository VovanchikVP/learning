from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from api.container import AppContainer
from config import Settings
from schemas import ImageRow
from services import PredictionService, StatusService
from training import ModelRegistry, TrainingService


def make_image(text: str, width: int = 64, height: int = 24) -> Image.Image:
    img = Image.new("L", (width, height), color=240)
    ImageDraw.Draw(img).text((2, 4), text, fill=20)
    return img


def image_base64(text: str = "test", **kwargs) -> str:
    buf = BytesIO()
    make_image(text, **kwargs).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeImageRepository:
    """Репозиторий в памяти: подменяет PostgreSQL в тестах."""

    def __init__(
        self, rows: list[ImageRow] | None = None, cursor_column: str | None = None
    ) -> None:
        self.rows = rows or []
        self.cursor_column = cursor_column
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def count(self) -> int:
        return len(self.rows)

    async def count_labeled(self) -> int:
        return len([row for row in self.rows if row.answer])

    async def fetch_ids(self) -> list[str]:
        return [row.id for row in self.rows]

    async def fetch_by_ids(self, ids: list[str]) -> list[ImageRow]:
        wanted = set(ids)
        return [row for row in self.rows if row.id in wanted]

    async def fetch_after(self, cursor: tuple[Any, str] | None, limit: int) -> list[ImageRow]:
        rows = sorted(self.rows, key=lambda r: (r.cursor, r.id))
        if cursor is not None:
            value, last_id = cursor
            rows = [r for r in rows if (r.cursor, r.id) > (value, last_id)]
        return rows[:limit]


def make_rows(count: int, *, with_cursor: bool = False, start: int = 0) -> list[ImageRow]:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(start, start + count):
        answer = str(10000 + i) if i % 2 else f"слово{i}"
        rows.append(
            ImageRow(
                id=f"0000000{i:04d}-0000-0000-0000-000000000000",
                image=image_base64(answer),
                answer=answer,
                cursor=base + timedelta(seconds=i) if with_cursor else None,
            )
        )
    return rows


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        workdir=tmp_path / "var",
        device="cpu",
        img_height=32,
        max_img_width=128,
        batch_size=8,
        initial_epochs=1,
        incremental_epochs=1,
        min_samples_to_train=8,
        retrain_threshold=4,
        early_stop_patience=1,
        num_workers=0,
    )


@pytest.fixture
def registry(settings: Settings) -> ModelRegistry:
    return ModelRegistry(settings)


@pytest.fixture
def repository() -> FakeImageRepository:
    return FakeImageRepository()


@pytest.fixture
def training(
    settings: Settings, registry: ModelRegistry, repository: FakeImageRepository
) -> TrainingService:
    return TrainingService(settings, registry, repository)


@pytest.fixture
def container(
    settings: Settings,
    registry: ModelRegistry,
    repository: FakeImageRepository,
    training: TrainingService,
) -> AppContainer:
    return AppContainer(
        settings=settings,
        registry=registry,
        repository=repository,
        training=training,
        prediction=PredictionService(registry),
        status=StatusService(registry, training),
    )
