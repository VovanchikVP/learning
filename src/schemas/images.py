from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from templates.api import FIELD_IMAGE_ROW_BASE64, FIELD_INGEST_ADDED, FIELD_ROW_CURSOR


class ImageRow(BaseModel):
    """Строка таблицы с размеченной картинкой."""

    id: str
    image: str = Field(description=FIELD_IMAGE_ROW_BASE64)
    answer: str
    cursor: Any = Field(default=None, description=FIELD_ROW_CURSOR)


class IngestStats(BaseModel):
    """Результат одного прохода по новым строкам базы."""

    added: dict[str, int] = Field(default_factory=dict, description=FIELD_INGEST_ADDED)
    skipped: int = 0

    @property
    def total(self) -> int:
        return sum(self.added.values())
