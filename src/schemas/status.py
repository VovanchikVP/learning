from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from constants.defaults import DEFAULT_RETRAIN_THRESHOLD
from templates.api import (
    FIELD_LABELED_ROWS,
    FIELD_LAST_ERROR,
    FIELD_LAST_POLL,
    FIELD_PENDING,
    FIELD_TABLE_ROWS,
)

from .common import GroupName
from .training import TrainResult


class CursorState(BaseModel):
    """Позиция keyset-курсора: значение cursor_column и id строки."""

    value: Any = None
    is_datetime: bool = False
    id: str = ""


class GroupStatus(BaseModel):
    group: GroupName
    ready: bool
    training: bool
    version: int
    device: str
    alphabet: str
    cached_samples: int
    trained_samples: int
    val_cer: float | None = None
    val_accuracy: float = 0.0
    updated_at: float = 0.0
    pending: int = Field(default=0, description=FIELD_PENDING)
    last_run: TrainResult | None = None


class ClassifierStatus(BaseModel):
    ready: bool
    training: bool
    version: int
    val_accuracy: float = 0.0
    trained_samples: int = 0
    updated_at: float = 0.0


class WatcherStatus(BaseModel):
    """Состояние фонового цикла: он наполняет кэш и запускает дообучение."""

    running: bool = False
    table_rows: int = Field(default=0, description=FIELD_TABLE_ROWS)
    labeled_rows: int = Field(default=0, description=FIELD_LABELED_ROWS)
    last_poll_at: float = Field(default=0.0, description=FIELD_LAST_POLL)
    last_error: str | None = Field(default=None, description=FIELD_LAST_ERROR)


class ServiceStatus(BaseModel):
    cursor_mode: str
    watcher: WatcherStatus = Field(default_factory=WatcherStatus)
    cursor: CursorState | None = None
    known_rows: int = 0
    skipped_rows: int = 0
    retrain_threshold: int = DEFAULT_RETRAIN_THRESHOLD
    classifier: ClassifierStatus
    groups: dict[str, GroupStatus] = Field(default_factory=dict)
