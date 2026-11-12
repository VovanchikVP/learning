from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from templates.api import FIELD_BASELINE_CER

from .common import GroupName


class TrainKind(StrEnum):
    FULL = "full"
    INCREMENTAL = "incremental"


class TrainResult(BaseModel):
    """Итог одного запуска обучения."""

    group: GroupName
    kind: TrainKind = TrainKind.FULL
    started_at: float = 0.0
    duration_sec: float = 0.0
    epochs: int = 0
    train_samples: int = 0
    new_samples: int = 0
    val_samples: int = 0
    train_loss: float | None = None
    val_cer: float | None = None
    val_accuracy: float | None = None
    baseline_cer: float | None = Field(default=None, description=FIELD_BASELINE_CER)
    accepted: bool = True
    note: str = ""


class GroupTrainingState(BaseModel):
    """Состояние модели группы, которое хранится в чекпоинте."""

    version: int = 0
    trained_samples: int = 0
    best_val_cer: float | None = None
    best_val_accuracy: float = 0.0
    updated_at: float = 0.0
    history: list[TrainResult] = Field(default_factory=list)


class EvalResult(BaseModel):
    """Метрики на выборке."""

    cer: float | None = None
    accuracy: float = 0.0
    count: int = 0

    @property
    def measured(self) -> bool:
        return self.count > 0 and self.cer is not None


class ClassifierTrainResult(BaseModel):
    """Итог переобучения классификатора групп."""

    trained: bool = False
    version: int = 0
    val_accuracy: float = 0.0
    samples: int = 0
    duration_sec: float = 0.0
    note: str = ""
