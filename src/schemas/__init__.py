"""Слой схем данных: DTO для API и сервисов."""

from .common import GroupName, RoutingMode
from .images import ImageRow, IngestStats
from .prediction import Candidate, PredictionResult, PredictRequest
from .status import ClassifierStatus, CursorState, GroupStatus, ServiceStatus, WatcherStatus
from .training import (
    ClassifierTrainResult,
    EvalResult,
    GroupTrainingState,
    TrainKind,
    TrainResult,
)

__all__ = [
    "Candidate",
    "ClassifierStatus",
    "ClassifierTrainResult",
    "CursorState",
    "EvalResult",
    "GroupName",
    "GroupStatus",
    "GroupTrainingState",
    "ImageRow",
    "IngestStats",
    "PredictRequest",
    "PredictionResult",
    "RoutingMode",
    "ServiceStatus",
    "TrainKind",
    "TrainResult",
    "WatcherStatus",
]
