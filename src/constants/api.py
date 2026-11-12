"""Константы HTTP-слоя."""

from typing import Final

API_VERSION: Final = "0.1.0"

ROUTE_HEALTH: Final = "/health"
ROUTE_STATUS: Final = "/status"
ROUTE_PREDICT: Final = "/predict"
ROUTE_TRAIN: Final = "/train/{group}"

TAG_SERVICE: Final = "service"
TAG_INFERENCE: Final = "inference"
TAG_TRAINING: Final = "training"

HEALTH_STATUS_FIELD: Final = "status"
HEALTH_STATUS_OK: Final = "ok"

BACKGROUND_TASK_NAME: Final = "continuous-learning"
TRAIN_TASK_NAME: Final = "train-{group}-{kind}"
