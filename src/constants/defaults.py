"""Значения по умолчанию для настроек приложения."""

from pathlib import Path
from typing import Final

ENV_PREFIX: Final = "CL_"
ENV_FILE: Final = ".env"

# база
DEFAULT_PG_DSN: Final = "postgresql://postgres:postgres@localhost:5432/postgres"
DEFAULT_TABLE: Final = "captcha"
DEFAULT_ID_COLUMN: Final = "id"
DEFAULT_IMAGE_COLUMN: Final = "image"
DEFAULT_ANSWER_COLUMN: Final = "answer"
DEFAULT_CURSOR_COLUMN: Final[str | None] = None
DEFAULT_FETCH_BATCH: Final = 500
DEFAULT_POLL_INTERVAL: Final = 5.0

# хранилище
DEFAULT_WORKDIR: Final = Path("var")

# препроцессинг
DEFAULT_IMG_HEIGHT: Final = 32
DEFAULT_MAX_IMG_WIDTH: Final = 384

# обучение
DEVICE_AUTO: Final = "auto"
DEFAULT_DEVICE: Final = DEVICE_AUTO
DEFAULT_BATCH_SIZE: Final = 32
DEFAULT_LR: Final = 3e-4
DEFAULT_INITIAL_EPOCHS: Final = 80
DEFAULT_INCREMENTAL_EPOCHS: Final = 10
DEFAULT_INCREMENTAL_LR: Final = 1e-4
DEFAULT_EARLY_STOP_PATIENCE: Final = 12
DEFAULT_VAL_FRACTION: Final = 0.1
DEFAULT_MIN_SAMPLES_TO_TRAIN: Final = 64
DEFAULT_RETRAIN_THRESHOLD: Final = 100
DEFAULT_REPLAY_RATIO: Final = 4.0
DEFAULT_CER_REGRESSION_TOLERANCE: Final = 0.01
DEFAULT_NUM_WORKERS: Final = 0

DEFAULT_LOG_LEVEL: Final = "INFO"

# API
DEFAULT_API_HOST: Final = "127.0.0.1"
DEFAULT_API_PORT: Final = 8000
