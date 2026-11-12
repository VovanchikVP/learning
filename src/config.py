from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from constants import defaults
from constants.db import (
    ASYNC_DRIVER_DSN_PREFIX,
    ASYNC_DRIVER_MARKER,
    IDENTIFIER_RE,
    SYNC_DSN_PREFIXES,
)
from constants.groups import DIGITS, WORDS
from constants.networks import HEIGHT_DIVISOR
from constants.storage import CHECKPOINT_DIRNAME, DATA_DIRNAME, STATE_FILENAME
from templates import errors


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=defaults.ENV_FILE,
        env_prefix=defaults.ENV_PREFIX,
        extra="ignore",
        case_sensitive=False,
    )

    # --- база ---
    pg_dsn: str = defaults.DEFAULT_PG_DSN
    table: str = defaults.DEFAULT_TABLE
    id_column: str = defaults.DEFAULT_ID_COLUMN
    image_column: str = defaults.DEFAULT_IMAGE_COLUMN
    answer_column: str = defaults.DEFAULT_ANSWER_COLUMN
    # Колонка для keyset-обхода новых строк (например created_at). Нужна потому,
    # что id в таблице — uuid и по нему нельзя понять, какие строки новые.
    # Если её нет, сервис переходит на сверку списка id с локальным кэшем.
    cursor_column: str | None = defaults.DEFAULT_CURSOR_COLUMN
    fetch_batch: int = defaults.DEFAULT_FETCH_BATCH
    poll_interval: float = defaults.DEFAULT_POLL_INTERVAL

    # --- хранилище ---
    workdir: Path = defaults.DEFAULT_WORKDIR

    # --- препроцессинг ---
    img_height: int = defaults.DEFAULT_IMG_HEIGHT
    max_img_width: int = defaults.DEFAULT_MAX_IMG_WIDTH

    # --- обучение ---
    device: str = defaults.DEFAULT_DEVICE
    batch_size: int = defaults.DEFAULT_BATCH_SIZE
    lr: float = defaults.DEFAULT_LR
    initial_epochs: int = defaults.DEFAULT_INITIAL_EPOCHS
    incremental_epochs: int = defaults.DEFAULT_INCREMENTAL_EPOCHS
    incremental_lr: float = defaults.DEFAULT_INCREMENTAL_LR
    early_stop_patience: int = defaults.DEFAULT_EARLY_STOP_PATIENCE
    val_fraction: float = defaults.DEFAULT_VAL_FRACTION
    min_samples_to_train: int = defaults.DEFAULT_MIN_SAMPLES_TO_TRAIN
    retrain_threshold: int = defaults.DEFAULT_RETRAIN_THRESHOLD
    # сколько старых примеров подмешивать на каждый новый при дообучении
    replay_ratio: float = defaults.DEFAULT_REPLAY_RATIO
    # насколько валидационный CER может ухудшиться, чтобы дообучение всё же приняли
    cer_regression_tolerance: float = defaults.DEFAULT_CER_REGRESSION_TOLERANCE
    num_workers: int = defaults.DEFAULT_NUM_WORKERS

    # --- API ---
    api_host: str = defaults.DEFAULT_API_HOST
    api_port: int = defaults.DEFAULT_API_PORT

    @field_validator("img_height")
    @classmethod
    def _check_img_height(cls, v: int) -> int:
        if v < HEIGHT_DIVISOR or v % HEIGHT_DIVISOR != 0:
            raise ValueError(errors.INVALID_IMG_HEIGHT.format(divisor=HEIGHT_DIVISOR))
        return v

    @field_validator("table", "id_column", "image_column", "answer_column", "cursor_column")
    @classmethod
    def _check_identifier(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if not IDENTIFIER_RE.match(v):
            raise ValueError(errors.INVALID_IDENTIFIER.format(name=v))
        return v

    @property
    def sqlalchemy_dsn(self) -> str:
        """DSN с драйвером asyncpg для SQLAlchemy."""
        dsn = self.pg_dsn
        if dsn.startswith(ASYNC_DRIVER_MARKER):
            return dsn
        for prefix in SYNC_DSN_PREFIXES:
            if dsn.startswith(prefix):
                return ASYNC_DRIVER_DSN_PREFIX + dsn[len(prefix) :]
        return dsn

    @property
    def data_dir(self) -> Path:
        return self.workdir / DATA_DIRNAME

    @property
    def checkpoint_dir(self) -> Path:
        return self.workdir / CHECKPOINT_DIRNAME

    @property
    def state_path(self) -> Path:
        return self.workdir / STATE_FILENAME


def classify_answer(answer: str) -> str:
    """Группа строки: только цифры -> digits, иначе -> words."""
    cleaned = (answer or "").strip()
    if cleaned and all(ch.isdigit() for ch in cleaned):
        return DIGITS
    return WORDS
