"""Имена файлов и директорий локального хранилища."""

import re
from typing import Final

DATA_DIRNAME: Final = "data"
CHECKPOINT_DIRNAME: Final = "checkpoints"
IMAGES_DIRNAME: Final = "images"
LABELS_FILENAME: Final = "labels.jsonl"
STATE_FILENAME: Final = "state.json"

CHECKPOINT_SUFFIX: Final = ".pt"
CLASSIFIER_CHECKPOINT_NAME: Final = f"group_classifier{CHECKPOINT_SUFFIX}"
#: во временный файл пишем и только потом переименовываем — чтобы не получить
#: обрезанный чекпоинт при падении посреди записи
TMP_SUFFIX: Final = ".tmp"

IMAGE_FORMAT: Final = "PNG"
IMAGE_EXTENSION: Final = ".png"

ENCODING: Final = "utf-8"
JSON_INDENT: Final = 2

#: безопасное имя файла для id; всё остальное заменяется на sha256
SAFE_NAME_RE: Final = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")

# ключи внутри чекпоинта модели
CKPT_GROUP: Final = "group"
CKPT_CHARS: Final = "chars"
CKPT_IMG_HEIGHT: Final = "img_height"
CKPT_MODEL: Final = "model"
CKPT_STATE: Final = "state"
CKPT_VERSION: Final = "version"
CKPT_VAL_ACCURACY: Final = "val_accuracy"
CKPT_TRAINED_SAMPLES: Final = "trained_samples"
CKPT_UPDATED_AT: Final = "updated_at"

# ключи в state.json
STATE_CURSOR: Final = "cursor"
STATE_PENDING: Final = "pending"
STATE_SKIPPED_IDS: Final = "skipped_ids"
STATE_UPDATED_AT: Final = "updated_at"
