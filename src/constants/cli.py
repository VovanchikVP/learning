"""Константы командной строки."""

from typing import Final

PROG_NAME: Final = "continuous-learning"
DEFAULT_LOG_LEVEL: Final = "INFO"
JSON_INDENT: Final = 2

DEFAULT_SEED_COUNT: Final = 300
DEFAULT_SEED_RANDOM_SEED: Final = 0

DEFAULT_SELFTEST_WORKDIR: Final = "var/selftest"
DEFAULT_SELFTEST_INITIAL: Final = 300
DEFAULT_SELFTEST_NEW: Final = 100
DEFAULT_SELFTEST_EPOCHS: Final = 25
SELFTEST_MIN_SAMPLES: Final = 16
SELFTEST_INCREMENTAL_EPOCH_DIVISOR: Final = 4
SELFTEST_MIN_INCREMENTAL_EPOCHS: Final = 2
SELFTEST_SEED_INITIAL: Final = 1
SELFTEST_SEED_NEW: Final = 2
SELFTEST_SEED_PREDICT: Final = 99

# имена команд
COMMAND_RUN: Final = "run"
COMMAND_WATCH: Final = "watch"
COMMAND_BOOTSTRAP: Final = "bootstrap"
COMMAND_TRAIN: Final = "train"
COMMAND_STATUS: Final = "status"
COMMAND_PREDICT: Final = "predict"
COMMAND_SEED_DB: Final = "seed-db"
COMMAND_SELFTEST: Final = "selftest"
