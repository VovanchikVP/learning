"""Константы процесса обучения."""

from typing import Final

GRAD_CLIP_NORM: Final = 5.0
WEIGHT_DECAY: Final = 1e-4
SCHEDULER_FACTOR: Final = 0.5
SCHEDULER_PATIENCE_DIVISOR: Final = 3
SCHEDULER_MIN_PATIENCE: Final = 2

#: улучшением считается только заметное падение CER
CER_IMPROVEMENT_EPS: Final = 1e-5
#: если валидационной выборки нет, качество меряем по части обучающей
FALLBACK_EVAL_SAMPLES: Final = 64
MIN_EVAL_BATCH_SIZE: Final = 8
#: сколько запусков обучения хранить в истории чекпоинта
HISTORY_LIMIT: Final = 50
INCREMENTAL_PATIENCE_DIVISOR: Final = 2
INCREMENTAL_MIN_PATIENCE: Final = 3

#: числа короче этого в обучение не берём: такие ответы приходят из полей,
#: где цифра — это счётчик или код, а не текст с картинки
MIN_DIGITS_ANSWER_LENGTH: Final = 5

#: детерминированное разбиение train/val: crc32(id) % VAL_BUCKETS
VAL_BUCKETS: Final = 1000

# классификатор групп
CLASSIFIER_EPOCHS: Final = 12
CLASSIFIER_LR: Final = 1e-3
CLASSIFIER_BATCH_SIZE: Final = 64
CLASSIFIER_MIN_SAMPLES: Final = 32
CLASSIFIER_EVAL_FALLBACK: Final = 128
PERFECT_ACCURACY: Final = 1.0

# округление метрик в ответах API
CONFIDENCE_ROUND_DIGITS: Final = 4
METRIC_ROUND_DIGITS: Final = 5
