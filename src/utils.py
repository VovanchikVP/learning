from __future__ import annotations

import logging
import zlib

from constants.defaults import DEFAULT_LOG_LEVEL
from constants.training import VAL_BUCKETS
from templates.logs import LOG_DATE_FORMAT, LOG_FORMAT


def setup_logging(level: str = DEFAULT_LOG_LEVEL) -> None:
    """Вывод логов в stdout.

    Под uvicorn root-логгер остаётся без обработчиков: свои INFO-сообщения
    сервиса тогда никуда не попадают, в логах видны только строки uvicorn.
    Поэтому фабрика приложения зовёт эту функцию так же, как CLI.
    """
    if logging.getLogger().handlers:  # уже настроено снаружи — не трогаем
        return
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, target: str) -> float:
    if not target:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, target) / len(target)


def is_val_sample(sample_id: str, val_fraction: float) -> bool:
    """Детерминированное разбиение train/val по id.

    Важно для непрерывного обучения: один и тот же пример всегда попадает в ту же
    часть, поэтому метрики сравнимы между версиями модели и нет утечки.
    """
    if val_fraction <= 0:
        return False
    bucket = zlib.crc32(str(sample_id).encode()) % VAL_BUCKETS
    return bucket < val_fraction * VAL_BUCKETS
