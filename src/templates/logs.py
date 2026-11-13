"""Шаблоны сообщений в лог.

Плейсхолдеры в стиле %, чтобы logging форматировал строку лениво — только если
сообщение действительно будет записано.
"""

from typing import Final

LOG_FORMAT: Final = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_DATE_FORMAT: Final = "%H:%M:%S"

# хранилище
STORE_LOADED: Final = "[%s] загружено из кэша: %d примеров"
STORE_BROKEN_LINE: Final = "[%s] пропущена битая строка labels.jsonl"

# база
DB_CONNECTED: Final = "подключение к PostgreSQL установлено (режим: %s)"
DB_MODE_KEYSET: Final = "keyset по %s"
DB_MODE_DIFF: Final = "сверка id"

# приём данных
STATE_RESTORED: Final = "состояние восстановлено: курсор=%s, ожидают дообучения %s"
ROW_SKIPPED: Final = "строка id=%s пропущена: %s"
ROW_IGNORED_SHORT_NUMBER: Final = "строка id=%s пропущена: число %r короче %d символов"
NEW_IDS_FOUND: Final = "новых id в базе: %d"
TABLE_SUMMARY: Final = "в таблице %s строк, из них с ответом %s; в локальном кэше %s"
BOOTSTRAP_DONE: Final = "загружено новых строк: %s"
POLL_RESULT: Final = "новых строк: %s; в очереди на дообучение: %s"
BOOTSTRAP_FAILED: Final = "не удалось прочитать таблицу, повтор через %.0fs"
POLL_FAILED: Final = "опрос базы не удался, повтор через %.0fs"
WATCHER_STOPPED: Final = "фоновый цикл остановлен"

# обучение
TRAIN_FULL_START: Final = "[%s] старт полного обучения на %d примерах"
TRAIN_INCREMENTAL_START: Final = "[%s] старт дообучения на %d новых примерах"
TRAIN_DONE: Final = "[%s] обучение завершено за %.1fs: val_cer=%s val_acc=%s принято=%s %s"
EPOCH: Final = "[%s] эпоха %d/%d loss=%.4f val_cer=%s val_acc=%.3f"
EARLY_STOP: Final = "[%s] ранняя остановка на эпохе %d"
INVALID_LOSS: Final = "[%s] пропущен батч с невалидным лоссом"
ALPHABET_EXTENDED: Final = "[%s] алфавит расширен: %s"
ROLLED_BACK: Final = "[%s] %s"
CHECKPOINT_SAVED: Final = "[%s] чекпоинт сохранён: v%d cer=%s"
CHECKPOINT_LOADED: Final = "[%s] загружен чекпоинт v%d (val_cer=%s, примеров=%d)"

# классификатор групп
CLASSIFIER_EPOCH: Final = "[classifier] эпоха %d/%d val_acc=%.4f"
CLASSIFIER_CHECKPOINT_LOADED: Final = "[classifier] загружен чекпоинт v%d (acc=%.4f)"
CLASSIFIER_TRAINED: Final = "[classifier] v%d, val_acc=%.4f"
CLASSIFIER_GROUP_NOT_READY: Final = "классификатор выбрал %s, но эта модель не обучена"

#: значение метрики, которую ещё не измеряли
METRIC_NOT_AVAILABLE: Final = "n/a"
