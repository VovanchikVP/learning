"""Текст командной строки: справка и вывод команд."""

from typing import Final

DESCRIPTION: Final = "Непрерывное обучение OCR-моделей по таблице в PostgreSQL"

# справка по командам
HELP_RUN: Final = "HTTP API + непрерывное обучение"
HELP_WATCH: Final = "только непрерывное обучение, без HTTP"
HELP_BOOTSTRAP: Final = "выкачать базу и обучить обе модели с нуля"
HELP_FORCE: Final = "переобучить, даже если чекпоинт есть"
HELP_TRAIN: Final = "запустить обучение вручную"
HELP_STATUS: Final = "показать состояние моделей"
HELP_PREDICT: Final = "распознать картинку из файла"
HELP_SEED_DB: Final = "создать демо-таблицу и налить синтетику"
HELP_SEED_COUNT: Final = "строк на группу"
HELP_SELFTEST: Final = "проверка пайплайна на синтетике без БД"

# вывод команд
SKIP_NOT_ENOUGH: Final = "[{group}] пропуск: всего {count} примеров"
SKIP_ALREADY_TRAINED: Final = "[{group}] модель уже обучена, используйте --force"
TRAIN_RESULT: Final = "[{group}] {payload}"
SEED_DONE: Final = "добавлено строк: {count} (по {per_group} на группу)"

SELFTEST_STEP_GENERATE: Final = "1) генерируем по {count} примеров на группу"
SELFTEST_ADDED: Final = "   добавлено: {added}"
SELFTEST_FULL_RESULT: Final = "   [{group}] полное обучение: cer={cer:.4f} acc={accuracy:.3f}"
SELFTEST_STEP_INCREMENT: Final = "2) доливаем по {count} новых примеров -> дообучение"
SELFTEST_INCREMENTAL_RESULT: Final = (
    "   [{group}] дообучение: новых={new} cer {baseline:.4f} -> {cer:.4f} "
    "acc={accuracy:.3f} принято={accepted} {note}"
)
SELFTEST_STEP_PREDICT: Final = "3) инференс по случайной картинке"
SELFTEST_PREDICTION: Final = "   [{group}] истина={answer!r} -> {predicted}:{text!r} (routed_by={routed_by})"
