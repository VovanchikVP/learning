"""Тексты ошибок и исключений."""

from typing import Final

# декодирование картинок
IMAGE_DECODE_FAILED: Final = "не удалось декодировать base64: {error}"
IMAGE_OPEN_FAILED: Final = "не удалось открыть изображение: {error}"

# настройки
INVALID_IDENTIFIER: Final = "недопустимое имя объекта БД: {name!r}"
INVALID_IMG_HEIGHT: Final = "img_height должна быть кратна {divisor} (архитектура CRNN)"
INVALID_NETWORK_HEIGHT: Final = "img_height должна быть кратна {divisor}"

# репозиторий
REPOSITORY_NOT_CONNECTED: Final = "репозиторий не подключён, вызовите connect()"
CURSOR_COLUMN_NOT_SET: Final = "cursor_column не настроена"

# инференс и обучение
GROUP_MODEL_NOT_READY: Final = "модель группы {group} ещё не обучена"
NO_MODEL_READY: Final = "ни одна модель ещё не обучена"
CLASSIFIER_NOT_READY: Final = "классификатор групп ещё не обучен"
TRAINING_IN_PROGRESS: Final = "обучение уже идёт"
UNKNOWN_GROUP: Final = "неизвестная группа {group}"
