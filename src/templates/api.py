"""Текст, который видит пользователь API: заголовки, описания, подписи полей."""

from typing import Final

API_TITLE: Final = "Continuous OCR Learning"

# эндпоинты
SUMMARY_HEALTH: Final = "Живость сервиса"
SUMMARY_STATUS: Final = "Состояние моделей и очередей"
SUMMARY_PREDICT: Final = "Распознать картинку"
SUMMARY_TRAIN: Final = "Запустить обучение вручную"

# поля схем
FIELD_IMAGE_BASE64: Final = "картинка в base64, допускается data-URL"
FIELD_IMAGE_ROW_BASE64: Final = "картинка в base64"
FIELD_PREDICT_GROUP: Final = "если не задана, группу выбирает классификатор"
FIELD_ROW_CURSOR: Final = "значение cursor_column, если она настроена"
FIELD_INGEST_ADDED: Final = "принято строк по группам"
FIELD_BASELINE_CER: Final = "val_cer до дообучения, с ним сравнивается результат"
FIELD_GROUP_CONFIDENCE: Final = "уверенность классификатора групп"
FIELD_PENDING: Final = "новых строк в очереди на дообучение"

#: режим поиска новых строк, когда настроена колонка-курсор
CURSOR_MODE_KEYSET: Final = "keyset:{column}"
