"""Пояснения к результату обучения (поле note у TrainResult)."""

from typing import Final

NOT_ENOUGH_DATA: Final = "мало данных: {count} < {minimum}"
ALL_NEW_IN_VALIDATION: Final = "все новые примеры попали в валидацию"
ROLLBACK: Final = "откат: val_cer {cer:.4f} > базового {baseline:.4f} + {tolerance}"
CLASSIFIER_NEEDS_BOTH_GROUPS: Final = "нужны данные обеих групп"
