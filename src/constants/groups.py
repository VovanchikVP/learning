"""Группы картинок и способы выбора модели."""

from typing import Final

WORDS: Final = "words"
DIGITS: Final = "digits"
GROUPS: Final[tuple[str, ...]] = (WORDS, DIGITS)

#: как была выбрана модель в ответе /predict
ROUTING_EXPLICIT: Final = "explicit"
ROUTING_CLASSIFIER: Final = "classifier"
ROUTING_OCR_CONFIDENCE: Final = "ocr_confidence"
