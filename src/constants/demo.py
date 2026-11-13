"""Константы генератора синтетических данных."""

from typing import Final

FONT_CANDIDATES: Final[tuple[str, ...]] = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

DEMO_WORDS: Final[tuple[str, ...]] = (
    "корова",
    "машина",
    "дорога",
    "солнце",
    "книга",
    "ветер",
    "яблоко",
    "город",
    "река",
    "лампа",
    "стол",
    "окно",
    "кошка",
    "собака",
    "берег",
    "поле",
    "зима",
    "осень",
    "весна",
    "лето",
    "море",
    "лес",
    "небо",
    "дом",
)

FONT_SIZE_RANGE: Final[tuple[int, int]] = (22, 30)
PADDING_RANGE: Final[tuple[int, int]] = (4, 10)
BACKGROUND_RANGE: Final[tuple[int, int]] = (215, 255)
TEXT_FILL_RANGE: Final[tuple[int, int]] = (0, 60)
NOISE_PROB: Final = 0.5
NOISE_SHARE: Final = 0.02
NOISE_VALUE_RANGE: Final[tuple[int, int]] = (0, 255)
ROTATE_DEGREES: Final = 2.0
DIGITS_LENGTH_RANGE: Final[tuple[int, int]] = (3, 6)
DIGIT_RANGE: Final[tuple[int, int]] = (0, 9)
