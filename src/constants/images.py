"""Константы декодирования, препроцессинга и аугментации картинок."""

from typing import Final

GRAYSCALE_MODE: Final = "L"
DATA_URL_PREFIX: Final = "data:"
DATA_URL_SEPARATOR: Final = ","
ASCII_ENCODING: Final = "ascii"

PIXEL_MAX: Final = 255.0
#: нормализация в диапазон [-1, 1]
NORM_SCALE: Final = 127.5
#: фон темнее этого порога считается инверсным, картинка переворачивается
DARK_BACKGROUND_THRESHOLD: Final = 128

MIN_IMAGE_WIDTH: Final = 16
#: ширины должно хватать на CTC-выравнивание
WIDTH_PER_CHAR: Final = 8
#: шаг, которым добирается ширина батча до нужной длины последовательности
WIDTH_STEP: Final = 4

AUG_MIN_WIDTH: Final = 8
AUG_ROTATE_PROB: Final = 0.7
AUG_ROTATE_DEGREES: Final = 2.5
AUG_SCALE_PROB: Final = 0.5
AUG_SCALE_RANGE: Final[tuple[float, float]] = (0.9, 1.12)
AUG_BRIGHTNESS_PROB: Final = 0.5
AUG_BRIGHTNESS_RANGE: Final[tuple[float, float]] = (0.8, 1.2)
AUG_CONTRAST_PROB: Final = 0.5
AUG_CONTRAST_RANGE: Final[tuple[float, float]] = (0.7, 1.3)
