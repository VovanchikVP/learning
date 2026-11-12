"""Константы архитектуры нейросетей."""

from typing import Final

#: индекс CTC-blank в алфавите
BLANK_INDEX: Final = 0

#: CNN-энкодер уменьшает ширину в 4 раза, последняя свёртка 2x2 без паддинга
#: снимает ещё один кадр
WIDTH_DOWNSCALE: Final = 4
#: четыре пулинга делят высоту на 16, финальная свёртка 2x2 — ещё на 2
HEIGHT_DIVISOR: Final = 32

CRNN_CHANNELS: Final[tuple[int, ...]] = (64, 128, 256, 512)
ENCODER_OUT_CHANNELS: Final = 512
LSTM_HIDDEN: Final = 256
LSTM_LAYERS: Final = 2

#: классификатор групп работает с картинкой фиксированной ширины
CLASSIFIER_INPUT_WIDTH: Final = 128
CLASSIFIER_CHANNELS: Final[tuple[int, ...]] = (16, 32, 64)
CLASSIFIER_CLASSES: Final = 2
