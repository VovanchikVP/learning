from __future__ import annotations

import torch
from torch import nn

from constants.networks import (
    BLANK_INDEX,
    CRNN_CHANNELS,
    ENCODER_OUT_CHANNELS,
    HEIGHT_DIVISOR,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    WIDTH_DOWNSCALE,
)
from templates import errors


def output_length(width: int) -> int:
    """Длина последовательности на выходе энкодера для картинки шириной width."""
    return max(1, width // WIDTH_DOWNSCALE - 1)


class CRNN(nn.Module):
    """Классический CRNN: CNN-энкодер + двухслойный BiLSTM + CTC-голова."""

    def __init__(
        self,
        n_classes: int,
        img_height: int = HEIGHT_DIVISOR,
        lstm_hidden: int = LSTM_HIDDEN,
    ) -> None:
        super().__init__()
        if img_height % HEIGHT_DIVISOR != 0:
            raise ValueError(errors.INVALID_NETWORK_HEIGHT.format(divisor=HEIGHT_DIVISOR))
        self.img_height = img_height
        self.n_classes = n_classes
        self.lstm_hidden = lstm_hidden

        c1, c2, c3, c4 = CRNN_CHANNELS
        # fmt: off
        self.cnn = nn.Sequential(
            nn.Conv2d(1, c1, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                   # H/2,  W/2
            nn.Conv2d(c1, c2, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                   # H/4,  W/4
            nn.Conv2d(c2, c3, 3, 1, 1), nn.BatchNorm2d(c3), nn.ReLU(True),
            nn.Conv2d(c3, c3, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                         # H/8,  W/4
            nn.Conv2d(c3, c4, 3, 1, 1), nn.BatchNorm2d(c4), nn.ReLU(True),
            nn.Conv2d(c4, c4, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                         # H/16, W/4
            nn.Conv2d(c4, ENCODER_OUT_CHANNELS, 2, 1, 0),
            nn.BatchNorm2d(ENCODER_OUT_CHANNELS), nn.ReLU(True),  # H/32, W/4-1
        )
        # fmt: on
        self.rnn = nn.LSTM(
            ENCODER_OUT_CHANNELS,
            lstm_hidden,
            num_layers=LSTM_LAYERS,
            bidirectional=True,
            batch_first=False,
        )
        self.head = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """[B,1,H,W] -> логиты [T,B,C]."""
        feats = self.cnn(images)
        h = feats.shape[2]
        if h != 1:
            feats = feats.mean(dim=2, keepdim=True)
        feats = feats.squeeze(2).permute(2, 0, 1)  # [W, B, C]
        out, _ = self.rnn(feats)
        return self.head(out)

    def expand_classes(self, n_classes: int) -> None:
        """Расширяет выходной слой под новые символы, сохраняя старые веса."""
        if n_classes <= self.n_classes:
            return
        old = self.head
        new = nn.Linear(old.in_features, n_classes).to(old.weight.device)
        with torch.no_grad():
            new.weight[: old.out_features] = old.weight
            new.bias[: old.out_features] = old.bias
        self.head = new
        self.n_classes = n_classes


@torch.no_grad()
def greedy_decode(
    logits: torch.Tensor, input_lengths: torch.Tensor, charset
) -> tuple[list[str], list[float]]:
    """CTC greedy decode. Возвращает тексты и среднюю уверенность по кадрам."""
    probs = logits.softmax(dim=2)  # [T, B, C]
    conf, indices = probs.max(dim=2)
    texts: list[str] = []
    confidences: list[float] = []
    for b in range(indices.shape[1]):
        t = int(input_lengths[b])
        seq = indices[:t, b].tolist()
        collapsed, prev = [], -1
        for i in seq:
            if i != prev and i != BLANK_INDEX:
                collapsed.append(i)
            prev = i
        texts.append(charset.decode(collapsed))
        confidences.append(float(conf[:t, b].mean()))
    return texts, confidences
