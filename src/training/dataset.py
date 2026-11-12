from __future__ import annotations

import base64
import binascii
import random
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from constants import images as img_const
from templates import errors

from .charset import CharSet
from .networks import output_length


class ImageDecodeError(ValueError):
    pass


def decode_base64_image(value: str | bytes | memoryview) -> Image.Image:
    """base64 (в т.ч. data-URL) -> полутоновое изображение PIL."""
    if isinstance(value, memoryview):
        value = bytes(value)
    if isinstance(value, bytes):
        try:
            value = value.decode(img_const.ASCII_ENCODING)
        except UnicodeDecodeError:
            # колонка может хранить уже сырые байты картинки
            return _open_bytes(value)
    text = value.strip()
    if text.startswith(img_const.DATA_URL_PREFIX):
        _, _, text = text.partition(img_const.DATA_URL_SEPARATOR)
    try:
        raw = base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ImageDecodeError(errors.IMAGE_DECODE_FAILED.format(error=exc)) from exc
    return _open_bytes(raw)


def _open_bytes(raw: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as exc:  # noqa: BLE001 - PIL бросает разные типы
        raise ImageDecodeError(errors.IMAGE_OPEN_FAILED.format(error=exc)) from exc
    return img.convert(img_const.GRAYSCALE_MODE)


def _background(arr: np.ndarray) -> float:
    """Оценка цвета фона по рамке изображения."""
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return float(np.median(border))


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    fill = int(_background(np.asarray(img, dtype=np.uint8)))
    if rng.random() < img_const.AUG_ROTATE_PROB:
        degrees = img_const.AUG_ROTATE_DEGREES
        img = img.rotate(rng.uniform(-degrees, degrees), resample=Image.BILINEAR, fillcolor=fill)
    if rng.random() < img_const.AUG_SCALE_PROB:
        w, h = img.size
        scale = rng.uniform(*img_const.AUG_SCALE_RANGE)
        img = img.resize((max(img_const.AUG_MIN_WIDTH, int(w * scale)), h), Image.BILINEAR)
    if rng.random() < img_const.AUG_BRIGHTNESS_PROB:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(*img_const.AUG_BRIGHTNESS_RANGE))
    if rng.random() < img_const.AUG_CONTRAST_PROB:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(*img_const.AUG_CONTRAST_RANGE))
    return img


def preprocess(
    img: Image.Image,
    height: int,
    max_width: int,
    min_width: int = img_const.MIN_IMAGE_WIDTH,
    augment_rng: random.Random | None = None,
) -> np.ndarray:
    """PIL -> массив float32 [1, H, W] в диапазоне [-1, 1]."""
    if img.mode != img_const.GRAYSCALE_MODE:
        img = img.convert(img_const.GRAYSCALE_MODE)
    if augment_rng is not None:
        img = augment(img, augment_rng)
    w, h = img.size
    new_w = max(min_width, min(max_width, int(round(w * height / max(h, 1)))))
    img = img.resize((new_w, height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    if _background(arr) < img_const.DARK_BACKGROUND_THRESHOLD:
        # тёмный фон -> инвертируем, модель ждёт тёмный текст на светлом
        arr = img_const.PIXEL_MAX - arr
    arr = arr / img_const.NORM_SCALE - 1.0
    return arr[None, :, :]


@dataclass(slots=True)
class Batch:
    images: torch.Tensor  # [B, 1, H, W]
    input_lengths: torch.Tensor  # [B]
    targets: torch.Tensor  # [sum(target_lengths)]
    target_lengths: torch.Tensor  # [B]
    texts: list[str]
    ids: list[int]


class OCRDataset(Dataset):
    """Датасет поверх закэшированных на диске PNG-файлов."""

    def __init__(
        self,
        samples: list,
        charset: CharSet,
        height: int,
        max_width: int,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.samples = samples
        self.charset = charset
        self.height = height
        self.max_width = max_width
        self.augment = augment
        self._seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample.path)
        img.load()
        rng = random.Random(hash((self._seed, idx, sample.id))) if self.augment else None
        # ширины должно хватать на CTC-выравнивание
        min_width = min(
            self.max_width,
            img_const.MIN_IMAGE_WIDTH + img_const.WIDTH_PER_CHAR * len(sample.text),
        )
        arr = preprocess(img, self.height, self.max_width, min_width, rng)
        return arr, sample.text, sample.id


def make_collate(charset: CharSet):
    def collate(items) -> Batch:
        max_w = max(a.shape[2] for a, _, _ in items)
        # длина выходной последовательности должна покрывать самую длинную метку
        longest = max(len(t) for _, t, _ in items)
        while output_length(max_w) < longest + 1:
            max_w += img_const.WIDTH_STEP
        padded, lengths, targets, target_lengths, texts, ids = [], [], [], [], [], []
        for arr, text, sample_id in items:
            w = arr.shape[2]
            if w < max_w:
                pad_value = float(arr[:, :, -1].mean())
                arr = np.pad(arr, ((0, 0), (0, 0), (0, max_w - w)), constant_values=pad_value)
            padded.append(arr)
            lengths.append(output_length(max_w))
            encoded = charset.encode(text)
            targets.extend(encoded)
            target_lengths.append(len(encoded))
            texts.append(text)
            ids.append(sample_id)
        return Batch(
            images=torch.from_numpy(np.stack(padded)),
            input_lengths=torch.tensor(lengths, dtype=torch.long),
            targets=torch.tensor(targets, dtype=torch.long),
            target_lengths=torch.tensor(target_lengths, dtype=torch.long),
            texts=texts,
            ids=ids,
        )

    return collate
