"""Синтетические данные: самопроверка пайплайна и наполнение демо-таблицы."""

from __future__ import annotations

import base64
import random
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from constants import demo as dc
from constants.groups import DIGITS
from constants.images import GRAYSCALE_MODE
from constants.storage import IMAGE_FORMAT


def find_fonts() -> list[Path]:
    return [Path(p) for p in dc.FONT_CANDIDATES if Path(p).exists()]


def render(text: str, rng: random.Random, fonts: list[Path]) -> Image.Image:
    size = rng.randint(*dc.FONT_SIZE_RANGE)
    font = ImageFont.truetype(str(rng.choice(fonts)), size) if fonts else ImageFont.load_default()
    pad = rng.randint(*dc.PADDING_RANGE)
    tmp = ImageDraw.Draw(Image.new(GRAYSCALE_MODE, (1, 1)))
    box = tmp.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
    bg = rng.randint(*dc.BACKGROUND_RANGE)
    img = Image.new(GRAYSCALE_MODE, (w, h), color=bg)
    draw = ImageDraw.Draw(img)
    draw.text((pad - box[0], pad - box[1]), text, fill=rng.randint(*dc.TEXT_FILL_RANGE), font=font)
    if rng.random() < dc.NOISE_PROB:  # лёгкий шум, чтобы данные не были стерильными
        px = img.load()
        for _ in range(int(w * h * dc.NOISE_SHARE)):
            px[rng.randrange(w), rng.randrange(h)] = rng.randint(*dc.NOISE_VALUE_RANGE)
    return img.rotate(
        rng.uniform(-dc.ROTATE_DEGREES, dc.ROTATE_DEGREES), resample=Image.BILINEAR, fillcolor=bg
    )


def sample_text(group: str, rng: random.Random) -> str:
    if group == DIGITS:
        length = rng.randint(*dc.DIGITS_LENGTH_RANGE)
        return "".join(str(rng.randint(*dc.DIGIT_RANGE)) for _ in range(length))
    return rng.choice(dc.DEMO_WORDS)


def generate(group: str, count: int, seed: int = 0) -> list[tuple[str, str]]:
    """Возвращает список (base64_png, answer)."""
    rng = random.Random(seed)
    fonts = find_fonts()
    out = []
    for _ in range(count):
        text = sample_text(group, rng)
        buf = BytesIO()
        render(text, rng, fonts).save(buf, format=IMAGE_FORMAT)
        out.append((base64.b64encode(buf.getvalue()).decode(), text))
    return out
