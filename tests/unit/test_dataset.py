import base64

import numpy as np
import pytest
from PIL import Image

from training.charset import CharSet
from training.dataset import (
    ImageDecodeError,
    decode_base64_image,
    make_collate,
    preprocess,
)
from training.networks import output_length
from tests.conftest import image_base64, make_image


def test_decode_plain_and_data_url():
    payload = image_base64("кот")
    assert decode_base64_image(payload).mode == "L"
    assert decode_base64_image(f"data:image/png;base64,{payload}").mode == "L"
    assert decode_base64_image(payload.encode()).mode == "L"


def test_decode_raw_image_bytes():
    raw = base64.b64decode(image_base64("кот"))
    assert decode_base64_image(raw).mode == "L"


@pytest.mark.parametrize("payload", ["не base64!!", "", "YWJj"])
def test_decode_rejects_garbage(payload):
    with pytest.raises(ImageDecodeError):
        decode_base64_image(payload)


def test_preprocess_normalizes_shape_and_range():
    arr = preprocess(make_image("кот", width=200, height=50), height=32, max_width=96)
    assert arr.shape[0] == 1 and arr.shape[1] == 32
    assert arr.shape[2] <= 96
    assert -1.0 <= arr.min() and arr.max() <= 1.0


def test_preprocess_inverts_dark_background():
    dark = Image.new("L", (40, 20), color=10)
    light = Image.new("L", (40, 20), color=245)
    assert preprocess(dark, 32, 64).mean() == pytest.approx(preprocess(light, 32, 64).mean(), abs=0.05)


def test_preprocess_respects_min_width():
    arr = preprocess(make_image("x", width=10, height=20), height=32, max_width=96, min_width=48)
    assert arr.shape[2] == 48


def test_collate_pads_and_fits_ctc_lengths():
    charset = CharSet("абв")
    items = [
        (preprocess(make_image("а", width=20), 32, 96), "а", "id-1"),
        (preprocess(make_image("абв", width=120), 32, 96), "абв", "id-2"),
    ]
    batch = make_collate(charset)(items)
    assert batch.images.shape[0] == 2
    assert len({batch.images.shape[3]}) == 1  # все картинки одной ширины
    assert batch.targets.tolist() == [1, 1, 2, 3]
    assert batch.target_lengths.tolist() == [1, 3]
    assert all(int(l) == output_length(batch.images.shape[3]) for l in batch.input_lengths)
    assert int(batch.input_lengths.min()) >= int(batch.target_lengths.max()) + 1
    assert np.isfinite(batch.images.numpy()).all()
