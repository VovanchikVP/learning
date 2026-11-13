"""Полный цикл обучения на синтетике: обучение -> чекпоинт -> дообучение -> инференс."""

from __future__ import annotations

import uuid

import pytest

from demo import generate
from schemas import TrainKind
from services import PredictionService
from training import ModelRegistry

pytestmark = pytest.mark.slow


def feed(training, count: int, seed: int) -> None:
    for group in ("words", "digits"):
        for image_b64, answer in generate(group, count, seed=seed):
            row_id = str(uuid.uuid4())
            got = training.ingest_row(row_id, image_b64, answer)
            if got:
                training.pending[got].append(row_id)


async def test_full_cycle(settings, registry, training):
    # по 24 на группу: id случайные, поэтому в train после отсечения валидации
    # всё равно останется >= 32 примеров, которые нужны классификатору
    feed(training, count=24, seed=1)
    assert len(registry.store("words")) == 24

    result = await training.train_group("words", TrainKind.FULL)
    assert result.accepted and result.kind is TrainKind.FULL
    assert result.val_cer is not None
    trainer = registry.trainer("words")
    assert trainer.is_ready and trainer.state.version == 1
    assert trainer.ckpt_path.exists()
    assert training.pending["words"] == []  # всё вошло в обучение

    # классификатор групп обучается сразу после OCR-модели
    await training.train_group("digits", TrainKind.FULL)
    assert registry.classifier.is_ready

    # инференс идёт на свежих весах
    image_b64, _ = generate("digits", 1, seed=5)[0]
    prediction = PredictionService(registry).predict_base64(image_b64)
    assert prediction.group in {"words", "digits"}
    assert 0.0 <= prediction.confidence <= 1.0
    assert {c.group for c in prediction.candidates} == {"words", "digits"}

    # дообучение на новой порции
    version_before = trainer.state.version
    feed(training, count=6, seed=2)
    incremental = await training.train_group("words", TrainKind.INCREMENTAL)
    assert incremental.kind is TrainKind.INCREMENTAL
    assert incremental.new_samples > 0
    assert incremental.baseline_cer is not None
    assert trainer.is_ready
    if incremental.accepted:
        assert trainer.state.version == version_before + 1
    else:
        assert "откат" in incremental.note

    # состояние переживает перезапуск процесса
    restored = ModelRegistry(settings)
    restored_trainer = restored.trainer("words")
    assert restored_trainer.is_ready
    assert restored_trainer.state.version == trainer.state.version
    assert restored_trainer.charset.chars == trainer.charset.chars
    assert restored.classifier.is_ready


async def test_alphabet_grows_without_retraining_from_scratch(settings, registry, training):
    feed(training, count=10, seed=3)
    await training.train_group("words", TrainKind.FULL)
    trainer = registry.trainer("words")
    before = set(trainer.charset.chars)

    training.ingest_row(str(uuid.uuid4()), generate("words", 1, seed=9)[0][0], "ёжикъ")
    added = trainer._ensure_model(["ёжикъ"])
    assert set(added) <= set("ёжикъ") and added
    assert before < set(trainer.charset.chars)
    assert trainer.model.head.out_features == trainer.charset.size
