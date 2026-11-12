from datetime import UTC, datetime

import pytest

from schemas import ImageRow, TrainKind
from training import TrainingService
from tests.conftest import FakeImageRepository, image_base64, make_rows


def test_ingest_row_routes_by_answer(training):
    assert training.ingest_row("id-1", image_base64("корова"), "корова") == "words"
    assert training.ingest_row("id-2", image_base64("1234"), "1234") == "digits"
    assert len(training.registry.store("words")) == 1
    assert len(training.registry.store("digits")) == 1


def test_ingest_row_is_idempotent(training):
    assert training.ingest_row("id-1", image_base64("кот"), "кот") == "words"
    assert training.ingest_row("id-1", image_base64("кот"), "кот") is None


def test_broken_rows_are_remembered_and_not_retried(training):
    assert training.ingest_row("bad", "не base64!!", "кот") is None
    assert training.ingest_row("empty", image_base64("x"), "   ") is None
    assert training.skipped_ids == {"bad", "empty"}
    assert {"bad", "empty"} <= training._known


async def test_ingest_by_diff_fetches_only_unknown_rows(settings, registry):
    rows = make_rows(6)
    repo = FakeImageRepository(rows)
    service = TrainingService(settings, registry, repo)

    first = await service.ingest_new()
    assert first.total == 6
    assert (await service.ingest_new()).total == 0  # повторный проход ничего не тянет

    repo.rows.extend(make_rows(2, start=6))
    assert (await service.ingest_new()).total == 2


async def test_ingest_keyset_advances_cursor(settings, registry):
    rows = make_rows(5, with_cursor=True)
    repo = FakeImageRepository(rows, cursor_column="created_at")
    service = TrainingService(settings, registry, repo)

    assert (await service.ingest_new()).total == 5
    assert service.cursor is not None
    assert service.cursor[1] == rows[-1].id
    assert (await service.ingest_new()).total == 0

    repo.rows.extend(make_rows(3, with_cursor=True, start=5))
    assert (await service.ingest_new()).total == 3
    assert service.cursor_mode == "keyset:created_at"


async def test_state_survives_restart(settings, registry):
    repo = FakeImageRepository(make_rows(4, with_cursor=True), cursor_column="created_at")
    service = TrainingService(settings, registry, repo)
    await service.ingest_new()
    cursor_before = service.cursor
    pending_before = service.pending_counts()

    restored = TrainingService(settings, registry, repo)
    assert restored.cursor == cursor_before
    assert isinstance(restored.cursor[0], datetime)
    assert restored.pending_counts() == pending_before
    assert restored.known_rows == service.known_rows
    assert (await restored.ingest_new()).total == 0


async def test_cursor_roundtrip_without_datetime(settings, registry):
    rows = [ImageRow(id="id-1", image=image_base64("1"), answer="1", cursor=42)]
    repo = FakeImageRepository(rows, cursor_column="seq")
    service = TrainingService(settings, registry, repo)
    await service.ingest_new()
    assert TrainingService(settings, registry, repo).cursor == (42, "id-1")


def test_needs_training_waits_for_minimum(training, settings):
    assert training.needs_training("words") is None
    for i in range(settings.min_samples_to_train):
        training.ingest_row(f"id-{i}", image_base64("кот"), "кот")
    assert training.needs_training("words") is TrainKind.FULL


def test_needs_incremental_only_after_threshold(training, settings, monkeypatch):
    trainer = training.registry.trainer("digits")
    monkeypatch.setattr(type(trainer), "is_ready", property(lambda self: True))
    assert training.needs_training("digits") is None
    training.pending["digits"] = [f"id-{i}" for i in range(settings.retrain_threshold)]
    assert training.needs_training("digits") is TrainKind.INCREMENTAL


def test_cursor_mode_without_cursor_column(training):
    assert training.cursor_mode == "id-diff"
    assert training.cursor_state is None


async def test_bootstrap_connects_repository(training, repository):
    await training.bootstrap()
    assert repository.connected


@pytest.mark.parametrize("kind", [TrainKind.FULL, TrainKind.INCREMENTAL])
async def test_training_without_data_is_rejected(training, kind):
    result = await training.train_group("words", kind)
    assert not result.accepted
    assert "мало данных" in result.note
