from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from datetime import datetime
from typing import TYPE_CHECKING, Any

from config import Settings, classify_answer, is_ignored_answer
from constants import storage
from constants.api import TRAIN_TASK_NAME
from constants.db import CURSOR_MODE_DIFF
from constants.groups import GROUPS
from constants.storage import ENCODING, JSON_INDENT, TMP_SUFFIX
from constants.training import MIN_DIGITS_ANSWER_LENGTH
from schemas import CursorState, GroupName, ImageRow, IngestStats, TrainKind, TrainResult
from templates import logs
from templates.api import CURSOR_MODE_KEYSET

from .dataset import ImageDecodeError, decode_base64_image
from .registry import ModelRegistry

if TYPE_CHECKING:  # только для аннотаций: services импортирует training
    from services.images import ImageRepository

log = logging.getLogger(__name__)


class TrainingService:
    """Непрерывное обучение: забирает новые строки из БД, копит их по группам
    и запускает (до)обучение соответствующей модели."""

    def __init__(
        self, settings: Settings, registry: ModelRegistry, repository: ImageRepository
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.repository = repository
        # keyset-курсор: (значение cursor_column, id последней обработанной строки)
        self.cursor: tuple[Any, str] | None = None
        self.pending: dict[str, list[str]] = {g: [] for g in GROUPS}
        self.skipped_ids: set[str] = set()
        self._train_locks = {g: asyncio.Lock() for g in GROUPS}
        self._tasks: dict[str, asyncio.Task] = {}
        self._stop = asyncio.Event()
        self._load_state()
        # всё, что уже не нужно тянуть из базы: лежит в кэше или не раскодировалось
        self._known: set[str] = set(self.skipped_ids) | registry.known_ids()

    # ------------------------------------------------------------- состояние

    @staticmethod
    def _encode_cursor(cursor: tuple[Any, str] | None) -> CursorState | None:
        if cursor is None:
            return None
        value, row_id = cursor
        if isinstance(value, datetime):
            return CursorState(value=value.isoformat(), is_datetime=True, id=row_id)
        return CursorState(value=value, is_datetime=False, id=row_id)

    @staticmethod
    def _decode_cursor(raw: dict | None) -> tuple[Any, str] | None:
        if not raw:
            return None
        state = CursorState.model_validate(raw)
        value = state.value
        if state.is_datetime and isinstance(value, str):
            value = datetime.fromisoformat(value)
        return (value, state.id)

    def _load_state(self) -> None:
        path = self.settings.state_path
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding=ENCODING))
        self.cursor = self._decode_cursor(data.get(storage.STATE_CURSOR))
        self.skipped_ids = {str(i) for i in data.get(storage.STATE_SKIPPED_IDS, [])}
        for group in GROUPS:
            self.pending[group] = [
                str(i) for i in data.get(storage.STATE_PENDING, {}).get(group, [])
            ]
        log.info(logs.STATE_RESTORED, self.cursor, self.pending_counts())

    def _save_state(self) -> None:
        cursor = self._encode_cursor(self.cursor)
        payload = {
            storage.STATE_CURSOR: cursor.model_dump(mode="json") if cursor else None,
            storage.STATE_PENDING: self.pending,
            storage.STATE_SKIPPED_IDS: sorted(self.skipped_ids),
            storage.STATE_UPDATED_AT: time.time(),
        }
        tmp = self.settings.state_path.with_suffix(TMP_SUFFIX)
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=JSON_INDENT), encoding=ENCODING
        )
        tmp.replace(self.settings.state_path)

    def pending_counts(self) -> dict[str, int]:
        return {group: len(ids) for group, ids in self.pending.items()}

    @property
    def cursor_mode(self) -> str:
        column = self.repository.cursor_column
        return CURSOR_MODE_KEYSET.format(column=column) if column else CURSOR_MODE_DIFF

    @property
    def cursor_state(self) -> CursorState | None:
        return self._encode_cursor(self.cursor)

    @property
    def known_rows(self) -> int:
        return len(self._known)

    # --------------------------------------------------------------- приём

    def ingest_row(self, row_id: str, image_b64: str, answer: str) -> GroupName | None:
        """Кладёт строку в кэш нужной группы. Возвращает группу или None."""
        row_id = str(row_id)
        self._known.add(row_id)
        answer = (answer or "").strip()
        if not answer:
            self.skipped_ids.add(row_id)
            return None
        if is_ignored_answer(answer):
            # помним id, чтобы не вычитывать строку из базы на каждом опросе
            self.skipped_ids.add(row_id)
            log.debug(logs.ROW_IGNORED_SHORT_NUMBER, row_id, answer, MIN_DIGITS_ANSWER_LENGTH)
            return None
        group = classify_answer(answer)
        try:
            image = decode_base64_image(image_b64)
        except ImageDecodeError as exc:
            self.skipped_ids.add(row_id)
            log.warning(logs.ROW_SKIPPED, row_id, exc)
            return None
        if not self.registry.store(group).add(row_id, image, answer):
            return None
        return group

    async def _accept(self, row: ImageRow, stats: IngestStats) -> None:
        group = await asyncio.to_thread(self.ingest_row, row.id, row.image, row.answer)
        if group:
            self.pending[group].append(row.id)
            stats.added[group] = stats.added.get(group, 0) + 1

    async def ingest_new(self) -> IngestStats:
        """Забирает из базы строки, которых ещё нет в локальном кэше."""
        if self.repository.cursor_column:
            return await self._ingest_keyset()
        return await self._ingest_by_diff()

    async def _ingest_keyset(self) -> IngestStats:
        """Постраничный обход по (cursor_column, id) — для больших таблиц."""
        stats = IngestStats(added=dict.fromkeys(GROUPS, 0))
        while not self._stop.is_set():
            rows = await self.repository.fetch_after(self.cursor, self.settings.fetch_batch)
            if not rows:
                break
            for row in rows:
                await self._accept(row, stats)
                self.cursor = (row.cursor, row.id)
            self._save_state()
            if len(rows) < self.settings.fetch_batch:
                break
        stats.skipped = len(self.skipped_ids)
        return stats

    async def _ingest_by_diff(self) -> IngestStats:
        """Сверка списка id из базы с локальным кэшем — когда cursor_column нет."""
        stats = IngestStats(added=dict.fromkeys(GROUPS, 0))
        all_ids = await self.repository.fetch_ids()
        unknown = [row_id for row_id in all_ids if row_id not in self._known]
        if unknown:
            log.info(logs.NEW_IDS_FOUND, len(unknown))
            batch = self.settings.fetch_batch
            for start in range(0, len(unknown), batch):
                if self._stop.is_set():
                    break
                rows = await self.repository.fetch_by_ids(unknown[start : start + batch])
                for row in rows:
                    await self._accept(row, stats)
                self._save_state()
        stats.skipped = len(self.skipped_ids)
        return stats

    # -------------------------------------------------------------- обучение

    def needs_training(self, group: GroupName) -> TrainKind | None:
        trainer = self.registry.trainer(group)
        if trainer.training:
            return None
        if not trainer.is_ready:
            if len(self.registry.store(group)) >= self.settings.min_samples_to_train:
                return TrainKind.FULL
            return None
        if len(self.pending[group]) >= self.settings.retrain_threshold:
            return TrainKind.INCREMENTAL
        return None

    async def train_group(self, group: GroupName, kind: TrainKind | str) -> TrainResult:
        kind = TrainKind(kind)
        trainer = self.registry.trainer(group)
        async with self._train_locks[group]:
            if kind is TrainKind.FULL:
                log.info(logs.TRAIN_FULL_START, group, len(self.registry.store(group)))
                result = await asyncio.to_thread(trainer.train_full)
                if result.accepted:
                    # всё накопленное уже вошло в обучение
                    self.pending[group] = []
            else:
                batch_ids = self.pending[group]
                self.pending[group] = []
                self._save_state()
                log.info(logs.TRAIN_INCREMENTAL_START, group, len(batch_ids))
                result = await asyncio.to_thread(trainer.train_incremental, batch_ids)
            log.info(
                logs.TRAIN_DONE,
                group,
                result.duration_sec,
                result.val_cer,
                result.val_accuracy,
                result.accepted,
                result.note,
            )
            self._save_state()
        if result.accepted:
            # данных стало больше — обновляем и роутер между моделями
            info = await asyncio.to_thread(self.registry.classifier.train)
            if info.trained:
                log.info(logs.CLASSIFIER_TRAINED, info.version, info.val_accuracy)
        return result

    def _schedule(self, group: GroupName, kind: TrainKind) -> None:
        task = self._tasks.get(group)
        if task is not None and not task.done():
            return
        self._tasks[group] = asyncio.create_task(
            self.train_group(group, kind), name=TRAIN_TASK_NAME.format(group=group, kind=kind)
        )

    # ------------------------------------------------------------ главный цикл

    async def bootstrap(self) -> IngestStats:
        await self.repository.connect()
        total = await self.repository.count()
        log.info(logs.TABLE_SUMMARY, total, {g: len(s) for g, s in self.registry.stores.items()})
        stats = await self.ingest_new()
        log.info(logs.BOOTSTRAP_DONE, stats.added)
        return stats

    async def run(self) -> None:
        await self.bootstrap()
        try:
            while not self._stop.is_set():
                for group in GROUPS:
                    kind = self.needs_training(group)
                    if kind:
                        self._schedule(group, kind)
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=self.settings.poll_interval)
                if self._stop.is_set():
                    break
                stats = await self.ingest_new()
                if stats.total:
                    log.info(logs.POLL_RESULT, stats.added, self.pending_counts())
        finally:
            for task in self._tasks.values():
                if not task.done():
                    await task
            await self.repository.close()
            self._save_state()

    def stop(self) -> None:
        self._stop.set()
