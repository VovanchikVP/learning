from __future__ import annotations

from typing import TYPE_CHECKING

from constants.groups import GROUPS
from schemas import ServiceStatus, WatcherStatus

if TYPE_CHECKING:  # только для аннотаций: training импортирует services
    from training.registry import ModelRegistry
    from training.service import TrainingService


class StatusService:
    """Сводное состояние сервиса для API."""

    def __init__(self, registry: ModelRegistry, training: TrainingService) -> None:
        self.registry = registry
        self.training = training

    def status(self) -> ServiceStatus:
        pending = self.training.pending_counts()
        return ServiceStatus(
            cursor_mode=self.training.cursor_mode,
            watcher=WatcherStatus(
                running=self.training.watching,
                table_rows=self.training.table_rows,
                labeled_rows=self.training.labeled_rows,
                last_poll_at=self.training.last_poll_at,
                last_error=self.training.last_error,
            ),
            cursor=self.training.cursor_state,
            known_rows=self.training.known_rows,
            skipped_rows=len(self.training.skipped_ids),
            retrain_threshold=self.registry.settings.retrain_threshold,
            classifier=self.registry.classifier_status(),
            groups={
                group: self.registry.group_status(group, pending.get(group, 0)) for group in GROUPS
            },
        )
