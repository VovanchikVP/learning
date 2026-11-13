from __future__ import annotations

import logging

from config import Settings
from constants.groups import GROUPS
from schemas import ClassifierStatus, GroupName, GroupStatus

from .classifier import GroupClassifier
from .store import SampleStore
from .trainer import GroupTrainer

log = logging.getLogger(__name__)


class ModelRegistry:
    """Живые модели и кэши датасетов в памяти процесса.

    Один объект на процесс: его читает слой API (инференс и статус) и пишет
    сервис обучения. Благодаря этому обновлённые после дообучения веса сразу
    используются в `/predict` — отдельной перезагрузки не требуется.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.workdir.mkdir(parents=True, exist_ok=True)
        self.stores: dict[str, SampleStore] = {
            group: SampleStore(settings.data_dir, group, settings.val_fraction) for group in GROUPS
        }
        self.trainers: dict[str, GroupTrainer] = {
            group: GroupTrainer(group, self.stores[group], settings) for group in GROUPS
        }
        self.classifier = GroupClassifier(settings, self.stores)

    @property
    def groups(self) -> tuple[GroupName, ...]:
        return GROUPS

    def trainer(self, group: GroupName) -> GroupTrainer:
        return self.trainers[group]

    def store(self, group: GroupName) -> SampleStore:
        return self.stores[group]

    def known_ids(self) -> set[str]:
        ids: set[str] = set()
        for store in self.stores.values():
            ids |= store.ids()
        return ids

    def group_status(self, group: GroupName, pending: int = 0) -> GroupStatus:
        return self.trainers[group].status(pending=pending)

    def classifier_status(self) -> ClassifierStatus:
        return self.classifier.status()
