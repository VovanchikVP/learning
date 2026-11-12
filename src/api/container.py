from __future__ import annotations

from dataclasses import dataclass

from config import Settings
from services import ImageRepository, PredictionService, StatusService
from training import ModelRegistry, TrainingService


@dataclass(slots=True)
class AppContainer:
    """Собранные зависимости приложения.

    Все слои связываются здесь, поэтому эндпоинты получают готовые сервисы и
    ничего не знают про их устройство.
    """

    settings: Settings
    registry: ModelRegistry
    repository: ImageRepository
    training: TrainingService
    prediction: PredictionService
    status: StatusService

    @classmethod
    def build(cls, settings: Settings) -> AppContainer:
        registry = ModelRegistry(settings)
        repository = ImageRepository(settings)
        training = TrainingService(settings, registry, repository)
        return cls(
            settings=settings,
            registry=registry,
            repository=repository,
            training=training,
            prediction=PredictionService(registry),
            status=StatusService(registry, training),
        )
