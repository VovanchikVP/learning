from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from services import PredictionService, StatusService
from training import ModelRegistry, TrainingService
from .container import AppContainer


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


ContainerDep = Annotated[AppContainer, Depends(get_container)]


def get_prediction_service(container: ContainerDep) -> PredictionService:
    return container.prediction


def get_status_service(container: ContainerDep) -> StatusService:
    return container.status


def get_training_service(container: ContainerDep) -> TrainingService:
    return container.training


def get_registry(container: ContainerDep) -> ModelRegistry:
    return container.registry


PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
StatusServiceDep = Annotated[StatusService, Depends(get_status_service)]
TrainingServiceDep = Annotated[TrainingService, Depends(get_training_service)]
RegistryDep = Annotated[ModelRegistry, Depends(get_registry)]
