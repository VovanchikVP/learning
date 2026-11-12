from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from constants.api import ROUTE_TRAIN, TAG_TRAINING
from schemas import GroupName, TrainKind, TrainResult
from templates import errors
from templates.api import SUMMARY_TRAIN

from ..deps import RegistryDep, TrainingServiceDep

router = APIRouter(tags=[TAG_TRAINING])


@router.post(ROUTE_TRAIN, response_model=TrainResult, summary=SUMMARY_TRAIN)
async def train(
    group: GroupName,
    service: TrainingServiceDep,
    registry: RegistryDep,
    kind: TrainKind = TrainKind.FULL,
) -> TrainResult:
    if registry.trainer(group).training:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=errors.TRAINING_IN_PROGRESS)
    return await service.train_group(group, kind)
