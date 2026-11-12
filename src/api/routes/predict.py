from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, status

from constants.api import ROUTE_PREDICT, TAG_INFERENCE
from schemas import PredictionResult, PredictRequest
from services import ModelNotReadyError
from templates.api import SUMMARY_PREDICT
from training.dataset import ImageDecodeError

from ..deps import PredictionServiceDep

router = APIRouter(tags=[TAG_INFERENCE])


@router.post(ROUTE_PREDICT, response_model=PredictionResult, summary=SUMMARY_PREDICT)
async def predict(request: PredictRequest, service: PredictionServiceDep) -> PredictionResult:
    try:
        return await asyncio.to_thread(service.predict_base64, request.image, request.group)
    except ImageDecodeError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
