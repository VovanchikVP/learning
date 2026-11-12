"""Слой API: только эндпоинты."""

from fastapi import APIRouter

from . import health, predict, status, training

router = APIRouter()
router.include_router(health.router)
router.include_router(status.router)
router.include_router(predict.router)
router.include_router(training.router)

__all__ = ["router"]
