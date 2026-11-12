from __future__ import annotations

from fastapi import APIRouter

from constants.api import HEALTH_STATUS_FIELD, HEALTH_STATUS_OK, ROUTE_HEALTH, TAG_SERVICE
from templates.api import SUMMARY_HEALTH

router = APIRouter(tags=[TAG_SERVICE])


@router.get(ROUTE_HEALTH, summary=SUMMARY_HEALTH)
async def health() -> dict[str, str]:
    return {HEALTH_STATUS_FIELD: HEALTH_STATUS_OK}
