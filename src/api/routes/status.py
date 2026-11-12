from __future__ import annotations

from fastapi import APIRouter

from constants.api import ROUTE_STATUS, TAG_SERVICE
from schemas import ServiceStatus
from templates.api import SUMMARY_STATUS

from ..deps import StatusServiceDep

router = APIRouter(tags=[TAG_SERVICE])


@router.get(ROUTE_STATUS, response_model=ServiceStatus, summary=SUMMARY_STATUS)
async def status(service: StatusServiceDep) -> ServiceStatus:
    return service.status()
