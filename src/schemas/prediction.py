from __future__ import annotations

from pydantic import BaseModel, Field

from templates.api import FIELD_GROUP_CONFIDENCE, FIELD_IMAGE_BASE64, FIELD_PREDICT_GROUP

from .common import GroupName, RoutingMode


class PredictRequest(BaseModel):
    image: str = Field(description=FIELD_IMAGE_BASE64)
    group: GroupName | None = Field(default=None, description=FIELD_PREDICT_GROUP)


class Candidate(BaseModel):
    group: GroupName
    text: str
    confidence: float


class PredictionResult(BaseModel):
    group: GroupName
    text: str
    confidence: float
    routed_by: RoutingMode
    group_confidence: float | None = Field(default=None, description=FIELD_GROUP_CONFIDENCE)
    candidates: list[Candidate] = Field(default_factory=list)
