from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PIL import Image

from constants.groups import (
    GROUPS,
    ROUTING_CLASSIFIER,
    ROUTING_EXPLICIT,
    ROUTING_OCR_CONFIDENCE,
)
from constants.training import CONFIDENCE_ROUND_DIGITS
from schemas import Candidate, GroupName, PredictionResult
from templates import errors, logs
from training.dataset import decode_base64_image

if TYPE_CHECKING:  # только для аннотаций
    from training.registry import ModelRegistry

log = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """Ни одна модель нужной группы ещё не обучена."""


class PredictionService:
    """Распознавание картинки и выбор модели под неё."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def predict_base64(self, image_b64: str, group: GroupName | None = None) -> PredictionResult:
        """То же, что predict, но на вход принимает base64 (бросает ImageDecodeError)."""
        return self.predict(decode_base64_image(image_b64), group)

    def predict(self, image: Image.Image, group: GroupName | None = None) -> PredictionResult:
        """Распознаёт картинку.

        Если группа не указана, её определяет классификатор; пока он не обучен —
        берётся ответ модели с большей уверенностью.
        """
        candidates = self._candidates(image, group)
        if not candidates:
            raise ModelNotReadyError(errors.NO_MODEL_READY)

        if group is not None:
            return self._result(candidates[0], candidates, ROUTING_EXPLICIT, None)

        if self.registry.classifier.is_ready:
            predicted, confidence = self.registry.classifier.predict(image)
            chosen = next((c for c in candidates if c.group == predicted), None)
            if chosen is not None:
                return self._result(
                    chosen,
                    candidates,
                    ROUTING_CLASSIFIER,
                    round(confidence, CONFIDENCE_ROUND_DIGITS),
                )
            log.debug(logs.CLASSIFIER_GROUP_NOT_READY, predicted)

        best = max(candidates, key=lambda c: c.confidence)
        return self._result(best, candidates, ROUTING_OCR_CONFIDENCE, None)

    def _candidates(self, image: Image.Image, group: GroupName | None) -> list[Candidate]:
        candidates: list[Candidate] = []
        for name in [group] if group else list(GROUPS):
            trainer = self.registry.trainer(name)
            if not trainer.is_ready:
                continue
            text, confidence = trainer.predict([image])[0]
            candidates.append(
                Candidate(group=name, text=text, confidence=round(confidence, CONFIDENCE_ROUND_DIGITS))
            )
        return candidates

    @staticmethod
    def _result(
        chosen: Candidate, candidates: list[Candidate], routed_by: str, group_confidence: float | None
    ) -> PredictionResult:
        return PredictionResult(
            group=chosen.group,
            text=chosen.text,
            confidence=chosen.confidence,
            routed_by=routed_by,
            group_confidence=group_confidence,
            candidates=candidates,
        )
