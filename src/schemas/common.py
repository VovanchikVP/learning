from __future__ import annotations

from typing import Literal

from constants.groups import (
    DIGITS,
    ROUTING_CLASSIFIER,
    ROUTING_EXPLICIT,
    ROUTING_OCR_CONFIDENCE,
    WORDS,
)

#: значения берутся из слоя констант, чтобы список групп был в одном месте
GroupName = Literal[WORDS, DIGITS]  # type: ignore[valid-type]
RoutingMode = Literal[  # type: ignore[valid-type]
    ROUTING_EXPLICIT, ROUTING_CLASSIFIER, ROUTING_OCR_CONFIDENCE
]
