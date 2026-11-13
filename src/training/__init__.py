"""Слой непрерывного обучения: сети, кэш датасета, тренеры и оркестратор."""

import os

# Часть CTC/LSTM-ядер ещё не реализована в бэкенде MPS — разрешаем молчаливый
# фолбэк на CPU для таких операций. Выставляется до первого импорта torch,
# который произойдёт при импорте любого модуля этого пакета.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .registry import ModelRegistry
from .service import TrainingService

__all__ = ["ModelRegistry", "TrainingService"]
