"""Слой моделей данных: описание таблиц базы данных."""

from .base import Base
from .labeled_image import build_labeled_image_model

__all__ = ["Base", "build_labeled_image_model"]
