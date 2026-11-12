"""Слой API."""

from .app import create_app
from .container import AppContainer

__all__ = ["AppContainer", "create_app"]
