from __future__ import annotations

from typing import Any, ClassVar

from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import NullType

from config import Settings


def build_labeled_image_model(settings: Settings) -> type[Any]:
    """Собирает ORM-модель таблицы с размеченными картинками.

    Имена таблицы и колонок задаются настройками, поэтому модель строится
    фабрикой: у каждой сборки свой registry и своя metadata, так что модель
    можно собрать под разные настройки (в том числе в тестах).

    Типы `id` и `cursor` намеренно не фиксируются (`NullType`): `id` — uuid, но
    в запросах он всегда приводится к тексту, а колонка курсора может быть и
    `timestamptz`, и чем-то ещё. Тип параметра выводит сам PostgreSQL.
    """

    class _Base(DeclarativeBase):
        # Mapped[Any] -> тип определяет сервер, см. docstring
        type_annotation_map: ClassVar[dict[Any, Any]] = {Any: NullType}

    class LabeledImage(_Base):
        __tablename__ = settings.table

        id: Mapped[Any] = mapped_column(settings.id_column, NullType, primary_key=True)
        image: Mapped[str] = mapped_column(settings.image_column, String)
        answer: Mapped[str] = mapped_column(settings.answer_column, String)

        if settings.cursor_column:
            cursor: Mapped[Any] = mapped_column(settings.cursor_column, NullType)

        def __repr__(self) -> str:  # pragma: no cover - отладочный вывод
            return f"<LabeledImage id={self.id!r} answer={self.answer!r}>"

    return LabeledImage
