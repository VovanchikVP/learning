"""Шаблоны SQL, которые нельзя собрать через ORM (DDL демо-таблицы)."""

from typing import Final

CREATE_DEMO_TABLE: Final = (
    'CREATE TABLE IF NOT EXISTS "{table}" ('
    '"{id_column}" uuid PRIMARY KEY DEFAULT gen_random_uuid(), '
    '"{image_column}" text NOT NULL, '
    '"{answer_column}" text NOT NULL, '
    "created_at timestamptz NOT NULL DEFAULT now())"
)
CREATE_CURSOR_INDEX: Final = (
    'CREATE INDEX IF NOT EXISTS "{table}_cursor_idx" '
    'ON "{table}" ("{cursor_column}", "{id_column}")'
)
INSERT_DEMO_ROW: Final = (
    'INSERT INTO "{table}" ("{image_column}", "{answer_column}") VALUES (:image, :answer)'
)
