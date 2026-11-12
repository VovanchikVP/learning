"""Константы работы с базой данных."""

import re
from typing import Final

#: имена таблицы и колонок подставляются в SQL, поэтому допускаются только
#: простые идентификаторы
IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

ASYNC_DRIVER_DSN_PREFIX: Final = "postgresql+asyncpg://"
ASYNC_DRIVER_MARKER: Final = "postgresql+"
SYNC_DSN_PREFIXES: Final[tuple[str, ...]] = ("postgresql://", "postgres://")

POOL_SIZE: Final = 2
POOL_MAX_OVERFLOW: Final = 2

#: режим поиска новых строк, когда колонки-курсора нет
CURSOR_MODE_DIFF: Final = "id-diff"

# имена колонок в результатах запросов
COLUMN_ALIAS_ID: Final = "id"
COLUMN_ALIAS_CURSOR: Final = "cursor"
