import pytest
from pydantic import ValidationError

from config import Settings, classify_answer
from constants.groups import DIGITS, WORDS


@pytest.mark.parametrize(
    ("answer", "group"),
    [
        ("12345", DIGITS),
        ("  42  ", DIGITS),
        ("корова", WORDS),
        ("дом2", WORDS),
        ("", WORDS),
        ("12 34", WORDS),
    ],
)
def test_classify_answer(answer, group):
    assert classify_answer(answer) == group


def test_identifiers_are_validated():
    with pytest.raises(ValidationError):
        Settings(table="images; drop table users")


def test_empty_cursor_column_means_disabled():
    assert Settings(cursor_column="").cursor_column is None


def test_sqlalchemy_dsn_gets_asyncpg_driver():
    assert Settings(pg_dsn="postgresql://u:p@h/db").sqlalchemy_dsn == "postgresql+asyncpg://u:p@h/db"
    assert Settings(pg_dsn="postgres://u:p@h/db").sqlalchemy_dsn == "postgresql+asyncpg://u:p@h/db"
    already = "postgresql+asyncpg://u:p@h/db"
    assert Settings(pg_dsn=already).sqlalchemy_dsn == already
