from datetime import UTC, datetime

from sqlalchemy.dialects import postgresql

from config import Settings
from models import build_labeled_image_model
from services import ImageRepository


def compiled(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


def test_model_uses_configured_names():
    model = build_labeled_image_model(
        Settings(table="ocr_rows", id_column="row_id", image_column="png", answer_column="label")
    )
    assert model.__tablename__ == "ocr_rows"
    assert {c.name for c in model.__table__.columns} == {"row_id", "png", "label"}
    assert not hasattr(model, "cursor")


def test_model_adds_cursor_column_when_configured():
    model = build_labeled_image_model(Settings(cursor_column="created_at"))
    assert "created_at" in {c.name for c in model.__table__.columns}


def test_keyset_statement_is_indexable_and_ordered():
    repo = ImageRepository(Settings(cursor_column="created_at"))
    sql = compiled(repo.keyset_statement((datetime(2026, 1, 1, tzinfo=UTC), "uuid-1"), 100))
    assert "captcha.created_at >= " in sql  # отдельное условие, чтобы работал индекс
    assert "captcha.created_at > " in sql
    assert "CAST(captcha.id AS TEXT) > " in sql
    assert "ORDER BY captcha.created_at ASC, CAST(captcha.id AS TEXT) ASC" in sql
    assert "LIMIT" in sql


def test_first_page_has_no_cursor_predicate():
    repo = ImageRepository(Settings(cursor_column="created_at"))
    sql = compiled(repo.keyset_statement(None, 10))
    assert "created_at >=" not in sql
    assert "answer IS NOT NULL" in sql


def test_ids_are_compared_as_text():
    repo = ImageRepository(Settings())
    sql = compiled(repo._base_select().where(repo._id_text.in_(["a", "b"])))
    assert "CAST(captcha.id AS TEXT) IN " in sql
