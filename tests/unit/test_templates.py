"""Слой шаблонов: тексты должны быть валидными и жить только в нём."""

import ast
import pathlib
import re

import pytest

from templates import api, cli, errors, logs, notes, sql

FORMAT_MODULES = [errors, notes, api, cli, sql]
CYRILLIC = re.compile(r"[а-яА-ЯёЁ]")
SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
#: в этих слоях текст и значения жить обязаны
TEXT_LAYERS = {"templates", "constants"}


class AnyValue(dict):
    """Подставляет значение, которое годится под любой формат-спецификатор."""

    def __missing__(self, key: str) -> float:
        return 0.0


def templates_of(module) -> list[tuple[str, str]]:
    return [
        (name, value)
        for name, value in vars(module).items()
        if not name.startswith("_") and isinstance(value, str)
    ]


@pytest.mark.parametrize("module", FORMAT_MODULES, ids=lambda m: m.__name__)
def test_format_templates_are_renderable(module):
    found = templates_of(module)
    assert found, f"в {module.__name__} нет шаблонов"
    for name, template in found:
        # ловит незакрытые скобки и неверные спецификаторы формата
        rendered = template.format_map(AnyValue())
        assert rendered, name


def test_log_templates_use_percent_style():
    for name, template in templates_of(logs):
        assert "{" not in template, f"{name}: logging форматирует через %"


def test_templates_are_not_empty():
    for module in [*FORMAT_MODULES, logs]:
        for name, template in templates_of(module):
            assert template.strip(), name


def test_known_templates_render_expected_text():
    assert errors.UNKNOWN_GROUP.format(group="letters") == "неизвестная группа letters"
    assert notes.NOT_ENOUGH_DATA.format(count=5, minimum=64) == "мало данных: 5 < 64"
    assert api.CURSOR_MODE_KEYSET.format(column="created_at") == "keyset:created_at"


def _python_files():
    for path in sorted(SRC.rglob("*.py")):
        if not TEXT_LAYERS & set(path.parts) and "__pycache__" not in path.parts:
            yield path


def test_no_russian_text_outside_template_layer():
    """Весь текст, который видит пользователь, должен лежать в templates/."""
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and CYRILLIC.search(node.value)
            ):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}: {node.value[:50]!r}")
    assert not offenders, "текст вне слоя шаблонов:\n" + "\n".join(offenders)
