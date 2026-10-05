from pathlib import Path

from codeguardian.scope import analyze_scope, find_unused_definitions


def write_python(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_module_definitions(tmp_path):
    path = write_python(
        tmp_path,
        """
import os

value = 10

def hello(name):
    result = name
    return result
""",
    )

    scope = analyze_scope(path)

    assert "os" in scope.defined
    assert "value" in scope.defined
    assert "hello" in scope.defined

    function = scope.children[0]

    assert function.name == "hello"
    assert function.kind == "function"
    assert "name" in function.defined
    assert "result" in function.defined


def test_nested_function_scope(tmp_path):
    path = write_python(
        tmp_path,
        """
def outer(value):
    def inner(item):
        return item + value

    return inner
""",
    )

    scope = analyze_scope(path)

    outer = scope.children[0]
    inner = outer.children[0]

    assert outer.name == "outer"
    assert "value" in outer.defined
    assert "inner" in outer.defined

    assert inner.name == "inner"
    assert "item" in inner.defined
    assert "value" in inner.used


def test_global_and_nonlocal(tmp_path):
    path = write_python(
        tmp_path,
        """
value = 0

def outer():
    count = 1

    def inner():
        global value
        nonlocal count
        value = 2
        count = 3
""",
    )

    scope = analyze_scope(path)

    outer = scope.children[0]
    inner = outer.children[0]

    assert "value" in inner.globals
    assert "count" in inner.nonlocals


def test_lambda_scope(tmp_path):
    path = write_python(
        tmp_path,
        """
factor = 2
double = lambda value: value * factor
""",
    )

    scope = analyze_scope(path)

    assert "factor" in scope.defined
    assert "double" in scope.defined

    lambda_scope = scope.children[0]

    assert lambda_scope.kind == "lambda"
    assert "value" in lambda_scope.defined
    assert "factor" in lambda_scope.used


def test_class_scope(tmp_path):
    path = write_python(
        tmp_path,
        """
class Example:
    value = 10

    def method(self, item):
        return item
""",
    )

    scope = analyze_scope(path)

    class_scope = scope.children[0]

    assert class_scope.name == "Example"
    assert class_scope.kind == "class"
    assert "value" in class_scope.defined
    assert "method" in class_scope.defined

    method = class_scope.children[0]

    assert method.name == "method"
    assert "self" in method.defined
    assert "item" in method.defined


def test_for_target_and_with_target(tmp_path):
    path = write_python(
        tmp_path,
        """
def process(items):
    for item in items:
        pass

    with open("file.txt") as handle:
        data = handle.read()

    return item, data
""",
    )

    scope = analyze_scope(path)

    function = scope.children[0]

    assert "item" in function.defined
    assert "handle" in function.defined
    assert "data" in function.defined


def test_exception_target(tmp_path):
    path = write_python(
        tmp_path,
        """
def process():
    try:
        work()
    except ValueError as error:
        print(error)
""",
    )

    scope = analyze_scope(path)

    function = scope.children[0]

    assert "error" in function.defined
    assert "work" in function.used
    assert "print" in function.used


def test_comprehension_scope(tmp_path):
    path = write_python(
        tmp_path,
        """
def process(items):
    return [item * 2 for item in items if item > 0]
""",
    )

    scope = analyze_scope(path)

    function = scope.children[0]

    assert "items" in function.defined
    assert "item" not in function.defined


def test_starred_arguments(tmp_path):
    path = write_python(
        tmp_path,
        """
def process(*args, **kwargs):
    return args, kwargs
""",
    )

    scope = analyze_scope(path)

    function = scope.children[0]

    assert "args" in function.defined
    assert "kwargs" in function.defined


def test_match_pattern(tmp_path):
    path = write_python(
        tmp_path,
        """
def process(value):
    match value:
        case {"name": name}:
            return name
        case _:
            return None
""",
    )

    scope = analyze_scope(path)

    function = scope.children[0]

    assert "value" in function.defined
    assert "name" in function.defined


def test_scope_tracks_used_names(tmp_path):
    path = write_python(
        tmp_path,
        """
value = 10

def process(item):
    result = item + value
    return result
""",
    )

    scope = analyze_scope(path)

    function = scope.children[0]

    assert "item" in function.defined
    assert "result" in function.defined
    assert "item" in function.used
    assert "value" in function.used


def test_finds_undefined_name(tmp_path):
    path = write_python(
        tmp_path,
        """
def process(value):
    return value + missing_value
""",
    )

    scope = analyze_scope(path)

    undefined = __import__(
        "codeguardian.scope",
        fromlist=["find_undefined_names"],
    ).find_undefined_names(scope)

    assert any(name == "missing_value" for name, _, _ in undefined)


def test_does_not_flag_builtin(tmp_path):
    path = write_python(
        tmp_path,
        """
def process(value):
    return len(value)
""",
    )

    scope = analyze_scope(path)

    assert __import__(
        "codeguardian.scope",
        fromlist=["find_undefined_names"],
    ).find_undefined_names(scope) == []


def test_resolves_enclosing_scope(tmp_path):
    path = write_python(
        tmp_path,
        """
def outer(value):
    def inner():
        return value
    return inner
""",
    )

    scope = analyze_scope(path)

    assert __import__(
        "codeguardian.scope",
        fromlist=["find_undefined_names"],
    ).find_undefined_names(scope) == []


def test_detects_unused_import(tmp_path):
    path = write_python(
        tmp_path,
        """
import os
import sys

def process():
    return 42
""",
    )

    scope = analyze_scope(path)

    assert "os" in scope.defined
    assert "sys" in scope.defined
    assert "os" not in scope.used
    assert "sys" not in scope.used


def test_tracks_assignment_locations(tmp_path):
    path = write_python(
        tmp_path,
        """
def process():
    unused_value = 42
    return 1
""",
    )

    scope = analyze_scope(path)
    function = scope.children[0]

    assert "unused_value" in function.defined


def test_unused_definitions_ignore_module_level_names(tmp_path):
    source = """\
def public_function():
    return 1


class PublicClass:
    pass
"""
    path = tmp_path / "sample.py"
    path.write_text(source)

    scope = __import__(
        "codeguardian.scope",
        fromlist=["analyze_scope"],
    ).analyze_scope(path)

    find_unused_definitions = __import__(
        "codeguardian.scope",
        fromlist=["find_unused_definitions"],
    ).find_unused_definitions

    assert find_unused_definitions(scope) == []


def test_unused_definitions_find_local_variable(tmp_path):
    source = """\
def example():
    used_value = 10
    unused_value = 20
    return used_value
"""
    path = tmp_path / "sample.py"
    path.write_text(source)

    scope = __import__(
        "codeguardian.scope",
        fromlist=["analyze_scope"],
    ).analyze_scope(path)

    find_unused_definitions = __import__(
        "codeguardian.scope",
        fromlist=["find_unused_definitions"],
    ).find_unused_definitions

    unused = find_unused_definitions(scope)

    assert any(name == "unused_value" for name, _, _ in unused)
    assert not any(name == "used_value" for name, _, _ in unused)


def test_parameter_annotation_counts_as_import_use(tmp_path):
    path = tmp_path / "parameter_annotation.py"
    path.write_text(
        "from pathlib import Path\n"
        "\n"
        "def load(path: Path):\n"
        "    return str(path)\n",
        encoding="utf-8",
    )
    scope = analyze_scope(path)
    assert "Path" in scope.defined
    assert "Path" in scope.used


def test_future_annotations_not_treated_as_unused_definition(tmp_path):
    path = tmp_path / "future_annotations.py"
    path.write_text(
        "from __future__ import annotations\n"
        "\n"
        "def identity(value):\n"
        "    return value\n",
        encoding="utf-8",
    )
    scope = analyze_scope(path)
    assert "annotations" not in scope.defined


def test_local_used_only_by_comprehension_is_not_unused(tmp_path):
    path = tmp_path / "comprehension_use.py"
    path.write_text(
        "def collect():\n"
        "    tokens = [1, 2, 3]\n"
        "    return [token * 2 for token in tokens]\n",
        encoding="utf-8",
    )
    scope = analyze_scope(path)
    assert "tokens" not in {
        name for name, _, _ in find_unused_definitions(scope)
    }


def test_local_used_only_by_nested_function_is_not_unused(tmp_path):
    path = tmp_path / "nested_use.py"
    path.write_text(
        "def outer():\n"
        "    value = 42\n"
        "    def inner():\n"
        "        return value\n"
        "    return inner()\n",
        encoding="utf-8",
    )
    scope = analyze_scope(path)
    assert "value" not in {
        name for name, _, _ in find_unused_definitions(scope)
    }
