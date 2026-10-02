from pathlib import Path

from codeguardian.scanner import PythonScanner


def test_finds_python_files(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()

    (project / "one.py").write_text("print('one')\n", encoding="utf-8")
    (project / "two.py").write_text("print('two')\n", encoding="utf-8")
    (project / "notes.txt").write_text("not Python\n", encoding="utf-8")

    scanner = PythonScanner(project)

    files = scanner.find_python_files()

    assert len(files) == 2
    assert {path.name for path in files} == {"one.py", "two.py"}


def test_valid_python_has_no_diagnostics(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()

    source = project / "good.py"
    source.write_text(
        """
def hello(name):
    return f"Hello {name}"
""",
        encoding="utf-8",
    )

    scanner = PythonScanner(project)

    diagnostics = scanner.check_file(source)

    assert diagnostics == []


def test_invalid_python_is_detected(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()

    source = project / "broken.py"
    source.write_text(
        """
def hello(name)
    return name
""",
        encoding="utf-8",
    )

    scanner = PythonScanner(project)

    diagnostics = scanner.check_file(source)

    assert len(diagnostics) == 1
    assert diagnostics[0].severity == "ERROR"
    assert "expected ':'" in diagnostics[0].message


def test_scan_checks_entire_project(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()

    (project / "good.py").write_text(
        "value = 42\n",
        encoding="utf-8",
    )

    (project / "bad.py").write_text(
        "def broken(\n",
        encoding="utf-8",
    )

    scanner = PythonScanner(project)

    diagnostics = scanner.scan()

    assert len(diagnostics) == 1
    assert diagnostics[0].file.name == "bad.py"


def test_ignores_pycache_and_git(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()

    (project / "good.py").write_text(
        "value = 42\n",
        encoding="utf-8",
    )

    pycache = project / "__pycache__"
    pycache.mkdir()
    (pycache / "ignored.py").write_text(
        "this should not be scanned\n",
        encoding="utf-8",
    )

    git = project / ".git"
    git.mkdir()
    (git / "ignored.py").write_text(
        "this should not be scanned\n",
        encoding="utf-8",
    )

    scanner = PythonScanner(project)

    files = scanner.find_python_files()

    assert files == [project / "good.py"]


def test_syntax_diagnostic_preserves_parser_error_span(tmp_path):
    from codeguardian.scanner import PythonScanner

    path = tmp_path / "broken.py"
    source = 'value = {"a": f() "b": g()}\n'
    path.write_text(source, encoding="utf-8")

    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        expected_end_line = exc.end_lineno
        expected_end_column = exc.end_offset
    else:
        raise AssertionError("Expected source to raise SyntaxError")

    diagnostics = PythonScanner(tmp_path).check_file(path)

    assert len(diagnostics) == 1

    diagnostic = diagnostics[0]

    assert diagnostic.line == 1
    assert diagnostic.column == 15
    assert diagnostic.end_line == expected_end_line
    assert diagnostic.end_column == expected_end_column


def test_non_parser_diagnostic_span_defaults_to_none(tmp_path):
    from codeguardian.scanner import Diagnostic

    diagnostic = Diagnostic(
        file=tmp_path / "example.py",
        line=1,
        column=1,
        severity="WARNING",
        message="example",
    )

    assert diagnostic.end_line is None
    assert diagnostic.end_column is None
