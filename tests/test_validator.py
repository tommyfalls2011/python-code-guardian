from pathlib import Path

from codeguardian.validator import RepairValidator


def test_validator_accepts_good_file(tmp_path: Path):
    path = tmp_path / "good.py"
    path.write_text(
        "def hello(name):\n"
        "    return f'Hello {name}'\n",
        encoding="utf-8",
    )

    result = RepairValidator().validate(path)

    assert result.passed is True
    assert result.diagnostics == []


def test_validator_rejects_syntax_error(tmp_path: Path):
    path = tmp_path / "broken.py"
    path.write_text(
        "def hello(name)\n"
        "    return name\n",
        encoding="utf-8",
    )

    result = RepairValidator().validate(path)

    assert result.passed is False
    assert any(
        diagnostic.message == "expected ':'"
        for diagnostic in result.diagnostics
    )


def test_validator_rejects_guardian_analysis_error(tmp_path: Path):
    path = tmp_path / "broken.py"
    path.write_text(
        "def broken():\n"
        "    return definitely_missing\n",
        encoding="utf-8",
    )

    result = RepairValidator().validate(path)

    assert result.passed is False
    assert any(
        "Undefined name: 'definitely_missing'" in diagnostic.message
        for diagnostic in result.diagnostics
    )


def test_validate_source_does_not_write_file():
    source = "def hello():\n    return 42\n"

    result = RepairValidator().validate_source(source)

    assert result.passed is True
    assert result.diagnostics == []


def test_validate_source_rejects_invalid_python():
    source = "def hello(\n"

    result = RepairValidator().validate_source(source)

    assert result.passed is False
    assert result.diagnostics


def test_validation_warnings_do_not_reject_valid_source(tmp_path):
    path = tmp_path / "unused_import.py"
    path.write_text(
        "import os\n",
        encoding="utf-8",
    )

    result = RepairValidator().validate(path)

    assert result.passed is True
    assert result.diagnostics
    assert all(
        diagnostic.severity.upper() != "ERROR"
        for diagnostic in result.diagnostics
    )


def test_validation_still_rejects_analysis_errors(tmp_path):
    path = tmp_path / "undefined.py"
    path.write_text(
        "def f():\n"
        "    return definitely_missing\n",
        encoding="utf-8",
    )

    result = RepairValidator().validate(path)

    assert result.passed is False
    assert any(
        diagnostic.severity.upper() == "ERROR"
        for diagnostic in result.diagnostics
    )
