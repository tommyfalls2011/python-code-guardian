from pathlib import Path

import pytest

from codeguardian.parser_guard import verify_current_diagnostic
from codeguardian.scanner import Diagnostic


def test_verify_current_diagnostic_accepts_matching_error(tmp_path):
    path = tmp_path / "example.py"
    source = "if True\n    pass\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=8,
        severity="ERROR",
        message="expected ':'",
    )

    verify_current_diagnostic(
        source,
        diagnostic,
    )


def test_verify_current_diagnostic_rejects_stale_location(tmp_path):
    path = tmp_path / "example.py"
    source = "if True\n    pass\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=999,
        severity="ERROR",
        message="expected ':'",
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        verify_current_diagnostic(
            source,
            diagnostic,
        )


def test_verify_current_diagnostic_rejects_stale_span(tmp_path):
    path = tmp_path / "example.py"
    source = "if True\n    pass\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=8,
        severity="ERROR",
        message="expected ':'",
        end_line=1,
        end_column=99,
    )

    with pytest.raises(
        ValueError,
        match="span does not match the current source",
    ):
        verify_current_diagnostic(
            source,
            diagnostic,
        )


def test_verify_current_diagnostic_can_ignore_span(tmp_path):
    path = tmp_path / "example.py"
    source = "if True\n    pass\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=8,
        severity="ERROR",
        message="expected ':'",
        end_line=1,
        end_column=99,
    )

    verify_current_diagnostic(
        source,
        diagnostic,
        check_span=False,
    )
