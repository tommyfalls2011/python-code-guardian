from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .analyzer import analyze_file
from .scanner import Diagnostic, PythonScanner


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a repaired Python file."""

    passed: bool
    diagnostics: list[Diagnostic]


class RepairValidator:
    """Validate repaired Python source before it is committed."""

    def validate(self, path: Path) -> ValidationResult:
        path = path.resolve()

        scanner = PythonScanner(path.parent)

        syntax_diagnostics = scanner.check_file(path)

        if syntax_diagnostics:
            return ValidationResult(
                passed=False,
                diagnostics=syntax_diagnostics,
            )

        analysis_diagnostics = analyze_file(path).diagnostics

        blocking_diagnostics = [
            diagnostic
            for diagnostic in analysis_diagnostics
            if diagnostic.severity.upper() == "ERROR"
        ]

        return ValidationResult(
            passed=not blocking_diagnostics,
            diagnostics=analysis_diagnostics,
        )

    def validate_source(
        self,
        source: str,
        filename: str = "<candidate>",
    ) -> ValidationResult:
        """Validate source without writing it to disk."""

        diagnostics: list[Diagnostic] = []

        try:
            ast.parse(source, filename=filename)
            compile(source, filename, "exec")
        except SyntaxError as exc:
            diagnostics.append(
                Diagnostic(
                    file=Path(filename),
                    line=exc.lineno or 1,
                    column=exc.offset or 1,
                    severity="ERROR",
                    message=exc.msg,
                )
            )

        return ValidationResult(
            passed=not diagnostics,
            diagnostics=diagnostics,
        )
