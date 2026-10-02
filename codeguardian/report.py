from __future__ import annotations

from dataclasses import dataclass

from .scanner import Diagnostic


@dataclass(frozen=True)
class Report:
    """Summarize diagnostics produced by Code Guardian."""

    diagnostics: list[Diagnostic]

    @property
    def errors(self) -> list[Diagnostic]:
        return [
            diagnostic
            for diagnostic in self.diagnostics
            if diagnostic.severity == "ERROR"
        ]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [
            diagnostic
            for diagnostic in self.diagnostics
            if diagnostic.severity == "WARNING"
        ]

    @property
    def infos(self) -> list[Diagnostic]:
        return [
            diagnostic
            for diagnostic in self.diagnostics
            if diagnostic.severity == "INFO"
        ]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def info_count(self) -> int:
        return len(self.infos)

    @property
    def total_count(self) -> int:
        return len(self.diagnostics)

    @property
    def passed(self) -> bool:
        """Return True when no ERROR diagnostics exist."""
        return self.error_count == 0
