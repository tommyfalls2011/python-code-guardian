from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Diagnostic:
    file: Path
    line: int
    column: int
    severity: str
    message: str
    end_line: int | None = None
    end_column: int | None = None


class PythonScanner:
    """Find and perform basic validation of Python source files."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def find_python_files(self) -> list[Path]:
        """Return Python files below the project root."""
        return sorted(
            path
            for path in self.root.rglob("*.py")
            if path.is_file()
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
            and "guardian-venv" not in path.parts
        )

    def check_file(self, path: Path) -> list[Diagnostic]:
        """Check one Python file for syntax errors."""
        diagnostics: list[Diagnostic] = []

        try:
            source = path.read_text(encoding="utf-8")

            # Parse the source with Python's AST parser.
            ast.parse(source, filename=str(path))

            # Compile without executing the program.
            compile(source, str(path), "exec")

        except SyntaxError as exc:
            diagnostics.append(
                Diagnostic(
                    file=path,
                    line=exc.lineno or 1,
                    column=exc.offset or 1,
                    severity="ERROR",
                    message=exc.msg,
                    end_line=exc.end_lineno,
                    end_column=exc.end_offset,
                )
            )

        except UnicodeDecodeError as exc:
            diagnostics.append(
                Diagnostic(
                    file=path,
                    line=1,
                    column=1,
                    severity="ERROR",
                    message=f"Unable to decode file as UTF-8: {exc}",
                )
            )

        except OSError as exc:
            diagnostics.append(
                Diagnostic(
                    file=path,
                    line=1,
                    column=1,
                    severity="ERROR",
                    message=f"Unable to read file: {exc}",
                )
            )

        return diagnostics

    def scan(self) -> list[Diagnostic]:
        """Scan every Python file in the project."""
        diagnostics: list[Diagnostic] = []

        for path in self.find_python_files():
            diagnostics.extend(self.check_file(path))

        return diagnostics
