from __future__ import annotations

from ..candidate import build_candidate
from ..parser_guard import verify_current_diagnostic
from ..plan import RepairPlan
from ..repair import Repair
from ..scanner import Diagnostic

def missing_colon(diagnostic: Diagnostic) -> RepairPlan:
    source = diagnostic.file.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)

    line_index = diagnostic.line - 1

    if line_index < 0 or line_index >= len(lines):
        raise ValueError("Diagnostic line is outside the source")

    line = lines[line_index].rstrip("\r\n")
    stripped = line.strip()

    if stripped.endswith(":"):
        raise ValueError("Line already ends with ':'")

    block_starters = (
        "def ",
        "async def ",
        "class ",
        "if ",
        "elif ",
        "else",
        "for ",
        "while ",
        "try",
        "except",
        "finally",
        "with ",
        "match ",
        "case ",
    )

    if not stripped.startswith(block_starters):
        raise ValueError(
            "Refusing to add ':' to a line that is not a "
            "recognized block header"
        )

    verify_current_diagnostic(
        source,
        diagnostic,
        check_span=False,
    )

    repair = Repair(
        file=diagnostic.file,
        start_line=diagnostic.line,
        start_column=len(line) + 1,
        end_line=diagnostic.line,
        end_column=len(line) + 1,
        replacement=":",
        reason="Add missing colon reported by the Python parser.",
    )

    build_candidate(source, repair)

    return RepairPlan(
        diagnostic=diagnostic,
        repair=repair,
    )
