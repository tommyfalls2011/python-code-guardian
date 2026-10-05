from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceWindow:
    """A bounded source region that an AI may replace."""

    start_line: int
    end_line: int
    source: str


def extract_source_window(
    source: str,
    line: int,
    *,
    context: int = 4,
) -> SourceWindow:
    if line < 1:
        raise ValueError("line must be >= 1")
    if context < 0:
        raise ValueError("context must be >= 0")

    lines = source.splitlines(keepends=True)
    if not lines:
        raise ValueError("source is empty")
    if line > len(lines):
        raise ValueError("line is outside the source")

    start = max(1, line - context)
    end = min(len(lines), line + context)

    return SourceWindow(
        start_line=start,
        end_line=end,
        source="".join(lines[start - 1:end]),
    )


def apply_source_window(
    source: str,
    window: SourceWindow,
    replacement: str,
) -> str:
    lines = source.splitlines(keepends=True)

    if window.start_line < 1 or window.end_line < window.start_line:
        raise ValueError("invalid source window")
    if window.end_line > len(lines):
        raise ValueError("source window is outside the source")

    current = "".join(
        lines[window.start_line - 1:window.end_line]
    )
    if current != window.source:
        raise ValueError("source changed after window extraction")

    if replacement and not replacement.endswith("\n"):
        replacement += "\n"

    return (
        "".join(lines[:window.start_line - 1])
        + replacement
        + "".join(lines[window.end_line:])
    )
