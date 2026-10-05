from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AIEdit:
    """A single bounded source edit proposed by an AI model."""

    operation: str
    line: int
    content: str = ""


def apply_ai_edit(source: str, edit: AIEdit) -> str:
    lines = source.splitlines(keepends=True)

    if edit.operation not in {"delete", "replace", "insert_before", "insert_after"}:
        raise ValueError(f"unsupported AI edit operation: {edit.operation}")

    if edit.line < 1 or edit.line > len(lines):
        raise ValueError("AI edit line is outside the source")

    index = edit.line - 1

    if edit.operation == "delete":
        if edit.content:
            raise ValueError("delete operation must not contain content")
        del lines[index]
        return "".join(lines)

    content = edit.content
    if not content:
        raise ValueError(f"{edit.operation} operation requires content")
    if not content.endswith("\n"):
        content += "\n"

    if edit.operation == "replace":
        lines[index] = content
    elif edit.operation == "insert_before":
        lines.insert(index, content)
    else:
        lines.insert(index + 1, content)

    return "".join(lines)
