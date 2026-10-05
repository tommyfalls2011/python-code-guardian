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

def apply_ai_edits(
    source: str,
    edits: list[AIEdit],
    *,
    max_edits: int = 3,
) -> str:
    """Apply a bounded edit transaction using original line numbers."""

    if not edits:
        raise ValueError("AI edit transaction is empty")

    if len(edits) > max_edits:
        raise ValueError(
            f"AI edit transaction exceeds maximum of {max_edits}"
        )

    lines = source.splitlines(keepends=True)
    line_count = len(lines)

    by_line: dict[int, list[AIEdit]] = {}

    for edit in edits:
        if edit.line < 1 or edit.line > line_count:
            raise ValueError(
                "AI edit line is outside the source"
            )

        by_line.setdefault(edit.line, []).append(edit)

    for line_edits in by_line.values():
        destructive = [
            edit
            for edit in line_edits
            if edit.operation in {"delete", "replace"}
        ]

        before = [
            edit
            for edit in line_edits
            if edit.operation == "insert_before"
        ]

        after = [
            edit
            for edit in line_edits
            if edit.operation == "insert_after"
        ]

        if len(destructive) > 1:
            raise ValueError(
                "AI edit transaction contains conflicting lines"
            )

        if len(before) > 1 or len(after) > 1:
            raise ValueError(
                "AI edit transaction contains conflicting lines"
            )

        if destructive and destructive[0].operation == "delete":
            if before or after:
                raise ValueError(
                    "AI edit transaction contains conflicting lines"
                )

    candidate = source

    operation_order = {
        "insert_after": 0,
        "replace": 1,
        "delete": 1,
        "insert_before": 2,
    }

    for edit in sorted(
        edits,
        key=lambda item: (
            item.line,
            operation_order.get(item.operation, -1),
        ),
        reverse=True,
    ):
        candidate = apply_ai_edit(candidate, edit)

    return candidate

def normalize_ai_edits(edits: list[AIEdit]) -> list[AIEdit]:
    """Normalize only structurally safe adjacent insertion blocks."""

    normalized = list(edits)

    changed = True
    while changed:
        changed = False

        for index, first in enumerate(normalized):
            if first.operation != "insert_after":
                continue

            first_lines = first.content.splitlines()
            if not first_lines:
                continue

            first_text = first_lines[-1]
            first_indent = len(first_text) - len(
                first_text.lstrip()
            )

            if not first_text.rstrip().endswith(":"):
                continue

            for second_index, second in enumerate(normalized):
                if second_index == index:
                    continue

                if second.operation != "insert_after":
                    continue

                if second.line != first.line + 1:
                    continue

                second_lines = second.content.splitlines()
                if not second_lines:
                    continue

                second_text = second_lines[0]
                second_indent = len(second_text) - len(
                    second_text.lstrip()
                )

                if second_indent <= first_indent:
                    continue

                merged_content = (
                    first.content.rstrip("\n")
                    + "\n"
                    + second.content.rstrip("\n")
                )

                merged = AIEdit(
                    operation="insert_after",
                    line=first.line,
                    content=merged_content,
                )

                low = min(index, second_index)
                high = max(index, second_index)

                normalized.pop(high)
                normalized.pop(low)
                normalized.insert(low, merged)

                changed = True
                break

            if changed:
                break

    return normalized
