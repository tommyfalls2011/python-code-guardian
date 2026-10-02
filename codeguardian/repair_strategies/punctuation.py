from __future__ import annotations

import tokenize
from io import StringIO

from ..candidate import build_candidate
from ..confidence import RepairConfidence
from ..parser_guard import verify_current_diagnostic
from ..plan import RepairPlan
from ..repair import Repair
from ..scanner import Diagnostic


def duplicate_punctuation(
    diagnostic: Diagnostic,
) -> RepairPlan:
    """Remove parser-identified duplicated punctuation."""

    source = diagnostic.file.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)

    line_index = diagnostic.line - 1

    if line_index < 0 or line_index >= len(lines):
        raise ValueError("Diagnostic line is outside the source")

    content = lines[line_index].rstrip("\r\n")

    if not content.strip():
        raise ValueError(
            "Refusing to repair punctuation on an empty line"
        )

    if diagnostic.column < 1 or diagnostic.column > len(content):
        raise ValueError(
            "Diagnostic column is outside the source"
        )

    verify_current_diagnostic(
        source,
        diagnostic,
        check_span=True,
    )

    try:
        tokens = list(
            tokenize.generate_tokens(
                StringIO(content + "\n").readline
            )
        )
    except (tokenize.TokenError, IndentationError) as exc:
        raise ValueError(
            f"Unable to safely analyze punctuation: {exc}"
        ) from exc

    significant = [
        token
        for token in tokens
        if token.type
        not in {
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.COMMENT,
        }
    ]

    target_index = None

    for index, token in enumerate(significant):
        if token.start[0] != 1:
            continue

        start_column = token.start[1] + 1
        end_column = token.end[1]

        if start_column <= diagnostic.column <= end_column:
            target_index = index
            break

    if target_index is None:
        raise ValueError(
            "Unable to identify the parser-reported token"
        )

    if target_index == 0:
        raise ValueError(
            "Refusing to repair because there is no preceding token"
        )

    target = significant[target_index]
    previous = significant[target_index - 1]

    allowed = {"=", ",", ":"}

    if target.string not in allowed:
        raise ValueError(
            "Refusing to repair unsupported punctuation"
        )

    if previous.string != target.string:
        raise ValueError(
            "Refusing to repair because punctuation is not duplicated"
        )

    repair = Repair(
        file=diagnostic.file,
        start_line=diagnostic.line,
        start_column=target.start[1] + 1,
        end_line=diagnostic.line,
        end_column=target.end[1] + 1,
        replacement="",
        reason=(
            "Remove duplicated punctuation identified by "
            "the Python parser."
        ),
        confidence=RepairConfidence.HIGH,
    )

    build_candidate(source, repair)

    return RepairPlan(
        diagnostic=diagnostic,
        repair=repair,
    )
