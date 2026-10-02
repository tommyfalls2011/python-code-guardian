from __future__ import annotations

import tokenize
from io import StringIO

from ..candidate import build_candidate
from ..confidence import RepairConfidence
from ..parser_guard import verify_current_diagnostic
from ..plan import RepairPlan
from ..repair import Repair
from ..scanner import Diagnostic


def missing_dict_comma(
    diagnostic: Diagnostic,
) -> RepairPlan:
    """Repair a missing comma between dictionary items."""

    source = diagnostic.file.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    line_index = diagnostic.line - 1

    if line_index < 0 or line_index >= len(lines):
        raise ValueError("Diagnostic line is outside the source")

    content = lines[line_index].rstrip("\r\n")

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
            f"Unable to safely analyze dictionary: {exc}"
        ) from exc

    significant = [
        token
        for token in tokens
        if token.type not in {
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.COMMENT,
        }
        and token.start[0] == 1
    ]

    candidates: list[tuple[int, int]] = []

    for index, token in enumerate(significant):
        if index == 0 or index + 1 >= len(significant):
            continue

        if token.type not in {
            tokenize.NAME,
            tokenize.NUMBER,
            tokenize.STRING,
        }:
            continue

        brace_depth = 0

        for prior in significant[:index]:
            if prior.string == "{":
                brace_depth += 1
            elif prior.string == "}":
                brace_depth -= 1

        if brace_depth <= 0:
            continue

        following = significant[index + 1]

        if following.string != ":":
            continue

        previous = significant[index - 1]

        gap_start = previous.end[1]
        gap_end = token.start[1]

        if gap_end <= gap_start:
            continue

        gap = content[gap_start:gap_end]

        if not gap or not gap.isspace():
            continue

        start_column = gap_start + 1
        end_column = gap_end + 1

        repair = Repair(
            file=diagnostic.file,
            start_line=diagnostic.line,
            start_column=start_column,
            end_line=diagnostic.line,
            end_column=end_column,
            replacement=", ",
            reason="Add missing comma between dictionary items.",
            confidence=RepairConfidence.MEDIUM,
        )

        try:
            build_candidate(source, repair)
        except (ValueError, SyntaxError):
            continue

        candidates.append(
            (start_column, end_column)
        )

    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        raise ValueError(
            "Unable to find a safe dictionary comma location"
        )

    if len(candidates) != 1:
        raise ValueError(
            "Refusing ambiguous dictionary comma repair"
        )

    start_column, end_column = candidates[0]

    repair = Repair(
        file=diagnostic.file,
        start_line=diagnostic.line,
        start_column=start_column,
        end_line=diagnostic.line,
        end_column=end_column,
        replacement=", ",
        reason="Add missing comma between dictionary items.",
        confidence=RepairConfidence.MEDIUM,
    )

    build_candidate(source, repair)

    return RepairPlan(
        diagnostic=diagnostic,
        repair=repair,
    )
