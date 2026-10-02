from __future__ import annotations

import tokenize
from io import StringIO

from ..candidate import build_candidate
from ..confidence import RepairConfidence
from ..parser_guard import verify_current_diagnostic
from ..plan import RepairPlan
from ..repair import Repair
from ..scanner import Diagnostic


def missing_comma(
    diagnostic: Diagnostic,
) -> RepairPlan:
    """Repair a parser-reported missing comma conservatively."""

    source = diagnostic.file.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)

    line_index = diagnostic.line - 1

    if line_index < 0 or line_index >= len(lines):
        raise ValueError("Diagnostic line is outside the source")

    content = lines[line_index].rstrip("\r\n")

    if not content.strip():
        raise ValueError(
            "Refusing to add ',' to an empty source line"
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
            f"Unable to safely analyze comma location: {exc}"
        ) from exc

    target = None

    for token in tokens:
        if token.type in {
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.COMMENT,
        }:
            continue

        if token.start[0] != 1:
            continue

        token_start = token.start[1] + 1
        token_end = token.end[1]

        if token_start <= diagnostic.column <= token_end:
            target = token
            break

    if target is None:
        raise ValueError(
            "Unable to identify the parser-reported token"
        )

    candidate_columns = [
        target.end[1] + 1,
        target.start[1] + 1,
    ]

    valid_columns: list[int] = []

    for column in dict.fromkeys(candidate_columns):
        if (
            diagnostic.end_line == diagnostic.line
            and diagnostic.end_column is not None
            and not (
                diagnostic.column
                <= column
                <= diagnostic.end_column
            )
        ):
            continue

        repair = Repair(
            file=diagnostic.file,
            start_line=diagnostic.line,
            start_column=column,
            end_line=diagnostic.line,
            end_column=column,
            replacement=",",
            reason="Add missing comma reported by the Python parser.",
        )

        try:
            build_candidate(source, repair)
        except (ValueError, SyntaxError):
            continue

        valid_columns.append(column)

    if not valid_columns:
        significant_tokens = [
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

        dict_candidate_columns: list[tuple[int, int]] = []

        for index, token in enumerate(significant_tokens):
            if index == 0:
                continue

            if token.type not in {
                tokenize.NAME,
                tokenize.NUMBER,
                tokenize.STRING,
            }:
                continue

            previous = significant_tokens[index - 1]

            gap_start = previous.end[1]
            gap_end = token.start[1]

            if gap_end <= gap_start:
                continue

            gap = content[gap_start:gap_end]

            if not gap or not gap.isspace():
                continue

            if index + 1 >= len(significant_tokens):
                continue

            following = significant_tokens[index + 1]

            if following.string != ":":
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
            )

            try:
                build_candidate(source, repair)
            except (ValueError, SyntaxError):
                continue

            dict_candidate_columns.append(
                (start_column, end_column)
            )

        unique_dict_candidates = list(
            dict.fromkeys(dict_candidate_columns)
        )

        if len(unique_dict_candidates) == 1:
            start_column, end_column = unique_dict_candidates[0]

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

        if len(unique_dict_candidates) > 1:
            raise ValueError(
                "Refusing to repair because multiple dictionary "
                "comma insertion points produce valid Python"
            )

    if len(valid_columns) != 1:
        raise ValueError(
            "Unable to find a safe comma insertion point"
        )

    repair = Repair(
        file=diagnostic.file,
        start_line=diagnostic.line,
        start_column=valid_columns[0],
        end_line=diagnostic.line,
        end_column=valid_columns[0],
        replacement=",",
        reason="Add missing comma reported by the Python parser.",
        confidence=RepairConfidence.MEDIUM,
    )

    build_candidate(source, repair)

    return RepairPlan(
        diagnostic=diagnostic,
        repair=repair,
    )
