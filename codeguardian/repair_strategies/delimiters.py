from __future__ import annotations

from ..candidate import build_candidate
from ..confidence import RepairConfidence
from ..parser_guard import verify_current_diagnostic
from ..plan import RepairPlan
from ..repair import Repair
from ..scanner import Diagnostic

def unclosed_delimiter(diagnostic: Diagnostic) -> RepairPlan:
    source = diagnostic.file.read_text(encoding="utf-8")

    pairs = {
        "(": ")",
        "[": "]",
        "{": "}",
    }

    stack: list[str] = []
    final_line_comment_column: int | None = None

    import tokenize
    from io import StringIO

    tokens = tokenize.generate_tokens(StringIO(source).readline)

    try:
        for token in tokens:
            if token.type == tokenize.COMMENT:
                if token.start[0] == source.count("\n") + 1:
                    final_line_comment_column = token.start[1] + 1
                continue

            if token.type != tokenize.OP:
                continue

            value = token.string

            if value in pairs:
                stack.append(value)
            elif value in pairs.values():
                if not stack or pairs[stack[-1]] != value:
                    raise ValueError(
                        "Delimiter structure is ambiguous"
                    )
                stack.pop()
    except tokenize.TokenError as exc:
        # An unexpected EOF is expected for the exact class of
        # incomplete source this strategy is designed to repair.
        if not str(exc.args[0]).startswith("unexpected EOF"):
            raise ValueError(
                f"Unable to safely analyze delimiters: {exc}"
            ) from exc

    if len(stack) != 1:
        raise ValueError(
            "Refusing repair because the source does not contain "
            "exactly one unmatched opening delimiter"
        )

    opening = stack[0]
    expected = pairs[opening]

    diagnostic_opening = {
        "'(' was never closed": "(",
        "'[' was never closed": "[",
        "'{' was never closed": "{",
    }.get(diagnostic.message)

    if diagnostic_opening is None:
        raise ValueError(
            "Unsupported unclosed-delimiter diagnostic"
        )

    if opening != diagnostic_opening:
        raise ValueError(
            "Refusing repair because the unmatched delimiter "
            "does not match the parser diagnostic"
        )

    verify_current_diagnostic(
        source,
        diagnostic,
        check_span=False,
    )

    lines = source.splitlines(keepends=True)

    if not lines:
        raise ValueError(
            "Refusing repair because the source is empty"
        )

    last_line_number = len(lines)
    last_line = lines[-1]
    last_content = last_line.rstrip("\r\n")

    insertion_column = len(last_content) + 1

    # If the final physical line contains a comment, the closing
    # delimiter must be inserted before the comment. Tokenize the
    # final line independently so an incomplete source TokenError
    # cannot prevent us from finding that comment.
    try:
        import tokenize
        from io import StringIO

        line_tokens = tokenize.generate_tokens(
            StringIO(last_content + "\n").readline
        )

        for token in line_tokens:
            if token.type == tokenize.COMMENT:
                insertion_column = token.start[1] + 1
                break
    except (tokenize.TokenError, IndentationError):
        pass

    repair = Repair(
        file=diagnostic.file,
        start_line=last_line_number,
        start_column=insertion_column,
        end_line=last_line_number,
        end_column=insertion_column,
        replacement=expected,
        reason=(
            f"Add missing closing delimiter '{expected}' "
            f"for unmatched '{opening}'."
        ),
    )

    build_candidate(source, repair)

    return RepairPlan(
        diagnostic=diagnostic,
        repair=repair,
    )

def unmatched_closing_delimiter(
    diagnostic: Diagnostic,
) -> RepairPlan:
    """Remove one parser-identified unmatched closing delimiter."""

    expected = {
        "unmatched ')'": ")",
        "unmatched ']'": "]",
        "unmatched '}'": "}",
    }.get(diagnostic.message)

    if expected is None:
        raise ValueError(
            "Unsupported unmatched-closing-delimiter diagnostic"
        )

    source = diagnostic.file.read_text(encoding="utf-8")

    verify_current_diagnostic(
        source,
        diagnostic,
        check_span=True,
        span_error=(
            "span does not identify exactly one "
            "unmatched closing delimiter"
        ),
    )

    lines = source.splitlines(keepends=True)
    line_index = diagnostic.line - 1

    if line_index < 0 or line_index >= len(lines):
        raise ValueError("Diagnostic line is outside the source")

    content = lines[line_index].rstrip("\r\n")
    character_index = diagnostic.column - 1

    if (
        character_index < 0
        or character_index >= len(content)
    ):
        raise ValueError(
            "Diagnostic column is outside the source"
        )

    if content[character_index] != expected:
        raise ValueError(
            "Parser diagnostic does not point at the expected "
            "closing delimiter"
        )

    if (
        diagnostic.end_line is not None
        and diagnostic.end_line != diagnostic.line
    ):
        raise ValueError(
            "Refusing unmatched delimiter repair across lines"
        )

    if (
        diagnostic.end_column is not None
        and diagnostic.end_column != diagnostic.column
    ):
        raise ValueError(
            "Parser span does not identify exactly one closing "
            "delimiter"
        )

    repair = Repair(
        file=diagnostic.file,
        start_line=diagnostic.line,
        start_column=diagnostic.column,
        end_line=diagnostic.line,
        end_column=diagnostic.column + 1,
        replacement="",
        reason=(
            f"Remove unmatched closing delimiter '{expected}' "
            "reported by the Python parser."
        ),
        confidence=RepairConfidence.HIGH,
    )

    build_candidate(source, repair)

    return RepairPlan(
        diagnostic=diagnostic,
        repair=repair,
    )

