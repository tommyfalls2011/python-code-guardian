from __future__ import annotations

from ..candidate import build_candidate
from ..confidence import RepairConfidence
from ..parser_guard import verify_current_diagnostic
from ..repair import Repair
from ..scanner import Diagnostic
from ..plan import RepairPlan

def missing_parameter_comma(
    diagnostic: Diagnostic,
) -> RepairPlan:
    """Insert a missing comma between bare function parameters."""

    import tokenize
    from io import StringIO

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
            f"Unable to safely analyze parameters: {exc}"
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

    strings = [token.string for token in significant]

    if len(strings) < 5:
        raise ValueError(
            "Function definition does not contain enough tokens"
        )

    if strings[0] == "def":
        def_index = 0
    elif (
        len(strings) >= 2
        and strings[0] == "async"
        and strings[1] == "def"
    ):
        def_index = 1
    else:
        raise ValueError(
            "Diagnostic is not on a function definition"
        )

    try:
        open_index = strings.index("(", def_index + 1)
    except ValueError as exc:
        raise ValueError(
            "Function parameter list has no opening parenthesis"
        ) from exc

    target_index = None

    for index, token in enumerate(significant):
        if index <= open_index:
            continue

        start_column = token.start[1] + 1
        end_column = token.end[1]

        if (
            start_column
            <= diagnostic.column
            <= end_column
        ):
            target_index = index
            break

    if target_index is None:
        raise ValueError(
            "Unable to identify invalid parameter token"
        )

    if target_index <= open_index + 1:
        raise ValueError(
            "There is no preceding parameter"
        )

    target = significant[target_index]
    previous = significant[target_index - 1]

    if target.type != tokenize.NAME:
        raise ValueError(
            "Invalid parameter token is not a bare name"
        )

    if previous.type != tokenize.NAME:
        raise ValueError(
            "Previous parameter token is not a bare name"
        )

    previous_end = previous.end[1]
    target_start = target.start[1]

    between = content[previous_end:target_start]

    if not between or not between.isspace():
        raise ValueError(
            "Parameters are not separated only by whitespace"
        )

    repair = Repair(
        file=diagnostic.file,
        start_line=diagnostic.line,
        start_column=previous_end + 1,
        end_line=diagnostic.line,
        end_column=target_start + 1,
        replacement=", ",
        reason="Insert missing comma between function parameters.",
        confidence=RepairConfidence.HIGH,
    )

    build_candidate(source, repair)

    return RepairPlan(
        diagnostic=diagnostic,
        repair=repair,
    )

def missing_import_comma(
    diagnostic: Diagnostic,
) -> RepairPlan:
    """Insert a missing comma between imported names."""

    import tokenize
    from io import StringIO

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
            f"Unable to safely analyze import: {exc}"
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

    strings = [token.string for token in significant]

    if not strings:
        raise ValueError("Import line contains no tokens")

    if strings[0] == "import":
        import_index = 0
    elif (
        strings[0] == "from"
        and "import" in strings
    ):
        import_index = strings.index("import")
    else:
        raise ValueError(
            "Diagnostic is not on an import statement"
        )

    target_index = None

    for index, token in enumerate(significant):
        if index <= import_index:
            continue

        start_column = token.start[1] + 1
        end_column = token.end[1]

        if (
            start_column
            <= diagnostic.column
            <= end_column
        ):
            target_index = index
            break

    if target_index is None:
        raise ValueError(
            "Unable to identify invalid import token"
        )

    if target_index <= import_index:
        raise ValueError(
            "Invalid token is not an imported name"
        )

    target = significant[target_index]
    previous = significant[target_index - 1]

    if target.type != tokenize.NAME:
        raise ValueError(
            "Invalid import token is not a name"
        )

    if previous.type != tokenize.NAME:
        raise ValueError(
            "Previous import token is not a name"
        )

    if previous.string in {"import", "as"}:
        raise ValueError(
            "Refusing ambiguous import repair"
        )

    if target.string == "as":
        raise ValueError(
            "Refusing ambiguous import repair"
        )

    previous_end = previous.end[1]
    target_start = target.start[1]

    between = content[previous_end:target_start]

    if not between or not between.isspace():
        raise ValueError(
            "Imported names are not separated only by whitespace"
        )

    repair = Repair(
        file=diagnostic.file,
        start_line=diagnostic.line,
        start_column=previous_end + 1,
        end_line=diagnostic.line,
        end_column=target_start + 1,
        replacement=", ",
        reason="Insert missing comma between imported names.",
        confidence=RepairConfidence.HIGH,
    )

    build_candidate(source, repair)

    return RepairPlan(
        diagnostic=diagnostic,
        repair=repair,
    )

