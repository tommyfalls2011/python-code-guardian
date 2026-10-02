from __future__ import annotations

from .scanner import Diagnostic


def verify_current_diagnostic(
    source: str,
    diagnostic: Diagnostic,
    *,
    check_span: bool = True,
    span_error: str | None = None,
) -> None:
    """
    Verify that a diagnostic still describes the current source.

    Raises ValueError when the source changed or the diagnostic
    no longer matches the parser output.
    """

    try:
        compile(
            source,
            str(diagnostic.file),
            "exec",
        )
    except SyntaxError as exc:
        if (
            exc.msg != diagnostic.message
            or exc.lineno != diagnostic.line
            or exc.offset != diagnostic.column
        ):
            raise ValueError(
                "Parser diagnostic does not match the current "
                "source location"
            ) from exc

        if check_span:
            if (
                diagnostic.end_line is not None
                and exc.end_lineno != diagnostic.end_line
            ):
                raise ValueError(
                    span_error
                    or "Parser diagnostic span does not match the "
                    "current source"
                ) from exc

            if (
                diagnostic.end_column is not None
                and exc.end_offset != diagnostic.end_column
            ):
                raise ValueError(
                    span_error
                    or "Parser diagnostic span does not match the "
                    "current source"
                ) from exc

        return

    raise ValueError(
        "Current source no longer has the reported syntax error"
    )
