from __future__ import annotations

from .candidate import build_candidate
from .confidence import RepairConfidence
from .plan import RepairPlan
from .parser_guard import verify_current_diagnostic
from .repair_strategies.dictionary import missing_dict_comma
from .repair_strategies.delimiters import unclosed_delimiter, unmatched_closing_delimiter
from .repair_strategies.comma import missing_comma
from .repair_strategies.punctuation import duplicate_punctuation
from .repair_strategies.imports_parameters import missing_import_comma, missing_parameter_comma
from .repair import Repair
from .scanner import Diagnostic
from .strategies import RepairStrategyRegistry


class RepairPlanner:
    """Create safe, deterministic repair proposals."""

    def __init__(self) -> None:
        self.registry = RepairStrategyRegistry()
        self.registry.register(
            "expected ':'",
            self._missing_colon,
        )
        self.registry.register(
            "'(' was never closed",
            self._unclosed_delimiter,
        )
        self.registry.register(
            "'[' was never closed",
            self._unclosed_delimiter,
        )
        self.registry.register(
            "'{' was never closed",
            self._unclosed_delimiter,
        )
        self.registry.register(
            "invalid syntax. Perhaps you forgot a comma?",
            missing_comma,
        )
        self.registry.register(
            "unmatched ')'",
            self._unmatched_closing_delimiter,
        )
        self.registry.register(
            "unmatched ']'",
            self._unmatched_closing_delimiter,
        )
        self.registry.register(
            "unmatched '}'",
            self._unmatched_closing_delimiter,
        )
        self.registry.register(
            "unexpected indent",
            self._unexpected_indent,
        )
        self.registry.register(
            "invalid syntax",
            self._plain_invalid_syntax,
        )
        self.registry.register(
            "unindent does not match any outer indentation level",
            self._unindent_mismatch,
        )
        self.registry.register_matcher(
            "expected-indented-block",
            lambda diagnostic: diagnostic.message.startswith(
                "expected an indented block after "
            ),
            self._expected_indented_block,
        )

    def plan(self, diagnostic: Diagnostic) -> RepairPlan | None:
        """Create a repair plan using the registered strategies."""
        return self.registry.resolve(diagnostic)

    def _unclosed_delimiter(self, diagnostic: Diagnostic) -> RepairPlan:
        return unclosed_delimiter(diagnostic)

    def _unmatched_closing_delimiter(
        self,
        diagnostic: Diagnostic,
    ) -> RepairPlan:
        return unmatched_closing_delimiter(diagnostic)

    def _unindent_mismatch(
        self,
        diagnostic: Diagnostic,
    ) -> RepairPlan:
        """Repair indentation that does not match an outer level."""

        source = diagnostic.file.read_text(encoding="utf-8")
        lines = source.splitlines(keepends=True)

        line_index = diagnostic.line - 1

        if line_index < 0 or line_index >= len(lines):
            raise ValueError("Diagnostic line is outside the source")

        line = lines[line_index].rstrip("\r\n")

        if not line.strip():
            raise ValueError(
                "Refusing to repair indentation on an empty line"
            )

        leading_text = line[: len(line) - len(line.lstrip(" \t"))]

        if not leading_text:
            raise ValueError(
                "Refusing to repair because the line has no indentation"
            )

        if " " in leading_text and "\t" in leading_text:
            raise ValueError(
                "Refusing to repair mixed tab and space indentation"
            )

        current_indent = len(leading_text.expandtabs(8))

        previous_levels: set[int] = {0}

        for previous_line in lines[:line_index]:
            text = previous_line.rstrip("\r\n")

            if not text.strip():
                continue

            prefix = text[: len(text) - len(text.lstrip(" \t"))]

            if " " in prefix and "\t" in prefix:
                continue

            previous_levels.add(len(prefix.expandtabs(8)))

        valid_outer_levels = {
            level
            for level in previous_levels
            if level < current_indent
        }

        if not valid_outer_levels:
            raise ValueError(
                "Refusing to repair because no valid outer "
                "indentation level was found"
            )

        target_indent = max(valid_outer_levels)

        if target_indent == current_indent:
            raise ValueError(
                "Refusing to repair because indentation already "
                "matches an outer level"
            )

        if "\t" in leading_text:
            replacement = "\t" * target_indent
        else:
            replacement = " " * target_indent

        verify_current_diagnostic(
            source,
            diagnostic,
            check_span=False,
        )

        repair = Repair(
            file=diagnostic.file,
            start_line=diagnostic.line,
            start_column=1,
            end_line=diagnostic.line,
            end_column=len(leading_text) + 1,
            replacement=replacement,
            reason=(
                "Correct indentation to the nearest valid outer "
                "indentation level reported by the Python parser."
            ),
        )

        build_candidate(source, repair)

        return RepairPlan(
            diagnostic=diagnostic,
            repair=repair,
        )

    def _expected_indented_block(
        self,
        diagnostic: Diagnostic,
    ) -> RepairPlan:
        """Indent the first statement after a block header."""

        source = diagnostic.file.read_text(encoding="utf-8")
        lines = source.splitlines(keepends=True)

        line_index = diagnostic.line - 1

        if line_index < 0 or line_index >= len(lines):
            raise ValueError("Diagnostic line is outside the source")

        line = lines[line_index].rstrip("\r\n")

        if not line.strip():
            raise ValueError(
                "Refusing to repair indentation on an empty line"
            )

        leading = len(line) - len(line.lstrip(" \t"))

        if leading != 0:
            raise ValueError(
                "Refusing to repair because the target line is "
                "already indented"
            )

        header_line_number = diagnostic.line - 1

        if header_line_number < 1:
            raise ValueError(
                "Refusing to repair because the block header "
                "cannot be identified"
            )

        header = lines[header_line_number - 1].rstrip("\r\n")

        if not header.strip():
            raise ValueError(
                "Refusing to repair because the block header is empty"
            )

        header_stripped = header.strip()

        block_prefixes = (
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

        if not header_stripped.startswith(block_prefixes):
            raise ValueError(
                "Refusing to repair because the preceding line "
                "is not a recognized block header"
            )

        header_indent = len(header) - len(header.lstrip(" \t"))

        # Do not guess mixed indentation. Use the same indentation
        # character as the block header, or four spaces for a
        # top-level block.
        if "\t" in header[:header_indent]:
            indentation = "\t"
        else:
            indentation = "    "

        verify_current_diagnostic(
            source,
            diagnostic,
            check_span=False,
        )

        repair = Repair(
            file=diagnostic.file,
            start_line=diagnostic.line,
            start_column=1,
            end_line=diagnostic.line,
            end_column=1,
            replacement=indentation,
            reason=(
                "Indent the first statement of the block reported "
                "by the Python parser."
            ),
        )

        build_candidate(source, repair)

        return RepairPlan(
            diagnostic=diagnostic,
            repair=repair,
        )

    def _unexpected_indent(self, diagnostic: Diagnostic) -> RepairPlan:
        """Repair an unexpected indentation conservatively."""

        source = diagnostic.file.read_text(encoding="utf-8")
        lines = source.splitlines(keepends=True)

        line_index = diagnostic.line - 1

        if line_index < 0 or line_index >= len(lines):
            raise ValueError("Diagnostic line is outside the source")

        line = lines[line_index].rstrip("\r\n")

        if not line.strip():
            raise ValueError(
                "Refusing to repair indentation on an empty line"
            )

        leading = len(line) - len(line.lstrip(" \t"))

        if leading == 0:
            raise ValueError(
                "Refusing to repair because the line has no indentation"
            )

        # Python reports the indentation column as 1-based. The
        # reported offset identifies the first non-whitespace column.
        if diagnostic.column != leading:
            raise ValueError(
                "Refusing to repair because the diagnostic column "
                "does not match the line indentation"
            )

        # Safely remove one indentation level. Four spaces is the
        # conventional Python indentation used by this project.
        if line.startswith("    "):
            replacement = line[4:]
            start_column = 1
            end_column = 5
        elif line.startswith("\t"):
            replacement = line[1:]
            start_column = 1
            end_column = 2
        else:
            raise ValueError(
                "Refusing to repair because the indentation does not "
                "start with a complete indentation level"
            )

        verify_current_diagnostic(
            source,
            diagnostic,
            check_span=False,
        )

        repair = Repair(
            file=diagnostic.file,
            start_line=diagnostic.line,
            start_column=start_column,
            end_line=diagnostic.line,
            end_column=end_column,
            replacement="",
            reason=(
                "Remove one excess indentation level reported by "
                "the Python parser."
            ),
        )

        build_candidate(source, repair)

        return RepairPlan(
            diagnostic=diagnostic,
            repair=repair,
        )

    def _plain_invalid_syntax(
        self,
        diagnostic: Diagnostic,
    ) -> RepairPlan:
        """Dispatch conservative repairs for plain invalid syntax."""

        try:
            return self._duplicate_punctuation(diagnostic)
        except ValueError:
            pass

        try:
            return self._missing_import_comma(diagnostic)
        except ValueError:
            pass

        try:
            return self._missing_parameter_comma(diagnostic)
        except ValueError:
            pass

        try:
            return self._missing_dict_comma(diagnostic)
        except ValueError:
            pass

        raise ValueError(
            "No safe deterministic repair for plain invalid syntax"
        )

    def _missing_parameter_comma(
        self,
        diagnostic: Diagnostic,
    ) -> RepairPlan:
        return missing_parameter_comma(diagnostic)

    def _missing_import_comma(
        self,
        diagnostic: Diagnostic,
    ) -> RepairPlan:
        return missing_import_comma(diagnostic)

    def _duplicate_punctuation(
        self,
        diagnostic: Diagnostic,
    ) -> RepairPlan:
        return duplicate_punctuation(diagnostic)

    def _missing_comma(
        self,
        diagnostic: Diagnostic,
    ) -> RepairPlan:
        return missing_comma(diagnostic)

    def _missing_dict_comma(
        self,
        diagnostic: Diagnostic,
    ) -> RepairPlan:
        return missing_dict_comma(diagnostic)

    def _missing_colon(self, diagnostic: Diagnostic) -> RepairPlan:
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
