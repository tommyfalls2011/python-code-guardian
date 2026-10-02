from __future__ import annotations

from .candidate import build_candidate
from .confidence import RepairConfidence
from .plan import RepairPlan
from .parser_guard import verify_current_diagnostic
from .repair_strategies.dictionary import missing_dict_comma
from .repair_strategies.comma import missing_comma
from .repair_strategies.punctuation import duplicate_punctuation
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

    def _unmatched_closing_delimiter(
        self,
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

    def _missing_import_comma(
        self,
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

    def _duplicate_punctuation(
        self,
        diagnostic: Diagnostic,
    ) -> RepairPlan:
        """Remove parser-identified duplicated punctuation."""

        import tokenize
        from io import StringIO

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

    def _missing_comma(self, diagnostic: Diagnostic) -> RepairPlan:
        """Repair a parser-reported missing comma conservatively."""

        import tokenize
        from io import StringIO

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

        candidate_columns = []

        # Python commonly points at the first expression when a
        # comma is missing between positional/container elements.
        candidate_columns.append(target.end[1] + 1)

        # For missing commas between keyword arguments, Python points
        # at the next argument, so the comma belongs before that token.
        candidate_columns.append(target.start[1] + 1)

        valid_columns: list[int] = []

        for column in dict.fromkeys(candidate_columns):
            # When Python provides an error span on this same line,
            # require the insertion point to stay inside that span.
            # SyntaxError offsets are 1-based and end_offset is
            # exclusive, so insertion columns may range from the
            # reported start through the reported end.
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
                reason=(
                    "Add missing comma reported by the Python parser."
                ),
            )

            try:
                build_candidate(source, repair)
            except (ValueError, SyntaxError):
                continue

            valid_columns.append(column)

        if not valid_columns:
            # Dictionary values can make Python report a span that does
            # not directly identify the missing comma. As a narrow
            # fallback, try whitespace boundaries immediately before a
            # potential dictionary key. The full candidate must still
            # parse, and exactly one candidate must succeed.
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

            dict_candidate_columns: list[int] = []

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

                # A likely dictionary key must be followed by ':'.
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
                    reason=(
                        "Add missing comma between dictionary items."
                    ),
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
                    reason=(
                        "Add missing comma between dictionary items."
                    ),
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
            if not valid_columns:
                raise ValueError(
                    "Unable to find a safe comma insertion point"
                )

            raise ValueError(
                "Refusing to repair because multiple comma insertion "
                "points produce valid Python"
            )

        insertion_column = valid_columns[0]

        repair = Repair(
            file=diagnostic.file,
            start_line=diagnostic.line,
            start_column=insertion_column,
            end_line=diagnostic.line,
            end_column=insertion_column,
            replacement=",",
            reason=(
                "Add missing comma reported by the Python parser."
            ),
            confidence=RepairConfidence.MEDIUM,
        )

        build_candidate(source, repair)

        return RepairPlan(
            diagnostic=diagnostic,
            repair=repair,
        )

    def _missing_dict_comma(
        self,
        diagnostic: Diagnostic,
    ) -> RepairPlan:
        """Repair a missing comma between dictionary items."""

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

            # This strategy is dictionary-specific. The candidate key
            # must occur inside an unmatched '{' at this token position.
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
                reason=(
                    "Add missing comma between dictionary items."
                ),
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
            reason=(
                "Add missing comma between dictionary items."
            ),
            confidence=RepairConfidence.MEDIUM,
        )

        build_candidate(source, repair)

        return RepairPlan(
            diagnostic=diagnostic,
            repair=repair,
        )

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
