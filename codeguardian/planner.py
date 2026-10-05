from __future__ import annotations

from .candidate import build_candidate
from .confidence import RepairConfidence
from .plan import RepairPlan
from .parser_guard import verify_current_diagnostic
from .repair_strategies.dictionary import missing_dict_comma
from .repair_strategies.delimiters import unclosed_delimiter, unmatched_closing_delimiter
from .repair_strategies.indentation import unindent_mismatch, expected_indented_block, unexpected_indent
from .repair_strategies.comma import missing_comma
from .repair_strategies.colon import missing_colon
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
        return unindent_mismatch(diagnostic)

    def _expected_indented_block(
        self,
        diagnostic: Diagnostic,
    ) -> RepairPlan:
        return expected_indented_block(diagnostic)

    def _unexpected_indent(self, diagnostic: Diagnostic) -> RepairPlan:
        return unexpected_indent(diagnostic)

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
        return missing_colon(diagnostic)
