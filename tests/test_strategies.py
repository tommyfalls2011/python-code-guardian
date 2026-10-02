from pathlib import Path

import pytest

from codeguardian.planner import RepairPlan
from codeguardian.scanner import Diagnostic
from codeguardian.strategies import RepairStrategyRegistry


def diagnostic(message: str) -> Diagnostic:
    return Diagnostic(
        file=Path("/tmp/example.py"),
        line=1,
        column=1,
        severity="ERROR",
        message=message,
    )


def test_registry_starts_empty():
    registry = RepairStrategyRegistry()

    assert registry.messages() == ()
    assert registry.resolve(diagnostic("missing")) is None


def test_registry_registers_and_resolves_strategy():
    registry = RepairStrategyRegistry()

    expected = object()

    def strategy(item):
        assert item.message == "test"
        return expected

    registry.register("test", strategy)

    assert registry.resolve(diagnostic("test")) is expected


def test_registry_rejects_duplicate_strategy():
    registry = RepairStrategyRegistry()

    def strategy(item):
        return None

    registry.register("test", strategy)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("test", strategy)


def test_registry_rejects_empty_message():
    registry = RepairStrategyRegistry()

    with pytest.raises(ValueError, match="must not be empty"):
        registry.register("   ", lambda item: None)


def test_registry_messages_are_sorted():
    registry = RepairStrategyRegistry()

    registry.register("z-message", lambda item: None)
    registry.register("a-message", lambda item: None)

    assert registry.messages() == (
        "a-message",
        "z-message",
    )


def test_missing_colon_strategy_rejects_unrecognized_line(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "value = some_expression\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=1,
        severity="ERROR",
        message="expected ':'",
    )

    with pytest.raises(ValueError, match="Refusing to add"):
        RepairPlanner().plan(item)


def test_missing_colon_strategy_accepts_async_function(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "async def hello(name)\n"
        "    return name\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=22,
        severity="ERROR",
        message="expected ':'",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == ":"


def test_unclosed_parenthesis_is_repaired(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "def hello(name):\n"
        "    return (name.upper()\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=2,
        column=12,
        severity="ERROR",
        message="'(' was never closed",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == ")"

    source = path.read_text(encoding="utf-8")
    assert plan.repair.start_line == 2
    assert plan.repair.start_column == 25
    assert plan.repair.end_line == 2
    assert plan.repair.end_column == 25


def test_unclosed_bracket_is_repaired(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "values = [1, 2, 3\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=10,
        severity="ERROR",
        message="'[' was never closed",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == "]"


def test_unclosed_brace_is_repaired(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "values = {1, 2, 3\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=10,
        severity="ERROR",
        message="'{' was never closed",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == "}"


def test_unclosed_delimiter_rejects_multiple_unmatched_openers(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "values = ([1, 2\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=10,
        severity="ERROR",
        message="'(' was never closed",
    )

    with pytest.raises(ValueError, match="exactly one unmatched"):
        RepairPlanner().plan(item)


def test_unclosed_delimiter_ignores_delimiters_in_strings(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        'text = "([{"\n'
        "values = [1, 2, 3\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=2,
        column=10,
        severity="ERROR",
        message="'[' was never closed",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == "]"


def test_unclosed_delimiter_ignores_delimiters_in_comments(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "# ([{\n"
        "values = [1, 2, 3  # ] })\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=2,
        column=10,
        severity="ERROR",
        message="'[' was never closed",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == "]"


def test_unclosed_delimiter_rejects_mismatched_structure(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "values = ([1, 2)\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=10,
        severity="ERROR",
        message="'(' was never closed",
    )

    with pytest.raises(ValueError, match="ambiguous"):
        RepairPlanner().plan(item)


def test_missing_comma_repairs_list(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "values = [1 2, 3]\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=11,
        severity="ERROR",
        message="invalid syntax. Perhaps you forgot a comma?",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == ","
    assert plan.repair.start_line == 1
    assert plan.repair.start_column == 12

    from codeguardian.candidate import build_candidate

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == "values = [1, 2, 3]\n"


def test_missing_comma_repairs_function_call(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "result = hello(1 2)\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=16,
        severity="ERROR",
        message="invalid syntax. Perhaps you forgot a comma?",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == ","

    from codeguardian.candidate import build_candidate

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == "result = hello(1, 2)\n"


def test_missing_comma_repairs_tuple(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "values = (1 2, 3)\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=11,
        severity="ERROR",
        message="invalid syntax. Perhaps you forgot a comma?",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == ","


def test_missing_comma_repairs_dict(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        'values = {"a": 1 "b": 2}\n',
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=16,
        severity="ERROR",
        message="invalid syntax. Perhaps you forgot a comma?",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == ","

    from codeguardian.candidate import build_candidate

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == 'values = {"a": 1, "b": 2}\n'


def test_missing_comma_repairs_set(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "values = {1 2, 3}\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=11,
        severity="ERROR",
        message="invalid syntax. Perhaps you forgot a comma?",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == ","


def test_missing_comma_rejects_invalid_column(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "values = [1 2, 3]\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=999,
        severity="ERROR",
        message="invalid syntax. Perhaps you forgot a comma?",
    )

    with pytest.raises(ValueError, match="outside the source"):
        RepairPlanner().plan(item)


def test_unclosed_delimiter_rejects_mismatched_diagnostic(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "values = [1, 2, 3\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=10,
        severity="ERROR",
        message="'(' was never closed",
    )

    with pytest.raises(
        ValueError,
        match="does not match the parser diagnostic",
    ):
        RepairPlanner().plan(item)


def test_unclosed_delimiter_accepts_matching_diagnostic(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "values = [1, 2, 3\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=10,
        severity="ERROR",
        message="'[' was never closed",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == "]"


def test_unexpected_indent_repairs_excess_level(tmp_path):
    from codeguardian.planner import RepairPlanner
    from codeguardian.candidate import build_candidate

    path = tmp_path / "example.py"
    path.write_text(
        "def hello():\n"
        "    print('x')\n"
        "        print('y')\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=3,
        column=8,
        severity="ERROR",
        message="unexpected indent",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == ""
    assert plan.repair.start_line == 3

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == (
        "def hello():\n"
        "    print('x')\n"
        "    print('y')\n"
    )


def test_unexpected_indent_rejects_bad_column(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "def hello():\n"
        "    print('x')\n"
        "        print('y')\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=3,
        column=999,
        severity="ERROR",
        message="unexpected indent",
    )

    with pytest.raises(
        ValueError,
        match="does not match the line indentation",
    ):
        RepairPlanner().plan(item)


def test_unexpected_indent_rejects_partial_indentation(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "def hello():\n"
        "    print('x')\n"
        "  print('y')\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=3,
        column=2,
        severity="ERROR",
        message="unexpected indent",
    )

    with pytest.raises(
        ValueError,
        match="complete indentation level",
    ):
        RepairPlanner().plan(item)


def test_expected_indented_block_repairs_function(tmp_path):
    from codeguardian.planner import RepairPlanner
    from codeguardian.candidate import build_candidate

    path = tmp_path / "example.py"
    path.write_text(
        "def hello():\n"
        "print('Hello')\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=2,
        column=1,
        severity="ERROR",
        message=(
            "expected an indented block after function definition "
            "on line 1"
        ),
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == "    "

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == (
        "def hello():\n"
        "    print('Hello')\n"
    )


def test_expected_indented_block_repairs_if(tmp_path):
    from codeguardian.planner import RepairPlanner
    from codeguardian.candidate import build_candidate

    path = tmp_path / "example.py"
    path.write_text(
        "if True:\n"
        "print('yes')\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=2,
        column=1,
        severity="ERROR",
        message=(
            "expected an indented block after 'if' statement "
            "on line 1"
        ),
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == (
        "if True:\n"
        "    print('yes')\n"
    )


def test_expected_indented_block_rejects_indented_target(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "def hello():\n"
        "  print('Hello')\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=2,
        column=1,
        severity="ERROR",
        message=(
            "expected an indented block after function definition "
            "on line 1"
        ),
    )

    with pytest.raises(
        ValueError,
        match="already indented",
    ):
        RepairPlanner().plan(item)


def test_expected_indented_block_matches_non_line_one(tmp_path):
    from codeguardian.planner import RepairPlanner
    from codeguardian.candidate import build_candidate

    path = tmp_path / "example.py"
    path.write_text(
        "x = 1\n"
        "y = 2\n"
        "\n"
        "def hello():\n"
        "print('Hello')\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=5,
        column=1,
        severity="ERROR",
        message=(
            "expected an indented block after function definition "
            "on line 4"
        ),
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == (
        "x = 1\n"
        "y = 2\n"
        "\n"
        "def hello():\n"
        "    print('Hello')\n"
    )


def test_unindent_mismatch_repairs_to_outer_level(tmp_path):
    from codeguardian.planner import RepairPlanner
    from codeguardian.candidate import build_candidate

    path = tmp_path / "example.py"
    path.write_text(
        "def hello():\n"
        "    if True:\n"
        "        print('x')\n"
        "      print('y')\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=4,
        column=17,
        severity="ERROR",
        message="unindent does not match any outer indentation level",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == (
        "def hello():\n"
        "    if True:\n"
        "        print('x')\n"
        "    print('y')\n"
    )


def test_unindent_mismatch_rejects_mixed_indentation(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "def hello():\n"
        "    print('x')\n"
        " \tprint('y')\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=3,
        column=2,
        severity="ERROR",
        message="unindent does not match any outer indentation level",
    )

    with pytest.raises(
        ValueError,
        match="mixed tab and space",
    ):
        RepairPlanner().plan(item)


def test_missing_comma_repairs_keyword_arguments(tmp_path):
    from codeguardian.planner import RepairPlanner
    from codeguardian.candidate import build_candidate

    path = tmp_path / "example.py"
    path.write_text(
        "result = hello(a=1 b=2)\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=18,
        severity="ERROR",
        message="invalid syntax. Perhaps you forgot a comma?",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.replacement == ","
    assert plan.repair.start_column == 19

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == (
        "result = hello(a=1, b=2)\n"
    )


def test_missing_comma_rejects_ambiguous_insertion(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "values = [1 2]\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=14,
        severity="ERROR",
        message="invalid syntax. Perhaps you forgot a comma?",
    )

    # Deliberately use a parser-reported location that does not
    # identify a useful token.
    with pytest.raises(ValueError):
        RepairPlanner().plan(item)


def test_repair_confidence_ordering():
    from codeguardian.confidence import RepairConfidence

    assert RepairConfidence.LOW < RepairConfidence.MEDIUM
    assert RepairConfidence.MEDIUM < RepairConfidence.HIGH


def test_parse_repair_confidence():
    from codeguardian.confidence import (
        RepairConfidence,
        parse_confidence,
    )

    assert parse_confidence("low") is RepairConfidence.LOW
    assert parse_confidence("medium") is RepairConfidence.MEDIUM
    assert parse_confidence("med") is RepairConfidence.MEDIUM
    assert parse_confidence("HIGH") is RepairConfidence.HIGH


def test_parse_repair_confidence_rejects_invalid_value():
    from codeguardian.confidence import parse_confidence

    with pytest.raises(
        ValueError,
        match="Unknown repair confidence",
    ):
        parse_confidence("certain")


def test_repair_defaults_to_high_confidence(tmp_path):
    from codeguardian.confidence import RepairConfidence
    from codeguardian.repair import Repair

    repair = Repair(
        file=tmp_path / "example.py",
        start_line=1,
        start_column=1,
        end_line=1,
        end_column=1,
        replacement="",
        reason="test repair",
    )

    assert repair.confidence is RepairConfidence.HIGH


def test_repair_accepts_explicit_confidence(tmp_path):
    from codeguardian.confidence import RepairConfidence
    from codeguardian.repair import Repair

    repair = Repair(
        file=tmp_path / "example.py",
        start_line=1,
        start_column=1,
        end_line=1,
        end_column=1,
        replacement="",
        reason="test repair",
        confidence=RepairConfidence.MEDIUM,
    )

    assert repair.confidence is RepairConfidence.MEDIUM


def test_missing_colon_has_high_confidence(tmp_path):
    from codeguardian.confidence import RepairConfidence
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "def hello()\n"
        "    pass\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=12,
        severity="ERROR",
        message="expected ':'",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.confidence is RepairConfidence.HIGH


def test_missing_comma_has_medium_confidence(tmp_path):
    from codeguardian.confidence import RepairConfidence
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(
        "values = [1 2, 3]\n",
        encoding="utf-8",
    )

    item = Diagnostic(
        file=path,
        line=1,
        column=11,
        severity="ERROR",
        message="invalid syntax. Perhaps you forgot a comma?",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.confidence is RepairConfidence.MEDIUM


@pytest.mark.parametrize(
    ("source", "column", "expected"),
    [
        (
            "x = = 1\n",
            5,
            "x =  1\n",
        ),
        (
            "values = [1,, 2]\n",
            13,
            "values = [1, 2]\n",
        ),
        (
            "if True::\n    pass\n",
            9,
            "if True:\n    pass\n",
        ),
    ],
)
def test_duplicate_punctuation_is_repaired(
    tmp_path,
    source,
    column,
    expected,
):
    from codeguardian.candidate import build_candidate
    from codeguardian.confidence import RepairConfidence
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    item = Diagnostic(
        file=path,
        line=1,
        column=column,
        severity="ERROR",
        message="invalid syntax",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.confidence is RepairConfidence.HIGH

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == expected


@pytest.mark.parametrize(
    ("source", "column"),
    [
        ("x =\n", 4),
        ("x = 1 +\n", 8),
        ("def f():\n    return +\n", 13),
    ],
)
def test_plain_invalid_syntax_does_not_guess(
    tmp_path,
    source,
    column,
):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    line = 2 if source.startswith("def ") else 1

    item = Diagnostic(
        file=path,
        line=line,
        column=column,
        severity="ERROR",
        message="invalid syntax",
    )

    with pytest.raises(ValueError):
        RepairPlanner().plan(item)


@pytest.mark.parametrize(
    "source",
    [
        "result = left == right\n",
        "result = left != right\n",
        "result = left <= right\n",
        "result = left >= right\n",
        "result = left // right\n",
        "result = left ** right\n",
        "result = left << 1\n",
        "result = left >> 1\n",
        "value += 1\n",
        "value -= 1\n",
        "value *= 2\n",
        "value /= 2\n",
        "value //= 2\n",
        "value **= 2\n",
    ],
)
def test_duplicate_punctuation_strategy_never_changes_valid_operators(
    tmp_path,
    source,
):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    item = Diagnostic(
        file=path,
        line=1,
        column=1,
        severity="ERROR",
        message="invalid syntax",
    )

    with pytest.raises(ValueError):
        RepairPlanner().plan(item)

    assert path.read_text(encoding="utf-8") == source


def test_duplicate_punctuation_rejects_nonduplicate_equals(
    tmp_path,
):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "x = 1 +\n"
    path.write_text(source, encoding="utf-8")

    item = Diagnostic(
        file=path,
        line=1,
        column=3,
        severity="ERROR",
        message="invalid syntax",
    )

    with pytest.raises(ValueError):
        RepairPlanner().plan(item)

    assert path.read_text(encoding="utf-8") == source


@pytest.mark.parametrize(
    ("source", "column", "expected"),
    [
        (
            "from os import path environ\n",
            21,
            "from os import path, environ\n",
        ),
        (
            "from os import (path environ)\n",
            22,
            "from os import (path, environ)\n",
        ),
        (
            "import os sys\n",
            11,
            "import os, sys\n",
        ),
    ],
)
def test_missing_import_comma_is_repaired(
    tmp_path,
    source,
    column,
    expected,
):
    from codeguardian.candidate import build_candidate
    from codeguardian.confidence import RepairConfidence
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    item = Diagnostic(
        file=path,
        line=1,
        column=column,
        severity="ERROR",
        message="invalid syntax",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.confidence is RepairConfidence.HIGH

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == expected


@pytest.mark.parametrize(
    ("source", "column"),
    [
        (
            "x =\n",
            4,
        ),
        (
            "x = 1 +\n",
            8,
        ),
    ],
)
def test_plain_invalid_syntax_still_refuses_ambiguous_cases(
    tmp_path,
    source,
    column,
):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    item = Diagnostic(
        file=path,
        line=1,
        column=column,
        severity="ERROR",
        message="invalid syntax",
    )

    with pytest.raises(ValueError):
        RepairPlanner().plan(item)

    assert path.read_text(encoding="utf-8") == source


@pytest.mark.parametrize(
    ("source", "column", "expected"),
    [
        (
            "def f(a b):\n    pass\n",
            9,
            "def f(a, b):\n    pass\n",
        ),
        (
            "async def f(a b):\n    pass\n",
            15,
            "async def f(a, b):\n    pass\n",
        ),
        (
            "def f(a b, /):\n    pass\n",
            9,
            "def f(a, b, /):\n    pass\n",
        ),
        (
            "def f(*, a b):\n    pass\n",
            12,
            "def f(*, a, b):\n    pass\n",
        ),
    ],
)
def test_missing_bare_parameter_comma_is_repaired(
    tmp_path,
    source,
    column,
    expected,
):
    from codeguardian.candidate import build_candidate
    from codeguardian.confidence import RepairConfidence
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    item = Diagnostic(
        file=path,
        line=1,
        column=column,
        severity="ERROR",
        message="invalid syntax",
    )

    plan = RepairPlanner().plan(item)

    assert plan is not None
    assert plan.repair.confidence is RepairConfidence.HIGH

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == expected


def test_plain_invalid_syntax_does_not_repair_lambda_parameters(
    tmp_path,
):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "f = lambda a b: a + b\n"
    path.write_text(source, encoding="utf-8")

    item = Diagnostic(
        file=path,
        line=1,
        column=14,
        severity="ERROR",
        message="invalid syntax",
    )

    with pytest.raises(ValueError):
        RepairPlanner().plan(item)

    assert path.read_text(encoding="utf-8") == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "def f(a=1 b=2):\n    return a + b\n",
            "def f(a=1, b=2):\n    return a + b\n",
        ),
        (
            "def f(a: int b: str):\n    return a\n",
            "def f(a: int, b: str):\n    return a\n",
        ),
        (
            "def f(a=1 b=2, *, c=3):\n    return a + b + c\n",
            "def f(a=1, b=2, *, c=3):\n    return a + b + c\n",
        ),
    ],
)
def test_existing_missing_comma_strategy_handles_parameters(
    tmp_path,
    source,
    expected,
):
    from codeguardian.candidate import build_candidate
    from codeguardian.planner import RepairPlanner
    from codeguardian.scanner import PythonScanner

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    diagnostics = PythonScanner(tmp_path).check_file(path)

    assert len(diagnostics) == 1
    assert diagnostics[0].message == (
        "invalid syntax. Perhaps you forgot a comma?"
    )

    plan = RepairPlanner().plan(diagnostics[0])

    assert plan is not None

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            'value = {"a": 1 "b": 2}\n',
            'value = {"a": 1, "b": 2}\n',
        ),
        (
            'value = {"a": f() "b": g()}\n',
            'value = {"a": f(), "b": g()}\n',
        ),
        (
            'value = {"a": x + 1 "b": y + 2}\n',
            'value = {"a": x + 1, "b": y + 2}\n',
        ),
        (
            'value = {"a": {"x": 1} "b": {"y": 2}}\n',
            'value = {"a": {"x": 1}, "b": {"y": 2}}\n',
        ),
    ],
)
def test_existing_missing_comma_strategy_handles_dict_items(
    tmp_path,
    source,
    expected,
):
    from codeguardian.candidate import build_candidate
    from codeguardian.planner import RepairPlanner
    from codeguardian.scanner import PythonScanner

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    diagnostics = PythonScanner(tmp_path).check_file(path)

    assert len(diagnostics) == 1
    assert diagnostics[0].message == (
        "invalid syntax. Perhaps you forgot a comma?"
    )

    plan = RepairPlanner().plan(diagnostics[0])

    assert plan is not None

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            'value = {"a": foo "b": bar}\n',
            'value = {"a": foo, "b": bar}\n',
        ),
        (
            'value = {"a": "x" "b": "y"}\n',
            'value = {"a": "x", "b": "y"}\n',
        ),
    ],
)
def test_plain_invalid_syntax_missing_dict_comma(
    tmp_path,
    source,
    expected,
):
    from codeguardian.candidate import build_candidate
    from codeguardian.planner import RepairPlanner
    from codeguardian.scanner import PythonScanner

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    diagnostics = PythonScanner(tmp_path).check_file(path)

    assert len(diagnostics) == 1
    assert diagnostics[0].message == "invalid syntax"

    plan = RepairPlanner().plan(diagnostics[0])

    assert plan is not None

    candidate = build_candidate(
        path.read_text(encoding="utf-8"),
        plan.repair,
    )

    assert candidate.proposed_source == expected


def test_valid_dict_unpack_is_not_diagnosed(tmp_path):
    from codeguardian.scanner import PythonScanner

    path = tmp_path / "example.py"
    path.write_text(
        "value = {**left **right}\n",
        encoding="utf-8",
    )

    diagnostics = PythonScanner(tmp_path).check_file(path)

    assert diagnostics == []


@pytest.mark.parametrize(
    "source",
    [
        "value = {x for x in items}\n",
        "value = {x: y for x, y in items}\n",
        "value = {**left, **right}\n",
        "value = {1, 2, 3}\n",
        'value = {"a": lambda x: x}\n',
        'value = {"a": (lambda x: x)}\n',
    ],
)
def test_valid_brace_constructs_are_not_diagnosed(
    tmp_path,
    source,
):
    from codeguardian.scanner import PythonScanner

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    diagnostics = PythonScanner(tmp_path).check_file(path)

    assert diagnostics == []


@pytest.mark.parametrize(
    ("source", "column"),
    [
        (
            "value = {foo bar}\n",
            14,
        ),
        (
            "value = {x for y z in items}\n",
            18,
        ),
    ],
)
def test_dict_comma_strategy_refuses_non_dict_brace_syntax(
    tmp_path,
    source,
    column,
):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    item = Diagnostic(
        file=path,
        line=1,
        column=column,
        severity="ERROR",
        message="invalid syntax",
    )

    with pytest.raises(ValueError):
        RepairPlanner().plan(item)

    assert path.read_text(encoding="utf-8") == source


def test_missing_comma_accepts_candidate_inside_parser_span(tmp_path):
    from codeguardian.planner import RepairPlanner
    from codeguardian.scanner import PythonScanner

    path = tmp_path / "example.py"
    path.write_text(
        "values = [1 2, 3]\n",
        encoding="utf-8",
    )

    diagnostics = PythonScanner(tmp_path).check_file(path)

    assert len(diagnostics) == 1

    diagnostic = diagnostics[0]

    assert diagnostic.end_line == 1
    assert diagnostic.end_column is not None

    plan = RepairPlanner().plan(diagnostic)

    assert plan.repair.start_line == 1
    assert (
        diagnostic.column
        <= plan.repair.start_column
        <= diagnostic.end_column
    )


def test_missing_comma_refuses_candidate_outside_parser_span(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "values = [1 2, 3]\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=14,
        severity="ERROR",
        message="invalid syntax. Perhaps you forgot a comma?",
        end_line=1,
        end_column=14,
    )

    with pytest.raises(ValueError):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "value = (1 + 2))\n",
            "value = (1 + 2)\n",
        ),
        (
            "value = [1, 2]]\n",
            "value = [1, 2]\n",
        ),
        (
            'value = {"a": 1}}\n',
            'value = {"a": 1}\n',
        ),
        (
            "result = f(1, 2))\n",
            "result = f(1, 2)\n",
        ),
        (
            "value = ((1 + 2)))\n",
            "value = ((1 + 2))\n",
        ),
        (
            "value = [1, [2, 3]]]\n",
            "value = [1, [2, 3]]\n",
        ),
        (
            "if (x > 1)):\n    pass\n",
            "if (x > 1):\n    pass\n",
        ),
    ],
)
def test_unmatched_closing_delimiter_repair(
    tmp_path,
    source,
    expected,
):
    from codeguardian.candidate import build_candidate
    from codeguardian.confidence import RepairConfidence
    from codeguardian.planner import RepairPlanner
    from codeguardian.scanner import PythonScanner

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    diagnostics = PythonScanner(tmp_path).check_file(path)

    assert len(diagnostics) == 1

    plan = RepairPlanner().plan(diagnostics[0])

    assert plan is not None
    assert plan.repair.confidence == RepairConfidence.HIGH

    candidate = build_candidate(source, plan.repair)

    assert candidate.proposed_source == expected


@pytest.mark.parametrize(
    "source",
    [
        "value = (1 + 2]\n",
        "value = [1, 2)\n",
        'value = {"a": 1]\n',
    ],
)
def test_mismatched_closing_delimiter_is_not_auto_repaired(
    tmp_path,
    source,
):
    from codeguardian.planner import RepairPlanner
    from codeguardian.scanner import PythonScanner

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    diagnostics = PythonScanner(tmp_path).check_file(path)

    assert len(diagnostics) == 1
    assert diagnostics[0].message.startswith(
        "closing parenthesis "
    )

    assert RepairPlanner().plan(diagnostics[0]) is None
    assert path.read_text(encoding="utf-8") == source


def test_unmatched_closing_delimiter_refuses_wrong_column(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "value = (1 + 2))\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=15,
        severity="ERROR",
        message="unmatched ')'",
        end_line=1,
        end_column=15,
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_unmatched_closing_delimiter_refuses_wide_span(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "value = (1 + 2))\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=16,
        severity="ERROR",
        message="unmatched ')'",
        end_line=1,
        end_column=17,
    )

    with pytest.raises(
        ValueError,
        match="span does not identify exactly one",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_unmatched_closing_delimiter_refuses_stale_message(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "value = [1, 2]]\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=15,
        severity="ERROR",
        message="unmatched ')'",
        end_line=1,
        end_column=15,
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_unmatched_closing_delimiter_refuses_stale_line(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "\nvalue = (1 + 2))\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=16,
        severity="ERROR",
        message="unmatched ')'",
        end_line=1,
        end_column=16,
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_unmatched_closing_delimiter_refuses_now_valid_source(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "value = (1 + 2)\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=15,
        severity="ERROR",
        message="unmatched ')'",
        end_line=1,
        end_column=15,
    )

    with pytest.raises(
        ValueError,
        match="no longer has the reported syntax error",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_unclosed_delimiter_refuses_stale_column(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "values = [1, 2, 3\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=11,
        severity="ERROR",
        message="'[' was never closed",
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_unclosed_delimiter_refuses_stale_message(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "values = [1, 2, 3\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=10,
        severity="ERROR",
        message="'(' was never closed",
    )

    with pytest.raises(
        ValueError,
        match="does not match the parser diagnostic",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_unclosed_delimiter_refuses_now_valid_source(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "values = [1, 2, 3]\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=10,
        severity="ERROR",
        message="'[' was never closed",
    )

    with pytest.raises(
        ValueError,
        match="exactly one unmatched opening delimiter",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source



def test_missing_colon_refuses_stale_column(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "if True\n"
        "    print('yes')\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=7,
        severity="ERROR",
        message="expected ':'",
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_missing_colon_refuses_stale_line(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "\n"
        "if True\n"
        "    print('yes')\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=8,
        severity="ERROR",
        message="expected ':'",
    )

    with pytest.raises(
        ValueError,
        match="Refusing to add",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_missing_colon_refuses_now_valid_source(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "if True:\n"
        "    print('yes')\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=8,
        severity="ERROR",
        message="expected ':'",
    )

    with pytest.raises(
        ValueError,
        match="already ends with",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_unexpected_indent_refuses_stale_message(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "def hello():\n"
        "    print('x')\n"
        "        print('y')\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=3,
        column=8,
        severity="ERROR",
        message="invalid syntax",
    )

    # Call the strategy directly because "invalid syntax" normally
    # dispatches through the plain-invalid-syntax strategy family.
    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner()._unexpected_indent(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_unexpected_indent_refuses_stale_line(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "\n"
        "def hello():\n"
        "    print('x')\n"
        "        print('y')\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=3,
        column=4,
        severity="ERROR",
        message="unexpected indent",
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_unexpected_indent_refuses_now_valid_source(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "def hello():\n"
        "    print('x')\n"
        "    print('y')\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=3,
        column=4,
        severity="ERROR",
        message="unexpected indent",
    )

    # Structural validation should reject this before any edit because
    # the current indentation is legitimate for this function body.
    with pytest.raises(ValueError):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_unexpected_indent_fresh_parser_guard_preserves_valid_source(
    tmp_path,
):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "if True:\n"
        "    if True:\n"
        "        print('valid')\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=3,
        column=8,
        severity="ERROR",
        message="unexpected indent",
    )

    with pytest.raises(
        ValueError,
        match="no longer has the reported syntax error",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source



def test_unindent_mismatch_refuses_stale_column(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "def hello():\n"
        "    if True:\n"
        "        print('x')\n"
        "      print('y')\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=4,
        column=7,
        severity="ERROR",
        message="unindent does not match any outer indentation level",
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_unindent_mismatch_refuses_stale_message(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "def hello():\n"
        "    if True:\n"
        "        print('x')\n"
        "      print('y')\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=4,
        column=17,
        severity="ERROR",
        message="unexpected indent",
    )

    # Call directly because this message normally dispatches to the
    # unexpected-indent strategy.
    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner()._unindent_mismatch(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_unindent_mismatch_refuses_valid_source(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "def hello():\n"
        "    if True:\n"
        "        print('x')\n"
        "    print('y')\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=4,
        column=17,
        severity="ERROR",
        message="unindent does not match any outer indentation level",
    )

    with pytest.raises(ValueError):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_expected_indented_block_refuses_stale_column(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "def hello():\n"
        "print('Hello')\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=2,
        column=2,
        severity="ERROR",
        message=(
            "expected an indented block after function definition "
            "on line 1"
        ),
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_expected_indented_block_refuses_stale_message(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "def hello():\n"
        "print('Hello')\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=2,
        column=1,
        severity="ERROR",
        message=(
            "expected an indented block after class definition "
            "on line 1"
        ),
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_expected_indented_block_refuses_valid_source(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "def hello():\n"
        "    print('Hello')\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=2,
        column=1,
        severity="ERROR",
        message=(
            "expected an indented block after function definition "
            "on line 1"
        ),
    )

    # Existing structural validation should reject an already
    # indented target before any edit is attempted.
    with pytest.raises(
        ValueError,
        match="already indented",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_expected_indented_block_refuses_forged_valid_top_level(
    tmp_path,
):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "x = 1\n"
        "print('Hello')\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=2,
        column=1,
        severity="ERROR",
        message=(
            "expected an indented block after function definition "
            "on line 1"
        ),
    )

    # The preceding line is not a block header, so structural
    # validation must refuse the forged diagnostic.
    with pytest.raises(
        ValueError,
        match="not a recognized block header",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_missing_comma_refuses_stale_column(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "values = [1 2, 3]\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=12,
        severity="ERROR",
        message="invalid syntax. Perhaps you forgot a comma?",
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_missing_comma_refuses_stale_message(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "values = [1 2, 3]\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=11,
        severity="ERROR",
        message="invalid syntax",
    )

    # Call directly because plain "invalid syntax" normally dispatches
    # through the plain-invalid-syntax strategy family.
    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner()._missing_comma(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_missing_comma_refuses_stale_parser_span(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "values = [1 2, 3]\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=11,
        severity="ERROR",
        message="invalid syntax. Perhaps you forgot a comma?",
        end_line=1,
        end_column=15,
    )

    with pytest.raises(
        ValueError,
        match="span does not match the current source",
    ):
        RepairPlanner().plan(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_missing_comma_accepts_fresh_parser_span(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "values = [1 2, 3]\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=11,
        severity="ERROR",
        message="invalid syntax. Perhaps you forgot a comma?",
        end_line=1,
        end_column=14,
    )

    plan = RepairPlanner().plan(diagnostic)

    assert plan is not None
    assert plan.repair.replacement == ","
    assert path.read_text(encoding="utf-8") == source


def test_duplicate_punctuation_refuses_stale_column(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "values = [1,, 2]\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=12,
        severity="ERROR",
        message="invalid syntax",
    )

    with pytest.raises(ValueError):
        RepairPlanner()._duplicate_punctuation(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_duplicate_punctuation_refuses_stale_message(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "values = [1,, 2]\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=13,
        severity="ERROR",
        message="unexpected indent",
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner()._duplicate_punctuation(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_duplicate_punctuation_refuses_stale_span(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "values = [1,, 2]\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=13,
        severity="ERROR",
        message="invalid syntax",
        end_line=1,
        end_column=15,
    )

    with pytest.raises(
        ValueError,
        match="span does not match the current source",
    ):
        RepairPlanner()._duplicate_punctuation(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_duplicate_punctuation_accepts_fresh_span(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "values = [1,, 2]\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=13,
        severity="ERROR",
        message="invalid syntax",
        end_line=1,
        end_column=14,
    )

    plan = RepairPlanner()._duplicate_punctuation(diagnostic)

    assert plan is not None
    assert plan.repair.replacement == ""
    assert plan.repair.start_column == 13
    assert plan.repair.end_column == 14
    assert path.read_text(encoding="utf-8") == source


def test_duplicate_punctuation_refuses_valid_source_directly(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "result = left == right\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=15,
        severity="ERROR",
        message="invalid syntax",
    )

    with pytest.raises(
        ValueError,
        match="Current source no longer has",
    ):
        RepairPlanner()._duplicate_punctuation(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_missing_parameter_comma_refuses_stale_column(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "def hello(first second):\n"
        "    pass\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=18,
        severity="ERROR",
        message="invalid syntax",
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner()._missing_parameter_comma(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_missing_parameter_comma_refuses_stale_span(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "def hello(first second):\n"
        "    pass\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=17,
        severity="ERROR",
        message="invalid syntax",
        end_line=1,
        end_column=24,
    )

    with pytest.raises(
        ValueError,
        match="span does not match the current source",
    ):
        RepairPlanner()._missing_parameter_comma(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_missing_parameter_comma_accepts_fresh_span(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = (
        "def hello(first second):\n"
        "    pass\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=17,
        severity="ERROR",
        message="invalid syntax",
        end_line=1,
        end_column=23,
    )

    plan = RepairPlanner()._missing_parameter_comma(diagnostic)

    assert plan is not None
    assert plan.repair.replacement == ", "
    assert path.read_text(encoding="utf-8") == source


def test_missing_import_comma_refuses_stale_column(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "from os import path environ\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=22,
        severity="ERROR",
        message="invalid syntax",
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner()._missing_import_comma(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_missing_import_comma_refuses_stale_span(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "from os import path environ\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=21,
        severity="ERROR",
        message="invalid syntax",
        end_line=1,
        end_column=29,
    )

    with pytest.raises(
        ValueError,
        match="span does not match the current source",
    ):
        RepairPlanner()._missing_import_comma(diagnostic)

    assert path.read_text(encoding="utf-8") == source


def test_missing_import_comma_accepts_fresh_span(tmp_path):
    from codeguardian.planner import RepairPlanner

    path = tmp_path / "example.py"
    source = "from os import path environ\n"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=21,
        severity="ERROR",
        message="invalid syntax",
        end_line=1,
        end_column=28,
    )

    plan = RepairPlanner()._missing_import_comma(diagnostic)

    assert plan is not None
    assert plan.repair.replacement == ", "
    assert path.read_text(encoding="utf-8") == source


def test_missing_dict_comma_refuses_stale_column(tmp_path):
    from codeguardian.planner import RepairPlanner
    from codeguardian.scanner import Diagnostic

    path = tmp_path / "example.py"
    source = 'value = {"a": foo "b": bar}\n'
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=999,
        severity="ERROR",
        message="invalid syntax",
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner()._missing_dict_comma(diagnostic)


def test_missing_dict_comma_refuses_stale_message(tmp_path):
    from codeguardian.planner import RepairPlanner
    from codeguardian.scanner import Diagnostic

    path = tmp_path / "example.py"
    source = 'value = {"a": foo "b": bar}\n'
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=15,
        severity="ERROR",
        message="invalid syntax. Perhaps you forgot a comma?",
    )

    with pytest.raises(
        ValueError,
        match="does not match the current source location",
    ):
        RepairPlanner().plan(diagnostic)


def test_missing_dict_comma_refuses_ambiguous_candidate(tmp_path):
    from codeguardian.planner import RepairPlanner
    from codeguardian.scanner import Diagnostic

    path = tmp_path / "example.py"
    source = (
        'value = {"a": foo "b": bar "c": baz}\n'
    )
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=15,
        severity="ERROR",
        message="invalid syntax",
    )

    with pytest.raises(ValueError):
        RepairPlanner().plan(diagnostic)
