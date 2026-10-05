from pathlib import Path

from codeguardian.ai.edit import AIEdit
from codeguardian.ai.engine import (
    AIEditEvaluation,
    apply_evaluated_ai_edit,
    evaluate_ai_edit,
    evaluate_ai_edits,
)
from codeguardian.analyzer import analyze_file


def diagnostic_for(tmp_path, source, text):
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")

    diagnostics = analyze_file(path).diagnostics

    return next(
        item
        for item in diagnostics
        if text in item.message
    )


def test_engine_accepts_single_line_improvement(tmp_path):
    source = (
        "def example():\n"
        "    definitely_not_defined\n"
        "    return 1\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Undefined name",
    )

    def provider(**kwargs):
        assert kwargs["line"] == 2
        return AIEdit("delete", 2)

    result = evaluate_ai_edit(
        source=source,
        diagnostic=diagnostic,
        model="test-model",
        provider=provider,
    )

    assert result.accepted is True
    assert result.candidate_source == (
        "def example():\n"
        "    return 1\n"
    )
    assert len(result.after_diagnostics) < len(
        result.before_diagnostics
    )


def test_engine_rejects_invalid_syntax(tmp_path):
    source = (
        "def example():\n"
        "    definitely_not_defined\n"
        "    return 1\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Undefined name",
    )

    def provider(**kwargs):
        return AIEdit("replace", 2, "    if True:")

    result = evaluate_ai_edit(
        source=source,
        diagnostic=diagnostic,
        model="test-model",
        provider=provider,
    )

    assert result.accepted is False
    assert result.candidate_source is None
    assert "syntax" in result.reason


def test_engine_rejects_new_error(tmp_path):
    source = (
        "def example():\n"
        "    unused = 1\n"
        "    return 1\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused definition",
    )

    def provider(**kwargs):
        return AIEdit(
            "replace",
            2,
            "    another_undefined_name",
        )

    result = evaluate_ai_edit(
        source=source,
        diagnostic=diagnostic,
        model="test-model",
        provider=provider,
    )

    assert result.accepted is False
    assert result.candidate_source is None
    assert "new error" in result.reason


def test_engine_rejects_no_improvement(tmp_path):
    source = (
        "def example():\n"
        "    unused = 1\n"
        "    return 1\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused definition",
    )

    def provider(**kwargs):
        return AIEdit(
            "replace",
            2,
            "    another_unused = 1",
        )

    result = evaluate_ai_edit(
        source=source,
        diagnostic=diagnostic,
        model="test-model",
        provider=provider,
    )

    assert result.accepted is False
    assert result.candidate_source is None
    assert (
        "not repaired" in result.reason
        or "does not reduce" in result.reason
    )

def test_transaction_applies_accepted_candidate(tmp_path):
    path = tmp_path / "target.py"
    original = (
        "def example():\n"
        "    definitely_not_defined\n"
        "    return 1\n"
    )
    candidate = (
        "def example():\n"
        "    return 1\n"
    )
    path.write_text(original, encoding="utf-8")

    evaluation = AIEditEvaluation(
        accepted=True,
        reason="candidate safely reduces diagnostics",
        edit=AIEdit("delete", 2),
        candidate_source=candidate,
        before_diagnostics=[],
        after_diagnostics=[],
    )

    result = apply_evaluated_ai_edit(path, evaluation)

    assert result.applied is True
    assert path.read_text(encoding="utf-8") == candidate
    assert result.backup is not None
    assert result.backup.exists()
    assert result.backup.read_text(encoding="utf-8") == original


def test_transaction_refuses_rejected_candidate(tmp_path):
    path = tmp_path / "target.py"
    original = "value = 1\n"
    path.write_text(original, encoding="utf-8")

    evaluation = AIEditEvaluation(
        accepted=False,
        reason="rejected",
        edit=AIEdit("delete", 1),
        candidate_source=None,
        before_diagnostics=[],
        after_diagnostics=[],
    )

    result = apply_evaluated_ai_edit(path, evaluation)

    assert result.applied is False
    assert result.backup is None
    assert path.read_text(encoding="utf-8") == original


def test_transaction_allows_unrelated_existing_error(
    tmp_path,
):
    path = tmp_path / "target.py"
    original = (
        "def example():\n"
        "    unused = 1\n"
        "    definitely_not_defined\n"
        "    return 1\n"
    )
    path.write_text(original, encoding="utf-8")

    before = analyze_file(path).diagnostics
    unused = next(
        item
        for item in before
        if "Unused definition" in item.message
    )

    def provider(**kwargs):
        return AIEdit("delete", unused.line)

    evaluation = evaluate_ai_edit(
        source=original,
        diagnostic=unused,
        model="test-model",
        provider=provider,
    )

    assert evaluation.accepted is True
    assert any(
        item.severity == "ERROR"
        for item in evaluation.after_diagnostics
    )

    result = apply_evaluated_ai_edit(
        path,
        evaluation,
    )

    assert result.applied is True
    assert "unused = 1" not in path.read_text(
        encoding="utf-8"
    )


def test_transaction_rolls_back_if_written_candidate_changes(
    tmp_path,
    monkeypatch,
):
    import codeguardian.ai.engine as engine

    path = tmp_path / "target.py"
    original = (
        "def example():\n"
        "    unused = 1\n"
        "    return 1\n"
    )
    path.write_text(original, encoding="utf-8")

    diagnostic = next(
        item
        for item in analyze_file(path).diagnostics
        if "Unused definition" in item.message
    )

    def provider(**kwargs):
        return AIEdit("delete", diagnostic.line)

    evaluation = evaluate_ai_edit(
        source=original,
        diagnostic=diagnostic,
        model="test-model",
        provider=provider,
    )

    real_analyze = engine.analyze_file
    calls = {"count": 0}

    def changed_analysis(target):
        calls["count"] += 1
        result = real_analyze(target)

        if calls["count"] == 1:
            target.write_text(
                target.read_text(encoding="utf-8")
                + "new_undefined_name\n",
                encoding="utf-8",
            )
            return real_analyze(target)

        return result

    monkeypatch.setattr(
        engine,
        "analyze_file",
        changed_analysis,
    )

    result = apply_evaluated_ai_edit(
        path,
        evaluation,
    )

    assert result.applied is False
    assert "rolled back" in result.reason
    assert path.read_text(encoding="utf-8") == original

def test_engine_rejects_single_edit_mutable_default(tmp_path):
    source = (
        "def example(items=[]):\n"
        "    return len(items)\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    called = False

    def provider(**kwargs):
        nonlocal called
        called = True
        return AIEdit(
            "replace",
            1,
            "def example(items=None):",
        )

    result = evaluate_ai_edit(
        source=source,
        diagnostic=diagnostic,
        model="test-model",
        provider=provider,
    )

    assert result.accepted is False
    assert result.edit is None
    assert called is False
    assert "multi-edit" in result.reason

def test_engine_accepts_safe_multi_edit_mutable_default(tmp_path):
    source = (
        "def example(items=[]):\n"
        "    return len(items)\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    def provider(**kwargs):
        assert kwargs["line"] == 1
        assert kwargs["max_edits"] == 3
        return [
            AIEdit(
                "replace",
                1,
                "def example(items=None):",
            ),
            AIEdit(
                "insert_after",
                1,
                (
                    "    if items is None:\n"
                    "        items = []"
                ),
            ),
        ]

    result = evaluate_ai_edits(
        source=source,
        diagnostic=diagnostic,
        model="test-model",
        provider=provider,
    )

    assert result.accepted is True
    assert len(result.edits) == 2
    assert result.candidate_source == (
        "def example(items=None):\n"
        "    if items is None:\n"
        "        items = []\n"
        "    return len(items)\n"
    )


def test_engine_multi_edit_rejects_outside_window(tmp_path):
    source = "".join(
        [
            "value_1 = 1\n",
            "value_2 = 2\n",
            "value_3 = 3\n",
            "value_4 = 4\n",
            "value_5 = 5\n",
            "value_6 = 6\n",
            "value_7 = 7\n",
            "value_8 = 8\n",
            "def example(items=[]):\n",
            "    return len(items)\n",
        ]
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    def provider(**kwargs):
        return [
            AIEdit(
                "replace",
                1,
                "value_1 = 999",
            ),
            AIEdit(
                "replace",
                9,
                "def example(items=None):",
            ),
        ]

    result = evaluate_ai_edits(
        source=source,
        diagnostic=diagnostic,
        model="test-model",
        provider=provider,
        context_lines=2,
    )

    assert result.accepted is False
    assert result.candidate_source is None
    assert "authorized source window" in result.reason


def test_engine_multi_edit_rejects_new_error(tmp_path):
    source = (
        "def example(items=[]):\n"
        "    return len(items)\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    def provider(**kwargs):
        return [
            AIEdit(
                "replace",
                1,
                "def example(items=None):",
            ),
            AIEdit(
                "insert_after",
                1,
                "    definitely_not_defined",
            ),
        ]

    result = evaluate_ai_edits(
        source=source,
        diagnostic=diagnostic,
        model="test-model",
        provider=provider,
    )

    assert result.accepted is False
    assert result.candidate_source is None
    assert (
        "new error" in result.reason
        or "not repaired" in result.reason
    )


def test_engine_multi_edit_rejects_empty_transaction(tmp_path):
    source = (
        "def example(items=[]):\n"
        "    return len(items)\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    def provider(**kwargs):
        return []

    result = evaluate_ai_edits(
        source=source,
        diagnostic=diagnostic,
        model="test-model",
        provider=provider,
    )

    assert result.accepted is False
    assert result.candidate_source is None
    assert "empty" in result.reason



def test_engine_rejects_exposing_if_false_body(tmp_path):
    source = (
        "def example(items=None):\n"
        "    if items is None:\n"
        "        items = []\n"
        "    if False:\n"
        "        print('unreachable')\n"
        "    return len(items)\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "if False",
    )

    def provider(**kwargs):
        return AIEdit("delete", 4)

    result = evaluate_ai_edit(
        source=source,
        diagnostic=diagnostic,
        model="test-model",
        provider=provider,
    )

    assert result.accepted is False
    assert result.candidate_source is None
    assert "exposes code" in result.reason


def test_engine_allows_deleting_entire_if_false_block(
    tmp_path,
):
    source = (
        "def example():\n"
        "    if False:\n"
        "        print('unreachable')\n"
        "    return 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "if False",
    )

    def provider(**kwargs):
        return AIEdit(
            "replace",
            2,
            "    return 1",
        )

    result = evaluate_ai_edit(
        source=source,
        diagnostic=diagnostic,
        model="test-model",
        provider=provider,
    )

    # A single-line replacement cannot safely remove both
    # the guard and its indented body, so syntax validation
    # must reject this candidate rather than expose the body.
    assert result.accepted is False


def test_deterministic_duplicate_import_edit_deletes_second_import(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_duplicate_import_edit,
    )

    source = (
        "import os\n"
        "import os\n"
        "\n"
        "print(os.getcwd())\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Duplicate import",
    )

    edit = deterministic_duplicate_import_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "delete"
    assert edit.line == 2


def test_deterministic_duplicate_import_edit_rejects_mixed_import(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_duplicate_import_edit,
    )

    source = (
        "import os\n"
        "import os, sys\n"
        "\n"
        "print(os.getcwd(), sys.version)\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Duplicate import",
    )

    edit = deterministic_duplicate_import_edit(
        source,
        diagnostic,
    )

    assert edit is None


def test_deterministic_duplicate_from_import_edit_is_safe(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_duplicate_import_edit,
    )

    source = (
        "from pathlib import Path\n"
        "from pathlib import Path\n"
        "\n"
        "print(Path('.'))\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Duplicate import",
    )

    edit = deterministic_duplicate_import_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "delete"
    assert edit.line == 2


def test_duplicate_import_evaluation_accepts_one_occurrence_reduction(
    tmp_path,
):
    from codeguardian.ai.edit import AIEdit
    from codeguardian.ai.engine import evaluate_ai_edit

    source = (
        "import os\n"
        "import os\n"
        "import os\n"
        "\n"
        "print(os.getcwd())\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Duplicate import",
    )

    evaluation = evaluate_ai_edit(
        source=source,
        diagnostic=diagnostic,
        model="unused",
        provider=lambda **kwargs: AIEdit(
            operation="delete",
            line=diagnostic.line,
        ),
    )

    assert evaluation.accepted
    assert evaluation.candidate_source is not None

    before_matching = [
        item
        for item in evaluation.before_diagnostics
        if item.message == diagnostic.message
    ]
    after_matching = [
        item
        for item in evaluation.after_diagnostics
        if item.message == diagnostic.message
    ]

    assert len(before_matching) == 2
    assert len(after_matching) == 1


def test_deterministic_unused_definition_deletes_literal_assignment(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_definition_edit,
    )

    source = (
        "def example():\n"
        "    unused_value = 123\n"
        "    return 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused definition: 'unused_value'",
    )

    edit = deterministic_unused_definition_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "delete"
    assert edit.line == 2


def test_deterministic_unused_definition_rejects_call_rhs(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_definition_edit,
    )

    source = (
        "def side_effect():\n"
        "    print('called')\n"
        "    return 1\n"
        "\n"
        "def example():\n"
        "    unused_value = side_effect()\n"
        "    return 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused definition: 'unused_value'",
    )

    assert (
        deterministic_unused_definition_edit(
            source,
            diagnostic,
        )
        is None
    )


def test_deterministic_unused_definition_rejects_attribute_access_rhs(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_definition_edit,
    )

    source = (
        "def example(obj):\n"
        "    unused_value = obj.value\n"
        "    return 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused definition: 'unused_value'",
    )

    assert (
        deterministic_unused_definition_edit(
            source,
            diagnostic,
        )
        is None
    )


def test_deterministic_unused_definition_rejects_unpacking(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_definition_edit,
    )

    source = (
        "def example():\n"
        "    first, second = (1, 2)\n"
        "    return 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused definition: 'first'",
    )

    assert (
        deterministic_unused_definition_edit(
            source,
            diagnostic,
        )
        is None
    )


def test_deterministic_unused_import_deletes_simple_import(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_import_edit,
    )

    source = (
        "import os\n"
        "\n"
        "value = 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused import: 'os'",
    )

    edit = deterministic_unused_import_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "delete"
    assert edit.line == 1


def test_deterministic_unused_import_deletes_simple_from_import(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_import_edit,
    )

    source = (
        "from pathlib import Path\n"
        "\n"
        "value = 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused import: 'Path'",
    )

    edit = deterministic_unused_import_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "delete"
    assert edit.line == 1


def test_deterministic_unused_import_supports_alias(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_import_edit,
    )

    source = (
        "import os as operating_system\n"
        "\n"
        "value = 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused import: 'operating_system'",
    )

    edit = deterministic_unused_import_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "delete"
    assert edit.line == 1


def test_deterministic_unused_import_rejects_mixed_import(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_import_edit,
    )

    source = (
        "import os, sys\n"
        "\n"
        "print(sys.version)\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused import: 'os'",
    )

    assert (
        deterministic_unused_import_edit(
            source,
            diagnostic,
        )
        is None
    )


def test_deterministic_unused_import_rejects_mixed_from_import(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_import_edit,
    )

    source = (
        "from pathlib import Path, PurePath\n"
        "\n"
        "print(PurePath('.'))\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused import: 'Path'",
    )

    assert (
        deterministic_unused_import_edit(
            source,
            diagnostic,
        )
        is None
    )


def test_deterministic_unused_import_not_offered_for_all_reexport(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_import_edit,
    )
    from codeguardian.analyzer import analyze_file

    path = tmp_path / "exports.py"
    source = (
        "from pathlib import Path\n"
        "__all__ = ['Path']\n"
    )
    path.write_text(source, encoding="utf-8")

    diagnostics = analyze_file(path).diagnostics

    unused = [
        item
        for item in diagnostics
        if item.message == "Unused import: 'Path'"
    ]

    assert unused == []

    assert deterministic_unused_import_edit(
        source,
        type(
            "FakeDiagnostic",
            (),
            {
                "message": "Unused import: 'Path'",
                "line": 2,
            },
        )(),
    ) is None


def test_same_alias_different_modules_are_not_duplicate_imports(
    tmp_path,
):
    source = (
        "import os as value\n"
        "import sys as value\n"
        "\n"
        "print(value.version)\n"
    )

    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")

    diagnostics = analyze_file(path).diagnostics

    assert not any(
        diagnostic.message == "Duplicate import: 'value'"
        for diagnostic in diagnostics
    )


def test_duplicate_from_import_repair_rejects_same_binding_different_modules(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_duplicate_import_edit,
    )

    source = (
        "from pathlib import Path as value\n"
        "from os import path as value\n"
        "\n"
        "print(value)\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Duplicate import: 'value'",
    )

    assert diagnostic.line == 2
    assert deterministic_duplicate_import_edit(
        source,
        diagnostic,
    ) is None


def test_duplicate_import_repair_accepts_identical_alias_sibling(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_duplicate_import_edit,
    )

    source = (
        "import os as operating_system\n"
        "import os as operating_system\n"
        "\n"
        "print(operating_system.getcwd())\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Duplicate import: 'operating_system'",
    )

    edit = deterministic_duplicate_import_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "delete"
    assert edit.line == 2


def test_duplicate_from_import_repair_accepts_identical_alias_sibling(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_duplicate_import_edit,
    )

    source = (
        "from pathlib import Path as FilePath\n"
        "from pathlib import Path as FilePath\n"
        "\n"
        "print(FilePath('.'))\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Duplicate import: 'FilePath'",
    )

    edit = deterministic_duplicate_import_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "delete"
    assert edit.line == 2


def test_duplicate_import_repair_rejects_identical_import_in_other_branch(
    tmp_path,
):
    from codeguardian.ai.engine import (
        Diagnostic,
        deterministic_duplicate_import_edit,
    )

    source = (
        "if condition:\n"
        "    import os\n"
        "else:\n"
        "    import os\n"
    )

    diagnostic = Diagnostic(
        file=tmp_path / "sample.py",
        line=4,
        column=5,
        severity="WARNING",
        message="Duplicate import: 'os'",
    )

    assert deterministic_duplicate_import_edit(
        source,
        diagnostic,
    ) is None


def test_unused_definition_repair_rejects_division_that_can_raise(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_definition_edit,
    )

    source = (
        "def example():\n"
        "    unused = 1 / 0\n"
        "    return 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused definition: 'unused'",
    )

    assert deterministic_unused_definition_edit(
        source,
        diagnostic,
    ) is None


def test_unused_definition_repair_rejects_set_literal(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_definition_edit,
    )

    source = (
        "def example():\n"
        "    unused = {1, 2}\n"
        "    return 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused definition: 'unused'",
    )

    assert deterministic_unused_definition_edit(
        source,
        diagnostic,
    ) is None


def test_unused_definition_repair_rejects_dict_literal(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_definition_edit,
    )

    source = (
        "def example():\n"
        "    unused = {'key': 'value'}\n"
        "    return 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused definition: 'unused'",
    )

    assert deterministic_unused_definition_edit(
        source,
        diagnostic,
    ) is None


def test_unused_definition_repair_accepts_nested_literal_sequence(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_definition_edit,
    )

    source = (
        "def example():\n"
        "    unused = [1, ('safe', None), [True, 3.5]]\n"
        "    return 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused definition: 'unused'",
    )

    edit = deterministic_unused_definition_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "delete"
    assert edit.line == 2


def test_unused_definition_repair_rejects_sequence_with_expression(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unused_definition_edit,
    )

    source = (
        "def example():\n"
        "    unused = [1, 2 / 0]\n"
        "    return 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unused definition: 'unused'",
    )

    assert deterministic_unused_definition_edit(
        source,
        diagnostic,
    ) is None


def test_unreachable_repair_deletes_statement_after_return(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unreachable_code_edit,
    )

    source = (
        "def example():\n"
        "    return 1\n"
        "    value = 2\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unreachable code after unconditional control transfer.",
    )

    edit = deterministic_unreachable_code_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "delete"
    assert edit.line == 3


def test_unreachable_repair_deletes_statement_after_raise(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unreachable_code_edit,
    )

    source = (
        "def example():\n"
        "    raise RuntimeError()\n"
        "    value = 2\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unreachable code after unconditional control transfer.",
    )

    edit = deterministic_unreachable_code_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.line == 3


def test_unreachable_repair_deletes_statement_after_break(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unreachable_code_edit,
    )

    source = (
        "for value in range(3):\n"
        "    break\n"
        "    print(value)\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unreachable code after unconditional control transfer.",
    )

    edit = deterministic_unreachable_code_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.line == 3


def test_unreachable_repair_deletes_statement_after_continue(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unreachable_code_edit,
    )

    source = (
        "for value in range(3):\n"
        "    continue\n"
        "    print(value)\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unreachable code after unconditional control transfer.",
    )

    edit = deterministic_unreachable_code_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.line == 3


def test_unreachable_repair_rejects_multiline_statement(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_unreachable_code_edit,
    )

    source = (
        "def example():\n"
        "    return 1\n"
        "    value = (\n"
        "        2\n"
        "    )\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Unreachable code after unconditional control transfer.",
    )

    assert deterministic_unreachable_code_edit(
        source,
        diagnostic,
    ) is None


def test_unreachable_repair_does_not_cross_branch_boundary(
    tmp_path,
):
    from codeguardian.ai.engine import (
        Diagnostic,
        deterministic_unreachable_code_edit,
    )

    source = (
        "def example(condition):\n"
        "    if condition:\n"
        "        return 1\n"
        "    value = 2\n"
        "    return value\n"
    )

    diagnostic = Diagnostic(
        file=tmp_path / "example.py",
        line=4,
        column=5,
        severity="WARNING",
        message=(
            "Unreachable code after unconditional control transfer."
        ),
    )

    assert deterministic_unreachable_code_edit(
        source,
        diagnostic,
    ) is None


def test_deterministic_mutable_default_list(tmp_path):
    from codeguardian.ai.engine import (
        deterministic_mutable_default_edits,
    )

    source = (
        "def collect(items=[]):\n"
        "    return items\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    edits = deterministic_mutable_default_edits(
        source,
        diagnostic,
    )

    assert edits is not None
    assert len(edits) == 2
    assert edits[0].content == "def collect(items=None):"
    assert edits[1].content == (
        "    if items is None:\n"
        "        items = []"
    )


def test_deterministic_mutable_default_dict(tmp_path):
    from codeguardian.ai.engine import (
        deterministic_mutable_default_edits,
    )

    source = (
        "def collect(options={}):\n"
        "    return options\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    edits = deterministic_mutable_default_edits(
        source,
        diagnostic,
    )

    assert edits is not None
    assert "options=None" in edits[0].content
    assert "options = {}" in edits[1].content


def test_deterministic_mutable_default_rejects_set_call(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_mutable_default_edits,
    )

    source = (
        "def collect(values=set()):\n"
        "    return values\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    assert deterministic_mutable_default_edits(
        source,
        diagnostic,
    ) is None


def test_deterministic_mutable_default_rejects_shadowed_set(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_mutable_default_edits,
    )

    source = (
        "def set():\n"
        '    return ["definition-time"]\n'
        "\n"
        "def collect(values=set()):\n"
        "    return values\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    assert deterministic_mutable_default_edits(
        source,
        diagnostic,
    ) is None


def test_deterministic_mutable_default_keyword_only(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_mutable_default_edits,
    )

    source = (
        "def collect(*, items=[]):\n"
        "    return items\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    edits = deterministic_mutable_default_edits(
        source,
        diagnostic,
    )

    assert edits is not None
    assert "items=None" in edits[0].content


def test_deterministic_mutable_default_rejects_nonempty(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_mutable_default_edits,
    )

    source = (
        "def collect(items=[1]):\n"
        "    return items\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    assert deterministic_mutable_default_edits(
        source,
        diagnostic,
    ) is None


def test_deterministic_mutable_default_rejects_multiple(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_mutable_default_edits,
    )

    source = (
        "def collect(items=[], options={}):\n"
        "    return items, options\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    assert deterministic_mutable_default_edits(
        source,
        diagnostic,
    ) is None


def test_deterministic_assert_tuple_edit(tmp_path):
    from codeguardian.ai.engine import (
        deterministic_assert_tuple_edit,
    )

    source = (
        "def check(value):\n"
        "    assert (value > 0, \"must be positive\")\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Assert condition is a non-empty tuple",
    )

    edit = deterministic_assert_tuple_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "replace"
    assert edit.line == 2
    assert edit.content == (
        '    assert value > 0, "must be positive"'
    )


def test_deterministic_assert_tuple_rejects_one_element(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_assert_tuple_edit,
    )

    source = (
        "def check(value):\n"
        "    assert (value,)\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Assert condition is a non-empty tuple",
    )

    assert deterministic_assert_tuple_edit(
        source,
        diagnostic,
    ) is None


def test_deterministic_assert_tuple_rejects_three_elements(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_assert_tuple_edit,
    )

    source = (
        "def check(value):\n"
        "    assert (value, \"message\", 3)\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Assert condition is a non-empty tuple",
    )

    assert deterministic_assert_tuple_edit(
        source,
        diagnostic,
    ) is None


def test_deterministic_assert_tuple_rejects_dynamic_message(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_assert_tuple_edit,
    )

    source = (
        "def check(value, message):\n"
        "    assert (value, message)\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Assert condition is a non-empty tuple",
    )

    assert deterministic_assert_tuple_edit(
        source,
        diagnostic,
    ) is None


def test_deterministic_assert_tuple_rejects_multiline(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_assert_tuple_edit,
    )

    source = (
        "def check(value):\n"
        "    assert (\n"
        "        value > 0,\n"
        "        \"must be positive\",\n"
        "    )\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Assert condition is a non-empty tuple",
    )

    assert deterministic_assert_tuple_edit(
        source,
        diagnostic,
    ) is None


def test_deterministic_mutable_default_preserves_docstring(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_mutable_default_edits,
    )

    source = (
        "def collect(items=[]):\n"
        '    """Return collected items."""\n'
        "    return items\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    edits = deterministic_mutable_default_edits(
        source,
        diagnostic,
    )

    assert edits is not None
    assert len(edits) == 2
    assert edits[0].content == "def collect(items=None):"
    assert edits[1].operation == "insert_after"
    assert edits[1].line == 2
    assert edits[1].content == (
        "    if items is None:\n"
        "        items = []"
    )


def test_deterministic_mutable_default_uses_actual_indentation(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_mutable_default_edits,
    )

    source = (
        "def collect(items=[]):\n"
        "\treturn items\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    edits = deterministic_mutable_default_edits(
        source,
        diagnostic,
    )

    assert edits is not None
    assert edits[1].content == (
        "\tif items is None:\n"
        "\t\titems = []"
    )


def test_deterministic_mutable_default_nested_indentation(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_mutable_default_edits,
    )

    source = (
        "class Collector:\n"
        "  def collect(items=[]):\n"
        "    return items\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    edits = deterministic_mutable_default_edits(
        source,
        diagnostic,
    )

    assert edits is not None
    assert edits[0].content == (
        "  def collect(items=None):"
    )
    assert edits[1].content == (
        "    if items is None:\n"
        "      items = []"
    )


def test_deterministic_mutable_default_rejects_multiline_docstring(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_mutable_default_edits,
    )

    source = (
        "def collect(items=[]):\n"
        '    """Return\n'
        "    collected items.\n"
        '    """\n'
        "    return items\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    assert deterministic_mutable_default_edits(
        source,
        diagnostic,
    ) is None


def test_deterministic_mutable_default_docstring_semantics(
    tmp_path,
):
    from codeguardian.ai.edit import apply_ai_edits
    from codeguardian.ai.engine import (
        deterministic_mutable_default_edits,
    )

    source = (
        "def collect(items=[]):\n"
        '    """Return collected items."""\n'
        "    return items\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Mutable default argument",
    )

    edits = deterministic_mutable_default_edits(
        source,
        diagnostic,
    )

    assert edits is not None

    repaired = apply_ai_edits(source, edits)

    namespace = {}
    exec(repaired, namespace)

    collect = namespace["collect"]

    assert collect.__doc__ == "Return collected items."
    assert collect() == []
    assert collect() is not collect()


def test_deterministic_bare_except_edit(tmp_path):
    from codeguardian.ai.engine import (
        deterministic_bare_except_edit,
    )

    source = (
        "try:\n"
        "    work()\n"
        "except:\n"
        "    recover()\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Bare except",
    )

    edit = deterministic_bare_except_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "replace"
    assert edit.line == 3
    assert edit.content == "except Exception:"


def test_deterministic_bare_except_preserves_indentation(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_bare_except_edit,
    )

    source = (
        "def run():\n"
        "    try:\n"
        "        work()\n"
        "    except:\n"
        "        recover()\n"
    )
    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "Bare except",
    )

    edit = deterministic_bare_except_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.content == "    except Exception:"


def test_deterministic_bare_except_rejects_specific_handler(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_bare_except_edit,
    )
    from codeguardian.scanner import Diagnostic

    source = (
        "try:\n"
        "    work()\n"
        "except ValueError:\n"
        "    recover()\n"
    )

    diagnostic = Diagnostic(
        file=str(tmp_path / "example.py"),
        line=3,
        column=1,
        severity="WARNING",
        message=(
            "Bare except catches BaseException; catch a specific "
            "exception type instead."
        ),
    )

    assert deterministic_bare_except_edit(
        source,
        diagnostic,
    ) is None


def test_deterministic_bare_except_rejects_nonliteral_header(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_bare_except_edit,
    )
    from codeguardian.scanner import Diagnostic

    source = (
        "try:\n"
        "    work()\n"
        "except   :  # unusual formatting\n"
        "    recover()\n"
    )

    diagnostic = Diagnostic(
        file=str(tmp_path / "example.py"),
        line=3,
        column=1,
        severity="WARNING",
        message=(
            "Bare except catches BaseException; catch a specific "
            "exception type instead."
        ),
    )

    assert deterministic_bare_except_edit(
        source,
        diagnostic,
    ) is None


def test_deterministic_identical_duplicate_function(tmp_path):
    from codeguardian.ai.engine import (
        deterministic_duplicate_definition_edit,
    )

    source = (
        "def run(value):\n"
        "    return value + 1\n"
        "\n"
        "def run(value):\n"
        "    return value + 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "defined more than once",
    )

    edit = deterministic_duplicate_definition_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "delete_range"
    assert edit.line == 4
    assert edit.end_line == 5


def test_deterministic_duplicate_function_rejects_different_body(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_duplicate_definition_edit,
    )

    source = (
        "def run(value):\n"
        "    return value + 1\n"
        "\n"
        "def run(value):\n"
        "    return value + 2\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "defined more than once",
    )

    assert (
        deterministic_duplicate_definition_edit(
            source,
            diagnostic,
        )
        is None
    )


def test_deterministic_identical_duplicate_class(tmp_path):
    from codeguardian.ai.engine import (
        deterministic_duplicate_definition_edit,
    )

    source = (
        "class Thing:\n"
        "    value = 1\n"
        "\n"
        "class Thing:\n"
        "    value = 1\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "defined more than once",
    )

    edit = deterministic_duplicate_definition_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "delete_range"
    assert edit.line == 4
    assert edit.end_line == 5


def test_deterministic_duplicate_definition_requires_same_scope(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_duplicate_definition_edit,
    )
    from codeguardian.scanner import Diagnostic

    source = (
        "class First:\n"
        "    def run(self):\n"
        "        return 1\n"
        "\n"
        "class Second:\n"
        "    def run(self):\n"
        "        return 1\n"
    )

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    diagnostic = Diagnostic(
        file=path,
        line=6,
        column=5,
        severity="WARNING",
        message="Name 'run' is defined more than once.",
    )

    assert (
        deterministic_duplicate_definition_edit(
            source,
            diagnostic,
        )
        is None
    )


def test_deterministic_duplicate_definition_includes_decorator(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_duplicate_definition_edit,
    )

    source = (
        "@staticmethod\n"
        "def run(value):\n"
        "    return value\n"
        "\n"
        "@staticmethod\n"
        "def run(value):\n"
        "    return value\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "defined more than once",
    )

    edit = deterministic_duplicate_definition_edit(
        source,
        diagnostic,
    )

    assert edit is not None
    assert edit.operation == "delete_range"
    assert edit.line == 5
    assert edit.end_line == 7


def test_deterministic_duplicate_definition_rejects_different_decorator(
    tmp_path,
):
    from codeguardian.ai.engine import (
        deterministic_duplicate_definition_edit,
    )

    source = (
        "@staticmethod\n"
        "def run(value):\n"
        "    return value\n"
        "\n"
        "@classmethod\n"
        "def run(value):\n"
        "    return value\n"
    )

    diagnostic = diagnostic_for(
        tmp_path,
        source,
        "defined more than once",
    )

    assert (
        deterministic_duplicate_definition_edit(
            source,
            diagnostic,
        )
        is None
    )
