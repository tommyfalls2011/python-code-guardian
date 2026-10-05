from pathlib import Path

from codeguardian.ai.edit import AIEdit
from codeguardian.ai.engine import (
    AIEditEvaluation,
    apply_evaluated_ai_edit,
    evaluate_ai_edit,
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

