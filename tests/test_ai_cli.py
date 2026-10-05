import sys

import pytest

from codeguardian.ai.edit import AIEdit
from codeguardian.ai.engine import AIEditEvaluation


def test_cli_rejects_deterministic_and_ai_modes_together(
    tmp_path,
    monkeypatch,
):
    from codeguardian.cli import main

    target = tmp_path / "sample.py"
    target.write_text(
        "def example():\n"
        "    definitely_not_defined\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codeguardian",
            str(target),
            "--repair",
            "--ai-repair",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2


def test_cli_ai_repair_applies_one_bounded_edit(
    tmp_path,
    monkeypatch,
    capsys,
):
    import codeguardian.cli as cli

    target = tmp_path / "sample.py"
    target.write_text(
        "def example():\n"
        "    definitely_not_defined\n"
        "    return 1\n",
        encoding="utf-8",
    )

    def fake_evaluate(
        *,
        source,
        diagnostic,
        model,
    ):
        assert model == "test-model"
        assert "definitely_not_defined" in source

        candidate = (
            "def example():\n"
            "    return 1\n"
        )

        return AIEditEvaluation(
            accepted=True,
            reason="safe",
            edit=AIEdit("delete", 2),
            candidate_source=candidate,
            before_diagnostics=[diagnostic],
            after_diagnostics=[],
        )

    monkeypatch.setattr(
        cli,
        "evaluate_ai_edit",
        fake_evaluate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codeguardian",
            str(target),
            "--ai-repair",
            "--ai-model",
            "test-model",
        ],
    )

    result = cli.main()
    output = capsys.readouterr().out

    assert result == 0
    assert "[AI REPAIRED]" in output
    assert "Operation: delete" in output
    assert target.read_text(
        encoding="utf-8"
    ) == (
        "def example():\n"
        "    return 1\n"
    )


def test_cli_ai_repair_leaves_rejected_source_unchanged(
    tmp_path,
    monkeypatch,
    capsys,
):
    import codeguardian.cli as cli

    target = tmp_path / "sample.py"
    original = (
        "def example():\n"
        "    definitely_not_defined\n"
        "    return 1\n"
    )
    target.write_text(original, encoding="utf-8")

    def fake_evaluate(
        *,
        source,
        diagnostic,
        model,
    ):
        return AIEditEvaluation(
            accepted=False,
            reason="candidate has invalid Python syntax",
            edit=AIEdit(
                "replace",
                diagnostic.line,
                "    if True:",
            ),
            candidate_source=None,
            before_diagnostics=[diagnostic],
            after_diagnostics=[],
        )

    monkeypatch.setattr(
        cli,
        "evaluate_ai_edit",
        fake_evaluate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codeguardian",
            str(target),
            "--ai-repair",
        ],
    )

    result = cli.main()
    output = capsys.readouterr().out

    assert result == 1
    assert "[AI REJECTED]" in output
    assert "AI repairs applied: 0" in output
    assert target.read_text(
        encoding="utf-8"
    ) == original
