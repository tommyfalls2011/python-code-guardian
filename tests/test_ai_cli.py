from codeguardian.analyzer import analyze_file
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

def test_cli_ai_repair_routes_mutable_default_to_multi_edit(
    tmp_path,
    monkeypatch,
    capsys,
):
    import codeguardian.cli as cli
    from codeguardian.ai.edit import AIEdit
    from codeguardian.ai.engine import AIEditsEvaluation

    target = tmp_path / "sample.py"
    original = (
        "def example(items=[]):\n"
        "    return len(items)\n"
    )
    candidate = (
        "def example(items=None):\n"
        "    if items is None:\n"
        "        items = []\n"
        "    return len(items)\n"
    )
    target.write_text(original, encoding="utf-8")

    diagnostics = analyze_file(target).diagnostics
    diagnostic = next(
        item
        for item in diagnostics
        if "Mutable default argument" in item.message
    )

    called = {
        "single": False,
        "multi": False,
    }

    def fake_single(**kwargs):
        called["single"] = True
        raise AssertionError(
            "mutable default used single-edit evaluator"
        )

    def fake_multi(
        *,
        source,
        diagnostic,
        model,
        provider=None,
    ):
        called["multi"] = True
        assert source == original
        assert model == "test-model"
        assert provider is not None

        edits = provider(
            diagnostic=diagnostic.message,
            line=diagnostic.line,
            context="",
            model=model,
            max_edits=3,
        )

        assert len(edits) == 2
        assert edits[0].content == (
            "def example(items=None):"
        )
        assert edits[1].content == (
            "    if items is None:\n"
            "        items = []"
        )

        return AIEditsEvaluation(
            accepted=True,
            reason="safe",
            edits=[
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
            ],
            candidate_source=candidate,
            before_diagnostics=[diagnostic],
            after_diagnostics=[],
        )

    monkeypatch.setattr(
        cli,
        "evaluate_ai_edit",
        fake_single,
    )
    monkeypatch.setattr(
        cli,
        "evaluate_ai_edits",
        fake_multi,
    )
    monkeypatch.setattr(
        cli,
        "analyze_file",
        analyze_file,
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
    assert called["multi"] is True
    assert called["single"] is False
    assert "[REPAIRED]" in output
    assert "[AI REPAIRED]" not in output
    assert "Deterministic repairs applied: 1" in output
    assert "AI repairs applied: 0" in output
    assert "Operation: replace" in output
    assert "Operation: insert_after" in output
    assert target.read_text(
        encoding="utf-8"
    ) == candidate


def test_cli_ai_repair_prioritizes_error_before_warning(
    tmp_path,
    monkeypatch,
    capsys,
):
    import codeguardian.cli as cli

    target = tmp_path / "sample.py"
    target.write_text(
        "import os\n"
        "import os\n"
        "def example():\n"
        "    definitely_not_defined\n",
        encoding="utf-8",
    )

    attempted = []

    def fake_evaluate(
        *,
        source,
        diagnostic,
        model,
    ):
        attempted.append(diagnostic.severity.upper())

        assert diagnostic.severity.upper() == "ERROR"

        candidate = (
            "import os\n"
            "import os\n"
            "def example():\n"
            "    return 1\n"
        )

        candidate_path = tmp_path / "candidate.py"
        candidate_path.write_text(
            candidate,
            encoding="utf-8",
        )

        return AIEditEvaluation(
            accepted=True,
            reason="safe",
            edit=AIEdit(
                "replace",
                4,
                "    return 1",
            ),
            candidate_source=candidate,
            before_diagnostics=analyze_file(
                target
            ).diagnostics,
            after_diagnostics=analyze_file(
                candidate_path
            ).diagnostics,
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
            "--max-repairs",
            "1",
        ],
    )

    cli.main()
    capsys.readouterr()

    assert attempted == ["ERROR"]


def test_cli_ai_repair_rescans_between_repairs(
    tmp_path,
    monkeypatch,
    capsys,
):
    import codeguardian.cli as cli

    target = tmp_path / "sample.py"
    target.write_text(
        "def example():\n"
        "    first_missing\n"
        "    second_missing\n"
        "    return 1\n",
        encoding="utf-8",
    )

    attempted_lines = []

    def fake_evaluate(
        *,
        source,
        diagnostic,
        model,
    ):
        attempted_lines.append(diagnostic.line)

        before = analyze_file(target).diagnostics

        if "first_missing" in diagnostic.message:
            assert diagnostic.line == 2
            candidate = source.replace(
                "    first_missing\n",
                "",
                1,
            )
            edit = AIEdit("delete", 2)
        elif "second_missing" in diagnostic.message:
            # The first deletion moved this from line 3 to line 2.
            assert diagnostic.line == 2
            candidate = source.replace(
                "    second_missing\n",
                "",
                1,
            )
            edit = AIEdit("delete", 2)
        else:
            raise AssertionError(
                f"unexpected diagnostic: {diagnostic.message}"
            )

        candidate_path = tmp_path / "candidate.py"
        candidate_path.write_text(
            candidate,
            encoding="utf-8",
        )

        after = analyze_file(
            candidate_path
        ).diagnostics

        # Transaction validation compares diagnostics from
        # the real target path. Preserve those paths while
        # using the freshly analyzed candidate diagnostics.
        after = [
            type(item)(
                file=target,
                line=item.line,
                column=item.column,
                severity=item.severity,
                message=item.message,
            )
            for item in after
        ]

        return AIEditEvaluation(
            accepted=True,
            reason="safe",
            edit=edit,
            candidate_source=candidate,
            before_diagnostics=before,
            after_diagnostics=after,
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
            "--max-repairs",
            "2",
        ],
    )

    result = cli.main()
    output = capsys.readouterr().out

    assert result == 0
    assert attempted_lines == [2, 2]
    assert output.count("[AI REPAIRED]") == 2
    assert "AI repairs applied: 2" in output
    assert target.read_text(
        encoding="utf-8"
    ) == (
        "def example():\n"
        "    return 1\n"
    )


def test_cli_ai_repair_respects_max_repairs(
    tmp_path,
    monkeypatch,
    capsys,
):
    import codeguardian.cli as cli

    target = tmp_path / "sample.py"
    target.write_text(
        "def example():\n"
        "    first_missing\n"
        "    second_missing\n"
        "    return 1\n",
        encoding="utf-8",
    )

    calls = []

    def fake_evaluate(
        *,
        source,
        diagnostic,
        model,
    ):
        calls.append(diagnostic.message)

        before = analyze_file(target).diagnostics

        candidate = source.replace(
            "    first_missing\n",
            "",
            1,
        )

        candidate_path = tmp_path / "candidate.py"
        candidate_path.write_text(
            candidate,
            encoding="utf-8",
        )

        after = [
            type(item)(
                file=target,
                line=item.line,
                column=item.column,
                severity=item.severity,
                message=item.message,
            )
            for item in analyze_file(
                candidate_path
            ).diagnostics
        ]

        return AIEditEvaluation(
            accepted=True,
            reason="safe",
            edit=AIEdit("delete", 2),
            candidate_source=candidate,
            before_diagnostics=before,
            after_diagnostics=after,
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
            "--max-repairs",
            "1",
        ],
    )

    result = cli.main()
    output = capsys.readouterr().out

    assert result == 1
    assert len(calls) == 1
    assert output.count("[AI REPAIRED]") == 1
    assert "AI repairs applied: 1" in output
    assert "second_missing" in target.read_text(
        encoding="utf-8"
    )


def test_cli_ai_repair_does_not_retry_unchanged_rejection(
    tmp_path,
    monkeypatch,
    capsys,
):
    import codeguardian.cli as cli

    target = tmp_path / "sample.py"
    target.write_text(
        "import os\n"
        "import os\n"
        "\n"
        "def example():\n"
        "    definitely_not_defined\n"
        "    return 1\n",
        encoding="utf-8",
    )

    attempts = []

    def fake_evaluate(
        *,
        source,
        diagnostic,
        model,
        provider=None,
    ):
        attempts.append(
            (
                diagnostic.message,
                diagnostic.line,
            )
        )

        if "Undefined name" in diagnostic.message:
            candidate = source.replace(
                "    definitely_not_defined\n",
                "",
                1,
            )

            candidate_path = tmp_path / "candidate.py"
            candidate_path.write_text(
                candidate,
                encoding="utf-8",
            )

            after = [
                type(item)(
                    file=target,
                    line=item.line,
                    column=item.column,
                    severity=item.severity,
                    message=item.message,
                )
                for item in analyze_file(
                    candidate_path
                ).diagnostics
            ]

            return AIEditEvaluation(
                accepted=True,
                reason="safe",
                edit=AIEdit("delete", 5),
                candidate_source=candidate,
                before_diagnostics=analyze_file(
                    target
                ).diagnostics,
                after_diagnostics=after,
            )

        return AIEditEvaluation(
            accepted=False,
            reason="target diagnostic was not repaired",
            edit=AIEdit(
                "replace",
                diagnostic.line,
                source.splitlines()[
                    diagnostic.line - 1
                ],
            ),
            candidate_source=None,
            before_diagnostics=analyze_file(
                target
            ).diagnostics,
            after_diagnostics=analyze_file(
                target
            ).diagnostics,
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
            "--max-repairs",
            "3",
        ],
    )

    result = cli.main()
    output = capsys.readouterr().out

    duplicate_attempts = [
        item
        for item in attempts
        if "Duplicate import" in item[0]
    ]

    assert result == 0
    assert len(duplicate_attempts) == 1
    assert "[AI REJECTED]" in output
    assert "definitely_not_defined" not in (
        target.read_text(encoding="utf-8")
    )


def test_cli_duplicate_import_uses_deterministic_repair(
    tmp_path,
    monkeypatch,
    capsys,
):
    import codeguardian.cli as cli

    target = tmp_path / "sample.py"
    target.write_text(
        "import os\n"
        "import os\n"
        "\n"
        "print(os.getcwd())\n",
        encoding="utf-8",
    )

    def forbidden_ollama(*args, **kwargs):
        raise AssertionError(
            "Ollama evaluator must not be used "
            "for a safe duplicate import"
        )

    monkeypatch.setattr(
        cli,
        "evaluate_ai_edits",
        forbidden_ollama,
    )

    original_evaluate = cli.evaluate_ai_edit

    calls = []

    def guarded_evaluate(
        *,
        source,
        diagnostic,
        model,
        provider=None,
    ):
        calls.append(provider)

        assert provider is not None

        return original_evaluate(
            source=source,
            diagnostic=diagnostic,
            model=model,
            provider=provider,
        )

    monkeypatch.setattr(
        cli,
        "evaluate_ai_edit",
        guarded_evaluate,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codeguardian",
            str(target),
            "--ai-repair",
            "--ai-model",
            "must-not-be-called",
            "--max-repairs",
            "1",
        ],
    )

    result = cli.main()
    output = capsys.readouterr().out

    assert result == 0
    assert len(calls) == 1
    assert calls[0] is not None
    assert "Deterministic repairs applied: 1" in output

    assert target.read_text(
        encoding="utf-8"
    ) == (
        "import os\n"
        "\n"
        "print(os.getcwd())\n"
    )


def test_cli_unused_literal_uses_deterministic_repair(
    tmp_path,
    monkeypatch,
    capsys,
):
    from codeguardian import cli

    path = tmp_path / "example.py"
    path.write_text(
        "def example():\n"
        "    unused_value = 123\n"
        "    return 1\n",
        encoding="utf-8",
    )

    def forbidden_ai(*args, **kwargs):
        raise AssertionError(
            "Ollama evaluator must not be used "
            "for safe unused definition"
        )

    real_evaluate = cli.evaluate_ai_edit

    def guarded_evaluate(*args, **kwargs):
        assert kwargs.get("provider") is not None
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        cli,
        "evaluate_ai_edit",
        guarded_evaluate,
    )
    monkeypatch.setattr(
        cli,
        "evaluate_ai_edits",
        forbidden_ai,
    )
    monkeypatch.setattr(
        "codeguardian.ai.engine.request_ai_edit",
        forbidden_ai,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "codeguardian",
            str(path),
            "--ai-repair",
            "--max-repairs",
            "1",
        ],
    )

    result = cli.main()

    assert result == 0
    assert "unused_value = 123" not in path.read_text(
        encoding="utf-8"
    )

    output = capsys.readouterr().out
    assert "Deterministic repairs applied: 1" in output


def test_cli_unused_literals_rescan_and_share_repair_budget(
    tmp_path,
    monkeypatch,
    capsys,
):
    from codeguardian import cli

    path = tmp_path / "example.py"
    path.write_text(
        "def example():\n"
        "    first_unused = 1\n"
        "    second_unused = 2\n"
        "    return 1\n",
        encoding="utf-8",
    )

    attempted_lines = []
    real_evaluate = cli.evaluate_ai_edit

    def guarded_evaluate(*args, **kwargs):
        assert kwargs.get("provider") is not None
        attempted_lines.append(
            kwargs["diagnostic"].line
        )
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        cli,
        "evaluate_ai_edit",
        guarded_evaluate,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "codeguardian",
            str(path),
            "--ai-repair",
            "--max-repairs",
            "2",
        ],
    )

    result = cli.main()

    assert result == 0
    assert attempted_lines == [2, 2]

    source = path.read_text(encoding="utf-8")
    assert "first_unused" not in source
    assert "second_unused" not in source

    output = capsys.readouterr().out
    assert "Deterministic repairs applied: 2" in output


def test_cli_unused_literal_respects_max_repairs(
    tmp_path,
    monkeypatch,
    capsys,
):
    from codeguardian import cli

    path = tmp_path / "example.py"
    path.write_text(
        "def example():\n"
        "    first_unused = 1\n"
        "    second_unused = 2\n"
        "    return 1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "codeguardian",
            str(path),
            "--ai-repair",
            "--max-repairs",
            "1",
        ],
    )

    result = cli.main()

    assert result == 0

    source = path.read_text(encoding="utf-8")

    remaining = sum(
        name in source
        for name in (
            "first_unused",
            "second_unused",
        )
    )

    assert remaining == 1

    output = capsys.readouterr().out
    assert "Deterministic repairs applied: 1" in output
    assert "Unused definition:" in output


def test_cli_safe_mode_does_not_auto_delete_unused_import(
    tmp_path,
    monkeypatch,
    capsys,
):
    from codeguardian import cli

    path = tmp_path / "side_effect_plugin.py"
    path.write_text(
        "import side_effect_plugin\n"
        "\n"
        "value = 1\n",
        encoding="utf-8",
    )

    def rejected_ai(*args, **kwargs):
        raise ValueError("AI unavailable for safety regression")

    monkeypatch.setattr(
        cli,
        "evaluate_ai_edit",
        rejected_ai,
    )
    monkeypatch.setattr(
        cli,
        "evaluate_ai_edits",
        rejected_ai,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "codeguardian",
            str(path),
            "--ai-repair",
            "--max-repairs",
            "1",
            "--policy",
            "safe",
        ],
    )

    result = cli.main()

    assert result == 0

    source = path.read_text(encoding="utf-8")
    assert "import side_effect_plugin" in source
    assert "value = 1" in source

    output = capsys.readouterr().out
    assert "AI repairs applied: 0" in output


def test_cli_unreachable_code_uses_deterministic_repair(
    tmp_path,
    monkeypatch,
    capsys,
):
    from codeguardian import cli

    path = tmp_path / "example.py"
    path.write_text(
        "def example():\n"
        "    return 1\n"
        "    value = 2\n",
        encoding="utf-8",
    )

    real_evaluate = cli.evaluate_ai_edit

    def guarded_evaluate(*args, **kwargs):
        provider = kwargs.get("provider")

        if provider is None:
            raise AssertionError(
                "Ollama evaluator must not be used "
                "for safe unreachable code"
            )

        return real_evaluate(*args, **kwargs)

    def forbidden_multi(*args, **kwargs):
        raise AssertionError(
            "multi-edit AI must not be used "
            "for safe unreachable code"
        )

    monkeypatch.setattr(
        cli,
        "evaluate_ai_edit",
        guarded_evaluate,
    )
    monkeypatch.setattr(
        cli,
        "evaluate_ai_edits",
        forbidden_multi,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "codeguardian",
            str(path),
            "--ai-repair",
            "--max-repairs",
            "1",
        ],
    )

    result = cli.main()

    assert result == 0

    source = path.read_text(encoding="utf-8")
    assert "return 1" in source
    assert "value = 2" not in source

    output = capsys.readouterr().out
    assert "Deterministic repairs applied: 1" in output


def test_cli_safe_mode_skips_if_false_ai_repair(
    tmp_path,
    monkeypatch,
    capsys,
):
    from codeguardian import cli

    path = tmp_path / "example.py"
    original = (
        "def example():\n"
        "    if False:\n"
        '        print("never")\n'
        "    return 1\n"
    )
    path.write_text(original, encoding="utf-8")

    def forbidden_ai(*args, **kwargs):
        raise AssertionError(
            "AI must not be called for if False "
            "under safe policy"
        )

    monkeypatch.setattr(
        cli,
        "evaluate_ai_edit",
        forbidden_ai,
    )
    monkeypatch.setattr(
        cli,
        "evaluate_ai_edits",
        forbidden_ai,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "codeguardian",
            str(path),
            "--ai-repair",
            "--policy",
            "safe",
            "--max-repairs",
            "3",
        ],
    )

    result = cli.main()
    output = capsys.readouterr().out

    assert result == 0
    assert path.read_text(encoding="utf-8") == original
    assert "[SKIPPED SAFE]" in output
    assert (
        "intentionally unreachable code"
        in output
    )
    assert "Deterministic repairs applied: 0" in output
    assert "AI repairs applied: 0" in output
    assert "Total repairs applied: 0" in output


def test_cli_safe_if_false_skip_does_not_block_other_repairs(
    tmp_path,
    monkeypatch,
    capsys,
):
    from codeguardian import cli

    path = tmp_path / "example.py"
    path.write_text(
        "import os\n"
        "import os\n"
        "\n"
        "def example():\n"
        "    if False:\n"
        '        print("never")\n'
        "    return os.getcwd()\n",
        encoding="utf-8",
    )

    real_evaluate = cli.evaluate_ai_edit

    def guarded_evaluate(*args, **kwargs):
        assert kwargs.get("provider") is not None
        return real_evaluate(*args, **kwargs)

    def forbidden_multi(*args, **kwargs):
        raise AssertionError(
            "AI must not be called during this safe test"
        )

    monkeypatch.setattr(
        cli,
        "evaluate_ai_edit",
        guarded_evaluate,
    )
    monkeypatch.setattr(
        cli,
        "evaluate_ai_edits",
        forbidden_multi,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "codeguardian",
            str(path),
            "--ai-repair",
            "--policy",
            "safe",
            "--max-repairs",
            "3",
        ],
    )

    result = cli.main()
    output = capsys.readouterr().out

    assert result == 0

    source = path.read_text(encoding="utf-8")

    assert source.count("import os") == 1
    assert "if False:" in source
    assert 'print("never")' in source

    assert "Deterministic repairs applied: 1" in output
    assert "AI repairs applied: 0" in output
    assert "[SKIPPED SAFE]" in output


def test_cli_safe_if_false_skip_reported_once_across_rescans(
    tmp_path,
    monkeypatch,
    capsys,
):
    from codeguardian import cli

    path = tmp_path / "example.py"
    path.write_text(
        "import os\n"
        "import os\n"
        "import sys\n"
        "import sys\n"
        "\n"
        "def example():\n"
        "    if False:\n"
        '        print("never")\n'
        "    return os.getcwd(), sys.version\n",
        encoding="utf-8",
    )

    real_evaluate = cli.evaluate_ai_edit

    def guarded_evaluate(*args, **kwargs):
        assert kwargs.get("provider") is not None
        return real_evaluate(*args, **kwargs)

    def forbidden_multi(*args, **kwargs):
        raise AssertionError(
            "AI must not be called during safe-skip test"
        )

    monkeypatch.setattr(
        cli,
        "evaluate_ai_edit",
        guarded_evaluate,
    )
    monkeypatch.setattr(
        cli,
        "evaluate_ai_edits",
        forbidden_multi,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "codeguardian",
            str(path),
            "--ai-repair",
            "--policy",
            "safe",
            "--max-repairs",
            "5",
        ],
    )

    result = cli.main()
    output = capsys.readouterr().out

    assert result == 0
    assert output.count("[SKIPPED SAFE]") == 1
    assert "Deterministic repairs applied: 2" in output
    assert "AI repairs applied: 0" in output

    source = path.read_text(encoding="utf-8")
    assert source.count("import os") == 1
    assert source.count("import sys") == 1
    assert "if False:" in source


def test_cli_bare_except_uses_deterministic_repair(
    tmp_path,
    monkeypatch,
    capsys,
):
    from codeguardian import cli

    path = tmp_path / "example.py"
    path.write_text(
        "def work():\n"
        "    return 1\n"
        "\n"
        "def recover():\n"
        "    return 2\n"
        "\n"
        "try:\n"
        "    work()\n"
        "except:\n"
        "    recover()\n",
        encoding="utf-8",
    )

    real_evaluate = cli.evaluate_ai_edit

    def guarded_evaluate(*args, **kwargs):
        provider = kwargs.get("provider")
        assert provider is not None
        return real_evaluate(*args, **kwargs)

    def forbidden_multi(*args, **kwargs):
        raise AssertionError(
            "AI multi-edit must not be called for bare except"
        )

    monkeypatch.setattr(
        cli,
        "evaluate_ai_edit",
        guarded_evaluate,
    )
    monkeypatch.setattr(
        cli,
        "evaluate_ai_edits",
        forbidden_multi,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "codeguardian",
            str(path),
            "--ai-repair",
            "--policy",
            "safe",
            "--max-repairs",
            "2",
        ],
    )

    result = cli.main()
    output = capsys.readouterr().out
    source = path.read_text(encoding="utf-8")

    assert result == 0
    assert "except Exception:" in source
    assert "\nexcept:\n" not in source
    assert "[REPAIRED]" in output
    assert "[AI REPAIRED]" not in output
    assert "Deterministic repairs applied: 1" in output
    assert "AI repairs applied: 0" in output
    assert "Total repairs applied: 1" in output
