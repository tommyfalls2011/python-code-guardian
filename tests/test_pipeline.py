

def test_pipeline_policy_off(tmp_path):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    path = tmp_path / "broken.py"
    original = "def hello(name)\n    return name\n"
    path.write_text(original, encoding="utf-8")

    result = GuardianPipeline(
        policy=RepairPolicy.OFF,
    ).repair(path)

    assert result.repairs_applied == 0
    assert path.read_text(encoding="utf-8") == original


def test_pipeline_policy_safe(tmp_path):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    path = tmp_path / "broken.py"
    path.write_text(
        "def hello(name)\n    return name\n",
        encoding="utf-8",
    )

    result = GuardianPipeline(
        policy=RepairPolicy.SAFE,
    ).repair(path)

    assert result.repairs_applied == 1
    assert "def hello(name):" in path.read_text(encoding="utf-8")


def test_pipeline_policy_tested_forces_tests(tmp_path):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    pipeline = GuardianPipeline(
        policy=RepairPolicy.TESTED,
    )

    assert pipeline.policy is RepairPolicy.TESTED
    assert pipeline.run_tests is True


def test_cli_policy_off_does_not_modify_project(tmp_path, capsys):
    from codeguardian.cli import main

    path = tmp_path / "broken.py"
    original = "def hello(name)\n    return name\n"
    path.write_text(original, encoding="utf-8")

    import sys
    old_argv = sys.argv
    try:
        sys.argv = [
            "codeguardian",
            str(tmp_path),
            "--repair",
            "--policy",
            "off",
        ]
        exit_code = main()
    finally:
        sys.argv = old_argv

    assert exit_code == 1
    assert path.read_text(encoding="utf-8") == original

    output = capsys.readouterr().out
    assert "Repair policy: off" in output
    assert "Repairs applied: 0" in output


def test_safe_policy_rejects_medium_confidence_repair(tmp_path):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    path = tmp_path / "example.py"
    original = "values = [1 2, 3]\n"
    path.write_text(original, encoding="utf-8")

    pipeline = GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    )

    result = pipeline.repair(path)

    assert result.repairs_applied == 0
    assert path.read_text(encoding="utf-8") == original


def test_safe_policy_accepts_high_confidence_repair(tmp_path):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    path = tmp_path / "example.py"
    path.write_text(
        "def hello()\n"
        "    return 1\n",
        encoding="utf-8",
    )

    pipeline = GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    )

    result = pipeline.repair(path)

    assert result.repairs_applied == 1
    assert path.read_text(encoding="utf-8") == (
        "def hello():\n"
        "    return 1\n"
    )


def test_pipeline_confidence_policy_thresholds():
    from codeguardian.confidence import RepairConfidence
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    safe = GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    )

    assert safe._confidence_allows(
        RepairConfidence.HIGH
    )
    assert not safe._confidence_allows(
        RepairConfidence.MEDIUM
    )
    assert not safe._confidence_allows(
        RepairConfidence.LOW
    )

    tested = GuardianPipeline(
        policy=RepairPolicy.TESTED,
        create_backups=False,
    )

    assert tested._confidence_allows(
        RepairConfidence.HIGH
    )
    assert tested._confidence_allows(
        RepairConfidence.MEDIUM
    )
    assert not tested._confidence_allows(
        RepairConfidence.LOW
    )


def test_safe_policy_reports_medium_confidence_skip(tmp_path):
    from codeguardian.confidence import RepairConfidence
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    path = tmp_path / "example.py"
    original = "values = [1 2, 3]\n"
    path.write_text(original, encoding="utf-8")

    pipeline = GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    )

    result = pipeline.repair(path)

    assert result.repairs_applied == 0
    assert len(result.skipped_repairs) == 1

    skipped = result.skipped_repairs[0]

    assert skipped.file == path
    assert skipped.confidence is RepairConfidence.MEDIUM
    assert "missing comma" in skipped.reason.lower()

    assert path.read_text(encoding="utf-8") == original


def test_high_confidence_repair_is_not_reported_as_skipped(tmp_path):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    path = tmp_path / "example.py"
    path.write_text(
        "def hello()\n"
        "    return 1\n",
        encoding="utf-8",
    )

    pipeline = GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    )

    result = pipeline.repair(path)

    assert result.repairs_applied == 1
    assert result.skipped_repairs == []


def test_cli_safe_policy_reports_medium_confidence_skip(
    tmp_path,
    monkeypatch,
    capsys,
):
    from codeguardian.cli import main

    path = tmp_path / "example.py"
    path.write_text(
        "values = [1 2, 3]\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "codeguardian",
            str(path),
            "--repair",
            "--policy",
            "safe",
        ],
    )

    exit_code = main()
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "[SKIPPED MEDIUM]" in output
    assert "Blocked by repair policy: safe" in output
    assert "Repairs applied: 0" in output
    assert "Repairs skipped: 1" in output


def test_cli_dry_run_displays_repair_confidence(
    tmp_path,
    monkeypatch,
    capsys,
):
    from codeguardian.cli import main

    path = tmp_path / "example.py"
    path.write_text(
        "values = [1 2, 3]\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "codeguardian",
            str(path),
            "--dry-run",
        ],
    )

    exit_code = main()
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "[WOULD REPAIR]" in output
    assert "Confidence: MEDIUM" in output


def test_safe_policy_skip_is_persisted_to_audit(tmp_path):
    from codeguardian.audit import RepairAudit
    from codeguardian.confidence import RepairConfidence
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    target = tmp_path / "example.py"
    original = "values = [1 2, 3]\n"
    target.write_text(original, encoding="utf-8")

    pipeline = GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    )

    result = pipeline.repair(target)

    assert result.repairs_applied == 0
    assert len(result.skipped_repairs) == 1

    entries = RepairAudit(
        tmp_path / ".guardian-audit.json"
    ).load()

    assert len(entries) == 1

    entry = entries[0]

    assert entry.status == "skipped"
    assert entry.stage == "policy"
    assert entry.confidence is RepairConfidence.MEDIUM
    assert entry.details == "Blocked by repair policy: safe"
    assert target.read_text(encoding="utf-8") == original


def test_cli_audit_displays_skipped_repair(
    tmp_path,
    monkeypatch,
    capsys,
):
    from codeguardian.cli import main
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    target = tmp_path / "example.py"
    target.write_text(
        "values = [1 2, 3]\n",
        encoding="utf-8",
    )

    GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    ).repair(target)

    monkeypatch.setattr(
        "sys.argv",
        [
            "codeguardian",
            str(target),
            "--audit",
        ],
    )

    exit_code = main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "REPAIR AUDIT:" in output
    assert "[SKIPPED]" in output
    assert "Confidence: MEDIUM" in output
    assert "Stage: policy" in output
    assert "Blocked by repair policy: safe" in output
    assert "Total audit records: 1" in output


def test_cli_audit_handles_empty_log(
    tmp_path,
    monkeypatch,
    capsys,
):
    from codeguardian.cli import main

    target = tmp_path / "example.py"
    target.write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "codeguardian",
            str(target),
            "--audit",
        ],
    )

    exit_code = main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "No audit records." in output


def test_cli_version(capsys):
    from codeguardian.cli import main

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "codeguardian",
            "--version",
        ]
        try:
            main()
        except SystemExit as exc:
            assert exc.code == 0
    finally:
        sys.argv = old_argv

    output = capsys.readouterr().out
    assert "codeguardian 1.0.0" in output
