import pytest
from pathlib import Path

from codeguardian.pipeline import GuardianPipeline


def write_broken_project(
    root: Path,
    test_body: str,
) -> Path:
    broken = root / "broken.py"

    broken.write_text(
        "def hello(name)\n"
        "    return f'Hello {name}'\n",
        encoding="utf-8",
    )

    (root / "test_broken.py").write_text(
        test_body,
        encoding="utf-8",
    )

    return broken


def test_pipeline_repairs_when_tests_pass(tmp_path: Path):
    broken = write_broken_project(
        tmp_path,
        (
            "from broken import hello\n\n"
            "def test_hello():\n"
            "    assert hello('World') == 'Hello World'\n"
        ),
    )

    result = GuardianPipeline(
        run_tests=True,
    ).repair(tmp_path)

    assert result.repairs_applied == 1
    assert result.report.passed is True
    assert "def hello(name):" in broken.read_text(
        encoding="utf-8"
    )


def test_pipeline_rolls_back_when_tests_fail(tmp_path: Path):
    broken = write_broken_project(
        tmp_path,
        (
            "def test_failure():\n"
            "    assert False\n"
        ),
    )

    original = broken.read_text(encoding="utf-8")

    result = GuardianPipeline(
        run_tests=True,
    ).repair(tmp_path)

    assert result.repairs_applied == 0
    assert broken.read_text(encoding="utf-8") == original


def test_pipeline_can_run_without_tests(tmp_path: Path):
    broken = tmp_path / "broken.py"

    broken.write_text(
        "def hello(name)\n"
        "    return name\n",
        encoding="utf-8",
    )

    result = GuardianPipeline(
        run_tests=False,
    ).repair(broken)

    assert result.repairs_applied == 1
    assert "def hello(name):" in broken.read_text(
        encoding="utf-8"
    )


def test_pipeline_rolls_back_without_backup(tmp_path: Path):
    broken = write_broken_project(
        tmp_path,
        (
            "def test_failure():\n"
            "    assert False\n"
        ),
    )

    original = broken.read_text(encoding="utf-8")

    result = GuardianPipeline(
        create_backups=False,
        run_tests=True,
    ).repair(tmp_path)

    assert result.repairs_applied == 0
    assert broken.read_text(encoding="utf-8") == original


def test_policy_off_does_not_apply_repairs(tmp_path):
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


def test_policy_safe_applies_deterministic_repairs(tmp_path):
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


def test_policy_tested_enables_test_validation(tmp_path):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    pipeline = GuardianPipeline(
        policy=RepairPolicy.TESTED,
    )

    assert pipeline.policy is RepairPolicy.TESTED
    assert pipeline.run_tests is True


def test_tested_policy_applies_medium_repair_when_tests_pass(tmp_path):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    target = tmp_path / "example.py"
    target.write_text(
        "values = [1 2, 3]\n",
        encoding="utf-8",
    )

    test_file = tmp_path / "test_example.py"
    test_file.write_text(
        "import example\n\n"
        "def test_values():\n"
        "    assert example.values == [1, 2, 3]\n",
        encoding="utf-8",
    )

    pipeline = GuardianPipeline(
        policy=RepairPolicy.TESTED,
        create_backups=False,
    )

    result = pipeline.repair(target)

    assert result.repairs_applied == 1
    assert target.read_text(encoding="utf-8") == (
        "values = [1, 2, 3]\n"
    )


def test_tested_policy_rolls_back_medium_repair_when_tests_fail(tmp_path):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    target = tmp_path / "example.py"
    original = "values = [1 2, 3]\n"
    target.write_text(original, encoding="utf-8")

    test_file = tmp_path / "test_example.py"
    test_file.write_text(
        "import example\n\n"
        "def test_values():\n"
        "    assert example.values == [99]\n",
        encoding="utf-8",
    )

    pipeline = GuardianPipeline(
        policy=RepairPolicy.TESTED,
        create_backups=False,
    )

    result = pipeline.repair(target)

    assert result.repairs_applied == 0
    assert target.read_text(encoding="utf-8") == original


def test_failed_tests_are_reported_as_rejected_repair(tmp_path):
    from codeguardian.confidence import RepairConfidence
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    target = tmp_path / "example.py"
    original = "values = [1 2, 3]\n"
    target.write_text(original, encoding="utf-8")

    test_file = tmp_path / "test_example.py"
    test_file.write_text(
        "import example\n\n"
        "def test_values():\n"
        "    assert example.values == [99]\n",
        encoding="utf-8",
    )

    pipeline = GuardianPipeline(
        policy=RepairPolicy.TESTED,
        create_backups=False,
    )

    result = pipeline.repair(target)

    assert result.repairs_applied == 0
    assert len(result.rejected_repairs) == 1

    rejected = result.rejected_repairs[0]

    assert rejected.stage == "tests"
    assert rejected.confidence is RepairConfidence.MEDIUM
    assert "missing comma" in rejected.reason.lower()
    assert rejected.details
    assert target.read_text(encoding="utf-8") == original


def test_successful_repair_has_no_rejection_record(tmp_path):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    target = tmp_path / "example.py"
    target.write_text(
        "def hello()\n"
        "    return 1\n",
        encoding="utf-8",
    )

    pipeline = GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    )

    result = pipeline.repair(target)

    assert result.repairs_applied == 1
    assert result.rejected_repairs == []


def test_cli_reports_test_rejected_repair(
    tmp_path,
    monkeypatch,
    capsys,
):
    from codeguardian.cli import main

    target = tmp_path / "example.py"
    original = "values = [1 2, 3]\n"
    target.write_text(original, encoding="utf-8")

    test_file = tmp_path / "test_example.py"
    test_file.write_text(
        "import example\n\n"
        "def test_values():\n"
        "    assert example.values == [99]\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "codeguardian",
            str(target),
            "--repair",
            "--policy",
            "tested",
        ],
    )

    exit_code = main()
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "[REJECTED MEDIUM]" in output
    assert "Stage: tests" in output
    assert "Repairs applied: 0" in output
    assert "Repairs rejected: 1" in output
    assert target.read_text(encoding="utf-8") == original


def test_test_rejection_is_persisted_to_audit(tmp_path):
    from codeguardian.audit import RepairAudit
    from codeguardian.confidence import RepairConfidence
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    target = tmp_path / "example.py"
    target.write_text(
        "values = [1 2, 3]\n",
        encoding="utf-8",
    )

    test_file = tmp_path / "test_example.py"
    test_file.write_text(
        "import example\n\n"
        "def test_values():\n"
        "    assert example.values == [99]\n",
        encoding="utf-8",
    )

    pipeline = GuardianPipeline(
        policy=RepairPolicy.TESTED,
        create_backups=False,
    )

    result = pipeline.repair(target)

    assert result.repairs_applied == 0

    entries = RepairAudit(
        tmp_path / ".guardian-audit.json"
    ).load()

    assert len(entries) == 1
    assert entries[0].status == "rejected"
    assert entries[0].stage == "tests"
    assert entries[0].confidence is RepairConfidence.MEDIUM
    assert entries[0].details


def test_pipeline_repairs_missing_import_comma(tmp_path):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    target = tmp_path / "example.py"
    target.write_text(
        "import os sys\n",
        encoding="utf-8",
    )

    pipeline = GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    )

    result = pipeline.repair(target)

    assert result.repairs_applied == 1
    assert result.skipped_repairs == []
    assert result.rejected_repairs == []
    assert result.report.error_count == 0

    assert target.read_text(
        encoding="utf-8"
    ) == "import os, sys\n"


def test_pipeline_repairs_from_import_missing_comma(tmp_path):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    target = tmp_path / "example.py"
    target.write_text(
        "from os import path environ\n",
        encoding="utf-8",
    )

    pipeline = GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    )

    result = pipeline.repair(target)

    assert result.repairs_applied == 1
    assert result.skipped_repairs == []
    assert result.rejected_repairs == []
    assert result.report.error_count == 0

    assert target.read_text(
        encoding="utf-8"
    ) == "from os import path, environ\n"


def test_import_comma_repair_is_recorded_in_history(tmp_path):
    from codeguardian.history import RepairHistory
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy
    from codeguardian.confidence import RepairConfidence

    target = tmp_path / "example.py"
    target.write_text(
        "import os sys\n",
        encoding="utf-8",
    )

    GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    ).repair(target)

    entries = RepairHistory(
        tmp_path / ".guardian-history.json"
    ).load()

    assert len(entries) == 1

    entry = entries[0]

    assert entry.confidence is RepairConfidence.HIGH
    assert "missing comma" in entry.reason.lower()
    assert entry.file == str(target.resolve())


def test_validation_error_rolls_back_repair_and_audits_it(
    tmp_path,
):
    from codeguardian.audit import RepairAudit
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    target = tmp_path / "example.py"

    original = (
        "def hello()\n"
        "    return definitely_missing\n"
    )

    target.write_text(
        original,
        encoding="utf-8",
    )

    pipeline = GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    )

    result = pipeline.repair(target)

    assert result.repairs_applied == 0
    assert len(result.rejected_repairs) == 1

    rejected = result.rejected_repairs[0]

    assert rejected.stage == "validation"
    assert "Undefined name" in rejected.details

    assert target.read_text(
        encoding="utf-8"
    ) == original

    entries = RepairAudit(
        tmp_path / ".guardian-audit.json"
    ).load()

    assert len(entries) == 1
    assert entries[0].status == "rejected"
    assert entries[0].stage == "validation"
    assert "Undefined name" in (entries[0].details or "")


def test_validation_warning_allows_repair_to_commit(
    tmp_path,
):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    target = tmp_path / "example.py"

    target.write_text(
        "import os sys\n",
        encoding="utf-8",
    )

    pipeline = GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    )

    result = pipeline.repair(target)

    assert result.repairs_applied == 1
    assert result.rejected_repairs == []

    assert target.read_text(
        encoding="utf-8"
    ) == "import os, sys\n"


def test_pipeline_repairs_missing_parameter_comma(tmp_path):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    target = tmp_path / "example.py"

    target.write_text(
        "def add(a b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )

    pipeline = GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    )

    result = pipeline.repair(target)

    assert result.repairs_applied == 1
    assert result.skipped_repairs == []
    assert result.rejected_repairs == []
    assert result.report.error_count == 0

    assert target.read_text(
        encoding="utf-8"
    ) == (
        "def add(a, b):\n"
        "    return a + b\n"
    )


def test_pipeline_repairs_async_parameter_comma(tmp_path):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    target = tmp_path / "example.py"

    target.write_text(
        "async def add(a b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )

    pipeline = GuardianPipeline(
        policy=RepairPolicy.SAFE,
        create_backups=False,
    )

    result = pipeline.repair(target)

    assert result.repairs_applied == 1
    assert result.rejected_repairs == []

    assert target.read_text(
        encoding="utf-8"
    ) == (
        "async def add(a, b):\n"
        "    return a + b\n"
    )


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
    ],
)
def test_pipeline_repairs_unmatched_closing_delimiter(
    tmp_path,
    source,
    expected,
):
    from codeguardian.pipeline import GuardianPipeline
    from codeguardian.policy import RepairPolicy

    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")

    pipeline = GuardianPipeline(
        policy=RepairPolicy.SAFE,
    )

    result = pipeline.repair(path)

    assert result.repairs_applied == 1
    assert result.report.passed
    assert result.skipped_repairs == []
    assert result.rejected_repairs == []
    assert path.read_text(encoding="utf-8") == expected
