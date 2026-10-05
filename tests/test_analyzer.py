from pathlib import Path

from codeguardian.analyzer import analyze_file


def test_detects_duplicate_import(tmp_path: Path):
    source = tmp_path / "duplicate.py"

    source.write_text(
        """
import os
import os
""",
        encoding="utf-8",
    )

    result = analyze_file(source)

    assert any(
        diagnostic.severity == "WARNING"
        and "Duplicate import" in diagnostic.message
        for diagnostic in result.diagnostics
    )


def test_detects_wildcard_import(tmp_path: Path):
    source = tmp_path / "wildcard.py"

    source.write_text(
        """
from os import *
""",
        encoding="utf-8",
    )

    result = analyze_file(source)

    assert any(
        "Wildcard import" in diagnostic.message
        for diagnostic in result.diagnostics
    )


def test_detects_duplicate_definition(tmp_path: Path):
    source = tmp_path / "duplicate_function.py"

    source.write_text(
        """
def hello():
    pass


def hello():
    pass
""",
        encoding="utf-8",
    )

    result = analyze_file(source)

    assert any(
        "defined more than once" in diagnostic.message
        for diagnostic in result.diagnostics
    )


def test_detects_if_false(tmp_path: Path):
    source = tmp_path / "unreachable.py"

    source.write_text(
        """
if False:
    print("never runs")
""",
        encoding="utf-8",
    )

    result = analyze_file(source)

    assert any(
        "if False" in diagnostic.message
        for diagnostic in result.diagnostics
    )


def test_detects_while_true(tmp_path: Path):
    source = tmp_path / "loop.py"

    source.write_text(
        """
while True:
    break
""",
        encoding="utf-8",
    )

    result = analyze_file(source)

    assert any(
        "while True" in diagnostic.message
        for diagnostic in result.diagnostics
    )


def test_analysis_result_separates_errors_and_warnings(tmp_path):
    source = """\
import os
import os


def example():
    unused_value = 123
    return definitely_missing
"""
    path = tmp_path / "sample.py"
    path.write_text(source)

    from codeguardian.analyzer import analyze_file

    result = analyze_file(path)

    errors = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.severity == "ERROR"
    ]

    warnings = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.severity == "WARNING"
    ]

    assert any("Undefined name" in diagnostic.message for diagnostic in errors)
    assert any("Duplicate import" in diagnostic.message for diagnostic in warnings)
    assert any("Unused definition" in diagnostic.message for diagnostic in warnings)


def test_report_summarizes_diagnostics(tmp_path):
    from codeguardian.report import Report
    from codeguardian.scanner import Diagnostic

    path = tmp_path / "sample.py"

    diagnostics = [
        Diagnostic(path, 1, 1, "ERROR", "error"),
        Diagnostic(path, 2, 1, "WARNING", "warning"),
        Diagnostic(path, 3, 1, "WARNING", "warning 2"),
        Diagnostic(path, 4, 1, "INFO", "info"),
    ]

    report = Report(diagnostics)

    assert report.error_count == 1
    assert report.warning_count == 2
    assert report.info_count == 1
    assert report.total_count == 4
    assert not report.passed


def test_report_passes_without_errors(tmp_path):
    from codeguardian.report import Report
    from codeguardian.scanner import Diagnostic

    path = tmp_path / "sample.py"

    report = Report([
        Diagnostic(path, 1, 1, "WARNING", "warning"),
        Diagnostic(path, 2, 1, "INFO", "info"),
    ])

    assert report.passed


def test_apply_repair_replaces_source_range(tmp_path):
    from codeguardian.repair import Repair, apply_repair

    path = tmp_path / "sample.py"

    repair = Repair(
        file=path,
        start_line=1,
        start_column=1,
        end_line=1,
        end_column=4,
        replacement="print",
        reason="Fix function name",
    )

    result = apply_repair("foo('hello')\n", repair)

    assert result == "print('hello')\n"


def test_repair_rejects_invalid_coordinates(tmp_path):
    from codeguardian.repair import Repair

    path = tmp_path / "sample.py"

    repair = Repair(
        file=path,
        start_line=0,
        start_column=1,
        end_line=1,
        end_column=2,
        replacement="x",
        reason="test",
    )

    try:
        repair.validate()
    except ValueError as exc:
        assert "start_line" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_repair_file_applies_valid_repair(tmp_path):
    from codeguardian.repair import Repair, repair_file

    path = tmp_path / "sample.py"
    path.write_text("foo('hello')\n")

    repair = Repair(
        file=path,
        start_line=1,
        start_column=1,
        end_line=1,
        end_column=4,
        replacement="print",
        reason="Fix function name",
    )

    result = repair_file(path, repair)

    assert result.applied
    assert path.read_text() == "print('hello')\n"


def test_repair_file_rejects_invalid_result(tmp_path):
    from codeguardian.repair import Repair, repair_file

    path = tmp_path / "sample.py"
    original = "print('hello')\n"
    path.write_text(original)

    repair = Repair(
        file=path,
        start_line=1,
        start_column=1,
        end_line=1,
        end_column=6,
        replacement="def",
        reason="Intentional invalid repair",
    )

    result = repair_file(path, repair)

    assert not result.applied
    assert path.read_text() == original
    assert "Repair rejected" in result.reason


def test_rollback_file_restores_original(tmp_path):
    from codeguardian.repair import rollback_file

    path = tmp_path / "sample.py"
    original = "print('original')\n"

    path.write_text("print('changed')\n")
    rollback_file(path, original)

    assert path.read_text() == original


def test_create_backup(tmp_path):
    from codeguardian.repair import create_backup

    path = tmp_path / "sample.py"
    original = b"print('original')\n"
    path.write_bytes(original)

    backup = create_backup(path)

    assert backup.exists()
    assert backup.read_bytes() == original


def test_restore_backup(tmp_path):
    from codeguardian.repair import create_backup, restore_backup

    path = tmp_path / "sample.py"
    path.write_text("print('original')\n")

    create_backup(path)
    path.write_text("print('changed')\n")

    restore_backup(path)

    assert path.read_text() == "print('original')\n"


def test_restore_backup_requires_backup(tmp_path):
    from codeguardian.repair import restore_backup

    path = tmp_path / "sample.py"
    path.write_text("print('original')\n")

    try:
        restore_backup(path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_repair_manager_applies_with_backup(tmp_path):
    from codeguardian.repair import Repair, RepairManager

    path = tmp_path / "sample.py"
    path.write_text("foo('hello')\n")

    repair = Repair(
        file=path,
        start_line=1,
        start_column=1,
        end_line=1,
        end_column=4,
        replacement="print",
        reason="Fix function name",
    )

    manager = RepairManager()
    result = manager.apply(repair)

    assert result.applied
    assert result.backup is not None
    assert result.backup.exists()
    assert path.read_text() == "print('hello')\n"


def test_repair_manager_rolls_back_failed_repair(tmp_path):
    from codeguardian.repair import Repair, RepairManager

    path = tmp_path / "sample.py"
    original = "print('hello')\n"
    path.write_text(original)

    repair = Repair(
        file=path,
        start_line=1,
        start_column=1,
        end_line=1,
        end_column=6,
        replacement="def",
        reason="Invalid repair",
    )

    manager = RepairManager()
    result = manager.apply(repair)

    assert not result.applied
    assert path.read_text() == original


def test_repair_manager_manual_rollback(tmp_path):
    from codeguardian.repair import Repair, RepairManager

    path = tmp_path / "sample.py"
    path.write_text("print('original')\n")

    repair = Repair(
        file=path,
        start_line=1,
        start_column=1,
        end_line=1,
        end_column=6,
        replacement="print",
        reason="No-op repair",
    )

    manager = RepairManager()
    manager.apply(repair)

    path.write_text("print('changed')\n")
    manager.rollback(path)

    assert path.read_text() == "print('original')\n"


def test_repair_planner_can_plan_missing_colon(tmp_path):
    from codeguardian.planner import RepairPlanner
    from codeguardian.scanner import Diagnostic

    path = tmp_path / "broken.py"
    path.write_text("def hello(name)\n    return name\n")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=16,
        severity="ERROR",
        message="expected ':'",
    )

    plan = RepairPlanner().plan(diagnostic)

    assert plan is not None
    assert plan.repair.replacement == ":"
    assert "missing colon" in plan.repair.reason.lower()


def test_repair_planner_ignores_unsupported_diagnostic(tmp_path):
    from codeguardian.planner import RepairPlanner
    from codeguardian.scanner import Diagnostic

    path = tmp_path / "sample.py"
    path.write_text("print(missing)\n")

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=7,
        severity="ERROR",
        message="Undefined name: 'missing'",
    )

    assert RepairPlanner().plan(diagnostic) is None


def test_repair_planner_and_manager_fix_broken_file(tmp_path):
    from codeguardian.planner import RepairPlanner
    from codeguardian.repair import RepairManager
    from codeguardian.scanner import PythonScanner

    path = tmp_path / "broken.py"
    original = "def hello(name)\n    return name\n"
    path.write_text(original)

    scanner = PythonScanner(tmp_path)
    diagnostics = scanner.check_file(path)

    assert len(diagnostics) == 1
    assert diagnostics[0].message == "expected ':'"

    plan = RepairPlanner().plan(diagnostics[0])
    assert plan is not None

    result = RepairManager().apply(plan.repair)

    assert result.applied
    assert path.read_text() == "def hello(name):\n    return name\n"
    assert result.backup is not None
    assert result.backup.read_text() == original

    assert scanner.check_file(path) == []


def test_repair_manager_preserves_backup_on_success(tmp_path):
    from codeguardian.repair import Repair, RepairManager

    path = tmp_path / "sample.py"
    original = "foo('hello')\n"
    path.write_text(original)

    repair = Repair(
        file=path,
        start_line=1,
        start_column=1,
        end_line=1,
        end_column=4,
        replacement="print",
        reason="Fix function name",
    )

    result = RepairManager().apply(repair)

    assert result.applied
    assert result.backup is not None
    assert result.backup.read_text() == original
    assert path.read_text() == "print('hello')\n"


def test_create_backup_creates_unique_backups(tmp_path):
    from codeguardian.repair import create_backup

    path = tmp_path / "sample.py"
    path.write_text("version one\n")

    first = create_backup(path)

    path.write_text("version two\n")

    second = create_backup(path)

    assert first != second
    assert first.exists()
    assert second.exists()
    assert first.read_text() == "version one\n"
    assert second.read_text() == "version two\n"


def test_repair_planner_does_not_modify_source(tmp_path):
    from codeguardian.planner import RepairPlanner
    from codeguardian.scanner import Diagnostic

    path = tmp_path / "broken.py"
    original = "def hello(name)\n    return name\n"
    path.write_text(original)

    diagnostic = Diagnostic(
        file=path,
        line=1,
        column=16,
        severity="ERROR",
        message="expected ':'",
    )

    plan = RepairPlanner().plan(diagnostic)

    assert plan is not None
    assert path.read_text() == original


def test_pipeline_scan(tmp_path):
    from codeguardian.pipeline import GuardianPipeline

    path = tmp_path / "sample.py"
    path.write_text("print('hello')\n")

    result = GuardianPipeline().scan(path)

    assert result.total_count == 0
    assert result.passed


def test_pipeline_repairs_and_rescans(tmp_path):
    from codeguardian.pipeline import GuardianPipeline

    path = tmp_path / "broken.py"
    path.write_text("def hello(name)\n    return name\n")

    result = GuardianPipeline().repair(path)

    assert result.repairs_applied == 1
    assert result.report.passed
    assert path.read_text() == "def hello(name):\n    return name\n"


def test_pipeline_repairs_multiple_files(tmp_path):
    from codeguardian.pipeline import GuardianPipeline

    first = tmp_path / "first.py"
    second = tmp_path / "second.py"

    first.write_text("def first(value)\n    return value\n")
    second.write_text("def second(value)\n    return value\n")

    result = GuardianPipeline().repair(tmp_path)

    assert result.repairs_applied == 2
    assert result.report.passed

    assert first.read_text() == (
        "def first(value):\n"
        "    return value\n"
    )

    assert second.read_text() == (
        "def second(value):\n"
        "    return value\n"
    )


def test_pipeline_respects_repair_limit(tmp_path):
    from codeguardian.pipeline import GuardianPipeline

    first = tmp_path / "first.py"
    second = tmp_path / "second.py"

    first.write_text("def first(value)\n    return value\n")
    second.write_text("def second(value)\n    return value\n")

    result = GuardianPipeline(
        max_repairs=1,
    ).repair(tmp_path)

    assert result.repairs_applied == 1
    assert not result.report.passed


def test_pipeline_rejects_invalid_repair_limit():
    from codeguardian.pipeline import GuardianPipeline

    try:
        GuardianPipeline(max_repairs=0)
    except ValueError as exc:
        assert "max_repairs" in str(exc)
    else:
        raise AssertionError("Expected ValueError")



def test_detects_unreachable_code_after_return(tmp_path):
    path = tmp_path / "after_return.py"
    path.write_text("def example():\n    return 1\n    value = 2\n", encoding="utf-8")
    result = analyze_file(path)
    assert any("Unreachable code" in d.message and d.line == 3 for d in result.diagnostics)


def test_detects_unreachable_code_after_raise(tmp_path):
    path = tmp_path / "after_raise.py"
    path.write_text("def example():\n    raise RuntimeError()\n    value = 2\n", encoding="utf-8")
    result = analyze_file(path)
    assert any("Unreachable code" in d.message and d.line == 3 for d in result.diagnostics)


def test_detects_unreachable_code_after_break(tmp_path):
    path = tmp_path / "after_break.py"
    path.write_text("for value in range(3):\n    break\n    print(value)\n", encoding="utf-8")
    result = analyze_file(path)
    assert any("Unreachable code" in d.message and d.line == 3 for d in result.diagnostics)


def test_detects_unreachable_code_after_continue(tmp_path):
    path = tmp_path / "after_continue.py"
    path.write_text("for value in range(3):\n    continue\n    print(value)\n", encoding="utf-8")
    result = analyze_file(path)
    assert any("Unreachable code" in d.message and d.line == 3 for d in result.diagnostics)
