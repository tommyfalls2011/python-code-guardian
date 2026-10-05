from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audit import RepairAudit
from .ai.edit import apply_ai_edits
from .ai.engine import deterministic_repair_edits
from .repair import create_backup
from .candidate import build_candidate
from .confidence import RepairConfidence
from .history import RepairHistory
from .planner import RepairPlanner
from .policy import RepairPolicy
from .repair import RepairManager, rollback_file
from .report import Report
from .scanner import PythonScanner
from .test_runner import PythonTestRunner
from .validator import RepairValidator


@dataclass(frozen=True)
class AppliedRepair:
    """Record of one successfully applied repair."""

    file: Path
    line: int
    column: int
    reason: str
    backup: Path | None


@dataclass(frozen=True)
class SkippedRepair:
    """Repair that was planned but blocked by policy."""

    file: Path
    line: int
    column: int
    reason: str
    confidence: RepairConfidence


@dataclass(frozen=True)
class RejectedRepair:
    """Repair attempted but rejected by a safety gate."""

    file: Path
    line: int
    column: int
    reason: str
    confidence: RepairConfidence
    stage: str
    details: str


@dataclass(frozen=True)
class PipelineResult:
    """Result of one Guardian analysis/repair pass."""

    report: Report
    repairs_applied: int
    repairs: list[AppliedRepair]
    skipped_repairs: list[SkippedRepair]
    rejected_repairs: list[RejectedRepair]


class GuardianPipeline:
    """Run analysis and optional deterministic repairs."""

    def __init__(
        self,
        create_backups: bool = True,
        max_repairs: int = 10,
        run_tests: bool = False,
        test_timeout: float = 120.0,
        policy: RepairPolicy = RepairPolicy.SAFE,
    ) -> None:
        if max_repairs < 1:
            raise ValueError("max_repairs must be >= 1")

        self.max_repairs = max_repairs
        self.run_tests = run_tests

        if not isinstance(policy, RepairPolicy):
            raise TypeError("policy must be a RepairPolicy")

        self.policy = policy

        if self.policy is RepairPolicy.TESTED:
            self.run_tests = True
        self.planner = RepairPlanner()
        self.repair_manager = RepairManager(
            create_backups=create_backups,
        )
        self.validator = RepairValidator()
        self.test_runner = PythonTestRunner(
            timeout=test_timeout,
        )

    def scan(self, path: Path) -> Report:
        target = path.resolve()

        if target.is_file():
            scanner = PythonScanner(target.parent)
            files = [target]
        else:
            scanner = PythonScanner(target)
            files = scanner.find_python_files()

        diagnostics = []

        for file_path in files:
            syntax_diagnostics = scanner.check_file(file_path)

            if syntax_diagnostics:
                diagnostics.extend(syntax_diagnostics)
                continue

            diagnostics.extend(
                self._analyze(file_path)
            )

        return Report(diagnostics)

    @staticmethod
    def _analyze(path: Path):
        from .analyzer import analyze_file

        return analyze_file(path).diagnostics

    @staticmethod
    def _audit_path(path: Path) -> Path:
        target = path.resolve()

        if target.is_dir():
            return target / ".guardian-audit.json"

        return target.parent / ".guardian-audit.json"

    def _test_root(self, path: Path) -> Path:
        target = path.resolve()
        return target if target.is_dir() else target.parent

    def repair(self, path: Path) -> PipelineResult:
        report = self.scan(path)
        repairs_applied = 0
        repairs: list[AppliedRepair] = []
        skipped_repairs: list[SkippedRepair] = []
        rejected_repairs: list[RejectedRepair] = []

        for diagnostic in list(report.diagnostics):
            if repairs_applied >= self.max_repairs:
                break

            if self.policy is RepairPolicy.OFF:
                break

            plan = self.planner.plan(diagnostic)

            if plan is None:
                source = diagnostic.file.read_text(
                    encoding="utf-8"
                )
                deterministic_edits = deterministic_repair_edits(
                    source,
                    diagnostic,
                )

                if deterministic_edits is None:
                    continue

                try:
                    candidate_source = apply_ai_edits(
                        source,
                        deterministic_edits,
                        max_edits=3,
                    )
                except ValueError:
                    continue

                candidate_validation = self.validator.validate_source(
                    candidate_source,
                    filename=str(diagnostic.file),
                )
                if not candidate_validation.passed:
                    continue

                before_errors = {
                    (item.line, item.column, item.severity, item.message)
                    for item in report.diagnostics
                    if item.file.resolve() == diagnostic.file.resolve()
                    and item.severity.upper() == "ERROR"
                }

                candidate_path = diagnostic.file
                backup = (
                    create_backup(candidate_path)
                    if self.repair_manager.create_backups
                    else None
                )

                candidate_path.write_text(
                    candidate_source,
                    encoding="utf-8",
                )

                final_diagnostics = self._analyze(candidate_path)
                after_errors = {
                    (item.line, item.column, item.severity, item.message)
                    for item in final_diagnostics
                    if item.severity.upper() == "ERROR"
                }

                if after_errors - before_errors:
                    self._rollback(
                        candidate_path,
                        source,
                        backup,
                    )
                    continue

                if self.run_tests:
                    test_result = self.test_runner.run(
                        self._test_root(path)
                    )
                    if not test_result.passed:
                        self._rollback(
                            candidate_path,
                            source,
                            backup,
                        )
                        continue

                repairs_applied += 1
                reason = (
                    "Deterministic repair: "
                    f"{diagnostic.message}"
                )
                repairs.append(
                    AppliedRepair(
                        file=diagnostic.file,
                        line=diagnostic.line,
                        column=diagnostic.column,
                        reason=reason,
                        backup=backup,
                    )
                )

                history_path = (
                    path / ".guardian-history.json"
                    if path.resolve().is_dir()
                    else path.parent / ".guardian-history.json"
                )
                RepairHistory(history_path).append(
                    file=str(diagnostic.file),
                    line=diagnostic.line,
                    column=diagnostic.column,
                    reason=reason,
                    backup=(
                        None if backup is None else str(backup)
                    ),
                    confidence=RepairConfidence.HIGH,
                )
                RepairAudit(self._audit_path(path)).append(
                    status="applied",
                    file=str(diagnostic.file),
                    line=diagnostic.line,
                    column=diagnostic.column,
                    reason=reason,
                    confidence=RepairConfidence.HIGH,
                    stage="deterministic",
                    details="Bounded deterministic edit transaction",
                )
                continue

            if not self._confidence_allows(plan.repair.confidence):
                skipped = SkippedRepair(
                    file=diagnostic.file,
                    line=diagnostic.line,
                    column=diagnostic.column,
                    reason=plan.repair.reason,
                    confidence=plan.repair.confidence,
                )
                skipped_repairs.append(skipped)

                RepairAudit(self._audit_path(path)).append(
                    status="skipped",
                    file=str(skipped.file),
                    line=skipped.line,
                    column=skipped.column,
                    reason=skipped.reason,
                    confidence=skipped.confidence,
                    stage="policy",
                    details=(
                        f"Blocked by repair policy: "
                        f"{self.policy.value}"
                    ),
                )
                continue

            source = plan.repair.file.read_text(
                encoding="utf-8"
            )

            # Final candidate safety gate before touching the file.
            build_candidate(source, plan.repair)

            result = self.repair_manager.apply(plan.repair)

            if not result.applied:
                rejected = RejectedRepair(
                    file=diagnostic.file,
                    line=diagnostic.line,
                    column=diagnostic.column,
                    reason=plan.repair.reason,
                    confidence=plan.repair.confidence,
                    stage="apply",
                    details=result.reason,
                )
                rejected_repairs.append(rejected)

                RepairAudit(self._audit_path(path)).append(
                    status="rejected",
                    file=str(rejected.file),
                    line=rejected.line,
                    column=rejected.column,
                    reason=rejected.reason,
                    confidence=rejected.confidence,
                    stage=rejected.stage,
                    details=rejected.details,
                )
                continue

            verification = self.validator.validate(
                plan.repair.file
            )

            if not verification.passed:
                details = "; ".join(
                    diagnostic.message
                    for diagnostic in verification.diagnostics
                )

                rejected = RejectedRepair(
                    file=diagnostic.file,
                    line=diagnostic.line,
                    column=diagnostic.column,
                    reason=plan.repair.reason,
                    confidence=plan.repair.confidence,
                    stage="validation",
                    details=details or "Validation failed",
                )
                rejected_repairs.append(rejected)

                RepairAudit(self._audit_path(path)).append(
                    status="rejected",
                    file=str(rejected.file),
                    line=rejected.line,
                    column=rejected.column,
                    reason=rejected.reason,
                    confidence=rejected.confidence,
                    stage=rejected.stage,
                    details=rejected.details,
                )

                self._rollback(
                    plan.repair.file,
                    source,
                    result.backup,
                )
                continue

            if self.run_tests:
                test_result = self.test_runner.run(
                    self._test_root(path)
                )

                if not test_result.passed:
                    details = (
                        test_result.stderr.strip()
                        or test_result.stdout.strip()
                        or (
                            "Test suite failed with return code "
                            f"{test_result.returncode}"
                        )
                    )

                    rejected = RejectedRepair(
                        file=diagnostic.file,
                        line=diagnostic.line,
                        column=diagnostic.column,
                        reason=plan.repair.reason,
                        confidence=plan.repair.confidence,
                        stage="tests",
                        details=details,
                    )
                    rejected_repairs.append(rejected)

                    RepairAudit(self._audit_path(path)).append(
                        status="rejected",
                        file=str(rejected.file),
                        line=rejected.line,
                        column=rejected.column,
                        reason=rejected.reason,
                        confidence=rejected.confidence,
                        stage=rejected.stage,
                        details=rejected.details,
                    )

                    self._rollback(
                        plan.repair.file,
                        source,
                        result.backup,
                    )
                    continue

            repairs_applied += 1

            repairs.append(
                AppliedRepair(
                    file=diagnostic.file,
                    line=diagnostic.line,
                    column=diagnostic.column,
                    reason=result.reason,
                    backup=result.backup,
                )
            )

            history_path = (
                path / ".guardian-history.json"
                if path.resolve().is_dir()
                else path.parent / ".guardian-history.json"
            )

            RepairHistory(history_path).append(
                file=str(diagnostic.file),
                line=diagnostic.line,
                column=diagnostic.column,
                reason=result.reason,
                backup=(
                    None
                    if result.backup is None
                    else str(result.backup)
                ),
                confidence=plan.repair.confidence,
            )

        return PipelineResult(
            report=self.scan(path),
            repairs_applied=repairs_applied,
            repairs=repairs,
            skipped_repairs=skipped_repairs,
            rejected_repairs=rejected_repairs,
        )

    def _confidence_allows(
        self,
        confidence: RepairConfidence,
    ) -> bool:
        """Return whether policy permits automatic application."""

        if not isinstance(confidence, RepairConfidence):
            raise TypeError(
                "repair confidence must be a RepairConfidence"
            )

        if self.policy is RepairPolicy.OFF:
            return False

        if self.policy is RepairPolicy.SAFE:
            return confidence >= RepairConfidence.HIGH

        if self.policy is RepairPolicy.TESTED:
            return confidence >= RepairConfidence.MEDIUM

        return False

    def _rollback(
        self,
        path: Path,
        original_source: str,
        backup: Path | None,
    ) -> None:
        if backup is not None:
            self.repair_manager.rollback(
                path,
                backup,
            )
        else:
            rollback_file(
                path,
                original_source,
            )
