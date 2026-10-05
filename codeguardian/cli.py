from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .analyzer import analyze_file
from .ai.engine import (
    apply_evaluated_ai_edit,
    evaluate_ai_edit,
    evaluate_ai_edits,
)
from .ai.ollama import OllamaError
from .audit import RepairAudit
from .history import RepairHistory
from .pipeline import GuardianPipeline
from .policy import RepairPolicy, parse_policy
from .report import Report
from .scanner import PythonScanner


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="codeguardian",
        description="Python 3 code checker and repair system.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"codeguardian {__version__}",
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Python file or project directory to scan.",
    )

    parser.add_argument(
        "--repair",
        action="store_true",
        help="Apply supported deterministic repairs automatically.",
    )

    parser.add_argument(
        "--max-repairs",
        type=int,
        default=10,
        help="Maximum automatic repairs per run (default: 10).",
    )

    parser.add_argument(
        "--ai-repair",
        action="store_true",
        help="Use a local AI model for bounded repairs.",
    )

    parser.add_argument(
        "--ai-model",
        default="qwen2.5-coder:7b",
        help=(
            "Ollama model for --ai-repair "
            "(default: qwen2.5-coder:7b)."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show supported repairs without modifying files.",
    )

    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the project's pytest suite after each repair.",
    )

    parser.add_argument(
        "--policy",
        choices=[policy.value for policy in RepairPolicy],
        default=RepairPolicy.SAFE.value,
        help="Repair policy: safe, tested, or off.",
    )

    parser.add_argument(
        "--history",
        action="store_true",
        help="Show repair history for the target.",
    )

    parser.add_argument(
        "--audit",
        action="store_true",
        help="Show skipped and rejected repair audit records.",
    )

    args = parser.parse_args()

    if args.repair and args.ai_repair:
        parser.error(
            "--repair and --ai-repair cannot be used together"
        )

    target = args.path.resolve()

    if not target.exists():
        print(f"ERROR: Path does not exist: {target}")
        return 2

    if args.audit:
        audit_path = (
            target / ".guardian-audit.json"
            if target.is_dir()
            else target.parent / ".guardian-audit.json"
        )

        entries = RepairAudit(audit_path).load()

        print("=" * 60)
        print(" PYTHON CODE GUARDIAN")
        print("=" * 60)
        print()
        print(f"Target: {target}")
        print()
        print("REPAIR AUDIT:")
        print()

        if not entries:
            print("No audit records.")
            return 0

        for entry in entries:
            print(
                f"[{entry.status.upper()}] "
                f"{entry.audit_id} {entry.timestamp}"
            )
            print(f"    File: {entry.file}")
            print(
                f"    Location: {entry.line}:{entry.column}"
            )
            print(f"    Reason: {entry.reason}")
            print(
                f"    Confidence: {entry.confidence.name}"
            )

            if entry.stage is not None:
                print(f"    Stage: {entry.stage}")

            if entry.details is not None:
                print(f"    Details: {entry.details}")

            print()

        print("-" * 60)
        print(f"Total audit records: {len(entries)}")
        return 0

    if args.history:
        history_path = (
            target / ".guardian-history.json"
            if target.is_dir()
            else target.parent / ".guardian-history.json"
        )

        history = RepairHistory(history_path).load()

        print("=" * 60)
        print(" PYTHON CODE GUARDIAN")
        print("=" * 60)
        print()
        print(f"Target: {target}")
        print()
        print("REPAIR HISTORY:")
        print()

        if not history:
            print("No repairs recorded.")
            return 0

        for entry in history:
            print(f"[{entry.repair_id}] {entry.timestamp}")
            print(f"    File: {entry.file}")
            print(f"    Location: {entry.line}:{entry.column}")
            print(f"    Reason: {entry.reason}")
            print(f"    Confidence: {entry.confidence.name}")

            if entry.backup is not None:
                print(f"    Backup: {entry.backup}")

            print()

        print("-" * 60)
        print(f"Total repairs: {len(history)}")
        return 0

    if target.is_file():
        scanner = PythonScanner(target.parent)
        files = [target]
    else:
        scanner = PythonScanner(target)
        files = scanner.find_python_files()

    print("=" * 60)
    print(" PYTHON CODE GUARDIAN")
    print("=" * 60)
    print()
    print(f"Target: {target}")
    print(f"Python files: {len(files)}")
    print(f"Repair policy: {args.policy}")
    print()

    diagnostics = []

    for path in files:
        syntax_diagnostics = scanner.check_file(path)

        if syntax_diagnostics:
            diagnostics.extend(syntax_diagnostics)
            continue

        analysis = analyze_file(path)
        diagnostics.extend(analysis.diagnostics)

    report = Report(diagnostics)

    if args.dry_run:
        planner = GuardianPipeline(
            max_repairs=args.max_repairs,
        ).planner

        print("DRY RUN:")
        print()

        planned = 0

        for diagnostic in report.diagnostics:
            if planned >= args.max_repairs:
                break

            plan = planner.plan(diagnostic)

            if plan is None:
                continue

            planned += 1

            print(
                f"[WOULD REPAIR] {diagnostic.file}:"
                f"{diagnostic.line}:{diagnostic.column}"
            )
            print(f"    {plan.repair.reason}")
            print(
                f"    Confidence: {plan.repair.confidence.name}"
            )
            print(f"    Replacement: {plan.repair.replacement!r}")
            print()

        print(f"Repairs available: {planned}")
        print()

    elif args.ai_repair and report.total_count:
        print("AI REPAIR MODE:")
        print(f"Model: {args.ai_model}")
        print()

        repairs_applied = 0
        rejected_attempts = set()

        while (
            report.total_count
            and repairs_applied < args.max_repairs
        ):
            repair_diagnostics = sorted(
                report.diagnostics,
                key=lambda item: {
                    "ERROR": 0,
                    "WARNING": 1,
                }.get(item.severity.upper(), 2),
            )

            applied_this_pass = False

            for diagnostic in repair_diagnostics:
                if diagnostic.severity.upper() not in {
                    "ERROR",
                    "WARNING",
                }:
                    continue

                source = diagnostic.file.read_text(
                    encoding="utf-8"
                )

                rejection_key = (
                    diagnostic.file,
                    diagnostic.line,
                    diagnostic.column,
                    diagnostic.severity.upper(),
                    diagnostic.message,
                    source,
                )

                if rejection_key in rejected_attempts:
                    continue

                try:
                    if diagnostic.message.startswith(
                        "Mutable default argument"
                    ):
                        evaluation = evaluate_ai_edits(
                            source=source,
                            diagnostic=diagnostic,
                            model=args.ai_model,
                        )
                    else:
                        evaluation = evaluate_ai_edit(
                            source=source,
                            diagnostic=diagnostic,
                            model=args.ai_model,
                        )
                except (
                    OllamaError,
                    ValueError,
                    OSError,
                ) as exc:
                    print(
                        f"[AI REJECTED] {diagnostic.file}:"
                        f"{diagnostic.line}:"
                        f"{diagnostic.column}"
                    )
                    print(f"    {exc}")
                    print()
                    rejected_attempts.add(
                        rejection_key
                    )
                    continue

                if not evaluation.accepted:
                    print(
                        f"[AI REJECTED] {diagnostic.file}:"
                        f"{diagnostic.line}:"
                        f"{diagnostic.column}"
                    )
                    print(f"    {evaluation.reason}")
                    print()
                    rejected_attempts.add(
                        rejection_key
                    )
                    continue

                transaction = apply_evaluated_ai_edit(
                    diagnostic.file,
                    evaluation,
                )

                if not transaction.applied:
                    print(
                        f"[AI REJECTED] {diagnostic.file}:"
                        f"{diagnostic.line}:"
                        f"{diagnostic.column}"
                    )
                    print(f"    {transaction.reason}")
                    print()
                    rejected_attempts.add(
                        rejection_key
                    )
                    continue

                print(
                    f"[AI REPAIRED] {diagnostic.file}:"
                    f"{diagnostic.line}:"
                    f"{diagnostic.column}"
                )

                if hasattr(evaluation, "edits"):
                    for edit in evaluation.edits:
                        print(
                            f"    Operation: {edit.operation}"
                        )
                        print(f"    Line: {edit.line}")
                else:
                    print(
                        f"    Operation: "
                        f"{evaluation.edit.operation}"
                    )
                    print(
                        f"    Line: {evaluation.edit.line}"
                    )

                if transaction.backup is not None:
                    print(
                        f"    Backup: {transaction.backup}"
                    )

                print()

                repairs_applied += 1
                applied_this_pass = True

                diagnostics = []

                for current_path in files:
                    syntax_diagnostics = scanner.check_file(
                        current_path
                    )

                    if syntax_diagnostics:
                        diagnostics.extend(
                            syntax_diagnostics
                        )
                        continue

                    diagnostics.extend(
                        analyze_file(
                            current_path
                        ).diagnostics
                    )

                report = Report(diagnostics)

                # Diagnostics and line numbers may have changed.
                # Restart from the freshly analyzed report.
                break

            if not applied_this_pass:
                break

        if repairs_applied == 0:
            print("AI repairs applied: 0")
            print()
        else:
            print(
                f"AI repairs applied: {repairs_applied}"
            )
            print()

    elif args.repair and report.total_count:
        pipeline = GuardianPipeline(
            max_repairs=args.max_repairs,
            run_tests=args.run_tests,
            policy=parse_policy(args.policy),
        )

        result = pipeline.repair(target)

        print("REPAIR MODE:")
        print()

        for repair in result.repairs:
            print(
                f"[REPAIRED] {repair.file}:"
                f"{repair.line}:{repair.column}"
            )
            print(f"    {repair.reason}")

            if repair.backup is not None:
                print(f"    Backup: {repair.backup}")

            print()

        for skipped in result.skipped_repairs:
            print(
                f"[SKIPPED {skipped.confidence.name}] "
                f"{skipped.file}:"
                f"{skipped.line}:{skipped.column}"
            )
            print(f"    {skipped.reason}")
            print(
                f"    Blocked by repair policy: {args.policy}"
            )
            print()

        for rejected in result.rejected_repairs:
            print(
                f"[REJECTED {rejected.confidence.name}] "
                f"{rejected.file}:"
                f"{rejected.line}:{rejected.column}"
            )
            print(f"    {rejected.reason}")
            print(f"    Stage: {rejected.stage}")
            print(f"    Details: {rejected.details}")
            print()

        print(f"Repairs applied: {result.repairs_applied}")
        print(
            f"Repairs skipped: {len(result.skipped_repairs)}"
        )
        print(
            f"Repairs rejected: {len(result.rejected_repairs)}"
        )
        print()

        diagnostics = result.report.diagnostics
        report = result.report

    if report.total_count == 0:
        print("PASS: No problems found.")
        return 0

    print("PROBLEMS FOUND:")
    print()

    for diagnostic in report.diagnostics:
        print(
            f"[{diagnostic.severity}] "
            f"{diagnostic.file}:{diagnostic.line}:{diagnostic.column}"
        )
        print(f"    {diagnostic.message}")
        print()

    print("-" * 60)
    print(f"Errors:   {report.error_count}")
    print(f"Warnings: {report.warning_count}")
    print(f"Info:     {report.info_count}")
    print(f"Total:    {report.total_count}")

    return 1 if not report.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
