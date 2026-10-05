from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..analyzer import analyze_file
from ..repair import create_backup, restore_backup
from ..scanner import Diagnostic
from ..validator import RepairValidator
from .edit import AIEdit, apply_ai_edit
from .localized import extract_source_window
from .ollama import request_ai_edit


@dataclass(frozen=True)
class AIEditEvaluation:
    accepted: bool
    reason: str
    edit: AIEdit | None
    candidate_source: str | None
    before_diagnostics: list[Diagnostic]
    after_diagnostics: list[Diagnostic]


AIEditProvider = Callable[..., AIEdit]


@dataclass(frozen=True)
class AIRepairTransaction:
    applied: bool
    reason: str
    backup: Path | None
    evaluation: AIEditEvaluation


def apply_evaluated_ai_edit(
    path: Path,
    evaluation: AIEditEvaluation,
) -> AIRepairTransaction:
    path = path.resolve()

    if not evaluation.accepted:
        return AIRepairTransaction(
            applied=False,
            reason="AI edit was not accepted",
            backup=None,
            evaluation=evaluation,
        )

    if evaluation.candidate_source is None:
        return AIRepairTransaction(
            applied=False,
            reason="accepted AI edit has no candidate source",
            backup=None,
            evaluation=evaluation,
        )

    original_source = path.read_text(encoding="utf-8")

    if not path.is_file():
        raise FileNotFoundError(path)

    backup = create_backup(path)

    try:
        path.write_text(
            evaluation.candidate_source,
            encoding="utf-8",
        )

        final_source = path.read_text(encoding="utf-8")

        if final_source != evaluation.candidate_source:
            restore_backup(path, backup)
            return AIRepairTransaction(
                applied=False,
                reason=(
                    "written source did not match candidate; "
                    "repair rolled back"
                ),
                backup=backup,
                evaluation=evaluation,
            )

        syntax = RepairValidator().validate_source(
            final_source,
            filename=str(path),
        )

        if not syntax.passed:
            restore_backup(path, backup)
            return AIRepairTransaction(
                applied=False,
                reason=(
                    "final syntax validation failed; "
                    "repair rolled back"
                ),
                backup=backup,
                evaluation=evaluation,
            )

        final_diagnostics = analyze_file(path).diagnostics

        expected = sorted(
            _diagnostic_key(item)
            for item in evaluation.after_diagnostics
        )
        actual = sorted(
            _diagnostic_key(item)
            for item in final_diagnostics
        )

        if actual != expected:
            restore_backup(path, backup)
            return AIRepairTransaction(
                applied=False,
                reason=(
                    "final diagnostics differed from approved "
                    "candidate; repair rolled back"
                ),
                backup=backup,
                evaluation=evaluation,
            )

    except Exception:
        try:
            restore_backup(path, backup)
        except Exception:
            path.write_text(original_source, encoding="utf-8")
        raise

    return AIRepairTransaction(
        applied=True,
        reason="AI repair applied and validated",
        backup=backup,
        evaluation=evaluation,
    )


def _diagnostic_key(diagnostic: Diagnostic) -> tuple[str, str]:
    return (
        diagnostic.severity.upper(),
        diagnostic.message,
    )


def _analyze_source(source: str) -> list[Diagnostic]:
    with tempfile.TemporaryDirectory(
        prefix="codeguardian-ai-"
    ) as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        return analyze_file(path).diagnostics


def _numbered_context(
    source: str,
    line: int,
    *,
    context_lines: int,
) -> str:
    window = extract_source_window(
        source,
        line,
        context=context_lines,
    )

    return "".join(
        f"{number}: {text}"
        for number, text in enumerate(
            window.source.splitlines(keepends=True),
            start=window.start_line,
        )
    )


def evaluate_ai_edit(
    *,
    source: str,
    diagnostic: Diagnostic,
    model: str,
    provider: AIEditProvider = request_ai_edit,
    context_lines: int = 3,
) -> AIEditEvaluation:
    before = _analyze_source(source)

    if diagnostic.message.startswith(
        "Mutable default argument"
    ):
        return AIEditEvaluation(
            accepted=False,
            reason=(
                "mutable-default repair requires a bounded "
                "multi-edit transaction"
            ),
            edit=None,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=before,
        )

    context = _numbered_context(
        source,
        diagnostic.line,
        context_lines=context_lines,
    )

    edit = provider(
        diagnostic=(
            f"{diagnostic.severity}: "
            f"line {diagnostic.line}: "
            f"{diagnostic.message}"
        ),
        line=diagnostic.line,
        context=context,
        model=model,
    )

    candidate = apply_ai_edit(source, edit)

    syntax = RepairValidator().validate_source(candidate)

    if not syntax.passed:
        return AIEditEvaluation(
            accepted=False,
            reason="candidate has invalid Python syntax",
            edit=edit,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=syntax.diagnostics,
        )

    after = _analyze_source(candidate)

    before_errors = {
        _diagnostic_key(item)
        for item in before
        if item.severity.upper() == "ERROR"
    }
    after_errors = {
        _diagnostic_key(item)
        for item in after
        if item.severity.upper() == "ERROR"
    }

    new_errors = after_errors - before_errors

    if new_errors:
        return AIEditEvaluation(
            accepted=False,
            reason="candidate introduces a new error",
            edit=edit,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=after,
        )

    target_key = _diagnostic_key(diagnostic)
    target_still_present = any(
        _diagnostic_key(item) == target_key
        for item in after
    )

    if target_still_present:
        return AIEditEvaluation(
            accepted=False,
            reason="target diagnostic was not repaired",
            edit=edit,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=after,
        )

    if len(after) >= len(before):
        return AIEditEvaluation(
            accepted=False,
            reason="candidate does not reduce diagnostics",
            edit=edit,
            candidate_source=None,
            before_diagnostics=before,
            after_diagnostics=after,
        )

    return AIEditEvaluation(
        accepted=True,
        reason="candidate safely reduces diagnostics",
        edit=edit,
        candidate_source=candidate,
        before_diagnostics=before,
        after_diagnostics=after,
    )
