from __future__ import annotations

import ast
from dataclasses import dataclass

from .proposal import AIRepairProposal


@dataclass(frozen=True)
class AIProposalValidation:
    """Result of validating an untrusted AI source proposal."""

    passed: bool
    stage: str
    details: str


def validate_ai_proposal(
    proposal: AIRepairProposal,
) -> AIProposalValidation:
    """Validate an AI proposal without modifying its target file."""

    try:
        proposal.validate_metadata()
    except (ValueError, FileNotFoundError) as exc:
        return AIProposalValidation(
            passed=False,
            stage="metadata",
            details=str(exc),
        )

    try:
        ast.parse(
            proposal.proposed_source,
            filename=str(proposal.file),
        )
    except SyntaxError as exc:
        return AIProposalValidation(
            passed=False,
            stage="syntax",
            details=f"{exc.msg} at line {exc.lineno}",
        )

    return AIProposalValidation(
        passed=True,
        stage="syntax",
        details="AI proposal parses successfully",
    )
