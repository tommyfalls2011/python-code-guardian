from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .repair import Repair, apply_repair


@dataclass(frozen=True)
class RepairCandidate:
    """A proposed source change that has not yet been applied."""

    repair: Repair
    original_source: str
    proposed_source: str

    def validate(self) -> None:
        """Validate that the proposed source is syntactically valid."""
        ast.parse(
            self.proposed_source,
            filename=str(self.repair.file),
        )

    @property
    def file(self) -> Path:
        return self.repair.file

    @property
    def reason(self) -> str:
        return self.repair.reason


def build_candidate(
    source: str,
    repair: Repair,
) -> RepairCandidate:
    """Build and validate a repair candidate without modifying files."""
    proposed_source = apply_repair(source, repair)

    candidate = RepairCandidate(
        repair=repair,
        original_source=source,
        proposed_source=proposed_source,
    )

    candidate.validate()
    return candidate
