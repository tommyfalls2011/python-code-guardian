from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AIRepairProposal:
    """An untrusted complete-source repair proposed by an AI model."""

    file: Path
    original_source: str
    proposed_source: str
    reason: str
    model: str

    def validate_metadata(self) -> None:
        if not self.reason.strip():
            raise ValueError("AI proposal reason must not be empty")

        if not self.model.strip():
            raise ValueError("AI proposal model must not be empty")

        if self.proposed_source == self.original_source:
            raise ValueError("AI proposal does not change the source")

        if not self.file.exists():
            raise FileNotFoundError(self.file)

        current_source = self.file.read_text(encoding="utf-8")

        if current_source != self.original_source:
            raise ValueError(
                "Target file changed after the AI proposal was created"
            )
