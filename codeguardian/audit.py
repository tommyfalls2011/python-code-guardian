from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .confidence import RepairConfidence


@dataclass(frozen=True)
class RepairAuditEntry:
    audit_id: str
    timestamp: str
    status: str
    file: str
    line: int
    column: int
    reason: str
    confidence: RepairConfidence
    stage: str | None = None
    details: str | None = None


class RepairAudit:
    """Persist non-committed repair decisions."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(
        self,
        *,
        status: str,
        file: str,
        line: int,
        column: int,
        reason: str,
        confidence: RepairConfidence,
        stage: str | None = None,
        details: str | None = None,
    ) -> RepairAuditEntry:
        if not status.strip():
            raise ValueError("status must not be empty")

        if not isinstance(confidence, RepairConfidence):
            raise TypeError(
                "confidence must be a RepairConfidence"
            )

        entry = RepairAuditEntry(
            audit_id=uuid4().hex,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            file=file,
            line=line,
            column=column,
            reason=reason,
            confidence=confidence,
            stage=stage,
            details=details,
        )

        entries = self.load()
        entries.append(entry)

        self.path.write_text(
            json.dumps(
                [
                    {
                        **asdict(item),
                        "confidence": item.confidence.name,
                    }
                    for item in entries
                ],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        return entry

    def load(self) -> list[RepairAuditEntry]:
        if not self.path.exists():
            return []

        data: Any = json.loads(
            self.path.read_text(encoding="utf-8")
        )

        if not isinstance(data, list):
            raise ValueError(
                "Repair audit must contain a JSON list"
            )

        entries: list[RepairAuditEntry] = []

        for item in data:
            if not isinstance(item, dict):
                raise ValueError(
                    "Repair audit entries must be JSON objects"
                )

            entries.append(
                RepairAuditEntry(
                    audit_id=str(item["audit_id"]),
                    timestamp=str(item["timestamp"]),
                    status=str(item["status"]),
                    file=str(item["file"]),
                    line=int(item["line"]),
                    column=int(item["column"]),
                    reason=str(item["reason"]),
                    confidence=RepairConfidence[
                        str(item["confidence"]).upper()
                    ],
                    stage=(
                        None
                        if item.get("stage") is None
                        else str(item["stage"])
                    ),
                    details=(
                        None
                        if item.get("details") is None
                        else str(item["details"])
                    ),
                )
            )

        return entries
