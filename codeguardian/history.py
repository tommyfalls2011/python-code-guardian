from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .confidence import RepairConfidence


@dataclass(frozen=True)
class RepairHistoryEntry:
    repair_id: str
    timestamp: str
    file: str
    line: int
    column: int
    reason: str
    backup: str | None
    confidence: RepairConfidence = RepairConfidence.HIGH


class RepairHistory:
    """Persist successful repair records as JSON."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(
        self,
        entry: RepairHistoryEntry | None = None,
        *,
        file: str | None = None,
        line: int | None = None,
        column: int | None = None,
        reason: str | None = None,
        backup: str | None = None,
        confidence: RepairConfidence = RepairConfidence.HIGH,
    ) -> RepairHistoryEntry:
        if entry is None:
            if (
                file is None
                or line is None
                or column is None
                or reason is None
            ):
                raise ValueError(
                    "file, line, column, and reason are required"
                )

            entry = RepairHistoryEntry(
                repair_id=uuid4().hex,
                timestamp=datetime.now(timezone.utc).isoformat(),
                file=file,
                line=line,
                column=column,
                reason=reason,
                backup=backup,
                confidence=confidence,
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

    def load(self) -> list[RepairHistoryEntry]:
        if not self.path.exists():
            return []

        data: Any = json.loads(
            self.path.read_text(encoding="utf-8")
        )

        if not isinstance(data, list):
            raise ValueError("Repair history must contain a JSON list")

        entries: list[RepairHistoryEntry] = []

        for item in data:
            if not isinstance(item, dict):
                raise ValueError(
                    "Repair history entries must be JSON objects"
                )

            entries.append(
                RepairHistoryEntry(
                    repair_id=str(item["repair_id"]),
                    timestamp=str(item["timestamp"]),
                    file=str(item["file"]),
                    line=int(item["line"]),
                    column=int(item["column"]),
                    reason=str(item["reason"]),
                    backup=(
                        None
                        if item.get("backup") is None
                        else str(item["backup"])
                    ),
                    confidence=RepairConfidence[
                        str(item.get("confidence", "HIGH")).upper()
                    ],
                )
            )

        return entries
