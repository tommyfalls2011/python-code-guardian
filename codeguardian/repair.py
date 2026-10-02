from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .confidence import RepairConfidence
from datetime import datetime, timezone


@dataclass(frozen=True)
class Repair:
    """Represent one proposed source-code change."""

    file: Path
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    replacement: str
    reason: str
    confidence: RepairConfidence = RepairConfidence.HIGH

    def validate(self) -> None:
        """Validate the repair coordinates and replacement."""
        if self.start_line < 1:
            raise ValueError("start_line must be >= 1")

        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")

        if self.start_column < 1:
            raise ValueError("start_column must be >= 1")

        if self.end_column < 1:
            raise ValueError("end_column must be >= 1")

        if not self.reason.strip():
            raise ValueError("reason must not be empty")


def apply_repair(source: str, repair: Repair) -> str:
    """Apply one validated repair to source text."""
    repair.validate()

    lines = source.splitlines(keepends=True)

    if repair.start_line > len(lines):
        raise ValueError("start_line is outside the source")

    if repair.end_line > len(lines):
        raise ValueError("end_line is outside the source")

    if repair.start_line != repair.end_line:
        raise ValueError("Multi-line repairs are not supported yet")

    line_index = repair.start_line - 1
    line = lines[line_index]

    start = repair.start_column - 1
    end = repair.end_column - 1

    if start > len(line):
        raise ValueError("start_column is outside the source")

    if end > len(line):
        raise ValueError("end_column is outside the source")

    lines[line_index] = line[:start] + repair.replacement + line[end:]

    return "".join(lines)


@dataclass(frozen=True)
class RepairResult:
    """Result of attempting a repair."""

    applied: bool
    original_source: str
    new_source: str
    reason: str


def repair_file(path: Path, repair: Repair) -> RepairResult:
    """Apply a repair only when the resulting source parses successfully."""
    import ast

    original_source = path.read_text(encoding="utf-8")

    if repair.file.resolve() != path.resolve():
        raise ValueError("Repair file does not match target path")

    new_source = apply_repair(original_source, repair)

    try:
        ast.parse(new_source, filename=str(path))
    except SyntaxError as exc:
        return RepairResult(
            applied=False,
            original_source=original_source,
            new_source=new_source,
            reason=f"Repair rejected: {exc.msg}",
        )

    path.write_text(new_source, encoding="utf-8")

    return RepairResult(
        applied=True,
        original_source=original_source,
        new_source=new_source,
        reason=repair.reason,
    )


def rollback_file(path: Path, original_source: str) -> None:
    """Restore a file to its original source."""
    path.write_text(original_source, encoding="utf-8")


def create_backup(path: Path) -> Path:
    """Create a unique backup beside the source file."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(
        f"{path.name}.guardian-backup-{timestamp}"
    )

    source = path.read_bytes()
    backup.write_bytes(source)

    return backup


def restore_backup(path: Path, backup: Path | None = None) -> None:
    """Restore a file from a specified or latest Guardian backup."""
    if backup is None:
        backups = sorted(
            path.parent.glob(f"{path.name}.guardian-backup-*")
        )

        if not backups:
            raise FileNotFoundError(
                f"No Guardian backup exists for: {path}"
            )

        backup = backups[-1]

    if not backup.exists():
        raise FileNotFoundError(f"Backup does not exist: {backup}")

    path.write_bytes(backup.read_bytes())


@dataclass(frozen=True)
class ManagedRepairResult:
    """Result of a managed repair operation."""

    applied: bool
    backup: Path | None
    reason: str


class RepairManager:
    """Safely manage source-code repairs."""

    def __init__(self, create_backups: bool = True) -> None:
        self.create_backups = create_backups

    def apply(self, repair: Repair) -> ManagedRepairResult:
        """Apply a repair with optional backup protection."""
        path = repair.file

        if not path.exists():
            raise FileNotFoundError(path)

        backup: Path | None = None

        if self.create_backups:
            backup = create_backup(path)

        result = repair_file(path, repair)

        if not result.applied and backup is not None:
            restore_backup(path, backup)

        return ManagedRepairResult(
            applied=result.applied,
            backup=backup,
            reason=result.reason,
        )

    def rollback(
        self,
        path: Path,
        backup: Path | None = None,
    ) -> None:
        """Rollback a file using a Guardian backup."""
        restore_backup(path, backup)
