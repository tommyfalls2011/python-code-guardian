from pathlib import Path

import pytest

from codeguardian.history import RepairHistory, RepairHistoryEntry


def test_history_starts_empty(tmp_path):
    history = RepairHistory(tmp_path / "history.json")

    assert history.load() == []


def test_history_append_and_load(tmp_path):
    path = tmp_path / "history.json"
    history = RepairHistory(path)

    entry = RepairHistoryEntry(
        repair_id="repair-001",
        timestamp="2026-10-01T17:00:00+00:00",
        file="/tmp/example.py",
        line=3,
        column=10,
        reason="Add missing colon.",
        backup="/tmp/example.py.guardian-backup",
    )

    history.append(entry)

    assert history.load() == [entry]


def test_history_preserves_multiple_entries(tmp_path):
    path = tmp_path / "history.json"
    history = RepairHistory(path)

    first = RepairHistoryEntry(
        repair_id="repair-001",
        timestamp="2026-10-01T17:00:00+00:00",
        file="/tmp/first.py",
        line=1,
        column=5,
        reason="First repair.",
        backup=None,
    )
    second = RepairHistoryEntry(
        repair_id="repair-002",
        timestamp="2026-10-01T17:01:00+00:00",
        file="/tmp/second.py",
        line=2,
        column=8,
        reason="Second repair.",
        backup="/tmp/second.backup",
    )

    history.append(first)
    history.append(second)

    assert history.load() == [first, second]


def test_history_generates_metadata(tmp_path):
    history = RepairHistory(tmp_path / "history.json")

    entry = history.append(
        file="/tmp/example.py",
        line=3,
        column=10,
        reason="Add missing colon.",
        backup=None,
    )

    assert entry.repair_id
    assert len(entry.repair_id) == 32
    assert entry.timestamp.endswith("+00:00")


def test_history_rejects_non_list(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{}\n", encoding="utf-8")

    history = RepairHistory(path)

    with pytest.raises(ValueError, match="JSON list"):
        history.load()


def test_history_persists_repair_confidence(tmp_path):
    from codeguardian.confidence import RepairConfidence
    from codeguardian.history import RepairHistory

    history = RepairHistory(tmp_path / "history.json")

    history.append(
        file="example.py",
        line=1,
        column=10,
        reason="test",
        confidence=RepairConfidence.MEDIUM,
    )

    entries = history.load()

    assert len(entries) == 1
    assert entries[0].confidence is RepairConfidence.MEDIUM


def test_history_old_entry_defaults_to_high_confidence(tmp_path):
    import json

    from codeguardian.confidence import RepairConfidence
    from codeguardian.history import RepairHistory

    path = tmp_path / "history.json"

    path.write_text(
        json.dumps(
            [
                {
                    "repair_id": "old-repair",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "file": "example.py",
                    "line": 1,
                    "column": 1,
                    "reason": "legacy repair",
                    "backup": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    entries = RepairHistory(path).load()

    assert len(entries) == 1
    assert entries[0].confidence is RepairConfidence.HIGH
