import json

from codeguardian.audit import RepairAudit
from codeguardian.confidence import RepairConfidence


def test_audit_append_and_load(tmp_path):
    path = tmp_path / ".guardian-audit.json"
    audit = RepairAudit(path)

    audit.append(
        status="rejected",
        file="example.py",
        line=1,
        column=10,
        reason="test repair",
        confidence=RepairConfidence.MEDIUM,
        stage="tests",
        details="pytest failed",
    )

    entries = audit.load()

    assert len(entries) == 1

    entry = entries[0]

    assert entry.status == "rejected"
    assert entry.file == "example.py"
    assert entry.confidence is RepairConfidence.MEDIUM
    assert entry.stage == "tests"
    assert entry.details == "pytest failed"


def test_audit_writes_confidence_as_name(tmp_path):
    path = tmp_path / ".guardian-audit.json"
    audit = RepairAudit(path)

    audit.append(
        status="rejected",
        file="example.py",
        line=1,
        column=1,
        reason="test",
        confidence=RepairConfidence.HIGH,
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data[0]["confidence"] == "HIGH"
