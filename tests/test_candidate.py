from pathlib import Path

import pytest

from codeguardian.candidate import RepairCandidate, build_candidate
from codeguardian.repair import Repair


def make_repair(path: Path, replacement: str) -> Repair:
    return Repair(
        file=path,
        start_line=1,
        start_column=16,
        end_line=1,
        end_column=16,
        replacement=replacement,
        reason="Test repair.",
    )


def test_build_candidate_does_not_modify_file(tmp_path):
    path = tmp_path / "example.py"
    source = "def hello(name)\n    return name\n"
    path.write_text(source, encoding="utf-8")

    repair = make_repair(path, ":")

    candidate = build_candidate(source, repair)

    assert path.read_text(encoding="utf-8") == source
    assert candidate.original_source == source
    assert candidate.proposed_source == (
        "def hello(name):\n    return name\n"
    )


def test_candidate_contains_repair_metadata(tmp_path):
    path = tmp_path / "example.py"
    source = "def hello(name)\n    return name\n"

    repair = make_repair(path, ":")

    candidate = build_candidate(source, repair)

    assert candidate.file == path
    assert candidate.reason == "Test repair."


def test_invalid_candidate_is_rejected(tmp_path):
    path = tmp_path / "example.py"
    source = "def hello(name):\n    return name\n"

    repair = Repair(
        file=path,
        start_line=1,
        start_column=1,
        end_line=1,
        end_column=2,
        replacement="def(",
        reason="Invalid test repair.",
    )

    with pytest.raises(SyntaxError):
        build_candidate(source, repair)


def test_candidate_validate_rechecks_source(tmp_path):
    path = tmp_path / "example.py"
    source = "def hello(name)\n    return name\n"

    repair = make_repair(path, ":")

    candidate = build_candidate(source, repair)

    candidate.validate()
