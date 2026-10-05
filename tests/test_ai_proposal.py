from pathlib import Path

from codeguardian.ai import (
    AIRepairProposal,
    validate_ai_proposal,
)


def make_proposal(
    path: Path,
    proposed_source: str,
) -> AIRepairProposal:
    return AIRepairProposal(
        file=path,
        original_source=path.read_text(encoding="utf-8"),
        proposed_source=proposed_source,
        reason="AI test repair",
        model="test-model",
    )


def test_valid_ai_proposal_passes_without_modifying_file(tmp_path):
    path = tmp_path / "sample.py"
    original = "value = 1\n"
    path.write_text(original, encoding="utf-8")

    proposal = make_proposal(path, "value = 2\n")
    result = validate_ai_proposal(proposal)

    assert result.passed is True
    assert result.stage == "syntax"
    assert path.read_text(encoding="utf-8") == original


def test_ai_proposal_with_invalid_python_is_rejected(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("value = 1\n", encoding="utf-8")

    proposal = make_proposal(path, "def broken(:\n")
    result = validate_ai_proposal(proposal)

    assert result.passed is False
    assert result.stage == "syntax"


def test_ai_proposal_rejects_stale_original_source(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("value = 1\n", encoding="utf-8")

    proposal = make_proposal(path, "value = 2\n")
    path.write_text("value = 99\n", encoding="utf-8")

    result = validate_ai_proposal(proposal)

    assert result.passed is False
    assert result.stage == "metadata"


def test_ai_proposal_rejects_no_change(tmp_path):
    path = tmp_path / "sample.py"
    original = "value = 1\n"
    path.write_text(original, encoding="utf-8")

    proposal = make_proposal(path, original)
    result = validate_ai_proposal(proposal)

    assert result.passed is False
    assert result.stage == "metadata"


def test_ai_proposal_requires_model_name(tmp_path):
    path = tmp_path / "sample.py"
    original = "value = 1\n"
    path.write_text(original, encoding="utf-8")

    proposal = AIRepairProposal(
        file=path,
        original_source=original,
        proposed_source="value = 2\n",
        reason="repair",
        model="",
    )

    result = validate_ai_proposal(proposal)

    assert result.passed is False
    assert result.stage == "metadata"
