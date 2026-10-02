import pytest

from codeguardian.policy import RepairPolicy, parse_policy


def test_policy_values():
    assert RepairPolicy.SAFE.value == "safe"
    assert RepairPolicy.TESTED.value == "tested"
    assert RepairPolicy.OFF.value == "off"


def test_parse_policy():
    assert parse_policy("safe") is RepairPolicy.SAFE
    assert parse_policy("SAFE") is RepairPolicy.SAFE
    assert parse_policy("tested") is RepairPolicy.TESTED
    assert parse_policy("off") is RepairPolicy.OFF


def test_parse_policy_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown repair policy"):
        parse_policy("dangerous")
