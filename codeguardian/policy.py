from __future__ import annotations

from enum import Enum


class RepairPolicy(str, Enum):
    """Controls how much validation is required before applying repairs."""

    SAFE = "safe"
    TESTED = "tested"
    OFF = "off"


def parse_policy(value: str) -> RepairPolicy:
    """Convert a CLI/configuration value into a repair policy."""
    try:
        return RepairPolicy(value.lower())
    except ValueError as exc:
        valid = ", ".join(policy.value for policy in RepairPolicy)
        raise ValueError(
            f"Unknown repair policy {value!r}; expected one of: {valid}"
        ) from exc
