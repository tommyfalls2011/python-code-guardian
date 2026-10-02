from __future__ import annotations

from enum import IntEnum


class RepairConfidence(IntEnum):
    """Confidence level for a proposed repair."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3


def parse_confidence(value: str) -> RepairConfidence:
    """Parse a confidence level from user/configuration input."""

    normalized = value.strip().lower()

    aliases = {
        "low": RepairConfidence.LOW,
        "medium": RepairConfidence.MEDIUM,
        "med": RepairConfidence.MEDIUM,
        "high": RepairConfidence.HIGH,
    }

    try:
        return aliases[normalized]
    except KeyError as exc:
        valid = ", ".join(
            level.name.lower()
            for level in RepairConfidence
        )
        raise ValueError(
            f"Unknown repair confidence {value!r}; "
            f"expected one of: {valid}"
        ) from exc
