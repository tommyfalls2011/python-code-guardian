from __future__ import annotations

from dataclasses import dataclass

from .repair import Repair
from .scanner import Diagnostic


@dataclass(frozen=True)
class RepairPlan:
    """A proposed repair and the diagnostic it addresses."""

    diagnostic: Diagnostic
    repair: Repair
