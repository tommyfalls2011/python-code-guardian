from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .plan import RepairPlan
from .scanner import Diagnostic

Strategy = Callable[[Diagnostic], RepairPlan | None]
Matcher = Callable[[Diagnostic], bool]


@dataclass(frozen=True)
class StrategyRule:
    matcher: Matcher
    strategy: Strategy
    name: str


@dataclass
class RepairStrategyRegistry:
    """Registry of deterministic repair strategies."""

    strategies: dict[str, Strategy]
    rules: list[StrategyRule]

    def __init__(self) -> None:
        self.strategies = {}
        self.rules = []

    def register(self, message: str, strategy: Strategy) -> None:
        if not message.strip():
            raise ValueError("Diagnostic message must not be empty")

        if message in self.strategies:
            raise ValueError(
                f"Repair strategy already registered: {message!r}"
            )

        self.strategies[message] = strategy

    def register_matcher(
        self,
        name: str,
        matcher: Matcher,
        strategy: Strategy,
    ) -> None:
        if not name.strip():
            raise ValueError(
                "Strategy matcher name must not be empty"
            )

        if any(rule.name == name for rule in self.rules):
            raise ValueError(
                f"Repair strategy matcher already registered: {name!r}"
            )

        self.rules.append(
            StrategyRule(
                matcher=matcher,
                strategy=strategy,
                name=name,
            )
        )

    def resolve(self, diagnostic: Diagnostic) -> RepairPlan | None:
        strategy = self.strategies.get(diagnostic.message)

        if strategy is not None:
            return strategy(diagnostic)

        for rule in self.rules:
            if rule.matcher(diagnostic):
                return rule.strategy(diagnostic)

        return None

    def messages(self) -> tuple[str, ...]:
        return tuple(sorted(self.strategies))
