"""Risk book: turn verified probabilities into constrained target weights.

Sizing is fractional Kelly on the verified win probability, scaled by the ML
verifier's conditional (aleatoric) volatility, then projected onto per-name,
gross, and net limits. A drawdown brake halves risk after a configured loss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SizedIdea:
    ticker: str
    side: int
    probability: float
    daily_volatility: float | None = None


@dataclass(frozen=True)
class RiskBook:
    max_position_weight: float = 0.1
    max_gross: float = 1.0
    max_net: float = 0.3
    kelly_fraction: float = 0.5
    target_daily_volatility: float = 0.012
    drawdown_brake: float = 0.08

    def __post_init__(self) -> None:
        values = (
            self.max_position_weight, self.max_gross, self.max_net,
            self.kelly_fraction, self.target_daily_volatility, self.drawdown_brake,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("risk limits must be finite and positive")
        if self.max_position_weight > self.max_gross:
            raise ValueError("max_position_weight cannot exceed max_gross")

    def weight(self, idea: SizedIdea) -> float:
        if idea.side not in (1, -1):
            raise ValueError("side must be +1 or -1")
        edge = 2.0 * idea.probability - 1.0
        if not math.isfinite(edge) or edge <= 0.0:
            return 0.0
        scale = 1.0
        if idea.daily_volatility is not None and idea.daily_volatility > 0.0:
            scale = min(2.0, max(0.25, self.target_daily_volatility / idea.daily_volatility))
        return idea.side * min(self.max_position_weight, self.kelly_fraction * edge * scale)

    def targets(self, ideas: list[SizedIdea], *, drawdown: float = 0.0) -> dict[str, float]:
        weights: dict[str, float] = {}
        for idea in ideas:
            value = self.weight(idea)
            if value != 0.0:
                weights[idea.ticker] = weights.get(idea.ticker, 0.0) + value
        weights = {
            ticker: max(-self.max_position_weight, min(self.max_position_weight, value))
            for ticker, value in weights.items()
        }
        if drawdown <= -self.drawdown_brake:
            weights = {ticker: value * 0.5 for ticker, value in weights.items()}
        net = sum(weights.values())
        if abs(net) > self.max_net:
            dominant = 1.0 if net > 0 else -1.0
            side_total = sum(value for value in weights.values() if value * dominant > 0)
            keep = (side_total - (abs(net) - self.max_net) * dominant) / side_total
            weights = {
                ticker: value * keep if value * dominant > 0 else value
                for ticker, value in weights.items()
            }
        gross = sum(abs(value) for value in weights.values())
        if gross > self.max_gross:
            # Leave a hair of headroom so rounding can never breach the kernel limit.
            factor = self.max_gross * (1.0 - 1e-9) / gross
            weights = {ticker: value * factor for ticker, value in weights.items()}
        return {ticker: round(value, 10) for ticker, value in sorted(weights.items()) if value}


__all__ = ["RiskBook", "SizedIdea"]
