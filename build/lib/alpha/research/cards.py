"""Signal cards: the common output of every research skill.

A card is deterministic analytics over point-in-time records. Its ``lines`` are
the exact sentences an LLM may quote, and its ``facts`` are the trusted numbers
an LLM citation must reconcile against. Cards therefore double as the evidence
layer: whatever the LLM claims about a card is checked against the card.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

Direction = Literal["bullish", "bearish", "neutral"]
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True)
class SignalCard:
    skill: str
    ticker: str
    as_of: str
    score: float                       # [-1, 1], bearish to bullish
    confidence: float                  # [0, 1]
    headline: str
    lines: tuple[str, ...] = ()
    facts: dict[str, float] = field(default_factory=dict)
    flags: tuple[str, ...] = ()
    record_ids: tuple[str, ...] = ()
    role: str = "market"               # analyst-desk role that reads this card

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not -1.0 <= self.score <= 1.0:
            raise ValueError("card score must be finite in [-1, 1]")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("card confidence must be finite in [0, 1]")
        for key, value in self.facts.items():
            if _KEY.fullmatch(key) is None or not math.isfinite(value):
                raise ValueError(f"invalid card fact {key}")

    @property
    def direction(self) -> Direction:
        if self.score >= 0.15:
            return "bullish"
        if self.score <= -0.15:
            return "bearish"
        return "neutral"

    @property
    def active(self) -> bool:
        return self.confidence > 0.0 and self.direction != "neutral"

    def to_record(self) -> dict:
        return {
            "skill": self.skill,
            "direction": self.direction,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "headline": self.headline,
            "lines": list(self.lines),
            "facts": {key: round(value, 6) for key, value in sorted(self.facts.items())},
            "flags": list(self.flags),
            "records": len(self.record_ids),
        }


def clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def empty_card(skill: str, ticker: str, as_of: str, reason: str, role: str) -> SignalCard:
    return SignalCard(
        skill=skill, ticker=ticker, as_of=as_of, score=0.0, confidence=0.0,
        headline=reason, lines=(reason,), role=role,
    )


def fmt_money(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if value >= scale:
            return f"{sign}${value / scale:.1f}{unit}"
    return f"{sign}${value:.0f}"


__all__ = ["Direction", "SignalCard", "clip", "empty_card", "fmt_money"]
