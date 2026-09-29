"""Shared contracts for the verifier committee.

A juror is one published, independently reproducible method that scores every
ticker in the cross-section at a decision time using only point-in-time data.
Jurors never see LLM text, embeddings, or conviction. Higher scores are more
bullish; ``NaN`` means the juror abstains on that name.

Each juror carries its provenance. Provenance names the paper (and, where the
method comes from a practitioner research group, the institution the authors
published from). It never claims to be an institution's proprietary model:
banks do not publish their trading models. It records *whose published research*
the implementation follows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol

import numpy as np
import pandas as pd

from alpha.research.store import StoreView

Family = Literal[
    "momentum", "reversal", "low_risk", "volume", "quality", "value", "accounting",
    "events", "smart_money", "analysts", "network", "ml",
]


@dataclass(frozen=True)
class Provenance:
    citation: str                 # authors (year), title, venue
    institution: str              # author affiliation(s) as best known; not an endorsement
    url: str = ""
    notes: str = ""


@dataclass(frozen=True)
class JurorSpec:
    name: str
    family: Family
    provenance: Provenance
    needs: tuple[str, ...] = ("prices",)   # prices, volume, benchmark, fundamentals, events
    learned: bool = False                  # True if the juror trains a model


@dataclass
class JuryContext:
    """Everything a juror may read at one decision time; sliced so the future is absent."""

    index: int
    as_of: str
    closes: pd.DataFrame               # rows <= index only
    volume: pd.DataFrame | None
    benchmark: pd.Series | None
    view: StoreView | None
    extras: dict = field(default_factory=dict)

    @property
    def tickers(self) -> list[str]:
        return list(self.closes.columns)

    def returns(self) -> pd.DataFrame:
        cache = self.extras.setdefault("_cache", {})
        if "returns" not in cache:
            cache["returns"] = self.closes.pct_change()
        return cache["returns"]


class Juror(Protocol):
    spec: JurorSpec

    def score(self, context: JuryContext) -> pd.Series: ...


class LearnedJuror(Protocol):
    spec: JurorSpec

    def score(self, context: JuryContext) -> pd.Series: ...

    def fit(self, rows: list[tuple[JuryContext, pd.Series, pd.DataFrame | None]]) -> None: ...


def cross_sectional_z(values: pd.Series) -> pd.Series:
    """Rank-based z-score, robust to outliers; NaN stays NaN."""
    clean = values.replace([np.inf, -np.inf], np.nan)
    ranks = clean.rank(pct=True)
    count = int(ranks.notna().sum())
    if count < 3:
        return pd.Series(np.nan, index=values.index)
    centred = (ranks - 0.5 / count - 0.5) * np.sqrt(12.0)
    return centred.clip(-3, 3)


@dataclass
class JuryData:
    """Full point-in-time tape plus a factory for sliced contexts."""

    dates: pd.DatetimeIndex
    closes: pd.DataFrame
    volume: pd.DataFrame | None = None
    benchmark: pd.Series | None = None
    view_at: Callable[[str], StoreView] | None = None
    as_of_at: Callable[[int], str] | None = None
    market_forecasts: dict[int, dict] = field(default_factory=dict)
    _contexts: dict[int, JuryContext] = field(default_factory=dict, repr=False)

    def context(self, index: int) -> JuryContext:
        if index not in self._contexts:
            as_of = (
                self.as_of_at(index) if self.as_of_at
                else f"{self.dates[index].date().isoformat()}T21:00:00Z"
            )
            self._contexts[index] = JuryContext(
                index=index,
                as_of=as_of,
                closes=self.closes.iloc[: index + 1],
                volume=None if self.volume is None else self.volume.iloc[: index + 1],
                benchmark=None if self.benchmark is None else self.benchmark.iloc[: index + 1],
                view=self.view_at(as_of) if self.view_at else None,
                extras={"market_forecast": self.market_forecasts.get(index)},
            )
        context = self._contexts[index]
        context.extras["market_forecast"] = self.market_forecasts.get(index)
        return context

    def forget(self, before: int) -> None:
        for index in [i for i in self._contexts if i < before]:
            del self._contexts[index]


__all__ = [
    "Family", "JurorSpec", "Juror", "JuryContext", "JuryData", "LearnedJuror",
    "Provenance", "cross_sectional_z",
]
