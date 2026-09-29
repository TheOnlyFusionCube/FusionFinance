"""Price and volume jurors: published return anomalies on the tape alone."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha.verifier.committee.base import JurorSpec, JuryContext, Provenance


def _need(context: JuryContext, sessions: int) -> bool:
    return len(context.closes) > sessions


@dataclass
class CrossSectionalMomentum:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="momentum_12_1", family="momentum",
        provenance=Provenance(
            citation="Jegadeesh & Titman (1993), Returns to Buying Winners and Selling Losers, "
                     "Journal of Finance; Asness, Moskowitz & Pedersen (2013), Value and "
                     "Momentum Everywhere, Journal of Finance",
            institution="UCLA (Jegadeesh & Titman); AQR Capital Management (Asness et al.)",
            notes="twelve-month return skipping the most recent month",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        if not _need(context, 253):
            return pd.Series(np.nan, index=context.tickers)
        c = context.closes
        return c.iloc[-22] / c.iloc[-253] - 1.0


@dataclass
class TimeSeriesMomentum:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="time_series_momentum", family="momentum",
        provenance=Provenance(
            citation="Moskowitz, Ooi & Pedersen (2012), Time Series Momentum, "
                     "Journal of Financial Economics",
            institution="AQR Capital Management / NYU Stern / Chicago Booth",
            notes="own twelve-month excess return scaled by ex-ante volatility",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        if not _need(context, 253):
            return pd.Series(np.nan, index=context.tickers)
        c = context.closes
        vol = context.returns().iloc[-63:].std() * np.sqrt(252)
        return (c.iloc[-1] / c.iloc[-253] - 1.0) / vol.where(vol > 0)


@dataclass
class ShortTermReversal:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="short_term_reversal", family="reversal",
        provenance=Provenance(
            citation="Jegadeesh (1990), Evidence of Predictable Behavior of Security Returns, "
                     "Journal of Finance; Lehmann (1990), Fads, Martingales, and Market "
                     "Efficiency, Quarterly Journal of Economics",
            institution="UCLA; UC San Diego",
            notes="negative of the one-week return",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        if not _need(context, 6):
            return pd.Series(np.nan, index=context.tickers)
        c = context.closes
        return -(c.iloc[-1] / c.iloc[-6] - 1.0)


@dataclass
class FiftyTwoWeekHigh:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="fifty_two_week_high", family="momentum",
        provenance=Provenance(
            citation="George & Hwang (2004), The 52-Week High and Momentum Investing, "
                     "Journal of Finance",
            institution="University of Houston and co-author",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        if not _need(context, 252):
            return pd.Series(np.nan, index=context.tickers)
        c = context.closes
        return c.iloc[-1] / c.iloc[-252:].max()


def _betas(context: JuryContext, window: int) -> pd.Series:
    returns = context.returns().iloc[-window:]
    bench = context.benchmark.pct_change().iloc[-window:] if context.benchmark is not None else returns.mean(axis=1)
    variance = float(bench.var())
    if not np.isfinite(variance) or variance <= 0:
        return pd.Series(np.nan, index=context.tickers)
    return returns.apply(lambda column: column.cov(bench)) / variance


@dataclass
class BettingAgainstBeta:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="betting_against_beta", family="low_risk", needs=("prices", "benchmark"),
        provenance=Provenance(
            citation="Frazzini & Pedersen (2014), Betting Against Beta, "
                     "Journal of Financial Economics",
            institution="AQR Capital Management / NYU Stern",
            notes="beta shrunk 0.6/0.4 toward one, as in the paper; low beta is bullish",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        if not _need(context, 253):
            return pd.Series(np.nan, index=context.tickers)
        return -(0.6 * _betas(context, 252) + 0.4)


@dataclass
class LowIdiosyncraticVolatility:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="low_idiosyncratic_volatility", family="low_risk", needs=("prices", "benchmark"),
        provenance=Provenance(
            citation="Ang, Hodrick, Xing & Zhang (2006), The Cross-Section of Volatility "
                     "and Expected Returns, Journal of Finance",
            institution="Columbia Business School and co-authors",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        if not _need(context, 64):
            return pd.Series(np.nan, index=context.tickers)
        returns = context.returns().iloc[-63:]
        bench = context.benchmark.pct_change().iloc[-63:] if context.benchmark is not None else returns.mean(axis=1)
        beta = _betas(context, 63)
        residual = returns - np.outer(bench.to_numpy(), beta.to_numpy())
        return -residual.std()


@dataclass
class HighVolumeReturnPremium:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="high_volume_return_premium", family="volume", needs=("prices", "volume"),
        provenance=Provenance(
            citation="Gervais, Kaniel & Mingelgrin (2001), The High-Volume Return Premium, "
                     "Journal of Finance",
            institution="Wharton / UT Austin",
            notes="abnormal recent volume against the trailing quarter",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        if context.volume is None or not _need(context, 64):
            return pd.Series(np.nan, index=context.tickers)
        volume = context.volume
        return np.log(volume.iloc[-5:].mean() / volume.iloc[-63:-5].mean())


__all__ = [
    "BettingAgainstBeta", "CrossSectionalMomentum", "FiftyTwoWeekHigh",
    "HighVolumeReturnPremium", "LowIdiosyncraticVolatility", "ShortTermReversal",
    "TimeSeriesMomentum",
]
