"""Price-bar helpers shared by the research skills (point-in-time only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.research.store import StoreView


def bars_frame(view: StoreView, ticker: str) -> pd.DataFrame:
    key = ("bars", ticker)
    if key not in view.cache:
        view.cache[key] = _bars_frame(view, ticker)
    return view.cache[key]


def _bars_frame(view: StoreView, ticker: str) -> pd.DataFrame:
    rows = view.records("price_bar", ticker)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(
        {
            "open": [r.open for r in rows],
            "high": [r.high for r in rows],
            "low": [r.low for r in rows],
            "close": [r.close for r in rows],
            "volume": [r.volume for r in rows],
        },
        index=pd.DatetimeIndex([pd.Timestamp(r.event_at) for r in rows]),
    )
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def atr(frame: pd.DataFrame, window: int = 14) -> float:
    if len(frame) < 2:
        return float("nan")
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - previous).abs(), (frame["low"] - previous).abs()],
        axis=1,
    ).max(axis=1)
    return float(true_range.iloc[-window:].mean())


def rsi(close: pd.Series, window: int = 14) -> float:
    delta = close.diff().iloc[-window:]
    gains = delta.clip(lower=0).mean()
    losses = (-delta.clip(upper=0)).mean()
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    return float(100.0 - 100.0 / (1.0 + gains / losses))


def trailing_return(close: pd.Series, sessions: int) -> float | None:
    if len(close) <= sessions:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - sessions] - 1.0)


def beta(asset: pd.Series, benchmark: pd.Series, window: int = 252) -> float | None:
    joined = pd.concat([asset.pct_change(), benchmark.pct_change()], axis=1).dropna().iloc[-window:]
    if len(joined) < 40:
        return None
    variance = float(np.var(joined.iloc[:, 1]))
    if variance <= 0:
        return None
    return float(np.cov(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1] / variance)


__all__ = ["atr", "bars_frame", "beta", "rsi", "trailing_return"]
