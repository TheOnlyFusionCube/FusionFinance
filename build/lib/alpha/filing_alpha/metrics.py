"""Dependency-light performance metrics for the filing-alpha demo seam.

Adapted from filing-alpha-cleanroom 0.6.0 under the MIT License.  See NOTICE.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def equity_curve(returns: Iterable[float], compound: bool = True) -> np.ndarray:
    """Build a unit-capital equity curve from period returns."""
    values = np.asarray(list(returns), dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.cumprod(1.0 + values) if compound else 1.0 + np.cumsum(values)


def max_drawdown(returns: Iterable[float], compound: bool = True) -> float:
    """Return maximum peak-to-trough drawdown as a positive magnitude."""
    curve = equity_curve(returns, compound=compound)
    if curve.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(curve)
    drawdowns = curve / np.where(peaks == 0.0, 1.0, peaks) - 1.0
    return float(-np.min(drawdowns))


def annualized_sharpe(returns: Iterable[float], periods_per_year: int = 252) -> float:
    """Return the sample annualized Sharpe ratio with zero risk-free rate."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    values = np.asarray(list(returns), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    standard_deviation = values.std(ddof=1)
    if standard_deviation <= 1e-12:
        return 0.0
    return float(math.sqrt(periods_per_year) * values.mean() / standard_deviation)


def annualized_return(returns: Iterable[float], periods_per_year: int = 252) -> float:
    """Return compounded annualized return."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    values = np.asarray(list(returns), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    ending_value = float(np.prod(1.0 + values))
    if ending_value <= 0:
        return -1.0
    return ending_value ** (periods_per_year / values.size) - 1.0


def summarize_daily_returns(
    returns: pd.Series,
    trades: pd.DataFrame | None = None,
) -> dict[str, float | int]:
    """Summarize portfolio returns and an optional trade ledger."""
    values = returns.fillna(0.0).astype(float)
    result: dict[str, float | int] = {
        "total_return": float((1.0 + values).prod() - 1.0),
        "additive_pnl": float(values.sum()),
        "annualized_return": annualized_return(values),
        "annualized_sharpe": annualized_sharpe(values),
        "max_drawdown": max_drawdown(values),
        "daily_volatility": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "positive_day_rate": float((values > 0).mean()) if len(values) else 0.0,
    }
    if trades is not None and not trades.empty:
        required = {"net_return", "holding_sessions"}
        missing = sorted(required.difference(trades.columns))
        if missing:
            raise ValueError(f"trades is missing required columns: {missing}")
        result.update(
            {
                "trade_count": int(len(trades)),
                "win_rate": float((trades["net_return"] > 0).mean()),
                "average_trade_return": float(trades["net_return"].mean()),
                "average_holding_sessions": float(trades["holding_sessions"].mean()),
            }
        )
    else:
        result.update(
            {
                "trade_count": 0,
                "win_rate": 0.0,
                "average_trade_return": 0.0,
                "average_holding_sessions": 0.0,
            }
        )
    return result


def information_coefficient(frame: pd.DataFrame, prediction: str, target: str) -> float:
    """Return pooled Spearman rank correlation for prediction and target."""
    clean = frame[[prediction, target]].dropna()
    if len(clean) < 3:
        return 0.0
    return float(clean[prediction].corr(clean[target], method="spearman"))
