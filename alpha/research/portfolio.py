"""Portfolio review: five lenses at once plus the core risk metrics.

Lenses: growth, risk, income, sector balance, and momentum. Metrics:
annualised volatility, Sharpe, beta, and maximum drawdown of the current
weights over trailing history. Concentration findings include the largest
weight, the Herfindahl index, average pairwise correlation, and the effective
number of independent bets (from the correlation matrix's eigenvalues), which
exposes "fifteen stocks that are really one bet".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha.research.market import bars_frame
from alpha.research.store import StoreView


@dataclass(frozen=True)
class PortfolioReview:
    metrics: dict[str, float]
    lenses: dict[str, str]
    findings: tuple[str, ...]

    def to_record(self) -> dict:
        return {
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "lenses": self.lenses,
            "findings": list(self.findings),
        }


def review_portfolio(
    view: StoreView,
    weights: dict[str, float],
    *,
    benchmark: str | None = None,
    window: int = 252,
    risk_free_rate: float = 0.0,
) -> PortfolioReview:
    if not weights:
        raise ValueError("portfolio is empty")
    total = sum(abs(w) for w in weights.values())
    weights = {t: w / total for t, w in weights.items()}
    closes = pd.DataFrame({t: bars_frame(view, t)["close"] for t in weights}).dropna().iloc[-window:]
    if len(closes) < 40:
        raise ValueError("not enough overlapping price history for a review")
    returns = closes.pct_change().dropna()
    vector = np.array([weights[t] for t in closes.columns])
    portfolio = returns.to_numpy() @ vector
    volatility = float(np.std(portfolio, ddof=1) * math.sqrt(252))
    excess = portfolio - risk_free_rate / 252
    sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(252)) if np.std(excess) > 0 else 0.0
    wealth = np.cumprod(1 + portfolio)
    drawdown = float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))
    metrics = {"volatility": volatility, "sharpe": sharpe, "max_drawdown": drawdown}
    if benchmark:
        bench = bars_frame(view, benchmark)["close"].pct_change().reindex(returns.index).dropna()
        if len(bench) > 40:
            aligned = pd.Series(portfolio, index=returns.index).loc[bench.index]
            metrics["beta"] = float(np.cov(aligned, bench)[0, 1] / np.var(bench, ddof=1))

    correlation = returns.corr().to_numpy()
    n = len(weights)
    if n > 1:
        off_diagonal = correlation[~np.eye(n, dtype=bool)]
        metrics["average_correlation"] = float(np.mean(off_diagonal))
        eigen = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
        metrics["effective_bets"] = float(eigen.sum() ** 2 / (eigen ** 2).sum())
    metrics["largest_weight"] = max(abs(w) for w in weights.values())
    metrics["herfindahl"] = float(sum(w * w for w in weights.values()))

    findings = []
    if n > 3 and metrics.get("effective_bets", n) < n / 3:
        findings.append(
            f"{n} holdings behave like {metrics['effective_bets']:.1f} independent bets; "
            "diversification is thinner than it looks."
        )
    if metrics["largest_weight"] > 0.25:
        findings.append(f"Largest position is {metrics['largest_weight']:.0%} of the book.")

    sectors: dict[str, float] = {}
    growth, yield_, payout_flags = [], [], []
    for ticker, weight in weights.items():
        filings = view.records("fundamentals", ticker)
        latest = max(filings, key=lambda f: (f.period_end, f.available_at)) if filings else None
        sectors[latest.sector if latest and latest.sector else "unclassified"] = (
            sectors.get(latest.sector if latest and latest.sector else "unclassified", 0.0) + abs(weight)
        )
        if latest and latest.revenue_growth_3y is not None:
            growth.append((weight, latest.revenue_growth_3y))
        if latest and latest.dividends_per_share and latest.shares_outstanding:
            price = float(closes[ticker].iloc[-1])
            yield_.append((weight, latest.dividends_per_share / price))
            if latest.net_income and latest.dividends_per_share * latest.shares_outstanding > latest.net_income:
                payout_flags.append(ticker)
    top_sector, top_share = max(sectors.items(), key=lambda item: item[1])
    trend_down = [
        t for t in closes.columns
        if closes[t].iloc[-1] < closes[t].rolling(50).mean().iloc[-1] < closes[t].rolling(min(150, len(closes) - 1)).mean().iloc[-1]
    ]
    lenses = {
        "growth": (
            f"Weighted revenue growth {sum(w * g for w, g in growth) / sum(abs(w) for w, _ in growth):.1%}."
            if growth else "No growth data on file."
        ),
        "risk": f"Volatility {volatility:.1%}, Sharpe {sharpe:.2f}, max drawdown {drawdown:.1%}.",
        "income": (
            f"Weighted dividend yield {sum(w * y for w, y in yield_):.2%}."
            + (f" Payout exceeds earnings at {', '.join(payout_flags)}." if payout_flags else "")
            if yield_ else "No dividend income."
        ),
        "sector_balance": f"Largest sector {top_sector} at {top_share:.0%} of gross exposure.",
        "momentum": (
            f"{len(trend_down)} of {n} holdings are in confirmed downtrends."
            if trend_down else "No holdings are in confirmed downtrends."
        ),
    }
    if top_share > 0.4:
        findings.append(f"Sector tilt: {top_sector} is {top_share:.0%} of the book.")
    findings.extend(f"{t} pays out more than it earns; the dividend is at risk." for t in payout_flags)
    return PortfolioReview(metrics=metrics, lenses=lenses, findings=tuple(findings))


__all__ = ["PortfolioReview", "review_portfolio"]
