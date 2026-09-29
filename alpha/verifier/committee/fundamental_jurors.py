"""Accounting, event, and information-flow jurors (read the PIT research store)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from alpha.research.records import parse_timestamp
from alpha.verifier.committee.base import JurorSpec, JuryContext, Provenance, cross_sectional_z

_RATING = {"strong_buy": 2, "buy": 1, "hold": 0, "sell": -1, "strong_sell": -2}


def _filings(context: JuryContext, ticker: str, count: int = 2) -> list:
    if context.view is None:
        return []
    rows = context.view.records("fundamentals", ticker)
    by_period = {}
    for row in rows:                       # later filings for a period supersede earlier ones
        by_period[row.period_end] = row
    return [by_period[key] for key in sorted(by_period)][-count:]


def _per_ticker(context: JuryContext, compute) -> pd.Series:
    values = {}
    for ticker in context.tickers:
        try:
            values[ticker] = compute(ticker)
        except (TypeError, ZeroDivisionError, ValueError):
            values[ticker] = np.nan
    return pd.Series(values, dtype=float)


def _ratio(numerator, denominator) -> float:
    if numerator is None or denominator in (None, 0):
        return np.nan
    return float(numerator) / float(denominator)


@dataclass
class GrossProfitability:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="gross_profitability", family="quality", needs=("fundamentals",),
        provenance=Provenance(
            citation="Novy-Marx (2013), The Other Side of Value: The Gross Profitability "
                     "Premium, Journal of Financial Economics",
            institution="University of Rochester",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        def compute(ticker):
            filings = _filings(context, ticker, 1)
            return _ratio(filings[-1].gross_profit, filings[-1].total_assets) if filings else np.nan
        return _per_ticker(context, compute)


@dataclass
class QualityMinusJunk:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="quality_minus_junk", family="quality", needs=("fundamentals", "prices"),
        provenance=Provenance(
            citation="Asness, Frazzini & Pedersen (2019), Quality Minus Junk, "
                     "Review of Accounting Studies",
            institution="AQR Capital Management",
            notes="z(profitability) + z(growth) + z(safety), a compact version of the paper's composite",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        rows = {}
        for ticker in context.tickers:
            filings = _filings(context, ticker, 2)
            if not filings:
                continue
            f = filings[-1]
            rows[ticker] = {
                "gpoa": _ratio(f.gross_profit, f.total_assets),
                "roe": _ratio(f.net_income, f.equity),
                "cfoa": _ratio(f.operating_cash_flow, f.total_assets),
                "growth": (
                    _ratio(f.gross_profit, filings[0].gross_profit) - 1.0
                    if len(filings) == 2 and filings[0].gross_profit else np.nan
                ),
                "leverage": -_ratio(f.total_debt, f.total_assets),
            }
        if not rows:
            return pd.Series(np.nan, index=context.tickers)
        frame = pd.DataFrame(rows).T
        z = frame.apply(cross_sectional_z)
        profitability = z[["gpoa", "roe", "cfoa"]].mean(axis=1)
        vol = context.returns().iloc[-252:].std()
        safety = pd.concat([z["leverage"], cross_sectional_z(-vol.reindex(frame.index))], axis=1).mean(axis=1)
        composite = pd.concat([profitability, z["growth"], safety], axis=1).mean(axis=1)
        return composite.reindex(context.tickers)


@dataclass
class PiotroskiFScore:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="piotroski_f_score", family="accounting", needs=("fundamentals",),
        provenance=Provenance(
            citation="Piotroski (2000), Value Investing: The Use of Historical Financial "
                     "Statement Information to Separate Winners from Losers, "
                     "Journal of Accounting Research",
            institution="University of Chicago",
            notes="nine binary signals on profitability, leverage/liquidity, and efficiency",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        def compute(ticker):
            filings = _filings(context, ticker, 2)
            if len(filings) < 2:
                return np.nan
            prior, now = filings
            roa, roa0 = _ratio(now.net_income, now.total_assets), _ratio(prior.net_income, prior.total_assets)
            cfo = _ratio(now.operating_cash_flow, now.total_assets)
            lev, lev0 = _ratio(now.total_debt, now.total_assets), _ratio(prior.total_debt, prior.total_assets)
            cur = _ratio(now.current_assets, now.current_liabilities)
            cur0 = _ratio(prior.current_assets, prior.current_liabilities)
            margin, margin0 = _ratio(now.gross_profit, now.revenue), _ratio(prior.gross_profit, prior.revenue)
            turn, turn0 = _ratio(now.revenue, now.total_assets), _ratio(prior.revenue, prior.total_assets)
            signals = (
                roa > 0, cfo > 0, roa > roa0, cfo > roa, lev <= lev0, cur > cur0,
                (now.shares_outstanding or 0) <= (prior.shares_outstanding or 0),
                margin > margin0, turn > turn0,
            )
            return float(sum(bool(s) for s in signals))
        return _per_ticker(context, compute)


@dataclass
class AccrualsAnomaly:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="accruals", family="accounting", needs=("fundamentals",),
        provenance=Provenance(
            citation="Sloan (1996), Do Stock Prices Fully Reflect Information in Accruals "
                     "and Cash Flows About Future Earnings?, The Accounting Review",
            institution="University of Pennsylvania (Wharton)",
            notes="low accruals (earnings backed by cash) are bullish",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        def compute(ticker):
            filings = _filings(context, ticker, 1)
            if not filings:
                return np.nan
            f = filings[-1]
            return -_ratio((f.net_income or 0) - (f.operating_cash_flow or 0), f.total_assets)
        return _per_ticker(context, compute)


@dataclass
class ValueComposite:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="value_composite", family="value", needs=("fundamentals", "prices"),
        provenance=Provenance(
            citation="Fama & French (1992), The Cross-Section of Expected Stock Returns, "
                     "Journal of Finance; Asness, Moskowitz & Pedersen (2013)",
            institution="University of Chicago; AQR Capital Management",
            notes="average rank of earnings, free-cash-flow, and book yields",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        rows = {}
        price = context.closes.iloc[-1]
        for ticker in context.tickers:
            filings = _filings(context, ticker, 1)
            if not filings or not filings[-1].shares_outstanding:
                continue
            f = filings[-1]
            cap = float(price[ticker]) * f.shares_outstanding
            rows[ticker] = {
                "ep": _ratio(f.net_income, cap),
                "fcfp": _ratio(f.free_cash_flow, cap),
                "bp": _ratio(f.equity, cap),
            }
        if not rows:
            return pd.Series(np.nan, index=context.tickers)
        return pd.DataFrame(rows).T.apply(cross_sectional_z).mean(axis=1).reindex(context.tickers)


@dataclass
class PostEarningsDrift:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="post_earnings_drift", family="events", needs=("events", "prices"),
        provenance=Provenance(
            citation="Bernard & Thomas (1989), Post-Earnings-Announcement Drift: Delayed "
                     "Price Response or Risk Premium?, Journal of Accounting Research",
            institution="University of Michigan",
            notes="surprise scaled by price, plus guidance revision, decaying over sixty days",
        ),
    ))
    horizon_days: int = 60

    def score(self, context: JuryContext) -> pd.Series:
        if context.view is None:
            return pd.Series(np.nan, index=context.tickers)
        cutoff = context.view.cutoff
        price = context.closes.iloc[-1]

        def compute(ticker):
            reports = context.view.records("earnings", ticker, since=cutoff - timedelta(days=self.horizon_days))
            if not reports:
                return 0.0
            report = reports[-1]
            age = (cutoff - parse_timestamp(report.available_at)).days
            surprise = (report.eps_actual - (report.eps_estimate or report.eps_actual)) / float(price[ticker])
            guidance = 0.0
            if report.guidance_mid and report.prior_guidance_mid:
                guidance = report.guidance_mid / report.prior_guidance_mid - 1.0
            return (surprise * 10 + guidance) * (1.0 - age / self.horizon_days)
        return _per_ticker(context, compute)


@dataclass
class OpportunisticInsiders:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="opportunistic_insiders", family="smart_money", needs=("events",),
        provenance=Provenance(
            citation="Cohen, Malloy & Pomorski (2012), Decoding Inside Information, "
                     "Journal of Finance; Lakonishok & Lee (2001), Are Insider Trades "
                     "Informative?, Review of Financial Studies",
            institution="Harvard Business School and co-authors; University of Illinois and co-author",
            notes="discretionary open-market trades only; plan and compensation lines ignored",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        if context.view is None:
            return pd.Series(np.nan, index=context.tickers)
        since = context.view.cutoff - timedelta(days=30)

        def compute(ticker):
            rows = context.view.records("insider", ticker, since=since)
            buyers = {r.insider for r in rows if r.discretionary_purchase}
            sellers = {r.insider for r in rows if r.discretionary_sale}
            return float(len(buyers) - len(sellers))
        return _per_ticker(context, compute)


@dataclass
class AnalystRevisions:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="analyst_revisions", family="analysts", needs=("events",),
        provenance=Provenance(
            citation="Womack (1996), Do Brokerage Analysts' Recommendations Have Investment "
                     "Value?, Journal of Finance; Jegadeesh, Kim, Krische & Lee (2004), "
                     "Analyzing the Analysts, Journal of Finance",
            institution="Dartmouth (Tuck); Emory University and co-authors",
            notes="net rating changes over thirty days; level of consensus is ignored",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        if context.view is None:
            return pd.Series(np.nan, index=context.tickers)
        since = context.view.cutoff - timedelta(days=30)

        def compute(ticker):
            actions = context.view.records("analyst_action", ticker, since=since)
            return float(sum(
                np.sign(_RATING[a.rating] - _RATING[a.prior_rating]) for a in actions if a.prior_rating
            ))
        return _per_ticker(context, compute)


@dataclass
class EconomicLinks:
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="economic_links", family="network", needs=("events", "prices"),
        provenance=Provenance(
            citation="Cohen & Frazzini (2008), Economic Links and Predictable Returns, "
                     "Journal of Finance",
            institution="Academic authors (Frazzini later a principal at AQR Capital Management)",
            notes="linked firms' one-month returns; competitor links enter with a negative sign",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        if context.view is None or len(context.closes) <= 22:
            return pd.Series(np.nan, index=context.tickers)
        month = context.closes.iloc[-1] / context.closes.iloc[-22] - 1.0

        def compute(ticker):
            links = context.view.records("relationship", ticker)
            values = [
                (-1.0 if link.relation == "competitor" else 1.0) * month[link.counterparty]
                for link in links if link.counterparty in month.index
            ]
            return float(np.mean(values)) if values else np.nan
        return _per_ticker(context, compute)


__all__ = [
    "AccrualsAnomaly", "AnalystRevisions", "EconomicLinks", "GrossProfitability",
    "OpportunisticInsiders", "PiotroskiFScore", "PostEarningsDrift", "QualityMinusJunk",
    "ValueComposite",
]
