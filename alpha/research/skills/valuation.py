"""Valuation skill: DCF, analyst consensus, and peer multiples, cross-checked.

DCF: free cash flow grows at a fitted rate that fades linearly to a terminal
rate over the projection period; the discount rate is a WACC with a CAPM cost
of equity (beta estimated from point-in-time bars against the benchmark). The
terminal value uses the Gordon growth formula. Each method has a known bias
(DCF is growth-sensitive, sell-side targets skew bullish, peer multiples fail
when a whole sector is rich), so the card reports agreement, not an average.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from statistics import mean, median

from alpha.research.cards import SignalCard, clip, empty_card
from alpha.research.market import bars_frame, beta
from alpha.research.store import StoreView


@dataclass(frozen=True)
class ValuationAssumptions:
    risk_free_rate: float = 0.04
    equity_risk_premium: float = 0.05
    terminal_growth: float = 0.025
    projection_years: int = 5
    tax_rate: float = 0.21
    credit_spread: float = 0.02
    growth_bounds: tuple[float, float] = (-0.05, 0.25)


def dcf_per_share(
    fcf: float, growth: float, wacc: float, *, years: int, terminal_growth: float,
    net_debt: float, shares: float,
) -> float:
    if wacc <= terminal_growth:
        raise ValueError("discount rate must exceed terminal growth")
    value, cash_flow = 0.0, fcf
    for year in range(1, years + 1):
        rate = growth + (terminal_growth - growth) * (year - 1) / max(1, years - 1)
        cash_flow *= 1.0 + rate
        value += cash_flow / (1.0 + wacc) ** year
    terminal = cash_flow * (1.0 + terminal_growth) / (wacc - terminal_growth)
    value += terminal / (1.0 + wacc) ** years
    return (value - net_debt) / shares


def analyze_valuation(
    view: StoreView,
    ticker: str,
    *,
    benchmark: str | None = None,
    peers: tuple[str, ...] = (),
    assumptions: ValuationAssumptions = ValuationAssumptions(),
) -> SignalCard:
    filings = view.records("fundamentals", ticker)
    bars = bars_frame(view, ticker)
    if not filings or bars.empty:
        return empty_card("valuation", ticker, view.as_of, "No filed fundamentals or prices.", "fundamentals")
    latest = max(filings, key=lambda f: (f.period_end, f.available_at))
    price = float(bars["close"].iloc[-1])
    lines, facts, flags, votes = [], {"price": round(price, 4)}, [], []
    ids = [latest.record_id]

    fcf = latest.free_cash_flow
    shares = latest.shares_outstanding
    if fcf is not None and fcf > 0 and shares:
        history = sorted({f.period_end: f for f in filings}.values(), key=lambda f: f.period_end)
        growth = latest.revenue_growth_3y
        if growth is None and len(history) >= 2 and history[0].revenue and latest.revenue:
            years = max(1, len(history) - 1)
            growth = (latest.revenue / history[0].revenue) ** (1 / years) - 1.0
        growth = clip(growth if growth is not None else 0.03, *assumptions.growth_bounds)
        fitted_beta = None
        if benchmark:
            bench = bars_frame(view, benchmark)
            if not bench.empty:
                fitted_beta = beta(bars["close"], bench["close"])
        stock_beta = clip(fitted_beta if fitted_beta is not None else 1.0, 0.5, 2.5)
        equity_cost = assumptions.risk_free_rate + stock_beta * assumptions.equity_risk_premium
        debt = latest.total_debt or 0.0
        cash = latest.cash or 0.0
        market_cap = price * shares
        debt_cost = (assumptions.risk_free_rate + assumptions.credit_spread) * (1 - assumptions.tax_rate)
        wacc = (market_cap * equity_cost + debt * debt_cost) / (market_cap + debt)
        wacc = max(wacc, assumptions.terminal_growth + 0.02)
        kwargs = dict(
            years=assumptions.projection_years, terminal_growth=assumptions.terminal_growth,
            net_debt=debt - cash, shares=shares,
        )
        intrinsic = dcf_per_share(fcf, growth, wacc, **kwargs)
        low = dcf_per_share(fcf, growth - 0.01, wacc, **kwargs)
        high = dcf_per_share(fcf, growth + 0.01, wacc, **kwargs)
        margin = intrinsic / price - 1.0
        facts.update({
            "dcf_value": round(intrinsic, 4), "dcf_margin_of_safety_pct": round(margin * 100, 2),
            "wacc_pct": round(wacc * 100, 3), "fcf_growth_pct": round(growth * 100, 2),
            "beta": round(stock_beta, 3),
        })
        lines.append(
            f"DCF value {intrinsic:.2f} per share versus price {price:.2f} "
            f"({margin:+.0%} margin of safety) at WACC {wacc:.1%} and growth {growth:.1%}."
        )
        lines.append(f"Growth one point lower or higher moves value to {low:.2f} or {high:.2f}.")
        votes.append(1 if margin > 0.15 else -1 if margin < -0.15 else 0)
        if abs(high - low) / max(abs(intrinsic), 1e-9) > 0.4:
            flags.append("DCF_GROWTH_SENSITIVE")
    else:
        lines.append("Free cash flow is negative or unavailable, so no DCF is reported.")

    actions = view.records("analyst_action", ticker, since=view.cutoff - timedelta(days=120))
    targets = {a.firm: a.price_target for a in actions if a.price_target}
    if targets:
        consensus = mean(targets.values())
        upside = consensus / price - 1.0
        facts["consensus_target"] = round(consensus, 4)
        facts["consensus_upside_pct"] = round(upside * 100, 2)
        lines.append(
            f"Consensus target {consensus:.2f} from {len(targets)} firms "
            f"({upside:+.0%}; sell-side targets skew bullish)."
        )
        votes.append(1 if upside > 0.2 else -1 if upside < -0.05 else 0)
        ids.extend(a.record_id for a in actions)

    if latest.net_income and shares and latest.net_income > 0:
        pe = price * shares / latest.net_income
        facts["pe"] = round(pe, 3)
        peer_pe = []
        for peer in peers:
            peer_filings = view.records("fundamentals", peer)
            peer_bars = bars_frame(view, peer)
            if not peer_filings or peer_bars.empty:
                continue
            p = max(peer_filings, key=lambda f: (f.period_end, f.available_at))
            if p.net_income and p.net_income > 0 and p.shares_outstanding:
                peer_pe.append(float(peer_bars["close"].iloc[-1]) * p.shares_outstanding / p.net_income)
        if peer_pe:
            relative = pe / median(peer_pe) - 1.0
            facts["pe_vs_peers_pct"] = round(relative * 100, 2)
            lines.append(
                f"P/E {pe:.1f} versus a peer median of {median(peer_pe):.1f} ({relative:+.0%})."
            )
            votes.append(-1 if relative > 0.3 else 1 if relative < -0.3 else 0)

    if not votes:
        return SignalCard(
            skill="valuation", ticker=ticker, as_of=view.as_of, score=0.0, confidence=0.1,
            headline="Valuation is indeterminate.", lines=tuple(lines), facts=facts,
            flags=tuple(flags), record_ids=tuple(ids), role="fundamentals",
        )
    agreement = sum(votes)
    if len(set(v for v in votes if v)) > 1:
        flags.append("METHODS_DISAGREE")
    verdict = "undervalued" if agreement > 0 else "overvalued" if agreement < 0 else "fairly valued"
    if agreement:
        side = 1 if agreement > 0 else -1
        lines.insert(0, f"{votes.count(side)} of {len(votes)} methods say {verdict}.")
    else:
        lines.insert(0, f"The {len(votes)} methods net to {verdict}.")
    return SignalCard(
        skill="valuation", ticker=ticker, as_of=view.as_of,
        score=clip(agreement / max(2, len(votes))),
        confidence=clip(0.25 * len(votes) + (0.2 if "METHODS_DISAGREE" not in flags else 0.0), 0.0, 1.0),
        headline=f"Stock screens {verdict} across {len(votes)} methods.",
        lines=tuple(lines), facts=facts, flags=tuple(flags), record_ids=tuple(ids),
        role="fundamentals",
    )


__all__ = ["ValuationAssumptions", "analyze_valuation", "dcf_per_share"]
