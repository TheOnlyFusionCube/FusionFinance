"""Smart-money skill: insiders, Congress, and 13F filers, filtered for intent.

Insiders: only discretionary open-market lines count (Form 4 code P or S, not
under a Rule 10b5-1 plan). Grants, option exercises, tax withholding, and gifts
are compensation mechanics, not conviction. Cluster buying (several distinct
insiders within a month), seven-figure size, and buying well below the 90-day
high strengthen the signal.

Congress: disclosures only count once disclosed (up to forty-five days late).
Trades by members whose committee oversees the issuer are flagged.

13F: new positions and full exits in the latest quarter, never routine resizing.
"""
from __future__ import annotations

from datetime import timedelta
from statistics import median

from alpha.research.cards import SignalCard, clip, empty_card, fmt_money
from alpha.research.market import bars_frame
from alpha.research.records import parse_timestamp
from alpha.research.store import StoreView


def analyze_smart_money(view: StoreView, ticker: str, *, insider_days: int = 30) -> SignalCard:
    lines, facts, flags, ids = [], {}, [], []
    score = 0.0
    evidence = 0.0

    insiders = view.records("insider", ticker, since=view.cutoff - timedelta(days=insider_days))
    buys = [r for r in insiders if r.discretionary_purchase]
    sells = [r for r in insiders if r.discretionary_sale]
    skipped = len(insiders) - len(buys) - len(sells)
    if insiders:
        ids.extend(r.record_id for r in insiders)
        buyers = {r.insider for r in buys}
        sellers = {r.insider for r in sells}
        bought = sum(r.value for r in buys)
        sold = sum(r.value for r in sells)
        facts["insider_buyers_30d"] = float(len(buyers))
        facts["insider_sellers_30d"] = float(len(sellers))
        facts["insider_buy_value"] = round(bought, 2)
        facts["insider_sell_value"] = round(sold, 2)
        lines.append(
            f"{len(buyers)} insiders bought {fmt_money(bought)} and {len(sellers)} sold "
            f"{fmt_money(sold)} in open-market, non-plan trades over {insider_days} days; "
            f"{skipped} routine lines were excluded."
        )
        insider_score = clip(0.35 * len(buyers) - 0.15 * len(sellers))
        if len(buyers) >= 2:
            flags.append("INSIDER_CLUSTER_BUY")
            insider_score = clip(insider_score + 0.2)
        if bought >= 1_000_000:
            insider_score = clip(insider_score + 0.15)
        bars = bars_frame(view, ticker)
        if buys and len(bars) >= 20:
            high = float(bars["high"].iloc[-63:].max())
            paid = min(r.price or high for r in buys)
            below = 1.0 - paid / high
            facts["buy_below_90d_high_pct"] = round(below * 100, 2)
            if below >= 0.2:
                flags.append("INSIDER_BUY_THE_DIP")
                lines.append(f"Insiders paid {below:.0%} below the ninety-day high.")
                insider_score = clip(insider_score + 0.15)
        score += insider_score
        evidence += 1.0 if buys or sells else 0.25

    congress = view.records("congress", ticker, since=view.cutoff - timedelta(days=120))
    if congress:
        ids.extend(r.record_id for r in congress)
        purchases = sum(1 for r in congress if r.direction == "purchase")
        sales = len(congress) - purchases
        lag = median(r.disclosure_lag_days for r in congress)
        facts["congress_net_buyers"] = float(purchases - sales)
        facts["congress_median_lag_days"] = round(lag, 1)
        lines.append(
            f"Members of Congress disclosed {purchases} purchases and {sales} sales; "
            f"median disclosure lag {lag:.0f} days."
        )
        if any(r.committee_oversees_issuer for r in congress):
            flags.append("COMMITTEE_OVERSIGHT_TRADE")
            lines.append("At least one trade came from a member whose committee oversees the issuer.")
        score += clip(0.15 * (purchases - sales), -0.3, 0.3)
        evidence += 0.5

    holdings = view.records("institutional", ticker, since=view.cutoff - timedelta(days=150))
    if holdings:
        ids.extend(r.record_id for r in holdings)
        latest_quarter = max(parse_timestamp(r.event_at) for r in holdings)
        quarter = [r for r in holdings if parse_timestamp(r.event_at) == latest_quarter]
        new = [r for r in quarter if r.is_new_position]
        exits = [r for r in quarter if r.is_exit]
        facts["funds_new_positions"] = float(len(new))
        facts["funds_exits"] = float(len(exits))
        lines.append(
            f"In the latest 13F quarter {len(new)} funds opened new positions and {len(exits)} exited."
        )
        score += clip(0.12 * (len(new) - len(exits)), -0.3, 0.3)
        evidence += 0.5

    if not lines:
        return empty_card("smart_money", ticker, view.as_of, "No insider, Congress, or 13F activity.", "market")
    return SignalCard(
        skill="smart_money", ticker=ticker, as_of=view.as_of, score=clip(score),
        confidence=clip(evidence / 2.0, 0.0, 1.0),
        headline=(
            "Smart money is accumulating." if score > 0.15
            else "Smart money is distributing." if score < -0.15
            else "Smart money is quiet or mixed."
        ),
        lines=tuple(lines), facts=facts, flags=tuple(flags), record_ids=tuple(ids), role="market",
    )


__all__ = ["analyze_smart_money"]
