"""Sentiment skill: institutional vs retail gauges and their divergence.

Institutional gauge: latest rating per covering firm, implied upside to the
average price target, net upgrades in the last month, and target dispersion.
Retail gauge: mention velocity against the trailing week, bullish share, and an
account-concentration filter (four hundred mentions from thirty accounts is
spam, not a conversation). A crowded, spammy, one-sided retail gauge is read as
a contrarian warning rather than confirmation.
"""
from __future__ import annotations

from datetime import timedelta
from statistics import mean, pstdev

from alpha.research.cards import SignalCard, clip, empty_card
from alpha.research.market import bars_frame
from alpha.research.records import parse_timestamp
from alpha.research.store import StoreView

_RATING = {"strong_buy": 1.0, "buy": 0.5, "hold": 0.0, "sell": -0.5, "strong_sell": -1.0}


def _gauge_label(value: float) -> str:
    if value >= 0.5:
        return "very bullish"
    if value >= 0.15:
        return "bullish"
    if value <= -0.5:
        return "very bearish"
    if value <= -0.15:
        return "bearish"
    return "neutral"


def analyze_sentiment(view: StoreView, ticker: str, *, recent_days: int = 1) -> SignalCard:
    lines, facts, flags, ids = [], {}, [], []
    institutional = None
    actions = view.records("analyst_action", ticker, since=view.cutoff - timedelta(days=90))
    if actions:
        latest = {}
        for action in actions:
            latest[action.firm] = action
            ids.append(action.record_id)
        ratings = [_RATING[a.rating] for a in latest.values()]
        targets = [a.price_target for a in latest.values() if a.price_target]
        recent = [a for a in actions if parse_timestamp(a.available_at) >= view.cutoff - timedelta(days=30)]
        upgrades = sum(
            1 for a in recent if a.prior_rating and _RATING[a.rating] > _RATING[a.prior_rating]
        )
        downgrades = sum(
            1 for a in recent if a.prior_rating and _RATING[a.rating] < _RATING[a.prior_rating]
        )
        bars = bars_frame(view, ticker)
        upside = None
        if targets and len(bars):
            upside = mean(targets) / float(bars["close"].iloc[-1]) - 1.0
            facts["target_upside_pct"] = round(upside * 100, 2)
            dispersion = pstdev(targets) / mean(targets) if len(targets) > 1 else 0.0
            facts["target_dispersion_pct"] = round(dispersion * 100, 2)
            if dispersion > 0.25:
                flags.append("HIGH_TARGET_DISPERSION")
        buy_share = sum(r > 0 for r in ratings) / len(ratings)
        facts["covering_firms"] = float(len(latest))
        facts["buy_share_pct"] = round(buy_share * 100, 2)
        facts["net_upgrades_30d"] = float(upgrades - downgrades)
        institutional = clip(
            0.6 * mean(ratings) + 0.25 * clip((upside or 0.0) * 3) + 0.15 * clip(upgrades - downgrades)
        )
        lines.append(
            f"{len(latest)} covering firms, {buy_share:.0%} at Buy or better, "
            f"net upgrades in the last month of {upgrades - downgrades}."
        )
        if upside is not None:
            lines.append(f"Average price target implies {upside:+.1%} versus the last close.")

    retail = None
    social = view.records("social", ticker, since=view.cutoff - timedelta(days=recent_days + 7))
    if social:
        ids.extend(item.record_id for item in social)
        recent_cut = view.cutoff - timedelta(days=recent_days)
        last_day = [s for s in social if parse_timestamp(s.available_at) >= recent_cut]
        prior = [s for s in social if parse_timestamp(s.available_at) < recent_cut]
        today = sum(s.mentions for s in last_day)
        recent_days_seen = max(1, len({s.event_at[:10] for s in last_day}))
        today = today / recent_days_seen
        baseline = sum(s.mentions for s in prior) / max(1, len({s.event_at[:10] for s in prior}))
        velocity = (today + 1.0) / (baseline + 1.0) - 1.0
        bulls = sum(s.bullish for s in last_day or social)
        bears = sum(s.bearish for s in last_day or social)
        accounts = sum(s.unique_accounts for s in last_day or social)
        mentions = sum(s.mentions for s in last_day or social)
        bull_share = bulls / (bulls + bears) if bulls + bears else 0.5
        account_ratio = accounts / mentions if mentions else 1.0
        facts["mention_velocity_pct"] = round(velocity * 100, 2)
        facts["retail_bull_share_pct"] = round(bull_share * 100, 2)
        facts["accounts_per_mention"] = round(account_ratio, 4)
        crowded = bull_share >= 0.8 or bull_share <= 0.2
        spammy = account_ratio < 0.3 and mentions >= 50
        retail = clip((bull_share - 0.5) * 2.0)
        state = "one-sided consensus" if crowded else "an open debate" if 0.35 <= bull_share <= 0.65 else "leaning"
        lines.append(
            f"Retail mentions per day changed {velocity:+.0%} versus the trailing week; "
            f"{bull_share:.0%} bullish, which reads as {state}."
        )
        if spammy:
            flags.append("SPAM_CONCENTRATED_MENTIONS")
            lines.append(
                f"Mentions come from few accounts ({account_ratio:.2f} accounts per mention), "
                "a promotion pattern rather than a conversation."
            )
        if crowded and velocity > 1.0:
            flags.append("CROWDED_RETAIL_SURGE")

    if institutional is None and retail is None:
        return empty_card("sentiment", ticker, view.as_of, "No analyst or social coverage.", "market")
    if institutional is not None:
        facts["institutional_gauge"] = round(institutional, 4)
    if retail is not None:
        facts["retail_gauge"] = round(retail, 4)
    divergence = ""
    if institutional is not None and retail is not None:
        if institutional >= 0.15 and retail <= -0.15:
            divergence = "Wall Street bullish while retail is bearish: possible early opportunity."
        elif institutional <= -0.15 and retail >= 0.15:
            divergence = "Retail bullish while Wall Street is bearish: momentum trade or trap."
            flags.append("RETAIL_VS_STREET_DIVERGENCE")
        if divergence:
            lines.append(divergence)

    contrarian = -0.6 if {"SPAM_CONCENTRATED_MENTIONS", "CROWDED_RETAIL_SURGE"} & set(flags) else 0.0
    retail_term = (retail or 0.0) * (0.25 if not contrarian else 0.0)
    # Fade a crowded, promotional retail gauge in whichever direction it points.
    score = clip((institutional or 0.0) * 0.75 + retail_term + contrarian * (retail or 0.0))
    confidence = clip(
        0.3 * (institutional is not None) + 0.2 * (retail is not None)
        + 0.3 * min(1.0, facts.get("covering_firms", 0.0) / 8), 0.0, 1.0,
    )
    headline = (
        f"Institutional {_gauge_label(institutional or 0.0)}, "
        f"retail {_gauge_label(retail or 0.0)}."
    )
    return SignalCard(
        skill="sentiment", ticker=ticker, as_of=view.as_of, score=score, confidence=confidence,
        headline=headline, lines=tuple(lines), facts=facts, flags=tuple(flags),
        record_ids=tuple(ids), role="market",
    )


__all__ = ["analyze_sentiment"]
