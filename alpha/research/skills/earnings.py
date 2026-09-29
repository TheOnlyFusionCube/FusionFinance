"""Earnings skill: beat/miss quality, guidance, priced-in risk, and bottlenecks.

Guidance outweighs the quarter because prices discount the future. A beat that
comes with a revenue beat is higher quality than one produced by cost cuts. A
large run-up into the report means the beat was probably priced in. Transcript
scanning looks for constraint language ("supply constrained", "lead times"),
the earliest public trace of a bottleneck someone else will be paid to fix.
"""
from __future__ import annotations

import re
from datetime import timedelta

import pandas as pd

from alpha.research.cards import SignalCard, clip, empty_card
from alpha.research.market import bars_frame
from alpha.research.store import StoreView

CONSTRAINT_PHRASES = (
    "supply constrained", "capacity constrained", "capacity limited", "lead times",
    "shortage", "bottleneck", "demand exceeds supply", "sold out", "allocation",
    "investing heavily", "cannot get enough",
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _surprise(actual: float, estimate: float | None) -> float | None:
    if estimate is None or estimate == 0:
        return None
    return (actual - estimate) / abs(estimate)


def constraint_sentences(text: str, limit: int = 3) -> list[str]:
    hits = []
    for sentence in _SENTENCE.split(text):
        lower = sentence.lower()
        if any(phrase in lower for phrase in CONSTRAINT_PHRASES):
            hits.append(sentence.strip())
        if len(hits) >= limit:
            break
    return hits


def analyze_earnings(view: StoreView, ticker: str, *, lookback_days: int = 100) -> SignalCard:
    reports = view.records("earnings", ticker, since=view.cutoff - timedelta(days=lookback_days))
    transcripts = view.records("transcript", ticker, since=view.cutoff - timedelta(days=lookback_days))
    if not reports and not transcripts:
        return empty_card("earnings", ticker, view.as_of, "No earnings report in the window.", "fundamentals")
    lines, facts, flags, ids = [], {}, [], []
    score = 0.0
    freshness = 1.0
    if reports:
        report = reports[-1]
        ids.append(report.record_id)
        eps = _surprise(report.eps_actual, report.eps_estimate)
        revenue = _surprise(report.revenue_actual, report.revenue_estimate)
        guidance = _surprise(report.guidance_mid, report.prior_guidance_mid) if report.guidance_mid else None
        if eps is not None:
            facts["eps_surprise_pct"] = round(eps * 100, 2)
        if revenue is not None:
            facts["revenue_surprise_pct"] = round(revenue * 100, 2)
        if guidance is not None:
            facts["guidance_change_pct"] = round(guidance * 100, 2)
        if report.gross_margin is not None and report.prior_gross_margin is not None:
            facts["gross_margin_change_bps"] = round((report.gross_margin - report.prior_gross_margin) * 1e4, 1)
        verdict = "beat" if (eps or 0) > 0 else "miss" if (eps or 0) < 0 else "in line"
        lines.append(
            f"{report.fiscal_period} EPS {verdict} ({(eps or 0):+.1%}), revenue {(revenue or 0):+.1%} "
            f"versus consensus."
        )
        if guidance is not None:
            lines.append(f"Guidance midpoint moved {guidance:+.1%} versus the prior outlook.")
        quality = "revenue-driven" if (eps or 0) > 0 and (revenue or 0) > 0 else "cost-driven" if (eps or 0) > 0 else ""
        if quality:
            lines.append(f"The beat was {quality}.")
            if quality == "cost-driven":
                flags.append("LOW_QUALITY_BEAT")
        bars = bars_frame(view, ticker)
        reported = pd.Timestamp(report.available_at).tz_convert(None)
        before = bars.loc[bars.index.tz_localize(None) < reported.normalize()] if len(bars) else bars
        if len(before) > 10:
            runup = float(before["close"].iloc[-1] / before["close"].iloc[-11] - 1.0)
            facts["pre_report_runup_pct"] = round(runup * 100, 2)
            if runup > 0.1 and (eps or 0) > 0:
                flags.append("PRICED_IN_RISK")
                lines.append(f"Shares ran {runup:+.0%} into the report, so the beat may be priced in.")
        age_days = (view.cutoff - pd.Timestamp(report.available_at).to_pydatetime()).days
        facts["days_since_report"] = float(age_days)
        freshness = 0.5 ** (age_days / 30.0)   # post-announcement drift fades
        score = freshness * clip(
            0.8 * clip((guidance or 0.0) * 10) + 0.4 * clip((revenue or 0.0) * 10)
            + 0.3 * clip((eps or 0.0) * 5) - (0.25 if "PRICED_IN_RISK" in flags else 0.0)
            - (0.15 if "LOW_QUALITY_BEAT" in flags else 0.0)
        )
    if transcripts:
        transcript = transcripts[-1]
        ids.append(transcript.record_id)
        hits = constraint_sentences(transcript.text)
        facts["constraint_mentions"] = float(
            sum(transcript.text.lower().count(p) for p in CONSTRAINT_PHRASES)
        )
        if hits:
            flags.append("BOTTLENECK_LANGUAGE")
            lines.append("Management flagged constraints on the call:")
            lines.extend(hits)
    return SignalCard(
        skill="earnings", ticker=ticker, as_of=view.as_of, score=score,
        confidence=(0.7 * freshness if reports else 0.3),
        headline=lines[0] if lines else "Earnings context only.",
        lines=tuple(lines), facts=facts, flags=tuple(flags), record_ids=tuple(ids),
        role="fundamentals",
    )


__all__ = ["CONSTRAINT_PHRASES", "analyze_earnings", "constraint_sentences"]
