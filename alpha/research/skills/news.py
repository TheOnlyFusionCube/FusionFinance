"""News impact skill: score every article 0-10 and map it to affected tickers.

Impact combines the factors a news desk weighs: magnitude (deal size, words like
merger or guidance), surprise versus expectations, breadth (how many tickers an
article touches), timing (in-session news prices faster than overnight news),
and source credibility. Direction comes from a finance lexicon. Second-order
effects propagate through the relationship graph: good news for a customer is
good news for its supplier, and good news for a competitor is bad news here.

The scorer is a protocol so an LLM scorer can replace the lexical default; any
replacement must still return the same bounded, structured fields.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

from alpha.research.cards import SignalCard, clip, empty_card
from alpha.research.records import NewsArticle, parse_timestamp
from alpha.research.store import StoreView

_POSITIVE = (
    "beat", "beats", "raised", "raises", "record", "upgrade", "approval", "approved",
    "accelerated", "strong", "surge", "wins", "awarded", "expands", "growth", "improved",
)
_NEGATIVE = (
    "miss", "missed", "cut", "cuts", "downgrade", "recall", "investigation", "lawsuit",
    "declined", "weak", "plunge", "delay", "halt", "fraud", "probe", "deteriorated",
)
_MAGNITUDE = (
    "merger", "acquire", "acquisition", "guidance", "fda", "buyback", "bankruptcy",
    "restatement", "tariff", "contract", "rate decision", "ceo",
)
_SURPRISE = ("unexpected", "surprise", "unexpectedly", "shock", "beat", "miss", "record")
_CREDIBILITY = {
    "wire": 1.0, "regulator": 1.0, "company": 0.9, "major": 0.9,
    "trade": 0.7, "blog": 0.35, "social": 0.25,
}
_DOLLARS = re.compile(r"\$\s?(\d+(?:\.\d+)?)\s?(billion|million|bn|b|m)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ArticleScore:
    impact: float        # 0-10
    direction: int       # -1, 0, +1
    rationale: str


class NewsScorer(Protocol):
    def score(self, article: NewsArticle) -> ArticleScore: ...


class LexicalNewsScorer:
    """Deterministic, explainable default scorer."""

    def score(self, article: NewsArticle) -> ArticleScore:
        text = f"{article.headline} {article.body}".lower()
        words = re.findall(r"[a-z]+", text)
        positive = sum(words.count(term) for term in _POSITIVE)
        negative = sum(words.count(term) for term in _NEGATIVE)
        direction = (positive > negative) - (negative > positive)
        magnitude = sum(term in text for term in _MAGNITUDE)
        dollars = [
            float(value) * (1e9 if unit.lower() in {"billion", "bn", "b"} else 1e6)
            for value, unit in _DOLLARS.findall(text)
        ]
        size = math.log10(max(dollars)) - 8.0 if dollars else 0.0     # $100M -> 0, $10B -> 2
        surprise = sum(term in text for term in _SURPRISE)
        breadth = len(set(article.mentioned_tickers) | {article.ticker})
        published = parse_timestamp(article.available_at)
        minutes = published.hour * 60 + published.minute
        in_session = 13 * 60 + 30 <= minutes <= 20 * 60 and published.weekday() < 5
        credibility = _CREDIBILITY.get(article.source.split(":")[0].lower(), 0.6)
        raw = (
            1.5 * min(magnitude, 3) + 1.2 * clip(size, 0.0, 3.0) + 1.0 * min(surprise, 2)
            + 0.4 * min(breadth - 1, 5) + (0.5 if in_session else 0.0)
            + 1.0 * min(abs(positive - negative), 3)
        )
        impact = round(clip(raw * credibility, 0.0, 10.0), 2)
        reasons = []
        if magnitude:
            reasons.append("material corporate event")
        if dollars:
            reasons.append("sizable dollar amount")
        if surprise:
            reasons.append("deviation from expectations")
        if credibility < 0.5:
            reasons.append("low-credibility source")
        return ArticleScore(impact, direction, ", ".join(reasons) or "routine coverage")


def impact_tier(impact: float) -> str:
    if impact >= 8:
        return "market-moving"
    if impact >= 6:
        return "important"
    if impact >= 3:
        return "relevant"
    return "background"


def analyze_news(
    view: StoreView,
    ticker: str,
    *,
    lookback_days: int = 5,
    scorer: NewsScorer | None = None,
    half_life_days: float = 2.0,
) -> SignalCard:
    scorer = scorer or LexicalNewsScorer()
    since = view.cutoff - timedelta(days=lookback_days)
    direct = view.records("news", ticker, since=since)
    links = {r.counterparty: r.relation for r in view.records("relationship", ticker)}
    secondary: list[tuple[NewsArticle, str]] = []
    for counterparty, relation in links.items():
        for article in view.records("news", counterparty, since=since):
            if ticker not in article.mentioned_tickers and article.ticker != ticker:
                secondary.append((article, relation))
    if not direct and not secondary:
        return empty_card("news_impact", ticker, view.as_of, "No news in the lookback window.", "news")

    weighted, weight_sum, lines, ids, facts = 0.0, 0.0, [], [], {}
    scored = []
    for article in direct:
        scored.append((article, scorer.score(article), 1.0, "direct"))
    for article, relation in secondary:
        score = scorer.score(article)
        sign = -1 if relation == "competitor" else 1
        scored.append((
            article,
            ArticleScore(score.impact * 0.6, score.direction * sign, score.rationale),
            0.6,
            relation,
        ))
    scored.sort(key=lambda item: -item[1].impact)
    for article, score, _discount, channel in scored:
        age = (view.cutoff - parse_timestamp(article.available_at)).total_seconds() / 86_400
        decay = 0.5 ** (age / half_life_days)
        weighted += score.direction * (score.impact / 10.0) * decay
        weight_sum += decay
        ids.append(article.record_id)
    top_article, top, _, top_channel = scored[0]
    facts["top_impact"] = top.impact
    facts["articles"] = float(len(direct))
    facts["secondary_articles"] = float(len(secondary))
    lines.append(f"{top_article.headline.strip().rstrip('.')}.")
    lines.append(
        f"Top story scores {top.impact:.1f} of 10 ({impact_tier(top.impact)}), "
        f"channel {top_channel}, because of {top.rationale}."
    )
    for article, score, _, channel in scored[1:4]:
        lines.append(
            f"Also {impact_tier(score.impact)} via {channel}: {article.headline.strip().rstrip('.')}."
        )
    score_value = clip(weighted / max(weight_sum, 1.0) * 2.0)
    confidence = clip(top.impact / 10.0, 0.0, 1.0)
    flags = tuple(
        flag for flag, hit in (
            ("LOW_CREDIBILITY_SOURCE", "low-credibility" in top.rationale),
            ("SECOND_ORDER_ONLY", not direct),
        ) if hit
    )
    direction = "bullish" if score_value > 0.15 else "bearish" if score_value < -0.15 else "mixed"
    return SignalCard(
        skill="news_impact", ticker=ticker, as_of=view.as_of, score=score_value,
        confidence=confidence, headline=f"News flow is {direction}; top impact {top.impact:.1f}/10.",
        lines=tuple(lines), facts=facts, flags=flags, record_ids=tuple(ids), role="news",
    )


__all__ = ["ArticleScore", "LexicalNewsScorer", "NewsScorer", "analyze_news", "impact_tier"]
