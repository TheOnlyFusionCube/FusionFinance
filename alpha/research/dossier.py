"""Research dossier: run every skill, measure convergence, seal for the desk.

Convergence follows the idea-generation rule of thumb: independent signal
families pointing the same way build conviction (one = watch, two = research,
three or more = eligible to size). Contradictions are surfaced, never averaged
away.

The dossier renders each card into a ``SourceDocument`` whose text is the
card's exact sentences and whose numeric values are the card's facts, routed to
the analyst-desk role that owns that evidence. The LLM desk can only quote
those sentences and only assert those numbers; the evidence audit rejects
anything else. The ``feature_vector`` exposes the same deterministic analytics
to the ML verifier, which never sees LLM output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from alpha.agents.models import NumericValue, SealedSourceSnapshot, SourceDocument
from alpha.research.cards import SignalCard
from alpha.research.skills.earnings import analyze_earnings
from alpha.research.skills.news import NewsScorer, analyze_news
from alpha.research.skills.sentiment import analyze_sentiment
from alpha.research.skills.smart_money import analyze_smart_money
from alpha.research.skills.technicals import analyze_technicals
from alpha.research.skills.valuation import ValuationAssumptions, analyze_valuation
from alpha.research.store import StoreView

SKILLS = ("news_impact", "sentiment", "smart_money", "earnings", "valuation", "technicals")
Tier = Literal["none", "watch", "research", "size"]
_FLAG_FEATURES = (
    "SPAM_CONCENTRATED_MENTIONS", "CROWDED_RETAIL_SURGE", "INSIDER_CLUSTER_BUY",
    "PRICED_IN_RISK", "STRETCHED_MOVE", "LOW_CREDIBILITY_SOURCE",
)


@dataclass(frozen=True)
class Convergence:
    direction: Literal["bullish", "bearish", "mixed", "none"]
    agreeing: tuple[str, ...]
    opposing: tuple[str, ...]
    tier: Tier

    def to_record(self) -> dict:
        return {
            "direction": self.direction, "agreeing": list(self.agreeing),
            "opposing": list(self.opposing), "tier": self.tier,
        }


def convergence(cards: tuple[SignalCard, ...]) -> Convergence:
    bullish = tuple(card.skill for card in cards if card.active and card.direction == "bullish")
    bearish = tuple(card.skill for card in cards if card.active and card.direction == "bearish")
    if not bullish and not bearish:
        return Convergence("none", (), (), "none")
    if len(bullish) == len(bearish):
        return Convergence("mixed", bullish, bearish, "watch")
    agreeing, opposing = (bullish, bearish) if len(bullish) > len(bearish) else (bearish, bullish)
    net = len(agreeing) - len(opposing)
    tier: Tier = "size" if net >= 3 else "research" if net == 2 else "watch"
    direction = "bullish" if agreeing is bullish else "bearish"
    return Convergence(direction, agreeing, opposing, tier)


@dataclass(frozen=True)
class Dossier:
    ticker: str
    as_of: str
    cards: tuple[SignalCard, ...]
    convergence: Convergence
    lineage: str
    risk_lines: tuple[str, ...] = field(default=())

    def card(self, skill: str) -> SignalCard | None:
        return next((card for card in self.cards if card.skill == skill), None)

    def feature_vector(self) -> tuple[float, ...]:
        """Deterministic analytics for the ML verifier (no LLM output)."""
        values = []
        for skill in SKILLS:
            card = self.card(skill)
            values.append(0.0 if card is None else card.score * card.confidence)
        flags = {flag for card in self.cards for flag in card.flags}
        values.extend(float(flag in flags) for flag in _FLAG_FEATURES)
        return tuple(values)

    @staticmethod
    def feature_names() -> tuple[str, ...]:
        return (*(f"{skill}_signal" for skill in SKILLS), *(f"flag_{f.lower()}" for f in _FLAG_FEATURES))

    def source_documents(self) -> tuple[SourceDocument, ...]:
        day = self.as_of[:10]
        documents = []
        by_role: dict[str, list[SignalCard]] = {}
        for card in self.cards:
            by_role.setdefault(card.role, []).append(card)
        for role in ("market", "news", "fundamentals"):
            for card in by_role.get(role, []):
                text = " ".join((card.headline, *card.lines))[:20_000]
                documents.append(SourceDocument(
                    document_id=f"{self.ticker}.{card.skill}.{day}",
                    available_at=self.as_of,
                    text=text,
                    roles=(role,),
                    numeric_values=(
                        NumericValue(key="signal_score", value=round(card.score, 6)),
                        NumericValue(key="signal_confidence", value=round(card.confidence, 6)),
                        *(
                            NumericValue(key=key, value=value)
                            for key, value in sorted(card.facts.items())
                            if key not in {"signal_score", "signal_confidence"}
                        ),
                    ),
                ))
        risk_text = " ".join(self.risk_lines) or "No red flags were raised by the research skills."
        documents.append(SourceDocument(
            document_id=f"{self.ticker}.risk_review.{day}",
            available_at=self.as_of,
            text=risk_text,
            roles=("risk",),
        ))
        return tuple(documents)

    def snapshot(self) -> SealedSourceSnapshot:
        return SealedSourceSnapshot.seal(self.source_documents())

    def to_record(self) -> dict:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "lineage": self.lineage,
            "convergence": self.convergence.to_record(),
            "cards": [card.to_record() for card in self.cards],
            "risk": list(self.risk_lines),
        }


_RISK_FLAGS = {
    "SPAM_CONCENTRATED_MENTIONS": "Retail chatter is concentrated in few accounts, a promotion risk.",
    "CROWDED_RETAIL_SURGE": "Retail sentiment is one-sided after a mention surge, so late buyers may be exit liquidity.",
    "PRICED_IN_RISK": "The earnings beat may already be priced in.",
    "STRETCHED_MOVE": "Price is overextended relative to normal volatility.",
    "LOW_CREDIBILITY_SOURCE": "The lead story comes from a low-credibility source.",
    "RETAIL_VS_STREET_DIVERGENCE": "Retail enthusiasm is not shared by the sell side.",
    "COMMITTEE_OVERSIGHT_TRADE": "A Congressional trade may reflect committee information and could reverse on disclosure scrutiny.",
    "METHODS_DISAGREE": "Valuation methods disagree.",
}


def build_dossier(
    view: StoreView,
    ticker: str,
    *,
    benchmark: str | None = None,
    peers: tuple[str, ...] = (),
    news_scorer: NewsScorer | None = None,
    valuation: ValuationAssumptions = ValuationAssumptions(),
    news_lookback_days: int = 5,
    retail_window_days: int = 1,
) -> Dossier:
    cards = (
        analyze_news(view, ticker, scorer=news_scorer, lookback_days=news_lookback_days),
        analyze_sentiment(view, ticker, recent_days=retail_window_days),
        analyze_smart_money(view, ticker),
        analyze_earnings(view, ticker),
        analyze_valuation(view, ticker, benchmark=benchmark, peers=peers, assumptions=valuation),
        analyze_technicals(view, ticker),
    )
    flags = [flag for card in cards for flag in card.flags]
    risk_lines = tuple(dict.fromkeys(_RISK_FLAGS[flag] for flag in flags if flag in _RISK_FLAGS))
    return Dossier(
        ticker=ticker,
        as_of=view.as_of,
        cards=cards,
        convergence=convergence(cards),
        lineage=view.lineage(ticker),
        risk_lines=risk_lines,
    )


__all__ = ["Convergence", "Dossier", "SKILLS", "build_dossier", "convergence"]
