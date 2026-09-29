"""Synthetic data vendor: raw research records from the planted-event world.

It emits the same record types the live collectors produce, consistent with
each planted event, so the whole collection -> analysis -> LLM -> ML pipeline
can run offline and leakage-free:

* genuine events carry corroborating footprints: a credible wire story, an
  earnings report whose surprise matches the true sign, cluster insider buying
  or selling filed on the day, sell-side rating changes, and a broad, split
  retail conversation;
* hype events carry the fingerprints of promotion: a low-credibility blog post,
  a retail mention spike from few accounts with one-sided bullishness, no
  discretionary insider buying, and no sell-side follow-through.

Background noise (routine plan sales, grants, rating reiterations, chatter,
Congressional and 13F activity) keeps every skill honest about base rates.
"""
from __future__ import annotations

import numpy as np

from alpha.fund.synthetic import BENCHMARK, SyntheticWorld
from alpha.research.records import (
    AnalystAction, CongressTrade, EarningsReport, EarningsTranscript, Fundamentals,
    InsiderTransaction, InstitutionalPosition, NewsArticle, PriceBar, Relationship,
    SocialMentions,
)

_SECTORS = ("technology", "healthcare", "industrials", "consumer", "financials")
_FIRMS = ("Atlas", "Birch", "Cobalt", "Delta", "Ember", "Fjord", "Garnet", "Harbor")
_RATINGS = ("strong_sell", "sell", "hold", "buy", "strong_buy")
_WIRE_UP = (
    "{t} beats estimates and raises guidance on strong demand",
    "{t} wins $2.5 billion contract; orders accelerated",
    "{t} record quarter as margins improved",
)
_WIRE_DOWN = (
    "{t} misses estimates and cuts guidance as demand weakens",
    "{t} faces investigation; orders declined",
    "{t} warns margins deteriorated after weak quarter",
)
_BLOG_UP = (
    "{t} is the next big thing, growth is strong and a surge is coming",
    "Why {t} could surge: strong story, record hype",
)
_BLOG_DOWN = (
    "{t} is doomed, demand is weak and a plunge is coming",
    "Why {t} could plunge: weak story, fraud whispers",
)
_CONSTRAINTS = (
    "We remain supply constrained and lead times have extended.",
    "Demand exceeds supply and we are investing heavily in capacity.",
)


def _day(world: SyntheticWorld, index: int, clock: str = "20:30:00") -> str:
    return f"{world.dates[index].date().isoformat()}T{clock}Z"


def world_records(world: SyntheticWorld, *, seed: int | None = None) -> list:
    rng = np.random.default_rng(world.config.seed + 1 if seed is None else seed)
    records: list = []
    n = len(world.dates)
    names = [*world.tickers, BENCHMARK]

    for name in names:
        opens = world.opens[name].to_numpy()
        closes = world.closes[name].to_numpy()
        volume = world.volume[name].to_numpy() if name in world.volume else np.full(n, 1e7)
        wiggle = np.abs(rng.normal(0.0, 0.004, size=(n, 2)))
        for index in range(n):
            top = max(opens[index], closes[index])
            bottom = min(opens[index], closes[index])
            records.append(PriceBar(
                ticker=name, event_at=_day(world, index, "20:00:00"),
                available_at=_day(world, index, "20:00:00"), source="synthetic-tape",
                open=float(opens[index]), close=float(closes[index]),
                high=float(top * (1 + wiggle[index, 0])), low=float(bottom * (1 - wiggle[index, 1])),
                volume=float(volume[index]),
            ))

    sector_of = {t: _SECTORS[i % len(_SECTORS)] for i, t in enumerate(world.tickers)}
    for position, ticker in enumerate(world.tickers):
        revenue = float(rng.uniform(2e9, 40e9))
        growth = float(rng.uniform(-0.02, 0.18))
        shares = float(rng.uniform(2e8, 2e9))
        for year, index in enumerate(range(0, n, 252)):
            revenue *= 1.0 + growth
            margin = float(rng.uniform(0.08, 0.2))
            records.append(Fundamentals(
                ticker=ticker, event_at=_day(world, max(0, index - 30)), available_at=_day(world, index),
                source="synthetic-filings", period_end=world.dates[max(0, index - 30)].date(),
                revenue=revenue, net_income=revenue * margin,
                operating_cash_flow=revenue * (margin + 0.05), capex=revenue * 0.04,
                shares_outstanding=shares, total_debt=revenue * 0.3, cash=revenue * 0.1,
                equity=revenue * 0.6, dividends_per_share=float(rng.choice([0.0, 0.5, 1.2])),
                revenue_growth_3y=growth, sector=sector_of[ticker],
            ))
        partner = world.tickers[(position + 7) % len(world.tickers)]
        records.append(Relationship(
            ticker=ticker, counterparty=partner, event_at=_day(world, 0), available_at=_day(world, 0),
            source="synthetic-graph", relation="supplier" if position % 2 else "competitor",
        ))
        price0 = float(world.closes[ticker].iloc[0])
        for index in range(0, n, 60):
            price = float(world.closes[ticker].iloc[index])
            for firm in rng.choice(_FIRMS, size=4, replace=False):
                rating = str(rng.choice(_RATINGS[1:4], p=(0.15, 0.5, 0.35)))
                records.append(AnalystAction(
                    ticker=ticker, event_at=_day(world, index), available_at=_day(world, index),
                    source="synthetic-ratings", firm=str(firm), rating=rating,
                    price_target=price * float(rng.uniform(0.95, 1.2)),
                ))
        for index in range(0, n):
            mentions = int(rng.poisson(20))
            bulls = int(rng.binomial(mentions, 0.5) * 0.7)
            bears = int((mentions - bulls) * 0.7)
            records.append(SocialMentions(
                ticker=ticker, event_at=_day(world, index, "20:00:00"),
                available_at=_day(world, index, "20:00:00"), source="synthetic-social",
                platform="reddit", mentions=mentions,
                unique_accounts=int(mentions * rng.uniform(0.6, 0.95)), bullish=bulls, bearish=bears,
            ))
        for index in range(20, n, 45):
            records.append(InsiderTransaction(
                ticker=ticker, event_at=_day(world, index - 2, "00:00:00"),
                available_at=_day(world, index, "22:00:00"), source="synthetic-form4",
                insider=f"Officer {position}", role="CFO", code="S", acquired=False,
                shares=float(rng.integers(1_000, 20_000)), price=price0, rule_10b5_1=True,
            ))
        if position % 6 == 0:
            for index in range(100, n, 63):
                records.append(InstitutionalPosition(
                    ticker=ticker, event_at=_day(world, index - 45), available_at=_day(world, index),
                    source="synthetic-13f", filer=f"Fund {int(rng.integers(1, 9))}",
                    shares=float(rng.integers(1, 5)) * 1e5, value=1e7, prior_shares=1e5,
                ))
        if position % 5 == 0:
            for index in range(50, n, 90):
                records.append(CongressTrade(
                    ticker=ticker, event_at=_day(world, index - 30), available_at=_day(world, index),
                    source="synthetic-stock-act", member=f"Member {position}", chamber="house",
                    direction="purchase" if rng.random() < 0.5 else "sale",
                    amount_low=1_001, amount_high=15_000,
                ))

    for event in world.events:
        index, ticker, sign = event.session_index, event.ticker, event.sign
        stamp = _day(world, index)
        price = float(world.closes[ticker].iloc[index])
        if event.kind == "genuine":
            text_up = event.text_sign > 0
            headline = str(rng.choice(_WIRE_UP if text_up else _WIRE_DOWN)).format(t=ticker)
            records.append(NewsArticle(
                ticker=ticker, event_at=stamp, available_at=stamp, source="wire",
                headline=headline, body=f"{headline}. Management said the shift was unexpected.",
            ))
            records.append(EarningsReport(
                ticker=ticker, event_at=stamp, available_at=stamp, source="synthetic-earnings",
                fiscal_period=f"Q{1 + (index // 63) % 4}", timing="post_market",
                eps_actual=1.0 + 0.08 * sign * event.strength, eps_estimate=1.0,
                revenue_actual=1e9 * (1 + 0.04 * sign), revenue_estimate=1e9,
                guidance_mid=1e9 * (1 + 0.06 * sign * event.strength), prior_guidance_mid=1e9,
                gross_margin=0.4 + 0.01 * sign, prior_gross_margin=0.4,
            ))
            if sign > 0 and rng.random() < 0.5:
                records.append(EarningsTranscript(
                    ticker=ticker, event_at=stamp, available_at=stamp, source="synthetic-transcripts",
                    fiscal_period="call", text=" ".join(("Thank you all for joining.", *_CONSTRAINTS)),
                ))
            for k in range(2 + int(rng.integers(0, 2))):
                records.append(InsiderTransaction(
                    ticker=ticker, event_at=_day(world, index, "00:00:00"), available_at=stamp,
                    source="synthetic-form4", insider=f"Director {ticker} {k}", role="Director",
                    code="P" if sign > 0 else "S", acquired=sign > 0,
                    shares=float(rng.integers(5_000, 40_000)), price=price, rule_10b5_1=False,
                ))
            for firm in rng.choice(_FIRMS, size=2, replace=False):
                records.append(AnalystAction(
                    ticker=ticker, event_at=stamp, available_at=stamp, source="synthetic-ratings",
                    firm=str(firm), rating="buy" if sign > 0 else "sell", prior_rating="hold",
                    price_target=price * (1 + 0.2 * sign),
                ))
            records.append(SocialMentions(
                ticker=ticker, event_at=stamp, available_at=stamp, source="synthetic-social",
                platform="x", mentions=120, unique_accounts=100,
                bullish=55 if sign > 0 else 35, bearish=35 if sign > 0 else 55,
            ))
        else:
            headline = str(rng.choice(_BLOG_UP if sign > 0 else _BLOG_DOWN)).format(t=ticker)
            records.append(NewsArticle(
                ticker=ticker, event_at=stamp, available_at=stamp, source="blog",
                headline=headline, body=headline,
            ))
            start = max(0, index - world.config.hype_runup_days + 1)
            for day in range(start, index + 1):
                mentions = int(400 + 150 * (day - start))
                records.append(SocialMentions(
                    ticker=ticker, event_at=_day(world, day), available_at=_day(world, day),
                    source="synthetic-social", platform="x", mentions=mentions,
                    unique_accounts=int(mentions * 0.12),
                    bullish=int(mentions * (0.9 if sign > 0 else 0.05)),
                    bearish=int(mentions * (0.05 if sign > 0 else 0.9)),
                ))
    return records


__all__ = ["world_records"]
