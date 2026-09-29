from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from alpha.agents.orchestrator import FusionOrchestrator
from alpha.fund.ideas import IdeaRequest, originate
from alpha.fund.synthetic import BENCHMARK, WorldConfig, build_world
from alpha.research.cards import SignalCard
from alpha.research.collectors.edgar import parse_companyfacts, parse_form4
from alpha.research.collectors.synthetic import world_records
from alpha.research.collectors.vendor import JsonlVendorCollector
from alpha.research.dossier import build_dossier, convergence
from alpha.research.offline import DossierAnalystProvider, DossierOriginator
from alpha.research.portfolio import review_portfolio
from alpha.research.records import (
    AnalystAction, CongressTrade, EarningsReport, EarningsTranscript, InsiderTransaction,
    NewsArticle, PriceBar, SocialMentions,
)
from alpha.research.skills.earnings import analyze_earnings
from alpha.research.skills.news import LexicalNewsScorer, analyze_news
from alpha.research.skills.sentiment import analyze_sentiment
from alpha.research.skills.smart_money import analyze_smart_money
from alpha.research.skills.technicals import analyze_technicals, cluster, pivots
from alpha.research.skills.valuation import analyze_valuation, dcf_per_share
from alpha.research.store import ResearchStore

FIXTURES = Path(__file__).parent / "fixtures"


def _bars(ticker: str, closes: list[float], start: str = "2025-01-02") -> list[PriceBar]:
    dates = pd.bdate_range(start, periods=len(closes))
    return [
        PriceBar(
            ticker=ticker, event_at=f"{d.date()}T20:00:00Z", available_at=f"{d.date()}T20:00:00Z",
            source="test", open=c, high=c * 1.01, low=c * 0.99, close=c, volume=1e6,
        )
        for d, c in zip(dates, closes)
    ]


@pytest.fixture(scope="module")
def world_store():
    world = build_world(WorldConfig(n_names=24, n_sessions=380, event_rate=0.04, seed=11))
    store = ResearchStore()
    store.add(world_records(world))
    return world, store


def test_records_reject_availability_before_event() -> None:
    with pytest.raises(ValidationError, match="available before"):
        InsiderTransaction(
            ticker="ACME", event_at="2026-01-05T00:00:00Z", available_at="2026-01-04T00:00:00Z",
            source="t", insider="A", code="P", acquired=True, shares=10, price=1.0,
        )


def test_store_is_point_in_time_and_deduplicates(tmp_path: Path) -> None:
    store = ResearchStore()
    trade = InsiderTransaction(
        ticker="ACME", event_at="2026-01-05T00:00:00Z", available_at="2026-01-07T22:00:00Z",
        source="t", insider="A", code="P", acquired=True, shares=10, price=1.0,
    )
    assert store.add([trade, trade]) == 1
    assert store.view("2026-01-06T00:00:00Z").records("insider", "ACME") == []
    assert store.view("2026-01-08T00:00:00Z").records("insider", "ACME") == [trade]
    store.save(tmp_path)
    reloaded = ResearchStore.load(tmp_path)
    assert len(reloaded) == 1
    view = reloaded.view("2026-01-08T00:00:00Z")
    assert view.lineage("ACME") == store.view("2026-01-08T00:00:00Z").lineage("ACME")


def test_news_is_indexed_under_every_mentioned_ticker() -> None:
    store = ResearchStore()
    store.add([NewsArticle(
        ticker="AAA", mentioned_tickers=("BBB",), event_at="2026-01-05T15:00:00Z",
        available_at="2026-01-05T15:00:00Z", source="wire", headline="AAA to acquire BBB for $5 billion",
    )])
    assert len(store.view("2026-01-06T00:00:00Z").records("news", "BBB")) == 1


def test_form4_parser_reads_a_real_edgar_document() -> None:
    rows = parse_form4(
        (FIXTURES / "sec_form4_aapl_sample.xml").read_bytes(),
        ticker="AAPL", accepted_at="2026-09-24T22:30:07Z", accession="0001140361-26-037584",
    )
    assert len(rows) == 1
    row = rows[0]
    assert (row.code, row.acquired, row.shares, row.price) == ("S", False, 2399.0, 340.06)
    assert row.rule_10b5_1 is True
    assert row.discretionary_sale is False
    assert row.available_at == "2026-09-24T22:30:07Z"


def test_companyfacts_parser_uses_filing_date_for_availability() -> None:
    payload = {"facts": {
        "us-gaap": {
            "Revenues": {"units": {"USD": [
                {"start": "2024-01-01", "end": "2024-12-31", "val": 100.0, "form": "10-K", "fp": "FY", "filed": "2025-02-20"},
                {"start": "2024-07-01", "end": "2024-09-30", "val": 30.0, "form": "10-Q", "fp": "Q3", "filed": "2024-11-01"},
            ]}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
                {"start": "2024-01-01", "end": "2024-12-31", "val": 40.0, "form": "10-K", "fp": "FY", "filed": "2025-02-20"},
            ]}},
        },
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2025-02-01", "val": 10.0, "filed": "2025-02-20"},
        ]}}},
    }}
    rows = parse_companyfacts(payload, ticker="ACME")
    assert len(rows) == 1
    assert rows[0].revenue == 100.0 and rows[0].operating_cash_flow == 40.0
    assert rows[0].shares_outstanding == 10.0
    assert rows[0].available_at.startswith("2025-02-21T04:59:59")


def test_vendor_collector_reports_invalid_rows(tmp_path: Path) -> None:
    good = SocialMentions(
        ticker="ACME", event_at="2026-01-05T20:00:00Z", available_at="2026-01-05T20:00:00Z",
        source="x", platform="x", mentions=10, unique_accounts=5, bullish=4, bearish=3,
    ).model_dump_json()
    (tmp_path / "social.jsonl").write_text(good + "\n" + '{"kind": "social", "ticker": "acme"}\n')
    collector = JsonlVendorCollector(tmp_path)
    assert len(collector.collect()) == 1
    assert collector.rejected and collector.rejected[0][1] == 2


def test_news_scorer_penalises_low_credibility_sources() -> None:
    scorer = LexicalNewsScorer()
    base = dict(ticker="ACME", event_at="2026-01-05T15:00:00Z", available_at="2026-01-05T15:00:00Z")
    wire = scorer.score(NewsArticle(source="wire", headline="ACME beats and raises guidance after $3 billion contract", **base))
    blog = scorer.score(NewsArticle(source="blog", headline="ACME beats and raises guidance after $3 billion contract", **base))
    assert wire.direction == 1 and wire.impact > blog.impact


def test_news_propagates_to_competitors_with_opposite_sign(world_store) -> None:
    from alpha.research.records import Relationship

    store = ResearchStore()
    store.add([
        Relationship(ticker="BBB", counterparty="AAA", relation="competitor", event_at="2026-01-01T00:00:00Z",
                     available_at="2026-01-01T00:00:00Z", source="t"),
        NewsArticle(ticker="AAA", event_at="2026-01-05T15:00:00Z", available_at="2026-01-05T15:00:00Z",
                    source="wire", headline="AAA beats estimates and raises guidance on record demand"),
    ])
    card = analyze_news(store.view("2026-01-06T00:00:00Z"), "BBB")
    assert card.score < 0 and "SECOND_ORDER_ONLY" in card.flags


def test_smart_money_ignores_plan_and_compensation_lines() -> None:
    store = ResearchStore()
    store.add(_bars("ACME", [100.0] * 60 + [70.0] * 5))
    common = dict(ticker="ACME", event_at="2025-03-28T00:00:00Z", available_at="2025-03-31T21:00:00Z", source="t")
    store.add([
        InsiderTransaction(insider="A", code="P", acquired=True, shares=20_000, price=70.0, **common),
        InsiderTransaction(insider="B", code="P", acquired=True, shares=10_000, price=70.0, **common),
        InsiderTransaction(insider="C", code="S", acquired=False, shares=50_000, price=70.0, rule_10b5_1=True, **common),
        InsiderTransaction(insider="D", code="A", acquired=True, shares=90_000, price=0.0, **common),
        CongressTrade(ticker="ACME", event_at="2025-02-20T00:00:00Z", available_at="2025-03-30T00:00:00Z",
                      source="t", member="M", chamber="senate", direction="purchase",
                      amount_low=1_001, amount_high=15_000, committee_oversees_issuer=True),
    ])
    card = analyze_smart_money(store.view("2025-04-01T00:00:00Z"), "ACME")
    assert card.facts["insider_buyers_30d"] == 2.0
    assert card.facts["insider_sellers_30d"] == 0.0
    assert {"INSIDER_CLUSTER_BUY", "INSIDER_BUY_THE_DIP", "COMMITTEE_OVERSIGHT_TRADE"} <= set(card.flags)
    assert card.direction == "bullish"


def test_sentiment_reads_spam_concentrated_euphoria_as_a_warning() -> None:
    store = ResearchStore()
    store.add(_bars("ACME", [50.0] * 30))
    for day in range(1, 8):
        stamp = f"2025-02-{day:02d}T20:00:00Z"
        store.add([SocialMentions(ticker="ACME", event_at=stamp, available_at=stamp, source="t",
                                  platform="x", mentions=20, unique_accounts=18, bullish=8, bearish=8)])
    stamp = "2025-02-10T20:00:00Z"
    store.add([SocialMentions(ticker="ACME", event_at=stamp, available_at=stamp, source="t",
                              platform="x", mentions=900, unique_accounts=60, bullish=850, bearish=20)])
    card = analyze_sentiment(store.view("2025-02-10T21:00:00Z"), "ACME")
    assert "SPAM_CONCENTRATED_MENTIONS" in card.flags and "CROWDED_RETAIL_SURGE" in card.flags
    assert card.score < 0


def test_earnings_weights_guidance_and_finds_bottleneck_language() -> None:
    store = ResearchStore()
    store.add(_bars("ACME", [40.0] * 30))
    stamp = "2025-02-10T21:05:00Z"
    store.add([
        EarningsReport(ticker="ACME", event_at=stamp, available_at=stamp, source="t", fiscal_period="Q4",
                       eps_actual=1.1, eps_estimate=1.0, revenue_actual=105, revenue_estimate=100,
                       guidance_mid=90, prior_guidance_mid=100),
        EarningsTranscript(ticker="ACME", event_at=stamp, available_at=stamp, source="t", fiscal_period="Q4",
                           text="Thanks. We remain supply constrained on memory. Lead times are long."),
    ])
    card = analyze_earnings(store.view("2025-02-11T00:00:00Z"), "ACME")
    assert card.score < 0, "a guidance cut outweighs a beat"
    assert "BOTTLENECK_LANGUAGE" in card.flags
    assert "We remain supply constrained on memory." in card.lines


def test_dcf_matches_a_hand_computation() -> None:
    value = dcf_per_share(100.0, 0.0, 0.1, years=1, terminal_growth=0.0, net_debt=0.0, shares=10.0)
    assert value == pytest.approx((100 / 1.1 + 1000 / 1.1) / 10)
    with pytest.raises(ValueError):
        dcf_per_share(1.0, 0.0, 0.02, years=5, terminal_growth=0.03, net_debt=0.0, shares=1.0)


def test_technical_pivots_never_use_unconfirmed_bars() -> None:
    frame = pd.DataFrame({"high": [1, 2, 3, 9, 3, 2, 1, 50.0], "low": [0.5] * 8})
    highs, _ = pivots(frame, 2)
    assert 9.0 in highs and 50.0 not in highs
    assert cluster([10.0, 10.2, 20.0], 0.5) == [(10.1, 2), (20.0, 1)]


def test_technicals_report_an_uptrend_plan() -> None:
    closes = list(np.linspace(50, 100, 260) + np.sin(np.arange(260) / 4) * 3)
    store = ResearchStore()
    store.add(_bars("ACME", closes))
    card = analyze_technicals(store.view("2026-12-31T00:00:00Z"), "ACME")
    assert card.direction == "bullish"
    assert card.facts["stop"] < card.facts["entry"] <= closes[-1] * 1.01 < card.facts["target"] + closes[-1]


def test_convergence_tiers() -> None:
    def card(skill, score):
        return SignalCard(skill=skill, ticker="A", as_of="x", score=score, confidence=0.8, headline="h")

    assert convergence((card("a", 0.5), card("b", 0.5), card("c", 0.5))).tier == "size"
    assert convergence((card("a", 0.5), card("b", 0.5))).tier == "research"
    mixed = convergence((card("a", 0.5), card("b", -0.5)))
    assert mixed.direction == "mixed" and mixed.tier == "watch"


def test_dossier_separates_genuine_catalysts_from_promotion(world_store) -> None:
    world, store = world_store
    genuine = next(e for e in world.events if e.kind == "genuine" and e.text_sign == e.sign)
    hype = next(e for e in world.events if e.kind == "hype")
    g = build_dossier(store.view(world.as_of(genuine.session_index)), genuine.ticker, benchmark=BENCHMARK)
    h = build_dossier(store.view(world.as_of(hype.session_index)), hype.ticker, benchmark=BENCHMARK)
    expected = "bullish" if genuine.sign > 0 else "bearish"
    assert g.convergence.direction == expected
    assert g.convergence.tier in {"research", "size"}
    assert "LOW_CREDIBILITY_SOURCE" in {f for c in h.cards for f in c.flags}
    assert h.risk_lines
    assert len(g.feature_vector()) == len(g.feature_names())


def test_offline_llm_reads_the_dossier_and_passes_the_evidence_audit(world_store) -> None:
    world, store = world_store
    event = next(e for e in world.events if e.kind == "genuine" and e.text_sign == e.sign)
    as_of = world.as_of(event.session_index)
    dossier = build_dossier(store.view(as_of), event.ticker, benchmark=BENCHMARK)
    snapshot = dossier.snapshot()
    originated = originate(DossierOriginator(), IdeaRequest(ticker=event.ticker, as_of=as_of, snapshot=snapshot))
    assert originated.proposal is not None
    assert originated.proposal.direction == ("positive" if event.sign > 0 else "negative")
    receipt = FusionOrchestrator(provider=DossierAnalystProvider()).run(originated.proposal, snapshot)
    assert receipt.evidence_audit is not None and receipt.evidence_audit.evidence_valid
    receipt.verify_receipt()


def test_portfolio_review_finds_hidden_concentration() -> None:
    rng = np.random.default_rng(0)
    common = np.cumsum(rng.normal(0, 0.01, 200))
    store = ResearchStore()
    for index, ticker in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE")):
        path = 50 * np.exp(common + np.cumsum(rng.normal(0, 0.001, 200)))
        store.add(_bars(ticker, list(path)))
    review = review_portfolio(store.view("2026-12-31T00:00:00Z"), {t: 0.2 for t in ("AAA", "BBB", "CCC", "DDD", "EEE")})
    assert review.metrics["effective_bets"] < 2.0
    assert any("independent bets" in finding for finding in review.findings)
    assert set(review.lenses) == {"growth", "risk", "income", "sector_balance", "momentum"}


def test_analyst_action_validates_rating_vocabulary() -> None:
    with pytest.raises(ValidationError):
        AnalystAction(ticker="ACME", event_at="2026-01-01T00:00:00Z", available_at="2026-01-01T00:00:00Z",
                      source="t", firm="F", rating="outperform")
    assert date(2026, 1, 1)


def test_valuation_cross_references_dcf_consensus_and_peers() -> None:
    from alpha.research.records import Fundamentals

    store = ResearchStore()
    stamp = "2025-02-01T00:00:00Z"
    for ticker, income in (("ACME", 100.0), ("PEER", 400.0)):
        store.add(_bars(ticker, [10.0] * 80))
        store.add([Fundamentals(
            ticker=ticker, event_at=stamp, available_at=stamp, source="t", period_end=date(2024, 12, 31),
            revenue=1_000.0, net_income=income, operating_cash_flow=200.0, capex=50.0,
            shares_outstanding=100.0, total_debt=0.0, cash=0.0, revenue_growth_3y=0.05,
        )])
    store.add([AnalystAction(ticker="ACME", event_at=stamp, available_at=stamp, source="t",
                             firm="F", rating="buy", price_target=14.0)])
    card = analyze_valuation(store.view("2025-06-01T00:00:00Z"), "ACME", peers=("PEER",))
    assert card.facts["dcf_value"] > 10.0
    assert card.facts["consensus_upside_pct"] == pytest.approx(40.0)
    assert card.facts["pe_vs_peers_pct"] == pytest.approx(300.0)
    assert "METHODS_DISAGREE" in card.flags


def test_cli_prints_a_dossier_and_a_portfolio_review() -> None:
    import contextlib
    import io

    from alpha.fund.__main__ import main

    research, review = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(research):
        assert main(["research", "S03", "--sessions", "300", "--as-of", "2024-02-01T21:00:00Z"]) == 0
    with contextlib.redirect_stdout(review):
        assert main(["review", "S01=0.5", "S02=0.5", "--sessions", "300"]) == 0
    assert "convergence:" in research.getvalue() and "[technicals]" in research.getvalue()
    assert '"lenses"' in review.getvalue()
