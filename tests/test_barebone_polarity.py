"""Frozen lexicon polarity and the narrative skill gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from demo.barebone_comparison import (
    LOCKED_NARRATIVE_SHA256,
    NARRATIVE_EVENTS,
    NARRATIVE_SCORES,
    load_barebone_comparison_config,
)
from demo.barebone_narrative_run import (
    LEDGER_RELATIVE,
    METRICS_RELATIVE,
    barebone_narrative_ledger,
    barebone_narrative_metrics,
    narrative_book,
    narrative_oos_pairs,
    narrative_rebalances,
    skill_before,
)
from demo.barebone_polarity import (
    POLARITY_MODEL_ID,
    load_polarity_lexicon,
    score_events,
    score_text,
    write_locked_polarity_sha256,
)
from demo.contracts import AssetBar, ExperimentConfig, MarketSession
from demo.controlled import ArmInput, benchmark_marks_from_closes, run_controlled_arm
from demo.pure_ml import skill_allows_book

_LEGACY_METRICS_SHA256 = "e7e5055ce9b4409d7941a71929f66261416b8d3c3f62423190edafd5bf5b1411"
_LOCKED_TAPE_SHA256 = "c29f4810a8433e0de286da46409dbe95c17c1fa09d25e7809bb5f9e73ad8a205"
_MOMENTUM_RETURN = 0.24597743358179014
_MOMENTUM_SHARPE = 1.6030790628765745
_SESSIONS = (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6))


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _event(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_id": "hn:1:AAPL",
        "ticker": "AAPL",
        "available_ts": "2025-01-02T15:00:00Z",
        "decision_session": "2025-01-03",
        "text": "Apple profit surged after the upgrade",
    }
    payload.update(updates)
    return payload


def _trading() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "schema_version": "1.0",
            "experiment_id": "barebone-comparison-v1",
            "start_date": "2025-01-02",
            "end_date": "2025-01-17",
            "starting_capital": 10000.0,
            "universe": ["AAA", "BBB"],
            "benchmark_ticker": "SPY",
            "rebalance_frequency_sessions": 10,
            "execution_lag_sessions": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_gross_leverage": 1.0,
            "max_position_weight": 0.1,
            "annualization_sessions": 252,
            "annual_risk_free_rate": 0.0,
        }
    )


def test_polarity_is_deterministic_and_negation_flips_the_sign() -> None:
    lexicon = load_polarity_lexicon()
    first = score_text("The company profit surged", lexicon)
    second = score_text("The company profit surged", lexicon)

    assert first == second
    assert first > 0.0
    assert score_text("The company did not report a profit", lexicon) < 0.0
    assert score_text("Banana bread recipe", lexicon) == 0.0
    assert lexicon.model_id == POLARITY_MODEL_ID
    assert lexicon.sha256 == hashlib.sha256(
        (_root() / "evidence" / "narrative" / "barebone_polarity_lexicon.json").read_bytes()
    ).hexdigest()


def test_look_ahead_missing_timestamp_and_unmapped_ticker_are_refused() -> None:
    with pytest.raises(ValueError, match="strictly after"):
        score_events([_event(decision_session="2025-01-02")], _SESSIONS)
    with pytest.raises(ValueError, match="strictly after"):
        score_events([_event(decision_session="2025-01-06")], _SESSIONS)
    with pytest.raises(ValueError, match="missing available_ts"):
        score_events([_event(available_ts="")], _SESSIONS)
    with pytest.raises(ValueError, match="sealed map universe"):
        score_events([_event(ticker="ZZZ", event_id="hn:1:ZZZ")], _SESSIONS)
    with pytest.raises(ValueError, match="return label"):
        score_events([_event(total_return=0.2)], _SESSIONS)
    rows = score_events([_event(), _event(event_id="hn:2:MSFT", ticker="MSFT", text="Microsoft loss warning")], _SESSIONS)
    assert [row["decision_session"] for row in rows] == ["2025-01-03", "2025-01-03"]
    assert rows[0]["polarity_model_id"] == POLARITY_MODEL_ID
    assert "text" not in rows[0]


def test_non_positive_skill_is_cash_and_future_prices_do_not_change_past_skill() -> None:
    config = _trading()
    means = {"AAA": 1.0, "BBB": -1.0}

    assert narrative_book(means, None, config) == {}
    assert narrative_book(means, 0.0, config) == {}
    assert narrative_book(means, -0.2, config) == {}
    sized = narrative_book(means, 0.2, config)
    assert set(sized) == {"AAA"}
    assert sized["AAA"] <= 0.1

    dates = tuple(date(2025, 1, 2) + timedelta(days=offset) for offset in range(16))
    closes = {
        ("AAA", index): 100.0 - index
        for index in range(16)
    }
    closes.update({("BBB", index): 100.0 + index for index in range(16)})
    by_session = {day: {"AAA": 1.0, "BBB": -1.0} for day in dates}
    pairs = narrative_oos_pairs(by_session, closes, dates)
    skill, count = skill_before(pairs, 8)
    assert count >= 8
    assert skill is not None and skill < 0.0
    assert skill_allows_book(skill) is False
    later = dict(closes)
    later[("AAA", 12)] = 1.0
    assert skill_before(narrative_oos_pairs(by_session, later, dates), 8) == (skill, count)
    changed = dict(closes)
    changed[("AAA", 4)] = 50.0
    assert skill_before(narrative_oos_pairs(by_session, changed, dates), 8) != (skill, count)
    rows = narrative_rebalances(by_session, closes, dates, config)
    assert rows
    assert all(row["sized"] is False for row in rows)


def test_aligned_polarity_can_size_and_the_kernel_accepts_the_narrative_arm() -> None:
    config = _trading()
    dates = tuple(date(2025, 1, 2) + timedelta(days=offset) for offset in range(16))
    closes = {("AAA", index): 100.0 + index for index in range(16)}
    closes.update({("BBB", index): 100.0 - index for index in range(16)})
    by_session = {day: {"AAA": 1.0, "BBB": -1.0} for day in dates}
    rows = narrative_rebalances(by_session, closes, dates, config)
    assert any(row["sized"] is True for row in rows)
    assert all(row["sized"] is False or (row["oos_skill"] is not None and float(row["oos_skill"]) > 0.0) for row in rows)

    sessions = tuple(
        MarketSession(
            session=day,
            bars=(
                AssetBar(ticker="AAA", open=100.0, close=101.0),
                AssetBar(ticker="BBB", open=100.0, close=99.0),
                AssetBar(ticker="SPY", open=100.0, close=100.0),
            ),
        )
        for day in (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6))
    )
    trading = ExperimentConfig.model_validate(
        {
            **config.model_dump(mode="json"),
            "start_date": "2025-01-02",
            "end_date": "2025-01-06",
            "rebalance_frequency_sessions": 1,
        }
    )
    marks = benchmark_marks_from_closes(
        sessions, benchmark_ticker="SPY", starting_capital=trading.starting_capital
    )
    sized = run_controlled_arm(
        config=trading,
        sessions=sessions,
        strategy_id="narrative",
        candidates=[
            ArmInput(
                strategy_id="narrative",
                decision_session=date(2025, 1, 2),
                ticker="AAA",
                structured_weight=0.1,
            )
        ],
        benchmark_marks=marks,
    )
    cash = run_controlled_arm(
        config=trading,
        sessions=sessions,
        strategy_id="narrative",
        candidates=[
            ArmInput(
                strategy_id="narrative",
                decision_session=date(2025, 1, 2),
                ticker="AAA",
                structured_weight=0.0,
            )
        ],
        benchmark_marks=marks,
    )
    assert sized.block_reason is None and sized.admitted_count == 1
    assert cash.block_reason is None and cash.admitted_count == 0
    with pytest.raises(ValueError, match="unknown strategy arm"):
        run_controlled_arm(
            config=trading,
            sessions=sessions,
            strategy_id="other",
            candidates=[],
            benchmark_marks=marks,
        )
    leaked = ArmInput.model_construct(
        strategy_id="narrative",
        decision_session=date(2025, 1, 2),
        ticker="AAA",
        structured_weight=0.1,
        receipt="sealed-later",
        market=None,
        outcome_ts=None,
    )
    with pytest.raises(ValueError, match="cannot take an evidence receipt"):
        run_controlled_arm(
            config=trading,
            sessions=sessions,
            strategy_id="narrative",
            candidates=[leaked],
            benchmark_marks=marks,
        )


def test_polarity_lock_refuses_a_changed_narrative_hash_or_a_claim(tmp_path: Path) -> None:
    source = json.loads((_root() / "configs" / "barebone-comparison-v1.json").read_text(encoding="utf-8"))
    source["evidence"]["narrative_sha256"] = "a" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="changed narrative_sha256"):
        write_locked_polarity_sha256(path, "b" * 64)
    source["evidence"]["narrative_sha256"] = LOCKED_NARRATIVE_SHA256
    source["comparable_performance_claim"] = True
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="comparable_performance_claim must be false"):
        write_locked_polarity_sha256(path, "b" * 64)


def test_scores_are_gitignored_and_the_momentum_control_stays_put() -> None:
    root = _root()
    ignored = subprocess.run(["git", "check-ignore", "-q", NARRATIVE_SCORES], cwd=root, check=False)
    events = subprocess.run(["git", "check-ignore", "-q", NARRATIVE_EVENTS], cwd=root, check=False)
    ohlcv = subprocess.run(
        ["git", "check-ignore", "-q", "evidence/market/barebone_window_ohlcv.json"],
        cwd=root,
        check=False,
    )
    assert ignored.returncode == 0
    assert events.returncode == 0
    assert ohlcv.returncode == 0
    config = load_barebone_comparison_config()
    metrics = json.loads((root / "results" / "barebone_three_arm_metrics.json").read_text(encoding="utf-8"))
    assert config.evidence.narrative_sha256 == LOCKED_NARRATIVE_SHA256
    assert config.evidence.tape_sha256 == _LOCKED_TAPE_SHA256
    assert metrics["tape_sha256"] == _LOCKED_TAPE_SHA256
    assert metrics["arms"]["pure_ml"]["statistics"]["total_return"] == _MOMENTUM_RETURN
    assert metrics["arms"]["pure_ml"]["statistics"]["sharpe_ratio"] == _MOMENTUM_SHARPE
    assert (
        hashlib.sha256((root / "results" / "metrics.json").read_bytes()).hexdigest()
        == _LEGACY_METRICS_SHA256
    )
    spec = importlib.util.spec_from_file_location(
        "build_submission_archive", root / "scripts" / "build_submission_archive.py"
    )
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    scores = root / NARRATIVE_SCORES
    created = not scores.exists()
    if created:
        scores.parent.mkdir(parents=True, exist_ok=True)
        scores.write_text("{}\n", encoding="utf-8")
    try:
        assert scores not in builder.collect_files()
    finally:
        if created:
            scores.unlink()
    polarity = config.evidence.polarity_sha256
    if polarity is None:
        return
    assert len(polarity) == 64
    provenance = json.loads(
        (root / "evidence" / "narrative" / "barebone_window_polarity_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["polarity_sha256"] == polarity
    assert provenance["narrative_events_sha256"] == LOCKED_NARRATIVE_SHA256
    assert provenance["comparable_performance_claim"] is False
    assert "text" not in provenance
    if scores.is_file() and (root / NARRATIVE_EVENTS).is_file() and (
        root / "evidence" / "market" / "barebone_window_ohlcv.json"
    ).is_file():
        assert hashlib.sha256(scores.read_bytes()).hexdigest() == polarity
        assert hashlib.sha256((root / NARRATIVE_EVENTS).read_bytes()).hexdigest() == LOCKED_NARRATIVE_SHA256
        ledger = json.loads((root / LEDGER_RELATIVE).read_text(encoding="utf-8"))
        arm_metrics = json.loads((root / METRICS_RELATIVE).read_text(encoding="utf-8"))
        assert ledger["comparable_performance_claim"] is False
        assert arm_metrics["comparable_performance_claim"] is False
        assert ledger["narrative_sha256"] == LOCKED_NARRATIVE_SHA256
        assert ledger["polarity_sha256"] == polarity
        assert ledger["tape_sha256"] == _LOCKED_TAPE_SHA256
        for row in ledger["oos_skill"]:
            if row["oos_skill"] is None or float(row["oos_skill"]) <= 0.0:
                assert row["sized"] is False
        assert (root / LEDGER_RELATIVE).read_text(encoding="utf-8") == json.dumps(
            barebone_narrative_ledger(), indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        assert (root / METRICS_RELATIVE).read_text(encoding="utf-8") == json.dumps(
            barebone_narrative_metrics(), indent=2, sort_keys=True, allow_nan=False
        ) + "\n"


def test_score_text_does_not_depend_on_the_clock() -> None:
    lexicon = load_polarity_lexicon()
    before = datetime.now(timezone.utc)
    value = score_text("profit", lexicon)
    after = datetime.now(timezone.utc)
    assert before <= after
    assert value == 1.0
