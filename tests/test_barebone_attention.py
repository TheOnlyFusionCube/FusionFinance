"""PIT attention counts, the skill gate, and the hybrid intersection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
from datetime import date, timedelta
from pathlib import Path

import pytest

import demo.barebone_attention as attention_module
import demo.barebone_attention_run as attention_run_module
from demo.barebone_attention import (
    ATTENTION_LOOKBACK_SESSIONS,
    ATTENTION_SIGNAL_ID,
    FROZEN_LEXICON_SHA256,
    HYBRID_SIGNAL_ID,
    attention_score_rows,
    demean_cross_section,
    write_locked_attention_sha256,
)
from demo.barebone_attention_run import (
    ATTENTION_LEDGER_RELATIVE,
    ATTENTION_METRICS_RELATIVE,
    HYBRID_LEDGER_RELATIVE,
    HYBRID_METRICS_RELATIVE,
    attention_book,
    attention_oos_pairs,
    attention_rebalances,
    barebone_attention_ledger,
    barebone_attention_metrics,
    barebone_hybrid_ledger,
    barebone_hybrid_metrics,
    hybrid_book,
    hybrid_rebalances,
    skill_before,
)
from demo.barebone_comparison import (
    ATTENTION_SCORES,
    LOCKED_NARRATIVE_SHA256,
    LOCKED_POLARITY_SHA256,
    LOCKED_TAPE_SHA256,
    NARRATIVE_EVENTS,
    NARRATIVE_LEXICON,
    load_barebone_comparison_config,
)
from demo.contracts import AssetBar, ExperimentConfig, MarketSession
from demo.controlled import ArmInput, benchmark_marks_from_closes, run_controlled_arm
from demo.pure_ml import skill_allows_book

_LEGACY_METRICS_SHA256 = "e7e5055ce9b4409d7941a71929f66261416b8d3c3f62423190edafd5bf5b1411"
_THREE_ARM_METRICS_SHA256 = "af40d4176d3e4705f799b3b640494777caa24ed81ea68f149a84bb8335d554f3"
_THREE_ARM_LEDGER_SHA256 = "07296e85b8930fa1eb88c8fcf0bc1f353d7a87fac249603ddddbe1ce2a5eae04"
_NARRATIVE_METRICS_SHA256 = "6d973eebffcacd70e4b786ccd3ee66b63feeb622db09bc67ac653d3fb44b4afd"
_MOMENTUM_RETURN = 0.24597743358179014
_MOMENTUM_SHARPE = 1.6030790628765745


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sessions(count: int = 30) -> tuple[date, ...]:
    return tuple(date(2025, 1, 2) + timedelta(days=offset) for offset in range(count))


def _event(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_id": "hn:1:AAPL",
        "ticker": "AAPL",
        "available_ts": "2025-01-02T12:00:00Z",
        "decision_session": "2025-01-03",
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


def _row_on(rows: list[dict[str, object]], session: str, ticker: str) -> dict[str, object]:
    found = [row for row in rows if row["session"] == session and row["ticker"] == ticker]
    assert len(found) == 1
    return found[0]


def test_modules_do_not_import_the_polarity_lexicon() -> None:
    attention = Path(attention_module.__file__ or "").read_text(encoding="utf-8")
    runner = Path(attention_run_module.__file__ or "").read_text(encoding="utf-8")
    for source in (attention, runner):
        assert "barebone_polarity" not in source
        assert "load_polarity_lexicon" not in source
        assert "score_text" not in source
    assert ATTENTION_LOOKBACK_SESSIONS == 21
    assert ATTENTION_SIGNAL_ID == "hn-attention-log1p-21s-v1"
    assert HYBRID_SIGNAL_ID == "hn-attention-momentum-intersection-v1"


def test_count_window_is_point_in_time_and_demeaned() -> None:
    sessions = _sessions()
    early = _event()
    future = _event(
        event_id="hn:2:AAPL",
        available_ts="2025-01-22T12:00:00Z",
        decision_session="2025-01-23",
    )
    base = attention_score_rows([early], sessions, ("AAPL", "MSFT"))
    with_future = attention_score_rows([early, future], sessions, ("AAPL", "MSFT"))
    assert _row_on(base, "2025-01-22", "AAPL") == _row_on(with_future, "2025-01-22", "AAPL")
    assert _row_on(base, "2025-01-22", "MSFT")["event_count"] == 0
    later = _row_on(with_future, "2025-01-23", "AAPL")
    assert later["event_count"] == 2
    dropped = _row_on(with_future, "2025-01-24", "AAPL")
    assert dropped["event_count"] == 1
    first = _row_on(with_future, "2025-01-22", "AAPL")
    assert first["event_count"] == 1
    assert first["signal_id"] == ATTENTION_SIGNAL_ID
    assert first["lookback_sessions"] == 21
    assert "text" not in first and "polarity" not in first
    raw = {"AAPL": math.log1p(1), "MSFT": math.log1p(0)}
    demeaned = demean_cross_section(raw)
    assert math.isclose(float(first["score"]), demeaned["AAPL"])
    assert math.isclose(float(_row_on(with_future, "2025-01-22", "MSFT")["score"]), demeaned["MSFT"])
    assert math.isclose(demeaned["AAPL"], -demeaned["MSFT"])
    odd = demean_cross_section({"AAPL": math.log1p(3), "MSFT": math.log1p(1), "NVDA": math.log1p(0)})
    assert math.isclose(odd["MSFT"], 0.0)
    assert demean_cross_section({"AAPL": 1.0}) == {}
    with pytest.raises(ValueError, match="locked at 21"):
        attention_score_rows([early], sessions, ("AAPL", "MSFT"), lookback=10)


def test_same_day_look_ahead_and_labels_are_refused() -> None:
    sessions = _sessions()
    with pytest.raises(ValueError, match="same-session"):
        attention_score_rows(
            [_event(available_ts="2025-01-03T12:00:00Z", decision_session="2025-01-03")],
            sessions,
            ("AAPL", "MSFT"),
        )
    with pytest.raises(ValueError, match="strictly after"):
        attention_score_rows([_event(decision_session="2025-01-06")], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="missing available_ts"):
        attention_score_rows([_event(available_ts="")], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="attention universe"):
        attention_score_rows([_event(ticker="ZZZ")], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="return label"):
        attention_score_rows([_event(polarity=0.4)], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="return label"):
        attention_score_rows([_event(total_return=0.2)], sessions, ("AAPL", "MSFT"))


def test_non_positive_skill_is_cash_and_future_prices_do_not_change_past_skill() -> None:
    config = _trading()
    raw = {"AAA": 1.0, "BBB": 0.0}
    assert attention_book(raw, None, config) == {}
    assert attention_book(raw, 0.0, config) == {}
    assert attention_book(raw, -0.2, config) == {}
    sized = attention_book(raw, 0.2, config)
    assert set(sized) == {"AAA"}
    assert sized["AAA"] <= 0.1

    dates = _sessions(16)
    closes = {("AAA", index): 100.0 - index for index in range(16)}
    closes.update({("BBB", index): 100.0 + index for index in range(16)})
    by_session = {day: {"AAA": 1.0, "BBB": 0.0} for day in dates}
    pairs = attention_oos_pairs(by_session, closes, dates)
    skill, count = skill_before(pairs, 8)
    assert count >= 8
    assert skill is not None and skill < 0.0
    assert skill_allows_book(skill) is False
    later = dict(closes)
    later[("AAA", 12)] = 1.0
    assert skill_before(attention_oos_pairs(by_session, later, dates), 8) == (skill, count)
    changed = dict(closes)
    changed[("AAA", 4)] = 50.0
    assert skill_before(attention_oos_pairs(by_session, changed, dates), 8) != (skill, count)
    rows = attention_rebalances(by_session, closes, dates, config)
    assert rows
    assert all(row["sized"] is False for row in rows)


def test_aligned_attention_can_size_and_hybrid_requires_both_gates() -> None:
    config = _trading()
    dates = _sessions(16)
    closes = {("AAA", index): 100.0 + index for index in range(16)}
    closes.update({("BBB", index): 100.0 - index for index in range(16)})
    by_session = {day: {"AAA": 1.0, "BBB": 0.0} for day in dates}
    rows = attention_rebalances(by_session, closes, dates, config)
    assert any(row["sized"] is True for row in rows)
    assert all(
        row["sized"] is False or (row["oos_skill"] is not None and float(row["oos_skill"]) > 0.0)
        for row in rows
    )

    momentum_scores = {"AAA": 1.0, "BBB": -1.0}
    attention_preferred = {"AAA": -1.0, "BBB": 1.0}
    both = hybrid_book(momentum_scores, True, True, config)
    assert set(both) == {"AAA"}
    assert "BBB" not in both
    assert both["AAA"] <= 0.1
    assert set(hybrid_book(attention_preferred, True, True, config)) == {"BBB"}
    assert hybrid_book(momentum_scores, True, False, config) == {}
    assert hybrid_book(momentum_scores, False, True, config) == {}
    cross_section = [
        {"ticker": "AAA", "score": 1.0},
        {"ticker": "BBB", "score": -1.0},
    ]
    hybrid_rows = hybrid_rebalances(
        [
            {"decision_session": "2025-01-02", "skill_pass": True, "oos_skill": 0.3},
            {"decision_session": "2025-01-16", "skill_pass": False, "oos_skill": -0.2},
        ],
        [
            {
                "decision_session": "2025-01-02",
                "skill_pass": True,
                "insufficient_history": False,
                "oos_skill": 0.4,
                "cross_section": cross_section,
            },
            {
                "decision_session": "2025-01-16",
                "skill_pass": True,
                "insufficient_history": False,
                "oos_skill": 0.4,
                "cross_section": cross_section,
            },
        ],
        config,
    )
    assert hybrid_rows[0]["sized"] is True
    assert hybrid_rows[0]["momentum_skill_pass"] is True
    assert hybrid_rows[0]["attention_skill_pass"] is True
    assert set(hybrid_rows[0]["book"]) == {"AAA"}
    assert hybrid_rows[1]["sized"] is False
    cash = hybrid_rebalances(
        [{"decision_session": "2025-01-02", "skill_pass": True, "oos_skill": 0.3}],
        [
            {
                "decision_session": "2025-01-02",
                "skill_pass": False,
                "insufficient_history": False,
                "oos_skill": -0.1,
                "cross_section": cross_section,
            }
        ],
        config,
    )
    assert cash[0]["sized"] is False

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
    sized_run = run_controlled_arm(
        config=trading,
        sessions=sessions,
        strategy_id="attention",
        candidates=[
            ArmInput(
                strategy_id="attention",
                decision_session=date(2025, 1, 2),
                ticker="AAA",
                structured_weight=0.1,
            )
        ],
        benchmark_marks=marks,
    )
    cash_run = run_controlled_arm(
        config=trading,
        sessions=sessions,
        strategy_id="hybrid",
        candidates=[
            ArmInput(
                strategy_id="hybrid",
                decision_session=date(2025, 1, 2),
                ticker="AAA",
                structured_weight=0.0,
            )
        ],
        benchmark_marks=marks,
    )
    assert sized_run.block_reason is None and sized_run.admitted_count == 1
    assert sized_run.lineage.proposals[0].reason == "attention count weight"
    assert cash_run.block_reason is None and cash_run.admitted_count == 0
    assert cash_run.lineage.proposals[0].reason == "hybrid skill gate cash"
    with pytest.raises(ValueError, match="unknown strategy arm"):
        run_controlled_arm(
            config=trading,
            sessions=sessions,
            strategy_id="other",
            candidates=[],
            benchmark_marks=marks,
        )


def test_attention_lock_refuses_changed_locks_or_a_claim(tmp_path: Path) -> None:
    source = json.loads((_root() / "configs" / "barebone-comparison-v1.json").read_text(encoding="utf-8"))
    path = tmp_path / "config.json"
    source["evidence"]["narrative_sha256"] = "a" * 64
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="changed narrative_sha256"):
        write_locked_attention_sha256(path, "b" * 64)
    source["evidence"]["narrative_sha256"] = LOCKED_NARRATIVE_SHA256
    source["evidence"]["tape_sha256"] = "a" * 64
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="changed tape_sha256"):
        write_locked_attention_sha256(path, "b" * 64)
    source["evidence"]["tape_sha256"] = LOCKED_TAPE_SHA256
    source["evidence"]["polarity_sha256"] = "d" * 64
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="changed polarity_sha256"):
        write_locked_attention_sha256(path, "b" * 64)
    source["evidence"]["polarity_sha256"] = LOCKED_POLARITY_SHA256
    source["comparable_performance_claim"] = True
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="comparable_performance_claim must be false"):
        write_locked_attention_sha256(path, "b" * 64)
    source["comparable_performance_claim"] = False
    path.write_text(json.dumps(source), encoding="utf-8")
    write_locked_attention_sha256(path, "c" * 64)
    locked = json.loads(path.read_text(encoding="utf-8"))
    assert locked["evidence"]["attention_sha256"] == "c" * 64
    assert locked["evidence"]["narrative_sha256"] == LOCKED_NARRATIVE_SHA256
    assert locked["evidence"]["tape_sha256"] == LOCKED_TAPE_SHA256
    assert locked["evidence"]["polarity_sha256"] == LOCKED_POLARITY_SHA256
    assert locked["comparable_performance_claim"] is False


def test_attention_file_is_gitignored_and_the_momentum_control_stays_put() -> None:
    root = _root()
    ignored = subprocess.run(["git", "check-ignore", "-q", ATTENTION_SCORES], cwd=root, check=False)
    events = subprocess.run(["git", "check-ignore", "-q", NARRATIVE_EVENTS], cwd=root, check=False)
    assert ignored.returncode == 0
    assert events.returncode == 0
    config = load_barebone_comparison_config()
    metrics = json.loads((root / "results" / "barebone_three_arm_metrics.json").read_text(encoding="utf-8"))
    assert config.evidence.narrative_sha256 == LOCKED_NARRATIVE_SHA256
    assert config.evidence.tape_sha256 == LOCKED_TAPE_SHA256
    assert config.evidence.polarity_sha256 == LOCKED_POLARITY_SHA256
    assert metrics["tape_sha256"] == LOCKED_TAPE_SHA256
    assert metrics["arms"]["pure_ml"]["statistics"]["total_return"] == _MOMENTUM_RETURN
    assert metrics["arms"]["pure_ml"]["statistics"]["sharpe_ratio"] == _MOMENTUM_SHARPE
    assert hashlib.sha256((root / "results" / "metrics.json").read_bytes()).hexdigest() == _LEGACY_METRICS_SHA256
    assert hashlib.sha256((root / "results" / "barebone_three_arm_metrics.json").read_bytes()).hexdigest() == (
        _THREE_ARM_METRICS_SHA256
    )
    assert hashlib.sha256((root / "results" / "barebone_three_arm_ledger.json").read_bytes()).hexdigest() == (
        _THREE_ARM_LEDGER_SHA256
    )
    assert hashlib.sha256((root / "results" / "barebone_narrative_arm_metrics.json").read_bytes()).hexdigest() == (
        _NARRATIVE_METRICS_SHA256
    )
    assert hashlib.sha256((root / NARRATIVE_LEXICON).read_bytes()).hexdigest() == FROZEN_LEXICON_SHA256
    spec = importlib.util.spec_from_file_location(
        "build_submission_archive", root / "scripts" / "build_submission_archive.py"
    )
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    scores = root / ATTENTION_SCORES
    created = not scores.exists()
    if created:
        scores.parent.mkdir(parents=True, exist_ok=True)
        scores.write_text("{}\n", encoding="utf-8")
    try:
        assert scores not in builder.collect_files()
    finally:
        if created:
            scores.unlink()
    attention = config.evidence.attention_sha256
    if attention is None:
        return
    provenance = json.loads(
        (root / "evidence" / "narrative" / "barebone_window_attention_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["attention_sha256"] == attention
    assert provenance["narrative_events_sha256"] == LOCKED_NARRATIVE_SHA256
    assert provenance["tape_sha256"] == LOCKED_TAPE_SHA256
    assert provenance["signal_id"] == ATTENTION_SIGNAL_ID
    assert provenance["lexicon_used_for_sizing"] is False
    assert provenance["comparable_performance_claim"] is False
    assert "text" not in provenance
    local_ready = (
        scores.is_file()
        and (root / NARRATIVE_EVENTS).is_file()
        and (root / "evidence" / "market" / "barebone_window_ohlcv.json").is_file()
    )
    if not local_ready:
        ledger = json.loads((root / ATTENTION_LEDGER_RELATIVE).read_text(encoding="utf-8"))
        hybrid = json.loads((root / HYBRID_LEDGER_RELATIVE).read_text(encoding="utf-8"))
        assert ledger["comparable_performance_claim"] is False
        assert hybrid["comparable_performance_claim"] is False
        assert ledger["lexicon_used_for_sizing"] is False
        assert hybrid["signal_id"] == HYBRID_SIGNAL_ID
        for row in ledger["oos_skill"]:
            if row["oos_skill"] is None or float(row["oos_skill"]) <= 0.0:
                assert row["sized"] is False
        for row in hybrid["oos_skill"]:
            if row["sized"]:
                assert row["momentum_skill_pass"] is True
                assert row["attention_skill_pass"] is True
        return
    assert hashlib.sha256(scores.read_bytes()).hexdigest() == attention
    assert hashlib.sha256((root / NARRATIVE_EVENTS).read_bytes()).hexdigest() == LOCKED_NARRATIVE_SHA256
    assert (root / ATTENTION_LEDGER_RELATIVE).read_text(encoding="utf-8") == json.dumps(
        barebone_attention_ledger(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    assert (root / ATTENTION_METRICS_RELATIVE).read_text(encoding="utf-8") == json.dumps(
        barebone_attention_metrics(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    assert (root / HYBRID_LEDGER_RELATIVE).read_text(encoding="utf-8") == json.dumps(
        barebone_hybrid_ledger(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    assert (root / HYBRID_METRICS_RELATIVE).read_text(encoding="utf-8") == json.dumps(
        barebone_hybrid_metrics(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
