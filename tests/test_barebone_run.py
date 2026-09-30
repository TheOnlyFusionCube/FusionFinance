"""Barebone-window three-arm baseline. Claim stays false."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import date, timedelta

import numpy as np
from pathlib import Path

import pandas as pd
import pytest

from demo.barebone_comparison import (
    BAREBONE_OHLCV,
    BAREBONE_PROVENANCE,
    load_barebone_comparison_config,
    refuse_fair_race_as_barebone_comparison,
)
from demo.barebone_run import (
    LEDGER_RELATIVE,
    METRICS_RELATIVE,
    barebone_three_arm_ledger,
    barebone_three_arm_metrics,
    load_barebone_market,
    parse_barebone_ohlcv,
)
from demo.barebone_tape import required_tickers
from demo.controlled import load_locked_config
from demo.pure_ml import (
    MOMENTUM_LOOKBACK_SESSIONS,
    _score_momentum,
    walk_forward_filing_proposals,
)

_LOCKED_TAPE_SHA256 = (
    "3bfacc31d366fd03c797c723686a5f528bd43b06e09284b8caca85709853caee"
)
_LEGACY_METRICS_SHA256 = (
    "e7e5055ce9b4409d7941a71929f66261416b8d3c3f62423190edafd5bf5b1411"
)
_FAIR_RACE_TAPE_HASH = (
    "3a2a378ce29b387b84f5345a7953028d478507f204b9f7be6eee609e0dd20c05"
)
_PERFORMANCE_KEYS = frozenset(
    {
        "sharpe_ratio",
        "sortino_ratio",
        "total_return",
        "annualized_return",
        "annualized_alpha",
        "information_ratio",
        "calmar_ratio",
        "max_drawdown",
        "wealth_relative_excess_return",
    }
)
_DATES = (date(2025, 1, 2), date(2026, 1, 12))


def _json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _mapping_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(value)
        for item in value.values():
            found.update(_mapping_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_mapping_keys(item))
    return found


def _bar(session: date, ticker: str, *, close: float = 100.0) -> dict[str, object]:
    return {
        "date": session.isoformat(),
        "ticker": ticker,
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "adjclose": close,
        "volume": 1000.0,
    }


def _payload_for(sessions: tuple[date, ...]) -> dict[str, object]:
    config = load_barebone_comparison_config()
    tickers = required_tickers(config)
    bars = [
        _bar(session, ticker, close=100.0 + offset)
        for offset, session in enumerate(sessions)
        for ticker in tickers
    ]
    return {
        "schema": "fusionfinance-evidence-ohlcv-v1",
        "adjustment": "raw OHLC plus Yahoo Finance Adj Close via yfinance; no second adjustment is applied",
        "bars": bars,
    }


def _install_repo_receipts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "evidence/amd/environment.json",
        "evidence/amd/hardware.json",
        "evidence/amd/training.json",
        "results/amd_compute.json",
        "results/fusion_policy_calibration.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / relative, destination)


def _write_locked_tree(tmp_path: Path, payload: dict[str, object]) -> str:
    _install_repo_receipts(tmp_path)
    raw = _json_text(payload).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    ohlcv = tmp_path / BAREBONE_OHLCV
    ohlcv.parent.mkdir(parents=True, exist_ok=True)
    ohlcv.write_bytes(raw)
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "barebone-comparison-v1.json").read_text(
            encoding="utf-8"
        )
    )
    config["evidence"]["tape_sha256"] = digest
    config_path = tmp_path / "configs" / "barebone-comparison-v1.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    tickers = required_tickers(load_barebone_comparison_config())
    provenance = {
        "schema": "fusionfinance-barebone-tape-provenance-v1",
        "experiment_id": "barebone-comparison-v1",
        "provider": "Yahoo Finance via yfinance",
        "fetched_at": "2026-09-30T04:12:52Z",
        "tickers": list(tickers),
        "window": ["2025-01-02", "2026-01-12"],
        "ohlcv": BAREBONE_OHLCV,
        "byte_sha256": digest,
        "adjustment": payload["adjustment"],
        "bar_count": len(payload["bars"]),
        "session_count": len({row["date"] for row in payload["bars"]}),
        "license_note": "not redistributed; local bind only",
        "disclaimer": (
            "not redistributed; local bind only. Yahoo Finance data is not "
            "for trading purposes and is not redistributed."
        ),
        "comparable_performance_claim": False,
    }
    provenance_path = tmp_path / BAREBONE_PROVENANCE
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    return digest


def test_walk_forward_stays_in_cash_when_the_tape_has_no_label() -> None:
    config = load_locked_config()
    rows = []
    for index, day in enumerate(_DATES):
        for ticker in config.universe:
            rows.append(
                {
                    "date": pd.Timestamp(day),
                    "ticker": ticker,
                    "session_index": index,
                    "close": 100.0,
                    "trail_return": 0.0,
                    "filing_signal_decayed": 0.0,
                }
            )
    panel = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="no pre-decision training rows"):
        walk_forward_filing_proposals(
            config,
            (_DATES[0],),
            panel=panel,
            tape_dates=_DATES,
        )
    manifest = walk_forward_filing_proposals(
        config,
        (_DATES[0],),
        panel=panel,
        tape_dates=_DATES,
        cash_when_untrained=True,
    )
    proposal = manifest["proposals"][0]
    assert proposal["untrained"] is True
    assert proposal["skill_pass"] is False
    assert proposal["targets"] == []
    assert proposal["fusion_targets"] == []


def _dated_panel(config, closes: list[list[float]]) -> pd.DataFrame:
    origin = date(2024, 1, 2)
    rows = []
    for index, cross_section in enumerate(closes):
        day = origin + timedelta(days=index)
        for ticker, close in zip(config.universe, cross_section, strict=True):
            rows.append(
                {
                    "date": pd.Timestamp(day),
                    "ticker": ticker,
                    "session_index": index,
                    "close": float(close),
                    "trail_return": 0.0,
                    "filing_signal_decayed": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_momentum_uses_only_prices_through_the_decision_and_skips_thin_history() -> None:
    config = load_locked_config()
    sessions = 80
    closes = [
        [100.0 * (1.0 + 0.001 * (offset + 1) * (index + 1)) for offset in range(len(config.universe))]
        for index in range(sessions)
    ]
    panel = _dated_panel(config, closes)
    decision = date(2024, 1, 2) + timedelta(days=70)
    original = _score_momentum(config, panel, decision)
    assert original["insufficient_history"] is False
    assert len(original["cross_section"]) == len(config.universe)
    shifted = closes.copy()
    shifted[71] = [price * 3.0 for price in shifted[71]]
    changed_future = _score_momentum(config, _dated_panel(config, shifted), decision)
    assert changed_future["cross_section"] == original["cross_section"]
    shifted_now = [row[:] for row in closes]
    shifted_now[70] = [price * 1.5 for price in shifted_now[70]]
    changed_now = _score_momentum(config, _dated_panel(config, shifted_now), decision)
    assert changed_now["cross_section"] != original["cross_section"]
    early = date(2024, 1, 2) + timedelta(days=10)
    thin = _score_momentum(config, panel, early)
    assert thin["insufficient_history"] is True
    assert thin["cross_section"] == []
    assert thin["lookback_sessions"] == MOMENTUM_LOOKBACK_SESSIONS


def test_momentum_skill_gate_keeps_a_positive_score_in_cash() -> None:
    config = load_locked_config()
    names = len(config.universe)
    sessions = 80
    closes = [[0.0 for _name in range(names)] for _session in range(sessions)]
    for index in range(MOMENTUM_LOOKBACK_SESSIONS + 1):
        for offset in range(names):
            closes[index][offset] = 100.0 * np.exp(0.01 * (offset - 8) * (index + 1) / 30.0)
    for index in range(MOMENTUM_LOOKBACK_SESSIONS + 1, sessions):
        for offset in range(names):
            momentum = np.log(
                closes[index - 1][offset] / closes[index - 1 - MOMENTUM_LOOKBACK_SESSIONS][offset]
            )
            closes[index][offset] = closes[index - 1][offset] * float(np.exp(-0.5 * momentum))
    panel = _dated_panel(config, closes)
    early = date(2024, 1, 2) + timedelta(days=10)
    live = date(2024, 1, 2) + timedelta(days=70)
    manifest = walk_forward_filing_proposals(
        config,
        (early, live),
        panel=panel,
        tape_dates=tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(sessions)),
        scorebook="momentum",
    )
    thin, scored = manifest["proposals"]
    assert thin["insufficient_history"] is True
    assert thin["skill_pass"] is False
    assert thin["targets"] == []
    assert scored["skill_pass"] is False
    assert scored["oos_skill"] is not None and scored["oos_skill"] <= 0.0
    assert scored["targets"] == []
    assert any(float(item["score"]) > 0.0 for item in scored["cross_section"])
    assert scored["fusion_targets"]


def test_barebone_parser_refuses_gaps_weekends_and_extra_names() -> None:
    config = load_barebone_comparison_config()
    weekend = _payload_for((date(2025, 1, 4),))
    with pytest.raises(ValueError, match="weekend"):
        parse_barebone_ohlcv(weekend, config)
    outside = _payload_for((date(2024, 12, 31), date(2026, 1, 12)))
    with pytest.raises(ValueError, match="outside the locked window"):
        parse_barebone_ohlcv(outside, config)
    partial = _payload_for(_DATES)
    partial["bars"] = [
        row for row in partial["bars"] if not (row["date"] == "2026-01-12" and row["ticker"] == "QQQ")
    ]
    with pytest.raises(ValueError, match="missing ticker"):
        parse_barebone_ohlcv(partial, config)
    extra = _payload_for(_DATES)
    extra["bars"].append(_bar(_DATES[0], "ZZZ"))
    with pytest.raises(ValueError, match="unexpected ticker"):
        parse_barebone_ohlcv(extra, config)


def test_short_tape_runs_the_three_arms_without_a_claim(tmp_path: Path) -> None:
    digest = _write_locked_tree(tmp_path, _payload_for(_DATES))
    ledger = barebone_three_arm_ledger(tmp_path)
    metrics = barebone_three_arm_metrics(tmp_path)

    assert ledger["tape_sha256"] == digest
    assert metrics["tape_sha256"] == digest
    assert ledger["comparable_performance_claim"] is False
    assert metrics["comparable_performance_claim"] is False
    assert ledger["experiment_id"] == "barebone-comparison-v1"
    assert ledger["window"] == ["2025-01-02", "2026-01-12"]
    assert ledger["max_position_weight"] == 0.1
    assert ledger["max_gross_leverage"] == 1.0
    assert _PERFORMANCE_KEYS.isdisjoint(_mapping_keys(ledger))
    refuse_fair_race_as_barebone_comparison(ledger)
    refuse_fair_race_as_barebone_comparison(metrics)
    assert ledger["arms"]["pure_ml"]["admitted_count"] == 0
    assert ledger["arms"]["pure_ml"]["oos_skill"][0]["insufficient_history"] is True
    binding = ledger["arms"]["pure_ml"]["model_binding"]
    assert binding["scorebook"] == "momentum"
    assert binding["lookback_sessions"] == 63
    assert binding["price"] == "adjclose"
    assert ledger["arms"]["pure_llm"]["admitted_count"] == 1
    assert ledger["arms"]["pure_llm"]["total_turnover"] > 0.0
    assert metrics["secondary_benchmark"]["ticker"] == "QQQ"
    assert "fund result" in metrics["secondary_benchmark"]["note"]
    for name in ("pure_ml", "pure_llm", "fusion"):
        assert math.isclose(
            metrics["arms"][name]["statistics"]["total_turnover"],
            ledger["arms"][name]["total_turnover"],
            rel_tol=1e-12,
            abs_tol=1e-9,
        )


def test_barebone_market_fails_closed_without_its_tape(tmp_path: Path) -> None:
    _install_repo_receipts(tmp_path)
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "barebone-comparison-v1.json").read_text(
            encoding="utf-8"
        )
    )
    destination = tmp_path / "configs" / "barebone-comparison-v1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="evidence is missing"):
        load_barebone_market(tmp_path)


def test_barebone_market_refuses_the_fair_race_extract(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fair = (root / "evidence" / "market" / "locked_ohlcv.json").read_bytes()
    digest = hashlib.sha256(fair).hexdigest()
    payload = json.loads(fair)
    _write_locked_tree(tmp_path, payload)
    ohlcv = tmp_path / BAREBONE_OHLCV
    ohlcv.write_bytes(fair)
    config_path = tmp_path / "configs" / "barebone-comparison-v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["evidence"]["tape_sha256"] = digest
    config_path.write_text(json.dumps(config), encoding="utf-8")
    provenance_path = tmp_path / BAREBONE_PROVENANCE
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["byte_sha256"] = digest
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="fair-race OHLCV"):
        load_barebone_market(tmp_path)


def test_checked_in_barebone_three_arm_matches_the_locked_yahoo_tape() -> None:
    root = Path(__file__).resolve().parents[1]
    ledger_path = root / LEDGER_RELATIVE
    metrics_path = root / METRICS_RELATIVE
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = load_barebone_comparison_config()

    assert ledger["tape_sha256"] == _LOCKED_TAPE_SHA256
    assert metrics["tape_sha256"] == _LOCKED_TAPE_SHA256
    assert config.evidence.tape_sha256 == _LOCKED_TAPE_SHA256
    assert ledger["comparable_performance_claim"] is False
    assert metrics["comparable_performance_claim"] is False
    assert ledger["experiment_id"] == "barebone-comparison-v1"
    assert ledger["window"] == ["2025-01-02", "2026-01-12"]
    assert ledger["max_position_weight"] == 0.1
    assert ledger["max_gross_leverage"] == 1.0
    assert ledger["session_count"] == 257
    assert _PERFORMANCE_KEYS.isdisjoint(_mapping_keys(ledger))
    refuse_fair_race_as_barebone_comparison(ledger)
    refuse_fair_race_as_barebone_comparison(metrics)
    assert _FAIR_RACE_TAPE_HASH not in ledger_path.read_text(encoding="utf-8")
    assert "locked_ohlcv.json" not in ledger_path.read_text(encoding="utf-8")
    for name in ("pure_ml", "pure_llm", "fusion"):
        arm = ledger["arms"][name]
        statistics = metrics["arms"][name]["statistics"]
        assert arm["comparable_performance_claim"] is False
        assert arm["post_cost_within_limit"] is True
        assert arm["tape_hash"] != _FAIR_RACE_TAPE_HASH
        assert metrics["arms"][name]["tape_sha256"] == _LOCKED_TAPE_SHA256
        assert statistics["trade_count"] == arm["trade_count"]
        assert math.isclose(statistics["total_turnover"], arm["total_turnover"])
        assert math.isclose(statistics["transaction_costs"], arm["transaction_costs"])
        values = metrics["arms"][name]["portfolio_values"]
        assert math.isclose(statistics["total_return"], values[-1] / values[0] - 1.0)
    assert metrics["secondary_benchmark"]["ticker"] == "QQQ"
    skill = ledger["arms"]["pure_ml"]["oos_skill"]
    binding = ledger["arms"]["pure_ml"]["model_binding"]
    assert binding["scorebook"] == "momentum"
    assert binding["lookback_sessions"] == 63
    assert binding["transform"] == "log_return_minus_cross_sectional_median"
    assert ledger["arms"]["pure_ml"]["config_hash"] == (
        "526d6eabc37eab8650876991e3822987eb857621c36d84069d2888c1b6c1b9b4"
    )
    assert any(row.get("insufficient_history") is True for row in skill)
    for row in skill:
        if row.get("insufficient_history") is True or (
            row["oos_skill"] is not None and float(row["oos_skill"]) <= 0.0
        ):
            assert row["sized"] is False
    sized_when_failed = [
        row
        for row in ledger["arms"]["pure_ml"]["decisions"]
        if row.get("oos_skill") is not None and float(row["oos_skill"]) <= 0.0
    ]
    assert sized_when_failed == []
    legacy = hashlib.sha256((root / "results" / "metrics.json").read_bytes()).hexdigest()
    assert legacy == _LEGACY_METRICS_SHA256
    ohlcv = root / BAREBONE_OHLCV
    if ohlcv.is_file():
        assert hashlib.sha256(ohlcv.read_bytes()).hexdigest() == _LOCKED_TAPE_SHA256
        assert ledger_path.read_text(encoding="utf-8") == _json_text(barebone_three_arm_ledger())
        assert metrics_path.read_text(encoding="utf-8") == _json_text(
            barebone_three_arm_metrics()
        )
