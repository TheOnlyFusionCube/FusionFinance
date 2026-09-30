"""Claim boundary for the barebone-comparison-v1 scaffold."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from demo.barebone_comparison import (
    BAREBONE_OHLCV,
    FAIR_RACE_EXPERIMENT_ID,
    FAIR_RACE_TAPE_HASH,
    load_barebone_comparison_config,
    refuse_barebone_performance_claim,
    refuse_fair_race_as_barebone_comparison,
    require_barebone_evidence,
    validate_barebone_payload,
)
from demo.controlled import load_locked_config

_LEGACY_METRICS_SHA256 = "e7e5055ce9b4409d7941a71929f66261416b8d3c3f62423190edafd5bf5b1411"


def _payload() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs" / "barebone-comparison-v1.json").read_text())


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "barebone-comparison-v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_barebone_comparison_v1_config_locks_the_window_risk_and_claim() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_barebone_comparison_config()
    trading = config.as_experiment_config()
    fair_race = load_locked_config()

    assert config.experiment.experiment_id == "barebone-comparison-v1"
    assert config.comparable_performance_claim is False
    assert trading.start_date.isoformat() == "2025-01-02"
    assert trading.end_date.isoformat() == "2026-01-12"
    assert trading.max_position_weight == 0.1
    assert trading.max_gross_leverage == 1.0
    assert trading.benchmark_ticker == "SPY"
    assert config.scorebook == "momentum"
    assert config.momentum_lookback_sessions == 63
    assert config.secondary_benchmark_ticker == "QQQ"
    assert config.secondary_benchmark_ticker not in trading.universe
    assert trading.benchmark_ticker not in trading.universe
    assert trading.universe == fair_race.universe
    assert trading.max_position_weight == fair_race.max_position_weight
    assert trading.max_gross_leverage == fair_race.max_gross_leverage
    assert config.evidence.ohlcv == BAREBONE_OHLCV
    digest = config.evidence.tape_sha256
    provenance = json.loads((root / "evidence/market/barebone_window_provenance.json").read_text())
    assert isinstance(digest, str) and len(digest) == 64
    assert digest != "3a2a378ce29b387b84f5345a7953028d478507f204b9f7be6eee609e0dd20c05"
    assert provenance["byte_sha256"] == digest
    assert provenance["provider"] == "Yahoo Finance via yfinance"
    assert "tiingo" not in provenance["provider"].casefold()
    assert "polygon" not in provenance["provider"].casefold()
    assert "not for trading" in provenance["disclaimer"].casefold()
    assert "not redistributed" in provenance["disclaimer"].casefold()
    assert provenance["comparable_performance_claim"] is False
    assert {"open", "high", "low", "close", "adjclose", "volume", "bars"}.isdisjoint(provenance)
    ohlcv = root / BAREBONE_OHLCV
    if ohlcv.is_file():
        assert hashlib.sha256(ohlcv.read_bytes()).hexdigest() == digest
    assert not (root / "results" / "barebone_comparison_ledger.json").exists()
    assert (
        hashlib.sha256((root / "results" / "metrics.json").read_bytes()).hexdigest()
        == _LEGACY_METRICS_SHA256
    )
    with pytest.raises(ValueError, match="keeps comparable_performance_claim false"):
        refuse_barebone_performance_claim(config, True)


def test_barebone_comparison_v1_refuses_a_true_claim_without_a_locked_tape() -> None:
    payload = _payload()
    payload["evidence"]["tape_sha256"] = None
    payload["comparable_performance_claim"] = True

    with pytest.raises(ValueError, match="comparable_performance_claim must be false"):
        validate_barebone_payload(payload)
    payload["comparable_performance_claim"] = False
    unlocked = validate_barebone_payload(payload)
    with pytest.raises(ValueError, match="without a locked tape_sha256"):
        refuse_barebone_performance_claim(unlocked, True)


def test_barebone_comparison_v1_refuses_a_true_claim_after_a_hash_is_locked(
    tmp_path: Path,
) -> None:
    payload = _payload()
    body = b"locked-placeholder\n"
    digest = hashlib.sha256(body).hexdigest()
    payload["evidence"]["tape_sha256"] = digest
    evidence = tmp_path / BAREBONE_OHLCV
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(body)
    config = load_barebone_comparison_config(_write_config(tmp_path, payload))

    assert require_barebone_evidence(config, root=tmp_path) == evidence
    with pytest.raises(ValueError, match="keeps comparable_performance_claim false"):
        refuse_barebone_performance_claim(config, True)


def test_barebone_comparison_v1_refuses_the_fair_race_tape_hash() -> None:
    payload = _payload()
    payload["evidence"]["tape_sha256"] = FAIR_RACE_TAPE_HASH
    payload["experiment_id"] = FAIR_RACE_EXPERIMENT_ID

    with pytest.raises(ValueError, match="barebone-comparison-v1"):
        validate_barebone_payload(payload)
    payload["experiment_id"] = "barebone-comparison-v1"
    with pytest.raises(ValidationError, match="fair-race tape_hash"):
        validate_barebone_payload(payload)


def test_barebone_comparison_v1_refuses_the_fair_race_ohlcv_path() -> None:
    payload = _payload()
    payload["evidence"]["ohlcv"] = "evidence/market/locked_ohlcv.json"

    with pytest.raises(ValidationError, match="fair-race OHLCV"):
        validate_barebone_payload(payload)


def test_barebone_comparison_v1_evidence_fails_closed_when_missing(tmp_path: Path) -> None:
    config = load_barebone_comparison_config()

    with pytest.raises(FileNotFoundError, match="barebone-comparison evidence is missing"):
        require_barebone_evidence(config, root=tmp_path)


def test_barebone_comparison_v1_evidence_fails_closed_when_the_hash_is_unlocked(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / BAREBONE_OHLCV
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b"present-but-unlocked\n")
    payload = _payload()
    payload["evidence"]["tape_sha256"] = None
    config = validate_barebone_payload(payload)

    with pytest.raises(ValueError, match="unlocked evidence tape"):
        require_barebone_evidence(config, root=tmp_path)


def test_barebone_comparison_v1_evidence_fails_closed_on_hash_mismatch(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["evidence"]["tape_sha256"] = "a" * 64
    evidence = tmp_path / BAREBONE_OHLCV
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b"different-bytes\n")
    config = load_barebone_comparison_config(_write_config(tmp_path, payload))

    with pytest.raises(ValueError, match="does not match the locked tape_sha256"):
        require_barebone_evidence(config, root=tmp_path)


def test_barebone_comparison_v1_refuses_reusing_the_fair_race_ohlcv_bytes(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    body = (root / "evidence" / "market" / "locked_ohlcv.json").read_bytes()
    payload = _payload()
    payload["evidence"]["tape_sha256"] = hashlib.sha256(body).hexdigest()
    evidence = tmp_path / BAREBONE_OHLCV
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(body)
    config = load_barebone_comparison_config(_write_config(tmp_path, payload))

    with pytest.raises(ValueError, match="fair-race OHLCV"):
        require_barebone_evidence(config, root=tmp_path)


def test_barebone_comparison_v1_refuses_fair_race_ledgers_as_comparable() -> None:
    root = Path(__file__).resolve().parents[1]
    ledger = json.loads(
        (root / "results" / "controlled_three_arm_ledger.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (root / "results" / "controlled_three_arm_metrics.json").read_text(encoding="utf-8")
    )

    with pytest.raises(ValueError, match="not comparable for barebone-comparison-v1"):
        refuse_fair_race_as_barebone_comparison(ledger)
    with pytest.raises(ValueError, match="not comparable for barebone-comparison-v1"):
        refuse_fair_race_as_barebone_comparison(metrics)
    with pytest.raises(ValueError, match="not comparable for barebone-comparison-v1"):
        refuse_fair_race_as_barebone_comparison(
            {"experiment_id": FAIR_RACE_EXPERIMENT_ID, "comparable_performance_claim": False}
        )
    with pytest.raises(ValueError, match="not comparable for barebone-comparison-v1"):
        refuse_fair_race_as_barebone_comparison({"tape_hash": FAIR_RACE_TAPE_HASH})
    with pytest.raises(ValueError, match="comparable_performance_claim must be false"):
        refuse_fair_race_as_barebone_comparison(
            {"experiment_id": "barebone-comparison-v1", "comparable_performance_claim": True}
        )


def test_barebone_comparison_v1_refuses_a_leverage_unlock_and_a_shifted_window() -> None:
    loosened = _payload()
    loosened["max_gross_leverage"] = 2.0
    shifted = _payload()
    shifted["end_date"] = "2026-07-09"
    borrowed = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "fusionfinance-demo.json").read_text()
    )

    with pytest.raises(ValueError, match="max_gross_leverage 1.0"):
        validate_barebone_payload(loosened)
    with pytest.raises(ValueError, match="2025-01-02 through 2026-01-12"):
        validate_barebone_payload(shifted)
    with pytest.raises(ValueError, match="experiment_id must be barebone-comparison-v1"):
        validate_barebone_payload(borrowed)
    unlocked_lookback = _payload()
    unlocked_lookback["momentum_lookback_sessions"] = 21
    with pytest.raises(ValueError, match="locked at 63"):
        validate_barebone_payload(unlocked_lookback)
    ridge = _payload()
    ridge["scorebook"] = "ridge"
    ridge["momentum_lookback_sessions"] = None
    assert validate_barebone_payload(ridge).scorebook == "ridge"
    missing_scorebook = _payload()
    del missing_scorebook["scorebook"]
    with pytest.raises(ValueError, match="scorebook is required"):
        validate_barebone_payload(missing_scorebook)


def test_barebone_comparison_v1_rejects_malformed_evidence_and_documents() -> None:
    escaped = _payload()
    escaped["evidence"]["ohlcv"] = "evidence/../market/barebone_window_ohlcv.json"
    unknown = _payload()
    unknown["alpha"] = 1
    missing_evidence = _payload()
    missing_evidence["evidence"] = None
    bad_hash = _payload()
    bad_hash["evidence"]["tape_sha256"] = "A" * 64
    cheaper = _payload()
    cheaper["transaction_cost_bps"] = 0.0
    dated = _payload()
    dated["start_date"] = date(2025, 1, 2)
    dated["end_date"] = date(2026, 1, 12)

    with pytest.raises(ValidationError, match="inside the repository"):
        validate_barebone_payload(escaped)
    with pytest.raises(ValueError, match="unknown barebone comparison fields"):
        validate_barebone_payload(unknown)
    with pytest.raises(ValueError, match="evidence path is required"):
        validate_barebone_payload(missing_evidence)
    with pytest.raises(ValidationError, match="lowercase sha256"):
        validate_barebone_payload(bad_hash)
    with pytest.raises(ValidationError, match="controlled-path locks"):
        validate_barebone_payload(cheaper)
    assert validate_barebone_payload(dated).experiment.start_date == date(2025, 1, 2)
    with pytest.raises(ValueError, match="must be a mapping"):
        refuse_fair_race_as_barebone_comparison(["not-a-document"])


def test_barebone_comparison_v1_secondary_benchmark_stays_optional_and_outside_the_book() -> None:
    missing = _payload()
    missing["secondary_benchmark_ticker"] = None
    inside = _payload()
    inside["secondary_benchmark_ticker"] = "AAPL"
    same = _payload()
    same["secondary_benchmark_ticker"] = "SPY"

    assert validate_barebone_payload(missing).secondary_benchmark_ticker is None
    with pytest.raises(ValidationError, match="outside the tradable book"):
        validate_barebone_payload(inside)
    with pytest.raises(ValidationError, match="outside the tradable book"):
        validate_barebone_payload(same)
