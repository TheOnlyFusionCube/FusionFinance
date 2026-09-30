from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from demo.controlled import load_locked_config
from demo.market_tape import evidence_price_sessions


def _copy_evidence(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "evidence", tmp_path / "evidence")
    return tmp_path


def test_evidence_tape_covers_the_sealed_calendar_without_holiday_fills() -> None:
    config = load_locked_config()
    sessions = evidence_price_sessions(config)
    dates = [session.session.isoformat() for session in sessions]

    assert dates[0] == "2026-02-02"
    assert dates[-1] == "2026-07-09"
    assert len(dates) == 109
    assert "2026-02-16" not in dates
    assert {bar.ticker for bar in sessions[0].bars} == set(config.universe) | {
        config.benchmark_ticker
    }


def test_evidence_tape_rejects_a_missing_ohlcv_file(tmp_path: Path) -> None:
    root = _copy_evidence(tmp_path)
    (root / "evidence/market/locked_ohlcv.json").unlink()

    with pytest.raises(ValueError, match="required market evidence is missing"):
        evidence_price_sessions(load_locked_config(), root=root)


def test_evidence_tape_rejects_a_missing_bar(tmp_path: Path) -> None:
    root = _copy_evidence(tmp_path)
    path = root / "evidence/market/locked_ohlcv.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["bars"] = [
        bar
        for bar in payload["bars"]
        if not (bar["date"] == "2026-02-02" and bar["ticker"] == "AAPL")
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing ticker"):
        evidence_price_sessions(load_locked_config(), root=root)


def test_evidence_tape_rejects_a_spy_return_that_is_not_in_the_replay(
    tmp_path: Path,
) -> None:
    root = _copy_evidence(tmp_path)
    path = root / "evidence/replay/v1_source.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["arms"]["benchmark"]["daily_returns"][1] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not reproduce the sealed benchmark"):
        evidence_price_sessions(load_locked_config(), root=root)
