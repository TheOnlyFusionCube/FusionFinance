"""Walk-forward filing scores for the controlled pure-ML arm.

Prices stay on the locked software tape. Scores come from
``alpha.filing_alpha.fit_fusion_model`` using only labels that end before
the decision. The proposal manifest is hash-bound to the AMD receipts.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.filing_alpha.filing_fusion import (
    add_filing_decay_features,
    fit_fusion_model,
    point_in_time_filing_join,
)
from demo.contracts import ExperimentConfig
from demo.controlled import canonical_hash, locked_weekday_sessions


_WARMUP_SESSIONS = 80
_HORIZON_SESSIONS = 10
_RIDGE_ALPHA = 10.0
_AMD_PATHS = (
    "evidence/amd/environment.json",
    "evidence/amd/hardware.json",
    "evidence/amd/training.json",
    "results/amd_compute.json",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _weekdays_before(day: date, count: int) -> list[date]:
    found: list[date] = []
    current = day
    while len(found) < count:
        current -= timedelta(days=1)
        if current.weekday() < 5:
            found.append(current)
    found.reverse()
    return found


def amd_evidence_sha256(root: Path | None = None) -> dict[str, str]:
    """Hash the checked-in AMD receipts and confirm ``amd_compute`` agrees."""

    base = _repo_root() if root is None else root
    records = {
        relative: hashlib.sha256((base / relative).read_bytes()).hexdigest()
        for relative in _AMD_PATHS
    }
    compute = json.loads((base / "results/amd_compute.json").read_text(encoding="utf-8"))
    receipts = compute.get("receipts")
    if not isinstance(receipts, dict):
        raise ValueError("amd_compute.json is missing receipt hashes")
    expected = {
        "environment": "evidence/amd/environment.json",
        "hardware": "evidence/amd/hardware.json",
        "training": "evidence/amd/training.json",
    }
    for key, relative in expected.items():
        recorded = receipts.get(key, {}).get("sha256")
        if recorded != records[relative]:
            raise ValueError(f"amd_compute.json hash does not match {relative}")
    if compute.get("workload") != "walk-forward quantitative model training":
        raise ValueError("amd_compute.json is not the walk-forward training receipt")
    return records


def walk_forward_filing_proposals(
    config: ExperimentConfig,
    decision_dates: tuple[date, ...] | list[date],
    *,
    root: Path | None = None,
) -> dict[str, object]:
    """Score each locked rebalance from an expanding pre-decision ridge fit."""

    if not decision_dates:
        raise ValueError("pure-ML proposals require at least one decision date")
    locked = locked_weekday_sessions(config)
    if any(day not in set(locked) for day in decision_dates):
        raise ValueError("pure-ML decisions must lie on the locked weekday tape")
    panel = _feature_panel(config, locked)
    proposals: list[dict[str, object]] = []
    for decision in decision_dates:
        proposals.append(_score_decision(config, panel, decision))
    amd = amd_evidence_sha256(root)
    payload = {
        "schema": "fusionfinance-pure-ml-proposal-manifest-v1",
        "model": "alpha.filing_alpha.fit_fusion_model",
        "ridge_alpha": _RIDGE_ALPHA,
        "horizon_sessions": _HORIZON_SESSIONS,
        "warmup_sessions": _WARMUP_SESSIONS,
        "amd_evidence_sha256": amd,
        "amd_compute_sha256": amd["results/amd_compute.json"],
        "proposals": proposals,
    }
    payload["proposal_manifest_hash"] = canonical_hash(payload)
    return payload


def _feature_panel(
    config: ExperimentConfig, locked: tuple[date, ...]
) -> pd.DataFrame:
    warmup = _weekdays_before(locked[0], _WARMUP_SESSIONS)
    calendar = [*warmup, *locked]
    index = {day: offset - len(warmup) for offset, day in enumerate(calendar)}
    market_rows: list[dict[str, object]] = []
    filing_rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(config.universe):
        for day in calendar:
            level = 100.0 + float(index[day])
            market_rows.append(
                {
                    "date": pd.Timestamp(day),
                    "ticker": ticker,
                    "open": level,
                    "high": level,
                    "low": level,
                    "close": level,
                    "volume": 1_000_000.0,
                }
            )
        for session_index, day in enumerate(calendar):
            if (session_index + ticker_index * 3) % 11 != 0:
                continue
            signal = ((ticker_index % 7) - 3) / 3.0 + 0.35 * np.sin(session_index / 5.0)
            filing_rows.append(
                {
                    "ticker": ticker,
                    "accepted_at": datetime(
                        day.year,
                        day.month,
                        day.day,
                        15,
                        0,
                        tzinfo=timezone.utc,
                    ),
                    "filing_signal": float(signal),
                }
            )
    market = pd.DataFrame(market_rows)
    market["trail_return"] = market.groupby("ticker", sort=False)["close"].pct_change(
        _HORIZON_SESSIONS
    )
    joined = point_in_time_filing_join(
        market,
        pd.DataFrame(filing_rows),
        ["filing_signal"],
    )
    decayed = add_filing_decay_features(joined, ["filing_signal"])
    decayed["session_index"] = decayed["date"].map(
        lambda value: index[pd.Timestamp(value).date()]
    )
    return decayed


def _score_decision(
    config: ExperimentConfig, panel: pd.DataFrame, decision: date
) -> dict[str, object]:
    decision_ts = pd.Timestamp(decision)
    history = panel.loc[panel["date"] < decision_ts].copy()
    history["label_index"] = history["session_index"] + _HORIZON_SESSIONS
    # A label is usable only after its horizon has finished, and only before
    # this decision. That set grows on later rebalances inside the window.
    decision_index = int(
        panel.loc[panel["date"] == decision_ts, "session_index"].iloc[0]
    )
    history = history.loc[history["label_index"] < decision_index].copy()
    if history.empty:
        raise ValueError(f"no pre-decision training rows for {decision.isoformat()}")
    start_close = history["close"].to_numpy(dtype=float)
    label_close = 100.0 + history["label_index"].to_numpy(dtype=float)
    history["target"] = label_close / start_close - 1.0
    fitted = fit_fusion_model(
        history,
        "target",
        ["trail_return"],
        ["filing_signal_decayed"],
        ridge_alpha=_RIDGE_ALPHA,
    )
    live = (
        panel.loc[panel["date"] == decision_ts]
        .set_index("ticker")
        .loc[list(config.universe)]
        .reset_index()
    )
    scores = fitted.predict(live)
    best = int(np.argmax(scores))
    score = float(scores[best])
    cap = float(config.max_position_weight)
    weight = float(np.clip(score, -cap, cap))
    coefficients = {
        column: float(value)
        for column, value in zip(
            ("trail_return", "filing_signal_decayed"),
            fitted.model.coef_,
            strict=True,
        )
    }
    return {
        "decision_session": decision.isoformat(),
        "ticker": str(live.iloc[best]["ticker"]),
        "model_score": score,
        "structured_weight": weight,
        "train_rows": int(len(history)),
        "intercept": float(fitted.model.intercept_),
        "coefficients": coefficients,
        "cross_section": [
            {"ticker": str(ticker), "score": float(value)}
            for ticker, value in zip(live["ticker"], scores, strict=True)
        ],
    }
