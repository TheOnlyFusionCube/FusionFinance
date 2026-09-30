"""Walk-forward filing scores for the controlled pure-ML arm.

Prices come from the evidence tape. Scores come from
``alpha.filing_alpha.fit_fusion_model`` using only labels whose forward close
is already in that tape and ends before the decision. The proposal manifest
is hash-bound to the AMD receipts.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.filing_alpha.filing_fusion import (
    add_filing_decay_features,
    fit_fusion_model,
    point_in_time_filing_join,
)
from demo.contracts import ExperimentConfig
from demo.controlled import canonical_hash, estimated_post_cost_gross_leverage
from demo.market_tape import adjusted_market_frame, evidence_price_sessions


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
    locked = tuple(
        session.session for session in evidence_price_sessions(config, root=root)
    )
    if any(day not in set(locked) for day in decision_dates):
        raise ValueError("pure-ML decisions must lie on the evidence tape")
    if len(set(decision_dates)) != len(list(decision_dates)):
        raise ValueError("pure-ML decisions must be unique")
    if list(decision_dates) != sorted(decision_dates):
        raise ValueError("pure-ML decisions must be chronological")
    panel = _feature_panel(config, root=root)
    proposals: list[dict[str, object]] = []
    previous: dict[str, float] = {}
    for decision in decision_dates:
        scored = _score_decision(config, panel, decision)
        scores = {
            str(item["ticker"]): float(item["score"])
            for item in scored["cross_section"]
        }
        book = allocate_positive_score_book(scores, config, previous)
        targets = [
            {
                "ticker": ticker,
                "model_score": scores[ticker],
                "structured_weight": weight,
            }
            for ticker, weight in sorted(book.items())
        ]
        positive = sum(score > 0.0 for score in scores.values())
        if positive > 1 and len(targets) < 2:
            raise ValueError(
                "positive scores support multiple names but the book has one"
            )
        scored["targets"] = targets
        scored["gross_exposure"] = float(sum(book.values()))
        proposals.append(scored)
        previous = dict(book)
    amd = amd_evidence_sha256(root)
    payload = {
        "schema": "fusionfinance-pure-ml-proposal-manifest-v1",
        "model": "alpha.filing_alpha.fit_fusion_model",
        "ridge_alpha": _RIDGE_ALPHA,
        "horizon_sessions": _HORIZON_SESSIONS,
        "amd_evidence_sha256": amd,
        "amd_compute_sha256": amd["results/amd_compute.json"],
        "proposals": proposals,
    }
    payload["proposal_manifest_hash"] = canonical_hash(payload)
    return payload


def _feature_panel(
    config: ExperimentConfig, *, root: Path | None = None
) -> pd.DataFrame:
    market = adjusted_market_frame(config, root=root)
    market = market.loc[market["ticker"].isin(config.universe)].copy()
    calendar = tuple(sorted({pd.Timestamp(value).date() for value in market["date"]}))
    index = {day: offset for offset, day in enumerate(calendar)}
    filing_rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(config.universe):
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
    # The forward close has to be a bar already stored in the evidence tape,
    # and that bar has to fall before this decision. Missing labels are
    # dropped; they are not filled with a synthetic price.
    decision_index = int(
        panel.loc[panel["date"] == decision_ts, "session_index"].iloc[0]
    )
    future = panel.loc[:, ["ticker", "session_index", "close"]].rename(
        columns={"session_index": "label_index", "close": "label_close"}
    )
    history = history.merge(future, on=["ticker", "label_index"], how="inner")
    history = history.loc[history["label_index"] < decision_index].copy()
    if history.empty:
        raise ValueError(f"no pre-decision training rows for {decision.isoformat()}")
    history["target"] = history["label_close"] / history["close"] - 1.0
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
        "train_rows": int(len(history)),
        "intercept": float(fitted.model.intercept_),
        "coefficients": coefficients,
        "cross_section": [
            {"ticker": str(ticker), "score": float(value)}
            for ticker, value in zip(live["ticker"], scores, strict=True)
        ],
    }


def allocate_positive_score_book(
    scores: dict[str, float],
    config: ExperimentConfig,
    previous: dict[str, float] | None = None,
) -> dict[str, float]:
    """Map positive scores to weights inside the locked position and gross caps.

    Names with a positive score share the gross budget in proportion to that
    score. No name exceeds ``max_position_weight``. The book is then scaled
    down until post-cost gross leverage is inside ``max_gross_leverage``.
    Non-positive scores are not given a position.
    """

    held = {} if previous is None else dict(previous)
    positive = [
        (ticker, float(score))
        for ticker, score in scores.items()
        if float(score) > 0.0
    ]
    if not positive:
        return {}
    cap = float(config.max_position_weight)
    full = _capped_proportional(positive, float(config.max_gross_leverage), cap)
    scale = _largest_feasible_scale(full, held, config)
    return {
        ticker: weight * scale
        for ticker, weight in sorted(full.items())
        if weight * scale > 0.0
    }


def _capped_proportional(
    pairs: list[tuple[str, float]], gross_target: float, cap: float
) -> dict[str, float]:
    remaining = {ticker: score for ticker, score in pairs}
    assigned = {ticker: 0.0 for ticker, _score in pairs}
    budget = float(gross_target)
    for _step in range(len(pairs) + 1):
        if not remaining or budget <= 1e-15:
            break
        mass = sum(remaining.values())
        if mass <= 0.0:
            break
        takes: dict[str, float] = {}
        saturated: list[str] = []
        for ticker, magnitude in remaining.items():
            room = cap - assigned[ticker]
            take = min(budget * magnitude / mass, room)
            if take > 0.0:
                takes[ticker] = take
            if take >= room - 1e-15:
                saturated.append(ticker)
        if not takes:
            break
        for ticker, take in takes.items():
            assigned[ticker] += take
            budget -= take
        for ticker in saturated:
            remaining.pop(ticker, None)
    return {
        ticker: weight for ticker, weight in assigned.items() if weight > 0.0
    }


def _largest_feasible_scale(
    weights: dict[str, float],
    previous: dict[str, float],
    config: ExperimentConfig,
) -> float:
    # Leave a hair under the cap so share-level cost accounting, and a full
    # replacement of the previous book, still land inside the locked limit.
    post_cost_limit = float(config.max_gross_leverage) - 1e-4
    previous_gross = sum(abs(weight) for weight in previous.values())

    def feasible(scale: float) -> bool:
        trial = {ticker: weight * scale for ticker, weight in weights.items()}
        if any(
            abs(weight) > config.max_position_weight + 1e-12
            for weight in trial.values()
        ):
            return False
        gross = sum(abs(weight) for weight in trial.values())
        if gross > config.max_gross_leverage + 1e-12:
            return False
        worst_previous = {"__prior__": previous_gross} if previous_gross else {}
        post_cost = estimated_post_cost_gross_leverage(trial, worst_previous, config)
        return post_cost <= post_cost_limit

    if feasible(1.0):
        return 1.0
    lo = 0.0
    hi = 1.0
    best = 0.0
    for _step in range(60):
        mid = (lo + hi) / 2.0
        if feasible(mid):
            best = mid
            lo = mid
        else:
            hi = mid
    return best
