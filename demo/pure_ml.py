"""Walk-forward filing scores for the controlled pure-ML arm.

Prices come from the evidence tape. Scores come from
``alpha.filing_alpha.fit_fusion_model`` using only labels whose forward close
is already in that tape and ends before the decision. Before sizing, an
expanding walk-forward Spearman check scores names left out of that fold's
fit against the next-session residual. Non-positive skill leaves the book in
cash. The proposal manifest is hash-bound to the AMD receipts.
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
_OOS_SKILL_THRESHOLD = 0.0
_MIN_OOS_PAIRS = 8
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
    oos_pairs = _expanding_oos_pairs(panel, config)
    proposals: list[dict[str, object]] = []
    previous_ml: dict[str, float] = {}
    previous_fusion: dict[str, float] = {}
    for decision in decision_dates:
        scored = _score_decision(config, panel, decision)
        scores = {
            str(item["ticker"]): float(item["score"])
            for item in scored["cross_section"]
        }
        decision_index = int(
            panel.loc[panel["date"] == pd.Timestamp(decision), "session_index"].iloc[0]
        )
        skill, pair_count = _skill_before(oos_pairs, decision_index)
        skill_pass = skill_allows_book(skill, _OOS_SKILL_THRESHOLD)
        fusion_book = allocate_positive_score_book(scores, config, previous_fusion)
        ml_book = (
            allocate_positive_score_book(scores, config, previous_ml)
            if skill_pass
            else {}
        )
        positive = sum(score > 0.0 for score in scores.values())
        if positive > 1 and len(fusion_book) < 2:
            raise ValueError(
                "positive scores support multiple names but the book has one"
            )
        if skill_pass and positive > 1 and len(ml_book) < 2:
            raise ValueError(
                "positive scores support multiple names but the book has one"
            )
        scored["oos_skill"] = skill
        scored["oos_skill_pairs"] = pair_count
        scored["oos_skill_threshold"] = _OOS_SKILL_THRESHOLD
        scored["skill_pass"] = skill_pass
        scored["targets"] = _target_rows(ml_book, scores)
        scored["fusion_targets"] = _target_rows(fusion_book, scores)
        scored["gross_exposure"] = float(sum(ml_book.values()))
        proposals.append(scored)
        previous_ml = dict(ml_book)
        previous_fusion = dict(fusion_book)
    amd = amd_evidence_sha256(root)
    payload = {
        "schema": "fusionfinance-pure-ml-proposal-manifest-v1",
        "model": "alpha.filing_alpha.fit_fusion_model",
        "ridge_alpha": _RIDGE_ALPHA,
        "horizon_sessions": _HORIZON_SESSIONS,
        "oos_skill_threshold": _OOS_SKILL_THRESHOLD,
        "oos_skill": "spearman_ic of held-out names versus next-session residual return",
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


def skill_allows_book(skill: float | None, threshold: float = _OOS_SKILL_THRESHOLD) -> bool:
    """True only when measured out-of-sample skill is strictly above the gate."""

    return skill is not None and skill > threshold


def spearman_ic(left: list[float], right: list[float]) -> float | None:
    """Spearman rank correlation. Undefined samples return ``None``."""

    if len(left) != len(right) or len(left) < _MIN_OOS_PAIRS:
        return None
    xs = np.asarray(left, dtype=float)
    ys = np.asarray(right, dtype=float)
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        return None
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ordered = values[order]
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def _target_rows(
    book: dict[str, float], scores: dict[str, float]
) -> list[dict[str, object]]:
    return [
        {
            "ticker": ticker,
            "model_score": scores[ticker],
            "structured_weight": weight,
        }
        for ticker, weight in sorted(book.items())
    ]


def _expanding_oos_pairs(
    panel: pd.DataFrame, config: ExperimentConfig
) -> list[tuple[int, float, float]]:
    """Score names that were excluded from each fold's fit.

    Each pair is a held-out name's score at session ``j`` and that name's
    next-session residual return. The fit uses only the other names, and
    only labels that ended before ``j``. The residual itself is not a
    training label.
    """

    universe = list(config.universe)
    midpoint = len(universe) // 2
    if midpoint < 1 or midpoint >= len(universe):
        raise ValueError("OOS skill folds require at least two names")
    folds = (universe[:midpoint], universe[midpoint:])
    labeled = panel.copy()
    labeled["label_index"] = labeled["session_index"] + _HORIZON_SESSIONS
    future = panel.loc[:, ["ticker", "session_index", "close"]].rename(
        columns={"session_index": "label_index", "close": "label_close"}
    )
    labeled = labeled.merge(future, on=["ticker", "label_index"], how="inner")
    labeled["target"] = labeled["label_close"] / labeled["close"] - 1.0
    closes = {
        (str(ticker), int(index)): float(close)
        for ticker, index, close in zip(
            panel["ticker"], panel["session_index"], panel["close"], strict=True
        )
    }
    indices = sorted({int(value) for value in panel["session_index"]})
    index_set = set(indices)
    pairs: list[tuple[int, float, float]] = []
    for session_index in indices:
        outcome_index = session_index + 1
        if outcome_index not in index_set:
            continue
        raw: dict[str, float] = {}
        for ticker in universe:
            start = closes.get((ticker, session_index))
            end = closes.get((ticker, outcome_index))
            if start is None or end is None or start <= 0.0:
                continue
            raw[ticker] = end / start - 1.0
        if len(raw) < 2:
            continue
        mean_return = sum(raw.values()) / len(raw)
        residual = {ticker: value - mean_return for ticker, value in raw.items()}
        live = panel.loc[panel["session_index"] == session_index]
        for fit_names, score_names in ((folds[0], folds[1]), (folds[1], folds[0])):
            if set(fit_names) & set(score_names):
                raise ValueError("OOS score names overlap the fit names")
            train = labeled.loc[
                labeled["ticker"].isin(fit_names)
                & (labeled["label_index"] < session_index)
            ]
            if train.empty:
                continue
            fitted = fit_fusion_model(
                train,
                "target",
                ["trail_return"],
                ["filing_signal_decayed"],
                ridge_alpha=_RIDGE_ALPHA,
            )
            present = [
                name
                for name in score_names
                if name in residual and name in set(live["ticker"])
            ]
            held_out = (
                live.loc[live["ticker"].isin(present)]
                .set_index("ticker")
                .loc[present]
                .reset_index()
            )
            if held_out.empty:
                continue
            predicted = fitted.predict(held_out)
            for ticker, score in zip(held_out["ticker"], predicted, strict=True):
                name = str(ticker)
                if name in fit_names or name not in residual:
                    continue
                pairs.append((outcome_index, float(score), float(residual[name])))
    return pairs


def _skill_before(
    pairs: list[tuple[int, float, float]], decision_index: int
) -> tuple[float | None, int]:
    usable = [pair for pair in pairs if pair[0] < decision_index]
    skill = spearman_ic(
        [pair[1] for pair in usable],
        [pair[2] for pair in usable],
    )
    return skill, len(usable)


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
