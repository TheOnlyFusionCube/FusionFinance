"""Size the Barebone narrative arm from the frozen polarity scores.

The book uses the same rebalance clock, 0.10 position cap, 1.0 gross cap, and
expanding Spearman gate as the momentum pure-ML arm. Skill at or below zero
is cash. This module does not rewrite the momentum ledger or the events file.
``comparable_performance_claim`` stays false.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from demo.barebone_comparison import (
    BAREBONE_EXPERIMENT_ID,
    LOCKED_NARRATIVE_SHA256,
    NARRATIVE_EVENTS,
    NARRATIVE_POLARITY_PROVENANCE,
    NARRATIVE_SCORES,
)
from demo.barebone_polarity import (
    POLARITY_AGGREGATION,
    POLARITY_MODEL_ID,
    POLARITY_SKILL_THRESHOLD,
    demean_cross_section,
    mean_polarity_by_session,
)
from demo.barebone_run import BareboneMarket, load_barebone_market
from demo.contracts import ExperimentConfig
from demo.controlled import (
    ArmInput,
    benchmark_marks_from_closes,
    load_policy_thresholds,
    run_controlled_arm,
)
from demo.pure_ml import allocate_positive_score_book, skill_allows_book, spearman_ic


LEDGER_RELATIVE = "results/barebone_narrative_arm_ledger.json"
METRICS_RELATIVE = "results/barebone_narrative_arm_metrics.json"
ARM_RELATIVE = "results/barebone_narrative_arm.json"
_LEDGER_SCHEMA = "fusionfinance-barebone-narrative-arm-ledger-v1"
_METRICS_SCHEMA = "fusionfinance-barebone-narrative-arm-metrics-v1"
_ARM_SCHEMA = "fusionfinance-barebone-narrative-arm-v1"
_CALIBRATION_RELATIVE = "results/fusion_policy_calibration.json"
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
_STATE: dict[str, dict[str, object]] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _mapping_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        found.update(str(key) for key in value)
        for item in value.values():
            found.update(_mapping_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_mapping_keys(item))
    return found


def narrative_oos_pairs(
    means_by_session: Mapping[date, Mapping[str, float]],
    closes: Mapping[tuple[str, int], float],
    dates: Sequence[date],
) -> list[tuple[int, float, float]]:
    """Demeaned polarity at session ``j`` versus the residual at ``j + 1``.

    The polarity was fixed before ``j``. The residual is the skill label.
    A pair is usable at a later decision only after ``j + 1`` is already past.
    """

    index = {day: offset for offset, day in enumerate(dates)}
    pairs: list[tuple[int, float, float]] = []
    for day, means in means_by_session.items():
        if day not in index:
            raise ValueError(f"narrative session {day.isoformat()} is not on the tape")
        session_index = index[day]
        outcome_index = session_index + 1
        if outcome_index >= len(dates):
            continue
        tradable: dict[str, tuple[float, float]] = {}
        for ticker, polarity in means.items():
            start = closes.get((ticker, session_index))
            end = closes.get((ticker, outcome_index))
            if start is None or end is None or start <= 0.0:
                continue
            tradable[ticker] = (float(polarity), end / start - 1.0)
        if len(tradable) < 2:
            continue
        scores = demean_cross_section({ticker: pair[0] for ticker, pair in tradable.items()})
        if len(scores) < 2:
            continue
        mean_return = sum(pair[1] for pair in tradable.values()) / len(tradable)
        for ticker, score in scores.items():
            pairs.append((outcome_index, float(score), float(tradable[ticker][1] - mean_return)))
    return pairs


def skill_before(
    pairs: Sequence[tuple[int, float, float]], decision_index: int
) -> tuple[float | None, int]:
    """Spearman of pairs whose outcome session is already before the decision."""

    usable = [pair for pair in pairs if pair[0] < decision_index]
    skill = spearman_ic([pair[1] for pair in usable], [pair[2] for pair in usable])
    return skill, len(usable)


def narrative_book(
    means: Mapping[str, float],
    skill: float | None,
    config: ExperimentConfig,
    previous: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Positive demeaned polarity, or cash when skill is not strictly above zero."""

    if not skill_allows_book(skill, POLARITY_SKILL_THRESHOLD):
        return {}
    scores = demean_cross_section(means)
    if len(scores) < 2:
        return {}
    return allocate_positive_score_book(dict(scores), config, dict(previous or {}))


def narrative_rebalances(
    means_by_session: Mapping[date, Mapping[str, float]],
    closes: Mapping[tuple[str, int], float],
    dates: Sequence[date],
    config: ExperimentConfig,
) -> list[dict[str, object]]:
    """One row per shared rebalance. Non-positive skill is an empty book."""

    pairs = narrative_oos_pairs(means_by_session, closes, dates)
    clock = [
        index
        for index in range(len(dates))
        if index % config.rebalance_frequency_sessions == 0
        and index + config.execution_lag_sessions < len(dates)
    ]
    rows: list[dict[str, object]] = []
    previous: dict[str, float] = {}
    for index in clock:
        skill, pair_count = skill_before(pairs, index)
        means = means_by_session.get(dates[index], {})
        book = narrative_book(means, skill, config, previous)
        scores = demean_cross_section(means) if book else {}
        rows.append(
            {
                "decision_session": dates[index].isoformat(),
                "oos_skill": skill,
                "oos_skill_pairs": pair_count,
                "oos_skill_threshold": POLARITY_SKILL_THRESHOLD,
                "skill_pass": skill_allows_book(skill, POLARITY_SKILL_THRESHOLD),
                "sized": bool(book),
                "name_count": len(means),
                "book": book,
                "scores": scores,
            }
        )
        previous = dict(book)
    return rows


def _closes(panel: pd.DataFrame) -> dict[tuple[str, int], float]:
    return {
        (str(ticker), int(index)): float(close)
        for ticker, index, close in zip(
            panel["ticker"], panel["session_index"], panel["close"], strict=True
        )
    }


def _load_score_rows(path: Path, digest: str) -> list[dict[str, object]]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("narrative scores bytes do not match the locked polarity_sha256")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("narrative scores file is empty")
    for row in rows:
        if row.get("polarity_model_id") != POLARITY_MODEL_ID:
            raise ValueError("narrative score is not the frozen lexicon")
        if "text" in row or _PERFORMANCE_KEYS.intersection(row):
            raise ValueError("narrative score carries text or a return label")
    return rows


def _arm_inputs(rows: Sequence[Mapping[str, object]], config: ExperimentConfig) -> list[ArmInput]:
    inputs: list[ArmInput] = []
    cash_ticker = config.universe[0]
    for row in rows:
        day = date.fromisoformat(str(row["decision_session"]))
        book = row["book"]
        if not isinstance(book, dict) or not book:
            inputs.append(
                ArmInput(
                    strategy_id="narrative",
                    decision_session=day,
                    ticker=cash_ticker,
                    structured_weight=0.0,
                )
            )
            continue
        for ticker, weight in sorted(book.items()):
            inputs.append(
                ArmInput(
                    strategy_id="narrative",
                    decision_session=day,
                    ticker=str(ticker),
                    structured_weight=float(weight),
                )
            )
    return inputs


def _refuse_sized_without_skill(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        skill = row["oos_skill"]
        if row["sized"] and not skill_allows_book(
            None if skill is None else float(skill), POLARITY_SKILL_THRESHOLD
        ):
            raise ValueError("narrative sized a book when OOS skill failed the gate")
        if skill is not None and float(skill) <= POLARITY_SKILL_THRESHOLD and row["sized"]:
            raise ValueError("narrative sized a book when OOS skill is not strictly positive")


def narrative_state(root: Path | None = None) -> dict[str, object]:
    """Load the locked tape and scores and execute the narrative arm."""

    base = _repo_root() if root is None else root
    cached = _STATE.get(str(base.resolve()))
    if cached is not None:
        return cached
    market = load_barebone_market(base)
    config = market.config
    if config.evidence.narrative_sha256 != LOCKED_NARRATIVE_SHA256:
        raise ValueError("narrative_sha256 is locked; refusing a new digest without a new experiment_id")
    if config.evidence.tape_sha256 != market.tape_sha256:
        raise ValueError("narrative arm tape_sha256 does not match the loaded tape")
    polarity_sha = config.evidence.polarity_sha256
    if polarity_sha is None:
        raise ValueError("barebone narrative arm refuses an unlocked polarity_sha256")
    events_path = base / NARRATIVE_EVENTS
    if hashlib.sha256(events_path.read_bytes()).hexdigest() != LOCKED_NARRATIVE_SHA256:
        raise ValueError("narrative events bytes do not match the locked narrative_sha256")
    provenance = json.loads((base / NARRATIVE_POLARITY_PROVENANCE).read_text(encoding="utf-8"))
    if provenance.get("polarity_sha256") != polarity_sha:
        raise ValueError("polarity provenance does not match the locked polarity_sha256")
    if provenance.get("narrative_events_sha256") != LOCKED_NARRATIVE_SHA256:
        raise ValueError("polarity provenance does not match the locked narrative_sha256")
    if provenance.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    rows = _load_score_rows(base / NARRATIVE_SCORES, polarity_sha)
    trading = config.as_experiment_config()
    dates = tuple(session.session for session in market.sessions)
    panel_dates = tuple(sorted({pd.Timestamp(value).date() for value in market.panel["date"]}))
    if panel_dates != dates:
        raise ValueError("narrative panel calendar does not match the tape")
    means = mean_polarity_by_session(rows)
    rebalances = narrative_rebalances(means, _closes(market.panel), dates, trading)
    _refuse_sized_without_skill(rebalances)
    run = run_controlled_arm(
        config=trading,
        sessions=market.sessions,
        strategy_id="narrative",
        candidates=_arm_inputs(rebalances, trading),
        benchmark_marks=benchmark_marks_from_closes(
            market.sessions,
            benchmark_ticker=trading.benchmark_ticker,
            starting_capital=trading.starting_capital,
        ),
        thresholds=load_policy_thresholds(base / _CALIBRATION_RELATIVE),
    )
    if run.block_reason is not None or run.ledger is None or run.metrics is None or run.result is None:
        raise ValueError(f"narrative arm blocked: {run.block_reason}")
    state = {
        "market": market,
        "trading": trading,
        "dates": dates,
        "rebalances": rebalances,
        "run": run,
        "polarity_sha256": polarity_sha,
        "provenance": provenance,
        "event_count": len(rows),
    }
    _STATE[str(base.resolve())] = state
    return state


def _skill_log(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "decision_session": row["decision_session"],
            "oos_skill": row["oos_skill"],
            "oos_skill_pairs": row["oos_skill_pairs"],
            "oos_skill_threshold": row["oos_skill_threshold"],
            "skill_pass": row["skill_pass"],
            "sized": row["sized"],
            "name_count": row["name_count"],
        }
        for row in rows
    ]


def barebone_narrative_ledger(root: Path | None = None) -> dict[str, object]:
    """Software ledger for the narrative arm. It does not carry a Sharpe."""

    state = narrative_state(root)
    market: BareboneMarket = state["market"]
    trading: ExperimentConfig = state["trading"]
    run = state["run"]
    rebalances: list[dict[str, object]] = state["rebalances"]
    by_session = {str(row["decision_session"]): row for row in rebalances}
    lineage = run.lineage
    ledger = run.ledger
    if ledger is None:
        raise ValueError("narrative arm is missing a ledger")
    decisions: list[dict[str, object]] = []
    for proposal in lineage.proposals:
        row = by_session[proposal.decision_session.isoformat()]
        if proposal.target_weight != 0.0 and not row["sized"]:
            raise ValueError("narrative sized a book when the skill gate was cash")
        if proposal.target_weight != 0.0 and (
            row["oos_skill"] is None or float(row["oos_skill"]) <= POLARITY_SKILL_THRESHOLD
        ):
            raise ValueError("narrative sized a book when OOS skill is not strictly positive")
        decisions.append(
            {
                "decision_session": proposal.decision_session.isoformat(),
                "ticker": proposal.ticker,
                "admitted": proposal.admitted,
                "target_weight": proposal.target_weight,
                "reason": proposal.reason,
                "model_score": None
                if not isinstance(row["scores"], dict)
                else row["scores"].get(proposal.ticker),
                "oos_skill": row["oos_skill"],
                "oos_skill_pairs": row["oos_skill_pairs"],
                "oos_skill_threshold": row["oos_skill_threshold"],
            }
        )
    document: dict[str, object] = {
        "schema": _LEDGER_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "claim_status": "controlled_software_ledger",
        "comparable_performance_claim": False,
        "description": (
            "Software ledger for the barebone narrative arm on the locked "
            "Hacker News events and the locked Yahoo tape. Polarity is the "
            "frozen lexicon scored once from text available before the "
            "decision session. The session score is that mean minus the "
            "cross-sectional median. The book is sized only when expanding "
            "Spearman skill versus the next-session residual is strictly "
            "above zero; otherwise it is cash. Position and gross caps stay "
            "0.10 and 1.0. This is not a performance claim."
        ),
        "strategy_id": "narrative",
        "tape_sha256": market.tape_sha256,
        "narrative_sha256": LOCKED_NARRATIVE_SHA256,
        "polarity_sha256": state["polarity_sha256"],
        "window": [trading.start_date.isoformat(), trading.end_date.isoformat()],
        "session_count": len(state["dates"]),
        "rebalance_count": len(rebalances),
        "sized_rebalances": sum(1 for row in rebalances if row["sized"]),
        "cash_rebalances": sum(1 for row in rebalances if not row["sized"]),
        "max_position_weight": trading.max_position_weight,
        "max_gross_leverage": trading.max_gross_leverage,
        "model_binding": {
            "model": POLARITY_MODEL_ID,
            "aggregation": POLARITY_AGGREGATION,
            "oos_skill_threshold": POLARITY_SKILL_THRESHOLD,
            "lexicon_sha256": state["provenance"]["lexicon_sha256"],
            "polarity_sha256": state["polarity_sha256"],
            "narrative_sha256": LOCKED_NARRATIVE_SHA256,
        },
        "config_hash": lineage.config_hash,
        "tape_hash": lineage.tape_hash,
        "lineage_hash": lineage.lineage_hash,
        "experiment_hash": lineage.experiment_hash,
        "admitted_count": run.admitted_count,
        "trade_count": ledger.trade_count,
        "total_turnover": ledger.total_turnover,
        "transaction_costs": ledger.transaction_costs,
        "slippage_costs": ledger.slippage_costs,
        "post_cost_within_limit": ledger.post_cost_within_limit,
        "oos_skill": _skill_log(rebalances),
        "decisions": decisions,
    }
    if document["comparable_performance_claim"] is not False:
        raise ValueError("comparable_performance_claim must be false")
    if _PERFORMANCE_KEYS.intersection(_mapping_keys(document)):
        raise ValueError("narrative ledger must not carry performance-claim fields")
    return document


def barebone_narrative_metrics(root: Path | None = None) -> dict[str, object]:
    """Fixture metrics for the narrative arm. The claim stays false."""

    state = narrative_state(root)
    market: BareboneMarket = state["market"]
    trading: ExperimentConfig = state["trading"]
    run = state["run"]
    if run.result is None or run.metrics is None or run.ledger is None:
        raise ValueError("narrative arm is missing an executed result")
    statistics = run.metrics.model_dump(mode="json")
    values = [point.portfolio_value for point in run.result.points]
    if not math.isfinite(float(statistics["total_return"])):
        raise ValueError("narrative return is not finite")
    if not math.isclose(float(statistics["total_return"]), values[-1] / values[0] - 1.0):
        raise ValueError("cumulative return does not match the wealth path")
    document: dict[str, object] = {
        "schema": _METRICS_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "claim_status": "controlled_software_fixture_metrics",
        "comparable_performance_claim": False,
        "description": (
            "Fixture metrics for the barebone narrative arm. Polarity is the "
            "frozen lexicon. The book is cash unless expanding Spearman skill "
            "is strictly above zero. Not a capital performance claim."
        ),
        "strategy_id": "narrative",
        "tape_sha256": market.tape_sha256,
        "narrative_sha256": LOCKED_NARRATIVE_SHA256,
        "polarity_sha256": state["polarity_sha256"],
        "source_ledger": LEDGER_RELATIVE,
        "window": [trading.start_date.isoformat(), trading.end_date.isoformat()],
        "session_count": len(state["dates"]),
        "max_position_weight": trading.max_position_weight,
        "max_gross_leverage": trading.max_gross_leverage,
        "config_hash": run.lineage.config_hash,
        "tape_hash": run.lineage.tape_hash,
        "experiment_hash": run.lineage.experiment_hash,
        "portfolio_sessions": [point.session.isoformat() for point in run.result.points],
        "portfolio_values": values,
        "statistics": statistics,
        "oos_skill": _skill_log(state["rebalances"]),
    }
    if document["comparable_performance_claim"] is not False:
        raise ValueError("comparable_performance_claim must be false")
    return document


def barebone_narrative_arm_index(root: Path | None = None) -> dict[str, object]:
    """Pointer at the narrative ledger. It does not publish a Sharpe."""

    state = narrative_state(root)
    rebalances: list[dict[str, object]] = state["rebalances"]
    document: dict[str, object] = {
        "schema": _ARM_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "claim_status": "controlled_software_ledger",
        "comparable_performance_claim": False,
        "strategy_id": "narrative",
        "narrative_sha256": LOCKED_NARRATIVE_SHA256,
        "narrative_events": NARRATIVE_EVENTS,
        "polarity_sha256": state["polarity_sha256"],
        "polarity_model_id": POLARITY_MODEL_ID,
        "tape_sha256": state["market"].tape_sha256,
        "event_count": state["event_count"],
        "sized_rebalances": sum(1 for row in rebalances if row["sized"]),
        "cash_rebalances": sum(1 for row in rebalances if not row["sized"]),
        "max_position_weight": 0.1,
        "max_gross_leverage": 1.0,
        "ledger": LEDGER_RELATIVE,
        "metrics": METRICS_RELATIVE,
    }
    if _PERFORMANCE_KEYS.intersection(document):
        raise ValueError("narrative arm index must not carry a performance field")
    return document


def write_barebone_narrative_artifacts(root: Path | None = None) -> tuple[Path, Path, Path]:
    """Write the narrative ledger, metrics, and index. Does not write dumps."""

    base = _repo_root() if root is None else root
    ledger = barebone_narrative_ledger(base)
    metrics = barebone_narrative_metrics(base)
    index = barebone_narrative_arm_index(base)
    ledger_path = base / LEDGER_RELATIVE
    metrics_path = base / METRICS_RELATIVE
    index_path = base / ARM_RELATIVE
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(_json_text(ledger), encoding="utf-8")
    metrics_path.write_text(_json_text(metrics), encoding="utf-8")
    index_path.write_text(_json_text(index), encoding="utf-8")
    return ledger_path, metrics_path, index_path


def main(argv: list[str] | None = None) -> int:
    """Execute the locked narrative arm. Does not call a live model."""

    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        ledger_path, metrics_path, _index_path = write_barebone_narrative_artifacts()
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(f"polarity_sha256={ledger['polarity_sha256']}")
    print(f"sized_rebalances={ledger['sized_rebalances']}")
    print(f"cash_rebalances={ledger['cash_rebalances']}")
    print(f"total_return={metrics['statistics']['total_return']}")
    print(f"sharpe_ratio={metrics['statistics']['sharpe_ratio']}")
    print(f"ledger={ledger_path}")
    print(f"metrics={metrics_path}")
    return 0
