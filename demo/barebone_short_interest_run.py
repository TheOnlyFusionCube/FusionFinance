"""Size the Barebone short-interest arm from the locked FINRA change ratio.

The score is the latest period change ratio whose decision session falls in
the 63 sessions ending at the decision session, minus the cross-sectional
median. Availability is the FINRA publication date, not the settlement date.
The book is cash unless expanding Spearman skill is strictly above zero.
The momentum ledger is not rewritten. ``comparable_performance_claim`` stays
false.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from demo.barebone_comparison import (
    BAREBONE_EXPERIMENT_ID,
    LOCKED_EDGAR_SHA256,
    LOCKED_FORM4_SHA256,
    LOCKED_NARRATIVE_SHA256,
    LOCKED_TAPE_SHA256,
    SHORT_INTEREST_EVENTS,
    SHORT_INTEREST_PROVENANCE,
    refuse_fair_race_as_barebone_comparison,
)
from demo.barebone_edgar import demean_cross_section
from demo.barebone_run import BareboneMarket, load_barebone_market
from demo.barebone_short_interest import (
    SI_LOOKBACK_SESSIONS,
    SI_SCORE,
    SI_SIGNAL_ID,
    SI_SKILL_THRESHOLD,
    SI_TRANSFORM,
    short_interest_score_rows,
)
from demo.contracts import ExperimentConfig
from demo.controlled import (
    ArmInput,
    benchmark_marks_from_closes,
    load_policy_thresholds,
    run_controlled_arm,
)
from demo.pure_ml import allocate_positive_score_book, skill_allows_book, spearman_ic

LEDGER_RELATIVE = "results/barebone_short_interest_arm_ledger.json"
METRICS_RELATIVE = "results/barebone_short_interest_arm_metrics.json"
ARM_RELATIVE = "results/barebone_short_interest_arm.json"
_LEDGER_SCHEMA = "fusionfinance-barebone-short-interest-arm-ledger-v1"
_METRICS_SCHEMA = "fusionfinance-barebone-short-interest-arm-metrics-v1"
_ARM_SCHEMA = "fusionfinance-barebone-short-interest-arm-v1"
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


def raw_ratio_by_session(rows: Sequence[Mapping[str, object]]) -> dict[date, dict[str, float]]:
    grouped: dict[date, dict[str, float]] = {}
    for row in rows:
        if row.get("signal_id") != SI_SIGNAL_ID:
            raise ValueError("short interest row is not the locked signal")
        if int(row["lookback_sessions"]) != SI_LOOKBACK_SESSIONS:
            raise ValueError("short interest lookback is locked at 63 sessions")
        if _PERFORMANCE_KEYS.intersection(row):
            raise ValueError("short interest row carries a return label")
        session = date.fromisoformat(str(row["session"]))
        grouped.setdefault(session, {})[str(row["ticker"])] = float(row["change_ratio"])
    return grouped


def short_interest_oos_pairs(
    raw_by_session: Mapping[date, Mapping[str, float]],
    closes: Mapping[tuple[str, int], float],
    dates: Sequence[date],
) -> list[tuple[int, float, float]]:
    """Demeaned change ratios at session ``j`` versus the residual at ``j + 1``."""

    index = {day: offset for offset, day in enumerate(dates)}
    pairs: list[tuple[int, float, float]] = []
    for day, raw in raw_by_session.items():
        if day not in index:
            raise ValueError(f"short interest session {day.isoformat()} is not on the tape")
        session_index = index[day]
        outcome_index = session_index + 1
        if outcome_index >= len(dates):
            continue
        tradable: dict[str, tuple[float, float]] = {}
        for ticker, feature in raw.items():
            start = closes.get((ticker, session_index))
            end = closes.get((ticker, outcome_index))
            if start is None or end is None or start <= 0.0:
                continue
            tradable[ticker] = (float(feature), end / start - 1.0)
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
    usable = [pair for pair in pairs if pair[0] < decision_index]
    skill = spearman_ic([pair[1] for pair in usable], [pair[2] for pair in usable])
    return skill, len(usable)


def short_interest_book(
    raw: Mapping[str, float],
    skill: float | None,
    config: ExperimentConfig,
    previous: Mapping[str, float] | None = None,
) -> dict[str, float]:
    if not skill_allows_book(skill, SI_SKILL_THRESHOLD):
        return {}
    scores = demean_cross_section(raw)
    if len(scores) < 2:
        return {}
    return allocate_positive_score_book(dict(scores), config, dict(previous or {}))


def short_interest_rebalances(
    raw_by_session: Mapping[date, Mapping[str, float]],
    closes: Mapping[tuple[str, int], float],
    dates: Sequence[date],
    config: ExperimentConfig,
) -> list[dict[str, object]]:
    pairs = short_interest_oos_pairs(raw_by_session, closes, dates)
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
        raw = raw_by_session.get(dates[index], {})
        passed = skill_allows_book(skill, SI_SKILL_THRESHOLD)
        book = short_interest_book(raw, skill, config, previous)
        scores = demean_cross_section(raw) if book else {}
        rows.append(
            {
                "decision_session": dates[index].isoformat(),
                "oos_skill": skill,
                "oos_skill_pairs": pair_count,
                "oos_skill_threshold": SI_SKILL_THRESHOLD,
                "skill_pass": passed,
                "insufficient_history": not raw,
                "sized": bool(book),
                "name_count": len(raw),
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


def _load_events(path: Path, digest: str) -> list[dict[str, object]]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("short interest events bytes do not match the locked short_interest_sha256")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("short interest events file is empty")
    for row in rows:
        if row.get("available_ts", "").startswith(str(row.get("settlement_date", "missing"))):
            raise ValueError("settlement date must not be available_ts")
        if "changePercent" in row or "html" in row or "accountingYearMonthNumber" in row:
            raise ValueError("short interest event carries a rounded percent, html, or accounting date")
        if int(row["change_previous_number"]) != int(row["current_short_position"]) - int(
            row["previous_short_position"]
        ):
            raise ValueError("short interest change does not match the quantities")
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
                    strategy_id="short_interest",
                    decision_session=day,
                    ticker=cash_ticker,
                    structured_weight=0.0,
                )
            )
            continue
        for ticker, weight in sorted(book.items()):
            inputs.append(
                ArmInput(
                    strategy_id="short_interest",
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
            None if skill is None else float(skill), SI_SKILL_THRESHOLD
        ):
            raise ValueError("short interest sized a book when OOS skill failed the gate")


def short_interest_state(root: Path | None = None) -> dict[str, object]:
    base = _repo_root() if root is None else root
    cached = _STATE.get(str(base.resolve()))
    if cached is not None:
        return cached
    market = load_barebone_market(base)
    config = market.config
    if config.evidence.tape_sha256 != LOCKED_TAPE_SHA256 or market.tape_sha256 != LOCKED_TAPE_SHA256:
        raise ValueError("tape_sha256 is locked; refusing a new digest without a new experiment_id")
    if config.evidence.narrative_sha256 != LOCKED_NARRATIVE_SHA256:
        raise ValueError("narrative_sha256 is locked; refusing a new digest without a new experiment_id")
    if config.evidence.edgar_sha256 != LOCKED_EDGAR_SHA256:
        raise ValueError("edgar_sha256 is locked; refusing a new digest without a new experiment_id")
    if config.evidence.edgar_form4_sha256 != LOCKED_FORM4_SHA256:
        raise ValueError("edgar_form4_sha256 is locked; refusing a new digest without a new experiment_id")
    digest = config.evidence.short_interest_sha256
    if digest is None:
        raise ValueError("barebone short interest arm refuses an unlocked short_interest_sha256")
    provenance = json.loads((base / SHORT_INTEREST_PROVENANCE).read_text(encoding="utf-8"))
    if provenance.get("short_interest_sha256") != digest:
        raise ValueError("short interest provenance does not match the locked short_interest_sha256")
    if provenance.get("edgar_form4_sha256") != LOCKED_FORM4_SHA256:
        raise ValueError("short interest provenance does not match the locked edgar_form4_sha256")
    if provenance.get("edgar_sha256") != LOCKED_EDGAR_SHA256:
        raise ValueError("short interest provenance does not match the locked edgar_sha256")
    if provenance.get("tape_sha256") != LOCKED_TAPE_SHA256:
        raise ValueError("short interest provenance does not match the locked tape_sha256")
    if provenance.get("narrative_sha256") != LOCKED_NARRATIVE_SHA256:
        raise ValueError("short interest provenance does not match the locked narrative_sha256")
    if provenance.get("settlement_used_as_available_ts") is not False:
        raise ValueError("settlement date must not be the availability timestamp")
    if provenance.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    events = _load_events(base / SHORT_INTEREST_EVENTS, digest)
    trading = config.as_experiment_config()
    dates = tuple(session.session for session in market.sessions)
    panel_dates = tuple(sorted({pd.Timestamp(value).date() for value in market.panel["date"]}))
    if panel_dates != dates:
        raise ValueError("short interest panel calendar does not match the tape")
    scored = short_interest_score_rows(events, dates, tuple(trading.universe))
    raw = raw_ratio_by_session(scored)
    rebalances = short_interest_rebalances(raw, _closes(market.panel), dates, trading)
    _refuse_sized_without_skill(rebalances)
    run = run_controlled_arm(
        config=trading,
        sessions=market.sessions,
        strategy_id="short_interest",
        candidates=_arm_inputs(rebalances, trading),
        benchmark_marks=benchmark_marks_from_closes(
            market.sessions,
            benchmark_ticker=trading.benchmark_ticker,
            starting_capital=trading.starting_capital,
        ),
        thresholds=load_policy_thresholds(base / _CALIBRATION_RELATIVE),
    )
    if run.block_reason is not None or run.ledger is None or run.metrics is None or run.result is None:
        raise ValueError(f"short interest arm blocked: {run.block_reason}")
    state = {
        "market": market,
        "trading": trading,
        "dates": dates,
        "rebalances": rebalances,
        "run": run,
        "short_interest_sha256": digest,
        "provenance": provenance,
        "event_count": len(events),
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
            "insufficient_history": row["insufficient_history"],
            "sized": row["sized"],
            "name_count": row["name_count"],
        }
        for row in rows
    ]


def _decisions(run, rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_session = {str(row["decision_session"]): row for row in rows}
    decisions: list[dict[str, object]] = []
    for proposal in run.lineage.proposals:
        row = by_session[proposal.decision_session.isoformat()]
        if proposal.target_weight != 0.0 and not row["sized"]:
            raise ValueError("short interest sized a book when the skill gate was cash")
        if proposal.target_weight != 0.0 and (
            row["oos_skill"] is None or float(row["oos_skill"]) <= SI_SKILL_THRESHOLD
        ):
            raise ValueError("short interest sized a book when OOS skill is not strictly positive")
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
    return decisions


def barebone_short_interest_ledger(
    root: Path | None = None, *, bound: Mapping[str, object] | None = None
) -> dict[str, object]:
    state = short_interest_state(root) if bound is None else bound
    market: BareboneMarket = state["market"]
    trading: ExperimentConfig = state["trading"]
    run = state["run"]
    rebalances: list[dict[str, object]] = state["rebalances"]
    ledger = run.ledger
    if ledger is None:
        raise ValueError("short interest arm is missing a ledger")
    document: dict[str, object] = {
        "schema": _LEDGER_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "claim_status": "controlled_software_ledger",
        "comparable_performance_claim": False,
        "description": (
            "Software ledger for the barebone FINRA short-interest arm. The "
            "score is the latest period change ratio in the 63 sessions ending "
            "at the decision session, minus the cross-sectional median. "
            "Availability is the publication date, not the settlement date. "
            "The log1p level is not used. The book is sized only when expanding "
            "Spearman skill versus the next-session residual is strictly above "
            "zero; otherwise it is cash. Position and gross caps stay 0.10 and "
            "1.0. This is not a performance claim."
        ),
        "strategy_id": "short_interest",
        "signal_id": SI_SIGNAL_ID,
        "tape_sha256": market.tape_sha256,
        "narrative_sha256": LOCKED_NARRATIVE_SHA256,
        "edgar_sha256": LOCKED_EDGAR_SHA256,
        "edgar_form4_sha256": LOCKED_FORM4_SHA256,
        "short_interest_sha256": state["short_interest_sha256"],
        "window": [trading.start_date.isoformat(), trading.end_date.isoformat()],
        "session_count": len(state["dates"]),
        "rebalance_count": len(rebalances),
        "sized_rebalances": sum(1 for row in rebalances if row["sized"]),
        "cash_rebalances": sum(1 for row in rebalances if not row["sized"]),
        "max_position_weight": trading.max_position_weight,
        "max_gross_leverage": trading.max_gross_leverage,
        "model_binding": {
            "signal_id": SI_SIGNAL_ID,
            "lookback_sessions": SI_LOOKBACK_SESSIONS,
            "transform": SI_TRANSFORM,
            "score": SI_SCORE,
            "available_ts": "publication date",
            "settlement_used_as_available_ts": False,
            "oos_skill_threshold": SI_SKILL_THRESHOLD,
            "short_interest_sha256": state["short_interest_sha256"],
            "edgar_form4_sha256": LOCKED_FORM4_SHA256,
            "edgar_sha256": LOCKED_EDGAR_SHA256,
            "narrative_sha256": LOCKED_NARRATIVE_SHA256,
            "tape_sha256": LOCKED_TAPE_SHA256,
        },
        "config_hash": run.lineage.config_hash,
        "tape_hash": run.lineage.tape_hash,
        "lineage_hash": run.lineage.lineage_hash,
        "experiment_hash": run.lineage.experiment_hash,
        "admitted_count": run.admitted_count,
        "trade_count": ledger.trade_count,
        "total_turnover": ledger.total_turnover,
        "transaction_costs": ledger.transaction_costs,
        "slippage_costs": ledger.slippage_costs,
        "post_cost_within_limit": ledger.post_cost_within_limit,
        "oos_skill": _skill_log(rebalances),
        "decisions": _decisions(run, rebalances),
    }
    if _PERFORMANCE_KEYS.intersection(_mapping_keys(document)):
        raise ValueError("short interest ledger must not carry performance-claim fields")
    refuse_fair_race_as_barebone_comparison(document)
    return document


def barebone_short_interest_metrics(
    root: Path | None = None, *, bound: Mapping[str, object] | None = None
) -> dict[str, object]:
    state = short_interest_state(root) if bound is None else bound
    market: BareboneMarket = state["market"]
    trading: ExperimentConfig = state["trading"]
    run = state["run"]
    if run.result is None or run.metrics is None or run.ledger is None:
        raise ValueError("short interest arm is missing an executed result")
    statistics = run.metrics.model_dump(mode="json")
    values = [point.portfolio_value for point in run.result.points]
    if not math.isfinite(float(statistics["total_return"])):
        raise ValueError("short interest return is not finite")
    if not math.isclose(float(statistics["total_return"]), values[-1] / values[0] - 1.0):
        raise ValueError("cumulative return does not match the wealth path")
    document: dict[str, object] = {
        "schema": _METRICS_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "claim_status": "controlled_software_fixture_metrics",
        "comparable_performance_claim": False,
        "description": (
            "Fixture metrics for the barebone FINRA short-interest arm. The "
            "feature is the publication-dated change ratio. The book is cash "
            "unless expanding Spearman skill is strictly above zero. Not a "
            "capital performance claim."
        ),
        "strategy_id": "short_interest",
        "signal_id": SI_SIGNAL_ID,
        "tape_sha256": market.tape_sha256,
        "narrative_sha256": LOCKED_NARRATIVE_SHA256,
        "edgar_sha256": LOCKED_EDGAR_SHA256,
        "edgar_form4_sha256": LOCKED_FORM4_SHA256,
        "short_interest_sha256": state["short_interest_sha256"],
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
    refuse_fair_race_as_barebone_comparison(document)
    return document


def barebone_short_interest_arm_index(
    root: Path | None = None, *, bound: Mapping[str, object] | None = None
) -> dict[str, object]:
    state = short_interest_state(root) if bound is None else bound
    rebalances: list[dict[str, object]] = state["rebalances"]
    document: dict[str, object] = {
        "schema": _ARM_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "claim_status": "controlled_software_ledger",
        "comparable_performance_claim": False,
        "strategy_id": "short_interest",
        "signal_id": SI_SIGNAL_ID,
        "score": SI_SCORE,
        "short_interest_sha256": state["short_interest_sha256"],
        "short_interest_events": SHORT_INTEREST_EVENTS,
        "edgar_form4_sha256": LOCKED_FORM4_SHA256,
        "edgar_sha256": LOCKED_EDGAR_SHA256,
        "narrative_sha256": LOCKED_NARRATIVE_SHA256,
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
        raise ValueError("short interest arm index must not carry a performance field")
    refuse_fair_race_as_barebone_comparison(document)
    return document


def write_barebone_short_interest_artifacts(
    root: Path | None = None, *, bound: Mapping[str, object] | None = None
) -> tuple[Path, Path, Path]:
    base = _repo_root() if root is None else root
    state = short_interest_state(base) if bound is None else bound
    ledger_path = base / LEDGER_RELATIVE
    metrics_path = base / METRICS_RELATIVE
    index_path = base / ARM_RELATIVE
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        _json_text(barebone_short_interest_ledger(base, bound=state)), encoding="utf-8"
    )
    metrics_path.write_text(
        _json_text(barebone_short_interest_metrics(base, bound=state)), encoding="utf-8"
    )
    index_path.write_text(
        _json_text(barebone_short_interest_arm_index(base, bound=state)), encoding="utf-8"
    )
    return ledger_path, metrics_path, index_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        ledger_path, metrics_path, _index_path = write_barebone_short_interest_artifacts()
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(f"signal_id={SI_SIGNAL_ID}")
    print(f"short_interest_sha256={ledger['short_interest_sha256']}")
    print(f"edgar_form4_sha256={ledger['edgar_form4_sha256']}")
    print(f"sized_rebalances={ledger['sized_rebalances']}")
    print(f"cash_rebalances={ledger['cash_rebalances']}")
    print(f"total_return={metrics['statistics']['total_return']}")
    print(f"sharpe_ratio={metrics['statistics']['sharpe_ratio']}")
    print(f"ledger={ledger_path}")
    print(f"metrics={metrics_path}")
    return 0
