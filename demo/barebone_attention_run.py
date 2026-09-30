"""Size attention and hybrid arms on the locked Barebone tape.

Attention sizes demeaned ``log1p`` event counts only when expanding Spearman
skill is strictly above zero. Hybrid sizes momentum scores only on the
intersection of the momentum skill gate and the attention skill gate.
Otherwise each arm is cash. The lexicon is not a feature. The momentum
three-arm ledger is not rewritten. ``comparable_performance_claim`` stays false.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from demo.barebone_attention import (
    ATTENTION_LOOKBACK_SESSIONS,
    ATTENTION_SIGNAL_ID,
    ATTENTION_SKILL_THRESHOLD,
    ATTENTION_TRANSFORM,
    FROZEN_LEXICON_SHA256,
    HYBRID_SIGNAL_ID,
    demean_cross_section,
)
from demo.barebone_comparison import (
    ATTENTION_PROVENANCE,
    ATTENTION_SCORES,
    BAREBONE_EXPERIMENT_ID,
    HYBRID_PROVENANCE,
    LOCKED_NARRATIVE_SHA256,
    LOCKED_POLARITY_SHA256,
    LOCKED_TAPE_SHA256,
    NARRATIVE_EVENTS,
    NARRATIVE_LEXICON,
    refuse_fair_race_as_barebone_comparison,
)
from demo.barebone_run import BareboneMarket, load_barebone_market
from demo.contracts import ExperimentConfig
from demo.controlled import (
    ArmInput,
    benchmark_marks_from_closes,
    load_policy_thresholds,
    run_controlled_arm,
)
from demo.pure_ml import (
    SCOREBOOK_MOMENTUM,
    allocate_positive_score_book,
    skill_allows_book,
    spearman_ic,
    walk_forward_filing_proposals,
)


ATTENTION_LEDGER_RELATIVE = "results/barebone_attention_arm_ledger.json"
ATTENTION_METRICS_RELATIVE = "results/barebone_attention_arm_metrics.json"
HYBRID_LEDGER_RELATIVE = "results/barebone_hybrid_arm_ledger.json"
HYBRID_METRICS_RELATIVE = "results/barebone_hybrid_arm_metrics.json"
INDEX_RELATIVE = "results/barebone_attention_hybrid.json"
_ATTENTION_LEDGER_SCHEMA = "fusionfinance-barebone-attention-arm-ledger-v1"
_ATTENTION_METRICS_SCHEMA = "fusionfinance-barebone-attention-arm-metrics-v1"
_HYBRID_LEDGER_SCHEMA = "fusionfinance-barebone-hybrid-arm-ledger-v1"
_HYBRID_METRICS_SCHEMA = "fusionfinance-barebone-hybrid-arm-metrics-v1"
_INDEX_SCHEMA = "fusionfinance-barebone-attention-hybrid-v1"
_HYBRID_PROVENANCE_SCHEMA = "fusionfinance-barebone-hybrid-v1"
_CALIBRATION_RELATIVE = "results/fusion_policy_calibration.json"
_HYBRID_RULE = "momentum_skill_pass AND attention_skill_pass"
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


def raw_log1p_by_session(
    rows: Sequence[Mapping[str, object]],
) -> dict[date, dict[str, float]]:
    """``log1p`` counts at each scored session. The lexicon is not a column."""

    grouped: dict[date, dict[str, float]] = {}
    for row in rows:
        if row.get("signal_id") != ATTENTION_SIGNAL_ID:
            raise ValueError("attention row is not the locked signal")
        if int(row["lookback_sessions"]) != ATTENTION_LOOKBACK_SESSIONS:
            raise ValueError("attention lookback is locked at 21 sessions")
        if "text" in row or "polarity" in row or _PERFORMANCE_KEYS.intersection(row):
            raise ValueError("attention row carries text, polarity, or a return label")
        session = date.fromisoformat(str(row["session"]))
        grouped.setdefault(session, {})[str(row["ticker"])] = float(row["log1p_count"])
    return grouped


def attention_oos_pairs(
    raw_by_session: Mapping[date, Mapping[str, float]],
    closes: Mapping[tuple[str, int], float],
    dates: Sequence[date],
) -> list[tuple[int, float, float]]:
    """Demeaned attention at session ``j`` versus the residual at ``j + 1``.

    The count window ends at ``j``. The residual is the skill label. A pair
    is usable at a later decision only after ``j + 1`` is already past.
    """

    index = {day: offset for offset, day in enumerate(dates)}
    pairs: list[tuple[int, float, float]] = []
    for day, raw in raw_by_session.items():
        if day not in index:
            raise ValueError(f"attention session {day.isoformat()} is not on the tape")
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
    """Spearman of pairs whose outcome session is already before the decision."""

    usable = [pair for pair in pairs if pair[0] < decision_index]
    skill = spearman_ic([pair[1] for pair in usable], [pair[2] for pair in usable])
    return skill, len(usable)


def attention_book(
    raw: Mapping[str, float],
    skill: float | None,
    config: ExperimentConfig,
    previous: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Positive demeaned attention, or cash when skill is not strictly above zero."""

    if not skill_allows_book(skill, ATTENTION_SKILL_THRESHOLD):
        return {}
    scores = demean_cross_section(raw)
    if len(scores) < 2:
        return {}
    return allocate_positive_score_book(dict(scores), config, dict(previous or {}))


def attention_rebalances(
    raw_by_session: Mapping[date, Mapping[str, float]],
    closes: Mapping[tuple[str, int], float],
    dates: Sequence[date],
    config: ExperimentConfig,
) -> list[dict[str, object]]:
    """One row per shared rebalance. Non-positive skill is an empty book."""

    pairs = attention_oos_pairs(raw_by_session, closes, dates)
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
        passed = skill_allows_book(skill, ATTENTION_SKILL_THRESHOLD)
        book = attention_book(raw, skill, config, previous)
        scores = demean_cross_section(raw) if book else {}
        rows.append(
            {
                "decision_session": dates[index].isoformat(),
                "oos_skill": skill,
                "oos_skill_pairs": pair_count,
                "oos_skill_threshold": ATTENTION_SKILL_THRESHOLD,
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


def hybrid_book(
    momentum_scores: Mapping[str, float],
    momentum_pass: bool,
    attention_pass: bool,
    config: ExperimentConfig,
    previous: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Momentum book only when both skill gates pass. Otherwise cash."""

    if not momentum_pass or not attention_pass:
        return {}
    if len(momentum_scores) < 2:
        return {}
    return allocate_positive_score_book(dict(momentum_scores), config, dict(previous or {}))


def _momentum_scores(proposal: Mapping[str, object]) -> dict[str, float]:
    cross = proposal.get("cross_section")
    if not isinstance(cross, list):
        raise ValueError("momentum proposal is missing a cross section")
    scores: dict[str, float] = {}
    for item in cross:
        if not isinstance(item, Mapping):
            raise ValueError("momentum cross section row must be an object")
        scores[str(item["ticker"])] = float(item["score"])
    return scores


def hybrid_rebalances(
    attention_rows: Sequence[Mapping[str, object]],
    momentum_proposals: Sequence[Mapping[str, object]],
    config: ExperimentConfig,
) -> list[dict[str, object]]:
    """Intersection gate. The sized scores are momentum, not attention or polarity."""

    attention = {str(row["decision_session"]): row for row in attention_rows}
    momentum = {str(row["decision_session"]): row for row in momentum_proposals}
    if set(attention) != set(momentum):
        raise ValueError("hybrid rebalance dates do not match the momentum clock")
    rows: list[dict[str, object]] = []
    previous: dict[str, float] = {}
    for day in sorted(attention):
        attention_row = attention[day]
        momentum_row = momentum[day]
        attention_pass = bool(attention_row["skill_pass"])
        momentum_pass = bool(momentum_row.get("skill_pass")) and momentum_row.get("insufficient_history") is not True
        scores = _momentum_scores(momentum_row)
        book = hybrid_book(scores, momentum_pass, attention_pass, config, previous)
        rows.append(
            {
                "decision_session": day,
                "momentum_oos_skill": momentum_row.get("oos_skill"),
                "attention_oos_skill": attention_row.get("oos_skill"),
                "momentum_skill_pass": momentum_pass,
                "attention_skill_pass": attention_pass,
                "skill_pass": momentum_pass and attention_pass,
                "sized": bool(book),
                "book": book,
                "scores": scores if book else {},
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


def _load_attention_rows(path: Path, digest: str) -> list[dict[str, object]]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("attention scores bytes do not match the locked attention_sha256")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("attention scores file is empty")
    for row in rows:
        if row.get("signal_id") != ATTENTION_SIGNAL_ID:
            raise ValueError("attention score is not the locked signal")
        if "text" in row or "polarity" in row or _PERFORMANCE_KEYS.intersection(row):
            raise ValueError("attention score carries text, polarity, or a return label")
    return rows


def _arm_inputs(
    rows: Sequence[Mapping[str, object]],
    config: ExperimentConfig,
    strategy_id: str,
) -> list[ArmInput]:
    inputs: list[ArmInput] = []
    cash_ticker = config.universe[0]
    for row in rows:
        day = date.fromisoformat(str(row["decision_session"]))
        book = row["book"]
        if not isinstance(book, dict) or not book:
            inputs.append(
                ArmInput(
                    strategy_id=strategy_id,
                    decision_session=day,
                    ticker=cash_ticker,
                    structured_weight=0.0,
                )
            )
            continue
        for ticker, weight in sorted(book.items()):
            inputs.append(
                ArmInput(
                    strategy_id=strategy_id,
                    decision_session=day,
                    ticker=str(ticker),
                    structured_weight=float(weight),
                )
            )
    return inputs


def _refuse_attention_without_skill(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        skill = row["oos_skill"]
        if row["sized"] and not skill_allows_book(
            None if skill is None else float(skill), ATTENTION_SKILL_THRESHOLD
        ):
            raise ValueError("attention sized a book when OOS skill failed the gate")


def _refuse_hybrid_without_both_gates(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        if row["sized"] and not (row["momentum_skill_pass"] and row["attention_skill_pass"]):
            raise ValueError("hybrid sized a book outside the skill-pass intersection")


def _marks(market: BareboneMarket, trading: ExperimentConfig):
    return benchmark_marks_from_closes(
        market.sessions,
        benchmark_ticker=trading.benchmark_ticker,
        starting_capital=trading.starting_capital,
    )


def attention_hybrid_state(root: Path | None = None) -> dict[str, object]:
    """Load the locked tape and attention counts and execute both arms."""

    base = _repo_root() if root is None else root
    cached = _STATE.get(str(base.resolve()))
    if cached is not None:
        return cached
    market = load_barebone_market(base)
    config = market.config
    if config.scorebook != SCOREBOOK_MOMENTUM:
        raise ValueError("hybrid momentum control requires the momentum scorebook")
    if config.evidence.narrative_sha256 != LOCKED_NARRATIVE_SHA256:
        raise ValueError("narrative_sha256 is locked; refusing a new digest without a new experiment_id")
    if config.evidence.tape_sha256 != LOCKED_TAPE_SHA256 or market.tape_sha256 != LOCKED_TAPE_SHA256:
        raise ValueError("tape_sha256 is locked; refusing a new digest without a new experiment_id")
    if config.evidence.polarity_sha256 != LOCKED_POLARITY_SHA256:
        raise ValueError("polarity_sha256 is locked; refusing a lexicon retune")
    lexicon_sha = hashlib.sha256((base / NARRATIVE_LEXICON).read_bytes()).hexdigest()
    if lexicon_sha != FROZEN_LEXICON_SHA256:
        raise ValueError("polarity lexicon bytes changed; refusing a retune")
    attention_sha = config.evidence.attention_sha256
    if attention_sha is None:
        raise ValueError("barebone attention arm refuses an unlocked attention_sha256")
    events_path = base / NARRATIVE_EVENTS
    if hashlib.sha256(events_path.read_bytes()).hexdigest() != LOCKED_NARRATIVE_SHA256:
        raise ValueError("narrative events bytes do not match the locked narrative_sha256")
    provenance = json.loads((base / ATTENTION_PROVENANCE).read_text(encoding="utf-8"))
    if provenance.get("attention_sha256") != attention_sha:
        raise ValueError("attention provenance does not match the locked attention_sha256")
    if provenance.get("narrative_events_sha256") != LOCKED_NARRATIVE_SHA256:
        raise ValueError("attention provenance does not match the locked narrative_sha256")
    if provenance.get("tape_sha256") != LOCKED_TAPE_SHA256:
        raise ValueError("attention provenance does not match the locked tape_sha256")
    if provenance.get("signal_id") != ATTENTION_SIGNAL_ID:
        raise ValueError("attention provenance is not the locked signal")
    if provenance.get("lookback_sessions") != ATTENTION_LOOKBACK_SESSIONS:
        raise ValueError("attention lookback is locked at 21 sessions")
    if provenance.get("lexicon_used_for_sizing") is not False:
        raise ValueError("attention sizing must not use the polarity lexicon")
    if provenance.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    if "text" in provenance:
        raise ValueError("attention provenance carries story text")
    rows = _load_attention_rows(base / ATTENTION_SCORES, attention_sha)
    trading = config.as_experiment_config()
    dates = tuple(session.session for session in market.sessions)
    panel_dates = tuple(sorted({pd.Timestamp(value).date() for value in market.panel["date"]}))
    if panel_dates != dates:
        raise ValueError("attention panel calendar does not match the tape")
    raw = raw_log1p_by_session(rows)
    attention_rows = attention_rebalances(raw, _closes(market.panel), dates, trading)
    _refuse_attention_without_skill(attention_rows)
    clock = [
        index
        for index in range(len(dates))
        if index % trading.rebalance_frequency_sessions == 0
        and index + trading.execution_lag_sessions < len(dates)
    ]
    manifest = walk_forward_filing_proposals(
        trading,
        tuple(dates[index] for index in clock),
        root=base,
        panel=market.panel,
        tape_dates=dates,
        cash_when_untrained=True,
        scorebook=SCOREBOOK_MOMENTUM,
    )
    hybrid_rows = hybrid_rebalances(attention_rows, manifest["proposals"], trading)
    _refuse_hybrid_without_both_gates(hybrid_rows)
    thresholds = load_policy_thresholds(base / _CALIBRATION_RELATIVE)
    marks = _marks(market, trading)
    attention_run = run_controlled_arm(
        config=trading,
        sessions=market.sessions,
        strategy_id="attention",
        candidates=_arm_inputs(attention_rows, trading, "attention"),
        benchmark_marks=marks,
        thresholds=thresholds,
    )
    hybrid_run = run_controlled_arm(
        config=trading,
        sessions=market.sessions,
        strategy_id="hybrid",
        candidates=_arm_inputs(hybrid_rows, trading, "hybrid"),
        benchmark_marks=marks,
        thresholds=thresholds,
    )
    if attention_run.block_reason is not None or attention_run.ledger is None or attention_run.metrics is None:
        raise ValueError(f"attention arm blocked: {attention_run.block_reason}")
    if hybrid_run.block_reason is not None or hybrid_run.ledger is None or hybrid_run.metrics is None:
        raise ValueError(f"hybrid arm blocked: {hybrid_run.block_reason}")
    state = {
        "market": market,
        "trading": trading,
        "dates": dates,
        "attention_rows": attention_rows,
        "hybrid_rows": hybrid_rows,
        "attention_run": attention_run,
        "hybrid_run": hybrid_run,
        "attention_sha256": attention_sha,
        "provenance": provenance,
        "event_count": int(provenance["event_count"]),
    }
    _STATE[str(base.resolve())] = state
    return state


def _attention_skill_log(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
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


def _hybrid_skill_log(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "decision_session": row["decision_session"],
            "momentum_oos_skill": row["momentum_oos_skill"],
            "attention_oos_skill": row["attention_oos_skill"],
            "momentum_skill_pass": row["momentum_skill_pass"],
            "attention_skill_pass": row["attention_skill_pass"],
            "skill_pass": row["skill_pass"],
            "sized": row["sized"],
        }
        for row in rows
    ]


def _decisions(
    run,
    rows: Sequence[Mapping[str, object]],
    *,
    hybrid: bool,
) -> list[dict[str, object]]:
    by_session = {str(row["decision_session"]): row for row in rows}
    decisions: list[dict[str, object]] = []
    for proposal in run.lineage.proposals:
        row = by_session[proposal.decision_session.isoformat()]
        if proposal.target_weight != 0.0 and not row["sized"]:
            raise ValueError("sized a book when the skill gate was cash")
        if hybrid and proposal.target_weight != 0.0 and not (
            row["momentum_skill_pass"] and row["attention_skill_pass"]
        ):
            raise ValueError("hybrid sized a book outside the skill-pass intersection")
        if not hybrid and proposal.target_weight != 0.0 and (
            row["oos_skill"] is None or float(row["oos_skill"]) <= ATTENTION_SKILL_THRESHOLD
        ):
            raise ValueError("attention sized a book when OOS skill is not strictly positive")
        item: dict[str, object] = {
            "decision_session": proposal.decision_session.isoformat(),
            "ticker": proposal.ticker,
            "admitted": proposal.admitted,
            "target_weight": proposal.target_weight,
            "reason": proposal.reason,
            "model_score": None
            if not isinstance(row["scores"], dict)
            else row["scores"].get(proposal.ticker),
        }
        if hybrid:
            item["momentum_skill_pass"] = row["momentum_skill_pass"]
            item["attention_skill_pass"] = row["attention_skill_pass"]
            item["momentum_oos_skill"] = row["momentum_oos_skill"]
            item["attention_oos_skill"] = row["attention_oos_skill"]
        else:
            item["oos_skill"] = row["oos_skill"]
            item["oos_skill_pairs"] = row["oos_skill_pairs"]
            item["oos_skill_threshold"] = row["oos_skill_threshold"]
        decisions.append(item)
    return decisions


def _ledger_shell(
    *,
    schema: str,
    strategy_id: str,
    description: str,
    state: Mapping[str, object],
    run,
    rebalances: Sequence[Mapping[str, object]],
    skill_log: list[dict[str, object]],
    model_binding: dict[str, object],
    hybrid: bool,
) -> dict[str, object]:
    market: BareboneMarket = state["market"]
    trading: ExperimentConfig = state["trading"]
    ledger = run.ledger
    if ledger is None:
        raise ValueError(f"{strategy_id} arm is missing a ledger")
    document: dict[str, object] = {
        "schema": schema,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "claim_status": "controlled_software_ledger",
        "comparable_performance_claim": False,
        "description": description,
        "strategy_id": strategy_id,
        "signal_id": HYBRID_SIGNAL_ID if hybrid else ATTENTION_SIGNAL_ID,
        "tape_sha256": market.tape_sha256,
        "narrative_sha256": LOCKED_NARRATIVE_SHA256,
        "attention_sha256": state["attention_sha256"],
        "polarity_sha256": LOCKED_POLARITY_SHA256,
        "lexicon_used_for_sizing": False,
        "window": [trading.start_date.isoformat(), trading.end_date.isoformat()],
        "session_count": len(state["dates"]),
        "rebalance_count": len(rebalances),
        "sized_rebalances": sum(1 for row in rebalances if row["sized"]),
        "cash_rebalances": sum(1 for row in rebalances if not row["sized"]),
        "max_position_weight": trading.max_position_weight,
        "max_gross_leverage": trading.max_gross_leverage,
        "model_binding": model_binding,
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
        "oos_skill": skill_log,
        "decisions": _decisions(run, rebalances, hybrid=hybrid),
    }
    if document["comparable_performance_claim"] is not False:
        raise ValueError("comparable_performance_claim must be false")
    if _PERFORMANCE_KEYS.intersection(_mapping_keys(document)):
        raise ValueError(f"{strategy_id} ledger must not carry performance-claim fields")
    refuse_fair_race_as_barebone_comparison(document)
    return document


def barebone_attention_ledger(
    root: Path | None = None, *, bound: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Software ledger for the attention arm. It does not carry a Sharpe."""

    state = attention_hybrid_state(root) if bound is None else bound
    rebalances: list[dict[str, object]] = state["attention_rows"]
    return _ledger_shell(
        schema=_ATTENTION_LEDGER_SCHEMA,
        strategy_id="attention",
        description=(
            "Software ledger for the barebone attention arm on the locked "
            "Hacker News events and the locked Yahoo tape. The score is "
            "log1p of the mapped event count in the 21 sessions ending at "
            "the decision session, minus the cross-sectional median. The "
            "book is sized only when expanding Spearman skill versus the "
            "next-session residual is strictly above zero; otherwise it is "
            "cash. The polarity lexicon is not used. Position and gross caps "
            "stay 0.10 and 1.0. This is not a performance claim."
        ),
        state=state,
        run=state["attention_run"],
        rebalances=rebalances,
        skill_log=_attention_skill_log(rebalances),
        model_binding={
            "signal_id": ATTENTION_SIGNAL_ID,
            "lookback_sessions": ATTENTION_LOOKBACK_SESSIONS,
            "transform": ATTENTION_TRANSFORM,
            "oos_skill_threshold": ATTENTION_SKILL_THRESHOLD,
            "lexicon_used_for_sizing": False,
            "attention_sha256": state["attention_sha256"],
            "narrative_sha256": LOCKED_NARRATIVE_SHA256,
            "tape_sha256": LOCKED_TAPE_SHA256,
        },
        hybrid=False,
    )


def barebone_hybrid_ledger(
    root: Path | None = None, *, bound: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Software ledger for the intersection hybrid. It does not carry a Sharpe."""

    state = attention_hybrid_state(root) if bound is None else bound
    rebalances: list[dict[str, object]] = state["hybrid_rows"]
    return _ledger_shell(
        schema=_HYBRID_LEDGER_SCHEMA,
        strategy_id="hybrid",
        description=(
            "Software ledger for the barebone hybrid arm. The book is cash "
            "unless the momentum skill gate and the attention skill gate both "
            "pass. When both pass, weights come from the momentum scores and "
            "the hybrid previous book. Attention is a gate. The polarity "
            "lexicon is not used. Position and gross caps stay 0.10 and 1.0. "
            "This is not a performance claim."
        ),
        state=state,
        run=state["hybrid_run"],
        rebalances=rebalances,
        skill_log=_hybrid_skill_log(rebalances),
        model_binding={
            "signal_id": HYBRID_SIGNAL_ID,
            "rule": _HYBRID_RULE,
            "sized_score": "momentum",
            "attention_signal_id": ATTENTION_SIGNAL_ID,
            "attention_lookback_sessions": ATTENTION_LOOKBACK_SESSIONS,
            "lexicon_used_for_sizing": False,
            "attention_sha256": state["attention_sha256"],
            "narrative_sha256": LOCKED_NARRATIVE_SHA256,
            "tape_sha256": LOCKED_TAPE_SHA256,
        },
        hybrid=True,
    )


def _metrics_shell(
    *,
    schema: str,
    strategy_id: str,
    description: str,
    source_ledger: str,
    state: Mapping[str, object],
    run,
    skill_log: list[dict[str, object]],
    hybrid: bool,
) -> dict[str, object]:
    market: BareboneMarket = state["market"]
    trading: ExperimentConfig = state["trading"]
    if run.result is None or run.metrics is None or run.ledger is None:
        raise ValueError(f"{strategy_id} arm is missing an executed result")
    statistics = run.metrics.model_dump(mode="json")
    values = [point.portfolio_value for point in run.result.points]
    if not math.isfinite(float(statistics["total_return"])):
        raise ValueError(f"{strategy_id} return is not finite")
    if not math.isclose(float(statistics["total_return"]), values[-1] / values[0] - 1.0):
        raise ValueError("cumulative return does not match the wealth path")
    document: dict[str, object] = {
        "schema": schema,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "claim_status": "controlled_software_fixture_metrics",
        "comparable_performance_claim": False,
        "description": description,
        "strategy_id": strategy_id,
        "signal_id": HYBRID_SIGNAL_ID if hybrid else ATTENTION_SIGNAL_ID,
        "tape_sha256": market.tape_sha256,
        "narrative_sha256": LOCKED_NARRATIVE_SHA256,
        "attention_sha256": state["attention_sha256"],
        "polarity_sha256": LOCKED_POLARITY_SHA256,
        "lexicon_used_for_sizing": False,
        "source_ledger": source_ledger,
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
        "oos_skill": skill_log,
    }
    if document["comparable_performance_claim"] is not False:
        raise ValueError("comparable_performance_claim must be false")
    refuse_fair_race_as_barebone_comparison(document)
    return document


def barebone_attention_metrics(
    root: Path | None = None, *, bound: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Fixture metrics for the attention arm. The claim stays false."""

    state = attention_hybrid_state(root) if bound is None else bound
    return _metrics_shell(
        schema=_ATTENTION_METRICS_SCHEMA,
        strategy_id="attention",
        description=(
            "Fixture metrics for the barebone attention arm. The score is "
            "log1p event counts over 21 sessions, demeaned in the cross "
            "section. The book is cash unless expanding Spearman skill is "
            "strictly above zero. The polarity lexicon is not used. Not a "
            "capital performance claim."
        ),
        source_ledger=ATTENTION_LEDGER_RELATIVE,
        state=state,
        run=state["attention_run"],
        skill_log=_attention_skill_log(state["attention_rows"]),
        hybrid=False,
    )


def barebone_hybrid_metrics(
    root: Path | None = None, *, bound: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Fixture metrics for the intersection hybrid. The claim stays false."""

    state = attention_hybrid_state(root) if bound is None else bound
    return _metrics_shell(
        schema=_HYBRID_METRICS_SCHEMA,
        strategy_id="hybrid",
        description=(
            "Fixture metrics for the barebone hybrid arm. The book is cash "
            "unless momentum skill and attention skill are both strictly "
            "above zero. Sized weights are momentum scores. The polarity "
            "lexicon is not used. Not a capital performance claim."
        ),
        source_ledger=HYBRID_LEDGER_RELATIVE,
        state=state,
        run=state["hybrid_run"],
        skill_log=_hybrid_skill_log(state["hybrid_rows"]),
        hybrid=True,
    )


def barebone_attention_hybrid_index(
    root: Path | None = None, *, bound: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Pointer at both arms. It does not publish a Sharpe."""

    state = attention_hybrid_state(root) if bound is None else bound
    attention_rows: list[dict[str, object]] = state["attention_rows"]
    hybrid_rows: list[dict[str, object]] = state["hybrid_rows"]
    document: dict[str, object] = {
        "schema": _INDEX_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "claim_status": "controlled_software_ledger",
        "comparable_performance_claim": False,
        "attention_signal_id": ATTENTION_SIGNAL_ID,
        "hybrid_signal_id": HYBRID_SIGNAL_ID,
        "hybrid_rule": _HYBRID_RULE,
        "narrative_sha256": LOCKED_NARRATIVE_SHA256,
        "narrative_events": NARRATIVE_EVENTS,
        "attention_sha256": state["attention_sha256"],
        "attention_scores": ATTENTION_SCORES,
        "tape_sha256": state["market"].tape_sha256,
        "polarity_sha256": LOCKED_POLARITY_SHA256,
        "lexicon_used_for_sizing": False,
        "event_count": state["event_count"],
        "attention_sized_rebalances": sum(1 for row in attention_rows if row["sized"]),
        "attention_cash_rebalances": sum(1 for row in attention_rows if not row["sized"]),
        "hybrid_sized_rebalances": sum(1 for row in hybrid_rows if row["sized"]),
        "hybrid_cash_rebalances": sum(1 for row in hybrid_rows if not row["sized"]),
        "max_position_weight": 0.1,
        "max_gross_leverage": 1.0,
        "attention_ledger": ATTENTION_LEDGER_RELATIVE,
        "attention_metrics": ATTENTION_METRICS_RELATIVE,
        "hybrid_ledger": HYBRID_LEDGER_RELATIVE,
        "hybrid_metrics": HYBRID_METRICS_RELATIVE,
    }
    if _PERFORMANCE_KEYS.intersection(document):
        raise ValueError("attention hybrid index must not carry a performance field")
    refuse_fair_race_as_barebone_comparison(document)
    return document


def hybrid_provenance(
    root: Path | None = None, *, bound: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Committed hybrid note. No story text and no prices."""

    state = attention_hybrid_state(root) if bound is None else bound
    document: dict[str, object] = {
        "schema": _HYBRID_PROVENANCE_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "signal_id": HYBRID_SIGNAL_ID,
        "rule": _HYBRID_RULE,
        "sized_score": "momentum",
        "provider": "momentum skill gate intersected with Hacker News attention counts; lexicon not used",
        "momentum_scorebook": SCOREBOOK_MOMENTUM,
        "attention_signal_id": ATTENTION_SIGNAL_ID,
        "attention_lookback_sessions": ATTENTION_LOOKBACK_SESSIONS,
        "attention_transform": ATTENTION_TRANSFORM,
        "attention_sha256": state["attention_sha256"],
        "narrative_sha256": LOCKED_NARRATIVE_SHA256,
        "tape_sha256": LOCKED_TAPE_SHA256,
        "polarity_sha256": LOCKED_POLARITY_SHA256,
        "lexicon_used_for_sizing": False,
        "license_note": "not redistributed; local bind only",
        "comparable_performance_claim": False,
    }
    refuse_fair_race_as_barebone_comparison(document)
    return document


def write_barebone_attention_artifacts(
    root: Path | None = None, *, bound: Mapping[str, object] | None = None
) -> tuple[Path, Path, Path, Path, Path]:
    """Write both ledgers, both metrics files, and the index. Does not write dumps."""

    base = _repo_root() if root is None else root
    state = attention_hybrid_state(base) if bound is None else bound
    attention_ledger = barebone_attention_ledger(base, bound=state)
    attention_metrics = barebone_attention_metrics(base, bound=state)
    hybrid_ledger = barebone_hybrid_ledger(base, bound=state)
    hybrid_metrics = barebone_hybrid_metrics(base, bound=state)
    index = barebone_attention_hybrid_index(base, bound=state)
    provenance = hybrid_provenance(base, bound=state)
    paths = {
        ATTENTION_LEDGER_RELATIVE: attention_ledger,
        ATTENTION_METRICS_RELATIVE: attention_metrics,
        HYBRID_LEDGER_RELATIVE: hybrid_ledger,
        HYBRID_METRICS_RELATIVE: hybrid_metrics,
        INDEX_RELATIVE: index,
    }
    written: list[Path] = []
    for relative, payload in paths.items():
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json_text(payload), encoding="utf-8")
        written.append(path)
    provenance_path = base / HYBRID_PROVENANCE
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(_json_text(provenance), encoding="utf-8")
    return written[0], written[1], written[2], written[3], written[4]


def main(argv: list[str] | None = None) -> int:
    """Execute the locked attention and hybrid arms. Does not call a live model."""

    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        attention_ledger_path, attention_metrics_path, hybrid_ledger_path, hybrid_metrics_path, _index = (
            write_barebone_attention_artifacts()
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    attention_ledger = json.loads(attention_ledger_path.read_text(encoding="utf-8"))
    attention_metrics = json.loads(attention_metrics_path.read_text(encoding="utf-8"))
    hybrid_ledger = json.loads(hybrid_ledger_path.read_text(encoding="utf-8"))
    hybrid_metrics = json.loads(hybrid_metrics_path.read_text(encoding="utf-8"))
    print(f"signal_id={ATTENTION_SIGNAL_ID}")
    print(f"hybrid_signal_id={HYBRID_SIGNAL_ID}")
    print(f"attention_sha256={attention_ledger['attention_sha256']}")
    print(f"attention_sized_rebalances={attention_ledger['sized_rebalances']}")
    print(f"attention_cash_rebalances={attention_ledger['cash_rebalances']}")
    print(f"attention_total_return={attention_metrics['statistics']['total_return']}")
    print(f"attention_sharpe_ratio={attention_metrics['statistics']['sharpe_ratio']}")
    print(f"hybrid_sized_rebalances={hybrid_ledger['sized_rebalances']}")
    print(f"hybrid_cash_rebalances={hybrid_ledger['cash_rebalances']}")
    print(f"hybrid_total_return={hybrid_metrics['statistics']['total_return']}")
    print(f"hybrid_sharpe_ratio={hybrid_metrics['statistics']['sharpe_ratio']}")
    print(f"attention_ledger={attention_ledger_path}")
    print(f"attention_metrics={attention_metrics_path}")
    print(f"hybrid_ledger={hybrid_ledger_path}")
    print(f"hybrid_metrics={hybrid_metrics_path}")
    return 0
