"""Controlled capital path: precheck, market verifier, risk, shared execution.

A sealed ``agent_evidence_precheck`` is not trading authorization. This module
is the downstream gate. It records proposal lineage and config, tape, and
experiment hashes, then executes only weights that survive the arm's gate and
the locked risk limits.

The resulting ledger is a software record. It is not a comparative performance
claim, and it does not relabel ``provisional_uncontrolled_legacy_race``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from numbers import Real
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alpha.agents.models import DecisionReceipt, EvidenceAuditSummary
from alpha.verifier.contract import ThesisContract, VerifierOutput
from alpha.verifier.policy import PolicyThresholds, adjudicate
from demo.contracts import (
    BenchmarkMark,
    ExperimentConfig,
    LedgerReconciliation,
    MarketSession,
    PerformanceMetrics,
    PositionTarget,
    SimulationResult,
    WeightProposal,
)
from demo.execution import reconcile_simulation, session_window, simulate_portfolio
from demo.metrics import compute_performance_metrics

ARM_IDS = ("pure_ml", "pure_llm", "fusion")
_EXTRA_STRUCTURED_ARMS = frozenset({"narrative", "attention", "hybrid", "edgar", "form4"})
CLAIM_STATUS = "controlled_software_ledger"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class MarketVerification(_FrozenModel):
    """Numeric market-head forecast. Thesis prose is not an input."""

    produced_at: str
    expected_residual_bps: dict[str, float]
    p_adverse: dict[str, float]
    out_of_distribution_score: float = Field(ge=0.0, le=1.0)
    calibration_hash: str = ""
    epistemic_var: dict[str, float] = Field(default_factory=dict)
    aleatoric_var: dict[str, float] = Field(default_factory=dict)
    epistemic_mi: dict[str, float] = Field(default_factory=dict)
    fundamental_confirm_prob: float | None = None

    @field_validator("produced_at")
    @classmethod
    def _produced_at(cls, value: str) -> str:
        return _aware_timestamp(value)

    @field_validator("out_of_distribution_score", "fundamental_confirm_prob", mode="before")
    @classmethod
    def _real(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("market forecast values must be real numbers")
        return float(value)


class ArmInput(_FrozenModel):
    """One name at one decision session for one strategy arm."""

    strategy_id: Literal["pure_ml", "pure_llm", "fusion", "narrative", "attention", "hybrid", "edgar", "form4"]
    decision_session: date
    ticker: str = Field(min_length=1)
    receipt: DecisionReceipt | None = None
    market: MarketVerification | None = None
    structured_weight: float | None = None
    outcome_ts: str | None = None

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker must not be blank")
        return normalized

    @field_validator("structured_weight", mode="before")
    @classmethod
    def _weight(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("structured weight must be a real number")
        return float(value)

    @field_validator("outcome_ts")
    @classmethod
    def _outcome(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _aware_timestamp(value)


class ProposalLineage(_FrozenModel):
    strategy_id: str
    decision_session: date
    ticker: str
    proposal_hash: str | None = None
    snapshot_hash: str | None = None
    thesis_hash: str | None = None
    receipt_hash: str | None = None
    market_input_hash: str | None = None
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    precheck_decision: str | None = None
    verifier_decision: str | None = None
    admitted: bool
    target_weight: float
    reason: str = Field(min_length=1)


class RunLineage(_FrozenModel):
    strategy_id: str
    experiment_id: str
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    tape_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposals: tuple[ProposalLineage, ...] = ()


class ControlledRun(_FrozenModel):
    claim_status: Literal["controlled_software_ledger"] = CLAIM_STATUS
    comparable_performance_claim: Literal[False] = False
    lineage: RunLineage
    ledger: LedgerReconciliation | None = None
    result: SimulationResult | None = None
    metrics: PerformanceMetrics | None = None
    block_reason: str | None = None
    candidate_count: int = Field(ge=0)
    admitted_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _Gate:
    ticker: str
    decision_session: date
    weight: float
    reason: str
    proposal_hash: str | None
    snapshot_hash: str | None
    thesis_hash: str | None
    receipt_hash: str | None
    market_input_hash: str | None
    precheck_decision: str | None
    verifier_decision: str | None
    structured_weight: float | None
    outcome_ts: str | None


def locked_weekday_sessions(config: ExperimentConfig) -> tuple[date, ...]:
    """Every weekday from the locked start through the locked end, inclusive."""

    days: list[date] = []
    current = config.start_date
    while current <= config.end_date:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    if len(days) < 2 or days[0] != config.start_date or days[-1] != config.end_date:
        raise ValueError("locked window must start and end on weekday sessions")
    return tuple(days)


def load_policy_thresholds(path: str | Path | None = None) -> PolicyThresholds:
    """Return calibrated thresholds only after the artifact verifies.

    Verification binds the hash to the receipt and requires the 10-day horizon.
    A missing market head still cannot be approved.
    """

    artifact_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[1] / "results" / "fusion_policy_calibration.json"
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    receipt_json = payload.get("receipt_json")
    artifact_hash = payload.get("artifact_hash")
    if not isinstance(receipt_json, str) or not isinstance(artifact_hash, str):
        raise ValueError("calibration artifact is incomplete")
    receipt = json.loads(receipt_json)
    from alpha.verifier.calibration import AffineLogitCalibrator

    parameters = tuple(
        (str(horizon), float(slope), float(intercept))
        for horizon, slope, intercept in receipt["parameters"]
    )
    horizons = tuple(str(horizon) for horizon in receipt["rows"])
    calibrator = AffineLogitCalibrator(
        parameters=parameters,
        device="cpu",
        strict_gpu=False,
        fitted_horizons=horizons,
        artifact_hash=artifact_hash,
        receipt_json=receipt_json,
    )
    calibrator.verify_artifact({"10d"})
    return PolicyThresholds(
        calibrated=True,
        calibration_hash=calibrator.artifact_hash,
        horizon_key="10d",
    )


def load_locked_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load the frozen three-arm contract, defaulting to the checked-in file."""

    config_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[1] / "configs" / "fusionfinance-demo.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(payload)


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def config_hash(config: ExperimentConfig) -> str:
    return canonical_hash(config.model_dump(mode="json"))


def tape_hash(sessions: Sequence[MarketSession]) -> str:
    return canonical_hash([session.model_dump(mode="json") for session in sessions])


def experiment_hash(
    *,
    strategy_id: str,
    config_hash: str,
    tape_hash: str,
    lineage_hash: str,
) -> str:
    return canonical_hash(
        {
            "schema": "fusionfinance-controlled-run-v1",
            "strategy_id": strategy_id,
            "config_hash": config_hash,
            "tape_hash": tape_hash,
            "lineage_hash": lineage_hash,
        }
    )


def benchmark_marks_from_closes(
    sessions: Sequence[MarketSession],
    *,
    benchmark_ticker: str,
    starting_capital: float,
) -> tuple[BenchmarkMark, ...]:
    """Price-only benchmark wealth indexed by the tape session dates."""

    closes: list[float] = []
    for session in sessions:
        bars = {bar.ticker: bar for bar in session.bars}
        bar = bars.get(benchmark_ticker)
        if bar is None:
            raise ValueError(
                f"market session {session.session} missing benchmark {benchmark_ticker}"
            )
        closes.append(bar.close)
    if not closes or closes[0] <= 0.0:
        raise ValueError("benchmark close must be positive")
    base = closes[0]
    return tuple(
        BenchmarkMark(
            session=session.session,
            value=starting_capital * close / base,
        )
        for session, close in zip(sessions, closes, strict=True)
    )


def estimated_post_cost_gross_leverage(
    targets: Mapping[str, float],
    previous: Mapping[str, float],
    config: ExperimentConfig,
) -> float:
    """Gross exposure after cost drag, assuming target weights are achieved."""

    names = set(targets) | set(previous)
    turnover = sum(
        abs(targets.get(name, 0.0) - previous.get(name, 0.0)) for name in names
    )
    gross = sum(abs(weight) for weight in targets.values())
    rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000.0
    cost_fraction = turnover * rate
    if cost_fraction >= 1.0:
        return math.inf
    if gross == 0.0:
        return 0.0
    return gross / (1.0 - cost_fraction)


def run_three_arms(
    *,
    config: ExperimentConfig,
    sessions: Sequence[MarketSession],
    arms: Mapping[str, Sequence[ArmInput]],
    benchmark_marks: Sequence[BenchmarkMark | Mapping[str, object]],
    thresholds: PolicyThresholds | None = None,
) -> tuple[ControlledRun, ControlledRun, ControlledRun]:
    """Run pure_ml, pure_llm, and fusion through one config and one tape."""

    if set(arms) != set(ARM_IDS):
        raise ValueError("three-arm run requires pure_ml, pure_llm, and fusion")
    return tuple(
        run_controlled_arm(
            config=config,
            sessions=sessions,
            strategy_id=strategy_id,
            candidates=arms[strategy_id],
            benchmark_marks=benchmark_marks,
            thresholds=thresholds,
        )
        for strategy_id in ARM_IDS
    )


def run_controlled_arm(
    *,
    config: ExperimentConfig,
    sessions: Sequence[MarketSession],
    strategy_id: str,
    candidates: Sequence[ArmInput | Mapping[str, object]],
    benchmark_marks: Sequence[BenchmarkMark | Mapping[str, object]],
    thresholds: PolicyThresholds | None = None,
) -> ControlledRun:
    """Execute one arm. Evidence failure stays in cash; limit breaches block."""

    if strategy_id not in ARM_IDS and strategy_id not in _EXTRA_STRUCTURED_ARMS:
        raise ValueError("unknown strategy arm")
    if isinstance(benchmark_marks, (str, bytes)) or len(tuple(benchmark_marks)) == 0:
        raise ValueError("controlled run requires date-bound benchmark marks")
    window = session_window(config, sessions)
    policy = thresholds if thresholds is not None else PolicyThresholds()
    arm_inputs = _arm_inputs(candidates)
    for candidate in arm_inputs:
        if candidate.strategy_id != strategy_id:
            raise ValueError("candidate strategy does not match the arm")
        _validate_arm_shape(candidate)
        _require_on_clock(candidate.decision_session, window, config)
        if candidate.ticker not in set(config.universe):
            raise ValueError(
                "proposal contains out-of-universe ticker(s): "
                f"{[candidate.ticker]}"
            )
    seen: set[tuple[date, str]] = set()
    for candidate in arm_inputs:
        key = (candidate.decision_session, candidate.ticker)
        if key in seen:
            raise ValueError("a strategy may submit only one proposal per decision session")
        seen.add(key)

    gates = tuple(
        _gate_candidate(candidate, config=config, window=window, thresholds=policy)
        for candidate in arm_inputs
    )
    block_reason = _book_block_reason(gates, config)
    if block_reason is not None:
        lineage = _lineage(strategy_id, config, window, gates, executed=False)
        return _run(lineage, gates, executed=False, block_reason=block_reason)

    proposals = _weight_proposals(strategy_id, gates)
    try:
        result = simulate_portfolio(
            config=config,
            sessions=window,
            proposals=proposals,
            strategy_id=strategy_id,
        )
    except ValueError as exc:
        lineage = _lineage(strategy_id, config, window, gates, executed=False)
        return _run(lineage, gates, executed=False, block_reason=str(exc))

    ledger = reconcile_simulation(result, config=config, sessions=window)
    metrics = compute_performance_metrics(
        result,
        config=config,
        benchmark_marks=benchmark_marks,
    )
    _require_metric_ledger_agreement(metrics, ledger, result)
    ledger = ledger.model_copy(
        update={"benchmark_sessions": metrics.benchmark_sessions}
    )
    lineage = _lineage(strategy_id, config, window, gates, executed=True)
    if not ledger.post_cost_within_limit:
        return _run(
            lineage,
            gates,
            executed=True,
            block_reason="post-cost gross leverage exceeds the locked limit",
            ledger=ledger,
            result=result,
        )
    return _run(
        lineage,
        gates,
        executed=True,
        ledger=ledger,
        result=result,
        metrics=metrics,
    )


def _arm_inputs(
    candidates: Sequence[ArmInput | Mapping[str, object]],
) -> tuple[ArmInput, ...]:
    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be arm inputs")
    return tuple(
        item if isinstance(item, ArmInput) else ArmInput.model_validate(item)
        for item in candidates
    )


def _structured_weight_reason(strategy_id: str, weight: float) -> str:
    """Gate label for a weight-only arm. Attention and hybrid do not use the lexicon."""

    if strategy_id == "narrative" and weight == 0.0:
        return "narrative skill gate cash"
    if strategy_id == "narrative":
        return "narrative polarity weight"
    if strategy_id == "attention" and weight == 0.0:
        return "attention skill gate cash"
    if strategy_id == "attention":
        return "attention count weight"
    if strategy_id == "hybrid" and weight == 0.0:
        return "hybrid skill gate cash"
    if strategy_id == "hybrid":
        return "hybrid momentum weight"
    if strategy_id == "edgar" and weight == 0.0:
        return "edgar skill gate cash"
    if strategy_id == "edgar":
        return "edgar filing weight"
    if strategy_id == "form4" and weight == 0.0:
        return "form4 skill gate cash"
    if strategy_id == "form4":
        return "form4 insider weight"
    return "structured weight"


def _validate_arm_shape(candidate: ArmInput) -> None:
    if candidate.strategy_id in {"pure_ml"} or candidate.strategy_id in _EXTRA_STRUCTURED_ARMS:
        if candidate.receipt is not None or candidate.market is not None:
            raise ValueError(f"{candidate.strategy_id} cannot take an evidence receipt or market forecast")
        if candidate.structured_weight is None:
            raise ValueError(f"{candidate.strategy_id} requires a structured weight")
        return
    if candidate.strategy_id == "pure_llm":
        if candidate.market is not None or candidate.structured_weight is not None:
            raise ValueError("pure_llm cannot take a market-verifier forecast as an input")
        if candidate.receipt is None:
            raise ValueError("pure_llm requires a sealed evidence precheck")
        return
    if candidate.receipt is None:
        raise ValueError("fusion requires a sealed evidence precheck")


def _require_on_clock(
    decision_session: date,
    window: tuple[MarketSession, ...],
    config: ExperimentConfig,
) -> None:
    dates = tuple(session.session for session in window)
    try:
        decision_index = dates.index(decision_session)
    except ValueError as exc:
        raise ValueError(
            "proposal decision session is outside the experiment market tape"
        ) from exc
    if decision_index % config.rebalance_frequency_sessions != 0:
        raise ValueError("proposal decision is off the shared rebalance clock")
    if decision_index + config.execution_lag_sessions >= len(dates):
        raise ValueError("proposal has no next market session for execution")


def _gate_candidate(
    candidate: ArmInput,
    *,
    config: ExperimentConfig,
    window: tuple[MarketSession, ...],
    thresholds: PolicyThresholds,
) -> _Gate:
    market_hash = (
        None
        if candidate.market is None
        else canonical_hash(candidate.market.model_dump(mode="json"))
    )
    base = {
        "ticker": candidate.ticker,
        "decision_session": candidate.decision_session,
        "market_input_hash": market_hash,
        "structured_weight": candidate.structured_weight,
        "outcome_ts": candidate.outcome_ts,
        "proposal_hash": None,
        "snapshot_hash": None,
        "thesis_hash": None,
        "receipt_hash": None,
        "precheck_decision": None,
        "verifier_decision": None,
    }
    if candidate.strategy_id in {"pure_ml"} or candidate.strategy_id in _EXTRA_STRUCTURED_ARMS:
        assert candidate.structured_weight is not None
        reason = _structured_weight_reason(candidate.strategy_id, candidate.structured_weight)
        return _Gate(
            weight=candidate.structured_weight,
            reason=reason,
            **base,
        )

    receipt = candidate.receipt
    assert receipt is not None
    receipt.verify_receipt()
    if receipt.proposal.ticker != candidate.ticker:
        raise ValueError("receipt ticker does not match the candidate")
    base.update(
        proposal_hash=receipt.proposal_hash,
        snapshot_hash=receipt.snapshot_hash,
        thesis_hash=None if receipt.thesis is None else receipt.thesis.thesis_hash,
        receipt_hash=receipt.receipt_hash,
        precheck_decision=receipt.decision,
    )
    as_of = datetime.fromisoformat(receipt.proposal.as_of).date()
    if as_of > candidate.decision_session:
        return _Gate(weight=0.0, reason="proposal as_of is after the decision session", **base)
    if receipt.decision != "approved" or receipt.thesis is None or receipt.evidence_audit is None:
        return _Gate(weight=0.0, reason=receipt.reasons[0], **base)

    if candidate.strategy_id == "pure_llm":
        _require_bound_outcome(candidate, window, config)
        if not receipt.thesis.is_prospective(candidate.outcome_ts or ""):
            return _Gate(weight=0.0, reason="prospective seal required", **base)
        return _Gate(
            weight=_signed_cap(receipt, config),
            reason="evidence precheck approved",
            **base,
        )

    if candidate.market is None:
        return _Gate(weight=0.0, reason="market verifier required", **base)
    _require_bound_outcome(candidate, window, config)
    verifier = _verifier_output(receipt.thesis, candidate.market, receipt.evidence_audit)
    adjudication = adjudicate(
        receipt.thesis,
        verifier,
        outcome_ts=candidate.outcome_ts,
        thr=thresholds,
    )
    base["verifier_decision"] = adjudication.decision
    if not adjudication.prospective:
        return _Gate(weight=0.0, reason="prospective seal required", **base)
    if adjudication.decision != "approved":
        return _Gate(weight=0.0, reason=adjudication.reason, **base)
    # A structured weight is the score-book size. It is applied only after
    # the market head approves. Missing that head still leaves the name in cash.
    weight = (
        candidate.structured_weight
        if candidate.structured_weight is not None
        else _signed_cap(receipt, config)
    )
    return _Gate(
        weight=weight,
        reason=adjudication.reason,
        **base,
    )


def _signed_cap(receipt: DecisionReceipt, config: ExperimentConfig) -> float:
    cap = min(receipt.provisional_weight_cap, config.max_position_weight)
    sign = 1.0 if receipt.proposal.direction == "positive" else -1.0
    return sign * cap


def _require_bound_outcome(
    candidate: ArmInput,
    window: tuple[MarketSession, ...],
    config: ExperimentConfig,
) -> None:
    if candidate.outcome_ts is None:
        raise ValueError("outcome timestamp is not bound to the market tape")
    outcome_date = datetime.fromisoformat(candidate.outcome_ts).date()
    dates = tuple(session.session for session in window)
    decision_index = dates.index(candidate.decision_session)
    execution_date = dates[decision_index + config.execution_lag_sessions]
    if outcome_date not in dates or outcome_date < execution_date:
        raise ValueError("outcome timestamp is not bound to the market tape")


def _verifier_output(
    thesis: ThesisContract,
    market: MarketVerification,
    audit: EvidenceAuditSummary,
) -> VerifierOutput:
    payload = market.model_dump(mode="json")
    return VerifierOutput(
        thesis_hash=thesis.thesis_hash,
        produced_at=payload["produced_at"],
        expected_residual_bps=payload["expected_residual_bps"],
        p_adverse=payload["p_adverse"],
        epistemic_var=payload["epistemic_var"],
        aleatoric_var=payload["aleatoric_var"],
        epistemic_mi=payload["epistemic_mi"],
        calibration_hash=payload["calibration_hash"],
        out_of_distribution_score=payload["out_of_distribution_score"],
        fundamental_confirm_prob=payload["fundamental_confirm_prob"],
        evidence_valid=audit.evidence_valid,
        citation_coverage=audit.citation_coverage,
        numeric_reconciliation=audit.numeric_reconciliation,
        timestamp_integrity=audit.timestamp_integrity,
    )


def _book_block_reason(gates: tuple[_Gate, ...], config: ExperimentConfig) -> str | None:
    previous: dict[str, float] = {}
    sessions = sorted({gate.decision_session for gate in gates})
    for session in sessions:
        weights = {
            gate.ticker: gate.weight
            for gate in gates
            if gate.decision_session == session and gate.weight != 0.0
        }
        for weight in weights.values():
            if abs(weight) > config.max_position_weight + 1e-12:
                return "position weight exceeds the locked limit"
        gross = sum(abs(weight) for weight in weights.values())
        if gross > config.max_gross_leverage + 1e-12:
            return "gross leverage exceeds the locked limit"
        post_cost = estimated_post_cost_gross_leverage(weights, previous, config)
        if post_cost > config.max_gross_leverage + 1e-9:
            return "post-cost gross leverage exceeds the locked limit"
        previous = weights
    return None


def _weight_proposals(
    strategy_id: str, gates: tuple[_Gate, ...]
) -> tuple[WeightProposal, ...]:
    sessions = sorted({gate.decision_session for gate in gates})
    proposals: list[WeightProposal] = []
    for session in sessions:
        weights = {
            gate.ticker: gate.weight
            for gate in gates
            if gate.decision_session == session and gate.weight != 0.0
        }
        proposals.append(
            WeightProposal(
                strategy_id=strategy_id,
                decision_session=session,
                targets=tuple(
                    PositionTarget(ticker=ticker, weight=weight)
                    for ticker, weight in sorted(weights.items())
                ),
            )
        )
    return tuple(proposals)


def _lineage(
    strategy_id: str,
    config: ExperimentConfig,
    window: tuple[MarketSession, ...],
    gates: tuple[_Gate, ...],
    *,
    executed: bool,
) -> RunLineage:
    rows = tuple(_proposal_row(strategy_id, gate, executed=executed) for gate in gates)
    config_digest = config_hash(config)
    tape_digest = tape_hash(window)
    lineage_digest = canonical_hash([row.model_dump(mode="json") for row in rows])
    return RunLineage(
        strategy_id=strategy_id,
        experiment_id=config.experiment_id,
        config_hash=config_digest,
        tape_hash=tape_digest,
        lineage_hash=lineage_digest,
        experiment_hash=experiment_hash(
            strategy_id=strategy_id,
            config_hash=config_digest,
            tape_hash=tape_digest,
            lineage_hash=lineage_digest,
        ),
        proposals=rows,
    )


def _proposal_row(strategy_id: str, gate: _Gate, *, executed: bool) -> ProposalLineage:
    admitted = executed and gate.weight != 0.0
    payload = {
        "strategy_id": strategy_id,
        "decision_session": gate.decision_session.isoformat(),
        "ticker": gate.ticker,
        "proposal_hash": gate.proposal_hash,
        "snapshot_hash": gate.snapshot_hash,
        "thesis_hash": gate.thesis_hash,
        "receipt_hash": gate.receipt_hash,
        "market_input_hash": gate.market_input_hash,
        "structured_weight": gate.structured_weight,
        "outcome_ts": gate.outcome_ts,
    }
    return ProposalLineage(
        strategy_id=strategy_id,
        decision_session=gate.decision_session,
        ticker=gate.ticker,
        proposal_hash=gate.proposal_hash,
        snapshot_hash=gate.snapshot_hash,
        thesis_hash=gate.thesis_hash,
        receipt_hash=gate.receipt_hash,
        market_input_hash=gate.market_input_hash,
        input_hash=canonical_hash(payload),
        precheck_decision=gate.precheck_decision,
        verifier_decision=gate.verifier_decision,
        admitted=admitted,
        target_weight=gate.weight,
        reason=gate.reason,
    )


def _run(
    lineage: RunLineage,
    gates: tuple[_Gate, ...],
    *,
    executed: bool,
    block_reason: str | None = None,
    ledger: LedgerReconciliation | None = None,
    result: SimulationResult | None = None,
    metrics: PerformanceMetrics | None = None,
) -> ControlledRun:
    admitted = sum(executed and gate.weight != 0.0 for gate in gates)
    return ControlledRun(
        lineage=lineage,
        ledger=ledger,
        result=result,
        metrics=metrics,
        block_reason=block_reason,
        candidate_count=len(gates),
        admitted_count=admitted,
    )


def _require_metric_ledger_agreement(
    metrics: PerformanceMetrics,
    ledger: LedgerReconciliation,
    result: SimulationResult,
) -> None:
    if not metrics.benchmark_sessions:
        raise ValueError("benchmark marks must bind to portfolio session dates")
    if metrics.benchmark_sessions != tuple(point.session for point in result.points):
        raise ValueError("benchmark marks must bind to portfolio session dates")
    if metrics.trade_count != ledger.trade_count:
        raise ValueError("metrics trade count does not match the ledger")
    if not math.isclose(
        metrics.total_turnover, ledger.total_turnover, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise ValueError("metrics turnover does not match the ledger")
    if not math.isclose(
        metrics.transaction_costs,
        ledger.transaction_costs,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise ValueError("metrics costs do not match the ledger")
    if metrics.total_turnover is None or metrics.transaction_costs is None:
        raise ValueError("ledger turnover and costs must be computed")


def _aware_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.isoformat().replace("+00:00", "Z")
