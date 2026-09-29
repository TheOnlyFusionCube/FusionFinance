"""The ML verification gate: independent quantitative challenge of LLM ideas.

The gate runs only after an idea has survived the analyst desk and the
deterministic evidence audit. It combines three machine-learned signals, none
of which read LLM prose, embeddings, or conviction:

1. the walk-forward market verifier's calibrated probability that the idea's
   side wins at the idea's horizon, with out-of-distribution scoring;
2. a meta-labeler trained on the fund's own resolved, leakage-clean LLM calls;
3. the LLM's empirical track record, which recalibrates stated conviction.

The gate is veto-first: the ML channel exists to falsify, so it blocks ideas it
confidently contradicts and scales the rest; it does not need to originate the
same idea independently. The legacy strict adjudication (which requires the
market model to agree on direction and materiality) is recorded beside every
verdict for comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

from alpha.agents.models import DecisionReceipt
from alpha.verifier.contract import VerifierOutput
from alpha.verifier.policy import PolicyThresholds, adjudicate

GateDecision = Literal["approved", "vetoed", "abstain"]


@dataclass(frozen=True)
class GateThresholds:
    ood_limit: float = 0.95
    ml_veto_probability: float = 0.42
    meta_veto_probability: float = 0.45
    min_edge_probability: float = 0.52
    calibrated: bool = False

    def __post_init__(self) -> None:
        values = (
            self.ood_limit, self.ml_veto_probability,
            self.meta_veto_probability, self.min_edge_probability,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("gate thresholds must be finite probabilities")


@dataclass(frozen=True)
class GateVerdict:
    decision: GateDecision
    reasons: tuple[str, ...]
    p_llm: float
    p_ml: float | None
    p_meta: float | None
    p_final: float
    expected_residual_bps: float | None
    ood_score: float | None
    strict_adjudication: str
    calibrated: bool
    extras: dict = field(default_factory=dict)
    committee: dict | None = None

    def to_record(self) -> dict:
        return {
            "decision": self.decision,
            "reasons": list(self.reasons),
            "p_llm": _round(self.p_llm),
            "p_ml": _round(self.p_ml),
            "p_meta": _round(self.p_meta),
            "p_final": _round(self.p_final),
            "expected_residual_bps": _round(self.expected_residual_bps),
            "ood_score": _round(self.ood_score),
            "strict_adjudication": self.strict_adjudication,
            "calibrated": self.calibrated,
            "committee": self.committee,
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-4), 1.0 - 1e-4)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def side_probability(forecast: dict, horizon_key: str, side: int) -> float | None:
    adverse = forecast.get("p_adverse", {}).get(horizon_key)
    if adverse is None or not math.isfinite(adverse):
        return None
    return float(1.0 - adverse) if side > 0 else float(adverse)


def desk_cleared(receipt: DecisionReceipt, *, require_consensus: bool = False) -> bool:
    """True when the desk found no fabrication, contradiction, or risk veto.

    With ``require_consensus`` the desk must also have approved outright (three
    supporting roles). Without it, a thin-support abstention may proceed to the
    ML gate, which then decides; evidence failures and vetoes never do.
    """
    if receipt.thesis is None or receipt.evidence_audit is None:
        return False
    if not receipt.evidence_audit.evidence_valid:
        return False
    if receipt.decision == "approved":
        return True
    return (
        not require_consensus
        and receipt.decision == "abstain"
        and receipt.reasons == ("INSUFFICIENT_ANALYST_SUPPORT",)
    )


def verify_idea(
    receipt: DecisionReceipt,
    *,
    forecast: dict | None,
    p_llm: float,
    p_meta: float | None,
    thresholds: GateThresholds = GateThresholds(),
    committee=None,
) -> GateVerdict:
    """Adjudicate one desk-cleared, evidence-valid idea against the ML channel.

    With a ``committee`` vote that reached quorum, the committee's pooled side
    probability is the ML channel (the market head sits on the committee as one
    juror). Without quorum, the single market-head forecast decides as before.
    """

    if not desk_cleared(receipt):
        raise ValueError("only desk-cleared, evidence-audited ideas reach the ML gate")
    receipt.verify_receipt()
    horizon_key = f"{receipt.proposal.horizon_days}d"
    side = 1 if receipt.proposal.direction == "positive" else -1
    strict = _strict_adjudication(receipt, forecast, horizon_key)

    def verdict(decision: GateDecision, reasons: tuple[str, ...], p_final: float,
                p_ml: float | None = None) -> GateVerdict:
        return GateVerdict(
            decision=decision,
            reasons=reasons,
            p_llm=p_llm,
            p_ml=p_ml,
            p_meta=p_meta,
            p_final=p_final,
            expected_residual_bps=(
                None if forecast is None
                else forecast.get("expected_residual_bps", {}).get(horizon_key)
            ),
            ood_score=None if forecast is None else forecast.get("out_of_distribution_score"),
            strict_adjudication=strict,
            calibrated=thresholds.calibrated,
            committee=None if committee is None else committee.to_record(),
        )

    if forecast is not None:
        ood = float(forecast.get("out_of_distribution_score", 1.0))
        if not math.isfinite(ood) or ood > thresholds.ood_limit:
            return verdict("abstain", (f"OUT_OF_DISTRIBUTION {ood:.2f}",), 0.5)
    if committee is not None and committee.decision != "no_quorum":
        if committee.side != side:
            raise ValueError("committee vote was taken for the other side")
        p_ml = committee.p_side
        if committee.decision == "veto":
            return verdict("vetoed", committee.reasons, p_ml, p_ml)
        return _pool(verdict, p_llm, p_ml, p_meta, thresholds, basis_prefix="COMMITTEE")
    if forecast is None:
        return verdict("abstain", ("ML_VERIFIER_UNAVAILABLE",), 0.5)
    p_ml = side_probability(forecast, horizon_key, side)
    if p_ml is None:
        return verdict("abstain", ("NO_ML_FORECAST_AT_HORIZON",), 0.5)
    if p_ml < thresholds.ml_veto_probability:
        return verdict("vetoed", (f"ML_CONTRADICTION p_side={p_ml:.2f}",), p_ml, p_ml)
    return _pool(verdict, p_llm, p_ml, p_meta, thresholds, basis_prefix="MARKET_HEAD")


def _pool(verdict, p_llm: float, p_ml: float, p_meta: float | None,
          thresholds: GateThresholds, *, basis_prefix: str) -> GateVerdict:
    # The meta-labeler is trained on market-head and dossier features, not on the
    # committee, so a committee probability is independent evidence and is added.
    committee_logit = _logit(p_ml) if basis_prefix == "COMMITTEE" else 0.0
    if p_meta is not None and p_meta < thresholds.meta_veto_probability:
        return verdict("vetoed", (f"META_LABEL_VETO p_correct={p_meta:.2f}",), p_meta, p_ml)
    if p_meta is not None:
        # The meta-labeler already conditions on the ML forecast, so it replaces
        # the ML term instead of being pooled with it a second time.
        p_final = _sigmoid(0.5 * _logit(p_meta) + 0.5 * _logit(p_llm) + committee_logit)
        basis = f"{basis_prefix}+META_LABEL"
    else:
        p_final = _sigmoid(_logit(p_llm) + _logit(p_ml))
        basis = f"{basis_prefix}+LLM_TRACK_RECORD"
    if p_final < thresholds.min_edge_probability:
        return verdict("abstain", (f"INSUFFICIENT_EDGE p={p_final:.2f}", basis), p_final, p_ml)
    return verdict("approved", ("ML_DID_NOT_FALSIFY", basis), p_final, p_ml)


def _strict_adjudication(receipt: DecisionReceipt, forecast: dict | None, horizon_key: str) -> str:
    """Legacy agreement-required policy, recorded for comparison only."""
    thesis = receipt.thesis
    audit = receipt.evidence_audit
    assert thesis is not None and audit is not None
    if forecast is None:
        return "unavailable"
    committed = datetime.fromisoformat(thesis.committed_at.replace("Z", "+00:00"))
    produced = max(datetime.now(timezone.utc), committed + timedelta(microseconds=1))
    output = VerifierOutput(
        thesis_hash=thesis.thesis_hash,
        produced_at=produced.isoformat().replace("+00:00", "Z"),
        expected_residual_bps=dict(forecast.get("expected_residual_bps", {})),
        p_adverse=dict(forecast.get("p_adverse", {})),
        epistemic_var=dict(forecast.get("epistemic_var", {})),
        aleatoric_var=dict(forecast.get("aleatoric_var", {})),
        epistemic_mi=dict(forecast.get("epistemic_mi", {})),
        out_of_distribution_score=min(1.0, max(0.0, float(forecast.get("out_of_distribution_score", 1.0)))),
        evidence_valid=audit.evidence_valid,
        citation_coverage=audit.citation_coverage,
        numeric_reconciliation=audit.numeric_reconciliation,
        timestamp_integrity=audit.timestamp_integrity,
    )
    return adjudicate(
        thesis,
        output,
        thr=PolicyThresholds(horizon_key=horizon_key, ood_limit=0.95),
        allow_uncalibrated=True,
    ).decision


__all__ = ["GateThresholds", "GateVerdict", "desk_cleared", "side_probability", "verify_idea"]
