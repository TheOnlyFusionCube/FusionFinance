"""Meta-policy: adjudicate a committed thesis against an independent verifier.

Four verdicts, never a naive binary (the second row of the matrix is the point:
"the LLM may be right but the market offers no actionable expression" is NOT the
same as "the LLM is wrong"). The verifier is used as a VETO before it is used as
a sizing signal.

Thresholds here are DELIBERATELY conservative placeholders. They must be
calibrated by walk-forward validation on the coverage / false-approval /
false-veto / net-IC / net-return trade-off before any live use. `PolicyThresholds`
carries `calibrated: bool` so an uncalibrated policy is self-documenting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from alpha.verifier.contract import Adjudication, ThesisContract, VerifierOutput


@dataclass(frozen=True)
class PolicyThresholds:
    ood_limit: float = 0.5              # abstain above this out-of-distribution score
    contradiction_prob: float = 0.35    # fundamental confirm-prob below -> reject
    confirmation_prob: float = 0.58     # market adverse-prob below -> inconclusive
    materiality_bps: float = 20.0       # |expected residual| below -> not economically actionable
    horizon_key: str = "10d"
    calibrated: bool = False            # MUST be set True only after walk-forward calibration
    calibration_hash: str = ""

    def __post_init__(self) -> None:
        probabilities = (self.ood_limit, self.contradiction_prob, self.confirmation_prob)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("policy probability thresholds must be finite in [0, 1]")
        if not math.isfinite(self.materiality_bps) or self.materiality_bps < 0.0:
            raise ValueError("materiality_bps must be finite and nonnegative")
        if self.horizon_key not in {"1d", "3d", "5d", "10d"}:
            raise ValueError("unknown policy horizon")
        if self.calibrated and (
            len(self.calibration_hash) != 64 or
            any(char not in "0123456789abcdef" for char in self.calibration_hash)
        ):
            raise ValueError("calibrated policy requires a lowercase SHA-256 artifact hash")


def _market_direction(v: VerifierOutput, horizon_key: str) -> str | None:
    bps = v.expected_residual_bps.get(horizon_key)
    if bps is None:
        return None
    return "negative" if bps < 0 else "positive"


def adjudicate(
    thesis: ThesisContract,
    v: VerifierOutput,
    *,
    outcome_ts: str | None = None,
    thr: PolicyThresholds = PolicyThresholds(),
    allow_uncalibrated: bool = False,
) -> Adjudication:
    """Apply the veto-first gate. `outcome_ts`, if given, records whether this is
    a leakage-safe prospective case (commit strictly before outcome)."""
    if v.thesis_hash != thesis.thesis_hash:
        raise ValueError("verifier output is not bound to this committed thesis")
    thesis.verify_commit()
    committed_at = datetime.fromisoformat(thesis.committed_at.replace("Z", "+00:00"))
    produced_at = datetime.fromisoformat(v.produced_at.replace("Z", "+00:00"))
    if produced_at <= committed_at:
        raise ValueError("verifier output must be produced after thesis commitment")
    prospective = False
    if outcome_ts:
        outcome = datetime.fromisoformat(outcome_ts.replace("Z", "+00:00"))
        prospective = thesis.is_prospective(outcome_ts) and produced_at < outcome

    def out(decision: str, reason: str) -> Adjudication:
        return Adjudication(thesis_hash=thesis.thesis_hash, decision=decision,
                            reason=reason, prospective=prospective)

    # 1. Evidence failure is a hard reject regardless of the return model.
    if (not v.evidence_valid or v.citation_coverage < 1.0 or
            v.numeric_reconciliation < 1.0 or not v.timestamp_integrity):
        return out("reject", "evidence layer failed (citation/reconciliation/timestamp)")
    if not allow_uncalibrated and (not thr.calibrated or not thr.calibration_hash):
        return out("abstain", "policy thresholds lack a calibration artifact")
    if thr.calibrated and v.calibration_hash != thr.calibration_hash:
        return out("abstain", "verifier probability is not bound to policy calibration")
    if thr.horizon_key != f"{thesis.horizon.value}d":
        return out("inconclusive", "policy horizon does not match committed thesis horizon")
    if thesis.expected_outcome.target not in {"sector_neutral_return", "future_residual_return"}:
        return out("inconclusive", "market policy target does not match committed outcome target")
    # 2. Out of distribution -> abstain / human review.
    if v.out_of_distribution_score > thr.ood_limit:
        return out("abstain", f"OOD {v.out_of_distribution_score:.2f} > {thr.ood_limit}")
    # 3. Fundamental verifier actively contradicts the stated fundamental claim.
    if v.fundamental_confirm_prob is not None and v.fundamental_confirm_prob < thr.contradiction_prob:
        return out("reject", f"fundamental confirm-prob {v.fundamental_confirm_prob:.2f} contradicts claim")
    # 4. Market direction disagrees with the thesis' stated trade direction.
    mdir = _market_direction(v, thr.horizon_key)
    if mdir is None:
        return out("inconclusive", "no market forecast at target horizon")
    bps = v.expected_residual_bps[thr.horizon_key]
    if not math.isfinite(bps):
        return out("abstain", "non-finite market forecast")
    if mdir != thesis.expected_outcome.direction:
        # LLM analysis may be right, but the market offers no actionable expression.
        return out("research_only", "independent market forecast disagrees on direction/timing")
    # 5. Direction agrees but conviction/materiality is weak.
    mbps = abs(v.expected_residual_bps.get(thr.horizon_key, 0.0))
    padv = v.p_adverse.get(thr.horizon_key)   # P(residual < 0) at horizon
    if padv is None:
        return out("inconclusive", "no adverse probability at target horizon")
    if mbps < thr.materiality_bps:
        return out("inconclusive", f"|residual| {mbps:.0f}bps below materiality {thr.materiality_bps:.0f}")
    # probability the market moves in the THESIS' stated direction
    if padv is not None:
        p_dir = padv if thesis.expected_outcome.direction == "negative" else (1.0 - padv)
        if p_dir < thr.confirmation_prob:
            return out("inconclusive", "directional probability below confirmation threshold")
    return out("approved", "evidence ok; fundamental not contradicted; market agrees, material")


def adjudicate_gpu(
    thesis: ThesisContract,
    v: VerifierOutput,
    *,
    outcome_ts: str | None = None,
    thr: PolicyThresholds = PolicyThresholds(),
    tensor_thr=None,
    device: str = "cuda:0",
    strict_gpu: bool = True,
    allow_uncalibrated: bool = False,
) -> Adjudication:
    """Apply the tensor uncertainty veto, then preserve the legacy contract gate."""
    import torch

    from alpha.gpu_ml import assert_rocm_stage
    from alpha.verifier.gpu_policy import (
        ABSTAIN, REJECT, TensorPolicyThresholds, tensor_policy,
    )

    if v.thesis_hash != thesis.thesis_hash:
        raise ValueError("verifier output is not bound to this committed thesis")
    base = adjudicate(
        thesis, v, outcome_ts=outcome_ts, thr=thr,
        allow_uncalibrated=allow_uncalibrated,
    )
    if base.decision == "reject":
        return base
    key = thr.horizon_key
    if key not in v.p_adverse or key not in v.epistemic_mi:
        if base.decision == "approved":
            return Adjudication(
                thesis_hash=thesis.thesis_hash, decision="abstain",
                reason="tensor policy inputs are incomplete", prospective=base.prospective,
            )
        return base
    if not strict_gpu and device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    adverse = v.p_adverse[key]
    contradiction = adverse if thesis.expected_outcome.direction == "positive" else 1.0 - adverse
    tensors = tuple(
        torch.tensor([value], dtype=torch.float32, device=device)
        for value in (contradiction, v.epistemic_mi[key], v.out_of_distribution_score)
    )
    assert_rocm_stage(
        "verifier-tensor-policy", *tensors, strict=strict_gpu,
    )
    result = tensor_policy(
        *tensors, thresholds=tensor_thr or TensorPolicyThresholds(),
        context={"thesis_hash": thesis.thesis_hash, "horizon": key},
    )
    decision = int(result.decisions.item())
    prospective = base.prospective
    suffix = result.decision_hash[:12]
    if decision == REJECT:
        return Adjudication(
            thesis_hash=thesis.thesis_hash, decision="reject",
            reason=f"tensor policy contradiction veto ({suffix})",
            prospective=prospective,
        )
    if decision == ABSTAIN:
        return Adjudication(
            thesis_hash=thesis.thesis_hash, decision="abstain",
            reason=f"tensor policy uncertainty/novelty veto ({suffix})",
            prospective=prospective,
        )
    return base
