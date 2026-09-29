"""Falsifiable thesis contract + commit-then-compare records for the
LLM-proposes / ML-challenges / meta-policy-decides architecture.

Design invariants (all enforced so agreement statistics stay meaningful):
  * An LLM thesis is a FALSIFIABLE proposition, not prose: claim_type,
    direction, target variable, horizon, magnitude, evidence locations,
    falsifiers. Without these the verifier does not know what it is testing.
  * The thesis is COMMITTED (frozen, content-hashed) BEFORE the verifier runs.
    A later LLM rebuttal is a SEPARATE object; the original is never rewritten,
    so the model cannot rationalise whatever the verifier says.
  * The verifier is an independent error channel. Its inputs are structured
    (accounting, filing-delta, price/volume, factor state) — NEVER the LLM's
    reasoning, embedding, or self-reported confidence. See market_head.py.
  * LLM pretraining leakage makes any HISTORICAL LLM-thesis backtest invalid
    (the model may already know the future). `committed_at` + `prompt_hash` +
    `model_version` exist so only PROSPECTIVE (commit-before-outcome) evidence
    is ever counted. `is_prospective(outcome_ts)` gates that.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        allow_inf_nan=False,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )

ClaimType = Literal[
    "liquidity_deterioration",
    "revenue_quality_weakening",
    "margin_improvement",
    "guidance_unreliable",
    "accounting_risk",
    "downside_risk",
    "near_term_catalyst",
]

# claim_type -> the verification TARGET it must be tested against. A company can
# have deteriorating liquidity while its stock rises for ten days; do NOT collapse
# the fundamental conclusion and the near-term trade into one flag.
CLAIM_TARGET: dict[str, str] = {
    "liquidity_deterioration": "future_cash_leverage_coverage",
    "revenue_quality_weakening": "future_cash_conversion_receivables",
    "margin_improvement": "subsequent_margin_persistence",
    "guidance_unreliable": "subsequent_guidance_revision_miss",
    "accounting_risk": "restatement_accrual_reversal",
    "downside_risk": "future_volatility_drawdown",
    "near_term_catalyst": "future_residual_return",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def content_hash(payload: dict) -> str:
    """Deterministic hash of a thesis' economic content (order-independent)."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class EvidenceRef(_FrozenModel):
    document_id: str
    section: str = ""
    location: str = ""            # e.g. "paragraph-42"
    quoted_text: str = ""         # must appear verbatim in the source (evidence layer)
    asserted_value: float | None = None   # must recompute from PIT data


class Horizon(_FrozenModel):
    type: Literal["trading_days"] = "trading_days"
    value: int = 10


class ExpectedOutcome(_FrozenModel):
    target: str = "sector_neutral_return"
    direction: Literal["negative", "positive"]
    magnitude_bps: float = 0.0    # signed; economic materiality checked vs costs


class ThesisContract(_FrozenModel):
    """A committed, falsifiable LLM conclusion. Immutable once committed."""
    ticker: str
    as_of: str                     # thesis timestamp (information availability bound)
    claim_type: ClaimType
    direction: Literal["negative", "positive"]
    horizon: Horizon = Field(default_factory=Horizon)
    expected_outcome: ExpectedOutcome
    evidence: tuple[EvidenceRef, ...] = Field(default_factory=tuple)
    causal_chain: tuple[str, ...] = Field(default_factory=tuple)
    falsifiers: tuple[str, ...] = Field(default_factory=tuple)
    llm_confidence: float = 0.5

    # commit metadata (leakage integrity)
    committed_at: str = ""
    model_version: str = ""
    prompt_hash: str = ""
    thesis_hash: str = ""

    @model_validator(mode="after")
    def _consistent_direction(self) -> "ThesisContract":
        if self.direction != self.expected_outcome.direction:
            raise ValueError("thesis and expected-outcome directions must agree")
        if self.horizon.value not in {1, 3, 5, 10}:
            raise ValueError("thesis horizon must match a verifier horizon")
        return self

    def _economic_body(self) -> dict:
        return self.model_dump(exclude={"committed_at", "model_version", "prompt_hash", "thesis_hash"})

    def commit(self, *, model_version: str, prompt_hash: str) -> "ThesisContract":
        """Freeze: stamp commit time + content hash. Call once, before verifying."""
        if self.thesis_hash:
            raise ValueError("thesis already committed; create a new object to amend")
        if not model_version or not prompt_hash:
            raise ValueError("commit requires non-empty model_version and prompt_hash")
        return self.model_copy(update={
            "committed_at": _utc(),
            "model_version": model_version,
            "prompt_hash": prompt_hash,
            "thesis_hash": content_hash(self._economic_body()),
        })

    def verify_commit(self) -> None:
        if not all((self.committed_at, self.model_version, self.prompt_hash, self.thesis_hash)):
            raise ValueError("thesis is not fully committed")
        if len(self.thesis_hash) != 64 or any(char not in "0123456789abcdef" for char in self.thesis_hash):
            raise ValueError("invalid thesis hash")
        if self.thesis_hash != content_hash(self._economic_body()):
            raise ValueError("thesis content no longer matches its commitment")
        committed = datetime.fromisoformat(self.committed_at.replace("Z", "+00:00"))
        asof = datetime.fromisoformat(self.as_of.replace("Z", "+00:00"))
        if asof.tzinfo is None:
            asof = asof.replace(tzinfo=timezone.utc)
        if asof > committed:
            raise ValueError("thesis as_of cannot follow its commitment")

    def target_variable(self) -> str:
        return CLAIM_TARGET[self.claim_type]

    def is_prospective(self, outcome_ts: str) -> bool:
        """True iff the outcome was realised strictly AFTER commit — the only
        leakage-safe evidence. Historical (pre-commit) outcomes are contaminated
        by LLM pretraining knowledge and must not be counted."""
        if not self.committed_at:
            raise ValueError("thesis not committed")
        return datetime.fromisoformat(outcome_ts.replace("Z", "+00:00")) > \
            datetime.fromisoformat(self.committed_at.replace("Z", "+00:00"))


class VerifierOutput(_FrozenModel):
    """Independent ML verifier verdict. Committed AFTER the thesis, separately."""
    thesis_hash: str
    produced_at: str = Field(default_factory=_utc)
    # market head: multi-horizon residual-return forecast (bps) + calibrated probs
    expected_residual_bps: Mapping[str, float] = Field(default_factory=dict)  # "1d","3d","5d","10d"
    p_adverse: Mapping[str, float] = Field(default_factory=dict)
    epistemic_var: Mapping[str, float] = Field(default_factory=dict)
    aleatoric_var: Mapping[str, float] = Field(default_factory=dict)
    epistemic_mi: Mapping[str, float] = Field(default_factory=dict)
    prediction_interval_bps: tuple[float, ...] | None = None
    calibration_hash: str = ""
    out_of_distribution_score: float = Field(default=1.0, ge=0.0, le=1.0)
    # fundamental head (optional): probability the stated fundamental claim confirms
    fundamental_confirm_prob: float | None = Field(default=None, ge=0.0, le=1.0)
    # evidence layer (deterministic)
    evidence_valid: bool = False
    citation_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    numeric_reconciliation: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp_integrity: bool = False

    @field_validator("thesis_hash")
    @classmethod
    def _bound_hash(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("thesis_hash must be lowercase SHA-256")
        return value

    @field_validator("calibration_hash")
    @classmethod
    def _calibration_hash(cls, value: str) -> str:
        if value and (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError("calibration_hash must be lowercase SHA-256")
        return value

    @field_validator("expected_residual_bps")
    @classmethod
    def _finite_horizon_values(cls, values: Mapping[str, float]) -> Mapping[str, float]:
        allowed = {"1d", "3d", "5d", "10d"}
        if not set(values).issubset(allowed):
            raise ValueError("unknown verifier horizon")
        return MappingProxyType(dict(values))

    @field_validator("p_adverse")
    @classmethod
    def _probabilities(cls, values: Mapping[str, float]) -> Mapping[str, float]:
        if any(key not in {"1d", "3d", "5d", "10d"} or not 0.0 <= value <= 1.0
               for key, value in values.items()):
            raise ValueError("adverse probabilities must be known horizons in [0, 1]")
        return MappingProxyType(dict(values))

    @field_validator("epistemic_var", "aleatoric_var", "epistemic_mi")
    @classmethod
    def _nonnegative_uncertainty(cls, values: Mapping[str, float]) -> Mapping[str, float]:
        if any(key not in {"1d", "3d", "5d", "10d"} or value < 0.0
               for key, value in values.items()):
            raise ValueError("uncertainty values must be nonnegative at known horizons")
        return MappingProxyType(dict(values))


Decision = Literal["approved", "reject", "research_only", "inconclusive", "abstain"]


class Adjudication(_FrozenModel):
    thesis_hash: str
    decision: Decision
    reason: str = ""
    decided_at: str = Field(default_factory=_utc)
    prospective: bool = False    # was this a leakage-safe (commit<outcome) case
