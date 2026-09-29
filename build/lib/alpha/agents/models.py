"""Strict immutable contracts for the lightweight FusionFinance agent desk."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime
from numbers import Real
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alpha.verifier.contract import CLAIM_TARGET, ClaimType, ThesisContract
from alpha.verifier.evidence import EvidenceRecord, EvidenceSource, audit_evidence


AnalystRole = Literal["market", "news", "fundamentals", "risk"]
ANALYST_ROLES: tuple[AnalystRole, ...] = (
    "market",
    "news",
    "fundamentals",
    "risk",
)
Direction = Literal["positive", "negative", "neutral"]
RuntimeDecision = Literal["approved", "reject", "abstain"]
FailureReason = Literal["ANALYST_OUTPUT_INVALID", "ORCHESTRATION_FAILED"]
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _json_ready(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _aware_iso8601(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.isoformat().replace("+00:00", "Z")


def _require_real_number(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("value must be a real number, not a coerced scalar")
    return value


def _require_optional_real_number(value: object) -> object:
    return value if value is None else _require_real_number(value)


def _require_boolean(value: object) -> object:
    if type(value) is not bool:
        raise ValueError("value must be a boolean, not a coerced scalar")
    return value


class NumericValue(_StrictFrozenModel):
    key: str
    value: float

    @field_validator("value", mode="before")
    @classmethod
    def _strict_value(cls, value: object) -> object:
        return _require_real_number(value)

    @field_validator("key")
    @classmethod
    def _valid_key(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("numeric key is invalid")
        return value


class SourceDocument(_StrictFrozenModel):
    document_id: str
    available_at: str
    text: str = Field(min_length=1, max_length=20_000)
    roles: tuple[AnalystRole, ...] = Field(min_length=1)
    numeric_values: tuple[NumericValue, ...] = ()

    @field_validator("document_id")
    @classmethod
    def _valid_document_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("document_id is invalid")
        return value

    @field_validator("available_at")
    @classmethod
    def _valid_available_at(cls, value: str) -> str:
        return _aware_iso8601(value)

    @model_validator(mode="after")
    def _unique_fields(self) -> "SourceDocument":
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("document roles must be unique")
        keys = tuple(item.key for item in self.numeric_values)
        if len(set(keys)) != len(keys):
            raise ValueError("numeric keys must be unique per document")
        return self


class SealedSourceSnapshot(_StrictFrozenModel):
    documents: tuple[SourceDocument, ...] = Field(min_length=1, max_length=32)
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @staticmethod
    def _digest(documents: tuple[SourceDocument, ...]) -> str:
        return _canonical_hash(
            {"documents": [item.model_dump(mode="json") for item in documents]}
        )

    @classmethod
    def seal(
        cls, documents: tuple[SourceDocument, ...]
    ) -> "SealedSourceSnapshot":
        frozen_documents = tuple(documents)
        return cls(
            documents=frozen_documents,
            snapshot_hash=cls._digest(frozen_documents),
        )

    @model_validator(mode="after")
    def _sealed_and_unique(self) -> "SealedSourceSnapshot":
        identifiers = tuple(item.document_id for item in self.documents)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("snapshot document IDs must be unique")
        if self.snapshot_hash != self._digest(self.documents):
            raise ValueError("snapshot content does not match its seal")
        return self

    def verify_seal(self) -> None:
        if self.snapshot_hash != self._digest(self.documents):
            raise ValueError("snapshot content does not match its seal")


class TradeProposal(_StrictFrozenModel):
    ticker: str
    as_of: str
    direction: Literal["positive", "negative"]
    horizon_days: Literal[1, 3, 5, 10]
    expected_move_bps: float
    confidence: float = Field(ge=0.0, le=1.0)
    claim_type: ClaimType
    target: Literal["sector_neutral_return", "future_residual_return"] = (
        "sector_neutral_return"
    )
    max_position_weight: float = Field(default=0.1, gt=0.0, le=0.25)

    @field_validator(
        "expected_move_bps",
        "confidence",
        "max_position_weight",
        mode="before",
    )
    @classmethod
    def _strict_numbers(cls, value: object) -> object:
        return _require_real_number(value)

    @field_validator("horizon_days", mode="before")
    @classmethod
    def _strict_horizon(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("horizon_days must be an integer")
        return value

    @field_validator("ticker")
    @classmethod
    def _valid_ticker(cls, value: str) -> str:
        if _TICKER.fullmatch(value) is None:
            raise ValueError("ticker must be an uppercase market symbol")
        return value

    @field_validator("as_of")
    @classmethod
    def _valid_as_of(cls, value: str) -> str:
        return _aware_iso8601(value)

    @model_validator(mode="after")
    def _direction_matches_move(self) -> "TradeProposal":
        if self.direction == "positive" and self.expected_move_bps <= 0.0:
            raise ValueError("positive proposal requires a positive expected move")
        if self.direction == "negative" and self.expected_move_bps >= 0.0:
            raise ValueError("negative proposal requires a negative expected move")
        return self

    def proposal_hash(self) -> str:
        return _canonical_hash(self.model_dump(mode="json"))

    def claim_target(self) -> str:
        """Return the fundamental target, distinct from the market trade target."""

        return CLAIM_TARGET[self.claim_type]


class AnalystCitation(_StrictFrozenModel):
    document_id: str
    quoted_text: str = Field(min_length=12, max_length=2_000)
    numeric_key: str = ""
    asserted_value: float | None = None

    @field_validator("asserted_value", mode="before")
    @classmethod
    def _strict_asserted_value(cls, value: object) -> object:
        return _require_optional_real_number(value)

    @field_validator("document_id")
    @classmethod
    def _valid_document_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("citation document_id is invalid")
        return value

    @model_validator(mode="after")
    def _numeric_pair(self) -> "AnalystCitation":
        if bool(self.numeric_key) != (self.asserted_value is not None):
            raise ValueError("numeric_key and asserted_value must appear together")
        if self.numeric_key and _IDENTIFIER.fullmatch(self.numeric_key) is None:
            raise ValueError("citation numeric_key is invalid")
        return self


class AnalystReport(_StrictFrozenModel):
    role: AnalystRole
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=1_000)
    citations: tuple[AnalystCitation, ...] = Field(min_length=1, max_length=8)
    causal_chain: tuple[str, ...] = Field(min_length=1, max_length=8)
    falsifiers: tuple[str, ...] = Field(min_length=1, max_length=8)
    risk_flags: tuple[str, ...] = Field(default=(), max_length=8)
    veto: bool = False

    @field_validator("confidence", mode="before")
    @classmethod
    def _strict_confidence(cls, value: object) -> object:
        return _require_real_number(value)

    @field_validator("veto", mode="before")
    @classmethod
    def _strict_veto(cls, value: object) -> object:
        return _require_boolean(value)

    @field_validator("causal_chain", "falsifiers", "risk_flags")
    @classmethod
    def _bounded_nonempty_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() or len(item) > 500 for item in values):
            raise ValueError("report list items must be non-empty and bounded")
        return values

    @model_validator(mode="after")
    def _veto_is_reserved_for_risk(self) -> "AnalystReport":
        if self.veto and self.role != "risk":
            raise ValueError("only the risk role may issue a veto")
        return self

    @model_validator(mode="after")
    def _numeric_claims_are_structured(self) -> "AnalystReport":
        narrative_fields = (
            self.summary,
            *self.causal_chain,
            *self.falsifiers,
            *self.risk_flags,
        )
        if any(re.search(r"\d", item) for item in narrative_fields):
            raise ValueError(
                "numeric claims must use citation numeric_key/asserted_value fields"
            )
        return self


class AnalystRequest(_StrictFrozenModel):
    role: AnalystRole
    proposal: TradeProposal
    snapshot: SealedSourceSnapshot


class EvidenceAuditSummary(_StrictFrozenModel):
    evidence_valid: bool
    citation_coverage: float = Field(ge=0.0, le=1.0)
    numeric_reconciliation: float = Field(ge=0.0, le=1.0)
    timestamp_integrity: bool
    errors: tuple[str, ...]

    @field_validator("citation_coverage", "numeric_reconciliation", mode="before")
    @classmethod
    def _strict_scores(cls, value: object) -> object:
        return _require_real_number(value)

    @field_validator("evidence_valid", "timestamp_integrity", mode="before")
    @classmethod
    def _strict_flags(cls, value: object) -> object:
        return _require_boolean(value)

    @model_validator(mode="after")
    def _valid_flag_matches_audit_fields(self) -> "EvidenceAuditSummary":
        if self.evidence_valid and (
            self.citation_coverage != 1.0
            or self.numeric_reconciliation != 1.0
            or not self.timestamp_integrity
            or self.errors
        ):
            raise ValueError("valid evidence requires consistent audit fields")
        return self


def finite_mean(values: tuple[float, ...]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("confidence inputs must be finite and non-empty")
    return sum(values) / len(values)


def derive_precheck_decision(
    proposal: TradeProposal,
    reports: tuple[AnalystReport, ...],
    audit: EvidenceAuditSummary,
) -> tuple[RuntimeDecision, float, tuple[str, ...]]:
    """Derive the only decision fields allowed in an evaluated precheck."""

    if tuple(report.role for report in reports) != ANALYST_ROLES:
        raise ValueError("decision derivation requires the four analyst roles")
    if not audit.evidence_valid:
        return "reject", 0.0, ("EVIDENCE_VETO", *audit.errors)
    risk_report = next(report for report in reports if report.role == "risk")
    if risk_report.veto:
        return "reject", 0.0, ("RISK_VETO", *risk_report.risk_flags)
    contradictions = tuple(
        report.role
        for report in reports
        if report.direction not in {proposal.direction, "neutral"}
    )
    if contradictions:
        return "abstain", 0.0, (
            "ANALYST_CONTRADICTION " + ",".join(contradictions),
        )
    support = tuple(
        report for report in reports if report.direction == proposal.direction
    )
    if len(support) < 3:
        return "abstain", 0.0, ("INSUFFICIENT_ANALYST_SUPPORT",)
    confidence = min(
        proposal.confidence,
        finite_mean(tuple(report.confidence for report in support)),
    )
    provisional_cap = round(proposal.max_position_weight * confidence, 6)
    return "approved", provisional_cap, (
        "CITATION_INTEGRITY_VALID",
        "NO_RISK_ROLE_VETO",
    )


def _evidence_tuple_from_reports(
    reports: tuple[AnalystReport, ...],
) -> tuple[tuple[str, str, str, str, float | None], ...]:
    return tuple(
        (
            citation.document_id,
            report.role,
            citation.numeric_key,
            citation.quoted_text,
            citation.asserted_value,
        )
        for report in reports
        for citation in report.citations
    )


def _evidence_tuple_from_thesis(
    thesis: ThesisContract,
) -> tuple[tuple[str, str, str, str, float | None], ...]:
    return tuple(
        (
            item.document_id,
            item.section,
            item.location,
            item.quoted_text,
            item.asserted_value,
        )
        for item in thesis.evidence
    )


def _audit_thesis_snapshot(
    thesis: ThesisContract,
    snapshot: SealedSourceSnapshot,
) -> EvidenceAuditSummary:
    records = tuple(
        EvidenceRecord(
            document_id=item.document_id,
            quoted_text=item.quoted_text,
            numeric_key=item.location,
            asserted_value=item.asserted_value,
        )
        for item in thesis.evidence
    )
    sources = tuple(
        EvidenceSource(
            document_id=item.document_id,
            available_at=item.available_at,
            text=item.text,
            numeric_values={value.key: value.value for value in item.numeric_values},
        )
        for item in snapshot.documents
    )
    result = audit_evidence(records, sources, thesis_as_of=thesis.as_of)
    return EvidenceAuditSummary(
        evidence_valid=result.evidence_valid,
        citation_coverage=result.citation_coverage,
        numeric_reconciliation=result.numeric_reconciliation,
        timestamp_integrity=result.timestamp_integrity,
        errors=result.errors,
    )


class DecisionReceipt(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    stage: Literal["agent_evidence_precheck"] = "agent_evidence_precheck"
    proposal: TradeProposal
    snapshot: SealedSourceSnapshot
    proposal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_model: str = Field(min_length=1, max_length=200)
    reports: tuple[AnalystReport, ...]
    thesis: ThesisContract | None = None
    evidence_audit: EvidenceAuditSummary | None = None
    decision: RuntimeDecision
    provisional_weight_cap: float = Field(ge=0.0, le=0.25)
    reasons: tuple[str, ...] = Field(min_length=1)
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("provisional_weight_cap", mode="before")
    @classmethod
    def _strict_provisional_cap(cls, value: object) -> object:
        return _require_real_number(value)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @staticmethod
    def _digest(payload: dict[str, object]) -> str:
        return _canonical_hash(payload)

    @classmethod
    def seal_evaluated(
        cls,
        *,
        proposal: TradeProposal,
        snapshot: SealedSourceSnapshot,
        provider_model: str,
        reports: tuple[AnalystReport, ...],
        thesis: ThesisContract,
        evidence_audit: EvidenceAuditSummary,
    ) -> "DecisionReceipt":
        decision, provisional_cap, reasons = derive_precheck_decision(
            proposal,
            reports,
            evidence_audit,
        )
        content = {
            "schema_version": 2,
            "stage": "agent_evidence_precheck",
            "proposal": proposal,
            "snapshot": snapshot,
            "proposal_hash": proposal.proposal_hash(),
            "snapshot_hash": snapshot.snapshot_hash,
            "provider_model": provider_model,
            "reports": reports,
            "thesis": thesis,
            "evidence_audit": evidence_audit,
            "decision": decision,
            "provisional_weight_cap": provisional_cap,
            "reasons": reasons,
        }
        digest = cls._digest(content)
        return cls(**content, receipt_hash=digest)

    @classmethod
    def seal_failure(
        cls,
        *,
        proposal: TradeProposal,
        snapshot: SealedSourceSnapshot,
        provider_model: str,
        reason: FailureReason,
    ) -> "DecisionReceipt":
        content = {
            "schema_version": 2,
            "stage": "agent_evidence_precheck",
            "proposal": proposal,
            "snapshot": snapshot,
            "proposal_hash": proposal.proposal_hash(),
            "snapshot_hash": snapshot.snapshot_hash,
            "provider_model": provider_model,
            "reports": (),
            "thesis": None,
            "evidence_audit": None,
            "decision": "reject",
            "provisional_weight_cap": 0.0,
            "reasons": (reason,),
        }
        digest = cls._digest(content)
        return cls(**content, receipt_hash=digest)

    @model_validator(mode="after")
    def _decision_invariants(self) -> "DecisionReceipt":
        self._verify_semantics()
        expected = self._digest(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        )
        if self.receipt_hash != expected:
            raise ValueError("receipt content does not match its seal")
        return self

    def _verify_semantics(self) -> None:
        self.snapshot.verify_seal()
        if self.proposal_hash != self.proposal.proposal_hash():
            raise ValueError("proposal hash does not match the bound proposal")
        if self.snapshot_hash != self.snapshot.snapshot_hash:
            raise ValueError("snapshot hash does not match the bound snapshot")

        is_failure = (
            not self.reports
            and self.thesis is None
            and self.evidence_audit is None
        )
        is_evaluated = bool(
            self.reports
            and self.thesis is not None
            and self.evidence_audit is not None
        )
        if not (is_failure or is_evaluated):
            raise ValueError(
                "receipt must contain a complete evaluated or complete failure tuple"
            )
        if is_failure:
            allowed = {
                ("ANALYST_OUTPUT_INVALID",),
                ("ORCHESTRATION_FAILED",),
            }
            if (
                self.decision != "reject"
                or self.provisional_weight_cap != 0.0
                or self.reasons not in allowed
            ):
                raise ValueError("failure receipt fields are not canonical")
            return

        assert self.thesis is not None
        assert self.evidence_audit is not None
        roles = tuple(report.role for report in self.reports)
        if roles != ANALYST_ROLES:
            raise ValueError("evaluated receipt requires the four analyst roles")
        self.thesis.verify_commit()
        if self.provider_model != self.thesis.model_version:
            raise ValueError("provider model must match the committed thesis")
        self._verify_thesis_binding()
        expected_audit = _audit_thesis_snapshot(self.thesis, self.snapshot)
        if self.evidence_audit != expected_audit:
            raise ValueError("evidence audit does not match the bound snapshot")
        expected = derive_precheck_decision(
            self.proposal,
            self.reports,
            self.evidence_audit,
        )
        actual = (
            self.decision,
            self.provisional_weight_cap,
            self.reasons,
        )
        if actual != expected:
            raise ValueError("decision, provisional cap, or reasons were not derived")

    def _verify_thesis_binding(self) -> None:
        assert self.thesis is not None
        expected_core = (
            self.proposal.ticker,
            self.proposal.as_of,
            self.proposal.claim_type,
            self.proposal.direction,
            self.proposal.horizon_days,
            self.proposal.target,
            self.proposal.expected_move_bps,
        )
        actual_core = (
            self.thesis.ticker,
            self.thesis.as_of,
            self.thesis.claim_type,
            self.thesis.direction,
            self.thesis.horizon.value,
            self.thesis.expected_outcome.target,
            self.thesis.expected_outcome.magnitude_bps,
        )
        if actual_core != expected_core:
            raise ValueError("thesis does not match the bound proposal")
        if self.thesis.expected_outcome.direction != self.proposal.direction:
            raise ValueError("thesis outcome direction does not match the proposal")
        if self.thesis.llm_confidence != finite_mean(
            tuple(report.confidence for report in self.reports)
        ):
            raise ValueError("thesis confidence does not match analyst reports")
        if self.thesis.causal_chain != tuple(
            item for report in self.reports for item in report.causal_chain
        ):
            raise ValueError("thesis causal chain does not match analyst reports")
        if self.thesis.falsifiers != tuple(
            item for report in self.reports for item in report.falsifiers
        ):
            raise ValueError("thesis falsifiers do not match analyst reports")
        if _evidence_tuple_from_thesis(self.thesis) != _evidence_tuple_from_reports(
            self.reports
        ):
            raise ValueError("thesis evidence does not match analyst citations")
        source_roles = {
            document.document_id: frozenset(document.roles)
            for document in self.snapshot.documents
        }
        for report in self.reports:
            for citation in report.citations:
                allowed_roles = source_roles.get(citation.document_id)
                if allowed_roles is not None and report.role not in allowed_roles:
                    raise ValueError("analyst evidence source is not role eligible")
        as_of = datetime.fromisoformat(self.proposal.as_of.replace("Z", "+00:00"))
        if any(
            datetime.fromisoformat(document.available_at.replace("Z", "+00:00"))
            > as_of
            for document in self.snapshot.documents
        ):
            raise ValueError("evaluated receipt contains a future source")

    def verify_receipt(self) -> None:
        self._verify_semantics()
        expected = self._digest(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        )
        if self.receipt_hash != expected:
            raise ValueError("receipt content does not match its seal")

__all__ = [
    "ANALYST_ROLES",
    "AnalystCitation",
    "AnalystReport",
    "AnalystRequest",
    "AnalystRole",
    "DecisionReceipt",
    "EvidenceAuditSummary",
    "FailureReason",
    "NumericValue",
    "RuntimeDecision",
    "SealedSourceSnapshot",
    "SourceDocument",
    "TradeProposal",
    "derive_precheck_decision",
    "finite_mean",
]
