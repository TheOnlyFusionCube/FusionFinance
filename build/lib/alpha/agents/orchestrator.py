"""Agent/evidence precheck orchestration; downstream capital gates are separate."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import alpha.agents.providers as provider_contracts
from alpha.agents.desk import AgentDesk, AnalystOutputError
from alpha.agents.models import (
    ANALYST_ROLES,
    AnalystReport,
    DecisionReceipt,
    EvidenceAuditSummary,
    FailureReason,
    SealedSourceSnapshot,
    TradeProposal,
    finite_mean,
)
from alpha.agents.providers import AnalystProvider
from alpha.verifier.contract import EvidenceRef, ExpectedOutcome, Horizon, ThesisContract
from alpha.verifier.evidence import (
    EvidenceRecord,
    EvidenceSource,
    audit_evidence,
)


@dataclass(frozen=True, slots=True)
class FusionOrchestrator:
    provider: AnalystProvider
    analyst_timeout_seconds: float = 20.0
    _provider_model: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            provider_model = self.provider.model_id
        except Exception:
            raise ValueError("provider model metadata is unavailable") from None
        if (
            not isinstance(provider_model, str)
            or not provider_model
            or len(provider_model) > 200
            or provider_model != provider_model.strip()
            or any(character.isspace() for character in provider_model)
        ):
            raise ValueError("provider model metadata is invalid")
        object.__setattr__(self, "_provider_model", provider_model)
        AgentDesk(
            provider=self.provider,
            timeout_seconds=self.analyst_timeout_seconds,
        )

    def run(
        self,
        proposal: TradeProposal,
        snapshot: SealedSourceSnapshot,
    ) -> DecisionReceipt:
        snapshot.verify_seal()
        try:
            reports = AgentDesk(
                provider=self.provider,
                timeout_seconds=self.analyst_timeout_seconds,
            ).run(proposal, snapshot)
        except AnalystOutputError:
            return self._failure(proposal, snapshot, "ANALYST_OUTPUT_INVALID")

        try:
            thesis = self._build_thesis(proposal, snapshot, reports)
            audit = self._audit(thesis, snapshot)
            return DecisionReceipt.seal_evaluated(
                proposal=proposal,
                snapshot=snapshot,
                provider_model=self._provider_model,
                reports=reports,
                thesis=thesis,
                evidence_audit=audit,
            )
        except Exception:
            return self._failure(proposal, snapshot, "ORCHESTRATION_FAILED")

    def _failure(
        self,
        proposal: TradeProposal,
        snapshot: SealedSourceSnapshot,
        reason: FailureReason,
    ) -> DecisionReceipt:
        return DecisionReceipt.seal_failure(
            proposal=proposal,
            snapshot=snapshot,
            provider_model=self._provider_model,
            reason=reason,
        )

    def _build_thesis(
        self,
        proposal: TradeProposal,
        snapshot: SealedSourceSnapshot,
        reports: tuple[AnalystReport, ...],
    ) -> ThesisContract:
        source_roles = {
            document.document_id: frozenset(document.roles)
            for document in snapshot.documents
        }
        for report in reports:
            for citation in report.citations:
                allowed_roles = source_roles.get(citation.document_id)
                if allowed_roles is not None and report.role not in allowed_roles:
                    raise ValueError("analyst cited a source outside its eligible role")
        citations = tuple(
            EvidenceRef(
                document_id=citation.document_id,
                section=report.role,
                location=citation.numeric_key,
                quoted_text=citation.quoted_text,
                asserted_value=citation.asserted_value,
            )
            for report in reports
            for citation in report.citations
        )
        thesis = ThesisContract(
            ticker=proposal.ticker,
            as_of=proposal.as_of,
            claim_type=proposal.claim_type,
            direction=proposal.direction,
            horizon=Horizon(value=proposal.horizon_days),
            expected_outcome=ExpectedOutcome(
                target=proposal.target,
                direction=proposal.direction,
                magnitude_bps=proposal.expected_move_bps,
            ),
            evidence=citations,
            causal_chain=tuple(
                item for report in reports for item in report.causal_chain
            ),
            falsifiers=tuple(item for report in reports for item in report.falsifiers),
            llm_confidence=finite_mean(
                tuple(report.confidence for report in reports)
            ),
        )
        committed = thesis.commit(
            model_version=self._provider_model,
            prompt_hash=_prompt_hash(
                proposal,
                snapshot,
                provider_model=self._provider_model,
            ),
        )
        committed.verify_commit()
        return committed

    @staticmethod
    def _audit(
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


def _prompt_hash(
    proposal: TradeProposal,
    snapshot: SealedSourceSnapshot,
    *,
    provider_model: str,
) -> str:
    payload = json.dumps(
        {
            "proposal_hash": proposal.proposal_hash(),
            "snapshot_hash": snapshot.snapshot_hash,
            "roles": ANALYST_ROLES,
            "provider_model": provider_model,
            "system_prompt": provider_contracts.analyst_system_prompt(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


__all__ = ["FusionOrchestrator"]
