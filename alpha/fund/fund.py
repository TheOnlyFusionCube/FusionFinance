"""The LLM-first fund: LLM originates, agents investigate, ML verifies, risk sizes.

Pipeline for one ticker at one decision time::

    sealed PIT sources ─► LLM originator ─► idea (or pass)
                                   │
                          analyst desk (market/news/fundamentals/risk)
                                   │
                        committed thesis + deterministic evidence audit
                                   │
            ML gate: market verifier + meta-labeler + LLM track record
                                   │
                         risk book ─► target weight ─► shared execution

Every stage is fail-closed and every idea, including passes and vetoes, is
appended to a hash-chained ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from alpha.agents.models import DecisionReceipt, SealedSourceSnapshot
from alpha.agents.orchestrator import FusionOrchestrator
from alpha.fund.ideas import IdeaOutputError, IdeaRequest, originate
from alpha.fund.leakage import KnowledgeCutoffGuard
from alpha.fund.learning import MetaLabeler, ReliabilityLedger, side_adjusted_features
from alpha.fund.ledger import DecisionLedger
from alpha.fund.sizing import RiskBook, SizedIdea
from alpha.fund.verification import (
    GateThresholds,
    GateVerdict,
    desk_cleared,
    side_probability,
    verify_idea,
)

STAGES = ("pass", "originator_failed", "desk_rejected", "vetoed", "abstain", "approved")


@dataclass(frozen=True)
class IdeaDecision:
    ticker: str
    as_of: str
    stage: str
    side: int = 0
    conviction: float = 0.0
    p_llm: float = 0.5
    claim_type: str = ""
    horizon_days: int = 0
    reasons: tuple[str, ...] = ()
    receipt: DecisionReceipt | None = None
    verdict: GateVerdict | None = None
    meta_features: tuple[float, ...] = ()
    daily_volatility: float | None = None
    contaminated: bool = False
    originator_model: str = ""
    analyst_model: str = ""

    @property
    def traded(self) -> bool:
        return self.stage == "approved"

    def to_record(self) -> dict:
        record = {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "stage": self.stage,
            "side": self.side,
            "conviction": round(self.conviction, 6),
            "p_llm": round(self.p_llm, 6),
            "claim_type": self.claim_type,
            "horizon_days": self.horizon_days,
            "reasons": list(self.reasons),
            "contaminated": self.contaminated,
            "originator_model": self.originator_model,
            "analyst_model": self.analyst_model,
        }
        if self.receipt is not None:
            record["proposal_hash"] = self.receipt.proposal_hash
            record["snapshot_hash"] = self.receipt.snapshot_hash
            record["receipt_hash"] = self.receipt.receipt_hash
            record["desk_decision"] = self.receipt.decision
            if self.receipt.thesis is not None:
                record["thesis_hash"] = self.receipt.thesis.thesis_hash
        if self.verdict is not None:
            record["ml_gate"] = self.verdict.to_record()
        return record


@dataclass
class LLMFirstFund:
    originator: object
    analyst: object
    guard: KnowledgeCutoffGuard = field(default_factory=KnowledgeCutoffGuard)
    thresholds: GateThresholds = field(default_factory=GateThresholds)
    risk: RiskBook = field(default_factory=RiskBook)
    reliability: ReliabilityLedger = field(default_factory=ReliabilityLedger)
    meta: MetaLabeler = field(default_factory=MetaLabeler)
    ledger: DecisionLedger = field(default_factory=DecisionLedger)
    analyst_timeout_seconds: float | None = None   # default: max(20s, provider timeout)
    require_desk_consensus: bool = False

    def __post_init__(self) -> None:
        if self.analyst_timeout_seconds is None:
            provider_timeout = getattr(self.analyst, "timeout_seconds", None) or 0.0
            self.analyst_timeout_seconds = max(20.0, float(provider_timeout))
        self._orchestrator = FusionOrchestrator(
            provider=self.analyst, analyst_timeout_seconds=self.analyst_timeout_seconds
        )

    def evaluate(
        self,
        *,
        ticker: str,
        as_of: str,
        snapshot: SealedSourceSnapshot,
        forecast: dict | None,
        market_state: tuple[float, ...] | None,
        synthetic_world: bool = False,
        jury=None,
    ) -> IdeaDecision:
        """Run one ticker through the pipeline. ``jury(side)`` returns a committee vote."""
        originator_model = self.originator.model_id
        analyst_model = self.analyst.model_id
        contaminated = self.guard.is_contaminated(
            originator_model, as_of, synthetic_world=synthetic_world
        ) or self.guard.is_contaminated(analyst_model, as_of, synthetic_world=synthetic_world)
        base = dict(
            ticker=ticker, as_of=as_of, contaminated=contaminated,
            originator_model=originator_model, analyst_model=analyst_model,
        )
        try:
            originated = originate(
                self.originator,
                IdeaRequest(
                    ticker=ticker,
                    as_of=as_of,
                    snapshot=snapshot,
                    max_position_weight=min(0.25, self.risk.max_position_weight),
                ),
            )
        except IdeaOutputError as exc:
            return self._record(IdeaDecision(stage="originator_failed", reasons=(str(exc),), **base))
        proposal = originated.proposal
        if proposal is None:
            return self._record(IdeaDecision(stage="pass", reasons=("ORIGINATOR_PASSED",), **base))

        side = 1 if proposal.direction == "positive" else -1
        horizon_key = f"{proposal.horizon_days}d"
        idea = dict(
            side=side,
            conviction=proposal.confidence,
            claim_type=proposal.claim_type,
            horizon_days=proposal.horizon_days,
            **base,
        )
        meta_features = self._meta_features(side, horizon_key, forecast, market_state)
        volatility = _daily_volatility(forecast, horizon_key, proposal.horizon_days)
        p_llm = self.reliability.calibrated_probability(
            originator_model, proposal.claim_type, proposal.confidence
        )

        receipt = self._orchestrator.run(proposal, snapshot)
        if not desk_cleared(receipt, require_consensus=self.require_desk_consensus):
            return self._record(IdeaDecision(
                stage="desk_rejected", reasons=receipt.reasons, receipt=receipt,
                meta_features=meta_features, daily_volatility=volatility,
                p_llm=p_llm, **idea,
            ))

        p_meta = self.meta.predict(meta_features) if meta_features else None
        verdict = verify_idea(
            receipt, forecast=forecast, p_llm=p_llm, p_meta=p_meta, thresholds=self.thresholds,
            committee=jury(side) if jury is not None else None,
        )
        return self._record(IdeaDecision(
            stage=verdict.decision, reasons=verdict.reasons, receipt=receipt, verdict=verdict,
            meta_features=meta_features, daily_volatility=volatility, p_llm=p_llm, **idea,
        ))

    def size(self, decisions: list[IdeaDecision], *, drawdown: float = 0.0) -> dict[str, float]:
        ideas = []
        for decision in decisions:
            if not decision.traded:
                continue
            assert decision.verdict is not None
            ideas.append(SizedIdea(
                ticker=decision.ticker,
                side=decision.side,
                probability=decision.verdict.p_final,
                daily_volatility=decision.daily_volatility,
            ))
        return self.risk.targets(ideas, drawdown=drawdown)

    def _record(self, decision: IdeaDecision) -> IdeaDecision:
        self.ledger.append(decision.to_record())
        return decision

    @staticmethod
    def _meta_features(
        side: int, horizon_key: str, forecast: dict | None,
        market_state: tuple[float, ...] | None,
    ) -> tuple[float, ...]:
        if forecast is None or market_state is None:
            return ()
        p_side = side_probability(forecast, horizon_key, side)
        bps = forecast.get("expected_residual_bps", {}).get(horizon_key)
        if p_side is None or bps is None:
            return ()
        return (
            *side_adjusted_features(side, market_state),
            side * float(bps) / 100.0,
            p_side - 0.5,
            float(forecast.get("out_of_distribution_score", 1.0)),
        )


def _daily_volatility(forecast: dict | None, horizon_key: str, horizon: int) -> float | None:
    if forecast is None:
        return None
    variance = forecast.get("aleatoric_var", {}).get(horizon_key)
    if variance is None or variance <= 0.0:
        return None
    return float((variance / horizon) ** 0.5)


__all__ = ["IdeaDecision", "LLMFirstFund", "STAGES"]
