from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha.agents.models import SealedSourceSnapshot, SourceDocument
from alpha.agents.orchestrator import FusionOrchestrator
from alpha.agents.providers import DeterministicOfflineProvider
from alpha.fund.backtest import ARMS, BacktestConfig, run_backtest
from alpha.fund.cache import CacheMiss, RecordReplayProvider
from alpha.fund.claude import ANALYST_SCHEMA, IDEA_SCHEMA, ClaudeProvider
from alpha.fund.fund import LLMFirstFund
from alpha.fund.ideas import (
    DeterministicOriginator, IdeaOutputError, IdeaRequest, OriginatorIdea, originate,
)
from alpha.fund.leakage import KnowledgeCutoffGuard
from alpha.fund.learning import MetaLabeler, ReliabilityLedger, ResolvedCall
from alpha.fund.ledger import DecisionLedger
from alpha.fund.sizing import RiskBook, SizedIdea
from alpha.fund.synthetic import WorldConfig
from alpha.fund.verification import GateThresholds, verify_idea

AS_OF = "2026-07-10T21:00:00Z"


def _snapshot(text: str = "Revenue growth accelerated and management raised guidance.") -> SealedSourceSnapshot:
    return SealedSourceSnapshot.seal(tuple(
        SourceDocument(document_id=f"ACME.{role}", available_at="2026-07-10T20:00:00Z",
                       text=text, roles=(role,))
        for role in ("market", "news", "fundamentals", "risk")
    ))


def _request(snapshot: SealedSourceSnapshot | None = None) -> IdeaRequest:
    return IdeaRequest(ticker="ACME", as_of=AS_OF, snapshot=snapshot or _snapshot())


class _Raw:
    model_id = "raw-test"

    def __init__(self, payload: str) -> None:
        self.payload = payload

    def originate(self, request: IdeaRequest) -> str:
        return self.payload


def _idea(**overrides) -> str:
    body = dict(
        ticker="ACME", action="long", horizon_days=5, expected_move_bps=120.0, conviction=0.7,
        thesis="Demand is inflecting.", falsifiers=("Residual return turns negative.",),
        cited_document_ids=("ACME.news",),
    )
    body.update(overrides)
    return json.dumps(body)


def test_originator_turns_text_into_a_residual_return_proposal() -> None:
    result = originate(DeterministicOriginator(), _request())
    assert result.proposal is not None
    assert result.proposal.direction == "positive"
    assert result.proposal.target == "future_residual_return"


@pytest.mark.parametrize("payload", [
    _idea(expected_move_bps=-5.0),
    _idea(thesis="Revenue grew 22 percent."),
    _idea(cited_document_ids=("OTHER.doc",)),
    _idea(ticker="ZZZ"),
    _idea(action="pass", expected_move_bps=10.0),
    '{"ticker": "ACME", "ticker": "ACME"}',
    "not json",
])
def test_originator_output_fails_closed(payload: str) -> None:
    with pytest.raises(IdeaOutputError):
        originate(_Raw(payload), _request())


def test_originator_rejects_future_sources() -> None:
    future = SealedSourceSnapshot.seal((SourceDocument(
        document_id="ACME.news", available_at="2026-07-11T00:00:00Z", text="Strong growth ahead.", roles=("news",),
    ),))
    with pytest.raises(IdeaOutputError, match="after the idea cutoff"):
        originate(DeterministicOriginator(), _request(future))


def test_leakage_guard_fails_closed_for_unknown_models() -> None:
    guard = KnowledgeCutoffGuard.from_env({"FUSION_LLM_KNOWLEDGE_CUTOFFS": "anthropic:m=2026-01-31"})
    assert guard.is_contaminated("anthropic:m", "2025-06-01T00:00:00Z", synthetic_world=False)
    assert not guard.is_contaminated("anthropic:m", "2026-03-01T00:00:00Z", synthetic_world=False)
    assert guard.is_contaminated("unknown", "2026-03-01T00:00:00Z", synthetic_world=False)
    assert not guard.is_contaminated("unknown", "2020-01-01T00:00:00Z", synthetic_world=True)
    assert not guard.is_contaminated("fusionfinance-offline-x", "2020-01-01T00:00:00Z", synthetic_world=False)


def test_reliability_ledger_shrinks_conviction_toward_track_record() -> None:
    ledger = ReliabilityLedger(prior_strength=10)
    assert ledger.calibrated_probability("m", "c", 0.8) == pytest.approx(0.8)
    calls = [ResolvedCall("m", "c", 1, 0.8, (0.0,), -0.01, 1) for _ in range(30)]
    calls.append(ResolvedCall("m", "c", 1, 0.8, (0.0,), 0.01, 1, contaminated=True))
    ledger.update(calls)
    assert ledger.observations("m", "c") == 30
    assert ledger.calibrated_probability("m", "c", 0.8) == pytest.approx(8 / 40)


def test_meta_labeler_learns_when_the_llm_side_is_wrong() -> None:
    calls = [
        ResolvedCall("m", "c", 1, 0.7, (x,), 0.01 if x < 0 else -0.01, 1)
        for x in [i / 10 - 2.0 for i in range(40)]
    ]
    meta = MetaLabeler(min_samples=20).fit(calls)
    assert meta.active
    assert meta.predict((-1.5,)) > 0.7 > 0.3 > meta.predict((1.5,))
    assert MetaLabeler(min_samples=100).fit(calls).predict((0.0,)) is None


def test_risk_book_respects_limits_and_drawdown_brake() -> None:
    book = RiskBook(max_position_weight=0.1, max_gross=0.25, max_net=0.1)
    ideas = [SizedIdea(t, 1, 0.9) for t in ("A", "B", "C", "D")] + [SizedIdea("E", -1, 0.9)]
    weights = book.targets(ideas)
    assert max(abs(w) for w in weights.values()) <= 0.1
    assert sum(abs(w) for w in weights.values()) <= 0.25
    assert abs(sum(weights.values())) <= 0.1 + 1e-9
    assert book.weight(SizedIdea("A", 1, 0.45)) == 0.0
    braked = RiskBook().targets([SizedIdea("A", 1, 0.6)], drawdown=-0.2)
    assert braked["A"] == pytest.approx(RiskBook().weight(SizedIdea("A", 1, 0.6)) / 2)


def test_ledger_hash_chain_detects_tampering(tmp_path: Path) -> None:
    ledger = DecisionLedger()
    ledger.append({"ticker": "A", "stage": "approved"})
    ledger.append({"ticker": "B", "stage": "vetoed"})
    ledger.write_jsonl(tmp_path / "l.jsonl")
    assert DecisionLedger.read_jsonl(tmp_path / "l.jsonl").head == ledger.head
    ledger.entries[0]["stage"] = "vetoed"
    with pytest.raises(ValueError, match="does not match"):
        ledger.verify()


def test_record_replay_cache_is_content_addressed(tmp_path: Path) -> None:
    recorder = RecordReplayProvider(DeterministicOriginator(), tmp_path, mode="record")
    first = recorder.originate(_request())
    replay = RecordReplayProvider(DeterministicOriginator(), tmp_path, mode="replay")
    assert replay.originate(_request()) == first and replay.hits == 1
    with pytest.raises(CacheMiss):
        replay.originate(_request(_snapshot("Margins deteriorated and guidance was cut.")))


def _approved_receipt():
    proposal = originate(DeterministicOriginator(), _request()).proposal
    receipt = FusionOrchestrator(provider=DeterministicOfflineProvider()).run(proposal, _snapshot())
    assert receipt.decision == "approved"
    return receipt


def _forecast(p_adverse: float, ood: float = 0.2) -> dict:
    return {
        "expected_residual_bps": {"5d": 50.0}, "p_adverse": {"5d": p_adverse},
        "epistemic_var": {"5d": 0.0}, "aleatoric_var": {"5d": 0.0005}, "epistemic_mi": {"5d": 0.0},
        "out_of_distribution_score": ood,
    }


def test_ml_gate_vetoes_contradiction_and_abstains_out_of_distribution() -> None:
    receipt = _approved_receipt()
    assert verify_idea(receipt, forecast=_forecast(0.8), p_llm=0.7, p_meta=None).decision == "vetoed"
    assert verify_idea(receipt, forecast=_forecast(0.4, ood=0.99), p_llm=0.7, p_meta=None).decision == "abstain"
    assert verify_idea(receipt, forecast=None, p_llm=0.7, p_meta=None).decision == "abstain"
    assert verify_idea(receipt, forecast=_forecast(0.45), p_llm=0.7, p_meta=0.2).decision == "vetoed"
    approved = verify_idea(receipt, forecast=_forecast(0.4), p_llm=0.7, p_meta=None)
    assert approved.decision == "approved" and approved.p_final > 0.7
    assert approved.strict_adjudication in {"approved", "inconclusive", "research_only", "abstain"}


def test_fund_records_every_stage_in_the_ledger() -> None:
    fund = LLMFirstFund(originator=DeterministicOriginator(), analyst=DeterministicOfflineProvider(),
                        thresholds=GateThresholds())
    passed = fund.evaluate(ticker="ACME", as_of=AS_OF, snapshot=_snapshot("Trading was orderly."),
                           forecast=None, market_state=None)
    traded = fund.evaluate(ticker="ACME", as_of=AS_OF, snapshot=_snapshot(), forecast=_forecast(0.3),
                           market_state=(0.0,) * 9)
    assert passed.stage == "pass" and traded.stage == "approved"
    assert fund.size([passed, traded])["ACME"] > 0
    fund.ledger.verify()
    assert [e["stage"] for e in fund.ledger.entries] == ["pass", "approved"]


def test_claude_provider_uses_structured_outputs_and_handles_refusals() -> None:
    class Block:
        type = "text"
        text = _idea()

    class Response:
        stop_reason = "end_turn"
        content = [Block()]

    class Messages:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return Response()

    class Client:
        def __init__(self):
            self.beta = type("B", (), {"messages": Messages()})()

    client = Client()
    provider = ClaudeProvider(client=client)
    result = originate(provider, _request())
    assert result.proposal is not None
    sent = client.beta.messages.kwargs
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_config"]["format"]["schema"] is IDEA_SCHEMA
    assert sent["extra_body"] == {"fallbacks": "default"}
    Response.stop_reason = "refusal"
    with pytest.raises(IdeaOutputError):
        originate(provider, _request())
    assert ANALYST_SCHEMA["additionalProperties"] is False
    OriginatorIdea.model_validate_json(_idea())


@pytest.mark.parametrize("snapshots", ["dossier", "narrative"])
def test_backtest_runs_all_arms_through_the_shared_kernel(snapshots: str) -> None:
    config = BacktestConfig(
        world=WorldConfig(n_names=24, n_sessions=360, event_rate=0.04, seed=3),
        train_sessions=50, retrain_every=40, gbm_iters=10, snapshots=snapshots,
    )
    report, fund = run_backtest(config)
    assert set(report.arms) == set(ARMS)
    for metrics in report.arms.values():
        assert metrics["periods"] > 0 and metrics["max_drawdown"] <= 0.0
    assert report.ideas["originated"] > 0
    assert report.ledger_entries == len(fund.ledger.entries)
    fund.ledger.verify()
    payload = report.to_dict()
    assert payload["claim_status"] == "synthetic_mechanism_test_not_market_evidence"
    assert "fusion" in report.table()


def test_ml_gate_defers_to_a_quorate_committee() -> None:
    from alpha.verifier.committee import ConsensusVote

    receipt = _approved_receipt()
    veto = ConsensusVote("ACME", 1, "veto", 0.4, 0.2, 0.8, 5, 3, reasons=("COMMITTEE_VETO p_side=0.40",))
    support = ConsensusVote("ACME", 1, "support", 0.58, 0.9, 0.1, 5, 3, reasons=("COMMITTEE_SUPPORT",))
    no_quorum = ConsensusVote("ACME", 1, "no_quorum", 0.5, 0.0, 0.0, 1, 1)
    vetoed = verify_idea(receipt, forecast=_forecast(0.3), p_llm=0.7, p_meta=None, committee=veto)
    assert vetoed.decision == "vetoed" and vetoed.committee["decision"] == "veto"
    approved = verify_idea(receipt, forecast=_forecast(0.8), p_llm=0.7, p_meta=None, committee=support)
    assert approved.decision == "approved" and "COMMITTEE+LLM_TRACK_RECORD" in approved.reasons
    fallback = verify_idea(receipt, forecast=_forecast(0.8), p_llm=0.7, p_meta=None, committee=no_quorum)
    assert fallback.decision == "vetoed", "without quorum the single market head decides"
    with pytest.raises(ValueError, match="other side"):
        verify_idea(receipt, forecast=None, p_llm=0.7, p_meta=None,
                    committee=ConsensusVote("ACME", -1, "support", 0.6, 1.0, 0.0, 5, 3))
