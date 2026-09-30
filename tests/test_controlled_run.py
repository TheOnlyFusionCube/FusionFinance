from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from alpha.agents.models import (
    ANALYST_ROLES,
    DecisionReceipt,
    NumericValue,
    SealedSourceSnapshot,
    SourceDocument,
    TradeProposal,
)
from alpha.agents.orchestrator import FusionOrchestrator
from alpha.agents.providers import DeterministicOfflineProvider
from alpha.verifier.policy import PolicyThresholds
from demo.contracts import ExperimentConfig
from demo.controlled import (
    ArmInput,
    MarketVerification,
    benchmark_marks_from_closes,
    load_locked_config,
    load_policy_thresholds,
    locked_software_tape,
    locked_weekday_sessions,
    run_controlled_arm,
    run_three_arms,
)
from demo.execution import (
    assert_post_cost_bound,
    reconcile_simulation,
    simulate_portfolio,
)

CALIBRATION = "ab" * 32


def _config() -> dict:
    return {
        "schema_version": "1.0",
        "experiment_id": "fair-race-test",
        "start_date": "2027-03-02",
        "end_date": "2027-03-04",
        "starting_capital": 10_000.0,
        "universe": ["AAA", "BBB"],
        "benchmark_ticker": "SPY",
        "rebalance_frequency_sessions": 5,
        "execution_lag_sessions": 1,
        "transaction_cost_bps": 5.0,
        "slippage_bps": 2.0,
        "max_gross_leverage": 1.0,
        "max_position_weight": 0.60,
        "annualization_sessions": 252,
        "annual_risk_free_rate": 0.0,
    }


def _config_model(**updates: object) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**_config(), **updates})


def _sessions(prices: tuple[float, float, float] = (100.0, 110.0, 121.0)):
    from demo.contracts import AssetBar, MarketSession

    dates = ("2027-03-02", "2027-03-03", "2027-03-04")
    return tuple(
        MarketSession(
            session=session,
            bars=(
                AssetBar(ticker="AAA", open=price, close=price),
                AssetBar(ticker="BBB", open=price, close=price),
                AssetBar(ticker="SPY", open=price, close=price + 1.0),
            ),
        )
        for session, price in zip(dates, prices, strict=True)
    )


def _marks(config: ExperimentConfig, sessions):
    return benchmark_marks_from_closes(
        sessions,
        benchmark_ticker=config.benchmark_ticker,
        starting_capital=config.starting_capital,
    )


def _proposal() -> TradeProposal:
    return TradeProposal(
        ticker="AAA",
        as_of="2026-07-10T21:00:00Z",
        direction="positive",
        horizon_days=10,
        expected_move_bps=120.0,
        confidence=0.8,
        claim_type="near_term_catalyst",
        max_position_weight=0.1,
    )


def _snapshot(*, risk_text: str | None = None) -> SealedSourceSnapshot:
    text = "Revenue growth improved and management raised guidance."
    documents = tuple(
        SourceDocument(
            document_id=f"{role}.source",
            available_at="2026-07-10T18:00:00Z",
            text=risk_text if role == "risk" and risk_text else text,
            roles=(role,),
            numeric_values=(NumericValue(key="growth_pct", value=12.0),),
        )
        for role in ANALYST_ROLES
    )
    return SealedSourceSnapshot.seal(documents)


def _receipt(*, risk_text: str | None = None) -> DecisionReceipt:
    return FusionOrchestrator(provider=DeterministicOfflineProvider()).run(
        _proposal(), _snapshot(risk_text=risk_text)
    )


def _thresholds() -> PolicyThresholds:
    return PolicyThresholds(
        calibrated=True,
        calibration_hash=CALIBRATION,
        horizon_key="10d",
    )


def _market(receipt: DecisionReceipt, *, residual: float = 80.0) -> MarketVerification:
    assert receipt.thesis is not None
    committed = datetime.fromisoformat(receipt.thesis.committed_at)
    produced = (committed + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    return MarketVerification(
        produced_at=produced,
        expected_residual_bps={"10d": residual},
        p_adverse={"10d": 0.2},
        out_of_distribution_score=0.1,
        calibration_hash=CALIBRATION,
        fundamental_confirm_prob=0.9,
    )


def _outcome() -> str:
    return "2027-03-03T21:00:00Z"


def test_pure_ml_hashes_are_stable_and_ledger_costs_are_populated() -> None:
    config = _config_model()
    sessions = _sessions()
    marks = _marks(config, sessions)
    candidate = ArmInput(
        strategy_id="pure_ml",
        decision_session="2027-03-02",
        ticker="AAA",
        structured_weight=0.20,
    )

    first = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="pure_ml",
        candidates=(candidate,),
        benchmark_marks=marks,
    )
    second = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="pure_ml",
        candidates=(candidate,),
        benchmark_marks=marks,
    )

    assert first.claim_status == "controlled_software_ledger"
    assert first.comparable_performance_claim is False
    assert first.block_reason is None
    assert first.lineage.experiment_hash == second.lineage.experiment_hash
    assert len(first.lineage.config_hash) == 64
    assert len(first.lineage.tape_hash) == 64
    assert first.lineage.proposals[0].admitted is True
    assert first.ledger is not None and first.metrics is not None
    assert first.ledger.total_turnover == pytest.approx(0.20)
    assert first.ledger.transaction_costs == pytest.approx(first.metrics.transaction_costs)
    assert first.ledger.transaction_costs > 0.0
    assert first.metrics.total_turnover is not None
    assert first.metrics.benchmark_sessions == (
        date(2027, 3, 2),
        date(2027, 3, 3),
        date(2027, 3, 4),
    )
    assert first.ledger.benchmark_sessions == first.metrics.benchmark_sessions
    assert first.ledger.post_cost_within_limit is True

    slipped = run_controlled_arm(
        config=config.model_copy(update={"slippage_bps": 9.0}),
        sessions=sessions,
        strategy_id="pure_ml",
        candidates=(candidate,),
        benchmark_marks=marks,
    )
    repriced = run_controlled_arm(
        config=config,
        sessions=_sessions((100.0, 110.0, 130.0)),
        strategy_id="pure_ml",
        candidates=(candidate,),
        benchmark_marks=_marks(config, _sessions((100.0, 110.0, 130.0))),
    )
    assert slipped.lineage.config_hash != first.lineage.config_hash
    assert repriced.lineage.tape_hash != first.lineage.tape_hash
    assert repriced.lineage.experiment_hash != first.lineage.experiment_hash


def test_pure_ml_matches_the_shared_kernel_and_refuses_agent_inputs() -> None:
    from demo.contracts import PositionTarget, WeightProposal

    config = _config_model()
    sessions = _sessions()
    controlled = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="pure_ml",
        candidates=(
            ArmInput(
                strategy_id="pure_ml",
                decision_session="2027-03-02",
                ticker="AAA",
                structured_weight=0.20,
            ),
            ArmInput(
                strategy_id="pure_ml",
                decision_session="2027-03-02",
                ticker="BBB",
                structured_weight=-0.20,
            ),
        ),
        benchmark_marks=_marks(config, sessions),
    )
    direct = simulate_portfolio(
        config=config,
        sessions=sessions,
        proposals=(
            WeightProposal(
                strategy_id="pure_ml",
                decision_session="2027-03-02",
                targets=(
                    PositionTarget(ticker="AAA", weight=0.20),
                    PositionTarget(ticker="BBB", weight=-0.20),
                ),
            ),
        ),
        strategy_id="pure_ml",
    )

    assert controlled.result is not None
    assert controlled.result.portfolio_values == direct.portfolio_values
    assert controlled.ledger is not None
    assert controlled.ledger.trade_count == 2

    with pytest.raises(ValueError, match="pure_ml cannot take"):
        run_controlled_arm(
            config=config,
            sessions=sessions,
            strategy_id="pure_ml",
            candidates=(
                ArmInput(
                    strategy_id="pure_ml",
                    decision_session="2027-03-02",
                    ticker="AAA",
                    structured_weight=0.10,
                    receipt=_receipt(),
                ),
            ),
            benchmark_marks=_marks(config, sessions),
        )


def test_limit_breaches_block_before_execution_and_still_emit_hashes() -> None:
    config = _config_model()
    sessions = _sessions()
    blocked = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="pure_ml",
        candidates=(
            ArmInput(
                strategy_id="pure_ml",
                decision_session="2027-03-02",
                ticker="AAA",
                structured_weight=0.50,
            ),
            ArmInput(
                strategy_id="pure_ml",
                decision_session="2027-03-02",
                ticker="BBB",
                structured_weight=-0.50,
            ),
        ),
        benchmark_marks=_marks(config, sessions),
    )

    assert blocked.result is None
    assert blocked.metrics is None
    assert blocked.block_reason == "post-cost gross leverage exceeds the locked limit"
    assert len(blocked.lineage.experiment_hash) == 64
    assert blocked.lineage.proposals[0].admitted is False
    assert blocked.admitted_count == 0

    gross = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="pure_ml",
        candidates=(
            ArmInput(
                strategy_id="pure_ml",
                decision_session="2027-03-02",
                ticker="AAA",
                structured_weight=0.60,
            ),
            ArmInput(
                strategy_id="pure_ml",
                decision_session="2027-03-02",
                ticker="BBB",
                structured_weight=0.60,
            ),
        ),
        benchmark_marks=_marks(config, sessions),
    )
    assert gross.block_reason == "gross leverage exceeds the locked limit"
    assert gross.lineage.config_hash == blocked.lineage.config_hash


def test_kernel_records_post_cost_leverage_and_reconciles_fills() -> None:
    from demo.contracts import PositionTarget, WeightProposal

    config = _config_model(start_date="2026-02-02", end_date="2026-02-04")
    sessions = _legacy_sessions()
    result = simulate_portfolio(
        config=config,
        sessions=sessions,
        proposals=(
            WeightProposal(
                strategy_id="fusion",
                decision_session="2026-02-02",
                targets=(
                    PositionTarget(ticker="AAA", weight=0.50),
                    PositionTarget(ticker="BBB", weight=-0.50),
                ),
            ),
        ),
        strategy_id="fusion",
    )
    ledger = reconcile_simulation(result, config=config, sessions=sessions)

    assert ledger.total_turnover == pytest.approx(1.0)
    assert ledger.transaction_costs == pytest.approx(7.0)
    assert ledger.trade_count == 2
    assert ledger.max_post_cost_gross_leverage > config.max_gross_leverage
    assert ledger.post_cost_within_limit is False
    assert result.rebalances[0].post_cost_gross_leverage == pytest.approx(
        ledger.max_post_cost_gross_leverage
    )

    tampered = result.model_copy(
        update={
            "rebalances": (
                result.rebalances[0].model_copy(
                    update={"transaction_cost": result.rebalances[0].transaction_cost + 1.0}
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="transaction cost"):
        reconcile_simulation(tampered, config=config, sessions=sessions)
    with pytest.raises(ValueError, match="post-cost gross leverage"):
        assert_post_cost_bound(post_gross=1.5, cost_fraction=0.0, max_gross_leverage=1.0)


def test_date_bound_benchmark_rejects_a_shifted_calendar() -> None:
    from demo.contracts import BenchmarkMark
    from demo.execution import simulate_portfolio
    from demo.metrics import compute_performance_metrics

    config = _config_model()
    sessions = _sessions()
    result = simulate_portfolio(
        config=config, sessions=sessions, proposals=(), strategy_id="pure_ml"
    )
    marks = _marks(config, sessions)
    shifted = (
        BenchmarkMark(session=date(2027, 3, 1), value=marks[0].value),
        *marks[1:],
    )

    bound = compute_performance_metrics(result, config=config, benchmark_marks=marks)
    assert bound.benchmark_sessions == tuple(point.session for point in result.points)
    with pytest.raises(ValueError, match="bind to portfolio session dates"):
        compute_performance_metrics(result, config=config, benchmark_marks=shifted)
    with pytest.raises(ValueError, match="bind to portfolio session dates"):
        run_controlled_arm(
            config=config,
            sessions=sessions,
            strategy_id="pure_ml",
            candidates=(),
            benchmark_marks=shifted,
        )


def test_fusion_without_a_prospective_seal_emits_a_cash_ledger() -> None:
    config = _config_model(
        start_date="2026-07-10",
        end_date="2026-07-14",
    )
    sessions = _dated_sessions(("2026-07-10", "2026-07-13", "2026-07-14"))
    receipt = _receipt()
    run = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="fusion",
        candidates=(
            ArmInput(
                strategy_id="fusion",
                decision_session="2026-07-10",
                ticker="AAA",
                receipt=receipt,
                market=_market(receipt),
                outcome_ts="2026-07-13T21:00:00Z",
            ),
        ),
        benchmark_marks=_marks(config, sessions),
        thresholds=_thresholds(),
    )

    assert run.block_reason is None
    assert run.admitted_count == 0
    assert run.ledger is not None and run.metrics is not None
    assert run.ledger.total_turnover == 0.0
    assert run.ledger.transaction_costs == 0.0
    assert run.metrics.transaction_costs == 0.0
    assert run.lineage.proposals[0].proposal_hash == receipt.proposal_hash
    assert run.lineage.proposals[0].receipt_hash == receipt.receipt_hash
    assert run.lineage.proposals[0].reason == "prospective seal required"
    assert run.claim_status != "provisional_uncontrolled_legacy_race"
    assert len(run.lineage.experiment_hash) == 64


def test_prospective_fusion_reaches_execution_and_shares_the_ml_book() -> None:
    config = _config_model()
    sessions = _sessions()
    marks = _marks(config, sessions)
    receipt = _receipt()
    assert receipt.decision == "approved"
    cap = receipt.provisional_weight_cap
    fusion = ArmInput(
        strategy_id="fusion",
        decision_session="2027-03-02",
        ticker="AAA",
        receipt=receipt,
        market=_market(receipt),
        outcome_ts=_outcome(),
    )
    pure_ml = ArmInput(
        strategy_id="pure_ml",
        decision_session="2027-03-02",
        ticker="AAA",
        structured_weight=cap,
    )
    arms = run_three_arms(
        config=config,
        sessions=sessions,
        arms={"fusion": (fusion,), "pure_ml": (pure_ml,), "pure_llm": ()},
        benchmark_marks=marks,
        thresholds=_thresholds(),
    )
    by_name = {run.lineage.strategy_id: run for run in arms}

    assert by_name["pure_ml"].lineage.config_hash == by_name["fusion"].lineage.config_hash
    assert by_name["pure_ml"].lineage.tape_hash == by_name["fusion"].lineage.tape_hash
    assert by_name["fusion"].lineage.experiment_hash != by_name["pure_ml"].lineage.experiment_hash
    assert by_name["fusion"].admitted_count == 1
    assert by_name["fusion"].lineage.proposals[0].verifier_decision == "approved"
    assert by_name["fusion"].result is not None and by_name["pure_ml"].result is not None
    assert (
        by_name["fusion"].result.portfolio_values
        == by_name["pure_ml"].result.portfolio_values
    )
    assert by_name["pure_llm"].ledger is not None
    assert by_name["pure_llm"].ledger.total_turnover == 0.0
    assert by_name["pure_llm"].ledger.transaction_costs == 0.0
    assert all(run.comparable_performance_claim is False for run in arms)


def test_fusion_gates_fail_closed_without_moving_capital() -> None:
    config = _config_model()
    sessions = _sessions()
    marks = _marks(config, sessions)
    receipt = _receipt()
    vetoed = _receipt(
        risk_text="A fraud investigation creates bankruptcy and liquidity crisis risk."
    )

    missing_market = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="fusion",
        candidates=(
            ArmInput(
                strategy_id="fusion",
                decision_session="2027-03-02",
                ticker="AAA",
                receipt=receipt,
                outcome_ts=_outcome(),
            ),
        ),
        benchmark_marks=marks,
        thresholds=_thresholds(),
    )
    uncalibrated = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="fusion",
        candidates=(
            ArmInput(
                strategy_id="fusion",
                decision_session="2027-03-02",
                ticker="AAA",
                receipt=receipt,
                market=_market(receipt),
                outcome_ts=_outcome(),
            ),
        ),
        benchmark_marks=marks,
    )
    disagreed = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="fusion",
        candidates=(
            ArmInput(
                strategy_id="fusion",
                decision_session="2027-03-02",
                ticker="AAA",
                receipt=receipt,
                market=_market(receipt, residual=-80.0),
                outcome_ts=_outcome(),
            ),
        ),
        benchmark_marks=marks,
        thresholds=_thresholds(),
    )
    veto = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="fusion",
        candidates=(
            ArmInput(
                strategy_id="fusion",
                decision_session="2027-03-02",
                ticker="AAA",
                receipt=vetoed,
                market=_market(vetoed),
                outcome_ts=_outcome(),
            ),
        ),
        benchmark_marks=marks,
        thresholds=_thresholds(),
    )

    assert missing_market.lineage.proposals[0].reason == "market verifier required"
    assert "calibration" in uncalibrated.lineage.proposals[0].reason
    assert disagreed.lineage.proposals[0].verifier_decision == "research_only"
    assert veto.lineage.proposals[0].precheck_decision == "reject"
    assert veto.lineage.proposals[0].reason == "RISK_VETO"
    for run in (missing_market, uncalibrated, disagreed, veto):
        assert run.admitted_count == 0
        assert run.ledger is not None
        assert run.ledger.total_turnover == 0.0
        assert run.ledger.transaction_costs == 0.0
        assert run.metrics is not None
        assert run.comparable_performance_claim is False


def test_pure_llm_refuses_verifier_input_and_trades_only_when_prospective() -> None:
    config = _config_model()
    sessions = _sessions()
    receipt = _receipt()
    with pytest.raises(ValueError, match="market-verifier forecast"):
        run_controlled_arm(
            config=config,
            sessions=sessions,
            strategy_id="pure_llm",
            candidates=(
                ArmInput(
                    strategy_id="pure_llm",
                    decision_session="2027-03-02",
                    ticker="AAA",
                    receipt=receipt,
                    market=_market(receipt),
                    outcome_ts=_outcome(),
                ),
            ),
            benchmark_marks=_marks(config, sessions),
        )

    traded = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="pure_llm",
        candidates=(
            ArmInput(
                strategy_id="pure_llm",
                decision_session="2027-03-02",
                ticker="AAA",
                receipt=receipt,
                outcome_ts=_outcome(),
            ),
        ),
        benchmark_marks=_marks(config, sessions),
    )
    assert traded.admitted_count == 1
    assert traded.ledger is not None
    assert traded.ledger.transaction_costs > 0.0
    assert traded.lineage.proposals[0].market_input_hash is None
    assert traded.lineage.proposals[0].verifier_decision is None


def test_locked_config_can_execute_and_legacy_metrics_stay_provisional() -> None:
    import json
    from pathlib import Path

    config = load_locked_config()
    names = (*config.universe, config.benchmark_ticker)
    sessions = _bars(config.start_date, config.end_date, names)
    run = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="pure_ml",
        candidates=(
            ArmInput(
                strategy_id="pure_ml",
                decision_session=config.start_date,
                ticker="AAPL",
                structured_weight=0.05,
            ),
        ),
        benchmark_marks=_marks(config, sessions),
    )
    legacy = json.loads(
        (Path(__file__).resolve().parents[1] / "results" / "demo_run.json").read_text(
            encoding="utf-8"
        )
    )

    assert config.experiment_id == "fusionfinance-fair-race-v1"
    assert run.block_reason is None
    assert run.ledger is not None and run.metrics is not None
    assert run.ledger.transaction_costs > 0.0
    assert run.ledger.total_turnover > 0.0
    assert run.metrics.benchmark_sessions == (config.start_date, config.end_date)
    assert run.claim_status != legacy["claim_status"]
    assert legacy["claim_status"] == "provisional_uncontrolled_legacy_race"


_FROZEN_COMMITTED_AT = "2026-02-02T20:00:00Z"
_FUSION_PRODUCED_AT = "2026-02-02T20:00:01Z"
_ENDPOINT_AS_OF = "2026-02-02T18:00:00Z"
_ENDPOINT_AVAILABLE_AT = "2026-02-02T15:00:00Z"
_ENDPOINT_OUTCOME_TS = "2026-07-09T21:00:00Z"
_SOFTWARE_LEDGER = Path("results/controlled_software_ledger.json")
_PRECHECK_RECEIPT = Path("results/controlled_precheck_receipt.json")
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


def desk_endpoint_receipt() -> DecisionReceipt:
    """Seal one precheck in the orchestrator, before the locked outcome."""

    proposal = TradeProposal(
        ticker="AAPL",
        as_of=_ENDPOINT_AS_OF,
        direction="positive",
        horizon_days=10,
        expected_move_bps=120.0,
        confidence=0.8,
        claim_type="near_term_catalyst",
        max_position_weight=0.1,
    )
    text = "Revenue growth improved and management raised guidance."
    documents = tuple(
        SourceDocument(
            document_id=f"{role}.source",
            available_at=_ENDPOINT_AVAILABLE_AT,
            text=text,
            roles=(role,),
            numeric_values=(NumericValue(key="growth_pct", value=12.0),),
        )
        for role in ANALYST_ROLES
    )
    receipt = FusionOrchestrator(
        provider=DeterministicOfflineProvider(),
        committed_at=_FROZEN_COMMITTED_AT,
    ).run(proposal, SealedSourceSnapshot.seal(documents))
    receipt.verify_receipt()
    assert receipt.thesis is not None
    assert receipt.thesis.committed_at == _FROZEN_COMMITTED_AT
    assert receipt.thesis.is_prospective(_ENDPOINT_OUTCOME_TS)
    committed = datetime.fromisoformat(receipt.thesis.committed_at)
    outcome = datetime.fromisoformat(_ENDPOINT_OUTCOME_TS)
    assert committed < outcome
    return receipt


def sealed_endpoint_ledger() -> dict[str, object]:
    """Prospective pure-LLM ledger on the locked window's two endpoints."""

    config = load_locked_config()
    names = (*config.universe, config.benchmark_ticker)
    sessions = _bars(config.start_date, config.end_date, names)
    receipt = desk_endpoint_receipt()
    assert receipt.thesis is not None
    marks = _marks(config, sessions)
    pure_llm = _execute_receipt(config, sessions, marks, receipt, strategy_id="pure_llm")
    fusion = _execute_receipt(config, sessions, marks, receipt, strategy_id="fusion")
    assert pure_llm.block_reason is None and pure_llm.ledger is not None
    assert fusion.ledger is not None
    row = pure_llm.lineage.proposals[0]
    fusion_row = fusion.lineage.proposals[0]
    ledger = pure_llm.ledger
    assert row.receipt_hash == receipt.receipt_hash
    assert fusion_row.receipt_hash == receipt.receipt_hash
    return {
        "schema": "fusionfinance-controlled-software-ledger-v1",
        "claim_status": pure_llm.claim_status,
        "comparable_performance_claim": pure_llm.comparable_performance_claim,
        "description": (
            "Two-session software ledger on the locked window endpoints "
            "2026-02-02 and 2026-07-09. AgentDesk and FusionOrchestrator seal "
            "the pure-LLM precheck before the outcome. Fusion sees that same "
            "receipt and abstains without a calibration artifact. This is not "
            "a full calendar and not a performance claim."
        ),
        "provenance": {
            "desk": "alpha.agents.desk.AgentDesk",
            "orchestrator": "alpha.agents.orchestrator.FusionOrchestrator",
            "provider": receipt.provider_model,
            "stage": receipt.stage,
            "seal": "orchestrator_first_commit",
            "receipt_path": _PRECHECK_RECEIPT.as_posix(),
        },
        "strategy_id": pure_llm.lineage.strategy_id,
        "experiment_id": pure_llm.lineage.experiment_id,
        "decision_session": config.start_date.isoformat(),
        "outcome_ts": _ENDPOINT_OUTCOME_TS,
        "committed_at": receipt.thesis.committed_at,
        "config_hash": pure_llm.lineage.config_hash,
        "tape_hash": pure_llm.lineage.tape_hash,
        "lineage_hash": pure_llm.lineage.lineage_hash,
        "experiment_hash": pure_llm.lineage.experiment_hash,
        "proposal_hash": row.proposal_hash,
        "receipt_hash": row.receipt_hash,
        "thesis_hash": row.thesis_hash,
        "precheck_decision": row.precheck_decision,
        "admitted": row.admitted,
        "target_weight": row.target_weight,
        "reason": row.reason,
        "session_count": ledger.session_count,
        "trade_count": ledger.trade_count,
        "total_turnover": ledger.total_turnover,
        "transaction_costs": ledger.transaction_costs,
        "slippage_costs": ledger.slippage_costs,
        "post_cost_within_limit": ledger.post_cost_within_limit,
        "benchmark_sessions": [
            day.isoformat() for day in ledger.benchmark_sessions
        ],
        "fusion": {
            "strategy_id": "fusion",
            "admitted": fusion_row.admitted,
            "reason": fusion_row.reason,
            "precheck_decision": fusion_row.precheck_decision,
            "verifier_decision": fusion_row.verifier_decision,
            "receipt_hash": fusion_row.receipt_hash,
            "experiment_hash": fusion.lineage.experiment_hash,
            "total_turnover": fusion.ledger.total_turnover,
            "transaction_costs": fusion.ledger.transaction_costs,
            "comparable_performance_claim": fusion.comparable_performance_claim,
        },
    }


def _execute_receipt(config, sessions, marks, receipt: DecisionReceipt, *, strategy_id: str):
    market = None
    if strategy_id == "fusion":
        market = MarketVerification(
            produced_at=_FUSION_PRODUCED_AT,
            expected_residual_bps={"10d": 80.0},
            p_adverse={"10d": 0.2},
            out_of_distribution_score=0.1,
            fundamental_confirm_prob=0.9,
        )
    return run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id=strategy_id,
        candidates=(
            ArmInput(
                strategy_id=strategy_id,
                decision_session=config.start_date,
                ticker="AAPL",
                receipt=receipt,
                market=market,
                outcome_ts=_ENDPOINT_OUTCOME_TS,
            ),
        ),
        benchmark_marks=marks,
    )


def _json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _mapping_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _mapping_keys(item)


def test_orchestrator_seals_the_commit_clock_before_the_outcome() -> None:
    receipt = desk_endpoint_receipt()
    assert receipt.provider_model == "fusionfinance-offline-lexical-v1"
    assert tuple(report.role for report in receipt.reports) == ANALYST_ROLES
    assert receipt.thesis is not None
    with pytest.raises(ValueError, match="already committed"):
        receipt.thesis.commit(model_version="x", prompt_hash="y" * 8)
    with pytest.raises(ValueError, match="timezone"):
        FusionOrchestrator(
            provider=DeterministicOfflineProvider(),
            committed_at="2026-02-02T20:00:00",
        )


def test_checked_in_software_ledger_matches_the_sealed_endpoint_run() -> None:
    root = Path(__file__).resolve().parents[1]
    ledger_path = root / _SOFTWARE_LEDGER
    receipt_path = root / _PRECHECK_RECEIPT
    receipt = desk_endpoint_receipt()
    ledger_text = _json_text(sealed_endpoint_ledger())
    receipt_text = _json_text(receipt.model_dump(mode="json"))
    document = json.loads(ledger_text)
    checked = json.loads(ledger_path.read_text(encoding="utf-8"))
    checked_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert ledger_path.read_text(encoding="utf-8") == ledger_text
    assert receipt_path.read_text(encoding="utf-8") == receipt_text
    assert checked == document
    assert checked_receipt == json.loads(receipt_text)
    loaded = DecisionReceipt.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    loaded.verify_receipt()
    assert loaded.thesis is not None
    assert loaded.thesis.is_prospective(checked["outcome_ts"])
    assert datetime.fromisoformat(loaded.thesis.committed_at) < datetime.fromisoformat(
        checked["outcome_ts"]
    )
    assert checked["committed_at"] == loaded.thesis.committed_at
    assert checked["receipt_hash"] == loaded.receipt_hash
    assert checked["provenance"]["seal"] == "orchestrator_first_commit"
    assert checked["provenance"]["desk"].endswith("AgentDesk")
    assert checked["provenance"]["orchestrator"].endswith("FusionOrchestrator")
    assert checked["claim_status"] == "controlled_software_ledger"
    assert checked["comparable_performance_claim"] is False
    assert checked["admitted"] is True
    assert checked["fusion"]["admitted"] is False
    assert "calibration" in checked["fusion"]["reason"]
    assert checked["fusion"]["total_turnover"] == 0.0
    assert checked["fusion"]["transaction_costs"] == 0.0
    for key in ("config_hash", "tape_hash", "experiment_hash"):
        assert isinstance(checked[key], str) and len(checked[key]) == 64
    assert isinstance(checked["total_turnover"], float)
    assert isinstance(checked["transaction_costs"], float)
    assert checked["total_turnover"] > 0.0
    assert checked["transaction_costs"] > 0.0
    assert _PERFORMANCE_KEYS.isdisjoint(_mapping_keys(checked))
    assert _PERFORMANCE_KEYS.isdisjoint(_mapping_keys(checked_receipt))


_THREE_ARM_LEDGER = Path("results/controlled_three_arm_ledger.json")
_THREE_ARM_METRICS = Path("results/controlled_three_arm_metrics.json")
_CALIBRATION_ARTIFACT = Path("results/fusion_policy_calibration.json")
_FIXTURE_METRICS_STATUS = "controlled_software_fixture_metrics"
_LEGACY_METRICS_SHA256 = (
    "e7e5055ce9b4409d7941a71929f66261416b8d3c3f62423190edafd5bf5b1411"
)


def test_calibrated_policy_still_requires_a_market_head_and_ood() -> None:
    thresholds = load_policy_thresholds()
    config = _config_model()
    sessions = _sessions()
    marks = _marks(config, sessions)
    receipt = _receipt()
    missing = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="fusion",
        candidates=(
            ArmInput(
                strategy_id="fusion",
                decision_session="2027-03-02",
                ticker="AAA",
                receipt=receipt,
                outcome_ts=_outcome(),
            ),
        ),
        benchmark_marks=marks,
        thresholds=thresholds,
    )
    no_forecast = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="fusion",
        candidates=(
            ArmInput(
                strategy_id="fusion",
                decision_session="2027-03-02",
                ticker="AAA",
                receipt=receipt,
                market=MarketVerification(
                    produced_at=_market(receipt).produced_at,
                    expected_residual_bps={},
                    p_adverse={},
                    out_of_distribution_score=0.1,
                    calibration_hash=thresholds.calibration_hash,
                    fundamental_confirm_prob=0.9,
                ),
                outcome_ts=_outcome(),
            ),
        ),
        benchmark_marks=marks,
        thresholds=thresholds,
    )
    ood = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="fusion",
        candidates=(
            ArmInput(
                strategy_id="fusion",
                decision_session="2027-03-02",
                ticker="AAA",
                receipt=receipt,
                market=MarketVerification(
                    produced_at=_market(receipt).produced_at,
                    expected_residual_bps={"10d": 80.0},
                    p_adverse={"10d": 0.2},
                    out_of_distribution_score=0.95,
                    calibration_hash=thresholds.calibration_hash,
                    fundamental_confirm_prob=0.9,
                ),
                outcome_ts=_outcome(),
            ),
        ),
        benchmark_marks=marks,
        thresholds=thresholds,
    )

    assert thresholds.calibrated is True
    assert missing.lineage.proposals[0].reason == "market verifier required"
    assert missing.admitted_count == 0
    assert no_forecast.lineage.proposals[0].verifier_decision == "inconclusive"
    assert no_forecast.admitted_count == 0
    assert ood.lineage.proposals[0].verifier_decision == "abstain"
    assert "OOD" in ood.lineage.proposals[0].reason
    assert ood.admitted_count == 0


def _locked_three_arm_state() -> dict[str, object]:
    config = load_locked_config()
    sessions = locked_software_tape(config)
    dates = locked_weekday_sessions(config)
    assert tuple(session.session for session in sessions) == dates
    marks = _marks(config, sessions)
    thresholds = load_policy_thresholds()
    clock = [
        index
        for index in range(len(dates))
        if index % config.rebalance_frequency_sessions == 0
        and index + config.execution_lag_sessions < len(dates)
    ]
    pure_ml: list[ArmInput] = []
    pure_llm: list[ArmInput] = []
    fusion: list[ArmInput] = []
    seals: list[dict[str, object]] = []
    for ordinal, index in enumerate(clock):
        decision = dates[index]
        ticker = config.universe[ordinal % len(config.universe)]
        outcome = f"{dates[index + config.execution_lag_sessions].isoformat()}T21:00:00Z"
        receipt = _seal_locked_receipt(ticker, decision)
        kind = ("approve", "reject", "research_only")[ordinal % 3]
        pure_ml.append(
            ArmInput(
                strategy_id="pure_ml",
                decision_session=decision,
                ticker=ticker,
                structured_weight=0.05,
            )
        )
        pure_llm.append(
            ArmInput(
                strategy_id="pure_llm",
                decision_session=decision,
                ticker=ticker,
                receipt=receipt,
                outcome_ts=outcome,
            )
        )
        fusion.append(
            ArmInput(
                strategy_id="fusion",
                decision_session=decision,
                ticker=ticker,
                receipt=receipt,
                market=_fusion_market(
                    decision, kind, thresholds.calibration_hash
                ),
                outcome_ts=outcome,
            )
        )
        assert receipt.thesis is not None
        seals.append(
            {
                "decision_session": decision.isoformat(),
                "committed_at": receipt.thesis.committed_at,
                "outcome_ts": outcome,
                "receipt_hash": receipt.receipt_hash,
                "market_case": kind,
            }
        )
    arms = run_three_arms(
        config=config,
        sessions=sessions,
        arms={"pure_ml": pure_ml, "pure_llm": pure_llm, "fusion": fusion},
        benchmark_marks=marks,
        thresholds=thresholds,
    )
    return {
        "config": config,
        "dates": dates,
        "thresholds": thresholds,
        "clock": clock,
        "seals": seals,
        "by_name": {run.lineage.strategy_id: run for run in arms},
    }


def locked_three_arm_ledger() -> dict[str, object]:
    """Full locked-window three-arm software ledger. Not a performance claim."""

    state = _locked_three_arm_state()
    config = state["config"]
    dates = state["dates"]
    thresholds = state["thresholds"]
    clock = state["clock"]
    by_name = state["by_name"]
    return {
        "schema": "fusionfinance-controlled-three-arm-ledger-v1",
        "claim_status": "controlled_software_ledger",
        "comparable_performance_claim": False,
        "description": (
            "Software ledger for every weekday session in the locked window "
            "2026-02-02 through 2026-07-09, rebalanced every 10 sessions. "
            "LLM arms use the desk's first orchestrator seal. Fusion uses the "
            "pre-window calibration artifact and still requires a market head. "
            "Prices are deterministic software marks. This is not a "
            "performance claim."
        ),
        "window": [config.start_date.isoformat(), config.end_date.isoformat()],
        "session_count": len(dates),
        "rebalance_frequency_sessions": config.rebalance_frequency_sessions,
        "rebalance_count": len(clock),
        "calibration_hash": thresholds.calibration_hash,
        "calibration_path": _CALIBRATION_ARTIFACT.as_posix(),
        "provenance": {
            "desk": "alpha.agents.desk.AgentDesk",
            "orchestrator": "alpha.agents.orchestrator.FusionOrchestrator",
            "provider": "fusionfinance-offline-lexical-v1",
            "seal": "orchestrator_first_commit",
        },
        "seals": state["seals"],
        "arms": {
            name: _arm_ledger(by_name[name])
            for name in ("pure_ml", "pure_llm", "fusion")
        },
    }


def _seal_locked_receipt(ticker: str, decision: date) -> DecisionReceipt:
    proposal = TradeProposal(
        ticker=ticker,
        as_of=f"{decision.isoformat()}T18:00:00Z",
        direction="positive",
        horizon_days=10,
        expected_move_bps=120.0,
        confidence=0.8,
        claim_type="near_term_catalyst",
        max_position_weight=0.1,
    )
    text = "Revenue growth improved and management raised guidance."
    documents = tuple(
        SourceDocument(
            document_id=f"{role}.source",
            available_at=f"{decision.isoformat()}T15:00:00Z",
            text=text,
            roles=(role,),
            numeric_values=(NumericValue(key="growth_pct", value=12.0),),
        )
        for role in ANALYST_ROLES
    )
    receipt = FusionOrchestrator(
        provider=DeterministicOfflineProvider(),
        committed_at=f"{decision.isoformat()}T20:00:00Z",
    ).run(proposal, SealedSourceSnapshot.seal(documents))
    receipt.verify_receipt()
    assert receipt.decision == "approved"
    assert receipt.thesis is not None
    assert receipt.thesis.committed_at == f"{decision.isoformat()}T20:00:00Z"
    return receipt


def _fusion_market(decision: date, kind: str, calibration_hash: str) -> MarketVerification:
    residual = 80.0
    fundamental = 0.9
    if kind == "reject":
        fundamental = 0.1
    elif kind == "research_only":
        residual = -80.0
    return MarketVerification(
        produced_at=f"{decision.isoformat()}T20:00:01Z",
        expected_residual_bps={"10d": residual},
        p_adverse={"10d": 0.2},
        out_of_distribution_score=0.1,
        calibration_hash=calibration_hash,
        fundamental_confirm_prob=fundamental,
    )


def _arm_ledger(run) -> dict[str, object]:
    assert run.block_reason is None and run.ledger is not None
    ledger = run.ledger
    return {
        "strategy_id": run.lineage.strategy_id,
        "claim_status": run.claim_status,
        "comparable_performance_claim": run.comparable_performance_claim,
        "config_hash": run.lineage.config_hash,
        "tape_hash": run.lineage.tape_hash,
        "lineage_hash": run.lineage.lineage_hash,
        "experiment_hash": run.lineage.experiment_hash,
        "admitted_count": run.admitted_count,
        "session_count": ledger.session_count,
        "trade_count": ledger.trade_count,
        "total_turnover": ledger.total_turnover,
        "transaction_costs": ledger.transaction_costs,
        "slippage_costs": ledger.slippage_costs,
        "post_cost_within_limit": ledger.post_cost_within_limit,
        "benchmark_sessions": [day.isoformat() for day in ledger.benchmark_sessions],
        "decisions": [
            {
                "decision_session": row.decision_session.isoformat(),
                "ticker": row.ticker,
                "admitted": row.admitted,
                "target_weight": row.target_weight,
                "reason": row.reason,
                "precheck_decision": row.precheck_decision,
                "verifier_decision": row.verifier_decision,
                "receipt_hash": row.receipt_hash,
            }
            for row in run.lineage.proposals
        ],
    }


def locked_three_arm_metrics() -> dict[str, object]:
    """Fixture metrics for the locked three-arm software tape. Not a claim."""

    state = _locked_three_arm_state()
    config = state["config"]
    dates = state["dates"]
    by_name = state["by_name"]
    arms: dict[str, object] = {}
    for name in ("pure_ml", "pure_llm", "fusion"):
        run = by_name[name]
        assert run.block_reason is None
        assert run.metrics is not None and run.result is not None and run.ledger is not None
        statistics = run.metrics.model_dump(mode="json")
        values = [point.portfolio_value for point in run.result.points]
        sessions = [point.session.isoformat() for point in run.result.points]
        if statistics["trade_count"] != run.ledger.trade_count:
            raise AssertionError("metrics trade count does not match the ledger")
        if not math.isclose(
            statistics["total_turnover"],
            run.ledger.total_turnover,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise AssertionError("metrics turnover does not match the ledger")
        if not math.isclose(
            statistics["transaction_costs"],
            run.ledger.transaction_costs,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise AssertionError("metrics costs do not match the ledger")
        if not math.isclose(statistics["total_return"], values[-1] / values[0] - 1.0):
            raise AssertionError("cumulative return does not match the wealth path")
        arms[name] = {
            "strategy_id": name,
            "claim_status": _FIXTURE_METRICS_STATUS,
            "comparable_performance_claim": False,
            "config_hash": run.lineage.config_hash,
            "tape_hash": run.lineage.tape_hash,
            "lineage_hash": run.lineage.lineage_hash,
            "experiment_hash": run.lineage.experiment_hash,
            "portfolio_sessions": sessions,
            "portfolio_values": values,
            "statistics": statistics,
        }
    return {
        "schema": "fusionfinance-controlled-three-arm-metrics-v1",
        "claim_status": _FIXTURE_METRICS_STATUS,
        "comparable_performance_claim": False,
        "description": (
            "Fixture metrics from the controlled three-arm software tape. "
            "Prices are deterministic software marks, the LLM desk is the "
            "offline lexical provider, and pure ML uses a fixed 0.05 weight. "
            "Not a capital performance claim."
        ),
        "source_ledger": _THREE_ARM_LEDGER.as_posix(),
        "fixture_context": {
            "prices": "deterministic software marks",
            "llm_provider": "fusionfinance-offline-lexical-v1",
            "pure_ml_weight": 0.05,
        },
        "window": [config.start_date.isoformat(), config.end_date.isoformat()],
        "session_count": len(dates),
        "arms": arms,
    }


def test_checked_in_three_arm_ledger_covers_the_locked_window() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / _THREE_ARM_LEDGER
    rendered = _json_text(locked_three_arm_ledger())
    document = json.loads(rendered)
    checked = json.loads(path.read_text(encoding="utf-8"))
    config = load_locked_config()
    dates = [day.isoformat() for day in locked_weekday_sessions(config)]
    fusion_decisions = {
        row["verifier_decision"] for row in document["arms"]["fusion"]["decisions"]
    }

    assert path.read_text(encoding="utf-8") == rendered
    assert checked == document
    assert document["comparable_performance_claim"] is False
    assert document["window"] == ["2026-02-02", "2026-07-09"]
    assert document["session_count"] == len(dates) == 114
    assert dates[0] == "2026-02-02" and dates[-1] == "2026-07-09"
    assert document["rebalance_frequency_sessions"] == 10
    assert document["rebalance_count"] == 12
    assert document["calibration_hash"] == load_policy_thresholds().calibration_hash
    assert len(document["calibration_hash"]) == 64
    assert "policy thresholds lack a calibration artifact" not in rendered
    assert fusion_decisions == {"approved", "reject", "research_only"}
    for seal in document["seals"]:
        assert seal["committed_at"] < seal["outcome_ts"]
    for name, arm in document["arms"].items():
        assert arm["comparable_performance_claim"] is False
        assert arm["benchmark_sessions"] == dates
        assert arm["session_count"] == 114
        assert arm["post_cost_within_limit"] is True
        assert isinstance(arm["total_turnover"], float)
        assert isinstance(arm["transaction_costs"], float)
        assert arm["total_turnover"] > 0.0
        assert arm["transaction_costs"] > 0.0
        assert len(arm["config_hash"]) == 64
        assert len(arm["tape_hash"]) == 64
        assert len(arm["experiment_hash"]) == 64
        assert arm["strategy_id"] == name
    assert document["arms"]["fusion"]["admitted_count"] > 0
    assert document["arms"]["pure_llm"]["admitted_count"] == 12
    assert _PERFORMANCE_KEYS.isdisjoint(_mapping_keys(document))


def test_checked_in_fixture_metrics_match_the_three_arm_ledger() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics_path = root / _THREE_ARM_METRICS
    ledger_path = root / _THREE_ARM_LEDGER
    legacy_path = root / "results" / "metrics.json"
    rendered = _json_text(locked_three_arm_metrics())
    document = json.loads(rendered)
    checked = json.loads(metrics_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    required = (
        "sharpe_ratio",
        "annualized_return",
        "total_return",
        "annualized_volatility",
        "max_drawdown",
        "sortino_ratio",
        "total_turnover",
        "transaction_costs",
        "trade_count",
    )

    assert metrics_path.read_text(encoding="utf-8") == rendered
    assert checked == document
    assert document["claim_status"] == _FIXTURE_METRICS_STATUS
    assert document["comparable_performance_claim"] is False
    assert document["source_ledger"] == _THREE_ARM_LEDGER.as_posix()
    assert hashlib.sha256(legacy_path.read_bytes()).hexdigest() == _LEGACY_METRICS_SHA256
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert legacy["fusion"]["sharpe"] != document["arms"]["fusion"]["statistics"]["sharpe_ratio"]
    for name, arm in document["arms"].items():
        ledger_arm = ledger["arms"][name]
        statistics = arm["statistics"]
        values = arm["portfolio_values"]
        assert arm["claim_status"] == _FIXTURE_METRICS_STATUS
        assert arm["comparable_performance_claim"] is False
        assert arm["config_hash"] == ledger_arm["config_hash"]
        assert arm["tape_hash"] == ledger_arm["tape_hash"]
        assert arm["experiment_hash"] == ledger_arm["experiment_hash"]
        assert len(arm["config_hash"]) == 64
        assert len(arm["tape_hash"]) == 64
        assert len(arm["experiment_hash"]) == 64
        assert arm["portfolio_sessions"] == ledger_arm["benchmark_sessions"]
        assert len(values) == document["session_count"] == 114
        assert statistics["trade_count"] == ledger_arm["trade_count"]
        assert math.isclose(statistics["total_turnover"], ledger_arm["total_turnover"])
        assert math.isclose(
            statistics["transaction_costs"], ledger_arm["transaction_costs"]
        )
        assert math.isclose(statistics["total_return"], values[-1] / values[0] - 1.0)
        assert statistics["starting_value"] == values[0]
        assert statistics["ending_value"] == values[-1]
        for key in required:
            assert statistics[key] is not None
            assert isinstance(statistics[key], (int, float))


def _legacy_sessions():
    from demo.contracts import AssetBar, MarketSession

    rows = (
        ("2026-02-02", (100.0, 100.0), (100.0, 100.0), 100.0),
        ("2026-02-03", (100.0, 110.0), (100.0, 90.0), 101.0),
        ("2026-02-04", (110.0, 121.0), (90.0, 81.0), 102.0),
    )
    return tuple(
        MarketSession(
            session=session,
            bars=(
                AssetBar(ticker="AAA", open=aaa[0], close=aaa[1]),
                AssetBar(ticker="BBB", open=bbb[0], close=bbb[1]),
                AssetBar(ticker="SPY", open=spy, close=spy),
            ),
        )
        for session, aaa, bbb, spy in rows
    )


def _dated_sessions(dates: tuple[str, ...]):
    from demo.contracts import AssetBar, MarketSession

    return tuple(
        MarketSession(
            session=session,
            bars=(
                AssetBar(ticker="AAA", open=100.0, close=100.0),
                AssetBar(ticker="BBB", open=100.0, close=100.0),
                AssetBar(ticker="SPY", open=100.0, close=101.0),
            ),
        )
        for session in dates
    )


def _bars(start: date, end: date, names: tuple[str, ...]):
    from demo.contracts import AssetBar, MarketSession

    def session(day: date, price: float) -> MarketSession:
        return MarketSession(
            session=day,
            bars=tuple(
                AssetBar(ticker=name, open=price, close=price) for name in names
            ),
        )

    return (session(start, 100.0), session(end, 101.0))
