from __future__ import annotations

import json
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
_ENDPOINT_AS_OF = "2026-02-02T18:00:00Z"
_ENDPOINT_AVAILABLE_AT = "2026-02-02T15:00:00Z"
_ENDPOINT_OUTCOME_TS = "2026-07-09T21:00:00Z"
_SOFTWARE_LEDGER = Path("results/controlled_software_ledger.json")
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


def sealed_endpoint_ledger() -> dict[str, object]:
    """Prospective pure-LLM ledger on the locked window's two endpoints."""

    config = load_locked_config()
    names = (*config.universe, config.benchmark_ticker)
    sessions = _bars(config.start_date, config.end_date, names)
    receipt = _freeze_receipt_commit(_endpoint_receipt(), _FROZEN_COMMITTED_AT)
    assert receipt.thesis is not None
    assert receipt.thesis.is_prospective(_ENDPOINT_OUTCOME_TS)
    run = run_controlled_arm(
        config=config,
        sessions=sessions,
        strategy_id="pure_llm",
        candidates=(
            ArmInput(
                strategy_id="pure_llm",
                decision_session=config.start_date,
                ticker="AAPL",
                receipt=receipt,
                outcome_ts=_ENDPOINT_OUTCOME_TS,
            ),
        ),
        benchmark_marks=_marks(config, sessions),
    )
    assert run.block_reason is None and run.ledger is not None
    row = run.lineage.proposals[0]
    ledger = run.ledger
    return {
        "schema": "fusionfinance-controlled-software-ledger-v1",
        "claim_status": run.claim_status,
        "comparable_performance_claim": run.comparable_performance_claim,
        "description": (
            "Two-session software ledger on the locked window endpoints "
            "2026-02-02 and 2026-07-09. The pure-LLM receipt is resealed at a "
            "fixed commit time before the execution session. This is not a "
            "full calendar and not a performance claim."
        ),
        "strategy_id": run.lineage.strategy_id,
        "experiment_id": run.lineage.experiment_id,
        "decision_session": config.start_date.isoformat(),
        "outcome_ts": _ENDPOINT_OUTCOME_TS,
        "committed_at": _FROZEN_COMMITTED_AT,
        "config_hash": run.lineage.config_hash,
        "tape_hash": run.lineage.tape_hash,
        "lineage_hash": run.lineage.lineage_hash,
        "experiment_hash": run.lineage.experiment_hash,
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
    }


def _endpoint_receipt() -> DecisionReceipt:
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
    return FusionOrchestrator(provider=DeterministicOfflineProvider()).run(
        proposal, SealedSourceSnapshot.seal(documents)
    )


def _freeze_receipt_commit(
    receipt: DecisionReceipt, committed_at: str
) -> DecisionReceipt:
    assert receipt.thesis is not None and receipt.evidence_audit is not None
    thesis = receipt.thesis.model_copy(update={"committed_at": committed_at})
    thesis.verify_commit()
    sealed = DecisionReceipt.seal_evaluated(
        proposal=receipt.proposal,
        snapshot=receipt.snapshot,
        provider_model=receipt.provider_model,
        reports=receipt.reports,
        thesis=thesis,
        evidence_audit=receipt.evidence_audit,
    )
    sealed.verify_receipt()
    return sealed


def test_checked_in_software_ledger_matches_the_sealed_endpoint_run() -> None:
    path = Path(__file__).resolve().parents[1] / _SOFTWARE_LEDGER
    rendered = json.dumps(
        sealed_endpoint_ledger(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    document = json.loads(rendered)
    checked = json.loads(path.read_text(encoding="utf-8"))

    assert path.read_text(encoding="utf-8") == rendered
    assert checked == document
    assert checked["claim_status"] == "controlled_software_ledger"
    assert checked["comparable_performance_claim"] is False
    assert checked["admitted"] is True
    for key in ("config_hash", "tape_hash", "experiment_hash"):
        assert isinstance(checked[key], str) and len(checked[key]) == 64
    assert isinstance(checked["total_turnover"], float)
    assert isinstance(checked["transaction_costs"], float)
    assert checked["total_turnover"] > 0.0
    assert checked["transaction_costs"] > 0.0
    assert _PERFORMANCE_KEYS.isdisjoint(checked)


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
