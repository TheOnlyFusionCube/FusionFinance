from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError


def _config_payload() -> dict:
    return {
        "schema_version": "1.0",
        "experiment_id": "fair-race-test",
        "start_date": "2026-02-02",
        "end_date": "2026-07-09",
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


def test_experiment_config_is_immutable_and_rejects_invalid_boundaries() -> None:
    from demo.contracts import ExperimentConfig

    config = ExperimentConfig.model_validate(_config_payload())

    assert config.start_date == date(2026, 2, 2)
    assert config.universe == ("AAA", "BBB")
    with pytest.raises(ValidationError):
        config.starting_capital = 1.0

    invalid = {**_config_payload(), "max_position_weight": 0.0}
    with pytest.raises(ValidationError, match="max_position_weight"):
        ExperimentConfig.model_validate(invalid)

    reversed_window = {
        **_config_payload(),
        "start_date": "2026-07-09",
        "end_date": "2026-02-02",
    }
    with pytest.raises(ValidationError, match="end_date"):
        ExperimentConfig.model_validate(reversed_window)

    benchmark_in_universe = {
        **_config_payload(),
        "benchmark_ticker": "AAA",
    }
    with pytest.raises(ValidationError, match="benchmark"):
        ExperimentConfig.model_validate(benchmark_in_universe)


def _market_sessions():
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


def _proposal(strategy_id: str):
    from demo.contracts import PositionTarget, WeightProposal

    return WeightProposal(
        strategy_id=strategy_id,
        decision_session="2026-02-02",
        targets=(
            PositionTarget(ticker="AAA", weight=0.50),
            PositionTarget(ticker="BBB", weight=-0.50),
        ),
    )


def test_identical_weights_receive_identical_shared_execution_pnl() -> None:
    from demo.contracts import ExperimentConfig
    from demo.execution import simulate_portfolio

    config = ExperimentConfig.model_validate(
        {
            **_config_payload(),
            "start_date": "2026-02-02",
            "end_date": "2026-02-04",
        }
    )
    first = simulate_portfolio(
        config=config,
        sessions=_market_sessions(),
        proposals=(_proposal("pure_ml"),),
        strategy_id="pure_ml",
    )
    second = simulate_portfolio(
        config=config,
        sessions=_market_sessions(),
        proposals=(_proposal("fusion"),),
        strategy_id="fusion",
    )

    assert first.portfolio_values == second.portfolio_values
    assert first.period_returns == second.period_returns
    rate = (5.0 + 2.0) / 10_000.0
    scale = 1.0 / (1.0 + rate)
    cost = 10_000.0 * scale * rate
    assert first.rebalances[0].applied_scale == pytest.approx(scale)
    assert first.rebalances[0].turnover == pytest.approx(scale)
    assert first.total_cost == pytest.approx(cost)
    assert first.points[1].portfolio_value == pytest.approx(10_000.0 - cost + 1_000.0 * scale)
    assert first.config_sha256 == second.config_sha256
    assert first.tape_sha256 == second.tape_sha256
    assert len(first.config_sha256) == 64
    assert len(first.tape_sha256) == 64


def test_future_market_mutation_cannot_change_earlier_execution() -> None:
    from demo.contracts import AssetBar, ExperimentConfig, MarketSession
    from demo.execution import simulate_portfolio

    config = ExperimentConfig.model_validate(
        {
            **_config_payload(),
            "start_date": "2026-02-02",
            "end_date": "2026-02-04",
        }
    )
    original = _market_sessions()
    changed_future = (
        *original[:2],
        MarketSession(
            session="2026-02-04",
            bars=(
                AssetBar(ticker="AAA", open=1_000_000.0, close=1_000_001.0),
                AssetBar(ticker="BBB", open=1.0, close=0.5),
                AssetBar(ticker="SPY", open=99_999.0, close=99_999.0),
            ),
        ),
    )

    baseline = simulate_portfolio(
        config=config,
        sessions=original,
        proposals=(_proposal("fusion"),),
        strategy_id="fusion",
    )
    mutated = simulate_portfolio(
        config=config,
        sessions=changed_future,
        proposals=(_proposal("fusion"),),
        strategy_id="fusion",
    )

    assert mutated.points[:2] == baseline.points[:2]
    assert mutated.rebalances == baseline.rebalances
    assert mutated.config_sha256 == baseline.config_sha256
    assert mutated.tape_sha256 != baseline.tape_sha256
    assert baseline.rebalances[0].execution_session == date(2026, 2, 3)
    assert {fill.execution_price for fill in baseline.rebalances[0].fills} == {100.0}


def test_execution_rejects_decisions_off_the_shared_rebalance_clock() -> None:
    from demo.contracts import ExperimentConfig, PositionTarget, WeightProposal
    from demo.execution import simulate_portfolio

    config = ExperimentConfig.model_validate(
        {
            **_config_payload(),
            "start_date": "2026-02-02",
            "end_date": "2026-02-04",
            "rebalance_frequency_sessions": 2,
        }
    )
    off_clock = WeightProposal(
        strategy_id="pure_llm",
        decision_session="2026-02-03",
        targets=(PositionTarget(ticker="AAA", weight=0.25),),
    )

    with pytest.raises(ValueError, match="rebalance clock"):
        simulate_portfolio(
            config=config,
            sessions=_market_sessions(),
            proposals=(off_clock,),
            strategy_id="pure_llm",
        )


def test_metrics_use_wealth_ratios_and_produce_benchmark_relative_fields() -> None:
    from demo.contracts import ExperimentConfig
    from demo.execution import simulate_portfolio
    from demo.metrics import compute_performance_metrics

    config = ExperimentConfig.model_validate(
        {
            **_config_payload(),
            "start_date": "2026-02-02",
            "end_date": "2026-02-04",
            "annualization_sessions": 2,
        }
    )
    result = simulate_portfolio(
        config=config,
        sessions=_market_sessions(),
        proposals=(_proposal("fusion"),),
        strategy_id="fusion",
    )

    metrics = compute_performance_metrics(
        result,
        config=config,
        benchmark_values=(10_000.0, 10_100.0, 10_200.0),
    )

    rate = (5.0 + 2.0) / 10_000.0
    scale = 1.0 / (1.0 + rate)
    cost = 10_000.0 * scale * rate
    final = 10_000.0 - cost + 2_000.0 * scale
    expected_total = final / 10_000.0 - 1.0
    expected_benchmark = 10_200.0 / 10_000.0 - 1.0
    assert metrics.total_return == pytest.approx(expected_total)
    assert metrics.annualized_return == pytest.approx(expected_total)
    assert metrics.max_drawdown == pytest.approx(0.0)
    assert metrics.benchmark_total_return == pytest.approx(expected_benchmark)
    assert metrics.wealth_relative_excess_return == pytest.approx(
        (1.0 + expected_total) / (1.0 + expected_benchmark) - 1.0
    )
    assert metrics.trade_count == 2
    assert metrics.transaction_costs == pytest.approx(cost)
    assert metrics.ending_cash_balance == pytest.approx(10_000.0 - cost)
    assert metrics.selection_excess_return == pytest.approx(
        metrics.wealth_relative_excess_return
    )


def test_drawdown_is_measured_from_the_running_wealth_peak() -> None:
    from demo.contracts import ExperimentConfig, PortfolioPoint, SimulationResult
    from demo.metrics import compute_performance_metrics

    config = ExperimentConfig.model_validate(
        {
            **_config_payload(),
            "start_date": "2026-02-02",
            "end_date": "2026-02-04",
            "starting_capital": 100.0,
            "annualization_sessions": 2,
        }
    )
    result = SimulationResult(
        strategy_id="drawdown_case",
        benchmark_ticker="SPY",
        starting_capital=100.0,
        points=(
            PortfolioPoint(
                session="2026-02-02",
                portfolio_value=100.0,
                cash_balance=100.0,
                period_return=0.0,
                gross_exposure=0.0,
                net_exposure=0.0,
            ),
            PortfolioPoint(
                session="2026-02-03",
                portfolio_value=120.0,
                cash_balance=0.0,
                period_return=0.20,
                gross_exposure=1.0,
                net_exposure=1.0,
            ),
            PortfolioPoint(
                session="2026-02-04",
                portfolio_value=90.0,
                cash_balance=0.0,
                period_return=-0.25,
                gross_exposure=1.0,
                net_exposure=1.0,
            ),
        ),
    )

    metrics = compute_performance_metrics(result, config=config)

    assert metrics.total_return == pytest.approx(-0.10)
    assert metrics.max_drawdown == pytest.approx(-0.25)
    assert metrics.benchmark_total_return is None
    assert metrics.wealth_relative_excess_return is None


def test_metrics_reject_misaligned_or_nonpositive_benchmark_wealth() -> None:
    from demo.contracts import ExperimentConfig
    from demo.execution import simulate_portfolio
    from demo.metrics import compute_performance_metrics

    config = ExperimentConfig.model_validate(
        {
            **_config_payload(),
            "start_date": "2026-02-02",
            "end_date": "2026-02-04",
        }
    )
    result = simulate_portfolio(
        config=config,
        sessions=_market_sessions(),
        proposals=(_proposal("fusion"),),
        strategy_id="fusion",
    )

    with pytest.raises(ValueError, match="align one-for-one"):
        compute_performance_metrics(
            result, config=config, benchmark_values=(10_000.0, 10_100.0)
        )
    with pytest.raises(ValueError, match="finite and positive"):
        compute_performance_metrics(
            result,
            config=config,
            benchmark_values=(10_000.0, 0.0, 10_200.0),
        )


def test_checked_in_demo_config_is_valid_and_frozen() -> None:
    import json
    from pathlib import Path

    from demo.contracts import ExperimentConfig

    path = Path(__file__).resolve().parents[1] / "configs/fusionfinance-demo.json"
    config = ExperimentConfig.model_validate(json.loads(path.read_text()))

    assert config.experiment_id == "fusionfinance-fair-race-v1"
    assert config.execution_lag_sessions == 1
    assert config.benchmark_ticker not in config.universe
    assert len(config.universe) == 16
