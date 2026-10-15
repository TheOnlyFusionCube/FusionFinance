"""Cash timing must not be reported as selection alpha."""

from __future__ import annotations

from datetime import date

import pytest


def _config():
    from demo.contracts import ExperimentConfig

    return ExperimentConfig.model_validate(
        {
            "schema_version": "1.0",
            "experiment_id": "exposure-reading",
            "start_date": "2026-02-02",
            "end_date": "2026-02-04",
            "starting_capital": 100.0,
            "universe": ["AAA"],
            "benchmark_ticker": "SPY",
            "rebalance_frequency_sessions": 1,
            "execution_lag_sessions": 1,
            "max_gross_leverage": 1.0,
            "max_position_weight": 1.0,
            "annualization_sessions": 2,
        }
    )


def _result(points):
    from demo.contracts import PortfolioPoint, SimulationResult

    built = tuple(
        PortfolioPoint(
            session=session,
            portfolio_value=value,
            cash_balance=value,
            period_return=ret,
            gross_exposure=gross,
            net_exposure=gross,
        )
        for session, value, ret, gross in points
    )
    return SimulationResult(
        strategy_id="timing",
        benchmark_ticker="SPY",
        starting_capital=100.0,
        points=built,
    )


def test_skipping_a_down_day_is_not_selection() -> None:
    from demo.metrics import compute_performance_metrics

    result = _result(
        (
            (date(2026, 2, 2), 100.0, 0.0, 0.0),
            (date(2026, 2, 3), 100.0, 0.0, 0.0),
            (date(2026, 2, 4), 100.0 * 110.0 / 90.0, 110.0 / 90.0 - 1.0, 1.0),
        )
    )
    metrics = compute_performance_metrics(
        result,
        config=_config(),
        benchmark_values=(100.0, 90.0, 110.0),
        benchmark_sessions=(
            date(2026, 2, 2),
            date(2026, 2, 3),
            date(2026, 2, 4),
        ),
    )

    assert metrics.selection_excess_return == pytest.approx(0.0, abs=1e-12)
    assert metrics.cash_replaced_excess_return == pytest.approx(0.0, abs=1e-12)
    assert metrics.same_day_benchmark_total_return == pytest.approx(110.0 / 90.0 - 1.0)
    assert metrics.sharpe_exposure_diluted is True
    assert metrics.annualized_alpha is not None
    assert metrics.annualized_alpha > 0.0


def test_fully_invested_book_may_quote_sharpe() -> None:
    from demo.metrics import compute_performance_metrics

    result = _result(
        (
            (date(2026, 2, 2), 100.0, 0.0, 0.0),
            (date(2026, 2, 3), 110.0, 0.10, 1.0),
            (date(2026, 2, 4), 121.0, 0.10, 1.0),
        )
    )
    metrics = compute_performance_metrics(
        result,
        config=_config(),
        benchmark_values=(100.0, 110.0, 121.0),
    )

    assert metrics.sharpe_exposure_diluted is False
    assert metrics.selection_excess_return == pytest.approx(0.0, abs=1e-12)
    assert metrics.cash_replaced_excess_return == pytest.approx(0.0, abs=1e-12)


def test_benchmark_sessions_must_match_the_portfolio() -> None:
    from demo.metrics import compute_performance_metrics

    result = _result(
        (
            (date(2026, 2, 2), 100.0, 0.0, 0.0),
            (date(2026, 2, 3), 100.0, 0.0, 0.0),
            (date(2026, 2, 4), 100.0, 0.0, 0.0),
        )
    )
    with pytest.raises(ValueError, match="benchmark sessions"):
        compute_performance_metrics(
            result,
            config=_config(),
            benchmark_values=(100.0, 100.0, 100.0),
            benchmark_sessions=(
                date(2026, 2, 2),
                date(2026, 2, 3),
                date(2026, 2, 5),
            ),
        )
    with pytest.raises(ValueError, match="benchmark_sessions requires"):
        compute_performance_metrics(
            result,
            config=_config(),
            benchmark_sessions=(date(2026, 2, 2),),
        )
