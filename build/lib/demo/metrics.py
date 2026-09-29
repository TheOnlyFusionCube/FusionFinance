"""Wealth-relative performance metrics for the controlled strategy race."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

from demo.contracts import ExperimentConfig, PerformanceMetrics, SimulationResult


_EPSILON = 1e-15


def compute_performance_metrics(
    result: SimulationResult,
    *,
    config: ExperimentConfig,
    benchmark_values: Sequence[float] | None = None,
) -> PerformanceMetrics:
    """Compute statistics from successive wealth ratios, never return deltas."""
    values = _validated_strategy_values(result, config)
    returns = _wealth_returns(values)
    periods = len(returns)
    annualization = config.annualization_sessions
    total_return = values[-1] / values[0] - 1.0
    annualized_return = _annualized_growth(total_return, periods, annualization)
    volatility = _sample_standard_deviation(returns)
    annualized_volatility = volatility * math.sqrt(annualization)
    risk_free_return = math.expm1(
        math.log1p(config.annual_risk_free_rate) / annualization
    )
    excess_returns = tuple(value - risk_free_return for value in returns)
    sharpe = _annualized_ratio(excess_returns, annualization)
    downside_deviation = math.sqrt(
        sum(min(value, 0.0) ** 2 for value in excess_returns) / periods
    )
    sortino = (
        _mean(excess_returns) / downside_deviation * math.sqrt(annualization)
        if downside_deviation > _EPSILON
        else None
    )
    max_drawdown = _maximum_drawdown(values)
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < -_EPSILON else None

    benchmark_statistics = _benchmark_statistics(
        benchmark_values=benchmark_values,
        strategy_returns=returns,
        strategy_total_return=total_return,
        expected_length=len(values),
        annualization=annualization,
        risk_free_return=risk_free_return,
    )
    total_turnover = sum(event.turnover for event in result.rebalances)
    transaction_costs = result.total_cost
    ending_point = result.points[-1]
    return PerformanceMetrics(
        strategy_id=result.strategy_id,
        benchmark_ticker=result.benchmark_ticker,
        periods=periods,
        starting_value=values[0],
        ending_value=values[-1],
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar,
        tail_loss_5pct=_linear_quantile(returns, 0.05),
        win_rate=sum(value > 0.0 for value in returns) / periods,
        trade_count=sum(len(event.fills) for event in result.rebalances),
        total_turnover=total_turnover,
        average_turnover=(
            total_turnover / len(result.rebalances) if result.rebalances else 0.0
        ),
        transaction_costs=transaction_costs,
        transaction_cost_pct=transaction_costs / values[0],
        average_gross_exposure=_mean(
            tuple(point.gross_exposure for point in result.points)
        ),
        ending_gross_exposure=ending_point.gross_exposure,
        ending_net_exposure=ending_point.net_exposure,
        ending_cash_balance=ending_point.cash_balance,
        **benchmark_statistics,
    )


def _validated_strategy_values(
    result: SimulationResult, config: ExperimentConfig
) -> tuple[float, ...]:
    if len(result.points) < 2:
        raise ValueError("metrics require at least two portfolio points")
    if result.benchmark_ticker != config.benchmark_ticker:
        raise ValueError("result benchmark does not match experiment config")
    if not math.isclose(
        result.starting_capital,
        config.starting_capital,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise ValueError("result starting capital does not match experiment config")
    sessions = tuple(point.session for point in result.points)
    if sessions != tuple(sorted(sessions)) or len(set(sessions)) != len(sessions):
        raise ValueError("portfolio points must be strictly increasing and unique")
    if sessions[0] != config.start_date or sessions[-1] != config.end_date:
        raise ValueError("portfolio points must cover the configured date window")
    values = tuple(point.portfolio_value for point in result.points)
    if not math.isclose(
        values[0], result.starting_capital, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise ValueError("first portfolio value must equal starting capital")
    if not math.isclose(result.points[0].period_return, 0.0, abs_tol=1e-12):
        raise ValueError("first portfolio point must have a zero period return")
    calculated_returns = _wealth_returns(values)
    for point, calculated in zip(result.points[1:], calculated_returns, strict=True):
        if not math.isclose(
            point.period_return, calculated, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("portfolio period return does not match its wealth ratio")
    return values


def _benchmark_statistics(
    *,
    benchmark_values: Sequence[float] | None,
    strategy_returns: tuple[float, ...],
    strategy_total_return: float,
    expected_length: int,
    annualization: int,
    risk_free_return: float,
) -> dict[str, float | None]:
    empty = {
        "benchmark_total_return": None,
        "benchmark_annualized_return": None,
        "wealth_relative_excess_return": None,
        "tracking_error": None,
        "information_ratio": None,
        "beta": None,
        "annualized_alpha": None,
    }
    if benchmark_values is None:
        return empty
    if isinstance(benchmark_values, (str, bytes)):
        raise ValueError("benchmark_values must be a numeric sequence")
    values = tuple(
        _finite_positive(value, "benchmark value") for value in benchmark_values
    )
    if len(values) != expected_length:
        raise ValueError(
            "benchmark_values must align one-for-one with portfolio points"
        )

    benchmark_returns = _wealth_returns(values)
    benchmark_total = values[-1] / values[0] - 1.0
    periods = len(benchmark_returns)
    active_returns = tuple(
        (1.0 + strategy) / (1.0 + benchmark) - 1.0
        for strategy, benchmark in zip(strategy_returns, benchmark_returns, strict=True)
    )
    tracking_period = _sample_standard_deviation(active_returns)
    tracking_error = tracking_period * math.sqrt(annualization)
    information_ratio = (
        _mean(active_returns) / tracking_period * math.sqrt(annualization)
        if tracking_period > _EPSILON
        else None
    )
    benchmark_variance = _sample_variance(benchmark_returns)
    beta = (
        _sample_covariance(strategy_returns, benchmark_returns) / benchmark_variance
        if benchmark_variance > _EPSILON
        else None
    )
    alpha = (
        (
            (_mean(strategy_returns) - risk_free_return)
            - beta * (_mean(benchmark_returns) - risk_free_return)
        )
        * annualization
        if beta is not None
        else None
    )
    return {
        "benchmark_total_return": benchmark_total,
        "benchmark_annualized_return": _annualized_growth(
            benchmark_total, periods, annualization
        ),
        "wealth_relative_excess_return": (
            (1.0 + strategy_total_return) / (1.0 + benchmark_total) - 1.0
        ),
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta": beta,
        "annualized_alpha": alpha,
    }


def _wealth_returns(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(
        current / previous - 1.0
        for previous, current in zip(values[:-1], values[1:], strict=True)
    )


def _annualized_growth(total_return: float, periods: int, annualization: int) -> float:
    try:
        value = math.expm1(math.log1p(total_return) * annualization / periods)
    except OverflowError as exc:
        raise ValueError(
            "annualized return exceeds the supported numeric range"
        ) from exc
    if not math.isfinite(value):
        raise ValueError("annualized return must be finite")
    return value


def _maximum_drawdown(values: tuple[float, ...]) -> float:
    peak = values[0]
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1.0)
    return drawdown


def _annualized_ratio(values: tuple[float, ...], annualization: int) -> float | None:
    deviation = _sample_standard_deviation(values)
    if deviation <= _EPSILON:
        return None
    return _mean(values) / deviation * math.sqrt(annualization)


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def _sample_variance(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _sample_standard_deviation(values: tuple[float, ...]) -> float:
    return math.sqrt(_sample_variance(values))


def _sample_covariance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("covariance inputs must have equal length")
    if len(left) < 2:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    return sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    ) / (len(left) - 1)


def _linear_quantile(values: tuple[float, ...], probability: float) -> float:
    ordered = tuple(sorted(values))
    location = (len(ordered) - 1) * probability
    lower_index = math.floor(location)
    upper_index = math.ceil(location)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = location - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def _finite_positive(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number
