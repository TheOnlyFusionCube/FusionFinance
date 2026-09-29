"""Immutable public contracts for the controlled FusionFinance experiment."""

from __future__ import annotations

from datetime import date
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ExperimentConfig(_FrozenModel):
    """One frozen set of assumptions shared by every strategy arm."""

    schema_version: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    start_date: date
    end_date: date
    starting_capital: float = Field(gt=0.0)
    universe: tuple[str, ...] = Field(min_length=1)
    benchmark_ticker: str = Field(min_length=1)
    rebalance_frequency_sessions: int = Field(gt=0)
    execution_lag_sessions: int = Field(default=1, ge=1)
    transaction_cost_bps: float = Field(default=0.0, ge=0.0, lt=10_000.0)
    slippage_bps: float = Field(default=0.0, ge=0.0, lt=10_000.0)
    max_gross_leverage: float = Field(gt=0.0, le=10.0)
    max_position_weight: float = Field(gt=0.0, le=1.0)
    annualization_sessions: int = Field(default=252, gt=1)
    annual_risk_free_rate: float = Field(default=0.0, gt=-1.0)

    @field_validator("schema_version", "experiment_id", "benchmark_ticker")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("benchmark_ticker")
    @classmethod
    def _normalize_benchmark(cls, value: str) -> str:
        return value.upper()

    @field_validator("universe", mode="before")
    @classmethod
    def _normalize_universe(cls, value) -> tuple[str, ...]:
        if isinstance(value, str):
            raise ValueError("universe must be a sequence of ticker symbols")
        try:
            names = tuple(str(item).strip().upper() for item in value)
        except TypeError as exc:
            raise ValueError("universe must be a sequence of ticker symbols") from exc
        if any(not name for name in names):
            raise ValueError("universe tickers must not be blank")
        if len(set(names)) != len(names):
            raise ValueError("universe tickers must be unique")
        return names

    @model_validator(mode="after")
    def _cross_field_boundaries(self) -> "ExperimentConfig":
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if self.max_position_weight > self.max_gross_leverage:
            raise ValueError("max_position_weight cannot exceed max_gross_leverage")
        if self.benchmark_ticker in self.universe:
            raise ValueError(
                "benchmark_ticker must not be part of the tradable universe"
            )
        if self.execution_lag_sessions != 1:
            raise ValueError("FusionFinance execution must use the next session")
        return self


class AssetBar(_FrozenModel):
    """One asset's tradable next-open and end-of-session mark."""

    ticker: str = Field(min_length=1)
    open: float = Field(gt=0.0)
    close: float = Field(gt=0.0)

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker must not be blank")
        return normalized


class MarketSession(_FrozenModel):
    """Deeply immutable cross-section for one exchange session."""

    session: date
    bars: tuple[AssetBar, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_tickers(self) -> Self:
        tickers = tuple(bar.ticker for bar in self.bars)
        if len(set(tickers)) != len(tickers):
            raise ValueError("market-session tickers must be unique")
        return self


class PositionTarget(_FrozenModel):
    ticker: str = Field(min_length=1)
    weight: float

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker must not be blank")
        return normalized


class WeightProposal(_FrozenModel):
    """Target weights decided at a close and executable next session."""

    strategy_id: str = Field(min_length=1)
    decision_session: date
    targets: tuple[PositionTarget, ...] = Field(default_factory=tuple)

    @field_validator("strategy_id")
    @classmethod
    def _strategy_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("strategy_id must not be blank")
        return normalized

    @model_validator(mode="after")
    def _unique_targets(self) -> Self:
        tickers = tuple(target.ticker for target in self.targets)
        if len(set(tickers)) != len(tickers):
            raise ValueError("proposal target tickers must be unique")
        return self


class Fill(_FrozenModel):
    strategy_id: str
    decision_session: date
    execution_session: date
    ticker: str
    execution_price: float = Field(gt=0.0)
    shares_delta: float
    traded_notional: float
    previous_weight: float
    target_weight: float
    transaction_cost: float = Field(ge=0.0)
    slippage_cost: float = Field(ge=0.0)


class RebalanceEvent(_FrozenModel):
    strategy_id: str
    decision_session: date
    execution_session: date
    turnover: float = Field(ge=0.0)
    transaction_cost: float = Field(ge=0.0)
    slippage_cost: float = Field(ge=0.0)
    fills: tuple[Fill, ...] = Field(default_factory=tuple)


class PortfolioPoint(_FrozenModel):
    session: date
    portfolio_value: float = Field(gt=0.0)
    cash_balance: float
    period_return: float = Field(gt=-1.0)
    gross_exposure: float = Field(ge=0.0)
    net_exposure: float


class SimulationResult(_FrozenModel):
    strategy_id: str
    benchmark_ticker: str
    starting_capital: float = Field(gt=0.0)
    points: tuple[PortfolioPoint, ...] = Field(min_length=1)
    rebalances: tuple[RebalanceEvent, ...] = Field(default_factory=tuple)

    @property
    def portfolio_values(self) -> tuple[float, ...]:
        return tuple(point.portfolio_value for point in self.points)

    @property
    def period_returns(self) -> tuple[float, ...]:
        return tuple(point.period_return for point in self.points)

    @property
    def total_cost(self) -> float:
        return sum(
            event.transaction_cost + event.slippage_cost for event in self.rebalances
        )


class PerformanceMetrics(_FrozenModel):
    """Auditable absolute and benchmark-relative experiment statistics."""

    strategy_id: str = Field(min_length=1)
    benchmark_ticker: str = Field(min_length=1)
    periods: int = Field(gt=0)
    starting_value: float = Field(gt=0.0)
    ending_value: float = Field(gt=0.0)
    total_return: float = Field(gt=-1.0)
    annualized_return: float = Field(gt=-1.0)
    annualized_volatility: float = Field(ge=0.0)
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown: float = Field(ge=-1.0, le=0.0)
    calmar_ratio: float | None = None
    tail_loss_5pct: float = Field(gt=-1.0)
    win_rate: float = Field(ge=0.0, le=1.0)
    trade_count: int = Field(ge=0)
    total_turnover: float = Field(ge=0.0)
    average_turnover: float = Field(ge=0.0)
    transaction_costs: float = Field(ge=0.0)
    transaction_cost_pct: float = Field(ge=0.0)
    average_gross_exposure: float = Field(ge=0.0)
    ending_gross_exposure: float = Field(ge=0.0)
    ending_net_exposure: float
    ending_cash_balance: float
    benchmark_total_return: float | None = Field(default=None, gt=-1.0)
    benchmark_annualized_return: float | None = Field(default=None, gt=-1.0)
    wealth_relative_excess_return: float | None = Field(default=None, gt=-1.0)
    tracking_error: float | None = Field(default=None, ge=0.0)
    information_ratio: float | None = None
    beta: float | None = None
    annualized_alpha: float | None = None
