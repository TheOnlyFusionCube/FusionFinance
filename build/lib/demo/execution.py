"""Shared, fail-fast portfolio execution for every FusionFinance arm.

Decisions are made at a session close and execute at the next session's open.
All arms therefore receive the same timing, cost, slippage, leverage, and
concentration treatment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

from demo.contracts import (
    ExperimentConfig,
    Fill,
    MarketSession,
    PortfolioPoint,
    RebalanceEvent,
    SimulationResult,
    WeightProposal,
)


def simulate_portfolio(
    *,
    config: ExperimentConfig,
    sessions: Sequence[MarketSession],
    proposals: Sequence[WeightProposal],
    strategy_id: str,
) -> SimulationResult:
    """Execute target-weight proposals without using information after decision time."""
    normalized_strategy = strategy_id.strip()
    if not normalized_strategy:
        raise ValueError("strategy_id must not be blank")
    window = _validated_sessions(config, sessions)
    scheduled = _schedule_proposals(
        config, window, proposals, strategy_id=normalized_strategy
    )

    cash = float(config.starting_capital)
    shares: dict[str, float] = {}
    first = window[0]
    points = [
        PortfolioPoint(
            session=first.session,
            portfolio_value=cash,
            cash_balance=cash,
            period_return=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
        )
    ]
    rebalances: list[RebalanceEvent] = []

    for market in window[1:]:
        bars = {bar.ticker: bar for bar in market.bars}
        previous_value = points[-1].portfolio_value
        pretrade_value = cash + sum(
            quantity * bars[ticker].open for ticker, quantity in shares.items()
        )
        if not math.isfinite(pretrade_value) or pretrade_value <= 0.0:
            raise ValueError("portfolio value must remain positive before execution")

        proposal = scheduled.get(market.session)
        if proposal is not None:
            cash, shares, event = _rebalance(
                config=config,
                proposal=proposal,
                market=market,
                cash=cash,
                shares=shares,
                pretrade_value=pretrade_value,
            )
            rebalances.append(event)

        close_notionals = {
            ticker: quantity * bars[ticker].close for ticker, quantity in shares.items()
        }
        portfolio_value = cash + sum(close_notionals.values())
        if not math.isfinite(portfolio_value) or portfolio_value <= 0.0:
            raise ValueError("portfolio value must remain positive after marking")
        gross = sum(abs(value) for value in close_notionals.values()) / portfolio_value
        net = sum(close_notionals.values()) / portfolio_value
        points.append(
            PortfolioPoint(
                session=market.session,
                portfolio_value=portfolio_value,
                cash_balance=cash,
                period_return=portfolio_value / previous_value - 1.0,
                gross_exposure=gross,
                net_exposure=net,
            )
        )

    return SimulationResult(
        strategy_id=normalized_strategy,
        benchmark_ticker=config.benchmark_ticker,
        starting_capital=config.starting_capital,
        points=tuple(points),
        rebalances=tuple(rebalances),
    )


def _validated_sessions(
    config: ExperimentConfig, sessions: Sequence[MarketSession]
) -> tuple[MarketSession, ...]:
    window = tuple(
        session
        for session in sessions
        if config.start_date <= session.session <= config.end_date
    )
    if len(window) < 2:
        raise ValueError("experiment requires at least two market sessions")
    dates = tuple(session.session for session in window)
    if dates != tuple(sorted(dates)) or len(set(dates)) != len(dates):
        raise ValueError("market sessions must be strictly increasing and unique")
    if dates[0] != config.start_date or dates[-1] != config.end_date:
        raise ValueError(
            "market sessions must cover the configured start and end dates"
        )
    required = set(config.universe) | {config.benchmark_ticker}
    for session in window:
        available = {bar.ticker for bar in session.bars}
        missing = required.difference(available)
        if missing:
            raise ValueError(
                f"market session {session.session} missing ticker(s): {sorted(missing)}"
            )
    return window


def _schedule_proposals(
    config: ExperimentConfig,
    sessions: tuple[MarketSession, ...],
    proposals: Sequence[WeightProposal],
    *,
    strategy_id: str,
) -> dict[date, WeightProposal]:
    index_by_date = {session.session: index for index, session in enumerate(sessions)}
    scheduled: dict[date, WeightProposal] = {}
    seen_decisions: set[date] = set()
    universe = set(config.universe)
    for proposal in proposals:
        if proposal.strategy_id != strategy_id:
            raise ValueError("all proposals must belong to the simulated strategy")
        if proposal.decision_session in seen_decisions:
            raise ValueError(
                "a strategy may submit only one proposal per decision session"
            )
        seen_decisions.add(proposal.decision_session)
        decision_index = index_by_date.get(proposal.decision_session)
        if decision_index is None:
            raise ValueError(
                "proposal decision session is outside the experiment market tape"
            )
        if decision_index % config.rebalance_frequency_sessions != 0:
            raise ValueError("proposal decision is off the shared rebalance clock")
        execution_index = decision_index + config.execution_lag_sessions
        if execution_index >= len(sessions):
            raise ValueError("proposal has no next market session for execution")
        unknown = {target.ticker for target in proposal.targets}.difference(universe)
        if unknown:
            raise ValueError(
                f"proposal contains out-of-universe ticker(s): {sorted(unknown)}"
            )
        for target in proposal.targets:
            if abs(target.weight) > config.max_position_weight + 1e-12:
                raise ValueError(f"target {target.ticker} exceeds max_position_weight")
        gross = sum(abs(target.weight) for target in proposal.targets)
        if gross > config.max_gross_leverage + 1e-12:
            raise ValueError("proposal exceeds max_gross_leverage")
        execution_date = sessions[execution_index].session
        if execution_date in scheduled:
            raise ValueError("multiple proposals resolve to one execution session")
        scheduled[execution_date] = proposal
    return scheduled


def _rebalance(
    *,
    config: ExperimentConfig,
    proposal: WeightProposal,
    market: MarketSession,
    cash: float,
    shares: dict[str, float],
    pretrade_value: float,
) -> tuple[float, dict[str, float], RebalanceEvent]:
    bars = {bar.ticker: bar for bar in market.bars}
    targets = {target.ticker: target.weight for target in proposal.targets}
    names = tuple(sorted(set(shares) | set(targets)))
    transaction_rate = config.transaction_cost_bps / 10_000.0
    slippage_rate = config.slippage_bps / 10_000.0
    new_cash = cash
    new_shares = dict(shares)
    fills: list[Fill] = []

    for ticker in names:
        price = bars[ticker].open
        current_quantity = shares.get(ticker, 0.0)
        current_notional = current_quantity * price
        desired_notional = targets.get(ticker, 0.0) * pretrade_value
        traded_notional = desired_notional - current_notional
        if abs(traded_notional) <= 1e-12:
            continue
        shares_delta = traded_notional / price
        transaction_cost = abs(traded_notional) * transaction_rate
        slippage_cost = abs(traded_notional) * slippage_rate
        new_cash -= traded_notional + transaction_cost + slippage_cost
        next_quantity = current_quantity + shares_delta
        if abs(next_quantity) <= 1e-12:
            new_shares.pop(ticker, None)
        else:
            new_shares[ticker] = next_quantity
        fills.append(
            Fill(
                strategy_id=proposal.strategy_id,
                decision_session=proposal.decision_session,
                execution_session=market.session,
                ticker=ticker,
                execution_price=price,
                shares_delta=shares_delta,
                traded_notional=traded_notional,
                previous_weight=current_notional / pretrade_value,
                target_weight=targets.get(ticker, 0.0),
                transaction_cost=transaction_cost,
                slippage_cost=slippage_cost,
            )
        )

    turnover = sum(abs(fill.traded_notional) for fill in fills) / pretrade_value
    event = RebalanceEvent(
        strategy_id=proposal.strategy_id,
        decision_session=proposal.decision_session,
        execution_session=market.session,
        turnover=turnover,
        transaction_cost=sum(fill.transaction_cost for fill in fills),
        slippage_cost=sum(fill.slippage_cost for fill in fills),
        fills=tuple(fills),
    )
    return new_cash, new_shares, event
