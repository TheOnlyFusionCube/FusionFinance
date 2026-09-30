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
    LedgerReconciliation,
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


def session_window(
    config: ExperimentConfig, sessions: Sequence[MarketSession]
) -> tuple[MarketSession, ...]:
    """Return the config-bounded tape used by execution and run hashes."""

    return _validated_sessions(config, sessions)


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
    cost_paid = sum(fill.transaction_cost + fill.slippage_cost for fill in fills)
    post_gross = _post_cost_gross_leverage(
        cash=new_cash,
        shares=new_shares,
        bars=bars,
    )
    assert_post_cost_bound(
        post_gross=post_gross,
        cost_fraction=cost_paid / pretrade_value,
        max_gross_leverage=config.max_gross_leverage,
    )
    event = RebalanceEvent(
        strategy_id=proposal.strategy_id,
        decision_session=proposal.decision_session,
        execution_session=market.session,
        turnover=turnover,
        transaction_cost=sum(fill.transaction_cost for fill in fills),
        slippage_cost=sum(fill.slippage_cost for fill in fills),
        post_cost_gross_leverage=post_gross,
        fills=tuple(fills),
    )
    return new_cash, new_shares, event


def assert_post_cost_bound(
    *,
    post_gross: float,
    cost_fraction: float,
    max_gross_leverage: float,
) -> None:
    """Reject exposure that costs alone cannot explain.

    A book already at the locked gross cap is slightly above that cap after
    costs reduce NAV. The allowed band is that cost drag. Anything wider is an
    accounting breach.
    """

    if (
        not math.isfinite(post_gross)
        or post_gross < 0.0
        or not math.isfinite(cost_fraction)
        or cost_fraction < 0.0
        or cost_fraction >= 1.0
    ):
        raise ValueError("post-cost gross leverage exceeds the locked limit")
    allowed = max_gross_leverage / (1.0 - cost_fraction)
    if post_gross > allowed and not math.isclose(
        post_gross, allowed, rel_tol=1e-9, abs_tol=1e-8
    ):
        raise ValueError("post-cost gross leverage exceeds the locked limit")


def reconcile_simulation(
    result: SimulationResult,
    *,
    config: ExperimentConfig,
    sessions: Sequence[MarketSession],
) -> LedgerReconciliation:
    """Replay fills against the tape and require the marked result to match."""

    if result.benchmark_ticker != config.benchmark_ticker:
        raise ValueError("result benchmark does not match experiment config")
    if not math.isclose(
        result.starting_capital,
        config.starting_capital,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise ValueError("result starting capital does not match experiment config")
    window = _validated_sessions(config, sessions)
    dates = tuple(session.session for session in window)
    if tuple(point.session for point in result.points) != dates:
        raise ValueError("portfolio points are not bound to the market tape")
    events: dict[date, RebalanceEvent] = {}
    for event in result.rebalances:
        if event.strategy_id != result.strategy_id:
            raise ValueError("rebalance strategy does not match the result")
        if event.execution_session in events:
            raise ValueError("multiple rebalances resolve to one execution session")
        _require_lagged_event(event, dates, config)
        events[event.execution_session] = event

    cash = float(config.starting_capital)
    shares: dict[str, float] = {}
    _require_mark(
        point=result.points[0],
        cash=cash,
        shares=shares,
        bars={},
        previous_value=None,
    )
    seen: set[date] = set()
    previous_value = result.points[0].portfolio_value
    for market, point in zip(window[1:], result.points[1:], strict=True):
        bars = {bar.ticker: bar for bar in market.bars}
        pretrade = cash + sum(
            quantity * bars[ticker].open for ticker, quantity in shares.items()
        )
        event = events.get(market.session)
        if event is not None:
            cash, shares = _replay_event(
                event,
                bars=bars,
                cash=cash,
                shares=shares,
                pretrade=pretrade,
                config=config,
            )
            seen.add(market.session)
        _require_mark(
            point=point,
            cash=cash,
            shares=shares,
            bars=bars,
            previous_value=previous_value,
        )
        previous_value = point.portfolio_value
    if set(events) != seen:
        raise ValueError("rebalance execution session is outside the market tape")

    post_cost = tuple(event.post_cost_gross_leverage for event in result.rebalances)
    max_post_cost = max(post_cost, default=0.0)
    return LedgerReconciliation(
        strategy_id=result.strategy_id,
        session_count=len(result.points),
        trade_count=sum(len(event.fills) for event in result.rebalances),
        total_turnover=sum(event.turnover for event in result.rebalances),
        transaction_costs=result.total_cost,
        slippage_costs=sum(event.slippage_cost for event in result.rebalances),
        max_post_cost_gross_leverage=max_post_cost,
        post_cost_within_limit=all(
            value <= config.max_gross_leverage + 1e-9 for value in post_cost
        ),
    )


def _post_cost_gross_leverage(
    *,
    cash: float,
    shares: dict[str, float],
    bars: dict,
) -> float:
    notionals = tuple(
        quantity * bars[ticker].open for ticker, quantity in shares.items()
    )
    post_value = cash + sum(notionals)
    if not math.isfinite(post_value) or post_value <= 0.0:
        raise ValueError("portfolio value must remain positive after costs")
    return sum(abs(value) for value in notionals) / post_value


def _require_lagged_event(
    event: RebalanceEvent,
    dates: tuple[date, ...],
    config: ExperimentConfig,
) -> None:
    try:
        execution_index = dates.index(event.execution_session)
    except ValueError as exc:
        raise ValueError(
            "rebalance execution session is outside the market tape"
        ) from exc
    decision_index = execution_index - config.execution_lag_sessions
    if decision_index < 0 or dates[decision_index] != event.decision_session:
        raise ValueError("fill decision session is not the lagged tape date")
    for fill in event.fills:
        if (
            fill.strategy_id != event.strategy_id
            or fill.decision_session != event.decision_session
            or fill.execution_session != event.execution_session
        ):
            raise ValueError("fill sessions do not match the rebalance event")


def _replay_event(
    event: RebalanceEvent,
    *,
    bars: dict,
    cash: float,
    shares: dict[str, float],
    pretrade: float,
    config: ExperimentConfig,
) -> tuple[float, dict[str, float]]:
    transaction_rate = config.transaction_cost_bps / 10_000.0
    slippage_rate = config.slippage_bps / 10_000.0
    if not math.isfinite(pretrade) or pretrade <= 0.0:
        raise ValueError("portfolio value must remain positive before execution")
    for fill in event.fills:
        bar = bars.get(fill.ticker)
        if bar is None or not math.isclose(
            fill.execution_price, bar.open, rel_tol=1e-12, abs_tol=1e-9
        ):
            raise ValueError("fill execution price does not match the tape open")
        if not math.isclose(
            fill.shares_delta * fill.execution_price,
            fill.traded_notional,
            rel_tol=1e-9,
            abs_tol=1e-8,
        ):
            raise ValueError("fill notional does not match shares and price")
        if not math.isclose(
            fill.transaction_cost,
            abs(fill.traded_notional) * transaction_rate,
            rel_tol=1e-9,
            abs_tol=1e-8,
        ):
            raise ValueError("fill transaction cost does not match the config")
        if not math.isclose(
            fill.slippage_cost,
            abs(fill.traded_notional) * slippage_rate,
            rel_tol=1e-9,
            abs_tol=1e-8,
        ):
            raise ValueError("fill slippage cost does not match the config")
        weight_delta = fill.target_weight - fill.previous_weight
        if abs(weight_delta) > 1e-12:
            implied = fill.traded_notional / weight_delta
            if not math.isclose(implied, pretrade, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError("fill weights do not reconcile to pre-trade value")
    if not math.isclose(
        sum(fill.transaction_cost for fill in event.fills),
        event.transaction_cost,
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        raise ValueError("event transaction cost does not match its fills")
    if not math.isclose(
        sum(fill.slippage_cost for fill in event.fills),
        event.slippage_cost,
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        raise ValueError("event slippage cost does not match its fills")
    turnover = sum(abs(fill.traded_notional) for fill in event.fills) / pretrade
    if not math.isclose(turnover, event.turnover, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError("event turnover does not match its fills")

    new_cash = cash
    new_shares = dict(shares)
    for fill in event.fills:
        new_cash -= fill.traded_notional + fill.transaction_cost + fill.slippage_cost
        quantity = new_shares.get(fill.ticker, 0.0) + fill.shares_delta
        if abs(quantity) <= 1e-12:
            new_shares.pop(fill.ticker, None)
        else:
            new_shares[fill.ticker] = quantity
    post_gross = _post_cost_gross_leverage(
        cash=new_cash, shares=new_shares, bars=bars
    )
    if not math.isclose(
        post_gross,
        event.post_cost_gross_leverage,
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        raise ValueError("post-cost leverage does not match fills")
    return new_cash, new_shares


def _require_mark(
    *,
    point: PortfolioPoint,
    cash: float,
    shares: dict[str, float],
    bars: dict,
    previous_value: float | None,
) -> None:
    if previous_value is None:
        if (
            not math.isclose(point.portfolio_value, cash, rel_tol=1e-12, abs_tol=1e-9)
            or not math.isclose(point.cash_balance, cash, rel_tol=1e-12, abs_tol=1e-9)
            or not math.isclose(point.period_return, 0.0, abs_tol=1e-12)
            or point.gross_exposure != 0.0
            or point.net_exposure != 0.0
        ):
            raise ValueError("starting portfolio point does not match capital")
        return
    notionals = tuple(
        quantity * bars[ticker].close for ticker, quantity in shares.items()
    )
    value = cash + sum(notionals)
    gross = sum(abs(item) for item in notionals) / value
    net = sum(notionals) / value
    if not math.isclose(point.portfolio_value, value, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError("portfolio value does not match fills and the tape")
    if not math.isclose(point.cash_balance, cash, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError("cash balance does not match fills")
    if not math.isclose(
        point.period_return, value / previous_value - 1.0, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("portfolio period return does not match its wealth ratio")
    if not math.isclose(point.gross_exposure, gross, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError("gross exposure does not match fills and the tape")
    if not math.isclose(point.net_exposure, net, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError("net exposure does not match fills and the tape")
