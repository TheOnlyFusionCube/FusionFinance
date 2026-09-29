"""Walk-forward ablation backtest for the LLM-first fund.

Six arms see the same ideas, the same point-in-time data, the same risk book,
and the same next-open execution kernel with costs and slippage:

``llm_only``        every originated idea is traded (LLM decides, as in popular
                    persona- or debate-style agent funds);
``llm_desk``        ideas must also survive the analyst desk and evidence audit;
``ml_only``         the single market verifier trades its own top forecasts;
``committee_only``  the verifier committee trades its own consensus, no LLM;
``fusion_single``   LLM-first, gated by the single market verifier;
``fusion``          LLM-first, gated by the verifier committee's consensus, with
                    meta-labeler and track-record recalibration.

The ML verifier is retrained on a fixed cadence using only labels whose window
closed before each decision. The meta-labeler and track record learn only from
ideas whose outcomes were known at decision time.
"""
from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass, field

import pandas as pd

from alpha.agents.models import SealedSourceSnapshot
from alpha.agents.providers import DeterministicOfflineProvider
from alpha.fund.fund import IdeaDecision, LLMFirstFund
from alpha.fund.ideas import DeterministicOriginator
from alpha.fund.learning import MetaLabeler, ResolvedCall
from alpha.fund.sizing import RiskBook, SizedIdea
from alpha.fund.synthetic import BENCHMARK, SyntheticWorld, WorldConfig, build_world
from alpha.fund.verification import GateThresholds, side_probability
from alpha.research.collectors.synthetic import world_records
from alpha.research.dossier import build_dossier
from alpha.research.offline import DossierAnalystProvider, DossierOriginator
from alpha.research.store import ResearchStore
from alpha.fund.verification import verify_idea
from alpha.verifier.committee import Committee, ConsensusRules, JuryData, default_jurors
from alpha.verifier.market_head import HORIZONS, MarketVerifier, structured_features
from demo.contracts import (
    AssetBar,
    ExperimentConfig,
    MarketSession,
    PositionTarget,
    WeightProposal,
)
from demo.execution import simulate_portfolio
from demo.metrics import compute_performance_metrics

ARMS = ("llm_only", "llm_desk", "ml_only", "committee_only", "fusion_single", "fusion")
_FEATURE_WARMUP = 252


@dataclass(frozen=True)
class BacktestConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    train_sessions: int = 120
    rebalance_every: int = 5
    retrain_every: int = 20
    panel_stride: int = 1
    gbm_iters: int = 40
    ensemble: int = 2
    starting_capital: float = 1_000_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    ml_only_min_probability: float = 0.58
    ml_only_materiality_bps: float = 25.0
    ml_only_per_side: int = 4
    ml_horizon_days: int = 5
    meta_min_samples: int = 40
    meta_regularization: float = 1.0
    snapshots: str = "dossier"          # "dossier" (research pipeline) or "narrative"
    use_committee: bool = True
    committee_rules: ConsensusRules = field(default_factory=ConsensusRules)
    committee_min_probability: float = 0.53
    risk: RiskBook = field(default_factory=RiskBook)
    thresholds: GateThresholds = field(default_factory=GateThresholds)


@dataclass
class BacktestReport:
    arms: dict[str, dict]
    ideas: dict[str, object]
    ledger_head: str
    ledger_entries: int
    config: dict
    committee: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim_status": "synthetic_mechanism_test_not_market_evidence",
            "config": self.config,
            "arms": self.arms,
            "ideas": self.ideas,
            "ledger": {"entries": self.ledger_entries, "head": self.ledger_head},
            "committee": self.committee,
        }

    def table(self) -> str:
        header = f"{'arm':<16}{'return':>9}{'sharpe':>8}{'max_dd':>9}{'turnover':>10}{'costs':>9}"
        lines = [header, "-" * len(header)]
        for arm in ARMS:
            m = self.arms[arm]
            sharpe = "n/a" if m["sharpe_ratio"] is None else f"{m['sharpe_ratio']:+.2f}"
            lines.append(
                f"{arm:<16}{m['total_return']:>+9.2%}{sharpe:>8}{m['max_drawdown']:>+9.2%}"
                f"{m['total_turnover']:>10.2f}{m['transaction_cost_pct']:>9.3%}"
            )
        ideas = self.ideas
        lines += [
            "",
            f"ideas originated: {ideas['originated']}  desk-approved: {ideas['desk_approved']}"
            f"  ML-approved: {ideas['ml_approved']}  ML-vetoed/abstained: {ideas['ml_blocked']}",
        ]
        for label, key in (
            ("hit rate, all LLM ideas", "hit_rate_originated"),
            ("hit rate, desk-approved", "hit_rate_desk_approved"),
            ("hit rate, ML-approved", "hit_rate_ml_approved"),
            ("hit rate, ML-blocked", "hit_rate_ml_blocked"),
        ):
            value = ideas[key]
            lines.append(f"{label:<26}{'n/a' if value is None else f'{value:.1%}':>8}")
        by_kind = ideas.get("ml_gate_pass_rate_by_event_kind", {})
        if by_kind:
            lines.append(
                "ML gate pass rate by planted event kind: "
                + ", ".join(f"{kind}={rate:.0%}" for kind, rate in sorted(by_kind.items()))
            )
        if self.committee:
            seated = sorted((r for r in self.committee if r["seated"]), key=lambda r: -r["weight"])
            lines.append(
                f"committee: {len(seated)} of {len(self.committee)} jurors seated at the end of the run"
            )
            for row in seated:
                lines.append(
                    f"  {row['juror']:<30}{row['family']:<13}IC {row['mean_ic']:+.3f}  t {row['ic_t']:+.1f}"
                )
        return "\n".join(lines)


@contextlib.contextmanager
def _gbm_iterations(iterations: int):
    previous = os.environ.get("FUSION_CPU_GBM_ITERS")
    os.environ["FUSION_CPU_GBM_ITERS"] = str(iterations)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("FUSION_CPU_GBM_ITERS", None)
        else:
            os.environ["FUSION_CPU_GBM_ITERS"] = previous


def run_backtest(
    config: BacktestConfig = BacktestConfig(),
    *,
    world: SyntheticWorld | None = None,
    originator=None,
    analyst=None,
) -> tuple[BacktestReport, LLMFirstFund]:
    if config.snapshots not in {"dossier", "narrative"}:
        raise ValueError("snapshots must be 'dossier' or 'narrative'")
    world = world or build_world(config.world)
    dossier_mode = config.snapshots == "dossier"
    store = ResearchStore()
    store.add(world_records(world))
    fund = LLMFirstFund(
        originator=originator or (DossierOriginator() if dossier_mode else DeterministicOriginator()),
        analyst=analyst or (DossierAnalystProvider() if dossier_mode else DeterministicOfflineProvider()),
        thresholds=config.thresholds,
        risk=config.risk,
        meta=MetaLabeler(
            min_samples=config.meta_min_samples, regularization=config.meta_regularization
        ),
    )
    tickers = list(world.tickers)
    closes = world.closes[tickers]
    total = len(world.dates)
    first_decision = _FEATURE_WARMUP + config.train_sessions
    if first_decision + config.rebalance_every >= total:
        raise ValueError("world is too short for the requested warm-up and training window")

    features: dict[int, pd.DataFrame] = {}
    labels: dict[int, dict[int, pd.Series]] = {}

    def features_at(index: int) -> pd.DataFrame:
        if index not in features:
            features[index] = structured_features(
                closes, world.volume, world.dates[index], tickers
            )
        return features[index]

    close_matrix = closes.to_numpy(dtype=float)

    def labels_at(index: int) -> dict[int, pd.Series]:
        # Vectorised equivalent of market_head.residual_forward for a complete,
        # survivor-free tape: close-to-close return strictly after ``index``,
        # demeaned across names.
        if index not in labels:
            labels[index] = {}
            for h in HORIZONS:
                if index + h >= total:
                    continue
                forward = close_matrix[index + h] / close_matrix[index] - 1.0
                labels[index][h] = pd.Series(forward - forward.mean(), index=tickers)
        return labels[index]

    decisions_by_index: dict[int, list[IdeaDecision]] = {}
    targets: dict[str, list[WeightProposal]] = {arm: [] for arm in ARMS}
    pending: list[tuple[int, IdeaDecision]] = []
    resolved: list[ResolvedCall] = []
    outcomes: list[tuple[IdeaDecision, float, str]] = []
    verifier: MarketVerifier | None = None
    last_fit = -10**9
    decision_indices = list(range(first_decision, total - 1, config.rebalance_every))

    horizon = config.committee_rules.horizon
    committee = Committee(jurors=default_jurors(), rules=config.committee_rules) if config.use_committee else None
    jury_data = JuryData(
        dates=world.dates, closes=closes, volume=world.volume[tickers],
        benchmark=world.closes[BENCHMARK], view_at=store.view, as_of_at=world.as_of,
    )

    def closed_label(i: int) -> pd.Series:
        return labels_at(i)[horizon]

    def closed_path(i: int) -> pd.DataFrame:
        return closes.iloc[i + 1:i + horizon + 1]

    if committee is not None:
        # Burn-in: score the committee on the same clock before the first decision
        # so every juror has an out-of-sample record when decisions begin.
        start = first_decision - config.rebalance_every * ((first_decision - 253) // config.rebalance_every)
        for index in range(start, first_decision, config.rebalance_every):
            committee.observe(jury_data, index, labels=closed_label, paths=closed_path)

    for index in decision_indices:
        if index - last_fit >= config.retrain_every:
            verifier = _fit_verifier(config, world, index, features_at, labels_at)
            last_fit = index
        frame = features_at(index)
        forecasts = verifier.forecast_frame(frame) if verifier is not None else {}
        if committee is not None:
            jury_data.market_forecasts[index] = forecasts
            committee.observe(jury_data, index, labels=closed_label, paths=closed_path)
            jury_data.forget(index - 2 * config.committee_rules.max_training_dates * config.rebalance_every)

        newly = [(start, item) for start, item in pending if start + item.horizon_days < index]
        pending = [(start, item) for start, item in pending if start + item.horizon_days >= index]
        new_calls = []
        for start, item in newly:
            realized = labels_at(start).get(item.horizon_days, pd.Series(dtype=float)).get(item.ticker)
            if realized is None or not math.isfinite(realized):
                continue
            outcomes.append((item, float(realized), _event_kind(world, item.ticker, start, config)))
            if item.meta_features:
                new_calls.append(ResolvedCall(
                    model_id=item.originator_model,
                    claim_type=item.claim_type,
                    side=item.side,
                    conviction=item.conviction,
                    features=item.meta_features,
                    realized_residual=float(realized),
                    label_end_index=start + item.horizon_days,
                    contaminated=item.contaminated,
                ))
        fund.reliability.update(new_calls)
        resolved.extend(new_calls)
        fund.meta.fit(resolved)

        as_of = world.as_of(index)
        view = store.view(as_of) if dossier_mode else None
        decisions = []
        for ticker in tickers:
            state = tuple(float(v) for v in frame.loc[ticker]) if ticker in frame.index else None
            if view is not None:
                dossier = build_dossier(
                    view, ticker, benchmark=BENCHMARK,
                    news_lookback_days=config.rebalance_every,
                    retail_window_days=config.rebalance_every,
                )
                snapshot = dossier.snapshot()
                if state is not None:
                    state = (*state, *dossier.feature_vector())
            else:
                snapshot = SealedSourceSnapshot.seal(
                    world.snapshot_documents(ticker, index, lookback=config.rebalance_every)
                )
            decision = fund.evaluate(
                ticker=ticker,
                as_of=as_of,
                snapshot=snapshot,
                forecast=forecasts.get(ticker),
                market_state=state,
                synthetic_world=True,
                jury=(lambda side, t=ticker: committee.vote(t, side)) if committee is not None else None,
            )
            decisions.append(decision)
            if decision.side:
                pending.append((index, decision))
        decisions_by_index[index] = decisions

        session = world.dates[index].date()
        arm_weights = {
            "llm_only": config.risk.targets([
                _sized(item, item.p_llm) for item in decisions if item.side
            ]),
            "llm_desk": config.risk.targets([
                _sized(item, item.p_llm) for item in decisions
                if item.side and item.stage != "desk_rejected"
            ]),
            "ml_only": config.risk.targets(_ml_only_ideas(config, forecasts)),
            "committee_only": config.risk.targets(_committee_only_ideas(config, committee, forecasts)),
            "fusion_single": config.risk.targets(_single_verifier_ideas(fund, decisions, forecasts)),
            "fusion": fund.size(decisions),
        }
        for arm, weights in arm_weights.items():
            targets[arm].append(WeightProposal(
                strategy_id=arm,
                decision_session=session,
                targets=tuple(PositionTarget(ticker=t, weight=w) for t, w in weights.items()),
            ))

    experiment = ExperimentConfig(
        schema_version="llm-first-1",
        experiment_id=f"llm-first-synthetic-seed{config.world.seed}",
        start_date=world.dates[first_decision].date(),
        end_date=world.dates[-1].date(),
        starting_capital=config.starting_capital,
        universe=tuple(tickers),
        benchmark_ticker=BENCHMARK,
        rebalance_frequency_sessions=config.rebalance_every,
        transaction_cost_bps=config.transaction_cost_bps,
        slippage_bps=config.slippage_bps,
        max_gross_leverage=config.risk.max_gross,
        max_position_weight=config.risk.max_position_weight,
    )
    sessions = _sessions(world, first_decision)
    benchmark = [float(world.closes.at[world.dates[i], BENCHMARK]) for i in range(first_decision, total)]
    arms = {}
    for arm in ARMS:
        result = simulate_portfolio(
            config=experiment, sessions=sessions, proposals=targets[arm], strategy_id=arm
        )
        metrics = compute_performance_metrics(result, config=experiment, benchmark_values=benchmark)
        arms[arm] = {
            key: (round(value, 6) if isinstance(value, float) else value)
            for key, value in metrics.model_dump(mode="json").items()
        }

    fund.ledger.verify()
    report = BacktestReport(
        arms=arms,
        ideas=_idea_diagnostics(decisions_by_index, outcomes),
        ledger_head=fund.ledger.head,
        ledger_entries=len(fund.ledger.entries),
        config={
            "seed": config.world.seed,
            "names": config.world.n_names,
            "sessions": config.world.n_sessions,
            "first_decision": world.dates[first_decision].date().isoformat(),
            "rebalance_every": config.rebalance_every,
            "retrain_every": config.retrain_every,
            "transaction_cost_bps": config.transaction_cost_bps,
            "slippage_bps": config.slippage_bps,
            "originator_model": fund.originator.model_id,
            "analyst_model": fund.analyst.model_id,
            "gate_calibrated": config.thresholds.calibrated,
            "snapshots": config.snapshots,
            "committee": config.use_committee,
        },
        committee=[] if committee is None else committee.roster(),
    )
    return report, fund


def _fit_verifier(config, world, index, features_at, labels_at) -> MarketVerifier | None:
    cutoff = world.dates[index]
    panels = []
    for start in range(_FEATURE_WARMUP, index - 1, config.panel_stride):
        available = {h: y for h, y in labels_at(start).items() if start + h < index}
        if not available:
            continue
        ends = {h: world.dates[start + h] for h in available}
        panels.append((world.dates[start], features_at(start), available, ends))
    if not panels:
        return None
    verifier = MarketVerifier(backend="cpu-gbm", ensemble=config.ensemble, seed0=config.world.seed)
    with _gbm_iterations(config.gbm_iters):
        verifier.fit(panels, decision_cutoff=cutoff)
    return verifier if verifier.models else None


def _sized(decision: IdeaDecision, probability: float) -> SizedIdea:
    return SizedIdea(
        ticker=decision.ticker, side=decision.side,
        probability=probability, daily_volatility=decision.daily_volatility,
    )


def _ml_only_ideas(config: BacktestConfig, forecasts: dict[str, dict]) -> list[SizedIdea]:
    key = f"{config.ml_horizon_days}d"
    longs, shorts = [], []
    for ticker, row in forecasts.items():
        bps = row.get("expected_residual_bps", {}).get(key)
        if bps is None or abs(bps) < config.ml_only_materiality_bps:
            continue
        side = 1 if bps > 0 else -1
        probability = side_probability(row, key, side)
        if probability is None or probability < config.ml_only_min_probability:
            continue
        variance = row.get("aleatoric_var", {}).get(key)
        volatility = (variance / config.ml_horizon_days) ** 0.5 if variance and variance > 0 else None
        (longs if side > 0 else shorts).append(
            (abs(bps), SizedIdea(ticker, side, probability, volatility))
        )
    picks = sorted(longs, key=lambda item: -item[0])[: config.ml_only_per_side]
    picks += sorted(shorts, key=lambda item: -item[0])[: config.ml_only_per_side]
    return [idea for _, idea in picks]


def _committee_only_ideas(config: BacktestConfig, committee, forecasts: dict) -> list[SizedIdea]:
    if committee is None:
        return []
    consensus = committee.consensus_up()
    key = f"{config.committee_rules.horizon}d"
    longs = consensus[consensus >= config.committee_min_probability].sort_values(ascending=False)
    shorts = consensus[consensus <= 1 - config.committee_min_probability].sort_values()
    ideas = []
    for side, picks in ((1, longs), (-1, shorts)):
        for ticker, p_up in picks.iloc[: config.ml_only_per_side].items():
            variance = forecasts.get(ticker, {}).get("aleatoric_var", {}).get(key)
            volatility = (variance / config.committee_rules.horizon) ** 0.5 if variance else None
            ideas.append(SizedIdea(ticker, side, p_up if side > 0 else 1 - p_up, volatility))
    return ideas


def _single_verifier_ideas(fund: LLMFirstFund, decisions: list[IdeaDecision], forecasts: dict) -> list[SizedIdea]:
    """Replay the gate for desk-cleared ideas with the single market head only."""
    ideas = []
    for item in decisions:
        if item.verdict is None or item.receipt is None:
            continue
        verdict = verify_idea(
            item.receipt, forecast=forecasts.get(item.ticker), p_llm=item.p_llm,
            p_meta=item.verdict.p_meta, thresholds=fund.thresholds,
        )
        if verdict.decision == "approved":
            ideas.append(SizedIdea(item.ticker, item.side, verdict.p_final, item.daily_volatility))
    return ideas


def _sessions(world: SyntheticWorld, first: int) -> list[MarketSession]:
    names = [*world.tickers, BENCHMARK]
    sessions = []
    for index in range(first, len(world.dates)):
        date = world.dates[index]
        sessions.append(MarketSession(
            session=date.date(),
            bars=tuple(
                AssetBar(
                    ticker=name,
                    open=float(world.opens.at[date, name]),
                    close=float(world.closes.at[date, name]),
                )
                for name in names
            ),
        ))
    return sessions


def _event_kind(world: SyntheticWorld, ticker: str, index: int, config: BacktestConfig) -> str:
    recent = [
        event for event in world.events
        if event.ticker == ticker and index - config.rebalance_every < event.session_index <= index
    ]
    return recent[-1].kind if recent else "none"


def _hit_rate(rows: list[tuple[IdeaDecision, float, str]]) -> float | None:
    if not rows:
        return None
    return round(sum(item.side * realized > 0 for item, realized, _ in rows) / len(rows), 6)


def _idea_diagnostics(decisions_by_index, outcomes) -> dict[str, object]:
    all_decisions = [item for items in decisions_by_index.values() for item in items]
    stages: dict[str, int] = {}
    for item in all_decisions:
        stages[item.stage] = stages.get(item.stage, 0) + 1
    originated = [row for row in outcomes if row[0].side]
    desk_ok = [row for row in originated if row[0].stage != "desk_rejected"]
    ml_ok = [row for row in desk_ok if row[0].stage == "approved"]
    ml_blocked = [row for row in desk_ok if row[0].stage in {"vetoed", "abstain"}]
    by_kind: dict[str, float] = {}
    for kind in sorted({row[2] for row in desk_ok}):
        rows = [row for row in desk_ok if row[2] == kind]
        by_kind[kind] = round(sum(row[0].stage == "approved" for row in rows) / len(rows), 6)
    return {
        "stages": dict(sorted(stages.items())),
        "originated": sum(1 for item in all_decisions if item.side),
        "desk_approved": sum(1 for item in all_decisions if item.side and item.stage != "desk_rejected"),
        "ml_approved": stages.get("approved", 0),
        "ml_blocked": stages.get("vetoed", 0) + stages.get("abstain", 0),
        "resolved": len(originated),
        "hit_rate_originated": _hit_rate(originated),
        "hit_rate_desk_approved": _hit_rate(desk_ok),
        "hit_rate_ml_approved": _hit_rate(ml_ok),
        "hit_rate_ml_blocked": _hit_rate(ml_blocked),
        "ml_gate_pass_rate_by_event_kind": by_kind,
    }


__all__ = ["ARMS", "BacktestConfig", "BacktestReport", "run_backtest"]
