"""Controlled three-arm baseline on the locked Barebone-window tape.

The tape is the gitignored Yahoo extract named by
``configs/barebone-comparison-v1.json``. This module refuses a missing file,
a hash mismatch, and the fair-race extract. It reuses the multi-name book and
the pure-ML out-of-sample skill gate. A rebalance with no completed
pre-decision label stays in cash. No price is invented.

The ledger and metrics are a software record. ``comparable_performance_claim``
stays false.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

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
from demo.barebone_comparison import (
    BAREBONE_EXPERIMENT_ID,
    BAREBONE_OHLCV,
    BAREBONE_PROVENANCE,
    BareboneComparisonConfig,
    load_barebone_comparison_config,
    refuse_fair_race_as_barebone_comparison,
    require_barebone_evidence,
)
from demo.barebone_tape import (
    BareboneProvenance,
    _OHLCV_SCHEMA,
    _YAHOO_PROVIDER,
    required_tickers,
)
from demo.contracts import AssetBar, ExperimentConfig, MarketSession
from demo.controlled import (
    ArmInput,
    MarketVerification,
    benchmark_marks_from_closes,
    load_policy_thresholds,
    run_three_arms,
)
from demo.market_tape import _adjusted_ohlcv
from demo.pure_ml import feature_panel_from_market, walk_forward_filing_proposals


LEDGER_RELATIVE = "results/barebone_three_arm_ledger.json"
METRICS_RELATIVE = "results/barebone_three_arm_metrics.json"
_FIXTURE_METRICS_STATUS = "controlled_software_fixture_metrics"
_CALIBRATION_RELATIVE = "results/fusion_policy_calibration.json"
_LEDGER_SCHEMA = "fusionfinance-barebone-three-arm-ledger-v1"
_METRICS_SCHEMA = "fusionfinance-barebone-three-arm-metrics-v1"
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

_STATE: dict[tuple[str, str, str], dict[str, object]] = {}


@dataclass(frozen=True, slots=True)
class BareboneMarket:
    """Adjusted sessions for the kernel, plus the QQQ close path."""

    config: BareboneComparisonConfig
    sessions: tuple[MarketSession, ...]
    panel: pd.DataFrame
    secondary_closes: tuple[float, ...]
    tape_sha256: str
    provenance: BareboneProvenance


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def parse_barebone_ohlcv(
    payload: Mapping[str, object], config: BareboneComparisonConfig
) -> tuple[tuple[MarketSession, ...], pd.DataFrame, tuple[float, ...]]:
    """Build adjusted sessions from one verified extract. Gaps are not filled."""

    if payload.get("schema") != _OHLCV_SCHEMA:
        raise ValueError("barebone ohlcv schema is missing or changed")
    bars = payload.get("bars")
    if not isinstance(bars, list) or not bars:
        raise ValueError("barebone ohlcv has no bars")
    trading = config.as_experiment_config()
    required = required_tickers(config)
    required_set = set(required)
    grouped: dict[date, dict[str, dict[str, float]]] = {}
    for record in bars:
        if not isinstance(record, dict):
            raise ValueError("barebone ohlcv bars must be JSON objects")
        session = date.fromisoformat(str(record["date"]))
        if session.weekday() >= 5:
            raise ValueError(
                f"barebone tape refuses a weekend session: {session.isoformat()}"
            )
        if session < trading.start_date or session > trading.end_date:
            raise ValueError(
                "barebone ohlcv session "
                f"{session.isoformat()} is outside the locked window"
            )
        ticker = str(record["ticker"]).strip().upper()
        if ticker not in required_set:
            raise ValueError(f"barebone ohlcv has an unexpected ticker: {ticker}")
        bucket = grouped.setdefault(session, {})
        if ticker in bucket:
            raise ValueError(
                f"duplicate ohlcv bar for {ticker} on {session.isoformat()}"
            )
        bucket[ticker] = _adjusted_ohlcv(record)
    if not grouped:
        raise ValueError("barebone ohlcv produced no sessions")
    ordered = tuple(sorted(grouped))
    if ordered[0] != trading.start_date or ordered[-1] != trading.end_date:
        raise ValueError(
            "barebone ohlcv must start and end on the locked window dates"
        )
    sessions: list[MarketSession] = []
    rows: list[dict[str, object]] = []
    secondary: list[float] = []
    tradable = (*trading.universe, trading.benchmark_ticker)
    for session in ordered:
        present = grouped[session]
        missing = [ticker for ticker in required if ticker not in present]
        if missing:
            raise ValueError(
                f"ohlcv session {session.isoformat()} missing ticker(s): {missing}"
            )
        if config.secondary_benchmark_ticker is None:
            raise ValueError("barebone-comparison-v1 requires the QQQ benchmark")
        secondary.append(present[config.secondary_benchmark_ticker]["close"])
        bar_rows = tuple(
            AssetBar(
                ticker=ticker,
                open=present[ticker]["open"],
                close=present[ticker]["close"],
            )
            for ticker in tradable
        )
        sessions.append(MarketSession(session=session, bars=bar_rows))
        for ticker in trading.universe:
            fields = present[ticker]
            rows.append(
                {
                    "date": pd.Timestamp(session),
                    "ticker": ticker,
                    "open": fields["open"],
                    "high": fields["high"],
                    "low": fields["low"],
                    "close": fields["close"],
                    "volume": fields["volume"],
                }
            )
    return tuple(sessions), pd.DataFrame(rows), tuple(secondary)


def load_barebone_market(root: Path | None = None) -> BareboneMarket:
    """Load the locked Yahoo tape. A missing or substituted file raises."""

    base = _repo_root() if root is None else root
    config = load_barebone_comparison_config(
        base / "configs" / "barebone-comparison-v1.json"
    )
    path = require_barebone_evidence(config, root=base)
    provenance_path = base / BAREBONE_PROVENANCE
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"barebone-comparison provenance is missing: {BAREBONE_PROVENANCE}"
        )
    provenance = BareboneProvenance.model_validate(
        json.loads(provenance_path.read_text(encoding="utf-8"))
    )
    if provenance.provider != _YAHOO_PROVIDER:
        raise ValueError("barebone three-arm run requires the Yahoo Finance tape")
    if provenance.byte_sha256 != config.evidence.tape_sha256:
        raise ValueError("provenance byte_sha256 does not match the locked tape")
    if provenance.ohlcv != BAREBONE_OHLCV:
        raise ValueError("provenance ohlcv path is not the barebone tape")
    if tuple(provenance.tickers) != required_tickers(config):
        raise ValueError("provenance tickers do not match the locked universe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("barebone ohlcv must be a JSON object")
    sessions, frame, secondary = parse_barebone_ohlcv(payload, config)
    bar_count = len(payload.get("bars", []))
    if bar_count != provenance.bar_count or len(sessions) != provenance.session_count:
        raise ValueError("provenance counts do not match the locked tape")
    if len(secondary) != len(sessions):
        raise ValueError("secondary benchmark path does not match the tape")
    panel = feature_panel_from_market(config.as_experiment_config(), frame)
    digest = config.evidence.tape_sha256
    if not isinstance(digest, str):
        raise ValueError("barebone-comparison-v1 refuses an unlocked evidence tape")
    return BareboneMarket(
        config=config,
        sessions=sessions,
        panel=panel,
        secondary_closes=secondary,
        tape_sha256=digest,
        provenance=provenance,
    )


def _seal_receipt(ticker: str, decision: date) -> DecisionReceipt:
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
    if receipt.decision != "approved" or receipt.thesis is None:
        raise ValueError("barebone desk seal did not produce an approved thesis")
    if receipt.thesis.committed_at != f"{decision.isoformat()}T20:00:00Z":
        raise ValueError("barebone desk seal committed_at does not match the decision")
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


def _secondary_wealth(
    closes: Sequence[float], sessions: Sequence[MarketSession], starting_capital: float
) -> dict[str, object]:
    if len(closes) != len(sessions) or not closes or closes[0] <= 0.0:
        raise ValueError("secondary benchmark closes must match the tape")
    base = closes[0]
    return {
        "ticker": "QQQ",
        "role": "adjusted-close price index on the locked tape",
        "sessions": [session.session.isoformat() for session in sessions],
        "values": [starting_capital * close / base for close in closes],
        "note": (
            "Close-to-close wealth of the secondary benchmark. "
            "Not an executed book and not a fund result."
        ),
    }


def barebone_three_arm_state(root: Path | None = None) -> dict[str, object]:
    """Run the three arms on the locked tape. The result is cached per root."""

    base = _repo_root() if root is None else root
    market = load_barebone_market(base)
    key = (str(base.resolve()), market.tape_sha256, market.config.scorebook)
    cached = _STATE.get(key)
    if cached is not None:
        return cached
    trading = market.config.as_experiment_config()
    dates = tuple(session.session for session in market.sessions)
    marks = benchmark_marks_from_closes(
        market.sessions,
        benchmark_ticker=trading.benchmark_ticker,
        starting_capital=trading.starting_capital,
    )
    thresholds = load_policy_thresholds(base / _CALIBRATION_RELATIVE)
    clock = [
        index
        for index in range(len(dates))
        if index % trading.rebalance_frequency_sessions == 0
        and index + trading.execution_lag_sessions < len(dates)
    ]
    decision_dates = tuple(dates[index] for index in clock)
    manifest = walk_forward_filing_proposals(
        trading,
        decision_dates,
        root=base,
        panel=market.panel,
        tape_dates=dates,
        cash_when_untrained=True,
        scorebook=market.config.scorebook,
    )
    proposals = {row["decision_session"]: row for row in manifest["proposals"]}
    pure_ml: list[ArmInput] = []
    pure_llm: list[ArmInput] = []
    fusion: list[ArmInput] = []
    seals: list[dict[str, object]] = []
    for ordinal, index in enumerate(clock):
        decision = dates[index]
        llm_ticker = trading.universe[ordinal % len(trading.universe)]
        proposal = proposals[decision.isoformat()]
        outcome = f"{dates[index + trading.execution_lag_sessions].isoformat()}T21:00:00Z"
        llm_receipt = _seal_receipt(llm_ticker, decision)
        kind = ("approve", "reject", "research_only")[ordinal % 3]
        market_head = _fusion_market(decision, kind, thresholds.calibration_hash)
        for target in proposal["targets"]:
            pure_ml.append(
                ArmInput(
                    strategy_id="pure_ml",
                    decision_session=decision,
                    ticker=str(target["ticker"]),
                    structured_weight=float(target["structured_weight"]),
                )
            )
        for target in proposal["fusion_targets"]:
            ticker = str(target["ticker"])
            fusion_receipt = _seal_receipt(ticker, decision)
            fusion.append(
                ArmInput(
                    strategy_id="fusion",
                    decision_session=decision,
                    ticker=ticker,
                    receipt=fusion_receipt,
                    market=market_head,
                    outcome_ts=outcome,
                    structured_weight=float(target["structured_weight"]),
                )
            )
            if fusion_receipt.thesis is None:
                raise ValueError("fusion seal is missing a thesis")
            seals.append(
                {
                    "arm": "fusion",
                    "decision_session": decision.isoformat(),
                    "ticker": ticker,
                    "committed_at": fusion_receipt.thesis.committed_at,
                    "outcome_ts": outcome,
                    "receipt_hash": fusion_receipt.receipt_hash,
                    "market_case": kind,
                }
            )
        pure_llm.append(
            ArmInput(
                strategy_id="pure_llm",
                decision_session=decision,
                ticker=llm_ticker,
                receipt=llm_receipt,
                outcome_ts=outcome,
            )
        )
        if llm_receipt.thesis is None:
            raise ValueError("pure_llm seal is missing a thesis")
        seals.append(
            {
                "arm": "pure_llm",
                "decision_session": decision.isoformat(),
                "ticker": llm_ticker,
                "committed_at": llm_receipt.thesis.committed_at,
                "outcome_ts": outcome,
                "receipt_hash": llm_receipt.receipt_hash,
                "market_case": None,
            }
        )
    arms = run_three_arms(
        config=trading,
        sessions=market.sessions,
        arms={"pure_ml": pure_ml, "pure_llm": pure_llm, "fusion": fusion},
        benchmark_marks=marks,
        thresholds=thresholds,
    )
    for run in arms:
        if run.block_reason is not None or run.ledger is None or run.metrics is None:
            raise ValueError(
                f"{run.lineage.strategy_id} blocked: {run.block_reason}"
            )
    state = {
        "market": market,
        "trading": trading,
        "dates": dates,
        "thresholds": thresholds,
        "clock": clock,
        "seals": seals,
        "manifest": manifest,
        "by_name": {run.lineage.strategy_id: run for run in arms},
        "secondary": _secondary_wealth(
            market.secondary_closes, market.sessions, trading.starting_capital
        ),
    }
    _STATE[key] = state
    return state


def _mapping_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        found.update(str(key) for key in value)
        for item in value.values():
            found.update(_mapping_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_mapping_keys(item))
    return found


def _arm_ledger(run: object) -> dict[str, object]:
    lineage = run.lineage
    ledger = run.ledger
    if ledger is None:
        raise ValueError("barebone arm is missing a ledger")
    return {
        "strategy_id": lineage.strategy_id,
        "claim_status": run.claim_status,
        "comparable_performance_claim": False,
        "config_hash": lineage.config_hash,
        "tape_hash": lineage.tape_hash,
        "lineage_hash": lineage.lineage_hash,
        "experiment_hash": lineage.experiment_hash,
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
            for row in lineage.proposals
        ],
    }


def _oos_skill_log(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in manifest["proposals"]:
        if not isinstance(row, Mapping):
            raise ValueError("pure-ML proposal rows must be objects")
        entry: dict[str, object] = {
            "decision_session": row["decision_session"],
            "oos_skill": row["oos_skill"],
            "oos_skill_pairs": row["oos_skill_pairs"],
            "oos_skill_threshold": row["oos_skill_threshold"],
            "sized": bool(row["skill_pass"]),
        }
        if row.get("untrained") is True:
            entry["untrained"] = True
        if row.get("insufficient_history") is True:
            entry["insufficient_history"] = True
        rows.append(entry)
    return rows


def _bind_pure_ml(document: dict[str, object], manifest: Mapping[str, object]) -> None:
    arms = document["arms"]
    if not isinstance(arms, dict):
        raise ValueError("barebone ledger is missing arms")
    arm = arms["pure_ml"]
    if manifest.get("scorebook") == "momentum":
        weight_rule = (
            "the score is the 63-session log return of adjusted close "
            "(Yahoo adjclose) minus the cross-sectional median at the "
            "decision session. Names without that history are excluded and "
            "not filled. Prices after the decision session are not features. "
            "An expanding Spearman skill of those scores versus the "
            "next-session residual must be strictly above "
            "oos_skill_threshold before sizing; a pair is used only once "
            "its outcome session is already before the decision. "
            "Non-positive skill leaves the book in cash. When the gate "
            "passes, positive scores share max_gross_leverage in proportion "
            "to score, each name capped at max_position_weight, then scaled "
            "so post-cost gross leverage stays inside the cap"
        )
        binding = {
            "model": manifest["model"],
            "scorebook": manifest["scorebook"],
            "lookback_sessions": manifest["lookback_sessions"],
            "price": manifest["price"],
            "transform": manifest["transform"],
            "proposal_manifest_hash": manifest["proposal_manifest_hash"],
            "oos_skill_threshold": manifest["oos_skill_threshold"],
            "oos_skill": manifest["oos_skill"],
            "weight_rule": weight_rule,
        }
    else:
        weight_rule = (
            "an expanding walk-forward Spearman skill of held-out "
            "fit_fusion_model scores versus next-session residual returns "
            "must be strictly above oos_skill_threshold before sizing; "
            "otherwise the book is cash. A rebalance with no completed "
            "pre-decision label is also cash. When the gate passes, positive "
            "scores share max_gross_leverage in proportion to score, each "
            "name capped at max_position_weight, then scaled so post-cost "
            "gross leverage stays inside the cap"
        )
        binding = {
            "model": manifest["model"],
            "scorebook": "ridge",
            "ridge_alpha": manifest["ridge_alpha"],
            "horizon_sessions": manifest["horizon_sessions"],
            "proposal_manifest_hash": manifest["proposal_manifest_hash"],
            "amd_evidence_sha256": manifest["amd_evidence_sha256"],
            "amd_compute_sha256": manifest["amd_compute_sha256"],
            "oos_skill_threshold": manifest["oos_skill_threshold"],
            "oos_skill": manifest["oos_skill"],
            "weight_rule": weight_rule,
        }
    arm["model_binding"] = binding
    arm["oos_skill"] = _oos_skill_log(manifest)
    by_session = {
        row["decision_session"]: row
        for row in manifest["proposals"]
        if isinstance(row, Mapping)
    }
    for decision in arm["decisions"]:
        proposal = by_session[decision["decision_session"]]
        if not proposal["skill_pass"]:
            raise ValueError("pure_ml sized a book when OOS skill failed the gate")
        if proposal.get("untrained") is True:
            raise ValueError("pure_ml sized a book with no pre-decision training rows")
        if proposal.get("insufficient_history") is True:
            raise ValueError("pure_ml sized a book without 63 sessions of history")
        match = next(
            target
            for target in proposal["targets"]
            if target["ticker"] == decision["ticker"]
        )
        decision["model_score"] = match["model_score"]
        decision["train_rows"] = proposal["train_rows"]
        decision["oos_skill"] = proposal["oos_skill"]
        decision["oos_skill_pairs"] = proposal["oos_skill_pairs"]
        decision["oos_skill_threshold"] = proposal["oos_skill_threshold"]
        if not math.isclose(decision["target_weight"], match["structured_weight"]):
            raise ValueError("executed pure_ml weight does not match the score book")


def _market_provenance(market: BareboneMarket) -> dict[str, object]:
    provenance = market.provenance
    return {
        "provider": provenance.provider,
        "fetched_at": provenance.fetched_at,
        "byte_sha256": provenance.byte_sha256,
        "disclaimer": provenance.disclaimer,
        "license_note": provenance.license_note,
        "adjustment": provenance.adjustment,
    }


def barebone_three_arm_ledger(root: Path | None = None) -> dict[str, object]:
    """Sealed baseline ledger for the locked Barebone window. Not a claim."""

    state = barebone_three_arm_state(root)
    market: BareboneMarket = state["market"]
    trading: ExperimentConfig = state["trading"]
    dates = state["dates"]
    thresholds = state["thresholds"]
    by_name = state["by_name"]
    document: dict[str, object] = {
        "schema": _LEDGER_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "claim_status": "controlled_software_ledger",
        "comparable_performance_claim": False,
        "description": (
            "Software ledger for the locked barebone-comparison-v1 window "
            "2025-01-02 through 2026-01-12, rebalanced every 10 sessions. "
            "Prices are the local Yahoo Finance tape after scaling OHLC by "
            "Adj Close over Close. Pure ML scores are the 63-session log "
            "return of that adjusted close, minus the cross-sectional median "
            "at the decision. A name without 63 earlier sessions is excluded "
            "and not filled. The next-session residual is the skill label, "
            "not a feature. Pure ML sizes a multi-name book only when the "
            "expanding Spearman skill of those scores is strictly above zero. "
            "Non-positive skill leaves that rebalance in cash. When the gate "
            "passes, positive scores share the gross budget under the locked "
            "position and gross caps. Market-approved fusion uses the same "
            "momentum scores without that skill gate. LLM arms use the desk's "
            "first orchestrator seal. Fusion still requires a market head. "
            "QQQ is a secondary price index on the metrics file, not a "
            "tradable name. This is not a performance claim."
        ),
        "tape_sha256": market.tape_sha256,
        "price_source": {
            "ohlcv": BAREBONE_OHLCV,
            "tape_sha256": market.tape_sha256,
            "market_data": _market_provenance(market),
        },
        "window": [trading.start_date.isoformat(), trading.end_date.isoformat()],
        "session_count": len(dates),
        "rebalance_frequency_sessions": trading.rebalance_frequency_sessions,
        "rebalance_count": len(state["clock"]),
        "max_position_weight": trading.max_position_weight,
        "max_gross_leverage": trading.max_gross_leverage,
        "calibration_hash": thresholds.calibration_hash,
        "calibration_path": _CALIBRATION_RELATIVE,
        "provenance": {
            "desk": "alpha.agents.desk.AgentDesk",
            "orchestrator": "alpha.agents.orchestrator.FusionOrchestrator",
            "provider": "fusionfinance-offline-lexical-v1",
            "seal": "orchestrator_first_commit",
            "market_data": _market_provenance(market),
        },
        "seals": state["seals"],
        "arms": {
            name: _arm_ledger(by_name[name])
            for name in ("pure_ml", "pure_llm", "fusion")
        },
    }
    _bind_pure_ml(document, state["manifest"])
    _refuse_claim_document(document)
    return document


def _fixture_context(
    market: BareboneMarket, manifest: Mapping[str, object]
) -> dict[str, object]:
    context: dict[str, object] = {
        "prices": f"{BAREBONE_OHLCV} adjusted OHLC",
        "ohlcv_sha256": market.tape_sha256,
        "market_data": _market_provenance(market),
        "llm_provider": "fusionfinance-offline-lexical-v1",
        "pure_ml_model": manifest["model"],
        "proposal_manifest_hash": manifest["proposal_manifest_hash"],
        "scorebook": market.config.scorebook,
    }
    if manifest.get("scorebook") == "momentum":
        context["lookback_sessions"] = manifest["lookback_sessions"]
        context["price"] = manifest["price"]
        context["transform"] = manifest["transform"]
    else:
        context["amd_compute_sha256"] = manifest["amd_compute_sha256"]
    return context


def barebone_three_arm_metrics(root: Path | None = None) -> dict[str, object]:
    """Fixture metrics for the locked Barebone-window tape. Not a claim."""

    state = barebone_three_arm_state(root)
    market: BareboneMarket = state["market"]
    trading: ExperimentConfig = state["trading"]
    dates = state["dates"]
    by_name = state["by_name"]
    manifest = state["manifest"]
    arms: dict[str, object] = {}
    for name in ("pure_ml", "pure_llm", "fusion"):
        run = by_name[name]
        if run.result is None or run.metrics is None or run.ledger is None:
            raise ValueError(f"{name} is missing an executed result")
        statistics = run.metrics.model_dump(mode="json")
        values = [point.portfolio_value for point in run.result.points]
        sessions = [point.session.isoformat() for point in run.result.points]
        if statistics["trade_count"] != run.ledger.trade_count:
            raise ValueError("metrics trade count does not match the ledger")
        if not math.isclose(
            statistics["total_turnover"],
            run.ledger.total_turnover,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError("metrics turnover does not match the ledger")
        if not math.isclose(
            statistics["transaction_costs"],
            run.ledger.transaction_costs,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError("metrics costs do not match the ledger")
        if not math.isclose(statistics["total_return"], values[-1] / values[0] - 1.0):
            raise ValueError("cumulative return does not match the wealth path")
        payload: dict[str, object] = {
            "strategy_id": name,
            "claim_status": _FIXTURE_METRICS_STATUS,
            "comparable_performance_claim": False,
            "config_hash": run.lineage.config_hash,
            "tape_hash": run.lineage.tape_hash,
            "tape_sha256": market.tape_sha256,
            "lineage_hash": run.lineage.lineage_hash,
            "experiment_hash": run.lineage.experiment_hash,
            "portfolio_sessions": sessions,
            "portfolio_values": values,
            "statistics": statistics,
        }
        if name == "pure_ml":
            payload["oos_skill"] = _oos_skill_log(manifest)
        arms[name] = payload
    document: dict[str, object] = {
        "schema": _METRICS_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "claim_status": _FIXTURE_METRICS_STATUS,
        "comparable_performance_claim": False,
        "description": (
            "Fixture metrics from the barebone-comparison-v1 controlled tape. "
            "Prices are the local Yahoo Finance extract. Pure ML scores are "
            "63-session adjusted-close momentum, demeaned by the "
            "cross-sectional median, and the book is sized only when "
            "expanding Spearman skill versus the next-session residual is "
            "strictly above zero. Thin history stays in cash and is not "
            "filled. Market-approved fusion uses those momentum scores "
            "without the skill gate. The LLM desk is the offline lexical "
            "provider. QQQ is a secondary adjusted-close index, not an "
            "executed book. Not a capital performance claim."
        ),
        "tape_sha256": market.tape_sha256,
        "source_ledger": LEDGER_RELATIVE,
        "fixture_context": _fixture_context(market, manifest),
        "window": [trading.start_date.isoformat(), trading.end_date.isoformat()],
        "session_count": len(dates),
        "max_position_weight": trading.max_position_weight,
        "max_gross_leverage": trading.max_gross_leverage,
        "secondary_benchmark": state["secondary"],
        "arms": arms,
    }
    _refuse_claim_document(document)
    return document


def _refuse_claim_document(document: Mapping[str, object]) -> None:
    refuse_fair_race_as_barebone_comparison(document)
    if document.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    if document.get("experiment_id") != BAREBONE_EXPERIMENT_ID:
        raise ValueError("experiment_id must be barebone-comparison-v1")
    if document.get("schema") == _LEDGER_SCHEMA and _PERFORMANCE_KEYS.intersection(
        _mapping_keys(document)
    ):
        raise ValueError("barebone ledger must not carry performance-claim fields")


def write_barebone_three_arm_artifacts(root: Path | None = None) -> tuple[Path, Path]:
    """Write the ledger and metrics. Does not write the OHLCV extract."""

    base = _repo_root() if root is None else root
    ledger_path = base / LEDGER_RELATIVE
    metrics_path = base / METRICS_RELATIVE
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(_json_text(barebone_three_arm_ledger(base)), encoding="utf-8")
    metrics_path.write_text(
        _json_text(barebone_three_arm_metrics(base)), encoding="utf-8"
    )
    return ledger_path, metrics_path
