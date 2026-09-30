"""Publication-time short interest, point-in-time scores, and the skill gate."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta, timezone
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest

import demo.barebone_short_interest as si_module
import demo.barebone_short_interest_run as si_run_module
from demo.barebone_comparison import (
    BAREBONE_UNIVERSE,
    LOCKED_EDGAR_SHA256,
    LOCKED_FORM4_SHA256,
    LOCKED_NARRATIVE_SHA256,
    LOCKED_TAPE_SHA256,
    SHORT_INTEREST_EVENTS,
    SHORT_INTEREST_PROVENANCE,
    SHORT_INTEREST_SYMBOL_MAP,
    load_barebone_comparison_config,
)
from demo.barebone_short_interest import (
    FINRA_ENDPOINT,
    SI_SIGNAL_ID,
    assert_official_publication_calendar,
    available_stamp,
    events_from_records,
    fetch_short_interest_events,
    load_symbol_map,
    main as ingest_main,
    publication_date,
    short_interest_provenance,
    short_interest_score_rows,
    write_locked_short_interest_sha256,
)
from demo.barebone_short_interest_run import (
    ARM_RELATIVE,
    LEDGER_RELATIVE,
    METRICS_RELATIVE,
    _arm_inputs,
    _load_events,
    _refuse_sized_without_skill,
    barebone_short_interest_arm_index,
    barebone_short_interest_ledger,
    barebone_short_interest_metrics,
    main as run_main,
    raw_ratio_by_session,
    short_interest_book,
    short_interest_oos_pairs,
    short_interest_rebalances,
    skill_before,
    write_barebone_short_interest_artifacts,
)
from demo.contracts import AssetBar, ExperimentConfig, MarketSession
from demo.controlled import ArmInput, benchmark_marks_from_closes, run_controlled_arm

_LEGACY_METRICS_SHA256 = "e7e5055ce9b4409d7941a71929f66261416b8d3c3f62423190edafd5bf5b1411"
_THREE_ARM_METRICS_SHA256 = "af40d4176d3e4705f799b3b640494777caa24ed81ea68f149a84bb8335d554f3"
_FORM4_METRICS_SHA256 = "fea41d3a60804900847ee621584683a73283957a67d1ed5718fa23a1efa6a4c9"
_FORM4_LEDGER_SHA256 = "dbe92a8b3fe0cfd3750e3b27afed740eb8dc836233d2e7f2a52fe32e1b58ae82"
_MOMENTUM_RETURN = 0.24597743358179014
_MOMENTUM_SHARPE = 1.6030790628765745


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _weekdays(start: date, count: int) -> tuple[date, ...]:
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return tuple(days)


def _record(
    symbol: str,
    settlement: str,
    current: int,
    previous: int,
    *,
    revision: str | None = None,
) -> dict[str, object]:
    return {
        "symbolCode": symbol,
        "settlementDate": settlement,
        "currentShortPositionQuantity": current,
        "previousShortPositionQuantity": previous,
        "changePreviousNumber": current - previous,
        "revisionFlag": revision,
        "stockSplitFlag": None,
        "marketClassCode": "NNM",
    }


def _trading() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "schema_version": "1.0",
            "experiment_id": "barebone-comparison-v1",
            "start_date": "2025-01-02",
            "end_date": "2025-03-22",
            "starting_capital": 10000.0,
            "universe": ["AAA", "BBB"],
            "benchmark_ticker": "SPY",
            "rebalance_frequency_sessions": 10,
            "execution_lag_sessions": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_gross_leverage": 1.0,
            "max_position_weight": 0.1,
            "annualization_sessions": 252,
            "annual_risk_free_rate": 0.0,
        }
    )


def test_publication_is_not_the_settlement_date() -> None:
    assert_official_publication_calendar()
    settlement = date(2025, 12, 31)
    published = publication_date(settlement)
    assert published == date(2026, 1, 12)
    assert published != settlement
    assert available_stamp(published) == "2026-01-12T00:00:00Z"
    assert not available_stamp(published).startswith(settlement.isoformat())
    assert publication_date(date(2024, 12, 31)) == date(2025, 1, 10)
    with pytest.raises(ValueError, match="missing"):
        publication_date(None)  # type: ignore[arg-type]


def test_missing_settlement_revisions_and_unmapped_symbols_fail() -> None:
    sessions = (date(2025, 11, 26),)
    kept, revised = events_from_records(
        [
            _record("AAPL", "2020-04-15", 10, 8),
            _record("AAPL", "2025-11-14", 110, 100),
            _record("AAPL", "2025-11-28", 90, 100, revision="R"),
        ],
        ticker="AAPL",
        symbol="AAPL",
        sessions=sessions,
    )
    assert revised == 1
    assert len(kept) == 1
    assert kept[0]["settlement_date"] == "2025-11-14"
    assert kept[0]["publication_date"] == "2025-11-25"
    assert kept[0]["available_ts"] == "2025-11-25T00:00:00Z"
    assert kept[0]["decision_session"] == "2025-11-26"
    assert kept[0]["change_ratio"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="settlementDate"):
        events_from_records(
            [{"symbolCode": "AAPL", "revisionFlag": None}],
            ticker="AAPL",
            symbol="AAPL",
            sessions=sessions,
        )
    with pytest.raises(ValueError, match="not the frozen map"):
        events_from_records(
            [_record("ZZZZ", "2025-11-14", 110, 100)],
            ticker="AAPL",
            symbol="AAPL",
            sessions=sessions,
        )
    broken = _record("AAPL", "2025-11-14", 110, 100)
    broken["changePreviousNumber"] = 1
    with pytest.raises(ValueError, match="change does not match"):
        events_from_records([broken], ticker="AAPL", symbol="AAPL", sessions=sessions)
    zero = _record("AAPL", "2025-11-14", 5, 0)
    with pytest.raises(ValueError, match="no previous"):
        events_from_records([zero], ticker="AAPL", symbol="AAPL", sessions=sessions)


def test_later_publication_does_not_change_the_earlier_score() -> None:
    sessions = _weekdays(date(2025, 8, 1), 120)
    early = [
        *_pair("2025-11-14", 1100, 1000, 900, 1000),
    ]
    late = [
        *_pair("2025-12-31", 800, 1000, 1200, 1000),
    ]
    early_events = _events(early, sessions)
    both = _events(early + late, sessions)
    early_rows = _by_session(short_interest_score_rows(early_events, sessions, ("AAPL", "MSFT")))
    both_rows = _by_session(short_interest_score_rows(both, sessions, ("AAPL", "MSFT")))
    early_day = date.fromisoformat(str(early_events[0]["decision_session"]))
    late_day = date.fromisoformat(str(both[-1]["decision_session"]))
    assert early_rows[early_day]["AAPL"] == pytest.approx(0.1)
    assert both_rows[early_day]["AAPL"] == pytest.approx(0.1)
    assert both_rows[early_day] == early_rows[early_day]
    settlement_gate = min(day for day in sessions if day > date(2025, 12, 31))
    assert settlement_gate < late_day
    assert both_rows[settlement_gate]["AAPL"] == pytest.approx(0.1)
    assert both_rows[late_day]["AAPL"] == pytest.approx(-0.2)
    forged = dict(both[0])
    forged["available_ts"] = "2025-11-14T00:00:00Z"
    with pytest.raises(ValueError, match="publication date"):
        short_interest_score_rows([forged, both[1]], sessions, ("AAPL", "MSFT"))


def _pair(
    settlement: str, aapl_current: int, aapl_previous: int, msft_current: int, msft_previous: int
) -> list[dict[str, object]]:
    return [
        _record("AAPL", settlement, aapl_current, aapl_previous),
        _record("MSFT", settlement, msft_current, msft_previous),
    ]


def _events(records: list[dict[str, object]], sessions: tuple[date, ...]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["symbolCode"]), []).append(record)
    rows: list[dict[str, object]] = []
    for symbol, items in grouped.items():
        kept, _revised = events_from_records(items, ticker=symbol, symbol=symbol, sessions=sessions)
        rows.extend(kept)
    return rows


def _by_session(rows: list[dict[str, object]]) -> dict[date, dict[str, float]]:
    found: dict[date, dict[str, float]] = {}
    for row in rows:
        found.setdefault(date.fromisoformat(str(row["session"])), {})[str(row["ticker"])] = float(
            row["change_ratio"]
        )
    return found


def _config_copy(tmp_path: Path, **evidence_updates: object) -> Path:
    source = json.loads((_root() / "configs" / "barebone-comparison-v1.json").read_text(encoding="utf-8"))
    source["evidence"].update(evidence_updates)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


def _install_map(root: Path) -> None:
    destination = root / SHORT_INTEREST_SYMBOL_MAP
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_root() / SHORT_INTEREST_SYMBOL_MAP, destination)


def test_symbol_map_refuses_unmapped_names_and_fetch_keeps_prior_locks(tmp_path: Path) -> None:
    mapping, _digest = load_symbol_map()
    assert mapping["BRK-B"] == "BRKB"
    assert mapping["AAPL"] == "AAPL"
    assert set(mapping) == set(BAREBONE_UNIVERSE)
    payload = json.loads((_root() / SHORT_INTEREST_SYMBOL_MAP).read_text(encoding="utf-8"))
    payload["entries"] = [entry for entry in payload["entries"] if entry["ticker"] != "NKE"]
    broken = tmp_path / "map.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing NKE"):
        load_symbol_map(broken)

    root = tmp_path / "repo"
    _install_map(root)
    sessions = (date(2025, 11, 26),)

    def _rows(symbol: str) -> list[object]:
        return [
            _record(symbol, "2020-04-15", 10, 8),
            _record(symbol, "2025-11-14", 110, 100),
            _record(symbol, "2025-11-28", 90, 100, revision="R"),
            _record(symbol, "2026-01-15", 50, 40),
        ]

    path = _config_copy(tmp_path, edgar_form4_sha256="a" * 64)
    with pytest.raises(ValueError, match="edgar_form4_sha256 is locked"):
        fetch_short_interest_events(
            config_path=path, root=root, lock_config=True, sessions=sessions, get_rows=_rows
        )
    path = _config_copy(tmp_path)
    digest = fetch_short_interest_events(
        config_path=path, root=root, lock_config=True, sessions=sessions, get_rows=_rows
    )
    locked = json.loads(path.read_text(encoding="utf-8"))
    assert locked["evidence"]["short_interest_sha256"] == digest
    assert locked["evidence"]["edgar_form4_sha256"] == LOCKED_FORM4_SHA256
    assert locked["evidence"]["edgar_sha256"] == LOCKED_EDGAR_SHA256
    assert locked["evidence"]["narrative_sha256"] == LOCKED_NARRATIVE_SHA256
    assert locked["evidence"]["tape_sha256"] == LOCKED_TAPE_SHA256
    events = (root / SHORT_INTEREST_EVENTS).read_bytes()
    assert hashlib.sha256(events).hexdigest() == digest
    assert b"2020-04-15" not in events
    sidecar = json.loads((root / SHORT_INTEREST_PROVENANCE).read_text(encoding="utf-8"))
    assert sidecar["settlement_used_as_available_ts"] is False
    assert sidecar["endpoint"] == FINRA_ENDPOINT
    assert sidecar["revised_rows_dropped"] == len(BAREBONE_UNIVERSE)
    assert sidecar["event_count"] == len(BAREBONE_UNIVERSE)

    empty_root = tmp_path / "empty"
    _install_map(empty_root)
    empty_dir = tmp_path / "empty-config"
    empty_dir.mkdir()
    empty_config = _config_copy(empty_dir)
    with pytest.raises(ValueError, match="no publication inside the window"):
        fetch_short_interest_events(
            config_path=empty_config,
            root=empty_root,
            lock_config=True,
            sessions=sessions,
            get_rows=lambda symbol: [_record(symbol, "2020-04-15", 10, 8)],
        )
    assert not (empty_root / SHORT_INTEREST_EVENTS).exists()


def test_rate_limit_and_symbol_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    def _rate(request: urllib.request.Request, timeout: int = 60) -> object:
        del timeout
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", _rate)
    with pytest.raises(ValueError, match="rate limit"):
        si_module._request({"limit": 1})
    with pytest.raises(ValueError, match="symbolCode"):
        si_module._default_get_rows("BRK.B")


def test_lock_refuses_changed_binds_and_controls_stay_put(tmp_path: Path) -> None:
    path = _config_copy(tmp_path, edgar_sha256="a" * 64)
    before = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="changed edgar_sha256"):
        write_locked_short_interest_sha256(path, "b" * 64)
    unchanged = json.loads(path.read_text(encoding="utf-8"))
    assert unchanged["evidence"]["edgar_sha256"] == "a" * 64
    assert unchanged["evidence"].get("short_interest_sha256") == before["evidence"].get(
        "short_interest_sha256"
    )
    path = _config_copy(tmp_path, tape_sha256="a" * 64)
    with pytest.raises(ValueError, match="changed tape_sha256"):
        write_locked_short_interest_sha256(path, "b" * 64)
    source = json.loads((_root() / "configs" / "barebone-comparison-v1.json").read_text(encoding="utf-8"))
    source["comparable_performance_claim"] = True
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="comparable_performance_claim must be false"):
        write_locked_short_interest_sha256(path, "b" * 64)
    path = _config_copy(tmp_path)
    write_locked_short_interest_sha256(path, "c" * 64)
    locked = json.loads(path.read_text(encoding="utf-8"))
    assert locked["evidence"]["short_interest_sha256"] == "c" * 64
    assert locked["evidence"]["edgar_form4_sha256"] == LOCKED_FORM4_SHA256
    with pytest.raises(ValueError, match="changed tape_sha256"):
        short_interest_provenance(
            events_sha256="a" * 64,
            symbol_map_sha256="b" * 64,
            tape_sha256="c" * 64,
            event_count=1,
            revised_dropped=0,
            fetched_at=datetime(2026, 9, 30, tzinfo=timezone.utc),
        )

    root = _root()
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", SHORT_INTEREST_EVENTS], cwd=root, check=False
    )
    assert ignored.returncode == 0
    config = load_barebone_comparison_config()
    metrics = json.loads((root / "results" / "barebone_three_arm_metrics.json").read_text(encoding="utf-8"))
    assert config.evidence.edgar_form4_sha256 == LOCKED_FORM4_SHA256
    assert config.evidence.narrative_sha256 == LOCKED_NARRATIVE_SHA256
    assert config.evidence.tape_sha256 == LOCKED_TAPE_SHA256
    assert config.comparable_performance_claim is False
    assert metrics["arms"]["pure_ml"]["statistics"]["total_return"] == _MOMENTUM_RETURN
    assert metrics["arms"]["pure_ml"]["statistics"]["sharpe_ratio"] == _MOMENTUM_SHARPE
    assert hashlib.sha256((root / "results" / "metrics.json").read_bytes()).hexdigest() == _LEGACY_METRICS_SHA256
    assert hashlib.sha256((root / "results" / "barebone_three_arm_metrics.json").read_bytes()).hexdigest() == (
        _THREE_ARM_METRICS_SHA256
    )
    assert hashlib.sha256((root / "results" / "barebone_form4_arm_metrics.json").read_bytes()).hexdigest() == (
        _FORM4_METRICS_SHA256
    )
    assert hashlib.sha256((root / "results" / "barebone_form4_arm_ledger.json").read_bytes()).hexdigest() == (
        _FORM4_LEDGER_SHA256
    )
    spec = importlib.util.spec_from_file_location(
        "build_submission_archive", root / "scripts" / "build_submission_archive.py"
    )
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    events = root / SHORT_INTEREST_EVENTS
    created = not events.exists()
    if created:
        events.write_text("{}\n", encoding="utf-8")
    try:
        assert events not in builder.collect_files()
    finally:
        if created:
            events.unlink()
    digest = config.evidence.short_interest_sha256
    if digest is None:
        return
    provenance = json.loads((root / SHORT_INTEREST_PROVENANCE).read_text(encoding="utf-8"))
    assert provenance["short_interest_sha256"] == digest
    assert provenance["settlement_used_as_available_ts"] is False
    ledger = json.loads((root / LEDGER_RELATIVE).read_text(encoding="utf-8"))
    assert ledger["comparable_performance_claim"] is False
    assert ledger["signal_id"] == SI_SIGNAL_ID
    assert ledger["max_position_weight"] == 0.1
    for row in ledger["oos_skill"]:
        if row["oos_skill"] is None or float(row["oos_skill"]) <= 0.0:
            assert row["sized"] is False
    local_ready = events.is_file() and (root / "evidence" / "market" / "barebone_window_ohlcv.json").is_file()
    if not local_ready:
        return
    assert hashlib.sha256(events.read_bytes()).hexdigest() == digest
    assert (root / LEDGER_RELATIVE).read_text(encoding="utf-8") == json.dumps(
        barebone_short_interest_ledger(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    assert (root / METRICS_RELATIVE).read_text(encoding="utf-8") == json.dumps(
        barebone_short_interest_metrics(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    assert (root / ARM_RELATIVE).read_text(encoding="utf-8") == json.dumps(
        barebone_short_interest_arm_index(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def test_skill_gate_ignores_later_prices() -> None:
    config = _trading()
    raw = {"AAA": 0.2, "BBB": -0.1}
    assert short_interest_book(raw, None, config) == {}
    assert short_interest_book(raw, 0.0, config) == {}
    sized = short_interest_book(raw, 0.2, config)
    assert set(sized) == {"AAA"}
    assert sized["AAA"] <= 0.1
    dates = _weekdays(date(2025, 1, 2), 80)
    raw_by_session = {day: {"AAA": 0.2, "BBB": -0.1} for day in dates[62:]}
    closes = {("AAA", index): 100.0 + index for index, _day in enumerate(dates)}
    closes.update({("BBB", index): 50.0 for index, _day in enumerate(dates)})
    pairs = short_interest_oos_pairs(raw_by_session, closes, dates)
    skill, count = skill_before(pairs, 70)
    closes[("AAA", 79)] = 1.0
    again, again_count = skill_before(short_interest_oos_pairs(raw_by_session, closes, dates), 70)
    assert skill == again
    assert count == again_count
    with pytest.raises(ValueError, match="failed the gate"):
        _refuse_sized_without_skill([{"sized": True, "oos_skill": 0.0}])
    with pytest.raises(ValueError, match="locked signal"):
        raw_ratio_by_session(
            [
                {
                    "signal_id": "other",
                    "lookback_sessions": 63,
                    "session": "2025-03-05",
                    "ticker": "AAA",
                    "change_ratio": 0.1,
                }
            ]
        )
    rebalances = short_interest_rebalances(raw_by_session, closes, dates, config)
    assert rebalances[0]["sized"] is False


def _kernel_sessions() -> tuple[MarketSession, ...]:
    return tuple(
        MarketSession(
            session=day,
            bars=(
                AssetBar(ticker="AAA", open=100.0, close=101.0),
                AssetBar(ticker="BBB", open=100.0, close=99.0),
                AssetBar(ticker="SPY", open=100.0, close=100.0),
            ),
        )
        for day in (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6))
    )


def test_document_builders_cover_the_arm_without_the_local_tape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = (
        json.dumps(
            {
                "available_ts": "2025-11-25T00:00:00Z",
                "settlement_date": "2025-11-14",
                "current_short_position": 110,
                "previous_short_position": 100,
                "change_previous_number": 10,
            }
        )
        + "\n"
    )
    events = tmp_path / "events.jsonl"
    events.write_text(payload, encoding="utf-8")
    assert _load_events(events, hashlib.sha256(payload.encode("utf-8")).hexdigest())[0]["settlement_date"]
    banned = payload.replace("2025-11-25T00:00:00Z", "2025-11-14T00:00:00Z")
    events.write_text(banned, encoding="utf-8")
    with pytest.raises(ValueError, match="settlement date"):
        _load_events(events, hashlib.sha256(banned.encode("utf-8")).hexdigest())

    trading = ExperimentConfig.model_validate(
        {
            **_trading().model_dump(mode="json"),
            "start_date": "2025-01-02",
            "end_date": "2025-01-06",
            "rebalance_frequency_sessions": 1,
        }
    )
    sessions = _kernel_sessions()
    marks = benchmark_marks_from_closes(
        sessions, benchmark_ticker="SPY", starting_capital=trading.starting_capital
    )
    run = run_controlled_arm(
        config=trading,
        sessions=sessions,
        strategy_id="short_interest",
        candidates=[
            ArmInput(
                strategy_id="short_interest",
                decision_session=date(2025, 1, 2),
                ticker="AAA",
                structured_weight=0.1,
            )
        ],
        benchmark_marks=marks,
    )
    row = {
        "decision_session": "2025-01-02",
        "oos_skill": 0.2,
        "oos_skill_pairs": 10,
        "oos_skill_threshold": 0.0,
        "skill_pass": True,
        "insufficient_history": False,
        "sized": True,
        "name_count": 2,
        "book": {"AAA": 0.1},
        "scores": {"AAA": 0.15},
    }
    cash = {**row, "sized": False, "book": {}, "scores": {}, "oos_skill": None, "skill_pass": False}
    inputs = _arm_inputs([row, cash], trading)
    assert inputs[0].structured_weight == 0.1
    assert inputs[1].structured_weight == 0.0
    state = {
        "market": SimpleNamespace(tape_sha256=LOCKED_TAPE_SHA256),
        "trading": trading,
        "dates": tuple(session.session for session in sessions),
        "rebalances": [row],
        "run": run,
        "short_interest_sha256": "e" * 64,
        "event_count": 4,
    }
    ledger = barebone_short_interest_ledger(bound=state)
    metrics = barebone_short_interest_metrics(bound=state)
    index = barebone_short_interest_arm_index(bound=state)
    assert ledger["comparable_performance_claim"] is False
    assert ledger["signal_id"] == SI_SIGNAL_ID
    assert ledger["model_binding"]["settlement_used_as_available_ts"] is False
    assert metrics["statistics"]["total_return"] == run.metrics.total_return
    assert index["event_count"] == 4
    written = write_barebone_short_interest_artifacts(tmp_path / "repo", bound=state)
    assert all(path.is_file() for path in written)
    refused = {
        **state,
        "run": SimpleNamespace(
            ledger=SimpleNamespace(
                trade_count=0,
                total_turnover=0.0,
                transaction_costs=0.0,
                slippage_costs=0.0,
                post_cost_within_limit=True,
            ),
            lineage=SimpleNamespace(
                proposals=[
                    SimpleNamespace(
                        decision_session=date(2025, 1, 2),
                        ticker="AAA",
                        admitted=False,
                        target_weight=0.1,
                        reason="short interest change weight",
                    )
                ],
                config_hash="a",
                tape_hash="b",
                lineage_hash="c",
                experiment_hash="d",
            ),
            admitted_count=0,
        ),
        "rebalances": [{**row, "sized": False}],
    }
    with pytest.raises(ValueError, match="skill gate was cash"):
        barebone_short_interest_ledger(bound=refused)
    with pytest.raises(ValueError, match="missing a ledger"):
        barebone_short_interest_ledger(bound={**state, "run": SimpleNamespace(ledger=None)})
    with pytest.raises(ValueError, match="missing an executed result"):
        barebone_short_interest_metrics(
            bound={**state, "run": SimpleNamespace(result=None, metrics=None, ledger=None)}
        )

    monkeypatch.setattr(
        si_run_module,
        "write_barebone_short_interest_artifacts",
        lambda root=None, bound=None: write_barebone_short_interest_artifacts(
            tmp_path / "printed", bound=state
        ),
    )
    printed = io.StringIO()
    with redirect_stdout(printed), redirect_stderr(printed):
        assert run_main([]) == 0
    assert "short_interest_sha256=" in printed.getvalue()
    monkeypatch.setattr(
        si_run_module,
        "write_barebone_short_interest_artifacts",
        lambda root=None, bound=None: (_ for _ in ()).throw(ValueError("short interest arm blocked")),
    )
    failed = io.StringIO()
    with redirect_stdout(failed), redirect_stderr(failed):
        assert run_main([]) == 1
    assert "short interest arm blocked" in failed.getvalue()
    monkeypatch.setattr(si_module, "fetch_short_interest_events", lambda **_kwargs: "f" * 64)
    ingested = io.StringIO()
    with redirect_stdout(ingested), redirect_stderr(ingested):
        assert ingest_main(["--lock-config", "--config", str(tmp_path / "config.json")]) == 0
    assert "short_interest_sha256=" in ingested.getvalue()
