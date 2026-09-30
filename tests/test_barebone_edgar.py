"""PIT SEC filing counts, the CIK map, and the EDGAR skill gate."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta, timezone
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest

import demo.barebone_edgar as edgar_module
import demo.barebone_edgar_run as edgar_run_module
from demo.barebone_comparison import (
    BAREBONE_UNIVERSE,
    EDGAR_CIK_MAP,
    EDGAR_EVENTS,
    EDGAR_PROVENANCE,
    LOCKED_NARRATIVE_SHA256,
    LOCKED_POLARITY_SHA256,
    LOCKED_TAPE_SHA256,
    load_barebone_comparison_config,
)
from demo.barebone_edgar import (
    EDGAR_EXACT_FORMS,
    EDGAR_LOOKBACK_SESSIONS,
    EDGAR_SIGNAL_ID,
    EDGAR_SUBMISSIONS,
    EDGAR_USER_AGENT,
    _default_get_json,
    _oldest_acceptance,
    demean_cross_section,
    edgar_provenance,
    edgar_score_rows,
    events_for_issuer,
    events_from_columns,
    fetch_edgar_events,
    load_cik_map,
    main as ingest_main,
    render_events_jsonl,
    write_locked_edgar_sha256,
)
from demo.barebone_edgar_run import (
    ARM_RELATIVE,
    LEDGER_RELATIVE,
    METRICS_RELATIVE,
    _arm_inputs,
    _load_events,
    _refuse_sized_without_skill,
    barebone_edgar_arm_index,
    barebone_edgar_ledger,
    barebone_edgar_metrics,
    edgar_book,
    edgar_oos_pairs,
    edgar_rebalances,
    main as run_main,
    raw_log1p_by_session,
    skill_before,
    write_barebone_edgar_artifacts,
)
from demo.contracts import AssetBar, ExperimentConfig, MarketSession
from demo.controlled import ArmInput, benchmark_marks_from_closes, run_controlled_arm

_LEGACY_METRICS_SHA256 = "e7e5055ce9b4409d7941a71929f66261416b8d3c3f62423190edafd5bf5b1411"
_THREE_ARM_METRICS_SHA256 = "af40d4176d3e4705f799b3b640494777caa24ed81ea68f149a84bb8335d554f3"
_THREE_ARM_LEDGER_SHA256 = "07296e85b8930fa1eb88c8fcf0bc1f353d7a87fac249603ddddbe1ce2a5eae04"
_NARRATIVE_METRICS_SHA256 = "6d973eebffcacd70e4b786ccd3ee66b63feeb622db09bc67ac653d3fb44b4afd"
_NARRATIVE_LEDGER_SHA256 = "95e342ad9cf35f3332faf82b99f37d12748e5131ca11b8ede1ccab145d04c2f3"
_ATTENTION_METRICS_SHA256 = "83eda47b45f5915817a6286e6515bcbc32119885709883bd0d86d47ff8cdc6a3"
_ATTENTION_LEDGER_SHA256 = "b6ac2081fdd237fb1100feff532b54c2c0f3c200e71f09adc656982d993cad5c"
_HYBRID_METRICS_SHA256 = "9e7b0ab941f73b4c2faf5de5299805f897f62c263d9acfc1386dd8dcf9545fad"
_HYBRID_LEDGER_SHA256 = "7166e38f870388117c8e6aceeb86cca80b133f7d9f6b1a899cc5e2a68fc1b57f"
_MOMENTUM_RETURN = 0.24597743358179014
_MOMENTUM_SHARPE = 1.6030790628765745
_XOM_WINDOW_CIK = "0000034088"


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sessions(count: int = 80) -> tuple[date, ...]:
    return tuple(date(2025, 1, 2) + timedelta(days=offset) for offset in range(count))


def _columns(
    forms: list[object],
    accessions: list[object],
    acceptances: list[object],
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "form": forms,
        "accessionNumber": accessions,
        "acceptanceDateTime": acceptances,
    }
    payload.update(extra)
    return payload


def _event(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_id": "edgar:0000320193-25-000001:AAPL",
        "ticker": "AAPL",
        "cik": "0000320193",
        "accession": "0000320193-25-000001",
        "form": "8-K",
        "acceptanceDateTime": "2025-01-02T15:00:00Z",
        "available_ts": "2025-01-02T15:00:00Z",
        "decision_session": "2025-01-03",
    }
    payload.update(updates)
    return payload


def _row_on(rows: list[dict[str, object]], session: str, ticker: str) -> dict[str, object]:
    found = [row for row in rows if row["session"] == session and row["ticker"] == ticker]
    assert len(found) == 1
    return found[0]


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


def _map_payload() -> dict[str, object]:
    return json.loads((_root() / EDGAR_CIK_MAP).read_text(encoding="utf-8"))


def _write_map(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_amendments_are_excluded_and_a_missing_acceptance_aborts() -> None:
    sessions = _sessions(5)
    rows = events_from_columns(
        _columns(
            ["8-K", "8-K/A", "10-Q", "10-K/A", "10-K", "4", "8-K"],
            [
                "0000320193-25-000001",
                "0000320193-25-000002",
                "0000320193-25-000003",
                "0000320193-25-000004",
                "0000320193-25-000005",
                "0000320193-25-000006",
                "0000320193-25-000007",
            ],
            [
                "2025-01-02T15:00:00.000Z",
                None,
                "2025-01-02T18:00:00.000Z",
                "",
                "2024-12-31T21:00:00.000Z",
                "2024-11-01T12:00:00.000Z",
                "2025-01-06T15:00:00.000Z",
            ],
            reportDate=["2024-09-28"] * 7,
        ),
        ticker="AAPL",
        cik="0000320193",
        sessions=sessions,
    )
    assert [row["form"] for row in rows] == ["8-K", "10-Q", "10-K"]
    assert all(row["form"] in EDGAR_EXACT_FORMS for row in rows)
    assert all("/A" not in str(row["form"]) for row in rows)
    assert all("reportDate" not in row for row in rows)
    assert rows[0]["available_ts"] == "2025-01-02T15:00:00Z"
    assert rows[0]["acceptanceDateTime"] == rows[0]["available_ts"]
    assert rows[0]["decision_session"] == "2025-01-03"
    assert rows[0]["decision_session"] > str(rows[0]["available_ts"])[:10]
    assert rows[2]["decision_session"] == "2025-01-02"
    assert "html" not in rows[0] and "text" not in rows[0]
    assert str(rows[0]["uri"]).startswith("https://www.sec.gov/Archives/edgar/data/")
    assert str(rows[0]["uri"]).endswith("-index.html")
    with pytest.raises(ValueError, match="missing acceptanceDateTime"):
        events_from_columns(
            _columns(["8-K"], ["0000320193-25-000008"], [None]),
            ticker="AAPL",
            cik="0000320193",
            sessions=sessions,
        )
    with pytest.raises(ValueError, match="missing acceptanceDateTime"):
        events_from_columns(
            _columns(["10-Q"], ["0000320193-25-000009"], ["  "]),
            ticker="AAPL",
            cik="0000320193",
            sessions=sessions,
        )
    with pytest.raises(ValueError, match="not a timestamp"):
        events_from_columns(
            _columns(["10-K"], ["0000320193-25-000010"], [20250102]),
            ticker="AAPL",
            cik="0000320193",
            sessions=sessions,
        )
    with pytest.raises(ValueError, match="missing acceptanceDateTime"):
        events_from_columns(
            {"form": ["8-K"], "accessionNumber": ["0000320193-25-000011"]},
            ticker="AAPL",
            cik="0000320193",
            sessions=sessions,
        )
    with pytest.raises(ValueError, match="missing form"):
        events_from_columns(
            {"acceptanceDateTime": ["2025-01-02T15:00:00Z"]},
            ticker="AAPL",
            cik="0000320193",
            sessions=sessions,
        )
    with pytest.raises(ValueError, match="different lengths"):
        events_from_columns(
            _columns(["8-K", "10-Q"], ["0000320193-25-000012"], ["2025-01-02T15:00:00Z"]),
            ticker="AAPL",
            cik="0000320193",
            sessions=sessions,
        )
    with pytest.raises(ValueError, match="must be lists"):
        events_from_columns(
            _columns("8-K", "0000320193-25-000013", "2025-01-02T15:00:00Z"),
            ticker="AAPL",
            cik="0000320193",
            sessions=sessions,
        )
    with pytest.raises(ValueError, match="not usable"):
        events_from_columns(
            _columns(["8-K"], ["not-an-accession"], ["2025-01-02T15:00:00Z"]),
            ticker="AAPL",
            cik="0000320193",
            sessions=sessions,
        )
    assert (
        events_from_columns(
            _columns(["8-K"], ["0000320193-25-000014"], ["2024-12-29T15:00:00Z"]),
            ticker="AAPL",
            cik="0000320193",
            sessions=sessions,
        )
        == []
    )


def test_count_window_is_point_in_time_and_ignores_report_date() -> None:
    sessions = _sessions()
    early = _event()
    scored = sessions[EDGAR_LOOKBACK_SESSIONS - 1].isoformat()
    future_session = sessions[EDGAR_LOOKBACK_SESSIONS].isoformat()
    future = _event(
        event_id="edgar:0000320193-25-000099:AAPL",
        accession="0000320193-25-000099",
        acceptanceDateTime=f"{scored}T15:00:00Z",
        available_ts=f"{scored}T15:00:00Z",
        decision_session=future_session,
    )
    base = edgar_score_rows([early], sessions, ("AAPL", "MSFT"))
    with_future = edgar_score_rows([early, future], sessions, ("AAPL", "MSFT"))
    assert _row_on(base, scored, "AAPL") == _row_on(with_future, scored, "AAPL")
    first = _row_on(with_future, scored, "AAPL")
    assert first["session_index"] == EDGAR_LOOKBACK_SESSIONS - 1
    assert first["event_count"] == 1
    assert _row_on(with_future, scored, "MSFT")["event_count"] == 0
    assert first["signal_id"] == EDGAR_SIGNAL_ID
    assert first["lookback_sessions"] == 63
    later = _row_on(with_future, future_session, "AAPL")
    assert later["event_count"] == 2
    assert "reportDate" not in first and "text" not in first
    raw = {"AAPL": first["log1p_count"], "MSFT": 0.0}
    demeaned = demean_cross_section(raw)
    assert first["score"] == demeaned["AAPL"]
    assert demean_cross_section({"AAPL": 1.0}) == {}
    odd = demean_cross_section({"AAPL": 3.0, "MSFT": 1.0, "NVDA": 0.0})
    assert odd["MSFT"] == 0.0
    with pytest.raises(ValueError, match="locked at 63"):
        edgar_score_rows([early], sessions, ("AAPL", "MSFT"), lookback=21)
    with pytest.raises(ValueError, match="same-session"):
        edgar_score_rows(
            [
                _event(
                    acceptanceDateTime="2025-01-03T15:00:00Z",
                    available_ts="2025-01-03T15:00:00Z",
                    decision_session="2025-01-03",
                )
            ],
            sessions,
            ("AAPL", "MSFT"),
        )
    with pytest.raises(ValueError, match="strictly after"):
        edgar_score_rows([_event(decision_session="2025-01-06")], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="equal acceptanceDateTime"):
        edgar_score_rows(
            [_event(acceptanceDateTime="2025-01-02T16:00:00Z")],
            sessions,
            ("AAPL", "MSFT"),
        )
    with pytest.raises(ValueError, match="forbidden field"):
        edgar_score_rows([_event(reportDate="2024-09-28")], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="not an exact"):
        edgar_score_rows([_event(form="8-K/A")], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="missing available_ts"):
        edgar_score_rows([_event(available_ts="")], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="edgar universe"):
        edgar_score_rows([_event(ticker="ZZZ")], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="missing event_id"):
        edgar_score_rows([_event(event_id=" ")], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="repeated"):
        edgar_score_rows([early, early], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="strictly increasing"):
        edgar_score_rows([], (date(2025, 1, 3), date(2025, 1, 2)), ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="at least two"):
        edgar_score_rows([], sessions, ("AAPL",))
    with pytest.raises(ValueError, match="must be an object"):
        edgar_score_rows(["not-an-event"], sessions, ("AAPL", "MSFT"))


def test_cik_map_refuses_unmapped_names_and_keeps_the_xom_window_registrant(tmp_path: Path) -> None:
    mapping, digest = load_cik_map()
    assert set(mapping) == set(BAREBONE_UNIVERSE)
    assert len(set(mapping.values())) == len(BAREBONE_UNIVERSE)
    assert mapping["XOM"] == _XOM_WINDOW_CIK
    assert mapping["AAPL"] == "0000320193"
    assert mapping["BRK-B"] == "0001067983"
    assert digest == hashlib.sha256((_root() / EDGAR_CIK_MAP).read_bytes()).hexdigest()
    payload = _map_payload()
    entries = list(payload["entries"])
    dropped = [entry for entry in entries if entry["ticker"] != "NKE"]
    path = tmp_path / "map.json"
    payload["entries"] = dropped
    _write_map(path, payload)
    with pytest.raises(ValueError, match="missing NKE"):
        load_cik_map(path)
    payload["entries"] = entries + [dict(entries[0])]
    _write_map(path, payload)
    with pytest.raises(ValueError, match="repeats AAPL"):
        load_cik_map(path)
    outside = [dict(entry) for entry in entries]
    outside[0]["ticker"] = "ZZZ"
    payload["entries"] = outside
    _write_map(path, payload)
    with pytest.raises(ValueError, match="outside the universe"):
        load_cik_map(path)
    shared = [dict(entry) for entry in entries]
    shared[1]["cik"] = shared[0]["cik"]
    payload["entries"] = shared
    _write_map(path, payload)
    with pytest.raises(ValueError, match="shared by"):
        load_cik_map(path)
    bad_cik = [dict(entry) for entry in entries]
    bad_cik[0]["cik"] = "320193"
    payload["entries"] = bad_cik
    _write_map(path, payload)
    with pytest.raises(ValueError, match="10 digits"):
        load_cik_map(path)
    payload["entries"] = ["not-an-object"]
    _write_map(path, payload)
    with pytest.raises(ValueError, match="must be an object"):
        load_cik_map(path)
    payload["entries"] = []
    _write_map(path, payload)
    with pytest.raises(ValueError, match="no entries"):
        load_cik_map(path)
    payload["schema"] = "other"
    payload["entries"] = entries
    _write_map(path, payload)
    with pytest.raises(ValueError, match="schema"):
        load_cik_map(path)
    payload = _map_payload()
    payload["experiment_id"] = "other"
    _write_map(path, payload)
    with pytest.raises(ValueError, match="experiment_id"):
        load_cik_map(path)
    payload = _map_payload()
    payload["comparable_performance_claim"] = True
    _write_map(path, payload)
    with pytest.raises(ValueError, match="comparable_performance_claim must be false"):
        load_cik_map(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_cik_map(path)


def test_overflow_is_fetched_only_when_recent_misses_the_floor() -> None:
    sessions = (date(2025, 6, 2), date(2025, 6, 3))
    recent = _columns(
        ["8-K"],
        ["0000320193-25-000001"],
        ["2025-06-02T16:00:00.000Z"],
        reportDate=["2025-06-01"],
    )
    payload = {"filings": {"recent": recent, "files": [{"name": "CIK0000320193-submissions-001.json"}]}}
    with pytest.raises(ValueError, match="need overflow"):
        events_for_issuer(payload, ticker="AAPL", cik="0000320193", sessions=sessions, get_json=None)

    def extra(url: str) -> dict[str, object]:
        assert url == EDGAR_SUBMISSIONS + "CIK0000320193-submissions-001.json"
        return _columns(
            ["10-Q", "10-K/A", "8-K"],
            ["0000320193-24-000010", "0000320193-24-000011", "0000320193-24-000012"],
            ["2024-12-31T20:00:00.000Z", "2024-12-31T20:00:00.000Z", "2025-06-02T12:00:00.000Z"],
            reportDate=["2024-09-30", "2024-09-30", "2025-06-02"],
        )

    rows = events_for_issuer(payload, ticker="AAPL", cik="0000320193", sessions=sessions, get_json=extra)
    assert [row["form"] for row in rows] == ["8-K", "10-Q", "8-K"]
    assert rows[1]["available_ts"] == "2024-12-31T20:00:00Z"
    assert rows[1]["decision_session"] == "2025-06-02"
    assert all("reportDate" not in row for row in rows)
    covered = _columns(
        ["8-K", "4"],
        ["0000320193-25-000001", "0000320193-24-000099"],
        ["2025-06-02T16:00:00.000Z", "2024-12-01T12:00:00.000Z"],
    )
    ignored = {"filings": {"recent": covered, "files": "not-a-list"}}
    kept = events_for_issuer(ignored, ticker="AAPL", cik="0000320193", sessions=sessions)
    assert [row["form"] for row in kept] == ["8-K"]
    with pytest.raises(ValueError, match="no overflow files"):
        events_for_issuer(
            {"filings": {"recent": recent, "files": []}},
            ticker="AAPL",
            cik="0000320193",
            sessions=sessions,
        )
    with pytest.raises(ValueError, match="refused"):
        events_for_issuer(
            {"filings": {"recent": recent, "files": [{"name": "../evil.json"}]}},
            ticker="AAPL",
            cik="0000320193",
            sessions=sessions,
            get_json=extra,
        )
    with pytest.raises(ValueError, match="not an object"):
        events_for_issuer(
            payload,
            ticker="AAPL",
            cik="0000320193",
            sessions=sessions,
            get_json=lambda _url: ["not-columns"],
        )
    with pytest.raises(ValueError, match="missing filings"):
        events_for_issuer({}, ticker="AAPL", cik="0000320193", sessions=sessions)
    with pytest.raises(ValueError, match="missing recent"):
        events_for_issuer({"filings": {}}, ticker="AAPL", cik="0000320193", sessions=sessions)
    with pytest.raises(ValueError, match="no acceptanceDateTime"):
        _oldest_acceptance({"acceptanceDateTime": [None, " "]})
    with pytest.raises(ValueError, match="not a timestamp"):
        _oldest_acceptance({"acceptanceDateTime": [20250102]})
    assert render_events_jsonl([]) == b""
    rendered = render_events_jsonl([{"b": 1, "a": 2}])
    assert rendered.startswith(b'{"a": 2, "b": 1}\n')


def test_submissions_reader_stays_on_data_sec_gov(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="non-data.sec.gov"):
        _default_get_json("https://www.sec.gov/Archives/edgar/data/320193/index.html")
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def _rate_limit(request: urllib.request.Request, timeout: int = 60) -> object:
        assert request.get_header("User-agent") == EDGAR_USER_AGENT
        assert timeout == 60
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", _rate_limit)
    with pytest.raises(ValueError, match="rate limit"):
        _default_get_json(EDGAR_SUBMISSIONS + "CIK0000320193.json")

    def _missing(request: urllib.request.Request, timeout: int = 60) -> object:
        raise urllib.error.HTTPError(request.full_url, 404, "missing", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", _missing)
    with pytest.raises(ValueError, match="failed: 404"):
        _default_get_json(EDGAR_SUBMISSIONS + "CIK0000320193.json")

    class _Body:
        def __enter__(self) -> "_Body":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def read(self) -> bytes:
            return b"[1]"

    monkeypatch.setattr(urllib.request, "urlopen", lambda _request, timeout=60: _Body())
    with pytest.raises(ValueError, match="not an object"):
        _default_get_json(EDGAR_SUBMISSIONS + "CIK0000320193.json")

    class _Object(_Body):
        def read(self) -> bytes:
            return b'{"cik": "320193"}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda _request, timeout=60: _Object())
    assert _default_get_json(EDGAR_SUBMISSIONS + "CIK0000320193.json")["cik"] == "320193"


def _config_copy(tmp_path: Path, **evidence_updates: object) -> Path:
    source = json.loads((_root() / "configs" / "barebone-comparison-v1.json").read_text(encoding="utf-8"))
    source["evidence"].update(evidence_updates)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


def _install_map(root: Path) -> None:
    destination = root / EDGAR_CIK_MAP
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_root() / EDGAR_CIK_MAP, destination)


def _fake_submissions(mapping: dict[str, str]) -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {}
    for offset, (ticker, cik) in enumerate(mapping.items()):
        accession = f"{cik}-25-{offset + 1:06d}"
        payloads[EDGAR_SUBMISSIONS + f"CIK{cik}.json"] = {
            "cik": str(int(cik)),
            "filings": {
                "recent": _columns(
                    ["8-K", "8-K/A", "10-Q", "4"],
                    [accession, f"{cik}-25-{offset + 1:06d}"[:-1] + "9", f"{cik}-25-100000", f"{cik}-24-000001"],
                    [
                        "2025-06-02T16:30:00.000Z",
                        None,
                        "2025-06-02T18:00:00.000Z",
                        "2024-12-01T12:00:00.000Z",
                    ],
                    reportDate=["2025-03-31", "2025-03-31", "2025-03-31", "2024-12-01"],
                    filingDate=["2025-06-02"] * 4,
                ),
                "files": [],
            },
        }
        payloads[ticker] = {"accession": accession}
    return payloads


def test_fetch_locks_metadata_without_moving_the_narrative_hash(tmp_path: Path) -> None:
    mapping, _cik_sha = load_cik_map()
    payloads = _fake_submissions(mapping)
    sessions = (date(2025, 6, 2), date(2025, 6, 3))
    root = tmp_path / "repo"
    _install_map(root)
    config_path = _config_copy(tmp_path, narrative_sha256="a" * 64)
    with pytest.raises(ValueError, match="narrative_sha256 is locked"):
        fetch_edgar_events(config_path=config_path, root=root, sessions=sessions, get_json=lambda _url: {})
    config_path = _config_copy(tmp_path, tape_sha256="b" * 64)
    with pytest.raises(ValueError, match="tape_sha256 is locked"):
        fetch_edgar_events(config_path=config_path, root=root, sessions=sessions, get_json=lambda _url: {})
    config_path = _config_copy(tmp_path)
    with pytest.raises(ValueError, match="does not match the frozen map"):
        fetch_edgar_events(
            config_path=config_path,
            root=root,
            sessions=sessions,
            get_json=lambda _url: {"cik": "0000000001", "filings": {}},
        )
    aapl = EDGAR_SUBMISSIONS + "CIK0000320193.json"
    duplicate = json.loads(json.dumps(payloads[aapl]))
    recent = duplicate["filings"]["recent"]
    recent["form"] = ["8-K", "8-K", "4"]
    recent["accessionNumber"] = ["0000320193-25-000001", "0000320193-25-000001", "0000320193-24-000001"]
    recent["acceptanceDateTime"] = [
        "2025-06-02T16:30:00.000Z",
        "2025-06-02T16:30:00.000Z",
        "2024-12-01T12:00:00.000Z",
    ]

    def _duplicate(url: str) -> dict[str, object]:
        if url == aapl:
            return duplicate
        return payloads[url]

    with pytest.raises(ValueError, match="repeated"):
        fetch_edgar_events(config_path=config_path, root=root, sessions=sessions, get_json=_duplicate)

    def _form_four(url: str) -> dict[str, object]:
        payload = json.loads(json.dumps(payloads[url]))
        payload["filings"]["recent"]["form"] = ["4", "4", "4", "4"]
        return payload

    with pytest.raises(ValueError, match="empty EDGAR tape"):
        fetch_edgar_events(config_path=config_path, root=root, sessions=sessions, get_json=_form_four)
    assert not (root / EDGAR_EVENTS).exists()

    seen: list[str] = []

    def _reader(url: str) -> dict[str, object]:
        seen.append(url)
        if not url.startswith(EDGAR_SUBMISSIONS):
            raise AssertionError(url)
        return payloads[url]

    digest = fetch_edgar_events(
        config_path=config_path,
        root=root,
        lock_config=True,
        sessions=sessions,
        get_json=_reader,
        now=datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
    )
    assert len(seen) == len(BAREBONE_UNIVERSE)
    events = (root / EDGAR_EVENTS).read_bytes()
    assert hashlib.sha256(events).hexdigest() == digest
    assert b"reportDate" not in events
    assert b"filingDate" not in events
    assert b"8-K/A" not in events
    assert b"<html" not in events.lower()
    lines = [json.loads(line) for line in events.decode("utf-8").splitlines()]
    assert len(lines) == len(BAREBONE_UNIVERSE) * 2
    assert {row["form"] for row in lines} == {"8-K", "10-Q"}
    assert all(row["available_ts"] == row["acceptanceDateTime"] for row in lines)
    assert all(row["decision_session"] == "2025-06-03" for row in lines)
    sidecar = json.loads((root / EDGAR_PROVENANCE).read_text(encoding="utf-8"))
    assert sidecar["edgar_sha256"] == digest
    assert sidecar["narrative_sha256"] == LOCKED_NARRATIVE_SHA256
    assert sidecar["tape_sha256"] == LOCKED_TAPE_SHA256
    assert sidecar["amendments"] == "excluded; form must be exactly 8-K, 10-Q, or 10-K"
    assert sidecar["report_date_used_as_available_ts"] is False
    assert sidecar["signal_id"] == EDGAR_SIGNAL_ID
    assert sidecar["lookback_sessions"] == 63
    assert sidecar["user_agent"] == EDGAR_USER_AGENT
    assert sidecar["comparable_performance_claim"] is False
    locked = json.loads(config_path.read_text(encoding="utf-8"))
    assert locked["evidence"]["edgar_sha256"] == digest
    assert locked["evidence"]["narrative_sha256"] == LOCKED_NARRATIVE_SHA256
    assert locked["evidence"]["tape_sha256"] == LOCKED_TAPE_SHA256
    assert locked["evidence"]["polarity_sha256"] == LOCKED_POLARITY_SHA256
    assert locked["comparable_performance_claim"] is False
    with pytest.raises(ValueError, match="changed tape_sha256"):
        edgar_provenance(
            events_sha256=digest,
            cik_map_sha256="c" * 64,
            tape_sha256="d" * 64,
            event_count=1,
            form_counts={},
            fetched_at=datetime(2026, 9, 30, tzinfo=timezone.utc),
        )


def test_non_positive_skill_is_cash_and_later_prices_do_not_change_past_skill() -> None:
    config = _trading()
    raw = {"AAA": 1.0, "BBB": 0.0}
    assert edgar_book(raw, None, config) == {}
    assert edgar_book(raw, 0.0, config) == {}
    assert edgar_book(raw, -0.2, config) == {}
    sized = edgar_book(raw, 0.2, config)
    assert set(sized) == {"AAA"}
    assert sized["AAA"] <= 0.1
    dates = _sessions()
    raw_by_session = {day: {"AAA": 1.0, "BBB": 0.0} for day in dates[62:]}
    closes = {("AAA", index): 100.0 + index for index, _day in enumerate(dates)}
    closes.update({("BBB", index): 50.0 for index, _day in enumerate(dates)})
    pairs = edgar_oos_pairs(raw_by_session, closes, dates)
    skill, count = skill_before(pairs, 75)
    closes[("AAA", 79)] = 1.0
    again, again_count = skill_before(edgar_oos_pairs(raw_by_session, closes, dates), 75)
    assert skill == again
    assert count == again_count
    assert count >= 8
    closes[("AAA", 74)] = 1.0
    changed, _changed_count = skill_before(edgar_oos_pairs(raw_by_session, closes, dates), 75)
    assert changed != skill
    with pytest.raises(ValueError, match="not on the tape"):
        edgar_oos_pairs({date(2024, 1, 1): raw}, closes, dates)
    assert skill_before([], 75) == (None, 0)
    rebalances = edgar_rebalances(raw_by_session, closes, dates, config)
    assert rebalances[0]["insufficient_history"] is True
    assert rebalances[0]["sized"] is False
    with pytest.raises(ValueError, match="failed the gate"):
        _refuse_sized_without_skill([{"sized": True, "oos_skill": 0.0}])
    with pytest.raises(ValueError, match="locked signal"):
        raw_log1p_by_session(
            [{"signal_id": "other", "lookback_sessions": 63, "session": "2025-03-05", "ticker": "AAA", "log1p_count": 0.0}]
        )
    with pytest.raises(ValueError, match="return label"):
        raw_log1p_by_session(
            [
                {
                    "signal_id": EDGAR_SIGNAL_ID,
                    "lookback_sessions": 63,
                    "session": "2025-03-05",
                    "ticker": "AAA",
                    "log1p_count": 0.0,
                    "total_return": 0.1,
                }
            ]
        )


def test_lock_refuses_changed_binds_and_the_momentum_control_stays_put(tmp_path: Path) -> None:
    path = _config_copy(tmp_path, narrative_sha256="a" * 64)
    with pytest.raises(ValueError, match="changed narrative_sha256"):
        write_locked_edgar_sha256(path, "b" * 64)
    path = _config_copy(tmp_path, tape_sha256="a" * 64)
    with pytest.raises(ValueError, match="changed tape_sha256"):
        write_locked_edgar_sha256(path, "b" * 64)
    path = _config_copy(tmp_path, polarity_sha256="d" * 64)
    with pytest.raises(ValueError, match="changed polarity_sha256"):
        write_locked_edgar_sha256(path, "b" * 64)
    source = json.loads((_root() / "configs" / "barebone-comparison-v1.json").read_text(encoding="utf-8"))
    source["comparable_performance_claim"] = True
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="comparable_performance_claim must be false"):
        write_locked_edgar_sha256(path, "b" * 64)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        write_locked_edgar_sha256(path, "b" * 64)
    path.write_text(json.dumps({"comparable_performance_claim": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence path is required"):
        write_locked_edgar_sha256(path, "b" * 64)
    path = _config_copy(tmp_path)
    write_locked_edgar_sha256(path, "c" * 64)
    locked = json.loads(path.read_text(encoding="utf-8"))
    assert locked["evidence"]["edgar_sha256"] == "c" * 64
    assert locked["evidence"]["edgar_events"] == EDGAR_EVENTS
    assert locked["evidence"]["narrative_sha256"] == LOCKED_NARRATIVE_SHA256
    assert locked["evidence"]["tape_sha256"] == LOCKED_TAPE_SHA256
    assert locked["evidence"]["polarity_sha256"] == LOCKED_POLARITY_SHA256

    root = _root()
    ignored = subprocess.run(["git", "check-ignore", "-q", EDGAR_EVENTS], cwd=root, check=False)
    assert ignored.returncode == 0
    config = load_barebone_comparison_config()
    metrics = json.loads((root / "results" / "barebone_three_arm_metrics.json").read_text(encoding="utf-8"))
    assert config.evidence.narrative_sha256 == LOCKED_NARRATIVE_SHA256
    assert config.evidence.tape_sha256 == LOCKED_TAPE_SHA256
    assert config.evidence.polarity_sha256 == LOCKED_POLARITY_SHA256
    assert config.comparable_performance_claim is False
    assert metrics["arms"]["pure_ml"]["statistics"]["total_return"] == _MOMENTUM_RETURN
    assert metrics["arms"]["pure_ml"]["statistics"]["sharpe_ratio"] == _MOMENTUM_SHARPE
    expected = {
        "results/metrics.json": _LEGACY_METRICS_SHA256,
        "results/barebone_three_arm_metrics.json": _THREE_ARM_METRICS_SHA256,
        "results/barebone_three_arm_ledger.json": _THREE_ARM_LEDGER_SHA256,
        "results/barebone_narrative_arm_metrics.json": _NARRATIVE_METRICS_SHA256,
        "results/barebone_narrative_arm_ledger.json": _NARRATIVE_LEDGER_SHA256,
        "results/barebone_attention_arm_metrics.json": _ATTENTION_METRICS_SHA256,
        "results/barebone_attention_arm_ledger.json": _ATTENTION_LEDGER_SHA256,
        "results/barebone_hybrid_arm_metrics.json": _HYBRID_METRICS_SHA256,
        "results/barebone_hybrid_arm_ledger.json": _HYBRID_LEDGER_SHA256,
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location(
        "build_submission_archive", root / "scripts" / "build_submission_archive.py"
    )
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    events = root / EDGAR_EVENTS
    created = not events.exists()
    if created:
        events.parent.mkdir(parents=True, exist_ok=True)
        events.write_text("{}\n", encoding="utf-8")
    try:
        assert events not in builder.collect_files()
    finally:
        if created:
            events.unlink()
    edgar_sha = config.evidence.edgar_sha256
    if edgar_sha is None:
        return
    provenance = json.loads((root / EDGAR_PROVENANCE).read_text(encoding="utf-8"))
    assert provenance["edgar_sha256"] == edgar_sha
    assert provenance["narrative_sha256"] == LOCKED_NARRATIVE_SHA256
    assert provenance["tape_sha256"] == LOCKED_TAPE_SHA256
    assert provenance["report_date_used_as_available_ts"] is False
    assert provenance["amendments"] == "excluded; form must be exactly 8-K, 10-Q, or 10-K"
    assert provenance["comparable_performance_claim"] is False
    local_ready = events.is_file() and (root / "evidence" / "market" / "barebone_window_ohlcv.json").is_file()
    ledger = json.loads((root / LEDGER_RELATIVE).read_text(encoding="utf-8"))
    assert ledger["comparable_performance_claim"] is False
    assert ledger["signal_id"] == EDGAR_SIGNAL_ID
    assert ledger["max_position_weight"] == 0.1
    assert ledger["max_gross_leverage"] == 1.0
    for row in ledger["oos_skill"]:
        if row["oos_skill"] is None or float(row["oos_skill"]) <= 0.0:
            assert row["sized"] is False
    if not local_ready:
        return
    assert hashlib.sha256(events.read_bytes()).hexdigest() == edgar_sha
    assert (root / LEDGER_RELATIVE).read_text(encoding="utf-8") == json.dumps(
        barebone_edgar_ledger(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    assert (root / METRICS_RELATIVE).read_text(encoding="utf-8") == json.dumps(
        barebone_edgar_metrics(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    assert (root / ARM_RELATIVE).read_text(encoding="utf-8") == json.dumps(
        barebone_edgar_arm_index(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


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


def _kernel_trading() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            **_trading().model_dump(mode="json"),
            "start_date": "2025-01-02",
            "end_date": "2025-01-06",
            "rebalance_frequency_sessions": 1,
        }
    )


def test_document_builders_cover_the_arm_without_the_local_tape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"form": "8-K", "event_id": "edgar:1:AAPL"}).encode("utf-8")
    events = tmp_path / "events.jsonl"
    events.write_bytes(payload)
    assert _load_events(events, hashlib.sha256(payload).hexdigest())[0]["form"] == "8-K"
    with pytest.raises(ValueError, match="do not match"):
        _load_events(events, "d" * 64)
    events.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        _load_events(events, hashlib.sha256(b"\n").hexdigest())
    banned = json.dumps({"form": "8-K", "reportDate": "2024-09-28"}) + "\n"
    events.write_text(banned, encoding="utf-8")
    with pytest.raises(ValueError, match="reportDate"):
        _load_events(events, hashlib.sha256(banned.encode("utf-8")).hexdigest())
    amended = json.dumps({"form": "8-K/A"}) + "\n"
    events.write_text(amended, encoding="utf-8")
    with pytest.raises(ValueError, match="not an exact"):
        _load_events(events, hashlib.sha256(amended.encode("utf-8")).hexdigest())

    trading = _kernel_trading()
    sessions = _kernel_sessions()
    marks = benchmark_marks_from_closes(sessions, benchmark_ticker="SPY", starting_capital=trading.starting_capital)
    run = run_controlled_arm(
        config=trading,
        sessions=sessions,
        strategy_id="edgar",
        candidates=[
            ArmInput(
                strategy_id="edgar",
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
        "scores": {"AAA": 0.4},
    }
    cash = {
        **row,
        "decision_session": "2025-01-03",
        "oos_skill": None,
        "skill_pass": False,
        "sized": False,
        "book": {},
        "scores": {},
    }
    inputs = _arm_inputs([row, cash], trading)
    assert inputs[0].structured_weight == 0.1
    assert inputs[1].structured_weight == 0.0
    assert inputs[1].ticker == "AAA"
    state = {
        "market": SimpleNamespace(tape_sha256=LOCKED_TAPE_SHA256),
        "trading": trading,
        "dates": tuple(session.session for session in sessions),
        "rebalances": [row],
        "run": run,
        "edgar_sha256": "e" * 64,
        "event_count": 4,
    }
    ledger = barebone_edgar_ledger(bound=state)
    metrics = barebone_edgar_metrics(bound=state)
    index = barebone_edgar_arm_index(bound=state)
    assert ledger["comparable_performance_claim"] is False
    assert ledger["signal_id"] == EDGAR_SIGNAL_ID
    assert ledger["narrative_sha256"] == LOCKED_NARRATIVE_SHA256
    assert ledger["strategy_id"] == "edgar"
    assert metrics["statistics"]["total_return"] == run.metrics.total_return
    assert metrics["comparable_performance_claim"] is False
    assert index["event_count"] == 4
    assert index["sized_rebalances"] == 1
    written = write_barebone_edgar_artifacts(tmp_path / "repo", bound=state)
    assert all(path.is_file() for path in written)

    cash_run = SimpleNamespace(
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
                    reason="edgar filing weight",
                )
            ],
            config_hash="a",
            tape_hash="b",
            lineage_hash="c",
            experiment_hash="d",
        ),
        admitted_count=0,
    )
    refused = {**state, "run": cash_run, "rebalances": [{**row, "sized": False, "oos_skill": 0.2}]}
    with pytest.raises(ValueError, match="skill gate was cash"):
        barebone_edgar_ledger(bound=refused)
    refused["rebalances"] = [{**row, "sized": True, "oos_skill": 0.0}]
    with pytest.raises(ValueError, match="not strictly positive"):
        barebone_edgar_ledger(bound=refused)
    with pytest.raises(ValueError, match="missing a ledger"):
        barebone_edgar_ledger(bound={**state, "run": SimpleNamespace(ledger=None)})
    with pytest.raises(ValueError, match="missing an executed result"):
        barebone_edgar_metrics(bound={**state, "run": SimpleNamespace(result=None, metrics=None, ledger=None)})

    def _fake_write() -> tuple[Path, Path, Path]:
        return write_barebone_edgar_artifacts(tmp_path / "printed", bound=state)

    monkeypatch.setattr(edgar_run_module, "write_barebone_edgar_artifacts", _fake_write)
    printed = io.StringIO()
    with redirect_stdout(printed), redirect_stderr(printed):
        assert run_main([]) == 0
    assert "edgar_sha256=" in printed.getvalue()

    def _blocked() -> tuple[Path, Path, Path]:
        raise ValueError("edgar arm blocked")

    monkeypatch.setattr(edgar_run_module, "write_barebone_edgar_artifacts", _blocked)
    failed = io.StringIO()
    with redirect_stdout(failed), redirect_stderr(failed):
        assert run_main([]) == 1
    assert "edgar arm blocked" in failed.getvalue()

    monkeypatch.setattr(edgar_module, "fetch_edgar_events", lambda **_kwargs: "f" * 64)
    ingested = io.StringIO()
    with redirect_stdout(ingested), redirect_stderr(ingested):
        assert ingest_main(["--lock-config", "--config", str(tmp_path / "config.json")]) == 0
    assert "edgar_sha256=" in ingested.getvalue()
    missing = io.StringIO()
    monkeypatch.setattr(
        edgar_module,
        "fetch_edgar_events",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("SEC submissions request failed: 404")),
    )
    with redirect_stdout(missing), redirect_stderr(missing):
        assert ingest_main(["--config", str(tmp_path / "missing.json")]) == 1
    assert "404" in missing.getvalue()
