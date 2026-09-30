"""PIT Form 4 counts, amendment exclusion, and the insider skill gate."""

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

import demo.barebone_form4 as form4_module
import demo.barebone_form4_run as form4_run_module
from demo.barebone_comparison import (
    BAREBONE_UNIVERSE,
    EDGAR_CIK_MAP,
    FORM4_EVENTS,
    FORM4_PROVENANCE,
    LOCKED_EDGAR_SHA256,
    LOCKED_NARRATIVE_SHA256,
    LOCKED_POLARITY_SHA256,
    LOCKED_TAPE_SHA256,
    load_barebone_comparison_config,
)
from demo.barebone_edgar import EDGAR_SUBMISSIONS, EDGAR_USER_AGENT, load_cik_map
from demo.barebone_form4 import (
    FORM4_ARCHIVE,
    FORM4_LOOKBACK_SESSIONS,
    FORM4_SIGNAL_ID,
    _default_get_json,
    _default_get_xml,
    form4_provenance,
    form4_rows_from_columns,
    form4_score_rows,
    main as ingest_main,
    open_market_counts,
    ownership_xml_url,
    render_events_jsonl,
    rows_for_issuer,
    write_locked_form4_sha256,
    fetch_form4_events,
)
from demo.barebone_form4_run import (
    ARM_RELATIVE,
    LEDGER_RELATIVE,
    METRICS_RELATIVE,
    _arm_inputs,
    _load_events,
    _refuse_sized_without_skill,
    barebone_form4_arm_index,
    barebone_form4_ledger,
    barebone_form4_metrics,
    form4_book,
    form4_oos_pairs,
    form4_rebalances,
    main as run_main,
    raw_net_by_session,
    skill_before,
    write_barebone_form4_artifacts,
)
from demo.contracts import AssetBar, ExperimentConfig, MarketSession
from demo.controlled import ArmInput, benchmark_marks_from_closes, run_controlled_arm

_LEGACY_METRICS_SHA256 = "e7e5055ce9b4409d7941a71929f66261416b8d3c3f62423190edafd5bf5b1411"
_THREE_ARM_METRICS_SHA256 = "af40d4176d3e4705f799b3b640494777caa24ed81ea68f149a84bb8335d554f3"
_EDGAR_METRICS_SHA256 = "d0a71a6f784d0dc57c683a504d787a51dc68e382f7111a01ae06e7ae8038bd24"
_EDGAR_LEDGER_SHA256 = "6889e94024ecf641ce881b020db16ce9890641010820d7171f991a8a37974338"
_MOMENTUM_RETURN = 0.24597743358179014
_MOMENTUM_SHARPE = 1.6030790628765745


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sessions(count: int = 80) -> tuple[date, ...]:
    return tuple(date(2025, 1, 2) + timedelta(days=offset) for offset in range(count))


def _xml(cik: str, transactions: str) -> bytes:
    return (
        "<?xml version='1.0'?>"
        "<ownershipDocument>"
        "<documentType>4</documentType>"
        "<periodOfReport>2020-01-01</periodOfReport>"
        f"<issuer><issuerCik>{cik}</issuerCik></issuer>"
        f"<nonDerivativeTable>{transactions}</nonDerivativeTable>"
        "</ownershipDocument>"
    ).encode("utf-8")


def _tx(code: str, side: str) -> str:
    return (
        "<nonDerivativeTransaction>"
        "<transactionDate><value>2025-06-01</value></transactionDate>"
        f"<transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>"
        "<transactionAmounts>"
        f"<transactionAcquiredDisposedCode><value>{side}</value></transactionAcquiredDisposedCode>"
        "</transactionAmounts>"
        "</nonDerivativeTransaction>"
    )


def _event(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_id": "form4:0000320193-25-000001:AAPL",
        "ticker": "AAPL",
        "cik": "0000320193",
        "accession": "0000320193-25-000001",
        "form": "4",
        "acceptanceDateTime": "2025-01-02T15:00:00Z",
        "available_ts": "2025-01-02T15:00:00Z",
        "decision_session": "2025-01-03",
        "buy_count": 2,
        "sell_count": 0,
        "net_count": 2,
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


def test_amendments_are_excluded_and_acceptance_is_required() -> None:
    sessions = _sessions(5)
    seen: list[str] = []

    def _xml_get(url: str) -> bytes:
        seen.append(url)
        assert "xslF345" not in url
        assert url.endswith("/wk-form4.xml")
        return _xml("0000320193", _tx("P", "A") + _tx("S", "D") + _tx("G", "D"))

    rows, dropped = form4_rows_from_columns(
        {
            "form": ["4", "4/A", "8-K", "4", "10-K"],
            "accessionNumber": [
                "0000320193-25-000001",
                "0000320193-25-000002",
                "0000320193-25-000003",
                "0000320193-25-000004",
                "0000320193-25-000005",
            ],
            "acceptanceDateTime": [
                "2025-01-02T15:00:00.000Z",
                None,
                "2025-01-02T18:00:00.000Z",
                "2024-12-29T15:00:00.000Z",
                "2025-01-03T15:00:00.000Z",
            ],
            "primaryDocument": [
                "xslF345X05/wk-form4.xml",
                "xslF345X05/amend.xml",
                "report.htm",
                "xslF345X05/old.xml",
                "xslF345X05/tenk.xml",
            ],
            "reportDate": ["2020-01-01"] * 5,
            "periodOfReport": ["2020-01-01"] * 5,
        },
        ticker="AAPL",
        cik="0000320193",
        sessions=sessions,
        get_xml=_xml_get,
    )
    assert dropped == 0
    assert len(rows) == 1
    assert rows[0]["form"] == "4"
    assert rows[0]["available_ts"] == "2025-01-02T15:00:00Z"
    assert rows[0]["acceptanceDateTime"] == rows[0]["available_ts"]
    assert rows[0]["available_ts"] != "2020-01-01T00:00:00Z"
    assert rows[0]["decision_session"] == "2025-01-03"
    assert rows[0]["buy_count"] == 1
    assert rows[0]["sell_count"] == 1
    assert rows[0]["net_count"] == 0
    assert "periodOfReport" not in rows[0]
    assert "transactionDate" not in rows[0]
    assert "html" not in rows[0]
    assert str(rows[0]["uri"]).startswith(FORM4_ARCHIVE)
    assert len(seen) == 1
    foreign, foreign_dropped = form4_rows_from_columns(
        {
            "form": ["4"],
            "accessionNumber": ["0000320193-25-000011"],
            "acceptanceDateTime": ["2025-01-02T15:00:00Z"],
            "primaryDocument": ["wk-form4.xml"],
        },
        ticker="AAPL",
        cik="0000320193",
        sessions=sessions,
        get_xml=lambda _url: _xml("0000789019", _tx("P", "A")),
    )
    assert foreign == []
    assert foreign_dropped == 1
    with pytest.raises(ValueError, match="missing acceptanceDateTime"):
        form4_rows_from_columns(
            {
                "form": ["4"],
                "accessionNumber": ["0000320193-25-000008"],
                "acceptanceDateTime": [None],
                "primaryDocument": ["wk-form4.xml"],
            },
            ticker="AAPL",
            cik="0000320193",
            sessions=sessions,
            get_xml=_xml_get,
        )
    with pytest.raises(ValueError, match="not an XML file"):
        ownership_xml_url("0000320193", "0000320193-25-000001", "xslF345X05/form4.html")
    assert (
        form4_rows_from_columns(
            {
                "form": ["4/A", "8-K"],
                "accessionNumber": ["0000320193-25-000009", "0000320193-25-000010"],
                "acceptanceDateTime": [None, None],
                "primaryDocument": ["amend.xml", "report.htm"],
            },
            ticker="AAPL",
            cik="0000320193",
            sessions=sessions,
            get_xml=None,
        )
        == ([], 0)
    )


def test_open_market_codes_ignore_gifts_and_refuse_html() -> None:
    body = _xml("0000320193", _tx("P", "A") + _tx("S", "D") + _tx("G", "D") + _tx("A", "A") + _tx("M", "A"))
    assert open_market_counts(body, cik="0000320193") == (1, 1)
    derivative = _xml("0000320193", "").replace(
        b"<nonDerivativeTable></nonDerivativeTable>",
        b"<derivativeTable><derivativeTransaction>"
        b"<transactionCoding><transactionCode>S</transactionCode></transactionCoding>"
        b"<transactionAmounts><transactionAcquiredDisposedCode><value>D</value>"
        b"</transactionAcquiredDisposedCode></transactionAmounts>"
        b"</derivativeTransaction></derivativeTable>",
    )
    assert open_market_counts(derivative, cik="0000320193") == (0, 1)
    assert open_market_counts(_xml("0000320193", _tx("P", "D")), cik="0000320193") == (0, 0)
    assert open_market_counts(_xml("0000320193", _tx("S", "A")), cik="0000320193") == (0, 0)
    assert open_market_counts(
        _xml("0000320193", "<nonDerivativeTransaction></nonDerivativeTransaction>"),
        cik="0000320193",
    ) == (0, 0)
    with pytest.raises(ValueError, match="not exactly 4"):
        open_market_counts(body.replace(b"<documentType>4</documentType>", b"<documentType>4/A</documentType>"), cik="0000320193")
    assert open_market_counts(body, cik="0000789019") is None
    with pytest.raises(ValueError, match="HTML"):
        open_market_counts(b"<!DOCTYPE html><html></html>", cik="0000320193")
    with pytest.raises(ValueError, match="not XML"):
        open_market_counts(b"not xml", cik="0000320193")
    with pytest.raises(ValueError, match="missing issuerCik"):
        open_market_counts(b"<ownershipDocument><documentType>4</documentType></ownershipDocument>", cik="0000320193")


def test_count_window_is_point_in_time() -> None:
    sessions = _sessions()
    early = _event()
    scored = sessions[FORM4_LOOKBACK_SESSIONS - 1].isoformat()
    future_session = sessions[FORM4_LOOKBACK_SESSIONS].isoformat()
    future = _event(
        event_id="form4:0000320193-25-000099:AAPL",
        accession="0000320193-25-000099",
        acceptanceDateTime=f"{scored}T15:00:00Z",
        available_ts=f"{scored}T15:00:00Z",
        decision_session=future_session,
        buy_count=5,
        sell_count=0,
        net_count=5,
    )
    base = form4_score_rows([early], sessions, ("AAPL", "MSFT"))
    with_future = form4_score_rows([early, future], sessions, ("AAPL", "MSFT"))
    assert _row_on(base, scored, "AAPL") == _row_on(with_future, scored, "AAPL")
    first = _row_on(with_future, scored, "AAPL")
    assert first["net_count"] == 2
    assert _row_on(with_future, scored, "MSFT")["net_count"] == 0
    assert first["signal_id"] == FORM4_SIGNAL_ID
    assert first["lookback_sessions"] == 63
    later = _row_on(with_future, future_session, "AAPL")
    assert later["net_count"] == 7
    assert "periodOfReport" not in first
    with pytest.raises(ValueError, match="locked at 63"):
        form4_score_rows([early], sessions, ("AAPL", "MSFT"), lookback=21)
    with pytest.raises(ValueError, match="same-session"):
        form4_score_rows(
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
    with pytest.raises(ValueError, match="equal acceptanceDateTime"):
        form4_score_rows([_event(acceptanceDateTime="2025-01-02T16:00:00Z")], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="forbidden field"):
        form4_score_rows([_event(periodOfReport="2025-06-01")], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="forbidden field"):
        form4_score_rows([_event(transactionDate="2025-06-01")], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="not an exact"):
        form4_score_rows([_event(form="4/A")], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="buy_count minus sell_count"):
        form4_score_rows([_event(net_count=9)], sessions, ("AAPL", "MSFT"))
    with pytest.raises(ValueError, match="missing event_id"):
        form4_score_rows([_event(event_id=" ")], sessions, ("AAPL", "MSFT"))
    assert render_events_jsonl([]) == b""


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


def test_cik_map_refuse_and_fetch_keeps_the_edgar_lock(tmp_path: Path) -> None:
    mapping, _digest = load_cik_map()
    assert mapping["XOM"] == "0000034088"
    payload = json.loads((_root() / EDGAR_CIK_MAP).read_text(encoding="utf-8"))
    payload["entries"] = [entry for entry in payload["entries"] if entry["ticker"] != "NKE"]
    broken = tmp_path / "map.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing NKE"):
        load_cik_map(broken)

    root = tmp_path / "repo"
    _install_map(root)
    sessions = (date(2025, 6, 2), date(2025, 6, 3))
    config_path = _config_copy(tmp_path, edgar_sha256="a" * 64)
    with pytest.raises(ValueError, match="edgar_sha256 is locked"):
        fetch_form4_events(
            config_path=config_path,
            root=root,
            sessions=sessions,
            get_json=lambda _url: {},
            get_xml=lambda _url: b"",
        )
    config_path = _config_copy(tmp_path)
    with pytest.raises(ValueError, match="does not match the frozen map"):
        fetch_form4_events(
            config_path=config_path,
            root=root,
            sessions=sessions,
            get_json=lambda _url: {"cik": "0000000001", "filings": {}},
            get_xml=lambda _url: b"",
        )

    def _submissions(cik: str, offset: int) -> dict[str, object]:
        accession = f"{cik}-25-{offset + 1:06d}"
        return {
            "cik": str(int(cik)),
            "filings": {
                "recent": {
                    "form": ["4", "4/A", "8-K"],
                    "accessionNumber": [accession, f"{cik}-25-900000", f"{cik}-24-000001"],
                    "acceptanceDateTime": [
                        "2025-06-02T16:30:00.000Z",
                        None,
                        "2024-12-01T12:00:00.000Z",
                    ],
                    "primaryDocument": ["xslF345X05/wk-form4.xml", "xslF345X05/amend.xml", "report.htm"],
                    "reportDate": ["2025-06-01", "2025-06-01", "2024-12-01"],
                },
                "files": [],
            },
        }

    payloads = {
        EDGAR_SUBMISSIONS + f"CIK{cik}.json": _submissions(cik, offset)
        for offset, cik in enumerate(mapping.values())
    }

    def _reader(url: str) -> dict[str, object]:
        assert url.startswith(EDGAR_SUBMISSIONS)
        return payloads[url]

    def _documents(url: str) -> bytes:
        assert url.startswith(FORM4_ARCHIVE)
        assert url.endswith("/wk-form4.xml")
        assert "xslF345" not in url
        cik = f"{int(url.split('/')[-3]):010d}"
        return _xml(cik, _tx("P", "A"))

    digest = fetch_form4_events(
        config_path=config_path,
        root=root,
        lock_config=True,
        sessions=sessions,
        get_json=_reader,
        get_xml=_documents,
        now=datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
    )
    events = (root / FORM4_EVENTS).read_bytes()
    assert hashlib.sha256(events).hexdigest() == digest
    assert b"periodOfReport" not in events
    assert b"transactionDate" not in events
    assert b"<html" not in events.lower()
    assert b"4/A" not in events
    lines = [json.loads(line) for line in events.decode("utf-8").splitlines()]
    assert len(lines) == len(BAREBONE_UNIVERSE)
    assert all(row["form"] == "4" for row in lines)
    assert all(row["net_count"] == 1 for row in lines)
    assert all(row["available_ts"] == "2025-06-02T16:30:00Z" for row in lines)
    assert all(row["decision_session"] == "2025-06-03" for row in lines)
    sidecar = json.loads((root / FORM4_PROVENANCE).read_text(encoding="utf-8"))
    assert sidecar["edgar_form4_sha256"] == digest
    assert sidecar["edgar_sha256"] == LOCKED_EDGAR_SHA256
    assert sidecar["narrative_sha256"] == LOCKED_NARRATIVE_SHA256
    assert sidecar["period_of_report_used_as_available_ts"] is False
    assert sidecar["transaction_date_used_as_available_ts"] is False
    assert sidecar["signal_id"] == FORM4_SIGNAL_ID
    assert sidecar["open_market_buys"] == len(BAREBONE_UNIVERSE)
    assert sidecar["open_market_sells"] == 0
    locked = json.loads(config_path.read_text(encoding="utf-8"))
    assert locked["evidence"]["edgar_form4_sha256"] == digest
    assert locked["evidence"]["edgar_sha256"] == LOCKED_EDGAR_SHA256
    assert locked["evidence"]["narrative_sha256"] == LOCKED_NARRATIVE_SHA256
    assert locked["evidence"]["tape_sha256"] == LOCKED_TAPE_SHA256
    assert locked["comparable_performance_claim"] is False

    def _none(url: str) -> dict[str, object]:
        payload = json.loads(json.dumps(payloads[url]))
        payload["filings"]["recent"]["form"] = ["8-K", "8-K", "8-K"]
        return payload

    empty_root = tmp_path / "empty"
    _install_map(empty_root)
    empty_config = tmp_path / "empty-config"
    empty_config.mkdir()
    with pytest.raises(ValueError, match="empty Form 4 tape"):
        fetch_form4_events(
            config_path=_config_copy(empty_config),
            root=empty_root,
            sessions=sessions,
            get_json=_none,
            get_xml=_documents,
        )
    assert not (empty_root / FORM4_EVENTS).exists()


def test_readers_stay_on_official_sec_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="non-data.sec.gov"):
        _default_get_json("https://www.sec.gov/Archives/edgar/data/320193/index.html")
    with pytest.raises(ValueError, match="non-XML"):
        _default_get_xml(FORM4_ARCHIVE + "320193/000146235625000012/xslF345X05/wk-form4.xml")
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def _rate(request: urllib.request.Request, timeout: int = 60) -> bytes:
        assert request.get_header("User-agent") == EDGAR_USER_AGENT
        assert timeout == 60
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", _rate)
    with pytest.raises(ValueError, match="rate limit"):
        _default_get_json(EDGAR_SUBMISSIONS + "CIK0000320193.json")
    with pytest.raises(ValueError, match="rate limit"):
        form4_module._request(EDGAR_SUBMISSIONS + "CIK0000320193.json", "application/json")

    class _Body:
        def __enter__(self) -> "_Body":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def read(self) -> bytes:
            return b"<html></html>"

    monkeypatch.setattr(urllib.request, "urlopen", lambda _request, timeout=60: _Body())
    xml_url = FORM4_ARCHIVE + "320193/000146235625000012/wk-form4.xml"
    with pytest.raises(ValueError, match="HTML"):
        _default_get_xml(xml_url)

    class _Json(_Body):
        def read(self) -> bytes:
            return b'{"cik": "320193"}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda _request, timeout=60: _Json())
    assert _default_get_json(EDGAR_SUBMISSIONS + "CIK0000320193.json")["cik"] == "320193"
    recent = {
        "form": ["4"],
        "accessionNumber": ["0000320193-25-000001"],
        "acceptanceDateTime": ["2025-06-02T16:00:00.000Z"],
        "primaryDocument": ["wk-form4.xml"],
    }
    with pytest.raises(ValueError, match="need overflow"):
        rows_for_issuer(
            {"filings": {"recent": recent, "files": [{"name": "CIK0000320193-submissions-001.json"}]}},
            ticker="AAPL",
            cik="0000320193",
            sessions=(date(2025, 6, 2), date(2025, 6, 3)),
            get_json=None,
            get_xml=lambda _url: _xml("0000320193", _tx("P", "A")),
        )
    with pytest.raises(ValueError, match="changed tape_sha256"):
        form4_provenance(
            events_sha256="a" * 64,
            cik_map_sha256="b" * 64,
            tape_sha256="c" * 64,
            event_count=1,
            buy_transactions=1,
            sell_transactions=0,
            issuer_mismatch_dropped=0,
            fetched_at=datetime(2026, 9, 30, tzinfo=timezone.utc),
        )


def test_lock_refuses_changed_binds_and_controls_stay_put(tmp_path: Path) -> None:
    path = _config_copy(tmp_path, edgar_sha256="a" * 64)
    before = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="changed edgar_sha256"):
        write_locked_form4_sha256(path, "b" * 64)
    unchanged = json.loads(path.read_text(encoding="utf-8"))
    assert unchanged["evidence"]["edgar_sha256"] == "a" * 64
    assert unchanged["evidence"].get("edgar_form4_sha256") == before["evidence"].get("edgar_form4_sha256")
    assert unchanged["evidence"].get("edgar_form4_sha256") != "b" * 64
    path = _config_copy(tmp_path, narrative_sha256="a" * 64)
    with pytest.raises(ValueError, match="changed narrative_sha256"):
        write_locked_form4_sha256(path, "b" * 64)
    path = _config_copy(tmp_path, tape_sha256="a" * 64)
    with pytest.raises(ValueError, match="changed tape_sha256"):
        write_locked_form4_sha256(path, "b" * 64)
    source = json.loads((_root() / "configs" / "barebone-comparison-v1.json").read_text(encoding="utf-8"))
    source["comparable_performance_claim"] = True
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="comparable_performance_claim must be false"):
        write_locked_form4_sha256(path, "b" * 64)
    path = _config_copy(tmp_path)
    write_locked_form4_sha256(path, "c" * 64)
    locked = json.loads(path.read_text(encoding="utf-8"))
    assert locked["evidence"]["edgar_form4_sha256"] == "c" * 64
    assert locked["evidence"]["edgar_sha256"] == LOCKED_EDGAR_SHA256
    assert locked["evidence"]["narrative_sha256"] == LOCKED_NARRATIVE_SHA256
    assert locked["evidence"]["polarity_sha256"] == LOCKED_POLARITY_SHA256
    assert locked["evidence"]["tape_sha256"] == LOCKED_TAPE_SHA256

    root = _root()
    ignored = subprocess.run(["git", "check-ignore", "-q", FORM4_EVENTS], cwd=root, check=False)
    assert ignored.returncode == 0
    config = load_barebone_comparison_config()
    metrics = json.loads((root / "results" / "barebone_three_arm_metrics.json").read_text(encoding="utf-8"))
    assert config.evidence.edgar_sha256 == LOCKED_EDGAR_SHA256
    assert config.evidence.narrative_sha256 == LOCKED_NARRATIVE_SHA256
    assert config.evidence.tape_sha256 == LOCKED_TAPE_SHA256
    assert config.comparable_performance_claim is False
    assert metrics["arms"]["pure_ml"]["statistics"]["total_return"] == _MOMENTUM_RETURN
    assert metrics["arms"]["pure_ml"]["statistics"]["sharpe_ratio"] == _MOMENTUM_SHARPE
    assert hashlib.sha256((root / "results" / "metrics.json").read_bytes()).hexdigest() == _LEGACY_METRICS_SHA256
    assert hashlib.sha256((root / "results" / "barebone_three_arm_metrics.json").read_bytes()).hexdigest() == (
        _THREE_ARM_METRICS_SHA256
    )
    assert hashlib.sha256((root / "results" / "barebone_edgar_arm_metrics.json").read_bytes()).hexdigest() == (
        _EDGAR_METRICS_SHA256
    )
    assert hashlib.sha256((root / "results" / "barebone_edgar_arm_ledger.json").read_bytes()).hexdigest() == (
        _EDGAR_LEDGER_SHA256
    )
    spec = importlib.util.spec_from_file_location(
        "build_submission_archive", root / "scripts" / "build_submission_archive.py"
    )
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    events = root / FORM4_EVENTS
    created = not events.exists()
    if created:
        events.write_text("{}\n", encoding="utf-8")
    try:
        assert events not in builder.collect_files()
    finally:
        if created:
            events.unlink()
    form4_sha = config.evidence.edgar_form4_sha256
    if form4_sha is None:
        return
    provenance = json.loads((root / FORM4_PROVENANCE).read_text(encoding="utf-8"))
    assert provenance["edgar_form4_sha256"] == form4_sha
    assert provenance["edgar_sha256"] == LOCKED_EDGAR_SHA256
    assert provenance["period_of_report_used_as_available_ts"] is False
    assert provenance["transaction_date_used_as_available_ts"] is False
    ledger = json.loads((root / LEDGER_RELATIVE).read_text(encoding="utf-8"))
    assert ledger["comparable_performance_claim"] is False
    assert ledger["signal_id"] == FORM4_SIGNAL_ID
    assert ledger["edgar_sha256"] == LOCKED_EDGAR_SHA256
    assert ledger["max_position_weight"] == 0.1
    for row in ledger["oos_skill"]:
        if row["oos_skill"] is None or float(row["oos_skill"]) <= 0.0:
            assert row["sized"] is False
    local_ready = events.is_file() and (root / "evidence" / "market" / "barebone_window_ohlcv.json").is_file()
    if not local_ready:
        return
    assert hashlib.sha256(events.read_bytes()).hexdigest() == form4_sha
    assert (root / LEDGER_RELATIVE).read_text(encoding="utf-8") == json.dumps(
        barebone_form4_ledger(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    assert (root / METRICS_RELATIVE).read_text(encoding="utf-8") == json.dumps(
        barebone_form4_metrics(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    assert (root / ARM_RELATIVE).read_text(encoding="utf-8") == json.dumps(
        barebone_form4_arm_index(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def test_skill_gate_ignores_later_prices() -> None:
    config = _trading()
    raw = {"AAA": 2.0, "BBB": -1.0}
    assert form4_book(raw, None, config) == {}
    assert form4_book(raw, 0.0, config) == {}
    sized = form4_book(raw, 0.2, config)
    assert set(sized) == {"AAA"}
    assert sized["AAA"] <= 0.1
    dates = _sessions()
    raw_by_session = {day: {"AAA": 1.0, "BBB": 0.0} for day in dates[62:]}
    closes = {("AAA", index): 100.0 + index for index, _day in enumerate(dates)}
    closes.update({("BBB", index): 50.0 for index, _day in enumerate(dates)})
    pairs = form4_oos_pairs(raw_by_session, closes, dates)
    skill, count = skill_before(pairs, 75)
    closes[("AAA", 79)] = 1.0
    again, again_count = skill_before(form4_oos_pairs(raw_by_session, closes, dates), 75)
    assert skill == again
    assert count == again_count
    closes[("AAA", 74)] = 1.0
    changed, _changed_count = skill_before(form4_oos_pairs(raw_by_session, closes, dates), 75)
    assert changed != skill
    with pytest.raises(ValueError, match="failed the gate"):
        _refuse_sized_without_skill([{"sized": True, "oos_skill": 0.0}])
    with pytest.raises(ValueError, match="locked signal"):
        raw_net_by_session(
            [{"signal_id": "other", "lookback_sessions": 63, "session": "2025-03-05", "ticker": "AAA", "net_count": 0}]
        )
    rebalances = form4_rebalances(raw_by_session, closes, dates, config)
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
    payload = json.dumps({"form": "4", "buy_count": 1, "sell_count": 0, "net_count": 1}) + "\n"
    events = tmp_path / "events.jsonl"
    events.write_text(payload, encoding="utf-8")
    assert _load_events(events, hashlib.sha256(payload.encode("utf-8")).hexdigest())[0]["form"] == "4"
    banned = json.dumps({"form": "4", "periodOfReport": "2025-06-01", "buy_count": 0, "sell_count": 0, "net_count": 0})
    banned += "\n"
    events.write_text(banned, encoding="utf-8")
    with pytest.raises(ValueError, match="report date"):
        _load_events(events, hashlib.sha256(banned.encode("utf-8")).hexdigest())

    trading = ExperimentConfig.model_validate(
        {**_trading().model_dump(mode="json"), "start_date": "2025-01-02", "end_date": "2025-01-06", "rebalance_frequency_sessions": 1}
    )
    sessions = _kernel_sessions()
    marks = benchmark_marks_from_closes(sessions, benchmark_ticker="SPY", starting_capital=trading.starting_capital)
    run = run_controlled_arm(
        config=trading,
        sessions=sessions,
        strategy_id="form4",
        candidates=[
            ArmInput(strategy_id="form4", decision_session=date(2025, 1, 2), ticker="AAA", structured_weight=0.1)
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
        "edgar_form4_sha256": "e" * 64,
        "event_count": 4,
    }
    ledger = barebone_form4_ledger(bound=state)
    metrics = barebone_form4_metrics(bound=state)
    index = barebone_form4_arm_index(bound=state)
    assert ledger["comparable_performance_claim"] is False
    assert ledger["signal_id"] == FORM4_SIGNAL_ID
    assert ledger["edgar_sha256"] == LOCKED_EDGAR_SHA256
    assert "notional" in str(ledger["description"]).lower() or "Notional" in str(ledger["description"])
    assert metrics["statistics"]["total_return"] == run.metrics.total_return
    assert index["event_count"] == 4
    written = write_barebone_form4_artifacts(tmp_path / "repo", bound=state)
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
                        reason="form4 insider weight",
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
        barebone_form4_ledger(bound=refused)
    with pytest.raises(ValueError, match="missing a ledger"):
        barebone_form4_ledger(bound={**state, "run": SimpleNamespace(ledger=None)})
    with pytest.raises(ValueError, match="missing an executed result"):
        barebone_form4_metrics(bound={**state, "run": SimpleNamespace(result=None, metrics=None, ledger=None)})

    monkeypatch.setattr(
        form4_run_module,
        "write_barebone_form4_artifacts",
        lambda root=None, bound=None: write_barebone_form4_artifacts(tmp_path / "printed", bound=state),
    )
    printed = io.StringIO()
    with redirect_stdout(printed), redirect_stderr(printed):
        assert run_main([]) == 0
    assert "edgar_form4_sha256=" in printed.getvalue()
    monkeypatch.setattr(
        form4_run_module,
        "write_barebone_form4_artifacts",
        lambda root=None, bound=None: (_ for _ in ()).throw(ValueError("form4 arm blocked")),
    )
    failed = io.StringIO()
    with redirect_stdout(failed), redirect_stderr(failed):
        assert run_main([]) == 1
    assert "form4 arm blocked" in failed.getvalue()
    monkeypatch.setattr(form4_module, "fetch_form4_events", lambda **_kwargs: "f" * 64)
    ingested = io.StringIO()
    with redirect_stdout(ingested), redirect_stderr(ingested):
        assert ingest_main(["--lock-config", "--config", str(tmp_path / "config.json")]) == 0
    assert "edgar_form4_sha256=" in ingested.getvalue()
