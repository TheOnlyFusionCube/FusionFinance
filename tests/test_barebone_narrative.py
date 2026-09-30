"""Fail-closed point-in-time narrative ingest for barebone-comparison-v1."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from demo.barebone_comparison import (
    BAREBONE_UNIVERSE,
    FAIR_RACE_TAPE_HASH,
    NARRATIVE_EVENTS,
    NARRATIVE_MAP,
    NARRATIVE_PROVENANCE,
    load_barebone_comparison_config,
)
from demo.barebone_narrative import (
    HN_PROVIDER,
    NARRATIVE_CASH_RECORD,
    _default_get_json,
    _hn_url,
    cash_narrative_record,
    collect_hn_query,
    decision_session_for,
    events_from_hn_hits,
    fetch_hn_events,
    ingest_hn_narrative,
    load_ticker_map,
    main,
    provenance_document,
    provider_skip_message,
    query_terms,
    render_events_jsonl,
    sessions_from_ohlcv,
    validate_event,
    write_locked_narrative_sha256,
)

_LEGACY_METRICS_SHA256 = "e7e5055ce9b4409d7941a71929f66261416b8d3c3f62423190edafd5bf5b1411"
_LOCKED_NARRATIVE_SHA256 = "860cb1d3a86fd4a3353a876d421618d69e76228c65e44b6dac7e28c820b9d2a9"
_LOCKED_TAPE_SHA256 = "c29f4810a8433e0de286da46409dbe95c17c1fa09d25e7809bb5f9e73ad8a205"
_SESSIONS = (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6))
_APPLE = "2025-01-02T15:00:00Z"


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_copy(tmp_path: Path) -> Path:
    source = _root() / "configs" / "barebone-comparison-v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["evidence"]["narrative_sha256"] = None
    destination = tmp_path / "barebone-comparison-v1.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    map_destination = tmp_path / NARRATIVE_MAP
    map_destination.parent.mkdir(parents=True, exist_ok=True)
    map_destination.write_text((_root() / NARRATIVE_MAP).read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def _apple_hit(object_id: str = "101") -> dict[str, object]:
    return {
        "objectID": object_id,
        "created_at": _APPLE,
        "title": "Apple announces a new chip",
        "url": "https://example.test/apple",
    }


def _parse_hn_url(url: str) -> tuple[str, int, int, int]:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    filters = query["numericFilters"][0]
    start_text, end_text = filters.split(",")
    return (
        query["query"][0],
        int(start_text.split(">=")[1]),
        int(end_text.split("<")[1]),
        int(query["page"][0]),
    )


def test_decision_session_is_strictly_after_the_utc_date() -> None:
    available = datetime(2025, 1, 2, 23, 0, tzinfo=timezone.utc)

    assert decision_session_for(available, _SESSIONS) == date(2025, 1, 3)
    assert decision_session_for(available, (date(2025, 1, 6), date(2025, 1, 3))) == date(2025, 1, 3)
    assert decision_session_for(datetime(2026, 1, 12, 12, tzinfo=timezone.utc), _SESSIONS) is None
    with pytest.raises(ValueError, match="timezone-aware"):
        decision_session_for(datetime(2025, 1, 2, 12), _SESSIONS)


def test_same_session_and_look_ahead_are_refused() -> None:
    event = {
        "event_id": "hn:1:AAPL",
        "ticker": "AAPL",
        "available_ts": _APPLE,
        "decision_session": "2025-01-02",
        "source_class": "social",
        "source_name": "hackernews-algolia",
        "text": "Apple announces a new chip",
    }

    with pytest.raises(ValueError, match="strictly after"):
        validate_event(event, _SESSIONS)
    event["decision_session"] = "2025-01-06"
    with pytest.raises(ValueError, match="strictly after"):
        validate_event(event, _SESSIONS)


def test_unmapped_ticker_and_missing_timestamp_are_refused() -> None:
    mapped = events_from_hn_hits([_apple_hit()], ticker_map=load_ticker_map(), sessions=_SESSIONS)
    dropped = events_from_hn_hits(
        [{**_apple_hit("202"), "title": "Banana bread recipe"}],
        ticker_map=load_ticker_map(),
        sessions=_SESSIONS,
    )
    blank = events_from_hn_hits(
        [{**_apple_hit("203"), "title": "   ", "story_text": ""}],
        ticker_map=load_ticker_map(),
        sessions=_SESSIONS,
    )

    assert [row["ticker"] for row in mapped] == ["AAPL"]
    assert mapped[0]["decision_session"] == "2025-01-03"
    assert dropped == []
    assert blank == []
    with pytest.raises(ValueError, match="missing available_ts"):
        events_from_hn_hits([{**_apple_hit(), "created_at": ""}], ticker_map=load_ticker_map(), sessions=_SESSIONS)
    with pytest.raises(ValueError, match="not in the sealed map universe"):
        validate_event(
            {
                "event_id": "hn:9:ZZZ",
                "ticker": "ZZZ",
                "available_ts": _APPLE,
                "decision_session": "2025-01-03",
                "source_class": "social",
                "source_name": "hackernews-algolia",
                "text": "unmapped",
            },
            _SESSIONS,
        )
    with pytest.raises(ValueError, match="missing available_ts"):
        validate_event({"event_id": "hn:9:AAPL", "ticker": "AAPL"}, _SESSIONS)


def test_return_labels_and_unfrozen_polarity_are_refused() -> None:
    event = {
        "event_id": "hn:1:AAPL",
        "ticker": "AAPL",
        "available_ts": _APPLE,
        "decision_session": "2025-01-03",
        "source_class": "social",
        "source_name": "hackernews-algolia",
        "text": "Apple announces a new chip",
        "total_return": 0.1,
    }
    with pytest.raises(ValueError, match="return label"):
        validate_event(event, _SESSIONS)
    del event["total_return"]
    event["polarity"] = 1
    with pytest.raises(ValueError, match="polarity_model_id"):
        validate_event(event, _SESSIONS)
    event["polarity_model_id"] = "frozen-v0"
    event["polarity_text_hash"] = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(ValueError, match="does not match"):
        validate_event(event, _SESSIONS)
    event["polarity_text_hash"] = hashlib.sha256(str(event["text"]).encode("utf-8")).hexdigest()
    cleaned = validate_event(event, _SESSIONS)
    assert cleaned["polarity_model_id"] == "frozen-v0"


def test_ambiguous_names_need_a_cashtag_or_a_distinctive_company_name() -> None:
    ticker_map = load_ticker_map()

    assert ticker_map.match("The cat sat on a low wall") == ()
    assert ticker_map.match("A meta analysis") == ()
    assert ticker_map.match("Caterpillar raised guidance") == ("CAT",)
    assert ticker_map.match("Watch $CAT and $V today") == ("V", "CAT")
    assert ticker_map.match("Lowe's opened a store") == ("LOW",)
    assert ticker_map.match("A meta analysis of Facebook ads") == ("META",)


def test_sealed_map_covers_the_barebone_universe(tmp_path: Path) -> None:
    payload = json.loads((_root() / NARRATIVE_MAP).read_text(encoding="utf-8"))
    tickers = [entry["ticker"] for entry in payload["entries"]]

    assert payload["comparable_performance_claim"] is False
    assert tickers == list(BAREBONE_UNIVERSE)
    assert load_ticker_map().match("$BRK.B Berkshire") == ("BRK-B",)
    broken = json.loads(json.dumps(payload))
    broken["entries"] = [entry for entry in broken["entries"] if entry["ticker"] != "PFE"]
    path = tmp_path / "broken_map.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="missing PFE"):
        load_ticker_map(path)


def test_hn_query_disables_typo_tolerance() -> None:
    url = _hn_url("AMD", 10, 20, 0)
    assert "tags=story" in url
    assert "typoTolerance=false" in url
    assert "hitsPerPage=1000" in url


def test_hn_fixture_splits_a_capped_window_and_paginates() -> None:
    from demo.barebone_narrative import _window_bounds

    start_i, end_i = _window_bounds()
    calls: list[tuple[str, int, int, int]] = []

    def get_json(url: str) -> dict[str, object]:
        parsed = _parse_hn_url(url)
        calls.append(parsed)
        _query, start, end, page = parsed
        if (start, end) == (start_i, end_i):
            return {"nbHits": 1001, "hits": [{"objectID": "ignored"}]}
        if page == 0:
            return {"nbHits": 2, "hits": [_apple_hit(f"{start}-a")]}
        if page == 1:
            return {"nbHits": 2, "hits": [_apple_hit(f"{start}-b")]}
        return {"nbHits": 2, "hits": []}

    rows = collect_hn_query("Apple", start_i, end_i, get_json=get_json, throttle_s=0)
    assert calls[0][1:] == (start_i, end_i, 0)
    assert len(rows) == 4
    assert collect_hn_query("Apple", 5, 5, get_json=get_json) == []
    with pytest.raises(ValueError, match="one second"):
        collect_hn_query("Apple", 10, 11, get_json=lambda _url: {"nbHits": 1001, "hits": []})
    with pytest.raises(ValueError, match="ended before nbHits"):
        collect_hn_query(
            "Apple",
            10,
            20,
            get_json=lambda url: {"nbHits": 2, "hits": [_apple_hit()] if _parse_hn_url(url)[3] == 0 else []},
        )


def test_fixture_ingest_locks_only_a_non_empty_file(tmp_path: Path) -> None:
    config_path = _config_copy(tmp_path)
    map_payload = json.loads((tmp_path / NARRATIVE_MAP).read_text(encoding="utf-8"))

    def get_json(url: str) -> dict[str, object]:
        query, _start, _end, page = _parse_hn_url(url)
        if page == 0 and query in {"Apple", "$AAPL"}:
            return {"nbHits": 1, "hits": [_apple_hit("501" if query == "Apple" else "501")]}
        return {"nbHits": 0, "hits": []}

    digest = ingest_hn_narrative(
        config_path=config_path,
        root=tmp_path,
        get_json=get_json,
        throttle_s=0,
        lock_config=True,
        sessions=_SESSIONS,
        now=datetime(2026, 9, 30, 5, 0, tzinfo=timezone.utc),
    )
    events_path = tmp_path / NARRATIVE_EVENTS
    body = events_path.read_bytes()
    sidecar = json.loads((tmp_path / NARRATIVE_PROVENANCE).read_text(encoding="utf-8"))
    record = json.loads((tmp_path / NARRATIVE_CASH_RECORD).read_text(encoding="utf-8"))
    locked = load_barebone_comparison_config(config_path)

    assert digest == hashlib.sha256(body).hexdigest()
    assert locked.evidence.narrative_sha256 == digest
    assert locked.evidence.tape_sha256 == _LOCKED_TAPE_SHA256
    assert locked.comparable_performance_claim is False
    assert locked.scorebook == "momentum"
    assert locked.experiment.max_position_weight == 0.1
    assert locked.experiment.max_gross_leverage == 1.0
    assert sidecar["provider"] == HN_PROVIDER
    assert sidecar["byte_sha256"] == digest
    assert sidecar["license_note"] == "not redistributed; local bind only"
    assert sidecar["comparable_performance_claim"] is False
    assert "text" not in sidecar
    assert "title" not in sidecar
    assert "hits" not in sidecar
    assert sidecar["event_count"] == 1
    assert sidecar["tickers_with_events"] == ["AAPL"]
    assert record["sized"] is False
    assert record["strategy_id"] == "narrative"
    assert record["narrative_sha256"] == digest
    assert record["tape_sha256"] == _LOCKED_TAPE_SHA256
    assert record["comparable_performance_claim"] is False
    assert "sharpe_ratio" not in record
    assert "total_return" not in record
    events = [json.loads(line) for line in body.decode("utf-8").splitlines()]
    assert events[0]["decision_session"] == "2025-01-03"
    assert "Apple" in str(events[0]["text"])

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    empty_config = _config_copy(empty_root)
    with pytest.raises(ValueError, match="empty narrative tape"):
        ingest_hn_narrative(
            config_path=empty_config,
            root=empty_root,
            get_json=lambda _url: {"nbHits": 0, "hits": []},
            throttle_s=0,
            lock_config=True,
            sessions=_SESSIONS,
        )
    assert load_barebone_comparison_config(empty_config).evidence.narrative_sha256 is None
    assert fetch_hn_events(
        sessions=_SESSIONS,
        ticker_map=load_ticker_map(),
        map_payload=map_payload,
        get_json=lambda _url: {"nbHits": 0, "hits": []},
        throttle_s=0,
    ) == []


def test_claim_true_is_refused(tmp_path: Path) -> None:
    config_path = _config_copy(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["comparable_performance_claim"] = True
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="comparable_performance_claim must be false"):
        load_barebone_comparison_config(config_path)
    with pytest.raises(ValueError, match="comparable_performance_claim must be false"):
        write_locked_narrative_sha256(config_path, "a" * 64)
    with pytest.raises(ValueError, match="lowercase sha256"):
        cash_narrative_record("not-a-hash")
    with pytest.raises(ValueError, match="Hacker News via Algolia"):
        provenance_document(
            digest="a" * 64,
            events=[],
            fetched_at=datetime(2026, 9, 30, tzinfo=timezone.utc),
            provider="X recent search",
        )


def test_reddit_and_x_skip_without_inventing_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT", "X_BEARER_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    events_path = _root() / NARRATIVE_EVENTS
    existed = events_path.exists()
    out = io.StringIO()
    with redirect_stdout(out):
        assert main(["--provider", "reddit", "--lock-config"]) == 0
    assert "reddit provider skipped" in out.getvalue()
    out = io.StringIO()
    with redirect_stdout(out):
        assert main(["--provider", "x"]) == 0
    assert "recent search is not a full-window archive" in out.getvalue()
    assert events_path.exists() is existed
    assert provider_skip_message("reddit", {}) is not None
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "agent")
    monkeypatch.setenv("X_BEARER_TOKEN", "token")
    assert provider_skip_message("reddit") is None
    assert provider_skip_message("x") is None
    out = io.StringIO()
    with redirect_stdout(out):
        assert main(["--provider", "reddit"]) == 0
    assert "does not call that API" in out.getvalue()
    out = io.StringIO()
    with redirect_stdout(out):
        assert main(["--provider", "x", "--lock-config"]) == 0
    assert "does not invent a full-window archive" in out.getvalue()
    assert not (tmp_path / NARRATIVE_EVENTS).exists()


def test_events_jsonl_is_gitignored_and_archive_skips_it() -> None:
    root = _root()
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", NARRATIVE_EVENTS],
        cwd=root,
        check=False,
    )
    provenance = subprocess.run(
        ["git", "check-ignore", "-q", NARRATIVE_PROVENANCE],
        cwd=root,
        check=False,
    )
    ticker_map = subprocess.run(
        ["git", "check-ignore", "-q", NARRATIVE_MAP],
        cwd=root,
        check=False,
    )
    raw = subprocess.run(
        ["git", "check-ignore", "-q", "evidence/narrative/hn_raw.json"],
        cwd=root,
        check=False,
    )

    assert ignored.returncode == 0
    assert raw.returncode == 0
    assert provenance.returncode == 1
    assert ticker_map.returncode == 1
    spec = importlib.util.spec_from_file_location(
        "build_submission_archive", root / "scripts" / "build_submission_archive.py"
    )
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    path = root / NARRATIVE_EVENTS
    created = not path.exists()
    if created:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"text":"do not archive"}\n', encoding="utf-8")
    try:
        assert path not in builder.collect_files()
    finally:
        if created:
            path.unlink()


def test_checked_in_narrative_hash_matches_provenance_and_momentum_tape_is_unchanged(
    tmp_path: Path,
) -> None:
    config = load_barebone_comparison_config()
    ledger = json.loads((_root() / "results" / "barebone_three_arm_ledger.json").read_text(encoding="utf-8"))
    metrics = json.loads((_root() / "results" / "barebone_three_arm_metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((_root() / NARRATIVE_PROVENANCE).read_text(encoding="utf-8"))
    copy = _config_copy(tmp_path)
    events_path = _root() / NARRATIVE_EVENTS

    assert config.evidence.narrative_sha256 == _LOCKED_NARRATIVE_SHA256
    assert provenance["byte_sha256"] == _LOCKED_NARRATIVE_SHA256
    assert provenance["provider"] == HN_PROVIDER
    if events_path.is_file():
        assert hashlib.sha256(events_path.read_bytes()).hexdigest() == _LOCKED_NARRATIVE_SHA256
    assert config.evidence.narrative_events == NARRATIVE_EVENTS
    assert config.evidence.tape_sha256 == _LOCKED_TAPE_SHA256
    assert ledger["tape_sha256"] == _LOCKED_TAPE_SHA256
    assert metrics["tape_sha256"] == _LOCKED_TAPE_SHA256
    assert (
        hashlib.sha256((_root() / "results" / "metrics.json").read_bytes()).hexdigest()
        == _LEGACY_METRICS_SHA256
    )
    assert render_events_jsonl([]) == b""
    with pytest.raises(ValueError, match="fair-race"):
        write_locked_narrative_sha256(copy, FAIR_RACE_TAPE_HASH)
    assert load_barebone_comparison_config().evidence.narrative_sha256 == _LOCKED_NARRATIVE_SHA256
    assert load_barebone_comparison_config(copy).evidence.narrative_sha256 is None


def test_calendar_and_hn_transport_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calendar = tmp_path / "tape.json"
    calendar.write_text(
        json.dumps({"bars": [{"date": "2025-01-02"}, {"date": "2026-01-12"}]}),
        encoding="utf-8",
    )
    assert sessions_from_ohlcv(calendar) == (date(2025, 1, 2), date(2026, 1, 12))
    calendar.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no bars"):
        sessions_from_ohlcv(calendar)

    def boom(request: urllib.request.Request, timeout: int = 60) -> object:
        del timeout
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many", hdrs=None, fp=None)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ValueError, match="HN rate limit"):
        _default_get_json("https://hn.algolia.com/api/v1/search_by_date?query=Apple")

    class _Body:
        def read(self) -> bytes:
            return b'{"nbHits": 0, "hits": []}'

        def __enter__(self) -> _Body:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: _Body())
    assert _default_get_json("https://hn.algolia.com/api/v1/search_by_date")["nbHits"] == 0
    terms = query_terms(json.loads((_root() / NARRATIVE_MAP).read_text(encoding="utf-8")))
    assert "$AAPL" in terms
    assert "Apple" in terms
    assert cash_narrative_record(None, event_count=0)["sized"] is False


def test_main_hn_refuses_a_missing_config(tmp_path: Path) -> None:
    err = io.StringIO()
    with redirect_stderr(err):
        code = main(["--provider", "hn", "--config", str(tmp_path / "missing.json")])

    assert code == 1
    assert err.getvalue()
    assert load_barebone_comparison_config().evidence.narrative_sha256 == _LOCKED_NARRATIVE_SHA256
