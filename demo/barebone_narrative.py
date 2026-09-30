"""Point-in-time narrative ingest for the Barebone window.

Hacker News stories come from the Algolia ``search_by_date`` API. A sealed
ticker map assigns cashtags and company names. Unmapped text is dropped.
A story with no ``created_at`` aborts the ingest. The decision session is the
first Barebone calendar session strictly after the UTC date of
``available_ts``. Same-session text is not a feature.

Reddit and X are stubs. Missing credentials skip the provider. They do not
invent events. The events file is gitignored. ``comparable_performance_claim``
stays false. This module does not score a book and does not publish a Sharpe.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from demo.barebone_comparison import (
    BAREBONE_EXPERIMENT_ID,
    BAREBONE_UNIVERSE,
    BAREBONE_WINDOW,
    NARRATIVE_EVENTS,
    NARRATIVE_MAP,
    NARRATIVE_PROVENANCE,
    load_barebone_comparison_config,
    validate_barebone_payload,
)

HN_ENDPOINT = "https://hn.algolia.com/api/v1/search_by_date"
HN_PROVIDER = "Hacker News via Algolia"
HN_SOURCE_NAME = "hackernews-algolia"
HN_HITS_CAP = 1000
NARRATIVE_LICENSE_NOTE = "not redistributed; local bind only"
NARRATIVE_CASH_RECORD = "results/barebone_narrative_arm.json"
_MAP_SCHEMA = "fusionfinance-barebone-ticker-map-v1"
_PROVENANCE_SCHEMA = "fusionfinance-barebone-narrative-provenance-v1"
_SOURCE_CLASSES = frozenset({"news", "social", "research", "other"})
_BANNED_FIELDS = frozenset(
    {
        "total_return",
        "forward_return",
        "next_return",
        "residual",
        "label",
        "sharpe_ratio",
        "sortino_ratio",
        "annualized_return",
    }
)
_REDDIT_ENV = ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT")
_X_ENV = ("X_BEARER_TOKEN",)
_JsonGet = Callable[[str], Mapping[str, object]]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _aware_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"available_ts is not a timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError("available_ts must be timezone-aware UTC")
    return parsed.astimezone(timezone.utc)


def decision_session_for(available: datetime, sessions: Sequence[date]) -> date | None:
    """First calendar session strictly after the UTC date of ``available``."""

    if available.tzinfo is None:
        raise ValueError("available_ts must be timezone-aware UTC")
    day = available.astimezone(timezone.utc).date()
    later = [session for session in sessions if session > day]
    if not later:
        return None
    return min(later)


def sessions_from_ohlcv(path: Path) -> tuple[date, ...]:
    """Read session dates from the local tape. Prices are not retained."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    bars = payload.get("bars") if isinstance(payload, dict) else None
    if not isinstance(bars, list) or not bars:
        raise ValueError("barebone calendar tape has no bars")
    found: set[date] = set()
    for bar in bars:
        if not isinstance(bar, dict) or "date" not in bar:
            raise ValueError("barebone calendar bar is missing a date")
        found.add(date.fromisoformat(str(bar["date"])))
    if not found:
        raise ValueError("barebone calendar tape has no sessions")
    ordered = tuple(sorted(found))
    start, end = BAREBONE_WINDOW
    if ordered[0].isoformat() != start or ordered[-1].isoformat() != end:
        raise ValueError("barebone calendar tape does not match the locked window")
    return ordered


class TickerMap:
    """Sealed cashtag and company-name patterns. Unmapped text stays unmapped."""

    def __init__(self, entries: Sequence[Mapping[str, object]]) -> None:
        compiled: list[tuple[str, tuple[re.Pattern[str], ...]]] = []
        seen: set[str] = set()
        for entry in entries:
            ticker = str(entry["ticker"]).strip().upper()
            if ticker in seen:
                raise ValueError(f"ticker map repeats {ticker}")
            seen.add(ticker)
            if ticker not in BAREBONE_UNIVERSE:
                raise ValueError(f"ticker map name {ticker} is outside the barebone universe")
            patterns: list[re.Pattern[str]] = []
            cashtags = entry.get("cashtags")
            names = entry.get("names")
            if not isinstance(cashtags, list) or not cashtags:
                raise ValueError(f"ticker map entry {ticker} needs a cashtag")
            if not isinstance(names, list):
                raise ValueError(f"ticker map entry {ticker} needs a name list")
            for tag in cashtags:
                token = str(tag).strip().upper().lstrip("$")
                if not token:
                    raise ValueError(f"ticker map cashtag for {ticker} is blank")
                patterns.append(
                    re.compile(rf"(?<![A-Z0-9])\${re.escape(token)}(?![A-Z0-9])", re.IGNORECASE)
                )
            for name in names:
                phrase = str(name).strip()
                if not phrase:
                    raise ValueError(f"ticker map name for {ticker} is blank")
                patterns.append(
                    re.compile(
                        rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])",
                        re.IGNORECASE,
                    )
                )
            compiled.append((ticker, tuple(patterns)))
        missing = [ticker for ticker in BAREBONE_UNIVERSE if ticker not in seen]
        if missing:
            raise ValueError("ticker map is missing " + ", ".join(missing))
        self._compiled = tuple(compiled)

    def match(self, text: str) -> tuple[str, ...]:
        found = [ticker for ticker, patterns in self._compiled if any(pattern.search(text) for pattern in patterns)]
        return tuple(found)


def load_ticker_map(path: Path | None = None) -> TickerMap:
    map_path = path or (_repo_root() / NARRATIVE_MAP)
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ticker map must be a JSON object")
    if payload.get("schema") != _MAP_SCHEMA:
        raise ValueError("ticker map schema is missing or changed")
    if payload.get("experiment_id") != BAREBONE_EXPERIMENT_ID:
        raise ValueError("ticker map experiment_id must be barebone-comparison-v1")
    if payload.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    if _BANNED_FIELDS.intersection(payload):
        raise ValueError("ticker map must not carry a return label")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("ticker map has no entries")
    return TickerMap(entries)


def validate_event(event: Mapping[str, object], sessions: Sequence[date]) -> dict[str, object]:
    """Refuse look-ahead, unmapped names, missing timestamps, and return labels."""

    if not isinstance(event, Mapping):
        raise ValueError("narrative event must be an object")
    banned = _BANNED_FIELDS.intersection(event)
    if banned:
        raise ValueError("narrative event carries a return label: " + ", ".join(sorted(banned)))
    if "available_ts" not in event or event.get("available_ts") in (None, ""):
        raise ValueError("narrative event is missing available_ts")
    available = _aware_utc(str(event["available_ts"]))
    ticker = str(event.get("ticker", "")).strip().upper()
    if ticker not in BAREBONE_UNIVERSE:
        raise ValueError(f"narrative event ticker {ticker or '(blank)'} is not in the sealed map universe")
    session_text = event.get("decision_session")
    if not isinstance(session_text, str) or not session_text:
        raise ValueError("narrative event is missing decision_session")
    session = date.fromisoformat(session_text)
    if session not in set(sessions):
        raise ValueError(f"decision_session {session.isoformat()} is not on the barebone calendar")
    expected = decision_session_for(available, sessions)
    if expected is None or session != expected:
        raise ValueError("decision_session must be the first session strictly after available_ts")
    if session <= available.date():
        raise ValueError("same-session narrative text is refused")
    source_class = event.get("source_class")
    if source_class not in _SOURCE_CLASSES:
        raise ValueError("narrative source_class is not news, social, research, or other")
    if not str(event.get("source_name", "")).strip():
        raise ValueError("narrative event is missing source_name")
    if not str(event.get("event_id", "")).strip():
        raise ValueError("narrative event is missing event_id")
    text = event.get("text")
    text_hash = event.get("text_hash")
    uri = event.get("uri")
    has_text = isinstance(text, str) and bool(text.strip())
    has_hash = isinstance(text_hash, str) and bool(text_hash) and isinstance(uri, str) and bool(uri)
    if not has_text and not has_hash:
        raise ValueError("narrative event needs text or text_hash plus a URI")
    if "polarity" in event:
        model_id = event.get("polarity_model_id")
        bound_hash = event.get("polarity_text_hash")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("polarity requires a frozen polarity_model_id")
        if not has_text or not isinstance(bound_hash, str):
            raise ValueError("polarity must be bound to text available at available_ts")
        digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
        if bound_hash != digest:
            raise ValueError("polarity text hash does not match the available text")
    cleaned = dict(event)
    cleaned["ticker"] = ticker
    cleaned["available_ts"] = available.strftime("%Y-%m-%dT%H:%M:%SZ")
    cleaned["decision_session"] = session.isoformat()
    return cleaned


def _hit_text(hit: Mapping[str, object]) -> str:
    parts: list[str] = []
    title = hit.get("title")
    story = hit.get("story_text")
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())
    if isinstance(story, str) and story.strip():
        parts.append(story.strip())
    return "\n".join(parts)


def events_from_hn_hits(
    hits: Sequence[Mapping[str, object]],
    *,
    ticker_map: TickerMap,
    sessions: Sequence[date],
) -> list[dict[str, object]]:
    """Map one Algolia page of stories. Missing timestamps raise."""

    events: list[dict[str, object]] = []
    seen: set[str] = set()
    for hit in hits:
        if not isinstance(hit, Mapping):
            raise ValueError("HN hit is not an object")
        created = hit.get("created_at")
        if not isinstance(created, str) or not created.strip():
            raise ValueError("HN hit is missing available_ts")
        available = _aware_utc(created)
        object_id = str(hit.get("objectID", "")).strip()
        if not object_id:
            raise ValueError("HN hit is missing objectID")
        text = _hit_text(hit)
        if not text:
            continue
        session = decision_session_for(available, sessions)
        if session is None:
            continue
        uri = hit.get("url") if isinstance(hit.get("url"), str) and hit.get("url") else None
        for ticker in ticker_map.match(text):
            event_id = f"hn:{object_id}:{ticker}"
            if event_id in seen:
                continue
            seen.add(event_id)
            events.append(
                validate_event(
                    {
                        "event_id": event_id,
                        "ticker": ticker,
                        "available_ts": available.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "decision_session": session.isoformat(),
                        "source_class": "social",
                        "source_name": HN_SOURCE_NAME,
                        "text": text,
                        "uri": uri,
                        "object_id": object_id,
                    },
                    sessions,
                )
            )
    events.sort(key=lambda row: (str(row["available_ts"]), str(row["event_id"])))
    return events


def _window_bounds() -> tuple[int, int]:
    start = datetime.fromisoformat(BAREBONE_WINDOW[0] + "T00:00:00+00:00")
    end = datetime.fromisoformat(BAREBONE_WINDOW[1] + "T00:00:00+00:00") + timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp())


def _hn_url(query: str, start_i: int, end_i: int, page: int) -> str:
    filters = f"created_at_i>={start_i},created_at_i<{end_i}"
    params = urllib.parse.urlencode(
        {
            "query": query,
            "tags": "story",
            "numericFilters": filters,
            "hitsPerPage": str(HN_HITS_CAP),
            "page": str(page),
            # Short tokens such as AMD otherwise match unrelated words.
            "typoTolerance": "false",
        }
    )
    return f"{HN_ENDPOINT}?{params}"


def _as_hit_page(payload: Mapping[str, object]) -> tuple[int, list[Mapping[str, object]]]:
    nb_hits = payload.get("nbHits")
    hits = payload.get("hits")
    if isinstance(nb_hits, bool) or not isinstance(nb_hits, int) or nb_hits < 0:
        raise ValueError("HN response is missing nbHits")
    if not isinstance(hits, list):
        raise ValueError("HN response is missing hits")
    rows = [hit for hit in hits if isinstance(hit, Mapping)]
    if len(rows) != len(hits):
        raise ValueError("HN hit is not an object")
    return nb_hits, rows


def collect_hn_query(
    query: str,
    start_i: int,
    end_i: int,
    *,
    get_json: _JsonGet,
    throttle_s: float = 0.0,
) -> list[Mapping[str, object]]:
    """Pull every story for one query. More than 1000 hits splits the window."""

    if end_i <= start_i:
        return []
    if throttle_s > 0:
        time.sleep(throttle_s)
    nb_hits, hits = _as_hit_page(get_json(_hn_url(query, start_i, end_i, 0)))
    if nb_hits > HN_HITS_CAP:
        if end_i - start_i <= 1:
            raise ValueError(
                f"HN query {query!r} exceeds {HN_HITS_CAP} hits in one second; "
                "refusing a partial page"
            )
        mid = (start_i + end_i) // 2
        left = collect_hn_query(query, start_i, mid, get_json=get_json, throttle_s=throttle_s)
        right = collect_hn_query(query, mid, end_i, get_json=get_json, throttle_s=throttle_s)
        return left + right
    rows = list(hits)
    page = 1
    while len(rows) < nb_hits:
        if throttle_s > 0:
            time.sleep(throttle_s)
        _count, more = _as_hit_page(get_json(_hn_url(query, start_i, end_i, page)))
        if not more:
            raise ValueError(f"HN query {query!r} ended before nbHits")
        rows.extend(more)
        page += 1
        if page > 50:
            raise ValueError(f"HN query {query!r} pagination did not finish")
    return rows


def query_terms(ticker_map_payload: Mapping[str, object]) -> tuple[str, ...]:
    entries = ticker_map_payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("ticker map has no entries")
    terms: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("ticker map entry is not an object")
        for tag in entry.get("cashtags", []):
            token = str(tag).strip().lstrip("$")
            if token:
                terms.append("$" + token)
        for name in entry.get("names", []):
            phrase = str(name).strip()
            if phrase:
                terms.append(phrase)
    ordered = tuple(dict.fromkeys(terms))
    if not ordered:
        raise ValueError("ticker map produced no HN queries")
    return ordered


def render_events_jsonl(events: Sequence[Mapping[str, object]]) -> bytes:
    lines = [
        json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False)
        for event in events
    ]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def _default_get_json(url: str) -> Mapping[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "FusionFinance-barebone-narrative/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise ValueError("HN rate limit; refusing a partial narrative tape") from None
        raise ValueError(f"HN request failed with HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        raise ValueError("HN request failed") from exc
    if not isinstance(payload, dict):
        raise ValueError("HN response is not an object")
    return payload


def fetch_hn_events(
    *,
    sessions: Sequence[date],
    ticker_map: TickerMap,
    map_payload: Mapping[str, object],
    get_json: _JsonGet | None = None,
    throttle_s: float = 0.25,
) -> list[dict[str, object]]:
    getter = get_json or _default_get_json
    start_i, end_i = _window_bounds()
    hits: list[Mapping[str, object]] = []
    seen_ids: set[str] = set()
    for term in query_terms(map_payload):
        batch = collect_hn_query(term, start_i, end_i, get_json=getter, throttle_s=throttle_s)
        print(f"hn {term}: {len(batch)} hits", file=sys.stderr)
        for hit in batch:
            object_id = str(hit.get("objectID", "")).strip()
            if object_id and object_id in seen_ids:
                continue
            if object_id:
                seen_ids.add(object_id)
            hits.append(hit)
    return events_from_hn_hits(hits, ticker_map=ticker_map, sessions=sessions)


def provenance_document(
    *,
    digest: str,
    events: Sequence[Mapping[str, object]],
    fetched_at: datetime,
    provider: str,
) -> dict[str, object]:
    if provider != HN_PROVIDER:
        raise ValueError("narrative provenance provider must name Hacker News via Algolia")
    tickers = sorted({str(event["ticker"]) for event in events})
    sessions = sorted({str(event["decision_session"]) for event in events})
    return {
        "schema": _PROVENANCE_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "provider": provider,
        "fetched_at": fetched_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window": list(BAREBONE_WINDOW),
        "events": NARRATIVE_EVENTS,
        "byte_sha256": digest,
        "event_count": len(events),
        "ticker_count": len(tickers),
        "session_count": len(sessions),
        "tickers_with_events": tickers,
        "license_note": NARRATIVE_LICENSE_NOTE,
        "disclaimer": (
            "not redistributed; local bind only. Hacker News story text is "
            "not redistributed."
        ),
        "comparable_performance_claim": False,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_locked_narrative_sha256(config_path: Path, digest: str) -> None:
    """Record the events-file digest. This does not invent one or set a claim."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("barebone comparison config must be a JSON object")
    if payload.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("evidence path is required")
    evidence["narrative_sha256"] = digest
    evidence["narrative_events"] = NARRATIVE_EVENTS
    validate_barebone_payload(payload)
    config_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def cash_narrative_record(narrative_sha256: str | None, *, event_count: int = 0) -> dict[str, object]:
    """Cash arm. No frozen polarity model means there are no scores to size."""

    if narrative_sha256 is not None and (
        len(narrative_sha256) != 64 or any(char not in "0123456789abcdef" for char in narrative_sha256)
    ):
        raise ValueError("narrative_sha256 must be a lowercase sha256 hex digest or null")
    document: dict[str, object] = {
        "schema": "fusionfinance-barebone-narrative-arm-v1",
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "claim_status": "controlled_software_ledger",
        "comparable_performance_claim": False,
        "strategy_id": "narrative",
        "sized": False,
        "reason": (
            "no frozen polarity model is bound to text available at available_ts, "
            "so the narrative arm stays in cash"
        ),
        "narrative_sha256": narrative_sha256,
        "narrative_events": NARRATIVE_EVENTS,
        "event_count": event_count,
        "max_position_weight": 0.1,
        "max_gross_leverage": 1.0,
        "tape_sha256": None,
    }
    if _BANNED_FIELDS.intersection(document):
        raise ValueError("narrative arm must not carry a performance field")
    return document


def provider_skip_message(provider: str, env: Mapping[str, str] | None = None) -> str | None:
    """Return a skip message when Reddit or X credentials are absent."""

    source = env if env is not None else os.environ
    if provider == "reddit":
        missing = [name for name in _REDDIT_ENV if not source.get(name)]
        if missing:
            return (
                "reddit provider skipped: "
                + ", ".join(missing)
                + " unset; no events were invented"
            )
        return None
    if provider == "x":
        missing = [name for name in _X_ENV if not source.get(name)]
        if missing:
            return (
                "x provider skipped: "
                + ", ".join(missing)
                + " unset; recent search is not a full-window archive and no events were invented"
            )
        return None
    return None


def ingest_hn_narrative(
    *,
    config_path: Path,
    root: Path | None = None,
    get_json: _JsonGet | None = None,
    throttle_s: float = 0.25,
    lock_config: bool = False,
    sessions: Sequence[date] | None = None,
    now: datetime | None = None,
) -> str:
    """Write the gitignored events file and its provenance. Return the digest."""

    base = root or _repo_root()
    config = load_barebone_comparison_config(config_path)
    if config.comparable_performance_claim is not False:
        raise ValueError("comparable_performance_claim must be false")
    calendar = sessions
    if calendar is None:
        calendar = sessions_from_ohlcv(base / config.evidence.ohlcv)
    map_path = base / NARRATIVE_MAP
    map_payload = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(map_payload, dict):
        raise ValueError("ticker map must be a JSON object")
    ticker_map = load_ticker_map(map_path)
    events = fetch_hn_events(
        sessions=calendar,
        ticker_map=ticker_map,
        map_payload=map_payload,
        get_json=get_json,
        throttle_s=throttle_s,
    )
    if lock_config and not events:
        raise ValueError("refusing to lock an empty narrative tape")
    payload = render_events_jsonl(events)
    digest = hashlib.sha256(payload).hexdigest()
    destination = base / NARRATIVE_EVENTS
    _atomic_write(destination, payload)
    fetched_at = now or datetime.now(timezone.utc)
    sidecar = provenance_document(
        digest=digest,
        events=events,
        fetched_at=fetched_at,
        provider=HN_PROVIDER,
    )
    _atomic_write(
        base / NARRATIVE_PROVENANCE,
        (json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )
    record = cash_narrative_record(digest if lock_config else config.evidence.narrative_sha256, event_count=len(events))
    record["tape_sha256"] = config.evidence.tape_sha256
    _atomic_write(
        base / NARRATIVE_CASH_RECORD,
        (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )
    if lock_config:
        write_locked_narrative_sha256(config_path, digest)
    return digest


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Prints a digest or a skip message. Does not invent events."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--provider", choices=("hn", "reddit", "x"), required=True)
    parser.add_argument("--lock-config", action="store_true")
    parser.add_argument("--throttle-seconds", type=float, default=0.25)
    args = parser.parse_args(argv)
    skip = provider_skip_message(args.provider)
    if skip is not None:
        print(skip)
        return 0
    if args.provider != "hn":
        print(
            f"{args.provider} credentials are present, but this build does not "
            "call that API and does not invent a full-window archive"
        )
        return 0
    config_path = args.config or (_repo_root() / "configs" / "barebone-comparison-v1.json")
    try:
        digest = ingest_hn_narrative(
            config_path=config_path,
            lock_config=args.lock_config,
            throttle_s=args.throttle_seconds,
        )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"sha256={digest}")
    if args.lock_config:
        print(f"locked narrative_sha256={digest} in {config_path}")
    else:
        print("narrative_sha256 remains null; re-run with --lock-config to record this digest")
    return 0
