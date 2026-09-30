"""Point-in-time FINRA consolidated short interest for the Barebone window.

The dataset is ``consolidatedShortInterest`` on ``api.finra.org``. Rows carry
a settlement date and no publication clock. FINRA publishes the compiled
report on the 7th business day after that settlement date. ``available_ts``
is that publication date at 00:00:00Z. The settlement date is never the
availability timestamp. A row with no settlement date aborts the ingest.

The feature, chosen before the arm is run, is the latest period change
ratio in the 63 sessions ending at the decision session:
``(current - previous) / previous``. The log1p level is not used. Names
with no in-window print are omitted. The events file is gitignored.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from demo.barebone_comparison import (
    BAREBONE_EXPERIMENT_ID,
    BAREBONE_UNIVERSE,
    LOCKED_ATTENTION_SHA256,
    LOCKED_EDGAR_SHA256,
    LOCKED_FORM4_SHA256,
    LOCKED_NARRATIVE_SHA256,
    LOCKED_POLARITY_SHA256,
    LOCKED_TAPE_SHA256,
    SHORT_INTEREST_EVENTS,
    SHORT_INTEREST_PROVENANCE,
    SHORT_INTEREST_SYMBOL_MAP,
    load_barebone_comparison_config,
    validate_barebone_payload,
)
from demo.barebone_edgar import demean_cross_section
from demo.barebone_narrative import decision_session_for, sessions_from_ohlcv

FINRA_ENDPOINT = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
FINRA_METADATA = "https://api.finra.org/metadata/group/otcMarket/name/consolidatedShortInterest"
FINRA_SCHEDULE = "https://www.finra.org/filing-reporting/regulatory-filing-systems/short-interest"
FINRA_USER_AGENT = "FusionFinance barebone-comparison-v1 research@theonlyfusioncube.com"
FINRA_SOURCE_NAME = "finra-consolidated-short-interest"
SI_SIGNAL_ID = "finra-si-change-ratio-63s-v1"
SI_LOOKBACK_SESSIONS = 63
SI_SKILL_THRESHOLD = 0.0
SI_TRANSFORM = "latest_change_ratio_minus_cross_sectional_median"
SI_SCORE = (
    "latest FINRA period change ratio, (current short position minus previous) "
    "divided by the previous short position, among publications whose decision "
    "session falls in the 63 sessions ending at the decision session; the "
    "cross-sectional median is removed; the log1p level is not used"
)
SI_PUBLICATION_RULE = (
    "7th weekday after settlementDate that is not an NYSE full-day holiday; "
    "FINRA states the report is provided for publication on the 7th business "
    "day after the reporting settlement date"
)
_MAP_SCHEMA = "fusionfinance-barebone-short-interest-symbol-map-v1"
_PROVENANCE_SCHEMA = "fusionfinance-barebone-short-interest-v1"
_SETTLEMENT_NOT_BEFORE = date(2024, 11, 1)
_SETTLEMENT_NOT_AFTER = date(2026, 1, 31)
_CALENDAR_LAST = date(2027, 6, 30)
_PAGE_LIMIT = 5000
_SYMBOL = re.compile(r"[A-Z0-9]{1,10}")
_BANNED_EVENT_FIELDS = frozenset(
    {
        "total_return",
        "forward_return",
        "next_return",
        "residual",
        "label",
        "sharpe_ratio",
        "sortino_ratio",
        "annualized_return",
        "html",
        "text",
        "accountingYearMonthNumber",
        "changePercent",
    }
)
# NYSE full-day holidays. Early closes stay business days. January 9, 2025
# is not listed: the December 31, 2024 cycle's 7th weekday is January 10, 2025,
# the date Nasdaq published that cycle.
_HOLIDAYS = frozenset(
    {
        date(2024, 1, 1),
        date(2024, 1, 15),
        date(2024, 2, 19),
        date(2024, 3, 29),
        date(2024, 5, 27),
        date(2024, 6, 19),
        date(2024, 7, 4),
        date(2024, 9, 2),
        date(2024, 11, 28),
        date(2024, 12, 25),
        date(2025, 1, 1),
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 4, 18),
        date(2025, 5, 26),
        date(2025, 6, 19),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 11, 27),
        date(2025, 12, 25),
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
        date(2027, 1, 1),
        date(2027, 1, 18),
        date(2027, 2, 15),
        date(2027, 3, 26),
        date(2027, 5, 31),
        date(2027, 6, 18),
    }
)
# Publication dates on FINRA's Short Interest Reporting page as fetched
# 2026-09-30. The 2025 heading on that page lists only the November and
# December cycles. Years are taken from the weekday names on the page.
OFFICIAL_PUBLICATION_DATES = (
    (date(2025, 11, 14), date(2025, 11, 25)),
    (date(2025, 11, 28), date(2025, 12, 9)),
    (date(2025, 12, 15), date(2025, 12, 24)),
    (date(2025, 12, 31), date(2026, 1, 12)),
    (date(2026, 1, 15), date(2026, 1, 27)),
    (date(2026, 1, 30), date(2026, 2, 10)),
    (date(2026, 2, 13), date(2026, 2, 25)),
    (date(2026, 2, 27), date(2026, 3, 10)),
    (date(2026, 3, 13), date(2026, 3, 24)),
    (date(2026, 3, 31), date(2026, 4, 10)),
    (date(2026, 4, 15), date(2026, 4, 24)),
    (date(2026, 4, 30), date(2026, 5, 11)),
    (date(2026, 5, 15), date(2026, 5, 27)),
    (date(2026, 5, 29), date(2026, 6, 9)),
    (date(2026, 6, 15), date(2026, 6, 25)),
    (date(2026, 6, 30), date(2026, 7, 10)),
    (date(2026, 7, 15), date(2026, 7, 24)),
    (date(2026, 7, 31), date(2026, 8, 11)),
    (date(2026, 8, 14), date(2026, 8, 25)),
    (date(2026, 8, 31), date(2026, 9, 10)),
    (date(2026, 9, 15), date(2026, 9, 24)),
    (date(2026, 9, 30), date(2026, 10, 9)),
    (date(2026, 10, 15), date(2026, 10, 26)),
    (date(2026, 10, 30), date(2026, 11, 10)),
    (date(2026, 11, 13), date(2026, 11, 24)),
    (date(2026, 11, 30), date(2026, 12, 9)),
    (date(2026, 12, 15), date(2026, 12, 24)),
    (date(2026, 12, 31), date(2027, 1, 12)),
)
_RowGet = Callable[[str], list[object]]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def publication_date(settlement: date) -> date:
    """Seventh business day after ``settlement``. The settlement date is not returned."""

    if not isinstance(settlement, date):
        raise ValueError("settlement date is missing; refusing to invent a publication time")
    found = 0
    cursor = settlement
    while found < 7:
        cursor += timedelta(days=1)
        if cursor > _CALENDAR_LAST:
            raise ValueError("publication calendar does not cover this settlement")
        if cursor.weekday() >= 5 or cursor in _HOLIDAYS:
            continue
        found += 1
    if cursor <= settlement:
        raise ValueError("publication date collapsed onto the settlement date")
    return cursor


def assert_official_publication_calendar() -> None:
    """Refuse the ingest if the business-day rule misses FINRA's published table."""

    for settlement, published in OFFICIAL_PUBLICATION_DATES:
        got = publication_date(settlement)
        if got != published or got == settlement:
            raise ValueError(
                f"publication rule missed FINRA's table for {settlement.isoformat()}"
            )


def available_stamp(publication: date) -> str:
    """Publication date at 00:00:00Z. FINRA does not provide an intraday clock."""

    if not isinstance(publication, date):
        raise ValueError("publication date is missing")
    return publication.strftime("%Y-%m-%dT00:00:00Z")


def load_symbol_map(path: Path | None = None) -> tuple[dict[str, str], str]:
    """Load the frozen ticker map. An unmapped universe name raises."""

    map_path = path or (_repo_root() / SHORT_INTEREST_SYMBOL_MAP)
    payload_bytes = map_path.read_bytes()
    payload = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("short interest symbol map must be a JSON object")
    if payload.get("schema") != _MAP_SCHEMA:
        raise ValueError("short interest symbol map schema is missing or changed")
    if payload.get("experiment_id") != BAREBONE_EXPERIMENT_ID:
        raise ValueError("short interest symbol map experiment_id must be barebone-comparison-v1")
    if payload.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("short interest symbol map has no entries")
    mapping: dict[str, str] = {}
    seen_symbol: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("short interest symbol map entry must be an object")
        ticker = str(entry.get("ticker", "")).strip().upper()
        symbol = str(entry.get("symbol_code", "")).strip().upper()
        if ticker not in BAREBONE_UNIVERSE:
            raise ValueError(f"short interest symbol {ticker or '(blank)'} is outside the universe")
        if ticker in mapping:
            raise ValueError(f"short interest symbol map repeats {ticker}")
        if _SYMBOL.fullmatch(symbol) is None:
            raise ValueError(f"FINRA symbol for {ticker} is not an official symbolCode")
        if symbol in seen_symbol:
            raise ValueError(f"FINRA symbol {symbol} is shared by {seen_symbol[symbol]} and {ticker}")
        mapping[ticker] = symbol
        seen_symbol[symbol] = ticker
    missing = [ticker for ticker in BAREBONE_UNIVERSE if ticker not in mapping]
    if missing:
        raise ValueError("short interest symbol map is missing " + ", ".join(missing))
    return mapping, hashlib.sha256(payload_bytes).hexdigest()


def _whole(value: object, label: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"FINRA {label} is missing")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise ValueError(f"FINRA {label} is not an integer share count")


def _settlement_date(record: Mapping[str, object]) -> date:
    value = record.get("settlementDate")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("FINRA row is missing settlementDate; refusing to invent a publication time")
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError("FINRA settlementDate is not a date") from exc


def events_from_records(
    records: Sequence[object],
    *,
    ticker: str,
    symbol: str,
    sessions: Sequence[date],
) -> tuple[list[dict[str, object]], int]:
    """Keep in-window prints. Revised rows are dropped. Settlement is not ``available_ts``."""

    if ticker not in BAREBONE_UNIVERSE:
        raise ValueError(f"short interest ticker {ticker} is outside the universe")
    rows: list[dict[str, object]] = []
    revised = 0
    seen_settlement: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"FINRA row for {ticker} is not an object")
        reported = str(record.get("symbolCode", "")).strip().upper()
        if reported != symbol:
            raise ValueError(f"FINRA symbol {reported or '(blank)'} is not the frozen map for {ticker}")
        settlement = _settlement_date(record)
        if settlement < _SETTLEMENT_NOT_BEFORE or settlement > _SETTLEMENT_NOT_AFTER:
            continue
        if "revisionFlag" not in record:
            raise ValueError(f"{ticker} {settlement.isoformat()} is missing revisionFlag")
        flag = record.get("revisionFlag")
        if flag == "R":
            revised += 1
            continue
        if flag not in (None,):
            raise ValueError(f"{ticker} {settlement.isoformat()} has an unknown revisionFlag")
        published = publication_date(settlement)
        if published == settlement:
            raise ValueError("publication date collapsed onto the settlement date")
        stamp = available_stamp(published)
        session = decision_session_for(datetime.fromisoformat(stamp.replace("Z", "+00:00")), sessions)
        if session is None:
            continue
        if session <= published:
            raise ValueError("same-session short interest print is refused")
        current = _whole(record.get("currentShortPositionQuantity"), "currentShortPositionQuantity")
        previous = _whole(record.get("previousShortPositionQuantity"), "previousShortPositionQuantity")
        change = _whole(record.get("changePreviousNumber"), "changePreviousNumber")
        if previous <= 0:
            raise ValueError(f"{ticker} {settlement.isoformat()} has no previous short position")
        if current - previous != change:
            raise ValueError(f"{ticker} {settlement.isoformat()} change does not match the quantities")
        key = settlement.isoformat()
        if key in seen_settlement:
            raise ValueError(f"{ticker} repeats settlement {key}")
        seen_settlement.add(key)
        ratio = (current - previous) / previous
        market = record.get("marketClassCode")
        if not isinstance(market, str) or not market.strip():
            raise ValueError(f"{ticker} {key} is missing marketClassCode")
        split = record.get("stockSplitFlag")
        if split not in (None, "S"):
            raise ValueError(f"{ticker} {key} has an unknown stockSplitFlag")
        identity = f"{symbol}|{key}|{published.isoformat()}|{current}|{previous}|{change}"
        rows.append(
            {
                "event_id": f"si:{key}:{ticker}",
                "ticker": ticker,
                "symbol_code": symbol,
                "settlement_date": key,
                "publication_date": published.isoformat(),
                "available_ts": stamp,
                "decision_session": session.isoformat(),
                "current_short_position": current,
                "previous_short_position": previous,
                "change_previous_number": change,
                "change_ratio": ratio,
                "market_class_code": market.strip(),
                "revision_flag": None,
                "stock_split_flag": split,
                "source_class": "other",
                "source_name": FINRA_SOURCE_NAME,
                "text_hash": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            }
        )
    return rows, revised


def _request(payload: dict[str, object]) -> tuple[list[object], int]:
    if not FINRA_ENDPOINT.startswith("https://api.finra.org/data/group/otcMarket/name/"):
        raise ValueError("short interest fetch refuses a non-FINRA endpoint")
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        FINRA_ENDPOINT,
        data=body,
        headers={
            "User-Agent": FINRA_USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total_header = response.headers.get("record-total")
            raw = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise ValueError("FINRA rate limit; refusing a partial short interest tape") from exc
        raise ValueError(f"FINRA short interest request failed: {exc.code}") from exc
    if total_header is None or not str(total_header).isdigit():
        raise ValueError("FINRA response is missing record-total")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("FINRA short interest response is not JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError("FINRA short interest response is not a list")
    return parsed, int(total_header)


def _default_get_rows(symbol: str) -> list[object]:
    if _SYMBOL.fullmatch(symbol) is None:
        raise ValueError("FINRA symbol is not an official symbolCode")
    rows: list[object] = []
    offset = 0
    total: int | None = None
    for _page in range(5):
        time.sleep(0.2)
        page, reported = _request(
            {
                "compareFilters": [
                    {"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": symbol}
                ],
                "limit": _PAGE_LIMIT,
                "offset": offset,
            }
        )
        if total is None:
            total = reported
        elif reported != total:
            raise ValueError(f"FINRA record-total changed while reading {symbol}")
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        if offset >= total:
            break
    else:
        raise ValueError("FINRA short interest paging did not finish")
    if total is None or len(rows) != total:
        raise ValueError(f"FINRA page did not return the full result for {symbol}")
    return rows


def render_events_jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) for row in rows]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def short_interest_score_rows(
    events: Sequence[Mapping[str, object]],
    sessions: Sequence[date],
    universe: Sequence[str] | None = None,
    *,
    lookback: int = SI_LOOKBACK_SESSIONS,
) -> list[dict[str, object]]:
    """Latest in-window change ratio, then the cross-sectional median.

    A publication whose decision session is later is not a feature. Names
    with no print in the window are omitted rather than filled with zero.
    """

    if lookback != SI_LOOKBACK_SESSIONS:
        raise ValueError("short interest lookback is locked at 63 sessions")
    names = tuple(universe) if universe is not None else BAREBONE_UNIVERSE
    if len(names) < 2:
        raise ValueError("short interest cross-section needs at least two names")
    ordered = tuple(sessions)
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if later <= earlier:
            raise ValueError("short interest calendar must be strictly increasing")
    index = {day: offset for offset, day in enumerate(ordered)}
    prints: dict[str, list[tuple[int, float]]] = {ticker: [] for ticker in names}
    seen: set[str] = set()
    allowed = set(names)
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("short interest event must be an object")
        banned = _BANNED_EVENT_FIELDS.intersection(event)
        if banned:
            raise ValueError("short interest event carries a forbidden field: " + ", ".join(sorted(banned)))
        ticker = str(event.get("ticker", "")).strip().upper()
        if ticker not in allowed:
            raise ValueError(f"short interest ticker {ticker or '(blank)'} is not in the universe")
        settlement_text = str(event.get("settlement_date", "")).strip()
        publication_text = str(event.get("publication_date", "")).strip()
        available_text = str(event.get("available_ts", "")).strip()
        if not settlement_text or not publication_text or not available_text:
            raise ValueError("short interest event is missing a publication timestamp")
        settlement = date.fromisoformat(settlement_text)
        published = date.fromisoformat(publication_text)
        if publication_date(settlement) != published or published == settlement:
            raise ValueError("available publication date is not the 7th business day")
        if available_text != available_stamp(published):
            raise ValueError("available_ts must be the publication date, not the settlement date")
        session_text = event.get("decision_session")
        if not isinstance(session_text, str) or not session_text:
            raise ValueError("short interest event is missing decision_session")
        session = date.fromisoformat(session_text)
        if session not in index:
            raise ValueError("short interest decision_session is not on the tape")
        if session <= published:
            raise ValueError("same-session short interest print is refused")
        expected = decision_session_for(
            datetime.fromisoformat(available_text.replace("Z", "+00:00")), ordered
        )
        if expected is None or session != expected:
            raise ValueError("decision_session must be the first session strictly after available_ts")
        current = _whole(event.get("current_short_position"), "current_short_position")
        previous = _whole(event.get("previous_short_position"), "previous_short_position")
        change = _whole(event.get("change_previous_number"), "change_previous_number")
        if previous <= 0 or current - previous != change:
            raise ValueError("short interest change ratio does not match the quantities")
        ratio = (current - previous) / previous
        try:
            stored = float(event["change_ratio"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("short interest event is missing change_ratio") from exc
        if abs(stored - ratio) > 1e-12:
            raise ValueError("short interest change_ratio does not match the quantities")
        event_id = str(event.get("event_id", "")).strip()
        if not event_id or event_id in seen:
            raise ValueError("short interest event_id is missing or repeated")
        seen.add(event_id)
        prints[ticker].append((index[session], ratio))
    rows: list[dict[str, object]] = []
    for session_index in range(lookback - 1, len(ordered)):
        start = session_index - lookback + 1
        raw: dict[str, float] = {}
        for ticker, series in prints.items():
            chosen = [item for item in series if start <= item[0] <= session_index]
            if not chosen:
                continue
            raw[ticker] = max(chosen, key=lambda item: item[0])[1]
        if len(raw) < 2:
            continue
        scores = demean_cross_section(raw)
        day = ordered[session_index]
        for ticker, ratio in raw.items():
            rows.append(
                {
                    "session": day.isoformat(),
                    "session_index": session_index,
                    "ticker": ticker,
                    "change_ratio": ratio,
                    "score": scores[ticker],
                    "signal_id": SI_SIGNAL_ID,
                    "lookback_sessions": lookback,
                }
            )
    return rows


def short_interest_provenance(
    *,
    events_sha256: str,
    symbol_map_sha256: str,
    tape_sha256: str,
    event_count: int,
    revised_dropped: int,
    fetched_at: datetime,
) -> dict[str, object]:
    if tape_sha256 != LOCKED_TAPE_SHA256:
        raise ValueError("short interest provenance refuses a changed tape_sha256")
    return {
        "schema": _PROVENANCE_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "provider": "FINRA consolidated short interest via api.finra.org",
        "endpoint": FINRA_ENDPOINT,
        "metadata_endpoint": FINRA_METADATA,
        "schedule_endpoint": FINRA_SCHEDULE,
        "user_agent": FINRA_USER_AGENT,
        "headers": ["User-Agent", "Accept: application/json", "Content-Type: application/json"],
        "score": SI_SCORE,
        "signal_id": SI_SIGNAL_ID,
        "lookback_sessions": SI_LOOKBACK_SESSIONS,
        "available_ts": "publication date at 00:00:00Z",
        "publication_rule": SI_PUBLICATION_RULE,
        "settlement_used_as_available_ts": False,
        "official_schedule_rows": len(OFFICIAL_PUBLICATION_DATES),
        "january_9_2025_counted_as_holiday": False,
        "symbol_map": SHORT_INTEREST_SYMBOL_MAP,
        "symbol_map_sha256": symbol_map_sha256,
        "events": SHORT_INTEREST_EVENTS,
        "short_interest_sha256": events_sha256,
        "tape_sha256": tape_sha256,
        "narrative_sha256": LOCKED_NARRATIVE_SHA256,
        "edgar_sha256": LOCKED_EDGAR_SHA256,
        "edgar_form4_sha256": LOCKED_FORM4_SHA256,
        "event_count": event_count,
        "revised_rows_dropped": revised_dropped,
        "revision_rule": "revisionFlag R is dropped because the revision clock is not in the file",
        "fetched_at": fetched_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "license_note": "not redistributed; local bind only; FINRA query rows are not committed",
        "comparable_performance_claim": False,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_locked_short_interest_sha256(config_path: Path, digest: str) -> None:
    """Record the short-interest digest without touching the earlier evidence locks."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("barebone comparison config must be a JSON object")
    if payload.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("evidence path is required")
    if evidence.get("tape_sha256") != LOCKED_TAPE_SHA256:
        raise ValueError("refusing to lock short interest against a changed tape_sha256")
    if evidence.get("narrative_sha256") != LOCKED_NARRATIVE_SHA256:
        raise ValueError("refusing to lock short interest against a changed narrative_sha256")
    if evidence.get("polarity_sha256") != LOCKED_POLARITY_SHA256:
        raise ValueError("refusing to lock short interest against a changed polarity_sha256")
    if evidence.get("attention_sha256") != LOCKED_ATTENTION_SHA256:
        raise ValueError("refusing to lock short interest against a changed attention_sha256")
    if evidence.get("edgar_sha256") != LOCKED_EDGAR_SHA256:
        raise ValueError("refusing to lock short interest against a changed edgar_sha256")
    if evidence.get("edgar_form4_sha256") != LOCKED_FORM4_SHA256:
        raise ValueError("refusing to lock short interest against a changed edgar_form4_sha256")
    evidence["short_interest_events"] = SHORT_INTEREST_EVENTS
    evidence["short_interest_sha256"] = digest
    validate_barebone_payload(payload)
    config_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fetch_short_interest_events(
    *,
    config_path: Path,
    root: Path | None = None,
    lock_config: bool = False,
    sessions: Sequence[date] | None = None,
    get_rows: _RowGet | None = None,
    now: datetime | None = None,
) -> str:
    """Download consolidated short interest and return the events digest."""

    assert_official_publication_calendar()
    base = root or _repo_root()
    config = load_barebone_comparison_config(config_path)
    if config.comparable_performance_claim is not False:
        raise ValueError("comparable_performance_claim must be false")
    if config.evidence.tape_sha256 != LOCKED_TAPE_SHA256:
        raise ValueError("tape_sha256 is locked; refusing a new digest without a new experiment_id")
    if config.evidence.narrative_sha256 != LOCKED_NARRATIVE_SHA256:
        raise ValueError("narrative_sha256 is locked; refusing a new digest without a new experiment_id")
    if config.evidence.edgar_sha256 != LOCKED_EDGAR_SHA256:
        raise ValueError("edgar_sha256 is locked; refusing a new digest without a new experiment_id")
    if config.evidence.edgar_form4_sha256 != LOCKED_FORM4_SHA256:
        raise ValueError("edgar_form4_sha256 is locked; refusing a new digest without a new experiment_id")
    mapping, map_sha = load_symbol_map(base / SHORT_INTEREST_SYMBOL_MAP)
    calendar = sessions if sessions is not None else sessions_from_ohlcv(base / config.evidence.ohlcv)
    reader = get_rows or _default_get_rows
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    revised = 0
    for ticker in BAREBONE_UNIVERSE:
        symbol = mapping[ticker]
        payload = reader(symbol)
        if not payload:
            raise ValueError(f"FINRA returned no short interest for {ticker}")
        kept, dropped = events_from_records(payload, ticker=ticker, symbol=symbol, sessions=calendar)
        revised += dropped
        if reader is _default_get_rows:
            print(
                f"short-interest {ticker} kept={len(kept)} revised={dropped}",
                file=sys.stderr,
            )
        if not kept:
            raise ValueError(f"FINRA short interest for {ticker} has no publication inside the window")
        for row in kept:
            event_id = str(row["event_id"])
            if event_id in seen:
                raise ValueError(f"short interest event_id {event_id} is repeated")
            seen.add(event_id)
            rows.append(row)
    if not rows:
        raise ValueError("refusing to lock an empty short interest tape")
    rows.sort(key=lambda row: (str(row["available_ts"]), str(row["event_id"])))
    rendered = render_events_jsonl(rows)
    digest = hashlib.sha256(rendered).hexdigest()
    _atomic_write(base / SHORT_INTEREST_EVENTS, rendered)
    sidecar = short_interest_provenance(
        events_sha256=digest,
        symbol_map_sha256=map_sha,
        tape_sha256=LOCKED_TAPE_SHA256,
        event_count=len(rows),
        revised_dropped=revised,
        fetched_at=now or datetime.now(timezone.utc),
    )
    _atomic_write(
        base / SHORT_INTEREST_PROVENANCE,
        (json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )
    if lock_config:
        write_locked_short_interest_sha256(config_path, digest)
    return digest


def main(argv: list[str] | None = None) -> int:
    """Fetch FINRA consolidated short interest. Does not commit the rows."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--lock-config", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config or (_repo_root() / "configs" / "barebone-comparison-v1.json")
    try:
        digest = fetch_short_interest_events(config_path=config_path, lock_config=args.lock_config)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"short_interest_sha256={digest}")
    print(f"signal_id={SI_SIGNAL_ID}")
    if args.lock_config:
        print(f"locked short_interest_sha256={digest} in {config_path}")
    else:
        print("short_interest_sha256 remains unset; re-run with --lock-config to record this digest")
    return 0
