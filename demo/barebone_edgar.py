"""Point-in-time SEC EDGAR filings for the locked Barebone window.

Filings come from ``data.sec.gov`` submissions JSON. ``available_ts`` is the
filing ``acceptanceDateTime`` in UTC. The decision session is the first tape
session strictly after that UTC date. ``reportDate`` is not an availability
timestamp. Amendments (``/A``) and every form outside 8-K, 10-Q, and 10-K are
excluded. The events file is gitignored. ``tape_sha256`` and the Hacker News
``narrative_sha256`` stay the locked binds.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path

from demo.barebone_comparison import (
    BAREBONE_EXPERIMENT_ID,
    BAREBONE_UNIVERSE,
    EDGAR_CIK_MAP,
    EDGAR_EVENTS,
    EDGAR_PROVENANCE,
    LOCKED_NARRATIVE_SHA256,
    LOCKED_POLARITY_SHA256,
    LOCKED_TAPE_SHA256,
    load_barebone_comparison_config,
    validate_barebone_payload,
)
from demo.barebone_narrative import _aware_utc, decision_session_for, sessions_from_ohlcv

EDGAR_SIGNAL_ID = "edgar-filings-log1p-63s-v1"
EDGAR_LOOKBACK_SESSIONS = 63
EDGAR_TRANSFORM = "log1p_count_minus_cross_sectional_median"
EDGAR_SKILL_THRESHOLD = 0.0
EDGAR_USER_AGENT = "FusionFinance barebone-comparison-v1 research@theonlyfusioncube.com"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/"
EDGAR_EXACT_FORMS = frozenset({"8-K", "10-Q", "10-K"})
EDGAR_ACCEPTANCE_NOT_BEFORE = date(2024, 12, 30)
EDGAR_SOURCE_NAME = "sec-edgar"
_MAP_SCHEMA = "fusionfinance-barebone-edgar-cik-map-v1"
_PROVENANCE_SCHEMA = "fusionfinance-barebone-edgar-v1"
_CIK = re.compile(r"[0-9]{10}")
_JsonGet = Callable[[str], Mapping[str, object]]
_BANNED_EVENT_FIELDS = frozenset(
    {
        "reportDate",
        "filingDate",
        "html",
        "text",
        "primaryDocument",
        "total_return",
        "forward_return",
        "next_return",
        "residual",
        "label",
        "sharpe_ratio",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def demean_cross_section(values: Mapping[str, float]) -> dict[str, float]:
    """Subtract the cross-sectional median. Fewer than two names is empty."""

    if len(values) < 2:
        return {}
    ordered = sorted(float(value) for value in values.values())
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return {ticker: float(value) - median for ticker, value in values.items()}


def load_cik_map(path: Path | None = None) -> tuple[dict[str, str], str]:
    """Load the frozen ticker map. An unmapped universe name raises."""

    map_path = path or (_repo_root() / EDGAR_CIK_MAP)
    payload_bytes = map_path.read_bytes()
    payload = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("edgar cik map must be a JSON object")
    if payload.get("schema") != _MAP_SCHEMA:
        raise ValueError("edgar cik map schema is missing or changed")
    if payload.get("experiment_id") != BAREBONE_EXPERIMENT_ID:
        raise ValueError("edgar cik map experiment_id must be barebone-comparison-v1")
    if payload.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("edgar cik map has no entries")
    mapping: dict[str, str] = {}
    seen_cik: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("edgar cik map entry must be an object")
        ticker = str(entry.get("ticker", "")).strip().upper()
        cik = str(entry.get("cik", "")).strip()
        if ticker not in BAREBONE_UNIVERSE:
            raise ValueError(f"edgar cik map ticker {ticker or '(blank)'} is outside the universe")
        if ticker in mapping:
            raise ValueError(f"edgar cik map repeats {ticker}")
        if _CIK.fullmatch(cik) is None:
            raise ValueError(f"edgar cik for {ticker} must be 10 digits")
        if cik in seen_cik:
            raise ValueError(f"edgar cik {cik} is shared by {seen_cik[cik]} and {ticker}")
        mapping[ticker] = cik
        seen_cik[cik] = ticker
    missing = [ticker for ticker in BAREBONE_UNIVERSE if ticker not in mapping]
    if missing:
        raise ValueError("edgar cik map is missing " + ", ".join(missing))
    return mapping, hashlib.sha256(payload_bytes).hexdigest()


def _normalize_acceptance(value: object, accession: str) -> datetime:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"filing {accession or '(blank)'} is missing acceptanceDateTime")
    if not isinstance(value, str):
        raise ValueError(f"filing {accession or '(blank)'} acceptanceDateTime is not a timestamp")
    return _aware_utc(value)


def events_from_columns(
    columns: Mapping[str, object],
    *,
    ticker: str,
    cik: str,
    sessions: Sequence[date],
    acceptance_not_before: date = EDGAR_ACCEPTANCE_NOT_BEFORE,
) -> list[dict[str, object]]:
    """Keep exact 8-K, 10-Q, and 10-K rows. ``/A`` and other forms are dropped.

    ``available_ts`` is ``acceptanceDateTime``. ``reportDate`` is ignored.
    A kept form with no acceptance timestamp aborts the parse.
    """

    if "acceptanceDateTime" not in columns:
        raise ValueError("submissions are missing acceptanceDateTime")
    if "form" not in columns or "accessionNumber" not in columns:
        raise ValueError("submissions are missing form or accessionNumber")
    forms = columns["form"]
    accessions = columns["accessionNumber"]
    acceptances = columns["acceptanceDateTime"]
    if not isinstance(forms, list) or not isinstance(accessions, list) or not isinstance(acceptances, list):
        raise ValueError("submissions columns must be lists")
    if not (len(forms) == len(accessions) == len(acceptances)):
        raise ValueError("submissions columns have different lengths")
    rows: list[dict[str, object]] = []
    for form_value, accession_value, accepted in zip(forms, accessions, acceptances, strict=True):
        form = str(form_value).strip()
        if form not in EDGAR_EXACT_FORMS:
            continue
        accession = str(accession_value).strip()
        available = _normalize_acceptance(accepted, accession)
        if available.date() < acceptance_not_before:
            continue
        session = decision_session_for(available, sessions)
        if session is None:
            continue
        if session <= available.date():
            raise ValueError("same-session filing is refused")
        accession_key = accession.replace("-", "")
        if not accession or not accession_key.isdigit():
            raise ValueError(f"filing accession {accession or '(blank)'} is not usable")
        uri = (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accession_key}/{accession}-index.html"
        )
        identity = f"{accession}|{form}|{available.strftime('%Y-%m-%dT%H:%M:%SZ')}|{cik}|{ticker}"
        rows.append(
            {
                "event_id": f"edgar:{accession}:{ticker}",
                "ticker": ticker,
                "cik": cik,
                "accession": accession,
                "form": form,
                "acceptanceDateTime": available.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "available_ts": available.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "decision_session": session.isoformat(),
                "source_class": "other",
                "source_name": EDGAR_SOURCE_NAME,
                "uri": uri,
                "text_hash": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            }
        )
    return rows


def _oldest_acceptance(columns: Mapping[str, object]) -> date:
    acceptances = columns.get("acceptanceDateTime")
    if not isinstance(acceptances, list) or not acceptances:
        raise ValueError("submissions recent has no acceptanceDateTime values")
    found: list[date] = []
    for value in acceptances:
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if not isinstance(value, str):
            raise ValueError("acceptanceDateTime is not a timestamp")
        found.append(_aware_utc(value).date())
    if not found:
        raise ValueError("submissions recent has no acceptanceDateTime values")
    return min(found)


def _overflow_names(files: object, *, oldest: date) -> tuple[str, ...]:
    if oldest <= EDGAR_ACCEPTANCE_NOT_BEFORE:
        return ()
    if not isinstance(files, list) or not files:
        raise ValueError(
            "submissions do not reach the acceptance floor and have no overflow files"
        )
    names: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("submissions overflow entry must be an object")
        name = str(item.get("name", "")).strip()
        if not name.startswith("CIK") or name.count("/") or ".." in name or not name.endswith(".json"):
            raise ValueError(f"submissions overflow name {name or '(blank)'} is refused")
        names.append(name)
    return tuple(names)


def events_for_issuer(
    payload: Mapping[str, object],
    *,
    ticker: str,
    cik: str,
    sessions: Sequence[date],
    get_json: _JsonGet | None = None,
) -> list[dict[str, object]]:
    """Parse one submissions document and any overflow files the window still needs."""

    filings = payload.get("filings")
    if not isinstance(filings, dict):
        raise ValueError(f"submissions for {ticker} are missing filings")
    recent = filings.get("recent")
    if not isinstance(recent, dict):
        raise ValueError(f"submissions for {ticker} are missing recent filings")
    rows = events_from_columns(recent, ticker=ticker, cik=cik, sessions=sessions)
    for name in _overflow_names(filings.get("files"), oldest=_oldest_acceptance(recent)):
        if get_json is None:
            raise ValueError(f"submissions for {ticker} need overflow {name}")
        extra = get_json(EDGAR_SUBMISSIONS + name)
        if not isinstance(extra, dict):
            raise ValueError(f"overflow {name} is not an object")
        rows.extend(events_from_columns(extra, ticker=ticker, cik=cik, sessions=sessions))
    return rows


def _default_get_json(url: str) -> Mapping[str, object]:
    if not url.startswith(EDGAR_SUBMISSIONS):
        raise ValueError("edgar fetch refuses a non-data.sec.gov submissions URL")
    time.sleep(0.2)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": EDGAR_USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise ValueError("SEC rate limit; refusing a partial EDGAR tape") from exc
        raise ValueError(f"SEC submissions request failed: {exc.code}") from exc
    if not isinstance(payload, dict):
        raise ValueError("SEC submissions response is not an object")
    return payload


def render_events_jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) for row in rows]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def edgar_score_rows(
    events: Sequence[Mapping[str, object]],
    sessions: Sequence[date],
    universe: Sequence[str] | None = None,
    *,
    lookback: int = EDGAR_LOOKBACK_SESSIONS,
) -> list[dict[str, object]]:
    """Count filings in ``[j - lookback + 1, j]`` and demean ``log1p``.

    The last session in the window is the decision session. A filing whose
    decision session is later is not a feature. Zero counts stay in the
    cross-section.
    """

    if lookback != EDGAR_LOOKBACK_SESSIONS:
        raise ValueError("edgar lookback is locked at 63 sessions")
    names = tuple(universe) if universe is not None else BAREBONE_UNIVERSE
    if len(names) < 2:
        raise ValueError("edgar cross-section needs at least two names")
    ordered = tuple(sessions)
    for earlier, later in zip(ordered, ordered[1:]):
        if later <= earlier:
            raise ValueError("edgar calendar must be strictly increasing")
    index = {day: offset for offset, day in enumerate(ordered)}
    counts = {ticker: [0] * len(ordered) for ticker in names}
    seen: set[str] = set()
    allowed = set(names)
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("edgar event must be an object")
        banned = _BANNED_EVENT_FIELDS.intersection(event)
        if banned:
            raise ValueError("edgar event carries a forbidden field: " + ", ".join(sorted(banned)))
        if event.get("available_ts") in (None, ""):
            raise ValueError("edgar event is missing available_ts")
        available = _aware_utc(str(event["available_ts"]))
        acceptance = event.get("acceptanceDateTime")
        if acceptance not in (None, ""):
            if _normalize_acceptance(acceptance, str(event.get("accession", ""))).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ) != available.strftime("%Y-%m-%dT%H:%M:%SZ"):
                raise ValueError("available_ts must equal acceptanceDateTime")
        ticker = str(event.get("ticker", "")).strip().upper()
        if ticker not in allowed:
            raise ValueError(f"edgar event ticker {ticker or '(blank)'} is not in the edgar universe")
        form = str(event.get("form", "")).strip()
        if form not in EDGAR_EXACT_FORMS:
            raise ValueError(f"edgar event form {form or '(blank)'} is not an exact 8-K, 10-Q, or 10-K")
        session_text = event.get("decision_session")
        if not isinstance(session_text, str) or not session_text:
            raise ValueError("edgar event is missing decision_session")
        session = date.fromisoformat(session_text)
        if session <= available.date():
            raise ValueError("same-session filing is refused")
        expected = decision_session_for(available, ordered)
        if expected is None or session != expected:
            raise ValueError("decision_session must be the first session strictly after available_ts")
        event_id = str(event.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("edgar event is missing event_id")
        if event_id in seen:
            raise ValueError(f"edgar event_id {event_id} is repeated")
        seen.add(event_id)
        counts[ticker][index[session]] += 1
    prefix = {ticker: [0] for ticker in names}
    for ticker in names:
        running = 0
        for count in counts[ticker]:
            running += count
            prefix[ticker].append(running)
    rows: list[dict[str, object]] = []
    for session_index in range(lookback - 1, len(ordered)):
        start = session_index - lookback + 1
        raw = {
            ticker: math.log1p(prefix[ticker][session_index + 1] - prefix[ticker][start])
            for ticker in names
        }
        scores = demean_cross_section(raw)
        day = ordered[session_index]
        for ticker in names:
            event_count = prefix[ticker][session_index + 1] - prefix[ticker][start]
            rows.append(
                {
                    "session": day.isoformat(),
                    "session_index": session_index,
                    "ticker": ticker,
                    "event_count": event_count,
                    "log1p_count": raw[ticker],
                    "score": scores[ticker],
                    "signal_id": EDGAR_SIGNAL_ID,
                    "lookback_sessions": lookback,
                }
            )
    return rows


def edgar_provenance(
    *,
    events_sha256: str,
    cik_map_sha256: str,
    tape_sha256: str,
    event_count: int,
    form_counts: Mapping[str, int],
    fetched_at: datetime,
) -> dict[str, object]:
    if tape_sha256 != LOCKED_TAPE_SHA256:
        raise ValueError("edgar provenance refuses a changed tape_sha256")
    return {
        "schema": _PROVENANCE_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "provider": "SEC EDGAR via data.sec.gov",
        "endpoint": EDGAR_SUBMISSIONS + "CIK##########.json",
        "user_agent": EDGAR_USER_AGENT,
        "forms": sorted(EDGAR_EXACT_FORMS),
        "amendments": "excluded; form must be exactly 8-K, 10-Q, or 10-K",
        "available_ts": "acceptanceDateTime converted to UTC",
        "report_date_used_as_available_ts": False,
        "acceptance_not_before": EDGAR_ACCEPTANCE_NOT_BEFORE.isoformat(),
        "signal_id": EDGAR_SIGNAL_ID,
        "lookback_sessions": EDGAR_LOOKBACK_SESSIONS,
        "fetched_at": fetched_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "events": EDGAR_EVENTS,
        "edgar_sha256": events_sha256,
        "cik_map": EDGAR_CIK_MAP,
        "cik_map_sha256": cik_map_sha256,
        "tape_sha256": tape_sha256,
        "narrative_sha256": LOCKED_NARRATIVE_SHA256,
        "event_count": event_count,
        "form_counts": {form: int(form_counts.get(form, 0)) for form in sorted(EDGAR_EXACT_FORMS)},
        "license_note": "not redistributed; local bind only; metadata only, no filing HTML",
        "comparable_performance_claim": False,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_locked_edgar_sha256(config_path: Path, digest: str) -> None:
    """Record the events digest without touching the tape or the Hacker News hash."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("barebone comparison config must be a JSON object")
    if payload.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("evidence path is required")
    if evidence.get("tape_sha256") != LOCKED_TAPE_SHA256:
        raise ValueError("refusing to lock edgar against a changed tape_sha256")
    if evidence.get("narrative_sha256") != LOCKED_NARRATIVE_SHA256:
        raise ValueError("refusing to lock edgar against a changed narrative_sha256")
    if evidence.get("polarity_sha256") != LOCKED_POLARITY_SHA256:
        raise ValueError("refusing to lock edgar against a changed polarity_sha256")
    evidence["edgar_events"] = EDGAR_EVENTS
    evidence["edgar_sha256"] = digest
    validate_barebone_payload(payload)
    config_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fetch_edgar_events(
    *,
    config_path: Path,
    root: Path | None = None,
    lock_config: bool = False,
    sessions: Sequence[date] | None = None,
    get_json: _JsonGet | None = None,
    now: datetime | None = None,
) -> str:
    """Download submissions metadata and return the events digest. Do not store HTML."""

    base = root or _repo_root()
    config = load_barebone_comparison_config(config_path)
    if config.comparable_performance_claim is not False:
        raise ValueError("comparable_performance_claim must be false")
    if config.evidence.tape_sha256 != LOCKED_TAPE_SHA256:
        raise ValueError("tape_sha256 is locked; refusing a new digest without a new experiment_id")
    if config.evidence.narrative_sha256 != LOCKED_NARRATIVE_SHA256:
        raise ValueError("narrative_sha256 is locked; refusing a new digest without a new experiment_id")
    mapping, cik_sha = load_cik_map(base / EDGAR_CIK_MAP)
    calendar = sessions if sessions is not None else sessions_from_ohlcv(base / config.evidence.ohlcv)
    reader = get_json or _default_get_json
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for ticker in BAREBONE_UNIVERSE:
        cik = mapping[ticker]
        payload = reader(EDGAR_SUBMISSIONS + f"CIK{cik}.json")
        if not isinstance(payload, dict):
            raise ValueError(f"submissions for {ticker} are not an object")
        reported = str(payload.get("cik", "")).strip()
        if reported.isdigit():
            reported = f"{int(reported):010d}"
        if reported != cik:
            raise ValueError(f"submissions cik for {ticker} does not match the frozen map")
        issuer_rows = events_for_issuer(
            payload,
            ticker=ticker,
            cik=cik,
            sessions=calendar,
            get_json=reader,
        )
        for row in issuer_rows:
            event_id = str(row["event_id"])
            if event_id in seen:
                raise ValueError(f"edgar event_id {event_id} is repeated")
            seen.add(event_id)
            rows.append(row)
    if not rows:
        raise ValueError("refusing to lock an empty EDGAR tape")
    rows.sort(key=lambda row: (str(row["available_ts"]), str(row["event_id"])))
    rendered = render_events_jsonl(rows)
    digest = hashlib.sha256(rendered).hexdigest()
    form_counts: dict[str, int] = {form: 0 for form in EDGAR_EXACT_FORMS}
    for row in rows:
        form_counts[str(row["form"])] += 1
    _atomic_write(base / EDGAR_EVENTS, rendered)
    sidecar = edgar_provenance(
        events_sha256=digest,
        cik_map_sha256=cik_sha,
        tape_sha256=LOCKED_TAPE_SHA256,
        event_count=len(rows),
        form_counts=form_counts,
        fetched_at=now or datetime.now(timezone.utc),
    )
    _atomic_write(
        base / EDGAR_PROVENANCE,
        (json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )
    if lock_config:
        write_locked_edgar_sha256(config_path, digest)
    return digest


def main(argv: list[str] | None = None) -> int:
    """Fetch SEC submissions metadata. Does not download filing HTML."""

    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--lock-config", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config or (_repo_root() / "configs" / "barebone-comparison-v1.json")
    try:
        digest = fetch_edgar_events(config_path=config_path, lock_config=args.lock_config)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"edgar_sha256={digest}")
    print(f"signal_id={EDGAR_SIGNAL_ID}")
    if args.lock_config:
        print(f"locked edgar_sha256={digest} in {config_path}")
    else:
        print("edgar_sha256 remains unset; re-run with --lock-config to record this digest")
    return 0
