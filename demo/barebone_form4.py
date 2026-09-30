"""Point-in-time SEC Form 4 insider filings for the locked Barebone window.

Filings are listed from ``data.sec.gov`` submissions JSON. The score uses
open-market transaction codes from the filing's raw ownership XML. ``P`` with
acquired code ``A`` is a buy. ``S`` with disposed code ``D`` is a sell. Other
codes are ignored. Notional is not used. ``available_ts`` is
``acceptanceDateTime`` in UTC. ``periodOfReport`` and ``transactionDate`` are
not availability timestamps. Amendments (``4/A``) are excluded. The events
file is gitignored. The 8-K/10-Q/10-K ``edgar_sha256`` stays the locked bind.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path

from demo.barebone_comparison import (
    BAREBONE_EXPERIMENT_ID,
    BAREBONE_UNIVERSE,
    EDGAR_CIK_MAP,
    FORM4_EVENTS,
    FORM4_PROVENANCE,
    LOCKED_ATTENTION_SHA256,
    LOCKED_EDGAR_SHA256,
    LOCKED_NARRATIVE_SHA256,
    LOCKED_POLARITY_SHA256,
    LOCKED_TAPE_SHA256,
    load_barebone_comparison_config,
    validate_barebone_payload,
)
from demo.barebone_edgar import (
    EDGAR_ACCEPTANCE_NOT_BEFORE,
    EDGAR_SUBMISSIONS,
    EDGAR_USER_AGENT,
    demean_cross_section,
    load_cik_map,
)
from demo.barebone_narrative import _aware_utc, decision_session_for, sessions_from_ohlcv

FORM4_SIGNAL_ID = "edgar-form4-netcount-63s-v1"
FORM4_LOOKBACK_SESSIONS = 63
FORM4_TRANSFORM = "open_market_buy_minus_sell_minus_cross_sectional_median"
FORM4_SKILL_THRESHOLD = 0.0
FORM4_SCORE = (
    "open-market buy count minus sell count; P counts only when marked acquired "
    "and S only when marked disposed; other codes and unsigned lines are ignored; "
    "notional is not used"
)
FORM4_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/"
FORM4_SOURCE_NAME = "sec-edgar"
_PROVENANCE_SCHEMA = "fusionfinance-barebone-form4-v1"
_XML_NAME = re.compile(r"[A-Za-z0-9._-]+\.xml")
_JsonGet = Callable[[str], Mapping[str, object]]
_XmlGet = Callable[[str], bytes]
_BANNED_EVENT_FIELDS = frozenset(
    {
        "periodOfReport",
        "transactionDate",
        "reportDate",
        "filingDate",
        "html",
        "text",
        "xml",
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


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _normalize_acceptance(value: object, accession: str) -> datetime:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"filing {accession or '(blank)'} is missing acceptanceDateTime")
    if not isinstance(value, str):
        raise ValueError(f"filing {accession or '(blank)'} acceptanceDateTime is not a timestamp")
    return _aware_utc(value)


def ownership_xml_url(cik: str, accession: str, primary_document: str) -> str:
    """Raw ownership XML. The stylesheet HTML view is not this URL."""

    name = str(primary_document).strip().replace("\\", "/")
    if not name or name.startswith("/") or ".." in name.split("/"):
        raise ValueError(f"Form 4 primary document path {name or '(blank)'} is refused")
    base = name.rsplit("/", 1)[-1]
    if _XML_NAME.fullmatch(base) is None:
        raise ValueError(f"Form 4 primary document {base or '(blank)'} is not an XML file name")
    accession_key = accession.replace("-", "")
    if not accession or not accession_key.isdigit():
        raise ValueError(f"filing accession {accession or '(blank)'} is not usable")
    return f"{FORM4_ARCHIVE}{int(cik)}/{accession_key}/{base}"


def _named_text(node: ET.Element, name: str) -> str | None:
    for element in node.iter():
        if element is node or _local(element.tag) != name:
            continue
        text = (element.text or "").strip()
        if text:
            return text
        for child in element:
            if _local(child.tag) == "value" and (child.text or "").strip():
                return child.text.strip()
    return None


def open_market_counts(payload: bytes, *, cik: str) -> tuple[int, int] | None:
    """Count open-market buys and sells. HTML and non-Form-4 documents raise.

    ``None`` means the ownership issuer is a different company, so this
    submissions row is a holder filing about someone else and is not a
    feature for ``cik``.
    """

    head = payload.lstrip()[:500].lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head[:200]:
        raise ValueError("Form 4 document is HTML; refusing a rendered page")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError("Form 4 document is not XML") from exc
    if _local(root.tag) != "ownershipDocument":
        raise ValueError("Form 4 document is not an ownershipDocument")
    document_type = _named_text(root, "documentType")
    if document_type != "4":
        raise ValueError("ownership document type is not exactly 4")
    issuer = _named_text(root, "issuerCik")
    if issuer is None or not issuer.isdigit():
        raise ValueError("ownership document is missing issuerCik")
    if f"{int(issuer):010d}" != cik:
        return None
    buys = 0
    sells = 0
    for element in root.iter():
        kind = _local(element.tag)
        if kind not in {"nonDerivativeTransaction", "derivativeTransaction"}:
            continue
        code = _named_text(element, "transactionCode")
        if code is None:
            continue
        code = code.strip().upper()
        if code not in {"P", "S"}:
            continue
        side = (_named_text(element, "transactionAcquiredDisposedCode") or "").strip().upper()
        if code == "P" and side == "A":
            buys += 1
        elif code == "S" and side == "D":
            sells += 1
    return buys, sells


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
        raise ValueError("submissions do not reach the acceptance floor and have no overflow files")
    names: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("submissions overflow entry must be an object")
        name = str(item.get("name", "")).strip()
        if not name.startswith("CIK") or name.count("/") or ".." in name or not name.endswith(".json"):
            raise ValueError(f"submissions overflow name {name or '(blank)'} is refused")
        names.append(name)
    return tuple(names)


def form4_rows_from_columns(
    columns: Mapping[str, object],
    *,
    ticker: str,
    cik: str,
    sessions: Sequence[date],
    get_xml: _XmlGet | None,
    acceptance_not_before: date = EDGAR_ACCEPTANCE_NOT_BEFORE,
) -> list[dict[str, object]]:
    """Keep exact Form 4 rows. ``4/A`` and every other form are dropped.

    A kept form with no acceptance timestamp aborts the parse. The signed
    count comes from the ownership XML. ``periodOfReport`` is not copied.
    """

    if "acceptanceDateTime" not in columns:
        raise ValueError("submissions are missing acceptanceDateTime")
    if "form" not in columns or "accessionNumber" not in columns or "primaryDocument" not in columns:
        raise ValueError("submissions are missing form, accessionNumber, or primaryDocument")
    forms = columns["form"]
    accessions = columns["accessionNumber"]
    acceptances = columns["acceptanceDateTime"]
    documents = columns["primaryDocument"]
    if not all(isinstance(column, list) for column in (forms, accessions, acceptances, documents)):
        raise ValueError("submissions columns must be lists")
    if not (len(forms) == len(accessions) == len(acceptances) == len(documents)):
        raise ValueError("submissions columns have different lengths")
    rows: list[dict[str, object]] = []
    dropped_other_issuer = 0
    for form_value, accession_value, accepted, primary in zip(
        forms, accessions, acceptances, documents, strict=True
    ):
        form = str(form_value).strip()
        if form != "4":
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
        url = ownership_xml_url(cik, accession, str(primary))
        if get_xml is None:
            raise ValueError(f"Form 4 {accession or '(blank)'} needs the ownership XML")
        try:
            parsed = open_market_counts(get_xml(url), cik=cik)
        except ValueError as exc:
            raise ValueError(f"{ticker} {accession}: {exc}") from exc
        if parsed is None:
            dropped_other_issuer += 1
            continue
        buys, sells = parsed
        stamp = available.strftime("%Y-%m-%dT%H:%M:%SZ")
        identity = f"{accession}|4|{stamp}|{cik}|{ticker}|{buys}|{sells}"
        rows.append(
            {
                "event_id": f"form4:{accession}:{ticker}",
                "ticker": ticker,
                "cik": cik,
                "accession": accession,
                "form": "4",
                "acceptanceDateTime": stamp,
                "available_ts": stamp,
                "decision_session": session.isoformat(),
                "buy_count": buys,
                "sell_count": sells,
                "net_count": buys - sells,
                "source_class": "other",
                "source_name": FORM4_SOURCE_NAME,
                "uri": url,
                "text_hash": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            }
        )
    return rows, dropped_other_issuer


def rows_for_issuer(
    payload: Mapping[str, object],
    *,
    ticker: str,
    cik: str,
    sessions: Sequence[date],
    get_json: _JsonGet | None = None,
    get_xml: _XmlGet | None = None,
) -> list[dict[str, object]]:
    """Parse one submissions document and any overflow files the window still needs."""

    filings = payload.get("filings")
    if not isinstance(filings, dict):
        raise ValueError(f"submissions for {ticker} are missing filings")
    recent = filings.get("recent")
    if not isinstance(recent, dict):
        raise ValueError(f"submissions for {ticker} are missing recent filings")
    rows, dropped = form4_rows_from_columns(
        recent, ticker=ticker, cik=cik, sessions=sessions, get_xml=get_xml
    )
    for name in _overflow_names(filings.get("files"), oldest=_oldest_acceptance(recent)):
        if get_json is None:
            raise ValueError(f"submissions for {ticker} need overflow {name}")
        extra = get_json(EDGAR_SUBMISSIONS + name)
        if not isinstance(extra, dict):
            raise ValueError(f"overflow {name} is not an object")
        extra_rows, extra_dropped = form4_rows_from_columns(
            extra, ticker=ticker, cik=cik, sessions=sessions, get_xml=get_xml
        )
        rows.extend(extra_rows)
        dropped += extra_dropped
    return rows, dropped


def _request(url: str, accept: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": EDGAR_USER_AGENT, "Accept": accept},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise ValueError("SEC rate limit; refusing a partial Form 4 tape") from exc
        raise ValueError(f"SEC Form 4 request failed: {exc.code}") from exc


def _default_get_json(url: str) -> Mapping[str, object]:
    if not url.startswith(EDGAR_SUBMISSIONS):
        raise ValueError("form4 fetch refuses a non-data.sec.gov submissions URL")
    time.sleep(0.2)
    payload = json.loads(_request(url, "application/json"))
    if not isinstance(payload, dict):
        raise ValueError("SEC submissions response is not an object")
    return payload


def _cache_path(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return Path("/tmp/form4-sec-cache") / digest


def _default_get_xml(url: str) -> bytes:
    if not url.startswith(FORM4_ARCHIVE) or not url.endswith(".xml"):
        raise ValueError("form4 fetch refuses a non-XML SEC archive URL")
    rest = url[len(FORM4_ARCHIVE) :]
    parts = rest.split("/")
    if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError("form4 fetch refuses a non-XML SEC archive URL")
    if _XML_NAME.fullmatch(parts[2]) is None:
        raise ValueError("form4 fetch refuses a non-XML SEC archive URL")
    cached = _cache_path(url)
    if cached.is_file():
        payload = cached.read_bytes()
    else:
        time.sleep(0.2)
        payload = _request(url, "application/xml")
    head = payload.lstrip()[:500].lower()
    if head.startswith(b"<!doctype html") or b"<html" in head[:200]:
        raise ValueError("Form 4 document is HTML; refusing a rendered page")
    if not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(payload)
    return payload


def render_events_jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) for row in rows]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def form4_score_rows(
    events: Sequence[Mapping[str, object]],
    sessions: Sequence[date],
    universe: Sequence[str] | None = None,
    *,
    lookback: int = FORM4_LOOKBACK_SESSIONS,
) -> list[dict[str, object]]:
    """Sum signed open-market counts in ``[j - lookback + 1, j]`` and demean.

    The last session in the window is the decision session. A filing whose
    decision session is later is not a feature. Zero nets stay in the
    cross-section. The feature is the signed count, not a price notional.
    """

    if lookback != FORM4_LOOKBACK_SESSIONS:
        raise ValueError("form4 lookback is locked at 63 sessions")
    names = tuple(universe) if universe is not None else BAREBONE_UNIVERSE
    if len(names) < 2:
        raise ValueError("form4 cross-section needs at least two names")
    ordered = tuple(sessions)
    for earlier, later in zip(ordered, ordered[1:]):
        if later <= earlier:
            raise ValueError("form4 calendar must be strictly increasing")
    index = {day: offset for offset, day in enumerate(ordered)}
    nets = {ticker: [0] * len(ordered) for ticker in names}
    seen: set[str] = set()
    allowed = set(names)
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("form4 event must be an object")
        banned = _BANNED_EVENT_FIELDS.intersection(event)
        if banned:
            raise ValueError("form4 event carries a forbidden field: " + ", ".join(sorted(banned)))
        if event.get("available_ts") in (None, ""):
            raise ValueError("form4 event is missing available_ts")
        available = _aware_utc(str(event["available_ts"]))
        acceptance = event.get("acceptanceDateTime")
        if acceptance in (None, ""):
            raise ValueError("form4 event is missing acceptanceDateTime")
        if _normalize_acceptance(acceptance, str(event.get("accession", ""))).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ) != available.strftime("%Y-%m-%dT%H:%M:%SZ"):
            raise ValueError("available_ts must equal acceptanceDateTime")
        ticker = str(event.get("ticker", "")).strip().upper()
        if ticker not in allowed:
            raise ValueError(f"form4 event ticker {ticker or '(blank)'} is not in the form4 universe")
        if str(event.get("form", "")).strip() != "4":
            raise ValueError("form4 event form is not an exact Form 4")
        try:
            buys = int(event["buy_count"])
            sells = int(event["sell_count"])
            net = int(event["net_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("form4 event is missing a signed open-market count") from exc
        if buys < 0 or sells < 0 or net != buys - sells:
            raise ValueError("form4 net_count must equal buy_count minus sell_count")
        session_text = event.get("decision_session")
        if not isinstance(session_text, str) or not session_text:
            raise ValueError("form4 event is missing decision_session")
        session = date.fromisoformat(session_text)
        if session <= available.date():
            raise ValueError("same-session filing is refused")
        expected = decision_session_for(available, ordered)
        if expected is None or session != expected:
            raise ValueError("decision_session must be the first session strictly after available_ts")
        event_id = str(event.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("form4 event is missing event_id")
        if event_id in seen:
            raise ValueError(f"form4 event_id {event_id} is repeated")
        seen.add(event_id)
        nets[ticker][index[session]] += net
    prefix = {ticker: [0] for ticker in names}
    for ticker in names:
        running = 0
        for count in nets[ticker]:
            running += count
            prefix[ticker].append(running)
    rows: list[dict[str, object]] = []
    for session_index in range(lookback - 1, len(ordered)):
        start = session_index - lookback + 1
        raw = {
            ticker: float(prefix[ticker][session_index + 1] - prefix[ticker][start]) for ticker in names
        }
        scores = demean_cross_section(raw)
        day = ordered[session_index]
        for ticker in names:
            net_count = int(raw[ticker])
            rows.append(
                {
                    "session": day.isoformat(),
                    "session_index": session_index,
                    "ticker": ticker,
                    "net_count": net_count,
                    "score": scores[ticker],
                    "signal_id": FORM4_SIGNAL_ID,
                    "lookback_sessions": lookback,
                }
            )
    return rows


def form4_provenance(
    *,
    events_sha256: str,
    cik_map_sha256: str,
    tape_sha256: str,
    event_count: int,
    buy_transactions: int,
    sell_transactions: int,
    issuer_mismatch_dropped: int,
    fetched_at: datetime,
) -> dict[str, object]:
    if tape_sha256 != LOCKED_TAPE_SHA256:
        raise ValueError("form4 provenance refuses a changed tape_sha256")
    return {
        "schema": _PROVENANCE_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "provider": "SEC EDGAR via data.sec.gov submissions plus raw ownership XML",
        "endpoint": EDGAR_SUBMISSIONS + "CIK##########.json",
        "ownership_xml": "SEC archive raw XML; stylesheet HTML refused; XML bytes not stored",
        "user_agent": EDGAR_USER_AGENT,
        "forms": ["4"],
        "amendments": "excluded; form must be exactly 4",
        "score": FORM4_SCORE,
        "available_ts": "acceptanceDateTime converted to UTC",
        "period_of_report_used_as_available_ts": False,
        "transaction_date_used_as_available_ts": False,
        "report_date_used_as_available_ts": False,
        "acceptance_not_before": EDGAR_ACCEPTANCE_NOT_BEFORE.isoformat(),
        "signal_id": FORM4_SIGNAL_ID,
        "lookback_sessions": FORM4_LOOKBACK_SESSIONS,
        "fetched_at": fetched_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "events": FORM4_EVENTS,
        "edgar_form4_sha256": events_sha256,
        "edgar_sha256": LOCKED_EDGAR_SHA256,
        "cik_map": EDGAR_CIK_MAP,
        "cik_map_sha256": cik_map_sha256,
        "tape_sha256": tape_sha256,
        "narrative_sha256": LOCKED_NARRATIVE_SHA256,
        "event_count": event_count,
        "open_market_buys": buy_transactions,
        "open_market_sells": sell_transactions,
        "issuer_mismatch_dropped": issuer_mismatch_dropped,
        "issuer_rule": "kept only when ownership issuerCik equals the frozen CIK",
        "license_note": "not redistributed; local bind only; metadata only, no filing HTML or XML",
        "comparable_performance_claim": False,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_locked_form4_sha256(config_path: Path, digest: str) -> None:
    """Record the Form 4 digest without touching the earlier evidence locks."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("barebone comparison config must be a JSON object")
    if payload.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("evidence path is required")
    if evidence.get("tape_sha256") != LOCKED_TAPE_SHA256:
        raise ValueError("refusing to lock form4 against a changed tape_sha256")
    if evidence.get("narrative_sha256") != LOCKED_NARRATIVE_SHA256:
        raise ValueError("refusing to lock form4 against a changed narrative_sha256")
    if evidence.get("polarity_sha256") != LOCKED_POLARITY_SHA256:
        raise ValueError("refusing to lock form4 against a changed polarity_sha256")
    if evidence.get("attention_sha256") != LOCKED_ATTENTION_SHA256:
        raise ValueError("refusing to lock form4 against a changed attention_sha256")
    if evidence.get("edgar_sha256") != LOCKED_EDGAR_SHA256:
        raise ValueError("refusing to lock form4 against a changed edgar_sha256")
    evidence["edgar_form4_events"] = FORM4_EVENTS
    evidence["edgar_form4_sha256"] = digest
    validate_barebone_payload(payload)
    config_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fetch_form4_events(
    *,
    config_path: Path,
    root: Path | None = None,
    lock_config: bool = False,
    sessions: Sequence[date] | None = None,
    get_json: _JsonGet | None = None,
    get_xml: _XmlGet | None = None,
    now: datetime | None = None,
) -> str:
    """Download Form 4 metadata and return the events digest. Do not store XML."""

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
    mapping, cik_sha = load_cik_map(base / EDGAR_CIK_MAP)
    calendar = sessions if sessions is not None else sessions_from_ohlcv(base / config.evidence.ohlcv)
    reader = get_json or _default_get_json
    xml_reader = get_xml or _default_get_xml
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    dropped_other_issuer = 0
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
        issuer_rows, dropped = rows_for_issuer(
            payload,
            ticker=ticker,
            cik=cik,
            sessions=calendar,
            get_json=reader,
            get_xml=xml_reader,
        )
        dropped_other_issuer += dropped
        if xml_reader is _default_get_xml:
            print(
                f"form4 {ticker} kept={len(issuer_rows)} other_issuer={dropped}",
                file=sys.stderr,
            )
        for row in issuer_rows:
            event_id = str(row["event_id"])
            if event_id in seen:
                raise ValueError(f"form4 event_id {event_id} is repeated")
            seen.add(event_id)
            rows.append(row)
    if not rows:
        raise ValueError("refusing to lock an empty Form 4 tape")
    rows.sort(key=lambda row: (str(row["available_ts"]), str(row["event_id"])))
    rendered = render_events_jsonl(rows)
    digest = hashlib.sha256(rendered).hexdigest()
    buys = sum(int(row["buy_count"]) for row in rows)
    sells = sum(int(row["sell_count"]) for row in rows)
    _atomic_write(base / FORM4_EVENTS, rendered)
    sidecar = form4_provenance(
        events_sha256=digest,
        cik_map_sha256=cik_sha,
        tape_sha256=LOCKED_TAPE_SHA256,
        event_count=len(rows),
        buy_transactions=buys,
        sell_transactions=sells,
        issuer_mismatch_dropped=dropped_other_issuer,
        fetched_at=now or datetime.now(timezone.utc),
    )
    _atomic_write(
        base / FORM4_PROVENANCE,
        (json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )
    if lock_config:
        write_locked_form4_sha256(config_path, digest)
    return digest


def main(argv: list[str] | None = None) -> int:
    """Fetch SEC Form 4 submissions and ownership XML codes. Does not store the XML."""

    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--lock-config", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config or (_repo_root() / "configs" / "barebone-comparison-v1.json")
    try:
        digest = fetch_form4_events(config_path=config_path, lock_config=args.lock_config)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"edgar_form4_sha256={digest}")
    print(f"signal_id={FORM4_SIGNAL_ID}")
    if args.lock_config:
        print(f"locked edgar_form4_sha256={digest} in {config_path}")
    else:
        print("edgar_form4_sha256 remains unset; re-run with --lock-config to record this digest")
    return 0
