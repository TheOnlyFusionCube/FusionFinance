"""Live SEC EDGAR collectors: Form 4 insider trades and XBRL fundamentals.

Both are free public sources. SEC fair-access rules require a descriptive
User-Agent with contact details and at most ten requests per second; the
collector enforces both. Availability is taken from EDGAR itself: a Form 4 is
available at its ``acceptanceDateTime``; an XBRL fact at the end of the day it
was filed (US Eastern, conservatively using standard time).
"""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from threading import Lock
from urllib.request import Request, urlopen

from alpha.filing_alpha.xbrl import DEFAULT_CONCEPT_MAP
from alpha.research.records import Fundamentals, InsiderTransaction

Fetch = Callable[[str], bytes]
_CODES = {"P", "S", "A", "M", "F", "G", "D", "C", "X", "J", "W"}


@dataclass
class SecClient:
    user_agent: str
    min_interval_seconds: float = 0.11
    timeout_seconds: float = 20.0
    max_bytes: int = 50_000_000
    _last: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def __post_init__(self) -> None:
        if "@" not in self.user_agent or len(self.user_agent) < 8:
            raise ValueError("SEC requires a User-Agent that includes a contact email")

    def __call__(self, url: str) -> bytes:
        if not url.startswith(("https://www.sec.gov/", "https://data.sec.gov/")):
            raise ValueError("SecClient only fetches sec.gov URLs")
        with self._lock:
            wait = self.min_interval_seconds - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
        request = Request(url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "identity"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = response.read(self.max_bytes + 1)
        if len(payload) > self.max_bytes:
            raise ValueError("SEC response exceeded the size limit")
        return payload


def ticker_to_cik(fetch: Fetch) -> dict[str, int]:
    rows = json.loads(fetch("https://www.sec.gov/files/company_tickers.json"))
    return {row["ticker"].upper(): int(row["cik_str"]) for row in rows.values()}


def _text(node: ET.Element | None, path: str) -> str:
    if node is None:
        return ""
    found = node.find(path)
    return "" if found is None or found.text is None else found.text.strip()


def _number(node: ET.Element, path: str) -> float | None:
    raw = _text(node, path)
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def parse_form4(
    xml_bytes: bytes, *, ticker: str, accepted_at: str, accession: str = ""
) -> list[InsiderTransaction]:
    """Parse the non-derivative lines of one Form 4 ownership document."""
    root = ET.fromstring(xml_bytes)
    owner = root.find("reportingOwner")
    name = _text(owner, "reportingOwnerId/rptOwnerName") or "unknown"
    relation = owner.find("reportingOwnerRelationship") if owner is not None else None
    roles = []
    if _text(relation, "isDirector").lower() in {"1", "true"}:
        roles.append("Director")
    if _text(relation, "isOfficer").lower() in {"1", "true"}:
        roles.append(_text(relation, "officerTitle") or "Officer")
    if _text(relation, "isTenPercentOwner").lower() in {"1", "true"}:
        roles.append("10% Owner")
    footnotes = " ".join(
        (item.text or "") for item in root.findall("footnotes/footnote")
    ).lower()
    plan = _text(root, "aff10b5One").lower() in {"1", "true"} or "10b5-1" in footnotes
    records = []
    for line in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _text(line, "transactionCoding/transactionCode").upper()
        trade_date = _text(line, "transactionDate/value")
        shares = _number(line, "transactionAmounts/transactionShares/value")
        if not trade_date or shares is None:
            continue
        records.append(InsiderTransaction(
            ticker=ticker,
            event_at=f"{trade_date[:10]}T00:00:00Z",
            available_at=accepted_at,
            source="sec-edgar-form4",
            insider=name,
            role=", ".join(roles),
            code=code if code in _CODES else "OTHER",
            acquired=_text(line, "transactionAmounts/transactionAcquiredDisposedCode/value") == "A",
            shares=abs(shares),
            price=_number(line, "transactionAmounts/transactionPricePerShare/value"),
            shares_after=_number(
                line, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"
            ),
            rule_10b5_1=plan,
            accession=accession,
        ))
    return records


@dataclass
class EdgarForm4Collector:
    fetch: Fetch
    max_filings: int = 40

    def collect(self, ticker: str, cik: int, *, since: date) -> list[InsiderTransaction]:
        submissions = json.loads(
            self.fetch(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        )
        recent = submissions["filings"]["recent"]
        records: list[InsiderTransaction] = []
        seen = 0
        for index, form in enumerate(recent["form"]):
            if form != "4":
                continue
            if date.fromisoformat(recent["filingDate"][index]) < since or seen >= self.max_filings:
                continue
            seen += 1
            accession = recent["accessionNumber"][index]
            document = recent["primaryDocument"][index].split("/")[-1]
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                f"{accession.replace('-', '')}/{document}"
            )
            accepted = recent["acceptanceDateTime"][index]
            accepted = accepted if accepted.endswith("Z") else accepted + "Z"
            try:
                records.extend(parse_form4(
                    self.fetch(url), ticker=ticker, accepted_at=accepted, accession=accession
                ))
            except ET.ParseError:
                continue
        return records


_EXTRA_CONCEPTS = {
    "eps": ("EarningsPerShareDiluted", "EarningsPerShareBasic"),
    "dividends_per_share": ("CommonStockDividendsPerShareDeclared",),
    "total_debt": ("LongTermDebt", "LongTermDebtNoncurrent"),
}


def parse_companyfacts(payload: dict, *, ticker: str, sector: str = "") -> list[Fundamentals]:
    """Annual 10-K facts grouped by (period end, filing date), first-filed first."""
    gaap = payload.get("facts", {}).get("us-gaap", {})
    dei = payload.get("facts", {}).get("dei", {})
    concepts = {
        "revenue": DEFAULT_CONCEPT_MAP["revenue"],
        "net_income": DEFAULT_CONCEPT_MAP["net_income"],
        "operating_cash_flow": DEFAULT_CONCEPT_MAP["operating_cash_flow"],
        "capex": DEFAULT_CONCEPT_MAP["capex"],
        "cash": DEFAULT_CONCEPT_MAP["cash"],
        "equity": DEFAULT_CONCEPT_MAP["equity"],
        "gross_profit": DEFAULT_CONCEPT_MAP["gross_profit"],
        "total_assets": DEFAULT_CONCEPT_MAP["total_assets"],
        "current_assets": DEFAULT_CONCEPT_MAP["current_assets"],
        "current_liabilities": DEFAULT_CONCEPT_MAP["current_liabilities"],
        **_EXTRA_CONCEPTS,
    }
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for field_name, names in concepts.items():
        for concept in names:
            units = gaap.get(concept, {}).get("units", {})
            facts = units.get("USD") or units.get("USD/shares") or []
            hits = [
                fact for fact in facts
                if fact.get("form") == "10-K" and fact.get("fp") == "FY" and "filed" in fact
                and (fact.get("start") is None or _days(fact["start"], fact["end"]) > 300)
            ]
            for fact in hits:
                slot = grouped[(fact["end"], fact["filed"])]
                slot.setdefault(field_name, float(fact["val"]))
            if hits:
                break
    shares = sorted(
        dei.get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", []),
        key=lambda fact: fact.get("filed", ""),
    )
    records = []
    for (period_end, filed), values in sorted(grouped.items()):
        if "revenue" not in values:
            continue
        known_shares = [fact["val"] for fact in shares if fact.get("filed", "") <= filed]
        records.append(Fundamentals(
            ticker=ticker,
            event_at=f"{period_end}T00:00:00Z",
            available_at=f"{filed}T23:59:59-05:00",
            source="sec-edgar-companyfacts",
            period_end=date.fromisoformat(period_end),
            shares_outstanding=float(known_shares[-1]) if known_shares else None,
            sector=sector,
            **values,
        ))
    return records


def _days(start: str, end: str) -> int:
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days


@dataclass
class EdgarFundamentalsCollector:
    fetch: Fetch

    def collect(self, ticker: str, cik: int, *, sector: str = "") -> list[Fundamentals]:
        payload = json.loads(
            self.fetch(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        )
        return parse_companyfacts(payload, ticker=ticker, sector=sector)


__all__ = [
    "EdgarForm4Collector", "EdgarFundamentalsCollector", "SecClient",
    "parse_companyfacts", "parse_form4", "ticker_to_cik",
]
