"""Local-hash-only ingest for the Barebone-window tape.

The OHLCV bytes stay outside git. A successful ingest prints their SHA-256
and writes a provenance sidecar with no prices. ``--lock-config`` records
that digest on the experiment config. This module does not invent a bar,
fill a gap, or authorize ``comparable_performance_claim``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from pydantic import ConfigDict, Field, field_validator, model_validator

from demo.barebone_comparison import (
    BAREBONE_EXPERIMENT_ID,
    BAREBONE_LICENSE_NOTE,
    BAREBONE_OHLCV,
    BAREBONE_PROVENANCE,
    BAREBONE_WINDOW,
    FAIR_RACE_OHLCV,
    FAIR_RACE_TAPE_HASH,
    BareboneComparisonConfig,
    _FrozenModel,
    _fair_race_ohlcv_sha256,
    _repo_root,
    load_barebone_comparison_config,
    validate_barebone_payload,
)


PROVIDER_ENV = {
    "tiingo": "TIINGO_API_KEY",
    "polygon": "POLYGON_API_KEY",
}
_OHLCV_SCHEMA = "fusionfinance-evidence-ohlcv-v1"
_PROVENANCE_SCHEMA = "fusionfinance-barebone-tape-provenance-v1"
_BAR_FIELDS = ("open", "high", "low", "close", "adjclose", "volume")
_CSV_REQUIRED = ("date", "ticker", "open", "high", "low", "close", "volume")
_CSV_OPTIONAL = ("adjclose",)
_RAW_ADJUSTMENT = (
    "raw OHLC; source supplied no adjusted close; adjclose equals close"
)
_VENDOR_ADJUSTMENT = "raw OHLC plus source adjclose; no second adjustment is applied"
_TIINGO_ADJUSTMENT = (
    "raw OHLC plus Tiingo adjClose; no second adjustment is applied"
)
_POLYGON_ADJUSTMENT = (
    "Polygon v2 aggs with adjusted=true; the payload has no separate "
    "unadjusted close; adjclose equals close"
)
_YAHOO_PROVIDER = "Yahoo Finance via yfinance"
_YAHOO_ADJUSTMENT = (
    "raw OHLC plus Yahoo Finance Adj Close via yfinance; "
    "no second adjustment is applied"
)
_YAHOO_DISCLAIMER = (
    "not redistributed; local bind only. "
    "Yahoo Finance data is not for trading purposes and is not redistributed."
)
_FETCHED_AT = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
_OPENER = Callable[[urllib.request.Request], bytes]


class BareboneProvenance(_FrozenModel):
    """Hash lock for a local tape. It stores no OHLC."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, populate_by_name=True
    )

    schema_name: str = Field(alias="schema")
    experiment_id: str
    provider: str
    fetched_at: str
    tickers: tuple[str, ...] = Field(min_length=1)
    window: tuple[str, str]
    ohlcv: str
    byte_sha256: str
    adjustment: str = Field(min_length=1)
    bar_count: int = Field(gt=0)
    session_count: int = Field(gt=0)
    license_note: str
    disclaimer: str = Field(min_length=1)
    comparable_performance_claim: bool

    @field_validator("schema_name")
    @classmethod
    def _schema(cls, value: str) -> str:
        if value != _PROVENANCE_SCHEMA:
            raise ValueError("barebone provenance schema is missing or changed")
        return value

    @field_validator("experiment_id")
    @classmethod
    def _experiment(cls, value: str) -> str:
        if value != BAREBONE_EXPERIMENT_ID:
            raise ValueError("provenance experiment_id must be barebone-comparison-v1")
        return value

    @field_validator("provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        if value not in {"local-csv", "tiingo", "polygon", _YAHOO_PROVIDER}:
            raise ValueError("provenance provider is not a local bind source")
        return value

    @field_validator("fetched_at")
    @classmethod
    def _fetched_at(cls, value: str) -> str:
        if len(value) != 20 or value[10] != "T" or not value.endswith("Z"):
            raise ValueError("fetched_at must be a UTC timestamp")
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return value

    @field_validator("window")
    @classmethod
    def _window(cls, value: tuple[str, str]) -> tuple[str, str]:
        if tuple(value) != BAREBONE_WINDOW:
            raise ValueError("provenance window does not match barebone-comparison-v1")
        return tuple(value)

    @field_validator("byte_sha256")
    @classmethod
    def _digest(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("byte_sha256 must be a lowercase sha256 hex digest")
        if value == FAIR_RACE_TAPE_HASH:
            raise ValueError("fair-race tape_hash is not the barebone-comparison tape")
        return value

    @field_validator("license_note")
    @classmethod
    def _license(cls, value: str) -> str:
        if value != BAREBONE_LICENSE_NOTE:
            raise ValueError("provenance license note is missing")
        return value

    @field_validator("comparable_performance_claim")
    @classmethod
    def _claim(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("comparable_performance_claim must be false")
        return False

    @model_validator(mode="after")
    def _yahoo_disclaimer(self) -> "BareboneProvenance":
        if self.provider != _YAHOO_PROVIDER:
            return self
        text = self.disclaimer.casefold()
        if "not for trading" not in text or "not redistributed" not in text:
            raise ValueError(
                "Yahoo provenance must include the not-for-trading "
                "and no-redistribute disclaimer"
            )
        if "tiingo" in text or "polygon" in text:
            raise ValueError("Yahoo provenance must not name another vendor as the source")
        return self


def required_tickers(config: BareboneComparisonConfig) -> tuple[str, ...]:
    """Tradable names plus SPY and the optional secondary benchmark."""

    names = [
        *config.experiment.universe,
        config.experiment.benchmark_ticker,
    ]
    if config.secondary_benchmark_ticker is not None:
        names.append(config.secondary_benchmark_ticker)
    if len(names) != len(set(names)):
        raise ValueError("barebone tape tickers must be unique")
    return tuple(names)


def refuse_software_mark_source(source: str) -> None:
    """This ingest has no synthetic price generator."""

    if source in {"software-marks", "synthetic", "generated"}:
        raise ValueError("barebone ingest refuses software-mark prices")


def read_ohlcv_csv(path: Path, *, adjustment: str) -> tuple[list[dict[str, object]], str]:
    """Read a user dump. Missing ``adjclose`` is refused unless adjustment is raw."""

    if adjustment not in {"auto", "raw"}:
        raise ValueError("adjustment must be auto or raw")
    _refuse_fair_race_path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("ohlcv csv is missing a header")
        fields = tuple(name.strip().lower() for name in reader.fieldnames)
        if any(not name for name in fields) or len(fields) != len(set(fields)):
            raise ValueError("ohlcv csv header names must be unique")
        unknown = set(fields).difference(_CSV_REQUIRED + _CSV_OPTIONAL)
        missing = [name for name in _CSV_REQUIRED if name not in fields]
        if missing or unknown:
            raise ValueError("ohlcv csv columns must be date,ticker,OHLCV, and optional adjclose")
        has_adj = "adjclose" in fields
        if adjustment == "raw" and has_adj:
            raise ValueError("raw adjustment refuses a csv that already has adjclose")
        if adjustment == "auto" and not has_adj:
            raise ValueError(
                "source has no adjclose; pass --adjustment raw to record unadjusted "
                "bars, or supply adjclose. Silent adjustment is refused"
            )
        rows: list[dict[str, object]] = []
        for raw in reader:
            item = {
                key.strip().lower(): value.strip()
                for key, value in raw.items()
                if key is not None
            }
            if not any(item.values()):
                continue
            row = _csv_row(item, has_adj=has_adj)
            rows.append(row)
    if not rows:
        raise ValueError("ohlcv csv has no bars")
    note = _VENDOR_ADJUSTMENT if has_adj else _RAW_ADJUSTMENT
    return rows, note


def assemble_ohlcv_document(
    config: BareboneComparisonConfig,
    rows: list[dict[str, object]],
    *,
    vendor: str,
    adjustment: str,
) -> dict[str, object]:
    """Validate a complete cross-section. Missing names and sessions raise."""

    if not adjustment.strip():
        raise ValueError("adjustment note must describe the source")
    expected = required_tickers(config)
    parsed = [_checked_bar(row) for row in rows]
    seen: set[tuple[str, str]] = set()
    by_session: dict[str, dict[str, dict[str, object]]] = {}
    start, end = BAREBONE_WINDOW
    for bar in parsed:
        session = str(bar["date"])
        ticker = str(bar["ticker"])
        if session < start or session > end:
            raise ValueError(f"ohlcv bar {ticker} on {session} is outside the barebone window")
        if ticker not in expected:
            raise ValueError(f"ohlcv bar ticker {ticker} is outside the barebone tape")
        key = (session, ticker)
        if key in seen:
            raise ValueError(f"duplicate ohlcv bar for {ticker} on {session}")
        seen.add(key)
        by_session.setdefault(session, {})[ticker] = bar
    if start not in by_session or end not in by_session:
        raise ValueError("barebone tape must include the window endpoints")
    for session, present in by_session.items():
        missing = [ticker for ticker in expected if ticker not in present]
        if missing:
            raise ValueError(
                f"ohlcv session {session} missing ticker(s): {', '.join(missing)}"
            )
    bars = [
        present[ticker]
        for session in sorted(by_session)
        for ticker in expected
        for present in (by_session[session],)
    ]
    return {
        "schema": _OHLCV_SCHEMA,
        "vendor": vendor,
        "interval": "1d",
        "currency": "USD",
        "adjustment": adjustment,
        "calendar_source": "configs/barebone-comparison-v1.json",
        "spy_return_source": None,
        "redistribution": BAREBONE_LICENSE_NOTE,
        "bars": bars,
    }


def render_ohlcv_bytes(document: dict[str, object]) -> bytes:
    """Canonical bytes whose SHA-256 is the lock."""

    return (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def ingest_barebone_tape(
    *,
    config_path: Path | None = None,
    source: str,
    csv_path: Path | None = None,
    adjustment: str = "auto",
    output_path: Path | None = None,
    provenance_path: Path | None = None,
    lock_config: bool = False,
    api_key: str | None = None,
    opener: _OPENER | None = None,
    now: datetime | None = None,
    root: Path | None = None,
) -> str:
    """Write the gitignored extract and its provenance. Return the byte SHA-256."""

    refuse_software_mark_source(source)
    base = _repo_root() if root is None else root
    config = load_barebone_comparison_config(config_path)
    disclaimer = BAREBONE_LICENSE_NOTE
    if source == "local-csv":
        if csv_path is None:
            raise ValueError("--from-csv is required for a local dump")
        if adjustment not in {"auto", "raw"}:
            raise ValueError("adjustment must be auto or raw")
        rows, note = read_ohlcv_csv(csv_path, adjustment=adjustment)
        vendor = "local-csv"
        provider = "local-csv"
    elif source in PROVIDER_ENV:
        if adjustment != "auto":
            raise ValueError("provider adjustment comes from the feed")
        key = _api_key(source, api_key)
        rows, note = fetch_provider_rows(
            source,
            required_tickers(config),
            api_key=key,
            opener=opener or _default_opener,
        )
        vendor = "Tiingo daily prices" if source == "tiingo" else "Polygon v2 aggs"
        provider = source
    elif source == "yfinance":
        if adjustment != "auto":
            raise ValueError("provider adjustment comes from the feed")
        rows, note = fetch_yfinance_rows(required_tickers(config))
        vendor = _YAHOO_PROVIDER
        provider = _YAHOO_PROVIDER
        disclaimer = _YAHOO_DISCLAIMER
    else:
        raise ValueError("source must be local-csv, tiingo, polygon, or yfinance")
    document = assemble_ohlcv_document(config, rows, vendor=vendor, adjustment=note)
    payload = render_ohlcv_bytes(document)
    _refuse_fair_race_bytes(payload)
    destination = output_path or (base / BAREBONE_OHLCV)
    sidecar = provenance_path or (base / BAREBONE_PROVENANCE)
    _refuse_fair_race_path(destination)
    digest = hashlib.sha256(payload).hexdigest()
    _atomic_write(destination, payload)
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    sessions = {str(bar["date"]) for bar in document["bars"]}
    provenance = BareboneProvenance.model_validate(
        {
            "schema": _PROVENANCE_SCHEMA,
            "experiment_id": BAREBONE_EXPERIMENT_ID,
            "provider": provider,
            "fetched_at": stamp,
            "tickers": list(required_tickers(config)),
            "window": list(BAREBONE_WINDOW),
            "ohlcv": config.evidence.ohlcv,
            "byte_sha256": digest,
            "adjustment": note,
            "bar_count": len(document["bars"]),
            "session_count": len(sessions),
            "license_note": BAREBONE_LICENSE_NOTE,
            "disclaimer": disclaimer,
            "comparable_performance_claim": False,
        }
    )
    _atomic_write(
        sidecar,
        (
            json.dumps(
                provenance.model_dump(by_alias=True, mode="json"),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    if lock_config:
        expected = (base / config.evidence.ohlcv).resolve()
        if destination.resolve() != expected:
            raise ValueError("--lock-config only locks the configured evidence path")
        if config_path is None:
            raise ValueError("--lock-config requires an explicit config path in tests and CLI")
        write_locked_tape_sha256(config_path, digest)
    return digest


def write_locked_tape_sha256(config_path: Path, digest: str) -> None:
    """Record an ingest digest. This does not invent one and does not set a claim."""

    if digest == FAIR_RACE_TAPE_HASH or digest == _fair_race_ohlcv_sha256():
        raise ValueError("fair-race OHLCV is not the barebone-comparison tape")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("barebone comparison config must be a JSON object")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("evidence path is required")
    evidence["tape_sha256"] = digest
    if payload.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    validate_barebone_payload(payload)
    config_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def fetch_provider_rows(
    provider: str,
    tickers: tuple[str, ...],
    *,
    api_key: str,
    opener: _OPENER,
) -> tuple[list[dict[str, object]], str]:
    """Download daily bars. The API key is not returned or stored."""

    if provider == "tiingo":
        rows = [
            bar
            for ticker in tickers
            for bar in _fetch_tiingo_ticker(ticker, api_key=api_key, opener=opener)
        ]
        return rows, _TIINGO_ADJUSTMENT
    if provider == "polygon":
        rows = [
            bar
            for ticker in tickers
            for bar in _fetch_polygon_ticker(ticker, api_key=api_key, opener=opener)
        ]
        return rows, _POLYGON_ADJUSTMENT
    raise ValueError("provider must be tiingo or polygon")


def fetch_yfinance_rows(
    tickers: tuple[str, ...],
    *,
    download: Callable[..., object] | None = None,
) -> tuple[list[dict[str, object]], str]:
    """Download Yahoo daily bars through yfinance. Missing fields raise."""

    start, end = BAREBONE_WINDOW
    end_exclusive = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
    fetcher = download or _yfinance_download
    try:
        frame = fetcher(list(tickers), start, end_exclusive)
    except Exception as exc:
        message = str(exc).splitlines()[0][:200]
        raise ValueError("yfinance price fetch failed: " + message) from None
    return rows_from_yfinance_frame(frame, tickers), _YAHOO_ADJUSTMENT


def rows_from_yfinance_frame(
    frame: object, tickers: tuple[str, ...]
) -> list[dict[str, object]]:
    """Map a yfinance frame to bars. A NaN field is a gap, not a fill."""

    import pandas as pd

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("yfinance returned no bars")
    if not isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("yfinance frame is missing the ticker level")
    present = {str(name) for name in frame.columns.get_level_values(0)}
    missing = [ticker for ticker in tickers if ticker not in present]
    if missing:
        raise ValueError("yfinance returned no bars for " + ", ".join(missing))
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        sub = frame[ticker]
        if "Adj Close" not in sub.columns:
            raise ValueError(
                f"yfinance bar for {ticker} has no Adj Close; refusing to invent an adjustment"
            )
        for label, record in sub.iterrows():
            session = pd.Timestamp(label).date().isoformat()
            values: dict[str, float] = {}
            for field, key in (
                ("Open", "open"),
                ("High", "high"),
                ("Low", "low"),
                ("Close", "close"),
                ("Adj Close", "adjclose"),
                ("Volume", "volume"),
            ):
                if field not in sub.columns:
                    raise ValueError(f"yfinance bar for {ticker} is missing {field}")
                raw = record[field]
                if pd.isna(raw):
                    raise ValueError(
                        f"yfinance bar for {ticker} on {session} is missing {field}; "
                        "refusing to fill the gap"
                    )
                values[key] = float(raw)
            rows.append({"date": session, "ticker": ticker, **values})
    return rows


def _yfinance_download(tickers: list[str], start: str, end: str) -> object:
    import yfinance as yf

    return yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        threads=True,
        progress=False,
        repair=False,
        keepna=True,
        interval="1d",
        timeout=60,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Prints the SHA-256 and never prints an API key."""

    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--provider", choices=("tiingo", "polygon", "yfinance"))
    parser.add_argument("--from-csv", type=Path, dest="csv_path")
    parser.add_argument("--adjustment", choices=("auto", "raw"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--provenance", type=Path, default=None)
    parser.add_argument("--lock-config", action="store_true")
    args = parser.parse_args(argv)
    if (args.provider is None) == (args.csv_path is None):
        print("pass exactly one of --provider or --from-csv", file=sys.stderr)
        return 2
    source = "local-csv" if args.csv_path is not None else str(args.provider)
    config_path = args.config or (_repo_root() / "configs" / "barebone-comparison-v1.json")
    try:
        digest = ingest_barebone_tape(
            config_path=config_path,
            source=source,
            csv_path=args.csv_path,
            adjustment=args.adjustment,
            output_path=args.output,
            provenance_path=args.provenance,
            lock_config=args.lock_config,
            root=_repo_root(),
        )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"sha256={digest}")
    if args.lock_config:
        print(f"locked tape_sha256={digest} in {config_path}")
    else:
        print("tape_sha256 remains null; re-run with --lock-config to record this digest")
    return 0


def _csv_row(item: dict[str, str], *, has_adj: bool) -> dict[str, object]:
    raw_date = _cell(item, "date")
    try:
        session = date.fromisoformat(raw_date).isoformat()
    except ValueError as exc:
        raise ValueError(f"ohlcv csv date is not YYYY-MM-DD: {raw_date}") from exc
    ticker = _cell(item, "ticker").upper()
    if not ticker:
        raise ValueError("ohlcv csv ticker must not be blank")
    numbers = {
        name: _finite_number(_cell(item, name), name)
        for name in ("open", "high", "low", "close", "volume")
    }
    if has_adj:
        numbers["adjclose"] = _finite_number(_cell(item, "adjclose"), "adjclose")
    else:
        numbers["adjclose"] = numbers["close"]
    return {"date": session, "ticker": ticker, **numbers}


def _cell(item: dict[str, str], name: str) -> str:
    value = item.get(name)
    if not isinstance(value, str):
        raise ValueError(f"ohlcv csv is missing {name}")
    return value.strip()


def _checked_bar(row: dict[str, object]) -> dict[str, object]:
    missing = [field for field in ("date", "ticker", *_BAR_FIELDS) if field not in row]
    if missing:
        raise ValueError(f"ohlcv bar is missing {missing}")
    try:
        values = {field: float(row[field]) for field in _BAR_FIELDS}
    except (TypeError, ValueError) as exc:
        raise ValueError("ohlcv bar contains a non-numeric field") from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("ohlcv bar contains a non-finite field")
    if min(values["open"], values["high"], values["low"], values["close"], values["adjclose"]) <= 0.0:
        raise ValueError("ohlcv prices must be positive")
    if values["volume"] < 0.0:
        raise ValueError("ohlcv volume must be non-negative")
    _require_high_low(values)
    scale = values["adjclose"] / values["close"]
    adjusted = {name: values[name] * scale for name in ("open", "high", "low", "close")}
    _require_high_low(adjusted)
    volume = values["volume"]
    stored_volume: int | float = int(volume) if volume.is_integer() else volume
    return {
        "date": str(row["date"]),
        "ticker": str(row["ticker"]),
        "open": values["open"],
        "high": values["high"],
        "low": values["low"],
        "close": values["close"],
        "adjclose": values["adjclose"],
        "volume": stored_volume,
    }


def _require_high_low(values: dict[str, float]) -> None:
    upper = max(values["open"], values["close"], values["low"])
    lower = min(values["open"], values["close"], values["high"])
    if values["high"] + 1e-12 < upper or values["low"] - 1e-12 > lower:
        raise ValueError("ohlcv bar violates high/low relationships")


def _finite_number(value: str, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ohlcv csv {name} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"ohlcv csv {name} is not finite")
    return number


def _fetch_tiingo_ticker(ticker: str, *, api_key: str, opener: _OPENER) -> list[dict[str, object]]:
    start, end = BAREBONE_WINDOW
    quoted = urllib.parse.quote(ticker.lower())
    url = (
        "https://api.tiingo.com/tiingo/daily/"
        f"{quoted}/prices?startDate={start}&endDate={end}&format=json"
    )
    request = urllib.request.Request(url, headers={"Authorization": f"Token {api_key}"})
    payload = _read_json_body(_read_http(request, api_key, opener), api_key, ticker)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"tiingo returned no bars for {ticker}")
    rows: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict) or "adjClose" not in item:
            raise ValueError(
                f"tiingo bar for {ticker} has no adjClose; refusing to invent an adjustment"
            )
        session = str(item.get("date", ""))[:10]
        rows.append(
            {
                "date": session,
                "ticker": ticker,
                "open": item.get("open"),
                "high": item.get("high"),
                "low": item.get("low"),
                "close": item.get("close"),
                "adjclose": item.get("adjClose"),
                "volume": item.get("volume"),
            }
        )
    return rows


def _fetch_polygon_ticker(ticker: str, *, api_key: str, opener: _OPENER) -> list[dict[str, object]]:
    start, end = BAREBONE_WINDOW
    symbol = urllib.parse.quote(ticker.replace("-", "."))
    url: str | None = (
        "https://api.polygon.io/v2/aggs/ticker/"
        f"{symbol}/range/1/day/{start}/{end}"
        "?adjusted=true&sort=asc&limit=50000&apiKey="
        + urllib.parse.quote(api_key)
    )
    rows: list[dict[str, object]] = []
    pages = 0
    while url is not None:
        pages += 1
        if pages > 10:
            raise ValueError("polygon response is truncated")
        request = urllib.request.Request(url)
        payload = _read_json_body(_read_http(request, api_key, opener), api_key, ticker)
        if not isinstance(payload, dict):
            raise ValueError(f"polygon returned an unexpected payload for {ticker}")
        status = payload.get("status")
        if status not in {"OK", "DELAYED"}:
            raise ValueError(f"polygon status for {ticker} was {status}")
        results = payload.get("results") or []
        if not isinstance(results, list):
            raise ValueError(f"polygon results for {ticker} are not a list")
        for item in results:
            if not isinstance(item, dict):
                raise ValueError(f"polygon bar for {ticker} is not an object")
            session = datetime.fromtimestamp(
                float(item["t"]) / 1000.0, timezone.utc
            ).date().isoformat()
            close = item.get("c")
            rows.append(
                {
                    "date": session,
                    "ticker": ticker,
                    "open": item.get("o"),
                    "high": item.get("h"),
                    "low": item.get("l"),
                    "close": close,
                    "adjclose": close,
                    "volume": item.get("v"),
                }
            )
        nxt = payload.get("next_url")
        if not nxt:
            url = None
        else:
            joined = str(nxt)
            if "apiKey=" not in joined:
                separator = "&" if "?" in joined else "?"
                joined = f"{joined}{separator}apiKey={urllib.parse.quote(api_key)}"
            url = joined
    if not rows:
        raise ValueError(f"polygon returned no bars for {ticker}")
    return rows


def _read_http(request: urllib.request.Request, api_key: str, opener: _OPENER) -> bytes:
    try:
        return opener(request)
    except Exception as exc:
        raise ValueError(
            "barebone price fetch failed: " + _redact(str(exc), api_key)
        ) from None


def _read_json_body(body: bytes, api_key: str, ticker: str) -> object:
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"barebone price fetch for {ticker} was not JSON: " + _redact(str(exc), api_key)
        ) from None


def _api_key(provider: str, api_key: str | None) -> str:
    env_name = PROVIDER_ENV[provider]
    value = api_key if api_key is not None else os.environ.get(env_name)
    if value is None or not value.strip():
        raise ValueError(f"{env_name} is required for --provider {provider}")
    return value.strip()


def _default_opener(request: urllib.request.Request) -> bytes:
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _redact(text: str, secret: str) -> str:
    if secret:
        return text.replace(secret, "***")
    return text


def _refuse_fair_race_path(path: Path) -> None:
    posix = path.as_posix().replace("\\", "/")
    if path.name == "locked_ohlcv.json" or posix.endswith(FAIR_RACE_OHLCV):
        raise ValueError("fair-race OHLCV is not the barebone-comparison tape")


def _refuse_fair_race_bytes(payload: bytes) -> None:
    if hashlib.sha256(payload).hexdigest() == FAIR_RACE_TAPE_HASH:
        raise ValueError("fair-race tape_hash is not the barebone-comparison tape")
    fair_path = _repo_root() / FAIR_RACE_OHLCV
    if fair_path.is_file() and payload == fair_path.read_bytes():
        raise ValueError("fair-race OHLCV is not the barebone-comparison tape")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
