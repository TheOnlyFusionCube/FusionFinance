"""Fail-closed OHLCV tape for the locked controlled window.

The session calendar and the SPY return path come from
``evidence/replay/v1_source.json``. Open, high, low, close, volume, and the
vendor adjusted close come from ``evidence/market/locked_ohlcv.json``. A
missing bar, a calendar disagreement, or a SPY return that does not reproduce
the sealed benchmark raises. Nothing in this module fills a gap or invents a
price.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd

from demo.contracts import AssetBar, ExperimentConfig, MarketSession


_REPLAY_RELATIVE = "evidence/replay/v1_source.json"
_OHLCV_RELATIVE = "evidence/market/locked_ohlcv.json"
_RETURN_ABS_TOL = 1e-5
_BAR_FIELDS = ("open", "high", "low", "close", "adjclose", "volume")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def evidence_paths() -> dict[str, str]:
    """Checked-in inputs the tape builder is allowed to read."""

    return {
        "calendar": _REPLAY_RELATIVE,
        "ohlcv": _OHLCV_RELATIVE,
    }


def evidence_file_sha256(root: Path | None = None) -> dict[str, str]:
    """Hash the calendar source and the OHLCV extract the builder reads."""

    base = _repo_root() if root is None else root
    return {
        relative: hashlib.sha256((base / relative).read_bytes()).hexdigest()
        for relative in evidence_paths().values()
    }


def evidence_price_sessions(
    config: ExperimentConfig, *, root: Path | None = None
) -> tuple[MarketSession, ...]:
    """Return the locked-window tape. Missing evidence raises."""

    base = _repo_root() if root is None else root
    replay = _read_json(base / _REPLAY_RELATIVE)
    dates = _sealed_dates(config, replay)
    sessions = _sessions_from_records(config, dates, _read_bars(base))
    _require_spy_returns(
        sessions,
        replay["arms"]["benchmark"]["daily_returns"],
        benchmark_ticker=config.benchmark_ticker,
    )
    return sessions


def adjusted_market_frame(
    config: ExperimentConfig, *, root: Path | None = None
) -> pd.DataFrame:
    """Adjusted bars on and before the locked window, with no filled gaps.

    Rows after ``config.end_date`` are ignored. A session on or before that
    date is kept only when every locked name and the benchmark are present.
    A partial session raises instead of being dropped.
    """

    base = _repo_root() if root is None else root
    replay = _read_json(base / _REPLAY_RELATIVE)
    sealed = set(_sealed_dates(config, replay))
    required = (*config.universe, config.benchmark_ticker)
    grouped: dict[date, dict[str, dict[str, float]]] = {}
    for record in _read_bars(base):
        session = date.fromisoformat(str(record["date"]))
        if session > config.end_date:
            raise ValueError(
                f"ohlcv evidence contains a session after the locked window: {session.isoformat()}"
            )
        ticker = str(record["ticker"]).strip().upper()
        adjusted = _adjusted_ohlcv(record)
        bucket = grouped.setdefault(session, {})
        if ticker in bucket:
            raise ValueError(
                f"duplicate ohlcv bar for {ticker} on {session.isoformat()}"
            )
        bucket[ticker] = adjusted
    rows: list[dict[str, object]] = []
    for session in sorted(grouped):
        present = grouped[session]
        missing = [ticker for ticker in required if ticker not in present]
        if missing:
            raise ValueError(
                f"ohlcv session {session.isoformat()} missing ticker(s): {missing}"
            )
        if config.start_date <= session <= config.end_date and session not in sealed:
            raise ValueError(
                "ohlcv session "
                f"{session.isoformat()} is inside the locked window but absent "
                "from the sealed replay calendar"
            )
        for ticker in required:
            fields = present[ticker]
            rows.append(
                {
                    "date": pd.Timestamp(session),
                    "ticker": ticker,
                    "open": fields["open"],
                    "high": fields["high"],
                    "low": fields["low"],
                    "close": fields["close"],
                    "volume": fields["volume"],
                }
            )
    if not rows:
        raise ValueError("ohlcv evidence produced no market rows")
    return pd.DataFrame(rows)


def _sessions_from_records(
    config: ExperimentConfig,
    dates: tuple[date, ...],
    records: list[dict[str, object]],
) -> tuple[MarketSession, ...]:
    required = (*config.universe, config.benchmark_ticker)
    index: dict[tuple[date, str], dict[str, float]] = {}
    for record in records:
        session = date.fromisoformat(str(record["date"]))
        ticker = str(record["ticker"]).strip().upper()
        key = (session, ticker)
        if key in index:
            raise ValueError(
                f"duplicate ohlcv bar for {ticker} on {session.isoformat()}"
            )
        index[key] = _adjusted_ohlcv(record)
    sealed = set(dates)
    for session, ticker in index:
        if config.start_date <= session <= config.end_date and session not in sealed:
            raise ValueError(
                "ohlcv session "
                f"{session.isoformat()} is inside the locked window but absent "
                "from the sealed replay calendar"
            )
    sessions: list[MarketSession] = []
    for session in dates:
        missing = [ticker for ticker in required if (session, ticker) not in index]
        if missing:
            raise ValueError(
                f"market session {session.isoformat()} missing ticker(s): {missing}"
            )
        sessions.append(
            MarketSession(
                session=session,
                bars=tuple(
                    AssetBar(
                        ticker=ticker,
                        open=index[(session, ticker)]["open"],
                        close=index[(session, ticker)]["close"],
                    )
                    for ticker in required
                ),
            )
        )
    return tuple(sessions)


def _sealed_dates(config: ExperimentConfig, replay: dict[str, object]) -> tuple[date, ...]:
    if replay.get("schema_version") != 1:
        raise ValueError("public replay source schema_version must equal 1")
    if replay.get("claim_status") != "provisional_uncontrolled_legacy_race":
        raise ValueError("public replay claim boundary is missing or changed")
    window = [config.start_date.isoformat(), config.end_date.isoformat()]
    if replay.get("window") != window:
        raise ValueError("sealed replay window does not match the locked config")
    raw_dates = replay.get("dates")
    if not isinstance(raw_dates, list) or not raw_dates:
        raise ValueError("sealed replay dates are missing")
    dates = tuple(date.fromisoformat(str(value)) for value in raw_dates)
    if dates != tuple(sorted(set(dates))):
        raise ValueError("sealed replay dates must be unique and increasing")
    if dates[0] != config.start_date or dates[-1] != config.end_date:
        raise ValueError("sealed replay dates must cover the locked window endpoints")
    returns = _benchmark_returns(replay)
    if len(returns) != len(dates):
        raise ValueError("sealed benchmark returns do not match the replay dates")
    return dates


def _benchmark_returns(replay: dict[str, object]) -> list[object]:
    arms = replay.get("arms")
    if not isinstance(arms, dict) or "benchmark" not in arms:
        raise ValueError("sealed replay is missing the benchmark arm")
    benchmark = arms["benchmark"]
    if not isinstance(benchmark, dict):
        raise ValueError("sealed benchmark arm must be an object")
    returns = benchmark.get("daily_returns")
    if not isinstance(returns, list) or not returns:
        raise ValueError("sealed benchmark returns are missing")
    return returns


def _require_spy_returns(
    sessions: tuple[MarketSession, ...],
    returns: list[object],
    *,
    benchmark_ticker: str,
) -> None:
    if len(returns) != len(sessions):
        raise ValueError("sealed benchmark returns do not match the tape")
    first = float(returns[0])
    if not math.isfinite(first) or abs(first) > 1e-12:
        raise ValueError("sealed benchmark first return is not the stored zero")
    closes: list[float] = []
    for session in sessions:
        bars = {bar.ticker: bar for bar in session.bars}
        bar = bars.get(benchmark_ticker)
        if bar is None:
            raise ValueError(
                f"market session {session.session.isoformat()} missing {benchmark_ticker}"
            )
        closes.append(bar.close)
    for index in range(1, len(sessions)):
        actual = closes[index] / closes[index - 1] - 1.0
        expected = float(returns[index])
        if not math.isfinite(expected) or abs(actual - expected) > _RETURN_ABS_TOL:
            day = sessions[index].session.isoformat()
            raise ValueError(
                f"SPY adjusted close on {day} does not reproduce the sealed benchmark return"
            )


def _read_bars(root: Path) -> list[dict[str, object]]:
    payload = _read_json(root / _OHLCV_RELATIVE)
    if payload.get("schema") != "fusionfinance-evidence-ohlcv-v1":
        raise ValueError("ohlcv evidence schema is missing or changed")
    if payload.get("calendar_source") != _REPLAY_RELATIVE:
        raise ValueError("ohlcv evidence is not bound to the sealed replay calendar")
    bars = payload.get("bars")
    if not isinstance(bars, list) or not bars:
        raise ValueError("ohlcv evidence has no bars")
    if not all(isinstance(bar, dict) for bar in bars):
        raise ValueError("ohlcv evidence bars must be JSON objects")
    return bars


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required market evidence is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON evidence at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"market evidence must be a JSON object: {path}")
    return payload


def _adjusted_ohlcv(record: dict[str, object]) -> dict[str, float]:
    missing = [field for field in _BAR_FIELDS if field not in record]
    if missing:
        raise ValueError(f"ohlcv bar is missing {missing}")
    try:
        values = {field: float(record[field]) for field in _BAR_FIELDS}
    except (TypeError, ValueError) as exc:
        raise ValueError("ohlcv bar contains a non-numeric field") from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("ohlcv bar contains a non-finite field")
    if values["close"] <= 0.0 or values["adjclose"] <= 0.0:
        raise ValueError("ohlcv close and adjclose must be positive")
    if values["volume"] < 0.0:
        raise ValueError("ohlcv volume must be non-negative")
    scale = values["adjclose"] / values["close"]
    adjusted = {
        name: values[name] * scale for name in ("open", "high", "low", "close")
    }
    if min(adjusted.values()) <= 0.0:
        raise ValueError("adjusted OHLC must be positive")
    upper = max(adjusted["open"], adjusted["close"], adjusted["low"])
    lower = min(adjusted["open"], adjusted["close"], adjusted["high"])
    if adjusted["high"] < upper or adjusted["low"] > lower:
        raise ValueError("ohlcv bar violates high/low relationships")
    adjusted["volume"] = values["volume"]
    return adjusted
