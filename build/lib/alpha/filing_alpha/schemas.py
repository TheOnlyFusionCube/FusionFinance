"""Immutable dataframe validation for filing-alpha transforms.

Adapted from filing-alpha-cleanroom 0.6.0 under the MIT License.  See NOTICE.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


MARKET_REQUIRED = ("date", "ticker", "open", "high", "low", "close", "volume")
FILING_REQUIRED = ("accepted_at", "ticker")


def require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    """Raise when ``frame`` lacks any required column."""
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a validated, normalized copy of split-adjusted daily OHLCV data."""
    require_columns(frame, MARKET_REQUIRED, "market data")
    result = frame.copy(deep=True)
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.tz_localize(None)
    result["ticker"] = result["ticker"].astype(str).str.upper().str.strip()
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    ohlc = result[["open", "high", "low", "close"]]
    if not np.isfinite(ohlc.to_numpy(dtype=float)).all():
        raise ValueError("OHLC columns must contain finite numeric values.")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be strictly positive.")
    if (
        not np.isfinite(result["volume"].to_numpy(dtype=float)).all()
        or (result["volume"] < 0).any()
    ):
        raise ValueError("Volume must be numeric, finite, and non-negative.")
    invalid_bars = (result["high"] < result[["open", "close", "low"]].max(axis=1)) | (
        result["low"] > result[["open", "close", "high"]].min(axis=1)
    )
    if invalid_bars.any():
        sample = result.loc[
            invalid_bars, ["date", "ticker", "open", "high", "low", "close"]
        ].head()
        raise ValueError(f"Invalid OHLC relationships detected. Sample:\n{sample}")
    if result.duplicated(["ticker", "date"], keep=False).any():
        raise ValueError("Market data must contain one row per ticker/date.")
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)


def validate_filing_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of one-row-per-filing aggregate features."""
    require_columns(frame, FILING_REQUIRED, "filing features")
    result = frame.copy(deep=True)
    result["accepted_at"] = pd.to_datetime(result["accepted_at"], errors="raise", utc=True)
    result["ticker"] = result["ticker"].astype(str).str.upper().str.strip()
    if "form" in result.columns:
        result["form"] = result["form"].astype(str).str.upper().str.strip()
    metadata = {"accepted_at", "ticker", "form", "period_end"}
    for column in (column for column in result.columns if column not in metadata):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result.duplicated(["ticker", "accepted_at"]).any():
        raise ValueError(
            "Filing features must contain at most one aggregate row per "
            "ticker/acceptance timestamp."
        )
    return result.sort_values(["ticker", "accepted_at"]).reset_index(drop=True)


def validate_predictions(frame: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """Return a validated copy of horizon prediction rows."""
    required = ["date", "ticker", *(f"pred_h{horizon}" for horizon in horizons)]
    require_columns(frame, required, "predictions")
    result = frame.copy(deep=True)
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.tz_localize(None)
    result["ticker"] = result["ticker"].astype(str).str.upper().str.strip()
    for column in required[2:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result.duplicated(["ticker", "date"]).any():
        raise ValueError("Predictions must contain one row per ticker/date.")
    if not np.isfinite(result[required[2:]].to_numpy(dtype=float)).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)
