"""Point-in-time filing availability, joining, decay, and ridge fusion.

Adapted from filing-alpha-cleanroom 0.6.0 under the MIT License.  See NOTICE.
This module intentionally does not parse SEC's raw, timezone-naive ``accepted``
field; callers must supply an explicit timezone-aware acceptance timestamp.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class FilingAvailabilityConfig:
    """Availability policy for an explicitly after-close signal session."""

    cutoff_hour_eastern: int = 16
    cutoff_minute_eastern: int = 0
    eastern_timezone: str = "America/New_York"

    def __post_init__(self) -> None:
        if not 0 <= self.cutoff_hour_eastern <= 23:
            raise ValueError("cutoff_hour_eastern must be in [0, 23]")
        if not 0 <= self.cutoff_minute_eastern <= 59:
            raise ValueError("cutoff_minute_eastern must be in [0, 59]")
        try:
            ZoneInfo(self.eastern_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {self.eastern_timezone}") from exc


def _timezone_aware_utc(values: pd.Series, *, name: str) -> pd.Series:
    timestamps: list[pd.Timestamp] = []
    for value in values:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError(f"{name} cannot contain missing timestamps")
        if timestamp.tzinfo is None:
            raise ValueError(
                f"{name} must contain timezone-aware timestamps; localize the source "
                "timezone explicitly before converting to UTC"
            )
        timestamps.append(timestamp.tz_convert("UTC"))
    return pd.Series(pd.DatetimeIndex(timestamps), index=values.index, name=name)


def map_filings_to_available_sessions(
    market: pd.DataFrame,
    filings: pd.DataFrame,
    config: FilingAvailabilityConfig | None = None,
    accepted_at_column: str = "accepted_at",
) -> pd.DataFrame:
    """Map each filing copy to the first after-close signal session that knows it.

    A filing accepted at or before the configured cutoff is available for that
    session's *after-close* signal.  A later filing becomes available on the
    next observed trading session.  This does not authorize a same-close fill.
    """
    policy = config or FilingAvailabilityConfig()
    required_market = {"date", "ticker"}
    required_filings = {accepted_at_column, "ticker"}
    if not required_market.issubset(market.columns):
        raise ValueError(f"market must contain {sorted(required_market)}")
    if not required_filings.issubset(filings.columns):
        raise ValueError(f"filings must contain {sorted(required_filings)}")
    sessions = market[["date", "ticker"]].copy(deep=True)
    sessions["date"] = pd.to_datetime(sessions["date"], errors="raise").dt.tz_localize(None)
    sessions = sessions.drop_duplicates().sort_values(["ticker", "date"])
    output: list[pd.DataFrame] = []
    cutoff = time(policy.cutoff_hour_eastern, policy.cutoff_minute_eastern)
    for ticker, ticker_filings in filings.groupby("ticker", sort=False):
        ticker_sessions = sessions.loc[
            sessions["ticker"] == ticker,
            "date",
        ].sort_values().to_numpy()
        if len(ticker_sessions) == 0:
            continue
        part = ticker_filings.copy(deep=True)
        accepted_utc = _timezone_aware_utc(part[accepted_at_column], name=accepted_at_column)
        accepted_eastern = accepted_utc.dt.tz_convert(policy.eastern_timezone)
        local_dates = accepted_eastern.dt.tz_localize(None).dt.normalize()
        after_cutoff = accepted_eastern.dt.time > cutoff
        candidate_dates = local_dates + pd.to_timedelta(after_cutoff.astype(int), unit="D")
        available_dates: list[pd.Timestamp | pd.NaT] = []
        for candidate in candidate_dates:
            position = int(
                np.searchsorted(ticker_sessions, np.datetime64(candidate), side="left")
            )
            available_dates.append(
                pd.Timestamp(ticker_sessions[position])
                if position < len(ticker_sessions)
                else pd.NaT
            )
        part["available_date"] = available_dates
        part["accepted_at_eastern"] = accepted_eastern.astype(str)
        output.append(part)
    if not output:
        return filings.assign(available_date=pd.NaT).iloc[0:0]
    return (
        pd.concat(output, ignore_index=True)
        .sort_values(["ticker", "available_date", accepted_at_column])
        .reset_index(drop=True)
    )


def point_in_time_filing_join(
    market: pd.DataFrame,
    filings: pd.DataFrame,
    feature_columns: Iterable[str],
    config: FilingAvailabilityConfig | None = None,
    accepted_at_column: str = "accepted_at",
) -> pd.DataFrame:
    """Backward-as-of join filing features onto a copy of the market panel."""
    mapped = map_filings_to_available_sessions(
        market,
        filings,
        config,
        accepted_at_column,
    )
    features = list(feature_columns)
    missing = [column for column in features if column not in mapped.columns]
    if missing:
        raise ValueError(f"Missing filing features: {missing}")
    left = market.copy(deep=True)
    left["date"] = pd.to_datetime(left["date"], errors="raise").dt.tz_localize(None)
    right_columns = ["ticker", "available_date", accepted_at_column, *features]
    right = mapped[right_columns].dropna(subset=["available_date"]).copy()
    pieces: list[pd.DataFrame] = []
    for ticker, ticker_market in left.groupby("ticker", sort=False):
        ticker_right = right.loc[right["ticker"] == ticker].sort_values(
            ["available_date", accepted_at_column],
            kind="stable",
        )
        ticker_left = ticker_market.sort_values("date")
        if ticker_right.empty:
            merged = ticker_left.copy()
            merged["available_date"] = pd.NaT
            merged[accepted_at_column] = pd.NaT
            for column in features:
                merged[column] = np.nan
        else:
            merged = pd.merge_asof(
                ticker_left,
                ticker_right.drop(columns="ticker"),
                left_on="date",
                right_on="available_date",
                direction="backward",
                allow_exact_matches=True,
            )
        pieces.append(merged)
    if not pieces:
        return left.assign(
            available_date=pd.NaT,
            **{accepted_at_column: pd.NaT, **{column: np.nan for column in features}},
        )
    result = pd.concat(pieces, ignore_index=True).sort_values(["ticker", "date"])
    result["filing_age_sessions"] = result.groupby(
        ["ticker", "available_date"],
        dropna=False,
    ).cumcount()
    result.loc[result["available_date"].isna(), "filing_age_sessions"] = np.nan
    return result.reset_index(drop=True)


def add_filing_decay_features(
    panel: pd.DataFrame,
    feature_columns: Iterable[str],
    half_life_sessions: float = 63.0,
) -> pd.DataFrame:
    """Return a copy with exponential filing-age decay features."""
    if half_life_sessions <= 0:
        raise ValueError("half_life_sessions must be positive")
    if "filing_age_sessions" not in panel.columns:
        raise ValueError("panel must contain filing_age_sessions")
    result = panel.copy(deep=True)
    age = pd.to_numeric(result["filing_age_sessions"], errors="coerce")
    decay = np.exp(-np.log(2.0) * age / float(half_life_sessions))
    result["filing_decay"] = decay
    for column in feature_columns:
        if column not in result.columns:
            raise ValueError(f"panel is missing filing feature: {column}")
        result[f"{column}_decayed"] = pd.to_numeric(
            result[column],
            errors="coerce",
        ) * decay
    return result


def robust_cross_sectional_standardize(
    frame: pd.DataFrame,
    columns: Iterable[str],
    group_columns: tuple[str, ...] = ("date",),
    clip: float = 5.0,
) -> pd.DataFrame:
    """Return median/MAD robust z-scores within each cross-section."""
    if clip <= 0:
        raise ValueError("clip must be positive")
    requested = [*group_columns, *columns]
    missing = sorted(set(requested).difference(frame.columns))
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")
    result = frame.copy(deep=True)
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        grouped = result.assign(_value=values).groupby(
            list(group_columns),
            dropna=False,
        )["_value"]
        median = grouped.transform("median")
        absolute_deviation = (values - median).abs()
        mad = result.assign(_mad=absolute_deviation).groupby(
            list(group_columns),
            dropna=False,
        )["_mad"].transform("median")
        scale = (1.4826 * mad).replace(0.0, np.nan)
        result[f"{column}_robust_z"] = (
            (values - median) / scale
        ).clip(-clip, clip).fillna(0.0)
    return result


@dataclass(frozen=True)
class FittedFusionModel:
    """Fitted ridge model and immutable feature-column contract."""

    market_columns: tuple[str, ...]
    filing_columns: tuple[str, ...]
    model: Ridge

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        columns = [*self.market_columns, *self.filing_columns]
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(f"frame is missing model columns: {missing}")
        return self.model.predict(frame[columns].fillna(0.0).to_numpy(dtype=float))


def fit_fusion_model(
    train: pd.DataFrame,
    target_column: str,
    market_columns: Iterable[str],
    filing_columns: Iterable[str],
    ridge_alpha: float = 10.0,
) -> FittedFusionModel:
    """Fit the archive's transparent ridge baseline on copied numeric arrays."""
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha cannot be negative")
    market = tuple(market_columns)
    filing = tuple(filing_columns)
    columns = [*market, *filing]
    missing = sorted(set([target_column, *columns]).difference(train.columns))
    if missing:
        raise ValueError(f"train is missing required columns: {missing}")
    clean = train.dropna(subset=[target_column]).copy(deep=True)
    if clean.empty:
        raise ValueError("train must contain at least one finite target")
    target = pd.to_numeric(clean[target_column], errors="coerce")
    finite_target = np.isfinite(target.to_numpy(dtype=float))
    clean = clean.loc[finite_target]
    target = target.loc[finite_target]
    if clean.empty:
        raise ValueError("train must contain at least one finite target")
    model = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    model.fit(
        clean[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float),
        target.to_numpy(dtype=float),
    )
    return FittedFusionModel(market, filing, model)
