"""Expanded point-in-time market feature families for the independent verifier."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


MOMENTUM_WINDOWS = (5, 10, 21, 63, 126, 252)
RISK_WINDOWS = (21, 63, 126)
LIQUIDITY_WINDOWS = (5, 21, 63)
EXPANDED_MARKET_FEATURES = tuple(
    [f"mom_{window}" for window in MOMENTUM_WINDOWS]
    + [f"market_rel_mom_{window}" for window in MOMENTUM_WINDOWS]
    + [f"market_beta_{window}" for window in RISK_WINDOWS]
    + [f"idio_vol_{window}" for window in RISK_WINDOWS]
    + ["vol_5", "vol_21", "vol_63", "vol_term_5_21", "vol_term_21_63"]
    + ["downside_semivar_21", "upside_semivar_21", "downside_semivar_63",
       "upside_semivar_63", "skew_21", "skew_63", "tail_freq_21", "tail_freq_63"]
    + [f"turnover_proxy_{window}" for window in LIQUIDITY_WINDOWS]
    + [f"amihud_{window}" for window in LIQUIDITY_WINDOWS]
    + ["roll_spread_21", "roll_spread_63"]
    + ["sector_rel_mom_21", "sector_rel_mom_63", "sector_rel_mom_126",
       "sector_beta_63"]
    + ["dispersion_loading_21", "breadth_alignment_21", "market_corr_21",
       "market_corr_63", "correlation_regime_loading", "staleness_sessions",
       "fresh_within_5d"]
)
SECTOR_FEATURES = (
    "sector_rel_mom_21", "sector_rel_mom_63", "sector_rel_mom_126",
    "sector_beta_63",
)


def _validate_panel(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name} index must be a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError(f"{name} index must be strictly increasing and unique")


def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=0)
    scale = frame.std(axis=0, ddof=0).where(lambda value: value > 0.0)
    return ((frame - mean) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _beta(returns: pd.DataFrame, market: pd.Series, window: int) -> pd.Series:
    stock = returns.iloc[-window:]
    benchmark = market.reindex(stock.index)
    centered_market = benchmark - benchmark.mean()
    variance = float(centered_market.pow(2).mean())
    if not np.isfinite(variance) or variance <= 0.0:
        return pd.Series(np.nan, index=returns.columns, dtype=float)
    centered_stock = stock - stock.mean(axis=0)
    return centered_stock.mul(centered_market, axis=0).mean(axis=0) / variance


def _correlation(returns: pd.DataFrame, market: pd.Series, window: int) -> pd.Series:
    return returns.iloc[-window:].corrwith(market.iloc[-window:])


def _labels_asof(sectors: pd.DataFrame | pd.Series | Mapping[str, str] | None,
                 asof: pd.Timestamp, names: Sequence[str]) -> pd.Series:
    if sectors is None:
        return pd.Series(index=names, dtype=object)
    if isinstance(sectors, pd.DataFrame):
        _validate_panel(sectors, "sectors")
        available = sectors.loc[sectors.index <= asof]
        if available.empty:
            return pd.Series(index=names, dtype=object)
        labels = available.ffill().iloc[-1]
    elif isinstance(sectors, pd.Series):
        labels = sectors
    else:
        labels = pd.Series(dict(sectors), dtype=object)
    return labels.reindex(names)


def _sector_series(returns: pd.DataFrame, labels: pd.Series) -> dict[str, pd.Series]:
    result = {}
    for sector in labels.dropna().unique():
        members = labels.index[labels == sector].intersection(returns.columns)
        if len(members):
            result[str(sector)] = returns[members].mean(axis=1)
    return result


def _record_sector_status(frame: pd.DataFrame, sectors, strict_pit: bool) -> pd.DataFrame:
    if isinstance(sectors, pd.DataFrame):
        status = "enabled_effective_dated"
        disabled = ()
    elif sectors is not None and not strict_pit:
        status = "enabled_non_pit_static_snapshot"
        disabled = ()
    else:
        status = "disabled_no_pit_sector_history"
        disabled = SECTOR_FEATURES
    frame.attrs["sector_feature_status"] = status
    frame.attrs["disabled_features"] = disabled
    return frame


def expanded_market_features(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    asof: pd.Timestamp,
    names: Sequence[str],
    *,
    sectors: pd.DataFrame | pd.Series | Mapping[str, str] | None = None,
    benchmark: str = "SPY",
    strict_pit: bool = True,
) -> pd.DataFrame:
    """Return cross-sectionally standardized features using only rows at/before asof.

    Strict PIT mode disables sector-relative features unless the caller supplies
    effective-dated sector history. A static map can only be used when strict PIT
    is explicitly disabled and is flagged as non-PIT in the returned attrs.
    Close/volume inputs cannot identify a true quoted spread, so the Roll estimate
    is explicitly a close-based spread proxy.
    """
    _validate_panel(prices, "prices")
    _validate_panel(volume, "volume")
    cutoff = pd.Timestamp(asof)
    px_all = prices.loc[prices.index <= cutoff]
    vol_all = volume.loc[volume.index <= cutoff]
    if len(px_all) < max(MOMENTUM_WINDOWS) + 1:
        return _record_sector_status(
            pd.DataFrame(columns=EXPANDED_MARKET_FEATURES, dtype=float),
            sectors, strict_pit,
        )
    requested = list(dict.fromkeys(str(name) for name in names))
    available = [name for name in requested if name in px_all.columns and px_all[name].notna().any()]
    if not available:
        return _record_sector_status(
            pd.DataFrame(
                index=pd.Index([], name="ticker"), columns=EXPANDED_MARKET_FEATURES
            ),
            sectors, strict_pit,
        )

    observed = px_all[available].notna()
    staleness = pd.Series({
        name: len(observed) - 1 - int(np.flatnonzero(observed[name].to_numpy())[-1])
        for name in available
    }, dtype=float)
    px = px_all[available].ffill(limit=5)
    vol = vol_all.reindex(index=px.index, columns=available).fillna(0.0)
    returns = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if benchmark in px_all.columns:
        market = px_all[benchmark].ffill().pct_change(fill_method=None)
    else:
        market = returns.mean(axis=1)
    raw = pd.DataFrame(index=pd.Index(available, name="ticker"), dtype=float)

    market_price = (1.0 + market.fillna(0.0)).cumprod()
    for window in MOMENTUM_WINDOWS:
        raw[f"mom_{window}"] = px.iloc[-1] / px.iloc[-1 - window] - 1.0
        market_momentum = market_price.iloc[-1] / market_price.iloc[-1 - window] - 1.0
        raw[f"market_rel_mom_{window}"] = raw[f"mom_{window}"] - market_momentum
    for window in RISK_WINDOWS:
        beta = _beta(returns, market, window)
        residual = returns.iloc[-window:].sub(
            market.iloc[-window:].to_numpy()[:, None] * beta.to_numpy()[None, :]
        )
        raw[f"market_beta_{window}"] = beta
        raw[f"idio_vol_{window}"] = residual.std(axis=0, ddof=0)

    volatility = {
        window: returns.iloc[-window:].std(axis=0, ddof=0)
        for window in LIQUIDITY_WINDOWS
    }
    for window, values in volatility.items():
        raw[f"vol_{window}"] = values
    raw["vol_term_5_21"] = volatility[5] / volatility[21].replace(0.0, np.nan)
    raw["vol_term_21_63"] = volatility[21] / volatility[63].replace(0.0, np.nan)

    for window in (21, 63):
        sample = returns.iloc[-window:]
        raw[f"downside_semivar_{window}"] = sample.clip(upper=0.0).pow(2).mean().pow(0.5)
        raw[f"upside_semivar_{window}"] = sample.clip(lower=0.0).pow(2).mean().pow(0.5)
        raw[f"skew_{window}"] = sample.skew(axis=0)
        threshold = sample.std(axis=0, ddof=0) * 2.0
        raw[f"tail_freq_{window}"] = sample.abs().gt(threshold, axis=1).mean(axis=0)

    dollar_volume = px * vol
    volume_baseline = vol.iloc[-63:].mean(axis=0).replace(0.0, np.nan)
    for window in LIQUIDITY_WINDOWS:
        raw[f"turnover_proxy_{window}"] = vol.iloc[-window:].mean(axis=0) / volume_baseline
        raw[f"amihud_{window}"] = (
            returns.iloc[-window:].abs() /
            dollar_volume.iloc[-window:].replace(0.0, np.nan)
        ).mean(axis=0)
    price_change = px.diff()
    for window in (21, 63):
        current = price_change.iloc[-window:]
        lagged = price_change.shift(1).iloc[-window:]
        covariance = ((current - current.mean()) * (lagged - lagged.mean())).mean()
        raw[f"roll_spread_{window}"] = (
            2.0 * np.sqrt((-covariance).clip(lower=0.0)) / px.iloc[-1].replace(0.0, np.nan)
        )

    dated_sector_history = isinstance(sectors, pd.DataFrame)
    sector_input = sectors if (not strict_pit or dated_sector_history) else None
    labels = _labels_asof(sector_input, cutoff, available)
    sector_returns = _sector_series(returns, labels)
    for window in (21, 63, 126):
        values = pd.Series(np.nan, index=available, dtype=float)
        for name in available:
            sector = labels.get(name)
            if pd.notna(sector) and str(sector) in sector_returns:
                sector_return = sector_returns[str(sector)].iloc[-window:]
                values[name] = raw.loc[name, f"mom_{window}"] - float(
                    (1.0 + sector_return.fillna(0.0)).prod() - 1.0
                )
        raw[f"sector_rel_mom_{window}"] = values
    sector_beta = pd.Series(np.nan, index=available, dtype=float)
    for name in available:
        sector = labels.get(name)
        if pd.notna(sector) and str(sector) in sector_returns:
            sector_beta[name] = _beta(
                returns[[name]], sector_returns[str(sector)], 63
            ).iloc[0]
    raw["sector_beta_63"] = sector_beta

    dispersion = returns.iloc[-21:].std(axis=1, ddof=0).mean()
    raw["dispersion_loading_21"] = volatility[21] / max(float(dispersion), 1e-12)
    breadth_sign = (returns.iloc[-21:] > 0.0).mean(axis=1) >= 0.5
    raw["breadth_alignment_21"] = returns.iloc[-21:].gt(0.0).eq(breadth_sign, axis=0).mean()
    raw["market_corr_21"] = _correlation(returns, market, 21)
    raw["market_corr_63"] = _correlation(returns, market, 63)
    cross_market = returns.iloc[-63:].mean(axis=1)
    raw["correlation_regime_loading"] = returns.iloc[-63:].corrwith(cross_market)
    raw["staleness_sessions"] = staleness
    raw["fresh_within_5d"] = (staleness <= 5).astype(float)

    raw = raw.reindex(columns=EXPANDED_MARKET_FEATURES)
    result = _cross_sectional_zscore(raw).astype(np.float32)
    return _record_sector_status(result, sectors, strict_pit)
