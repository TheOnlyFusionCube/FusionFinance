"""Learned jurors: published ML asset-pricing recipes, trained walk-forward.

Each learned juror trains only on (context, label) rows whose label window
closed before the decision it will later score; the committee supplies those
rows. Until a juror has been fitted it abstains.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha.verifier.committee.base import JurorSpec, JuryContext, Provenance, cross_sectional_z

_WINDOWS = (5, 10, 20, 30, 60)


def alpha158_features(context: JuryContext) -> pd.DataFrame:
    """A documented subset of Qlib's Alpha158 price/volume factor family.

    Rolling operators (ROC, MA, STD, MAX, MIN, QTLU, QTLD, RSV, IMAX, IMIN,
    CNTP, SUMP, BETA-as-slope, VMA, VSTD, CORR of price and log volume) over
    5/10/20/30/60-session windows, each normalised by the latest close or
    volume as in Qlib's definitions.
    """
    cache = context.extras.setdefault("_cache", {})
    if "alpha158" in cache:
        return cache["alpha158"]
    close = context.closes
    if len(close) < 61:
        frame = pd.DataFrame(index=context.tickers)
        cache["alpha158"] = frame
        return frame
    last = close.iloc[-1]
    returns = context.returns()
    columns: dict[str, pd.Series] = {}
    volume = context.volume
    for w in _WINDOWS:
        window = close.iloc[-w:]
        columns[f"ROC{w}"] = close.iloc[-w - 1] / last
        columns[f"MA{w}"] = window.mean() / last
        columns[f"STD{w}"] = window.std() / last
        columns[f"MAX{w}"] = window.max() / last
        columns[f"MIN{w}"] = window.min() / last
        columns[f"QTLU{w}"] = window.quantile(0.8) / last
        columns[f"QTLD{w}"] = window.quantile(0.2) / last
        span = (window.max() - window.min()).replace(0, np.nan)
        columns[f"RSV{w}"] = (last - window.min()) / span
        columns[f"IMAX{w}"] = pd.Series(np.argmax(window.to_numpy(), axis=0) / w, index=close.columns)
        columns[f"IMIN{w}"] = pd.Series(np.argmin(window.to_numpy(), axis=0) / w, index=close.columns)
        recent = returns.iloc[-w:]
        columns[f"CNTP{w}"] = (recent > 0).mean()
        gains = recent.clip(lower=0).sum()
        columns[f"SUMP{w}"] = gains / (recent.abs().sum().replace(0, np.nan))
        x = np.arange(w) - (w - 1) / 2
        columns[f"BETA{w}"] = pd.Series((x @ window.to_numpy()) / (x @ x), index=close.columns) / last
        if volume is not None:
            vwin = volume.iloc[-w:]
            vlast = volume.iloc[-1].replace(0, np.nan)
            columns[f"VMA{w}"] = vwin.mean() / vlast
            columns[f"VSTD{w}"] = vwin.std() / vlast
            columns[f"CORR{w}"] = window.corrwith(np.log1p(vwin))
    frame = pd.DataFrame(columns).replace([np.inf, -np.inf], np.nan)
    cache["alpha158"] = frame
    return frame


@dataclass
class _GradientBoostedJuror:
    spec: JurorSpec
    max_iter: int = 60
    seed: int = 7
    _model: object | None = None
    _columns: tuple[str, ...] = ()

    def features(self, context: JuryContext) -> pd.DataFrame:
        raise NotImplementedError

    def target(self, context: JuryContext, label: pd.Series, path: pd.DataFrame | None) -> pd.Series:
        return label

    def fit(self, rows: list[tuple[JuryContext, pd.Series, pd.DataFrame | None]]) -> None:
        """Train on (context, closed label, closed forward price path) rows."""
        from sklearn.ensemble import HistGradientBoostingRegressor

        frames, targets = [], []
        for context, label, path in rows:
            features = self.features(context)
            if features.empty:
                continue
            target = self.target(context, label, path).reindex(features.index)
            keep = target.notna()
            if keep.sum() < 10:
                continue
            frames.append(features[keep])
            # Cross-sectional rank target, as in the ML asset-pricing literature.
            targets.append(cross_sectional_z(target[keep]))
        if len(frames) < 10:
            return
        matrix = pd.concat(frames)
        self._columns = tuple(matrix.columns)
        self._model = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=self.max_iter,
            l2_regularization=1.0, random_state=self.seed,
        ).fit(matrix.to_numpy(np.float64), pd.concat(targets).to_numpy(np.float64))

    def score(self, context: JuryContext) -> pd.Series:
        if self._model is None:
            return pd.Series(np.nan, index=context.tickers)
        features = self.features(context).reindex(columns=list(self._columns))
        if features.empty:
            return pd.Series(np.nan, index=context.tickers)
        prediction = self._model.predict(features.to_numpy(np.float64))
        return pd.Series(prediction, index=features.index).reindex(context.tickers)


@dataclass
class Alpha158Boosting(_GradientBoostedJuror):
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="alpha158_boosting", family="ml", needs=("prices", "volume"), learned=True,
        provenance=Provenance(
            citation="Yang, Liu, Zhou, Bian & Liu (2020), Qlib: An AI-oriented Quantitative "
                     "Investment Platform, arXiv:2009.11189 (Alpha158 factors with a "
                     "gradient-boosted tree baseline)",
            institution="Microsoft Research Asia",
            url="https://github.com/microsoft/qlib",
            notes="sklearn histogram GBRT stands in for LightGBM; no new dependency",
        ),
    ))

    def features(self, context: JuryContext) -> pd.DataFrame:
        return alpha158_features(context)


@dataclass
class GuKellyXiuTrees(_GradientBoostedJuror):
    """GBRT on a panel of firm characteristics built from the other jurors' raw signals."""

    characteristic_jurors: tuple = ()
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="gu_kelly_xiu_trees", family="ml", needs=("prices", "fundamentals"), learned=True,
        provenance=Provenance(
            citation="Gu, Kelly & Xiu (2020), Empirical Asset Pricing via Machine Learning, "
                     "Review of Financial Studies",
            institution="Chicago Booth / Yale (Kelly also AQR Capital Management)",
            notes="regression trees on rank-normalised characteristics",
        ),
    ))

    def features(self, context: JuryContext) -> pd.DataFrame:
        cache = context.extras.setdefault("_cache", {})
        columns = {}
        for juror in self.characteristic_jurors:
            key = ("raw", juror.spec.name)
            if key not in cache:
                cache[key] = juror.score(context)
            columns[juror.spec.name] = cross_sectional_z(cache[key])
        frame = pd.DataFrame(columns, index=context.tickers)
        return frame if frame.notna().any().any() else pd.DataFrame(index=context.tickers)


@dataclass
class TripleBarrierClassifier(_GradientBoostedJuror):
    """Predicts which barrier is hit first; labels follow the triple-barrier method."""

    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="triple_barrier", family="ml", needs=("prices", "volume"), learned=True,
        provenance=Provenance(
            citation="Lopez de Prado (2018), Advances in Financial Machine Learning, "
                     "Wiley, ch. 3 (triple-barrier labelling)",
            institution="Marcos Lopez de Prado (practitioner-academic; Cornell University)",
            notes="volatility-scaled profit-take and stop-loss barriers with a vertical time barrier",
        ),
    ))
    horizon: int = 5
    width: float = 1.5

    def features(self, context: JuryContext) -> pd.DataFrame:
        return alpha158_features(context)

    def target(self, context: JuryContext, label: pd.Series, path: pd.DataFrame | None) -> pd.Series:
        if path is None or path.empty:
            return label
        daily_vol = context.returns().iloc[-20:].std()
        barrier = self.width * daily_vol * np.sqrt(self.horizon)
        cumulative = (path / context.closes.iloc[-1] - 1.0).to_numpy(dtype=float)
        outcome = {}
        for column, ticker in enumerate(path.columns):
            series = cumulative[:, column]
            finite = series[np.isfinite(series)]
            if not len(finite):
                outcome[ticker] = np.nan
                continue
            level = float(barrier[ticker])
            ups = np.flatnonzero(finite >= level)
            downs = np.flatnonzero(finite <= -level)
            first_up = ups[0] if len(ups) else None
            first_down = downs[0] if len(downs) else None
            if first_up is not None and (first_down is None or first_up < first_down):
                outcome[ticker] = 1.0
            elif first_down is not None:
                outcome[ticker] = -1.0
            else:          # vertical barrier: half-weight sign of the final return
                outcome[ticker] = 0.5 * float(np.sign(finite[-1]))
        return pd.Series(outcome)


@dataclass
class MarketHeadJuror:
    """The repository's own walk-forward ensemble, seated as one juror among many."""

    horizon_key: str = "5d"
    spec: JurorSpec = field(default_factory=lambda: JurorSpec(
        name="fusion_market_head", family="ml", needs=("prices", "volume"), learned=True,
        provenance=Provenance(
            citation="FusionFinance MarketVerifier: bootstrapped GBRT ensemble on structured "
                     "price/volume features with calibrated adverse probability and OOD score",
            institution="FusionFinance (this repository)",
        ),
    ))

    def score(self, context: JuryContext) -> pd.Series:
        forecasts = context.extras.get("market_forecast") or {}
        return pd.Series({
            ticker: 0.5 - row["p_adverse"][self.horizon_key]
            for ticker, row in forecasts.items() if self.horizon_key in row.get("p_adverse", {})
        }, dtype=float).reindex(context.tickers)


__all__ = [
    "Alpha158Boosting", "GuKellyXiuTrees", "MarketHeadJuror", "TripleBarrierClassifier",
    "alpha158_features",
]
