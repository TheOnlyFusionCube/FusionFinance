"""Independent market-verification head (Increase-Alpha-style).

A compact ensemble on CURATED STRUCTURED features (price/volume sequence, momentum
across windows, volatility, liquidity, distance-from-high, volume surge) that
forecasts multi-horizon RESIDUAL returns (cross-sectionally market-neutral). It is
a distinct error channel: it consumes NO LLM text, embedding, or confidence, so it
cannot become a learned rubber stamp for the LLM. Every feature and label is
point-in-time — features from data on/before t, labels strictly after t.

v1 residualisation removes the market (cross-sectional demean per date). Full
sector/industry/common-factor neutralisation (the paper's target) additionally
needs GICS labels + a factor model; that is a documented extension, not silently
assumed. Standalone cross-sectional alpha on this survivorship-limited S&P
universe is expected to be ~0 (prior sharpe3/fundrank/quality reads) — the head's
job here is calibrated verification, not standalone return.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from alpha.gpu_ml import DeepEnsemble, EnsembleConfig, TorchOOD, assert_rocm_stage

HORIZONS = (1, 3, 5, 10)
_FEATURES = ("mom21", "mom63", "mom126", "mom252", "rev5", "vol21", "dvol", "dist_high", "vsurge")


def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mu = frame.mean()
    sd = frame.std(ddof=0)
    return ((frame - mu) / sd.where(sd > 0, 1.0)).clip(-4, 4).fillna(0.0)


def structured_features(prices: pd.DataFrame, volume: pd.DataFrame, asof: pd.Timestamp,
                        names: list[str]) -> pd.DataFrame:
    """Per-name structured features from data on/before ``asof`` (no lookahead)."""
    for label, frame in (("prices", prices), ("volume", volume)):
        if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates or
                not frame.index.is_monotonic_increasing):
            raise ValueError(f"{label} index must be strictly increasing and unique")
    px = prices.loc[prices.index <= asof]
    vol = volume.loc[volume.index <= asof]
    if len(px) < 253:
        return pd.DataFrame(columns=_FEATURES)
    names = [n for n in names if n in px.columns]
    c = px[names]
    def ret(k): return c.iloc[-1] / c.iloc[-1 - k] - 1.0
    daily = c.pct_change()
    raw = pd.DataFrame(index=pd.Index(names, name="ticker"))
    raw["mom21"], raw["mom63"], raw["mom126"], raw["mom252"] = ret(21), ret(63), ret(126), ret(252)
    raw["rev5"] = ret(5)
    raw["vol21"] = daily.iloc[-21:].std(ddof=0)
    dv = (c * vol[names]).iloc[-21:].median()
    raw["dvol"] = np.log(dv.where(dv > 0, np.nan))
    raw["dist_high"] = c.iloc[-1] / c.iloc[-252:].max()
    v = vol[names]
    raw["vsurge"] = v.iloc[-5:].mean() / v.iloc[-63:].mean().where(v.iloc[-63:].mean() > 0, np.nan)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw.dropna(thresh=6)
    return _zscore(raw[list(_FEATURES)])


def market_features(prices: pd.DataFrame, volume: pd.DataFrame, asof: pd.Timestamp,
                    names: list[str], *, feature_set: str = "basic", **kwargs) -> pd.DataFrame:
    """Build the stable nine-feature set or the expanded PIT feature family."""
    if feature_set == "basic":
        if kwargs:
            raise ValueError("expanded feature options require feature_set='expanded'")
        return structured_features(prices, volume, asof, names)
    if feature_set == "expanded":
        from alpha.features_market import expanded_market_features
        return expanded_market_features(prices, volume, asof, names, **kwargs)
    raise ValueError("feature_set must be 'basic' or 'expanded'")


def residual_forward(prices: pd.DataFrame, asof: pd.Timestamp, horizon: int,
                     names: list[str], *, membership: pd.DataFrame | None = None) -> pd.Series:
    """Market-neutral (cross-sectionally demeaned) forward return over ``horizon``
    sessions strictly AFTER ``asof``. No lookahead."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("price index must be strictly increasing and unique")
    idx = prices.index
    pos = idx.searchsorted(asof, side="right")
    if pos == 0 or pos - 1 + horizon >= len(idx):
        return pd.Series(dtype=float)
    entry, exit_ = idx[pos - 1], idx[pos - 1 + horizon]
    names = list(dict.fromkeys(str(name) for name in names))
    end_by_name = pd.Series(pd.NaT, index=names, dtype="datetime64[ns]")
    if membership is not None:
        required = {"ticker", "start_date", "end_date"}
        if not required.issubset(membership.columns):
            raise ValueError("membership requires ticker, start_date, and end_date")
        frame = membership.loc[:, ["ticker", "start_date", "end_date"]].copy()
        if frame["ticker"].isna().any():
            raise ValueError("membership tickers must be present")
        original_start = frame["start_date"].copy()
        original_end = frame["end_date"].copy()
        frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
        frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
        frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce")
        invalid_start = original_start.notna() & frame["start_date"].isna()
        invalid_end = original_end.notna() & frame["end_date"].isna()
        if (frame["ticker"].eq("").any() or frame["start_date"].isna().any() or
                invalid_start.any() or invalid_end.any()):
            raise ValueError("membership tickers and dates must be valid and present")
        if ((frame["end_date"].notna()) &
                (frame["end_date"] <= frame["start_date"])).any():
            raise ValueError("membership end dates must be after start dates")
        active = frame[
            (frame["start_date"] <= entry) &
            (frame["end_date"].isna() | (frame["end_date"] > entry))
        ].drop_duplicates("ticker", keep="last").set_index("ticker")
        names = list(dict.fromkeys(active.index.tolist()))
        end_by_name = pd.Series(pd.NaT, index=names, dtype="datetime64[ns]")
        end_by_name.update(active["end_date"])
    else:
        names = [name for name in names if name in prices.columns]
    realized = {}
    for name in names:
        if name not in prices.columns:
            realized[name] = np.nan
            continue
        entry_price = prices.at[entry, name]
        if not np.isfinite(entry_price):
            realized[name] = np.nan
            continue
        terminal = exit_
        membership_end = end_by_name.get(name)
        if pd.notna(membership_end) and membership_end <= exit_:
            eligible = idx[(idx >= entry) & (idx <= membership_end)]
        else:
            eligible = idx[(idx >= entry) & (idx <= terminal)]
        observed = prices.loc[eligible, name].dropna()
        realized[name] = (
            float(observed.iloc[-1] / entry_price - 1.0) if len(observed) else np.nan
        )
    r = pd.Series(realized, dtype=float)
    r = r.replace([np.inf, -np.inf], np.nan)
    finite_mean = r.mean(skipna=True)
    return r - finite_mean if len(r) and np.isfinite(finite_mean) else r


@dataclass
class _OOD:
    mean: np.ndarray
    inv_cov: np.ndarray
    def score(self, X: np.ndarray) -> np.ndarray:
        d = X - self.mean
        m = np.einsum("ij,jk,ik->i", d, self.inv_cov, d)
        # squash Mahalanobis^2 to [0,1) vs the feature dimension (chi2 scale)
        return 1.0 - np.exp(-np.maximum(m, 0.0) / (2.0 * X.shape[1]))


@dataclass(frozen=True)
class _CPUProbabilisticModel:
    regressors: tuple[object, ...]
    classifiers: tuple[object | float, ...]
    conditional_variance: object | float


def _date_block_bootstrap_rows(groups: np.ndarray, *, seed: int,
                               block_len: int = 20) -> np.ndarray:
    # Integer-encode first: object/Timestamp groups make np.unique + equality O(n)
    # comparisons on Python objects and dominate CPU demo fits.
    group_values = np.asarray(groups).reshape(-1)
    _, inverse = np.unique(group_values, return_inverse=True)
    inverse = inverse.astype(np.int64, copy=False)
    n_dates = int(inverse.max()) + 1 if len(inverse) else 0
    if n_dates <= 0:
        raise ValueError("date groups cannot be empty")
    length = min(block_len, max(1, n_dates // 3))
    blocks = int(np.ceil(n_dates / length))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_dates, size=blocks)
    order = np.arange(n_dates, dtype=np.int64)
    sampled = np.concatenate([
        order[(int(start) + np.arange(length)) % n_dates] for start in starts
    ])[:n_dates]
    # Build row lists per date once, then index — avoids rescanning all rows per date.
    order_rows = np.argsort(inverse, kind="mergesort")
    sorted_ids = inverse[order_rows]
    boundaries = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1], True])
    rows_by_date = [
        order_rows[boundaries[i]:boundaries[i + 1]] for i in range(n_dates)
    ]
    return np.concatenate([rows_by_date[int(date)] for date in sampled])


def _cpu_tree_iters() -> int:
    # Demo/CPU path can drop tree budget without changing the PIT contract.
    raw = os.environ.get("FUSION_CPU_GBM_ITERS", "").strip()
    if raw:
        return max(10, int(raw))
    return 200


def _oof_squared_residuals(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           *, seed: int) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("conditional variance requires at least two date groups")
    residuals = np.full(len(y), np.nan, dtype=np.float64)
    n_splits = min(2 if _cpu_tree_iters() < 100 else 5, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    max_iter = _cpu_tree_iters()
    for fold, (train, held_out) in enumerate(splitter.split(X, y, groups)):
        model = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.08, max_iter=max_iter,
            l2_regularization=1.0, random_state=seed + fold,
        ).fit(X[train], y[train])
        residuals[held_out] = np.square(y[held_out] - model.predict(X[held_out]))
    if not np.isfinite(residuals).all():
        raise RuntimeError("OOF residual generation left uncovered rows")
    return residuals


def _train_cpu_probabilistic(X: np.ndarray, y: np.ndarray, *, seed0: int,
                             ensemble: int, groups: np.ndarray) -> _CPUProbabilisticModel:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    adverse = (y < 0).astype(np.int8)
    regressors = []
    classifiers: list[object | float] = []
    count = max(2, ensemble)
    max_iter = _cpu_tree_iters()
    for member in range(count):
        rows = _date_block_bootstrap_rows(groups, seed=seed0 + member)
        regressor = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.08, max_iter=max_iter,
            l2_regularization=1.0, random_state=seed0 + member,
        ).fit(X[rows], y[rows])
        regressors.append(regressor)
        member_adverse = adverse[rows]
        if np.unique(member_adverse).size < 2:
            classifiers.append(float(member_adverse[0]))
        else:
            classifier = HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.08, max_iter=max_iter,
                l2_regularization=1.0, random_state=seed0 + member,
            )
            classifier.fit(X[rows], member_adverse)
            classifiers.append(classifier)
    squared_residuals = _oof_squared_residuals(X, y, groups, seed=seed0 + 10_000)
    variance_floor = max(float(np.quantile(squared_residuals, 0.05)), 1e-12)
    variance_target = np.log(np.maximum(squared_residuals, variance_floor))
    if len(X) >= 20:
        conditional_variance: object | float = HistGradientBoostingRegressor(
            max_depth=2, learning_rate=0.08, max_iter=max(20, max_iter // 2),
            l2_regularization=2.0, random_state=seed0 + 20_000,
        ).fit(X, variance_target)
    else:
        conditional_variance = float(np.mean(squared_residuals))
    return _CPUProbabilisticModel(
        regressors=tuple(regressors),
        classifiers=tuple(classifiers),
        conditional_variance=conditional_variance,
    )


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))


def _predict_cpu_probabilistic(model: _CPUProbabilisticModel,
                               X: np.ndarray) -> dict[str, np.ndarray]:
    regression_members = np.stack([member.predict(X) for member in model.regressors])
    classifier_members = []
    for member in model.classifiers:
        if isinstance(member, float):
            classifier_members.append(np.full(len(X), member, dtype=np.float64))
        else:
            classifier_members.append(member.predict_proba(X)[:, 1])
    probabilities = np.stack(classifier_members)
    p_adverse = probabilities.mean(axis=0)
    if isinstance(model.conditional_variance, float):
        conditional_variance = np.full(len(X), model.conditional_variance)
    else:
        conditional_variance = np.exp(model.conditional_variance.predict(X))
    return {
        "p_adverse": p_adverse,
        "mu": regression_members.mean(axis=0),
        "epistemic_var": regression_members.var(axis=0),
        "aleatoric_var": np.maximum(conditional_variance, 1e-12),
        "epistemic_mi": np.maximum(
            _binary_entropy(p_adverse) - _binary_entropy(probabilities).mean(axis=0), 0.0
        ),
    }


class MarketVerifier:
    """Walk-forward multi-horizon residual-return forecaster. Trained only on
    panels whose label window closes on/before the decision date."""

    def __init__(self, ensemble: int = 32, seed0: int = 7,
                 backend: str = "gpu-ensemble", device: str = "cuda:0",
                 hidden: tuple[int, ...] = (256, 256, 128), epochs: int = 200,
                 strict_gpu: bool | None = None, feature_set: str = "basic",
                 calibrator=None, strict_pit: bool | None = None):
        if feature_set not in {"basic", "expanded"}:
            raise ValueError("feature_set must be 'basic' or 'expanded'")
        if backend not in {"gpu-ensemble", "rocm-mlp", "cpu-gbm"}:
            raise ValueError("backend must be 'gpu-ensemble' or 'cpu-gbm'")
        resolved_strict = backend != "cpu-gbm" if strict_gpu is None else strict_gpu
        if backend == "cpu-gbm" and resolved_strict:
            raise ValueError("strict_gpu cannot be combined with the CPU backend")
        self.ensemble = ensemble
        self.seed0 = seed0
        self.backend = backend
        self.device = device
        self.hidden = hidden
        self.epochs = epochs
        self.strict_gpu = resolved_strict
        self.strict_pit = resolved_strict if strict_pit is None else strict_pit
        self.feature_set = feature_set
        if calibrator is not None:
            expected_horizons = {f"{horizon}d" for horizon in HORIZONS}
            if not hasattr(calibrator, "verify_artifact"):
                raise ValueError("verifier requires a verifiable calibration artifact")
            calibrator.verify_artifact(expected_horizons)
            if not getattr(calibrator, "artifact_hash", ""):
                raise ValueError("verifier requires a complete fitted calibration artifact")
            if self.strict_gpu and (
                not getattr(calibrator, "strict_gpu", False) or
                getattr(calibrator, "device", None) != "cuda:0"
            ):
                raise ValueError("strict verifier requires a fitted strict cuda:0 calibrator")
        self.calibrator = calibrator
        self.models: dict[int, object] = {}
        self.ood: _OOD | TorchOOD | None = None
        self.feature_names: tuple[str, ...] = _FEATURES
        self.label_exclusions: dict[int, int] = {}

    def build_features(self, prices: pd.DataFrame, volume: pd.DataFrame,
                       asof: pd.Timestamp, names: list[str], **kwargs) -> pd.DataFrame:
        if self.feature_set == "expanded":
            if self.strict_pit and kwargs.get("strict_pit") is False:
                raise ValueError("strict verifier cannot disable PIT feature gating")
            if "strict_pit" not in kwargs:
                kwargs = {**kwargs, "strict_pit": self.strict_pit}
        return market_features(
            prices, volume, asof, names, feature_set=self.feature_set, **kwargs
        )

    def _train(self, X: np.ndarray, Y: np.ndarray, groups: np.ndarray):
        if self.backend == "cpu-gbm":
            return _train_cpu_probabilistic(
                X, Y, seed0=self.seed0, ensemble=self.ensemble, groups=groups
            )
        if self.backend in {"gpu-ensemble", "rocm-mlp"}:
            assert_rocm_stage(
                "market-verifier-fit", strict=self.strict_gpu,
            )
            model = DeepEnsemble(EnsembleConfig(
                members=self.ensemble, hidden=self.hidden, epochs=self.epochs,
                seed0=self.seed0, device=self.device, strict=self.strict_gpu,
            ))
            model.fit(X, Y, groups=groups)
            params = [
                *((f"W{index}", value) for index, value in enumerate(model._W)),
                *((f"b{index}", value) for index, value in enumerate(model._b)),
            ]
            assert_rocm_stage(
                "market-verifier-trained", params=params, strict=self.strict_gpu,
            )
            return model
        raise ValueError("backend must be 'gpu-ensemble' or 'cpu-gbm'")

    def _predict(self, model, X):
        if self.backend == "cpu-gbm":
            return _predict_cpu_probabilistic(model, X)
        prediction = model.predict(X)
        assert_rocm_stage(
            "market-verifier-predict", strict=self.strict_gpu,
        )
        return prediction

    def _member_adverse_probabilities(self, model, X: np.ndarray) -> np.ndarray:
        if self.backend == "cpu-gbm":
            members = []
            for classifier in model.classifiers:
                if isinstance(classifier, float):
                    members.append(np.full(len(X), classifier, dtype=np.float64))
                else:
                    members.append(classifier.predict_proba(X)[:, 1])
            return np.stack(members)
        import torch

        values = torch.from_numpy(np.ascontiguousarray(X, np.float32)).to(model.device)
        with torch.no_grad():
            logits = model._forward(
                values.unsqueeze(0).expand(model.cfg.members, -1, -1), train=False
            )[..., 0].float()
            return torch.sigmoid(logits).detach().cpu().numpy()

    @staticmethod
    def _unpack_panels(panels: Iterable[tuple]) -> list[tuple[Any, pd.DataFrame, dict[int, pd.Series], dict[int, pd.Timestamp]]]:
        unpacked = []
        for ordinal, panel in enumerate(panels):
            if len(panel) == 2:
                feat, labs = panel
                panel_id = ordinal
                label_ends = {}
            elif len(panel) == 3:
                panel_id, feat, labs = panel
                label_ends = {}
            elif len(panel) == 4:
                panel_id, feat, labs, label_ends = panel
            else:
                raise ValueError("invalid panel tuple length")
            if not isinstance(feat, pd.DataFrame) or not isinstance(labs, dict) or not isinstance(label_ends, dict):
                raise TypeError("invalid verifier panel")
            unpacked.append((panel_id, feat, labs, label_ends))
        return unpacked

    def fit(self, panels: list[tuple], *, decision_cutoff: pd.Timestamp | None = None):
        """panels: list of (features_df, {horizon: residual_fwd Series}) already
        realised (label window closed). Trains one ensemble per horizon."""
        if self.strict_pit and decision_cutoff is None:
            raise ValueError("strict PIT fitting requires an explicit decision_cutoff")
        unpacked = self._unpack_panels(panels)
        if not unpacked:
            self.models = {}
            self.ood = None
            self.label_exclusions = {}
            return self
        feature_names = tuple(str(column) for column in unpacked[0][1].columns)
        if not feature_names:
            raise ValueError("verifier feature set cannot be empty")
        cutoff = pd.Timestamp(decision_cutoff) if decision_cutoff is not None else None
        if cutoff is not None and pd.isna(cutoff):
            raise ValueError("decision_cutoff cannot be NaT")
        local_models: dict[int, object] = {}
        exclusions: dict[int, int] = {}
        Xall: np.ndarray | None = None
        for h in HORIZONS:
            Xs, ys, groups = [], [], []
            for panel_id, feat, labs, label_ends in unpacked:
                if tuple(str(column) for column in feat.columns) != feature_names:
                    raise ValueError("all verifier panels must share ordered feature columns")
                y = labs.get(h)
                if y is None:
                    continue
                if cutoff is not None:
                    label_end = pd.Timestamp(label_ends[h]) if h in label_ends else pd.NaT
                    panel_time = pd.Timestamp(panel_id)
                    if (pd.isna(label_end) or pd.isna(panel_time) or
                            not panel_time < label_end < cutoff):
                        raise ValueError("verifier received a label unavailable at decision cutoff")
                common = feat.index.intersection(y.index)
                if len(common) < 20:
                    continue
                matrix = feat.loc[common].to_numpy(np.float32)
                target = y.loc[common].to_numpy(np.float32)
                valid = np.isfinite(matrix).all(axis=1) & np.isfinite(target)
                exclusions[h] = exclusions.get(h, 0) + int((~valid).sum())
                if int(valid.sum()) < 20:
                    continue
                Xs.append(matrix[valid])
                ys.append(target[valid])
                groups.append(np.repeat(panel_id, int(valid.sum())))
            if not Xs:
                continue
            X = np.vstack(Xs)
            Y = np.concatenate(ys)
            group_ids = np.concatenate(groups)
            # Integer group codes: object/Timestamp ids thrash unique/equality in bagging + OOF.
            _, group_codes = np.unique(group_ids, return_inverse=True)
            local_models[h] = self._train(X, Y, group_codes.astype(np.int64, copy=False))
            if h == HORIZONS[0]:
                Xall = X
        local_ood: _OOD | TorchOOD | None = None
        if Xall is not None and len(Xall):
            if self.backend == "cpu-gbm":
                mu = Xall.mean(0)
                cov = np.cov(Xall.T) + 1e-3 * np.eye(Xall.shape[1])
                local_ood = _OOD(mu, np.linalg.pinv(cov))
            else:
                local_ood = TorchOOD(Xall, device=self.device, strict=self.strict_gpu)
                assert_rocm_stage(
                    "market-verifier-ood", local_ood.mean, local_ood.whitener,
                    strict=self.strict_gpu,
                )
        self.feature_names = feature_names
        self.models = local_models
        self.ood = local_ood
        self.label_exclusions = exclusions
        return self

    def forecast_frame(self, features: pd.DataFrame) -> dict[str, dict]:
        """Batch multi-horizon forecast for many names. One model call per horizon."""
        if not isinstance(features, pd.DataFrame) or features.empty:
            return {}
        missing = [name for name in self.feature_names if name not in features.columns]
        if missing:
            raise ValueError(f"forecast frame missing feature(s): {missing}")
        matrix = features.loc[:, list(self.feature_names)].to_numpy(np.float32)
        valid = np.isfinite(matrix).all(axis=1)
        if not valid.any():
            return {}
        index = features.index[valid]
        X = matrix[valid]
        mi_basis = (
            "calibrated_member_probabilities" if self.calibrator is not None
            else "pre_calibration_member_probabilities"
        )
        horizon_payload: dict[str, dict[str, np.ndarray]] = {}
        for h in HORIZONS:
            model = self.models.get(h)
            if model is None:
                continue
            prediction = self._predict(model, X)
            key = f"{h}d"
            payload = {
                "expected_residual_bps": prediction["mu"] * 1e4,
                "p_adverse": prediction["p_adverse"],
                "epistemic_var": prediction["epistemic_var"],
                "aleatoric_var": prediction["aleatoric_var"],
                "epistemic_mi": prediction["epistemic_mi"],
            }
            if self.calibrator is not None:
                member_probabilities = self._member_adverse_probabilities(model, X)
                calibrated_members = self.calibrator.transform(
                    member_probabilities.reshape(-1), key
                ).reshape(member_probabilities.shape)
                calibrated_mean = calibrated_members.mean(axis=0)
                payload["p_adverse"] = calibrated_mean
                payload["epistemic_mi"] = np.maximum(
                    _binary_entropy(calibrated_mean) -
                    _binary_entropy(calibrated_members).mean(axis=0),
                    0.0,
                )
            horizon_payload[key] = payload
        ood = (
            self.ood.score(X) if self.ood is not None
            else np.ones(len(X), dtype=np.float64)
        )
        calibration_hash = (
            getattr(self.calibrator, "artifact_hash", "") if self.calibrator else ""
        )
        out: dict[str, dict] = {}
        for i, ticker in enumerate(index):
            row = {
                "expected_residual_bps": {}, "p_adverse": {}, "epistemic_var": {},
                "aleatoric_var": {}, "epistemic_mi": {},
                "epistemic_mi_basis": mi_basis,
                "out_of_distribution_score": float(ood[i]),
                "calibration_hash": calibration_hash,
            }
            for key, payload in horizon_payload.items():
                row["expected_residual_bps"][key] = float(payload["expected_residual_bps"][i])
                for name in ("p_adverse", "epistemic_var", "aleatoric_var", "epistemic_mi"):
                    row[name][key] = float(payload[name][i])
            out[str(ticker)] = row
        return out

    def forecast(self, feat_row: pd.Series) -> dict:
        """Multi-horizon residual-return forecast (bps) + adverse prob + OOD for
        ONE name. Used to build a VerifierOutput for a thesis."""
        if not isinstance(feat_row, pd.Series):
            raise TypeError("forecast expects a feature Series")
        batch = self.forecast_frame(feat_row.to_frame().T)
        if not batch:
            raise ValueError("forecast features must be finite")
        return next(iter(batch.values()))
