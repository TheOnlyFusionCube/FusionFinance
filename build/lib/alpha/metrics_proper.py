"""Proper probabilistic scores and multiplicity-aware Sharpe diagnostics."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd


def _arrays(*values) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value).reshape(-1) for value in values)
    if not arrays or len({len(value) for value in arrays}) != 1 or not len(arrays[0]):
        raise ValueError("metric inputs must be non-empty and equally sized")
    if any(not np.isfinite(value.astype(float)).all() for value in arrays):
        raise ValueError("metric inputs must be finite")
    return arrays


def _date_average(loss: np.ndarray, date_ids=None) -> float:
    if date_ids is None:
        return float(np.mean(loss, dtype=np.float64))
    dates = np.asarray(date_ids).reshape(-1)
    if len(dates) != len(loss):
        raise ValueError("date_ids must match score rows")
    if bool(pd.isna(dates).any()):
        raise ValueError("date_ids cannot be missing")
    return float(np.mean([loss[dates == date].mean() for date in np.unique(dates)]))


def date_averaged_log_loss(outcomes, probabilities, date_ids=None) -> float:
    outcomes, probabilities = _arrays(outcomes, probabilities)
    if not np.isin(outcomes, (0, 1)).all():
        raise ValueError("binary outcomes must be 0 or 1")
    if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
        raise ValueError("probabilities must be in [0, 1]")
    probability = np.clip(probabilities.astype(float), 1e-12, 1.0 - 1e-12)
    loss = -(outcomes * np.log(probability) + (1.0 - outcomes) * np.log(1.0 - probability))
    return _date_average(loss, date_ids)


def date_averaged_brier(outcomes, probabilities, date_ids=None) -> float:
    outcomes, probabilities = _arrays(outcomes, probabilities)
    if not np.isin(outcomes, (0, 1)).all():
        raise ValueError("binary outcomes must be 0 or 1")
    if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
        raise ValueError("probabilities must be in [0, 1]")
    return _date_average(np.square(probabilities - outcomes), date_ids)


def _normal_pdf(values: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(values)) / math.sqrt(2.0 * math.pi)


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    erf = np.frompyfunc(math.erf, 1, 1)
    return 0.5 * (1.0 + erf(values / math.sqrt(2.0)).astype(float))


def _gaussian_inputs(outcomes, means, scales) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outcomes, means, scales = _arrays(outcomes, means, scales)
    outcomes = outcomes.astype(float)
    means = means.astype(float)
    scales = scales.astype(float)
    if not (scales > 0.0).all():
        raise ValueError("Gaussian scales must be positive")
    return outcomes, means, scales


def gaussian_nll(outcomes, means, scales, date_ids=None) -> float:
    outcomes, means, scales = _gaussian_inputs(outcomes, means, scales)
    z = (outcomes - means) / scales
    loss = np.log(scales) + 0.5 * np.square(z) + 0.5 * math.log(2.0 * math.pi)
    return _date_average(loss, date_ids)


def gaussian_crps(outcomes, means, scales, date_ids=None) -> float:
    outcomes, means, scales = _gaussian_inputs(outcomes, means, scales)
    z = (outcomes - means) / scales
    score = scales * (
        z * (2.0 * _normal_cdf(z) - 1.0) + 2.0 * _normal_pdf(z) - 1.0 / math.sqrt(math.pi)
    )
    return _date_average(score, date_ids)


def pit_histogram(outcomes, means, scales, *, bins: int = 10) -> dict[str, np.ndarray]:
    if bins <= 1:
        raise ValueError("PIT histogram needs at least two bins")
    outcomes, means, scales = _gaussian_inputs(outcomes, means, scales)
    pit = np.clip(_normal_cdf((outcomes - means) / scales), 0.0, 1.0)
    counts, edges = np.histogram(pit, bins=np.linspace(0.0, 1.0, bins + 1))
    return {"pit": pit, "counts": counts, "edges": edges}


def equal_mass_reliability(outcomes, probabilities, *, bins: int = 10) -> list[dict[str, float | int]]:
    outcomes, probabilities = _arrays(outcomes, probabilities)
    if bins <= 0:
        raise ValueError("reliability bins must be positive")
    if not np.isin(outcomes, (0, 1)).all() or not (
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ).all():
        raise ValueError("invalid reliability inputs")
    order = np.argsort(probabilities, kind="mergesort")
    result = []
    for indices in np.array_split(order, min(bins, len(order))):
        result.append({
            "count": int(len(indices)),
            "mean_probability": float(probabilities[indices].mean()),
            "event_rate": float(outcomes[indices].mean()),
            "lower": float(probabilities[indices].min()),
            "upper": float(probabilities[indices].max()),
        })
    return result


def deflated_sharpe_ratio(
    returns,
    *,
    trial_sharpes=None,
    trials: int = 1,
    sharpe_dispersion: float | None = None,
) -> float:
    """Probability a per-period Sharpe exceeds cross-trial selection luck."""
    values = np.asarray(returns, dtype=float).reshape(-1)
    if len(values) < 3 or not np.isfinite(values).all() or trials <= 0:
        raise ValueError("DSR requires finite returns, n>=3, and positive trials")
    if trial_sharpes is not None:
        observed_trials = np.asarray(trial_sharpes, dtype=float).reshape(-1)
        if len(observed_trials) < 2 or not np.isfinite(observed_trials).all():
            raise ValueError("trial_sharpes must contain at least two finite Sharpes")
        if trials != 1 and trials != len(observed_trials):
            raise ValueError("trials conflicts with trial_sharpes length")
        if sharpe_dispersion is not None:
            raise ValueError("provide trial_sharpes or sharpe_dispersion, not both")
        trials = len(observed_trials)
        dispersion = float(observed_trials.std(ddof=1))
    elif trials > 1:
        if sharpe_dispersion is None:
            raise ValueError("multiple trials require cross-trial Sharpe dispersion")
        dispersion = float(sharpe_dispersion)
    else:
        if sharpe_dispersion is not None:
            raise ValueError("Sharpe dispersion requires trials > 1")
        dispersion = 0.0
    if not math.isfinite(dispersion) or dispersion < 0.0:
        raise ValueError("Sharpe dispersion must be finite and nonnegative")
    standard_deviation = float(values.std(ddof=1))
    if standard_deviation <= 0.0:
        raise ValueError("DSR is undefined for constant returns")
    sharpe = float(values.mean() / standard_deviation)
    centered = (values - values.mean()) / standard_deviation
    skewness = float(np.mean(np.power(centered, 3)))
    kurtosis = float(np.mean(np.power(centered, 4)))
    sharpe_variance = max(
        (1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe ** 2) /
        (len(values) - 1),
        1e-16,
    )
    expected_max = 0.0
    if trials > 1:
        gamma = 0.5772156649015329
        normal = NormalDist()
        expected_max = dispersion * (
            (1.0 - gamma) * normal.inv_cdf(1.0 - 1.0 / trials) +
            gamma * normal.inv_cdf(1.0 - 1.0 / (trials * math.e))
        )
    return float(
        NormalDist().cdf((sharpe - expected_max) / math.sqrt(sharpe_variance))
    )
