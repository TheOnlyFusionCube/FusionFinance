"""Ensemble-member ablation using identical data, groups, seeds, and proper scores."""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Sequence

import numpy as np
import pandas as pd

from alpha.gpu_ml import DeepEnsemble, EnsembleConfig
from alpha.metrics_proper import (
    date_averaged_brier,
    date_averaged_log_loss,
    deflated_sharpe_ratio,
    gaussian_crps,
    gaussian_nll,
)


MEMBER_GRID = (4, 8, 16, 32, 64, 96)


def select_smallest_tied(rows: Sequence[dict], *, tolerance: float = 0.0) -> int:
    successful = [row for row in rows if "log_loss" in row]
    if not successful:
        raise ValueError("no successful ablation rows")
    if tolerance < 0.0:
        raise ValueError("tie tolerance cannot be negative")
    best = min(float(row["log_loss"]) for row in successful)
    tied = [int(row["members"]) for row in successful
            if float(row["log_loss"]) <= best + tolerance]
    return min(tied)


def run_m_ablation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: np.ndarray,
    X_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    evaluation_groups: np.ndarray,
    *,
    config: EnsembleConfig,
    member_grid: Sequence[int] = MEMBER_GRID,
    tie_tolerance: float = 0.0,
) -> dict:
    grid = tuple(int(member) for member in member_grid)
    if grid != MEMBER_GRID:
        raise ValueError(f"member grid must be exactly {MEMBER_GRID}")
    numeric_arrays = tuple(np.asarray(value) for value in (
        X_train, y_train, X_evaluation, y_evaluation,
    ))
    if len(X_train) != len(y_train) or len(X_train) != len(train_groups):
        raise ValueError("training arrays must align")
    if len(X_evaluation) != len(y_evaluation) or len(X_evaluation) != len(evaluation_groups):
        raise ValueError("evaluation arrays must align")
    if any(not np.isfinite(value.astype(float)).all() for value in numeric_arrays):
        raise ValueError("ablation arrays must be finite")
    for groups in (train_groups, evaluation_groups):
        flattened = np.asarray(groups).reshape(-1)
        if bool(pd.isna(flattened).any()):
            raise ValueError("ablation groups cannot be missing")
    if config.strict:
        from alpha.gpu_ml import assert_rocm_stage
        assert_rocm_stage("gpu-m-ablation", strict=True)

    adverse = (np.asarray(y_evaluation) < 0.0).astype(np.float32)
    rows = []
    strategy_paths = []
    for members in grid:
        model = DeepEnsemble(replace(config, members=members))
        fit_receipt = model.fit(X_train, y_train, groups=train_groups)
        prediction = model.predict(X_evaluation)
        scale = np.sqrt(
            np.maximum(prediction["epistemic_var"] + prediction["aleatoric_var"], 1e-12)
        )
        row_returns = np.sign(prediction["mu"]) * np.asarray(y_evaluation)
        groups = np.asarray(evaluation_groups).reshape(-1)
        strategy_returns = np.asarray([
            row_returns[groups == group].mean() for group in np.unique(groups)
        ])
        strategy_paths.append(strategy_returns)
        row = {
            "members": members,
            "log_loss": date_averaged_log_loss(
                adverse, prediction["p_adverse"], evaluation_groups
            ),
            "brier": date_averaged_brier(
                adverse, prediction["p_adverse"], evaluation_groups
            ),
            "crps": gaussian_crps(
                y_evaluation, prediction["mu"], scale, evaluation_groups
            ),
            "gaussian_nll": gaussian_nll(
                y_evaluation, prediction["mu"], scale, evaluation_groups
            ),
            "fit": fit_receipt,
        }
        rows.append(row)
    trial_sharpes = np.asarray([
        values.mean() / values.std(ddof=1) for values in strategy_paths
    ])
    scored_rows = [
        {
            **row,
            "deflated_sharpe_ratio": deflated_sharpe_ratio(
                values, trial_sharpes=trial_sharpes
            ),
        }
        for row, values in zip(rows, strategy_paths)
    ]
    return {
        "member_grid": list(grid),
        "tie_tolerance": float(tie_tolerance),
        "selected_members": select_smallest_tied(scored_rows, tolerance=tie_tolerance),
        "config": asdict(config),
        "results": scored_rows,
        "selection_metric": "date_averaged_log_loss",
    }
