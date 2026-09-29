"""Purged walk-forward folds and calibration-slice-only affine-logit scaling."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedFold:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    calibration: tuple[int, ...]
    test: tuple[int, ...]


def _causal_prefix(indices: range, label_ends: pd.DatetimeIndex,
                   next_decision: pd.Timestamp) -> tuple[int, ...]:
    return tuple(index for index in indices if label_ends[index] < next_decision)


def purged_walk_forward_folds(
    decision_dates: Sequence,
    label_end_dates: Sequence,
    *,
    min_train_size: int,
    validation_size: int,
    calibration_size: int,
    test_size: int,
    purge: int = 10,
    max_label_horizon: int = 10,
    step_size: int | None = None,
) -> list[PurgedFold]:
    """Create expanding folds with three embargoes and strict label-close checks."""
    dates = pd.DatetimeIndex(decision_dates)
    label_ends = pd.DatetimeIndex(label_end_dates)
    sizes = (min_train_size, validation_size, calibration_size, test_size)
    if len(dates) != len(label_ends) or not len(dates):
        raise ValueError("decision and label-end dates must be non-empty and aligned")
    if dates.hasnans or label_ends.hasnans or bool((label_ends <= dates).any()):
        raise ValueError("label dates must be present and strictly forward after decisions")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("decision dates must be strictly increasing")
    if dates.normalize().has_duplicates:
        raise ValueError("purged folds require at most one decision per trading session")
    if any(size <= 0 for size in sizes):
        raise ValueError("fold slice sizes must be positive")
    if purge < max(max_label_horizon, 10):
        raise ValueError("purge must be at least max label horizon and 10 sessions")
    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    first_test = min_train_size + purge + validation_size + purge + calibration_size + purge
    folds = []
    for test_start in range(first_test, len(dates) - test_size + 1, step):
        calibration_end = test_start - purge
        calibration_start = calibration_end - calibration_size
        validation_end = calibration_start - purge
        validation_start = validation_end - validation_size
        train_end = validation_start - purge
        if train_end < min_train_size:
            continue
        validation = _causal_prefix(
            range(validation_start, validation_end), label_ends, dates[calibration_start]
        )
        calibration = _causal_prefix(
            range(calibration_start, calibration_end), label_ends, dates[test_start]
        )
        train = _causal_prefix(range(0, train_end), label_ends, dates[validation_start])
        test = tuple(range(test_start, test_start + test_size))
        if not all((train, validation, calibration, test)):
            continue
        folds.append(PurgedFold(train, validation, calibration, test))
    return folds


@dataclass(frozen=True)
class AffineLogitCalibrator:
    parameters: tuple[tuple[str, float, float], ...]
    device: str = "cuda:0"
    strict_gpu: bool = False
    fitted_horizons: tuple[str, ...] = ()
    artifact_hash: str = ""
    receipt_json: str = ""

    def __post_init__(self) -> None:
        if bool(self.artifact_hash) != bool(self.receipt_json):
            raise ValueError("calibration hash and canonical receipt must be paired")
        if self.artifact_hash:
            self.verify_artifact()

    def verify_artifact(self, required_horizons: set[str] | None = None) -> None:
        if (len(self.artifact_hash) != 64 or
                any(char not in "0123456789abcdef" for char in self.artifact_hash)):
            raise ValueError("calibration artifact hash must be lowercase SHA-256")
        encoded = self.receipt_json.encode()
        if hashlib.sha256(encoded).hexdigest() != self.artifact_hash:
            raise ValueError("calibration artifact hash does not match its receipt")
        try:
            receipt = json.loads(self.receipt_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("calibration receipt must be canonical JSON") from exc
        if json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) != self.receipt_json:
            raise ValueError("calibration receipt JSON is not canonical")
        parameters = tuple(
            (str(horizon), float(slope), float(intercept))
            for horizon, slope, intercept in receipt.get("parameters", ())
        )
        if parameters != self.parameters:
            raise ValueError("calibration receipt parameters do not match artifact")
        rows = receipt.get("rows")
        if not isinstance(rows, dict) or set(rows) != set(self.fitted_horizons):
            raise ValueError("calibration receipt horizons do not match fitted horizons")
        if required_horizons is not None and set(self.fitted_horizons) != required_horizons:
            raise ValueError("calibration artifact does not cover required horizons")
        for metadata in rows.values():
            for key in ("probability_sha256", "outcome_sha256"):
                digest = metadata.get(key, "")
                if (len(digest) != 64 or
                        any(char not in "0123456789abcdef" for char in digest)):
                    raise ValueError("calibration receipt contains an invalid input hash")

    @classmethod
    def identity(cls, horizons: Sequence[str], *, device: str = "cuda:0",
                 strict_gpu: bool = False) -> "AffineLogitCalibrator":
        unique = tuple(dict.fromkeys(str(horizon) for horizon in horizons))
        if not unique:
            raise ValueError("calibrator horizons cannot be empty")
        return cls(tuple((horizon, 1.0, 0.0) for horizon in unique), device, strict_gpu)

    def _parameter(self, horizon: str) -> tuple[float, float]:
        for key, slope, intercept in self.parameters:
            if key == horizon:
                return slope, intercept
        raise KeyError(f"unknown calibration horizon {horizon!r}")

    def transform(self, probabilities, horizon: str):
        import torch

        slope, intercept = self._parameter(horizon)
        if self.strict_gpu and not self.artifact_hash:
            raise RuntimeError("strict calibration transform requires a fitted artifact")
        if self.artifact_hash and horizon not in self.fitted_horizons:
            raise RuntimeError("fitted calibration artifact is missing this horizon")
        is_tensor = torch.is_tensor(probabilities)
        device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            if self.strict_gpu:
                raise RuntimeError("affine calibration requires ROCm cuda:0")
            device = "cpu"
        value = torch.as_tensor(probabilities, dtype=torch.float32, device=device)
        if not bool(torch.isfinite(value).all()) or not bool(
            ((value >= 0.0) & (value <= 1.0)).all()
        ):
            raise ValueError("calibration probabilities must be finite and in [0, 1]")
        logits = torch.logit(value.clamp(1e-6, 1.0 - 1e-6))
        calibrated = torch.sigmoid(slope * logits + intercept)
        if self.strict_gpu:
            from alpha.gpu_ml import assert_rocm_stage
            assert_rocm_stage(
                f"calibration-transform-{horizon}", value, calibrated,
                strict=True,
            )
        if is_tensor:
            return calibrated.to(probabilities.device)
        return calibrated.cpu().numpy()

    def fit(
        self,
        calibration_data: Mapping[str, tuple[np.ndarray, np.ndarray]],
        *,
        fold_ids: Mapping[str, Sequence],
        decision_dates: Mapping[str, Sequence],
        label_end_dates: Mapping[str, Sequence],
        deployment_cutoff,
        epochs: int = 200,
        lr: float = 0.05,
    ) -> "AffineLogitCalibrator":
        import torch
        import torch.nn.functional as functional

        if epochs <= 0 or lr <= 0.0:
            raise ValueError("calibrator epochs and learning rate must be positive")
        required = {horizon for horizon, _, _ in self.parameters}
        if set(calibration_data) != required:
            raise ValueError("calibration artifacts require data for every horizon")
        cutoff = pd.Timestamp(deployment_cutoff)
        if pd.isna(cutoff):
            raise ValueError("calibration deployment cutoff must be present")
        metadata = {}
        for horizon, values in calibration_data.items():
            if horizon not in fold_ids or horizon not in decision_dates or horizon not in label_end_dates:
                raise ValueError("calibration requires fold ids and decision/label-end dates")
            probability_values, outcome_values = values
            probability_array = np.asarray(
                probability_values, dtype="<f8"
            ).reshape(-1)
            outcome_array = np.asarray(outcome_values, dtype="<f8").reshape(-1)
            row_count = len(probability_array)
            folds = np.asarray(fold_ids[horizon]).reshape(-1)
            decisions = pd.DatetimeIndex(decision_dates[horizon])
            label_ends = pd.DatetimeIndex(label_end_dates[horizon])
            if not (len(folds) == len(decisions) == len(label_ends) == row_count):
                raise ValueError("calibration receipt metadata must align with rows")
            if (not row_count or bool(pd.isna(folds).any()) or decisions.hasnans or
                    label_ends.hasnans):
                raise ValueError("calibration receipt metadata cannot be empty or missing")
            if bool((label_ends <= decisions).any()):
                raise ValueError("calibration labels must be strictly forward")
            if bool((label_ends >= cutoff).any()):
                raise ValueError("calibration labels must end before deployment cutoff")
            metadata[horizon] = {
                "fold_ids": [str(value) for value in folds.tolist()],
                "decision_dates": [value.isoformat() for value in decisions],
                "label_end_dates": [value.isoformat() for value in label_ends],
                "probability_sha256": hashlib.sha256(
                    probability_array.tobytes()
                ).hexdigest(),
                "outcome_sha256": hashlib.sha256(outcome_array.tobytes()).hexdigest(),
                "row_count": row_count,
                "canonical_dtype": "float64-le",
            }
        device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            if self.strict_gpu:
                raise RuntimeError("affine calibration requires ROCm cuda:0")
            device = "cpu"
        if self.strict_gpu and device != "cuda:0":
            raise RuntimeError("strict calibration requires device='cuda:0'")
        fitted = []
        fitted_horizons = []
        for horizon, initial_slope, initial_intercept in self.parameters:
            if horizon not in calibration_data:
                fitted.append((horizon, initial_slope, initial_intercept))
                continue
            probabilities, outcomes = calibration_data[horizon]
            probability = torch.as_tensor(probabilities, dtype=torch.float32, device=device).reshape(-1)
            target = torch.as_tensor(outcomes, dtype=torch.float32, device=device).reshape(-1)
            if len(probability) != len(target) or not len(probability):
                raise ValueError("calibration probabilities and outcomes must align")
            if not bool(torch.isfinite(probability).all()) or not bool(torch.isfinite(target).all()):
                raise ValueError("calibration inputs must be finite")
            if not bool(((probability >= 0.0) & (probability <= 1.0)).all()) or not bool(
                ((target == 0.0) | (target == 1.0)).all()
            ):
                raise ValueError("invalid calibration probability or outcome")
            slope = torch.tensor(initial_slope, device=device, requires_grad=True)
            intercept = torch.tensor(initial_intercept, device=device, requires_grad=True)
            optimizer = torch.optim.Adam((slope, intercept), lr=lr)
            logits = torch.logit(probability.clamp(1e-6, 1.0 - 1e-6))
            if self.strict_gpu:
                from alpha.gpu_ml import assert_rocm_stage
                assert_rocm_stage(
                    f"calibration-{horizon}", probability, target,
                    params=(("slope", slope), ("intercept", intercept)), strict=True,
                )
            for _ in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                loss = functional.binary_cross_entropy_with_logits(
                    slope.float() * logits.float() + intercept.float(), target.float()
                )
                loss.backward()
                optimizer.step()
            fitted.append((horizon, float(slope.detach()), float(intercept.detach())))
            fitted_horizons.append(horizon)
        receipt = {
            "parameters": fitted,
            "deployment_cutoff": cutoff.isoformat(),
            "rows": metadata,
        }
        encoded = json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        artifact_hash = hashlib.sha256(encoded).hexdigest()
        return AffineLogitCalibrator(
            tuple(fitted), self.device, self.strict_gpu,
            tuple(fitted_horizons), artifact_hash, encoded.decode(),
        )
