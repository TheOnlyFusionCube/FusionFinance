"""What the fund learns about its own LLM: track record and meta-labels.

Both learners consume only *resolved, leakage-clean* LLM calls, i.e. ideas whose
outcome window closed before the current decision and that were made on data
the model could not have seen during pretraining.

* ``ReliabilityLedger`` is a Beta-Binomial track record per (model, claim type).
  It turns self-reported conviction into an empirically shrunk probability.
* ``MetaLabeler`` is the ML verifier's learned half (meta-labelling in the sense
  of Lopez de Prado): the LLM sets the side, and a classifier trained on
  structured market state predicts whether that side will be right. It never
  sees LLM text, embeddings, or conviction, so it stays an independent channel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class ResolvedCall:
    model_id: str
    claim_type: str
    side: int                       # +1 long, -1 short
    conviction: float
    features: tuple[float, ...]     # side-adjusted structured market state
    realized_residual: float        # realised residual return over the horizon
    label_end_index: int            # session index at which the outcome is known
    contaminated: bool = False

    @property
    def correct(self) -> bool:
        return self.side * self.realized_residual > 0.0


@dataclass
class ReliabilityLedger:
    """Empirical-Bayes shrinkage of LLM conviction toward its realised hit rate."""

    prior_strength: float = 20.0
    _hits: dict[tuple[str, str], int] = field(default_factory=dict)
    _count: dict[tuple[str, str], int] = field(default_factory=dict)

    def update(self, calls: list[ResolvedCall]) -> None:
        for call in calls:
            if call.contaminated:
                continue
            key = (call.model_id, call.claim_type)
            self._count[key] = self._count.get(key, 0) + 1
            self._hits[key] = self._hits.get(key, 0) + int(call.correct)

    def observations(self, model_id: str, claim_type: str) -> int:
        return self._count.get((model_id, claim_type), 0)

    def hit_rate(self, model_id: str, claim_type: str) -> float | None:
        count = self.observations(model_id, claim_type)
        return None if count == 0 else self._hits[(model_id, claim_type)] / count

    def calibrated_probability(self, model_id: str, claim_type: str, conviction: float) -> float:
        """Posterior mean hit probability with the stated conviction as prior."""
        if not math.isfinite(conviction) or not 0.0 <= conviction <= 1.0:
            raise ValueError("conviction must be finite in [0, 1]")
        key = (model_id, claim_type)
        count = self._count.get(key, 0)
        hits = self._hits.get(key, 0)
        return (hits + self.prior_strength * conviction) / (count + self.prior_strength)


@dataclass
class MetaLabeler:
    """P(LLM side is right | structured market state), fit on clean history."""

    min_samples: int = 40
    regularization: float = 1.0
    _mean: np.ndarray | None = None
    _scale: np.ndarray | None = None
    _model: object | None = None
    trained_on: int = 0

    @property
    def active(self) -> bool:
        return self._model is not None

    def fit(self, calls: list[ResolvedCall]) -> "MetaLabeler":
        from sklearn.linear_model import LogisticRegression

        usable = [call for call in calls if not call.contaminated]
        labels = np.array([int(call.correct) for call in usable], dtype=np.int8)
        if len(usable) < self.min_samples or len(np.unique(labels)) < 2:
            self._model = None
            self.trained_on = 0
            return self
        matrix = np.array([call.features for call in usable], dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError("meta-label features must be finite")
        self._mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        self._scale = np.where(scale > 0.0, scale, 1.0)
        self._model = LogisticRegression(C=self.regularization, max_iter=1_000).fit(
            (matrix - self._mean) / self._scale, labels
        )
        self.trained_on = len(usable)
        return self

    def predict(self, features: tuple[float, ...]) -> float | None:
        if self._model is None:
            return None
        row = (np.asarray(features, dtype=np.float64) - self._mean) / self._scale
        if not np.isfinite(row).all():
            return None
        return float(self._model.predict_proba(row.reshape(1, -1))[0, 1])


def side_adjusted_features(side: int, raw: tuple[float, ...]) -> tuple[float, ...]:
    """Express market state relative to the LLM's side (long-positive)."""
    if side not in (1, -1):
        raise ValueError("side must be +1 or -1")
    return tuple(side * value for value in raw)


__all__ = ["MetaLabeler", "ReliabilityLedger", "ResolvedCall", "side_adjusted_features"]
