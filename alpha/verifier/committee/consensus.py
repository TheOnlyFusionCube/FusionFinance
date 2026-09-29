"""The verifier committee: jurors earn votes out of sample, then reach consensus.

Protocol at every decision time ``t``:

1. **Retrain** learned jurors on a schedule, using only rows whose label window
   (and forward path) closed before ``t``.
2. **Score** every juror on the cross-section at ``t`` and store the rank z-scores.
3. **Audit** each juror on its own past scores whose outcomes are now known:
   per-date Spearman information coefficient (IC), its t-statistic, and a
   one-feature logistic calibration from z-score to P(residual return > 0).
   These scores were produced before their outcomes, so the audit is out of sample.
4. **Seat** only jurors that earned a vote: enough history, positive mean IC,
   and an IC t-statistic above the bar. Weight = mean IC shrunk by history
   length. Jurors of the same family (e.g. two momentum variants) share one
   family vote so correlated methods cannot outvote the rest.
5. **Decide** by a log-odds opinion pool of calibrated side probabilities. The
   committee needs a quorum of seated jurors across at least two families. It
   vetoes when the pooled probability is below the veto line, or when a weighted
   supermajority votes against the side and the pooled probability is under one half.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd

from alpha.verifier.committee.base import JuryData, cross_sectional_z

LabelFn = Callable[[int], pd.Series]
PathFn = Callable[[int], pd.DataFrame]
CommitteeDecision = Literal["support", "veto", "no_quorum"]


@dataclass(frozen=True)
class ConsensusRules:
    horizon: int = 5
    min_history_dates: int = 8
    # A t-statistic of two is the conventional bar; Harvey, Liu & Zhu (2016) argue
    # for about three once many candidate signals are tested at once.
    min_ic_t: float = 2.0
    shrink_dates: float = 20.0
    audit_window_dates: int = 120
    quorum_jurors: int = 3
    quorum_families: int = 2
    veto_probability: float = 0.45
    supermajority: float = 2.0 / 3.0
    retrain_every: int = 20
    min_training_dates: int = 20
    max_training_dates: int = 150


@dataclass(frozen=True)
class JurorStats:
    name: str
    family: str
    dates: int
    mean_ic: float
    ic_t: float
    slope: float
    intercept: float
    weight: float

    def p_up(self, z: float) -> float:
        return 1.0 / (1.0 + math.exp(-(self.slope * z + self.intercept)))


@dataclass(frozen=True)
class JurorVote:
    name: str
    family: str
    z: float
    p_side: float
    weight: float

    def to_record(self) -> dict:
        return {"juror": self.name, "family": self.family, "z": round(self.z, 4),
                "p_side": round(self.p_side, 4), "weight": round(self.weight, 5)}


@dataclass(frozen=True)
class ConsensusVote:
    ticker: str
    side: int
    decision: CommitteeDecision
    p_side: float
    support_share: float
    against_share: float
    jurors_seated: int
    families_seated: int
    votes: tuple[JurorVote, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_record(self) -> dict:
        ranked = sorted(self.votes, key=lambda v: -v.weight)
        return {
            "decision": self.decision,
            "p_side": round(self.p_side, 4),
            "support_share": round(self.support_share, 4),
            "against_share": round(self.against_share, 4),
            "jurors_seated": self.jurors_seated,
            "families_seated": self.families_seated,
            "reasons": list(self.reasons),
            "votes": [vote.to_record() for vote in ranked[:8]],
        }


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


@dataclass
class Committee:
    jurors: list
    rules: ConsensusRules = field(default_factory=ConsensusRules)
    _scores: dict[int, dict[str, pd.Series]] = field(default_factory=dict)
    _stats: dict[str, JurorStats] = field(default_factory=dict)
    _last_fit: int | None = None
    _scored_index: int | None = None

    # ------------------------------------------------------------------ observe
    def observe(self, data: JuryData, index: int, *, labels: LabelFn, paths: PathFn | None = None) -> None:
        """Advance the committee to decision time ``index`` (no lookahead)."""
        if self._scored_index is not None and index <= self._scored_index:
            raise ValueError("committee time must move forward")
        closed = [i for i in sorted(self._scores) if i + self.rules.horizon < index]
        if self._due_for_training(index) and len(closed) >= self.rules.min_training_dates:
            rows = [
                (data.context(i), labels(i), paths(i) if paths else None)
                for i in closed[-self.rules.max_training_dates:]
            ]
            for juror in self.jurors:
                if juror.spec.learned and hasattr(juror, "fit"):
                    juror.fit(rows)
            self._last_fit = index
        context = data.context(index)
        self._scores[index] = {
            juror.spec.name: cross_sectional_z(juror.score(context).reindex(context.tickers))
            for juror in self.jurors
        }
        self._scored_index = index
        self._audit(closed[-self.rules.audit_window_dates:], labels)

    def _due_for_training(self, index: int) -> bool:
        return self._last_fit is None or index - self._last_fit >= self.rules.retrain_every

    def _audit(self, dates: list[int], labels: LabelFn) -> None:
        stats: dict[str, JurorStats] = {}
        label_cache = {i: labels(i) for i in dates}
        for juror in self.jurors:
            name = juror.spec.name
            ics, pooled_z, pooled_y = [], [], []
            for i in dates:
                z = self._scores[i].get(name)
                if z is None:
                    continue
                y = label_cache[i].reindex(z.index)
                keep = z.notna() & y.notna()
                if keep.sum() < 8 or z[keep].nunique() < 3:
                    continue
                ic = z[keep].corr(y[keep].rank(), method="spearman")
                if np.isfinite(ic):
                    ics.append(ic)
                    pooled_z.append(z[keep].to_numpy())
                    pooled_y.append((y[keep] > 0).to_numpy())
            stats[name] = self._fit_stats(juror.spec, ics, pooled_z, pooled_y)
        family_counts: dict[str, int] = {}
        for item in stats.values():
            if item.weight > 0:
                family_counts[item.family] = family_counts.get(item.family, 0) + 1
        self._stats = {
            name: JurorStats(**{**item.__dict__, "weight": item.weight / family_counts[item.family]})
            if item.weight > 0 else item
            for name, item in stats.items()
        }

    def _fit_stats(self, spec, ics, pooled_z, pooled_y) -> JurorStats:
        n = len(ics)
        if n < 2:
            return JurorStats(spec.name, spec.family, n, 0.0, 0.0, 0.0, 0.0, 0.0)
        mean = float(np.mean(ics))
        sd = float(np.std(ics, ddof=1))
        t = mean / sd * math.sqrt(n) if sd > 0 else 0.0
        z = np.concatenate(pooled_z)
        y = np.concatenate(pooled_y)
        slope, intercept = 0.0, 0.0
        if len(np.unique(y)) == 2 and len(y) >= 50:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(C=1.0).fit(z.reshape(-1, 1), y)
            slope, intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
        earned = n >= self.rules.min_history_dates and mean > 0 and t >= self.rules.min_ic_t and slope > 0
        weight = mean * n / (n + self.rules.shrink_dates) if earned else 0.0
        return JurorStats(spec.name, spec.family, n, mean, t, slope, intercept, weight)

    # --------------------------------------------------------------------- vote
    def vote(self, ticker: str, side: int, index: int | None = None) -> ConsensusVote:
        if side not in (1, -1):
            raise ValueError("side must be +1 or -1")
        index = self._scored_index if index is None else index
        if index is None or index not in self._scores:
            return ConsensusVote(ticker, side, "no_quorum", 0.5, 0.0, 0.0, 0, 0, reasons=("COMMITTEE_NOT_SCORED",))
        votes = []
        for name, stats in self._stats.items():
            if stats.weight <= 0:
                continue
            z = self._scores[index][name].get(ticker, np.nan)
            if not np.isfinite(z):
                continue
            p_up = stats.p_up(float(z))
            votes.append(JurorVote(name, stats.family, float(z), p_up if side > 0 else 1 - p_up, stats.weight))
        families = {vote.family for vote in votes}
        if len(votes) < self.rules.quorum_jurors or len(families) < self.rules.quorum_families:
            return ConsensusVote(
                ticker, side, "no_quorum", 0.5, 0.0, 0.0, len(votes), len(families), tuple(votes),
                (f"COMMITTEE_NO_QUORUM jurors={len(votes)} families={len(families)}",),
            )
        total = sum(v.weight for v in votes)
        pooled = sum(v.weight * _logit(v.p_side) for v in votes) / total
        p_side = 1.0 / (1.0 + math.exp(-pooled))
        support = sum(v.weight for v in votes if v.p_side > 0.5) / total
        against = sum(v.weight for v in votes if v.p_side < 0.5) / total
        if p_side < self.rules.veto_probability:
            decision, reason = "veto", f"COMMITTEE_VETO p_side={p_side:.2f}"
        elif against >= self.rules.supermajority and p_side < 0.5:
            decision, reason = "veto", f"COMMITTEE_SUPERMAJORITY_AGAINST {against:.0%}"
        else:
            decision, reason = "support", f"COMMITTEE_SUPPORT p_side={p_side:.2f} support={support:.0%}"
        return ConsensusVote(
            ticker, side, decision, p_side, support, against, len(votes), len(families), tuple(votes), (reason,)
        )

    def consensus_up(self, index: int | None = None) -> pd.Series:
        """Pooled P(residual > 0) for every scored ticker (for a committee-only book)."""
        index = self._scored_index if index is None else index
        if index is None:
            return pd.Series(dtype=float)
        tickers = next(iter(self._scores[index].values())).index
        values = {}
        for ticker in tickers:
            vote = self.vote(ticker, 1, index)
            if vote.decision != "no_quorum":
                values[ticker] = vote.p_side
        return pd.Series(values, dtype=float)

    # ------------------------------------------------------------------- report
    def roster(self) -> list[dict]:
        rows = []
        for juror in self.jurors:
            stats = self._stats.get(juror.spec.name)
            rows.append({
                "juror": juror.spec.name,
                "family": juror.spec.family,
                "learned": juror.spec.learned,
                "citation": juror.spec.provenance.citation,
                "institution": juror.spec.provenance.institution,
                "audited_dates": 0 if stats is None else stats.dates,
                "mean_ic": None if stats is None else round(stats.mean_ic, 4),
                "ic_t": None if stats is None else round(stats.ic_t, 2),
                "weight": 0.0 if stats is None else round(stats.weight, 5),
                "seated": bool(stats and stats.weight > 0),
            })
        return rows


__all__ = ["Committee", "ConsensusRules", "ConsensusVote", "JurorStats", "JurorVote"]
