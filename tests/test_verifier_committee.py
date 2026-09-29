from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.fund.synthetic import BENCHMARK, WorldConfig, build_world
from alpha.research.collectors.synthetic import world_records
from alpha.research.records import InsiderTransaction
from alpha.research.store import ResearchStore
from alpha.verifier.committee import (
    Committee, ConsensusRules, JurorSpec, JuryData, Provenance, default_jurors,
)
from alpha.verifier.committee.base import cross_sectional_z
from alpha.verifier.committee.fundamental_jurors import OpportunisticInsiders
from alpha.verifier.committee.ml_jurors import TripleBarrierClassifier, alpha158_features


@pytest.fixture(scope="module")
def world():
    world = build_world(WorldConfig(n_names=24, n_sessions=330, event_rate=0.04, seed=5))
    store = ResearchStore()
    store.add(world_records(world))
    return world, store


def _data(world, store, closes=None) -> JuryData:
    tickers = list(world.tickers)
    return JuryData(
        dates=world.dates,
        closes=world.closes[tickers] if closes is None else closes,
        volume=world.volume[tickers],
        benchmark=world.closes[BENCHMARK],
        view_at=store.view,
        as_of_at=world.as_of,
    )


def test_rank_z_is_centred_and_bounded() -> None:
    z = cross_sectional_z(pd.Series([1.0, 2.0, 3.0, np.nan, 100.0]))
    assert z.isna().sum() == 1
    assert abs(z.mean()) < 1e-9 and z.abs().max() <= 3
    assert cross_sectional_z(pd.Series([1.0, np.nan])).isna().all()


def test_every_juror_cites_published_work() -> None:
    jurors = default_jurors()
    assert len(jurors) == 20
    assert len({j.spec.name for j in jurors}) == len(jurors)
    for juror in jurors:
        assert juror.spec.provenance.citation and juror.spec.provenance.institution
    families = {j.spec.family for j in jurors}
    assert {"momentum", "low_risk", "quality", "events", "smart_money", "ml"} <= families


def test_rule_jurors_cannot_see_the_future(world) -> None:
    world_, store = world
    index = 300
    tickers = list(world_.tickers)
    corrupted = world_.closes[tickers].copy()
    corrupted.iloc[index + 1:] *= 3.0
    honest = _data(world_, store).context(index)
    tampered = _data(world_, store, closes=corrupted).context(index)
    for juror in default_jurors(include_market_head=False):
        if juror.spec.learned:
            continue
        a, b = juror.score(honest), juror.score(tampered)
        pd.testing.assert_series_equal(a.reindex(tickers), b.reindex(tickers), check_names=False)


def test_alpha158_features_are_finite_ratios(world) -> None:
    world_, store = world
    frame = alpha158_features(_data(world_, store).context(300))
    assert frame.shape[0] == 24 and frame.shape[1] >= 60
    assert np.isfinite(frame.to_numpy(dtype=float)).mean() > 0.95


def test_triple_barrier_labels_the_first_barrier_touched(world) -> None:
    world_, store = world
    context = _data(world_, store).context(300)
    juror = TripleBarrierClassifier()
    last = context.closes.iloc[-1]
    path = pd.DataFrame([last * 0.9, last * 1.5], columns=context.tickers)
    labels = juror.target(context, pd.Series(0.0, index=context.tickers), path)
    assert (labels == -1.0).all()


def test_insider_juror_counts_only_discretionary_trades() -> None:
    store = ResearchStore()
    common = dict(event_at="2026-03-01T00:00:00Z", available_at="2026-03-02T22:00:00Z", source="t")
    store.add([
        InsiderTransaction(ticker="AAA", insider="x", code="P", acquired=True, shares=1, price=1, **common),
        InsiderTransaction(ticker="AAA", insider="y", code="P", acquired=True, shares=1, price=1, **common),
        InsiderTransaction(ticker="BBB", insider="z", code="S", acquired=False, shares=1, price=1,
                           rule_10b5_1=True, **common),
    ])
    dates = pd.bdate_range("2026-02-02", periods=30)
    closes = pd.DataFrame(10.0, index=dates, columns=["AAA", "BBB", "CCC"])
    data = JuryData(dates=dates, closes=closes, view_at=store.view)
    scores = OpportunisticInsiders().score(data.context(29))
    assert scores.to_dict() == {"AAA": 2.0, "BBB": 0.0, "CCC": 0.0}


class _Oracle:
    """A juror whose score is the realised future return (only valid in a test)."""

    def __init__(self, name, family, future, noise=0.0, seed=0):
        self.spec = JurorSpec(name=name, family=family, provenance=Provenance("test", "test"))
        self.future, self.noise = future, noise
        self.rng = np.random.default_rng(seed)

    def score(self, context):
        signal = self.future.iloc[context.index]
        return signal + self.noise * self.rng.normal(size=len(signal)) * signal.std()


def _oracle_setup(world):
    world_, store = world
    tickers = list(world_.tickers)
    closes = world_.closes[tickers]
    forward = closes.shift(-5) / closes - 1.0
    residual = forward.sub(forward.mean(axis=1), axis=0)
    data = _data(world_, store)

    def labels(i):
        return residual.iloc[i]

    return data, residual, labels


def test_committee_seats_skill_rejects_noise_and_vetoes(world) -> None:
    data, residual, labels = _oracle_setup(world)
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.normal(size=residual.shape), index=residual.index, columns=residual.columns)
    jurors = [
        _Oracle("skilled_a", "events", residual, noise=1.0, seed=2),
        _Oracle("skilled_b", "analysts", residual, noise=1.0, seed=3),
        _Oracle("skilled_c", "smart_money", residual, noise=1.0, seed=4),
        _Oracle("coin_flip", "value", noise),
    ]
    committee = Committee(jurors=jurors, rules=ConsensusRules(min_history_dates=6))
    for index in range(40, 300, 5):
        committee.observe(data, index, labels=labels)
    roster = {row["juror"]: row for row in committee.roster()}
    assert roster["skilled_a"]["seated"] and roster["skilled_b"]["seated"]
    assert not roster["coin_flip"]["seated"]
    ticker = residual.iloc[295].idxmax()
    assert committee.vote(ticker, 1).decision == "support"
    against = committee.vote(ticker, -1)
    assert against.decision == "veto" and against.p_side < 0.45
    assert against.jurors_seated >= 3 and against.families_seated >= 2


def test_committee_reports_no_quorum_before_it_has_history(world) -> None:
    data, residual, labels = _oracle_setup(world)
    committee = Committee(jurors=[_Oracle("a", "events", residual)])
    committee.observe(data, 40, labels=labels)
    assert committee.vote(data.closes.columns[0], 1).decision == "no_quorum"
    with pytest.raises(ValueError, match="forward"):
        committee.observe(data, 40, labels=labels)


def test_committee_only_audits_closed_labels(world) -> None:
    data, residual, _ = _oracle_setup(world)
    asked: list[tuple[int, int]] = []

    def labels(i):
        asked.append((i, current[0]))
        return residual.iloc[i]

    current = [0]
    committee = Committee(jurors=[_Oracle("a", "events", residual)])
    for index in range(40, 200, 5):
        current[0] = index
        committee.observe(data, index, labels=labels)
    assert asked and all(i + 5 < now for i, now in asked)


def test_learned_jurors_train_on_closed_rows_and_score(world) -> None:
    from alpha.verifier.committee.ml_jurors import Alpha158Boosting, GuKellyXiuTrees
    from alpha.verifier.committee.price_jurors import CrossSectionalMomentum, ShortTermReversal

    world_, store = world
    data = _data(world_, store)
    tickers = list(world_.tickers)
    closes = world_.closes[tickers]
    rows = []
    for index in range(260, 300, 2):
        forward = closes.iloc[index + 5] / closes.iloc[index] - 1.0
        rows.append((data.context(index), forward - forward.mean(), closes.iloc[index + 1:index + 6]))
    jurors = [
        Alpha158Boosting(max_iter=15),
        GuKellyXiuTrees(max_iter=15, characteristic_jurors=(CrossSectionalMomentum(), ShortTermReversal())),
        TripleBarrierClassifier(max_iter=15),
    ]
    later = data.context(315)
    for juror in jurors:
        assert juror.score(later).isna().all(), "an unfitted juror abstains"
        juror.fit(rows)
        assert juror.score(later).notna().sum() >= 20
