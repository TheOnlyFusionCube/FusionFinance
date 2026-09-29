"""Verifier committee: published quant methods that must earn their vote."""
from __future__ import annotations

from alpha.verifier.committee.base import JurorSpec, JuryContext, JuryData, Provenance
from alpha.verifier.committee.consensus import Committee, ConsensusRules, ConsensusVote
from alpha.verifier.committee.fundamental_jurors import (
    AccrualsAnomaly, AnalystRevisions, EconomicLinks, GrossProfitability,
    OpportunisticInsiders, PiotroskiFScore, PostEarningsDrift, QualityMinusJunk,
    ValueComposite,
)
from alpha.verifier.committee.ml_jurors import (
    Alpha158Boosting, GuKellyXiuTrees, MarketHeadJuror, TripleBarrierClassifier,
)
from alpha.verifier.committee.price_jurors import (
    BettingAgainstBeta, CrossSectionalMomentum, FiftyTwoWeekHigh, HighVolumeReturnPremium,
    LowIdiosyncraticVolatility, ShortTermReversal, TimeSeriesMomentum,
)


def default_jurors(*, include_market_head: bool = True) -> list:
    """The full bench, rule-based jurors first; GKX trees read the rule jurors' raw signals."""
    rules = [
        CrossSectionalMomentum(), TimeSeriesMomentum(), FiftyTwoWeekHigh(), ShortTermReversal(),
        BettingAgainstBeta(), LowIdiosyncraticVolatility(), HighVolumeReturnPremium(),
        GrossProfitability(), QualityMinusJunk(), PiotroskiFScore(), AccrualsAnomaly(),
        ValueComposite(), PostEarningsDrift(), OpportunisticInsiders(), AnalystRevisions(),
        EconomicLinks(),
    ]
    learned = [
        Alpha158Boosting(),
        GuKellyXiuTrees(characteristic_jurors=tuple(rules)),
        TripleBarrierClassifier(),
    ]
    if include_market_head:
        learned.append(MarketHeadJuror())
    return [*rules, *learned]


__all__ = [
    "Committee", "ConsensusRules", "ConsensusVote", "JurorSpec", "JuryContext", "JuryData",
    "Provenance", "default_jurors",
]
