# Verifier committee

The ML verifier is a committee of twenty jurors, not a single model. Each juror
is one published, reproducible method that scores the whole cross-section using
only point-in-time structured data. Jurors never see LLM text, embeddings, or
conviction. They must earn their vote out of sample before they count, and
correlated jurors cannot outvote the rest.

Code: [`alpha/verifier/committee/`](../alpha/verifier/committee/).

## What "from the banks" can and cannot mean

Goldman Sachs, Morgan Stanley, JPMorgan, and other dealers do not publish the
models they trade with, and nothing here claims to reproduce a proprietary
model. `gs-quant`, the best-known bank toolkit, is an open-source API client
for Goldman's own data and analytics service. It needs institutional
credentials and has no trading models to import.

What is public, and what institutional quant desks build on, is peer-reviewed
research. Much of it was written by practitioner researchers at AQR Capital
Management, and much of it comes from academics whose work sell-side quant
teams implement as standard factors. The committee implements those methods
and records the paper and authors for each. The `institution` field is a best-effort author
affiliation, not an endorsement.

## Roster

| Juror | Family | Method | Author affiliation |
|---|---|---|---|
| `momentum_12_1` | momentum | Jegadeesh & Titman (1993), Returns to Buying Winners and Selling Losers, Journal of Finance; Asness, Moskowitz & Pedersen (2013), Value and Momentum Everywhere, Journal of Finance | UCLA (Jegadeesh & Titman); AQR Capital Management (Asness et al.) |
| `time_series_momentum` | momentum | Moskowitz, Ooi & Pedersen (2012), Time Series Momentum, Journal of Financial Economics | AQR Capital Management / NYU Stern / Chicago Booth |
| `fifty_two_week_high` | momentum | George & Hwang (2004), The 52-Week High and Momentum Investing, Journal of Finance | University of Houston and co-author |
| `short_term_reversal` | reversal | Jegadeesh (1990), Evidence of Predictable Behavior of Security Returns, Journal of Finance; Lehmann (1990), Fads, Martingales, and Market Efficiency, Quarterly Journal of Economics | UCLA; UC San Diego |
| `betting_against_beta` | low_risk | Frazzini & Pedersen (2014), Betting Against Beta, Journal of Financial Economics | AQR Capital Management / NYU Stern |
| `low_idiosyncratic_volatility` | low_risk | Ang, Hodrick, Xing & Zhang (2006), The Cross-Section of Volatility and Expected Returns, Journal of Finance | Columbia Business School and co-authors |
| `high_volume_return_premium` | volume | Gervais, Kaniel & Mingelgrin (2001), The High-Volume Return Premium, Journal of Finance | Wharton / UT Austin |
| `gross_profitability` | quality | Novy-Marx (2013), The Other Side of Value: The Gross Profitability Premium, Journal of Financial Economics | University of Rochester |
| `quality_minus_junk` | quality | Asness, Frazzini & Pedersen (2019), Quality Minus Junk, Review of Accounting Studies | AQR Capital Management |
| `piotroski_f_score` | accounting | Piotroski (2000), Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers, Journal of Accounting Research | University of Chicago |
| `accruals` | accounting | Sloan (1996), Do Stock Prices Fully Reflect Information in Accruals and Cash Flows About Future Earnings?, The Accounting Review | University of Pennsylvania (Wharton) |
| `value_composite` | value | Fama & French (1992), The Cross-Section of Expected Stock Returns, Journal of Finance; Asness, Moskowitz & Pedersen (2013) | University of Chicago; AQR Capital Management |
| `post_earnings_drift` | events | Bernard & Thomas (1989), Post-Earnings-Announcement Drift: Delayed Price Response or Risk Premium?, Journal of Accounting Research | University of Michigan |
| `opportunistic_insiders` | smart_money | Cohen, Malloy & Pomorski (2012), Decoding Inside Information, Journal of Finance; Lakonishok & Lee (2001), Are Insider Trades Informative?, Review of Financial Studies | Harvard Business School and co-authors; University of Illinois and co-author |
| `analyst_revisions` | analysts | Womack (1996), Do Brokerage Analysts' Recommendations Have Investment Value?, Journal of Finance; Jegadeesh, Kim, Krische & Lee (2004), Analyzing the Analysts, Journal of Finance | Dartmouth (Tuck); Emory University and co-authors |
| `economic_links` | network | Cohen & Frazzini (2008), Economic Links and Predictable Returns, Journal of Finance | Academic authors (Frazzini later a principal at AQR Capital Management) |
| `alpha158_boosting` | ml | Yang, Liu, Zhou, Bian & Liu (2020), Qlib: An AI-oriented Quantitative Investment Platform, arXiv:2009.11189 (Alpha158 factors with a gradient-boosted tree baseline) | Microsoft Research Asia |
| `gu_kelly_xiu_trees` | ml | Gu, Kelly & Xiu (2020), Empirical Asset Pricing via Machine Learning, Review of Financial Studies | Chicago Booth / Yale (Kelly also AQR Capital Management) |
| `triple_barrier` | ml | Lopez de Prado (2018), Advances in Financial Machine Learning, Wiley, ch. 3 (triple-barrier labelling) | Marcos Lopez de Prado (practitioner-academic; Cornell University) |
| `fusion_market_head` | ml | FusionFinance MarketVerifier: bootstrapped GBRT ensemble on structured price/volume features with calibrated adverse probability and OOD score | FusionFinance (this repository) |

Sixteen jurors are transparent rules; four learn walk-forward. Gu–Kelly–Xiu
trees are trained on the rule jurors' rank-normalised signals as firm
characteristics. Alpha158 and triple-barrier use Qlib-style price/volume
operators. The repository's own market head sits as one juror among twenty.

## Consensus protocol

At every decision time the committee:

1. **Retrains** learned jurors on a schedule, using only rows whose label
   window and forward price path closed before the decision.
2. **Scores** every juror and stores rank z-scores.
3. **Audits** each juror on its own past scores whose outcomes are now known:
   per-date Spearman information coefficient (IC), IC t-statistic, and a
   one-feature logistic calibration from z-score to P(residual return > 0).
   Scores were produced before their outcomes, so the audit is out of sample.
4. **Seats** a juror only with at least eight audited dates, positive mean IC,
   an IC t-statistic of at least two, and a positive calibration slope. The
   bar follows Harvey, Liu & Zhu (2016), *...and the Cross-Section of Expected
   Returns*, on the risk of testing many signals at once. Weight is mean IC
   shrunk by history length, and a seated family shares one family weight.
5. **Decides** with a log-odds opinion pool of calibrated side probabilities.
   A decision needs a quorum of three seated jurors across two families. The
   committee vetoes when pooled P(side) < 0.45, or when a weighted two-thirds
   supermajority votes against and pooled P(side) < 0.5.

In the fund, a quorate committee vote replaces the single market head as the
ML channel. Every idea's ledger entry records the committee decision and its top
weighted votes. Without quorum, the single market head decides as before. The
market head's out-of-distribution guard still applies either way.

## What the audit caught

During development the audit seated `accruals` with t = 2.7 on synthetic data
that carried no accruals signal. A floating-point artifact ranked names in a
fixed order that happened to correlate with outcomes over thirty dates. Fixing
the synthetic data and raising the bar from t ≥ 1 to t ≥ 2 removed it. This is
exactly the false discovery the seating rule exists to stop, and it is why a
committee must be audited rather than simply assembled.

## Results on the synthetic world

Same world, costs, and clock as [the fund doc](llm-first-fund.md). At the end
of the run four jurors are seated, all with the planted catalyst signal:
analyst revisions (IC 0.14, t 5.9), opportunistic insiders (0.13, 5.8),
post-earnings drift (0.13, 5.3), and Gu–Kelly–Xiu trees (0.10, 3.9). The
anomalies this world does not contain (momentum, value, quality, low risk)
correctly earn no vote.

| Arm | Dossier: return / Sharpe / max DD | Narrative: return / Sharpe / max DD |
|---|---|---|
| LLM decides alone | +26.3% / 4.48 / −2.8% | −6.9% / −3.26 / −7.6% |
| Committee trades alone | +24.7% / 4.49 / −2.7% | +24.7% / 4.49 / −2.7% |
| LLM-first, single market head | +22.7% / 4.35 / −1.6% | −2.9% / −1.43 / −4.0% |
| **LLM-first, committee** | **+26.9% / 4.69** / −1.9% | **+1.1% / 0.58 / −1.6%** |

With the committee, the gate passed 90% of genuine ideas but only 38% of hype
in dossier mode, and 59% versus 23% in narrative mode. In narrative mode,
committee-approved ideas hit 56% and blocked ideas 29%. The committee-only
book is strong here because this world plants its signal in structured event
data. On real data, expect the seated set, and whether the committee alone
adds anything, to look very different. This is a mechanism test, not market
evidence.

## Adding a juror

Implement `spec` (name, family, provenance) and `score(context) -> pd.Series`
using only `context.closes`, `context.volume`, `context.benchmark`, and
`context.view` (the point-in-time research store). Learned jurors also
implement `fit(rows)`. `test_rule_jurors_cannot_see_the_future` checks every
rule juror against a tape whose future has been corrupted.
