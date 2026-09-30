# FusionFinance methodology

## Controlled comparison contract

The next claim-valid three-arm run must use
`configs/fusionfinance-demo.json` without changing assumptions after seeing the
result.

| Assumption | Locked value |
|---|---:|
| Window | 2026-02-02 through 2026-07-09 |
| Starting capital | $10,000 |
| Benchmark | SPY |
| Rebalance cadence | Every 10 sessions |
| Execution | Next session open |
| Transaction cost | 5 basis points of traded notional |
| Slippage | 2 basis points of traded notional |
| Maximum gross leverage | 1.0x |
| Maximum absolute position | 10% |
| Annualization | 252 sessions |
| Annual risk-free rate | 0% |

The locked 16-name universe is AAPL, AMZN, AVGO, BRK-B, COST, GOOGL, HD, JPM,
LLY, META, MSFT, NFLX, NVDA, ORCL, TSLA, and WMT. SPY is benchmark-only and
cannot be proposed as a position.

## Strategy arms

The controlled experiment requires three proposal generators feeding the same
execution function:

1. **Pure ML** uses structured quantitative signals and no LLM recommendation.
2. **Pure LLM** uses the LLM recommendation and no proprietary ML signal or
   verifier forecast as an input.
3. **FusionFinance** begins with a proposal, audits its evidence, challenges it
   with the independent structured verifier, and permits only policy-approved
   target weights to reach execution.

The implementation must not average arms or give FusionFinance a different cost,
timing, or leverage model. Empty or rejected proposals remain cash.

## Timing and look-ahead controls

- Market features are built from rows on or before the decision timestamp.
- Forward labels used to train the verifier begin strictly after their feature
  timestamp.
- Filing availability is derived from an explicit timezone-aware acceptance
  timestamp and the observed exchange-session calendar.
- Decisions are associated with a close and execute one full session later at
  the next open.
- Proposals outside the shared rebalance clock, outside the universe, after the
  final executable session, or above risk limits fail rather than being clipped
  silently.
- A thesis must be committed before verification. Evidence sources must already
  be available at the thesis timestamp.

The tests demonstrate that identical weights receive identical P&L across arms
and that changing later market data cannot alter an earlier fill. These tests
validate the kernel; they do not prove that a proposal generator itself is free
of leakage.

## Accounting and metrics

Positions are expressed as target weights of pre-trade portfolio value. The
kernel trades at the next open, deducts transaction cost and slippage from cash,
and marks holdings at each close. Portfolio return for session `t` is:

```text
wealth[t] / wealth[t - 1] - 1
```

Total and annualized returns are compounded from wealth, not summed. Maximum
drawdown is the worst wealth-to-running-peak ratio. Volatility uses sample
standard deviation; Sharpe and Sortino use the locked risk-free rate. The ledger
also records fills, turnover, total execution costs, cash, gross exposure, and
net exposure.

When benchmark wealth is supplied, the metrics layer additionally reports
benchmark return, wealth-relative excess return, tracking error, information
ratio, beta, and annualized alpha. Misaligned, non-finite, or non-positive
benchmark wealth is rejected. A controlled run must pass date-bound benchmark
marks; a length-only wealth sequence does not date-bind a claim.

## Evidence levels

| Artifact or result | Current evidence level |
|---|---|
| Shared execution invariants | Implemented and unit-tested |
| Wealth and benchmark metrics | Implemented and unit-tested |
| Exact-citation, numeric, and timestamp audit | Implemented and unit-tested |
| Filing transforms | Implemented from an attributed subset and integration-tested |
| AMD training workload | Semantically cross-checked self-reported receipts; hashes protect integrity, not independent attestation |
| Controlled-path wiring, hashes, and ledger reconciliation | Implemented and unit-tested; not a comparative performance result |
| Checked-in three-arm replay | Provisional legacy visualization only; not a comparable Sharpe claim |

The controlled path records proposal/input lineage, config and market-tape
hashes, post-cost leverage, ledger/result reconciliation, and date-bound
benchmark marks. Those checks make a software ledger auditable. They do not by
themselves produce a comparative performance claim. That claim still requires a
prospective three-arm run on the locked tape. The checked-in replay remains a
provisional legacy visualization. The offline desk seals one pure-LLM
precheck inside `FusionOrchestrator` before the locked outcome. That receipt
and the software ledger that consumes it are checked in at
[`results/controlled_precheck_receipt.json`](../results/controlled_precheck_receipt.json)
and
[`results/controlled_software_ledger.json`](../results/controlled_software_ledger.json).
The ledger carries experiment, config, and tape hashes and computed turnover
and costs. It is not a performance claim. The sealed replay calendar for the locked window 2026-02-02 through
2026-07-09, rebalanced every 10 sessions, is checked in at
[`results/controlled_three_arm_ledger.json`](../results/controlled_three_arm_ledger.json).
Fusion uses
[`results/fusion_policy_calibration.json`](../results/fusion_policy_calibration.json),
a pre-window calibration receipt, and still requires a market head. That
three-arm file is a software ledger, not a performance claim. Fixture
statistics from that same wealth path are checked in at
[`results/controlled_three_arm_metrics.json`](../results/controlled_three_arm_metrics.json).
Prices are the checked-in OHLCV tape bound to that sealed calendar and to
the sealed SPY returns. The desk is the offline lexical provider. Before
pure ML sizes a book, an expanding walk-forward check measures Spearman
skill of `fit_fusion_model` scores versus next-session residual returns on
names that were not used to fit that fold. Skill at or below zero leaves
that rebalance in cash. When skill passes, positive scores share the gross
budget under the locked position and gross caps, including post-cost
leverage. Fusion uses that score book only after the market head approves,
and does not apply the skill gate. `comparable_performance_claim` stays false.

`configs/barebone-comparison-v1.json` is a separate contract shell for the
window 2025-01-02 through 2026-01-12. Its tradable universe is the original
16 names plus a frozen supplemental list of liquid US names. SPY and QQQ
stay outside that book. It keeps the controlled path's 0.10 position cap
and 1.0 gross cap, with SPY as the benchmark and QQQ as an optional
secondary benchmark. Its evidence path is
`evidence/market/barebone_window_ohlcv.json`. That file is not checked in.
Resolving it fails closed when the file is missing, when `tape_sha256` is
null, or when the bytes do not match the locked hash. The fair-race tape
hash `3a2a378ce29b387b84f5345a7953028d478507f204b9f7be6eee609e0dd20c05` and
`evidence/market/locked_ohlcv.json` are not that tape. The fair-race ledger
and fixture metrics are not a `barebone-comparison-v1` result.
`comparable_performance_claim` stays false. A locked local tape can run the
same controlled three-arm path, including the multi-name book and the pure-ML
out-of-sample skill gate. On this experiment `scorebook` is `momentum`: the
pure-ML score is the 63-session log return of adjusted close, minus the
cross-sectional median at the decision session. A name without that history
is excluded and not filled. The next-session residual is only the skill
label. Skill at or below zero leaves the rebalance in cash. The ridge
scorebook remains available for tests and is still the fair-race default.
The results are `results/barebone_three_arm_ledger.json` and
`results/barebone_three_arm_metrics.json`. Both record that file's
`tape_sha256` and keep the claim false. The ledger does not carry Sharpe or
return fields. QQQ is a secondary adjusted-close index, not a tradable name.
`make test` rebuilds those results when the gitignored extract is present and
checks the checked-in JSON when it is absent. The fair-race three-arm files
are left in place.

`scripts/ingest_barebone_tape.py` can bind a local extract for that window.
`--provider yfinance` reads Yahoo Finance daily bars through yfinance.
`--provider tiingo` reads `TIINGO_API_KEY` and `--provider polygon` reads
`POLYGON_API_KEY`. `--from-csv` reads a user dump and does not call a vendor.
Adjusted close is stored when the source supplies it. A dump without
`adjclose` is refused unless `--adjustment raw` records that the bars are
unadjusted and sets `adjclose` equal to `close`. The script does not fill a
missing bar and does not generate software marks. It writes
`evidence/market/barebone_window_ohlcv.json`, which is gitignored, and
`evidence/market/barebone_window_provenance.json`, which has the byte SHA-256
and no prices. `tape_sha256` in the config stays null until `--lock-config`
records the digest of the file just written. That flag does not invent a
digest and does not set a performance claim.

`scripts/ingest_barebone_narrative.py` binds point-in-time Hacker News stories
for the same window. `--provider hn` reads the Algolia `search_by_date` API.
A sealed ticker map assigns cashtags and company names to the barebone
universe and drops unmapped text. The decision session is the first calendar
session strictly after the UTC date of `available_ts`. A missing timestamp
aborts the ingest. The events file
`evidence/narrative/barebone_window_events.jsonl` is gitignored.
`evidence/narrative/barebone_window_narrative_provenance.json` records the
provider, fetch time, window, and byte SHA-256 and does not carry the story
text. `narrative_sha256` stays null until `--lock-config` records the digest
of a non-empty file just written. `--provider reddit` and `--provider x`
require their own credentials and skip when those are unset. This build does
not call those APIs and does not treat an X recent search as a full-window
archive. `comparable_performance_claim` stays false.

Polarity for that narrative arm is the frozen lexicon in
`evidence/narrative/barebone_polarity_lexicon.json`. Scoring is offline and
does not call a live model. `scripts/score_barebone_narrative.py` reads the
locked events file, refuses a look-ahead timestamp or an unmapped ticker, and
writes gitignored `evidence/narrative/barebone_window_scores.jsonl`.
`polarity_sha256` is the digest of that scores file. `narrative_sha256` stays
the locked Hacker News digest. The session score is the mean polarity minus
the cross-sectional median of names with an event that session. The narrative
arm sizes only when expanding Spearman skill versus the next-session residual
is strictly above zero; otherwise it is cash. The position cap stays 0.10 and
the gross cap stays 1.0. Results are
`results/barebone_narrative_arm_ledger.json` and
`results/barebone_narrative_arm_metrics.json`. The momentum pure-ML book is
unchanged. `comparable_performance_claim` stays false.

Attention for a separate arm is the mapped Hacker News event count, not the
lexicon. `scripts/score_barebone_attention.py` counts events whose decision
session falls in the 21 sessions ending at the decision session, takes
`log1p`, and subtracts the cross-sectional median. A later session is not a
feature. Zero counts stay in the cross-section. The scores file
`evidence/narrative/barebone_window_attention.jsonl` is gitignored.
`attention_sha256` is the digest of that file. The book is cash unless
expanding Spearman skill versus the next-session residual is strictly above
zero. The hybrid arm is cash unless that attention gate and the momentum
skill gate both pass. When both pass, the weights are the momentum scores.
The polarity lexicon is not used for either book and is not retuned. Results
are `results/barebone_attention_arm_ledger.json`,
`results/barebone_attention_arm_metrics.json`,
`results/barebone_hybrid_arm_ledger.json`, and
`results/barebone_hybrid_arm_metrics.json`. The momentum pure-ML book is
unchanged. `comparable_performance_claim` stays false.

EDGAR for a separate arm is SEC submissions metadata, not filing HTML.
`scripts/ingest_barebone_edgar.py` reads `data.sec.gov` submissions JSON for
a frozen ticker-to-CIK map and keeps forms that are exactly 8-K, 10-Q, or
10-K. Amendments (`/A`) are excluded. `available_ts` is `acceptanceDateTime`
in UTC. `reportDate` is not an availability timestamp. The decision session
is the first tape session strictly after that UTC date. A kept form with no
acceptance timestamp aborts the ingest. The events file
`evidence/narrative/barebone_window_edgar.jsonl` is gitignored.
`edgar_sha256` is the digest of that file. The Hacker News `narrative_sha256`
stays the locked narrative bind. The score is `log1p` of the filing count in
the 63 sessions ending at the decision session, minus the cross-sectional
median. Zero counts stay in the cross-section. The book is cash unless
expanding Spearman skill versus the next-session residual is strictly above
zero. Results are `results/barebone_edgar_arm_ledger.json` and
`results/barebone_edgar_arm_metrics.json`. The momentum pure-ML book is
unchanged. `comparable_performance_claim` stays false.

Form 4 for a separate arm is SEC insider filings, not the 8-K/10-Q/10-K tape.
`scripts/ingest_barebone_form4.py` reads `data.sec.gov` submissions JSON for
the same frozen CIK map and keeps forms that are exactly 4. Amendments
(`4/A`) are excluded. `available_ts` is `acceptanceDateTime` in UTC.
`periodOfReport`, `transactionDate`, and `reportDate` are not availability
timestamps. Open-market buys are transaction code P marked acquired. Sells
are code S marked disposed. Other codes are ignored. The feature is that buy
count minus that sell count. Notional is not used, because many Form 4 prices
are blank. The raw ownership XML is read to obtain those codes and is not
stored. Stylesheet HTML is refused. The events file
`evidence/narrative/barebone_window_form4.jsonl` is gitignored.
`edgar_form4_sha256` is the digest of that file. `edgar_sha256` stays the
locked 8-K/10-Q/10-K digest. The score subtracts the cross-sectional median
over the 63 sessions ending at the decision session. Zero nets stay in the
cross-section. The book is cash unless expanding Spearman skill versus the
next-session residual is strictly above zero. Results are
`results/barebone_form4_arm_ledger.json` and
`results/barebone_form4_arm_metrics.json`. The momentum pure-ML book is
unchanged. `comparable_performance_claim` stays false.

FINRA short interest for a separate arm is the consolidated query on
`api.finra.org`, not a scrape. `scripts/ingest_barebone_short_interest.py`
reads `consolidatedShortInterest` for a frozen ticker-to-symbol map.
`available_ts` is the publication date, the 7th business day after
`settlementDate`, at 00:00:00Z. The settlement date is not an availability
timestamp. A row with no settlement date aborts the ingest. The feature is
the latest period change ratio, current short position minus previous,
divided by the previous short position, in the 63 sessions ending at the
decision session. The log1p level is not used. Names with no in-window print
are omitted. The events file
`evidence/market/barebone_window_short_interest.jsonl` is gitignored.
`short_interest_sha256` is the digest of that file. `tape_sha256`,
`narrative_sha256`, `edgar_sha256`, and `edgar_form4_sha256` stay locked.
The book is cash unless expanding Spearman skill versus the next-session
residual is strictly above zero. Results are
`results/barebone_short_interest_arm_ledger.json` and
`results/barebone_short_interest_arm_metrics.json`. The momentum pure-ML
book is unchanged. `comparable_performance_claim` stays false.

## Reproduction checks

```bash
pytest -q tests/test_fusion_experiment.py tests/test_controlled_run.py \
  tests/test_fusion_evidence.py tests/test_filing_alpha_integration.py \
  tests/test_barebone_comparison.py tests/test_barebone_tape.py \
  tests/test_barebone_run.py tests/test_barebone_narrative.py \
  tests/test_barebone_polarity.py tests/test_barebone_attention.py \
  tests/test_barebone_edgar.py tests/test_barebone_form4.py \
  tests/test_barebone_short_interest.py
make artifacts
make verify
make test
```

`build_fusion_artifacts.py` reads only checked-in evidence, fails closed on
missing fields, and deterministically preserves the legacy replay's provisional
flags. It does not convert that replay into a controlled experiment.
