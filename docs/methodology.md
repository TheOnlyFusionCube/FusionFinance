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

## Reproduction checks

```bash
pytest -q tests/test_fusion_experiment.py tests/test_controlled_run.py \
  tests/test_fusion_evidence.py tests/test_filing_alpha_integration.py \
  tests/test_barebone_comparison.py tests/test_barebone_tape.py \
  tests/test_barebone_run.py
make artifacts
make verify
make test
```

`build_fusion_artifacts.py` reads only checked-in evidence, fails closed on
missing fields, and deterministically preserves the legacy replay's provisional
flags. It does not convert that replay into a controlled experiment.
