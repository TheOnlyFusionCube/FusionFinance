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
provisional legacy visualization.

## Reproduction checks

```bash
pytest -q tests/test_fusion_experiment.py tests/test_controlled_run.py \
  tests/test_fusion_evidence.py tests/test_filing_alpha_integration.py
make artifacts
make verify
make test
```

`build_fusion_artifacts.py` reads only checked-in evidence, fails closed on
missing fields, and deterministically preserves the legacy replay's provisional
flags. It does not convert that replay into a controlled experiment.
