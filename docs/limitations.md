# FusionFinance limitations and claim policy

FusionFinance is research and decision-support software. It is not investment
advice, a live brokerage system, or evidence of guaranteed profitability.

## Current replay is provisional

The curves in `demo/replay.json`, `results/demo_run.json`, and
`results/metrics.json` may be used to demonstrate the interface, but not to claim
that FusionFinance beat the other arms in a fair experiment. The Sharpe ratios
in `results/metrics.json` are `provisional_uncontrolled_legacy_race` statistics.
Do not compare them.

Specifically:

- the legacy arms were not run through `demo/execution.py`;
- the arms do not share a verified transaction-cost ledger;
- turnover, trade count, win rate, cash, and exposure were not stored in the
  source curves;
- cached historical LLM theses were generated after their outcomes and were not
  prospectively sealed;
- those theses do not contain the citation evidence required by the current
  evidence contract;
- every displayed LLM decision therefore fails the evidence gate; and
- the legacy Fusion curve contains an ML champion, so it does not isolate the
  incremental effect of LLM reasoning or verification.
- the legacy Fusion curve has only 21 nonzero changes across 109 stored account
  values, in an apparent five-session marking cadence, while comparison arms
  change more frequently; sparse marking can hide intra-interval drawdowns.

The replay correctly labels itself `provisional_uncontrolled_legacy_race`. Its
annualized metrics cover only 109 observations and are especially sensitive to
the selected window.

## Product gaps

- The stored LLM sleeve came from one historical batched analyst prompt, not a
  prospective run of the implemented market/news/fundamentals/risk desk.
- The deterministic offline provider is a reproducible lexical fallback for
  tests and judge interaction, not a substitute for a calibrated language
  model or independent financial research.
- The agent receipt is an evidence precheck, not a trading authorization.
  `demo/controlled.py` is the separate path that can call the market verifier,
  portfolio limits, and execution. Capital there still requires a prospective
  seal and, for the fusion arm, a calibrated market adjudication.
- Citation auditing proves exact substring, source-role, structured-number, and
  timestamp integrity, not semantic entailment. A well-formed but irrelevant
  quotation can still accompany an unsupported qualitative conclusion.
- The evidence auditor is implemented. A controlled run can consume sealed
  receipts, but those receipts have not been integrated into a regenerated
  three-arm performance claim on the locked tape.
- The kernel records post-cost gross leverage and rejects accounting breaches
  beyond cost drag. The controlled path additionally blocks books that would
  exceed the locked gross cap after costs, and reconciles fills, events, and
  marked results. The legacy replay still has no trade ledger.
- Controlled runs carry experiment, config, and market-tape hashes. The legacy
  replay result does not.
- Claim-bearing benchmark fields are bound to portfolio session dates.
  Length-only benchmark wealth remains available to kernel tests and is not
  sufficient for a comparative claim.
- The broader covariance-, liquidity-, and volatility-aware sizing engine in
  the product vision is not yet part of the controlled path.
- No controlled live or paper-trading comparison has been completed with all
  three arms.
- External model and market-data services can introduce latency, outages, and
  nondeterminism. Cached LLM outputs improve replayability but are not
  prospective evidence.
- The built-in HTTP provider has a hard request timeout. Python cannot kill an
  arbitrary third-party provider call already running in a thread; the desk
  stops waiting at its deadline and bounds such work to four daemon workers and
  a bounded fail-closed queue. Stuck workers cannot block process exit, but they
  reduce capacity until the provider returns, so custom providers must implement
  their own cooperative timeout.

## Data and research risks

- The controlled fixed universe can introduce selection and survivorship bias.
- The repository does not have a complete delisting-return source. Membership
  exits are handled with available prices, which is not equivalent to a complete
  delisting model.
- Corporate-action, missing-price, and timestamp errors remain material risks.
- A five-month window does not establish robustness across market regimes.
- Confidence intervals, repeated prospective windows, and a locked untouched
  holdout are still required for a strong performance claim.
- Backtest overfitting, prompt sensitivity, model calibration drift, and LLM
  hallucination remain possible even with a verifier.

Runtime SHA-256 seals detect post-hoc mutation but are not signatures or
independent attestation; any caller able to create a receipt can recompute them.

The imported filing-alpha transforms are not evidence of alpha. The source
archive rejected its own Sharpe6 candidate; FusionFinance retained only narrow,
auditable software transforms and excluded the strategy and faulty timestamp
parser.

## AMD evidence boundary

Self-reported receipts record an AMD vendor identifier, `gfx1100`, a ROCm/HIP
runtime, and a walk-forward training workload. The publication builder derives
and cross-checks those fields; SHA-256 protects byte integrity but is not
independent attestation. The receipts do not establish the commercial device
SKU, MI300 usage, real-time utilization, lineage to the displayed legacy curves,
or that every application stage ran on the accelerator. Public materials must
retain this distinction.

## Conditions for a comparative claim

A statement that FusionFinance improved risk-adjusted decision quality requires
a new run in which all three arms use the locked config, shared execution kernel,
identical market tape, identical costs, complete ledgers, and prospectively
committed evidence. Until then, describe the architecture and verified software
properties—not comparative investment performance.
