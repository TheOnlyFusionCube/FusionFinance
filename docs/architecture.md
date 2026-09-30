# FusionFinance architecture

FusionFinance separates proposal generation, evidence checking, independent
quantitative verification, and execution. The separation is deliberate: an LLM
cannot approve its own claims, and no strategy arm receives different execution
rules.

## Claim boundary

The shared contracts, execution kernel, metrics, evidence audit, filing
transforms, and controlled-run wiring described below are implemented and
tested. That wiring does **not** regenerate the checked-in three-arm replay.
The current `demo/replay.json` is a legacy retrospective artifact and identifies
itself as `provisional_uncontrolled_legacy_race`. Sharpe ratios from that
artifact, including `results/metrics.json`, are not a comparable claim. A sealed
`agent_evidence_precheck` is not trading authorization.

```text
Point-in-time market and filing data
             |
       +-----+--------------------------+
       |                                |
 Quantitative models              LLM analyst
       |                                |
       |                         committed thesis
       |                                |
       |                    deterministic evidence audit
       |                                |
       +----------> independent market verifier
                                |
                         veto-first policy
                                |
                         target-weight proposal
                                |
                    shared next-session execution
                                |
                  immutable ledger and metrics
```

## Implemented components

### Data and filing features

Point-in-time market and filing inputs feed the controlled-path foundation.
`alpha/filing_alpha/` contains the narrow MIT-licensed subset
ported from `filing_alpha_sharpe6`: XBRL ratios and deltas, deterministic filing
text features, point-in-time joins, freshness decay, robust standardization, and
a transparent ridge baseline. Raw SEC timestamp parsing and the source archive's
rejected trading strategy were intentionally excluded.

Filing acceptance times must be timezone-aware. A filing known by an after-close
signal may only influence a later execution; it never authorizes a same-close
fill.

### Quantitative models

The repository contains point-in-time feature builders, walk-forward model code,
and an independent multi-horizon residual-return verifier. The verifier consumes
structured market features, not LLM prose, embeddings, or self-reported
confidence. This preserves a distinct error channel.

### Concurrent analyst runtime

`alpha/agents/` implements the previously architectural agent boundary without
vendoring an external framework. One `AgentDesk` invokes market, news,
fundamentals, and risk roles concurrently against the same immutable proposal
and SHA-256-sealed source snapshot. It rejects the snapshot before agent
execution if any source became available after the proposal cutoff. Role
outputs are strict, frozen schemas; citations must use a source eligible for the
role, narrative numbers are forbidden, and quantitative assertions use audited
`numeric_key`/`asserted_value` pairs. The canonical instruction and exact input
hashes are bound into the committed thesis prompt hash. Role-output parsing and
validation run inside the desk deadline. A shared four-worker pool bounds
timed-out custom-provider work instead of creating unbounded threads;
malformed JSON, role spoofing, missing fields, unsafe numeric claims, or
provider failure cannot reach approval. Malformed role output rejects before a
thesis exists; exact-quote, numeric, and timestamp failures reject during the
subsequent evidence audit.

The deterministic offline provider gives judges a dependency-free path whose
output changes with the supplied point-in-time text and numeric values. The
optional OpenAI-compatible provider uses `urllib`, strict response-size and
timeout limits, disabled redirects, and HTTPS validation. The recommended
factory reads credentials from `FUSION_LLM_*` environment variables; credentials
are excluded from representations and receipts. It adds no model SDK dependency.

`FusionOrchestrator` converts validated role citations into one committed
`ThesisContract`, runs the existing exact-quote, numeric, and timestamp audit,
then applies a small veto-first candidate decision. Evidence failure or a risk
veto forces a zero position. Candidate approval requires valid citation
integrity, no contradiction, and support from at least three roles. Every
outcome is returned as a sealed, immutable `agent_evidence_precheck` receipt.

This runtime is an upstream pre-gate. It does not call `market_head`, the
calibrated/OOD adjudication policy, portfolio construction, or execution, so its
`approved` value must not be interpreted as permission to move capital.
`demo/controlled.py` is the downstream path that applies those gates. The
evidence audit proves exact-quote, numeric, timestamp, and source-role integrity;
it does not prove that prose semantically entails an analyst conclusion. Receipt
hashes prove self-consistency, not signer identity or external authenticity.
The receipt carries a `provisional_weight_cap`, never an executable position
target; only a distinct downstream policy can convert it into one. The current receipt
embeds the complete immutable proposal and sealed source snapshot. Its validator
reruns the evidence audit and derives the decision, reasons, and cap rather than
trusting caller-supplied values. Evaluated receipts require all four reports,
the matching committed thesis, and an audit; failure receipts require all three
to be absent.

### Historical LLM replay boundary

The public replay stores the projected output of one historical batched analyst,
not a prospectively executed run of the new agent desk. A claim-bearing run must
still execute and seal those roles before outcomes are known. The implemented
runtime makes that run possible; it does not rewrite the provenance of the
checked-in curves.

### Evidence and adjudication

`alpha/verifier/contract.py` defines immutable thesis and verifier records with a
commit-then-compare flow. `alpha/verifier/evidence.py` independently checks:

- exact cited text against a known source snapshot;
- asserted numbers against trusted extracted values;
- whether every source was available by the thesis timestamp; and
- malformed, missing, unknown, or duplicate evidence.

The audit fails closed. `alpha/verifier/policy.py` then applies evidence,
calibration, out-of-distribution, contradiction, direction, materiality, and
confidence gates. Possible outcomes are `approved`, `reject`, `research_only`,
`inconclusive`, and `abstain`.

### Shared execution and metrics

`demo/contracts.py`, `demo/execution.py`, and `demo/metrics.py` provide the
foundation of the controlled experiment boundary. All records are frozen. A decision made at a
session close executes at the next session's open. The kernel applies one
rebalance clock, cost model, slippage model, concentration limit, and gross
leverage check to every arm, then records fills, cash, exposure, turnover,
costs, and wealth.

`demo/controlled.py` runs one arm, or all three, through that kernel. It records
proposal and input lineage and hashes the locked config, the market tape, and
the experiment. Each rebalance stores post-cost gross leverage. The kernel
rejects exposure that cost drag cannot explain. The controlled path also refuses
a book whose post-cost gross leverage would exceed the locked cap, reconciles
fills to events and marked results, and requires benchmark wealth to be bound
to portfolio session dates. Turnover and costs on that ledger are computed
values.

That software record is not a comparative performance claim. A full weekday
calendar for the locked February–July window is checked in as a software
ledger, not as a published performance result. Uncalibrated policy thresholds
abstain. A calibrated artifact does not approve fusion without a market head
and an in-distribution score. A thesis sealed after its outcome does not reach
capital.

Metrics are recomputed from successive wealth ratios. They include return,
annualized volatility, Sharpe, Sortino, Calmar, running-peak drawdown, tail loss,
turnover, costs, exposure, and benchmark-relative fields.

## AMD compute evidence

The checked-in receipt chain records an AMD accelerator with architecture
`gfx1100`, ROCm/HIP `7.2.53211-e1a6bc5663`, PyTorch
`2.9.1+gitff65f5b`, and a 231.29-second expanding walk-forward MLP training
workload. `scripts/verify_amd.py` verifies the receipt paths and SHA-256 hashes.

The source receipt did not expose a commercial device name and contains no
utilization sample. Therefore the repository claims **AMD gfx1100 compute**, not
a specific MI300 or Radeon SKU.
