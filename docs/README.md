# FusionFinance repository guide

This guide identifies the source of truth for each public claim. The public
tree contains only the runnable submission and its evidence closure; historical
research families and unused vendored frameworks are intentionally excluded.

## Fast review path

| Question | Canonical artifact |
|---|---|
| What does the project do? | Root [`README.md`](../README.md) and the [live demo](https://fusionfinance2.vercel.app) |
| How are ML, agents, verification, and risk connected? | [`architecture.md`](architecture.md) |
| Is the three-arm comparison fair? | [`methodology.md`](methodology.md) and [`../configs/fusionfinance-demo.json`](../configs/fusionfinance-demo.json) |
| What is actually proven about AMD compute? | [`amd-compute.md`](amd-compute.md), [`../results/amd_compute.json`](../results/amd_compute.json), and `python scripts/verify_amd.py` |
| Which conclusions are provisional? | [`limitations.md`](limitations.md) and the claim-status field in [`../results/demo_run.json`](../results/demo_run.json) |
| Where are the submitted media? | [`../presentation/`](../presentation/) |

## Controlled-run path

The following is the implemented architecture for a controlled run. It is
**not** the provenance of the checked-in legacy replay, whose explicit
provisional status remains attached to `demo/replay.json` and
`results/demo_run.json`. Legacy Sharpe ratios in `results/metrics.json` are
not a comparable claim.

```text
configs/fusionfinance-demo.json
        │ freezes universe, window, costs, lag, and risk limits
        ▼
demo/contracts.py ──► demo/execution.py ──► demo/metrics.py
        ▲                       ▲
        │                       │ shared across all three arms
alpha/verifier/                 │
alpha/filing_alpha/ ────────────┘
        │
        ▼
controlled-run artifacts ──► replay payload ──► demo/index.html
```

- `demo/` contains immutable experiment contracts, the shared execution and
  metric kernel, the no-auth interface, and the separate legacy replay
  payload.
- `alpha/verifier/` contains thesis, evidence, calibration, market-check, and
  veto policies. Agents cannot execute trades directly.
- `alpha/agents/` contains a provider-agnostic four-role desk. It runs market,
  news, fundamentals, and risk analysis concurrently, validates strict JSON,
  commits one immutable thesis, and emits a fail-closed evidence-precheck
  receipt. That receipt is not trading authorization. `demo/controlled.py`
  is the downstream market, risk, and execution path and does not relabel the
  legacy replay.
- `alpha/filing_alpha/` contains the narrowly selected filing/XBRL/text
  transforms imported from the supplied archive, with its own MIT notice.
- `configs/` is the experiment-control source of truth.
- `results/` currently contains public, machine-readable **legacy replay**
  outputs; it must not be relabeled as shared-kernel output. `presentation/`
  contains the submitted video, deck, and narration script.

## Focused public surface

- `alpha/` retains only the filing transforms, focused agent runtime,
  verification layer, and quantitative modules needed to substantiate the
  AMD-backed workload.
- `evidence/` contains the sealed legacy replay source and exactly three
  byte-preserved AMD receipts.
- `scripts/` contains exactly three release tools: deterministic artifact
  build, AMD verification, and curated archive construction.
- `tests/` covers the retained implementation, publication integrity, static
  release surface, and video storyboard.
- External agent frameworks are referenced in `NOTICE`; unused upstream source
  is not vendored into the judge path.

## Source-of-truth rules

1. Experiment assumptions come from `configs/fusionfinance-demo.json`, not
   prose or UI labels.
2. Public metrics are recomputed from stored curves; the UI does not embed a
   separate set of performance numbers.
3. Claim status travels with the replay artifact. A provisional run may be
   visualized, but it may not be described as causal evidence.
4. AMD claims must be derivable from hashed receipts and pass
   `scripts/verify_amd.py`.
5. Third-party license boundaries come from root `NOTICE` and the filing-alpha
   MIT notice.

## Generated files

`demo/replay.json`, `results/demo_run.json`, `results/metrics.json`,
`results/decisions.jsonl`, and `results/amd_compute.json` are rebuilt from
`evidence/replay/v1_source.json` and the three `evidence/amd/` receipts by
`python scripts/build_fusion_artifacts.py`. The GitHub-ready archive is built
by `python scripts/build_submission_archive.py`; its `SHA256SUMS.txt` manifest
binds every included file.

Do not hand-edit generated performance numbers. Change their source inputs,
rebuild, run the verification checks, and review the resulting diff.
