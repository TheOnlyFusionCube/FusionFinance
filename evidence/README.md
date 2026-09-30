# Public evidence

This directory contains the smallest checked-in evidence set needed to rebuild
and verify the public FusionFinance replay without private caches, API keys, or
network access.

## `replay/v1_source.json`

This is a sealed copy of the published v1 retrospective replay input. It keeps
the four stored return/equity curves and all 12 projected decision events. The
artifact builder validates the structure, recomputes public metrics, refreshes
the AMD receipt manifest, and fails closed if required evidence is absent.

The source remains labeled `provisional_uncontrolled_legacy_race`. It is a
product/replay artifact, not causal proof that the hybrid architecture
outperformed either baseline.

## `market/locked_ohlcv.json`

Daily OHLC, volume, and vendor adjusted close for the locked universe and
SPY. `demo/market_tape.py` builds the controlled tape from this file and from
the sealed replay calendar. It scales OHLC by `adjclose/close`, requires
every sealed session, and requires the SPY adjusted-close returns to
reproduce `arms.benchmark.daily_returns`. A missing bar is an error. The
builder does not fill one.

## `amd/`

The three JSON files are byte-preserved receipts from the recorded ROCm/HIP
workload:

- `environment.json` — PyTorch/ROCm environment and visible architecture;
- `hardware.json` — recorded AMD hardware probe output;
- `training.json` — walk-forward training measurements.

`python3 scripts/verify_amd.py` verifies every published receipt against the
SHA-256 digests in `results/amd_compute.json`. The recorded device-name field is
blank, so FusionFinance claims AMD `gfx1100` only—not an unproven commercial
SKU.
