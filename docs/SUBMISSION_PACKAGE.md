# FusionFinance Submission Package

> **Judge-facing endpoints:** [Live demo](https://fusionfinance2.vercel.app) · [Public GitHub repository](https://github.com/FusionCube18712/FusionFinance). Recheck both in a signed-out browser after every deployment.

FusionFinance is prepared as a local hackathon submission bundle for **Track 3: Unicorn — Open Innovation**. The bundle presents Pure ML, Pure LLM, and FusionFinance side by side while keeping the current retrospective replay clearly separated from causal performance claims.

## Submission copy

**Title:** FusionFinance  
**Tagline:** Quantitative discipline meets agentic intelligence.  
**Track:** Track 3 — Unicorn: Open Innovation

**Short description:** FusionFinance is an auditable autonomous-trading research architecture in which quantitative ML proposes opportunities, specialized LLM agents investigate context and falsifiers, deterministic verification challenges their evidence, and independent market and portfolio gates are required before capital. The agent runtime emits a sealed evidence precheck, which is not trading authorization. A separate controlled path can run that receipt through market verification, risk, and shared execution. The checked-in curves remain a provisional legacy replay and are not that path's output.

**AMD compute statement:** The recorded AMD-backed workload is an expanding walk-forward MLP ranker trained with PyTorch on ROCm/HIP. Hash-bound receipts prove one AMD `gfx1100` accelerator, 51,522,830,336 bytes of reported VRAM, 72 walk-forward training runs, and 231.29 seconds of training. The device-name field was unavailable, so FusionFinance does not claim an exact commercial SKU. See [`amd-compute.md`](amd-compute.md).

**Closing statement:**

> Autonomous finance should not be controlled by an unverified language model or an isolated statistical model. FusionFinance combines machine learning, agentic reasoning, quantitative verification, and risk control into one auditable decision system.

Research and paper-trading software only. No financial advice, profit guarantee, or real-money execution is authorized.

## Local artifact inventory

| Artifact | Local status | Path |
|---|---|---|
| Public GitHub repository | Published; signed-out check passed | [github.com/FusionCube18712/FusionFinance](https://github.com/FusionCube18712/FusionFinance) |
| Hosted demo URL | Deployed; responsive signed-out QA passed | [fusionfinance2.vercel.app](https://fusionfinance2.vercel.app) |
| Interactive replay dashboard | Ready locally | [`demo/index.html`](../demo/index.html) |
| Dashboard replay payload | Ready locally | [`demo/replay.json`](../demo/replay.json) |
| Narrated demo video | Refreshed 1080p repository master ready; replace the earlier external upload if edits remain open | [`presentation/FusionFinance_Demo.mp4`](../presentation/FusionFinance_Demo.mp4) |
| Submission deck | Submitted; repository copy included | [`presentation/FusionFinance_Submission.pdf`](../presentation/FusionFinance_Submission.pdf) |
| Deck HTML source | Ready locally | [`presentation/FusionFinance_Submission.html`](../presentation/FusionFinance_Submission.html) |
| Video script | Ready locally | [`presentation/video-script.md`](../presentation/video-script.md) |
| Replay result | Ready locally | [`results/demo_run.json`](../results/demo_run.json) |
| Metric summary | Ready locally | [`results/metrics.json`](../results/metrics.json) |
| AMD receipt manifest | Ready locally | [`results/amd_compute.json`](../results/amd_compute.json) |

The refreshed repository video is 4:10.15 at 1920×1080 and the PDF contains 10 slides. Its SHA-256 is `c76c5e3f98e0f15a99d3238d275bfc20792e4fc7e9f212842e202eb74eb0ec47`; use that digest to confirm any external video upload matches the repository master.
The timed, machine-readable edit plan is checked in as
[`presentation/video-storyboard.json`](../presentation/video-storyboard.json).
Raw presenter, voice, and generation intermediates remain private and are not
part of the repository.

## Judge-facing public links

| Required link | Current status |
|---|---|
| GitHub repository | [https://github.com/FusionCube18712/FusionFinance](https://github.com/FusionCube18712/FusionFinance) — published and checked signed out |
| Hosted demo | [https://fusionfinance2.vercel.app](https://fusionfinance2.vercel.app) — production QA passed at 375, 768, and 1440 pixels |
| Demo video | Earlier version submitted through the hackathon form; replace it with the refreshed repository master if the form remains editable |
| Slide deck | Submitted through the hackathon form; repository copy included |

The repository and demo URLs are the official submission endpoints. Confirm both open without authentication immediately before final judging.

## Provisional replay boundary

The checked-in February–July 2026 comparison is a **retrospective point-in-time replay**. Its stored curves report the following legacy statistics. These Sharpe ratios are `provisional_uncontrolled_legacy_race` figures, not a comparable performance claim:

| Arm | Return | Sharpe | Sortino | Maximum drawdown | Volatility |
|---|---:|---:|---:|---:|---:|
| Pure ML | +1.7% | +0.76 | +1.03 | −2.9% | 5.4% |
| Pure LLM | −10.3% | −0.90 | −1.28 | −23.8% | 24.5% |
| Legacy Fusion line | +10.1% | +2.05 | +3.48 | −3.9% | 11.1% |
| SPY benchmark | +8.7% | +1.39 | +2.10 | −8.9% | 14.6% |

These results are **provisional and are not causal proof that FusionFinance outperforms**. The legacy arms were not generated by the new shared execution kernel, the historical LLM theses were generated after their outcomes rather than prospectively sealed, and those theses fail the current mandatory evidence contract. The dashboard, deck, video, README, and submission form must preserve this boundary. The replay demonstrates the product, audit, and metric pipeline; a prospective sealed run is still required for a credible outperformance claim.

## Judge quickstart

Install Python 3.11+ and FFmpeg (`ffprobe`) before running the full verification
path. The no-auth browser demo itself remains static and offline.

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
make artifacts
make verify
make test
python3 -m http.server 8000 --bind 127.0.0.1 --directory demo
```

Open `http://127.0.0.1:8000/index.html`. The default scenario requires no login, API key, external model service, or live market connection and serves only the public `demo/` directory.

## Final upload checklist

### 1. Publish safely

- [x] Select the public GitHub repository: [FusionCube18712/FusionFinance](https://github.com/FusionCube18712/FusionFinance).
- [x] Publish from a curated release tree rather than the dirty research worktree.
- [x] Secret-scan the selected files before packaging.
- [x] Exclude `.env` files, data caches, fine-tuning corpora, scratch receipts, and local media intermediates.
- [x] Preserve the GPL root license, the filing-alpha MIT notice, and linked architectural references in `NOTICE`.
- [x] License original FusionFinance work under GPL-3.0-only at the repository root and preserve all third-party notices.
- [x] Push the intended commits and confirm the repository opens without authentication.
- [x] Confirm the README prominently links both the public repository and live demo.

### 2. Reproduce and verify

- [x] Run `python3 scripts/build_fusion_artifacts.py` and inspect the generated diff.
- [x] Run `python3 scripts/verify_amd.py`; confirm all three receipt hashes match.
- [x] Run `make artifacts`, `make verify`, and `make test`; retain the final pass output.
- [x] Serve only `demo/` and replay the default scenario in a clean browser.
- [x] Confirm startup is under 60 seconds and each interactive request completes within 30 seconds.
- [ ] Test an unseen input or scenario variant and confirm outputs remain dynamic rather than hardcoded.

### 3. Inspect the media

- [x] Run `make verify-media`; confirm the MP4 container's default H.264/AAC
  streams are 1920×1080 at 30 fps with `yuv420p` compatibility, span
  250.00±0.25 seconds, and remain below GitHub's 100 MiB per-file limit.
- [x] Open all 10 PDF slides at full resolution and check clipping, contrast, links, and AMD attribution.
- [ ] Watch the complete refreshed MP4 with sound; confirm H.264 1920×1080 video and audible AAC narration.
- [x] Confirm the deck and video show the provisional replay boundary and do not imply causal superiority.
- [x] Confirm AMD language is limited to the proven vendor, `gfx1100` architecture, runtime, VRAM, training count, and duration.
- [x] Confirm no slide, narration, or dashboard label claims W7900, MI300, MI300X, or an unproven utilization figure.
- [x] Use [`docs/assets/dashboard.png`](assets/dashboard.png) as the legible submission cover image.

### 4. Publish public artifacts

- [x] Deploy the no-auth demo to [fusionfinance2.vercel.app](https://fusionfinance2.vercel.app) and verify the production build.
- [x] Submit the original MP4 through the hackathon form; keep the repository copy available.
- [ ] If submission edits remain open, replace that upload with the refreshed 1080p repository master and confirm SHA-256 `c76c5e3f98e0f15a99d3238d275bfc20792e4fc7e9f212842e202eb74eb0ec47`.
- [x] Publish the PDF in the repository and submit it through the hackathon form.
- [x] Test the public GitHub and live-demo links without authentication.
- [x] Record the verified repository and demo URLs in this package and the README.

### 5. Final claim and form audit

- [ ] Select **Track 3: Unicorn — Open Innovation**.
- [ ] Copy the title, short description, AMD statement, and closing message from this document.
- [ ] State that the system is research and paper-trading software, not financial advice or a profit guarantee.
- [ ] Keep retrospective replay results distinct from prospective or live evidence.
- [ ] Ensure every important video claim also appears in the public repository, deck, or hosted application.
- [ ] Verify that the repository prominently explains what AMD compute performed and how to validate the receipt.
- [ ] Submit the public GitHub, demo, video, deck, and cover-image fields before the deadline.
- [ ] Reopen the completed submission once, signed out, and test every judge-facing path.
