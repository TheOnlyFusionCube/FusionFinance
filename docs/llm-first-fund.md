# The LLM-first fund with an ML verifier

FusionFinance inverts the usual hybrid design. The **language model originates**
every idea from a point-in-time research dossier. An analyst desk and a
deterministic evidence audit check what it says. An **independent ML verifier**
then tries to falsify the idea using data the LLM never produced. Only ideas
the verifier fails to falsify are sized and traded, through the same execution
kernel as every other arm.

```text
research dossier (sealed, PIT) ─► LLM originator ─► idea or pass
                                        │
                     analyst desk: market / news / fundamentals / risk
                                        │
                committed thesis + exact-quote / numeric / timestamp audit
                                        │
      ML gate: verifier committee of twenty published methods that must earn
               their vote out of sample (see verifier-committee.md), with the
               walk-forward market head as one juror and its OOD guard
               + meta-labeler trained on the fund's own resolved LLM calls
               + Beta-Binomial LLM track record (recalibrates conviction)
                                        │
          risk book: fractional Kelly × volatility scaling, name/gross/net caps
                                        │
               shared next-open execution with costs ─► metrics
                                        │
                     hash-chained ledger of every decision, passes included
```

Code: [`alpha/fund/`](../alpha/fund/) (fund loop, gate, learning, sizing,
ledger, cache, leakage guard, Claude provider, backtest, CLI) on top of
[`alpha/agents/`](../alpha/agents/) (desk and receipts),
[`alpha/verifier/`](../alpha/verifier/) (market head, evidence audit, strict
policy), [`alpha/research/`](../alpha/research/) (see
[research pipeline](research-pipeline.md)), and [`demo/`](../demo/) (execution
kernel and metrics).

## Design choices that differ from popular agent funds

Multi-agent LLM trading projects such as
[TradingAgents](https://github.com/tauricresearch/tradingagents) and
[ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) organise LLM roles
(analysts, debaters, investor personas, a risk manager, a portfolio manager)
whose collective output becomes the trade. FusionFinance keeps the LLM in
charge of ideas but not of approval. We have not benchmarked those projects;
the table lists design properties, not measured performance.

| Property | FusionFinance |
|---|---|
| Who approves capital | An ML verifier the LLM cannot see or influence; the LLM only proposes |
| Error channels | LLM reads text; the verifier reads price/volume structure and deterministic analytics, never LLM prose, embeddings, or conviction |
| Learning from mistakes | Meta-labeler and track record learn only from resolved, leakage-clean calls whose outcome window closed before the decision |
| Hallucination control | Exact-quote, numeric-reconciliation, timestamp, and role-eligibility audit; narrative fields may not contain numbers |
| Pretraining leakage | `KnowledgeCutoffGuard` marks decisions dated inside a model's training window as contaminated and excludes them from learning; unknown cutoffs fail closed |
| Reproducibility | Content-addressed record/replay cache for every LLM call; deterministic offline providers; hash-chained ledger |
| Fair comparison | Every arm shares one execution kernel, clock, cost, slippage, and limit set |
| Structured outputs | Strict JSON schemas; duplicate keys, coerced numbers, spoofed roles, and oversize outputs fail closed |

## Ablation on the synthetic world

The synthetic world ([`alpha/fund/synthetic.py`](../alpha/fund/synthetic.py))
plants two event types. **Genuine** catalysts are disclosed after the close,
leave no price footprint, and drift afterwards; their sign is knowable from
text, filings, insiders, and estimates. **Hype** events follow a quiet run-up,
arrive as promotion (low-credibility posts, spam-concentrated one-sided retail
chatter, no insider buying), and reverse. Because the world is fictional, no
pretrained model can know its future: it is the one setting where an LLM-first
backtest is leakage-free by construction. It is a **mechanism test, not market
evidence**.

Run with `make backtest` (30 names, 640 sessions, seed 7, tests from
2024-06-05 for 267 sessions, 5-session rebalance, 5 bps cost plus 2 bps
slippage, offline deterministic LLM stand-ins). Output is byte-identical
across runs.

**Dossier mode** (LLM reads the research pipeline):

| Arm | Return | Sharpe | Max drawdown | Turnover | Costs |
|---|---:|---:|---:|---:|---:|
| `llm_only` (LLM decides) | +29.0% | 4.91 | −2.1% | 25.1 | 1.95% |
| `llm_desk` (+ desk and evidence audit) | +25.4% | 4.41 | −2.1% | 24.5 | 1.88% |
| `ml_only` (market head trades alone) | −0.3% | −0.04 | −4.0% | 23.1 | 1.62% |
| `committee_only` (committee trades alone) | +24.7% | 4.49 | −2.7% | 15.5 | 1.23% |
| `fusion_single` (LLM-first, market-head gate) | +23.7% | 4.58 | −1.6% | 21.6 | 1.64% |
| `fusion` (LLM-first, committee gate) | **+29.4%** | **5.12** | −1.9% | 21.6 | 1.71% |

Ideas the gate approved hit 63.9%; ideas it blocked hit 47.0%. In this mode the
research pipeline already screens out most promotion: only 10 hype ideas
reached the gate over the whole run (8 passed, 2 vetoed), against 62 genuine
ones (46 approved, 5 vetoed, 11 rejected by the desk). Each arm's risk book
applies the drawdown brake using its own point-in-time paper mark.

**Narrative mode** (LLM reads raw text only, so it falls for promotion):

| Arm | Return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| `llm_only` | −6.9% | −3.26 | −7.6% |
| `fusion_single` | −2.9% | −1.43 | −4.0% |
| `fusion` (committee) | **+1.1%** | **0.58** | **−1.6%** |

Committee-approved ideas hit 56.0% and blocked ideas 29.0%; the gate passed 59%
of genuine ideas and 23% of hype. A verifier cannot create an edge the LLM
lacks. What it does here is turn a fooled LLM's losing book into a roughly flat one.

Read these numbers as evidence that the plumbing does what it claims, not as
expected returns. The Sharpe ratios are high because the world plants strong,
clean signals. Gate thresholds are hand-set (`gate_calibrated: false`), and the
offline providers are rule-based stand-ins for an LLM.

## Running a real model

```bash
pip install -e '.[claude]'
export ANTHROPIC_API_KEY=...            # or an `ant auth login` profile
export FUSION_LLM_KNOWLEDGE_CUTOFFS="anthropic:claude-opus-5-5=YYYY-MM-DD"
```

```python
from alpha.fund.backtest import BacktestConfig, run_backtest
from alpha.fund.cache import RecordReplayProvider
from alpha.fund.claude import ClaudeProvider

claude = RecordReplayProvider(ClaudeProvider(), "cache/llm", mode="record")
report, fund = run_backtest(BacktestConfig(), originator=claude, analyst=claude)
```

`ClaudeProvider` defaults to `claude-opus-5-5` with structured outputs and
server-side refusal fallbacks (`fallbacks: "default"`). It is untrusted like
any provider: its JSON is revalidated, audited, and verified. On the synthetic
world, a real model is leakage-free. On real historical data, set the model's
knowledge cutoff. The guard then marks every earlier decision as contaminated
and excludes it from learning. Only decisions made after the cutoff, and
ideally sealed before their outcomes, count as evidence.

## Limits

- The verifier's market head is trained on structured price and volume
  features. On real, survivorship-limited universes its standalone alpha is
  expected to be near zero; its job is falsification, not origination.
- Gate thresholds, skill weights, and risk parameters are uncalibrated
  defaults. A walk-forward calibration artifact is required before any
  claim-bearing run.
- No live broker integration exists. The system produces target weights and a
  ledger; execution is simulated.
- Research and paper trading only. Not investment advice.
