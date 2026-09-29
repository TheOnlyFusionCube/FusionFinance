# Research data collection and analysis pipeline

The research pipeline turns raw market, filing, news, and crowd data into a
per-ticker **dossier** that the LLM reads and the ML verifier can check. Its
layers follow the workflow Barebone AI describes in its public
[resources library](https://barebone.ai/resources): impact-scored news with
affected tickers, institutional-versus-retail sentiment, insider, Congress, and
13F "smart money", earnings and transcript analysis, three-method valuation,
confirmed technical levels, signal convergence, and a multi-lens portfolio
review. FusionFinance adds the parts a fund needs and a research app does not:
strict point-in-time availability, content-hashed lineage, and outputs that
double as the evidence an LLM citation must reconcile against.

```text
collectors ─► typed PIT records ─► ResearchStore ─► view(as_of) ─► six skills ─► dossier
 (EDGAR,        (event_at +          (dedup, JSONL,    (nothing later   (signal       │
  vendor JSONL,  available_at)        lineage hash)     than as_of)      cards)        ├─► sealed SourceDocuments ─► LLM originator + analyst desk
  synthetic)                                                                           └─► feature_vector ─────────► ML verifier / meta-labeler
```

## 1. Collection

| Collector | Module | Records | Availability clock |
|---|---|---|---|
| SEC Form 4 (live, free) | `alpha/research/collectors/edgar.py` | `InsiderTransaction` | EDGAR `acceptanceDateTime` |
| SEC XBRL company facts (live, free) | `alpha/research/collectors/edgar.py` | `Fundamentals` (annual 10-K) | end of filing day, US Eastern |
| Vendor file drop | `alpha/research/collectors/vendor.py` | any record kind, one JSON object per line | as supplied, validated |
| Synthetic vendor | `alpha/research/collectors/synthetic.py` | every kind, consistent with planted events | simulated |

The EDGAR client enforces SEC fair-access rules (a contact User-Agent and at
most ten requests per second) and only fetches `sec.gov` hosts. Form 4 lines
keep the transaction code, the acquired/disposed flag, and the Rule 10b5-1 plan
flag (from `aff10b5One` or the footnotes). News, analyst ratings, social
mentions, Congressional disclosures, 13F holdings, earnings, and transcripts
normally come from licensed vendors, so they enter through the JSONL drop and
must validate against the same schemas. Invalid rows are reported, never
coerced.

```bash
python -m alpha.fund collect AAPL MSFT NVDA --out data/store \
  --user-agent "Your Name you@example.com" --since 2026-01-01
python -m alpha.fund research AAPL --data data/store --benchmark SPY
```

## 2. Point-in-time records and store

`alpha/research/records.py` defines eleven frozen record types. Each has an
`event_at` (when it happened) and an `available_at` (when the fund could know
it); a record whose availability precedes its event is rejected. The store
(`alpha/research/store.py`) deduplicates by content hash, persists one JSONL
file per kind, indexes news under every mentioned ticker, and answers queries
only through `view(as_of)`. A Form 4 trade dated Monday but accepted on
Wednesday does not exist on Tuesday; a Congressional trade disclosed forty days
late does not exist for forty days. `view.lineage(ticker)` hashes the exact
record set an analysis saw.

## 3. Analysis skills

Every skill returns a `SignalCard` (`alpha/research/cards.py`): a score in
[-1, 1], a confidence, quotable sentences, trusted numeric facts, and flags.

| Skill | What it measures | Notable guards |
|---|---|---|
| `news_impact` | 0–10 impact from magnitude, deal size, surprise, breadth, session timing, source credibility; bullish/bearish direction; second-order effects through supplier/customer/competitor links | pluggable scorer protocol; low-credibility sources are down-weighted and flagged |
| `sentiment` | institutional gauge (ratings, target upside, net upgrades, dispersion) and retail gauge (mention velocity, bull share, accounts per mention) with divergence | spam-concentrated or crowded one-sided retail euphoria counts **against** the trade |
| `smart_money` | discretionary open-market insider buys and sells, cluster buys, buying below the 90-day high; Congress purchases/sales with disclosure lag; 13F new positions and exits | grants, option exercises, tax withholding, gifts, and 10b5-1 plan trades are excluded; committee-oversight trades are flagged |
| `earnings` | EPS/revenue surprise, guidance change, beat quality, run-up into the report; transcript constraint language ("supply constrained", "lead times") | guidance outweighs the quarter; priced-in and cost-driven beats are penalised; drift fades with a 30-day half-life |
| `valuation` | DCF (fitted growth fading to terminal, CAPM WACC with PIT beta, Gordon terminal value), analyst consensus, peer P/E | reports agreement across methods and growth sensitivity rather than an average |
| `technicals` | rule-based pivots, zones clustered by ATR, confirmation across daily and weekly views, trend with a strictness bar, RSI-adaptive entries, entry/stop/target, 0–100 confidence | pivots never use unconfirmed bars; below forty the call is "wait"; all-time-high fallback projects targets; overextended moves are flagged |

`build_dossier` runs all six, computes **convergence** (independent families
agreeing: one = watch, two = research, three or more = eligible to size;
contradictions are listed, not averaged), and assembles a risk review from the
flags. `review_portfolio` (`alpha/research/portfolio.py`) gives the five-lens
portfolio view: growth, risk (volatility, Sharpe, beta, max drawdown), income
(yield, payout above earnings), sector balance, and momentum, plus hidden
concentration via average correlation and the effective number of independent
bets.

## 4. Hand-off to the fund

`Dossier.snapshot()` seals one `SourceDocument` per card, routed to the analyst
role that owns it (news, market, fundamentals) plus a risk-review document. The
card's sentences are the only text an LLM may quote and its facts (including
`signal_score`) are the only numbers it may assert; the existing evidence audit
rejects anything else, which is FusionFinance's version of "every figure is
checked against the underlying data". `Dossier.feature_vector()` exposes the
same deterministic analytics to the ML meta-labeler, which never sees LLM text.

## Limits

- The live collectors cover SEC data only. News, ratings, social, Congress,
  13F, earnings estimates, and transcripts require a vendor feed in the JSONL
  format; none is bundled.
- The news scorer is lexical by default. An LLM scorer can be plugged in, but
  it then becomes a second LLM channel and should be evaluated as one.
- Skill weights and thresholds are hand-set, not fitted. They are transparent
  defaults, not calibrated parameters.
