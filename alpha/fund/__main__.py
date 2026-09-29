"""Command line for the LLM-first fund and its research pipeline.

    python -m alpha.fund backtest [--snapshots dossier|narrative] [--json PATH] [--ledger PATH]
    python -m alpha.fund research TICKER [--data DIR] [--as-of ISO] [--benchmark MKT]
    python -m alpha.fund collect AAPL MSFT --out DIR --user-agent "Name email@example.com"
    python -m alpha.fund review AAPL=0.5 MSFT=0.5 --data DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


def _world_store(seed: int, sessions: int):
    from alpha.fund.synthetic import WorldConfig, build_world
    from alpha.research.collectors.synthetic import world_records
    from alpha.research.store import ResearchStore

    world = build_world(WorldConfig(seed=seed, n_sessions=sessions))
    store = ResearchStore()
    store.add(world_records(world))
    return world, store


def _backtest(args) -> int:
    from alpha.fund.backtest import BacktestConfig, run_backtest
    from alpha.fund.synthetic import WorldConfig

    config = BacktestConfig(
        world=WorldConfig(seed=args.seed, n_sessions=args.sessions),
        snapshots=args.snapshots,
    )
    report, fund = run_backtest(config)
    print(report.table())
    print(f"\nledger: {report.ledger_entries} entries, head {report.ledger_head[:16]}")
    print("claim status: synthetic mechanism test, not market evidence")
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if args.ledger:
        fund.ledger.write_jsonl(Path(args.ledger))
    return 0


def _print_dossier(dossier) -> None:
    record = dossier.to_record()
    conv = record["convergence"]
    print(f"{record['ticker']} as of {record['as_of']}  lineage {record['lineage'][:12]}")
    print(f"convergence: {conv['direction']} ({conv['tier']}); agreeing {conv['agreeing']}, opposing {conv['opposing']}")
    for card in record["cards"]:
        print(f"\n[{card['skill']}] {card['direction']} score {card['score']:+.2f} conf {card['confidence']:.2f}")
        print(f"  {card['headline']}")
        for line in card["lines"]:
            print(f"  - {line}")
        if card["flags"]:
            print(f"  flags: {', '.join(card['flags'])}")
    if record["risk"]:
        print("\nrisk review:")
        for line in record["risk"]:
            print(f"  ! {line}")


def _research(args) -> int:
    from alpha.research.dossier import build_dossier
    from alpha.research.store import ResearchStore

    if args.data:
        store = ResearchStore.load(Path(args.data))
        as_of = args.as_of or f"{date.today().isoformat()}T23:59:59Z"
    else:
        world, store = _world_store(args.seed, args.sessions)
        as_of = args.as_of or world.as_of(len(world.dates) - 1)
    dossier = build_dossier(store.view(as_of), args.ticker.upper(), benchmark=args.benchmark)
    if args.json:
        print(json.dumps(dossier.to_record(), indent=2))
    else:
        _print_dossier(dossier)
    return 0


def _collect(args) -> int:
    from alpha.research.collectors.edgar import (
        EdgarForm4Collector, EdgarFundamentalsCollector, SecClient, ticker_to_cik,
    )
    from alpha.research.store import ResearchStore

    client = SecClient(args.user_agent)
    ciks = ticker_to_cik(client)
    out = Path(args.out)
    store = ResearchStore.load(out) if out.exists() else ResearchStore()
    since = date.fromisoformat(args.since)
    for ticker in (t.upper() for t in args.tickers):
        if ticker not in ciks:
            print(f"{ticker}: no SEC CIK found", file=sys.stderr)
            continue
        insiders = EdgarForm4Collector(client, max_filings=args.max_filings).collect(
            ticker, ciks[ticker], since=since
        )
        fundamentals = EdgarFundamentalsCollector(client).collect(ticker, ciks[ticker])
        added = store.add([*insiders, *fundamentals])
        print(f"{ticker}: {len(insiders)} Form 4 lines, {len(fundamentals)} annual filings, {added} new")
    store.save(out)
    print(f"store: {len(store)} records in {out}")
    return 0


def _review(args) -> int:
    from alpha.research.portfolio import review_portfolio
    from alpha.research.store import ResearchStore

    weights = {}
    for item in args.weights:
        ticker, _, weight = item.partition("=")
        weights[ticker.upper()] = float(weight)
    if args.data:
        store = ResearchStore.load(Path(args.data))
        as_of = args.as_of or f"{date.today().isoformat()}T23:59:59Z"
    else:
        world, store = _world_store(args.seed, args.sessions)
        as_of = args.as_of or world.as_of(len(world.dates) - 1)
    review = review_portfolio(store.view(as_of), weights, benchmark=args.benchmark)
    print(json.dumps(review.to_record(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m alpha.fund", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def world_options(p):
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--sessions", type=int, default=640)

    bt = sub.add_parser("backtest", help="walk-forward ablation on the synthetic world")
    world_options(bt)
    bt.add_argument("--snapshots", choices=("dossier", "narrative"), default="dossier")
    bt.add_argument("--json")
    bt.add_argument("--ledger")
    bt.set_defaults(func=_backtest)

    rs = sub.add_parser("research", help="print a research dossier for one ticker")
    world_options(rs)
    rs.add_argument("ticker")
    rs.add_argument("--data", help="store directory (default: synthetic world)")
    rs.add_argument("--as-of")
    rs.add_argument("--benchmark", default="MKT")
    rs.add_argument("--json", action="store_true")
    rs.set_defaults(func=_research)

    co = sub.add_parser("collect", help="collect SEC Form 4 and XBRL fundamentals")
    co.add_argument("tickers", nargs="+")
    co.add_argument("--out", required=True)
    co.add_argument("--user-agent", required=True, help="SEC requires a name and contact email")
    co.add_argument("--since", default=f"{date.today().year - 1}-01-01")
    co.add_argument("--max-filings", type=int, default=40)
    co.set_defaults(func=_collect)

    rv = sub.add_parser("review", help="five-lens portfolio review")
    world_options(rv)
    rv.add_argument("weights", nargs="+", help="TICKER=WEIGHT")
    rv.add_argument("--data")
    rv.add_argument("--as-of")
    rv.add_argument("--benchmark", default="MKT")
    rv.set_defaults(func=_review)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
