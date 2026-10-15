"""CLI.

  python -m ae_signal run --posts posts.csv --meddra-dir /path/to/MedAscii --out outputs/
  python -m ae_signal prr --out outputs/ --min-confidence 0.7    # re-score without new API calls
"""

from __future__ import annotations

import argparse
import asyncio

import pandas as pd

from .config import Config
from .pipeline import run, signal_tables


def main() -> None:
    p = argparse.ArgumentParser(prog="ae_signal")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Process posts with the agent panel and build signal tables")
    r.add_argument("--posts", required=True, help="CSV / JSONL / XLSX with a 'text' column (post_id optional)")
    r.add_argument("--meddra-dir", required=True, help="Directory holding llt.asc and mdhier.asc")
    r.add_argument("--meddra-version", default="26.1")
    r.add_argument("--coders", type=int, default=3)
    r.add_argument("--verifiers", type=int, default=2)
    r.add_argument("--raters", type=int, default=2)
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--limit", type=int, help="Only process the first N posts")

    s = sub.add_parser("prr", help="Recompute PRR tables from an existing coded_events.csv")
    s.add_argument("--meddra-dir", default="")

    for sp in (r, s):
        sp.add_argument("--out", default="outputs")
        sp.add_argument("--min-confidence", type=float, default=0.5)
        sp.add_argument("--comparators", nargs="*", help="Comparator products (default: all non-target)")

    args = p.parse_args()
    cfg = Config(meddra_dir=args.meddra_dir, min_confidence=args.min_confidence,
                 comparator_products=args.comparators or None)

    if args.cmd == "run":
        cfg.meddra_version = args.meddra_version
        cfg.coders.n_agents, cfg.verifiers.n_agents, cfg.raters.n_agents = args.coders, args.verifiers, args.raters
        cfg.concurrency = args.concurrency
        tables = asyncio.run(run(args.posts, args.out, cfg, limit=args.limit))
    else:
        events = pd.read_csv(f"{args.out}/coded_events.csv", dtype={"pt_code": str, "soc_code": str})
        tables = signal_tables(events, args.out, cfg)

    prr = tables.get("prr_pt")
    if prr is not None and not prr.empty:
        cols = ["pt_name", "a_target_with_term", "c_comparator_with_term", "prr",
                "prr_lower95", "prr_upper95", "chi2_yates", "prr_confidence_weighted", "signal"]
        print(prr[cols].head(25).to_string(index=False))
    print(f"Tables written to {args.out}/")


if __name__ == "__main__":
    main()
