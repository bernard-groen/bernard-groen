"""End-to-end run: load posts -> prefilter -> coders -> verifiers + raters -> tables -> PRR.

Every processed post is appended to ``audit.jsonl`` (raw agent outputs included),
so a 20k-post run can be stopped and resumed without repeating API calls.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from .agents import Agents
from .aggregate import merge_coders, score_events
from .config import GENERIC_RSV_TERMS, PRODUCTS, UNSPECIFIED_RSV, Config
from .meddra import MedDRA
from .prr import compute_prr

_PATTERNS = {
    name: re.compile(r"\b(" + "|".join(re.escape(f) for f in forms) + r")\b", re.I)
    for name, forms in {**PRODUCTS, UNSPECIFIED_RSV: GENERIC_RSV_TERMS}.items()
}


def load_posts(path: str) -> list[dict]:
    if path.endswith((".jsonl", ".ndjson")):
        df = pd.read_json(path, lines=True)
    elif path.endswith(".json"):
        df = pd.read_json(path)
    elif path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "text" not in df.columns:
        raise ValueError(f"{path} needs a 'text' column; found {list(df.columns)}")
    if "post_id" not in df.columns:
        df["post_id"] = [f"P{i:06d}" for i in range(1, len(df) + 1)]
    df["post_id"] = df["post_id"].astype(str)
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["post_id"])
    return df.where(df.notna(), None).to_dict("records")


def matched_products(text: str) -> list[str]:
    return [name for name, pat in _PATTERNS.items() if pat.search(text)]


async def process_post(post: dict, agents: Agents, cfg: Config, meddra: MedDRA) -> dict:
    record = {"post_id": post["post_id"], "matched_products": matched_products(post["text"])}
    if not record["matched_products"]:
        return {**record, "status": "no_product_mention", "events": []}
    try:
        coder_outputs = await asyncio.gather(*(agents.code(post) for _ in range(cfg.coders.n_agents)))
        events = merge_coders(coder_outputs, meddra, cfg.min_coder_agreement)
        record["coders"] = [o.model_dump() for o in coder_outputs]
        record["post_type"] = Counter(o.post_type for o in coder_outputs).most_common(1)[0][0]
        if not events:
            return {**record, "status": "no_events", "events": []}

        verifier_outputs, rater_outputs = await asyncio.gather(
            asyncio.gather(*(agents.verify(post, events) for _ in range(cfg.verifiers.n_agents))),
            asyncio.gather(*(agents.rate(post, events) for _ in range(cfg.raters.n_agents))),
        )
        record["verifiers"] = [o.model_dump() for o in verifier_outputs]
        record["raters"] = [o.model_dump() for o in rater_outputs]
        record["events"] = score_events(events, verifier_outputs, rater_outputs)
        record["status"] = "ok"
    except Exception as e:  # one bad post must not stop a 20k run
        record.update(status="error", error=f"{type(e).__name__}: {e}", events=[])
    return record


def _load_done(audit_path: str) -> dict[str, dict]:
    done = {}
    if os.path.exists(audit_path):
        with open(audit_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("status") != "error":
                    done[rec["post_id"]] = rec
    return done


async def run(posts_path: str, out_dir: str, cfg: Config, limit: int | None = None,
              agents: Agents | None = None) -> dict[str, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    meddra = MedDRA(cfg.meddra_dir)
    if meddra.version and meddra.version != cfg.meddra_version:
        print(f"warning: MedDRA files are version {meddra.version}, config says {cfg.meddra_version}")
    cfg.meddra_version = meddra.version or cfg.meddra_version
    agents = agents or Agents(cfg, meddra)

    posts = load_posts(posts_path)[:limit]
    audit_path = os.path.join(out_dir, "audit.jsonl")
    done = _load_done(audit_path)
    todo = [p for p in posts if p["post_id"] not in done]
    print(f"{len(posts)} posts, {len(done)} already processed, {len(todo)} to go")

    by_id = {p["post_id"]: p for p in posts}
    with open(audit_path, "a") as audit:
        tasks = [asyncio.create_task(process_post(p, agents, cfg, meddra)) for p in todo]
        for i, fut in enumerate(asyncio.as_completed(tasks), start=1):
            rec = await fut
            rec["processed_at"] = datetime.now(timezone.utc).isoformat()
            rec["meddra_version"] = cfg.meddra_version
            audit.write(json.dumps(rec, default=str) + "\n")
            audit.flush()
            done[rec["post_id"]] = rec
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} processed")

    return write_tables([done[pid] for pid in by_id if pid in done], by_id, out_dir, cfg)


def write_tables(records: list[dict], posts: dict[str, dict], out_dir: str, cfg: Config) -> dict[str, pd.DataFrame]:
    post_rows, event_rows = [], []
    for rec in records:
        post = posts.get(rec["post_id"], {})
        post_rows.append({
            "post_id": rec["post_id"], "status": rec["status"], "post_type": rec.get("post_type"),
            "matched_products": "; ".join(rec.get("matched_products", [])),
            "n_events": len(rec.get("events", [])), "error": rec.get("error"),
            **{k: v for k, v in post.items() if k not in ("post_id", "text")},
            "text": post.get("text"),
        })
        for ev in rec.get("events", []):
            event_rows.append({"post_id": rec["post_id"], "meddra_version": rec.get("meddra_version"),
                               **{k: "; ".join(v) if isinstance(v, list) else v for k, v in ev.items()}})

    posts_df = pd.DataFrame(post_rows)
    events_df = pd.DataFrame(event_rows)
    posts_df.to_csv(os.path.join(out_dir, "posts.csv"), index=False)
    events_df.to_csv(os.path.join(out_dir, "coded_events.csv"), index=False)
    tables = {"posts": posts_df, "coded_events": events_df}
    tables.update(signal_tables(events_df, out_dir, cfg))
    return tables


def signal_tables(events_df: pd.DataFrame, out_dir: str, cfg: Config) -> dict[str, pd.DataFrame]:
    if events_df.empty:
        return {}
    qualifying = events_df[(events_df["confidence"] >= cfg.min_confidence)
                           & (events_df["suspect_product"] != UNSPECIFIED_RSV)]
    tables = {}
    levels = ["pt"] + (["soc"] if events_df["soc_name"].fillna("").astype(str).str.len().gt(0).any() else [])
    for level in levels:
        df = compute_prr(qualifying, cfg.target_product, cfg.comparator_products, level,
                         cfg.prr_threshold, cfg.chi2_threshold, cfg.min_cases)
        df.to_csv(os.path.join(out_dir, f"prr_{level}.csv"), index=False)
        tables[f"prr_{level}"] = df

    summary = (events_df.groupby(["suspect_product", "soc_name", "pt_name"])
               .agg(n_posts=("post_id", "nunique"), mean_confidence=("confidence", "mean"),
                    n_qualifying=("confidence", lambda s: int((s >= cfg.min_confidence).sum())),
                    n_serious=("seriousness_indicated", "sum"))
               .reset_index().sort_values(["suspect_product", "n_posts"], ascending=[True, False]))
    summary["mean_confidence"] = summary["mean_confidence"].round(3)
    summary.to_csv(os.path.join(out_dir, "event_summary.csv"), index=False)
    tables["event_summary"] = summary
    return tables
