"""Score a pipeline run against a human-coded gold standard: precision, recall, F1.

Gold file (CSV/XLSX), one row per gold event:

    post_id, pt, suspect_product
    S001,    Injection site pain, Abrysvo
    S001,    Pyrexia,             Abrysvo
    S002,    ,                                <- post reviewed, no adverse event

``pt`` may be a PT name or an LLT (mapped to its PT); ``suspect_product`` is
optional and only used by the strict metric. Every gold post_id defines the
evaluation set, so posts with no AE must be listed with an empty ``pt``.

Metrics (all at PT level):
  post_detection   does the post contain >= 1 AE?           (binary, with specificity)
  event            (post, PT) pairs                          (headline metric)
  event_strict     (post, suspect product, PT) triples
Each comes as a confidence-threshold sweep, with a bootstrap 95% CI on the
headline F1. Agent-level competency is computed from audit.jsonl: each coder
alone, each verifier as a filter, each rater as a ranker (AUROC, Brier).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .meddra import MedDRA

THRESHOLDS = [round(x, 2) for x in np.arange(0.0, 1.0, 0.05)]


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def load_gold(path: str, meddra: MedDRA) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(path, dtype=str) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    if "pt" not in df.columns:
        df = df.rename(columns={"llt": "pt", "term": "pt", "pt_name": "pt"})
    if "suspect_product" not in df.columns:
        df["suspect_product"] = None
    df["post_id"] = df["post_id"].astype(str).str.strip()
    unknown, pts = [], []
    for term in df["pt"]:
        if term is None or pd.isna(term) or not str(term).strip():
            pts.append(None)
            continue
        pt = meddra.to_pt(term)
        if pt is None:
            unknown.append(str(term))
        pts.append(pt.pt_name if pt else None)
    df["pt_name"] = pts
    df = df[df["pt_name"].notna() | df["pt"].isna() | (df["pt"].astype(str).str.strip() == "")]
    return df[["post_id", "suspect_product", "pt_name"]], sorted(set(unknown))


def _pairs(df: pd.DataFrame, strict: bool) -> set[tuple]:
    df = df[df["pt_name"].notna()]
    cols = ["post_id", "suspect_product", "pt_name"] if strict else ["post_id", "pt_name"]
    return set(map(tuple, df[cols].astype(str).values))


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pt_name"] = df["pt_name"].where(df["pt_name"].isna(), df["pt_name"].astype(str).str.strip().str.lower())
    df["suspect_product"] = df["suspect_product"].fillna("").astype(str).str.strip().str.lower()
    return df


def score(pred: pd.DataFrame, gold: pd.DataFrame, posts: list[str]) -> dict:
    """pred/gold: columns post_id, suspect_product, pt_name (already thresholded)."""
    pred, gold = _norm(pred), _norm(gold)
    out = {}
    for name, strict in (("event", False), ("event_strict", True)):
        p, g = _pairs(pred, strict), _pairs(gold, strict)
        out[name] = prf(len(p & g), len(p - g), len(g - p))
    p_posts = set(pred.loc[pred["pt_name"].notna(), "post_id"])
    g_posts = set(gold.loc[gold["pt_name"].notna(), "post_id"])
    all_posts = set(posts)
    det = prf(len(p_posts & g_posts), len(p_posts - g_posts), len(g_posts - p_posts))
    tn = len(all_posts - p_posts - g_posts)
    det["tn"] = tn
    det["specificity"] = round(tn / (tn + det["fp"]), 4) if tn + det["fp"] else 0.0
    out["post_detection"] = det
    return out


def bootstrap_f1(pred: pd.DataFrame, gold: pd.DataFrame, posts: list[str], n: int = 2000, seed: int = 0):
    """95% CI for event-level F1, resampling posts with replacement."""
    pred, gold = _norm(pred), _norm(gold)
    p, g = _pairs(pred, False), _pairs(gold, False)
    idx = {pid: i for i, pid in enumerate(posts)}
    tp, fp, fn = (np.zeros(len(posts)) for _ in range(3))
    for pid, _ in p & g:
        tp[idx[pid]] += 1
    for pid, _ in p - g:
        fp[idx[pid]] += 1
    for pid, _ in g - p:
        fn[idx[pid]] += 1
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(posts), size=(n, len(posts)))
    T, F, N = tp[samples].sum(1), fp[samples].sum(1), fn[samples].sum(1)
    f1 = np.where(2 * T + F + N > 0, 2 * T / np.maximum(2 * T + F + N, 1), 0.0)
    return round(float(np.percentile(f1, 2.5)), 4), round(float(np.percentile(f1, 97.5)), 4)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _agent_metrics(audit_path: str, gold: pd.DataFrame, posts: list[str], meddra: MedDRA) -> pd.DataFrame:
    gold_pairs = _pairs(_norm(gold), False)
    records = {}
    with open(audit_path) as f:
        for line in f:
            rec = json.loads(line)
            records[rec["post_id"]] = rec
    rows = []

    # Coders: each one on its own, before any panel consensus.
    n_coders = max((len(r.get("coders") or []) for r in records.values()), default=0)
    for k in range(n_coders):
        pred = set()
        for pid in posts:
            coders = records.get(pid, {}).get("coders") or []
            if k < len(coders):
                for ev in coders[k]["events"]:
                    if (t := meddra.get(ev["llt_code"])) is not None:
                        pred.add((pid, t.pt_name.strip().lower()))
        rows.append({"role": "coder", "agent": k + 1, "metric": "event P/R/F1 (alone)",
                     **prf(len(pred & gold_pairs), len(pred - gold_pairs), len(gold_pairs - pred))})

    # Verifiers and raters: judged on the candidate events the coder panel produced.
    cands = []
    for pid in posts:
        rec = records.get(pid, {})
        for ev in rec.get("events", []):
            cands.append((pid, ev["event_id"], (pid, str(ev["pt_name"]).strip().lower()) in gold_pairs))
    y = np.array([c[2] for c in cands], dtype=int)
    n_ver = max((len(r.get("verifiers") or []) for r in records.values()), default=0)
    for k in range(n_ver):
        keep = []
        for pid, eid, _ in cands:
            vs = records[pid]["verifiers"]
            v = next((v for v in vs[k]["verdicts"] if v["event_id"] == eid), None) if k < len(vs) else None
            keep.append(bool(v and v["is_genuine_adverse_experience"] and v["coding_correct"]))
        keep = np.array(keep, dtype=bool)
        rows.append({"role": "verifier", "agent": k + 1, "metric": "accept/reject candidates",
                     **prf(int((keep & (y == 1)).sum()), int((keep & (y == 0)).sum()), int((~keep & (y == 1)).sum())),
                     "accuracy": round(float((keep == (y == 1)).mean()), 4) if len(y) else None})
    n_rat = max((len(r.get("raters") or []) for r in records.values()), default=0)
    for k in range(n_rat):
        s = []
        for pid, eid, _ in cands:
            rs = records[pid]["raters"]
            r = next((r for r in rs[k]["ratings"] if r["event_id"] == eid), None) if k < len(rs) else None
            s.append(r["p_adverse_event"] * r["p_product_attribution"] * r["p_coding_accuracy"] if r else 0.0)
        s = np.clip(np.array(s, dtype=float), 0, 1)
        rows.append({"role": "rater", "agent": k + 1, "metric": "rank candidates",
                     "auroc": round(auroc(y, s), 4) if len(y) else None,
                     "brier": round(float(((s - y) ** 2).mean()), 4) if len(y) else None,
                     "n_candidates": len(y), "n_true": int(y.sum())})
    return pd.DataFrame(rows)


def evaluate(out_dir: str, gold_path: str, meddra: MedDRA, threshold: float = 0.5) -> dict[str, pd.DataFrame]:
    events = pd.read_csv(os.path.join(out_dir, "coded_events.csv"), dtype={"post_id": str})
    processed = pd.read_csv(os.path.join(out_dir, "posts.csv"), dtype={"post_id": str})
    gold, unknown = load_gold(gold_path, meddra)
    if unknown:
        print(f"warning: {len(unknown)} gold terms not found in MedDRA and ignored: {unknown[:10]}")

    gold_posts = sorted(set(gold["post_id"]))
    posts = [p for p in gold_posts if p in set(processed["post_id"])]
    if missing := sorted(set(gold_posts) - set(posts)):
        print(f"warning: {len(missing)} gold posts were not in the run and are excluded: {missing[:10]}")
    gold = gold[gold["post_id"].isin(posts)]
    events = events[events["post_id"].isin(posts)]

    # Threshold sweep
    sweep = []
    for t in THRESHOLDS:
        m = score(events[events["confidence"] >= t], gold, posts)
        for level, vals in m.items():
            sweep.append({"threshold": t, "level": level, **vals})
    sweep = pd.DataFrame(sweep)
    ev = sweep[sweep["level"] == "event"]
    best_t = float(ev.loc[ev["f1"] == ev["f1"].max(), "threshold"].max())  # highest threshold among ties

    # Headline at the operating threshold and at the best-F1 threshold
    summary = []
    for label, t in (("operating", threshold), ("best_f1", best_t)):
        pred = events[events["confidence"] >= t]
        lo, hi = bootstrap_f1(pred, gold, posts)
        for level, vals in score(pred, gold, posts).items():
            row = {"threshold_type": label, "threshold": t, "level": level, **vals}
            if level == "event":
                row.update(f1_ci95_low=lo, f1_ci95_high=hi)
            summary.append(row)
    summary = pd.DataFrame(summary)

    # Per-PT performance at the operating threshold
    pred = _norm(events[events["confidence"] >= threshold])
    g = _norm(gold)
    p_pairs, g_pairs = _pairs(pred, False), _pairs(g, False)
    per_pt = pd.DataFrame([{"pt_name": pt, "support": sum(1 for x in g_pairs if x[1] == pt),
                            **prf(sum(1 for x in p_pairs & g_pairs if x[1] == pt),
                                  sum(1 for x in p_pairs - g_pairs if x[1] == pt),
                                  sum(1 for x in g_pairs - p_pairs if x[1] == pt))}
                           for pt in sorted({x[1] for x in p_pairs | g_pairs})])
    per_pt = per_pt.sort_values(["support", "f1"], ascending=[False, True])

    # Calibration: is a 0.8 right 80% of the time?
    allev = _norm(events)
    allev["correct"] = [(pid, pt) in g_pairs for pid, pt in zip(allev["post_id"], allev["pt_name"].astype(str))]
    allev["bin"] = pd.cut(allev["confidence"], [0, 0.2, 0.4, 0.6, 0.8, 1.0001], right=False, include_lowest=True)
    calibration = (allev.groupby("bin", observed=False)
                   .agg(n=("correct", "size"), mean_confidence=("confidence", "mean"), observed_precision=("correct", "mean"))
                   .round(4).reset_index())
    calibration["bin"] = calibration["bin"].astype(str)

    # Error listing for review: false positives and false negatives with the post text
    text = processed.set_index("post_id")["text"].to_dict() if "text" in processed else {}
    status = processed.set_index("post_id")["status"].to_dict()
    errors = [{"post_id": pid, "error": "false_positive", "pt_name": pt,
               "confidence": float(pred.loc[(pred.post_id == pid) & (pred.pt_name == pt), "confidence"].max()),
               "verbatims": "; ".join(map(str, pred.loc[(pred.post_id == pid) & (pred.pt_name == pt), "verbatims"])),
               "text": text.get(pid)} for pid, pt in sorted(p_pairs - g_pairs)]
    errors += [{"post_id": pid, "error": "false_negative", "pt_name": pt,
                "missed_by": "prefilter" if status.get(pid) == "no_product_mention" else "pipeline",
                "text": text.get(pid)} for pid, pt in sorted(g_pairs - p_pairs)]
    errors = pd.DataFrame(errors)

    tables = {"eval_summary": summary, "eval_threshold_sweep": sweep, "eval_per_pt": per_pt,
              "eval_calibration": calibration, "eval_errors": errors}
    audit = os.path.join(out_dir, "audit.jsonl")
    if os.path.exists(audit):
        tables["eval_agents"] = _agent_metrics(audit, gold, posts, meddra)
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    return tables
