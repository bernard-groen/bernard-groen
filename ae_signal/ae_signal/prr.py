"""Proportional Reporting Ratio (Evans, Waller & Davis 2001).

A "report" is one (post, product) pair with at least one qualifying event, the
social-media analogue of a spontaneous report. For target product T and term X:

                    term X    other terms
    target T          a           b
    comparators       c           d

    PRR = [a / (a + b)] / [c / (c + d)]
    95% CI = exp( ln(PRR) +/- 1.96 * sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d)) )
    chi2 (Yates) = N (|ad - bc| - N/2)^2 / ((a+b)(c+d)(a+c)(b+d))

Signal (Evans criteria): PRR >= 2, chi2 >= 4 and a >= 3.

A confidence-weighted PRR is reported alongside: each report contributes its
event confidence instead of 1, so low-confidence posts move the ratio less.
"""

from __future__ import annotations

import math

import pandas as pd


def prr_2x2(a: float, b: float, c: float, d: float) -> dict:
    corrected = min(a, b, c, d) == 0
    if corrected:  # Haldane-Anscombe correction so the ratio and CI stay finite
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    prr = (a / (a + b)) / (c / (c + d))
    se = math.sqrt(max(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d), 0.0))
    n = a + b + c + d
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    chi2 = n * max(abs(a * d - b * c) - n / 2, 0) ** 2 / denom if denom else 0.0
    return {
        "prr": prr,
        "prr_lower95": math.exp(math.log(prr) - 1.96 * se),
        "prr_upper95": math.exp(math.log(prr) + 1.96 * se),
        "chi2_yates": chi2,
        "zero_cell_corrected": corrected,
    }


def compute_prr(events: pd.DataFrame, target: str, comparators: list[str] | None = None,
                level: str = "pt", prr_threshold: float = 2.0, chi2_threshold: float = 4.0,
                min_cases: int = 3) -> pd.DataFrame:
    """events: one row per (post_id, suspect_product, pt) with confidence and hierarchy columns.

    level: "pt" or "soc".
    """
    code_col, name_col = {"pt": ("pt_code", "pt_name"), "soc": ("soc_code", "soc_name")}[level]
    df = events.copy()
    if comparators is not None:
        df = df[df["suspect_product"].isin([target, *comparators])]
    df["is_target"] = df["suspect_product"] == target

    # Collapse to one row per report per term, keeping the best confidence.
    per_term = (df.groupby(["post_id", "suspect_product", "is_target", code_col, name_col],
                           as_index=False)["confidence"].max())
    reports = per_term.groupby(["post_id", "suspect_product", "is_target"], as_index=False)["confidence"].max()

    n_target = int(reports["is_target"].sum())
    n_comp = int((~reports["is_target"]).sum())
    w_target = reports.loc[reports["is_target"], "confidence"].sum()
    w_comp = reports.loc[~reports["is_target"], "confidence"].sum()

    rows = []
    for (code, name), g in per_term.groupby([code_col, name_col]):
        a = int(g["is_target"].sum())
        if a == 0:
            continue
        c = int((~g["is_target"]).sum())
        stats = prr_2x2(a, n_target - a, c, n_comp - c)

        a_w = g.loc[g["is_target"], "confidence"].sum()
        c_w = g.loc[~g["is_target"], "confidence"].sum()
        weighted = prr_2x2(a_w, w_target - a_w, c_w, w_comp - c_w)["prr"] if w_target and w_comp else float("nan")

        rows.append({
            code_col: code, name_col: name,
            "a_target_with_term": a, "b_target_other": n_target - a,
            "c_comparator_with_term": c, "d_comparator_other": n_comp - c,
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()},
            "prr_confidence_weighted": round(weighted, 3),
            "mean_confidence": round(g.loc[g["is_target"], "confidence"].mean(), 3),
            "signal": stats["prr"] >= prr_threshold and stats["chi2_yates"] >= chi2_threshold and a >= min_cases,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["signal", "prr", "a_target_with_term"], ascending=False).reset_index(drop=True)
