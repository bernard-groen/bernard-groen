"""Combine the independent agent outputs into scored events.

Unit of analysis: one (post, suspect product, PT). Coders that pick different
LLTs under the same PT agree; coders that pick different PTs for the same words
produce separate candidate events, each with lower coder agreement.

Final confidence = p_ae x p_product x p_coding, where each component averages
the raters' probabilities with the verifiers' votes (and, for coding, the coder
agreement). Temporal plausibility is reported alongside but not multiplied in,
because most posts give no timing and that should not suppress them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

from .meddra import MedDRA
from .schemas import CoderOutput, RaterOutput, VerifierOutput


def _clip(p: float) -> float:
    return min(1.0, max(0.0, float(p)))


def merge_coders(coder_outputs: list[CoderOutput], meddra: MedDRA, min_agreement: float) -> list[dict]:
    n = len(coder_outputs)
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for coder_idx, out in enumerate(coder_outputs):
        seen = set()
        for ev in out.events:
            term = meddra.get(ev.llt_code)
            if term is None:
                continue
            key = (ev.suspect_product, term.pt_code)
            if key in seen:  # one vote per coder per PT
                continue
            seen.add(key)
            groups[key].append((coder_idx, ev, term))

    events = []
    for i, ((product, pt_code), votes) in enumerate(sorted(groups.items()), start=1):
        agreement = len(votes) / n
        if agreement < min_agreement:
            continue
        llt_code = Counter(t.llt_code for _, _, t in votes).most_common(1)[0][0]
        term = meddra.get(llt_code)

        def mode(attr):
            return Counter(getattr(ev, attr) for _, ev, _ in votes).most_common(1)[0][0]

        events.append({
            "event_id": f"E{i}",
            "suspect_product": product,
            "verbatims": sorted({ev.verbatim for _, ev, _ in votes}),
            "llt_code": term.llt_code, "llt_name": term.llt_name,
            "pt_code": term.pt_code, "pt_name": term.pt_name,
            "hlt_name": term.hlt_name, "hlgt_name": term.hlgt_name,
            "soc_code": term.soc_code, "soc_name": term.soc_name,
            "event_kind": mode("event_kind"),
            "experiencer": mode("experiencer"),
            "onset_after_dose": mode("onset_after_dose"),
            "seriousness_indicated": sum(ev.seriousness_indicated for _, ev, _ in votes) * 2 > len(votes),
            "coder_votes": len(votes),
            "coder_agreement": round(agreement, 3),
        })
    return events


def score_events(events: list[dict], verifier_outputs: list[VerifierOutput],
                 rater_outputs: list[RaterOutput]) -> list[dict]:
    verdicts = defaultdict(list)
    for out in verifier_outputs:
        for v in out.verdicts:
            verdicts[v.event_id].append(v)
    ratings = defaultdict(list)
    for out in rater_outputs:
        for r in out.ratings:
            ratings[r.event_id].append(r)

    scored = []
    for ev in events:
        vs, rs = verdicts.get(ev["event_id"], []), ratings.get(ev["event_id"], [])
        # A missing vote counts against the event rather than being ignored.
        n_v, n_r = max(len(verifier_outputs), 1), max(len(rater_outputs), 1)
        v_ae = sum(v.is_genuine_adverse_experience for v in vs) / n_v
        v_prod = sum(v.product_attribution_correct for v in vs) / n_v
        v_code = sum(v.coding_correct for v in vs) / n_v
        r_ae = sum(_clip(r.p_adverse_event) for r in rs) / n_r
        r_prod = sum(_clip(r.p_product_attribution) for r in rs) / n_r
        r_code = sum(_clip(r.p_coding_accuracy) for r in rs) / n_r
        r_temp = sum(_clip(r.p_temporal_association) for r in rs) / n_r

        p_ae = mean([v_ae, r_ae])
        p_product = mean([v_prod, r_prod])
        p_coding = mean([v_code, r_code, ev["coder_agreement"]])
        scored.append({
            **ev,
            "verifier_ae_agree": round(v_ae, 3),
            "verifier_product_agree": round(v_prod, 3),
            "verifier_coding_agree": round(v_code, 3),
            "verifier_suggestions": " | ".join(v.better_term_suggestion for v in vs if v.better_term_suggestion),
            "rater_p_ae": round(r_ae, 3),
            "rater_p_product": round(r_prod, 3),
            "rater_p_coding": round(r_code, 3),
            "p_temporal": round(r_temp, 3),
            "p_ae": round(p_ae, 3),
            "p_product": round(p_product, 3),
            "p_coding": round(p_coding, 3),
            "confidence": round(p_ae * p_product * p_coding, 3),
            "rater_rationales": " | ".join(r.rationale for r in rs if r.rationale),
        })
    return scored
