"""Structured-output schemas for the three agent roles.

Each role answers in JSON constrained by these models (``output_config.format``).
Probabilities are plain floats; range checks happen in ``aggregate`` because
structured outputs do not enforce numeric bounds.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# ---- Coder ---------------------------------------------------------------

class ProductMention(BaseModel):
    product: str  # canonical name from the product list, or the unspecified-RSV label
    evidence: str  # the words in the post that identify it
    patient_received: bool  # did the person whose experience is described receive it?


class ProposedEvent(BaseModel):
    verbatim: str  # the reporter's own words for the event
    llt_code: str  # chosen from search_meddra results only
    suspect_product: str
    event_kind: Literal["adverse_event", "lack_of_efficacy", "medication_error", "exposure_in_pregnancy", "other"]
    experiencer: Literal["poster", "third_party", "unclear"]
    onset_after_dose: Literal["yes", "no", "unclear"]
    seriousness_indicated: bool  # hospitalisation, death, disability, life-threatening, etc.
    rationale: str


class CoderOutput(BaseModel):
    post_type: Literal[
        "first_hand_experience", "second_hand_report", "news_or_media",
        "question_or_speculation", "promotional", "unrelated",
    ]
    product_mentions: list[ProductMention]
    events: list[ProposedEvent]


# ---- Verifier ------------------------------------------------------------

class EventVerdict(BaseModel):
    event_id: str
    is_genuine_adverse_experience: bool  # false if negated, hypothetical, an indication, a feared effect, etc.
    product_attribution_correct: bool
    coding_correct: bool  # is this LLT/PT the best MedDRA representation of the verbatim?
    better_term_suggestion: str  # free text, empty if none
    comment: str


class VerifierOutput(BaseModel):
    verdicts: list[EventVerdict]
    missed_events: list[str]  # verbatims the coders failed to capture


# ---- Confidence rater ----------------------------------------------------

class EventRating(BaseModel):
    event_id: str
    p_adverse_event: float  # probability the post describes this event actually happening
    p_product_attribution: float  # probability the suspect product is correctly identified
    p_temporal_association: float  # probability onset plausibly follows the dose
    p_coding_accuracy: float  # probability the MedDRA term is correct
    rationale: str


class RaterOutput(BaseModel):
    ratings: list[EventRating]


def strict_json_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema with $refs inlined, titles stripped and objects closed."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            out = {k: walk(v) for k, v in node.items() if k not in ("title", "default")}
            if out.get("type") == "object":
                out["additionalProperties"] = False
                out["required"] = list(out.get("properties", {}))
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)
