"""Run configuration: products in scope, agent panel sizes, models and thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field

# Product dictionary. Keys are the canonical product names used in every output
# table; values are the surface forms matched (case-insensitive, word-bounded)
# during prefiltering and given to the agents as context. Extend freely.
TARGET_PRODUCT = "Abrysvo"

PRODUCTS: dict[str, list[str]] = {
    # Target: Pfizer RSV vaccine (RSVpreF, bivalent prefusion F)
    "Abrysvo": ["abrysvo", "pfizer rsv", "rsvpref", "pfizer's rsv", "pfizer rsv vaccine"],
    # Comparators - the background against which PRR is computed
    "Arexvy": ["arexvy", "gsk rsv", "gsk's rsv", "rsvpref3"],
    "mRESVIA": ["mresvia", "moderna rsv", "moderna's rsv", "mrna-1345"],
    "Beyfortus (nirsevimab)": ["beyfortus", "nirsevimab"],
    "COVID-19 vaccine": ["covid vaccine", "covid jab", "covid shot", "covid booster",
                         "comirnaty", "spikevax", "novavax", "pfizer covid", "moderna covid"],
    "Influenza vaccine": ["flu shot", "flu jab", "flu vaccine", "influenza vaccine",
                          "fluzone", "flublok", "fluad"],
    "Shingles vaccine": ["shingrix", "shingles vaccine", "shingles shot", "shingles jab"],
    "Tdap vaccine": ["tdap", "boostrix", "adacel", "whooping cough vaccine"],
    "Pneumococcal vaccine": ["prevnar", "pneumovax", "capvaxive", "pneumonia vaccine", "pneumonia shot"],
}

# Generic RSV mentions ("got my RSV shot") are ambiguous between brands. They are
# kept for the agents to resolve from context; unresolved ones are reported as
# "RSV vaccine (unspecified)" and excluded from the target/comparator split.
UNSPECIFIED_RSV = "RSV vaccine (unspecified)"
GENERIC_RSV_TERMS = ["rsv vaccine", "rsv shot", "rsv jab", "rsv vax", "rsv injection"]


@dataclass
class RoleConfig:
    n_agents: int
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_tokens: int = 16000


@dataclass
class Config:
    # ASCII distribution directory, or an LLT/PT table (.xlsx/.csv)
    meddra_dir: str
    meddra_version: str = "28.0"

    coders: RoleConfig = field(default_factory=lambda: RoleConfig(n_agents=3))
    verifiers: RoleConfig = field(default_factory=lambda: RoleConfig(n_agents=2, effort="high"))
    raters: RoleConfig = field(default_factory=lambda: RoleConfig(n_agents=2))

    # Max concurrent API calls across all agents.
    concurrency: int = 8
    # Max search_meddra round-trips a coder may make for one post.
    max_coder_turns: int = 8
    # Candidate LLTs returned per search_meddra call.
    search_top_k: int = 15

    # A coded event enters the signal tables only if its final confidence clears this.
    min_confidence: float = 0.5
    # An event is carried forward only if at least this fraction of coders proposed its PT.
    # The default keeps single-coder events (recall first): verifiers and raters then
    # judge them, and the low coder agreement pulls their confidence down.
    min_coder_agreement: float = 0.3

    # PRR signal criteria (Evans et al. 2001).
    prr_threshold: float = 2.0
    chi2_threshold: float = 4.0
    min_cases: int = 3

    target_product: str = TARGET_PRODUCT
    # None = every product other than the target is a comparator.
    comparator_products: list[str] | None = None
