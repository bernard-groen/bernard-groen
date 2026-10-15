"""Offline tests: no API calls. Agents are replaced by fakes."""

import asyncio
import json
import math
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from ae_signal.agents import Agents
from ae_signal.aggregate import merge_coders, score_events
from ae_signal.config import Config
from ae_signal.meddra import MedDRA
from ae_signal.pipeline import matched_products, run
from ae_signal.prr import prr_2x2
from ae_signal.schemas import (CoderOutput, EventRating, EventVerdict, ProposedEvent, RaterOutput,
                               VerifierOutput, strict_json_schema)

HERE = os.path.dirname(__file__)
MEDDRA_DIR = os.path.join(HERE, "fixtures", "meddra_mini")
POSTS = os.path.join(HERE, "..", "data", "sample_posts_synthetic.csv")


@pytest.fixture(scope="module")
def meddra():
    return MedDRA(MEDDRA_DIR)


def test_meddra_loads_current_terms_only(meddra):
    assert meddra.version == "26.1"
    assert meddra.get("91999999") is None  # non-current LLT
    fever = next(t for t in meddra.terms.values() if t.llt_name == "Fever")
    assert fever.pt_name == "Pyrexia"
    assert fever.soc_name.startswith("General disorders")


def test_meddra_search_ranks_relevant_term_first(meddra):
    assert meddra.search("fever", 3)[0]["pt_name"] == "Pyrexia"
    assert meddra.search("Guillain Barre", 3)[0]["pt_name"] == "Guillain-Barre syndrome"


def test_prr_matches_hand_calculation():
    r = prr_2x2(10, 90, 20, 880)
    assert r["prr"] == pytest.approx((10 / 100) / (20 / 900))
    n, ad_bc = 1000, 10 * 880 - 90 * 20
    expected_chi2 = n * (abs(ad_bc) - n / 2) ** 2 / (100 * 900 * 30 * 970)
    assert r["chi2_yates"] == pytest.approx(expected_chi2)
    se = math.sqrt(1 / 10 - 1 / 100 + 1 / 20 - 1 / 900)
    assert r["prr_lower95"] == pytest.approx(r["prr"] * math.exp(-1.96 * se))
    assert not r["zero_cell_corrected"]


def test_prr_zero_cell_is_corrected():
    r = prr_2x2(3, 5, 0, 40)
    assert r["zero_cell_corrected"] and math.isfinite(r["prr"])


def test_strict_schema_is_closed_and_inlined():
    s = json.dumps(strict_json_schema(CoderOutput))
    assert "$ref" not in s and "$defs" not in s
    events = strict_json_schema(CoderOutput)["properties"]["events"]["items"]
    assert events["additionalProperties"] is False
    assert set(events["required"]) == set(ProposedEvent.model_fields)


def test_prefilter():
    assert matched_products("Got Abrysvo and a flu shot") == ["Abrysvo", "Influenza vaccine"]
    assert matched_products("my headache today is unreal") == []
    assert matched_products("had my RSV jab") == ["RSV vaccine (unspecified)"]


def _ev(llt, product="Abrysvo", verbatim="x"):
    return ProposedEvent(verbatim=verbatim, llt_code=llt, suspect_product=product, event_kind="adverse_event",
                         experiencer="poster", onset_after_dose="yes", seriousness_indicated=False, rationale="")


def _coder(*events):
    return CoderOutput(post_type="first_hand_experience", product_mentions=[], events=list(events))


def test_merge_coders_groups_llts_under_same_pt(meddra):
    # "Fever" (91000003) and "Pyrexia" (90000002) share a PT -> one event, full agreement.
    fever = next(c for c, t in meddra.terms.items() if t.llt_name == "Fever")
    outs = [_coder(_ev(fever)), _coder(_ev(fever)), _coder(_ev("90000002"), _ev("90000005"))]
    events = merge_coders(outs, meddra, min_agreement=0.3)
    by_pt = {e["pt_name"]: e for e in events}
    assert by_pt["Pyrexia"]["coder_agreement"] == 1.0
    assert by_pt["Pyrexia"]["llt_name"] == "Fever"  # majority LLT
    assert by_pt["Headache"]["coder_agreement"] == pytest.approx(1 / 3, abs=1e-3)
    assert len(merge_coders(outs, meddra, min_agreement=0.5)) == 1


def test_score_events_combines_votes():
    events = [{"event_id": "E1", "coder_agreement": 1.0}]
    v = [VerifierOutput(verdicts=[EventVerdict(event_id="E1", is_genuine_adverse_experience=ok,
                                               product_attribution_correct=True, coding_correct=True,
                                               better_term_suggestion="", comment="")], missed_events=[])
         for ok in (True, False)]
    r = [RaterOutput(ratings=[EventRating(event_id="E1", p_adverse_event=0.9, p_product_attribution=1.0,
                                          p_temporal_association=0.5, p_coding_accuracy=1.0, rationale="")])]
    [s] = score_events(events, v, r)
    assert s["p_ae"] == pytest.approx((0.5 + 0.9) / 2)
    assert s["confidence"] == pytest.approx(0.7, abs=1e-3)


# ---- coder tool loop against a fake Anthropic client -----------------------

class FakeClient:
    def __init__(self, responses):
        self.calls = []
        self._responses = iter(responses)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self._responses)


def _text(obj):
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=obj.model_dump_json())])


def test_coder_searches_then_rejects_invented_codes(meddra):
    tool_call = SimpleNamespace(stop_reason="tool_use", content=[
        SimpleNamespace(type="tool_use", id="tu1", name="search_meddra", input={"query": "fever"})])
    client = FakeClient([tool_call, _text(_coder(_ev("12345678"))), _text(_coder(_ev("90000002")))])
    agents = Agents(Config(meddra_dir=MEDDRA_DIR), meddra, client=client)

    out = asyncio.run(agents.code({"post_id": "1", "text": "Abrysvo gave me a fever"}))
    assert out.events[0].llt_code == "90000002"
    tool_result = client.calls[1]["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result" and "Pyrexia" in tool_result["content"]
    assert "not current MedDRA LLTs" in client.calls[2]["messages"][4]["content"]
    assert client.calls[0]["fallbacks"] == "default"


# ---- full pipeline with fake agents ---------------------------------------

class FakeAgents:
    """Codes by keyword so the sample file produces a deterministic event table."""

    KEYWORDS = {"fever": "90000002", "sore": "90000001", "headache": "90000005", "guillain": "90000006",
                "faint": "90000007", "joint pain": "90000009", "preterm": "90000010", "tired": "90000003"}

    async def code(self, post):
        text = post["text"].lower()
        products = matched_products(post["text"])
        product = "Abrysvo" if "Abrysvo" in products else (products[0] if products else "")
        if "terrified" in text or "news" in text or "zero side effects" in text:
            return _coder()
        return _coder(*[_ev(code, product, kw) for kw, code in self.KEYWORDS.items() if kw in text])

    async def verify(self, post, events):
        return VerifierOutput(verdicts=[EventVerdict(event_id=e["event_id"], is_genuine_adverse_experience=True,
                                                     product_attribution_correct=True, coding_correct=True,
                                                     better_term_suggestion="", comment="") for e in events],
                              missed_events=[])

    async def rate(self, post, events):
        return RaterOutput(ratings=[EventRating(event_id=e["event_id"], p_adverse_event=0.9, p_product_attribution=0.9,
                                                p_temporal_association=0.8, p_coding_accuracy=0.9, rationale="")
                                    for e in events])


def test_pipeline_end_to_end(tmp_path):
    cfg = Config(meddra_dir=MEDDRA_DIR, min_cases=1)
    tables = asyncio.run(run(POSTS, str(tmp_path), cfg, agents=FakeAgents()))

    posts = tables["posts"].set_index("post_id")
    assert posts.loc["S015", "status"] == "no_product_mention"
    assert posts.loc["S004", "status"] == "no_events"
    events = tables["coded_events"]
    assert set(events.loc[events.post_id == "S003", "pt_name"]) == {"Guillain-Barre syndrome"}
    prr = tables["prr_pt"].set_index("pt_name")
    assert prr.loc["Pyrexia", "a_target_with_term"] == 1
    for name in ("posts.csv", "coded_events.csv", "prr_pt.csv", "prr_soc.csv", "event_summary.csv", "audit.jsonl"):
        assert (tmp_path / name).exists()

    # Resume: a second run makes no agent calls and reproduces the same tables.
    again = asyncio.run(run(POSTS, str(tmp_path), cfg, agents=_NoCalls()))
    pd.testing.assert_frame_equal(again["prr_pt"], tables["prr_pt"])


class _NoCalls:
    def __getattr__(self, name):
        raise AssertionError("agents should not be called on resume")
