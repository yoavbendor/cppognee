"""--link-weights det: the pure weighting (no DB). Run from apps/docmem: python -m pytest tests/test_link_weights.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docmem.ask import LEAD_BONUS, OWN_BONUS, OWN_SUFFIX, weigh_links  # noqa: E402


def test_link_score_is_idf_times_support_times_lead():
    rec = weigh_links({"p": ["T"]}, {"p": 0.5}, {"T": 2.0}, {"T": 3}, {"p": True}, 0.5)
    assert rec["p"]["targets"][0]["contribution"] == round(2.0 * 3 * LEAD_BONUS, 4)
    assert rec["p"]["link_score"] == round(2.0 * 3 * LEAD_BONUS, 4)
    rec = weigh_links({"p": ["T"]}, {"p": 0.5}, {"T": 2.0}, {"T": 3}, {"p": False}, 0.5)
    assert rec["p"]["link_score"] == 6.0


def test_link_scores_sum_over_targets_and_own_article_bonus():
    cands = {"p": ["A", "B" + OWN_SUFFIX]}
    rec = weigh_links(cands, {"p": 0.0}, {"A": 1.0, "B": 1.0}, {"A": 1, "B": 2}, {"p": False}, 0.0)
    assert rec["p"]["link_score"] == round(1.0 + 1.0 * 2 * OWN_BONUS, 4)
    own = rec["p"]["targets"][1]
    assert own["own"] and own["target"] == "B" + OWN_SUFFIX and own["support"] == 2


def test_normalized_best_gets_one_and_lambda_mixes():
    cands = {"a": ["T"], "b": ["T"], "c": ["T"]}
    lead = {"a": False, "b": False, "c": False}
    sims = {"a": 0.9, "b": 0.5, "c": 0.1}
    support = {"T": 1}
    # a link-free candidate scores 0 links, so only the similarity counts
    cands["d"] = []
    rec = weigh_links({k: v for k, v in cands.items() if v}, sims, {"T": 1.0}, support, {**lead, "d": False},
                      0.5)
    assert rec["a"]["norm_link"] == 1.0 and rec["b"]["norm_link"] == 1.0
    assert rec["a"]["final"] == round(0.5 * 1.0 + 0.5 * 0.9, 4)


def test_lambda_zero_is_similarity_and_lambda_one_is_links():
    cands = {"x": ["T"], "y": ["U"]}
    sims = {"x": 0.2, "y": 0.9}
    idf, support, lead = {"T": 3.0, "U": 1.0}, {"T": 1, "U": 1}, {"x": False, "y": False}
    by_sim = weigh_links(cands, sims, idf, support, lead, 0.0)
    by_link = weigh_links(cands, sims, idf, support, lead, 1.0)
    assert max(by_sim, key=lambda p: by_sim[p]["final"]) == "y"
    assert max(by_link, key=lambda p: by_link[p]["final"]) == "x"


def test_no_links_anywhere_is_safe():
    rec = weigh_links({"p": ["T"]}, {"p": 0.3}, {"T": 1.0}, {"T": 1}, {"p": True}, 0.5)
    assert rec["p"]["norm_link"] == 1.0 and rec["p"]["final"] == round(0.5 * 1.0 + 0.5 * 0.3, 4)
