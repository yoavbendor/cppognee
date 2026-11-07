"""Alignment: name splitting and matching, score combination, pair selection, the matching on a small example, and
the eval's matching of model items. No cognee, no network.

Run from apps/docmem:  python -m pytest tests/test_align.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from docmem import ask, evaluate  # noqa: E402
from docmem.align import (WEIGHTS, align_rows, combined_score, jaccard, name_tokens, parse_label,  # noqa: E402
                          select_pairs, _best_rows)


def test_name_tokens_split_camel_case_digits_quotes_and_hyphens():
    assert name_tokens("ApolloLunarModule") == ("apollo", "lunar", "module")
    assert name_tokens("apollo_lunar_module") == ("apollo", "lunar", "module")
    assert name_tokens("Apollo11Mission") == ("apollo", "11", "mission")
    assert name_tokens("'Walt Cunningham'") == ("walt", "cunningham")
    assert name_tokens("clr-R114") == ("clr", "114")  # single letters are dropped
    assert name_tokens("LMDSThrottleableEngine") == ("lmds", "throttleable", "engine")
    assert name_tokens("S-IVB") == ("ivb",)
    assert name_tokens("") == ()


def test_title_and_name_match_by_jaccard():
    assert jaccard(name_tokens("ApolloLunarModule"), name_tokens("Apollo Lunar Module")) == 1.0
    assert jaccard(name_tokens("Apollo11MissionSystem"), name_tokens("Apollo 11")) == 2 / 4
    assert jaccard(name_tokens("Mission"), name_tokens("Apollo 11")) == 0.0
    assert jaccard((), ()) == 0.0


def test_parse_label_gives_kind_and_name():
    assert parse_label("part def ApolloLunarModule") == ("part def", "ApolloLunarModule")
    assert parse_label("requirement 'clr-R114'") == ("requirement", "'clr-R114'")
    assert parse_label("individual part def 'Walt Cunningham'") == ("individual part def", "'Walt Cunningham'")
    assert parse_label("ApolloLunarModule") is None  # no kind: not a declaration


def test_combined_score_is_the_weighted_sum_and_a_missing_signal_counts_zero():
    assert WEIGHTS == {"name": 0.5, "keyword": 0.2, "vector": 0.3}
    assert combined_score({"name": 1.0, "keyword": 1.0, "vector": 1.0}, WEIGHTS) == 1.0
    assert abs(combined_score({"name": 0.5}, WEIGHTS) - 0.25) < 1e-12
    assert combined_score({}, WEIGHTS) == 0.0
    assert combined_score({"name": 1.0, "other": 9.0}, {"name": 0.5}) == 0.5


def test_select_pairs_keeps_the_top_above_threshold_best_first():
    scores = {"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.2}
    assert select_pairs(scores, 3, 0.35) == [("a", 0.9), ("b", 0.5), ("c", 0.5)]
    assert select_pairs(scores, 1, 0.35) == [("a", 0.9)]
    assert select_pairs(scores, 3, 0.95) == []


def test_best_rows_keeps_one_row_per_key_by_score():
    rows = [{"id": 1, "score": 0.4}, {"id": 1, "score": 0.9}, {"id": 2, "score": 0.5}, {"id": 3, "score": 0.1}]
    assert [r["score"] for r in _best_rows(rows, 5, lambda r: r["id"])] == [0.9, 0.5, 0.1]
    assert len(_best_rows(rows, 2, lambda r: r["id"])) == 2


def test_align_rows_matches_by_name_heading_and_vector():
    docs = [
        {"title_tokens": ("apollo", "lunar", "module"), "heading_tokens": [("descent",)]},
        {"title_tokens": ("saturn",), "heading_tokens": [("crew",)]},
    ]
    kb = np.eye(4, 3, dtype=np.float32)  # passages p0, p1 (doc 0), p2, p3 (doc 1), orthogonal directions
    passage_doc = np.array([0, 0, 1, 1])
    ids = ["p0", "p1", "p2", "p3"]
    model = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)
    elements = [
        {"name_tokens": ("apollo", "lunar", "module"), "query": "apollo lunar module", "vec_rows": [0]},
        {"name_tokens": ("crew",), "query": "crew", "vec_rows": [1]},
    ]
    bm25 = {"apollo lunar module": [("p0", 6.0), ("p1", 3.0)], "crew": [("p3", 2.0)]}
    rows, best, info = align_rows(elements, docs, kb, passage_doc, ids, model, bm25, WEIGHTS, 3, 0.0)
    by_el = {}
    for e, d, score, signals, pids in rows:
        by_el.setdefault(e, []).append((d, score, signals, pids))
    # element 0 is named like doc 0's title and matches its passage vector best
    top0 = max(by_el[0], key=lambda x: x[1])
    assert top0[0] == 0 and top0[2]["name"] == 1.0 and top0[2]["vector"] > 0
    assert top0[3][0] == "p0"
    # element 1 matches doc 1 by its heading "Crew" (name), and by keyword on p3
    top1 = max(by_el[1], key=lambda x: x[1])
    assert top1[0] == 1 and top1[2]["name"] == 1.0 and abs(top1[2]["keyword"] - 2 / 6) < 1e-3
    # the BM25 scale is the best score any name reaches; the vector baseline is the mean element-passage cosine
    assert info["bm25_scale"] == 6.0
    assert len(best) == 2 and best[0] >= best[1]


def test_align_rows_drops_pairs_below_threshold():
    docs = [{"title_tokens": ("moon",), "heading_tokens": []}]
    kb = np.eye(2, 2, dtype=np.float32)
    elements = [{"name_tokens": ("sun",), "query": "sun", "vec_rows": []}]
    rows, best, _ = align_rows(elements, docs, kb, np.array([0, 0]), ["p0", "p1"],
                               np.zeros((0, 2), dtype=np.float32), {}, WEIGHTS, 3, 0.1)
    assert rows == [] and best == [0.0]


def test_eval_matches_a_model_item_by_file_and_heading():
    hit = {"uri": "file:///models/TechnicalRequirementsPackage.sysml", "rank": 2,
           "path": "model › apollo11 › package TechnicalRequirementsPackage › requirement def HatchClearanceDimensions"}
    src = {"file": "TechnicalRequirementsPackage.sysml", "heading": "requirement def HatchClearanceDimensions"}
    assert evaluate.source_rank([hit], src) == 2
    assert evaluate.source_rank([hit], {**src, "heading": "requirement def SlaSeparationVelocity"}) is None
    assert evaluate.source_rank([hit], {"file": "SystemSpecificationPackage.sysml"}) is None
    assert evaluate.heading_in(hit["path"], "Requirement Def HatchClearanceDimensions")


def test_aligned_model_item_is_labelled_as_model():
    item = {"uri": "file:///models/TechnicalComponentsPackage.sysml", "line": 99, "end_line": 110,
            "path": "model › apollo11 › package P › part def ApolloLunarModule", "aligned": "model"}
    assert ask.label(item).startswith("model: TechnicalComponentsPackage.sysml:99-110")
    assert ask.label({**item, "aligned": None}).startswith("TechnicalComponentsPackage.sysml:99-110")


# -- propagation through model relations (no cognee) ---------------------------------------------------------------

from docmem.align import DECAY, KEEP_PROPAGATED, MAX_HOPS, model_links, propagate  # noqa: E402


def _entry(score, passages=("p",)):
    return {"score": score, "signals": {}, "passages": list(passages)}


def _el(sid, qname, kind, metaclass=""):
    return {"section_id": sid, "qname": qname, "kind": kind, "metaclass": metaclass}


def test_propagate_takes_the_source_documents_with_decay_and_a_path():
    direct = {0: {7: _entry(0.5)}}  # element 0 is aligned to doc 7
    prop = propagate(direct, [(1, 0, "satisfied by")])
    assert set(prop) == {1}
    assert prop[1][7]["score"] == round(0.5 * DECAY, 4)
    assert prop[1][7]["origin"] == 0
    assert prop[1][7]["path"] == [{"rule": "satisfied by", "from": 0, "to": 1}]


def test_propagate_never_overwrites_direct_rows_or_feeds_them():
    direct = {0: {7: _entry(0.5)}, 1: {8: _entry(0.4)}}
    prop = propagate(direct, [(1, 0, "owned by"), (0, 1, "owned by")])
    assert prop == {}  # both elements have direct rows: neither takes anything


def test_propagate_ownership_gives_a_child_its_owners_documents():
    direct = {0: {3: _entry(0.6)}}  # the owner (a part def) is aligned
    prop = propagate(direct, [(1, 0, "owned by"), (2, 1, "owned by")])
    assert prop[1][3]["score"] == round(0.6 * DECAY, 4)
    # the grandchild is two hops away: decayed twice
    assert prop[2][3]["score"] == round(0.6 * DECAY * DECAY, 4)
    assert len(prop[2][3]["path"]) == 2


def test_propagate_stops_after_max_hops():
    direct = {0: {3: _entry(0.9)}}
    links = [(1, 0, "typed by"), (2, 1, "typed by"), (3, 2, "typed by")]
    prop = propagate(direct, links)
    assert 1 in prop and 2 in prop and 3 not in prop
    assert MAX_HOPS == 2


def test_propagate_score_never_exceeds_the_source_score():
    direct = {0: {3: _entry(0.9)}, 1: {4: _entry(0.2)}}
    prop = propagate(direct, [(2, 0, "satisfied by"), (2, 1, "satisfied by")])
    for d, v in prop[2].items():  # never above the direct score it is inherited from
        assert v["score"] <= direct[v["origin"]][d]["score"]
    assert prop[2][3]["score"] == round(0.9 * DECAY, 4)  # the better path wins
    assert prop[2][4]["score"] == round(0.2 * DECAY, 4)


def test_propagate_keeps_the_top_documents_per_element():
    direct = {s: {d: _entry(0.4 + 0.01 * d) for d in range(5)} for s in (0,)}
    prop = propagate(direct, [(1, 0, "owned by")])
    assert len(prop[1]) == KEEP_PROPAGATED == 3
    assert set(prop[1]) == {4, 3, 2}  # the three best documents


def test_model_links_follow_satisfy_typing_specialization_ownership_and_usages():
    els = [_el("s0", "P", "part def"),
           _el("s1", "P::R", "requirement def", "RequirementDefinition"),
           _el("s2", "P::R2", "requirement", "RequirementUsage"),
           _el("s3", "P::A", "part def")]
    rels = [{"rel": "r_satisfy", "src": "s3", "dst": "s1"},      # A satisfies R
            {"rel": "r_typed_by", "src": "s2", "dst": "s1"},     # usage R2 typed by R
            {"rel": "r_specializes", "src": "s3", "dst": "s0"}]  # A specializes P
    links = set(model_links(els, rels))
    assert (1, 3, "satisfied by") in links      # the requirement takes its satisfier's documents
    assert (2, 1, "typed by") in links          # the usage takes its definition's documents
    assert (1, 2, "usage of") in links          # the requirement def takes its usages'
    assert (3, 0, "specializes") in links       # A takes P's documents
    assert (1, 0, "owned by") in links          # P::R is owned by P (by qualified name)
    assert (3, 0, "owned by") in links          # P::A too


def test_model_links_ignore_relations_to_elements_not_in_the_model():
    els = [_el("s0", "P", "part def")]
    assert model_links(els, [{"rel": "r_typed_by", "src": "s0", "dst": "lib"}]) == []
