"""Ontology: IPTC and SKOS parsing, levels, centroid and top-k selection. No DB, no network, no LLM.

Run from apps/docmem:  python -m pytest tests/test_ontology.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from docmem.ontology import (  # noqa: E402
    Concept,
    centroid,
    levels,
    load_concepts,
    parse_iptc,
    parse_skos,
    top_k,
)

BASE = "http://cv.example/mt/"


def iptc_concept(n, label, broader=None, retired=False, definition=None):
    c = {"uri": BASE + n, "qcode": "mt:" + n, "prefLabel": {"en-GB": label},
         "definition": {"en-GB": definition or f"about {label}"}}
    if broader:
        c["broader"] = [BASE + broader]
    if retired:
        c["retired"] = "2021-09-30T12:00:00+00:00"
    return c


SAMPLE = {
    "@context": "https://www.iptc.org/std/IKOS/IKOS.jsonld",
    "conceptSet": [
        iptc_concept("01", "arts"),
        iptc_concept("02", "music", broader="01"),
        iptc_concept("03", "old music form", broader="02", retired=True),
        iptc_concept("04", "opera", broader="02"),
        iptc_concept("05", "sport"),
        iptc_concept("06", "football", broader="05"),
    ],
}


def test_iptc_levels_broader_and_retired_skipped():
    concepts = parse_iptc(SAMPLE)
    by = {c.label: c for c in concepts}
    assert set(by) == {"arts", "music", "opera", "sport", "football"}  # the retired one is gone
    assert by["music"].broader == (BASE + "01",)
    assert by["arts"].broader == ()
    lv = levels(concepts)
    assert lv[BASE + "01"] == 0 and lv[BASE + "05"] == 0
    assert lv[BASE + "02"] == 1 and lv[BASE + "06"] == 1
    assert lv[BASE + "04"] == 2  # opera under music under arts


def test_iptc_definition_and_label_english_only():
    doc = {"conceptSet": [{"uri": BASE + "x", "prefLabel": {"fr": "art", "en-GB": "arts"},
                           "definition": {"fr": "l'art", "en-GB": "the arts"}}]}
    (c,) = parse_iptc(doc)
    assert (c.label, c.definition) == ("arts", "the arts")


def test_iptc_missing_definition_is_empty():
    doc = {"conceptSet": [{"uri": BASE + "x", "prefLabel": {"en-GB": "arts"}}]}
    (c,) = parse_iptc(doc)
    assert c.definition == ""


def test_iptc_parent_retired_is_an_error():
    doc = {"conceptSet": [iptc_concept("01", "arts", retired=True), iptc_concept("02", "music", broader="01")]}
    with pytest.raises(ValueError, match="retired"):
        parse_iptc(doc)


def test_iptc_dangling_parent_is_an_error():
    doc = {"conceptSet": [iptc_concept("02", "music", broader="99")]}
    with pytest.raises(ValueError, match="not in the scheme"):
        parse_iptc(doc)


def test_cycle_is_an_error():
    cs = [Concept(BASE + "a", "a", "", (BASE + "b",)), Concept(BASE + "b", "b", "", (BASE + "a",))]
    with pytest.raises(ValueError, match="cycle"):
        levels(cs)


def test_load_concepts_detects_iptc_json(tmp_path):
    import json

    p = tmp_path / "mediatopic.json"
    p.write_text(json.dumps(SAMPLE), encoding="utf-8")
    assert {c.label for c in load_concepts(p)} == {"arts", "music", "opera", "sport", "football"}


TTL = """
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <http://example.org/t/> .

ex:top a skos:Concept ;
    skos:prefLabel "Top"@en ;
    skos:definition "The top concept"@en .

ex:mid a skos:Concept ;
    skos:prefLabel "Middle"@en ;
    skos:prefLabel "Mitte"@de ;
    skos:broader ex:top .

ex:leaf a skos:Concept ;
    skos:prefLabel "Leaf"@en ;
    skos:broader ex:mid .

ex:old a skos:Concept ;
    skos:prefLabel "Old"@en ;
    owl:deprecated true .
"""


def test_skos_turtle_parsing(tmp_path):
    p = tmp_path / "scheme.ttl"
    p.write_text(TTL, encoding="utf-8")
    concepts = parse_skos(p)
    by = {c.label: c for c in concepts}
    assert set(by) == {"Top", "Middle", "Leaf"}  # the deprecated one is skipped
    assert by["Middle"].label == "Middle"  # the English label, not the German one
    assert by["Top"].definition == "The top concept"
    assert by["Middle"].definition == ""
    assert by["Leaf"].broader == ("http://example.org/t/mid",)
    lv = levels(concepts)
    assert [lv[c.uri] for c in (by["Top"], by["Middle"], by["Leaf"])] == [0, 1, 2]


def test_skos_unknown_suffix_is_an_error(tmp_path):
    p = tmp_path / "scheme.txt"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="suffix"):
        parse_skos(p)


def test_centroid_is_the_normalized_mean():
    v = centroid(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    assert np.allclose(v, [2 ** -0.5, 2 ** -0.5])
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_centroid_of_one_vector_is_that_direction():
    v = centroid(np.array([[3.0, 4.0]], dtype=np.float32))
    assert np.allclose(v, [0.6, 0.8])


def test_top_k_picks_best_first_and_applies_min_sim():
    row = np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32)
    assert top_k(row, 2, 0.0) == [1, 3]
    assert top_k(row, 3, 0.6) == [1, 3]  # 0.5 is below the minimum
    assert top_k(row, 4, 0.95) == []
    assert top_k(row, 10, -1.0) == [1, 3, 2, 0]


def test_label_matches_whole_words_only():
    from docmem.ontology import label_matches

    assert label_matches("What is Bohmian mechanics?", {"ohm": "Ohm"}) == []  # inside a word
    assert label_matches("Ohm's law and a Bohmian view", {"ohm": "Ohm", "bo": "Bohmian"}) == ["bo", "ohm"]
    assert label_matches("the ohm-metre", {"ohm": "Ohm"}) == ["ohm"]  # a hyphen ends a word


def test_label_matches_is_case_insensitive_and_once_per_concept():
    from docmem.ontology import label_matches

    assert label_matches("VISCOSITY, viscosity and more viscosity", {"v": "Viscosity"}) == ["v"]


def test_label_matches_longest_first_and_no_overlap():
    from docmem.ontology import label_matches

    labels = {"effect": "effect", "hall": "Hall effect", "ohm": "Ohm"}
    # "Hall effect" is taken first, so "effect" inside it does not match; the result is longest first
    assert label_matches("Ohm and the Hall effect", labels) == ["hall", "ohm"]
    assert label_matches("only the effect here", labels) == ["effect"]
    assert label_matches("motor oil", {"oil": "oil", "mo": "motor oil"}) == ["mo"]


def test_label_matches_skips_empty_labels():
    from docmem.ontology import label_matches

    assert label_matches("anything at all", {"x": "  "}) == []


def test_best_per_document_keeps_best_passage_then_top_documents():
    from docmem.ask import best_per_document

    ranked = [  # best first
        {"id": "a1", "uri": "A"}, {"id": "a2", "uri": "A"}, {"id": "b1", "uri": "B"},
        {"id": "c1", "uri": "C"}, {"id": "b2", "uri": "B"}, {"id": "d1", "uri": "D"},
    ]
    assert [h["id"] for h in best_per_document(ranked, 2)] == ["a1", "b1"]
    assert [h["id"] for h in best_per_document(ranked, 3)] == ["a1", "b1", "c1"]
    assert [h["id"] for h in best_per_document(ranked, 10)] == ["a1", "b1", "c1", "d1"]


def test_best_per_document_one_document_gives_one_passage():
    from docmem.ask import best_per_document

    ranked = [{"id": "x1", "uri": "X"}, {"id": "x2", "uri": "X"}]
    assert [h["id"] for h in best_per_document(ranked, 3)] == ["x1"]
    assert best_per_document([], 3) == []
