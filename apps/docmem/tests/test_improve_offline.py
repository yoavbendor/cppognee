"""Pure-function tests for docmem.improve: no database, no network, no cognee import.

Run from apps/docmem:  python -m pytest tests/test_improve_offline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from docmem import improve as im  # noqa: E402


def ent(i, name, kind="organization", description="", doc="doc://a"):
    return {"id": f"e{i}", "name": name, "kind": kind, "description": description, "doc_uri": doc}


# union-find

def test_union_find_groups_transitive_merges_in_order():
    uf = im.UnionFind(6)
    uf.union(0, 2)
    uf.union(2, 4)
    uf.union(5, 1)
    assert uf.groups() == [[0, 2, 4], [1, 5], [3]]


def test_union_find_is_idempotent_and_order_free():
    uf = im.UnionFind(3)
    uf.union(2, 0)
    uf.union(0, 2)
    uf.union(1, 1)
    assert uf.groups() == [[0, 2], [1]]


def test_union_find_no_merges_is_singletons():
    assert im.UnionFind(3).groups() == [[0], [1], [2]]


# blocking

@pytest.mark.parametrize("a,b", [
    ("The Court", "court"),
    ("  THE  court ", "Court"),
    ("the  Ministry of Defence", "ministry of defence"),
])
def test_block_ignores_case_spacing_and_leading_the(a, b):
    assert im.block_of(a) == im.block_of(b)


def test_block_keeps_words_that_only_start_with_the():
    assert im.block_of("Theatre") == "theatre"
    assert im.block_of("Thes") != im.block_of("Thesis")


def test_units_split_a_name_block_by_kind():
    entities = [
        ent(0, "Apple", kind="organization"),
        ent(1, "the apple", kind="Organization"),
        ent(2, "apple", kind="food"),
        ent(3, "Berlin", kind="place"),
    ]
    units = im.units_of(entities)
    by = {(u["block"], u["kind"]): u["members"] for u in units}
    assert by[("apple", "organization")] == [0, 1]
    assert by[("apple", "food")] == [2]
    assert by[("berlin", "place")] == [3]
    assert len(units) == 3


def test_unit_name_and_description_come_from_members():
    entities = [
        ent(0, "Apple Inc", description="short"),
        ent(1, "Apple Inc", description="a much longer description"),
        ent(2, "apple inc", description="mid length"),
    ]
    (unit,) = im.units_of(entities)
    assert unit["name"] == "Apple Inc"
    assert unit["description"] == "a much longer description"


# canonical name

def test_canonical_most_frequent_wins():
    assert im.canonical(["IDF", "Israel Defense Forces", "IDF"]) == "IDF"


def test_canonical_tie_goes_to_shortest_then_alphabetical():
    assert im.canonical(["Long Name", "Short"]) == "Short"
    assert im.canonical(["bbb", "aaa"]) == "aaa"


def test_canonical_ignores_empty_values():
    assert im.canonical(["", "x", ""]) == "x"


# batching of pairs

def test_batches_split_evenly_and_keep_order():
    assert im.batches(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]


def test_batches_of_nothing_is_nothing():
    assert im.batches([], 15) == []


def test_pair_batch_size_is_fifteen_or_less():
    pairs = [(i,) for i in range(40)]
    sizes = [len(b) for b in im.batches(pairs, im.PAIR_BATCH)]
    assert im.PAIR_BATCH == 15
    assert sizes == [15, 15, 10]


def test_topic_batch_size_is_eighty_or_less():
    names = [f"t{i}" for i in range(161)]
    assert [len(b) for b in im.batches(names, im.TOPIC_BATCH)] == [80, 80, 1]


# reply parsing

def test_parse_same_accepts_numbers_strings_and_prefixes():
    assert im.parse_same({"same": [1, "3", "P4", "pair 2"]}, 5) == {0, 1, 2, 3}


def test_parse_same_ignores_out_of_range_and_garbage():
    assert im.parse_same({"same": [0, 9, "x", None]}, 3) == set()


def test_parse_same_accepts_a_bare_list():
    assert im.parse_same([2], 2) == {1}


def test_parse_same_rejects_a_reply_without_a_list():
    with pytest.raises(ValueError):
        im.parse_same({"answer": "yes"}, 2)


def test_parse_categories_assigns_and_tracks_known():
    reply = {"categories": {"Security": [1, 2], "Economy": [3]}}
    assign, known, rejected = im.parse_categories(reply, 3, [], 15)
    assert assign == {0: "Security", 1: "Security", 2: "Economy"}
    assert known == ["Security", "Economy"]
    assert rejected == []


def test_parse_categories_reuses_known_names_case_insensitively():
    reply = {"categories": {"security": [1]}}
    assign, known, _ = im.parse_categories(reply, 1, ["Security"], 15)
    assert assign == {0: "Security"}
    assert known == ["Security"]


def test_parse_categories_enforces_the_cap():
    reply = {"categories": {"A": [1], "B": [2], "C": [3]}}
    assign, known, rejected = im.parse_categories(reply, 3, ["X", "Y"], 3)
    assert assign == {0: "A"}
    assert known == ["X", "Y", "A"]
    assert rejected == ["B", "C"]


def test_parse_categories_skips_out_of_range_numbers_and_empty_categories():
    reply = {"categories": {"Keep": [1, 7], "Empty": [9]}}
    assign, known, _ = im.parse_categories(reply, 2, [], 15)
    assert assign == {0: "Keep"}
    assert known == ["Keep"]


def test_parse_categories_rejects_a_reply_without_categories():
    with pytest.raises(ValueError):
        im.parse_categories({"groups": {}}, 2, [], 15)


# near pairs and existing links

def test_near_pairs_finds_only_close_vectors():
    vectors = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    pairs = im.near_pairs(vectors, 0.05)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    assert pairs[0][2] < 0.05


def test_near_pairs_identical_names_are_distance_zero():
    assert [(i, j) for i, j, _ in im.near_pairs([[0.3, 0.4], [0.3, 0.4]], 0.01)] == [(0, 1)]


def test_near_pairs_needs_two_vectors():
    assert im.near_pairs([[1.0, 0.0]], 0.5) == []


def test_existing_links_join_units_sharing_a_concept():
    entities = [ent(0, "A"), ent(1, "a", kind="other"), ent(2, "B")]
    units = im.units_of(entities)
    linked = {"e0": {"c1"}, "e1": {"c1"}, "e2": {"c9"}}
    pairs = im.existing_links_pairs(units, entities, linked)
    assert len(pairs) == 1
    uf = im.UnionFind(len(units))
    for x, y in pairs:
        uf.union(x, y)
    assert [len(g) for g in uf.groups()] == [2, 1]


# concepts and orphans

def test_concepts_need_two_entities_and_take_the_frequent_name():
    entities = [
        ent(0, "Ministry of Health", description="short", doc="doc://a"),
        ent(1, "Health Ministry", description="the longest description here", doc="doc://a"),
        ent(2, "Ministry of Health", description="", doc="doc://b"),
        ent(3, "Lone", doc="doc://c"),
    ]
    (c,) = im.concepts_of(entities, [[0, 1, 2], [3]])
    assert c["name"] == "Ministry of Health"
    assert c["kind"] == "organization"
    assert c["aliases"] == ["Health Ministry"]
    assert c["n_docs"] == 2
    assert c["description"] == "the longest description here"


def test_concept_key_is_stable_for_the_same_members():
    entities = [ent(0, "X"), ent(1, "x")]
    a = im.concepts_of(entities, [[0, 1]])[0]["key"]
    b = im.concepts_of(list(reversed(entities)), [[1, 0]])[0]["key"]
    assert a == b


def test_orphans_entities_need_a_mention_or_a_relation():
    edges = [
        ("p1", "e1", "mentions", {}),          # e1 is mentioned
        ("e2", "e3", "works_for", {}),         # e2 and e3 have a relation
        ("e4", "e1", "same_as", {}),           # same_as alone does not count as a relation
        ("f1", "t1", "topics", {}),            # t1 is linked from a file
    ]
    ents, topics = im.orphans(["e1", "e2", "e3", "e4", "e5"], ["t1", "t2"], edges)
    assert ents == ["e4", "e5"]
    assert topics == ["t2"]


def test_orphans_relation_must_join_two_entities():
    edges = [("p1", "e1", "relates", {})]  # the source is not an entity
    ents, _ = im.orphans(["e1"], [], edges)
    assert ents == ["e1"]
