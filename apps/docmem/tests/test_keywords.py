"""Keyword index: tokenizer, BM25 ranking, deletion, segment merge and compaction. No network, no cognee.

Run from apps/docmem:  python -m pytest tests/test_keywords.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from docmem.keywords import KeywordIndex, tokenize  # noqa: E402


@pytest.mark.parametrize("text, expected", [
    ("REQ-1234 is open", ["req-1234", "is", "open"]),
    ("o'neill and O’Neill", ["o'neill", "and", "o’neill"]),
    ("version 3.5 and 1.0.2", ["version", "3.5", "and", "1.0.2"]),
    ("John's house, it's late", ["john", "house", "it", "late"]),
    ("well-known. end-", ["well-known", "end"]),
    ("snake_case x_y", ["snake", "case", "x", "y"]),
    ("e.g. and i.e", ["e.g", "and", "i.e"]),
    ("ＡＢＣ ｶﾀｶﾅ", ["abc", "カタカナ"]),
    ("café Zürich", ["café", "zürich"]),
    ("", []),
    ("--- ... !!!", []),
])
def test_tokenize(text, expected):
    assert tokenize(text) == expected


def build(tmp_path, docs, flush_each=False):
    kw = KeywordIndex(tmp_path, create=True)
    for pid, text in docs:
        kw.add(pid, text)
        if flush_each:
            kw.flush()
    kw.flush()
    return kw


CORPUS = [
    ("d1", "cat zebra"),
    ("d2", "cat cat"),
    ("d3", "cat"),
    ("d4", "cat dog"),
    ("d5", "dog bird"),
]


def test_rare_term_outranks_common_and_tf_counts(tmp_path):
    kw = build(tmp_path, CORPUS)
    hits = kw.search("cat zebra")
    assert [p for p, _ in hits][0] == "d1"  # the rare term carries it
    assert hits[0][1] > hits[1][1]
    assert [p for p, _ in kw.search("zebra")] == ["d1"]
    cat_only = [p for p, _ in kw.search("cat")]
    assert cat_only.index("d2") < cat_only.index("d3")  # higher tf, same length


def test_unknown_and_empty_queries(tmp_path):
    kw = build(tmp_path, CORPUS)
    assert kw.search("nonexistent") == []
    assert kw.search("!!!") == []


def test_add_skips_present_passage(tmp_path):
    kw = build(tmp_path, CORPUS)
    assert kw.add("d1", "something else entirely") is False
    kw.flush()
    assert kw.stats()["passages"] == 5
    assert kw.search("entirely") == []


def test_delete_then_search(tmp_path):
    kw = build(tmp_path, CORPUS)
    kw.delete(["d1"])
    kw.flush()
    assert [p for p, _ in kw.search("zebra")] == []
    assert "d1" not in [p for p, _ in kw.search("cat")]
    assert kw.stats()["passages"] == 4
    # a deleted passage can come back, with new postings, and is found again
    assert kw.add("d1", "zebra again") is True
    kw.flush()
    assert [p for p, _ in kw.search("zebra")] == ["d1"]


def test_delete_before_flush_never_writes(tmp_path):
    kw = KeywordIndex(tmp_path, create=True)
    kw.add("x", "unique token")
    kw.delete(["x"])
    kw.flush()
    assert kw.stats()["passages"] == 0
    assert kw.search("unique") == []


def test_two_segments_merge_on_compact(tmp_path):
    kw = build(tmp_path, CORPUS[:2], flush_each=True)  # two flushes, so two segments for "cat"
    kw.add("d3", "cat")
    kw.add("d4", "cat dog")
    kw.flush()
    assert len(kw._postings.query().where("term = 'cat'").limit(None).select(["segment"]).to_arrow()) == 3
    before = kw.search("cat zebra")
    kw.compact()
    assert len(kw._postings.query().where("term = 'cat'").limit(None).select(["segment"]).to_arrow()) == 1
    after = kw.search("cat zebra")
    assert [p for p, _ in before] == [p for p, _ in after]
    assert all(abs(a[1] - b[1]) < 1e-9 for a, b in zip(before, after))


def test_compact_drops_deleted(tmp_path):
    kw = build(tmp_path, CORPUS)
    kw.delete(["d2", "d5"])
    kw.flush()
    assert kw.stats()["deleted_rows"] == 2
    kw.compact()
    s = kw.stats()
    assert s["passages"] == 3 and s["deleted_rows"] == 0
    postings = kw.postings(live_only=False)
    assert all(n not in (1, 4) for nums, _ in postings.values() for n in nums.tolist())
    assert [p for p, _ in kw.search("bird")] == []
    assert {p for p, _ in kw.search("cat")} == {"d1", "d3", "d4"}


def test_rare_terms_of_passages(tmp_path):
    kw = build(tmp_path, CORPUS)
    top = kw.rare_terms(["d1"], n=2)
    assert [t for t, _ in top][0] == "zebra"


def test_search_reads_only_query_terms(tmp_path):
    kw = build(tmp_path, CORPUS)
    got = kw._postings_for(["zebra"])
    assert [t for t, _, _ in got] == ["zebra"]


def test_missing_index_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        KeywordIndex(tmp_path)
