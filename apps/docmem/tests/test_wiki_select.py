"""Topic-seeded Wikipedia selection and the category concept tree. No network: the dump is replaced by FakeSource,
which has the two methods DumpSource has (locate, read).

Run from apps/docmem:  python -m pytest tests/test_wiki_select.py
"""

import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from docmem.ontology import levels, parse_skos  # noqa: E402
from docmem.scan_wiki import (  # noqa: E402
    allocate,
    article_problem,
    candidate_articles,
    is_disambiguation,
    link_targets,
    read_seeds,
    redirect_of,
    stratified_sample,
    titles_selection,
)
from docmem.wiki_categories import break_cycles, build, is_maintenance, parent_names, write_turtle  # noqa: E402

LONG = " ".join(["word"] * 350)  # well over MIN_WORDS once extracted
SHORT = " ".join(["word"] * 10)


def page(pid, title, text, ns="0"):
    """A record as scan_wiki.page_of makes it."""
    return {"page_id": pid, "title": title, "ns": ns, "redirect": text.lstrip().upper().startswith("#REDIRECT"),
            "revid": pid, "timestamp": "2020-01-01T00:00:00Z", "text": text}


class FakeSource:
    def __init__(self, pages):
        self.by_pid = {r["page_id"]: r for r in pages}
        self.by_title = {r["title"]: r["page_id"] for r in pages}

    def locate(self, titles):
        return {t: (0, self.by_title[t]) for t in titles if t in self.by_title}

    def read(self, wanted, fn):
        for _stream, pids in wanted.items():
            for pid in pids:
                fn(pid, self.by_pid[pid])


# seeds

def test_read_seeds_labels_comments_and_blanks(tmp_path):
    f = tmp_path / "seeds.txt"
    f.write_text("# a comment\n\nOutline of robotics | robotics\nList of ships|maritime\nGlossary of nautical terms\n")
    assert read_seeds(f) == [("Outline of robotics", "robotics"), ("List of ships", "maritime"),
                             ("Glossary of nautical terms", "")]


def test_read_seeds_rejects_duplicates_and_empty(tmp_path):
    f = tmp_path / "dup.txt"
    f.write_text("Outline of robotics|a\nOutline of robotics|b\n")
    with pytest.raises(ValueError, match="listed twice"):
        read_seeds(f)
    g = tmp_path / "empty.txt"
    g.write_text("# nothing\n")
    with pytest.raises(ValueError, match="no seed pages"):
        read_seeds(g)


def test_link_targets_normalizes_and_drops_non_articles():
    text = "[[Foo_bar#Sec|x]] [[:Baz]] [[Category:X]] [[File:y.png]] [[Foo bar]] [[#Self]] [[Talk:Q]]"
    assert link_targets(text) == ["Foo bar", "Baz"]


# redirects and candidates

def test_redirect_of_reads_articles_categories_and_others():
    assert redirect_of("#REDIRECT [[Foo bar#Sec]]") == ("wiki", "Foo bar")
    assert redirect_of("#redirect [[Foo|bar]]") == ("wiki", "Foo")
    assert redirect_of("#REDIRECT [[:Category:Engineers]]") == ("category", "Engineers")
    assert redirect_of("#REDIRECT [[Wikipedia:Help]]") == ("", "")
    assert redirect_of("Just an article.") is None


def test_candidates_follow_a_redirect_once_and_drop_the_rest():
    pages = [
        page(1, "Alpha", "#REDIRECT [[Gamma Target]]"),
        page(2, "Gamma Target", LONG),
        page(3, "Delta", "#REDIRECT [[Alpha]]"),  # redirect to a redirect: not followed
        page(4, "Short one", SHORT),
        page(5, "List of things", LONG),
        page(6, "Name (disambiguation)", "Some name.\n" + LONG),
        page(7, "Beta", "{{Infobox person}}\n" + LONG),
    ]
    targets = {"Alpha": {"mech"}, "Delta": {"mech"}, "Short one": {"elec"}, "List of things": {"elec"},
               "Name (disambiguation)": {"elec"}, "Beta": {"mech", "elec"}, "Missing": {"mech"}}
    cands, report = candidate_articles(FakeSource(pages), targets)
    assert {c["title"]: c["fields"] for c in cands.values()} == {
        "Gamma Target": {"mech"},  # reached through Alpha, which takes the fields of its redirect
        "Beta": {"mech", "elec"},
    }
    assert report["not in dump"] == 1
    assert report["redirects followed"] == 2
    assert report["double redirect"] == 1
    assert report["short"] == 1
    assert report["list"] == 1
    assert report["disambiguation"] == 1


def test_article_problem_checks_each_rule():
    assert article_problem(page(1, "Fine", LONG)) is None
    assert article_problem(page(2, "Glossary of x", LONG)) == "list"
    assert article_problem(page(3, "Thing", LONG, ns="14")) == "not an article"
    assert article_problem(page(4, "Thing", SHORT)) == "short"


def test_disambiguation_by_title_and_template():
    assert is_disambiguation("Mercury (disambiguation)", "")
    assert is_disambiguation("Mercury", "{{disambig|geo}}")
    assert is_disambiguation("Mercury", "{{ Hndis | x }}")
    assert not is_disambiguation("Mercury", "{{Infobox element}}")


# stratified sampler

def test_allocate_equal_shares_rounded_and_capped():
    quota = allocate(1000, {f"f{i}": 100 for i in range(12)})
    assert sum(quota.values()) == 1000
    assert sorted(quota.values()) == [83] * 8 + [84] * 4


def test_allocate_passes_a_small_pool_share_to_the_others():
    quota = allocate(60, {"small": 3, "a": 100, "c": 200})
    assert quota == {"small": 3, "a": 29, "c": 28}


def test_stratified_sample_draws_each_id_once_for_the_first_field():
    pools = {"mech": [2, 7], "elec": [7]}  # 7 is linked from both
    chosen = stratified_sample(pools, 2, random.Random(1))
    assert chosen == {"mech": [2], "elec": [7]}


def test_stratified_sample_redistributes_shortfall_and_respects_the_total():
    pools = {"B": [1, 2, 3], "A": list(range(1, 101)), "C": list(range(1, 201))}
    chosen = stratified_sample(pools, 60, random.Random(7))
    ids = [p for pids in chosen.values() for p in pids]
    assert len(ids) == 60 and len(set(ids)) == 60
    assert sorted(chosen["B"]) == [1, 2, 3]
    assert len(chosen["A"]) == 29 and len(chosen["C"]) == 28
    short = stratified_sample({"a": [1], "b": [1]}, 5, random.Random(1))
    assert sum(len(v) for v in short.values()) == 1


def test_stratified_sample_is_deterministic_for_a_seed():
    pools = {"a": list(range(0, 500)), "b": list(range(250, 900))}
    first = stratified_sample(pools, 100, random.Random(3))
    assert first == stratified_sample(pools, 100, random.Random(3))
    assert first != stratified_sample(pools, 100, random.Random(4))


# maintenance categories

def test_maintenance_filter_uses_whole_words_and_substrings():
    for name in ["Articles with short description", "All articles lacking reliable references", "CS1 maint: x",
                 "Pages with broken links", "Harv and Sfn no-target errors", "Use dmy dates", "Hidden categories",
                 "Wikipedia categories named after people", "Commons category link is on Wikidata",
                 "Vague or ambiguous time", "Accuracy disputes from March 2020", "Engineering stubs",
                 "Articles needing additional references"]:
        assert is_maintenance(name), name
    for name in ["Harvard University alumni", "Pagers", "Mechanical engineering", "Machine learning"]:
        assert not is_maintenance(name), name


def test_parent_names_skips_colon_links_and_repeats():
    text = "[[Category:A]]\n[[Category:A|sort]]\n[[:Category:Not a parent]]\n[[Category:B_c]]"
    assert parent_names(text) == ["A", "B c"]


# cycles

def test_break_cycles_drops_back_edges_only():
    keep, dropped = break_cycles({"A": ["B"], "B": ["A"]})
    assert dropped == 1 and keep == {"A": ["B"]}
    keep, dropped = break_cycles({"A": ["A", "B"], "B": []})
    assert dropped == 1 and keep == {"A": ["B"]}
    keep, dropped = break_cycles({"A": ["B"], "B": ["C"], "C": []})
    assert dropped == 0 and keep == {"A": ["B"], "B": ["C"]}
    keep, dropped = break_cycles({"A": ["B"], "B": ["C"], "C": ["A"]})
    assert dropped == 1 and keep == {"A": ["B"], "B": ["C"]}


# the category tree

CATEGORY_PAGES = [
    page(101, "Category:Engineering", "[[Category:Applied sciences]]\n[[Category:Articles with short description]]\n"
                                      "[[:Category:Not a parent]]", ns="14"),
    page(102, "Category:Applied sciences", "[[Category:Sciences]]\n[[Category:Engineering]]", ns="14"),
    page(103, "Category:Sciences", "[[Category:Knowledge]]", ns="14"),
    page(104, "Category:Engineering disciplines", "#REDIRECT [[:Category:Engineering]]", ns="14"),
    page(105, "Category:Not a parent", "Plain text.", ns="14"),
    page(106, "Category:Knowledge", "[[Category:Everything]]", ns="14"),
]
DOCS = {
    "Doc A": {"Engineering", "Engineering disciplines", "Articles with short description", "Mechanical engineering"},
    "Doc B": {"Applied sciences"},
}


def test_build_category_tree_breaks_the_cycle_and_follows_the_redirect():
    built = build(DOCS, FakeSource(CATEGORY_PAGES), depth=2)
    assert built["nodes"] == ["Applied sciences", "Engineering", "Knowledge", "Mechanical engineering", "Sciences"]
    assert built["level"] == {"Applied sciences": 0, "Engineering": 0, "Mechanical engineering": 0,
                              "Sciences": 1, "Knowledge": 2}
    assert built["cycles"] == 1
    # the walk starts at "Applied sciences"; Engineering -> Applied sciences closes the cycle, so that edge goes
    assert built["parents"] == {"Applied sciences": ["Engineering", "Sciences"], "Sciences": ["Knowledge"]}
    assert built["doc_categories"] == {"Doc A": ["Engineering", "Mechanical engineering"],
                                       "Doc B": ["Applied sciences"]}
    assert built["missing_pages"] == 1 and built["redirects"] == 1


def test_build_depth_zero_keeps_only_document_categories():
    built = build(DOCS, FakeSource(CATEGORY_PAGES), depth=0)
    assert built["nodes"] == ["Applied sciences", "Engineering", "Mechanical engineering"]
    # edges among the document categories stay; the cycle between the two is broken as at depth 2
    assert built["parents"] == {"Applied sciences": ["Engineering"]} and built["cycles"] == 1


def test_turtle_loads_with_the_ontology_reader(tmp_path):
    built = build(DOCS, FakeSource(CATEGORY_PAGES), depth=2)
    out = tmp_path / "scheme.ttl"
    write_turtle(out, built["nodes"], built["parents"])
    concepts = {c.label: c for c in parse_skos(out)}
    assert set(concepts) == set(built["nodes"])
    assert all(c.definition == "" for c in concepts.values())  # the loader embeds the label alone
    assert concepts["Sciences"].broader
    lv = levels(list(concepts.values()))  # the ontology's level counts from the scheme's roots, not the documents
    assert lv[concepts["Knowledge"].uri] == 0 and lv[concepts["Sciences"].uri] == 1


# titles file

def test_titles_selection_reads_listed_titles_in_file_order(capsys):
    pages = [
        page(1, "Alpha", LONG),
        page(2, "Beta", LONG),
        page(3, "Gamma", "#REDIRECT [[Alpha]]"),  # not an article: reported, not written
        page(4, "Delta", LONG),
    ]
    titles = [("Delta", "electrical"), ("Gamma", ""), ("Missing", "x"), ("Alpha", ""), ("Beta", "mech")]
    out = io.StringIO()
    uris = titles_selection(FakeSource(pages), titles, out, 180, 320)
    docs = [json.loads(line) for line in out.getvalue().splitlines()]
    docs = [d for d in docs if d["type"] == "doc"]
    assert [d["title"] for d in docs] == ["Delta", "Alpha", "Beta"]
    assert [d.get("field") for d in docs] == ["electrical", None, "mech"]
    assert uris == [d["uri"] for d in docs]
    err = capsys.readouterr().err
    assert "not in the dump index: 'Missing'" in err
    assert "not an article" in err and "'Gamma'" in err
