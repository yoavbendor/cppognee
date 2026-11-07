"""wikiplain/2: value-bearing templates rendered as text; wikiplain/1 unchanged; read_span and the scan use both.

No network: read_span is given a fake fetch. Run from apps/docmem:  python -m pytest tests/test_wikivalues.py
"""

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from docmem import scan_wiki, sources  # noqa: E402
from docmem.extract import RENDERERS, to_markdown_with_links as render_any  # noqa: E402
from docmem.extract import wikiplain, wikivalues  # noqa: E402


def md_of(wikitext: str) -> str:
    return wikivalues.to_markdown("T", wikitext)


def body_of(wikitext: str) -> str:
    """The markdown without its "# T" heading and blank line."""
    return md_of(wikitext).split("\n\n", 1)[1].rstrip("\n")


def links_match(wikitext: str):
    md, links = wikivalues.to_markdown_with_links("T", wikitext)
    data = md.encode("utf-8")
    for r in links:
        if r["kind"] == "wiki":
            assert data[r["offset"]: r["offset"] + r["length"]].decode("utf-8") == r["anchor"], r
    return md, links


# Each template kind, as the article writes it

@pytest.mark.parametrize(
    "wikitext, expected",
    [
        ("{{convert|65|%}}", "65%"),
        ("{{cvt|65|%}}", "65%"),
        ("{{convert|10500|lbf|N|abbr=on}}", "10500 lbf"),
        ("{{convert|10,500|lbf|N|sp=us}}", "10,500 lbf"),
        ("{{convert|5|ft|6|in|m}}", "5 ft 6 in"),
        ("{{val|1.5|e=9}}", "1.5 × 10^9"),
        ("{{val|1.5|e=9|u=m}}", "1.5 × 10^9 m"),
        ("{{val|10,500|u=lbf}}", "10,500 lbf"),
        ("{{nowrap|10 lbf}}", "10 lbf"),
        ("{{nobr|a b}}", "a b"),
        ("{{frac|1|2}}", "1/2"),
        ("{{sfrac|3|4}}", "3/4"),
        ("{{frac|3|1|2}}", "3 1/2"),
        ("{{circa|1969}}", "c. 1969"),
        ("{{lang|de|Mondlandung}}", "Mondlandung"),
        ("{{transl|ja|Tsuki}}", "Tsuki"),
        ("{{abbr|NASA|National Aeronautics and Space Administration}}", "NASA"),
        ("{{mvar|x}}", "x"),
        ("{{math|x^2 + y}}", "x^2 + y"),
        ("{{start date|1969|7|16}}", "1969-07-16"),
        ("{{Start date|1969|7}}", "1969-07"),
        ("{{start date|1969}}", "1969"),
        ("{{start date|df=yes|1969|7|16}}", "1969-07-16"),
        ("{{end date|1972|12|19}}", "1972-12-19"),
        ("{{dts|1969|7|16}}", "1969-07-16"),
        ("{{birth date|1930|6|7}}", "1930-06-07"),
        ("{{death date|1930|6|7|df=y}}", "1930-06-07"),
    ],
)
def test_template_renders_its_value(wikitext, expected):
    assert body_of(wikitext) == expected


def test_percent_and_degree_are_written_against_the_number():
    assert body_of("{{convert|20|°}}") == "20°"
    assert body_of("{{convert|20|km}}") == "20 km"


# Ranges and conversions: the source value and unit only

@pytest.mark.parametrize(
    "wikitext, expected",
    [
        ("{{convert|1000|to|10000|lbf}}", "1000 to 10000 lbf"),
        ("{{convert|1,000|to|10,000|lbf|N}}", "1,000 to 10,000 lbf"),
        ("{{convert|1000|-|10000|lbf}}", "1000-10000 lbf"),
        ("{{convert|1000|–|10000|lbf}}", "1000–10000 lbf"),
        ("{{convert|10|and|20|m}}", "10 and 20 m"),
        ("{{convert|10|-|20|%}}", "10-20%"),
        # the Descent propulsion system article writes its range this way
        ("{{convert|1050|and(-)|10125|lbf|kN}}", "1050 and(-) 10125 lbf"),
        ("{{cvt|1050|and(-)|6825|lbf|kN}}", "1050 and(-) 6825 lbf"),
    ],
)
def test_ranges(wikitext, expected):
    assert body_of(wikitext) == expected


def test_no_conversion_arithmetic():
    # the target unit and value are not computed, only the source's are written
    assert body_of("{{convert|1|km|mi}}") == "1 km"
    assert body_of("{{convert|10500|lbf|kN|1}}") == "10500 lbf"


def test_malformed_value_templates_are_stripped_as_before():
    # without its source unit a convert has nothing to render; a frac needs two parts
    for wikitext in ("a {{convert|5}} b", "a {{frac|1}} b", "a {{convert}} b", "a {{val}} b"):
        assert body_of(wikitext) == wikiplain.to_markdown("T", wikitext).split("\n\n", 1)[1].rstrip("\n")
        assert "5" not in body_of(wikitext) and "1" not in body_of(wikitext)


def test_endash_connector_in_a_range():
    assert body_of("{{convert|10|{{endash}}|15|m|abbr=on}} deep") == "10–15 m deep"
    assert body_of("{{convert|4|{{ndash}}|5|m}}") == "4–5 m"


def test_value_with_an_unrendered_template_is_stripped_as_before():
    # {{#expr:...}} is a parser function docmem does not evaluate: the convert is left to the stripping
    wikitext = "a {{cvt|{{#expr:1400000+3178000}}|lb|t|order=flip}} b"
    assert body_of(wikitext) == wikiplain.to_markdown("T", wikitext).split("\n\n", 1)[1].rstrip("\n")
    assert "#expr" not in body_of(wikitext) and "lb" not in body_of(wikitext)


def test_empty_source_unit_is_stripped_as_before():
    wikitext = "a {{convert|5||ft}} b"
    assert body_of(wikitext) == wikiplain.to_markdown("T", wikitext).split("\n\n", 1)[1].rstrip("\n")


def test_unknown_templates_are_still_stripped():
    wikitext = "a {{Infobox|x=5 km}} b {{cite web|url=u|title=t}} c"
    assert body_of(wikitext) == wikiplain.to_markdown("T", wikitext).split("\n\n", 1)[1].rstrip("\n")


def test_template_renders_inside_a_stripped_template_is_stripped_with_it():
    wikitext = "a {{cite web|title={{convert|3|m}}}} b"
    assert "3" not in body_of(wikitext)
    assert body_of(wikitext) == wikiplain.to_markdown("T", wikitext).split("\n\n", 1)[1].rstrip("\n")


def test_throttle_sentence_of_the_descent_propulsion_article():
    wikitext = ("The throttle could be set between {{convert|10|%}} and {{convert|65|%}}"
                " of the {{convert|10500|lbf|N|abbr=on}} thrust.")
    assert body_of(wikitext) == "The throttle could be set between 10% and 65% of the 10500 lbf thrust."
    assert wikiplain.to_markdown("T", wikitext) == "# T\n\nThe throttle could be set between  and  of the  thrust.\n"


# Nesting and links

def test_nested_value_templates_render_inside_out():
    assert body_of("{{nowrap|{{convert|10|lbf}}}}") == "10 lbf"
    assert body_of("{{nowrap|{{val|1.5|e=9|u=m}}}}") == "1.5 × 10^9 m"
    assert body_of("{{abbr|{{mvar|x}}|x}}") == "x"


def test_value_template_inside_an_unknown_template_argument_is_stripped_with_it():
    assert body_of("{{Infobox|x={{convert|1|km}}}} after") == "after"


def test_link_inside_a_rendered_template_keeps_its_record():
    md, links = links_match("Thrust was [[Rocketdyne F-1|F-1]] at {{nowrap|[[pound-force|lbf]] {{convert|5|x}}}}.")
    assert md.split("\n\n", 1)[1].strip() == "Thrust was F-1 at lbf 5 x."
    wiki = [(r["anchor"], r["target"]) for r in links if r["kind"] == "wiki"]
    assert wiki == [("F-1", "Rocketdyne F-1"), ("lbf", "Pound-force")]


def test_abbr_expansion_link_is_rendered_only_as_the_abbreviation():
    md, links = links_match("{{abbr|NASA|[[National Aeronautics and Space Administration]]}} works")
    assert md.split("\n\n", 1)[1].strip() == "NASA works"
    assert links == []


def test_link_as_the_rendered_value_gets_a_record():
    md, links = links_match("Lunar {{nowrap|[[Apollo 11|Apollo]]}} mission")
    assert md.split("\n\n", 1)[1].strip() == "Lunar Apollo mission"
    # offsets count the "# T\n\n" heading: "Apollo" starts at byte 5 + len("Lunar ")
    assert [(r["anchor"], r["target"], r["offset"]) for r in links] == [("Apollo", "Apollo 11", 11)]


def test_value_inside_a_heading_and_a_section_is_rendered():
    md = md_of("== Thrust {{nowrap|{{convert|1|lbf}}}} ==\nAt {{convert|65|%}}.")
    assert md == "# T\n\n## Thrust 1 lbf\n\nAt 65%.\n"


# wikiplain/1 is unchanged

def test_wikiplain_1_still_drops_value_templates():
    assert wikiplain.to_markdown("T", "throttle {{convert|65|%}} and {{start date|1969|7|16}}") == (
        "# T\n\nthrottle  and\n"
    )
    assert wikiplain.to_markdown("T", "{{nowrap|x}} {{frac|1|2}}") == "# T\n\n"


def test_wikiplain_1_and_2_differ_only_where_a_value_template_is_present():
    plain = "Lead [[Moon]] and text.\n\n== Sec ==\nSee [[Apollo 11|it]].\n"
    assert wikiplain.to_markdown("T", plain) == wikivalues.to_markdown("T", plain)


def test_registry_names_both_extractors():
    assert set(RENDERERS) == {"wikiplain/1", "wikiplain/2"}
    assert render_any("wikiplain/2", "T", "{{convert|1|lbf}}")[0] == "# T\n\n1 lbf\n"
    assert render_any("wikiplain/1", "T", "{{convert|1|lbf}}")[0] == "# T\n\n"
    with pytest.raises(ValueError):
        render_any("wikiplain/3", "T", "x")


# read_span dispatches on the extractor (fetch is faked: no network)

RAW = "Throttle {{convert|65|%}} of [[Thrust|the thrust]].\n"


@pytest.fixture
def fake_wiki(monkeypatch):
    sources._markdown.cache_clear()
    monkeypatch.setattr(sources, "fetch", lambda url, headers=None: RAW.encode("utf-8"))
    yield
    sources._markdown.cache_clear()


def test_read_span_reads_wikiplain_2_passages(fake_wiki):
    md = wikivalues.to_markdown("T", RAW)
    data = md.encode("utf-8")
    start = data.index(b"Throttle")
    text, status = sources.read_span(
        "https://en.wikipedia.org/w/index.php?oldid=1", start, len(data) - start,
        sha256=None, extractor="wikiplain/2", doc_sha256=None, title="T")
    assert status == "ok" and text.startswith("Throttle 65% of the thrust")


def test_read_span_reads_wikiplain_1_passages_with_their_own_rendering(fake_wiki):
    text, status = sources.read_span(
        "https://en.wikipedia.org/w/index.php?oldid=1", 0, 40,
        extractor="wikiplain/1", doc_sha256=None, title="T")
    assert status == "ok" and "65" not in text


def test_read_span_rejects_an_unknown_extractor(fake_wiki):
    with pytest.raises(ValueError):
        sources.read_span("https://en.wikipedia.org/w/index.php?oldid=1", 0, 1, extractor="wikiplain/3",
                          title="T")


def test_read_span_reports_changed_revision_by_hash(monkeypatch):
    sources._markdown.cache_clear()
    monkeypatch.setattr(sources, "fetch", lambda url, headers=None: RAW.encode("utf-8"))
    try:
        _, status = sources.read_span("https://en.wikipedia.org/w/index.php?oldid=1", 0, 5,
                                      extractor="wikiplain/2", doc_sha256="0" * 64, title="T")
        assert status == "changed"
    finally:
        sources._markdown.cache_clear()


# the scan writes the extractor on the doc and renders with it

def test_emit_records_the_extractor_and_renders_values():
    rec = {"revid": 7, "title": "Descent propulsion system", "mtime": 0, "page_id": 1,
           "text": "The throttle was " + "{{convert|65|%}} " + "word " * 400}
    for extractor, has_value in (("wikiplain/2", True), ("wikiplain/1", False)):
        out = io.StringIO()
        scan_wiki.emit(out, rec, 180, 320, extractor=extractor)
        records = [json.loads(line) for line in out.getvalue().splitlines()]
        doc = records[0]  # emit writes no scan_start; the doc record comes first
        assert doc["type"] == "doc" and doc["extractor"] == extractor
        text = "".join(r["text"] for r in records if r["type"] == "passage")
        assert ("65%" in text) is has_value


def test_extractor_default_is_wikiplain_1():
    assert scan_wiki.DEFAULT_EXTRACTOR == "wikiplain/1"
