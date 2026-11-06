"""Links: Markdown and HTML links, wiki link targets, and the marks wikiplain uses to find link text. No network.

Run from apps/docmem:  python -m pytest tests/test_links.py

The wikitext goldens were produced by the wikiplain/1 renderer before links were added (its to_markdown), so
they pin that to_markdown output unchanged.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from docmem.extract.wikiplain import to_markdown, to_markdown_with_links  # noqa: E402
from docmem.links import markdown_links, wiki_target  # noqa: E402

BASE = "file:///docs/a/b.md"
ID = r"\b[A-Z]{2,8}-\d{1,6}\b"


def links_of(text, ids=()):
    """The links of a Markdown text; each anchor must be the bytes its offset and length point at."""
    data = text.encode("utf-8") if isinstance(text, str) else text
    out = markdown_links(data, BASE, list(ids))
    for r in out:
        assert data[r["offset"]: r["offset"] + r["length"]].decode("utf-8", "replace") == r["anchor"]
    return out


def one(text, **fields):
    out = links_of(text)
    assert len(out) == 1, out
    for key, value in fields.items():
        assert out[0][key] == value, (key, out[0])
    return out[0]


# Markdown and HTML

def test_relative_file_link():
    one("See [Home](index.md).", kind="file", target="file:///docs/a/index.md", anchor="Home", fragment="")


def test_title_fragment_and_parent_directory():
    one('[x](../up.md#sec "Title")', kind="file", target="file:///docs/up.md", fragment="sec", anchor="x")


def test_absolute_path_and_percent_encoding():
    one("[r](/srv/r.md)", kind="file", target="file:///srv/r.md")
    one("[s](my%20file.md)", kind="file", target="file:///docs/a/my%20file.md")


def test_fragment_alone_is_the_document():
    one("[s](#Intro)", kind="file", target=BASE, fragment="Intro")


def test_images_are_not_links():
    assert links_of("![pic](p.png) ![x](https://e.org/i.png)") == []


def test_url_with_fragment_and_mailto():
    one("[w](https://e.org/a?b=1#c)", kind="url", target="https://e.org/a?b=1", fragment="c")
    one("[m](mailto:a@b.c)", kind="url", target="mailto:a@b.c", anchor="m")


def test_bare_url_drops_trailing_punctuation():
    one("Visit https://example.org/page.", kind="url", anchor="https://example.org/page",
        target="https://example.org/page")


def test_bare_url_keeps_balanced_parenthesis():
    one("(see https://e.org/a_(b))", kind="url", anchor="https://e.org/a_(b)")


def test_link_text_url_is_not_counted_twice():
    out = links_of("[https://a.example/](https://a.example/)")
    assert [(r["kind"], r["target"]) for r in out] == [("url", "https://a.example/")]


def test_html_anchor_and_img_is_not_a_link():
    one('<a href="other.md">other doc</a>', kind="file", target="file:///docs/a/other.md", anchor="other doc")
    assert links_of('<img src="https://x.example/y.png">') == []


def test_wiki_style_links_in_markdown():
    out = links_of("[[Target Page|shown]] [[target_name#Sec]]")
    assert [(r["kind"], r["anchor"], r["target"], r["fragment"]) for r in out] == [
        ("wiki", "shown", "Target Page", ""),
        ("wiki", "target_name#Sec", "Target name", "Sec"),
    ]
    assert out[1]["length"] == len("target_name#Sec")


def test_wiki_category_has_no_span():
    one("[[Category:Greek]]", kind="category", target="Greek", offset=0, length=0, anchor="")


def test_id_patterns_and_code():
    out = links_of("ABC-123 and `XYZ-9` and PROJ-42", ids=[ID])
    assert [(r["kind"], r["anchor"]) for r in out] == [("id", "ABC-123"), ("id", "PROJ-42")]


def test_code_is_ignored():
    text = "```\n[f](no.md) https://no.example ABC-7\n```\n`[i](no2.md)` [yes](yes.md)"
    out = links_of(text, ids=[ID])
    assert [r["anchor"] for r in out] == ["yes"]


def test_empty_anchor_or_target_is_skipped():
    assert links_of("[](x.md) [ ](y.md) [x]()") == []


def test_offsets_count_utf8_bytes():
    text = "\u05e9\u05dc\u05d5\u05dd [link](a.md) and ABC-1"
    out = links_of(text, ids=[ID])
    assert [r["anchor"] for r in out] == ["link", "ABC-1"]
    assert out[0]["offset"] == len("\u05e9\u05dc\u05d5\u05dd [".encode("utf-8"))


def test_invalid_utf8_is_kept_as_replacement():
    data = b"x [a\xffb](c.md)"
    out = markdown_links(data, BASE)
    assert len(out) == 1 and out[0]["anchor"] == "a\ufffdb"
    assert out[0]["offset"] == 3 and out[0]["length"] == 3


# Wiki link targets

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Bobby_Wanbon", ("wiki", "Bobby Wanbon", "")),
        (":john  millington_(rugby league)", ("wiki", "John millington (rugby league)", "")),
        ("Strikeforce: Carano vs. Cyborg", ("wiki", "Strikeforce: Carano vs. Cyborg", "")),
        ("iPod: a", ("wiki", "IPod: a", "")),
        ("Foo#Bar_baz", ("wiki", "Foo", "Bar baz")),
        ("#Section", ("wiki", "", "Section")),
        ("Category:Living_people", ("category", "Living people", "")),
        ("File:x.jpg", ("", "", "")),
        ("Image:x.png", ("", "", "")),
        ("Media:x.ogg", ("", "", "")),
        ("fr:Paris", ("", "", "")),
        ("Talk:Foo", ("", "", "")),
        ("Wikipedia:Notability", ("", "", "")),
        ("category:", ("", "", "")),
    ],
)
def test_wiki_target(raw, expected):
    assert wiki_target(raw) == expected


# wikiplain: the marks

def test_marks_never_reach_the_output():
    md, links = to_markdown_with_links("T", "[[A]] and [[B|b]]\n== [[C]] ==\nx")
    assert not re.search("[\U000f0000-\U000f01ff]", md)
    assert all(r["anchor"] and not re.search("[\U000f0000-\U000f01ff]", r["anchor"]) for r in links
               if r["kind"] == "wiki")


def test_mark_characters_in_wikitext_are_refused():
    with pytest.raises(ValueError):
        to_markdown("T", "a \U000f0000 b")


def wiki_links_match_md(md, links):
    data = md.encode("utf-8")
    for r in links:
        if r["kind"] == "wiki":
            assert data[r["offset"]: r["offset"] + r["length"]].decode("utf-8") == r["anchor"], r


SNIPPETS = [
    ('sections_files_refs', "'''[[Alpha]]''' is a [[beta|Beta letter]].\n\n== History ==\nSee [[Gamma#x|]] and [[File:a.jpg|thumb|cap [[Delta]]]]. [[Category:Greek]]\n\n== References ==\n<ref>[[Hidden]]</ref>\n* [[Skipped]]\n",
     '# T\n\nAlpha is a Beta letter.\n\n## History\n\nSee  and .\n',
     [('wiki', 'Alpha', 'Alpha', ''), ('wiki', 'Beta letter', 'Beta', ''), ('category', '', 'Greek', '')]),
    ('blank_lines', 'Lead with [[ a ]] and [[: foo_bar ]].\n\n\n\n== Sec ==\n\n\n[[Eta]]\n',
     '# T\n\nLead with  a  and : foo_bar .\n\n## Sec\n\nEta\n',
     [('wiki', ' a ', 'A', ''), ('wiki', ': foo_bar ', 'Foo bar', ''), ('wiki', 'Eta', 'Eta', '')]),
    ('templates_tags', '{{Infobox|x=[[Theta]]}}\nA [[Iota]] {{cite|[[Kappa]]}} line.\n\n<span>[[Lambda]]</span>\n',
     '# T\n\nA Iota  line.\n\nLambda\n',
     [('wiki', 'Iota', 'Iota', ''), ('wiki', 'Lambda', 'Lambda', '')]),
    ('heading_links', '== [[Mu]] ==\ntext\n== [[References]] ==\n[[X]] here\n',
     '# T\n\n## Mu\n\ntext\n',
     [('wiki', 'Mu', 'Mu', '')]),
    ('empty_display', "Nu [[Nu|'']] and [[Xi|]] x\n[[Pi|\n]]\n\n[[Rho|\n\n\nb]] end\n\n\n",
     "# T\n\nNu '' and  x\n\nb end\n",
     [('wiki', "''", 'Nu', ''), ('wiki', 'b', 'Rho', '')]),
    ('newlines_around', 'a\n[[Omicron]]\n\n\nb\n\n\n\n[[Sigma|\nsx]]\n',
     '# T\n\na\n\nOmicron\n\nb\n\nsx\n',
     [('wiki', 'Omicron', 'Omicron', ''), ('wiki', 'sx', 'Sigma', '')]),
    ('line_start_link', 'A paragraph.\n[[Link]]\n\n\n\nAfter\n',
     '# T\n\nA paragraph.\n\nLink\n\nAfter\n',
     [('wiki', 'Link', 'Link', '')]),
    ('nested_visible_tag', '<span>\n[[Inner]]\n</span>\n\nafter [[Outer]]\n',
     '# T\n\nInner\n\nafter Outer\n',
     [('wiki', 'Inner', 'Inner', ''), ('wiki', 'Outer', 'Outer', '')]),
    ('bold_italic', "'''[[Bold]]'''\n\n\n''[[Italic|\n]]''\n\nend\n",
     '# T\n\nBold\n\nend\n',
     [('wiki', 'Bold', 'Bold', '')]),
    ('notes_section', 'Intro [[One]].\n== Notes ==\n[[Hidden note]]\n== After ==\n[[Shown]] text\n',
     '# T\n\nIntro One.\n\n## After\n\nShown text\n',
     [('wiki', 'One', 'One', ''), ('wiki', 'Shown', 'Shown', '')]),
]


@pytest.mark.parametrize("name, wikitext, golden, expected_links", SNIPPETS, ids=[s[0] for s in SNIPPETS])
def test_wikitext_golden(name, wikitext, golden, expected_links):
    assert to_markdown("T", wikitext) == golden
    md, links = to_markdown_with_links("T", wikitext)
    assert md == golden
    assert [(r["kind"], r["anchor"], r["target"], r["fragment"]) for r in links] == expected_links
    wiki_links_match_md(md, links)
