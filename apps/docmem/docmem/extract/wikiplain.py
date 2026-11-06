"""wikiplain/1: a Wikipedia article's wikitext as plain markdown, deterministically.

Output: "# Title", the lead, then each section as a heading of depth = the wikitext level ("==" is "##")
followed by its paragraphs. Templates, references, files and images, categories, tables, comments and
galleries are dropped (mwparserfromhell's strip_code does the rest). Sections titled as bibliography or
link lists are skipped with their subsections. Each non-blank line of a section is one paragraph.

to_markdown_with_links also returns the article's links (docmem.links): the display text of each wiki link
that stays in the markdown, with its byte range there, and one "category" link per category. The rendering is
the same either way: each wiki link's display is bracketed by marks while the text is built, and the marks are
located and removed afterwards. Every whitespace and newline rule looks through the marks, so removing them
gives exactly the text the rules give without them.
"""

import re

import mwparserfromhell

from ..links import wiki_target

EXTRACTOR = "wikiplain/1"
HEADING = re.compile(r"^(={2,6})[ \t]*(.+?)[ \t]*\1[ \t]*$", re.MULTILINE)
TABLE = re.compile(
    r"\{\|(?:(?!\{\|).)*?\|\}", re.DOTALL
)  # innermost wiki table, repeated until none is left
DROP_TAGS = {"ref", "references", "gallery", "math", "chem", "timeline", "score", "table"}
DROP_LINK_PREFIXES = ("file:", "image:", "media:", "category:")
SKIP_SECTIONS = {
    "references",
    "external links",
    "see also",
    "further reading",
    "notes",
    "bibliography",
    "sources",
    "citations",
    "footnotes",
}

# Marks, in plane 15 private use. A wiki link's display text starts with START and four DIGIT characters
# (the link's index, most significant byte first) and ends with END. The range is never used in article text;
# a wikitext that contains it is refused.
MARK_LO, MARK_HI = "\U000f0000", "\U000f01ff"
START, END, DIGIT0 = "\U000f0000", "\U000f0001", 0xF0100
MARK_RE = re.compile("[\U000f0000-\U000f01ff]")
MARK_CLASS = "[\U000f0000-\U000f01ff]"


def _is_mark(c: str) -> bool:
    return MARK_LO <= c <= MARK_HI


def _is_space(c: str) -> bool:
    return c.isspace()


def _clean(s: str) -> str:
    return MARK_RE.sub("", s)


def _mstrip(s: str, drop) -> str:
    """s without the characters that drop() accepts at either end, as str.strip does, looking through marks:
    the marks stay where they are."""
    content = [i for i, c in enumerate(s) if not (_is_mark(c) or drop(c))]
    if not content:
        return "".join(c for c in s if _is_mark(c))
    first, last = content[0], content[-1]
    head = "".join(c for c in s[:first] if _is_mark(c))
    tail = "".join(c for c in s[last + 1 :] if _is_mark(c))
    return head + s[first : last + 1] + tail


def _squeeze(s: str) -> str:
    """s with every run of newlines cut to two, looking through marks (what the three-newline replacement does)."""
    out, run = [], 0
    for c in s:
        if _is_mark(c):
            out.append(c)
            continue
        if c == "\n":
            run += 1
            if run > 2:
                continue
        else:
            run = 0
        out.append(c)
    return "".join(out)


def _start(k: int) -> str:
    return START + "".join(chr(DIGIT0 + ((k >> shift) & 0xFF)) for shift in (24, 16, 8, 0))


def _remove(code, node) -> None:
    try:
        code.remove(node)
    except ValueError:
        pass


def _mark(code, infos: list) -> None:
    """Bracket the display text of each article link in the tree (nested ones too) with its marks; infos gets the
    link's target. Namespace links (other than Category:) and categories get no marks."""
    for link in list(code.filter_wikilinks()):
        kind, target, fragment = wiki_target(str(link.title))
        if kind != "wiki":
            continue
        infos.append({"kind": kind, "target": target, "fragment": fragment})
        code.insert_before(link, _start(len(infos) - 1))
        code.insert_after(link, END)


def _plain(wikitext: str, infos: list) -> str:
    """Plain text of a fragment, with the marks of its article links."""
    wikitext = wikitext.replace("\r", "")
    while True:
        cut = TABLE.sub("", wikitext)
        if cut == wikitext:
            break
        wikitext = cut
    code = mwparserfromhell.parse(wikitext)
    for link in code.filter_wikilinks(recursive=False):
        if str(link.title).strip().lower().startswith(DROP_LINK_PREFIXES):
            _remove(code, link)
    for tag in code.filter_tags(recursive=False):
        if str(tag.tag).strip().lower() in DROP_TAGS:
            _remove(code, tag)
    _mark(code, infos)
    # strip_code, node by node: its join and collapse step is redone here so that marks are transparent to it
    kwargs = {"normalize": True, "collapse": True, "keep_template_params": False}
    parts = []
    for node in code.nodes:
        stripped = node.__strip__(**kwargs)
        if stripped:
            parts.append(str(stripped))
    return _squeeze(_mstrip("".join(parts), lambda c: c == "\n"))


def _sections(wikitext: str):
    """(level, heading wikitext, body wikitext) for the lead (level 0, heading None) and each section."""
    out, level, heading, last = [], 0, None, 0
    for m in HEADING.finditer(wikitext):
        out.append((level, heading, wikitext[last : m.start()]))
        level, heading, last = len(m.group(1)), m.group(2), m.end()
    out.append((level, heading, wikitext[last:]))
    return out


def _categories(wikitext: str) -> list[str]:
    """Each category of the article once, in order of first appearance."""
    names = []
    for link in mwparserfromhell.parse(wikitext.replace("\r", "")).filter_wikilinks():
        kind, name, _ = wiki_target(str(link.title))
        if kind == "category" and name not in names:
            names.append(name)
    return names


def _render(title: str, wikitext: str, infos: list) -> str:
    """The article as markdown, marks included."""
    blocks, skip_level = [], None
    for level, heading, body in _sections(wikitext):
        if skip_level is not None:
            if level > skip_level:
                continue
            skip_level = None
        if heading is not None:
            name = _mstrip(_plain(heading, infos), _is_space)
            if _clean(name).lower() in SKIP_SECTIONS:
                skip_level = level
                continue
            if _clean(name):
                blocks.append("#" * level + " " + name)
        for line in _plain(body, infos).split("\n"):
            if _clean(line).strip():
                blocks.append(_mstrip(line, _is_space))
    out = (
        f"# {title}\n\n"
        + "".join(b + "\n\n" for b in blocks[:-1])
        + (blocks[-1] + "\n" if blocks else "")
    )
    out = re.sub(r"[ \t]+(?=" + MARK_CLASS + r"*$)", "", out, flags=re.MULTILINE)
    return _squeeze(out)


def _extract(marked: str):
    """(markdown, [(index, start byte, end byte)]) with the marks removed; the spans are in bytes of the markdown."""
    out, spans, stack, byte, i = [], [], [], 0, 0
    while i < len(marked):
        ch = marked[i]
        if ch == START:
            k = 0
            for d in marked[i + 1 : i + 5]:
                k = (k << 8) | (ord(d) - DIGIT0)
            stack.append((k, byte))
            i += 5
            continue
        if ch == END:
            k, a = stack.pop()
            spans.append((k, a, byte))
        elif _is_mark(ch):
            raise ValueError(f"unexpected mark in rendered text at {i}")
        else:
            out.append(ch)
            byte += len(ch.encode("utf-8"))
        i += 1
    if stack:
        raise ValueError("unclosed link mark in rendered text")
    return "".join(out), spans


def to_markdown_with_links(title: str, wikitext: str):
    """(markdown, links): to_markdown(title, wikitext) and its links, as dicts without "type" or "uri"
    (docmem.links): "wiki" links with offset and length in UTF-8 bytes of the markdown, then "category" links."""
    if MARK_RE.search(wikitext):
        raise ValueError(f"{title!r}: the wikitext contains characters docmem uses for link marks")
    infos: list = []
    md, spans = _extract(_render(title, wikitext, infos))
    md_bytes = md.encode("utf-8")
    links = []
    for k, a, b in spans:
        if b == a:
            continue
        info = infos[k]
        links.append({"offset": a, "length": b - a, "anchor": md_bytes[a:b].decode("utf-8", "replace"),
                      "target": info["target"] or title, "kind": "wiki", "fragment": info["fragment"]})
    links.sort(key=lambda r: r["offset"])
    links += [{"offset": 0, "length": 0, "anchor": "", "target": name, "kind": "category", "fragment": ""}
              for name in _categories(wikitext)]
    return md, links


def to_markdown(title: str, wikitext: str) -> str:
    return to_markdown_with_links(title, wikitext)[0]
