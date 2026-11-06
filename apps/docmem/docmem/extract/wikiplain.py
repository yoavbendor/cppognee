"""wikiplain/1: a Wikipedia article's wikitext as plain markdown, deterministically.

Output: "# Title", the lead, then each section as a heading of depth = the wikitext level ("==" is "##")
followed by its paragraphs. Templates, references, files and images, categories, tables, comments and
galleries are dropped (mwparserfromhell's strip_code does the rest). Sections titled as bibliography or
link lists are skipped with their subsections. Each non-blank line of a section is one paragraph.
"""

import re

import mwparserfromhell

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


def _plain(wikitext: str) -> str:
    """Plain text of a fragment: no templates, tags, media, categories, tables or comments."""
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
    return code.strip_code(normalize=True, collapse=True)


def _remove(code, node) -> None:
    try:
        code.remove(node)
    except ValueError:
        pass


def _sections(wikitext: str):
    """(level, heading wikitext, body wikitext) for the lead (level 0, heading None) and each section."""
    out, level, heading, last = [], 0, None, 0
    for m in HEADING.finditer(wikitext):
        out.append((level, heading, wikitext[last : m.start()]))
        level, heading, last = len(m.group(1)), m.group(2), m.end()
    out.append((level, heading, wikitext[last:]))
    return out


def to_markdown(title: str, wikitext: str) -> str:
    blocks, skip_level = [], None
    for level, heading, body in _sections(wikitext):
        if skip_level is not None:
            if level > skip_level:
                continue
            skip_level = None
        if heading is not None:
            name = _plain(heading).strip()
            if name.lower() in SKIP_SECTIONS:
                skip_level = level
                continue
            if name:
                blocks.append("#" * level + " " + name)
        blocks.extend(line.strip() for line in _plain(body).split("\n") if line.strip())
    out = (
        f"# {title}\n\n"
        + "".join(b + "\n\n" for b in blocks[:-1])
        + (blocks[-1] + "\n" if blocks else "")
    )
    out = re.sub(r"[ \t]+$", "", out, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", out)
