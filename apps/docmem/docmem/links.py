"""Links in a document, as the "link" records of the scan stream.

A link is {"offset", "length", "anchor", "target", "kind", "fragment"}: the byte range of its anchor text in the
bytes the document's passages refer to, the anchor text itself, the normalized target, the kind and the part after
"#". Kinds:
  "wiki"      an article title (wiki link targets are normalized the MediaWiki way, see wiki_target)
  "url"       a URI with a scheme other than file: (http, https, mailto, ...), without its fragment
  "file"      a path, resolved against the document's uri into a file:// URI; "#frag" alone is the document itself
  "id"        a match of one --id-pattern; target and anchor are the match
  "category"  a wiki category: offset 0, length 0, empty anchor

markdown_links reads Markdown and HTML: [text](target), [text](target "title"), <a href="target">text</a>,
[[Target|text]] (kind "wiki") and bare http(s) URLs (the anchor is the URL). Image syntax is not a link, and code
(fenced blocks and inline code spans) is ignored. Offsets count UTF-8 bytes; a byte that is not valid UTF-8 is kept
as it is and shows as U+FFFD in the anchor, like docmem.store.read_span does.
"""

import itertools
import re
from urllib.parse import quote, unquote, urljoin

# MediaWiki namespaces of English Wikipedia (lowercase, spaces as in the prefix). A link into one of them is not
# an article, so it gets no record. Category: is handled before this list.
NAMESPACES = frozenset(
    {
        "media", "special", "talk", "user", "user talk", "wikipedia", "wikipedia talk", "wp", "wt", "project",
        "file", "file talk", "image", "image talk", "mediawiki", "mediawiki talk", "template", "template talk",
        "help", "help talk", "category talk", "portal", "portal talk", "draft", "draft talk", "module", "timedtext",
        "book", "education program", "gadget", "gadget talk", "gadget definition", "topic",
    }
)
KINDS = ("wiki", "url", "file", "id", "category")
INTERLANGUAGE = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]+)*|simple")
FENCE = re.compile(r"^[ \t]{0,3}(```|~~~)")
CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.S)
IMAGE = re.compile(r"!\[[^\[\]]*\]\([^()]*(?:\([^()]*\)[^()]*)*\)")
WIKI = re.compile(r"\[\[([^\[\]|\n]+)(?:\|([^\[\]\n]*))?\]\]")
MD_LINK = re.compile(
    r"(?<!!)\[([^\[\]]*)\]\(\s*(<[^<>\n]*>|[^\s()<>]*(?:\([^\s()<>]*\)[^\s()<>]*)*)"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'))?\s*\)"
)
HTML_A = re.compile(
    r"<a\s[^>]*?href\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))[^>]*>(.*?)</a\s*>", re.I | re.S
)
TAG = re.compile(r"<[a-zA-Z][a-zA-Z0-9-]*(?:\s[^<>]*)?/?>")
BARE_URL = re.compile(r"https?://[^\s<>\"]+")
SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


def _text(s: str) -> str:
    """s as docmem reads bytes: invalid UTF-8 becomes U+FFFD, the same as bytes.decode('utf-8', 'replace')."""
    return s.encode("utf-8", "surrogateescape").decode("utf-8", "replace")


def _title(name: str) -> str:
    name = " ".join(name.replace("_", " ").split())
    return name[:1].upper() + name[1:]


def _fragment(frag: str) -> str:
    return " ".join(frag.replace("_", " ").split())


def wiki_target(raw: str) -> tuple[str, str, str]:
    """(kind, target, fragment) of a wiki link's target text (what sits before the pipe).

    Leading colons and whitespace go; the part after "#" is the fragment; underscores become spaces, runs of spaces
    collapse and the first letter is capitalized. Kind "category" for Category: (target = the category name);
    kind "" for a link that is not an article (File:, Image:, Media:, other namespaces, interlanguage links such as
    "fr:"). The target is "" for a link to a section of the current page ("#Section")."""
    text = re.sub(r"^[:\s]+", "", raw)
    name, _, frag = text.partition("#")
    kind = "wiki"
    prefix, colon, rest = name.partition(":")
    if colon:
        p = prefix.strip()
        if p.lower() == "category":
            kind, name = "category", rest
        elif p.lower() in NAMESPACES or INTERLANGUAGE.fullmatch(p):
            return "", "", ""
    target = _title(name)
    if kind == "category" and not target:
        return "", "", ""
    return kind, target, _fragment(frag)


def _blank(s: str) -> str:
    """s with every character but newlines replaced by a space (same length)."""
    return re.sub(r"[^\n]", " ", s)


def _mask_code(text: str) -> str:
    """text with fenced code blocks and inline code spans blanked, so no link is searched inside them."""
    out, fence = [], None
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if fence is not None:
            if bare.lstrip().startswith(fence):
                fence = None
            out.append(_blank(line))
            continue
        m = FENCE.match(bare)
        if m:
            fence = m.group(1)
            out.append(_blank(line))
            continue
        out.append(line)
    return CODE_SPAN.sub(lambda m: _blank(m.group(0)), "".join(out))


def _bare_url(s: str) -> str:
    """A bare URL without the punctuation a sentence puts after it (and an unbalanced closing parenthesis)."""
    while s:
        if s[-1] in ".,;:!?'\"*":
            s = s[:-1]
        elif s[-1] == ")" and s.count(")") > s.count("("):
            s = s[:-1]
        else:
            break
    return s


def _target(raw: str, base_uri: str):
    """(kind, target, fragment) of a Markdown or HTML link target, or None when there is no target."""
    t = raw.strip()
    if t.startswith("<") and t.endswith(">"):
        t = t[1:-1].strip()
    if not t:
        return None
    path, _, frag = t.partition("#")
    if not path:
        return "file", base_uri, frag
    if SCHEME.match(path):
        return ("file" if path.lower().startswith("file:") else "url"), path, frag
    return "file", urljoin(base_uri, quote(unquote(path))), frag


def markdown_links(data: bytes, base_uri: str, id_patterns=()) -> list[dict]:
    """Links of a Markdown (or plain text) document, sorted by offset. base_uri is the document's uri;
    id_patterns are regular expressions whose every match is an "id" link."""
    text = data.decode("utf-8", "surrogateescape")
    masked = _mask_code(text)
    offs = range(len(text) + 1) if text.isascii() else list(
        itertools.accumulate((len(c.encode("utf-8", "surrogateescape")) for c in text), initial=0)
    )
    taken = bytearray(len(text))
    records = []
    categories = set()

    def cover(cs, ce):
        taken[cs:ce] = b"\x01" * (ce - cs)

    def take(cs, ce, a, b, kind, target, fragment, check=True):
        """Record the link whose anchor is text[a:b], unless its span (cs, ce) overlaps a link already taken."""
        if check and taken.find(1, cs, ce) != -1:
            return
        cover(cs, ce)
        anchor = _text(text[a:b])
        if b <= a or not anchor.strip() or not target:
            return
        records.append({"offset": offs[a], "length": offs[b] - offs[a], "anchor": anchor,
                        "target": _text(target), "kind": kind, "fragment": _text(fragment)})

    for m in IMAGE.finditer(masked):
        cover(m.start(), m.end())
    for m in WIKI.finditer(masked):
        kind, name, frag = wiki_target(m.group(1))
        if kind == "category":
            if name not in categories:
                categories.add(name)
                records.append({"offset": 0, "length": 0, "anchor": "", "target": name, "kind": "category",
                                "fragment": ""})
            cover(m.start(), m.end())
            continue
        if not kind:
            cover(m.start(), m.end())
            continue
        a, b = m.span(2) if m.group(2) is not None else m.span(1)
        if not name:
            target = _target("#" + frag, base_uri)
        else:
            target = ("wiki", name, frag)
        if target:
            take(m.start(), m.end(), a, b, *target)
    for m in MD_LINK.finditer(masked):
        target = _target(m.group(2), base_uri)
        if target:
            take(m.start(), m.end(), *m.span(1), *target)
    for m in HTML_A.finditer(masked):
        href = next(g for g in m.groups()[:3] if g is not None)
        target = _target(href, base_uri)
        if target:
            take(m.start(), m.end(), *m.span(4), *target)
    url_view = TAG.sub(lambda m: _blank(m.group(0)), masked)
    for m in BARE_URL.finditer(url_view):
        url = _bare_url(m.group(0))
        target, _, frag = url.partition("#")
        if url:
            take(m.start(), m.start() + len(url), m.start(), m.start() + len(url), "url", target, frag)
    for pattern in id_patterns:
        for m in re.compile(pattern).finditer(masked):
            take(m.start(), m.end(), m.start(), m.end(), "id", m.group(0), "", check=False)
    return sorted(records, key=lambda r: r["offset"])
