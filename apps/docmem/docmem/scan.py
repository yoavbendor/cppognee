"""Scanner: cut markdown files into documents, sections and passages, and write them to stdout as JSON lines.

The stream (protocol 1), per document:
  {"type": "doc", "uri", "sha256", "size", "mtime", "title"}
  {"type": "section", "uri", "path": [...], "level", "offset", "length", "line"}      (one per heading)
  {"type": "passage", "uri", "path": [...], "offset", "length", "line", "end_line", "sha256", "text"}
  {"type": "end", "uri", "sections", "passages"}
and once at the end: {"type": "scan_end", "roots": [...], "uris": [...]}.

Offsets and lengths are in bytes of the file as it was scanned; "sha256" of a passage hashes exactly
those bytes, so a reader can tell when the file has changed under the index. "text" is the passage's
text for the indexer to embed; the indexer does not store it.

Passages ("meaning" units): a section's blocks (paragraphs, lists, tables, code fences) are merged
while they fit --target-words, and a block longer than --max-words is split at sentence ends. A
passage never crosses a heading.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from . import PROTOCOL

HEADING = re.compile(rb"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
FENCE = re.compile(rb"^[ \t]{0,3}(```|~~~)")
SENTENCE_END = re.compile(rb"[.!?][\"')\]]*[ \t\n]+(?=[A-Z0-9\"'(\[])")
SUFFIXES = (".md", ".markdown", ".mdx", ".txt")


def blocks(data: bytes):
    """Yield ("heading", level, title, start, end, line) and ("block", None, None, start, end, line):
    byte ranges of headings and of blank-line separated blocks (a fenced code block is one block)."""
    pos, line_no = 0, 1
    cur_start = cur_line = None
    in_fence = None
    for raw in data.splitlines(keepends=True):
        start, end = pos, pos + len(raw)
        line = raw.rstrip(b"\r\n")
        pos = end
        this_line = line_no
        line_no += 1
        if in_fence is not None:
            if line.lstrip().startswith(in_fence):
                in_fence = None
            continue
        m = FENCE.match(line)
        if m:
            in_fence = m.group(1)
            if cur_start is None:
                cur_start, cur_line = start, this_line
            continue
        h = HEADING.match(line)
        if h or not line.strip():
            if cur_start is not None:
                yield ("block", None, None, cur_start, start, cur_line)
                cur_start = None
            if h:
                yield ("heading", len(h.group(1)), h.group(2).decode("utf-8", "replace").strip(),
                       start, end, this_line)
            continue
        if cur_start is None:
            cur_start, cur_line = start, this_line
    if cur_start is not None:
        yield ("block", None, None, cur_start, pos, cur_line)


def trim(data: bytes, start: int, end: int):
    while end > start and data[end - 1] in b" \t\r\n":
        end -= 1
    return start, end


def split_long(data: bytes, start: int, end: int, max_words: int, target_words: int):
    """Split one long block at sentence ends into pieces of about target_words (never above max_words
    unless a single sentence is)."""
    cuts = [start] + [start + m.end() for m in SENTENCE_END.finditer(data[start:end])] + [end]
    pieces, piece_start, words = [], start, 0
    for a, b in zip(cuts, cuts[1:]):
        w = len(data[a:b].split())
        if words and words + w > target_words:
            pieces.append((piece_start, a))
            piece_start, words = a, 0
        words += w
    pieces.append((piece_start, end))
    return [trim(data, a, b) for a, b in pieces if data[a:b].strip()]


def passages_of(data: bytes, items, target_words: int, max_words: int):
    """Merge a section's blocks into passages; items are (start, end, line) byte ranges."""
    out, cur, words = [], None, 0
    for start, end, _ in items:
        start, end = trim(data, start, end)
        w = len(data[start:end].split())
        if w > max_words:
            if cur:
                out.append(cur)
                cur, words = None, 0
            out.extend(split_long(data, start, end, max_words, target_words))
            continue
        if cur and words + w > target_words:
            out.append(cur)
            cur, words = None, 0
        cur = (cur[0], end) if cur else (start, end)
        words += w
    if cur:
        out.append(cur)
    return out


def line_at(data: bytes, offset: int, base_offset: int, base_line: int) -> int:
    return base_line + data.count(b"\n", base_offset, offset)


def scan_file(path: Path, target_words: int, max_words: int):
    data = path.read_bytes()
    doc = {"uri": path.resolve().as_uri(), "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
           "mtime": path.stat().st_mtime, "title": None}
    yield from scan_bytes(data, doc, path.stem, target_words, max_words)


def scan_bytes(data: bytes, doc: dict, fallback_title: str, target_words: int, max_words: int):
    """Cut markdown ``data`` into the doc's records. ``doc`` holds the doc record's fields (uri, sha256,
    size, mtime, and for a derived text "format", "extractor", ...); title None means the first "# "
    heading, else fallback_title. Offsets are into ``data``."""
    uri = doc["uri"]
    events = list(blocks(data))
    title = doc.get("title") or next((t for kind, lvl, t, *_ in events if kind == "heading" and lvl == 1),
                                     fallback_title)
    yield {"type": "doc", **doc, "title": title}

    # sections: (path, level, start, line) in order; a section spans to the next heading of the same or
    # a higher level. The text before the first heading is the section with path [title].
    heads = [(lvl, t, s, ln) for kind, lvl, t, s, e, ln in events if kind == "heading"]
    stack, sections = [], []
    for lvl, t, s, ln in heads:
        stack = [x for x in stack if x[0] < lvl] + [(lvl, t)]
        sections.append(([x[1] for x in stack], lvl, s, ln))
    n_sections = 0
    for i, (spath, lvl, s, ln) in enumerate(sections):
        end = next((sections[j][2] for j in range(i + 1, len(sections)) if sections[j][1] <= lvl), len(data))
        n_sections += 1
        yield {"type": "section", "uri": uri, "path": spath, "level": lvl, "offset": s, "length": end - s,
               "line": ln}

    # passages, section by section (a heading closes the current group of blocks)
    n_passages = 0
    group, cur_path = [], [title] if not heads or heads[0][0] != 1 else []
    head_i = -1

    def flush():
        nonlocal n_passages
        for a, b in passages_of(data, group, target_words, max_words):
            span = data[a:b]
            line = line_at(data, a, 0, 1)
            n_passages += 1
            yield {"type": "passage", "uri": uri, "path": cur_path or [title], "offset": a, "length": b - a,
                   "line": line, "end_line": line + span.count(b"\n"),
                   "sha256": hashlib.sha256(span).hexdigest(), "text": span.decode("utf-8", "replace")}

    for kind, lvl, t, s, e, ln in events:
        if kind == "heading":
            yield from flush()
            group = []
            head_i += 1
            cur_path = sections[head_i][0]
        else:
            group.append((s, e, ln))
    yield from flush()
    yield {"type": "end", "uri": uri, "sections": n_sections, "passages": n_passages}


def iter_files(roots):
    for root in roots:
        p = Path(root)
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUFFIXES
                              and not any(part.startswith(".") for part in f.relative_to(p).parts))
        elif p.is_file():
            yield p
        else:
            raise SystemExit(f"docmem scan: no such file or directory: {root}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docmem scan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="markdown files or directories (scanned recursively)")
    ap.add_argument("--target-words", type=int, default=180, help="merge blocks up to this many words")
    ap.add_argument("--max-words", type=int, default=320, help="split blocks longer than this")
    a = ap.parse_args(argv)
    out = sys.stdout
    uris = []
    out.write(json.dumps({"type": "scan_start", "protocol": PROTOCOL}) + "\n")
    for f in iter_files(a.roots):
        for rec in scan_file(f, a.target_words, a.max_words):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        uris.append(f.resolve().as_uri())
    out.write(json.dumps({"type": "scan_end", "roots": [Path(r).resolve().as_uri() for r in a.roots],
                          "uris": uris}) + "\n")
    out.flush()


if __name__ == "__main__":
    main()
