"""Make a test corpus of big markdown files from a Wikipedia dump prefix: the longest articles, grouped into
"books" of several articles each (# Article, ## Section, ### Subsection, ...). Needs mwparserfromhell and
webcognee's pipeline/ directory (for its wikitext parser) on PYTHONPATH.

    python tools/make_wiki_corpus.py --dump head192.xml.bz2 --books 6 --per-book 8 --out corpus/
"""
import argparse
from pathlib import Path

import build_wiki as bw
import build_wiki_live as bl


def article_markdown(title: str, wikitext: str) -> str:
    out = [f"# {title}\n"]
    for number, path, heading, body in bl.numbered_sections(wikitext):
        text, _, _ = bw.clean(body)
        if number:
            if heading.lower() in ("references", "external links", "see also", "further reading", "notes",
                                   "bibliography", "sources", "citations", "footnotes"):
                continue
            out.append("#" * (path.count(" › ") + 2) + f" {heading}\n")
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        if paras:
            out.append("\n\n".join(paras) + "\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--scan", type=int, default=3000, help="articles to read from the dump")
    ap.add_argument("--books", type=int, default=6)
    ap.add_argument("--per-book", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    raw, _redirects = bl.parse_dump(a.dump, a.scan)
    longest = sorted(raw, key=lambda r: -len(r[3]))[: a.books * a.per_book]
    longest.sort(key=lambda r: r[0])
    a.out.mkdir(parents=True, exist_ok=True)
    for b in range(a.books):
        part = longest[b * a.per_book:(b + 1) * a.per_book]
        name = f"book{b + 1:02d}-{bl.plain(part[0][0])[:20].replace(' ', '_')}.md"
        body = "\n\n".join(article_markdown(bl.plain(t), w) for t, _, _, w in part)
        (a.out / name).write_text(body)
        print(name, len(body.encode()), "bytes:", ", ".join(bl.plain(t) for t, *_ in part))


if __name__ == "__main__":
    main()
