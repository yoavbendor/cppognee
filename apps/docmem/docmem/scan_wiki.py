"""Scanner for English Wikipedia articles, read from the official compressed dump.

    docmem scan-wiki --sample 1000 --seed 1 --cache DIR > wiki.jsonl
    docmem scan-wiki --seed-pages examples/engineering_seeds.txt --sample 1000 --seed 1 --cache DIR > wiki.jsonl

The dump index (offset:page_id:title per page) is downloaded once into --cache and parsed. Its distinct
offsets start the dump's independent bz2 streams (about 100 pages each). Each needed stream is fetched with one
Range request (up to 4 at a time) and decompressed in memory. Article text is never written to disk.

Without --seed-pages, entries are drawn uniformly from the index with random.Random(seed); a draw is kept only if
its page is in the namespace 0, not a redirect and has text, otherwise another draw replaces it (no page is kept
twice).

With --seed-pages FILE, the sample is drawn from the articles the seed pages link to. FILE has one seed title per
line, optionally followed by "|" and a field label ("Outline of mechanical engineering|mechanical"); blank lines
and lines starting with "#" are skipped, and a seed without a label is in the field "". Each seed must be a page
of the dump index under exactly that title. Its article links (docmem.links.wiki_target, namespace 0) are the
candidates: a redirect is followed once to its target; disambiguation pages, list / outline / glossary / index
pages and pages under MIN_WORDS words of extracted text are left out. The sample is stratified by field: equal
shares of --sample per field (rounded; a field whose pool is too small passes its share to the others), drawn
with random.Random(seed). An article linked from several fields is drawn once, for the first field drawn (the
smallest pool first). Each doc record of such a sample has "field": label.

With --titles FILE, exactly the listed articles are read, in file order, with the same line format as a seed file
("Title" or "Title|field"; "field" is set on the doc only when given). Titles the dump index does not list, and
listed pages that are not articles (a redirect, a non-article namespace), are reported on stderr and skipped.

The stream is the scan.py protocol (scan_start, doc, section, passage, end per article, scan_end). A doc's
uri is the revision it was read from, https://en.wikipedia.org/w/index.php?oldid=N, and its sha256 is that of the
wikitext (see docmem.sources.read_span for how passages are read back). Progress and the seeded pool sizes go to
stderr.
"""

import argparse
import bz2
import calendar
import hashlib
import json
import multiprocessing
import os
import random
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import mwparserfromhell

from . import PROTOCOL
from .extract.wikiplain import EXTRACTOR, to_markdown_with_links
from .links import wiki_target
from .scan import scan_bytes
from .sources import USER_AGENT, fetch

BASE = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream"
INDEX_URL = BASE + "-index.txt.bz2"
DUMP_URL = BASE + ".xml.bz2"
WIKI = "https://en.wikipedia.org/"
MAX_CONCURRENT = 4
JUDGE_BATCH = 64  # candidate pages whose word counts are checked together, in worker processes
MIN_WORDS = 300  # a seeded article has at least this many words of extracted text
REDIRECT = re.compile(r"\s*#REDIRECT\s*:?\s*\[\[([^\[\]\n]+)\]\]", re.IGNORECASE)
TEMPLATE = re.compile(r"\{\{\s*([^|{}\n]+?)\s*[|}]")
DISAMBIGUATION_TEMPLATES = frozenset({
    "dab", "hndis", "geodis", "numberdis", "sia", "shipindex", "surname", "given name",
    "letter-number combination disambiguation", "place name disambiguation",
})
LIST_TITLE = re.compile(r"(lists?|outline|glossary|index) of\b", re.IGNORECASE)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def ensure_index(cache: Path) -> Path:
    """The index file in cache, downloaded once."""
    path = cache / INDEX_URL.rsplit("/", 1)[1]
    if path.exists():
        return path
    cache.mkdir(parents=True, exist_ok=True)
    log(f"downloading {INDEX_URL} (once; kept in {cache})")
    part = path.with_name(path.name + ".part")
    req = urllib.request.Request(INDEX_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as r, part.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    part.replace(path)
    return path


class Index:
    """Every index line: the stream it lives in (an index into `starts`) and its page id. Two arrays of 4-byte
    ints, about 200 MB for the whole dump; the titles are not kept."""

    def __init__(self, path: Path):
        self.starts: list[int] = []
        self.stream_of = array("I")
        self.page_id = array("I")
        last = -1
        with bz2.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                off, pid, _title = line.split(":", 2)
                off = int(off)
                if off != last:
                    if off < last:
                        raise ValueError(f"{path}: offsets are not increasing at {off}")
                    self.starts.append(off)
                    last = off
                self.stream_of.append(len(self.starts) - 1)
                self.page_id.append(int(pid))
        if array("I").itemsize != 4:
            raise RuntimeError("array('I') is not 4 bytes on this platform")

    def __len__(self):
        return len(self.page_id)


def find_titles(index: Index, path: Path, want: set[str]) -> dict[str, tuple[int, int]]:
    """{title: (stream, page id)} for the titles in want that the index lists, in one pass over the index file.
    Streams are numbered as Index numbers them; each title's stream start is checked against index.starts."""
    found: dict[str, tuple[int, int]] = {}
    stream, last = -1, -1
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            off, pid, title = line.split(":", 2)
            off = int(off)
            if off != last:
                stream += 1
                last = off
            title = title.rstrip("\n")
            if title in want:
                if index.starts[stream] != off:
                    raise RuntimeError(f"index stream {stream} of {title!r} does not start at {off}")
                found[title] = (stream, int(pid))
    return found


def dump_size() -> int:
    req = urllib.request.Request(DUMP_URL, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers["Content-Length"])


def page_of(page) -> dict | None:
    """A dump page as a record (any namespace), or None when it has no text. "redirect" is whether the dump marks
    it as a redirect; redirect_of reads where it points."""
    text = page.findtext("revision/text")
    if not text or not text.strip():
        return None
    return {
        "page_id": int(page.findtext("id")),
        "title": page.findtext("title"),
        "ns": page.findtext("ns"),
        "redirect": page.find("redirect") is not None,
        "revid": int(page.findtext("revision/id")),
        "timestamp": page.findtext("revision/timestamp"),
        "text": text,
    }


def article_of(rec: dict | None) -> dict | None:
    """(ns 0, not a redirect, text) as a record, or None when the page is not a usable article."""
    if rec is None or rec["ns"] != "0" or rec["redirect"]:
        return None
    ts = time.strptime(rec["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
    return {
        "page_id": rec["page_id"],
        "title": rec["title"],
        "revid": rec["revid"],
        "mtime": calendar.timegm(ts),
        "text": rec["text"],
    }


def redirect_of(wikitext: str) -> tuple[str, str] | None:
    """(kind, title) of the page a "#REDIRECT [[...]]" text points to, read as docmem.links.wiki_target reads a
    link: kind "wiki" (an article), "category", or "" (another namespace). None when the text is not a redirect."""
    m = REDIRECT.match(wikitext)
    if m is None:
        return None
    kind, title, _ = wiki_target(m.group(1).split("|")[0])
    return kind, title


def fetch_stream(index: Index, stream: int, size: int, wanted: set[int]):
    """({page_id: page_of record or None} for the wanted pages of one stream, bytes downloaded)."""
    start = index.starts[stream]
    end = index.starts[stream + 1] - 1 if stream + 1 < len(index.starts) else size - 1
    body = fetch(DUMP_URL, {"Range": f"bytes={start}-{end}"})
    if body is None or len(body) != end - start + 1:
        raise RuntimeError(f"range {start}-{end} of the dump did not return the stream")
    root = ET.fromstring(b"<root>" + bz2.decompress(body) + b"</root>")
    found = {}
    for page in root.findall("page"):
        pid = int(page.findtext("id"))
        if pid in wanted:
            found[pid] = page_of(page)
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"pages {sorted(missing)} are not in the stream at offset {start}")
    return found, len(body)


def map_pages(index: Index, size: int, wanted: dict[int, set[int]], fn, stats: dict) -> None:
    """Call fn(page id, record) for every wanted page (wanted: {stream: page ids}). Streams are fetched up to
    MAX_CONCURRENT at a time, in batches, so the records of a batch are the only ones held."""
    streams = sorted(wanted)
    batch = MAX_CONCURRENT * 4
    with ThreadPoolExecutor(MAX_CONCURRENT) as pool:
        for at in range(0, len(streams), batch):
            part = streams[at : at + batch]
            fetched = pool.map(lambda s: fetch_stream(index, s, size, wanted[s]), part)
            for found, nbytes in fetched:
                stats["streams"] += 1
                stats["bytes"] += nbytes
                for pid, rec in found.items():
                    fn(pid, rec)


def sample(index: Index, n: int, seed: int, size: int):
    """Yield (article records) in draw order, until n valid articles; counts the streams and bytes read."""
    rng = random.Random(seed)
    seen: set[int] = set()
    stats = {"streams": 0, "bytes": 0, "draws": 0}
    with ThreadPoolExecutor(MAX_CONCURRENT) as pool:
        while len(seen) < n:
            draws = [rng.randrange(len(index)) for _ in range(n - len(seen))]
            stats["draws"] += len(draws)
            wanted: dict[int, set[int]] = {}
            for k in draws:
                wanted.setdefault(index.stream_of[k], set()).add(index.page_id[k])
            streams = sorted(wanted)
            jobs = [(index, s, size, wanted[s]) for s in streams]
            fetched = pool.map(lambda job: fetch_stream(*job), jobs)
            results = {}
            for s, (found, nbytes) in zip(streams, fetched):
                results.update(found)
                stats["streams"] += 1
                stats["bytes"] += nbytes
            for k in draws:
                rec = article_of(results[index.page_id[k]])
                if rec is None or rec["page_id"] in seen:
                    continue
                seen.add(rec["page_id"])
                yield rec, stats
                if len(seen) == n:
                    return


def emit(out, rec: dict, target_words: int, max_words: int, field: str | None = None) -> str:
    text = rec["text"]
    data = text.encode("utf-8")
    revid, title = rec["revid"], rec["title"]
    doc = {
        "uri": f"https://en.wikipedia.org/w/index.php?oldid={revid}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "mtime": rec["mtime"],
        "title": title,
        "format": "wikitext",
        "extractor": EXTRACTOR,
        "page_id": rec["page_id"],
        "revid": revid,
        "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
    }
    if field is not None:
        doc["field"] = field
    md, links = to_markdown_with_links(title, text)
    for r in scan_bytes(md.encode("utf-8"), doc, title, target_words, max_words, links=links):
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
    return doc["uri"]


# ---- topic-seeded selection ------------------------------------------------------------------------

def read_seeds(path: Path) -> list[tuple[str, str]]:
    """(title, field) per seed line: "Title" or "Title|field". Blank lines and "#" comments are skipped; a title
    listed twice is an error."""
    seeds, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        title, _, field = line.partition("|")
        title, field = title.strip(), field.strip()
        if not title:
            raise ValueError(f"{path}:{n}: no title")
        if title in seen:
            raise ValueError(f"{path}:{n}: {title!r} is listed twice")
        seen.add(title)
        seeds.append((title, field))
    if not seeds:
        raise ValueError(f"{path}: no seed pages")
    return seeds


def link_targets(wikitext: str) -> list[str]:
    """The article titles a page links to (namespace 0, fragments dropped), each once, in order."""
    titles = []
    for link in mwparserfromhell.parse(wikitext.replace("\r", "")).filter_wikilinks():
        kind, title, _ = wiki_target(str(link.title))
        if kind == "wiki" and title:
            titles.append(title)
    return list(dict.fromkeys(titles))


def is_disambiguation(title: str, wikitext: str) -> bool:
    """A disambiguation page: its title ends in "(disambiguation)", or it carries a disambiguation template
    (the "disambig..." family and the templates in DISAMBIGUATION_TEMPLATES)."""
    if title.lower().endswith("(disambiguation)"):
        return True
    for m in TEMPLATE.finditer(wikitext):
        name = m.group(1).strip().lower()
        if name.startswith("disambig") or name in DISAMBIGUATION_TEMPLATES:
            return True
    return False


def article_problem(rec: dict) -> str | None:
    """Why a page record (page_of) is not a candidate article, or None when it is one: namespace 0, not a
    redirect, neither a disambiguation nor a list / outline / glossary / index page, and at least MIN_WORDS
    words of extracted text."""
    if rec["ns"] != "0":
        return "not an article"
    if rec["redirect"]:
        return "redirect"
    if LIST_TITLE.match(rec["title"]):
        return "list"
    if is_disambiguation(rec["title"], rec["text"]):
        return "disambiguation"
    md, _ = to_markdown_with_links(rec["title"], rec["text"])
    if len(md.split()) < MIN_WORDS:
        return "short"
    return None


def allocate(n: int, caps: dict[str, int]) -> dict[str, int]:
    """Equal shares of n across the fields, each capped at its cap. The fields are taken smallest cap first; a
    share is rounded up while the remainder is not divided evenly, and what a field cannot take passes on to the
    fields after it. The quotas sum to min(n, sum of caps)."""
    order = sorted(caps, key=lambda f: (caps[f], f))
    quota, left = {}, n
    for i, f in enumerate(order):
        share, extra = divmod(left, len(order) - i)
        quota[f] = min(caps[f], share + (1 if extra else 0))
        left -= quota[f]
    return quota


def stratified_sample(pools: dict[str, list[int]], n: int, rng: random.Random) -> dict[str, list[int]]:
    """Up to n ids: equal shares per field (see allocate), each id at most once. An id in several pools is drawn
    for the first field drawn, the smallest pool first. A field whose free ids run short passes its share on in
    the next round."""
    pools = {f: sorted(ids) for f, ids in pools.items()}
    chosen: dict[str, list[int]] = {f: [] for f in pools}
    taken: set[int] = set()
    left = n
    while left > 0:
        free = {f: [p for p in pools[f] if p not in taken] for f in pools}
        caps = {f: len(ids) for f, ids in free.items() if ids}
        if not caps:
            break
        quota = allocate(left, caps)
        for f in sorted(quota, key=lambda f: (caps[f], f)):
            avail = [p for p in pools[f] if p not in taken]
            k = min(quota[f], len(avail))
            if k == 0:
                continue
            pick = rng.sample(avail, k)
            taken.update(pick)
            chosen[f].extend(pick)
            left -= k
    return chosen


class DumpSource:
    """The dump as the seeded selection reads it: titles located through the index, pages read by stream."""

    def __init__(self, index: Index, path: Path, size: int, stats: dict):
        self.index, self.path, self.size, self.stats = index, path, size, stats

    def locate(self, titles: set[str]) -> dict[str, tuple[int, int]]:
        """{title: (stream, page id)} for the titles the index has."""
        return find_titles(self.index, self.path, titles)

    def read(self, wanted: dict[int, set[int]], fn) -> None:
        """fn(page id, record) for each wanted page ({stream: page ids})."""
        map_pages(self.index, self.size, wanted, fn, self.stats)


def wanted_of(located: dict[str, tuple[int, int]]) -> dict[int, set[int]]:
    wanted: dict[int, set[int]] = {}
    for stream, pid in located.values():
        wanted.setdefault(stream, set()).add(pid)
    return wanted


def seed_targets(source: DumpSource, seeds: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Article titles the seed pages link to, with the fields of the seeds that link them."""
    located = source.locate({t for t, _ in seeds})
    missing = [t for t, _ in seeds if t not in located]
    if missing:
        raise SystemExit(f"docmem scan-wiki: not in the dump index (exact titles): {missing}")
    pages: dict[int, dict] = {}
    source.read(wanted_of(located), lambda pid, rec: pages.__setitem__(pid, rec))
    targets: dict[str, set[str]] = {}
    for title, field in seeds:
        rec = pages[located[title][1]]
        if rec["ns"] != "0":
            raise SystemExit(f"docmem scan-wiki: seed {title!r} is not an article (namespace {rec['ns']})")
        if rec["redirect"]:
            raise SystemExit(f"docmem scan-wiki: seed {title!r} is a redirect; name its target instead")
        links = link_targets(rec["text"])
        log(f"seed {title!r} [{field}]: {len(links)} article links")
        for t in links:
            targets.setdefault(t, set()).add(field)
    log(f"all {len(seeds)} seeds are in the dump index; {len(targets)} distinct article links")
    return targets


def candidate_articles(source: DumpSource, targets: dict[str, set[str]]):
    """The candidate articles the seeds link to, by page id: {pid: {"title", "stream", "fields"}}, with the
    fields each one is linked from; and a count of what was left out, by reason. A redirect among the targets is
    followed once; its target takes the fields of the redirect. The word counts (article_problem) run in worker
    processes, a batch at a time; the records are read in this process."""
    report: Counter = Counter()
    judged: dict[int, dict | None] = {}
    redirects: dict[str, set[str]] = {}
    pending: list[tuple[int, dict]] = []

    # forkserver, not fork: the pool starts while the dump's download threads are alive, and forking a
    # multi-threaded process can deadlock
    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(min(4, os.cpu_count() or 1), mp_context=context) as procs:

        def flush() -> None:
            if not pending:
                return
            for (pid, _), why in zip(pending, procs.map(article_problem, [rec for _, rec in pending])):
                if why:
                    report[why] += 1
                    judged[pid] = None
            pending.clear()

        def judge(pid: int, stream: int, rec: dict, fields: set[str]) -> None:
            if pid in judged:
                if judged[pid] is not None:
                    judged[pid]["fields"] |= fields
                return
            judged[pid] = {"title": rec["title"], "stream": stream, "fields": set(fields)}
            pending.append((pid, rec))
            if len(pending) >= JUDGE_BATCH:
                flush()

        located = source.locate(set(targets))
        report["not in dump"] = len(targets) - len(located)
        where = {pid: (t, s) for t, (s, pid) in located.items()}

        def first(pid: int, rec: dict | None) -> None:
            title, stream = where[pid]
            if rec is None or rec["ns"] != "0":
                report["not an article"] += 1
            elif rec["redirect"]:
                dest = redirect_of(rec["text"])
                if dest is not None and dest[0] == "wiki" and dest[1]:
                    redirects.setdefault(dest[1], set()).update(targets[title])
                    report["redirects followed"] += 1
                else:
                    report["redirect elsewhere"] += 1
            else:
                judge(pid, stream, rec, targets[title])

        source.read(wanted_of(located), first)

        todo = {}
        for title, fields in redirects.items():
            if title in located and located[title][1] in judged:
                if judged[located[title][1]] is not None:
                    judged[located[title][1]]["fields"] |= fields
            else:
                todo[title] = fields
        located2 = source.locate(set(todo))
        report["redirect targets not in dump"] = len(todo) - len(located2)
        where2 = {pid: (t, s) for t, (s, pid) in located2.items()}

        def second(pid: int, rec: dict | None) -> None:
            title, stream = where2[pid]
            if rec is None or rec["ns"] != "0":
                report["redirect target not an article"] += 1
            elif rec["redirect"]:
                report["double redirect"] += 1
            else:
                judge(pid, stream, rec, todo[title])

        source.read(wanted_of(located2), second)
        flush()
    return {pid: c for pid, c in judged.items() if c is not None}, report


def titles_selection(source: DumpSource, titles: list[tuple[str, str]], out, target_words: int,
                     max_words: int) -> list[str]:
    """Write the listed articles to out, in the order of titles; returns the revision URIs written."""
    located = source.locate({t for t, _ in titles})
    for t, _ in titles:
        if t not in located:
            log(f"not in the dump index: {t!r}")
    pages: dict[int, dict] = {}
    source.read(wanted_of(located), lambda pid, rec: pages.__setitem__(pid, rec))
    uris = []
    for title, field in titles:
        if title not in located:
            continue
        art = article_of(pages[located[title][1]])
        if art is None:
            log(f"not an article (a redirect, another namespace or empty): {title!r}")
            continue
        uris.append(emit(out, art, target_words, max_words, field=field or None))
    return uris


def seeded_selection(source: DumpSource, seeds, n: int, rng: random.Random, out, target_words: int,
                     max_words: int) -> list[str]:
    """Write the stratified sample of the seeds' articles to out; returns the revision URIs written."""
    fields = list(dict.fromkeys(f for _, f in seeds))
    targets = seed_targets(source, seeds)
    cands, report = candidate_articles(source, targets)
    log(f"distinct candidate articles: {len(cands)}; left out: {dict(report)}")
    pools = {f: sorted(pid for pid, c in cands.items() if f in c["fields"]) for f in fields}
    for f in fields:
        log(f"pool [{f}]: {len(pools[f])} articles")
    chosen = stratified_sample(pools, n, rng)
    for f in fields:
        log(f"sample [{f}]: {len(chosen[f])} articles")
    field_of = {pid: f for f in fields for pid in chosen[f]}
    wanted: dict[int, set[int]] = {}
    for pid in field_of:
        wanted.setdefault(cands[pid]["stream"], set()).add(pid)
    uris = []

    def write(pid: int, rec: dict) -> None:
        art = article_of(rec)
        if art is None:
            raise RuntimeError(f"article {pid} was a candidate but is not an article on the second read")
        uris.append(emit(out, art, target_words, max_words, field=field_of[pid]))

    source.read(wanted, write)
    return uris


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="docmem scan-wiki",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--sample", type=int, default=1000, help="articles to draw (ns 0, not redirects)"
    )
    ap.add_argument("--seed", type=int, default=1, help="random seed for the draws")
    ap.add_argument("--cache", type=Path, required=True, help="directory for the dump index")
    ap.add_argument("--seed-pages", type=Path, default=None,
                    help="seed pages (one title per line, optionally 'title|field'): sample their links, "
                         "stratified by field, instead of drawing at random")
    ap.add_argument("--titles", type=Path, default=None,
                    help="article titles, one per line, optionally 'title|field': read exactly these, in file "
                         "order, instead of drawing at random")
    ap.add_argument(
        "--target-words", type=int, default=180, help="merge blocks up to this many words"
    )
    ap.add_argument("--max-words", type=int, default=320, help="split blocks longer than this")
    a = ap.parse_args(argv)

    t0 = time.time()
    path = ensure_index(a.cache)
    index = Index(path)
    size = dump_size()
    log(f"index: {len(index)} entries in {len(index.starts)} streams ({time.time() - t0:.0f} s)")

    out = sys.stdout
    out.write(json.dumps({"type": "scan_start", "protocol": PROTOCOL}) + "\n")
    if a.titles is not None and a.seed_pages is not None:
        ap.error("--titles and --seed-pages are exclusive")
    if a.titles is not None:
        stats = {"streams": 0, "bytes": 0, "draws": 0}
        source = DumpSource(index, path, size, stats)
        uris = titles_selection(source, read_seeds(a.titles), out, a.target_words, a.max_words)
    elif a.seed_pages is not None:
        stats = {"streams": 0, "bytes": 0, "draws": 0}
        source = DumpSource(index, path, size, stats)
        seeds = read_seeds(a.seed_pages)
        uris = seeded_selection(source, seeds, a.sample, random.Random(a.seed), out, a.target_words,
                                a.max_words)
    else:
        uris, stats = [], {"streams": 0, "bytes": 0, "draws": 0}
        for i, (rec, stats) in enumerate(sample(index, a.sample, a.seed, size), 1):
            uris.append(emit(out, rec, a.target_words, a.max_words))
            if i % 100 == 0:
                log(
                    f"{i}/{a.sample} articles, {stats['draws']} draws, {stats['streams']} streams, "
                    f"{stats['bytes'] / 2**20:.1f} MB, {time.time() - t0:.0f} s"
                )
    out.write(json.dumps({"type": "scan_end", "roots": [WIKI], "uris": uris}) + "\n")
    out.flush()
    log(
        f"done: {len(uris)} articles, {stats['streams']} streams, "
        f"{stats['bytes'] / 2**20:.1f} MB downloaded (dump), {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    main()
