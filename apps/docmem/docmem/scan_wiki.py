"""Scanner for a random sample of English Wikipedia articles, read from the official compressed dump.

    docmem scan-wiki --sample 1000 --seed 1 --cache DIR > wiki.jsonl

The dump index (offset:page_id:title per page) is downloaded once into --cache and parsed. Its distinct
offsets start the dump's independent bz2 streams (about 100 pages each). Entries are drawn uniformly from the
index with random.Random(seed); a draw is kept only if its page is in the namespace 0, not a redirect and has
text, otherwise another draw replaces it (no page is kept twice). Each needed stream is fetched with one Range
request (up to 4 at a time) and decompressed in memory. Article text is never written to disk.

The stream is the scan.py protocol (scan_start, doc, section, passage, end per article, scan_end). A doc's
uri is the revision it was read from, https://en.wikipedia.org/w/index.php?oldid=N, and its sha256 is that of the
wikitext (see docmem.sources.read_span for how passages are read back).
"""

import argparse
import bz2
import calendar
import hashlib
import json
import random
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import PROTOCOL
from .extract.wikiplain import EXTRACTOR, to_markdown_with_links
from .scan import scan_bytes
from .sources import USER_AGENT, fetch

BASE = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream"
INDEX_URL = BASE + "-index.txt.bz2"
DUMP_URL = BASE + ".xml.bz2"
WIKI = "https://en.wikipedia.org/"
MAX_CONCURRENT = 4


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


def dump_size() -> int:
    req = urllib.request.Request(DUMP_URL, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers["Content-Length"])


def article_of(page) -> dict | None:
    """(ns 0, not a redirect, text) as a record, or None when the page is not a usable article."""
    if page.findtext("ns") != "0" or page.find("redirect") is not None:
        return None
    text = page.findtext("revision/text")
    if not text or not text.strip():
        return None
    ts = time.strptime(page.findtext("revision/timestamp"), "%Y-%m-%dT%H:%M:%SZ")
    return {
        "page_id": int(page.findtext("id")),
        "title": page.findtext("title"),
        "revid": int(page.findtext("revision/id")),
        "mtime": calendar.timegm(ts),
        "text": text,
    }


def fetch_stream(index: Index, stream: int, size: int, wanted: set[int]):
    """({page_id: record or None} for the wanted pages of one stream, bytes downloaded)."""
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
            found[pid] = article_of(page)
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"pages {sorted(missing)} are not in the stream at offset {start}")
    return found, len(body)


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
                rec = results[index.page_id[k]]
                if rec is None or rec["page_id"] in seen:
                    continue
                seen.add(rec["page_id"])
                yield rec, stats
                if len(seen) == n:
                    return


def emit(out, rec: dict, target_words: int, max_words: int) -> str:
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
    md, links = to_markdown_with_links(title, text)
    for r in scan_bytes(md.encode("utf-8"), doc, title, target_words, max_words, links=links):
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
    return doc["uri"]


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
    ap.add_argument(
        "--target-words", type=int, default=180, help="merge blocks up to this many words"
    )
    ap.add_argument("--max-words", type=int, default=320, help="split blocks longer than this")
    a = ap.parse_args(argv)

    t0 = time.time()
    index = Index(ensure_index(a.cache))
    size = dump_size()
    log(f"index: {len(index)} entries in {len(index.starts)} streams ({time.time() - t0:.0f} s)")

    out = sys.stdout
    out.write(json.dumps({"type": "scan_start", "protocol": PROTOCOL}) + "\n")
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
        f"done: {len(uris)} articles, {stats['draws']} draws, {stats['streams']} streams, "
        f"{stats['bytes'] / 2**20:.1f} MB downloaded (dump), {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    main()
