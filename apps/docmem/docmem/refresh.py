"""Refresh: close the index's gaps (work the one read could not do, e.g. enrichment over budget) with one
more read per source, all of a document's gaps in that read.

    docmem refresh --db kb --api-base ... --model ... [--max-docs 50] [--enrich-max-calls 200]

Documents with the most open gaps go first. Each is read once (a local file, or a wiki revision fetched
by its oldid), cut again exactly as at index time, and only its gap sections are enriched; nothing is
re-embedded. If the source has changed since indexing, it is indexed as an update in the same read.
"""
import asyncio
import io
import json
import os
import sys

from . import index, store
from .store import log


async def open_gaps(graph):
    """{uri: {section path, ...}} for every DocGap, and the DocFile properties of those URIs."""
    nodes, _ = await graph.get_filtered_graph_data([{"type": ["DocGap"]}], include_edges=False)
    gaps = {}
    for n in nodes:
        p = store.node_props({"id": n[0], **n[1]}) if isinstance(n, (tuple, list)) else store.node_props(n)
        gaps.setdefault(p["name"], set()).add(p.get("path", ""))
    files = {n["name"]: n for n in await store.nodes_of(graph, "DocFile", list(gaps))} if gaps else {}
    return gaps, files


def records(uri: str, f: dict, a):
    """The scan records of one source, read now (once)."""
    from . import scan

    if uri.startswith("file://"):
        path = store.uri_path(uri)
        if not path.exists():
            log(f"docmem refresh: {uri} is gone; re-scan with --prune to drop it")
            return
        yield from scan.scan_file(path, a.target_words, a.max_words)
        return
    from . import sources
    from .scan_wiki import emit

    m = sources.WIKI_OLDID.match(uri)
    if not m:
        raise SystemExit(f"docmem refresh: no reader for {uri}")
    raw = sources.fetch(f"https://en.wikipedia.org/w/index.php?oldid={m.group(1)}&action=raw")
    if raw is None:
        log(f"docmem refresh: {uri} is missing online; its gaps stay open")
        return
    buf = io.StringIO()
    emit(buf, {"text": raw.decode("utf-8"), "revid": int(m.group(1)), "title": f["title"],
               "mtime": f.get("mtime", 0), "page_id": f.get("page_id", 0)}, a.target_words, a.max_words)
    for line in buf.getvalue().splitlines():
        yield json.loads(line)


def main(argv=None):
    ap = index.parser("docmem refresh", __doc__)
    ap.add_argument("--max-docs", type=int, help="read at most this many sources this run")
    ap.add_argument("--target-words", type=int, default=180, help="as given to scan")
    ap.add_argument("--max-words", type=int, default=320, help="as given to scan")
    a = ap.parse_args(argv)
    store.configure(a.db, api_base=a.api_base or os.environ.get("DOCMEM_API_BASE"),
                    api_key=a.api_key or os.environ.get("DOCMEM_API_KEY"))
    a.enrich, a.prune = True, False

    async def go():
        import cognee

        await cognee.run_migrations()
        graph, _ = await store.engines()
        gaps, files = await open_gaps(graph)
        order = sorted(gaps, key=lambda u: -len(gaps[u]))[: a.max_docs]
        log(f"docmem refresh: {sum(map(len, gaps.values()))} gap sections in {len(gaps)} documents; "
            f"reading {len(order)}")
        a.gaps = {u: gaps[u] for u in order}

        def lines():
            yield json.dumps({"type": "scan_start", "protocol": 1})
            for u in order:
                for r in records(u, files.get(u, {}), a):
                    yield json.dumps(r, ensure_ascii=False)

        await index.run(lines(), a)

    asyncio.run(go())


if __name__ == "__main__":
    sys.exit(main())
