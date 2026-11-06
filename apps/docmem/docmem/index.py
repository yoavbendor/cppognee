"""Indexer: read the scanner's JSON lines on stdin and store them in cognee, keeping vectors and
provenance, not text.

    docmem scan docs/ | docmem index --db ./kb

Run it again after editing files: a document whose SHA-256 is unchanged is skipped; in a changed one
only new passages are embedded (a passage is identified by its file, its bytes' hash and which
occurrence of them it is), passages still present get their new offsets, and passages gone are
deleted from the graph and the vector table. With --prune, files under the scanned roots that the
scan no longer saw are removed from the index.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from . import PROTOCOL, store
from .store import log


class Stats(dict):
    def __missing__(self, k):
        return 0


async def existing(graph, ids: list[str]) -> dict:
    return {n["id"]: store.node_props(n) for n in await graph.get_nodes(ids)} if ids else {}


async def plan_doc(graph, doc: dict, sections: list[dict], passages: list[dict], force: bool, st: Stats):
    """(embed, graph_only, delete_ids) for one scanned document."""
    DocFile, DocSection, DocPassage = store.models()
    uri = doc["uri"]
    f = DocFile(name=uri, key=store.key("file", uri), title=doc["title"], sha256=doc["sha256"],
                size=doc["size"], mtime=doc["mtime"])
    old = await existing(graph, [str(f.id)])
    if old and old[str(f.id)].get("sha256") == doc["sha256"] and not force:
        st["docs_unchanged"] += 1
        return [], [], []
    st["docs_new" if not old else "docs_changed"] += 1

    sec_by_path, secs, seen = {}, [], {}
    for s in sections:
        p = " › ".join(s["path"])
        n = seen[p] = seen.get(p, -1) + 1  # repeated heading paths stay distinct
        node = DocSection(name=uri, key=store.key("section", uri, p, n), path=p, level=s["level"],
                          offset=s["offset"], length=s["length"], line=s["line"], doc=f)
        secs.append(node)
        sec_by_path.setdefault(p, []).append(node)

    def section_for(path: str, offset: int):
        cands = sec_by_path.get(path) or []
        best = None
        for c in cands:
            if c.offset <= offset < c.offset + c.length:
                best = c
        if best is None:  # text before the first heading: a section for the whole file
            best = DocSection(name=uri, key=store.key("section", uri, path, "lead"), path=path, level=0,
                              offset=0, length=doc["size"], line=1, doc=f)
            secs.append(best)
            sec_by_path[path] = [best]
        return best

    pas, occ = [], {}
    for p in passages:
        path = " › ".join(p["path"])
        n = occ[p["sha256"]] = occ.get(p["sha256"], -1) + 1
        sec = section_for(path, p["offset"])
        pas.append(DocPassage(name=uri, key=store.key("passage", uri, p["sha256"], n), path=path,
                              offset=p["offset"], length=p["length"], line=p["line"], end_line=p["end_line"],
                              sha256=p["sha256"], embed_text=f"{path}\n\n{p['text']}",
                              section_offset=sec.offset, section_length=sec.length, section=sec))

    old_pas = {n["id"] for n in await store.nodes_of(graph, "DocPassage", [uri])} if old else set()
    old_secs = {n["id"] for n in await store.nodes_of(graph, "DocSection", [uri])} if old else set()
    embed = [p for p in pas if str(p.id) not in old_pas]
    keep = [p for p in pas if str(p.id) in old_pas]
    new_ids = {str(p.id) for p in pas} | {str(s.id) for s in secs}
    delete = sorted((old_pas | old_secs) - new_ids)
    st["passages_embedded"] += len(embed)
    st["passages_kept"] += len(keep)
    st["nodes_deleted"] += len(delete)
    # sections are re-embedded with the doc's new passages (their vector is just the heading path);
    # a doc whose passages all survived keeps its section vectors too
    sec_embed = [s for s in secs if str(s.id) not in old_secs]
    sec_keep = [s for s in secs if str(s.id) in old_secs]
    return embed + sec_embed + [f], keep + sec_keep, delete


async def write(plan_embed, plan_graph, delete_ids, st: Stats):
    """One cognee pipeline run: embed + store, graph-only updates, then deletions."""
    import cognee
    from cognee.modules.pipelines import Task
    from cognee.tasks.storage import add_data_points

    async def store_batch(items, ctx=None):
        if delete_ids:
            graph, vector = await store.engines()
            await graph.delete_nodes(delete_ids)
            for coll in (store.PASSAGES, store.SECTIONS):
                if await vector.has_collection(coll):
                    await vector.delete_data_points(coll, delete_ids)
        if plan_embed:
            await add_data_points(plan_embed, ctx=ctx, index_edges=False)
        if plan_graph:
            await add_data_points(plan_graph, ctx=ctx, graph_only=True)
        return []

    t = time.time()
    result = await cognee.run_custom_pipeline(tasks=[Task(store_batch, needs_llm=False)],
                                              data=[{"batch": st["batches"]}], dataset=store.DATASET)
    for run in (result or {}).values():
        status = getattr(run, "status", "")
        if "Errored" in str(status) or "ERRORED" in str(status):
            raise SystemExit(f"docmem index: the cognee pipeline failed: {getattr(run, 'payload', run)}")
    st["batches"] += 1
    st["store_seconds"] += time.time() - t


async def prune(graph, roots: list[str], seen: set[str], st: Stats):
    nodes, _ = await graph.get_filtered_graph_data([{"type": ["DocFile"]}], include_edges=False)
    gone = []
    for n in nodes:
        props = store.node_props({"id": n[0], **n[1]}) if isinstance(n, (tuple, list)) else store.node_props(n)
        uri = props.get("name") or ""
        if any(uri == r or uri.startswith(r.rstrip("/") + "/") for r in roots) and uri not in seen:
            gone.append(uri)
    if not gone:
        return
    ids = [n["id"] for t in ("DocPassage", "DocSection", "DocFile") for n in await store.nodes_of(graph, t, gone)]
    st["docs_removed"] += len(gone)
    await write([], [], ids, st)


async def run(lines, a):
    import cognee

    await cognee.run_migrations()
    graph, _ = await store.engines()
    st = Stats(batches=0)
    t0 = time.time()
    pend_embed, pend_graph, pend_delete = [], [], []
    doc, sections, passages = None, [], []
    scan_end = None

    async def flush():
        nonlocal pend_embed, pend_graph, pend_delete
        if pend_embed or pend_graph or pend_delete:
            await write(pend_embed, pend_graph, pend_delete, st)
            log(f"docmem index: {st['docs_new'] + st['docs_changed'] + st['docs_unchanged']} docs, "
                f"{st['passages_embedded']} passages embedded, {st['passages_kept']} kept, "
                f"{time.time() - t0:.0f}s")
        pend_embed, pend_graph, pend_delete = [], [], []

    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        r = json.loads(line)
        t = r.get("type")
        if t == "scan_start":
            if r.get("protocol") != PROTOCOL:
                raise SystemExit(f"docmem index: stream protocol {r.get('protocol')}, expected {PROTOCOL}")
        elif t == "doc":
            doc, sections, passages = r, [], []
        elif t == "section":
            sections.append(r)
        elif t == "passage":
            passages.append(r)
        elif t == "end":
            if doc is None or r["uri"] != doc["uri"]:
                raise SystemExit(f"docmem index: line {n}: 'end' without its 'doc'")
            e, g, d = await plan_doc(graph, doc, sections, passages, a.force, st)
            pend_embed += e
            pend_graph += g
            pend_delete += d
            doc = None
            if len(pend_embed) + len(pend_graph) >= a.batch:
                await flush()
        elif t == "scan_end":
            scan_end = r
        else:
            raise SystemExit(f"docmem index: line {n}: unknown record type {t!r}")
    await flush()
    if a.prune:
        if scan_end is None:
            raise SystemExit("docmem index: --prune needs a complete scan (no scan_end record)")
        await prune(graph, scan_end["roots"], set(scan_end["uris"]), st)
    st["seconds"] = round(time.time() - t0, 1)
    st["store_seconds"] = round(st["store_seconds"], 1)
    print(json.dumps(dict(st)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docmem index", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--embed", help="embedding model for a new index: fastembed:<model> (default "
                    "fastembed:BAAI/bge-small-en-v1.5) or openai:<model> on --api-base")
    ap.add_argument("--embed-dims", type=int)
    ap.add_argument("--api-base", default=None, help="OpenAI-compatible base URL (or DOCMEM_API_BASE)")
    ap.add_argument("--api-key", default=None, help="API key (or DOCMEM_API_KEY)")
    ap.add_argument("--batch", type=int, default=1000, help="DataPoints per cognee pipeline run")
    ap.add_argument("--force", action="store_true", help="re-index documents whose hash is unchanged")
    ap.add_argument("--prune", action="store_true", help="remove indexed files the scan no longer saw")
    ap.add_argument("input", nargs="?", type=Path, help="read this JSONL file instead of stdin")
    a = ap.parse_args(argv)
    import os

    store.configure(a.db, a.embed, a.embed_dims, a.api_base or os.environ.get("DOCMEM_API_BASE"),
                    a.api_key or os.environ.get("DOCMEM_API_KEY"), create=True)
    lines = open(a.input, encoding="utf-8") if a.input else sys.stdin
    asyncio.run(run(lines, a))


if __name__ == "__main__":
    main()
