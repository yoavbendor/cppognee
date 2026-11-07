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
from .keywords import KeywordIndex
from .store import log


class Stats(dict):
    def __missing__(self, k):
        return 0


async def existing(graph, ids: list[str]) -> dict:
    return {n["id"]: store.node_props(n) for n in await graph.get_nodes(ids)} if ids else {}


async def plan_doc(graph, doc: dict, sections: list[dict], passages: list[dict], force: bool, st: Stats,
                   gap_paths: set | None = None, links: list[dict] | None = None):
    """The writes for one scanned document: {"embed", "graph", "delete", ...}, or None when it is unchanged.
    With gap_paths (a refresh), an unchanged document is not skipped: its passages in those sections are
    enriched again, and nothing is re-embedded."""
    DocFile, DocSection, DocPassage, *_ = store.models()
    uri = doc["uri"]
    f = DocFile(name=uri, key=store.key("file", uri), title=doc["title"], sha256=doc["sha256"],
                size=doc["size"], mtime=doc["mtime"], url=doc.get("url", ""))
    old = await existing(graph, [str(f.id)])
    unchanged = old and old[str(f.id)].get("sha256") == doc["sha256"] and not force
    if unchanged and not gap_paths:
        st["docs_unchanged"] += 1
        return None
    if unchanged:
        st["docs_refreshed"] += 1
    else:
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
                              section_offset=sec.offset, section_length=sec.length,
                              extractor=doc.get("extractor", ""), doc_sha256=doc["sha256"], section=sec))

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
    link_nodes, link_edges = plan_links(doc, f, pas, links or [], st)
    new_idx = {i for i, p in enumerate(pas) if str(p.id) not in old_pas}
    if gap_paths:
        new_idx |= {i for i, p in enumerate(pas) if p.path in gap_paths}
    if unchanged:  # a refresh of an unchanged document: graph writes only, keep its earlier enrichment
        f.enriched_by = old[str(f.id)].get("enriched_by", "")
        return {"embed": [], "graph": pas + secs + [f] + link_nodes, "delete": [], "file": f, "passages": pas,
                "raw": passages, "new": new_idx, "edges": link_edges}
    # the keyword index takes each new passage's text once, and forgets the passages that are gone
    kw_add = [(str(p.id), passages[i]["text"]) for i, p in enumerate(pas) if str(p.id) not in old_pas]
    kw_del = sorted(old_pas - {str(p.id) for p in pas})
    return {"embed": embed + sec_embed + [f], "graph": keep + sec_keep + link_nodes, "delete": delete, "file": f,
            "edges": link_edges, "kw_add": kw_add, "kw_del": kw_del,
            "passages": pas, "raw": passages, "new": new_idx}


def plan_links(doc, f, pas, links, st: Stats):
    """DocTarget nodes and edges for a document's link records: passage -links_to-> target (the anchor
    text on the edge; a link outside every passage hangs off the file), file -in_category-> category,
    and file -is_target-> the target the document itself is (its title or URI)."""
    import bisect

    starts = [p.offset for p in pas]
    nodes, edges = {}, []
    me = store.self_target(doc)
    nodes[me.key] = me
    edges.append((str(f.id), str(me.id), "is_target", {"edge_text": f"{f.title} is {me.name}"}))
    for ln in links:
        t = store.target(ln["kind"], ln["target"])
        if t.key == me.key:
            continue  # a link to the document itself
        nodes.setdefault(t.key, t)
        t = nodes[t.key]
        if ln["kind"] == "category":
            edges.append((str(f.id), str(t.id), "in_category", {"edge_text": f"{f.title} in {t.name}"}))
            continue
        i = bisect.bisect_right(starts, ln["offset"]) - 1
        src = pas[i] if i >= 0 and ln["offset"] < pas[i].offset + pas[i].length else f
        props = {"edge_text": f"{ln.get('anchor') or t.name} -> {t.name}", "anchor": ln.get("anchor", "")}
        if ln.get("fragment"):
            props["fragment"] = ln["fragment"]
        edges.append((str(src.id), str(t.id), "links_to", props))
    st["links"] += len(links)
    st["link_targets"] += len(nodes) - 1
    return list(nodes.values()), edges


async def enrich_plans(plans, enricher, a, st: Stats):
    """Run the enrichment calls of a window of documents concurrently and turn the replies into
    DataPoints and edges on each plan (mutating its passages' mentions and its file's topics)."""
    from . import enrich as en

    DocFile, DocSection, DocPassage, DocEntity, DocSummary, DocTopic, DocGap = store.models()
    jobs = []
    for plan in plans:
        texts = plan["raw"]
        paths = [p.path for p in plan["passages"]]
        for g in en.groups(paths, texts, a.enrich_words, plan["new"]):
            jobs.append((plan, g))
    results = await asyncio.gather(*(enricher.group(plan["file"].title, g, plan["raw"]) for plan, g in jobs))
    for plan in plans:
        plan.setdefault("edges", [])
        plan.setdefault("topics", [])
        plan.setdefault("ents", {})
    for (plan, g), (rec, reason) in zip(jobs, results):
        f, pas = plan["file"], plan["passages"]
        uri = f.name

        def gap_for(path, part, idx, why):
            sec = pas[idx[0]].section
            return DocGap(name=uri, key=store.key("gap", uri, sec.key, part, "enrich"), kind="enrich",
                          reason=why or "", path=path, part=part, section=sec)

        if rec is None:
            plan["embed"] += [gap_for(path, part, idx, reason) for path, part, idx in g]
            st["enrich_gap_sections"] += len(g)
            continue
        st["enrich_groups"] += 1
        # a section enriched now closes its earlier gap, if any
        plan["delete"] += [str(gap_for(path, part, idx, "").id) for path, part, idx in g]
        where = {(path, part): idx for path, part, idx in g}
        for sm in rec["summaries"]:
            idx = where.get((sm["path"], sm["part"]))
            if idx:
                sec = pas[idx[0]].section
                plan["embed"].append(DocSummary(name=uri, key=store.key("summary", uri, sec.key, sm["part"]),
                                                text=sm["text"], part=sm["part"], section=sec))
                st["summaries"] += 1
        ents = plan["ents"]

        def entity(name, kind="other", description=""):
            k = en.norm(name)
            if k not in ents:
                ents[k] = DocEntity(name=name, key=store.key("entity", uri, k), kind=kind, description=description,
                                    doc_uri=uri)
                st["entities"] += 1
            elif description and not ents[k].description:
                ents[k].description = description
            return ents[k]

        for e in rec["entities"]:
            ent = entity(e["name"], e["type"], e["description"])
            for i in e["passages"]:
                if all(m.id != ent.id for m in pas[i].mentions):
                    pas[i].mentions.append(ent)
        for r in rec["relations"]:
            s_, t_ = entity(r["source"]), entity(r["target"])
            props = {"edge_text": f"{s_.name} {r['relation'].replace('_', ' ')} {t_.name}"}
            if r["passages"]:
                props["passage_key"] = pas[r["passages"][0]].key
            plan["edges"].append((str(s_.id), str(t_.id), r["relation"], props))
            st["relations"] += 1
        plan["topics"] += rec["topics"]
    for plan in plans:
        f = plan["file"]
        if plan["topics"]:
            seen = {}
            for t in plan["topics"]:
                seen.setdefault(en.norm(t), t)
            f.topics = [DocTopic(name=t, key=store.key("topic", k)) for k, t in seen.items()]
        f.enriched_by = enricher.version
        # entities go in with the graph-only writes: they have no vector of their own at ingest
        plan["graph"] += list(plan["ents"].values())


async def write(plan_embed, plan_graph, delete_ids, st: Stats, edges=None):
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
        if plan_graph or edges:
            # custom edges go last, once both calls have written their endpoints
            await add_data_points(plan_graph, ctx=ctx, graph_only=True, custom_edges=edges or None)
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


async def prune(graph, roots: list[str], seen: set[str], st: Stats, kw: KeywordIndex | None = None):
    nodes, _ = await graph.get_filtered_graph_data([{"type": ["DocFile"]}], include_edges=False)
    gone = []
    for n in nodes:
        props = store.node_props({"id": n[0], **n[1]}) if isinstance(n, (tuple, list)) else store.node_props(n)
        uri = props.get("name") or ""
        if any(uri == r or uri.startswith(r.rstrip("/") + "/") for r in roots) and uri not in seen:
            gone.append(uri)
    if not gone:
        return
    passage_ids = [n["id"] for n in await store.nodes_of(graph, "DocPassage", gone)]
    ids = passage_ids + [n["id"] for t in ("DocSection", "DocFile") for n in await store.nodes_of(graph, t, gone)]
    st["docs_removed"] += len(gone)
    await write([], [], ids, st)
    if kw:
        kw.delete(passage_ids)
        kw.flush()


async def run(lines, a):
    import cognee

    await cognee.run_migrations()
    graph, _ = await store.engines()
    st = Stats(batches=0)
    t0 = time.time()
    kw = None if a.no_keywords else KeywordIndex(a.db, create=True)
    pend_embed, pend_graph, pend_delete, pend_edges = [], [], [], []
    pend_kw_add, pend_kw_del = [], []
    doc, sections, passages, links = None, [], [], []
    scan_end = None
    window = []
    enricher = None
    if a.enrich:
        from . import enrich as en

        enricher = en.Enricher(a.api_base, a.api_key, a.model, a.enrich_concurrency,
                               en.Budget(a.enrich_max_calls, a.enrich_max_tokens), json_mode=not a.no_json_mode)

    async def drain_window():
        nonlocal pend_embed, pend_graph, pend_delete, pend_edges, pend_kw_add, pend_kw_del, window
        if enricher and window:
            t = time.time()
            await enrich_plans(window, enricher, a, st)
            st["enrich_seconds"] += time.time() - t
        for plan in window:
            pend_embed += plan["embed"]
            pend_graph += plan["graph"]
            pend_delete += plan["delete"]
            pend_edges += plan.get("edges", [])
            pend_kw_add += plan.get("kw_add", [])
            pend_kw_del += plan.get("kw_del", [])
        window = []

    async def flush():
        nonlocal pend_embed, pend_graph, pend_delete, pend_edges, pend_kw_add, pend_kw_del
        if pend_embed or pend_graph or pend_delete:
            await write(pend_embed, pend_graph, pend_delete, st, pend_edges)
            if kw:  # only once the batch is in the graph and the vector store
                kw.delete(pend_kw_del)
                for pid, text in pend_kw_add:
                    kw.add(pid, text)
                kw.flush()
            log(f"docmem index: {st['docs_new'] + st['docs_changed'] + st['docs_unchanged']} docs, "
                f"{st['passages_embedded']} passages embedded, {st['passages_kept']} kept, "
                f"{time.time() - t0:.0f}s")
        pend_embed, pend_graph, pend_delete, pend_edges = [], [], [], []
        pend_kw_add, pend_kw_del = [], []

    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        r = json.loads(line)
        t = r.get("type")
        if t == "scan_start":
            if r.get("protocol") != PROTOCOL:
                raise SystemExit(f"docmem index: stream protocol {r.get('protocol')}, expected {PROTOCOL}")
        elif t == "doc":
            doc, sections, passages, links = r, [], [], []
        elif t == "section":
            sections.append(r)
        elif t == "passage":
            passages.append(r)
        elif t == "link":
            links.append(r)
        elif t == "end":
            if doc is None or r["uri"] != doc["uri"]:
                raise SystemExit(f"docmem index: line {n}: 'end' without its 'doc'")
            plan = await plan_doc(graph, doc, sections, passages, a.force, st,
                                  (getattr(a, "gaps", None) or {}).get(doc["uri"]), links)
            doc = None
            if plan:
                window.append(plan)
            if len(window) >= (a.enrich_window if enricher else 1):
                await drain_window()
            if len(pend_embed) + len(pend_graph) >= a.batch:
                await flush()
        elif t == "scan_end":
            scan_end = r
        else:
            raise SystemExit(f"docmem index: line {n}: unknown record type {t!r}")
    await drain_window()
    await flush()
    if a.prune:
        if scan_end is None:
            raise SystemExit("docmem index: --prune needs a complete scan (no scan_end record)")
        await prune(graph, scan_end["roots"], set(scan_end["uris"]), st, kw)
    st["seconds"] = round(time.time() - t0, 1)
    st["store_seconds"] = round(st["store_seconds"], 1)
    if kw:
        kw.tidy()
        st["keywords"] = kw.stats()
    if enricher:
        st["enrich_seconds"] = round(st["enrich_seconds"], 1)
        st["enrich_usage"] = enricher.budget.as_dict()
    print(json.dumps(dict(st)))


def parser(prog="docmem index", doc=__doc__):
    ap = argparse.ArgumentParser(prog=prog, description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--embed", help="embedding model for a new index: fastembed:<model> (default "
                    "fastembed:BAAI/bge-small-en-v1.5) or openai:<model> on --api-base")
    ap.add_argument("--embed-dims", type=int)
    ap.add_argument("--api-base", default=None, help="OpenAI-compatible base URL (or DOCMEM_API_BASE)")
    ap.add_argument("--api-key", default=None, help="API key (or DOCMEM_API_KEY)")
    ap.add_argument("--batch", type=int, default=1000, help="DataPoints per cognee pipeline run")
    ap.add_argument("--force", action="store_true", help="re-index documents whose hash is unchanged")
    ap.add_argument("--prune", action="store_true", help="remove indexed files the scan no longer saw")
    ap.add_argument("--enrich", action="store_true",
                    help="have the chat model (--api-base, --model) write summaries, entities, relations and "
                         "topics while the text passes through")
    ap.add_argument("--model", help="chat model for --enrich (or DOCMEM_MODEL)")
    ap.add_argument("--enrich-words", type=int, default=900, help="passage words per enrichment call")
    ap.add_argument("--enrich-concurrency", type=int, default=2, help="enrichment calls in flight")
    ap.add_argument("--enrich-window", type=int, default=8, help="documents enriched together")
    ap.add_argument("--enrich-max-calls", type=int, help="stop calling after this many (the rest become gaps)")
    ap.add_argument("--enrich-max-tokens", type=int, help="stop after this many prompt+completion tokens")
    ap.add_argument("--no-json-mode", action="store_true", help="the server does not take response_format")
    ap.add_argument("--no-keywords", action="store_true",
                    help="do not build the keyword (BM25) index; passages it would have covered are not searchable "
                         "by kwsearch, so re-index into a new --db to add them later")
    return ap


def main(argv=None):
    ap = parser()
    ap.add_argument("input", nargs="?", type=Path, help="read this JSONL file instead of stdin")
    a = ap.parse_args(argv)
    import os

    store.configure(a.db, a.embed, a.embed_dims, a.api_base or os.environ.get("DOCMEM_API_BASE"),
                    a.api_key or os.environ.get("DOCMEM_API_KEY"), create=True)
    lines = open(a.input, encoding="utf-8") if a.input else sys.stdin
    asyncio.run(run(lines, a))


if __name__ == "__main__":
    main()
