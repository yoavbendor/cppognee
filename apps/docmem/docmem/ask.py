"""Ask: embed the question with the index's model, find the nearest passages in cognee's vector table,
read their byte ranges back from the files (checking each range's SHA-256), and have the chat model
answer from them, citing file and line.

    docmem ask --db ./kb --api-base https://host/v1 --model gpt-4.1-mini "what does X do?"
    docmem ask --db ./kb --no-answer "..."      # retrieval only: no API needed
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from . import store

ANSWER_SYSTEM = (
    "You answer questions using only the numbered excerpts from the user's documents. "
    "Cite the excerpts you use as [1], [2]. Be concise: a sentence or a short paragraph. "
    "If the excerpts do not contain the answer, reply with 'NOT FOUND:' and one sentence on what they "
    "do cover. Do not use outside knowledge.")

# --hops: the same answer prompt, plus a SEARCH option for a fact the excerpts lack.
ANSWER_SYSTEM_HOPS = (
    "You answer questions using only the numbered excerpts from the user's documents. "
    "Cite the excerpts you use as [1], [2]. Be concise: a sentence or a short paragraph. "
    "If a fact the question needs is missing from the excerpts (another subject, a place, a date, or the "
    "thing a found fact points to), reply ONLY with 1 or 2 lines, each 'SEARCH: <short search query for "
    "that fact>', and nothing else. Otherwise answer. Reply 'NOT FOUND:' and one sentence only if the "
    "excerpts clearly do not cover it. Do not use outside knowledge.")

RERANK_SYSTEM = (
    "You rank document excerpts by how well they answer a question. Reply with JSON only: "
    '{"scores": [s1, s2, ...]}, one integer per excerpt in order, 0 = irrelevant, 1 = related, '
    "2 = partly answers, 3 = answers it.")


async def retrieve(question: str, k: int, candidates: int | None = None):
    """The nearest passages: [{rank, distance, uri, path, offset, length, line, end_line, sha256, ...}]."""
    graph, vector = await store.engines()
    t = time.time()
    results = await vector.search(store.PASSAGES, query_text=question, limit=candidates or k)
    t_vec = time.time() - t
    ids = [str(r.id) for r in results]
    nodes = {n["id"]: store.node_props(n) for n in await graph.get_nodes(ids)}
    hits = []
    for r in results:
        n = nodes.get(str(r.id))
        if n is None:
            continue  # a vector without its graph node: the indexer was interrupted mid-write
        hits.append({"distance": round(float(r.score), 4), **{k_: n[k_] for k_ in (
            "id", "name", "path", "offset", "length", "line", "end_line", "sha256", "section_offset",
            "section_length")}, "extractor": n.get("extractor", ""), "doc_sha256": n.get("doc_sha256", "")})
    for i, h in enumerate(hits, 1):
        h["rank"] = i
        h["uri"] = h.pop("name")
    return hits, {"vector_ms": round(t_vec * 1000, 1), "total_ms": round((time.time() - t) * 1000, 1)}


def span(h, offset, length, sha256=None):
    from . import sources

    return sources.read_span(h["uri"], offset, length, sha256, extractor=h["extractor"] or None,
                             doc_sha256=h["doc_sha256"] or None, title=h["path"].split(" › ")[0])


async def link_neighbours(graph, passage_ids: list[str]):
    """Passages one link away from the given ones, with no LLM: through every target they link to
    (wiki title, URL, file, ID), the other passages linking to that target and the passages of the
    document that is the target. {passage id: [target name, ...]} and {target name: in-degree}."""
    if not passage_ids:
        return {}, {}
    rows = await graph.query(
        "MATCH (p:Node)-[r:EDGE]->(t:Node) WHERE p.id IN $ids AND r.relationship_name = 'links_to' "
        "RETURN DISTINCT t.id, t.name", {"ids": passage_ids})
    targets = {tid: name for tid, name in rows}
    if not targets:
        return {}, {}
    tids = list(targets)
    via, degree = {}, {}
    # other passages linking to the same targets
    for pid, tid in await graph.query(
            "MATCH (q:Node)-[r:EDGE]->(t:Node) WHERE t.id IN $tids AND r.relationship_name = 'links_to' "
            "AND q.type = 'DocPassage' RETURN q.id, t.id", {"tids": tids}):
        degree[targets[tid]] = degree.get(targets[tid], 0) + 1
        via.setdefault(pid, []).append(targets[tid])
    # passages of documents that are the targets
    for pid, tid in await graph.query(
            "MATCH (f:Node)-[r:EDGE]->(t:Node) WHERE t.id IN $tids AND r.relationship_name = 'is_target' "
            "MATCH (p:Node)-[r1:EDGE]->(s:Node)-[r2:EDGE]->(f) WHERE r1.relationship_name = 'section' "
            "AND r2.relationship_name = 'doc' RETURN p.id, t.id", {"tids": tids}):
        via.setdefault(pid, []).append(targets[tid] + " (its article)")
    for pid in passage_ids:
        via.pop(pid, None)
    return via, degree


async def expand_links(question: str, hits, n: int, max_degree: int):
    """Add up to n passages reached through links from the hits, ranked by the question's similarity to
    their stored vectors. Targets linked from more than max_degree passages (e.g. a country) are too
    common to say anything and are not followed."""
    import numpy as np

    graph, vector = await store.engines()
    via, degree = await link_neighbours(graph, [h["id"] for h in hits])
    common = {t for t, d in degree.items() if d > max_degree}
    cands = {pid: ts for pid, ts in via.items() if any(t not in common for t in ts)}
    if not cands:
        return hits, []
    rows = await vector.retrieve(store.PASSAGES, list(cands), include_vector=True)
    q = np.asarray((await vector.embedding_engine.embed_text([question]))[0], dtype=np.float32)
    q /= np.linalg.norm(q) or 1
    scored = []
    for r in rows:
        v = np.asarray(r.payload["vector"], dtype=np.float32)
        scored.append((1 - float(q @ v / (np.linalg.norm(v) or 1)), str(r.id)))
    scored.sort()
    chosen = [pid for _, pid in scored[:n]]
    nodes = {nd["id"]: store.node_props(nd) for nd in await graph.get_nodes(chosen)}
    dist = {pid: d for d, pid in scored}
    added = []
    for pid in chosen:
        nd = nodes.get(pid)
        if nd is None:
            continue
        h = {"distance": round(dist[pid], 4), **{k_: nd[k_] for k_ in (
            "id", "path", "offset", "length", "line", "end_line", "sha256", "section_offset", "section_length")},
            "extractor": nd.get("extractor", ""), "doc_sha256": nd.get("doc_sha256", ""), "uri": nd["name"],
            "via": [t for t in cands[pid] if t not in common]}
        added.append(h)
    out = hits + added
    for i, h in enumerate(out, 1):
        h["rank"] = i
    return out, [h["via"] for h in added]


def read_hits(hits, context: str, max_bytes: int):
    """Attach each hit's text. context "section" reads the enclosing section instead of the passage
    (once per section), cut at max_bytes."""
    seen = set()
    out = []
    for h in hits:
        if context == "section":
            sec = (h["uri"], h["section_offset"])
            if sec in seen:
                continue
            seen.add(sec)
            text, status = span(h, h["section_offset"], min(h["section_length"], max_bytes))
            # the section's bytes are not hashed; the passage's are, so check the passage
            _, status = span(h, h["offset"], h["length"], h["sha256"])
        else:
            text, status = span(h, h["offset"], h["length"], h["sha256"])
        out.append({**h, "text": text, "status": status})
    return out


def where(h) -> str:
    """file.md:12-14 for a local file; the revision URL for a web source."""
    if h["uri"].startswith("file:"):
        return f"{store.uri_path(h['uri']).name}:{h['line']}-{h['end_line']}"
    return h["uri"]


def label(h) -> str:
    return f"{where(h)} — {h['path']}"


def excerpts(hits) -> str:
    return "\n\n".join(f"[{i}] {label(h)}\n{h['text']}" for i, h in enumerate(hits, 1))


def rerank(chat, question: str, hits, k: int):
    scores = chat.complete_json(RERANK_SYSTEM, f"Question: {question}\n\nExcerpts:\n\n{excerpts(hits)}")
    scores = scores["scores"] if isinstance(scores, dict) else scores
    if len(scores) != len(hits):
        raise ValueError(f"rerank: {len(scores)} scores for {len(hits)} excerpts")
    order = sorted(range(len(hits)), key=lambda i: (-int(scores[i]), i))
    return [{**hits[i], "rerank": int(scores[i])} for i in order[:k]]


def answer(chat, question: str, hits, system: str = ANSWER_SYSTEM) -> str:
    usable = [h for h in hits if h["status"] == "ok"]
    if not usable:
        return "NOT FOUND: no passage could be read back from its file."
    return chat.complete(system, f"Excerpts:\n\n{excerpts(usable)}\n\nQuestion: {question}")


def search_queries(reply: str) -> list[str]:
    """The queries of a reply's SEARCH: lines; [] when the reply is an answer."""
    out = []
    for line in reply.splitlines():
        s = line.strip()
        if s[:7].upper() == "SEARCH:" and s[7:].strip():
            out.append(s[7:].strip())
    return out


async def follow(queries: list[str], hits, a):
    """Retrieve for each query, add the passages not already gathered (in order), cap the list at
    --max-passages and renumber the ranks."""
    have = {h["id"] for h in hits}
    for q in queries:
        new, _ = await retrieve(q, a.hop_k)
        for h in read_hits(new, a.context, a.max_bytes):
            if h["id"] not in have:
                have.add(h["id"])
                hits.append(h)
    hits = hits[:a.max_passages]
    for i, h in enumerate(hits, 1):
        h["rank"] = i
    return hits


async def ask_one(question: str, a, chat):
    before = chat.usage["calls"] if chat else 0
    hits, timing = await retrieve(question, a.k, a.rerank or a.k)
    linked = []
    if a.links:
        t = time.time()
        hits, linked = await expand_links(question, hits, a.links, a.link_max_degree)
        timing["links_ms"] = round((time.time() - t) * 1000, 1)
    hits = read_hits(hits, a.context, a.max_bytes)
    t = time.time()
    if a.rerank and chat:
        hits = rerank(chat, question, hits, a.k)
        timing["rerank_s"] = round(time.time() - t, 2)
    text = None
    searches = []
    if chat:
        t = time.time()
        if a.hops > 0:
            for _ in range(a.hops):
                reply = answer(chat, question, hits, ANSWER_SYSTEM_HOPS)
                queries = search_queries(reply)
                if not queries:
                    text = reply
                    break
                searches.append(queries)
                hits = await follow(queries, hits, a)
            else:
                # still asking after the last hop: answer from everything gathered, without the option
                text = answer(chat, question, hits)
        else:
            text = answer(chat, question, hits)
        timing["answer_s"] = round(time.time() - t, 2)
    calls = chat.usage["calls"] - before if chat else 0
    return {"question": question, "answer": text, "hits": hits, "timing": timing,
            "searches": searches, "calls": calls, "linked": linked}


def add_common(ap):
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--api-base", help="OpenAI-compatible base URL, e.g. https://api.openai.com/v1 (or DOCMEM_API_BASE)")
    ap.add_argument("--api-key", help="its key (or DOCMEM_API_KEY); never stored")
    ap.add_argument("--model", help="chat model name (or DOCMEM_MODEL)")
    ap.add_argument("-k", type=int, default=6, help="passages given to the model")
    ap.add_argument("--rerank", type=int, default=0, metavar="N",
                    help="retrieve N candidates and let the chat model pick the best k (one extra call)")
    ap.add_argument("--context", choices=("passage", "section"), default="passage",
                    help="give the model each hit's passage, or its whole enclosing section")
    ap.add_argument("--max-bytes", type=int, default=6000, help="cut a section read at this many bytes")
    ap.add_argument("--no-answer", action="store_true", help="retrieval only; no chat model needed")
    ap.add_argument("--links", type=int, default=0, metavar="N",
                    help="add up to N passages one link away from the hits (shared link targets, the "
                         "linked document itself), ranked by similarity; no LLM; 0 = off")
    ap.add_argument("--link-max-degree", type=int, default=50,
                    help="do not follow targets linked from more passages than this (too common)")
    ap.add_argument("--hops", type=int, default=0, metavar="N",
                    help="multi-hop: when the model asks for SEARCH:, retrieve for each query and ask again, "
                         "at most N times, then answer (at most N+1 answer calls); 0 = off")
    ap.add_argument("--hop-k", type=int, default=3, help="passages retrieved per follow-up SEARCH query")
    ap.add_argument("--max-passages", type=int, default=10,
                    help="cap on the passages given to the model in total (multi-hop)")


def open_db(a):
    if a.hops < 0 or a.hop_k < 1 or a.max_passages < 1:
        raise SystemExit("docmem: --hops must be 0 or more; --hop-k and --max-passages at least 1")
    store.configure(a.db, api_base=a.api_base or os.environ.get("DOCMEM_API_BASE"),
                    api_key=a.api_key or os.environ.get("DOCMEM_API_KEY"))
    from .llm import Chat

    return None if a.no_answer else Chat(a.api_base, a.api_key, a.model)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docmem ask", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(ap)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("question", nargs="+")
    a = ap.parse_args(argv)
    chat = open_db(a)
    r = asyncio.run(ask_one(" ".join(a.question), a, chat))
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return
    for i, qs in enumerate(r["searches"], 1):
        print(f"search {i}: " + " | ".join(qs))
    if r["searches"]:
        print()
    if r["answer"] is not None:
        print(r["answer"], "\n")
    for i, h in enumerate(r["hits"], 1):
        flag = "" if h["status"] == "ok" else f"  [{h['status'].upper()}: re-run scan | index]"
        if h.get("via"):
            flag += f"  [linked via {', '.join(h['via'][:3])}]"
        link = f"{store.uri_path(h['uri'])}#L{h['line']}" if h["uri"].startswith("file:") else h["uri"]
        print(f"[{i}] {h['distance']:.3f}  {label(h)}{flag}\n     {link}")
    print(f"\n{json.dumps(r['timing'])} calls={r['calls']}", file=sys.stderr)


def stats_main(argv=None):
    ap = argparse.ArgumentParser(prog="docmem stats")
    ap.add_argument("--db", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = store.configure(a.db)

    async def go():
        graph, vector = await store.engines()
        out = {"embed": cfg["embed"], "dims": cfg["dims"], "vector_backend": cfg["vector_backend"]}
        for t in ("DocFile", "DocSection", "DocPassage", "DocSummary", "DocEntity", "DocTopic", "DocGap"):
            nodes, _ = await graph.get_filtered_graph_data([{"type": [t]}], include_edges=False)
            out[t] = len(nodes)
        size = {}
        for sub in (a.db / "system" / "databases").iterdir():
            size[sub.name] = sum(f.stat().st_size for f in ([sub] if sub.is_file() else sub.rglob("*")) if f.is_file())
        out["bytes"] = size
        print(json.dumps(out, indent=1))

    asyncio.run(go())
