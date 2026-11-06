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


async def hits_for(ids, distance: dict):
    """Hit dicts (as retrieve makes them) for passage ids, in the given order."""
    graph, _ = await store.engines()
    nodes = {nd["id"]: store.node_props(nd) for nd in await graph.get_nodes(list(ids))}
    out = []
    for pid in ids:
        nd = nodes.get(pid)
        if nd is None:
            continue
        out.append({"distance": distance.get(pid), **{k_: nd[k_] for k_ in (
            "id", "path", "offset", "length", "line", "end_line", "sha256", "section_offset", "section_length")},
            "extractor": nd.get("extractor", ""), "doc_sha256": nd.get("doc_sha256", ""), "uri": nd["name"]})
    return out


async def best_of(question: str, cands, n: int):
    """The n candidate passages nearest the question, by their stored vectors (no re-embedding)."""
    import numpy as np

    _, vector = await store.engines()
    rows = await vector.retrieve(store.PASSAGES, list(cands), include_vector=True)
    q = np.asarray((await vector.embedding_engine.embed_text([question]))[0], dtype=np.float32)
    q /= np.linalg.norm(q) or 1
    scored = sorted((1 - float(q @ (v := np.asarray(r.payload["vector"], dtype=np.float32)) /
                               (np.linalg.norm(v) or 1)), str(r.id)) for r in rows)[:n]
    return await hits_for([pid for _, pid in scored], {pid: round(d, 4) for d, pid in scored})


async def hybrid(question: str, k: int, db):
    """Vector and keyword (BM25) results fused by reciprocal rank (1 / (60 + rank) summed), top k."""
    from .keywords import KeywordIndex

    vec, timing = await retrieve(question, 3 * k)
    t = time.time()
    kw = KeywordIndex(db).search(question, 3 * k)
    timing["keyword_ms"] = round((time.time() - t) * 1000, 1)
    score = {}
    for i, h in enumerate(vec, 1):
        score[h["id"]] = score.get(h["id"], 0) + 1 / (60 + i)
    for i, (pid, _) in enumerate(kw, 1):
        score[pid] = score.get(pid, 0) + 1 / (60 + i)
    top = sorted(score, key=lambda p: -score[p])[:k]
    by_id = {h["id"]: h for h in vec}
    missing = [p for p in top if p not in by_id]
    by_id.update({h["id"]: h for h in await hits_for(missing, {})})
    out = [by_id[p] for p in top if p in by_id]
    for i, h in enumerate(out, 1):
        h["rank"] = i
    return out, timing


async def expand_terms(question: str, hits, n: int, db, terms: int = 8):
    """Add up to n passages of OTHER documents that share the hits' rarest terms (BM25 over those
    terms), ranked by the question's similarity to their stored vectors; no LLM."""
    from .keywords import KeywordIndex

    kw = KeywordIndex(db)
    rare = kw.rare_terms([h["id"] for h in hits], terms)
    if not rare:
        return hits, []
    docs = {h["uri"] for h in hits}
    found = kw.search(" ".join(rare), 10 * n)
    cands = {pid: rare for pid, _ in found if pid not in {h["id"] for h in hits}}
    added = [h for h in await best_of(question, cands, 3 * n) if h["uri"] not in docs][:n]
    for h in added:
        h["via"] = ["terms: " + ", ".join(rare[:4])]
    out = hits + added
    for i, h in enumerate(out, 1):
        h["rank"] = i
    return out, [h["via"] for h in added]


async def expand_links(question: str, hits, n: int, max_degree: int):
    """Add up to n passages reached through links from the hits, ranked by the question's similarity to
    their stored vectors. Targets linked from more than max_degree passages (e.g. a country) are too
    common to say anything and are not followed."""
    graph, _ = await store.engines()
    via, degree = await link_neighbours(graph, [h["id"] for h in hits])
    common = {t for t, d in degree.items() if d > max_degree}
    cands = {pid: ts for pid, ts in via.items() if any(t not in common for t in ts)}
    if not cands:
        return hits, []
    added = await best_of(question, cands, n)
    for h in added:
        h["via"] = [t for t in cands[h["id"]] if t not in common]
    out = hits + added
    for i, h in enumerate(out, 1):
        h["rank"] = i
    return out, [h["via"] for h in added]


async def link_notes(hits) -> list[str]:
    """Sentences naming how the excerpts are linked, from the link graph (no LLM): excerpts of different
    documents that link to the same target, and an excerpt linking to the document another one is from."""
    graph, _ = await store.engines()
    ids = [h["id"] for h in hits]
    if not ids:
        return []
    rows = await graph.query(
        "MATCH (p:Node)-[r:EDGE]->(t:Node) WHERE p.id IN $ids AND r.relationship_name = 'links_to' "
        "RETURN p.id, t.id, t.name", {"ids": ids})
    pos = {h["id"]: i for i, h in enumerate(hits, 1)}
    doc_of = {i: h["uri"] for i, h in enumerate(hits, 1)}
    by_target, names = {}, {}
    for pid, tid, name in rows:
        by_target.setdefault(tid, set()).add(pos[pid])
        names[tid] = name
    notes = []
    for tid, ps in by_target.items():
        if len({doc_of[i] for i in ps}) > 1:
            ps = sorted(ps)
            notes.append(f"{', '.join(f'[{i}]' for i in ps[:-1])} and [{ps[-1]}] all mention {names[tid]}."
                         if len(ps) > 2 else f"[{ps[0]}] and [{ps[1]}] both mention {names[tid]}.")
    # an excerpt linking to the subject of another excerpt's document
    subj = await graph.query(
        "MATCH (f:Node)-[r:EDGE]->(t:Node) WHERE t.id IN $tids AND r.relationship_name = 'is_target' "
        "RETURN f.name, t.id", {"tids": list(by_target)}) if by_target else []
    for furi, tid in subj:
        for j, h in enumerate(hits, 1):
            if h["uri"] == furi:
                for i in sorted(by_target[tid]):
                    if doc_of[i] != furi:
                        notes.append(f"[{i}] mentions {names[tid]}, which is what [{j}] is about.")
                break
    return notes


async def notes_for(hits, a):
    """--link-notes: the notes for the excerpts the model will see (stale ones are not shown)."""
    return await link_notes([h for h in hits if h["status"] == "ok"]) if a.link_notes else None


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


def answer(chat, question: str, hits, system: str = ANSWER_SYSTEM, notes=None) -> str:
    usable = [h for h in hits if h["status"] == "ok"]
    if not usable:
        return "NOT FOUND: no passage could be read back from its file."
    extra = ("\n\nHow the excerpts are linked:\n" + "\n".join(notes)) if notes else ""
    return chat.complete(system, f"Excerpts:\n\n{excerpts(usable)}{extra}\n\nQuestion: {question}")


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
    if a.keywords:
        hits, timing = await hybrid(question, a.rerank or a.k, a.db)
    else:
        hits, timing = await retrieve(question, a.k, a.rerank or a.k)
    linked = []
    if a.terms:
        t = time.time()
        hits, by_terms = await expand_terms(question, hits, a.terms, a.db)
        linked += by_terms
        timing["terms_ms"] = round((time.time() - t) * 1000, 1)
    if a.links:
        t = time.time()
        hits, by_links = await expand_links(question, hits, a.links, a.link_max_degree)
        linked += by_links
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
                reply = answer(chat, question, hits, ANSWER_SYSTEM_HOPS, await notes_for(hits, a))
                queries = search_queries(reply)
                if not queries:
                    text = reply
                    break
                searches.append(queries)
                hits = await follow(queries, hits, a)
            else:
                # still asking after the last hop: answer from everything gathered, without the option
                text = answer(chat, question, hits, notes=await notes_for(hits, a))
        else:
            text = answer(chat, question, hits, notes=await notes_for(hits, a))
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
    ap.add_argument("--keywords", action="store_true",
                    help="hybrid retrieval: fuse vector and keyword (BM25) results by reciprocal rank")
    ap.add_argument("--terms", type=int, default=0, metavar="N",
                    help="add up to N passages of other documents sharing the hits' rarest terms; no LLM")
    ap.add_argument("--links", type=int, default=0, metavar="N",
                    help="add up to N passages one link away from the hits (shared link targets, the "
                         "linked document itself), ranked by similarity; no LLM; 0 = off")
    ap.add_argument("--link-max-degree", type=int, default=50,
                    help="do not follow targets linked from more passages than this (too common)")
    ap.add_argument("--link-notes", action="store_true",
                    help="tell the model how the excerpts are linked (shared link targets), from the graph")
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
        dist = f"{h['distance']:.3f}" if h.get('distance') is not None else "  kw "
        print(f"[{i}] {dist}  {label(h)}{flag}\n     {link}")
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
