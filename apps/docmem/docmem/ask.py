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


def read_hits(hits, context: str, max_bytes: int):
    """Attach each hit's text. context "section" reads the enclosing section instead of the passage
    (once per section), cut at max_bytes."""
    seen = set()
    out = []
    for h in hits:
        if context == "section":
            span = (h["uri"], h["section_offset"])
            if span in seen:
                continue
            seen.add(span)
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


def answer(chat, question: str, hits) -> str:
    usable = [h for h in hits if h["status"] == "ok"]
    if not usable:
        return "NOT FOUND: no passage could be read back from its file."
    return chat.complete(ANSWER_SYSTEM, f"Excerpts:\n\n{excerpts(usable)}\n\nQuestion: {question}")


async def ask_one(question: str, a, chat):
    hits, timing = await retrieve(question, a.k, a.rerank or a.k)
    hits = read_hits(hits, a.context, a.max_bytes)
    t = time.time()
    if a.rerank and chat:
        hits = rerank(chat, question, hits, a.k)
        timing["rerank_s"] = round(time.time() - t, 2)
    text = None
    if chat:
        t = time.time()
        text = answer(chat, question, hits)
        timing["answer_s"] = round(time.time() - t, 2)
    return {"question": question, "answer": text, "hits": hits, "timing": timing}


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


def open_db(a):
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
    if r["answer"] is not None:
        print(r["answer"], "\n")
    for i, h in enumerate(r["hits"], 1):
        flag = "" if h["status"] == "ok" else f"  [{h['status'].upper()}: re-run scan | index]"
        link = f"{store.uri_path(h['uri'])}#L{h['line']}" if h["uri"].startswith("file:") else h["uri"]
        print(f"[{i}] {h['distance']:.3f}  {label(h)}{flag}\n     {link}")
    print(f"\n{json.dumps(r['timing'])}", file=sys.stderr)


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
