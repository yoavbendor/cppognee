"""Alignment between the prose KB and the SysML model DB: which model elements a prose document is about.

    docmem align --kb KBDB --model MODELDB --model-name apollo11 --out ALIGNDIR
    docmem ask --db KBDB --with-model ALIGNDIR --model-db MODELDB "question"
    docmem ask --db MODELDB --with-prose ALIGNDIR --prose-db KBDB "question"

The alignment is a third store: ALIGNDIR holds only ``alignment.jsonl`` (one row per element and document
pair, with the evidence) and ``meta.json`` (inputs, thresholds, counts). It can be deleted or rebuilt; neither
the prose KB nor the model DB is written. Nothing is used from it unless ``--with-model`` / ``--with-prose``
is given.

No LLM. A model element is scored against each prose document on three signals, each in [0, 1]:
  name     the element's name tokens against the document title and its section headings (Jaccard)
  keyword  BM25 of the name in the KB keyword index: the document's best passage, divided by the best BM25 score
           any name reaches in this run (one scale for all elements)
  vector   the element's centroid (mean of its passage vectors) against the document's passage vectors: cosine,
           rescaled so the mean element-passage pair scores 0 and identical text 1
score = sum(weight * signal). The top documents per element above the threshold are kept, with the best
prose passages of each (by vector similarity).

Cognee keeps one database per process, so the two stores are read by child processes
(``python -m docmem.align export-model|export-kb|spans|prose``); the matching itself reads only their files.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = {"name": 0.5, "keyword": 0.2, "vector": 0.3}
TOP = 3
BM25_K = 50
THRESHOLD = 0.35  # chosen from the score distribution (see meta.json): below ~0.3 most pairs share one word
DECAY = 0.8  # score kept per model relation a document travels along (--propagate)
MAX_HOPS = 2  # longest chain of model relations a propagated document may come along
KEEP_PROPAGATED = 3  # documents kept per element that inherits them
SECTION_SEP = " › "
_LABEL = re.compile(r"^(?P<kind>.*?)\s+(?P<name>'(?:[^'\\]|\\.)*'|\S+)$")  # "part def Foo", "requirement 'clr-R1'"
_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")


# -- pure functions ---------------------------------------------------------------------------------------------

def name_tokens(text: str) -> tuple[str, ...]:
    """Lowercase words of a name: split at CamelCase, letter/digit boundaries, underscores, hyphens and quotes.
    "ApolloLunarModule" -> (apollo, lunar, module); "Apollo11Mission" -> (apollo, 11, mission); single letters
    are dropped ("clr-R114" -> (clr, 114))."""
    spaced = _BOUNDARY.sub(" ", text.strip().strip("'\""))
    return tuple(t.lower() for t in re.split(r"[^0-9A-Za-z]+", spaced) if len(t) >= 2)


def jaccard(a, b) -> float:
    """Jaccard similarity of two token collections; 0 when both are empty."""
    a, b = set(a), set(b)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def parse_label(label: str) -> tuple[str, str] | None:
    """(kind, name) of one section label, e.g. "part def Foo" -> ("part def", "Foo"); None when it has no name."""
    m = _LABEL.match(label.strip())
    if not m or not m.group("kind"):
        return None
    return m.group("kind"), m.group("name")


def combined_score(signals: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted sum of the signals; a signal that is absent counts as 0."""
    return sum(w * signals.get(k, 0.0) for k, w in weights.items())


def select_pairs(scores: dict, top: int, threshold: float) -> list:
    """The best `top` keys of scores ({doc: score}) whose score is at least `threshold`, best first."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return [(k, s) for k, s in ranked if s >= threshold][:top]


def quantiles(values, points=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)) -> dict[str, float]:
    if not len(values):
        return {}
    arr = np.sort(np.asarray(values, dtype=np.float64))
    return {f"p{int(p * 100)}": round(float(np.quantile(arr, p)), 4) for p in points}


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(n == 0, 1, n)


def align_rows(elements: list[dict], docs: list[dict], kb_vecs: np.ndarray, passage_doc: np.ndarray,
               passage_ids: list[str], model_vecs: np.ndarray, bm25: dict, weights: dict, top: int,
               threshold: float, passages_per_doc: int = 2):
    """Score every element against every prose document.

    elements: {"name_tokens", "query", "vec_rows": [row indices of model_vecs]}
    docs: {"title_tokens", "heading_tokens": [token tuples]}; passage_doc: the doc index of each KB passage row.
    bm25: {query: [(passage id, score), ...]}. Returns (rows, all_top_scores) where rows is a list of
    (element index, doc index, score, signals, passage ids) for the kept pairs, the best score of each element, and
    the BM25 scale and vector baseline used."""
    pid_row = {pid: i for i, pid in enumerate(passage_ids)}
    doc_rows = [np.flatnonzero(passage_doc == d) for d in range(len(docs))]
    centroids = np.zeros((len(elements), kb_vecs.shape[1]), dtype=np.float32)
    for e, el in enumerate(elements):
        if el["vec_rows"]:
            centroids[e] = _normalize(model_vecs[el["vec_rows"]].mean(axis=0))
    sims = _normalize(kb_vecs) @ centroids.T if kb_vecs.size else None  # passages x elements

    # BM25 scale: the best score any element's name reaches (one scale for the run, so an element's keyword
    # signal does not depend on its own best match). Vector baseline: the mean element-passage cosine.
    kw_scale = max((hits[0][1] for hits in bm25.values() if hits), default=0.0)
    has_vec = [bool(el["vec_rows"]) for el in elements]
    baseline = float(sims[:, has_vec].mean()) if sims is not None and any(has_vec) else 0.0

    rows, best = [], []
    for e, el in enumerate(elements):
        # keyword: the best BM25 passage of each document
        kw_hits = [(p, s) for p, s in bm25.get(el["query"], []) if p in pid_row]
        kw_doc: dict[int, float] = {}
        for pid, s in kw_hits:
            d = int(passage_doc[pid_row[pid]])
            if d >= 0 and kw_scale > 0:
                kw_doc[d] = max(kw_doc.get(d, 0.0), s / kw_scale)
        scores = {}
        for d, doc in enumerate(docs):
            signals = {"name": max([jaccard(el["name_tokens"], doc["title_tokens"])] +
                                   [jaccard(el["name_tokens"], h) for h in doc["heading_tokens"]])}
            signals["keyword"] = kw_doc.get(d, 0.0)
            pids = []
            if sims is not None and len(doc_rows[d]):
                col = sims[doc_rows[d], e]
                signals["vector"] = float(min(1.0, max(0.0, (col.max() - baseline) / (1 - baseline))))
                order = np.argsort(-col, kind="stable")[:passages_per_doc]
                pids = [passage_ids[int(doc_rows[d][i])] for i in order]
            else:
                signals["vector"] = 0.0
            signals = {k: round(float(v), 4) for k, v in signals.items()}
            scores[d] = (round(combined_score(signals, weights), 4), signals, pids)
        best.append(max((v[0] for v in scores.values()), default=0.0))
        for d, _ in select_pairs({d: v[0] for d, v in scores.items()}, top, threshold):
            _, signals, pids = scores[d]
            rows.append((e, d, scores[d][0], signals, pids))
    return rows, best, {"bm25_scale": round(kw_scale, 4), "vector_baseline": round(baseline, 4)}


def model_links(elements: list[dict], relations: list[dict]) -> list[tuple[int, int, str]]:
    """(inheritor, source, rule) triples: the inheritor takes the documents of the source element.
      satisfied by  a requirement takes its satisfier's documents (r_satisfy runs satisfier -> requirement)
      typed by      a usage takes its definition's documents; specializes: a specialization takes its general's
      usage of      a requirement definition takes its usages' documents (the reverse of typed by)
      owned by      an element takes its owner's documents (the owner is the element named by its qualified name's prefix)
    relations: {"rel": "r_satisfy" | "r_typed_by" | "r_specializes", "src": section id, "dst": section id}."""
    index = {el["section_id"]: i for i, el in enumerate(elements)}
    by_qname = defaultdict(list)
    for i, el in enumerate(elements):
        by_qname[el["qname"]].append(i)
    links = []
    for r in relations:
        s, d = index.get(r["src"]), index.get(r["dst"])
        if s is None or d is None:
            continue
        if r["rel"] == "r_satisfy":
            links.append((d, s, "satisfied by"))
        elif r["rel"] == "r_typed_by":
            links.append((s, d, "typed by"))
            if elements[d]["metaclass"] == "RequirementDefinition":
                links.append((d, s, "usage of"))
        elif r["rel"] == "r_specializes":
            links.append((s, d, "specializes"))
    for i, el in enumerate(elements):
        for o in by_qname.get(el["qname"].rpartition("::")[0], []):
            links.append((i, o, "owned by"))
    return links


def propagate(direct: dict, links: list[tuple[int, int, str]], decay: float = DECAY,
              max_hops: int = MAX_HOPS, keep: int = KEEP_PROPAGATED) -> dict:
    """The documents elements inherit from related elements, for elements with no direct row.

    direct: {element: {doc: {"score", "signals", "passages"}}} (the alignment's rows). Round k takes the documents the
    sources hold after round k-1, so a chain has at most max_hops links, and each link multiplies the score by decay:
    a propagated score is below the direct score it comes from. An element with direct rows keeps them and takes
    nothing. Returns {element: {doc: entry}} for the elements that got documents, at most keep each; an entry has the
    score, the passages of the direct row it comes from, origin (that element) and path ([{rule, from, to}])."""
    best = {e: {d: {**v, "origin": e, "path": []} for d, v in docs.items()} for e, docs in direct.items()}
    inherits = defaultdict(list)
    for inheritor, source, rule in links:
        if inheritor != source and inheritor not in direct:
            inherits[source].append((inheritor, rule))
    for _ in range(max_hops):
        held = {e: sorted(docs.items(), key=lambda kv: (-kv[1]["score"], kv[0]))[:keep]
                for e, docs in best.items()}
        for source, items in held.items():
            for inheritor, rule in inherits.get(source, []):
                for d, v in items:
                    score = round(v["score"] * decay, 4)
                    cur = best.setdefault(inheritor, {}).get(d)
                    if cur is None or score > cur["score"]:
                        best[inheritor][d] = {"score": score, "passages": v["passages"], "origin": v["origin"],
                                              "path": v["path"] + [{"rule": rule, "from": source, "to": inheritor}]}
    out = {}
    for e, docs in best.items():
        if e not in direct and docs:
            out[e] = dict(sorted(docs.items(), key=lambda kv: (-kv[1]["score"], kv[0]))[:keep])
    return out


# -- child processes: read one store each ------------------------------------------------------------------------

def _child(args: list[str], payload=None) -> str:
    """Run `python -m docmem.align ARGS` (its own process, its own cognee configuration). Its stdout is returned."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PKG_ROOT), env.get("PYTHONPATH")]))
    r = subprocess.run([sys.executable, "-m", "docmem.align", *map(str, args)],
                       input=None if payload is None else json.dumps(payload), capture_output=True, text=True,
                       env=env)
    if r.returncode:
        raise SystemExit(f"docmem align: {args[0]} failed:\n{r.stderr[-3000:]}")
    return r.stdout


def _child_json(args: list[str], payload=None):
    return json.loads(_child(args, payload))


def _load_cfg(db: Path) -> dict:
    return json.loads((Path(db) / "docmem.json").read_text())


RELATIONS = ("r_satisfy", "r_typed_by", "r_specializes")  # the model relations propagation follows


async def _model_elements(graph, vector, model_name: str):
    """The model's elements (one per labelled section of model_name, with the row indices of their passage vectors),
    the count of unparsed sections, the passage vector matrix, and the model relations among the elements."""
    from . import store
    from .ontology import nodes_of_type
    from .sysml_resolve import _relation_rows, metaclass_for

    secs = [s for s in await nodes_of_type(graph, "DocSection")
            if s.get("path", "").split(SECTION_SEP)[1:2] == [model_name]]
    elements, unparsed = [], 0
    for s in secs:
        labels = s["path"].split(SECTION_SEP)[2:]
        parsed = [parse_label(x) for x in labels]
        if not labels:
            continue
        if any(p is None for p in parsed):
            unparsed += 1
            continue
        elements.append({"section_id": str(s["id"]), "qname": "::".join(p[1].strip("'") for p in parsed),
                         "kind": parsed[-1][0], "metaclass": metaclass_for(parsed[-1][0])[0] or "",
                         "name": parsed[-1][1].strip("'"), "uri": s["name"],
                         "line": s.get("line"), "path": s["path"], "passages": []})
    sids = [e["section_id"] for e in elements]
    pairs = await graph.query(
        "MATCH (p:Node)-[r:EDGE]->(s:Node) WHERE r.relationship_name = 'section' AND s.id IN $sids "
        "RETURN p.id, s.id", {"sids": sids}) if sids else []
    by_sid = {e["section_id"]: e for e in elements}
    pid_list = []
    for pid, sid in pairs:
        by_sid[str(sid)]["passages"].append(str(pid))
        pid_list.append(str(pid))
    vecs = {str(r.id): r.payload["vector"] for r in
            (await vector.retrieve(store.PASSAGES, sorted(set(pid_list)), include_vector=True) if pid_list else [])}
    mat, index = [], {}
    for pid in sorted(vecs):
        index[pid] = len(mat)
        mat.append(vecs[pid])
    for e in elements:
        e["vec_rows"] = [index[p] for p in e.pop("passages") if p in index]
    relations = []
    sid_set = set(sids)
    for rel in RELATIONS:
        for a, b, _, _ in await _relation_rows(graph, rel):
            if a in sid_set and b in sid_set:
                relations.append({"rel": rel, "src": a, "dst": b})
    return elements, unparsed, np.asarray(mat, dtype=np.float32).reshape(len(mat), -1), relations


def _export_model(db: Path, model_name: str, out: Path) -> None:
    import asyncio

    from . import store

    store.configure(db)

    async def go():
        graph, vector = await store.engines()
        elements, unparsed, mat, relations = await _model_elements(graph, vector, model_name)
        np.save(out / "model_vectors.npy", mat)
        (out / "model.json").write_text(json.dumps({
            "model_name": model_name, "elements": elements, "unparsed_sections": unparsed,
            "relations": relations}, ensure_ascii=False))
        print(json.dumps({"elements": len(elements), "unparsed": unparsed, "passages": len(mat),
                          "relations": len(relations)}))

    asyncio.run(go())


def _export_kb(db: Path, queries: list[str], out: Path) -> None:
    import asyncio

    from . import store
    from .keywords import KeywordIndex
    from .ontology import nodes_of_type

    store.configure(db)

    async def go():
        graph, vector = await store.engines()
        files = await nodes_of_type(graph, "DocFile")
        docs = {f["name"]: {"uri": f["name"], "title": f["title"], "url": f.get("url", ""), "headings": []}
                for f in files}
        for s in await nodes_of_type(graph, "DocSection"):
            parts = s.get("path", "").split(SECTION_SEP)
            if len(parts) > 1 and s["name"] in docs:
                docs[s["name"]]["headings"].append(parts[-1])
        passages = [{"id": str(p["id"]), "uri": p["name"], "path": p["path"], "line": p["line"],
                     "end_line": p["end_line"]} for p in await nodes_of_type(graph, "DocPassage")]
        res = await vector.retrieve(store.PASSAGES, [p["id"] for p in passages], include_vector=True)
        vecs = {str(r.id): r.payload["vector"] for r in res}
        passages = [p for p in passages if p["id"] in vecs]
        kb = KeywordIndex(Path(db))
        bm25 = {q: [[pid, s] for pid, s in kb.search(q, BM25_K)] for q in queries}
        np.save(out / "kb_vectors.npy", np.asarray([vecs[p["id"]] for p in passages], dtype=np.float32))
        (out / "kb.json").write_text(json.dumps({"docs": list(docs.values()), "passages": passages},
                                                ensure_ascii=False))
        (out / "bm25.json").write_text(json.dumps(bm25, ensure_ascii=False))
        print(json.dumps({"documents": len(docs), "passages": len(passages), "queries": len(queries)}))

    asyncio.run(go())


async def _read_sections(graph, section_ids: list[str], max_bytes: int, db: Path) -> list[dict]:
    """The model sections' text: each section's bytes in its .sysml file (cut at max_bytes)."""
    from . import store

    nodes = {str(n["id"]): store.node_props(n) for n in await graph.get_nodes(section_ids)}
    out = []
    for sid in section_ids:
        n = nodes.get(sid)
        if n is None:
            raise SystemExit(f"docmem align: no section {sid} in {db}")
        full, status = store.read_span(n["name"], int(n["offset"]), int(n["length"]))
        text = full.encode("utf-8")[:max_bytes].decode("utf-8", "replace")
        out.append({"id": sid, "uri": n["name"], "path": n["path"], "line": n["line"],
                    "end_line": int(n["line"]) + full.count("\n"), "text": text, "status": status})
    return out


def _spans(db: Path, section_ids: list[str], max_bytes: int) -> list[dict]:
    import asyncio

    from . import store

    store.configure(db)

    async def go():
        graph, _ = await store.engines()
        return await _read_sections(graph, section_ids, max_bytes, db)

    return asyncio.run(go())


def _query_model(db: Path, model_name: str, question: str, n: int, max_bytes: int) -> list[dict]:
    """The n model elements nearest the question: the cosine between the question's vector and each element's centroid
    (the normalized mean of its passage vectors, as align_rows does). Each comes with its text and score; no threshold."""
    import asyncio

    from . import store

    store.configure(db)

    async def go():
        graph, vector = await store.engines()
        elements, _, mat, _ = await _model_elements(graph, vector, model_name)
        have = [e for e in elements if e["vec_rows"]]
        if not have:
            return []
        q = np.asarray((await vector.embedding_engine.embed_text([question]))[0], dtype=np.float32)
        q /= np.linalg.norm(q) or 1
        cent = _normalize(np.stack([mat[e["vec_rows"]].mean(axis=0) for e in have]))
        sims = cent @ q
        order = np.argsort(-sims, kind="stable")[:n]
        picks = [have[i] for i in order]
        spans = await _read_sections(graph, [e["section_id"] for e in picks], max_bytes, db)
        return [{**sp, "score": round(float(sims[i]), 4)} for sp, i in zip(spans, order)]

    return asyncio.run(go())


def _query_prose(db: Path, question: str, n: int, max_bytes: int) -> list[dict]:
    """The n prose passages nearest the question (ask's retrieval), with their text read as ask reads them."""
    import asyncio

    from . import ask, store

    store.configure(db)

    async def go():
        hits, _ = await ask.retrieve(question, n)
        return ask.read_hits(hits, "passage", max_bytes)

    return asyncio.run(go())


def _prose(db: Path, passage_ids: list[str], max_bytes: int) -> list[dict]:
    """The prose passages' hits with their text, read the way ask reads them."""
    import asyncio

    from . import ask, store

    store.configure(db)

    async def go():
        hits = await ask.hits_for(passage_ids, {})
        return ask.read_hits(hits, "passage", max_bytes)

    return asyncio.run(go())


def child_main(argv: list[str]) -> None:
    """Entry point of the child processes (docmem.align export-model, export-kb, spans, prose)."""
    ap = argparse.ArgumentParser(prog="docmem.align")
    sub = ap.add_subparsers(dest="cmd", required=True)
    em = sub.add_parser("export-model")
    em.add_argument("--db", type=Path, required=True)
    em.add_argument("--model-name", required=True)
    em.add_argument("--out", type=Path, required=True)
    ek = sub.add_parser("export-kb")
    ek.add_argument("--db", type=Path, required=True)
    ek.add_argument("--queries", type=Path, required=True)
    ek.add_argument("--out", type=Path, required=True)
    sp = sub.add_parser("spans", help="model sections (ids on stdin) -> their text")
    sp.add_argument("--db", type=Path, required=True)
    pr = sub.add_parser("prose", help="prose passages (ids on stdin) -> hits with their text")
    pr.add_argument("--db", type=Path, required=True)
    qm = sub.add_parser("query-model", help="question (stdin) -> the nearest model elements with their text")
    qm.add_argument("--db", type=Path, required=True)
    qm.add_argument("--model-name", required=True)
    qp = sub.add_parser("query-prose", help="question (stdin) -> the nearest prose passages with their text")
    qp.add_argument("--db", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "export-model":
        _export_model(a.db, a.model_name, a.out)
    elif a.cmd == "export-kb":
        _export_kb(a.db, json.loads(a.queries.read_text()), a.out)
    else:
        req = json.loads(sys.stdin.read())
        if a.cmd == "spans":
            res = _spans(a.db, req["section_ids"], req["max_bytes"])
        elif a.cmd == "query-model":
            res = _query_model(a.db, a.model_name, req["question"], req["n"], req["max_bytes"])
        elif a.cmd == "query-prose":
            res = _query_prose(a.db, req["question"], req["n"], req["max_bytes"])
        else:
            res = _prose(a.db, req["passage_ids"], req["max_bytes"])
        print(json.dumps(res, ensure_ascii=False))


# -- the alignment -----------------------------------------------------------------------------------------------

def align(kb: Path, model: Path, model_name: str, out: Path, weights: dict, top: int, threshold: float,
          bm25_k: int = BM25_K, propagation: bool = True) -> dict:
    kb_cfg, model_cfg = _load_cfg(kb), _load_cfg(model)
    if kb_cfg["embed"] != model_cfg["embed"] or kb_cfg["dims"] != model_cfg["dims"]:
        raise SystemExit(f"docmem align: the KB uses {kb_cfg['embed']} and the model {model_cfg['embed']}; "
                         "vectors from different models cannot be compared")
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".work-", dir=out))
    try:
        _child(["export-model", "--db", model, "--model-name", model_name, "--out", work])
        mj = json.loads((work / "model.json").read_text())
        elements = mj["elements"]
        queries = sorted({" ".join(name_tokens(e["name"])) for e in elements} - {""})
        (work / "queries.json").write_text(json.dumps(queries))
        _child(["export-kb", "--db", kb, "--queries", work / "queries.json", "--out", work])
        kj = json.loads((work / "kb.json").read_text())
        kb_vecs = np.load(work / "kb_vectors.npy")
        bm25 = {q: [tuple(x) for x in v] for q, v in json.loads((work / "bm25.json").read_text()).items()}
        model_vecs = np.load(work / "model_vectors.npy")
        doc_index = {d["uri"]: i for i, d in enumerate(kj["docs"])}
        docs = [{"title_tokens": name_tokens(d["title"]),
                 "heading_tokens": [name_tokens(h) for h in d["headings"]]} for d in kj["docs"]]
        passage_ids = [p["id"] for p in kj["passages"]]
        passage_doc = np.asarray([doc_index.get(p["uri"], -1) for p in kj["passages"]], dtype=np.int64)
        els = [{"name_tokens": name_tokens(e["name"]), "query": " ".join(name_tokens(e["name"])),
                "vec_rows": e["vec_rows"]} for e in elements]
        rows, best, scale = align_rows(els, docs, kb_vecs, passage_doc, passage_ids, model_vecs, bm25, weights,
                                       top, threshold)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    direct = defaultdict(dict)
    for e, d, score, signals, pids in rows:
        direct[e][d] = {"score": score, "signals": signals, "passages": pids}
    prop = propagate(direct, model_links(elements, mj["relations"])) if propagation else {}

    def line(e, d, entry, basis):
        el, doc = elements[e], kj["docs"][d]
        out_ = {"element": el["qname"], "section_id": el["section_id"], "kind": el["kind"],
                "metaclass": el["metaclass"], "model_uri": el["uri"], "model_line": el["line"],
                "model_path": el["path"], "doc": doc["title"], "uri": doc["uri"], "url": doc["url"],
                "score": entry["score"], "signals": entry.get("signals", {}), "passages": entry["passages"],
                "basis": basis}
        if basis == "propagated":
            out_["origin"] = elements[entry["origin"]]["qname"]
            out_["path"] = [{"rule": x["rule"], "from": elements[x["from"]]["qname"],
                             "to": elements[x["to"]]["qname"]} for x in entry["path"]]
            out_["hops"] = len(entry["path"])
            out_["signals"] = {}
        return out_

    lines = [line(e, d, {"signals": signals, "passages": pids, "score": score}, "direct")
             for e, d, score, signals, pids in rows]
    for e in sorted(prop):
        for d, entry in prop[e].items():
            lines.append(line(e, d, entry, "propagated"))
    direct_ids = {elements[e]["section_id"] for e in direct}
    prop_ids = {elements[e]["section_id"] for e in prop}
    by_kind = defaultdict(lambda: {"elements": 0, "direct": set(), "total": set()})
    for el in elements:
        k = by_kind[el["kind"]]
        k["elements"] += 1
        if el["section_id"] in direct_ids:
            k["direct"].add(el["section_id"])
        if el["section_id"] in direct_ids | prop_ids:
            k["total"].add(el["section_id"])
    (out / "alignment.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    meta = {
        "kb": str(kb), "model": str(model), "model_name": model_name, "embed": kb_cfg["embed"],
        "dims": kb_cfg["dims"], "weights": weights, "top": top, "threshold": threshold, "bm25_k": bm25_k,
        "propagation": {"decay": DECAY, "max_hops": MAX_HOPS, "keep": KEEP_PROPAGATED} if propagation else None,
        "elements": len(elements), "unparsed_sections": mj["unparsed_sections"],
        "elements_with_passages": sum(1 for e in elements if e["vec_rows"]),
        "aligned_elements": len(direct_ids), "propagated_elements": len(prop_ids - direct_ids),
        "aligned_elements_total": len(direct_ids | prop_ids),
        "rows": len(lines), "propagated_rows": sum(1 for x in lines if x["basis"] == "propagated"),
        "documents": len(kj["docs"]), "kb_passages": len(passage_ids),
        "top_score_quantiles": quantiles(best),
        "elements_with_top_score_at_least": {str(t): sum(1 for b in best if b >= t)
                                             for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)},
        "by_kind": {k: {"elements": v["elements"], "aligned": len(v["direct"]), "aligned_total": len(v["total"])}
                    for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1]["elements"])},
        **scale, "seconds": round(time.time() - started, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    return {"meta": meta, "rows": lines}


def load_meta(aligndir: Path) -> dict:
    path = Path(aligndir) / "meta.json"
    if not path.exists():
        raise SystemExit(f"docmem: no alignment at {aligndir} (run docmem align first)")
    return json.loads(path.read_text())


def load_rows(aligndir: Path) -> list[dict]:
    path = Path(aligndir) / "alignment.jsonl"
    if not path.exists():
        raise SystemExit(f"docmem: no alignment at {aligndir} (run docmem align first)")
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def align_main(argv=None):
    ap = argparse.ArgumentParser(prog="docmem align", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb", type=Path, required=True, help="the prose KB")
    ap.add_argument("--model", type=Path, required=True, help="the model DB")
    ap.add_argument("--model-name", required=True, help="the model's name in its section paths, e.g. apollo11")
    ap.add_argument("--out", type=Path, required=True, help="the alignment directory (rebuilt each run)")
    ap.add_argument("--weights", default=",".join(f"{k}={v}" for k, v in WEIGHTS.items()),
                    help="signal weights, name=..,keyword=..,vector=..")
    ap.add_argument("--top", type=int, default=TOP, help="prose documents kept per element")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="minimum score of a kept pair")
    ap.add_argument("--propagate", action=argparse.BooleanOptionalAction, default=True,
                    help="elements with no direct alignment inherit their model relations' documents (satisfied "
                         "by, typed by, specializes, usage of, owned by): decay 0.8 per link, at most 2 links, "
                         "3 documents each (default on)")
    a = ap.parse_args(argv)
    weights = {k.strip(): float(v) for k, v in (x.split("=") for x in a.weights.split(","))}
    if set(weights) != set(WEIGHTS):
        raise SystemExit(f"docmem align --weights: name the signals {', '.join(WEIGHTS)}")
    r = align(a.kb, a.model, a.model_name, a.out, weights, a.top, a.threshold, propagation=a.propagate)
    m = r["meta"]
    print(f"elements {m['elements']} ({m['elements_with_passages']} with passages), aligned directly "
          f"{m['aligned_elements']}, propagated to {m['propagated_elements']} more, rows {m['rows']} "
          f"({m['propagated_rows']} propagated), documents {m['documents']}, {m['seconds']}s")
    print("top score per element:", json.dumps(m["top_score_quantiles"]))
    print("elements whose top score reaches:", json.dumps(m["elements_with_top_score_at_least"]))
    print("by kind (elements / aligned directly / aligned with propagation):")
    for k, v in list(m["by_kind"].items())[:25]:
        print(f"  {k:24s} {v['elements']:5d} {v['aligned']:5d} {v['aligned_total']:5d}")


# -- ask: the aligned items for a question's hits ----------------------------------------------------------------

def _best_rows(rows: list[dict], n: int, key) -> list[dict]:
    """The rows by score, best first, one per key, at most n."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: -r["score"]):
        k = key(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out[:n]


def aligned_items(question: str, hits: list[dict], a) -> list[dict]:
    """Items to add to a question's context (ask --with-model / --with-prose), read through a child process on the
    other DB. Items are marked with "aligned": "model" or "prose", and "found_by": "alignment" (the stored rows of
    the hits' documents, or of the hits' elements) or "query" (the question's nearest elements or passages, unless
    --no-query-align). An item the query also finds is "alignment+query". At most a.aligned of each kind per source.
    Nothing already in the context is added twice."""
    taken = {h["id"]: None for h in hits}  # id -> the added item (None: already a hit)
    items = []

    def add(item: dict, found_by: str) -> None:
        if item["id"] in taken:
            if taken[item["id"]] is not None and found_by == "query":
                taken[item["id"]]["found_by"] = "alignment+query"
            return
        item["found_by"] = found_by
        taken[item["id"]] = item
        items.append(item)

    if a.with_model:
        rows = load_rows(a.with_model)
        by_uri = defaultdict(list)
        for r in rows:
            by_uri[r["uri"]].append(r)
        cands = [r for h in hits if h.get("status") == "ok" for r in by_uri.get(h["uri"], [])]
        pick = _best_rows(cands, a.aligned, lambda r: r["section_id"])
        if pick:
            spans = _child_json(["spans", "--db", a.model_db], {"section_ids": [r["section_id"] for r in pick],
                                                                  "max_bytes": a.max_bytes})
            for r, sp in zip(pick, spans):
                add({"id": sp["id"], "uri": sp["uri"], "path": sp["path"], "line": sp["line"],
                     "end_line": sp["end_line"], "text": sp["text"], "status": sp["status"],
                     "distance": None, "aligned": "model",
                     "via": [f"aligned to {r['doc']} (score {r['score']})"]}, "alignment")
        if not a.no_query_align:
            model_name = load_meta(a.with_model)["model_name"]
            for g in _child_json(["query-model", "--db", a.model_db, "--model-name", model_name],
                                 {"question": question, "n": a.aligned, "max_bytes": a.max_bytes}):
                score = g.pop("score")
                add({**g, "distance": None, "aligned": "model", "via": ["query"], "query_score": score}, "query")
    if a.with_prose:
        rows = load_rows(a.with_prose)
        by_key = defaultdict(list)
        for r in rows:
            by_key[(r["model_uri"], r["model_path"])].append(r)
        cands = [r for h in hits for r in by_key.get((h["uri"], h["path"]), []) if r["passages"]]
        pick = _best_rows(cands, a.aligned, lambda r: r["passages"][0])
        pick = [r for r in pick if r["passages"][0] not in taken]
        if pick:
            got = {h["id"]: h for h in _child_json(["prose", "--db", a.prose_db], {
                "passage_ids": [r["passages"][0] for r in pick], "max_bytes": a.max_bytes})}
            for r in pick:
                if r["passages"][0] in got:
                    add({**got[r["passages"][0]], "aligned": "prose",
                         "via": [f"aligned to {r['element']} (score {r['score']})"]}, "alignment")
        if not a.no_query_align:
            for g in _child_json(["query-prose", "--db", a.prose_db],
                                 {"question": question, "n": a.aligned, "max_bytes": a.max_bytes}):
                add({**g, "aligned": "prose", "via": ["query"]}, "query")
    return items


if __name__ == "__main__":
    child_main(sys.argv[1:])
