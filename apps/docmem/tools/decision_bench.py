#!/usr/bin/env python3
"""Decision benchmark: a System One-style decision model (the decider) against each task's baseline.

Each task is a plug-in in TASKS. Its build() returns items; the same run() and score() code serves every task:

    item = {id, state, question: {type, instructions, criteria?}, gold, baseline_pred, baseline_score?,
            baseline_conf?, extra_baselines?, meta?}

* noul: gold is a bool; the decider's p_yes > 0.5 is its answer. Brier = (p - y)^2.
* choice: gold is a list of acceptable labels (a single label, or several when any of them is right).
  The decider's choice must be one of them. Brier = sum over labels of (p_label - target_label)^2, where the
  target spreads 1 over the acceptable labels.
* accuracy, Brier and "automatable at 5% error" (the largest fraction of items, taken by descending decider
  confidence, whose error rate stays at or below 5%) are reported for the decider and for the baseline.
  AUC (noul only) is the rank AUC of p_yes, or of the baseline score when the baseline has one.

    python tools/decision_bench.py --task routing --build --run --score [--limit N]
    python tools/decision_bench.py --summary          # rebuild summary.md from the reports

Outputs under $DBENCH_SCRATCH/dbench (default: ./dbench_work/dbench):
    <task>/items.jsonl, <task>/preds.jsonl, <task>/report.json, summary.md, cache.jsonl (decider cache)

DB content is read in child processes, one per DB, because docmem.store.configure must run before cognee is
imported. Child results are cached under dbench/_work, keyed by their spec, so a rebuild does not reopen a DB.
The decider is the Kev stand-in server (DOCMEM_DECIDER_URL, DOCMEM_DECIDER_FLAVOR, DOCMEM_DECIDER_MODEL); the
defaults below are used when those variables are unset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCMEM = HERE.parent
# Where outputs go (and, for --build, where the source DBs are): $DBENCH_SCRATCH, else ./dbench_work.
SCRATCH = Path(os.environ.get("DBENCH_SCRATCH", Path.cwd() / "dbench_work"))
BENCH = SCRATCH / "dbench"
WORK = BENCH / "_work"
CACHE = BENCH / "cache.jsonl"
PY = os.environ.get("DBENCH_PY", str(SCRATCH / "cvenv" / "bin" / "python"))
EXAMPLES = DOCMEM / "examples"
N3X = SCRATCH / "n3x"
N3X_KB = N3X / "kb"
N3X_MODEL = N3X / "model"
APOLLO_DB = SCRATCH / "dm" / "model_apollo"
WIKI_DB = SCRATCH / "dm" / "dbr"
ENG_DB = SCRATCH / "dm" / "dbe1000"
ONTO = SCRATCH / "onto"
SEP = " › "
SEED = 7
CAP_CHARS = 6000       # answerability state: ~1500 tokens at ~4 characters per token
LEAD_CHARS = (600, 1200)  # concept state: lead passages are read until 600 characters, cut at 1200
NOUL_ANSWERABLE = "Do these passages contain the answer to the question?"


def decider_env() -> None:
    os.environ.setdefault("DOCMEM_DECIDER_URL", "http://127.0.0.1:8009")
    os.environ.setdefault("DOCMEM_DECIDER_FLAVOR", "kev")
    os.environ.setdefault("DOCMEM_DECIDER_MODEL", "kev-latest")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def cap(text: str, n: int) -> tuple[str, bool]:
    """Cut text to at most n characters at a whitespace; (text, truncated)."""
    if len(text) <= n:
        return text, False
    cut = text[:n]
    return cut[: cut.rfind(" ")] if " " in cut else cut, True


def passages_in(excerpts: str) -> list[str]:
    """The passage texts of docmem's excerpt block ("[i] label" line, then the text)."""
    parts = re.split(r"(?m)^\[\d+\] [^\n]*\n", excerpts)
    return [p.strip() for p in parts if p.strip()]


def stable_hash(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


# ---- child processes: DB reads -------------------------------------------------------------------------------

def run_child(op: str, db: Path, spec: dict) -> dict | list:
    """Run `--child OP` on db with spec in its own process; cached by (op, db, spec)."""
    WORK.mkdir(parents=True, exist_ok=True)
    key = stable_hash(op, str(db), spec)
    out_path = WORK / f"{op}-{key}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    spec_path = WORK / f"{op}-{key}.spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(DOCMEM), env.get("PYTHONPATH")]))
    t = time.time()
    r = subprocess.run([PY, str(Path(__file__).resolve()), "--child", op, "--db", str(db),
                        "--spec", str(spec_path), "--out", str(out_path)],
                       env=env, capture_output=True, text=True)
    if r.returncode:
        out_path.unlink(missing_ok=True)
        raise SystemExit(f"decision_bench: child {op} on {db} failed:\n{r.stderr[-3000:]}")
    print(f"  child {op} on {db.name}: {time.time() - t:.0f}s", file=sys.stderr)
    return json.loads(out_path.read_text(encoding="utf-8"))


def child_ask(db: str, spec: dict) -> dict:
    """docmem ask --no-answer -k K for each question, in this process: the hits with their text."""
    import argparse as ap_
    import asyncio
    from docmem import ask

    ap = ap_.ArgumentParser()
    ask.add_common(ap)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("question", nargs="+")
    a = ap.parse_args(["--db", db, "--no-answer", "-k", str(spec["k"]), "x"])
    ask.open_db(a)

    async def go():
        out = {}
        for q in spec["questions"]:
            r = await ask.ask_one(q["question"], a, None)
            out[q["id"]] = [{"distance": h.get("distance"), "text": h["text"], "status": h["status"],
                             "label": ask.label(h), "path": h["path"], "uri": h["uri"]} for h in r["hits"]]
        return out

    return asyncio.run(go())


async def op_kb(db: str, spec: dict) -> list:
    """Every DocPassage of a prose DB, with its text read from the source."""
    from docmem import sources, store
    from docmem.ontology import nodes_of_type

    graph, _ = await store.engines()
    out = []
    for p in await nodes_of_type(graph, "DocPassage"):
        path = p.get("path", "")
        text, status = sources.read_span(p["name"], int(p["offset"]), int(p["length"]), p.get("sha256"),
                                         extractor=p.get("extractor") or None,
                                         doc_sha256=p.get("doc_sha256") or None,
                                         title=path.split(SEP)[0])
        if status == "ok":
            out.append({"id": str(p["id"]), "uri": p["name"], "path": path, "text": text})
    return out


async def op_model(db: str, spec: dict) -> dict:
    """The sections and passages of one model (the second path segment) in a model DB, passage text read from
    the .sysml source. A passage's path is the path of the element it belongs to."""
    from docmem import store
    from docmem.ontology import nodes_of_type

    name = spec["model_name"]
    graph, _ = await store.engines()
    secs = [s for s in await nodes_of_type(graph, "DocSection")
            if s.get("path", "").split(SEP)[1:2] == [name]]
    pas = [p for p in await nodes_of_type(graph, "DocPassage")
           if p.get("path", "").split(SEP)[1:2] == [name]]
    passages = []
    for p in pas:
        text, status = store.read_span(p["name"], int(p["offset"]), int(p["length"]), p.get("sha256"))
        if status == "ok":
            passages.append({"id": str(p["id"]), "path": p["path"].split(SEP), "text": text})
    return {"sections": [{"id": str(s["id"]), "path": s["path"].split(SEP)} for s in secs],
            "passages": passages}


async def op_concepts(db: str, spec: dict) -> dict:
    """Concept placement inputs for a prose DB with a concept scheme.

    The document's centroid is the normalized mean of its passage vectors (docmem classify's). Its top concepts
    are the nearest scheme concepts by cosine. Field scores are leave-one-out: a document is scored against each
    field's centroid computed without it. Lead text: the document's first passages (by offset) up to LEAD_CHARS."""
    import numpy as np
    from docmem import sources, store
    from docmem.ontology import CONCEPTS, centroid, nodes_of_type, passages_of_docs, scheme_concepts, vectors

    graph, vector = await store.engines()
    _, concepts = await scheme_concepts(graph, spec["scheme"])
    cids = list(concepts)
    cvec = await vectors(vector, CONCEPTS, cids)
    missing = [c for c in cids if c not in cvec]
    if missing:
        raise SystemExit(f"concept vectors missing for {len(missing)} concepts")
    C = np.stack([cvec[c] for c in cids]).astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    docs = await nodes_of_type(graph, "DocFile")
    pmap = await passages_of_docs(graph, [str(d["id"]) for d in docs])
    pvec = await vectors(vector, store.PASSAGES, sorted({p for ps in pmap.values() for p in ps}))
    by_title, cent = {}, {}
    for d in docs:
        vs = [pvec[p] for p in pmap.get(str(d["id"]), []) if p in pvec]
        if vs and d.get("title") is not None:
            by_title[d["title"]] = d
            cent[d["title"]] = centroid(np.stack(vs))

    def top(v, k):
        sims = C @ v
        order = np.argsort(-sims, kind="stable")[:k]
        return [{"label": concepts[cids[i]]["name"],
                 "definition": (concepts[cids[i]].get("definition") or "")[:160],
                 "cos": round(float(sims[i]), 4)} for i in order]

    # field centroids with leave-one-out (a document never scores against a field that contains only itself)
    fields = {t: f for t, f in spec["fields"].items() if t in cent}
    fsum, fn = defaultdict(lambda: np.zeros(C.shape[1], dtype=np.float32)), Counter()
    for t, f in fields.items():
        fsum[f] += cent[t]
        fn[f] += 1

    def field_pred(t: str) -> str:
        best, best_s = None, -2.0
        for f in fsum:
            n = fn[f] - (1 if fields.get(t) == f else 0)
            if n <= 0:
                continue
            s = fsum[f] - (cent[t] if fields.get(t) == f else 0)
            s = s / np.linalg.norm(s)
            score = float(s @ cent[t])
            if score > best_s:
                best, best_s = f, score
        return best

    async def lead(title: str):
        d = by_title[title]
        nodes = [store.node_props(n) for n in await graph.get_nodes(pmap.get(str(d["id"]), []))]
        out = ""
        for p in sorted(nodes, key=lambda n: int(n["offset"])):
            text, status = sources.read_span(p["name"], int(p["offset"]), int(p["length"]), p.get("sha256"),
                                             extractor=p.get("extractor") or None,
                                             doc_sha256=p.get("doc_sha256") or None,
                                             title=p["path"].split(SEP)[0])
            if status != "ok":
                return None
            out += text.strip() + "\n\n"
            if len(out) >= LEAD_CHARS[0]:
                break
        out = out.strip()
        return cap(out, LEAD_CHARS[1])[0] if out else None

    rng = random.Random(spec["seed"])
    cats = spec["doc_categories"]
    elig_a = []
    for t in sorted(by_title):
        if t not in cats:
            continue
        tops = top(cent[t], spec["top"])
        names = [x["label"] for x in tops]
        gold = [n for n in names if n in cats[t]]
        if gold and len(set(names)) == len(names):
            elig_a.append((t, tops, gold))
    res_a = []
    for t, tops, gold in rng.sample(elig_a, min(spec["n_a"], len(elig_a))):
        lead_text = await lead(t)
        if lead_text:
            res_a.append({"title": t, "lead": lead_text, "candidates": tops, "gold": gold,
                          "baseline": tops[0]["label"]})
    res_b = []
    for t in sorted(rng.sample(sorted(fields), min(spec["n_b"], len(fields)))):
        lead_text = await lead(t)
        if lead_text:
            res_b.append({"title": t, "lead": lead_text, "gold": [fields[t]], "baseline": field_pred(t)})
    return {"scheme": spec["scheme"], "docs": len(by_title), "eligible_a": len(elig_a), "a": res_a, "b": res_b}


CHILD_OPS = {"kb": op_kb, "model": op_model, "concepts": op_concepts}


def child_main(argv: list[str]) -> None:
    import asyncio

    ap = argparse.ArgumentParser()
    ap.add_argument("--child", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(DOCMEM))
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    if a.child == "ask":
        res = child_ask(a.db, spec)
    else:
        from docmem import store

        store.configure(Path(a.db))  # before cognee is imported
        res = asyncio.run(CHILD_OPS[a.child](a.db, spec))
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


# ---- the decision question: items -----------------------------------------------------------------------------

ROUTE_STRATEGIES = {
    "vector": "one passage holds the answer: nearest-passage vector search",
    "links+split": "facts from several articles or passages: split the question into parts and follow links",
    "links": "a chain of linked facts: follow the links from the first passage to the next",
    "structured list": "a list or a count over many items: gather every match into a structured list",
    "model": "about the SysML model: query the model elements and their requirements",
    "unanswerable": "the documents cannot answer it: say so rather than guess",
}
NAME_AND = re.compile(r"\b[A-Z][\w'.-]*(?: [A-Z][\w'.-]*)* and [A-Z][\w'.-]*")


def route_gold(q: dict) -> str:
    kind, db = q.get("kind"), q.get("db")
    if db in ("model", "both"):
        return "model"
    if kind == "unanswerable" or not q.get("sources"):
        return "unanswerable"
    if kind in ("multi2", "multi3", "concept"):
        return "links+split"
    if kind == "aggregate":
        return "structured list"
    if kind == "relational":
        return "links"
    if kind in ("single", "two-source"):
        return "vector"
    return "links+split" if len({s["file"] for s in q["sources"]}) >= 2 else "vector"


def rule_route(question: str) -> str:
    """Baseline router: the first matching regex decides."""
    if re.search(r"\b(which of|list|how many)\b", question, re.I):
        return "structured list"
    if NAME_AND.search(question):
        return "links+split"
    return "vector"


def build_routing(limit: int | None) -> list[dict]:
    files = [("wiki1000", EXAMPLES / "wiki1000_questions.jsonl"), ("eng1000", EXAMPLES / "eng1000_questions.jsonl"),
             ("apollo_prose", EXAMPLES / "apollo_prose_questions.jsonl"),
             ("apollo_model", EXAMPLES / "apollo_model_questions.jsonl"),
             ("apollo_cross", EXAMPLES / "apollo_cross_questions.jsonl"), ("n3x", N3X / "questions.jsonl")]
    criteria = dict(ROUTE_STRATEGIES)
    items = []
    for src, path in files:
        for q in read_jsonl(path):
            question = q["question"]
            items.append({
                "id": f"{src}:{q['id']}", "state": question,
                "question": {"type": "choice",
                             "instructions": "Which retrieval strategy should answer this question?",
                             "criteria": criteria},
                "gold": [route_gold(q)], "baseline_pred": rule_route(question),
                "meta": {"source": src, "kind": q.get("kind"), "db": q.get("db"), "sources": len(q.get("sources", []))}})
    return items[:limit] if limit else items


def answerability_item(src: str, qid: str, kind: str, must: list[str], excerpts: str, question: str,
                       distance: float | None, meta: dict) -> dict:
    capped, truncated = cap(excerpts, CAP_CHARS)
    texts = passages_in(capped)
    body = "\n".join(texts).lower()
    present = [m for m in must if m.lower() in body]
    unanswerable = kind == "unanswerable"
    gold = (not unanswerable) and bool(must) and len(present) == len(must)
    from_guard = _answer_guard(question, texts)
    return {
        "id": f"{src}:{qid}",
        "state": f"{capped}\n\nQuestion: {question}",
        "question": {"type": "noul", "instructions": NOUL_ANSWERABLE},
        "gold": gold,
        "baseline_pred": not from_guard["offTopic"],
        "baseline_score": from_guard["coverage"],
        "extra_baselines": {"vector_threshold": {"pred": None, "score": None if distance is None else -distance}},
        "meta": {**meta, "source": src, "kind": kind, "truncated": truncated, "must": len(must),
                 "present": len(present), "distance": distance},
    }


SKIP_WORDS = set(("about above after again against among another anyone anything around because been before "
                  "being below between both could describe difference does doing during each either every explain "
                  "from have having their there these thing things those through under until which while whose "
                  "would what when where whom why with within without your yours known called used using often "
                  "usually mean means much many most other others some such than that them then they this tell "
                  "happen happened happens list name named kind kinds type types example").split())


def _stem(word: str) -> str:
    """docmem's answer-guard stemmer (webcognee answer-guard.js), ported."""
    w = word.lower()
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 5 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 5 and re.search(r"(ches|shes|sses|xes|zes)$", w):
        w = w[:-2]
    elif len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 6 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 5 and w.endswith("ed"):
        w = w[:-2]
    return w


def _words(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9À-ɏ']+", text.lower())]


def _answer_guard(question: str, texts: list[str]) -> dict:
    """answer-guard's check (webcognee/answer-guard.js): the question's distinctive words (5+ letters, not filler)
    must appear in the passages; off topic when every one is missing, or the longest one (7+ letters) is."""
    terms, seen_t = [], set()
    for raw in re.findall(r"[a-z0-9À-ɏ']+", question.lower()):
        if len(raw) < 5 or raw in SKIP_WORDS:
            continue
        s = _stem(raw)
        if s not in seen_t:
            seen_t.add(s)
            terms.append(s)
    seen = set()
    for t in texts:
        seen.update(_words(t))

    def present(term: str) -> bool:
        return term in seen or any(w.startswith(term) and len(w) > len(term) for w in seen)

    missing = [t for t in terms if not present(t)]
    longest = max(terms, key=len, default="")
    off = bool(terms) and (len(missing) == len(terms) or (len(longest) >= 7 and longest in missing))
    coverage = 1.0 if not terms else 1 - len(missing) / len(terms)
    return {"offTopic": off, "coverage": round(coverage, 4)}


def build_answerability(limit: int | None) -> list[dict]:
    items = []
    qs = {q["id"]: q for q in read_jsonl(N3X / "questions.jsonl")}
    key = json.loads((N3X / "ctx_key.json").read_text(encoding="utf-8"))
    raw_dir = N3X / "raw"
    for rid in sorted(key):
        rec = key[rid]
        text = (N3X / "ctx" / f"{rid}.txt").read_text(encoding="utf-8")
        excerpts, _, question = text.rpartition("\n\nQuestion: ")
        question = question.strip()
        raw = json.loads((raw_dir / f"{rec['condition']}_{rec['question_id']}.json").read_text(encoding="utf-8"))
        ok = [h for h in raw["hits"] if h.get("status", "ok") == "ok"]
        dist = float(ok[0]["distance"]) if ok and ok[0].get("distance") is not None else None
        items.append(answerability_item(
            "n3x", f"{rec['question_id']}:{rec['condition']}", qs[rec["question_id"]].get("kind", ""),
            qs[rec["question_id"]]["must_include"], excerpts, question, dist,
            {"condition": rec["condition"], "qid": rec["question_id"]}))
    for src, db, path in [("eng", ENG_DB, EXAMPLES / "eng1000_questions.jsonl"),
                          ("wiki", WIKI_DB, EXAMPLES / "wiki1000_questions.jsonl")]:
        qlist = read_jsonl(path)
        res = run_child("ask", db, {"k": 4, "questions": [{"id": q["id"], "question": q["question"]} for q in qlist]})
        hit = answerable = 0
        for q in qlist:
            if not q.get("sources") or q.get("kind") == "unanswerable":
                continue
            answerable += 1
            hay = " ".join(h["path"] + " " + h["label"] for h in res[q["id"]]).lower()
            if any(s["file"].lower() in hay or s.get("heading", "").lower() in hay for s in q["sources"]):
                hit += 1
        print(f"  {src}: retrieval reached a source for {hit}/{answerable} answerable questions", file=sys.stderr)
        for q in qlist:
            hits = res[q["id"]]
            excerpts = "\n\n".join(f"[{i}] {h['label']}\n{h['text']}" for i, h in enumerate(hits, 1))
            dist = float(hits[0]["distance"]) if hits and hits[0].get("distance") is not None else None
            items.append(answerability_item(src, q["id"], q.get("kind", ""), q.get("must_include", []),
                                            excerpts, q["question"], dist, {"qid": q["id"]}))
    # vector-threshold baseline: answerable when the top hit is at most thr away (thr fitted on these items:
    # an optimistic baseline)
    scored = [(it["extra_baselines"]["vector_threshold"]["score"], it["gold"]) for it in items
              if it["extra_baselines"]["vector_threshold"]["score"] is not None]
    if scored:
        thr = max({-s for s, _ in scored}, key=lambda t: sum((-s <= t) == g for s, g in scored))
        for it in items:
            s = it["extra_baselines"]["vector_threshold"]["score"]
            it["extra_baselines"]["vector_threshold"]["pred"] = None if s is None else (-s <= thr)
            it["extra_baselines"]["vector_threshold"]["threshold_distance"] = thr
    return items[:limit] if limit else items


def load_fields() -> dict:
    fields = {}
    for line in (ONTO / "selected_titles.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip():
            field, title = line.split("\t")[:2]
            fields[title] = field
    return fields


def concept_result() -> dict:
    """Both concept tasks read one child result (one DB open)."""
    spec = {"scheme": "wikicat", "doc_categories": json.loads(
        (ONTO / "wikicat_eng.doc_categories.json").read_text(encoding="utf-8")),
        "fields": load_fields(), "top": 8, "n_a": 240, "n_b": 240, "seed": SEED}
    return run_child("concepts", ENG_DB, spec)


def build_concept_a(limit: int | None) -> list[dict]:
    res = concept_result()
    items = []
    for r in res["a"]:
        criteria = {c["label"]: (c["definition"] or f"Wikipedia category {c['label']}") for c in r["candidates"]}
        items.append({"id": f"a:{r['title']}", "state": r["lead"],
                      "question": {"type": "choice",
                                   "instructions": "Which of these Wikipedia category concepts best describes "
                                                   "the subject of this article?",
                                   "criteria": criteria},
                      "gold": r["gold"], "baseline_pred": r["baseline"],
                      "meta": {"title": r["title"], "candidates": [c["label"] for c in r["candidates"]],
                               "cos": [c["cos"] for c in r["candidates"]]}})
    return items[:limit] if limit else items


FIELD_DESC = "the seed field of Wikipedia articles on engineering topics"


def build_concept_b(limit: int | None) -> list[dict]:
    res = concept_result()
    fields = sorted(set(load_fields().values()))
    criteria = {f: f"{FIELD_DESC}: {f}" for f in fields}
    items = [{"id": f"b:{r['title']}", "state": r["lead"],
              "question": {"type": "choice", "instructions": "Which seed field is this article from?",
                           "criteria": criteria},
              "gold": r["gold"], "baseline_pred": r["baseline"], "meta": {"title": r["title"]}} for r in res["b"]]
    return items[:limit] if limit else items


SYSML_KINDS = [
    (r"^requirement def\b", "requirement definition"),
    (r"^action def\b", "action definition"),
    (r"^item def\b", "item definition"),
    (r"^part def\b", "part definition"),
    (r"^constraint(?: def)?\b", "constraint"),
    (r"^attribute(?: def)?\b", "attribute"),
    (r"^port(?: def)?\b", "port"),
    (r"^state(?: def)?\b", "state"),
]
SYSML_STATES = {
    "requirement definition": "requirement definition", "action definition": "action definition",
    "item definition": "item definition", "part definition": "part definition", "constraint": "constraint",
    "attribute": "attribute", "port": "port", "state": "state",
}
DECL_LINE = re.compile(r"^\s*(package|part|requirement|action|item|constraint|attribute|port|state|analysis|"
                       r"calc|enum|import|subject|out|in|inout|ref|doc|connection|interface|flow|allocation|"
                       r"satisfy|perform|exhibit|verification|use|view|alias|bind|connect|first|then|entry|"
                       r"do|accept|send|assert|assume|require|objective|concern|frame)\b[^\n]*$", re.M)


def element_text(passages: list[dict]) -> str:
    """The element's own text: its passages with the declaration line of each removed (only the doc comment,
    requirement text and string values remain)."""
    kept = []
    for p in passages:
        lines = p["text"].split("\n")
        if lines and DECL_LINE.match(lines[0]) and not lines[0].lstrip().startswith("doc"):
            lines = lines[1:]
        text = "\n".join(lines).strip()
        if len(re.findall(r"[A-Za-z]{3,}", text)) >= 3:
            kept.append(text)
    return "\n".join(kept)


def kind_rules(text: str) -> str:
    """Baseline: keyword rules on the element's text."""
    t = text.lower()
    if re.search(r"\bshall\b", t):
        return "requirement definition"
    if re.search(r"\b(must|less than|greater than|at most|at least|within)\b|[<>]=?", t):
        return "constraint"
    if re.search(r"(^|\n|\.\s)\s*(perform|compute|determine|check|send|load|build|validate|generate|derive|"
                 r"calculate|execute|transfer)\b", t) or re.search(r"\b(then|sequence|steps?)\b", t):
        return "action definition"
    if re.search(r"\b(port|interface|connect|connection|pin)s?\b", t):
        return "port"
    if re.search(r"\b(state|mode|phase|transition)s?\b", t):
        return "state"
    if re.search(r"\b(item|message|payload|data|packet|record)s?\b", t):
        return "item definition"
    if re.search(r"\b(value|unit|measure|quantity|number|kg|seconds?)\b", t):
        return "attribute"
    return "part definition"


def build_sysml_kind(limit: int | None) -> list[dict]:
    res = run_child("model", APOLLO_DB, {"model_name": "apollo11"})
    by_path = defaultdict(list)
    for p in res["passages"]:
        by_path[tuple(p["path"])].append(p)
    rng = random.Random(SEED)
    per_kind = defaultdict(list)
    for s in res["sections"]:
        label = s["path"][-1] if s["path"] else ""
        kind = next((k for rx, k in SYSML_KINDS if re.match(rx, label)), None)
        if kind is None:
            continue
        text = element_text(by_path.get(tuple(s["path"]), []))
        if text:
            per_kind[kind].append((s["id"], text))
    items = []
    for kind in sorted(per_kind):
        pool = per_kind[kind]
        for sid, text in rng.sample(pool, min(30, len(pool))):
            text, trunc = cap(text, 1200)
            items.append({"id": f"el:{sid}", "state": text,
                          "question": {"type": "choice",
                                       "instructions": "Which kind of SysML v2 element is this text describing?",
                                       "criteria": {k: f"a SysML v2 {k}" for k in SYSML_STATES}},
                          "gold": [kind], "baseline_pred": kind_rules(text),
                          "meta": {"element": s_label(res, sid), "kind": kind, "truncated": trunc,
                                   "pool_per_kind": {k: len(v) for k, v in per_kind.items()}}})
    return items[:limit] if limit else items


def s_label(res: dict, sid: str) -> str:
    for s in res["sections"]:
        if s["id"] == sid:
            return s["path"][-1]
    return ""


def build_alignment(limit: int | None) -> list[dict]:
    import numpy as np
    from fastembed import TextEmbedding

    kb = run_child("kb", N3X_KB, {})
    mdl = run_child("model", N3X_MODEL, {"model_name": "nanos3reader"})
    kb_by_id = {p["id"]: p for p in kb}
    kb_index = {p["id"]: j for j, p in enumerate(kb)}
    el_texts = defaultdict(list)
    for p in mdl["passages"]:
        el_texts[tuple(p["path"])].append(p["text"])
    rows = read_jsonl(N3X / "align" / "alignment.jsonl")
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    docs = [f"{p['path']}\n\n{p['text']}" for p in kb]
    D = np.asarray(list(model.embed(docs)), dtype=np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    rng = random.Random(SEED)
    items, skipped = [], Counter()
    for i, row in enumerate(rows):
        el_path = tuple(row["model_path"].split(SEP))
        el = "\n".join(el_texts.get(el_path, []))
        if not el.strip():
            skipped["element text missing"] += 1
            continue
        el_text, _ = cap(el, 1200)
        prov = [pid for pid in row["passages"] if pid in kb_by_id]
        if not prov:
            skipped["no provenance passage in kb"] += 1
            continue
        e = np.asarray(list(model.embed([el_text])), dtype=np.float32)[0]
        e /= np.linalg.norm(e)
        sims = D @ e
        prov_set = set(row["passages"])
        cands = [j for j in np.argsort(-sims, kind="stable") if kb[j]["id"] not in prov_set][:2]
        for pid in prov:
            ptext, _ = cap(kb_by_id[pid]["text"], 1200)
            items.append(_align_item(el_text, ptext, True, row["element"], float(sims[kb_index[pid]]), pid))
        for j in cands:
            ptext, _ = cap(kb[j]["text"], 1200)
            items.append(_align_item(el_text, ptext, False, row["element"], float(sims[j]), kb[j]["id"]))
    rng.shuffle(items)
    thr = _best_threshold([(it["baseline_score"], it["gold"]) for it in items])
    for it in items:
        it["baseline_pred"] = it["baseline_score"] >= thr
    print(f"  alignment: skipped {dict(skipped)}; threshold {thr:.4f} (fitted on these items)", file=sys.stderr)
    return items[:limit] if limit else items


def _align_item(el: str, passage: str, gold: bool, element: str, cos: float, pid: str) -> dict:
    return {"id": f"aln:{element}:{pid}:{int(gold)}",
            "state": f"Model element: {el}\n\nPassage: {passage}",
            "question": {"type": "noul", "instructions": "Does this passage describe this model element?"},
            "gold": gold, "baseline_pred": None, "baseline_score": round(cos, 4),
            "meta": {"element": element, "passage_id": pid}}


def _best_threshold(pairs: list[tuple[float, bool]]) -> float:
    cands = sorted({s for s, _ in pairs})
    return max(cands, key=lambda t: sum((s >= t) == g for s, g in pairs))


MATURITY_RX = re.compile(r'maturity\s*=\s*"(\w+)"')
MATURITY_LEVELS = {"concept": "an idea, not yet designed", "designed": "designed, not yet implemented",
                   "implemented": "implemented, not yet verified", "verified": "implemented and verified"}


def maturity_rules(text: str) -> str:
    """Baseline: modal and status words in the element's text."""
    t = text.lower()
    if re.search(r"\b(verif\w*|test\w*|validat\w*|proven)\b", t):
        return "verified"
    if re.search(r"\b(will|shall|planned|design\w*|proposed)\b", t):
        return "designed"
    if re.search(r"\b(idea|concept|may|might|could|explor\w*)\b", t):
        return "concept"
    return "implemented"


def build_maturity(limit: int | None) -> list[dict]:
    """The n3x model's @Provenance maturity tags. State: the element's text with its maturity clause removed
    (the source reference stays). Gold: the tag."""
    res = run_child("model", N3X_MODEL, {"model_name": "nanos3reader"})
    by_path = defaultdict(list)
    for p in res["passages"]:
        by_path[tuple(p["path"])].append(p)
    items = []
    for path in sorted(by_path):
        ps = by_path[path]
        tags = [m.group(1) for p in ps for m in [MATURITY_RX.search(p["text"])] if m]
        if not tags:
            continue
        cleaned = [{"text": re.sub(r';?\s*maturity\s*=\s*"\w+";?', "", p["text"])} for p in ps]
        text = element_text(cleaned)
        if not text:
            continue
        text, trunc = cap(text, 1200)
        items.append({"id": f"mat:{'/'.join(path[2:])}", "state": text,
                      "question": {"type": "choice",
                                   "instructions": "Which maturity level does this model element have?",
                                   "criteria": MATURITY_LEVELS},
                      "gold": [tags[0]], "baseline_pred": maturity_rules(text),
                      "meta": {"element": path[-1], "tags_on_element": len(tags), "truncated": trunc}})
    return items[:limit] if limit else items


TASKS = {
    "routing": {"title": "1 routing: retrieval strategy", "qtype": "choice", "build": build_routing,
                "gold": "route_gold() from the question's kind/db fields: single, two-source and single-article "
                        "cases give vector; multi2/multi3/concept give links+split; aggregate gives structured list; "
                        "relational gives links; no sources or unanswerable gives unanswerable; model and cross DB "
                        "questions give model. Two sources in different files and no kind give links+split.",
                "baseline": "rule_route(): regex. 'which of', 'list', 'how many' give structured list; capitalized "
                            "names joined by 'and' give links+split; otherwise vector."},
    "answerability": {"title": "2 answerability: do the passages contain the answer", "qtype": "noul",
                      "build": build_answerability,
                      "gold": "every must_include string appears in the passage text the decider sees (state capped "
                              "at 6000 characters, about 1500 tokens); unanswerable questions are False.",
                      "baseline": "answer-guard (webcognee answer-guard.js, ported): off topic when no distinctive "
                                  "question word appears in the passages (or the longest one is missing). "
                                  "Extra baseline: top-hit vector distance threshold, fitted on these items "
                                  "(optimistic)."},
    "concept_a": {"title": "4a concept placement: which Wikipedia category", "qtype": "choice",
                  "build": build_concept_a,
                  "gold": "a candidate (top 8 wikicat concepts by the article's centroid cosine) that is one of the "
                          "article's real Wikipedia categories (wikicat_eng.doc_categories.json); only articles with "
                          "at least one such candidate are used.",
                  "baseline": "top-1 candidate by cosine."},
    "concept_b": {"title": "4b concept placement: which seed field", "qtype": "choice", "build": build_concept_b,
                  "gold": "the field the article was sampled under (onto/selected_titles.tsv).",
                  "baseline": "nearest field centroid, leave-one-out (the article is not in its own field's centroid)."},
    "sysml_kind": {"title": "5 SysML element kind", "qtype": "choice", "build": build_sysml_kind,
                   "gold": "the element's declaration kind (apollo11 model DB). Declaration lines are removed from the "
                           "text; elements with no remaining text are dropped; at most 30 per kind.",
                   "baseline": "kind_rules(): keyword rules on the text (shall, constraint words, action verbs, port, "
                               "state, item, value words; default part definition)."},
    "maturity": {"title": "10 maturity level", "qtype": "choice", "build": build_maturity,
                 "gold": "the element's @Provenance maturity tag in the n3x model (33 tags; concept never occurs).",
                 "baseline": "maturity_rules(): modal and status words (verified/test, will/shall/planned, "
                             "may/idea; default implemented)."},
    "alignment": {"title": "7 alignment verification", "qtype": "noul", "build": build_alignment,
                  "gold": "positive: a provenance passage of the element (alignment.jsonl); negative: the 2 passages "
                          "nearest the element by vector that are not its provenance.",
                  "baseline": "cosine of bge-small embeddings (element vs passage) with a threshold fitted on these "
                              "items (optimistic)."},
}


# ---- run, score ---------------------------------------------------------------------------------------------

def run_task(name: str, items: list[dict]) -> dict:
    from docmem.decide import Decider

    decider_env()
    decider = Decider.from_env(cache=CACHE)
    before = dict(decider.stats)
    groups = defaultdict(list)
    for it in items:
        groups[it["state"]].append(it)
    preds = {}
    t0 = time.time()
    for n, (state, its) in enumerate(groups.items(), 1):
        out = decider.decide(state, {it["id"]: it["question"] for it in its})
        for it in its:
            d = out[it["id"]]
            preds[it["id"]] = {"id": it["id"], "p_yes": d.p_yes, "probabilities": d.probabilities,
                               "choice": d.choice, "confidence": d.confidence, "score": d.score}
        if n % 50 == 0:
            print(f"  {name}: {n}/{len(groups)} states", file=sys.stderr)
    d_stats = {k: round(decider.stats[k] - before[k], 3) for k in before}
    d_stats.update({"states": len(groups), "wall_seconds": round(time.time() - t0, 1)})
    with open(BENCH / name / "preds.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(preds[it["id"]], ensure_ascii=False) + "\n")
    return d_stats


def roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = sum(ranks[k] for k in range(len(labels)) if labels[k])
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def automatable(conf: list[float], correct: list[bool], limit: float = 0.05) -> float:
    """Largest fraction of items, taken by descending confidence, whose error rate stays at or below limit."""
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    errors, best = 0, 0
    for k, i in enumerate(order, 1):
        errors += not correct[i]
        if errors / k <= limit:
            best = k
    return best / len(conf) if conf else 0.0


def evaluate(items: list[dict], preds: dict, qtype: str) -> dict:
    """Metrics for the decider, the primary baseline (baseline_pred) and any extra baselines."""
    rows = []
    for it in items:
        p = preds[it["id"]]
        rows.append((it, p))
    out = {"n": len(items), "qtype": qtype}
    if qtype == "noul":
        y = [bool(it["gold"]) for it, _ in rows]
        pd_ = [p["p_yes"] for _, p in rows]
        dec_correct = [(pv > 0.5) == g for pv, g in zip(pd_, y)]
        out["decider"] = {"acc": _mean(dec_correct), "brier": _mean([(pv - g) ** 2 for pv, g in zip(pd_, y)]),
                          "auc": roc_auc(pd_, y), "auto5": automatable([max(pv, 1 - pv) for pv in pd_], dec_correct)}
        out["baseline"] = _noul_block(rows, y, "baseline", "baseline_pred", "baseline_score", "baseline_conf")
        extras = {}
        for name in {k for it, _ in rows for k in it.get("extra_baselines", {})}:
            extras[name] = _noul_block(rows, y, name, None, None, None, extra=True)
        out["extra_baselines"] = extras
    else:
        gold_sets = [set(it["gold"]) for it, _ in rows]
        dec_correct = [p["choice"] in g for (it, p), g in zip(rows, gold_sets)]
        briers, conf = [], []
        for (it, p), g in zip(rows, gold_sets):
            labels = list(it["question"]["criteria"])
            probs = p["probabilities"] or {}
            target = {lab: (1 / len(g) if lab in g else 0.0) for lab in labels}
            briers.append(sum((probs.get(lab, 0.0) - target[lab]) ** 2 for lab in labels))
            conf.append(probs.get(p["choice"], max(probs.values()) if probs else 0.0))
        out["decider"] = {"acc": _mean(dec_correct), "brier": _mean(briers),
                          "auto5": automatable(conf, dec_correct), "auc": None}
        base_correct = [it["baseline_pred"] in g for (it, _), g in zip(rows, gold_sets)]
        base_brier = []
        for (it, _), g in zip(rows, gold_sets):
            labels = list(it["question"]["criteria"])
            target = {lab: (1 / len(g) if lab in g else 0.0) for lab in labels}
            base_brier.append(sum(((1.0 if lab == it["baseline_pred"] else 0.0) - target[lab]) ** 2
                                  for lab in labels))
        out["baseline"] = {"acc": _mean(base_correct), "brier": _mean(base_brier), "auc": None, "auto5": None}
        out["chance"] = _mean([1 / len(it["question"]["criteria"]) for it, _ in rows])
    return out


def _mean(xs: list[float]) -> float:
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else float("nan")


def _noul_block(rows, y, key, pred_key, score_key, conf_key, extra=False) -> dict:
    """Metrics of one noul baseline: its predictions, rank AUC of its scores, and automatable@5% when it has a
    confidence. key names the extra baseline when extra is True."""
    if extra:
        preds_ = [it["extra_baselines"][key]["pred"] for it, _ in rows]
        scores = [it["extra_baselines"][key]["score"] for it, _ in rows]
    else:
        preds_ = [it[pred_key] for it, _ in rows]
        scores = [it.get(score_key) for it, _ in rows]
    if any(p is None for p in preds_):
        return {"acc": None, "brier": None, "auc": None, "auto5": None}
    correct = [bool(p) == g for p, g in zip(preds_, y)]
    has_scores = all(s is not None for s in scores)
    confs = [it.get(conf_key) for it, _ in rows] if conf_key else [None] * len(rows)
    auc = roc_auc(scores, y) if has_scores else None
    auto = None
    if all(c is not None for c in confs):
        auto = automatable(confs, correct)
    return {"acc": _mean(correct), "brier": _mean([(int(bool(p)) - int(g)) ** 2 for p, g in zip(preds_, y)]),
            "auc": None if auc is None else round(auc, 4), "auto5": auto}


def disagreements(items: list[dict], preds: dict, qtype: str, base_key: str = "baseline_pred") -> list[dict]:
    """Up to two items where the decider and the baseline pick differently; decider-right ones first."""
    out = []
    for it in items:
        p = preds[it["id"]]
        if qtype == "noul":
            d_pick, b_pick = p["p_yes"] > 0.5, it[base_key]
            if b_pick is None or d_pick == bool(b_pick):
                continue
            right = d_pick == bool(it["gold"])
        else:
            d_pick, b_pick = p["choice"], it[base_key]
            if b_pick is None or d_pick == b_pick:
                continue
            right = d_pick in it["gold"]
        out.append({"id": it["id"], "decider_right": right, "decider": d_pick, "baseline": b_pick,
                    "gold": it["gold"], "instructions": it["question"]["instructions"],
                    "state": it["state"][:300]})
    out.sort(key=lambda x: not x["decider_right"])
    return out[:2]


def score_task(name: str, items: list[dict], run_stats: dict | None) -> dict:
    spec = TASKS[name]
    preds = {r["id"]: r for r in read_jsonl(BENCH / name / "preds.jsonl")}
    base_key = "baseline_pred"
    metrics = evaluate(items, preds, spec["qtype"])
    ex = disagreements(items, preds, spec["qtype"], base_key)
    report = {"task": name, "title": spec["title"], "n": len(items), "qtype": spec["qtype"],
              "gold_how": spec["gold"], "baseline_how": spec["baseline"],
              "metrics": metrics, "decider_run": run_stats or read_run(name),
              "disagreement_examples": ex,
              "sources": dict(Counter((it.get("meta") or {}).get("source", "-") for it in items))}
    (BENCH / name / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def read_run(name: str) -> dict | None:
    p = BENCH / name / "run.json"
    return json.loads(p.read_text()) if p.exists() else None


def write_summary() -> None:
    rows = []
    for name in TASKS:
        p = BENCH / name / "report.json"
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    lines = ["# Decision benchmark: decider vs baseline", "",
             "| task | n | baseline acc | decider acc | Brier base | Brier dec | auto@5% base | auto@5% dec | "
             "AUC base | AUC dec | decider s/item |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m, run = r["metrics"], r["decider_run"] or {}
        n = max(1, r["n"])
        lines.append(
            f"| {r['title']} | {r['n']} | {m['baseline']['acc']} | {m['decider']['acc']} | "
            f"{m['baseline']['brier']} | {m['decider']['brier']} | {m['baseline'].get('auto5')} | "
            f"{m['decider'].get('auto5')} | {m['baseline'].get('auc')} | {m['decider'].get('auc')} | "
            f"{round(run.get('seconds', 0) / n, 2) if run else 'n/a'} |")
    lines += ["", "Brier: noul = (p - y)^2; choice = sum over options of (p - target)^2 (range 0 to 2). "
              "Baseline Brier uses the hard prediction (0 or 1). Choice tasks: chance accuracy is in "
              "report.json (metrics.chance). Decider s/item = server seconds per item over the whole run."]
    for r in rows:
        lines += ["", f"## {r['title']}", "", f"- gold: {r['gold_how']}", f"- baseline: {r['baseline_how']}",
                  f"- sources: {r['sources']}"]
        for ex in r["disagreement_examples"]:
            state = " ".join(ex["state"].split())[:200]
            lines.append(f"- example {ex['id']} (decider right: {ex['decider_right']}): state \"{state}\"; "
                         f"gold {ex['gold']}; decider {ex['decider']}; baseline {ex['baseline']}")
    (BENCH / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv: list[str]) -> None:
    if argv and argv[0] == "--child":
        child_main(argv)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="at most N items (applied at build and run)")
    ap.add_argument("--summary", action="store_true", help="rebuild summary.md from the reports")
    a = ap.parse_args(argv)
    if a.summary:
        write_summary()
        return
    if not a.task:
        ap.error("--task NAME (or --summary)")
    do_all = not (a.build or a.run or a.score)
    spec = TASKS[a.task]
    tdir = BENCH / a.task
    tdir.mkdir(parents=True, exist_ok=True)
    if a.build or do_all:
        items = spec["build"](a.limit or None)
        with open(tdir / "items.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"{a.task}: built {len(items)} items; gold labels: "
              f"{dict(Counter(str(it['gold']) for it in items).most_common(8))}", file=sys.stderr)
    items = read_jsonl(tdir / "items.jsonl")
    if a.limit:
        items = items[:a.limit]
    run_stats = None
    if a.run or do_all:
        run_stats = run_task(a.task, items)
        (tdir / "run.json").write_text(json.dumps(run_stats, indent=1), encoding="utf-8")
        print(f"{a.task}: decided {len(items)} items: {run_stats}", file=sys.stderr)
    if a.score or do_all:
        r = score_task(a.task, items, run_stats)
        print(json.dumps(r["metrics"], indent=1, ensure_ascii=False))
        write_summary()


if __name__ == "__main__":
    main(sys.argv[1:])
