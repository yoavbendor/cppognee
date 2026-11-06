"""Concepts from an ontology: load a SKOS concept scheme into cognee and classify documents by it.

    docmem ontology load --db DB --skos FILE [--scheme NAME] [--query-prefix]
    docmem classify --db DB [--scheme NAME] [--top 3] [--min-sim 0.0] [--level doc|section|both]
    docmem concepts --db DB [--scheme NAME] [--doc TITLE] [--tree]

``load`` reads IPTC Media Topics JSON (detected by its "conceptSet" key) or any SKOS RDF/Turtle/JSON-LD that
rdflib parses. Each live concept becomes a DocConcept node (its label and definition are embedded) with a
"broader" edge to its parent. Retired concepts are skipped.

``classify`` takes each document's centroid (the normalized mean of its passages' stored vectors; likewise
for each section) and links it to its nearest concepts: item -"about_concept"-> concept, with the cosine
similarity as "score". Re-running replaces the scheme's earlier about_concept edges.

Nothing here calls an LLM or reads a source file: only the graph and the stored vectors are used.
"""

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import store

CONCEPTS = "DocConcept_embed_text"
ABOUT = "about_concept"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # the bge query instruction
CONCEPT_MIN_LEVEL = 2  # ask --concepts follows concepts at this level or deeper (not "sport" or "politics")
RDF_FORMATS = {".ttl": "turtle", ".rdf": "xml", ".owl": "xml", ".xml": "xml", ".nt": "nt", ".n3": "n3",
               ".jsonld": "json-ld", ".json": "json-ld"}


@dataclass(frozen=True)
class Concept:
    uri: str
    label: str
    definition: str
    broader: tuple[str, ...] = ()  # the URIs of the parents


def _pick(pairs: list[tuple[str, str]], what: str, required: bool) -> str:
    """The English text of (language, text) pairs, else an untagged one; an error when required and absent."""
    for lang, text in pairs:
        if lang.lower().startswith("en"):
            return text
    for lang, text in pairs:
        if not lang:
            return text
    if required:
        raise ValueError(f"{what}: no English or untagged label")
    return ""


def parse_iptc(doc: dict) -> list[Concept]:
    """IPTC NewsCodes JSON: a "conceptSet" of SKOS concepts (prefLabel and definition by language, broader
    as the parent's URI). Concepts with a "retired" date are skipped."""
    retired = {c["uri"] for c in doc["conceptSet"] if c.get("retired")}
    out = []
    for c in doc["conceptSet"]:
        if c.get("retired"):
            continue
        out.append(Concept(
            uri=c["uri"],
            label=_pick(list(c["prefLabel"].items()), c["uri"], required=True),
            definition=_pick(list(c.get("definition", {}).items()), c["uri"], required=False),
            broader=tuple(c.get("broader", []))))
    return _checked(out, retired)


def parse_skos(path: Path) -> list[Concept]:
    """SKOS concepts from an RDF file (rdflib guesses nothing: the format comes from the suffix). A concept
    with owl:deprecated true is skipped, as IPTC's retired ones are."""
    from rdflib import RDF, Graph, Namespace

    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    OWL = Namespace("http://www.w3.org/2002/07/owl#")
    fmt = RDF_FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"{path}: unknown RDF suffix (use one of {', '.join(RDF_FORMATS)})")
    g = Graph()
    g.parse(path, format=fmt)

    def texts(subject, pred):
        return [(lit.language or "", str(lit)) for lit in g.objects(subject, pred)]

    every = list(g.subjects(RDF.type, SKOS.Concept))
    retired = {str(s) for s in every if any(str(o).lower() in ("true", "1") for o in g.objects(s, OWL.deprecated))}
    out = []
    for s in every:
        uri = str(s)
        if uri in retired:
            continue
        out.append(Concept(
            uri=uri,
            label=_pick(texts(s, SKOS.prefLabel), uri, required=True),
            definition=_pick(texts(s, SKOS.definition), uri, required=False),
            broader=tuple(sorted(str(o) for o in g.objects(s, SKOS.broader)))))
    return _checked(out, retired)


def load_concepts(path: Path) -> list[Concept]:
    """IPTC JSON when the file is JSON with a "conceptSet", otherwise SKOS RDF (JSON-LD for other .json)."""
    if path.suffix.lower() == ".json":
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and "conceptSet" in doc:
            return parse_iptc(doc)
    return parse_skos(path)


def _checked(concepts: list[Concept], retired: set[str]) -> list[Concept]:
    """Every parent must be a live concept of the scheme; a retired parent is an error, not a silent re-parent."""
    live = {c.uri for c in concepts}
    for c in concepts:
        for p in c.broader:
            if p in retired:
                raise ValueError(f"{c.label} ({c.uri}) is under retired concept {p}")
            if p not in live:
                raise ValueError(f"{c.label} ({c.uri}) is under {p}, which is not in the scheme")
    levels(concepts)  # raises on a cycle
    return concepts


def levels(concepts: list[Concept]) -> dict[str, int]:
    """Level 0 = no parent; otherwise one more than the least level of its parents. A cycle is an error."""
    by = {c.uri: c for c in concepts}
    out: dict[str, int] = {}

    def level(uri: str, path: frozenset) -> int:
        if uri in out:
            return out[uri]
        if uri in path:
            raise ValueError(f"broader cycle through {uri}")
        parents = by[uri].broader
        out[uri] = 0 if not parents else 1 + min(level(p, path | {uri}) for p in parents)
        return out[uri]

    for c in concepts:
        level(c.uri, frozenset())
    return out


def centroid(vectors: np.ndarray) -> np.ndarray:
    """The normalized mean of the given vectors (rows)."""
    v = vectors.mean(axis=0)
    return v / (np.linalg.norm(v) or 1.0)


def top_k(row: np.ndarray, k: int, min_sim: float) -> list[int]:
    """Indices of the k largest similarities that are at least min_sim, best first."""
    order = np.argsort(-row, kind="stable")[:k]
    return [int(i) for i in order if row[i] >= min_sim]


# ---- the cognee side -------------------------------------------------------------------------------

_model = None


def concept_model():
    """DocConcept, defined once cognee is importable (like store.models)."""
    global _model
    if _model is None:
        from cognee.low_level import DataPoint

        class DocConcept(DataPoint):
            name: str  # the English label
            key: str
            uri: str
            scheme: str
            definition: str
            level: int  # 0 = top
            embed_text: str  # "label: definition", with the query prefix when loaded with it; embedded
            metadata: dict = {"index_fields": ["embed_text"], "identity_fields": ["key"]}

        _model = DocConcept
    return _model


def _props(node) -> dict:
    """A node from get_filtered_graph_data as a flat dict (it comes as (id, props) or as a dict)."""
    if isinstance(node, (tuple, list)):
        return store.node_props({"id": node[0], **node[1]})
    return store.node_props(node)


async def nodes_of_type(graph, type_name: str) -> list[dict]:
    nodes, _ = await graph.get_filtered_graph_data([{"type": [type_name]}], include_edges=False)
    return [_props(n) for n in nodes]


async def scheme_concepts(graph, scheme: str | None) -> tuple[str, dict]:
    """(scheme, {concept id: props}) for the named scheme, or the only one in the index."""
    nodes = await nodes_of_type(graph, "DocConcept")
    if not nodes:
        raise SystemExit("docmem: this index has no ontology (run docmem ontology load first)")
    schemes = sorted({n.get("scheme", "") for n in nodes})
    if scheme is None:
        if len(schemes) != 1:
            raise SystemExit(f"docmem: the index has {len(schemes)} schemes ({', '.join(schemes)}): pass --scheme")
        scheme = schemes[0]
    elif scheme not in schemes:
        raise SystemExit(f"docmem: no scheme {scheme!r} in this index (have: {', '.join(schemes)})")
    return scheme, {n["id"]: n for n in nodes if n.get("scheme") == scheme}


async def vectors(vector, collection: str, ids: list[str]) -> dict[str, np.ndarray]:
    """Stored vectors by id (ids missing from the collection are absent from the result)."""
    out = {}
    for start in range(0, len(ids), 1000):
        for r in await vector.retrieve(collection, ids[start:start + 1000], include_vector=True):
            out[str(r.id)] = np.asarray(r.payload["vector"], dtype=np.float32)
    return out


async def broader_parents(graph, concept_ids: list[str]) -> dict[str, list[str]]:
    """child concept id -> parent concept ids, from the broader edges."""
    out: dict[str, list[str]] = defaultdict(list)
    if concept_ids:
        for child, parent in await graph.query(
                "MATCH (c:Node)-[r:EDGE]->(p:Node) WHERE r.relationship_name = 'broader' AND c.id IN $ids "
                "RETURN c.id, p.id", {"ids": concept_ids}):
            out[str(child)].append(str(parent))
    return out


async def docs_with_concepts(graph, concept_ids: list[str]) -> dict[str, list[str]]:
    """concept id -> ids of the DocFiles that are about it."""
    out: dict[str, list[str]] = defaultdict(list)
    if concept_ids:
        for fid, cid in await graph.query(
                "MATCH (f:Node)-[r:EDGE]->(c:Node) WHERE c.id IN $cids AND r.relationship_name = $rel "
                "AND f.type = 'DocFile' RETURN f.id, c.id", {"cids": concept_ids, "rel": ABOUT}):
            out[str(cid)].append(str(fid))
    return out


async def docs_of_passages(graph, passage_ids: list[str]) -> dict[str, str]:
    """passage id -> its DocFile id."""
    rows = await graph.query(
        "MATCH (p:Node)-[r1:EDGE]->(s:Node)-[r2:EDGE]->(f:Node) WHERE p.id IN $ids "
        "AND r1.relationship_name = 'section' AND r2.relationship_name = 'doc' RETURN p.id, f.id",
        {"ids": passage_ids})
    return {str(p): str(f) for p, f in rows}


async def passages_of_docs(graph, doc_ids: list[str]) -> dict[str, list[str]]:
    """DocFile id -> its passage ids."""
    out: dict[str, list[str]] = defaultdict(list)
    rows = await graph.query(
        "MATCH (p:Node)-[r1:EDGE]->(s:Node)-[r2:EDGE]->(f:Node) WHERE f.id IN $ids "
        "AND r1.relationship_name = 'section' AND r2.relationship_name = 'doc' AND p.type = 'DocPassage' "
        "RETURN p.id, f.id", {"ids": doc_ids})
    for p, f in rows:
        out[str(f)].append(str(p))
    return out


async def concepts_of_docs(graph, doc_ids: list[str], min_level: int,
                           scheme: str | None = None) -> dict[str, dict[str, str]]:
    """DocFile id -> {concept id: label} for its about_concept concepts at min_level or deeper (of one
    scheme, if given)."""
    rows = await graph.query(
        "MATCH (f:Node)-[r:EDGE]->(c:Node) WHERE f.id IN $ids AND r.relationship_name = $rel RETURN f.id, c.id",
        {"ids": doc_ids, "rel": ABOUT})
    cids = list({str(c) for _, c in rows})
    props = {n["id"]: store.node_props(n) for n in await graph.get_nodes(cids)} if cids else {}
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for f, c in rows:
        p = props.get(str(c))
        if p is not None and p.get("level", 0) >= min_level and (scheme is None or p.get("scheme") == scheme):
            out[str(f)][str(c)] = p["name"]
    return out


def _check_pipeline(result, what: str):
    for run in (result or {}).values():
        status = str(getattr(run, "status", ""))
        if "Errored" in status or "ERRORED" in status:
            raise SystemExit(f"docmem {what}: the cognee pipeline failed: {getattr(run, 'payload', run)}")


# ---- ontology load -----------------------------------------------------------------------------------

async def load(concepts: list[Concept], scheme: str, prefix: str):
    """Store the concepts (embedded) and their broader edges through index.write, as index does."""
    import cognee
    from .index import Stats, write

    await cognee.run_migrations()
    DocConcept = concept_model()
    lv = levels(concepts)
    nodes = {}
    for c in concepts:
        text = f"{c.label}: {c.definition}" if c.definition else c.label
        nodes[c.uri] = DocConcept(name=c.label, key=store.key("concept", scheme, c.uri), uri=c.uri, scheme=scheme,
                                  definition=c.definition, level=lv[c.uri], embed_text=prefix + text)
    edges = [(str(nodes[c.uri].id), str(nodes[p].id), "broader", {"edge_text": f"{c.label} is under {nodes[p].name}"})
             for c in concepts for p in c.broader]
    await write(list(nodes.values()), [], [], Stats(batches=0), edges)
    return lv, len(edges)


def ontology_main(argv=None):
    ap = argparse.ArgumentParser(prog="docmem ontology", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ld = sub.add_parser("load", help="store an ontology's concepts and their broader edges")
    ld.add_argument("--db", type=Path, required=True)
    ld.add_argument("--skos", type=Path, required=True, help="IPTC Media Topics JSON, or SKOS RDF/Turtle/JSON-LD")
    ld.add_argument("--scheme", help="name the concepts are stored under (default: the file name without suffix)")
    ld.add_argument("--query-prefix", action="store_true",
                    help="embed the concepts with the bge query instruction (the question side) instead of plain")
    a = ap.parse_args(argv)
    store.configure(a.db)
    t = time.time()
    concepts = load_concepts(a.skos)
    scheme = a.scheme or a.skos.stem
    prefix = QUERY_PREFIX if a.query_prefix else ""
    lv, n_broader = asyncio.run(load(concepts, scheme, prefix))
    print(json.dumps({"scheme": scheme, "concepts": len(concepts), "levels": dict(sorted(Counter(lv.values()).items())),
                      "broader_edges": n_broader, "query_prefix": a.query_prefix,
                      "seconds": round(time.time() - t, 1)}))


# ---- classify ----------------------------------------------------------------------------------------

async def replace_about(concept_ids: list[str], edges: list[tuple]) -> int:
    """Delete the scheme's about_concept edges and write the new ones, in one cognee pipeline run. Returns the
    number deleted. The graph adapter's delete_edge_triples removes only those relationships."""
    import cognee
    from cognee.infrastructure.databases.provenance import EdgeIdentity
    from cognee.modules.pipelines import Task

    counts = {}

    async def store_batch(items, ctx=None):
        graph, _ = await store.engines()
        old = await graph.query(
            "MATCH (i:Node)-[r:EDGE]->(c:Node) WHERE c.id IN $cids AND r.relationship_name = $rel RETURN i.id, c.id",
            {"cids": concept_ids, "rel": ABOUT})
        await graph.delete_edge_triples(
            [EdgeIdentity(source_id=str(i), target_id=str(c), relationship_name=ABOUT) for i, c in old])
        await graph.add_edges(edges)
        counts["replaced"] = len(old)
        return []

    result = await cognee.run_custom_pipeline(tasks=[Task(store_batch, needs_llm=False)], data=[{"batch": 0}],
                                              dataset=store.DATASET)
    _check_pipeline(result, "classify")
    return counts["replaced"]


async def classify(scheme: str | None, top: int, min_sim: float, level: str) -> dict:
    import cognee

    await cognee.run_migrations()
    graph, vector = await store.engines()
    scheme, concepts = await scheme_concepts(graph, scheme)
    cids = list(concepts)
    cvec = await vectors(vector, CONCEPTS, cids)
    missing = [c for c in cids if c not in cvec]
    if missing:
        raise SystemExit(f"docmem classify: {len(missing)} concepts have no stored vector (re-run ontology load)")
    C = np.stack([cvec[c] for c in cids])
    C /= np.linalg.norm(C, axis=1, keepdims=True)

    t = time.time()
    rows = await graph.query(
        "MATCH (p:Node)-[r1:EDGE]->(s:Node)-[r2:EDGE]->(f:Node) WHERE r1.relationship_name = 'section' "
        "AND r2.relationship_name = 'doc' AND p.type = 'DocPassage' RETURN p.id, s.id, f.id")
    pvec = await vectors(vector, store.PASSAGES, [str(p) for p, _, _ in rows])
    by_doc, by_sec = defaultdict(list), defaultdict(list)
    no_vector = 0
    for p, s, f in rows:
        v = pvec.get(str(p))
        if v is None:
            no_vector += 1  # a passage whose vector row is missing: the index was interrupted
            continue
        by_doc[str(f)].append(v)
        by_sec[str(s)].append(v)
    all_docs = await nodes_of_type(graph, "DocFile")
    n_docs_all = len(all_docs)
    titles = {n["id"]: n for n in all_docs if n["id"] in by_doc}
    paths = {n["id"]: n for n in await nodes_of_type(graph, "DocSection") if n["id"] in by_sec}

    items = []  # (kind, id, label, centroid)
    if level in ("doc", "both"):
        items += [("doc", i, titles[i].get("title", i), centroid(np.stack(vs))) for i, vs in by_doc.items()
                  if i in titles]
    if level in ("section", "both"):
        items += [("section", i, paths[i].get("path", i), centroid(np.stack(vs))) for i, vs in by_sec.items()
                  if i in paths]

    parents = await broader_parents(graph, cids)
    names = {c: p["name"] for c, p in concepts.items()}
    edges, top_level = [], Counter()
    kinds = {"doc": [], "section": []}
    for start in range(0, len(items), 4096):
        block = items[start:start + 4096]
        S = np.stack([it[3] for it in block]) @ C.T
        for (kind, iid, label, _), row in zip(block, S):
            kinds[kind].append(float(row.max()))
            for j in top_k(row, top, min_sim):
                edges.append((iid, cids[j], ABOUT, {"score": round(float(row[j]), 4),
                                                    "edge_text": f"{label} about {names[cids[j]]}"}))
            if kind == "doc":
                # the level-0 ancestor of the best concept, by the first parent at each step
                root = cids[int(np.argmax(row))]
                while parents.get(root):
                    root = parents[root][0]
                top_level[names[root]] += 1

    deleted = await replace_about(cids, edges)
    return {
        "scheme": scheme, "level": level, "top": top, "min_sim": min_sim, "concepts": len(cids),
        "classified": {k: len(v) for k, v in kinds.items()},
        "docs_without_passages": n_docs_all - len(titles),
        "passages_without_vector": no_vector,
        "mean_best_sim": {k: round(float(np.mean(v)), 4) if v else None for k, v in kinds.items()},
        "edges_written": len(edges), "edges_replaced": deleted,
        "docs_per_top_level_concept": dict(top_level.most_common()),
        "seconds": round(time.time() - t, 1),
    }


def classify_main(argv=None):
    ap = argparse.ArgumentParser(prog="docmem classify", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--scheme", help="the scheme to classify with (needed when the index has more than one)")
    ap.add_argument("--top", type=int, default=3, help="concepts kept per item")
    ap.add_argument("--min-sim", type=float, default=0.0, help="cosine similarity a concept must reach")
    ap.add_argument("--level", choices=("doc", "section", "both"), default="doc",
                    help="classify documents, sections, or both")
    a = ap.parse_args(argv)
    store.configure(a.db)
    print(json.dumps(asyncio.run(classify(a.scheme, a.top, a.min_sim, a.level)), indent=1))


# ---- concepts ----------------------------------------------------------------------------------------

async def show(scheme: str | None, doc: str | None, tree: bool):
    graph, _ = await store.engines()
    scheme, concepts = await scheme_concepts(graph, scheme)
    cids = list(concepts)
    parents = await broader_parents(graph, cids)
    names = {c: p["name"] for c, p in concepts.items()}
    lv = {c: p.get("level", 0) for c, p in concepts.items()}

    def ancestors(c: str) -> list[str]:
        out, stack = [], [c]
        while stack:
            for p in parents.get(stack.pop(), []):
                if p not in out:
                    out.append(p)
                    stack.append(p)
        return out

    if tree:
        by_concept = await docs_with_concepts(graph, cids)
        under: dict[str, set] = defaultdict(set)  # a concept's documents: those assigned to it or below it
        for c, fids in by_concept.items():
            for anc in [c] + ancestors(c):
                under[anc].update(fids)
        children: dict[str, list[str]] = defaultdict(list)
        for c in cids:
            for p in parents.get(c, []):
                children[p].append(c)
        for top in sorted((c for c in cids if lv[c] == 0), key=lambda c: (-len(under[c]), names[c])):
            print(f"{len(under[top]):6d}  {names[top]}")
            for ch in sorted(children[top], key=lambda c: (-len(under[c]), names[c])):
                if lv[ch] == 1:
                    print(f"{len(under[ch]):8d}    {names[ch]}")
        return

    docs = [n for n in await nodes_of_type(graph, "DocFile") if n.get("title", "").lower() == doc.lower()]
    if not docs:
        raise SystemExit(f"docmem concepts: no document titled {doc!r}")
    rows = await graph.query(
        "MATCH (f:Node)-[r:EDGE]->(c:Node) WHERE f.id IN $ids AND r.relationship_name = $rel "
        "RETURN f.id, c.id, r.properties", {"ids": [d["id"] for d in docs], "rel": ABOUT})
    per_doc: dict[str, list] = defaultdict(list)
    for f, c, props in rows:
        p = json.loads(props) if isinstance(props, str) else (props or {})
        per_doc[str(f)].append((p.get("score", 0.0), str(c)))
    for d in docs:
        print(f"{d['title']}  ({d['name']})")
        for score, c in sorted(per_doc.get(d["id"], []), reverse=True):
            chain = " > ".join(names[a] for a in reversed(ancestors(c)) if a in names) + f" > {names[c]}"
            print(f"  {score:.3f}  {names[c]}  [level {lv[c]}]  {chain}")


def concepts_main(argv=None):
    ap = argparse.ArgumentParser(prog="docmem concepts", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--scheme", help="the scheme (needed when the index has more than one)")
    ap.add_argument("--doc", help="show this document's concepts, with scores and ancestor chains")
    ap.add_argument("--tree", action="store_true",
                    help="the top two levels, with the number of documents under each concept")
    a = ap.parse_args(argv)
    if bool(a.doc) == a.tree:
        raise SystemExit("docmem concepts: give --doc TITLE or --tree")
    store.configure(a.db)
    asyncio.run(show(a.scheme, a.doc, a.tree))
