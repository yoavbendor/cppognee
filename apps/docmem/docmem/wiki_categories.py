"""Wikipedia's category tree as a SKOS concept scheme, for docmem's ontology loader.

    docmem wiki-categories --db DB --cache DIR --out FILE.ttl [--depth 2]

The concepts are the categories the indexed documents are in (DocFile -in_category-> DocTarget). Each category's
page is read from the dump through the index, as scan-wiki reads articles, and its [[Category:...]] links are its
parents. --depth N reads the pages of the categories up to N levels from the documents: depth 2 keeps the
document categories, their parents and their grandparents, with the broader edges among them (a grandparent's
own parents are not read, so the top level has no edges up). A category whose page is missing from the dump is
kept without parents. A category that is a redirect is followed once, and its target is the concept. Maintenance
categories (MAINTENANCE, and any name with "stub" or "articles needing") are dropped. A cycle is broken: the
broader edge that closes it is dropped and counted.

Output, FILE.ttl: SKOS Turtle, each category a skos:Concept with skos:prefLabel (its name) and skos:broader to its
kept parents. There is no skos:definition: ontology.load embeds the label alone. FILE.doc_categories.json maps each
indexed document title to its kept direct categories, for evaluation. Category pages are read in memory only.
"""

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

import mwparserfromhell

from . import store
from .links import wiki_target
from .ontology import nodes_of_type
from .scan_wiki import DumpSource, Index, dump_size, ensure_index, redirect_of, log

MAINTENANCE = (
    "Articles", "All articles", "Pages", "CS1", "Wikipedia", "Webarchive", "Commons", "Use dmy", "Use mdy",
    "Short description", "Harv", "Accuracy disputes", "Vague or ambiguous", "Hidden categories",
)
CATEGORY_NS = "14"
URI_BASE = "https://en.wikipedia.org/wiki/Category:"


def is_maintenance(name: str) -> bool:
    """A maintenance or tracking category: its name is one of MAINTENANCE or starts with one of them as whole
    words ("Harvard ..." is not "Harv ..."), or contains "stub" or "articles needing"."""
    low = name.lower()
    if "stub" in low or "articles needing" in low:
        return True
    return any(low == p.lower() or low.startswith(p.lower() + " ") for p in MAINTENANCE)


def parent_names(wikitext: str) -> list[str]:
    """The categories a category page is in: its [[Category:...]] links, each once. A link written with a leading
    colon ([[:Category:...]]) links to the category and is not a membership."""
    names = []
    for link in mwparserfromhell.parse(wikitext.replace("\r", "")).filter_wikilinks():
        raw = str(link.title)
        if raw.lstrip().startswith(":"):
            continue
        kind, name, _ = wiki_target(raw)
        if kind == "category" and name:
            names.append(name)
    return list(dict.fromkeys(names))


def break_cycles(parents: dict[str, list[str]]) -> tuple[dict[str, list[str]], int]:
    """parents: child -> parent names (an edge per pair). A depth-first walk from the sorted names keeps every edge
    except the back edges, each of which closes a cycle (a self-loop is one). Returns the kept edges and the number
    of back edges dropped; the kept edges form no cycle."""
    keep: dict[str, list[str]] = {}
    dropped = 0
    state: dict[str, int] = {}  # 1 on the walk's stack, 2 finished
    for root in sorted(parents):
        if root in state:
            continue
        state[root] = 1
        stack = [(root, iter(sorted(parents[root])))]
        while stack:
            node, children = stack[-1]
            for p in children:
                if state.get(p) == 1:
                    dropped += 1
                    continue
                keep.setdefault(node, []).append(p)
                if p not in state:
                    state[p] = 1
                    stack.append((p, iter(sorted(parents.get(p, [])))))
                    break
            else:
                state[node] = 2
                stack.pop()
    return keep, dropped


def fetch_pages(source: DumpSource, names: list[str]) -> dict[str, tuple]:
    """For each name, what its category page says: ("page", parent names), ("redirect", target name) or
    ("missing",). Pages are read through the index in one pass and one read of their streams."""
    located = source.locate({"Category:" + n for n in names})
    out: dict[str, tuple] = {n: ("missing",) for n in names}
    name_of: dict[int, str] = {}
    wanted: dict[int, set[int]] = {}
    for n in names:
        if "Category:" + n in located:
            stream, pid = located["Category:" + n]
            name_of[pid] = n
            wanted.setdefault(stream, set()).add(pid)

    def take(pid: int, rec: dict | None) -> None:
        if rec is None or rec["ns"] != CATEGORY_NS:
            return
        dest = redirect_of(rec["text"])
        if dest is None:
            out[name_of[pid]] = ("page", parent_names(rec["text"]))
        elif dest[0] == "category" and dest[1]:
            out[name_of[pid]] = ("redirect", dest[1])
        else:
            out[name_of[pid]] = ("page", [])

    source.read(wanted, take)
    return out


def read_categories(source: DumpSource, names: list[str]) -> dict[str, dict]:
    """{name: {"canon", "parents", "missing"}}: the concept a category name stands for (its redirect target, followed
    once; the name itself otherwise), its parents, and whether its page is missing from the dump."""
    first = fetch_pages(source, names)
    hop = {n: got[1] for n, got in first.items() if got[0] == "redirect"}
    second_names = sorted({d for d in hop.values() if d not in first})
    second = fetch_pages(source, second_names) if second_names else {}
    out: dict[str, dict] = {}
    for n, got in first.items():
        if got[0] == "page":
            out[n] = {"canon": n, "parents": got[1], "missing": False}
        elif got[0] == "missing":
            out[n] = {"canon": n, "parents": [], "missing": True}
    for n, d in hop.items():
        got = first[d] if d in first else second[d]
        if got[0] == "page":
            out[n] = {"canon": d, "parents": got[1], "missing": False}
            out.setdefault(d, {"canon": d, "parents": got[1], "missing": False})
        else:  # a missing target or a redirect to a redirect: n stays a concept without parents
            out[n] = {"canon": n, "parents": [], "missing": True}
    return out


def walk(source: DumpSource, doc_cats: set[str], depth: int) -> tuple[dict[str, dict], dict[str, int]]:
    """Read the pages of the categories up to depth levels from the documents. Returns (entries by name as
    read_categories gives them, the distance from the documents of each name seen)."""
    dist = {n: 0 for n in doc_cats}
    entries: dict[str, dict] = {}
    for d in range(depth):
        layer = sorted(n for n, k in dist.items() if k == d and n not in entries)
        if not layer:
            break
        entries.update(read_categories(source, layer))
        log(f"level {d}: {len(layer)} category pages read")
        for n in layer:
            e = entries[n]
            if e["canon"] != n:
                dist.setdefault(e["canon"], d)
            for p in e["parents"]:
                if p not in dist and not is_maintenance(p):
                    dist[p] = d + 1
    return entries, dist


async def document_categories() -> dict[str, set[str]]:
    """{document title: the category names its DocFile links to with in_category}."""
    import cognee

    await cognee.run_migrations()
    graph, _ = await store.engines()
    docs = await nodes_of_type(graph, "DocFile")
    rows = await graph.query(
        "MATCH (f:Node)-[r:EDGE]->(t:Node) WHERE r.relationship_name = 'in_category' RETURN f.id, t.id")
    tids = sorted({str(t) for _, t in rows})
    names = {str(n["id"]): store.node_props(n)["name"] for n in await graph.get_nodes(tids)} if tids else {}
    cats: dict[str, set[str]] = defaultdict(set)
    for f, t in rows:
        cats[str(f)].add(names[str(t)])
    return {d["title"]: cats.get(d["id"], set()) for d in docs}


def write_turtle(path: Path, nodes: list[str], parents: dict[str, list[str]]) -> None:
    from rdflib import RDF, Graph, Literal, Namespace, URIRef

    skos = Namespace("http://www.w3.org/2004/02/skos/core#")
    g = Graph()
    g.bind("skos", skos)

    def uri(name: str) -> URIRef:
        return URIRef(URI_BASE + quote(name.replace(" ", "_"), safe="/:()',!*+;=&$@-._~"))

    for name in nodes:
        g.add((uri(name), RDF.type, skos.Concept))
        g.add((uri(name), skos.prefLabel, Literal(name, lang="en")))
    for child, ps in parents.items():
        for p in ps:
            g.add((uri(child), skos.broader, uri(p)))
    g.serialize(destination=str(path), format="turtle")


def build(docs: dict[str, set[str]], source: DumpSource, depth: int) -> dict:
    """The concept scheme for the documents' categories: nodes with their level, broader edges without cycles,
    and the documents' kept direct categories."""
    doc_cats = {n for names in docs.values() for n in names if not is_maintenance(n)}
    entries, dist = walk(source, doc_cats, depth)
    canon_of = {n: (entries[n]["canon"] if n in entries else n) for n in dist}
    level: dict[str, int] = {}
    for n, k in dist.items():
        c = canon_of[n]
        level[c] = min(level.get(c, k), k)
    raw: dict[str, list[str]] = {}
    for c in sorted(level):
        e = entries.get(c)
        parents = []
        for p in (e["parents"] if e else []):
            cp = canon_of.get(p, p)
            if cp in level:  # a self-loop is kept here and counted as a cycle by break_cycles
                parents.append(cp)
        raw[c] = list(dict.fromkeys(parents))
    keep, cycles = break_cycles(raw)
    keep = {c: ps for c, ps in keep.items() if ps}
    doc_out = {t: sorted({canon_of[n] for n in names if not is_maintenance(n)}) for t, names in docs.items()}
    return {
        "nodes": sorted(level),
        "level": level,
        "parents": keep,
        "cycles": cycles,
        "doc_categories": doc_out,
        "dropped_maintenance": len({n for names in docs.values() for n in names if is_maintenance(n)}),
        "missing_pages": sum(1 for e in entries.values() if e["missing"]),
        "redirects": sum(1 for n, e in entries.items() if e["canon"] != n),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docmem wiki-categories", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True, help="an index whose documents' categories to read")
    ap.add_argument("--cache", type=Path, required=True, help="directory for the dump index")
    ap.add_argument("--depth", type=int, default=2, help="levels of parent categories above the documents' own")
    ap.add_argument("--out", type=Path, required=True, help="the SKOS Turtle file to write")
    a = ap.parse_args(argv)
    if a.depth < 0:
        raise SystemExit("docmem wiki-categories: --depth must be 0 or more")
    store.configure(a.db)
    t0 = time.time()
    docs = asyncio.run(document_categories())
    log(f"{len(docs)} documents; {len({n for v in docs.values() for n in v})} distinct categories")

    path = ensure_index(a.cache)
    index = Index(path)
    stats = {"streams": 0, "bytes": 0, "draws": 0}
    source = DumpSource(index, path, dump_size(), stats)
    built = build(docs, source, a.depth)

    write_turtle(a.out, built["nodes"], built["parents"])
    doc_path = a.out.with_name(a.out.stem + ".doc_categories.json")
    doc_path.write_text(json.dumps(built["doc_categories"], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    report = {
        "documents": len(docs),
        "concepts": len(built["nodes"]),
        "concepts_per_level": dict(sorted(Counter(built["level"].values()).items())),
        "broader_edges": sum(len(v) for v in built["parents"].values()),
        "cycles_broken": built["cycles"],
        "missing_pages": built["missing_pages"],
        "redirects_followed": built["redirects"],
        "maintenance_dropped": built["dropped_maintenance"],
        "dump_streams": stats["streams"],
        "dump_mb": round(stats["bytes"] / 2**20, 1),
        "seconds": round(time.time() - t0, 1),
        "turtle": str(a.out),
        "doc_categories": str(doc_path),
    }
    print(json.dumps(report), file=sys.stdout)
