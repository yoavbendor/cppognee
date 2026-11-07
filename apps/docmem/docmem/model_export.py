"""Export a model DB's resolved model as a nomograph index, for sysmldiag and nomograph-sysml. Model DBs only.

    docmem model-export --db MODELDB --model apollo11 --out index.json [--include-library]

The file has the shape ``nomograph-sysml index`` writes, ``{"elements": [...], "relationships": [...]}``, so
``nomograph-sysml stat|check|inspect|trace|query --index`` and ``sysmldiag --index`` read it. Everything comes from the
DB: elements from the DocSection nodes (qualified names from their heading paths, spans from their byte ranges), the
relationships from the resolved edges that ``model-resolve`` wrote (``r_typed_by`` and the rest, with their targets
fully qualified), Member from each element's owner, and the doc comment of an element from its own passages, read back
from the source file (the DB keeps no passage text).

Library elements (the other models in the DB, e.g. sysml-library) are exported only where a model element refers to
them, with their owners so the Member tree holds; ``--include-library`` exports all of them. Library relationships
are exported only with ``--include-library``.

A SysML keyword that KEYWORD_KIND does not list is exported as ``generic_usage`` (nomograph's catch-all) and counted
in the report. The layer follows nomograph's own rule, which is by kind (LAYER_OF_KIND).

Spans: an element's span covers its declaration (columns in bytes, end exclusive). A relationship's span is the
whole line of the reference, since the DB keeps the line but not the columns.
"""

import argparse
import asyncio
import bisect
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import store
from .ontology import nodes_of_type
from .sysml_resolve import _edges

# docmem's relation fragments (resolved edges) -> nomograph's relationship kinds
TYPE_CHAIN = {"TypedBy", "Specialize", "Redefine"}  # nomograph kinds that carry a type
REL_KIND = {
    "r_typed_by": "TypedBy",
    "r_specializes": "Specialize",
    "r_subsets": "Specialize",
    "r_redefines": "Redefine",
    "r_satisfy": "Satisfy",
    "r_perform": "Perform",
    "r_allocate": "Allocate",
}
MEMBER = "Member"
FALLBACK_KIND = (
    "generic_usage"  # nomograph's catch-all kind, for a keyword KEYWORD_KIND does not list
)

# the scanner's kind (keywords, modifiers in front) -> nomograph's element kind
KEYWORD_KIND = {
    "package": "package_definition",
    "library package": "library_package",
    "part def": "part_definition",
    "part": "part_usage",
    "item def": "item_definition",
    "item": "item_usage",
    "port def": "port_definition",
    "port": "port_usage",
    "interface def": "interface_definition",
    "interface": "interface_usage",
    "connection def": "connection_definition",
    "connection": "connection_usage",
    "action def": "action_definition",
    "action": "action_usage",
    "state def": "state_definition",
    "state": "state_usage",
    "calc def": "calc_definition",
    "calc": "calc_usage",
    "analysis def": "analysis_definition",
    "analysis": "analysis_usage",
    "requirement def": "requirement_definition",
    "requirement": "requirement_usage",
    "attribute def": "attribute_definition",
    "attribute": "attribute_usage",
    "enum def": "enumeration_definition",
    "view": "view_usage",
    "snapshot": "snapshot_usage",
    "timeslice": "timeslice_usage",
    "end": "end_usage",
    "exhibit state": "exhibit_usage",
    "perform action": "action_usage",  # a PerformActionUsage is an action usage
    # a usage with no keyword (x :>> y = 5, or a KerML feature): the Feature that docmem's metaclass also gives it
    "redefinition": "feature_usage",
    "specialization": "feature_usage",
    "feature": "feature_usage",
}
# a direction word starts a parameter: in x, out item y, inout z, return w
PARAMETER_WORDS = frozenset({"in", "out", "inout", "return"})
# modifiers that change no kind; "then" and "do" are sequence markers in front of a usage
ANNOTATION_WORDS = frozenset(
    {
        "abstract",
        "individual",
        "variation",
        "ref",
        "derived",
        "readonly",
        "portion",
        "constant",
        "var",
        "const",
        "composite",
        "standard",
        "all",
        "first",
        "then",
        "do",
        "private",
        "public",
        "protected",
    }
)
# the layer of a kind, as nomograph assigns it (a kind not listed has none)
LAYER_OF_KIND = {
    **{k: "Requirements" for k in ("requirement_definition", "requirement_usage")},
    **{
        k: "Functional"
        for k in (
            "action_definition",
            "action_usage",
            "calc_definition",
            "calc_usage",
            "state_definition",
            "state_usage",
            "analysis_definition",
            "analysis_usage",
            "exhibit_usage",
        )
    },
    **{
        k: "Logical"
        for k in (
            "part_definition",
            "part_usage",
            "item_definition",
            "item_usage",
            "port_definition",
            "port_usage",
            "interface_definition",
            "interface_usage",
            "connection_definition",
            "connection_usage",
            "attribute_definition",
            "attribute_usage",
            "end_usage",
            "parameter_usage",
        )
    },
}
LABEL = re.compile(
    r"^(?P<kind>.*?)\s+(?P<name>'(?:[^'\\]|\\.)*'|\S+)$"
)  # "kind name" of a section's last label
DOC_COMMENT = re.compile(rb"\bdoc\s*/\*(.*?)\*/", re.S)
PATH_SEP = " › "


def nomo_kind(raw: str) -> str | None:
    """The nomograph kind of a scanner kind, or None when no keyword in KEYWORD_KIND matches it."""
    words = [w for w in raw.split() if w not in ANNOTATION_WORDS]
    if not words:
        return None
    if words[0] in PARAMETER_WORDS:
        return "parameter_usage"
    return KEYWORD_KIND.get(" ".join(words))


def layer_of(kind: str) -> str | None:
    return LAYER_OF_KIND.get(kind)


def parse_label(label: str) -> tuple[str, str]:
    """(scanner kind, name) of a section label such as "part def Car" or "attribute 'per cent'". The name's quotes go."""
    m = LABEL.match(label)
    if m is None:
        raise ValueError(f"not a declaration label: {label!r}")
    name = m.group("name")
    if len(name) > 1 and name[0] == name[-1] == "'":
        name = name[1:-1]
    return m.group("kind"), name


def split_path(path: str) -> tuple[str, str, str]:
    """(model, qualified name, scanner kind) of a section's heading path: ["model", NAME, label, ...]."""
    parts = path.split(PATH_SEP)
    if len(parts) < 3 or parts[0] != "model":
        raise ValueError(f"not a model section path: {path!r}")
    parsed = [parse_label(p) for p in parts[2:]]
    return parts[1], "::".join(name for _, name in parsed), parsed[-1][0]


def line_starts(data: bytes) -> list[int]:
    return [0] + [m.end() for m in re.finditer(rb"\n", data)]


def _line_col(starts: list[int], offset: int) -> tuple[int, int]:
    line = bisect.bisect_right(starts, offset) - 1
    return line, offset - starts[line]


def element_span(starts: list[int], offset: int, length: int) -> tuple[int, int, int, int]:
    """(start_line, start_col, end_line, end_col), lines from 0, columns in bytes, the end exclusive."""
    sl, sc = _line_col(starts, offset)
    el, ec = _line_col(starts, offset + length - 1)
    return sl, sc, el, ec + 1


def line_span(data: bytes, starts: list[int], line1: int) -> tuple[int, int, int, int]:
    """The span of a relationship's line (1-based in the DB): from its first to its last non-blank byte."""
    line0 = line1 - 1
    end = starts[line0 + 1] - 1 if line0 + 1 < len(starts) else len(data)
    text = data[starts[line0] : end].rstrip(b"\r\n")
    return line0, len(text) - len(text.lstrip()), line0, len(text.rstrip())


def doc_comment(data: bytes) -> str | None:
    m = DOC_COMMENT.search(data)
    return m.group(1).decode("utf-8", "replace").strip() if m else None


@dataclass
class Element:
    sid: str  # the DocSection id
    model: str
    qname: str
    raw_kind: str
    uri: str
    span: tuple
    doc: str | None = None


@dataclass
class Relation:
    source: str  # sid
    target: str  # sid
    rel: str  # nomograph kind
    uri: str
    span: tuple


@dataclass
class Export:
    index: dict
    report: dict
    section_of: dict = field(
        default_factory=dict
    )  # (model, qualified name) -> sid of the first such element


def select(
    elements: list[Element],
    relations: list[Relation],
    model: str,
    include_library: bool,
    owner_of: dict[str, str],
) -> set[str]:
    """The sids to export: the model's elements, the library elements they refer to and their owners (or every
    element with include_library)."""
    if include_library:
        return {e.sid for e in elements}
    keep = {e.sid for e in elements if e.model == model}
    for r in relations:
        if r.source in keep:
            keep.add(r.target)
    # the type chain of every referred library element (its typing, specialization, redefinition),
    # so a view can name the type of `attribute dryMass :> ISQ::mass;`
    chain = [r for r in relations if r.rel in TYPE_CHAIN]
    grew = True
    while grew:
        grew = False
        for r in chain:
            if r.source in keep and r.target not in keep:
                keep.add(r.target)
                grew = True
    frontier = list(keep)
    while frontier:  # the owners of the referred elements, so that the Member tree is whole
        owner = owner_of.get(frontier.pop())
        if owner is not None and owner not in keep:
            keep.add(owner)
            frontier.append(owner)
    return keep


def build_index(
    elements: list[Element],
    relations: list[Relation],
    model: str,
    include_library: bool,
) -> Export:
    """The nomograph index of one model (see the module docstring). file_path is relative to the directory of the
    model's files; a file outside it keeps its absolute path."""
    by_qname: dict[tuple[str, str], list[str]] = defaultdict(list)
    for e in elements:
        by_qname[(e.model, e.qname)].append(e.sid)
    owner_of = {}
    for e in elements:
        parent = e.qname.rpartition("::")[0]
        if parent and (e.model, parent) in by_qname:
            owner_of[e.sid] = by_qname[(e.model, parent)][0]
    el_of = {e.sid: e for e in elements}
    keep = select(elements, relations, model, include_library, owner_of)
    kept = sorted((el_of[s] for s in keep), key=lambda e: (e.uri, e.span[0], e.span[1], e.qname))
    root = _root_of(model, kept)

    unmapped: Counter = Counter()
    records, members = [], defaultdict(list)
    kinds = Counter()
    for e in kept:
        nomo = nomo_kind(e.raw_kind)
        if nomo is None:
            unmapped[e.raw_kind] += 1
            nomo = FALLBACK_KIND
        kinds[nomo] += 1
        if e.sid in owner_of and owner_of[e.sid] in keep:
            members[owner_of[e.sid]].append(e)
        rec = {
            "qualified_name": e.qname,
            "kind": nomo,
            "file_path": _file_path(e.uri, root),
            "span": _span_dict(e.span),
            "doc": e.doc,
            "attributes": [],
            "members": [],
        }
        layer = layer_of(nomo)
        if layer is not None:
            rec["layer"] = layer
        records.append(rec)
    qname_of = {e.sid: e.qname for e in elements}
    for rec, e in zip(records, kept):
        rec["members"] = [
            qname_of[c.sid]
            for c in sorted(members.get(e.sid, []), key=lambda c: (c.uri, c.span[0], c.span[1]))
        ]

    rels = []
    rel_kinds = Counter()
    for m in sorted(members):
        for c in members[m]:
            rels.append(
                {
                    "source": qname_of[m],
                    "target": c.qname,
                    "kind": MEMBER,
                    "file_path": _file_path(c.uri, root),
                    "span": _span_dict(c.span),
                }
            )
    for r in relations:
        if r.source not in keep or r.target not in keep:
            continue
        if not include_library and el_of[r.source].model != model and r.rel not in TYPE_CHAIN:
            continue
        rel_kinds[r.rel] += 1
        rels.append(
            {
                "source": qname_of[r.source],
                "target": qname_of[r.target],
                "kind": r.rel,
                "file_path": _file_path(r.uri, root),
                "span": _span_dict(r.span),
            }
        )
    report = {
        "model": model,
        "root": root,
        "elements": len(records),
        "relationships": len(rels),
        "elements_by_kind": dict(kinds.most_common()),
        "member_relationships": sum(len(v) for v in members.values()),
        "relationships_by_kind": dict(sorted(rel_kinds.items())),
        "library_elements": sum(1 for e in kept if e.model != model),
        "unmapped_keywords": dict(unmapped.most_common()),
        "duplicate_qualified_names": sum(
            1 for q in {(e.model, e.qname) for e in kept} if len(by_qname[q]) > 1
        ),
    }
    index = {"elements": records, "relationships": rels}
    return Export(index, report, {k: v[0] for k, v in by_qname.items()})


def _root_of(model: str, kept: list[Element]) -> str:
    """The directory of the model's files: the common parent of its files."""
    dirs = [str(store.uri_path(e.uri).parent) for e in kept if e.model == model]
    return os.path.commonpath(dirs) if dirs else ""


def _file_path(uri: str, root: str) -> str:
    """./relative/path under the model's root (nomograph's form), else the absolute path."""
    path = store.uri_path(uri)
    rel = os.path.relpath(path, root) if root else str(path)
    return str(path) if rel.startswith("..") else "./" + rel.replace(os.sep, "/")


def _span_dict(span: tuple) -> dict:
    sl, sc, el, ec = span
    return {"start_line": sl, "start_col": sc, "end_line": el, "end_col": ec}


async def read_db(graph) -> tuple[list[Element], list[Relation]]:
    """Every section of the DB as an Element (its span and doc comment read from its file) and every resolved
    relation as a Relation. The file bytes are checked against the SHA-256 the DB stores: a file that changed since
    the scan is refused, as model-resolve refuses it."""
    files = {f["name"]: f for f in await nodes_of_type(graph, "DocFile")}
    sections = await nodes_of_type(graph, "DocSection")
    passages = await nodes_of_type(graph, "DocPassage")
    data, starts = {}, {}

    def source(uri: str) -> tuple[bytes, list[int]]:
        if uri not in data:
            if uri not in files:
                raise SystemExit(f"docmem model-export: no DocFile for {uri}")
            raw = store.uri_path(uri).read_bytes()
            if hashlib.sha256(raw).hexdigest() != files[uri]["sha256"]:
                raise SystemExit(
                    f"docmem model-export: {uri} changed since it was indexed; re-scan and re-index"
                )
            data[uri], starts[uri] = raw, line_starts(raw)
        return data[uri], starts[uri]

    by_path: dict[str, list[dict]] = defaultdict(list)
    for p in passages:
        by_path[p["path"]].append(p)
    elements, sid_info = [], {}
    for s in sections:
        model, qname, raw_kind = split_path(s["path"])
        raw, st = source(s["name"])
        doc = None
        for p in by_path.get(s["path"], []):
            pd, _ = source(p["name"])
            doc = doc or doc_comment(pd[int(p["offset"]) : int(p["offset"]) + int(p["length"])])
        el = Element(
            str(s["id"]),
            model,
            qname,
            raw_kind,
            s["name"],
            element_span(st, int(s["offset"]), int(s["length"])),
            doc,
        )
        elements.append(el)
        sid_info[el.sid] = el
    relations = []
    for rel, kind in REL_KIND.items():
        async for a, b, props in _edges(graph, rel):
            if a not in sid_info or b not in sid_info:
                raise SystemExit(
                    f"docmem model-export: a {rel} edge names a section that is not in the DB"
                )
            raw, st = source(props["file"])
            relations.append(
                Relation(a, b, kind, props["file"], line_span(raw, st, int(props["line"])))
            )
    return elements, relations


async def export_db(model: str, include_library: bool, graph=None) -> Export:
    if graph is None:
        graph, _ = await store.engines()
    elements, relations = await read_db(graph)
    if not any(e.model == model for e in elements):
        models = sorted({e.model for e in elements})
        raise SystemExit(
            f"docmem model-export: no elements of model {model!r} in this DB (models: {models})"
        )
    return build_index(elements, relations, model, include_library)


def export_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="docmem model-export",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--db", type=Path, required=True, help="a model DB (scan-sysml, index, model-resolve first)"
    )
    ap.add_argument("--model", required=True, help="the model's name (scan-sysml --model-name)")
    ap.add_argument("--out", type=Path, required=True, help="the nomograph index file to write")
    ap.add_argument(
        "--include-library",
        action="store_true",
        help="export every library element, not only the used",
    )
    a = ap.parse_args(argv)
    store.configure(a.db)
    result = asyncio.run(export_db(a.model, a.include_library))
    a.out.write_text(json.dumps(result.index, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result.report, indent=1, ensure_ascii=False))
    print(f"docmem model-export: wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    export_main()
