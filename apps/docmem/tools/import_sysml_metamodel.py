"""Import the SysML v2 and KerML abstract syntax as a SKOS concept scheme, for `docmem ontology load`.

    python tools/import_sysml_metamodel.py --source PATH --out FILE.ttl

PATH is a checkout of github.com/Systems-Modeling/SysML-v2-Release (or its sysml.library directory). The release
has no .ecore or .uml metamodel. Its abstract syntax is in the reflective libraries instead: KerML.kerml and
Metaobjects.kerml declare the KerML metaclasses ("a reflective KerML model of the KerML abstract syntax"), and
SysML.sysml declares the SysML ones ("a reflective SysML model of the SysML abstract syntax"). Each metaclass
becomes one skos:Concept (URI urn:sysml:meta:<qualified name>):

  skos:prefLabel    the metaclass name
  skos:definition   the text of its doc comment, or its qualified name when it has none (the class path)
  skos:broader      each supertype that is itself a metaclass (multiple inheritance keeps all of them)
  skos:scopeNote    the library, the qualified name and file:line

Supertypes are resolved with docmem's own name resolution (docmem.sysml_resolve), over every library file. A
supertype that is not a metaclass (KerML's Objects::Object, for instance, is a library classifier) is not a
concept of this scheme; it is counted and listed, not kept as a broader edge. A supertype that does not resolve
stops the import.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from rdflib import RDF, Graph, Literal, Namespace, URIRef

from docmem.scan_sysml import iter_files, parse_model
from docmem.sysml_resolve import Universe

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
BASE = "urn:sysml:meta:"
REFLECTIVE = {"KerML.kerml": "KerML", "Metaobjects.kerml": "KerML", "SysML.sysml": "SysML"}
METACLASS_KINDS = {"metaclass", "abstract metaclass", "metadata def", "abstract metadata def"}


def doc_text(f, el) -> str:
    """The element's own doc comment, as plain text ("" when it has none)."""
    for s, e, header in sorted(el.pieces):
        if header or f.data[s:s + 3] != b"doc":
            continue
        body = f.data[s + 3:e].decode("utf-8", "replace").strip()
        body = body.removeprefix("/*").removesuffix("*/")
        lines = [ln.strip().removeprefix("*").strip() for ln in body.splitlines()]
        return " ".join(" ".join(lines).split())
    return ""


def load_universe(source: Path):
    sources = []
    for path in iter_files([source]):
        data = path.read_bytes()
        sources.append(({"uri": path.resolve().as_uri()}, data))
    errors: list[str] = []
    model = parse_model(sources, "metamodel", errors)
    if errors:
        raise SystemExit("import_sysml_metamodel: files failed to parse:\n" + "\n".join(errors))
    return model.files, Universe(model.files)


def build(source: Path):
    files, uni = load_universe(source)
    meta: dict[str, tuple] = {}
    for f in files:
        lib = REFLECTIVE.get(Path(f.uri).name.replace("%20", " "))
        if lib is None:
            continue
        for el in f.elements:
            if el.kind in METACLASS_KINDS:
                meta[el.qname] = (f, el, lib)
    broader: dict[str, list[str]] = defaultdict(list)
    dropped = []
    for qn, (f, el, _) in meta.items():
        for r in uni.header_refs.get(id(el), []):
            if r.fragment != "specializes":
                continue
            res = uni.resolve(r.text, r.chain)
            if res.target is None:
                raise SystemExit(f"import_sysml_metamodel: {qn} specializes {r.text!r}, which does not resolve: "
                                 f"{res.why} ({f.uri}:{f.line_of(r.start)})")
            if res.target in meta:
                if res.target not in broader[qn]:
                    broader[qn].append(res.target)
            else:
                dropped.append((qn, res.target))
    return meta, broader, dropped


def to_graph(meta, broader) -> Graph:
    g = Graph()
    g.bind("skos", SKOS)
    for qn in sorted(meta):
        f, el, lib = meta[qn]
        c = URIRef(BASE + quote(qn, safe=":"))
        g.add((c, RDF.type, SKOS.Concept))
        g.add((c, SKOS.prefLabel, Literal(el.name, lang="en")))
        g.add((c, SKOS.definition, Literal(doc_text(f, el) or qn, lang="en")))
        g.add((c, SKOS.scopeNote, Literal(f"{lib} library: {qn} ({Path(f.uri).name}:{el.line})", lang="en")))
        for p in broader.get(qn, []):
            g.add((c, SKOS.broader, URIRef(BASE + quote(p, safe=":"))))
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True,
                    help="the SysML-v2-Release checkout, or its sysml.library directory")
    ap.add_argument("--out", type=Path, required=True, help="the Turtle file to write")
    a = ap.parse_args(argv)
    src = a.source / "sysml.library" if (a.source / "sysml.library").is_dir() else a.source
    meta, broader, dropped = build(src)
    g = to_graph(meta, broader)
    g.serialize(destination=str(a.out), format="turtle")

    labels = Counter(el.name for _, el, _ in meta.values())
    depth: dict[str, int] = {}

    def level(qn: str, path=frozenset()) -> int:
        if qn not in depth:
            if qn in path:
                raise SystemExit(f"import_sysml_metamodel: broader cycle through {qn}")
            depth[qn] = 0 if not broader.get(qn) else 1 + min(level(p, path | {qn}) for p in broader[qn])
        return depth[qn]

    for qn in meta:
        level(qn)
    print(json.dumps({
        "out": str(a.out),
        "metaclasses": len(meta),
        "by_library": dict(Counter(lib for _, _, lib in meta.values())),
        "with_doc_comment": sum(1 for f, el, _ in meta.values() if doc_text(f, el)),
        "broader_edges": sum(len(v) for v in broader.values()),
        "multiple_supertypes": sum(1 for v in broader.values() if len(v) > 1),
        "roots": sum(1 for qn in meta if not broader.get(qn)),
        "levels": dict(sorted(Counter(depth.values()).items())),
        "supertypes_that_are_not_metaclasses": dict(Counter(t for _, t in dropped).most_common()),
        "duplicate_labels": {k: v for k, v in labels.items() if v > 1},
    }, indent=1))


if __name__ == "__main__":
    sys.exit(main())
