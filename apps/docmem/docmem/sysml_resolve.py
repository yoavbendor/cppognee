"""Name resolution for SysML v2 and KerML models in a model DB (never a prose DB).

    docmem model-resolve --db MODELDB              resolve the references, write resolved edges and metaclasses
    docmem model-show --db MODELDB "Qualified::Name"  one element: metaclass, owner, types, owned and inherited features

``model-resolve`` reads the graph: the sections of every model file (DocSection: the element's offset, path and
line). The link records keep anchors and fragments but no offsets, and an import keeps no visibility, so the
source files are parsed again with the scanner (they are public model files, read by their file:// URIs). Every
parsed element must match a stored section, or the command stops: the index is stale. Nothing calls an LLM.

Resolution follows SysML v2 and KerML as far as the references here need (the approximations are in the README):
- a name's head is looked up in the enclosing namespaces, innermost first. In each namespace the order is: owned
  members, then inherited members (the supertypes and typings, breadth first, so the most specific wins), then
  imports;
- ``import X::*`` brings in the public members of X, ``import X::**`` also those of the namespaces X owns, and
  ``import X::Y`` brings in Y alone. An import is private unless ``public import`` says otherwise; only public
  imports are re-exported to the namespaces that import from this one;
- a segment after ``::`` is looked up among the members of the namespace before it: its public members, unless the
  requester is inside it;
- a name that matches two different elements is ambiguous and stays unresolved, with the candidates reported.
"""

import argparse
import asyncio
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from . import store
from .ontology import _check_pipeline, broader_parents, nodes_of_type, scheme_concepts
from .scan_sysml import Element, Ref, _key, parse_model

SCHEME = "sysmlmeta"
TYPE_FRAGS = ("typed_by", "specializes", "subsets", "redefines")  # the source is the named element that owns them
# what an element inherits members from: its types, and the features it specializes, subsets or redefines (their
# members are its members too, as a redefinition of a part's part is that part's part)
SUPER_FRAGS = ("specializes", "typed_by", "subsets", "redefines")
SPLIT = re.compile(r"::|\.(?=(?:[^']*'[^']*')*[^']*$)")  # a path's separators, not those inside a quoted name
ROOT_TYPE = "Base::Anything"  # the top of the KerML types: what a type with no declared supertype specializes
# the library base of each SysML keyword (the implicit supertype of a definition or usage without a declared one)
DEFAULT_SUPERTYPE = {
    "part": "Parts::Part", "item": "Items::Item", "port": "Ports::Port", "action": "Actions::Action",
    "connection": "Connections::Connection", "calc": "Calculations::Calculation",
    "requirement": "Requirements::RequirementCheck",
}
PAIRED_FRAGS = ("satisfy", "satisfy_by", "allocate", "allocate_to")  # a statement's two halves, joined by stmt
WRITTEN = {"typed_by": "r_typed_by", "specializes": "r_specializes", "subsets": "r_subsets",
           "redefines": "r_redefines", "perform": "r_perform", "satisfy": "r_satisfy", "allocate": "r_allocate"}
METACLASS_REL = "metaclass"
EDGE_RELS = tuple(WRITTEN.values()) + (METACLASS_REL,)
EXAMPLES = 5  # unresolved references shown per fragment

# The scanner's kind (keywords, with the modifiers in front) -> the metaclass of the metamodel (sysml_metamodel.ttl).
KIND_METACLASS = {
    # KerML declarations
    "classifier": "Classifier", "feature": "Feature", "type": "Type", "datatype": "DataType", "class": "Class",
    "struct": "Structure", "assoc": "Association", "assoc struct": "AssociationStructure",
    "behavior": "Behavior", "function": "Function",
    "predicate": "Predicate", "connector": "Connector", "binding": "BindingConnector",
    "succession": "Succession", "metaclass": "Metaclass", "namespace": "Namespace", "package": "Package",
    "library package": "LibraryPackage", "expr": "Expression", "step": "Step", "interaction": "Interaction",
    # SysML definitions and usages
    "part def": "PartDefinition", "part": "PartUsage", "item def": "ItemDefinition", "item": "ItemUsage",
    "port def": "PortDefinition", "port": "PortUsage", "action def": "ActionDefinition", "action": "ActionUsage",
    "perform action": "PerformActionUsage", "state def": "StateDefinition", "state": "StateUsage",
    "exhibit state": "ExhibitStateUsage", "constraint def": "ConstraintDefinition", "constraint": "ConstraintUsage",
    # subject, actor and stakeholder have no usage metaclass in the reflective model (only memberships), so they stay
    # unmapped and are listed in the report
    "requirement def": "RequirementDefinition", "requirement": "RequirementUsage",
    "attribute def": "AttributeDefinition",
    "attribute": "AttributeUsage", "enum def": "EnumerationDefinition", "enum": "EnumerationUsage",
    "calc def": "CalculationDefinition", "calc": "CalculationUsage", "analysis def": "AnalysisCaseDefinition",
    "analysis": "AnalysisCaseUsage", "use case def": "UseCaseDefinition", "use case": "UseCaseUsage",
    "view def": "ViewDefinition", "view": "ViewUsage", "viewpoint def": "ViewpointDefinition",
    "viewpoint": "ViewpointUsage", "concern def": "ConcernDefinition", "concern": "ConcernUsage",
    "interface def": "InterfaceDefinition", "interface": "InterfaceUsage", "connection def": "ConnectionDefinition",
    "connection": "ConnectionUsage", "allocation def": "AllocationDefinition", "allocation": "AllocationUsage",
    "flow def": "FlowDefinition", "flow": "FlowUsage", "occurrence def": "OccurrenceDefinition",
    "occurrence": "OccurrenceUsage", "metadata def": "MetadataDefinition",
    "verification def": "VerificationCaseDefinition", "verification": "VerificationCaseUsage",
    "case def": "CaseDefinition", "case": "CaseUsage",
    # a usage the scanner names after what it redefines (:>> x = 5) or specializes (:> x): a Feature
    "redefinition": "Feature", "specialization": "Feature",
}
# Modifiers that change no metaclass: they are recorded as notes ("individual part def" is a PartDefinition).
NOTE_WORDS = frozenset({
    "abstract", "individual", "variation", "ref", "derived", "readonly", "portion", "constant", "var", "const",
    "composite", "end", "in", "out", "inout", "return", "first", "then", "do", "standard", "all",
})


def metaclass_for(kind: str) -> tuple[str | None, tuple[str, ...]]:
    """(metaclass name or None, notes) for a scanner kind. A kind with no keyword (a bare `in x : T`) is a Feature,
    as in KerML; a keyword not in KIND_METACLASS is None and is listed as unmapped."""
    words = kind.split()
    notes = tuple(w for w in words if w in NOTE_WORDS)
    base = " ".join(w for w in words if w not in NOTE_WORDS)
    if base == "":
        return "Feature", notes + ("no keyword",)
    return KIND_METACLASS.get(base), notes


def _type_like(el: Element) -> bool:
    """A declaration that is a type or feature: not a package, not an alias."""
    return el.kind != "alias" and not el.kind.endswith("package")


def _unique(items):
    seen, out = set(), []
    for x in items:
        k = x.qname if isinstance(x, Element) else x
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


@dataclass
class Resolution:
    target: str | None  # the qualified name, when exactly one element matches
    why: str  # the reason it did not resolve ("" when it did)


@dataclass
class Found:
    """One reference and its outcome: status resolved, unresolved or out_of_scope. rel is the edge written (when
    resolved), source the element it leaves, target the qualified name it points at."""

    fragment: str
    as_written: str
    uri: str
    line: int
    status: str
    why: str = ""
    rel: str = ""
    source: Element | None = None
    target: str | None = None


class Universe:
    """Every element and reference of a set of parsed files, with the name lookups of SysML v2 and KerML."""

    def __init__(self, files):
        self.files = files
        self.by_qname: dict[str, list[Element]] = defaultdict(list)
        self.children: dict[str, dict[str, list[Element]]] = defaultdict(lambda: defaultdict(list))
        self.in_file: dict[tuple[str, str], list[Element]] = defaultdict(list)
        self.header_refs: dict[int, list[Ref]] = defaultdict(list)  # id(element) -> refs in its header
        self.imports: dict[str, list[Ref]] = defaultdict(list)  # namespace qname -> its import refs
        self.elements: list[Element] = []
        self._supers: dict[str, list[str]] = {}
        self._busy_supers: set[str] = set()
        # Memos of member lookups and import results. A result computed while a cycle was cut (a name reached again
        # while it is being resolved) may be incomplete: _cuts counts the cuts. Such a result is kept only for the
        # top-level resolve() that computed it (_q*, cleared when the next one starts); the rest are kept for good.
        self._memo: dict = {}
        self._imp_memo: dict = {}
        self._qmemo: dict = {}
        self._qimp_memo: dict = {}
        self._cuts = 0
        for f in files:
            for el in f.elements:
                self.elements.append(el)
                self.by_qname[el.qname].append(el)
                self.children[el.parent][_key(el.name)].append(el)
                if el.alias is not None:  # a short name is a second key of the same element
                    self.children[el.parent][_key(el.alias)].append(el)
                self.in_file[(el.uri, el.qname)].append(el)
            for el in f.anon:  # a scope only: its names are its body's members, found through its parent's chain
                self.by_qname[el.qname].append(el)
                self.in_file[(el.uri, el.qname)].append(el)
            for r in f.refs:
                if r.fragment == "import":
                    self.imports[r.chain[-1]].append(r)
                elif r.owner is not None:
                    self.header_refs[id(r.owner)].append(r)
        self.alias_of = {id(r.owner): r for f in files for r in f.refs if r.fragment == "alias_for"}

    # -- lookups ----------------------------------------------------------------------------------------

    def owned(self, ns: str, key: str) -> list[Element]:
        return self.children.get(ns, {}).get(key, [])

    def supertypes(self, qname: str) -> list[str]:
        """The qualified names of qname's specializations and typings, in the order written. A cycle back to an
        element being resolved yields nothing for that inner step (it is not cached)."""
        if qname in self._supers:
            return self._supers[qname]
        if qname in self._busy_supers:
            self._cuts += 1
            return []
        self._busy_supers.add(qname)
        before = self._cuts
        out: list[str] = []
        declared = False
        try:
            for el in self.by_qname.get(qname, []):
                for r in self.header_refs.get(id(el), []):
                    if r.fragment in SUPER_FRAGS:
                        declared = True
                        res = self._resolve(r.text, r.chain, None, set())
                        if res.target is not None and res.target not in out:
                            out.append(res.target)
            first = self.by_qname.get(qname)
            if first and not declared and self.default_supertype(first[0]) is not None:
                out.append(self.default_supertype(first[0]))
            # every type is a specialization of Base::Anything, the least specific supertype (KerML)
            if first and _type_like(first[0]) and ROOT_TYPE in self.by_qname and ROOT_TYPE != qname \
                    and ROOT_TYPE not in out:
                out.append(ROOT_TYPE)
        finally:
            self._busy_supers.discard(qname)
        if self._cuts == before:
            self._supers[qname] = out
        return out

    def default_supertype(self, el: Element) -> str | None:
        """The library base of the element's keyword (Parts::Part for a part), the implicit supertype of a definition
        or usage with no declared one; None for the others (their supertype is Base::Anything, see supertypes)."""
        words = [w for w in el.kind.split() if w not in NOTE_WORDS]
        if not _type_like(el) or not words:
            return None
        target = DEFAULT_SUPERTYPE.get(words[0])
        return target if target in self.by_qname and target != el.qname else None

    def inherited(self, ns: str, key: str, exclude, busy: set) -> list[Element]:
        """Public members named key that ns inherits: the first level of its types (breadth first) that has one."""
        frontier, seen = self.supertypes(ns), {ns}
        while frontier:
            found, nxt = [], []
            for s in frontier:
                if s in seen:
                    continue
                seen.add(s)
                found += [e for e in self.owned(s, key) if e.vis == "public" and e.qname != exclude]
                nxt += self.supertypes(s)
            if found:
                return self._deref(found, busy)
            frontier = nxt
        return []

    def _deref(self, els: list[Element], busy: set) -> list[Element]:
        """The elements a lookup names: an alias (alias X for Y) stands for the element it names."""
        out: list[Element] = []
        for e in els:
            if e.kind != "alias":
                out.append(e)
                continue
            ref = self.alias_of.get(id(e))
            if ref is None:
                continue
            if id(e) in busy:
                self._cuts += 1
                continue
            busy.add(id(e))
            try:
                target = self._resolve(ref.text, ref.chain, None, busy).target
            finally:
                busy.discard(id(e))
            if target is not None:
                out += self._deref(self.by_qname.get(target, []), busy)
        return _unique(out)

    def imported(self, ns: str, key: str, inside: bool, exclude, busy: set) -> list[Element]:
        """Members named key that ns gets from its imports. From outside ns, only its public imports count."""
        out = []
        for r in self.imports.get(ns, []):
            if not inside and r.vis != "public":
                continue
            if id(r) in busy:
                self._cuts += 1
                continue
            k = (id(r), key)
            if k not in self._imp_memo and k not in self._qimp_memo:
                before = self._cuts
                busy.add(id(r))
                try:
                    res = self._import_provides(r, key, None, busy)
                finally:
                    busy.discard(id(r))
                (self._imp_memo if self._cuts == before else self._qimp_memo)[k] = res
            res = self._imp_memo.get(k, self._qimp_memo.get(k, []))
            out += [e for e in res if e.qname != exclude]
        return _unique(out)

    def _import_provides(self, r: Ref, key: str, exclude, busy: set) -> list[Element]:
        text = r.text
        if text.endswith("::**"):
            base, kind = text[:-4], "recursive"
        elif text.endswith("::*"):
            base, kind = text[:-3], "all"
        else:
            base, _, last = text.rpartition("::")
            kind = "single"
            if _key(last or text) != key:
                return []
            if not base:  # a top-level name: import Y
                return self.member_of("", _key(text), False, exclude, busy)
        target = self._resolve(base, r.chain, None, busy).target
        if target is None:
            return []
        if kind == "recursive":
            return self._recursive(target, key, set(), busy)
        return self.member_of(target, key, False, exclude, busy)

    def _recursive(self, ns: str, key: str, seen: set, busy: set) -> list[Element]:
        seen.add(ns)
        out = self.member_of(ns, key, False, None, busy)
        for kids in self.children.get(ns, {}).values():
            for c in kids:
                if c.vis == "public" and c.qname not in seen:
                    out += self._recursive(c.qname, key, seen, busy)
        return _unique(out)

    def member_of(self, ns: str, key: str, inside: bool, exclude, busy: set) -> list[Element]:
        """Members of ns named key, in shadowing order: owned, then inherited, then imported."""
        k = (ns, key, inside, exclude)
        if k in self._memo:
            return self._memo[k]
        if k in self._qmemo:
            return self._qmemo[k]
        before = self._cuts
        owned = [e for e in self.owned(ns, key) if e.qname != exclude and (inside or e.vis == "public")]
        if owned:
            out = self._deref(owned, busy)
        else:
            out = self.inherited(ns, key, exclude, busy) or self.imported(ns, key, inside, exclude, busy)
        (self._memo if self._cuts == before else self._qmemo)[k] = out
        return out

    def resolve(self, text: str, chain: list, exclude: str | None = None) -> Resolution:
        """The qualified name of text as written inside the namespaces of chain (innermost last). exclude is a
        qualified name that does not count (a redefinition does not redefine itself)."""
        self._qmemo.clear()  # a new top-level query: the results of cut cycles from the last one do not carry over
        self._qimp_memo.clear()
        return self._resolve(text, chain, exclude, set())

    def _resolve(self, text: str, chain: list, exclude, busy: set) -> Resolution:
        parts = SPLIT.split(text)  # a::b and the feature chain a.b.c
        head = parts[0]
        if head in ("", "*", "**"):
            return Resolution(None, f"not a name: {text!r}")
        cands: list[Element] = []
        for ns in reversed(chain):
            cands = self.member_of(ns, _key(head), True, exclude, busy)
            if cands:
                break
        if not cands:
            return Resolution(None, f"no declaration of '{head}' in its enclosing namespaces or their imports")
        if len({c.qname for c in cands}) > 1:
            return Resolution(None, "ambiguous: " + ", ".join(sorted({c.qname for c in cands})))
        cur = cands[0].qname
        for seg in parts[1:]:
            if seg in ("*", "**"):
                return Resolution(None, f"'{seg}' is only valid in an import")
            cands = self.member_of(cur, _key(seg), cur in chain, None, busy)
            if not cands:
                return Resolution(None, f"no member '{seg}' in '{cur}'")
            if len({c.qname for c in cands}) > 1:
                return Resolution(None, f"ambiguous member '{seg}' of '{cur}'")
            cur = cands[0].qname
        return Resolution(cur, "")

    def owner_of(self, f, r: Ref) -> Element | None:
        """The named element whose statement holds r: its header owner, else the innermost element of the chain's
        last namespace that encloses r. None at the top level or in an anonymous statement."""
        if r.owner is not None:
            return r.owner
        if r.chain[-1] == "":
            return None
        best = None
        for el in self.in_file.get((f.uri, r.chain[-1]), []):
            if el.start <= r.start < el.end and (best is None or el.start > best.start):
                best = el
        return best

    # -- references ---------------------------------------------------------------------------------------

    def references(self) -> list[Found]:
        out: list[Found] = []
        for f in self.files:
            by_stmt: dict[int, dict[str, list[Ref]]] = defaultdict(lambda: defaultdict(list))
            for r in f.refs:
                if r.fragment in PAIRED_FRAGS and r.stmt is not None:
                    by_stmt[r.stmt][r.fragment].append(r)
            for r in sorted(f.refs, key=lambda r: r.start):
                line = f.line_of(r.start)
                if r.fragment == "import":
                    out.append(self._import(f, r, line))
                elif r.fragment == "alias_for":
                    res = self.resolve(r.text, r.chain)
                    out.append(Found("alias_for", r.text, f.uri, line, "resolved" if res.target else "unresolved",
                                     res.why, target=res.target))
                elif r.fragment in TYPE_FRAGS or r.fragment == "perform":
                    out.append(self._single(f, r, line))
                elif r.fragment not in PAIRED_FRAGS:
                    out.append(Found(r.fragment, r.text, f.uri, line, "out_of_scope",
                                     why="a fragment that is not an edge in this model"))
            for stmt in sorted(by_stmt):
                out += self._pairs(f, by_stmt[stmt])
        return out

    def _import(self, f, r: Ref, line: int) -> Found:
        base = r.text[:-4] if r.text.endswith("::**") else r.text[:-3] if r.text.endswith("::*") else r.text
        res = self.resolve(base, r.chain)
        return Found("import", r.text, f.uri, line, "resolved" if res.target else "unresolved", res.why)

    def _single(self, f, r: Ref, line: int) -> Found:
        src = self.owner_of(f, r)
        if src is None:
            return Found(r.fragment, r.text, f.uri, line, "unresolved",
                         why="the statement is not inside a named element")
        res = self.resolve(r.text, r.chain, src.qname if r.fragment == "redefines" else None)
        if res.target is None:
            return Found(r.fragment, r.text, f.uri, line, "unresolved", res.why, source=src)
        if src.name is None:  # its owner is an unnamed usage: resolved, but it has no section to draw an edge from
            return Found(r.fragment, r.text, f.uri, line, "anonymous", "", "", src, res.target)
        return Found(r.fragment, r.text, f.uri, line, "resolved", "", WRITTEN[r.fragment], src, res.target)

    def _pairs(self, f, group: dict[str, list[Ref]]) -> list[Found]:
        """satisfy X by Y: Y satisfies X (the owner does, without by). allocate A to B: A is allocated to B. A
        `by` or `to` half is reported with its own resolution; its edge belongs to the other half."""
        out: list[Found] = []
        satisfiers = {id(b): self.resolve(b.text, b.chain) for b in group.get("satisfy_by", [])}
        for b in group.get("satisfy_by", []):
            res = satisfiers[id(b)]
            out.append(Found("satisfy_by", b.text, f.uri, f.line_of(b.start),
                             "resolved" if res.target else "unresolved", res.why, target=res.target))
        for r in group.get("satisfy", []):
            if group.get("satisfy_by"):
                sres = satisfiers[id(group["satisfy_by"][0])]
                src = self.by_qname[sres.target][0] if sres.target else None
                src_why = sres.why
            else:
                src = self.owner_of(f, r)
                src_why = "the statement is not inside a named element"
            out.append(self._edge(f, r, "satisfy", src, src_why, self.resolve(r.text, r.chain), WRITTEN["satisfy"]))
        for a in group.get("allocate", []):
            if not group.get("allocate_to"):
                out.append(Found("allocate", a.text, f.uri, f.line_of(a.start), "unresolved",
                                 why="allocate with no 'to' target"))
                continue
            sres = self.resolve(a.text, a.chain)
            src = self.by_qname[sres.target][0] if sres.target else None
            to = group["allocate_to"][0]
            out.append(self._edge(f, a, "allocate", src, sres.why, self.resolve(to.text, to.chain),
                                  WRITTEN["allocate"]))
        if not group.get("allocate"):
            for to in group.get("allocate_to", []):
                out.append(Found("allocate_to", to.text, f.uri, f.line_of(to.start), "unresolved",
                                 why="a 'to' target with no allocated element"))
        return out

    def _edge(self, f, r: Ref, frag: str, src: Element | None, src_why: str, tres: Resolution, rel: str) -> Found:
        line = f.line_of(r.start)
        if src is None:
            return Found(frag, r.text, f.uri, line, "unresolved", src_why, target=tres.target)
        if tres.target is None:
            return Found(frag, r.text, f.uri, line, "unresolved", tres.why, source=src)
        if src.name is None:
            return Found(frag, r.text, f.uri, line, "anonymous", "", "", src, tres.target)
        return Found(frag, r.text, f.uri, line, "resolved", "", rel, src, tres.target)

    def metaclasses(self):
        """(element, metaclass name or None, notes) for every element."""
        for el in self.elements:
            name, notes = metaclass_for(el.kind)
            yield el, name, notes


# -- the model DB: resolve -------------------------------------------------------------------------------

def _short(qname: str) -> str:
    return qname.rpartition("::")[2]


def _section_lookup(sections: list[dict]) -> dict:
    """(uri, offset) -> the DocSection nodes there."""
    out: dict = defaultdict(list)
    for s in sections:
        out[(s["name"], int(s["offset"]))].append(s)
    return out


def _path_tail(path: str) -> str:
    """The element's own labels: a stored path is "model › NAME › …", the scanner's path is a list."""
    return " › ".join(path.split(" › ")[2:])


def _section_for(lookup: dict, el: Element) -> dict | None:
    for s in lookup.get((el.uri, el.start), []):
        if _path_tail(s.get("path", "")) == " › ".join(el.path[2:]):
            return s
    return None


async def replace_edges(edges: list[tuple]) -> int:
    """Delete the resolver's edges (its relations and metaclasses) and write the new ones, in one pipeline run.
    Returns the number deleted. The other relations of the graph are not touched."""
    import cognee
    from cognee.infrastructure.databases.provenance import EdgeIdentity
    from cognee.modules.pipelines import Task

    counts = {}

    async def store_batch(items, ctx=None):
        graph, _ = await store.engines()
        old = []
        for rel in EDGE_RELS:
            rows = await graph.query(
                "MATCH (a:Node)-[r:EDGE]->(b:Node) WHERE r.relationship_name = $rel RETURN a.id, b.id", {"rel": rel})
            old += [EdgeIdentity(source_id=str(a), target_id=str(b), relationship_name=rel) for a, b in rows]
        await graph.delete_edge_triples(old)
        await graph.add_edges(edges)
        counts["replaced"] = len(old)
        return []

    result = await cognee.run_custom_pipeline(tasks=[Task(store_batch, needs_llm=False)], data=[{"batch": 0}],
                                              dataset=store.DATASET)
    _check_pipeline(result, "model-resolve")
    return counts["replaced"]


def _parse_files(uris: list[str]):
    """The scanner's parse of each model file, read from its file:// URI."""
    sources = []
    for uri in uris:
        path = store.uri_path(uri)
        if not path.is_file():
            raise SystemExit(f"docmem model-resolve: {path} is gone; re-scan and re-index the model first")
        sources.append(({"uri": uri}, path.read_bytes()))
    errors: list[str] = []
    model = parse_model(sources, "resolve", errors)
    if errors:
        raise SystemExit("docmem model-resolve: the files no longer parse:\n" + "\n".join(errors))
    return model.files


async def resolve_async(db: Path, out) -> dict:
    import cognee

    await cognee.run_migrations()
    graph, _ = await store.engines()
    _, concepts = await scheme_concepts(graph, SCHEME)
    meta_id = {}
    for cid, p in concepts.items():
        if p["name"] in meta_id:
            raise SystemExit(f"docmem model-resolve: two metaclasses named {p['name']!r} in {SCHEME}")
        meta_id[p["name"]] = cid

    t = time.time()
    uris = [n["name"] for n in await nodes_of_type(graph, "DocFile")]
    if not uris:
        raise SystemExit(f"docmem model-resolve: no model files in {db} (scan-sysml and index them first)")
    sections = await store.nodes_of(graph, "DocSection", uris)
    lookup = _section_lookup(sections)
    uni = Universe(_parse_files(uris))

    sec_of: dict[int, dict] = {}
    missing = []
    for el in uni.elements:
        s = _section_for(lookup, el)
        if s is None:
            missing.append(f"{el.uri}:{el.line} {el.kind} {el.name}")
        else:
            sec_of[id(el)] = s
    if missing:
        raise SystemExit("docmem model-resolve: %d elements have no DocSection in the DB (the index is stale; "
                         "re-run scan-sysml and index):\n%s" % (len(missing), "\n".join(missing[:10])))
    matched = {s["id"] for s in sec_of.values()}
    extra = [s for s in sections if s["id"] not in matched]  # a lead section of text, or a file no longer parsed

    found = uni.references()
    edges, unmapped, mapped = [], Counter(), 0
    for f in found:
        if f.status != "resolved" or not f.rel or f.source is None:
            continue
        tgt = uni.by_qname[f.target][0]
        props = {"edge_text": f"{_short(f.source.qname)} {f.rel} {_short(f.target)}",
                 "as_written": f.as_written, "file": f.uri, "line": f.line}
        edges.append((str(sec_of[id(f.source)]["id"]), str(sec_of[id(tgt)]["id"]), f.rel, props))
    for el, name, notes in uni.metaclasses():
        if name is None:
            unmapped[el.kind] += 1
            continue
        if name not in meta_id:
            raise SystemExit(f"docmem model-resolve: metaclass {name!r} is not in the {SCHEME} scheme "
                             "(re-run tools/import_sysml_metamodel.py and ontology load)")
        mapped += 1
        props = {"edge_text": f"{_short(el.qname)} is a {name}", "file": el.uri, "line": el.line,
                 "notes": " ".join(notes)}
        edges.append((str(sec_of[id(el)]["id"]), str(meta_id[name]), METACLASS_REL, props))

    replaced = await replace_edges(edges)
    report = _report(uni, found, len(uni.elements), len(sections), extra, mapped, unmapped, len(edges), replaced)
    report["sections_not_in_files_examples"] = [f"{_uri_name(s['name'])} {s.get('path', '')}" for s in extra[:EXAMPLES]]
    report["seconds"] = round(time.time() - t, 1)
    out.write(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return report


def _report(uni, found, n_elements, n_sections, extra, mapped, unmapped, n_edges, replaced) -> dict:
    by_frag: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    for f in found:
        by_frag[f.fragment][f.status] += 1
        if f.status == "unresolved" and len(examples[f.fragment]) < EXAMPLES:
            examples[f.fragment].append(f"{f.as_written!r} at {_uri_name(f.uri)}:{f.line}: {f.why}")
    # "anonymous": resolved, but the reference belongs to an unnamed usage, which has no section to write an edge from
    ok = ("resolved", "anonymous")
    core = [f for f in found if f.fragment in ("typed_by", "specializes")]
    core_ok = sum(1 for f in core if f.status in ok)
    in_scope = [f for f in found if f.status != "out_of_scope"]
    return {
        "model_files": len(uni.files), "elements": n_elements, "sections_in_db": n_sections,
        "sections_not_in_files": len(extra),
        "typed_by_and_specializes": {"total": len(core), "resolved": core_ok,
                                     "resolved_pct": round(100.0 * core_ok / len(core), 1) if core else None},
        "in_scope_resolved_pct": round(100.0 * sum(f.status in ok for f in in_scope) / len(in_scope), 1)
        if in_scope else None,
        "references_by_fragment": {k: dict(v) for k, v in sorted(by_frag.items())},
        "edges_written": n_edges, "edges_replaced": replaced,
        "metaclass_mapped": mapped, "metaclass_unmapped_kinds": dict(unmapped.most_common()),
        "unresolved_examples": dict(examples),
    }


def _uri_name(uri: str) -> str:
    return unquote(uri.rsplit("/", 1)[-1])


def resolve_main(argv=None):
    import sys

    ap = argparse.ArgumentParser(prog="docmem model-resolve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True, help="a model DB (its files were scanned with scan-sysml)")
    a = ap.parse_args(argv)
    store.configure(a.db)
    asyncio.run(resolve_async(a.db, sys.stdout))


# -- the model DB: show ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Member:
    name: str
    kind: str
    qname: str
    where: str  # file:line
    sid: str  # the DocSection id


def is_feature(kind: str) -> bool:
    """A usage (a feature), as opposed to a definition or a package."""
    return not (kind.endswith(" def") or kind.endswith("package"))


def inherited_members(qname: str, owned: dict[str, list[Member]], supers: dict[str, list[str]]):
    """The features qname inherits, as (member, the type it comes from). Its types are walked breadth first, so a
    feature of a more specific type is found first; a name the element declares itself (a redefinition) is not
    inherited again, and a name seen at a closer type is not repeated from a farther one."""
    seen = {m.name for m in owned.get(qname, [])}
    out, visited = [], {qname}
    frontier = list(supers.get(qname, []))
    while frontier:
        nxt = []
        for t in frontier:
            if t in visited:
                continue
            visited.add(t)
            for m in owned.get(t, []):
                if m.name not in seen and is_feature(m.kind):
                    seen.add(m.name)
                    out.append((m, t))
            nxt.extend(supers.get(t, []))
        frontier = nxt
    return out


def _where(uri: str, line) -> str:
    return f"{_uri_name(uri)}:{line}"


async def show_async(name: str, out) -> None:
    graph, _ = await store.engines()
    secs = await nodes_of_type(graph, "DocSection")
    qn: dict[str, str] = {}
    info: dict[str, dict] = {}
    for s in secs:
        labels = s.get("path", "").split(" › ")[2:]
        parts = [re.match(r"^(?P<kind>.*?)\s+(?P<name>'(?:[^'\\]|\\.)*'|\S+)$", x) for x in labels]
        if not labels or any(p is None for p in parts):
            continue
        q = "::".join(p.group("name") for p in parts)
        kind = parts[-1].group("kind")
        qn[str(s["id"])] = q
        info[str(s["id"])] = {"kind": kind, "uri": s.get("name", ""), "line": s.get("line"), "qname": q}
    hits = [i for i, q in qn.items() if q == name]
    if not hits:
        near = sorted({q for q in qn.values() if q.endswith(name)})[:10]
        raise SystemExit(f"docmem model-show: no element {name!r}" + (f" (did you mean: {', '.join(near)})" if near
                                                                    else ""))
    sid = hits[0]
    rels: dict[str, dict[str, list]] = {r: defaultdict(list) for r in EDGE_RELS}
    for rel in EDGE_RELS:
        async for a, b, props in _edges(graph, rel):
            rels[rel][a].append((b, props))
    concept_ids = sorted({b for lst in rels[METACLASS_REL].values() for b, _ in lst})
    names: dict[str, str] = {}
    parents: dict[str, list[str]] = {}
    if concept_ids:
        _, concepts = await scheme_concepts(graph, SCHEME)
        names = {c: p["name"] for c, p in concepts.items()}
        parents = await broader_parents(graph, concept_ids)

    def up(c: str, seen: frozenset = frozenset()) -> list[str]:
        """The names of c's ancestors in the metamodel, nearest first, depth first."""
        out_: list[str] = []
        for p in parents.get(c, []):
            if p not in seen:
                out_ += [names.get(p, p)] + up(p, seen | {p})
        return out_

    me = info[sid]
    out.write(f"{me['kind']} {name}   ({_where(me['uri'], me['line'])})\n")
    for cid, _ in rels[METACLASS_REL].get(sid, []):
        chain = " < ".join(dict.fromkeys(up(cid)))
        out.write(f"metaclass   {names.get(cid, cid)}" + (f"  (up: {chain})" if chain else "") + "\n")
    owner_q = name.rpartition("::")[0]
    owner = next((i for i, q in qn.items() if q == owner_q), None)
    out.write(f"owner       {owner_q or '(top level)'}" + (f"  ({info[owner]['kind']})" if owner else "") + "\n")

    def walk(node: str, depth: int, seen: frozenset):
        for rel in ("r_typed_by", "r_specializes"):
            for tid, props in rels[rel].get(node, []):
                if tid not in qn:
                    continue
                is_root = not any(rels[x].get(tid) for x in ("r_typed_by", "r_specializes"))
                out.write(f"{'  ' * depth}{rel[2:]} {qn[tid]}   [{_where(props.get('file', ''), props.get('line'))}]"
                          + ("   (root)" if is_root else "") + "\n")
                if tid not in seen:
                    walk(tid, depth + 1, seen | {tid})

    out.write("types (up to the library roots):\n")
    walk(sid, 1, frozenset({sid}))

    owned: dict[str, list[Member]] = defaultdict(list)
    for i, q in qn.items():
        owned[q.rpartition("::")[0]].append(Member(q.rpartition("::")[2], info[i]["kind"], q,
                                                   _where(info[i]["uri"], info[i]["line"]), i))
    supers = {q: [qn[str(b)] for b, _ in rels["r_specializes"].get(i, []) + rels["r_typed_by"].get(i, [])
                  if str(b) in qn] for i, q in qn.items()}
    out.write("owned features:\n")
    for m in sorted(owned.get(name, []), key=lambda m: m.where):
        if is_feature(m.kind):
            redef = [qn[b] for b, _ in rels["r_redefines"].get(m.sid, []) if b in qn]
            out.write(f"  {m.name}  ({m.kind})  [{m.where}]" + (f"  redefines {', '.join(redef)}" if redef else "")
                      + "\n")
    out.write("inherited features:\n")
    for m, src in inherited_members(name, owned, supers):
        out.write(f"  {m.name}  ({m.kind})  from {src}  [{m.where}]\n")


async def _edges(graph, rel: str):
    rows = await graph.query("MATCH (a:Node)-[r:EDGE]->(b:Node) WHERE r.relationship_name = $rel "
                             "RETURN a.id, b.id, r.properties", {"rel": rel})
    for a, b, props in rows:
        p = json.loads(props) if isinstance(props, str) else (props or {})
        yield str(a), str(b), p


def show_main(argv=None):
    import sys

    ap = argparse.ArgumentParser(prog="docmem model-show", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("name", help="the qualified name, e.g. 'Apollo11::LunarModule'")
    a = ap.parse_args(argv)
    store.configure(a.db)
    asyncio.run(show_async(a.name, sys.stdout))


if __name__ == "__main__":
    resolve_main()
