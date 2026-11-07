"""Scanner: a SysML v2 textual model as docmem protocol-1 records, for a model DB (never a prose DB).

    docmem scan-sysml ROOT [ROOT ...] [--model-name NAME]

ROOT is a .sysml file or a directory (scanned recursively, hidden directories skipped). The model is only read and
tokenized; nothing in it is executed. Output is the protocol stream: one "doc" per file (uri, sha256 of the file,
extractor "", "format": "sysml", "model": NAME), then per file

  section  one per named element (package, part def, part, requirement, attribute, ...): path = ["model", NAME,
           "package A", "requirement def B", ..., "<kind> <name>"] (the named enclosing elements, then its own
           label). offset and length cover its declaration, from its first modifier or keyword to the closing
           brace or ';'.
  passage  the declaration header, doc comments, comments, requirement text ("require constraint {...}",
           "assume", "assert"), and string-literal values, inside the element. Pieces separated only by whitespace
           form one passage; any other gap starts a new one. An attribute with no text beyond its header has no
           passage. Offsets are into the file.
  link     kind "model". fragment is the relation: specializes, redefines, subsets, typed_by, satisfy / satisfy_by,
           verify / verify_by, allocate / allocate_to, connect / connect_to, bind / bind_to, flow / flow_to,
           perform, exhibit, import, or the metadata name (refinement) with "_to" for the second name. target is
           the name as written (whitespace removed), or the qualified name of the element it refers to when the
           name resolves uniquely in an enclosing namespace.

Statements the scanner does not model (transitions, successions, expressions, ...) are skipped; their leading
words are reported on stderr with file:line examples.
"""

import argparse
import bisect
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

from . import PROTOCOL

SUFFIX = ".sysml"
TOKEN = re.compile(
    rb"(?P<ws>\s+)"
    rb"|(?P<line>//[^\n]*)"
    rb"|(?P<cmt>/\*.*?\*/)"
    rb'|(?P<str>"(?:[^"\\]|\\.)*")'
    rb"|(?P<q>'(?:[^'\\\n]|\\.)*')"
    rb"|(?P<num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[A-Za-z_]*)"
    rb"|(?P<id>[A-Za-z_][A-Za-z0-9_]*)"
    rb"|(?P<p>::>|::|:>>|:>|->|\.\.|\*\*|==|!=|<=|>=|=>|[\x80-\xff]+|.)",
    re.S,
)
HEAD = re.compile(r"'(?:[^'\\\n]|\\.)*'|[A-Za-z_][A-Za-z0-9_]*")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

VISIBILITY = frozenset({"private", "public", "protected"})
MODIFIERS = VISIBILITY | {
    "abstract", "individual", "ref", "variation", "in", "out", "inout", "first", "then", "end", "return",
    "derived", "readonly", "portion", "constant", "standard", "library",
}
RELATION_PREFIX = frozenset({"perform", "exhibit", "do"})  # a modifier only before an element keyword: do action x
ELEMENT_KEYWORDS = frozenset({
    "package", "part", "port", "action", "state", "requirement", "item", "attribute", "interface",
    "connection", "constraint", "analysis", "verification", "view", "viewpoint", "subject", "actor",
    "stakeholder", "concern", "enum", "calc", "flow", "allocation", "occurrence", "metadata", "timeslice",
    "snapshot", "event", "message",
})
VERBS = frozenset({"satisfy", "verify", "allocate", "connect", "bind", "flow", "perform", "exhibit"})
REDEFINES = frozenset({":>", ":>>", "::>"})  # specialization operators; ":>>" redefines, the others specialize
TEXT_WORDS = frozenset({"require", "assume", "assert"})
STOP = VERBS | {"by", "to", "from", "of", "def", "specializes", "subsets", "redefines", "ordered", "nonunique",
                "unique"}
HEADER_SKIP = frozenset({"ordered", "nonunique", "unique", "dependency", "def", "ref", "in", "out", "inout",
                         "derived", "readonly", "abstract", "individual", "variation", "end", "constant",
                         "requirement"})
OPENERS = ("(", "[", "{")
CLOSERS = (")", "]", "}")


class ScanError(ValueError):
    """A model the scanner cannot tokenize or whose braces do not balance."""


class Tok(NamedTuple):
    kind: str  # id, q (quoted name), str, num, cmt (block comment), p (punctuation)
    text: str
    start: int
    end: int


def is_p(t: Tok, *texts: str) -> bool:
    return t.kind == "p" and t.text in texts


def tokenize(data: bytes, where: str) -> list[Tok]:
    """Tokens of a SysML file. Comments and whitespace are tokens too (block comments), line comments are not."""
    toks, pos = [], 0
    while pos < len(data):
        m = TOKEN.match(data, pos)
        group = m.lastgroup
        if group == "p" and (data[pos:pos + 1] in (b'"', b"'") or data.startswith(b"/*", pos)):
            what = "string" if data[pos:pos + 1] == b'"' else "quoted name" if data[pos:pos + 1] == b"'" else "comment"
            raise ScanError(f"{where}:{_line(data, pos)}: unterminated {what}")
        if group not in ("ws", "line"):
            toks.append(Tok(group, data[m.start():m.end()].decode("utf-8", "replace"), m.start(), m.end()))
        pos = m.end()
    return toks


def _line(data: bytes, offset: int) -> int:
    return data.count(b"\n", 0, offset) + 1


def _key(name: str) -> str:
    """The lookup key of a declared name: a quoted identifier is the same name as its unquoted form."""
    if len(name) > 2 and name[0] == "'" and name[-1] == "'" and IDENT.match(name[1:-1]):
        return name[1:-1]
    return name


class Element:
    def __init__(self, kind, keyword, name, qname, path, start, line):
        self.kind, self.keyword, self.name, self.qname = kind, keyword, name, qname
        self.path, self.start, self.line = path, start, line
        self.end = start
        self.pieces: list[tuple[int, int, bool]] = []  # (start, end, is_header)


class Ref:
    __slots__ = ("start", "end", "text", "fragment", "chain", "target")

    def __init__(self, start, end, text, fragment, chain):
        self.start, self.end, self.text, self.fragment, self.chain = start, end, text, fragment, chain
        self.target = text


class _File:
    """Parses one file. Declared names go to the shared Model; references wait there for resolution."""

    def __init__(self, doc: dict, data: bytes, model: "Model"):
        self.doc, self.data, self.model = doc, data, model
        self.uri = doc["uri"]
        self.toks = tokenize(data, self.uri)
        self.n = len(self.toks)
        self.root_path = ["model", model.name]
        self.elements: list[Element] = []
        self.refs: list[Ref] = []
        self.unmodelled: list[tuple[int, str]] = []
        self.stray = 0
        self.line_starts = [0] + [m.end() for m in re.finditer(rb"\n", data)]
        self.members(0, None, [""], until=False)

    # -- positions and helpers --------------------------------------------------------------------------------

    def line_of(self, offset: int) -> int:
        return bisect.bisect_right(self.line_starts, offset)

    def fail(self, index: int, msg: str):
        off = self.toks[index].start if index < self.n else len(self.data)
        raise ScanError(f"{self.uri}:{self.line_of(off)}: {msg}")

    def piece(self, owner, start: int, end: int, header: bool = False):
        if owner is None:
            self.stray += 1  # text outside every named element (a file-level comment): not a passage
            return
        owner.pieces.append((start, end, header))

    def skip_balanced(self, m: int) -> int:
        """Index just after the bracket that closes the one at m (any of ( [ {)."""
        depth, k = 0, m
        while k < self.n:
            t = self.toks[k]
            if t.kind == "p" and t.text in OPENERS:
                depth += 1
            elif t.kind == "p" and t.text in CLOSERS:
                depth -= 1
                if depth == 0:
                    return k + 1
            k += 1
        self.fail(m, "unbalanced bracket")

    def qname(self, m: int) -> tuple[int, str]:
        """(index after, text) of the name starting at m: a::b, 'q'::c, a.b.c, a::* and a::**."""
        toks, e = self.toks, m + 1
        while e + 1 < self.n:
            a, b = toks[e], toks[e + 1]
            if a.kind == "p" and a.text == "::" and (b.kind in ("id", "q") or is_p(b, "*", "**")):
                e += 2
            elif a.kind == "p" and a.text == "." and b.kind in ("id", "q"):
                e += 2
            else:
                break
        return e, "".join(t.text for t in toks[m:e])

    def is_modifier(self, k: int) -> bool:
        t = self.toks[k]
        if t.kind != "id":
            return False
        if t.text in MODIFIERS:
            return True
        return t.text in RELATION_PREFIX and self.is_keyword(k + 1)

    def is_keyword(self, k: int) -> bool:
        if k >= self.n or self.toks[k].kind != "id":
            return False
        t = self.toks[k]
        if t.text in ELEMENT_KEYWORDS:
            return True
        return t.text == "use" and k + 1 < self.n and self.toks[k + 1].kind == "id" and self.toks[k + 1].text == "case"

    def is_name(self, m: int) -> bool:
        t = self.toks[m]
        if t.kind not in ("id", "q") or t.text in STOP:
            return False
        nxt = self.toks[m + 1] if m + 1 < self.n else None
        return not (nxt is not None and nxt.kind == "id" and nxt.text in ("to", "by", "of"))

    # -- members and statements -------------------------------------------------------------------------------

    def members(self, i: int, owner, chain: list, until: bool) -> int:
        """Parse members from i. Returns the index after the closing brace when until, else self.n."""
        toks, n = self.toks, self.n
        while True:
            if i >= n:
                if until:
                    self.fail(n, "unclosed '{'")
                return i
            t = toks[i]
            if is_p(t, "}"):
                if until:
                    return i + 1
                self.fail(i, "unexpected '}'")
            if t.kind == "cmt":
                self.piece(owner, t.start, t.end)
                i += 1
            elif is_p(t, ";"):
                i += 1
            elif is_p(t, "{"):  # a body with no declaration: its text belongs to the enclosing element
                i = self.members(i + 1, owner, chain, True)
            elif t.kind == "id" and t.text == "doc" and i + 1 < n and toks[i + 1].kind == "cmt":
                self.piece(owner, t.start, toks[i + 1].end)
                i += 2
            elif is_p(t, "#"):
                i = self.metadata(i, owner, chain)
            elif t.kind == "id" and t.text == "comment":
                i = self.comment(i, owner)
            else:
                i = self.statement(i, owner, chain)

    def statement(self, i: int, owner, chain: list) -> int:
        toks, n = self.toks, self.n
        if is_p(toks[i], "@"):  # a metadata usage: its body is parsed as members of the enclosing element
            return self.unknown(i, owner, chain, warn=False)
        j = i
        while j < n and toks[j].kind == "id" and toks[j].text in VISIBILITY:
            j += 1
        if j < n and toks[j].kind == "id" and toks[j].text == "import":
            return self.import_stmt(j, chain)
        if j < n and toks[j].kind == "id" and toks[j].text in TEXT_WORDS:
            return self.text_stmt(j, owner)
        if j + 3 < n and toks[j].kind == "id" and is_p(toks[j + 1], "=") and toks[j + 2].kind == "str" \
                and is_p(toks[j + 3], ";"):
            self.piece(owner, toks[j].start, toks[j + 3].end)  # text = "..."; in a metadata body
            return j + 4
        if j + 1 < n and toks[j].kind == "p" and toks[j].text in REDEFINES and toks[j + 1].kind in ("id", "q"):
            return self.redefinition(i, j, owner, chain)  # :>> name : T { ... }
        k = j
        while k < n and self.is_modifier(k):
            k += 1
        if self.is_keyword(k):
            return self.declaration(i, k, owner, chain, generic=False)
        if k < n and toks[k].kind == "id" and toks[k].text in VERBS:
            return self.relation(k, owner, chain)
        if k > i and k + 1 < n and toks[k].kind == "id" and toks[k].text not in STOP \
                and toks[k + 1].kind == "p" and toks[k + 1].text in (":", ":>", ":>>", "::>", "[", "="):
            return self.declaration(i, k, owner, chain, generic=True)  # a usage with a direction: in sys : T
        return self.unknown(i, owner, chain)

    def redefinition(self, i: int, j: int, owner, chain: list) -> int:
        """:>> name ... / :> name ...: a usage that redefines or specializes a feature of its owner's type."""
        name_tok = self.toks[j + 1]
        frag, kind = ("redefines", "redefinition") if self.toks[j].text == ":>>" else ("specializes", "specialization")
        self.refs.append(Ref(name_tok.start, name_tok.end, name_tok.text, frag, chain))
        return self.usage(i, kind, "", name_tok, j + 2, None, owner, chain)

    def declaration(self, mstart: int, k: int, owner, chain: list, generic: bool) -> int:
        toks, n = self.toks, self.n
        mods = [t.text for t in toks[mstart:k] if t.text not in VISIBILITY]
        short = None
        if generic:
            kw, kind, name_tok, hstart = "", " ".join(mods), toks[k], k + 1
        else:
            kw = toks[k].text
            m = k + 1
            if kw == "use":
                kw, m = "use case", k + 2
            words = mods + [kw]
            if m < n and toks[m].kind == "id" and toks[m].text == "def":
                words.append("def")
                m += 1
            kind = " ".join(words)
            if m < n and is_p(toks[m], "<"):  # a short name: <'FLR-R001'> Name
                close = next((x for x in range(m + 1, n) if is_p(toks[x], ">")), None)
                if close is None:
                    self.fail(m, "unclosed '<'")
                short = "".join(t.text for t in toks[m + 1:close])
                m = close + 1
            if m < n and self.is_name(m):
                name_tok, hstart = toks[m], m + 1
            else:
                name_tok, hstart = None, m
        return self.usage(mstart, kind, kw, name_tok, hstart, short, owner, chain)

    def usage(self, mstart: int, kind: str, kw: str, name_tok, hstart: int, short, owner, chain: list) -> int:
        """A declaration's header and body. A usage with no name (no name_tok and no short name) is not a section:
        its text and links still belong to the enclosing element."""
        toks = self.toks
        if name_tok is not None:
            name, alias = name_tok.text, short
        else:
            name, alias = short, None
        verb = kw if kw in VERBS else None
        hend, term = self.header_end(hstart, eq_term=(kw != "bind"))
        self.header_refs(hstart, hend, verb, verb if name_tok is None else None, chain)
        if name is None:
            _, nxt = self.finish(term, hend, owner, chain, chain)
            return nxt
        parent = chain[-1]
        qname = f"{parent}::{name}" if parent else name
        start = toks[mstart].start
        el = Element(kind, kw, name, qname, (owner.path if owner else self.root_path) + [f"{kind} {name}"],
                     start, self.line_of(start))
        self.elements.append(el)
        self.model.declare(parent, name, qname)
        if alias is not None:
            self.model.declare(parent, alias, qname)
        head_end = hend if term in ("{", ";") else hend - 1
        el.pieces.append((start, toks[head_end].end, True))
        end_tok, nxt = self.finish(term, hend, el, chain + [qname], chain)
        el.end = toks[end_tok].end
        return nxt

    def finish(self, term: str, hend: int, owner, chain: list, ref_chain: list):
        """(index of the last token of the statement, index after it) for the header's terminator. ref_chain is
        the namespace the header's names resolve in; after a body, `} :> x;` adds specializations to the header."""
        toks = self.toks
        if term == "{":
            close = self.members(hend + 1, owner, chain, True)
            if close < self.n and toks[close].kind == "p" and toks[close].text in REDEFINES:
                hend2, term2 = self.header_end(close, eq_term=True)
                self.header_refs(close, hend2, None, None, ref_chain)
                return self.finish(term2, hend2, owner, chain, ref_chain)
            return close - 1, close
        if term == ";":
            return hend, hend + 1
        if term == "}":
            return hend - 1, hend
        # "=": a value up to ';'; a string value is text of the element
        if hend + 2 < self.n and toks[hend + 1].kind == "str" and is_p(toks[hend + 2], ";", "}"):
            self.piece(owner, toks[hend].start, toks[hend + 1].end)
        m = hend
        while m < self.n:
            t = toks[m]
            if t.kind == "p" and t.text in OPENERS:
                m = self.skip_balanced(m)
                continue
            if t.kind == "p" and t.text in (";", "}"):
                break
            m += 1
        if m >= self.n:
            self.fail(hend, "unexpected end of file")
        if is_p(toks[m], ";"):
            return m, m + 1
        return m - 1, m

    def header_end(self, m: int, eq_term: bool):
        """(index, terminator) of the header that starts at m: '{', ';', '}' or '=' outside brackets."""
        toks, k = self.toks, m
        while k < self.n:
            t = toks[k]
            if t.kind == "p":
                if t.text in ("(", "["):
                    k = self.skip_balanced(k)
                    continue
                if t.text in ("{", ";", "}"):
                    return k, t.text
                if eq_term and t.text == "=":
                    return k, "="
            k += 1
        self.fail(m, "unexpected end of file")

    def header_refs(self, k: int, hend: int, verb, frag, chain: list):
        """Record the names in a header: typing (:), specialization (:>, :>>, specializes, subsets, redefines) and
        the relation words (to, by, from, of) with the verb in front of them."""
        toks, m = self.toks, k
        while m < hend:
            t = toks[m]
            if t.kind == "p":
                if t.text == ":":
                    frag = "typed_by"
                elif t.text in (":>", "::>"):
                    frag = "specializes"
                elif t.text == ":>>":
                    frag = "redefines"
                elif t.text == "=" and verb == "bind":
                    frag = "bind_to"
                elif t.text in ("(", "["):
                    m = self.skip_balanced(m)
                    continue
                m += 1
                continue
            if t.kind == "id" and t.text in ("specializes", "subsets", "redefines"):
                frag = t.text
            elif t.kind == "id" and t.text == "of":
                frag = "typed_by"
            elif t.kind == "id" and t.text in VERBS:
                verb, frag = t.text, t.text
            elif t.kind == "id" and t.text == "to":
                frag = f"{verb}_to" if verb else None
            elif t.kind == "id" and t.text == "by":
                frag = f"{verb}_by" if verb else None
            elif t.kind == "id" and t.text == "from":
                frag = verb
            elif frag is not None and t.kind in ("id", "q") and t.text not in HEADER_SKIP:
                e, text = self.qname(m)
                self.refs.append(Ref(t.start, self.toks[e - 1].end, text, frag, chain))
                m = e
                if not (m < hend and is_p(toks[m], ",")):
                    frag = None
                continue
            m += 1

    def relation(self, k: int, owner, chain: list) -> int:
        """A relation statement with no declaration: satisfy X by Y; allocate A to B; connect a to b; perform X;"""
        verb, m = self.toks[k].text, k + 1
        if verb in ("satisfy", "verify") and m < self.n and self.toks[m].kind == "id" \
                and self.toks[m].text == "requirement":
            m += 1
        hend, term = self.header_end(m, eq_term=(verb != "bind"))
        self.header_refs(m, hend, verb, verb, chain)
        return self.finish(term, hend, owner, chain, chain)[1]

    def metadata(self, i: int, owner, chain: list) -> int:
        """#name [kind] X to Y; a metadata reference (#refinement dependency A to B)."""
        if i + 1 >= self.n or self.toks[i + 1].kind not in ("id", "q"):
            self.fail(i, "'#' is not followed by a metadata name")
        name = self.toks[i + 1].text
        hend, term = self.header_end(i + 2, eq_term=True)
        self.header_refs(i + 2, hend, name, name, chain)
        return self.finish(term, hend, owner, chain, chain)[1]

    def import_stmt(self, j: int, chain: list) -> int:
        toks, m = self.toks, j + 1
        if m < self.n and toks[m].kind in ("id", "q"):
            e, text = self.qname(m)
            self.refs.append(Ref(toks[m].start, toks[e - 1].end, text, "import", chain))
            m = e
        while m < self.n and not is_p(toks[m], ";", "}"):
            m += 1
        return m + 1 if m < self.n and is_p(toks[m], ";") else m

    def text_stmt(self, j: int, owner) -> int:
        """require / assume / assert statements: their text is text of the enclosing element."""
        toks, m, end_tok = self.toks, j + 1, None
        while m < self.n:
            t = toks[m]
            if is_p(t, "(", "["):
                m = self.skip_balanced(m)
                continue
            if is_p(t, "{"):
                m = self.skip_balanced(m)
                end_tok = m - 1
                break
            if is_p(t, ";"):
                end_tok, m = m, m + 1
                break
            if is_p(t, "}"):
                end_tok = m - 1
                break
            m += 1
        if end_tok is None:
            self.fail(j, "unexpected end of file")
        self.piece(owner, toks[j].start, toks[end_tok].end)
        return m

    def comment(self, i: int, owner) -> int:
        """comment name about X /* ... */: the block comment is text of the enclosing element."""
        m = i + 1
        while m < self.n:
            t = self.toks[m]
            if t.kind == "cmt":
                self.piece(owner, t.start, t.end)
                return m + 1
            if is_p(t, ";"):
                return m + 1
            if is_p(t, "{", "}"):
                return m
            m += 1
        return m

    def unknown(self, i: int, owner, chain: list, warn: bool = True) -> int:
        """Skip a statement this scanner does not model; a body in it is still parsed for text."""
        toks, m = self.toks, i
        if warn:
            self.unmodelled.append((self.line_of(toks[i].start), toks[i].text))
        while m < self.n:
            t = toks[m]
            if is_p(t, "}"):
                return m
            if is_p(t, ";"):
                return m + 1
            if is_p(t, "{"):
                m = self.members(m + 1, owner, chain, True)
                if m < self.n and is_p(toks[m], "->"):
                    m += 1
                    continue
                return m
            if is_p(t, "(", "["):
                m = self.skip_balanced(m)
                continue
            m += 1
        return m

    # -- passages ---------------------------------------------------------------------------------------------

    def passage_spans(self, el: Element) -> list[list[int]]:
        """[start, end] byte spans of an element's passages: contiguous pieces (gaps of whitespace only) merged."""
        if el.keyword == "attribute" and not any(not header for _, _, header in el.pieces):
            return []
        spans: list[list[int]] = []
        for s, e, _ in sorted(el.pieces):
            if spans and self.data[spans[-1][1]:s].strip() == b"":
                spans[-1][1] = max(spans[-1][1], e)
            else:
                spans.append([s, e])
        return spans


class Model:
    """All files of one model. Names are declared while the files parse; references resolve after that."""

    def __init__(self, name: str):
        self.name = name
        self.decls: dict[str, dict[str, list[str]]] = {}  # namespace qualified name -> name key -> qnames
        self.files: list[_File] = []

    def declare(self, namespace: str, name: str, qname: str):
        self.decls.setdefault(namespace, {}).setdefault(_key(name), []).append(qname)

    def resolve(self, text: str, chain: list) -> str:
        """The qualified name of a reference: its head is looked up from the innermost namespace outwards, and
        taken only when that namespace declares exactly one element of that name; otherwise the text as written."""
        head = HEAD.match(text)
        if head is None:
            return text
        key, tail = _key(head.group(0)), text[head.end():]
        for ns in reversed(chain):
            found = self.decls.get(ns, {}).get(key)
            if found:  # two declarations of one name in one namespace are ambiguous, even if identical
                return f"{found[0]}{tail}" if len(found) == 1 else text
        return text

    def records(self):
        for f in self.files:
            yield {"type": "doc", **f.doc, "extractor": "", "format": "sysml", "model": self.name}
            for el in f.elements:
                yield {"type": "section", "uri": f.uri, "path": el.path, "level": len(el.path) - 1,
                       "offset": el.start, "length": el.end - el.start, "line": el.line}
            passages = 0
            for el in f.elements:
                for s, e in f.passage_spans(el):
                    passages += 1
                    span = f.data[s:e]
                    yield {"type": "passage", "uri": f.uri, "path": el.path, "offset": s, "length": e - s,
                           "line": f.line_of(s), "end_line": f.line_of(e - 1),
                           "sha256": hashlib.sha256(span).hexdigest(), "text": span.decode("utf-8", "replace")}
            for r in sorted(f.refs, key=lambda r: r.start):
                span = f.data[r.start:r.end]
                yield {"type": "link", "uri": f.uri, "offset": r.start, "length": r.end - r.start,
                       "anchor": span.decode("utf-8", "replace"), "target": r.target, "kind": "model",
                       "fragment": r.fragment}
            yield {"type": "end", "uri": f.uri, "sections": len(f.elements), "passages": passages,
                   "links": len(f.refs)}

    def summary(self) -> dict:
        kinds, links, unmodelled = Counter(), Counter(), Counter()
        examples: dict[str, list[str]] = defaultdict(list)
        passages = stray = 0
        for f in self.files:
            kinds.update(el.kind for el in f.elements)
            links.update(r.fragment for r in f.refs)
            passages += sum(len(f.passage_spans(el)) for el in f.elements)
            stray += f.stray
            for line, word in f.unmodelled:
                unmodelled[word] += 1
                if len(examples[word]) < 3:
                    examples[word].append(f"{f.uri}:{line}")
        return {"files": len(self.files), "elements": sum(kinds.values()), "elements_by_kind": dict(kinds.most_common()),
                "passages": passages, "links": sum(links.values()), "links_by_fragment": dict(links.most_common()),
                "stray_text_pieces": stray, "unmodelled_statements": dict(unmodelled.most_common(20)),
                "unmodelled_examples": {w: examples[w] for w, _ in unmodelled.most_common(20)}}


def parse_model(sources: list, model_name: str) -> Model:
    """sources: (doc, bytes) per file, doc holding uri, sha256, size, mtime and title."""
    model = Model(model_name)
    for doc, data in sources:
        model.files.append(_File(doc, data, model))
    for f in model.files:
        for r in f.refs:
            r.target = model.resolve(r.text, r.chain)
    return model


def iter_files(roots):
    for root in roots:
        p = Path(root)
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() == SUFFIX
                              and not any(part.startswith(".") for part in f.relative_to(p).parts))
        elif p.is_file():
            yield p
        else:
            raise SystemExit(f"docmem scan-sysml: no such file or directory: {root}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docmem scan-sysml", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help=".sysml files or directories (scanned recursively)")
    ap.add_argument("--model-name", default=None,
                    help="the model's name, the root of every section path (default: the first root's name)")
    a = ap.parse_args(argv)
    paths = list(iter_files(a.roots))
    first = Path(a.roots[0]).resolve()
    name = a.model_name or (first.stem if first.is_file() else first.name)
    sources = []
    for p in paths:
        data = p.read_bytes()
        doc = {"uri": p.resolve().as_uri(), "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
               "mtime": p.stat().st_mtime, "title": p.stem}
        sources.append((doc, data))
    try:
        model = parse_model(sources, name)
    except ScanError as e:
        raise SystemExit(f"docmem scan-sysml: {e}")
    out = sys.stdout
    out.write(json.dumps({"type": "scan_start", "protocol": PROTOCOL}) + "\n")
    for rec in model.records():
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.write(json.dumps({"type": "scan_end", "roots": [Path(r).resolve().as_uri() for r in a.roots],
                          "uris": [f.uri for f in model.files]}) + "\n")
    out.flush()
    print(json.dumps(model.summary(), indent=1, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
