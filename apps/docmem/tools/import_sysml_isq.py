"""Import the SysML v2 ISQ quantities (ISO 80000) as a SKOS scheme for `docmem ontology load`.

    python tools/import_sysml_isq.py --out FILE --source PATH

PATH is a checkout of github.com/Systems-Modeling/SysML-v2-Release (or its
"sysml.library/Domain Libraries/Quantities and Units" directory, or a zip of either). Each ISQ<Part>
package becomes a level-1 concept under the root "physical quantity (ISO 80000)"; each scalar or
dimension-one quantity value (`attribute def XxxValue :> ScalarQuantityValue | DimensionOneValue`)
becomes a level-2 concept under its package. Units, vector and tensor values are not quantities
here and are skipped. A quantity's skos:definition is its `definition:` field; without one it is
the label path "Parent > Child", so the embedded text always has context.

Concept URIs are the SysML qualified names: urn:sysml:isq:ISQMechanics:MassValue. Output is
deterministic for a given source.
"""

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

SCHEME = "urn:sysml:isq"
ROOT = SCHEME + ":physical-quantity"
ROOT_LABEL = "physical quantity (ISO 80000)"
ROOT_DEFINITION = "International system of quantities (ISQ), as defined in ISO/IEC 80000"
REPO = "https://github.com/Systems-Modeling/SysML-v2-Release"
LICENSE = "https://www.eclipse.org/legal/epl-2.0/"
QUANTITY_SUPERTYPES = ("ScalarQuantityValue", "DimensionOneValue")
PACKAGE_RE = re.compile(r"^\s*standard library package\s+(ISQ\w*)\s*\{", re.M)
VALUE_DEF_RE = re.compile(r"^\s*attribute def\s+(\w+Value)\s*:>\s*(\w+)\s*\{", re.M)
DOC_RE = re.compile(r"\s*doc\s*/\*(.*?)\*/", re.S)
FIELD_RE = re.compile(
    r"^(source|symbol\(s\)|application domain|name|quantity dimension|"
    r"measurement unit\(s\)|tensor order|definition|remarks):\s*(.*)$"
)
STANDARD_RE = re.compile(
    r"from standard (.+)$", re.M
)  # "Generated on ... from standard ISO-80000-4:2019 "Mechanics""
STANDARD_TITLE_RE = re.compile(r'"([^"]+)"')


def words(camel: str) -> str:
    """'ThermodynamicTemperature' -> 'thermodynamic temperature'."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", camel).lower()


def doc_fields(text: str, pos: int) -> dict:
    """The `key: value` fields of the doc comment that starts at pos (empty when there is none).
    A value continues on the following lines until the next known key."""
    m = DOC_RE.match(text, pos)
    if not m:
        return {}
    fields, key = {}, None
    for raw in m.group(1).splitlines():
        line = raw.strip().lstrip("*").strip()
        if not line:
            continue
        f = FIELD_RE.match(line)
        if f:
            key = f.group(1)
            fields[key] = f.group(2).strip()
        elif key:
            fields[key] = (fields[key] + " " + line).strip()
    return fields


def ttl(s: str) -> str:
    """A Turtle string literal."""
    esc = (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{esc}"'


def parse_isq(text: str) -> dict:
    """One ISQ<Part> file -> {"package", "label", "definition", "quantities": [(uri, label, definition or None)]}."""
    pm = PACKAGE_RE.search(text)
    if not pm:
        raise ValueError("no 'standard library package ISQ...' header")
    pkg = pm.group(1)
    standard = ""
    if m := DOC_RE.match(text, pm.end()):
        found = STANDARD_RE.search(m.group(1))
        standard = found.group(1).strip() if found else ""
    title = STANDARD_TITLE_RE.search(standard)
    quantities = []
    for m in VALUE_DEF_RE.finditer(text):
        if m.group(2) not in QUANTITY_SUPERTYPES:
            continue
        name = m.group(1)[: -len("Value")]
        definition = doc_fields(text, m.end()).get("definition") or None
        quantities.append((f"{SCHEME}:{pkg}:{m.group(1)}", words(name), definition))
    return {
        "package": pkg,
        "label": title.group(1).lower() if title else words(pkg[len("ISQ") :]),
        "definition": standard or None,
        "quantities": quantities,
    }


def read_sources(source: Path) -> list[tuple[str, str]]:
    """(file name, text) for every ISQ*.sysml under source: a directory, one .sysml file, or a .zip."""
    if source.is_file() and source.suffix == ".zip":
        with zipfile.ZipFile(source) as z:
            members = [
                n
                for n in sorted(z.namelist())
                if n.endswith(".sysml")
                and n.split("/")[-2:-1] == ["Quantities and Units"]
                and Path(n).name.startswith("ISQ")
            ]
            return [(Path(n).name, z.read(n).decode("utf-8")) for n in members]
    if source.is_file():
        return [(source.name, source.read_text(encoding="utf-8"))]
    files = [
        p for p in sorted(source.rglob("ISQ*.sysml")) if p.parent.name == "Quantities and Units"
    ]
    return [(p.name, p.read_text(encoding="utf-8")) for p in files]


def build(parts: list[dict]) -> list[dict]:
    """Concepts as dicts {uri, label, definition, broader}, root first, then packages and quantities in
    source order. A concept without a definition gets its label path as its definition."""
    root_path = ROOT_LABEL
    out = [
        {
            "uri": ROOT,
            "label": ROOT_LABEL,
            "definition": ROOT_DEFINITION,
            "broader": None,
            "path": root_path,
        }
    ]
    for part in parts:
        if (
            part["package"] == "ISQ"
        ):  # the umbrella package: its own quantities sit directly under the root
            pkg_uri, pkg_path = ROOT, root_path
        else:
            pkg_uri = f"{SCHEME}:{part['package']}"
            pkg_path = f"{root_path} > {part['label']}"
            out.append(
                {
                    "uri": pkg_uri,
                    "label": part["label"],
                    "definition": part["definition"],
                    "broader": ROOT,
                    "path": pkg_path,
                }
            )
        for uri, label, definition in part["quantities"]:
            out.append(
                {
                    "uri": uri,
                    "label": label,
                    "definition": definition,
                    "broader": pkg_uri,
                    "path": f"{pkg_path} > {label}",
                }
            )
    seen = set()
    for c in out:
        if c["uri"] in seen:
            raise ValueError(f"duplicate concept URI {c['uri']}")
        seen.add(c["uri"])
        if c["definition"] is None:
            c["definition"] = c["path"]
    return out


def turtle(concepts: list[dict]) -> str:
    lines = [
        "# SysML v2 ISQ quantities (ISO 80000), from the SysML-v2-Release repository (EPL-2.0).",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "@prefix dcterms: <http://purl.org/dc/terms/> .",
        "",
        f"<{SCHEME}> a skos:ConceptScheme ;",
        f"    skos:prefLabel {ttl('SysML v2 ISQ quantities')}@en ;",
        f"    dcterms:source <{REPO}> ;",
        f"    dcterms:license <{LICENSE}> .",
        "",
    ]
    for c in concepts:
        lines.append(f"<{c['uri']}> a skos:Concept ;")
        lines.append(f"    skos:inScheme <{SCHEME}> ;")
        if c["broader"] is None:
            lines.append(f"    skos:topConceptOf <{SCHEME}> ;")
        else:
            lines.append(f"    skos:broader <{c['broader']}> ;")
        lines.append(f"    skos:prefLabel {ttl(c['label'])}@en ;")
        lines.append(f"    skos:definition {ttl(c['definition'])}@en .")
        lines.append("")
    return "\n".join(lines)


def summarize(parts: list[dict], concepts: list[dict]) -> dict:
    with_def = sum(1 for p in parts for q in p["quantities"] if q[2])
    return {
        "concepts": len(concepts),
        "packages": {p["package"]: len(p["quantities"]) for p in parts},
        "quantities": sum(len(p["quantities"]) for p in parts),
        "quantities_with_definition": with_def,
        "quantities_with_path_definition": sum(len(p["quantities"]) for p in parts) - with_def,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True, help="the Turtle file to write")
    ap.add_argument(
        "--source",
        type=Path,
        required=True,
        help="SysML-v2-Release checkout, its Quantities and Units directory, a .sysml file or a zip",
    )
    a = ap.parse_args(argv)
    files = read_sources(a.source)
    if not files:
        sys.exit(f"import_sysml_isq: no ISQ*.sysml files under {a.source}")
    parts = [parse_isq(text) for _, text in sorted(files)]
    concepts = build(parts)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(turtle(concepts), encoding="utf-8")
    print(json.dumps(summarize(parts, concepts), indent=1))


if __name__ == "__main__":
    main()
