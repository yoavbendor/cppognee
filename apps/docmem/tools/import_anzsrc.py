"""Import the ANZSRC 2020 Fields of Research (FoR) as a SKOS scheme for `docmem ontology load`.

    python tools/import_anzsrc.py --out FILE --source PATH

PATH is the ABS workbook anzsrc2020_for.xlsx (catalogue 1297.0, released 24 October 2025), or a zip
that holds it. Table 3 gives the hierarchy and the labels: divisions (2 digits), groups (4 digits) and
fields (6 digits). Table 4 gives the definitions and exclusions of divisions and groups. Fields have no
published definition, so a field's skos:definition is its label path ("Engineering > Mechanical
engineering > Tribology"), and the embedded text always has context. A division's or group's
skos:definition is "<label path>: <definition>"; its exclusions go in skos:scopeNote, not in the
definition.

Concept URIs are https://linked.data.gov.au/def/anzsrc-for/2020/<code>, with skos:notation = code.
Where Table 3 and Table 4 give different names for one group, the Table 3 label is used and the
code is listed under name_mismatches in the summary. Output is deterministic for a given workbook.
"""

import argparse
import json
import sys
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

SCHEME = "https://linked.data.gov.au/def/anzsrc-for/2020"
CONCEPT_BASE = SCHEME + "/"
WORKBOOK = "anzsrc2020_for.xlsx"
SOURCE = "1297.0 ANZSRC - Australian and New Zealand Standard Research Classification, 2020"
ISSUED = "2025-10-24"
RIGHTS = "© Commonwealth of Australia 2025; © Crown Copyright New Zealand 2025"
CODE_LENGTH_LEVEL = {2: 1, 4: 2, 6: 3}  # division, group, field


def _cells(row) -> list[tuple[int, object]]:
    """(column index, value) for the non-empty cells of a row."""
    return [(i, c) for i, c in enumerate(row) if c not in (None, "")]


def _code(cells) -> int | None:
    """The row's code: its first non-empty cell when that is a whole number, else None."""
    if not cells:
        return None
    v = cells[0][1]
    if isinstance(v, str) and v.strip().isdigit():
        return int(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v).is_integer():
        return int(v)
    return None


def _text(v) -> str:
    """One line of text: runs of whitespace collapsed (the workbook has trailing spaces)."""
    return " ".join(str(v).split())


def _block(v) -> str | None:
    """Multi-line text with each line stripped and empty lines dropped; None when empty."""
    lines = [line.strip() for line in str(v).splitlines() if line.strip()] if v else []
    return "\n".join(lines) or None


def read_rows(source: Path) -> dict[str, list[tuple]]:
    """The rows of Table 3 and Table 4 of the FoR workbook, from the .xlsx or from a .zip that holds it."""
    import openpyxl  # imported here: the parsers below need no spreadsheet library

    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            members = [n for n in z.namelist() if Path(n).name == WORKBOOK]
            if len(members) != 1:
                raise ValueError(f"{source}: expected one {WORKBOOK}, found {len(members)}")
            data = BytesIO(z.read(members[0]))
    else:
        data = source
    wb = openpyxl.load_workbook(data, read_only=True, data_only=True)
    try:
        for name in ("Table 3", "Table 4"):
            if name not in wb.sheetnames:
                raise ValueError(f"{source}: no sheet {name!r}")
        return {name: list(wb[name].iter_rows(values_only=True)) for name in ("Table 3", "Table 4")}
    finally:
        wb.close()


def table3(rows: list[tuple]) -> list[tuple[int, str]]:
    """(code, label) for every division, group and field, in workbook order. The code is the row's first
    non-empty cell and the label the one after it; other rows (headers) have no numeric code."""
    out = []
    for row in rows:
        cells = _cells(row)
        code = _code(cells)
        if code is None:
            continue
        if len(cells) < 2 or not isinstance(cells[1][1], str):
            raise ValueError(f"Table 3: code {code} has no label")
        out.append((code, _text(cells[1][1])))
    return out


def table4(rows: list[tuple]) -> dict[int, tuple[str, str | None, str | None]]:
    """code -> (name, definition, exclusions) for each division and group. The Definition and Exclusions
    columns are found from the header row."""
    header = next((r for r in rows if "Definition" in r and "Exclusions" in r), None)
    if header is None:
        raise ValueError("Table 4: no header row with Definition and Exclusions")
    def_col, excl_col = list(header).index("Definition"), list(header).index("Exclusions")
    out: dict[int, tuple[str, str | None, str | None]] = {}
    for row in rows:
        cells = _cells(row)
        code = _code(cells)
        if code is None:
            continue
        if code in out:
            raise ValueError(f"Table 4 lists {code} twice")
        if len(cells) < 2 or not isinstance(cells[1][1], str):
            raise ValueError(f"Table 4: code {code} has no name")
        definition = _text(row[def_col]) if def_col < len(row) and row[def_col] else None
        exclusions = _block(row[excl_col]) if excl_col < len(row) else None
        out[code] = (_text(cells[1][1]), definition, exclusions)
    return out


def name_mismatches(t3: list[tuple[int, str]], t4: dict) -> list[str]:
    """Codes whose Table 4 name differs from the Table 3 label (ignoring case)."""
    labels = dict(t3)
    return [
        str(code)
        for code, (name, _, _) in sorted(t4.items())
        if code in labels and name.lower() != labels[code].lower()
    ]


def build(t3: list[tuple[int, str]], t4: dict) -> list[dict]:
    """Concepts as dicts {uri, code, label, definition, scope_note, broader, level, path}, in workbook order.
    A parent must come before its children, which the workbook guarantees."""
    codes = [code for code, _ in t3]
    if len(set(codes)) != len(codes):
        dup = sorted({c for c in codes if codes.count(c) > 1})
        raise ValueError(f"Table 3 lists codes twice: {dup}")
    for code in codes:
        if len(str(code)) not in CODE_LENGTH_LEVEL:
            raise ValueError(f"code {code} is not 2, 4 or 6 digits")
    extra = sorted(set(t4) - set(codes))
    if extra:
        raise ValueError(f"Table 4 has codes that are not in Table 3: {extra}")

    path: dict[int, str] = {}
    out = []
    for code, label in t3:
        length = len(str(code))
        level = CODE_LENGTH_LEVEL[length]
        parent = code // 100 if level > 1 else None
        if parent is not None and parent not in path:
            raise ValueError(
                f"{code} {label}: parent {parent} is not a division or group in Table 3"
            )
        own_path = label if parent is None else f"{path[parent]} > {label}"
        path[code] = own_path
        if level == 3:
            definition, scope = own_path, None
        else:
            if code not in t4 or not t4[code][1]:
                raise ValueError(f"Table 4 has no definition for {code} {label}")
            _, text, scope = t4[code]
            definition = f"{own_path}: {text}"
        out.append(
            {
                "uri": f"{CONCEPT_BASE}{code}",
                "code": str(code),
                "label": label,
                "definition": definition,
                "scope_note": scope,
                "broader": f"{CONCEPT_BASE}{parent}" if parent is not None else None,
                "level": level,
                "path": own_path,
            }
        )
    return out


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


def turtle(concepts: list[dict]) -> str:
    lines = [
        f"# ANZSRC 2020 Fields of Research, {SOURCE}, released {ISSUED}. {RIGHTS}.",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "@prefix dcterms: <http://purl.org/dc/terms/> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
        f"<{SCHEME}> a skos:ConceptScheme ;",
        f"    skos:prefLabel {ttl('ANZSRC 2020 Fields of Research (FoR)')}@en ;",
        f"    dcterms:source {ttl(SOURCE)} ;",
        '    dcterms:hasVersion "2020" ;',
        f'    dcterms:issued "{ISSUED}"^^xsd:date ;',
        f"    dcterms:rights {ttl(RIGHTS)} .",
        "",
    ]
    for c in concepts:
        pairs = [("a", "skos:Concept"), ("skos:inScheme", f"<{SCHEME}>")]
        if c["broader"] is None:
            pairs.append(("skos:topConceptOf", f"<{SCHEME}>"))
        else:
            pairs.append(("skos:broader", f"<{c['broader']}>"))
        pairs.append(("skos:notation", ttl(c["code"])))
        pairs.append(("skos:prefLabel", ttl(c["label"]) + "@en"))
        pairs.append(("skos:definition", ttl(c["definition"]) + "@en"))
        if c["scope_note"]:
            pairs.append(("skos:scopeNote", ttl(c["scope_note"]) + "@en"))
        lines.append(f"<{c['uri']}>")
        lines.append(" ;\n".join(f"    {p} {o}" for p, o in pairs) + " .")
        lines.append("")
    return "\n".join(lines)


def summarize(concepts: list[dict], mismatches: list[str]) -> dict:
    levels = Counter(c["level"] for c in concepts)
    divisions = [c for c in concepts if c["level"] == 1]
    return {
        "concepts": len(concepts),
        "divisions": levels[1],
        "groups": levels[2],
        "fields": levels[3],
        "groups_with_exclusions": sum(1 for c in concepts if c["level"] == 2 and c["scope_note"]),
        "groups_per_division": {
            d["code"]: sum(1 for g in concepts if g["level"] == 2 and g["broader"] == d["uri"])
            for d in divisions
        },
        "name_mismatches": mismatches,
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
        help="anzsrc2020_for.xlsx, or a zip that holds it",
    )
    a = ap.parse_args(argv)
    tables = read_rows(a.source)
    t3 = table3(tables["Table 3"])
    t4 = table4(tables["Table 4"])
    concepts = build(t3, t4)
    mismatches = name_mismatches(t3, t4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(turtle(concepts), encoding="utf-8")
    for code in mismatches:
        print(
            f"import_anzsrc: {code}: Table 3 and Table 4 names differ, Table 3 label used",
            file=sys.stderr,
        )
    print(json.dumps(summarize(concepts, mismatches), indent=1))


if __name__ == "__main__":
    main()
