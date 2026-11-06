"""Importers for SKOS schemes (tools/import_*.py): parsing of inline samples. No network.

Run from apps/docmem:  python -m pytest tests/test_importers.py
"""

import json
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "tools"))

import pytest  # noqa: E402
from import_sysml_isq import build, parse_isq, read_sources, summarize, turtle  # noqa: E402

ISQ_SAMPLE = """standard library package ISQMechanics {
    doc
    /*
     * International System of Quantities and Units
     * Generated on 2025-03-13T15:00:05Z from standard ISO-80000-4:2019 "Mechanics"
     * see also https://www.iso.org/standard/64975.html
     */

    /* ISO-80000-4 item 4-1 mass */
    attribute def MassValue :> ScalarQuantityValue {
        doc
        /*
         * source: item 4-1 mass
         * symbol(s): `m`
         * name: Mass
         * quantity dimension: M^1
         * definition: property of a body which expresses "inertia" with regard to changes in its
         * state of motion
         * remarks: The kilogram (kg) is a base unit.
         */
        attribute :>> num: Real;
    }

    attribute def MassUnit :> SimpleUnit {
        attribute :>> quantityDimension { }
    }

    attribute def PositionVector3dValue :> '3dVectorQuantityValue' {
        doc /* definition: a vector, not a scalar quantity */
    }

    attribute def PackingFractionValue :> DimensionOneValue {
        doc
        /*
         * name: PackingFraction
         * definition: ratio of the volume of the particles to the volume of the cell
         */
    }

    attribute def UndocumentedValue :> ScalarQuantityValue {
        attribute :>> num: Real;
    }
}
"""


def test_parse_keeps_scalar_and_dimension_one_quantities_only():
    part = parse_isq(ISQ_SAMPLE)
    assert part["package"] == "ISQMechanics"
    assert part["label"] == "mechanics"
    assert part["definition"] == 'ISO-80000-4:2019 "Mechanics"'
    assert [q[0] for q in part["quantities"]] == [
        "urn:sysml:isq:ISQMechanics:MassValue",
        "urn:sysml:isq:ISQMechanics:PackingFractionValue",
        "urn:sysml:isq:ISQMechanics:UndocumentedValue",
    ]


def test_definition_spans_lines_and_keeps_quotes():
    mass = parse_isq(ISQ_SAMPLE)["quantities"][0]
    assert mass[1] == "mass"
    assert (
        mass[2]
        == 'property of a body which expresses "inertia" with regard to changes in its state of motion'
    )


def test_build_uses_label_path_when_definition_missing():
    concepts = {c["uri"]: c for c in build([parse_isq(ISQ_SAMPLE)])}
    assert concepts["urn:sysml:isq:physical-quantity"]["broader"] is None
    assert concepts["urn:sysml:isq:ISQMechanics"]["broader"] == "urn:sysml:isq:physical-quantity"
    undocumented = concepts["urn:sysml:isq:ISQMechanics:UndocumentedValue"]
    assert undocumented["label"] == "undocumented"
    assert undocumented["definition"] == "physical quantity (ISO 80000) > mechanics > undocumented"


def test_turtle_loads_in_rdflib_with_labels_and_broader(tmp_path):
    rdflib = pytest.importorskip("rdflib")
    from rdflib import RDF, Namespace

    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    concepts = build([parse_isq(ISQ_SAMPLE)])
    out = tmp_path / "isq.ttl"
    out.write_text(turtle(concepts), encoding="utf-8")
    g = rdflib.Graph()
    g.parse(out, format="turtle")
    mass = rdflib.URIRef("urn:sysml:isq:ISQMechanics:MassValue")
    assert (mass, RDF.type, SKOS.Concept) in g
    assert (mass, SKOS.broader, rdflib.URIRef("urn:sysml:isq:ISQMechanics")) in g
    labels = {str(o): o.language for o in g.objects(mass, SKOS.prefLabel)}
    assert labels == {"mass": "en"}
    assert "inertia" in str(next(g.objects(mass, SKOS.definition)))


def test_summary_counts_per_package():
    parts = [parse_isq(ISQ_SAMPLE)]
    s = summarize(parts, build(parts))
    assert s["packages"] == {"ISQMechanics": 3}
    assert s["quantities"] == 3
    assert s["quantities_with_definition"] == 2
    assert s["concepts"] == 1 + 1 + 3


def test_read_sources_from_zip_and_directory(tmp_path):
    lib = "sysml.library/Domain Libraries/Quantities and Units/"
    zpath = tmp_path / "release.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr(f"SysML-v2-Release-abc/{lib}ISQMechanics.sysml", ISQ_SAMPLE)
        z.writestr(f"SysML-v2-Release-abc/{lib}ISQ.sysml", "standard library package ISQ {}")
        z.writestr(f"SysML-v2-Release-abc/{lib}ISQMechanics.sysmlx", "not a source file")
    names = [n for n, _ in read_sources(zpath)]
    assert names == ["ISQ.sysml", "ISQMechanics.sysml"]

    d = tmp_path / "Quantities and Units"
    d.mkdir()
    (d / "ISQMechanics.sysml").write_text(ISQ_SAMPLE, encoding="utf-8")
    (d / "ISQ.sysml").write_text("standard library package ISQ {}", encoding="utf-8")
    assert [n for n, _ in read_sources(tmp_path)] == ["ISQ.sysml", "ISQMechanics.sysml"]


def test_umbrella_package_quantities_sit_under_root():
    umbrella = "standard library package ISQ {\n    attribute def TemperatureDifferenceValue :> ScalarQuantityValue {\n        doc /* definition: difference of two temperatures */\n    }\n}\n"
    concepts = {c["uri"]: c for c in build([parse_isq(umbrella)])}
    assert set(concepts) == {
        "urn:sysml:isq:physical-quantity",
        "urn:sysml:isq:ISQ:TemperatureDifferenceValue",
    }
    assert (
        concepts["urn:sysml:isq:ISQ:TemperatureDifferenceValue"]["broader"]
        == "urn:sysml:isq:physical-quantity"
    )


# ---- ANZSRC 2020 Fields of Research (tools/import_anzsrc.py) -------------------------------------

import import_anzsrc as anzsrc  # noqa: E402


def t3_div(code, label):
    return (code, label, None, None, None)


def t3_grp(code, label):
    return (None, code, label, None, None)


def t3_fld(code, label):
    return (None, None, code, label, None)


def t4_div(code, name, definition, exclusions=None):
    return (code, name, None, definition, exclusions)


def t4_grp(code, name, definition, exclusions=None):
    return (None, code, name, definition, exclusions)


FOR_TABLE3 = [
    ("Table 3 title",),
    (None, None, "Field"),
    t3_div(40, "ENGINEERING "),
    t3_grp(4001, "Civil engineering"),
    t3_fld(400101, "Structural engineering"),
    t3_grp(4002, "Mechanical engineering"),
    t3_fld(400201, "Tribology  "),
]
FOR_TABLE4 = [
    (None, "Group", None, "Definition", "Exclusions"),
    t4_div(40, "ENGINEERING", "This division covers engineering.", None),
    t4_grp(
        4001,
        "Civil engineering",
        "This group covers civil engineering.\nIt includes:\n• bridges;",
        "a) Surveying is included in Group 4013 Geomatic engineering.\nb) Mining is excluded.",
    ),
    t4_grp(4002, "Mechanical engineering and design", "This group covers machines.", None),
]


def test_anzsrc_hierarchy_labels_and_definitions():
    concepts = {
        c["code"]: c for c in anzsrc.build(anzsrc.table3(FOR_TABLE3), anzsrc.table4(FOR_TABLE4))
    }
    assert list(concepts) == ["40", "4001", "400101", "4002", "400201"]
    assert concepts["40"]["broader"] is None and concepts["40"]["level"] == 1
    assert concepts["40"]["label"] == "ENGINEERING"
    assert concepts["40"]["definition"] == "ENGINEERING: This division covers engineering."
    assert concepts["40"]["scope_note"] is None
    civil = concepts["4001"]
    assert civil["broader"] == "https://linked.data.gov.au/def/anzsrc-for/2020/40"
    assert civil["definition"] == (
        "ENGINEERING > Civil engineering: This group covers civil engineering. It includes: • bridges;"
    )
    assert civil["scope_note"] == (
        "a) Surveying is included in Group 4013 Geomatic engineering.\nb) Mining is excluded."
    )
    assert "Surveying" not in civil["definition"]
    field = concepts["400101"]
    assert field["definition"] == "ENGINEERING > Civil engineering > Structural engineering"
    assert field["scope_note"] is None and field["level"] == 3
    assert concepts["400201"]["label"] == "Tribology"
    assert concepts["400201"]["definition"] == "ENGINEERING > Mechanical engineering > Tribology"


def test_anzsrc_name_difference_keeps_table3_label_and_is_reported():
    t3, t4 = anzsrc.table3(FOR_TABLE3), anzsrc.table4(FOR_TABLE4)
    assert anzsrc.name_mismatches(t3, t4) == ["4002"]
    concepts = {c["code"]: c for c in anzsrc.build(t3, t4)}
    assert concepts["4002"]["label"] == "Mechanical engineering"


def test_anzsrc_rejects_broken_hierarchy_and_missing_definitions():
    orphan = [t3_div(40, "ENGINEERING"), t3_fld(400101, "Structural engineering")]
    with pytest.raises(ValueError, match="parent 4001"):
        anzsrc.build(anzsrc.table3(orphan), anzsrc.table4(FOR_TABLE4[:2]))
    no_definition = [t3_div(40, "ENGINEERING"), t3_grp(4001, "Civil engineering")]
    with pytest.raises(ValueError, match="no definition for 4001"):
        anzsrc.build(anzsrc.table3(no_definition), anzsrc.table4(FOR_TABLE4[:2]))
    with pytest.raises(ValueError, match="not in Table 3"):
        anzsrc.build(anzsrc.table3(FOR_TABLE3[:2]), anzsrc.table4(FOR_TABLE4))


def write_for_workbook(path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in (("Table 3", FOR_TABLE3), ("Table 4", FOR_TABLE4)):
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(list(row))
    wb.save(path)
    return path


def test_anzsrc_reads_workbook_and_zip_and_writes_loadable_turtle(tmp_path, capsys):
    rdflib = pytest.importorskip("rdflib")
    from rdflib import RDF, Namespace

    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    xlsx = write_for_workbook(tmp_path / "anzsrc2020_for.xlsx")
    zpath = tmp_path / "Download-all.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.write(xlsx, "bundle/anzsrc2020_for.xlsx")
        z.writestr("bundle/anzsrc2020_toa.xlsx", "not the FoR workbook")
    assert anzsrc.read_rows(zpath)["Table 3"] == anzsrc.read_rows(xlsx)["Table 3"]

    out = tmp_path / "anzsrc_for.ttl"
    anzsrc.main(["--out", str(out), "--source", str(zpath)])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["divisions"] == 1 and summary["groups"] == 2 and summary["fields"] == 2
    assert summary["groups_per_division"] == {"40": 2}
    assert summary["name_mismatches"] == ["4002"]
    assert "4002: Table 3 and Table 4 names differ" in captured.err

    g = rdflib.Graph()
    g.parse(out, format="turtle")
    base = "https://linked.data.gov.au/def/anzsrc-for/2020/"
    scheme = rdflib.URIRef(base.rstrip("/"))
    civil = rdflib.URIRef(base + "4001")
    assert (civil, RDF.type, SKOS.Concept) in g
    assert (civil, SKOS.broader, rdflib.URIRef(base + "40")) in g
    assert (rdflib.URIRef(base + "40"), SKOS.topConceptOf, scheme) in g
    assert str(g.value(civil, SKOS.notation)) == "4001"
    assert "Mining is excluded" in str(g.value(civil, SKOS.scopeNote))
    assert "Mining" not in str(g.value(civil, SKOS.definition))
    assert str(g.value(civil, SKOS.prefLabel)) == "Civil engineering"
    assert str(g.value(scheme, rdflib.URIRef("http://purl.org/dc/terms/hasVersion"))) == "2020"
    assert "Crown Copyright New Zealand 2025" in str(
        g.value(scheme, rdflib.URIRef("http://purl.org/dc/terms/rights"))
    )


def test_anzsrc_read_rows_rejects_zip_without_workbook(tmp_path):
    zpath = tmp_path / "other.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("readme.txt", "no workbook here")
    with pytest.raises(ValueError, match="expected one anzsrc2020_for.xlsx"):
        anzsrc.read_rows(zpath)
