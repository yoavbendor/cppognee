"""Importers for SKOS schemes (tools/import_*.py): parsing of inline samples. No network.

Run from apps/docmem:  python -m pytest tests/test_importers.py
"""

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
