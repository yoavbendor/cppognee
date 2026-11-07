"""Model export and model checks: kind mapping, labels and spans, the nomograph index shape, finding parsing.
No DB and no nomograph binary.

Run from apps/docmem:  python -m pytest tests/test_model_export.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from docmem.model_check import parse_findings, select_gaps, severity_of  # noqa: E402
from docmem.model_export import (  # noqa: E402
    Element,
    Relation,
    build_index,
    doc_comment,
    element_span,
    layer_of,
    line_span,
    line_starts,
    nomo_kind,
    parse_label,
    split_path,
)

MODEL_URI = "file:///proj/m/a.sysml"
LIB_URI = "file:///lib/l.sysml"
SRC = b"package P {\n\tpart def Car {\n\t\tattribute mass;\n\t}\n\tpart car : Car;\n}\n"


@pytest.mark.parametrize(
    "raw, kind",
    [
        ("part def", "part_definition"),
        ("individual part def", "part_definition"),  # a modifier changes no kind
        ("abstract part def", "part_definition"),
        ("part", "part_usage"),
        ("then timeslice", "timeslice_usage"),
        ("then action", "action_usage"),
        ("do action", "action_usage"),
        ("perform action", "action_usage"),
        ("in", "parameter_usage"),
        ("out item", "parameter_usage"),
        ("inout item", "parameter_usage"),
        ("return", "parameter_usage"),
        ("end", "end_usage"),
        ("exhibit state", "exhibit_usage"),
        ("library package", "library_package"),
        ("package", "package_definition"),
        ("enum def", "enumeration_definition"),
        ("redefinition", "feature_usage"),
        ("specialization", "feature_usage"),
        ("snapshot", "snapshot_usage"),
    ],
)
def test_keyword_maps_to_nomograph_kind(raw, kind):
    assert nomo_kind(raw) == kind


@pytest.mark.parametrize("raw", ["subject", "datatype", "enum", "", "behavior"])
def test_unlisted_keyword_is_unmapped(raw):
    assert nomo_kind(raw) is None


def test_layer_follows_kind():
    assert layer_of("requirement_usage") == "Requirements"
    assert layer_of("action_usage") == "Functional"
    assert layer_of("part_definition") == "Logical"
    assert layer_of("package_definition") is None


def test_labels_and_paths():
    assert parse_label("part def Car") == ("part def", "Car")
    assert parse_label("individual part def 'James Lovell'") == (
        "individual part def",
        "James Lovell",
    )
    assert parse_label("attribute 'per cent'") == ("attribute", "per cent")
    assert split_path("model › m › package P › part def 'a b'") == ("m", "P::a b", "part def")
    with pytest.raises(ValueError):
        split_path("model › m")


def test_spans_are_zero_based_and_end_exclusive():
    starts = line_starts(SRC)
    assert element_span(starts, 0, len(SRC) - 1) == (
        0,
        0,
        5,
        1,
    )  # the package: line 1 to its closing brace
    mass = SRC.index(b"attribute mass;")
    assert element_span(starts, mass, len(b"attribute mass;")) == (2, 2, 2, 17)
    car_line = line_span(SRC, starts, 5)  # line 5 is "\tpart car : Car;"
    assert car_line == (4, 1, 4, 16)


def test_doc_comment():
    assert doc_comment(b"doc /* Apollo 11  group. */ part x;") == "Apollo 11  group."
    assert doc_comment(b"part x;") is None


def _tiny_model():
    pkg = Element("p", "m", "P", "package", MODEL_URI, (0, 0, 5, 1))
    car = Element("car_def", "m", "P::Car", "part def", MODEL_URI, (1, 1, 3, 2), doc="A car.")
    mass = Element("mass", "m", "P::Car::mass", "attribute", MODEL_URI, (2, 2, 2, 17))
    part = Element("car", "m", "P::car", "part", MODEL_URI, (4, 1, 4, 16))
    base = Element("base", "lib", "L::Base", "part def", LIB_URI, (0, 0, 0, 10))
    weird = Element("w", "m", "P::Weird", "subject", MODEL_URI, (0, 0, 0, 5))
    rels = [
        Relation("car", "car_def", "TypedBy", MODEL_URI, (4, 1, 4, 16)),
        Relation("car_def", "base", "Specialize", MODEL_URI, (1, 1, 1, 20)),
        Relation(
            "p", "base", "TypedBy", MODEL_URI, (0, 0, 0, 1)
        ),  # from the package: kept only if its target is in
    ]
    return [pkg, car, mass, part, base, weird], rels


def test_export_shape_and_library_closure():
    export = build_index(*_tiny_model(), model="m", include_library=False)
    index = export.index
    assert set(index) == {"elements", "relationships"}
    names = {e["qualified_name"] for e in index["elements"]}
    assert names == {
        "P",
        "P::Car",
        "P::Car::mass",
        "P::car",
        "P::Weird",
        "L::Base",
    }  # L::Base is referred to
    by = {e["qualified_name"]: e for e in index["elements"]}
    assert by["P::Car"] == {
        "qualified_name": "P::Car",
        "kind": "part_definition",
        "file_path": "./a.sysml",
        "span": {"start_line": 1, "start_col": 1, "end_line": 3, "end_col": 2},
        "doc": "A car.",
        "attributes": [],
        "members": ["P::Car::mass"],
        "layer": "Logical",
    }
    assert "layer" not in by["P"]  # a package has no layer
    assert by["P::Weird"]["kind"] == "generic_usage"
    assert by["L::Base"]["file_path"] == "/lib/l.sysml"  # outside the model's root: absolute
    assert export.report["unmapped_keywords"] == {"subject": 1}
    assert export.report["library_elements"] == 1


def test_export_relationships_and_members():
    export = build_index(*_tiny_model(), model="m", include_library=False)
    rels ={(r["source"], r["target"], r["kind"]) for r in export.index["relationships"]}
    assert ("P", "P::Car", "Member") in rels
    assert ("P", "P::car", "Member") in rels
    assert ("P::Car", "P::Car::mass", "Member") in rels
    assert ("P::car", "P::Car", "TypedBy") in rels
    assert ("P::Car", "L::Base", "Specialize") in rels
    # the package's TypedBy names the library element, but the package is not a source of a kept edge: it is kept
    assert ("P", "L::Base", "TypedBy") in rels
    typed = next(
        r
        for r in export.index["relationships"]
        if r["kind"] == "TypedBy" and r["source"] == "P::car"
    )
    assert typed["span"] == {"start_line": 4, "start_col": 1, "end_line": 4, "end_col": 16}


def test_library_excluded_without_reference():
    elements, rels = _tiny_model()
    export = build_index(elements, [r for r in rels if r.source != "car_def"], "m", False)
    names = {e["qualified_name"] for e in export.index["elements"]}
    assert "L::Base" in names  # still referenced by P's TypedBy
    export = build_index(elements, [], "m", False)
    assert "L::Base" not in {e["qualified_name"] for e in export.index["elements"]}
    assert export.report["library_elements"] == 0
    export = build_index(elements, [], "m", True)
    assert "L::Base" in {e["qualified_name"] for e in export.index["elements"]}


def test_findings_from_check_document():
    doc = {
        "counts": {"UnverifiedRequirements": 1, "DanglingReferences": 1},
        "finding_count": 2,
        "findings": [
            {
                "check_type": "UnverifiedRequirements",
                "element": "P::clr-R001",
                "file_path": "./Requirements/T.sysml",
                "message": "Requirement has no verify relationship",
                "span": {"start_line": 12, "end_line": 20},
            },
            {
                "check_type": "DanglingReferences",
                "element": "P::StagingPort",
                "file_path": "./Technical/Ports.sysml",
                "message": "port has no TypedBy relationship",
                "span": {"start_line": 25, "end_line": 29},
            },
        ],
    }
    found = parse_findings(doc)
    assert [(f.check, f.element, f.line, f.severity) for f in found] == [
        ("UnverifiedRequirements", "P::clr-R001", 13, "warning"),
        ("DanglingReferences", "P::StagingPort", 26, "error"),
    ]
    assert found[0].file_path == "./Requirements/T.sysml"
    assert severity_of("MetamodelConformance") == "error"


def test_select_gaps_by_check_and_prefix():
    rows = [
        {"check": "UnverifiedRequirements", "name": "Tech::Pkg::clr-R001"},
        {"check": "UnverifiedRequirements", "name": "Tech::PkgOther::x"},
        {"check": "OrphanRequirements", "name": "Tech::Pkg::clr-R002"},
    ]
    assert len(select_gaps(rows, None, None)) == 3
    assert [r["name"] for r in select_gaps(rows, "UnverifiedRequirements", "Tech::Pkg")] == [
        "Tech::Pkg::clr-R001"
    ]
    assert len(select_gaps(rows, None, "Tech::Pk")) == 0  # a prefix must end at a segment
