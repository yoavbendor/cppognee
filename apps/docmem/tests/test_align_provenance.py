"""Provenance alignment: @Provenance annotations in a model file, their owner element, their source strings, and how
an anchor (a markdown heading or a code symbol) names passages of a prose file. No cognee, no network.

Run from apps/docmem:  python -m pytest tests/test_align_provenance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from docmem.align import heading_slug, owner_of, parse_source, provenance_annotations, resolve_anchor  # noqa: E402

MODEL = b"""package P {
    part def Engine {
        @Provenance { source = "nanos3reader@ff28f0b:src/s3_reader.cpp#CredentialProvider::resolve"; maturity = "implemented"; }
        part pump : Pump {
            @Provenance{source="nanos3reader@ff28f0b:README.md#why";maturity="verified"}
        }
    }
    // a comment that says @Provenance without a brace is not an annotation
}
"""


def test_annotations_are_found_with_their_source_and_maturity():
    anns = provenance_annotations(MODEL)
    assert [a["source"] for a in anns] == [
        "nanos3reader@ff28f0b:src/s3_reader.cpp#CredentialProvider::resolve", "nanos3reader@ff28f0b:README.md#why"]
    assert [a["maturity"] for a in anns] == ["implemented", "verified"]
    assert MODEL[anns[0]["offset"]:].startswith(b"@Provenance")


def test_annotation_without_source_is_a_model_error():
    with pytest.raises(ValueError, match="no source"):
        provenance_annotations(b'part def X { @Provenance { maturity = "verified"; } }')


def test_source_splits_into_repo_path_and_anchor():
    assert parse_source("nanos3reader@ff28f0b:src/s3_reader.cpp#CredentialProvider::resolve") == (
        "nanos3reader@ff28f0b", "src/s3_reader.cpp", "CredentialProvider::resolve")
    assert parse_source("nanos3reader@ff28f0b:tests/sigv4_test.cpp") == (
        "nanos3reader@ff28f0b", "tests/sigv4_test.cpp", None)
    with pytest.raises(ValueError):
        parse_source("no-colon-here")


def test_heading_slug_matches_github_anchors():
    assert heading_slug("Why") == "why"
    assert heading_slug("Scope & non-goals") == "scope--non-goals"
    assert heading_slug("Configuration (environment)") == "configuration-environment"
    assert heading_slug("Small static builds (mbedTLS)") == "small-static-builds-mbedtls"
    assert heading_slug("SigV4 crypto backend") == "sigv4-crypto-backend"


def test_owner_is_the_innermost_element_whose_body_holds_the_annotation():
    uri = "file:///m/a.sysml"
    els = [
        {"uri": uri, "offset": 0, "length": 200, "section_id": "pkg", "qname": "P"},
        {"uri": uri, "offset": 10, "length": 150, "section_id": "eng", "qname": "P::Engine"},
        {"uri": uri, "offset": 60, "length": 40, "section_id": "pump", "qname": "P::Engine::pump"},
        {"uri": "file:///m/b.sysml", "offset": 0, "length": 999, "section_id": "other", "qname": "Q"},
    ]
    assert owner_of(els, uri, 70)["section_id"] == "pump"
    assert owner_of(els, uri, 20)["section_id"] == "eng"
    assert owner_of(els, uri, 5)["section_id"] == "pkg"
    assert owner_of(els, uri, 500) is None


PASSAGES = [  # a markdown document: the section path is the passage's path
    {"id": "p1", "path": "nanos3reader › Why", "line": 1, "end_line": 2},
    {"id": "p2", "path": "nanos3reader › Features", "line": 4, "end_line": 5},
    {"id": "p3", "path": "nanos3reader › Features › Sub", "line": 6, "end_line": 6},
]
README = ["# nanos3reader", "", "## Why", "", "## Features", "- one", "### Sub", "x"]


def test_heading_anchor_names_the_passages_of_that_section_only():
    assert resolve_anchor(PASSAGES, README, "features") == ("heading", ["p2"])
    assert resolve_anchor(PASSAGES, README, "why") == ("heading", ["p1"])


def test_file_without_anchor_names_every_passage():
    assert resolve_anchor(PASSAGES, README, None) == ("file", ["p1", "p2", "p3"])


CODE = [
    {"id": "c1", "path": "s3_reader.cpp", "line": 1, "end_line": 3},
    {"id": "c2", "path": "s3_reader.cpp", "line": 4, "end_line": 6},
    {"id": "c3", "path": "s3_reader.cpp", "line": 7, "end_line": 8},
]
SRC = [
    "class S3Streambuf : public std::streambuf {",   # 1  declaration of the class
    "    int underflow() override {",                # 2
    "        if (resolve(fresh, diag)) {",           # 3  a call, inside an if
    "bool CredentialProvider::resolve(Credentials& out) {",  # 4 definition, qualified
    "    return resolve(out);",                      # 5  a call
    "    std::unique_ptr<X> open(const std::string& uri);",  # 6  a prototype
    "    buf->open(uri);",                           # 7  a mention
    "}",                                             # 8
]


def test_code_anchor_finds_the_definition_not_the_calls():
    assert resolve_anchor(CODE, SRC, "S3Streambuf") == ("definition", ["c1"])


def test_qualified_anchor_is_searched_whole_when_the_file_has_it():
    assert resolve_anchor(CODE, SRC, "CredentialProvider::resolve") == ("definition", ["c2"])


def test_qualified_anchor_falls_back_to_its_last_part():
    lines = ["int resolve(int x) {", "  return x;", "}"]
    passages = [{"id": "x", "path": "f.cpp", "line": 1, "end_line": 3}]
    assert resolve_anchor(passages, lines, "Provider::resolve") == ("definition", ["x"])


def test_prototype_is_a_mention_and_a_call_is_a_mention():
    assert resolve_anchor(CODE, SRC, "open") == ("mention", ["c2", "c3"])


def test_unknown_anchor_falls_back_to_the_whole_document_at_a_lower_score():
    mode, ids = resolve_anchor(CODE, SRC, "no_such_symbol")
    assert (mode, ids) == ("anchor-unmatched", ["c1", "c2", "c3"])


def test_definition_needs_a_return_type_or_class_qualifier_before_the_name():
    from docmem.align import _is_definition

    assert _is_definition("std::string build_authorization(const CanonicalRequest& req,", "build_authorization")
    assert _is_definition("bool CredentialProvider::resolve(Credentials& out) {", "resolve")
    assert not _is_definition("const std::string authz = s3v4::build_authorization(req, cred.access_key,",
                              "build_authorization")  # a call in an expression
    assert not _is_definition("    // Human-readable description of the most recent open() failure", "open")
    assert not _is_definition("        if (resolve(fresh, diag)) {", "resolve")
    assert not _is_definition("    std::unique_ptr<X> open(const std::string& uri);", "open")  # a prototype
