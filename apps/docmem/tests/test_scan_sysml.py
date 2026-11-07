"""SysML v2 scanner: tokenizer edge cases, elements, passages, each relation kind, resolution and the stream.

Run from apps/docmem:  python -m pytest tests/test_scan_sysml.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from docmem.scan_sysml import ScanError, main, parse_model, tokenize  # noqa: E402

URI = "file:///models/t.sysml"


def model_of(src: bytes, name: str = "m"):
    doc = {"uri": URI, "sha256": "0" * 64, "size": len(src), "mtime": 0.0, "title": "t"}
    return parse_model([(doc, src)], name)


def records_of(src: bytes, name: str = "m"):
    return list(model_of(src, name).records())


def of_type(recs, kind):
    return [r for r in recs if r["type"] == kind]


def text_at(src: bytes, rec: dict) -> str:
    return src[rec["offset"]: rec["offset"] + rec["length"]].decode("utf-8")


def links(src: bytes, name: str = "m"):
    """(anchor, target, fragment) of each link, and every anchor must be the bytes its offset and length point at."""
    out = []
    for r in of_type(records_of(src, name), "link"):
        assert text_at(src, r) == r["anchor"]
        out.append((r["anchor"], r["target"], r["fragment"]))
    return out


def link_frags(src: bytes):
    return {(a, f) for a, _, f in links(src)}


# -- tokenizer ------------------------------------------------------------------------------------------------

def kinds(src: bytes):
    return [(t.kind, t.text) for t in tokenize(src, "t")]


def test_tokenizer_comments_strings_and_names():
    toks = kinds(b"part 'a b' /* x { ; } */ \"s;{\" // /* not a comment\n :>> :> ::> :: 1.5E-3 ;")
    assert toks == [
        ("id", "part"), ("q", "'a b'"), ("cmt", "/* x { ; } */"), ("str", '"s;{"'),
        ("p", ":>>"), ("p", ":>"), ("p", "::>"), ("p", "::"), ("num", "1.5E-3"), ("p", ";"),
    ]


def test_unterminated_string_and_comment_are_errors():
    with pytest.raises(ScanError, match="unterminated string"):
        tokenize(b'attribute x = "open;\n', "t")
    with pytest.raises(ScanError, match="unterminated comment"):
        tokenize(b"part x { doc /* open }", "t")


def test_nested_braces_and_brace_in_comment_do_not_end_a_body():
    src = b"""package P {
  part def A {
    doc /* a } brace in a comment */
    part b { part c { } }
    attribute s = "x;}y";
  }
  part after : A;
}
"""
    names = {r["path"][-1] for r in of_type(records_of(src), "section")}
    assert {"part def A", "part b", "part c", "attribute s", "part after"} <= names
    a = next(r for r in of_type(records_of(src), "section") if r["path"][-1] == "part def A")
    assert text_at(src, a).endswith("\n  }")  # the brace that closes A, not the one in the comment or the string


# -- elements and sections -------------------------------------------------------------------------------------

SMALL = b"""package P {
\tpart def A { doc /* the a */ }
\tpart a : A;
}
"""


def test_sections_form_a_tree_under_the_model_root():
    secs = of_type(records_of(SMALL, "apollo11"), "section")
    assert [s["path"] for s in secs] == [
        ["model", "apollo11", "package P"],
        ["model", "apollo11", "package P", "part def A"],
        ["model", "apollo11", "package P", "part a"],
    ]
    pkg, a, part_a = secs
    assert text_at(SMALL, pkg).startswith("package P {") and text_at(SMALL, pkg).endswith("}")
    assert text_at(SMALL, a).startswith("part def A {") and text_at(SMALL, a).endswith("}")
    assert text_at(SMALL, part_a) == "part a : A;"
    assert part_a["line"] == 3


def test_element_kinds_include_modifiers_and_generic_usages():
    src = b"""package P {
  abstract part def B;
  individual part def 'SA-1' :> B;
  in item fuel : Fuel;
  end capa ::> goal;
  :>> unitConversion: Conv { :>> prefix = kilo; }
  calc def c { in sys : Part; }
}
"""
    labels = [s["path"][-1] for s in of_type(records_of(src), "section")]
    assert "abstract part def B" in labels
    assert "individual part def 'SA-1'" in labels
    assert "in item fuel" in labels
    assert "end capa" in labels
    assert "redefinition unitConversion" in labels
    assert "in sys" in labels


def test_short_name_and_quoted_name_are_declared():
    src = b"""package P {
  requirement def <'R-1'> Req1 { doc /* r */ }
  requirement 'r-2' : Req1;
}
"""
    labels = [s["path"][-1] for s in of_type(records_of(src), "section")]
    assert "requirement def Req1" in labels
    assert "requirement 'r-2'" in labels


def test_unclosed_body_is_an_error():
    with pytest.raises(ScanError, match="unclosed"):
        records_of(b"package P { part a : A;\n")


# -- passages --------------------------------------------------------------------------------------------------

def test_header_and_doc_form_one_passage_and_rationale_is_its_own():
    src = b"""package P {
  requirement def R {
    doc /* The system shall fly. */
    @Rationale {
      text = "Because it must; really.";
    }
    require constraint { speed <= 10 }
  }
}
"""
    pas = [r for r in of_type(records_of(src), "passage") if r["path"][-1] == "requirement def R"]
    texts = [r["text"] for r in pas]
    assert texts[0] == "requirement def R {\n    doc /* The system shall fly. */"
    assert texts[1] == 'text = "Because it must; really.";'
    assert texts[2] == "require constraint { speed <= 10 }"
    for r in pas:
        assert text_at(src, r) == r["text"]
        assert r["sha256"] and r["end_line"] >= r["line"]


def test_bare_attribute_has_no_passage_but_attribute_with_text_does():
    src = b"""package P {
  attribute bare :> ISQ::mass = 5 [kg];
  attribute label: String = "Burning; hot";
  part p;
}
"""
    passages = of_type(records_of(src), "passage")
    paths = [r["path"][-1] for r in passages]
    assert "attribute bare" not in paths
    label = next(r for r in passages if r["path"][-1] == "attribute label")
    assert label["text"] == 'attribute label: String = "Burning; hot"'
    assert "part p" in paths  # a bare part keeps its header passage


def test_line_comments_and_file_level_comments_are_not_passages():
    src = b"// a header comment with /* no end\npackage P {\n  // ignored\n  part a;\n}\n"
    assert [r["path"][-1] for r in of_type(records_of(src), "passage")] == ["package P", "part a"]


# -- relationships -------------------------------------------------------------------------------------------

def test_specialization_redefinition_subsetting_and_typing():
    src = b"""package P {
  part def Vehicle :> Base, Other { }
  part def Base { }
  part def Other { }
  part v : Vehicle :>> mass;
  part w subsets v;
  attribute :>> failure = 5E-6 [1/h];
}
"""
    got = link_frags(src)
    assert ("Base", "specializes") in got
    assert ("Other", "specializes") in got
    assert ("Vehicle", "typed_by") in got
    assert ("mass", "redefines") in got
    assert ("v", "subsets") in got
    assert ("failure", "redefines") in got


def test_specialization_after_a_body_belongs_to_the_element():
    src = b"package P {\n  part x { doc /* d */ } :> missions;\n}\n"
    lk = links(src)
    assert ("missions", "missions", "specializes") in lk


def test_satisfy_names_the_requirement_and_the_satisfier():
    src = b"""package P {
  requirement def <'R-1'> Req1 { }
  part v;
  satisfy 'R-1' by v;
  satisfy requirement Req1 by P::v;
}
"""
    got = links(src)
    assert ("'R-1'", "P::Req1", "satisfy") in got
    assert ("v", "P::v", "satisfy_by") in got
    assert ("Req1", "P::Req1", "satisfy") in got
    assert ("P::v", "P::v", "satisfy_by") in got


def test_allocate_connect_bind_and_flow_name_both_ends():
    src = b"""package P {
  part a; part b;
  allocate a to b;
  connect a to b;
  bind a = b;
  flow of Fuel from a to b;
  interface cm : Iface connect a.x to b.y;
}
"""
    got = link_frags(src)
    assert ("a", "allocate") in got and ("b", "allocate_to") in got
    assert ("a", "connect") in got and ("b", "connect_to") in got
    assert ("a", "bind") in got and ("b", "bind_to") in got
    assert ("Fuel", "typed_by") in got
    assert ("a", "flow") in got and ("b", "flow_to") in got
    assert ("Iface", "typed_by") in got
    assert ("a.x", "connect") in got and ("b.y", "connect_to") in got


def test_perform_exhibit_and_named_usages_with_a_type():
    src = b"""package P {
  action def Act { }
  part def Eng { perform action run : Act; }
  state def S { }
  part host { exhibit state st : S; }
  perform Act;
}
"""
    got = link_frags(src)
    assert ("Act", "typed_by") in got
    assert ("S", "typed_by") in got
    assert ("Act", "perform") in got


def test_import_and_metadata_reference():
    src = b"""package P {
  private import SI::*;
  part a;
  part b;
  #refinement dependency a to P::b;
}
"""
    got = link_frags(src)
    assert ("SI::*", "import") in got
    assert ("a", "refinement") in got
    assert ("P::b", "refinement_to") in got


# -- resolution ----------------------------------------------------------------------------------------------

def test_names_resolve_in_the_enclosing_namespace_when_unique():
    src = b"""package P {
  part def Engine { }
  part def Vehicle {
    part def Inner { }
    part e : Engine;
    part i : Inner;
  }
}
package Q {
  part def Engine { }
  part x : Engine;
}
"""
    targets = [t for a, t, f in links(src) if f == "typed_by"]
    assert "P::Engine" in targets  # from inside Vehicle, P's Engine
    assert "Q::Engine" in targets  # from inside Q, its own Engine
    assert "P::Vehicle::Inner" in targets


def test_ambiguous_names_stay_as_written():
    src = b"""package P {
  part def Dup { }
  part def Dup { }
  part y : Dup;
}
"""
    assert ("Dup", "Dup", "typed_by") in links(src)


def test_unknown_names_stay_as_written():
    src = b"package P {\n  part z : Imported::Thing;\n}\n"
    assert ("Imported::Thing", "Imported::Thing", "typed_by") in links(src)


# -- the stream ----------------------------------------------------------------------------------------------

def test_doc_record_and_end_counts_match_the_records():
    recs = records_of(SMALL, "apollo11")
    doc = recs[0]
    assert doc["type"] == "doc"
    assert doc["extractor"] == "" and doc["format"] == "sysml" and doc["model"] == "apollo11"
    end = recs[-1]
    assert end["type"] == "end"
    assert end["sections"] == len(of_type(recs, "section"))
    assert end["passages"] == len(of_type(recs, "passage"))
    assert end["links"] == len(of_type(recs, "link"))


def test_every_passage_and_link_reads_back_from_the_file():
    src = b"""package P {
  requirement def <'R-1'> Req1 {
    doc /* The system shall work. */
    subject s : Thing;
    satisfy 'R-1' by v;
  }
  part v : Thing :> Base;
  attribute label: String = "x; y";
}
"""
    recs = records_of(src)
    for r in of_type(recs, "passage"):
        assert text_at(src, r) == r["text"]
    for r in of_type(recs, "section"):
        assert text_at(src, r).split()[0] in ("package", "requirement", "part", "attribute", "subject")
    links(src)  # asserts each anchor


def test_command_line_writes_one_stream_and_summary_to_stderr(tmp_path, capsys):
    (tmp_path / "a.sysml").write_bytes(SMALL)
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "b.sysml").write_bytes(SMALL)
    main([str(tmp_path), "--model-name", "apollo11"])
    out, err = capsys.readouterr()
    lines = [json.loads(line) for line in out.splitlines()]
    assert lines[0] == {"type": "scan_start", "protocol": lines[0]["protocol"]}
    assert lines[-1]["type"] == "scan_end"
    docs = [r for r in lines if r["type"] == "doc"]
    assert len(docs) == 1 and docs[0]["uri"].endswith("/a.sysml")
    assert json.loads(err)["files"] == 1



# -- unnamed redefinitions: named after the feature they redefine ------------------------------------------

REDEFINITIONS = b"""package P {
  part def Car { part crew; attribute powerLoad; ref y; }
  part def Racer :> Car {
    part :>> crew { part spaceSuits; }
    attribute :>> powerLoad = 450 [W];
    ref :>> y;
    :>> crew;
  }
}
"""


def test_unnamed_redefinitions_are_sections_named_after_the_redefined_feature():
    recs = records_of(REDEFINITIONS)
    secs = {r["path"][-1]: r for r in of_type(recs, "section")}
    for label in ("part crew", "attribute powerLoad", "ref redefinition y", "redefinition crew"):
        assert secs[label]["unnamed_redefinition"] is True, label
    # Car's own features are declarations, not redefinitions
    car_crew = next(r for r in of_type(recs, "section") if r["path"][-2:] == ["part def Car", "part crew"])
    assert "unnamed_redefinition" not in car_crew


def test_a_redefinition_body_is_a_scope_for_its_children():
    paths = [r["path"] for r in of_type(records_of(REDEFINITIONS), "section")]
    assert ["model", "m", "package P", "part def Racer", "part crew", "part spaceSuits"] in paths


def test_a_redefinition_is_the_source_of_its_link_and_never_its_own_target():
    lk = links(REDEFINITIONS)
    # the scanner resolves names by declaration only: the redefined name is not found, so it stays as written
    assert ("crew", "crew", "redefines") in lk
    assert ("powerLoad", "powerLoad", "redefines") in lk
    assert ("y", "y", "redefines") in lk
    assert all(t != "P::Racer::crew" for a, t, _ in lk if a == "crew")


def test_kerml_unnamed_redefinition_is_named_after_the_feature_and_a_list_redefines_each():
    src = (b"package P {\n  feature def Q {\n    in feature monitor redefines onOccurrence {\n"
           b"      feature redefines startingAt {\n        feature redefines accessedFeature, observations;\n"
           b"      }\n    }\n  }\n}\n")
    recs = records_of(src)
    unnamed = [r["path"][-1] for r in of_type(recs, "section") if r.get("unnamed_redefinition")]
    assert unnamed == ["feature startingAt", "feature accessedFeature"]
    assert ["model", "m", "package P", "feature def Q", "in feature monitor", "feature startingAt",
            "feature accessedFeature"] in [r["path"] for r in of_type(recs, "section")]
    lk = links(src)
    assert ("onOccurrence", "onOccurrence", "redefines") in lk  # the named monitor redefines it
    assert ("accessedFeature", "accessedFeature", "redefines") in lk
    assert ("observations", "observations", "redefines") in lk


def test_a_value_after_an_unnamed_redefinition_is_not_a_redefinition():
    lk = links(b"package P { part def A { attribute :>> mass = dryMass + fuel; } }\n")
    assert ("mass", "mass", "redefines") in lk
    assert not any(a in ("dryMass", "fuel") for a, _, _ in lk)
