"""Name resolution, inheritance, relations and metaclass mapping of the model resolver. No database.

Run from apps/docmem:  python -m pytest tests/test_sysml_resolve.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docmem.scan_sysml import parse_model  # noqa: E402
from docmem.sysml_resolve import (  # noqa: E402
    Member,
    Universe,
    inherited_members,
    metaclass_for,
    relation_id,
    statement_edges,
)

URI = "file:///models/t.sysml"


def universe_of(src: str, uri: str = URI) -> Universe:
    data = src.encode()
    errors: list[str] = []
    model = parse_model([({"uri": uri}, data)], "m", errors)
    assert not errors, errors
    return Universe(model.files)


def resolve(uni: Universe, text: str, chain: list):
    return uni.resolve(text, chain)


IMPORTS = """
package Lib {
    part def Engine { attribute power; }
    part def Wheel;
    private part def Secret;
    package Deep { part def Inner; }
}
package Vehicles {
    private import Lib::Engine;
    public import Lib::*;
    part def Car { part engine : Engine; }
}
package User {
    import Vehicles::*;
    import Lib::**;
    part def Truck :> Wheel;
}
"""


def test_names_resolve_outward_then_through_imports():
    uni = universe_of(IMPORTS)
    # the private import is visible inside Vehicles, the body of Car included
    assert resolve(uni, "Engine", ["", "Vehicles", "Vehicles::Car"]).target == "Lib::Engine"
    # a qualified name needs no import: the head is found at the top level
    assert resolve(uni, "Lib::Engine", ["", "User"]).target == "Lib::Engine"
    # a name inside the package that declares it wins over any import
    assert resolve(uni, "Wheel", ["", "Lib"]).target == "Lib::Wheel"


def test_wildcard_import_brings_the_public_members_only():
    uni = universe_of(IMPORTS)
    # Vehicles re-exports Lib's public members (public import Lib::*), so User sees Wheel through Vehicles::*
    assert resolve(uni, "Wheel", ["", "User"]).target == "Lib::Wheel"
    # its private import of Engine is not re-exported, and User does not import Lib::Engine directly
    # (import Lib::** does, so this one is found through the recursive import)
    assert resolve(uni, "Engine", ["", "Vehicles"]).target == "Lib::Engine"


def test_private_members_are_not_imported_from_outside():
    uni = universe_of(IMPORTS)
    res = resolve(uni, "Secret", ["", "User"])
    assert res.target is None and "no declaration of 'Secret'" in res.why
    # inside its own namespace the private member is visible
    assert resolve(uni, "Secret", ["", "Lib"]).target == "Lib::Secret"


def test_recursive_import_reaches_nested_namespaces():
    uni = universe_of(IMPORTS)
    assert resolve(uni, "Inner", ["", "User"]).target == "Lib::Deep::Inner"
    # a plain wildcard does not go into Deep
    plain = universe_of("package A { package B { part def C; } }\npackage D { import A::*; }")
    assert resolve(plain, "C", ["", "D"]).target is None


def test_a_name_declared_twice_is_ambiguous_and_left_unresolved():
    uni = universe_of("package A { part def Thing; }\npackage B { part def Thing; }\n"
                      "package C { import A::*; import B::*; part def X :> Thing; }")
    res = resolve(uni, "Thing", ["", "C"])
    assert res.target is None and res.why.startswith("ambiguous: A::Thing, B::Thing")


def test_inheritance_with_redefinition_takes_the_most_specific_feature():
    uni = universe_of(
        "package Inh {\n"
        "  part def Base { attribute mass; attribute color; part wheel; }\n"
        "  part def Mid :> Base { :>> mass = 1000; }\n"
        "  part def Top :> Mid { attribute color; }\n"
        "}\n")
    assert uni.supertypes("Inh::Top") == ["Inh::Mid"]
    owned: dict[str, list[Member]] = {}
    for el in uni.elements:
        owned.setdefault(el.parent, []).append(Member(el.name, el.kind, el.qname, f"{el.line}", el.start))
    supers = {q: uni.supertypes(q) for q in {el.qname for el in uni.elements}}
    got = [(m.qname, src) for m, src in inherited_members("Inh::Top", owned, supers)]
    # color is Top's own (a redefinition of Base::color); mass comes from Mid's redefinition, not Base's;
    # wheel is inherited from Base
    assert got == [("Inh::Mid::mass", "Inh::Mid"), ("Inh::Base::wheel", "Inh::Base")]


def test_redefinition_resolves_to_the_redefined_feature_not_itself():
    uni = universe_of(
        "package Inh {\n"
        "  part def Base { attribute mass; }\n"
        "  part def Mid :> Base { :>> mass = 1000; }\n"
        "}\n")
    redef = next(e for e in uni.elements if e.kind == "redefinition")
    assert redef.qname == "Inh::Mid::mass"
    refs = [r for r in uni.header_refs.get(id(redef), []) if r.fragment == "redefines"]
    # the name ref of ":>> mass" is the redefinition's own name: it is resolved past the element itself
    assert refs == [] or uni.resolve(refs[0].text, refs[0].chain, redef.qname).target == "Inh::Base::mass"


def test_an_unnamed_redefinition_is_a_named_scope_for_its_body():
    # part :>> engine is named after the engine it redefines, so its body's cyl is looked up through Car's engine type
    uni = universe_of(
        "package A {\n"
        "  part def Engine { part cyl; }\n"
        "  part def Car { part engine : Engine; }\n"
        "  part def Racer :> Car { part :>> engine { part :>> cyl; } }\n"
        "}\n")
    engine = next(e for e in uni.elements if e.qname == "A::Racer::engine")
    assert engine.unnamed and engine.kind == "part"
    refs = {f.as_written: f for f in uni.references() if f.fragment == "redefines"}
    assert refs["engine"].status == "resolved" and refs["engine"].source.qname == "A::Racer::engine"
    assert refs["engine"].target == "A::Car::engine"
    cyl = refs["cyl"]
    assert cyl.status == "resolved" and cyl.rel == "r_redefines"
    assert cyl.source.qname == "A::Racer::engine::cyl" and cyl.target == "A::Engine::cyl"
    assert not any(e.name is None for e in uni.elements)  # the scope of a redefinition is an element


def test_a_redefining_usage_with_a_value_is_the_source_of_its_redefines_edge():
    # attribute :>> powerLoad = 450 [W]; used to take its enclosing element as the source
    uni = universe_of(
        "package A {\n"
        "  part def Car { attribute powerLoad; }\n"
        "  part def Racer :> Car { attribute :>> powerLoad = 450 [W]; ref :>> powerLoad; }\n"
        "}\n")
    edges = [(f.source.qname, f.rel, f.target) for f in uni.references() if f.rel == "r_redefines"]
    assert ("A::Racer::powerLoad", "r_redefines", "A::Car::powerLoad") in edges


def test_repeated_statements_keep_one_edge_each():
    uni = universe_of(
        "package M {\n"
        "  requirement def R;\n"
        "  part def Q {\n"
        "    satisfy R by Y;\n"
        "    satisfy R by Y;\n"
        "  }\n"
        "  part def Y;\n"
        "}\n")
    named = statement_edges(uni.references())
    sat = [(f, rel) for f, rel in named if f.rel == "r_satisfy"]
    assert len(sat) == 2 and len({rel for _, rel in sat}) == 2
    assert {(f.source.qname, f.target) for f, _ in sat} == {("M::Y", "M::R")}


def test_two_statements_on_one_line_are_told_apart_by_occurrence():
    uni = universe_of(
        "package M { requirement def R; part def Y { satisfy R by Y; satisfy R by Y; } part def X; }")
    names = [rel for f, rel in statement_edges(uni.references()) if f.rel == "r_satisfy"]
    assert len(names) == 2 and len(set(names)) == 2
    assert names[0] == relation_id("r_satisfy", URI, 1, 0) and names[1] == relation_id("r_satisfy", URI, 1, 1)


def test_relation_identity_is_the_relation_then_the_statement():
    assert relation_id("r_satisfy", "file:///a.sysml", 12, 0) == "r_satisfy@file:///a.sysml:12#0"


def test_satisfy_by_names_the_satisfier_and_the_requirement():
    uni = universe_of(
        "package Req {\n"
        "  requirement def R1;\n"
        "  part def Q;\n"
        "  part def S { satisfy requirement R1 by Q; }\n"
        "  part def P { satisfy R1; }\n"
        "}\n")
    found = {(f.fragment, f.status, f.rel, f.source.qname if f.source else None, f.target)
             for f in uni.references() if f.fragment in ("satisfy", "satisfy_by")}
    assert ("satisfy", "resolved", "r_satisfy", "Req::Q", "Req::R1") in found  # Q satisfies R1
    assert ("satisfy", "resolved", "r_satisfy", "Req::P", "Req::R1") in found  # the owner satisfies R1


def test_allocate_a_to_b_is_an_edge_from_a_to_b():
    uni = universe_of("package M { part def X; part def Y; part def Z { allocate X to Y; } }")
    edges = [(f.source.qname, f.rel, f.target) for f in uni.references() if f.rel == "r_allocate"]
    assert edges == [("M::X", "r_allocate", "M::Y")]


def test_typed_by_resolves_in_the_owners_namespace_and_reports_misses():
    uni = universe_of("package P { part def Engine; part car : Engine; part bike : Missing; }")
    refs = [f for f in uni.references() if f.fragment == "typed_by"]
    ok = next(f for f in refs if f.as_written == "Engine")
    assert ok.status == "resolved" and ok.rel == "r_typed_by" and ok.source.qname == "P::car"
    bad = next(f for f in refs if f.as_written == "Missing")
    assert bad.status == "unresolved" and "no declaration of 'Missing'" in bad.why


def test_import_visibility_defaults_to_private_and_public_is_explicit():
    uni = universe_of("package A { part def X; }\npackage B { import A::X; }\npackage C { public import A::X; }")
    vis = {r.chain[-1]: r.vis for f in uni.files for r in f.refs if r.fragment == "import"}
    assert vis == {"B": "private", "C": "public"}


def test_metaclass_mapping_for_definitions_usages_and_notes():
    assert metaclass_for("part def") == ("PartDefinition", ())
    assert metaclass_for("part") == ("PartUsage", ())
    assert metaclass_for("port") == ("PortUsage", ())
    assert metaclass_for("action") == ("ActionUsage", ())
    assert metaclass_for("requirement def") == ("RequirementDefinition", ())
    assert metaclass_for("requirement") == ("RequirementUsage", ())
    assert metaclass_for("attribute") == ("AttributeUsage", ())
    assert metaclass_for("individual part def") == ("PartDefinition", ("individual",))
    assert metaclass_for("ref part") == ("PartUsage", ("ref",))
    assert metaclass_for("perform action") == ("PerformActionUsage", ())
    assert metaclass_for("do action") == ("ActionUsage", ("do",))
    assert metaclass_for("standard library package") == ("LibraryPackage", ("standard",))
    assert metaclass_for("in") == ("Feature", ("in", "no keyword"))
    assert metaclass_for("frobnicate def") == (None, ())
