"""Nanolance vs Ladybug parity for the provenance, tag, learned-value, metadata and timestamp methods.

Each scenario runs the same writes and reads on a fresh nanolance adapter and a fresh Ladybug
adapter, and the normalized outputs must be equal. Ids are deterministic (uuid5), so both
runs see identical inputs. Dataset and run id lists are compared sorted (Ladybug's folded
writes append in insertion order, the shared transition sorts); ref-key lists keep their order.
"""

from types import SimpleNamespace
from uuid import NAMESPACE_OID, uuid5

import pytest

pytest.importorskip("ladybug")

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.infrastructure.databases.provenance import (
    EdgeIdentity,
    make_source_ref_key,
)
from cognee.infrastructure.databases.provenance.markers import (
    mark_graph_provenance_if_empty,
    stores_provenance_in_graph,
)
from cognee_nanograph import NanolanceGraphAdapter

DS_A = uuid5(NAMESPACE_OID, "dataset-a")
DS_B = uuid5(NAMESPACE_OID, "dataset-b")
DATA_1 = uuid5(NAMESPACE_OID, "data-1")
DATA_2 = uuid5(NAMESPACE_OID, "data-2")
DATA_3 = uuid5(NAMESPACE_OID, "data-3")
RUN_1 = str(uuid5(NAMESPACE_OID, "run-1"))
RUN_2 = str(uuid5(NAMESPACE_OID, "run-2"))
RUN_3 = str(uuid5(NAMESPACE_OID, "run-3"))
REF_A1 = make_source_ref_key(DS_A, DATA_1)
REF_A2 = make_source_ref_key(DS_A, DATA_2)
REF_B3 = make_source_ref_key(DS_B, DATA_3)


def _node(node_id, name=None, typ="Node", props=None):
    data = {"id": node_id, "name": name or node_id, "type": typ, **(props or {})}
    return SimpleNamespace(model_dump=lambda: dict(data))


def _edge_name(edge: EdgeIdentity) -> str:
    return f"{edge.source_id}|{edge.relationship_name}|{edge.target_id}"


def _prov(data) -> dict:
    return {
        "keys": data.source_ref_keys,
        "datasets": sorted(data.source_dataset_ids),
        "runs": sorted(data.source_run_ids),
        "run_refs": data.source_run_refs,
    }


def _node_delete(result: dict) -> dict:
    return {
        nid: {
            "type": d.node_type,
            "indexed": d.indexed_fields,
            "props": d.node_properties,
            **_prov(d),
        }
        for nid, d in result.items()
    }


def _edge_delete(result: dict) -> dict:
    return {
        _edge_name(e): {"text": d.edge_text, "props": d.edge_properties, **_prov(d)}
        for e, d in result.items()
    }


def _edge_keyed(result: dict) -> dict:
    return {_edge_name(e): value for e, value in result.items()}


def _edge_list(edges) -> list[str]:
    return sorted(_edge_name(e) for e in edges)


async def scenario_attach_find_remove(g) -> dict:
    await g.add_nodes([_node("n1"), _node("n2")])
    await g.add_edges([("n1", "n2", "r", {})])
    edge = EdgeIdentity("n1", "n2", "r")
    await g.attach_node_source_refs(["n1", "n2"], [REF_A1], pipeline_run_id=RUN_1)
    await g.attach_node_source_refs(["n1"], [REF_B3], pipeline_run_id=RUN_2)
    await g.attach_edge_source_refs([edge], [REF_A1, REF_B3], pipeline_run_id=RUN_1)
    out = {
        "find_nodes_a1": sorted(await g.find_nodes_by_source_ref(REF_A1)),
        "find_nodes_b3": sorted(await g.find_nodes_by_source_ref(REF_B3)),
        "find_edges_a1": _edge_list(await g.find_edges_by_source_ref(REF_A1)),
        "dataset_nodes_a": await g.find_node_source_refs_by_dataset(str(DS_A)),
        "dataset_edges_b": _edge_keyed(await g.find_edge_source_refs_by_dataset(str(DS_B))),
        "run1_nodes": await g.find_node_source_refs_by_pipeline_run(RUN_1),
        "run1_edges": _edge_keyed(await g.find_edge_source_refs_by_pipeline_run(RUN_1)),
        "run2_nodes": await g.find_node_source_refs_by_pipeline_run(RUN_2),
        "n1_delete": _node_delete(await g.get_node_delete_data(["n1"])),
        "edge_delete": _edge_delete(await g.get_edge_delete_data([edge])),
    }
    await g.remove_node_source_refs(["n1"], [REF_B3])
    await g.remove_edge_source_refs([edge], [REF_A1])
    out["after_remove_n1"] = _node_delete(await g.get_node_delete_data(["n1"]))
    out["after_remove_edge"] = _edge_delete(await g.get_edge_delete_data([edge]))
    out["find_nodes_b3_after"] = sorted(await g.find_nodes_by_source_ref(REF_B3))
    out["missing_node"] = _node_delete(await g.get_node_delete_data(["nope"]))
    return out


async def scenario_fold_stamps_on_write(g) -> dict:
    await g.add_nodes(
        [_node("n1"), _node("n2"), _node("n3")],
        source_ref_key=REF_A1,
        pipeline_run_id=RUN_1,
    )
    # A second dataset joins n1; the already-owned A1 on the third write adds no run ref.
    await g.add_nodes([_node("n1")], source_ref_key=REF_B3, pipeline_run_id=RUN_2)
    await g.add_nodes([_node("n1")], source_ref_key=REF_A1, pipeline_run_id=RUN_3)
    # Dict form: each node carries its own ref.
    await g.add_nodes(
        [_node("n4", typ="Entity")],
        source_ref_key={"n4": REF_A2},
        pipeline_run_id=RUN_2,
    )
    # A plain re-add keeps the stored provenance and updates the name.
    await g.add_nodes([_node("n1", name="renamed")])
    await g.add_edges(
        [("n1", "n2", "r", {}), ("n2", "n3", "r", {})],
        source_ref_key=REF_A1,
        pipeline_run_id=RUN_1,
    )
    await g.add_edges(
        [("n2", "n4", "s", {})],
        source_ref_key={("n2", "n4", "s"): REF_B3},
    )
    nodes = _node_delete(await g.get_node_delete_data(["n1", "n2", "n3", "n4"]))
    edges = _edge_delete(
        await g.get_edge_delete_data(
            [
                EdgeIdentity("n1", "n2", "r"),
                EdgeIdentity("n2", "n3", "r"),
                EdgeIdentity("n2", "n4", "s"),
            ]
        )
    )
    return {"nodes": nodes, "edges": edges}


async def scenario_fold_dict_missing_entry_raises(g) -> dict:
    await g.add_nodes([_node("n1"), _node("n2")])
    with pytest.raises(KeyError):
        await g.add_nodes([_node("n1"), _node("n2")], source_ref_key={"n1": REF_A1})
    return {"raised": True}


async def scenario_delete_edge_triples(g) -> dict:
    await g.add_nodes([_node("n1"), _node("n2")])
    await g.add_edges([("n1", "n2", "r", {}), ("n1", "n2", "s", {})])
    await g.delete_edge_triples([EdgeIdentity("n1", "n2", "r")])
    return {
        "has": await g.has_edges([("n1", "n2", "r"), ("n1", "n2", "s")]),
        "nodes": sorted(n["id"] for n in await g.get_nodes(["n1", "n2"])),
        "edges_of_n1": sorted(rel for _, rel, _ in await g.get_edges("n1")),
    }


async def scenario_belongs_to_set_tags(g) -> dict:
    await g.add_nodes(
        [
            _node("n1", props={"belongs_to_set": ["x", "y"]}),
            _node("n2", props={"belongs_to_set": ["x"]}),
            _node("n3", props={"belongs_to_set": ["z"]}),
            _node("n4"),
        ]
    )

    def tags(nodes):
        return {n["id"]: n.get("belongs_to_set") for n in nodes}

    await g.remove_belongs_to_set_tags(["x"], node_ids=["n1"])
    scoped = tags(await g.get_nodes(["n1", "n2", "n3", "n4"]))
    await g.remove_belongs_to_set_tags(["y", "z"])
    return {"scoped": scoped, "all": tags(await g.get_nodes(["n1", "n2", "n3", "n4"]))}


async def scenario_feedback_weights(g) -> dict:
    await g.add_nodes([_node("n1"), _node("n2"), _node("n3")])
    await g.add_edges(
        [
            ("n1", "n2", "r", {"edge_object_id": "eo-1"}),
            ("n2", "n3", "s", {"edge_object_id": "eo-2"}),
        ]
    )
    out = {
        "set_nodes": await g.set_node_feedback_weights({"n1": 0.9, "n2": 0.2, "ghost": 0.4}),
        "get_nodes": await g.get_node_feedback_weights(["n1", "n2", "n3", "ghost"]),
        "set_edges": await g.set_edge_feedback_weights({"eo-1": 0.7, "missing": 0.1}),
        "get_edges": await g.get_edge_feedback_weights(["eo-1", "eo-2", "missing"]),
    }
    # Learned values survive a re-add that carries no feedback of its own.
    await g.add_nodes([_node("n1", name="again")])
    await g.add_edges([("n1", "n2", "r", {"edge_object_id": "eo-1"})])
    out["get_nodes_after_readd"] = await g.get_node_feedback_weights(["n1"])
    out["get_edges_after_readd"] = await g.get_edge_feedback_weights(["eo-1"])
    return out


async def scenario_timestamps_in_range(g) -> dict:
    stamps = {
        "t1": (1000, 2000, "1969"),
        "t2": (1500, None, "mid"),  # no time_until: defaults to time_at + 1000
        "t3": (3000, 4000, "later"),
    }
    nodes = [
        _node(name, name, "Timestamp", {"time_at": at, "time_until": until, "timestamp_str": s})
        if until is not None
        else _node(name, name, "Timestamp", {"time_at": at, "timestamp_str": s})
        for name, (at, until, s) in stamps.items()
    ]
    nodes.append(_node("t4", "t4", "Timestamp", {"timestamp_str": "no time"}))
    nodes.append(_node("p1", "p1", "Person", {"time_at": 1200, "time_until": 1300}))
    await g.add_nodes(nodes)
    windows = [
        (None, None),
        (1600, 1700),
        (None, 1500),
        (2000, None),
        (2000, 2000),
        (4000, None),
        (0, 1000),
    ]
    return {str(w): await g.get_timestamps_in_range(*w) for w in windows}


async def scenario_metadata_and_provenance_marker(g) -> dict:
    out = {"empty": await g.get_graph_metadata()}
    # A non-empty graph is never marked (it stays on the relational ledger).
    await g.add_nodes([_node("n1")])
    out["mark_nonempty"] = await mark_graph_provenance_if_empty(g)
    out["marked_nonempty"] = await stores_provenance_in_graph(g)
    await g.delete_graph()
    out["mark_empty"] = await mark_graph_provenance_if_empty(g)
    out["marked"] = await stores_provenance_in_graph(g)
    await g.set_graph_metadata({"k": "1"})
    await g.set_graph_metadata({"k": "2", "other": "x"})
    out["overwritten"] = await g.get_graph_metadata()
    await g.delete_graph()
    out["after_delete"] = await g.get_graph_metadata()
    return out


SCENARIOS = [
    scenario_attach_find_remove,
    scenario_fold_stamps_on_write,
    scenario_fold_dict_missing_entry_raises,
    scenario_delete_edge_triples,
    scenario_belongs_to_set_tags,
    scenario_feedback_weights,
    scenario_timestamps_in_range,
    scenario_metadata_and_provenance_marker,
]


async def _open(kind: str, root):
    if kind == "nanolance":
        adapter = NanolanceGraphAdapter(graph_database_url=str(root / "graph"))
        await adapter.initialize()
        return adapter
    return LadybugAdapter(str(root / "graph_db"))


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.__name__)
async def test_nanolance_matches_ladybug(scenario, tmp_path):
    results = {}
    for kind in ("nanolance", "ladybug"):
        root = tmp_path / kind
        root.mkdir()
        adapter = await _open(kind, root)
        try:
            results[kind] = await scenario(adapter)
        finally:
            if kind == "ladybug":
                await adapter.close()
    assert results["nanolance"] == results["ladybug"]
