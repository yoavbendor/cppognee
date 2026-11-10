"""nanolance adapter behaviour that is specific to it: registration, Ladybug-shaped results,
dangling-edge skipping, index state, and methods that stay inherited for later stages."""

import asyncio
from uuid import uuid4

import pytest
import pytest_asyncio

import cognee_nanograph
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.graph.supported_databases import supported_databases
from cognee.infrastructure.databases.provenance import EdgeIdentity, make_source_ref_key
from cognee_nanograph import NanolanceGraphAdapter


@pytest_asyncio.fixture
async def graph(tmp_path):
    adapter = NanolanceGraphAdapter(graph_database_url=str(tmp_path / "graph"))
    await adapter.initialize()
    return adapter


class Person:
    """Stand-in node type: get_nodeset_subgraph takes the label from the class name."""


def test_register_maps_provider_name_to_the_adapter():
    cognee_nanograph.register()
    assert supported_databases["nanolance"] is NanolanceGraphAdapter


def test_create_graph_engine_builds_the_adapter_from_the_url(tmp_path):
    from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine

    cognee_nanograph.register()
    engine = create_graph_engine(
        graph_database_provider="nanolance",
        graph_file_path="",
        graph_database_url=str(tmp_path / "via-engine"),
    )
    assert isinstance(engine, NanolanceGraphAdapter)
    assert engine.database_path == str(tmp_path / "via-engine")


def test_missing_directory_is_an_error():
    with pytest.raises(OSError, match="GRAPH_DATABASE_URL"):
        NanolanceGraphAdapter(graph_database_url="")


def test_remote_key_is_refused(tmp_path):
    with pytest.raises(OSError, match="Remote nanolance"):
        NanolanceGraphAdapter(graph_database_url=str(tmp_path), graph_database_key="secret")


def test_no_cypher_capability():
    assert NanolanceGraphAdapter.supports_cypher_queries is False


@pytest.mark.asyncio
async def test_query_names_cypher_in_its_error(graph):
    with pytest.raises(NotImplementedError, match="does not support raw Cypher"):
        await graph.query("MATCH (n) RETURN n")


@pytest.mark.asyncio
async def test_edges_with_a_missing_endpoint_are_skipped(graph):
    await graph.add_nodes([("a", {"name": "a", "type": "Node"})])
    await graph.add_edges([("a", "ghost", "rel", {}), ("ghost", "a", "rel", {})])
    assert await graph.get_graph_counts() == (1, 0)


@pytest.mark.asyncio
async def test_edge_key_separator_in_ids_is_refused(graph):
    with pytest.raises(ValueError, match="'\\|'"):
        await graph.add_edges([("a|b", "c", "rel", {})])


@pytest.mark.asyncio
async def test_connections_and_edges_put_the_node_first(graph):
    """Ladybug's undirected match: the queried node is always the first element."""
    await graph.add_nodes([(i, {"name": i, "type": "Node"}) for i in ["a", "b"]])
    await graph.add_edge("a", "b", "rel", {"w": 2})
    (edge,) = await graph.get_edges("b")
    me, rel, other = edge
    assert (me["id"], rel, other["id"]) == ("b", "rel", "a")
    ((me, props, other),) = await graph.get_connections("b")
    assert me["id"] == "b" and other["id"] == "a"
    assert props == {"relationship_name": "rel", "w": 2}


@pytest.mark.asyncio
async def test_nodeset_subgraph_or_and_operators(graph):
    await graph.add_nodes(
        [(i, {"name": i, "type": "Person"}) for i in ["alice", "bob"]]
        + [(i, {"name": i, "type": "Node"}) for i in ["carol", "dave"]]
    )
    await graph.add_edges(
        [
            ("alice", "bob", "knows", {}),
            ("alice", "carol", "rel", {}),
            ("bob", "carol", "rel", {}),
            ("alice", "dave", "rel", {}),
        ]
    )
    nodes, _ = await graph.get_nodeset_subgraph(Person, ["alice", "bob"], "OR")
    assert {node_id for node_id, _ in nodes} == {"alice", "bob", "carol", "dave"}
    nodes, edges = await graph.get_nodeset_subgraph(Person, ["alice", "bob"], "AND")
    assert {node_id for node_id, _ in nodes} == {"alice", "bob", "carol"}
    assert {(s, t) for s, t, _, _ in edges} == {
        ("alice", "bob"),
        ("alice", "carol"),
        ("bob", "carol"),
    }
    assert await graph.get_nodeset_subgraph(Person, [], "OR") == ([], [])


@pytest.mark.asyncio
async def test_invalid_filter_attribute_is_refused(graph):
    with pytest.raises(CogneeValidationError, match="Invalid attribute filter"):
        await graph.get_filtered_graph_data([{"color": ["red"]}])


@pytest.mark.asyncio
async def test_upsert_keeps_created_at_and_overwrites_properties(graph):
    await graph.add_nodes([("a", {"name": "a", "type": "Node", "v": 1})])
    first = graph._open("nodes").to_table(columns=["created_at"], filter="id = 'a'")
    await graph.add_nodes([("a", {"name": "a", "type": "Node", "v": 2})])
    second = graph._open("nodes").to_table(columns=["created_at", "updated_at"], filter="id = 'a'")
    assert first["created_at"][0].as_py() == second["created_at"][0].as_py()
    assert (await graph.get_node("a"))["v"] == 2


@pytest.mark.asyncio
async def test_scalar_indexes_exist_and_reads_stay_correct_after_appends(graph):
    await graph.add_nodes([(f"n{i}", {"name": f"n{i}", "type": "Node"}) for i in range(20)])
    await graph.add_edges([(f"n{i}", f"n{i + 1}", "next", {}) for i in range(19)])
    await graph.add_nodes([("late", {"name": "late", "type": "Person"})])
    await graph.add_edge("late", "n0", "next")
    indexed = {
        field
        for index in graph._open("nodes").list_indices() + graph._open("edges").list_indices()
        for field in index["fields"]
    }
    assert {"id", "type", "source_id", "target_id", "edge_key", "relationship_name"} <= indexed
    assert await graph.has_edge("late", "n0", "next") is True
    assert (await graph.get_node("late"))["type"] == "Person"
    assert len((await graph.get_neighborhood(["n0"], depth=1))[1]) == 2


@pytest.mark.asyncio
async def test_metrics_report_components_and_selfloops(graph):
    await graph.add_nodes([(i, {"name": i, "type": "Node"}) for i in ["a", "b", "c", "d"]])
    await graph.add_edges([("a", "b", "r", {}), ("b", "b", "r", {})])
    metrics = await graph.get_graph_metrics(include_optional=True)
    assert metrics["num_nodes"] == 4 and metrics["num_edges"] == 2
    assert metrics["num_connected_components"] == 3
    assert metrics["sizes_of_connected_components"] == [2, 1, 1]
    assert metrics["num_selfloops"] == 1
    assert metrics["diameter"] == -1


@pytest.mark.asyncio
async def test_has_edges_dedupes_and_keeps_input_order(graph):
    await graph.add_nodes([(i, {"name": i, "type": "Node"}) for i in ["a", "b"]])
    await graph.add_edge("a", "b", "r")
    await graph.add_edge("b", "a", "r")
    result = await graph.has_edges([("b", "a", "r"), ("a", "b", "r"), ("a", "b", "r")])
    assert result == [("b", "a", "r"), ("a", "b", "r")]


@pytest.mark.asyncio
async def test_concurrent_writes_are_serialized(graph):
    await asyncio.gather(
        *[graph.add_nodes([(f"n{i}", {"name": f"n{i}", "type": "Node"})]) for i in range(25)]
    )
    assert await graph.get_graph_counts() == (25, 0)


@pytest.mark.asyncio
async def test_a_second_adapter_on_the_same_directory_sees_writes(graph):
    other = NanolanceGraphAdapter(graph_database_url=graph.database_path)
    await graph.add_nodes([("a", {"name": "a", "type": "Node"})])
    assert await other.get_node("a") is not None


@pytest.mark.asyncio
async def test_delete_graph_empties_both_tables(graph):
    await graph.add_nodes(
        [("a", {"name": "a", "type": "Node"}), ("b", {"name": "b", "type": "Node"})]
    )
    await graph.add_edge("a", "b", "r")
    await graph.delete_graph()
    assert await graph.get_graph_counts() == (0, 0)
    await graph.add_nodes([("a", {"name": "a", "type": "Node"})])
    assert await graph.get_graph_counts() == (1, 0)


@pytest.mark.asyncio
async def test_provenance_stamp_survives_plain_reupsert_and_learned_values(graph):
    """A plain re-add keeps stored provenance and learned values; a fold adds its ref once."""
    ref = make_source_ref_key(uuid4(), uuid4())
    await graph.add_nodes([("a", {"name": "a", "type": "Node"})], source_ref_key=ref)
    await graph.set_node_feedback_weights({"a": 0.8})
    await graph.add_nodes([("a", {"name": "a", "type": "Node", "v": 2})])
    node = (await graph.get_node_delete_data(["a"]))["a"]
    assert node.source_ref_keys == [ref]
    assert node.node_properties["feedback_weight"] == 0.8
    assert node.node_properties["v"] == 2
    await graph.add_nodes([("a", {"name": "a", "type": "Node"})], source_ref_key=ref)
    assert (await graph.get_node_delete_data(["a"]))["a"].source_ref_keys == [ref]


@pytest.mark.asyncio
async def test_delete_edge_triples_keeps_endpoints(graph):
    await graph.add_nodes(
        [("a", {"name": "a", "type": "Node"}), ("b", {"name": "b", "type": "Node"})]
    )
    await graph.add_edges([("a", "b", "r", {}), ("a", "b", "s", {})])
    await graph.delete_edge_triples([EdgeIdentity("a", "b", "r")])
    assert await graph.has_edges([("a", "b", "r"), ("a", "b", "s")]) == [("a", "b", "s")]
    assert {n["id"] for n in await graph.get_nodes(["a", "b"])} == {"a", "b"}
    assert graph.supports_per_row_source_refs is True
    assert graph.supports_incremental_chunk_updates is False
