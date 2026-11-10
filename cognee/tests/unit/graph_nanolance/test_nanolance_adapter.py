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


@pytest.mark.asyncio
async def test_top_degree_seeds_rank_by_degree_then_id_then_fill_with_isolated(graph):
    await graph.add_nodes([(i, {"name": i, "type": "Node"}) for i in ["hub", "a", "b", "c", "iso"]])
    # Degrees: hub 3, a 2, b 2 (a tie, so id order), c 1, iso 0 (fill only).
    await graph.add_edges(
        [("hub", "a", "r", {}), ("hub", "b", "r", {}), ("hub", "c", "r", {}), ("b", "a", "s", {})]
    )
    assert await graph.get_top_degree_node_ids(1) == ["hub"]
    assert await graph.get_top_degree_node_ids(3) == ["hub", "a", "b"]
    everything = await graph.get_top_degree_node_ids(10)
    assert len(everything) == 5 and everything[:4] == ["hub", "a", "b", "c"]
    assert set(everything) == {"hub", "a", "b", "c", "iso"}


@pytest.mark.asyncio
async def test_top_degree_seeds_on_empty_and_edgeless_graphs(graph):
    assert await graph.get_top_degree_node_ids(5) == []
    await graph.add_nodes(
        [("x", {"name": "x", "type": "Node"}), ("y", {"name": "y", "type": "Node"})]
    )
    assert set(await graph.get_top_degree_node_ids(5)) == {"x", "y"}
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        await graph.get_top_degree_node_ids(0)


@pytest.mark.asyncio
async def test_entity_type_names_follow_one_is_a_hop_to_entity_types(graph):
    await graph.add_nodes(
        [
            ("e1", {"name": "e1", "type": "Entity"}),
            ("e2", {"name": "e2", "type": "Entity"}),
            ("e3", {"name": "e3", "type": "Entity"}),
            ("person", {"name": "Person", "type": "EntityType"}),
            ("thing", {"name": "Thing", "type": "Node"}),
        ]
    )
    await graph.add_edges(
        [
            ("e1", "person", "is_a", {}),
            ("e2", "thing", "is_a", {}),  # target is not an EntityType
            ("e3", "person", "rel", {}),  # not is_a
            ("person", "e2", "is_a", {}),  # wrong direction
        ]
    )
    assert await graph.get_entity_type_names(["e1", "e2", "e3", "person", "ghost"]) == {
        "e1": "Person"
    }
    assert await graph.get_entity_type_names([]) == {}


@pytest.mark.asyncio
async def test_temporal_anchors_direct_and_through_entities(graph):
    await graph.add_nodes(
        [
            ("c1", {"name": "c1", "type": "DocumentChunk"}),
            ("c2", {"name": "c2", "type": "DocumentChunk"}),
            ("c3", {"name": "c3", "type": "DocumentChunk"}),
            ("e1", {"name": "e1", "type": "Entity"}),
            ("e2", {"name": "e2", "type": "Entity"}),
            ("t_in", {"name": "t_in", "type": "Timestamp", "time_at": 1000, "time_until": 2000}),
            ("t_out", {"name": "t_out", "type": "Timestamp", "time_at": 5000, "time_until": 6000}),
            ("p", {"name": "p", "type": "Person", "time_at": 1200, "time_until": 1300}),
        ]
    )
    await graph.add_edges(
        [
            ("c1", "e1", "contains", {}),  # c1 reaches t_in through e1
            ("e1", "t_in", "born_at", {}),
            ("c2", "t_in", "mentions", {}),  # c2 directly
            ("c3", "e2", "contains", {}),
            ("e2", "t_out", "died_at", {}),  # outside the window below
            ("c3", "p", "mentions", {}),  # a Person, not a Timestamp
        ]
    )
    anchors = await graph.get_temporal_anchors(["c1", "c2", "c3"], ["e1", "e2"], 1500, 1600)
    assert anchors == {
        "timestamp_ids": {"t_in"},
        "chunk_ids": {"c1", "c2"},
        "entity_ids": {"e1"},
        "chunk_timestamps": {"c1": {"t_in"}, "c2": {"t_in"}},
    }
    # The window is half-open: t_in ends at 2000, so [2000, ...) does not reach it.
    assert (await graph.get_temporal_anchors(["c1"], [], 2000, None))["chunk_ids"] == set()
    empty = await graph.get_temporal_anchors([], [], None, None)
    assert empty["chunk_ids"] == set() and empty["timestamp_ids"] == set()


@pytest.mark.asyncio
async def test_read_helpers_on_an_empty_graph(graph):
    assert await graph.get_triplets_batch(offset=0, limit=5) == []
    assert await graph.get_entity_type_names(["a"]) == {}
    anchors = await graph.get_temporal_anchors(["c"], ["e"], 0, None)
    assert anchors["chunk_ids"] == set() and anchors["timestamp_ids"] == set()


@pytest.mark.asyncio
async def test_triplets_batch_pages_in_key_order_and_refuses_negative_bounds(graph):
    await graph.add_nodes([(i, {"name": f"name-{i}", "type": "Node", "v": i}) for i in "abc"])
    await graph.add_edges(
        [("b", "a", "z", {"w": 1}), ("a", "c", "r", {}), ("a", "b", "r", {"w": 2})]
    )
    page = await graph.get_triplets_batch(offset=0, limit=2)
    assert [(t["start_node"]["id"], t["end_node"]["id"]) for t in page] == [("a", "b"), ("a", "c")]
    assert page[0]["relationship_properties"] == {"relationship_name": "r", "w": 2}
    assert page[0]["start_node"] == {"id": "a", "name": "name-a", "type": "Node", "v": "a"}
    rest = await graph.get_triplets_batch(offset=2, limit=5)
    assert [t["relationship_properties"]["relationship_name"] for t in rest] == ["z"]
    assert await graph.get_triplets_batch(offset=10, limit=5) == []
    assert await graph.get_triplets_batch(offset=0, limit=0) == []
    with pytest.raises(ValueError, match="Offset"):
        await graph.get_triplets_batch(offset=-1, limit=1)
    with pytest.raises(ValueError, match="Limit"):
        await graph.get_triplets_batch(offset=0, limit=-1)
