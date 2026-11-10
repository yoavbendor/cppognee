"""Generic graph-adapter contract, run against nanolance and (when pyturso is installed) Turso.

Ported from the Turso-applicable cases of the in-tree graph unit tests (native counts,
entity type lookup, node-only filtered reads, top-degree seeds) plus the write and delete
semantics every in-tree adapter shares. The originals are untouched.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

NODES = {
    "alice": {"name": "Alice", "type": "Person"},
    "bob": {"name": "Bob", "type": "Person"},
    "carol": {"name": "Carol", "type": "Place"},
    "person": {"name": "Person", "type": "EntityType"},
}
EDGES = [
    ("alice", "bob", "knows"),
    ("bob", "carol", "visited"),
    ("alice", "person", "is_a"),
    ("bob", "person", "is_a"),
]


def _node(node_id, props):
    """A DataPoint-shaped object, as add_nodes receives from the pipeline."""
    return SimpleNamespace(model_dump=lambda: {"id": node_id, **props})


async def _fill(adapter):
    await adapter.add_nodes([_node(node_id, props) for node_id, props in NODES.items()])
    for source, target, relation in EDGES:
        await adapter.add_edge(source, target, relation)


@pytest.fixture(params=["nanolance", "turso"])
def backend(request):
    if request.param == "turso":
        pytest.importorskip("turso", reason="pyturso not installed")
    return request.param


@pytest_asyncio.fixture
async def adapter(backend, tmp_path):
    if backend == "nanolance":
        from cognee_nanograph import NanolanceGraphAdapter

        instance = NanolanceGraphAdapter(graph_database_url=str(tmp_path / "graph"))
    else:
        from cognee.infrastructure.databases.graph.turso.adapter import TursoAdapter

        instance = TursoAdapter(database_path=str(tmp_path / "graph.db"))
    await instance.initialize()
    yield instance
    if backend == "turso":
        await instance.engine.dispose()


@pytest.mark.asyncio
async def test_empty_graph_reads(adapter):
    assert await adapter.is_empty() is True
    assert await adapter.get_graph_counts() == (0, 0)
    assert await adapter.get_graph_data() == ([], [])
    assert await adapter.get_neighborhood(["missing"], depth=2) == ([], [])
    assert await adapter.get_node("missing") is None


@pytest.mark.asyncio
async def test_writes_are_idempotent_upserts(adapter):
    await _fill(adapter)
    first = await adapter.get_graph_data()
    await _fill(adapter)
    assert await adapter.get_graph_counts() == (len(NODES), len(EDGES))
    assert await adapter.get_graph_data() == first
    assert await adapter.is_empty() is False


@pytest.mark.asyncio
async def test_properties_round_trip_through_json(adapter):
    payload = {
        "name": "Alice",
        "type": "Person",
        "tags": ["a", "b"],
        "meta": {"n": 1.5, "ok": True},
    }
    await adapter.add_nodes([("alice", {**payload, "unicode": "שלום — ✓"})])
    nodes, _ = await adapter.get_graph_data()
    ((node_id, props),) = nodes
    assert node_id == "alice"
    assert props == {**payload, "unicode": "שלום — ✓"}
    assert (await adapter.get_node("alice"))["meta"] == {"n": 1.5, "ok": True}


@pytest.mark.asyncio
async def test_has_edges_returns_only_existing_triples(adapter):
    await _fill(adapter)
    present = await adapter.has_edges([("alice", "bob", "knows"), ("alice", "bob", "visited")])
    assert present == [("alice", "bob", "knows")]
    assert await adapter.has_edge("bob", "carol", "visited") is True
    assert await adapter.has_edge("carol", "bob", "visited") is False


@pytest.mark.asyncio
async def test_get_edges_and_connections_name_the_neighbours(adapter):
    await _fill(adapter)
    edges = await adapter.get_edges("bob")
    assert sorted(rel for _, rel, _ in edges) == ["is_a", "knows", "visited"]
    connections = await adapter.get_connections("bob")
    assert len(connections) == 3
    assert {other["id"] for _, _, other in connections} == {"alice", "carol", "person"}
    assert {edge["relationship_name"] for _, edge, _ in connections} == {
        "is_a",
        "knows",
        "visited",
    }


@pytest.mark.asyncio
async def test_neighbors_are_distinct_nodes(adapter):
    await _fill(adapter)
    neighbours = await adapter.get_neighbors("alice")
    assert sorted(node["id"] for node in neighbours) == ["bob", "person"]


@pytest.mark.asyncio
async def test_delete_nodes_removes_incident_edges(adapter):
    await _fill(adapter)
    await adapter.delete_nodes(["bob", "never-existed"])
    nodes, edges = await adapter.get_graph_data()
    assert sorted(node_id for node_id, _ in nodes) == ["alice", "carol", "person"]
    assert edges == [("alice", "person", "is_a", {})]
    assert await adapter.get_graph_counts() == (3, 1)


@pytest.mark.asyncio
async def test_filtered_reads_return_nodes_only_without_edges(adapter):
    await _fill(adapter)
    nodes, edges = await adapter.get_filtered_graph_data(
        [{"type": ["Person"]}], include_edges=False
    )
    assert sorted(node_id for node_id, _ in nodes) == ["alice", "bob"]
    assert edges == []


@pytest.mark.asyncio
async def test_filtered_reads_keep_edges_between_matched_nodes(adapter):
    await _fill(adapter)
    nodes, edges = await adapter.get_filtered_graph_data([{"type": ["Person"]}])
    assert sorted(node_id for node_id, _ in nodes) == ["alice", "bob"]
    assert edges == [("alice", "bob", "knows", {})]


@pytest.mark.asyncio
async def test_query_is_not_supported(adapter):
    assert adapter.supports_cypher_queries is False
    with pytest.raises(NotImplementedError, match="Cypher"):
        await adapter.query("MATCH (n) RETURN n", {})


@pytest.mark.asyncio
async def test_nanolance_counts_are_native():
    """The adapter overrides the interface default, so counts never read full metrics."""
    from cognee_nanograph import NanolanceGraphAdapter

    assert NanolanceGraphAdapter.get_graph_counts is not GraphDBInterface.get_graph_counts


@pytest.mark.asyncio
async def test_turso_counts_are_native():
    pytest.importorskip("turso", reason="pyturso not installed")
    from cognee.infrastructure.databases.graph.turso.adapter import TursoAdapter

    assert TursoAdapter.get_graph_counts is not GraphDBInterface.get_graph_counts


@pytest.mark.asyncio
async def test_counts_agree_with_metrics(adapter):
    await _fill(adapter)
    metrics = await adapter.get_graph_metrics()
    assert await adapter.get_graph_counts() == (metrics["num_nodes"], metrics["num_edges"])


@pytest.mark.asyncio
async def test_entity_type_names_map_entities_to_their_type(adapter):
    await _fill(adapter)
    assert await adapter.get_entity_type_names(["alice", "bob", "carol"]) == {
        "alice": "Person",
        "bob": "Person",
    }
    # Never outward from a type node.
    assert await adapter.get_entity_type_names(["person"]) == {}


@pytest.mark.asyncio
async def test_top_degree_seeds_rank_by_degree(adapter):
    assert await adapter.get_top_degree_node_ids(5) == []
    for node_id in ["hub", "incoming", "outgoing", "isolated"]:
        await adapter.add_node(_node(node_id, {"name": node_id, "type": "Node"}))
    assert len(await adapter.get_top_degree_node_ids(5)) == 4
    await adapter.add_edge("incoming", "hub", "rel")
    await adapter.add_edge("hub", "outgoing", "rel")
    assert await adapter.get_top_degree_node_ids(1) == ["hub"]
    assert set(await adapter.get_top_degree_node_ids(5)) == {
        "hub",
        "incoming",
        "outgoing",
        "isolated",
    }
