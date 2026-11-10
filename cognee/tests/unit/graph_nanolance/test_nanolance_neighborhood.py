"""get_neighborhood on nanolance must equal a brute-force breadth-first search."""

import random
from collections import deque

import pytest
import pytest_asyncio

from cognee_nanograph import NanolanceGraphAdapter

RELATIONS = ["knows", "visited", "is_a"]


def _random_graph(seed: int, nodes: int = 40, edges: int = 90):
    rng = random.Random(seed)
    ids = [f"n{i}" for i in range(nodes)]
    triples = set()
    while len(triples) < edges:
        s, t = rng.choice(ids), rng.choice(ids)
        triples.add((s, t, rng.choice(RELATIONS)))
    return ids, sorted(triples)


def _brute_force(ids, triples, seeds, depth, edge_types):
    """Nodes within ``depth`` hops of the seeds (undirected, optionally only ``edge_types``)
    and the edges with both endpoints among them."""
    allowed = [t for t in triples if edge_types is None or t[2] in edge_types]
    adjacency = {}
    for s, t, _ in allowed:
        adjacency.setdefault(s, set()).add(t)
        adjacency.setdefault(t, set()).add(s)
    dist = {seed: 0 for seed in seeds}
    queue = deque(seeds)
    while queue:
        node = queue.popleft()
        if dist[node] == depth:
            continue
        for nxt in adjacency.get(node, ()):
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    members = set(dist)
    induced = {(s, t, r) for s, t, r in triples if s in members and t in members}
    return members, induced


@pytest_asyncio.fixture
async def graph(tmp_path):
    adapter = NanolanceGraphAdapter(graph_database_url=str(tmp_path / "g"))
    await adapter.initialize()
    return adapter


async def _load(adapter, ids, triples):
    await adapter.add_nodes([(i, {"name": i, "type": "Node"}) for i in ids])
    await adapter.add_edges([(s, t, r, {}) for s, t, r in triples])


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("depth", [1, 2, 3])
async def test_neighborhood_matches_brute_force_bfs(graph, seed, depth):
    ids, triples = _random_graph(seed)
    await _load(graph, ids, triples)
    seeds = ["n0", "n7"]
    nodes, edges = await graph.get_neighborhood(seeds, depth=depth)
    expected_nodes, expected_edges = _brute_force(ids, triples, seeds, depth, None)
    assert {node_id for node_id, _ in nodes} == expected_nodes
    assert {(s, t, r) for s, t, r, _ in edges} == expected_edges


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [4, 5])
async def test_neighborhood_with_edge_types_traverses_only_those_types(graph, seed):
    ids, triples = _random_graph(seed)
    await _load(graph, ids, triples)
    seeds = ["n3"]
    nodes, edges = await graph.get_neighborhood(seeds, depth=3, edge_types=["knows"])
    expected_nodes, _ = _brute_force(ids, triples, seeds, 3, ["knows"])
    assert {node_id for node_id, _ in nodes} == expected_nodes
    # Edges are induced over the returned nodes, whatever their relation.
    _, expected_edges = _brute_force(ids, triples, seeds, 3, None)
    assert {(s, t, r) for s, t, r, _ in edges} <= expected_edges | {
        t for t in triples if t[0] in expected_nodes and t[1] in expected_nodes
    }


@pytest.mark.asyncio
async def test_neighborhood_of_unknown_seed_is_empty(graph):
    await _load(graph, ["a", "b"], [("a", "b", "knows")])
    assert await graph.get_neighborhood(["zzz"], depth=2) == ([], [])


@pytest.mark.asyncio
async def test_neighborhood_rejects_depth_below_one(graph):
    with pytest.raises(ValueError, match="depth"):
        await graph.get_neighborhood(["a"], depth=0)
