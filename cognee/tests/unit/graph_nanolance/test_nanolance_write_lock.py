"""Writes to one nanolance graph directory from several processes are serialized.

Each write is a read-modify-write (provenance folding reads the stored row, then merges),
so two processes writing the same directory without a shared lock lose updates.
"""

import subprocess
import sys
from uuid import NAMESPACE_URL, uuid5

import pytest

from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee_nanograph import NanolanceGraphAdapter

HUB = "hub"
WRITES_PER_PROCESS = 30

# Child process: for each write, upserts a new node and an edge to the shared hub, then
# attaches a new provenance ref to the hub (a read-modify-write of its provenance columns).
WORKER = """
import asyncio, sys
from uuid import NAMESPACE_URL, uuid5
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee_nanograph import NanolanceGraphAdapter

async def main(path, tag, count):
    adapter = NanolanceGraphAdapter(graph_database_url=path)
    await adapter.initialize()
    for i in range(count):
        await adapter.add_nodes([(f"{tag}-{i}", {"name": f"{tag}-{i}", "type": "Item"})])
        await adapter.add_edges([(f"{tag}-{i}", "hub", "member_of", {})])
        ref = make_source_ref_key(uuid5(NAMESPACE_URL, tag), uuid5(NAMESPACE_URL, f"{tag}-{i}"))
        await adapter.attach_node_source_refs(["hub"], [ref])

asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3])))
"""


def _ref(tag: str, i: int) -> str:
    return make_source_ref_key(uuid5(NAMESPACE_URL, tag), uuid5(NAMESPACE_URL, f"{tag}-{i}"))


@pytest.mark.asyncio
async def test_write_lock_file_is_created_in_graph_directory(tmp_path):
    graph_dir = tmp_path / "graph"
    adapter = NanolanceGraphAdapter(graph_database_url=str(graph_dir))

    await adapter.add_node("n1", {"name": "n1", "type": "Item"})

    assert (graph_dir / ".write.lock").is_file()


@pytest.mark.asyncio
async def test_concurrent_processes_lose_no_writes(tmp_path):
    graph_dir = str(tmp_path / "graph")
    adapter = NanolanceGraphAdapter(graph_database_url=graph_dir)
    await adapter.add_nodes([(HUB, {"name": HUB, "type": "Hub"})])

    tags = ("a", "b")
    children = [
        subprocess.Popen(
            [sys.executable, "-c", WORKER, graph_dir, tag, str(WRITES_PER_PROCESS)],
            stderr=subprocess.PIPE,
            text=True,
        )
        for tag in tags
    ]
    for tag, child in zip(tags, children):
        _, stderr = child.communicate(timeout=120)
        assert child.returncode == 0, f"writer {tag} failed: {stderr[-2000:]}"

    expected = [f"{tag}-{i}" for tag in tags for i in range(WRITES_PER_PROCESS)]
    nodes = await adapter.get_nodes(expected)
    assert sorted(node["id"] for node in nodes) == sorted(expected)

    edges = await adapter.get_edges(HUB)
    assert len(edges) == len(expected)

    hub = (await adapter.get_node_delete_data([HUB]))[HUB]
    expected_refs = {_ref(tag, i) for tag in tags for i in range(WRITES_PER_PROCESS)}
    assert set(hub.source_ref_keys) == expected_refs
    assert len(hub.source_ref_keys) == len(expected_refs)
