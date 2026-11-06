import pytest
from pydantic import Field

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils import get_graph_from_model


class Passage(DataPoint):
    name: str
    embed_text: str = Field(default="", exclude=True)
    metadata: dict = {"index_fields": ["embed_text"], "store_index_text": False}


@pytest.mark.asyncio
async def test_excluded_field_is_embeddable_but_not_dumped():
    p = Passage(name="file:///a.md", embed_text="only embedded")
    nodes, _ = await get_graph_from_model(p, added_nodes={}, added_edges={}, visited_properties={})
    node = nodes[0]
    assert DataPoint.get_embeddable_data(node) == "only embedded"  # the vector index reads it
    assert "embed_text" not in node.model_dump()  # graph adapters store model_dump()


def test_edge_label_never_uses_unstored_index_text():
    from cognee.modules.graph.utils.prepare_edges_for_storage import _get_node_label

    p = Passage(name="file:///a.md", embed_text="secret passage text")
    assert _get_node_label(p, p.id) == "file:///a.md"
