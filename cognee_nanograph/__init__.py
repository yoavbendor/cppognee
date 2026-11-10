"""nanolance graph backend for cognee, registered from outside cognee's core.

Call ``register()`` before cognee builds a graph engine, then set
``GRAPH_DATABASE_PROVIDER=nanolance`` and ``GRAPH_DATABASE_URL`` to a directory.
"""

from cognee.infrastructure.databases.graph.use_graph_adapter import use_graph_adapter

from .adapter import NanolanceGraphAdapter


def register() -> None:
    """Make ``GRAPH_DATABASE_PROVIDER=nanolance`` resolve to ``NanolanceGraphAdapter``."""
    use_graph_adapter("nanolance", NanolanceGraphAdapter)


__all__ = ["NanolanceGraphAdapter", "register"]
