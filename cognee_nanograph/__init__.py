"""nanolance graph backend for cognee, registered from outside cognee's core.

Call ``register()`` before cognee builds a graph engine, then set
``GRAPH_DATABASE_PROVIDER=nanolance``. ``GRAPH_DATABASE_URL`` is optional: when unset the graph
lives at ``<system root>/databases/cognee_graph_nanolance``. With backend access control on
(the default), each user+dataset gets its own directory through the registered dataset handler.
"""

from cognee.infrastructure.databases.dataset_database_handler.use_dataset_database_handler import (
    use_dataset_database_handler,
)
from cognee.infrastructure.databases.graph.use_graph_adapter import use_graph_adapter

from .adapter import NanolanceGraphAdapter
from .dataset_database_handler import NanolanceDatasetDatabaseHandler


def register() -> None:
    """Make ``GRAPH_DATABASE_PROVIDER=nanolance`` resolve to ``NanolanceGraphAdapter``.

    Also registers ``NanolanceDatasetDatabaseHandler`` under the ``nanolance`` provider.
    """
    use_graph_adapter("nanolance", NanolanceGraphAdapter)
    use_dataset_database_handler("nanolance", NanolanceDatasetDatabaseHandler, "nanolance")


__all__ = ["NanolanceDatasetDatabaseHandler", "NanolanceGraphAdapter", "register"]
