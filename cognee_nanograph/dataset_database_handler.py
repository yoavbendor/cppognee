"""Per-dataset nanolance graph directories for backend access control."""

import os
from uuid import UUID

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.dataset_database_handler import DatasetDatabaseHandlerInterface
from cognee.infrastructure.databases.graph.get_graph_engine import graph_engine_cache
from cognee.infrastructure.files.storage.get_file_storage import get_file_storage
from cognee.modules.users.models import DatasetDatabase, User

NANOLANCE_DATASET_DATABASE_HANDLER = "nanolance"


def _owner_databases_directory(owner_id) -> str:
    """Per-owner directory holding that owner's dataset graph directories."""
    return os.path.join(get_base_config().system_root_directory, "databases", str(owner_id))


class NanolanceDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """Gives each user+dataset its own nanolance directory under the owner's databases folder."""

    @classmethod
    async def create_dataset(cls, dataset_id: UUID | None, user: User | None) -> dict:
        """Return connection info pointing the dataset at its own nanolance directory.

        The directory is ``<system root>/databases/<owner id>/<dataset id>.nanolance``.
        Nothing is created here; the adapter creates its tables on first use.
        """
        from cognee.infrastructure.databases.graph.config import get_graph_config

        if get_graph_config().graph_database_provider != NANOLANCE_DATASET_DATABASE_HANDLER:
            raise ValueError(
                "NanolanceDatasetDatabaseHandler can only be used with the nanolance graph "
                "database provider."
            )
        if dataset_id is None or user is None:
            raise ValueError("Creating a nanolance dataset database needs a dataset id and owner.")

        graph_db_name = f"{dataset_id}.nanolance"
        return {
            "graph_database_name": graph_db_name,
            "graph_database_url": os.path.join(_owner_databases_directory(user.id), graph_db_name),
            "graph_database_provider": NANOLANCE_DATASET_DATABASE_HANDLER,
            # Local directories only; the adapter refuses a remote key.
            "graph_database_key": "",
            "graph_dataset_database_handler": NANOLANCE_DATASET_DATABASE_HANDLER,
            "graph_database_connection_info": {},
        }

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase):
        """Evict the dataset's cached engines, then remove its directory.

        Evicting first means no open adapter still holds the directory's files; the
        directory is never opened just to be dropped.
        """
        await graph_engine_cache.aevict_for_database(dataset_database.graph_database_name)
        file_storage = get_file_storage(_owner_databases_directory(dataset_database.owner_id))
        await file_storage.remove_all(dataset_database.graph_database_name)
