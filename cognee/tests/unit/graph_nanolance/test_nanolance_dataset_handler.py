"""NanolanceDatasetDatabaseHandler, register(), and provider -> handler derivation."""

import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

import cognee_nanograph
from cognee.context_global_variables import multi_user_support_possible
from cognee.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
    supported_dataset_database_handlers,
)
from cognee.infrastructure.databases.graph.config import GraphConfig, get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import (
    create_graph_engine,
    graph_engine_cache,
)
from cognee.infrastructure.databases.graph.supported_databases import supported_databases
from cognee.infrastructure.databases.vector.config import get_vectordb_config
from cognee_nanograph import NanolanceDatasetDatabaseHandler, NanolanceGraphAdapter
from cognee_nanograph import dataset_database_handler as handler_module


@pytest.fixture
def system_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        handler_module,
        "get_base_config",
        lambda: SimpleNamespace(system_root_directory=str(tmp_path)),
    )
    return tmp_path


@pytest.fixture
def nanolance_provider(monkeypatch):
    monkeypatch.setenv("GRAPH_DATABASE_PROVIDER", "nanolance")
    get_graph_config.cache_clear()
    yield
    get_graph_config.cache_clear()


def _owner(owner_id=None):
    return SimpleNamespace(id=owner_id or uuid4())


def _dataset_database(info, owner_id):
    return SimpleNamespace(
        graph_database_name=info["graph_database_name"],
        graph_database_url=info["graph_database_url"],
        owner_id=owner_id,
    )


@pytest.mark.asyncio
async def test_create_dataset_points_at_a_directory_per_owner_and_dataset(
    system_root, nanolance_provider
):
    dataset_id = uuid4()
    owner = _owner()

    info = await NanolanceDatasetDatabaseHandler.create_dataset(dataset_id, owner)

    expected_dir = os.path.join(
        str(system_root), "databases", str(owner.id), f"{dataset_id}.nanolance"
    )
    assert info["graph_database_url"] == expected_dir
    assert info["graph_database_name"] == f"{dataset_id}.nanolance"
    assert info["graph_database_provider"] == "nanolance"
    assert info["graph_dataset_database_handler"] == "nanolance"
    assert info["graph_database_key"] == ""


@pytest.mark.asyncio
async def test_create_dataset_gives_distinct_directories_per_dataset_and_owner(
    system_root, nanolance_provider
):
    owner = _owner()
    first = await NanolanceDatasetDatabaseHandler.create_dataset(uuid4(), owner)
    second = await NanolanceDatasetDatabaseHandler.create_dataset(uuid4(), owner)
    other_owner = await NanolanceDatasetDatabaseHandler.create_dataset(uuid4(), _owner())

    urls = {
        first["graph_database_url"],
        second["graph_database_url"],
        other_owner["graph_database_url"],
    }
    assert len(urls) == 3


@pytest.mark.asyncio
async def test_create_dataset_refuses_another_provider(system_root, monkeypatch):
    monkeypatch.setenv("GRAPH_DATABASE_PROVIDER", "ladybug")
    get_graph_config.cache_clear()
    try:
        with pytest.raises(ValueError, match="nanolance graph database provider"):
            await NanolanceDatasetDatabaseHandler.create_dataset(uuid4(), _owner())
    finally:
        get_graph_config.cache_clear()


@pytest.mark.asyncio
async def test_created_directory_is_a_working_graph(system_root, nanolance_provider):
    dataset_id = uuid4()
    info = await NanolanceDatasetDatabaseHandler.create_dataset(dataset_id, _owner())

    adapter = NanolanceGraphAdapter(graph_database_url=info["graph_database_url"])
    await adapter.initialize()

    assert os.path.isdir(info["graph_database_url"])
    assert await adapter.is_empty()


@pytest.mark.asyncio
async def test_delete_dataset_evicts_the_cached_engine_and_removes_the_directory(
    system_root, nanolance_provider, monkeypatch
):
    owner = _owner()
    info = await NanolanceDatasetDatabaseHandler.create_dataset(uuid4(), owner)
    database = _dataset_database(info, owner.id)
    engine_args = {
        "graph_database_provider": "nanolance",
        "graph_file_path": "",
        "graph_database_url": info["graph_database_url"],
        "graph_database_name": info["graph_database_name"],
    }
    # A live cached engine for the dataset, as a running search would hold.
    engine = create_graph_engine(**engine_args)
    await engine.initialize()
    assert graph_engine_cache.is_cached(**engine_args)

    await NanolanceDatasetDatabaseHandler.delete_dataset(database)

    assert not graph_engine_cache.is_cached(**engine_args)
    assert not os.path.exists(info["graph_database_url"])
    # The owner's folder stays; only the dataset's directory goes.
    assert os.path.isdir(os.path.dirname(info["graph_database_url"]))


def test_register_registers_adapter_and_dataset_handler():
    cognee_nanograph.register()

    assert supported_databases["nanolance"] is NanolanceGraphAdapter
    assert supported_dataset_database_handlers["nanolance"] == {
        "handler_instance": NanolanceDatasetDatabaseHandler,
        "handler_provider": "nanolance",
    }


def test_provider_derives_the_nanolance_handler_without_setting_it(monkeypatch):
    monkeypatch.setenv("GRAPH_DATABASE_PROVIDER", "nanolance")
    monkeypatch.delenv("GRAPH_DATASET_DATABASE_HANDLER", raising=False)

    assert GraphConfig().graph_dataset_database_handler == "nanolance"


def test_derivation_does_not_depend_on_registration_order(monkeypatch):
    # The config can be built (and cached) before register() runs; the derived name must
    # still be right, and the check must pass once the handler is registered.
    monkeypatch.delitem(supported_dataset_database_handlers, "nanolance", raising=False)
    monkeypatch.setenv("GRAPH_DATABASE_PROVIDER", "nanolance")
    monkeypatch.setenv("VECTOR_DB_PROVIDER", "lancedb")
    monkeypatch.setenv("VECTOR_DATASET_DATABASE_HANDLER", "lancedb")
    get_graph_config.cache_clear()
    get_vectordb_config.cache_clear()
    try:
        cached = get_graph_config()
        assert cached.graph_dataset_database_handler == "nanolance"

        cognee_nanograph.register()

        assert multi_user_support_possible() is True
    finally:
        get_graph_config.cache_clear()
        get_vectordb_config.cache_clear()


def test_unregistered_provider_fails_the_multi_user_check(monkeypatch):
    monkeypatch.delitem(supported_dataset_database_handlers, "nanolance", raising=False)
    monkeypatch.setenv("GRAPH_DATABASE_PROVIDER", "nanolance")
    monkeypatch.setenv("VECTOR_DB_PROVIDER", "lancedb")
    monkeypatch.setenv("VECTOR_DATASET_DATABASE_HANDLER", "lancedb")
    get_graph_config.cache_clear()
    get_vectordb_config.cache_clear()
    try:
        with pytest.raises(OSError, match="Unsupported graph dataset to database handler"):
            multi_user_support_possible()
    finally:
        get_graph_config.cache_clear()
        get_vectordb_config.cache_clear()
