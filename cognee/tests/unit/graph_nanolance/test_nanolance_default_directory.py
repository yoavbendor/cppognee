"""Default graph directory: no URL means <system root>/databases/cognee_graph_nanolance."""

import os
from types import SimpleNamespace

import pytest

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.graph.config import GraphConfig
from cognee_nanograph import NanolanceGraphAdapter


@pytest.fixture
def system_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cognee_nanograph.adapter.get_base_config",
        lambda: SimpleNamespace(system_root_directory=str(tmp_path)),
    )
    return tmp_path


def test_empty_url_uses_the_default_directory_under_the_system_root(system_root):
    adapter = NanolanceGraphAdapter(graph_database_url="")

    assert adapter.database_path == os.path.join(
        str(system_root), "databases", "cognee_graph_nanolance"
    )


def test_given_url_is_used_as_is(system_root, tmp_path):
    given = tmp_path / "elsewhere"

    adapter = NanolanceGraphAdapter(graph_database_url=str(given))

    assert adapter.database_path == str(given)


def test_default_matches_the_location_ladybug_style_config_derives(monkeypatch):
    # The adapter and GraphConfig.fill_derived must agree on where an unconfigured
    # nanolance graph lives, so switching the provider does not move the default.
    monkeypatch.setenv("GRAPH_DATABASE_PROVIDER", "nanolance")
    monkeypatch.delenv("GRAPH_FILE_PATH", raising=False)
    monkeypatch.delenv("GRAPH_FILENAME", raising=False)

    config = GraphConfig()
    adapter = NanolanceGraphAdapter(graph_database_url="")

    assert adapter.database_path == config.graph_file_path
    assert os.path.isabs(adapter.database_path)
    assert get_base_config().system_root_directory in adapter.database_path
