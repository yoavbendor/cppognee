"""pytest plugin that runs cognee's suites on the nanolance graph.

    pytest -p cognee_nanograph.pytest_plugin ...

It calls ``register()`` before any test builds a graph engine, so
``GRAPH_DATABASE_PROVIDER=nanolance`` resolves to ``NanolanceGraphAdapter`` and its
dataset handler. With that provider set it also fails the run when a test built a
Ladybug graph engine or no nanolance one, so a green run means the graph really was
nanolance.
"""

import os

import pytest

import cognee_nanograph
from cognee_nanograph import NanolanceGraphAdapter

cognee_nanograph.register()

_built = {"nanolance": 0, "ladybug": 0}


def _count(cls, name: str) -> None:
    original = cls.__init__

    def init(self, *args, **kwargs):
        _built[name] += 1
        original(self, *args, **kwargs)

    cls.__init__ = init


_count(NanolanceGraphAdapter, "nanolance")


def pytest_configure(config):
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    _count(LadybugAdapter, "ladybug")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("GRAPH_DATABASE_PROVIDER", "").lower() != "nanolance":
        return
    if session.testscollected == 0 or session.config.option.collectonly:
        return
    problem = None
    if _built["ladybug"]:
        problem = f"{_built['ladybug']} Ladybug graph engine(s) were built"
    elif not _built["nanolance"]:
        problem = "no nanolance graph engine was built"
    if problem:
        session.config.get_terminal_writer().line(
            f"cognee_nanograph: GRAPH_DATABASE_PROVIDER=nanolance, but {problem}.", red=True
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
