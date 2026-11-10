"""The nanolance graph adapter's tests need nanolance, which is not a cognee dependency.

Without it this directory is not collected, so cognee's own unit run stays green; the
nanolance-graph workflow installs it and runs these tests.
"""

import importlib.util

if importlib.util.find_spec("nanolance") is None:
    collect_ignore_glob = ["test_*.py"]
