"""Ladybug's remote extension installs are gated by the egress policy. Native code downloads the
extension, so the Python guard cannot see it; the helpers check the policy before INSTALL. Every case
runs in a child process because an audit hook cannot be removed."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

POLICY_WITHOUT_LADYBUG = {
    "mode": "enforce",
    "runtime": {"hosts": [{"host": "llm.corp.example", "ports": [443], "role": "llm"}]},
}

POLICY_WITH_LADYBUG = {
    "mode": "enforce",
    "runtime": {"hosts": [{"host": "extension.ladybugdb.com", "ports": [80], "role": "download"}]},
}

PRELUDE = """
import sys
import cognee_egress
from cognee_egress import EgressDenied
cognee_egress.install_from_env()

def denied(fn):
    try:
        fn()
    except EgressDenied:
        return True
    return False
"""


def run_child(tmp_path, body, policy=POLICY_WITHOUT_LADYBUG):
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    env.pop("COGNEE_EGRESS_POLICY", None)
    env["PYTHONPATH"] = str(REPO)
    if policy is not None:
        pol = tmp_path / "policy.json"
        pol.write_text(json.dumps(policy))
        env["COGNEE_EGRESS_POLICY"] = str(pol)
    code = PRELUDE + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def assert_ok(proc):
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_denied_policy_blocks_remote_json_install(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            from cognee_db_workers import _kuzu_helpers as h
            assert denied(h.check_remote_extension_install)

            h.bundled_json_extension_path = lambda execute, bundled_dir=None: None
            calls = []

            def execute(statement):
                calls.append(statement)
                if statement == "LOAD EXTENSION JSON;":
                    raise Exception("Extension: json has not been installed.")

            try:
                h.load_json_extension(execute)
            except EgressDenied as error:
                assert "not installed locally" in str(error), str(error)
                assert "extension.ladybugdb.com" in str(error), str(error)
            else:
                raise SystemExit("load_json_extension did not raise")
            assert "INSTALL JSON;" not in calls, calls
            """,
        )
    )


def test_listed_repo_allows_remote_json_install(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            from cognee_db_workers import _kuzu_helpers as h
            h.check_remote_extension_install()

            h.bundled_json_extension_path = lambda execute, bundled_dir=None: None
            calls = []
            state = {"installed": False}

            def execute(statement):
                calls.append(statement)
                if statement == "LOAD EXTENSION JSON;" and not state["installed"]:
                    raise Exception("Extension: json has not been installed.")
                if statement == "INSTALL JSON;":
                    state["installed"] = True

            h.load_json_extension(execute)
            assert "INSTALL JSON;" in calls, calls
            """,
            policy=POLICY_WITH_LADYBUG,
        )
    )


def test_no_policy_allows_remote_json_install(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            from cognee_db_workers import _kuzu_helpers as h
            h.check_remote_extension_install()

            h.bundled_json_extension_path = lambda execute, bundled_dir=None: None
            calls = []
            state = {"installed": False}

            def execute(statement):
                calls.append(statement)
                if statement == "LOAD EXTENSION JSON;" and not state["installed"]:
                    raise Exception("Extension: json has not been installed.")
                if statement == "INSTALL JSON;":
                    state["installed"] = True

            h.load_json_extension(execute)
            assert "INSTALL JSON;" in calls, calls
            """,
            policy=None,
        )
    )


def test_local_warmup_returns_without_opening_a_database_when_denied(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            import types
            from cognee_db_workers import _kuzu_helpers as h

            class Database:
                def __init__(self, *args, **kwargs):
                    raise AssertionError("should not open")

            sys.modules["ladybug"] = types.SimpleNamespace(Database=Database)
            h.bundled_extensions_present = lambda bundled_dir=None: False
            h.install_json_extension_local(64 * 1024 * 1024)
            """,
        )
    )
