"""Tests for check_provider_endpoint. An audit hook cannot be removed, so every case runs in a child process."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

POLICY = {
    "mode": "enforce",
    "runtime": {
        "hosts": [
            {"host": "llm.corp.example", "ports": [443], "role": "llm"},
            {"host": "emb.corp.example", "ports": [443], "role": "embedding"},
        ],
        "subprocesses": ["true"],
    },
}

PRELUDE = """
import cognee_egress
cognee_egress.install_from_env()
from cognee_egress import EgressDenied
from cognee.infrastructure.llm.egress_check import check_provider_endpoint

def denied(fn):
    try:
        fn()
    except EgressDenied:
        return True
    return False

def message(fn):
    try:
        fn()
    except EgressDenied as error:
        return str(error)
    raise AssertionError("expected EgressDenied")
"""


def run_child(tmp_path, body, policy=POLICY, env_extra=None):
    # Hermetic: the child reads no repo .env and no inherited LLM, embedding or AWS settings.
    env = {
        k: v
        for k, v in os.environ.items()
        if "proxy" not in k.lower() and not k.startswith(("LLM_", "EMBEDDING_", "AWS_", "COGNEE_"))
    }
    empty_env_file = tmp_path / "empty.env"
    empty_env_file.write_text("")
    env.update(PYTHONPATH=str(REPO), COGNEE_ENV_FILE=str(empty_env_file), TELEMETRY_DISABLED="1")
    if policy is not None:
        pol = tmp_path / "policy.json"
        pol.write_text(json.dumps(policy))
        env["COGNEE_EGRESS_POLICY"] = str(pol)
    env.update(env_extra or {})
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


def test_policy_decisions(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            assert denied(lambda: check_provider_endpoint("openai", None, "llm")), "default OpenAI URL"
            assert "api.openai.com" in message(lambda: check_provider_endpoint("openai", None, "llm"))
            check_provider_endpoint("openai", "https://llm.corp.example/v1", "llm")
            assert denied(
                lambda: check_provider_endpoint("openai", "https://emb.corp.example/v1", "llm")
            ), "listed for another role"
            check_provider_endpoint("ollama", None, "llm")
            check_provider_endpoint("fastembed", None, "embedding")
            assert denied(lambda: check_provider_endpoint("someunknown", None, "llm")), "no default"
            """,
        )
    )


def test_bedrock_endpoint_is_checked_by_region(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            assert "region is unknown" in message(lambda: check_provider_endpoint("bedrock", None, "llm"))
            """,
        )
    )
    assert_ok(
        run_child(
            tmp_path,
            """
            msg = message(lambda: check_provider_endpoint("bedrock", None, "llm"))
            assert "https://bedrock-runtime.us-east-1.amazonaws.com" in msg, msg
            """,
            env_extra={"AWS_REGION": "us-east-1"},
        )
    )


def test_no_policy_is_a_no_op(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            check_provider_endpoint("openai", None, "llm")
            check_provider_endpoint("someunknown", None, "llm")
            """,
            policy=None,
        )
    )


def test_default_clients_check_endpoint_at_construction(tmp_path):
    # The default framework (litellm_native), the instructor framework and the embedding engine
    # all construct their client with the OpenAI default URL, which the policy does not list.
    assert_ok(
        run_child(
            tmp_path,
            """
            from cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client import (
                get_native_client,
            )
            from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
                get_llm_client,
            )
            from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
                get_embedding_engine,
            )

            for build in (get_native_client, get_llm_client, get_embedding_engine):
                msg = message(build)
                assert "api.openai.com" in msg, (build.__name__, msg)
            """,
            env_extra={
                "LLM_PROVIDER": "openai",
                "LLM_API_KEY": "sk-test",
                "LLM_MODEL": "openai/gpt-4o-mini",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "openai/text-embedding-3-large",
            },
        )
    )
