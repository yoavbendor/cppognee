"""Tests for cognee_egress. An audit hook cannot be removed, so every case runs in a child process."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]

POLICY = {
    "mode": "enforce",
    "runtime": {
        "hosts": [
            {"host": "10.255.255.1", "ports": [443], "role": "llm"},
            {"host": "llm.corp.example", "ports": [443], "role": "llm"},
            {
                "host": ".mirror.corp.example",
                "ports": [443],
                "role": "download",
                "download_only": True,
            },
        ],
        "subprocesses": ["true"],
    },
}

PRELUDE = """
import socket, subprocess, sys
import cognee_egress
from cognee_egress import EgressDenied, guard
cognee_egress.install_from_env()

def denied(fn):
    try:
        fn()
    except EgressDenied:
        return True
    except OSError:
        return False
    return False

def tcp(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        s.connect((host, port))
    finally:
        s.close()
"""


def run_child(tmp_path, body, policy=POLICY, env_extra=None):
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps(policy))
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    env.update(PYTHONPATH=str(REPO), COGNEE_EGRESS_POLICY=str(pol))
    env.update(env_extra or {})
    code = PRELUDE + textwrap.dedent(body)
    return subprocess.run(  # noqa: PLW1510 - callers check returncode
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60
    )


def assert_ok(proc):
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_dns_of_unlisted_host_is_denied(tmp_path):
    assert_ok(run_child(tmp_path, "assert denied(lambda: socket.getaddrinfo('example.com', 443))"))


def test_connect_rules(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            assert denied(lambda: tcp('10.255.255.2', 443)), 'unlisted IP'
            assert denied(lambda: tcp('10.255.255.1', 80)), 'listed IP, wrong port'
            assert not denied(lambda: tcp('10.255.255.1', 443)), 'listed IP and port'
            assert denied(lambda: tcp('example.com', 443)), 'host name in connect()'
            """,
        )
    )


def test_ip_from_allowed_lookup_is_allowed_and_others_are_not(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            fake = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('10.255.255.3', 443))]
            guard._real_getaddrinfo = lambda *a, **k: fake
            socket.getaddrinfo('llm.corp.example', 443)
            assert not denied(lambda: tcp('10.255.255.3', 443))
            assert denied(lambda: tcp('10.255.255.4', 443))
            """,
        )
    )


def test_loopback_and_unix_sockets_are_allowed(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1)
            tcp('127.0.0.1', srv.getsockname()[1])
            socket.getaddrinfo('localhost', 80)
            a, b = socket.socketpair(); a.close(); b.close()
            """,
        )
    )


def test_bind_to_all_interfaces_is_denied(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            assert denied(lambda: socket.socket().bind(('0.0.0.0', 0)))
            assert denied(lambda: socket.socket().bind(('', 0)))
            """,
        )
    )


def test_subprocess_rules(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            subprocess.run(['true'], check=True)
            import ctypes.util; ctypes.util.find_library('c')  # runs ldconfig -p
            assert not denied(lambda: subprocess.run(['ldconfig', '-p'], capture_output=True))
            assert denied(lambda: subprocess.run(['git', '--version']))
            assert denied(lambda: subprocess.run('true', shell=True))
            assert denied(lambda: subprocess.run([sys.executable, '-m', 'pip', '--version']))
            assert denied(lambda: subprocess.run([sys.executable, '-c', 'print(1)']))
            assert denied(lambda: __import__('os').system('true'))
            subprocess.run([sys.executable, '-m', 'cognee_egress.policy'], check=True)
            """,
        )
    )


PROBE = """
import socket
import cognee_egress

def probe(q):
    cognee_egress.install_from_env()  # what cognee / cognee_db_workers do on import
    try:
        socket.getaddrinfo('example.com', 443)
        q.put('reached')
    except cognee_egress.EgressDenied:
        q.put('denied')
"""


def test_python_child_inherits_the_guard(tmp_path):
    (tmp_path / "egress_probe.py").write_text(PROBE)
    assert_ok(
        run_child(
            tmp_path,
            """
            import multiprocessing as mp
            from egress_probe import probe
            ctx = mp.get_context('spawn'); q = ctx.Queue()
            p = ctx.Process(target=probe, args=(q,)); p.start(); p.join(30)
            assert q.get(timeout=5) == 'denied'
            """,
            env_extra={"PYTHONPATH": f"{REPO}{os.pathsep}{tmp_path}"},
        )
    )


def test_download_only_and_proxy_tunnels(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            g = guard._guard
            class Conn:
                host, port = 'files.mirror.corp.example', 443
            g.on_http_send((Conn(), b'GET /x HTTP/1.1\\r\\nHost: x\\r\\n\\r\\n'))
            assert denied(lambda: g.on_http_send((Conn(), b'POST /x HTTP/1.1\\r\\n\\r\\n')))
            assert denied(lambda: g.on_http_send((Conn(), b'CONNECT evil.example:443 HTTP/1.1\\r\\n\\r\\n')))
            g.on_http_send((Conn(), b'CONNECT llm.corp.example:443 HTTP/1.1\\r\\n\\r\\n'))
            g.on_http_send((Conn(), b'some body bytes'))
            assert denied(lambda: cognee_egress.check_url('https://files.mirror.corp.example/a', method='PUT'))
            """,
        )
    )


def test_check_url_roles_and_hosts(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            cognee_egress.check_url('https://llm.corp.example/v1', role='llm')
            cognee_egress.check_url('http://localhost:11434', role='llm')
            assert denied(lambda: cognee_egress.check_url('https://api.openai.com/v1', role='llm'))
            assert denied(lambda: cognee_egress.check_url('https://llm.corp.example/v1', role='decider'))
            assert denied(lambda: cognee_egress.check_url('http://llm.corp.example/v1', role='llm'))
            """,
        )
    )


def test_record_mode_logs_and_does_not_block(tmp_path):
    log = tmp_path / "egress.jsonl"
    policy = dict(POLICY, mode="record", log=str(log))
    assert_ok(
        run_child(
            tmp_path,
            "assert not denied(lambda: socket.getaddrinfo('nonexistent.invalid', 443))",
            policy,
        )
    )
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(
        r["decision"] == "deny" and r["dest"].startswith("nonexistent.invalid") for r in rows
    )


def test_outside_proxy_variable_is_refused(tmp_path):
    proc = run_child(tmp_path, "pass", env_extra={"HTTPS_PROXY": "http://proxy.corp.example:8080"})
    assert proc.returncode != 0 and "proxy" in proc.stderr
    assert_ok(run_child(tmp_path, "pass", env_extra={"HTTPS_PROXY": "http://127.0.0.1:3128"}))


def test_policy_cannot_be_switched(tmp_path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps(dict(POLICY, mode="record")))
    assert_ok(
        run_child(
            tmp_path,
            f"""
            import os
            os.environ['COGNEE_EGRESS_POLICY'] = {str(other)!r}
            try:
                cognee_egress.install_from_env(); raise SystemExit('switched')
            except cognee_egress.PolicyError:
                pass
            """,
        )
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"runtime": {"hosts": [{"host": "*.corp.example"}]}},
        {"runtime": {"hosts": [{"host": "https://x.example/"}]}},
        {"runtime": {"hosts": [{"host": "x.example", "ports": [0]}]}},
        {"runtime": {"subprocesses": ["/usr/bin/git"]}},
        {"mode": "off", "runtime": {}},
        {"install": {}},
    ],
)
def test_malformed_policies_are_rejected(tmp_path, bad):
    sys.path.insert(0, str(REPO))
    from cognee_egress import PolicyError, load_policy

    p = tmp_path / "p.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(PolicyError):
        load_policy(p)


def _userns_available() -> bool:
    import shutil

    if shutil.which("unshare") is None:
        return False
    return (
        subprocess.run(["unshare", "-rn", "true"], capture_output=True, check=False).returncode == 0
    )


@pytest.mark.skipif(
    not _userns_available(), reason="needs unprivileged user namespaces (unshare -rn)"
)
def test_isolated_runner_has_no_direct_route_and_gateway_refuses_unlisted_hosts(tmp_path):
    log = tmp_path / "egress.jsonl"
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps(dict(POLICY, log=str(log))))
    probe = textwrap.dedent(
        """
        import os, socket, http.client
        assert [n for _, n in socket.if_nameindex()] == ['lo']
        try:
            socket.create_connection(('192.0.2.1', 443), 2)
            raise SystemExit('direct route exists')
        except OSError:
            pass
        c = http.client.HTTPConnection('127.0.0.1', 3128, timeout=10)
        c.request('CONNECT', 'example.com:443')
        assert c.getresponse().status == 403
        c = http.client.HTTPConnection('127.0.0.1', 3128, timeout=10)
        c.request('GET', 'http://example.com/')
        assert c.getresponse().status == 405
        assert os.environ['COGNEE_EGRESS_POLICY'] == os.environ['EXPECT_POLICY']
        """
    )
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    env.update(PYTHONPATH=str(REPO), EXPECT_POLICY=str(pol))
    proc = subprocess.run(  # noqa: PLW1510 - returncode asserted below
        [
            sys.executable,
            "-m",
            "cognee_egress.isolate",
            "--policy",
            str(pol),
            "--",
            sys.executable,
            "-c",
            probe,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(r["decision"] == "deny" and r["dest"] == "example.com:443" for r in rows)


def test_hardened_environment_is_forced_and_reapplied(tmp_path):
    assert_ok(
        run_child(
            tmp_path,
            """
            import os
            assert os.environ['TELEMETRY_DISABLED'] == '1'
            assert os.environ['GIT_ALLOW_PROTOCOL'] == 'file'
            os.environ['GLINER_AUTO_INSTALL'] = 'true'   # e.g. from a .env file
            cognee_egress.install_from_env()
            assert os.environ['GLINER_AUTO_INSTALL'] == 'false'
            """,
            env_extra={"TELEMETRY_DISABLED": "", "GLINER_AUTO_INSTALL": "true"},
        )
    )
