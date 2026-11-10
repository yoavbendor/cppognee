"""Run a program with no network except an allowlisting proxy (egress level 3). No root, no Docker.

    python -m cognee_egress.isolate --policy policy.json -- pytest cognee/tests/unit
    python -m cognee_egress.isolate --policy policy.json --phase install -- bash setup.sh

How it works:

- ``unshare -rn`` gives the program its own network namespace, in which only the loopback interface
  exists. It needs unprivileged user namespaces (check: ``unshare -rn true && echo ok``). Nothing,
  native code and child programs included, can reach any address directly.
- Outside the namespace, ``cognee_egress.gateway`` serves CONNECT for the hosts of the chosen policy
  phase, on a Unix socket in a private temporary directory.
- Inside, a relay listens on 127.0.0.1:3128 and forwards to that socket; HTTP(S)_PROXY point at it.
  Programs that ignore the proxy variables simply fail to connect.
- Unless ``--without-guard`` is given, ``COGNEE_EGRESS_POLICY`` is also set inside, so Python code
  that loads cognee gets the in-process guard (level 2) as well.

The program runs as root of its own user namespace (your user outside it): files it writes are yours.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

from .gateway import splice
from .policy import load_policy

RELAY_PORT = 3128
PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


class IsolationError(RuntimeError):
    pass


def _bring_up_loopback() -> None:
    SIOCGIFFLAGS, SIOCSIFFLAGS, IFF_UP = 0x8913, 0x8914, 0x1
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        req = struct.pack("16sH22x", b"lo", 0)
        flags = struct.unpack("16sH22x", fcntl.ioctl(s, SIOCGIFFLAGS, req))[1]
        fcntl.ioctl(s, SIOCSIFFLAGS, struct.pack("16sH22x", b"lo", flags | IFF_UP))


def _check_isolated() -> None:
    names = [n for _, n in socket.if_nameindex()]
    if names != ["lo"]:
        raise IsolationError(f"not in an isolated network namespace: interfaces {names}")


def _start_relay(sock_path: str, port: int) -> None:
    async def handle(reader, writer):
        try:
            up_r, up_w = await asyncio.open_unix_connection(sock_path)
        except OSError:
            writer.close()
            return
        await splice(reader, writer, up_r, up_w)

    ready = threading.Event()

    def run():
        async def main():
            server = await asyncio.start_server(handle, "127.0.0.1", port)
            ready.set()
            async with server:
                await server.serve_forever()

        asyncio.run(main())

    threading.Thread(target=run, daemon=True, name="egress-relay").start()
    if not ready.wait(10):
        raise IsolationError("relay did not start")


def inside(a) -> int:
    _bring_up_loopback()
    _check_isolated()
    _start_relay(a.socket, RELAY_PORT)
    env = dict(os.environ)
    proxy = f"http://127.0.0.1:{RELAY_PORT}"
    env.update({v: proxy for v in PROXY_VARS})
    env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,::1"
    if a.with_guard:
        env["COGNEE_EGRESS_POLICY"] = a.policy
        env["COGNEE_EGRESS_PHASE"] = a.phase
    else:
        env.pop("COGNEE_EGRESS_POLICY", None)
        env.pop("COGNEE_EGRESS_PHASE", None)
    return subprocess.run(a.command, env=env, check=False).returncode


def outside(a) -> int:
    unshare = shutil.which("unshare")
    if unshare is None:
        raise IsolationError("the 'unshare' program (util-linux) is required")
    load_policy(a.policy)  # fail fast on a bad policy, before starting anything
    tmp = tempfile.mkdtemp(prefix="cognee-egress-")
    os.chmod(tmp, 0o700)
    sock = os.path.join(tmp, "gateway.sock")
    gw = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cognee_egress.gateway",
            "--policy",
            a.policy,
            "--phase",
            a.phase,
            "--socket",
            sock,
        ]
    )
    try:
        deadline = time.monotonic() + 15
        while not os.path.exists(sock):
            if gw.poll() is not None or time.monotonic() > deadline:
                raise IsolationError("egress gateway failed to start")
            time.sleep(0.05)
        cmd = [
            unshare,
            "-rn",
            sys.executable,
            "-m",
            "cognee_egress.isolate",
            "--inside",
            "--socket",
            sock,
            "--policy",
            a.policy,
            "--phase",
            a.phase,
        ]
        if not a.with_guard:
            cmd.append("--without-guard")
        return subprocess.run([*cmd, "--", *a.command], check=False).returncode
    finally:
        gw.terminate()
        gw.wait(10)
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--policy",
        default=os.environ.get("COGNEE_EGRESS_POLICY"),
        help="policy file (default: $COGNEE_EGRESS_POLICY)",
    )
    ap.add_argument("--phase", default="runtime", choices=("runtime", "install"))
    ap.add_argument(
        "--without-guard",
        dest="with_guard",
        action="store_false",
        help="do not also enable the in-process guard inside",
    )
    ap.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--socket", help=argparse.SUPPRESS)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.command[:1] == ["--"]:
        a.command = a.command[1:]
    if not a.command:
        ap.error("give the command to run after --")
    if not a.policy:
        ap.error("--policy or COGNEE_EGRESS_POLICY is required")
    a.policy = os.path.abspath(a.policy)
    return inside(a) if a.inside else outside(a)


if __name__ == "__main__":
    sys.exit(main())
