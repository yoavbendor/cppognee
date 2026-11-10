"""Allowlisting HTTPS proxy for the isolated runner. Runs OUTSIDE the network namespace and listens only
on a Unix socket, which the isolated program reaches through a loopback relay.

Only ``CONNECT host:port`` is served: the host must match the policy phase, the port must be allowed,
and the resolved address must not be loopback, link-local (cloud metadata), multicast or unspecified.
The gateway connects to the address it checked, without resolving again. Plain-HTTP proxy requests are
refused, so every allowed connection is TLS end to end. The gateway never sees inside TLS, so a
host's ``download_only`` flag cannot be enforced here (only the in-process guard checks HTTP methods).

    python -m cognee_egress.gateway --policy policy.json --phase runtime --socket /path/gw.sock
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import sys
import time

from .policy import EgressPolicy, Phase, load_policy

MAX_HEAD = 65536  # a request head larger than this is answered with 431


def _bad_address(ip: str) -> bool:
    a = ipaddress.ip_address(ip)
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped:
        a = a.ipv4_mapped
    return a.is_loopback or a.is_link_local or a.is_multicast or a.is_unspecified or a.is_reserved


def _split_target(target: str) -> tuple[str, int] | None:
    if target.startswith("["):
        host, _, rest = target[1:].partition("]")
        port = rest[1:] if rest.startswith(":") else ""
    else:
        host, _, port = target.rpartition(":")
    if not host or not port.isdigit():
        return None
    return host.lower().rstrip("."), int(port)


class Gateway:
    def __init__(self, policy: EgressPolicy, phase: str):
        self.policy = policy
        self.phase_name = phase
        self.phase: Phase = policy.phase(phase)
        self.log = open(policy.log, "a", encoding="utf-8", buffering=1) if policy.log else None  # noqa: SIM115 - process-lifetime log

    def record(self, dest: str, ok: bool, reason: str = "") -> None:
        if self.log is None:
            return
        rec = {
            "ts": round(time.time(), 3),
            "pid": os.getpid(),
            "mode": "gateway",
            "phase": self.phase_name,
            "decision": "allow" if ok else "deny",
            "kind": "gateway",
            "dest": dest,
            "reason": reason,
            "origin": "",
        }
        try:
            self.log.write(json.dumps(rec) + "\n")
        except (OSError, ValueError):
            pass  # best-effort log

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self._handle(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    async def _reply(self, writer, status: str) -> None:
        writer.write(
            f"HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode()
        )
        await writer.drain()

    async def _handle(self, reader, writer) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except asyncio.LimitOverrunError:
            await self._reply(writer, "431 Request Header Fields Too Large")
            return
        line = head.split(b"\r\n", 1)[0].decode("latin-1")
        parts = line.split(" ")
        if len(parts) != 3 or parts[0] != "CONNECT":
            self.record(line[:200], False, "only CONNECT is served")
            await self._reply(writer, "405 Method Not Allowed")
            return
        target = _split_target(parts[1])
        if target is None:
            await self._reply(writer, "400 Bad Request")
            return
        host, port = target
        rule = self.phase.rule_for(host)
        if rule is None or not rule.allows_port(port):
            self.record(
                f"{host}:{port}",
                False,
                "host not in policy" if rule is None else "port not allowed",
            )
            await self._reply(writer, "403 Forbidden")
            return
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, port, type=1)  # SOCK_STREAM
        except OSError as e:
            self.record(f"{host}:{port}", False, f"resolve failed: {e}")
            await self._reply(writer, "502 Bad Gateway")
            return
        addrs = [i[4][0] for i in infos if not _bad_address(i[4][0])]
        if not addrs:
            self.record(
                f"{host}:{port}", False, "resolves only to loopback/link-local/reserved addresses"
            )
            await self._reply(writer, "403 Forbidden")
            return
        try:
            up_r, up_w = await asyncio.open_connection(addrs[0], port)
        except OSError as e:
            self.record(f"{host}:{port}", False, f"connect failed: {e}")
            await self._reply(writer, "502 Bad Gateway")
            return
        self.record(f"{host}:{port}", True)
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        await splice(reader, writer, up_r, up_w)


async def _pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
    try:
        while data := await src.read(65536):
            dst.write(data)
            await dst.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            dst.write_eof() if dst.can_write_eof() else dst.close()
        except (OSError, RuntimeError):
            pass


async def splice(a_r, a_w, b_r, b_w) -> None:
    """Copy bytes both ways until both directions are closed."""
    await asyncio.gather(_pipe(a_r, b_w), _pipe(b_r, a_w))
    b_w.close()


async def serve(policy: EgressPolicy, phase: str, sock_path: str) -> None:
    gw = Gateway(policy, phase)
    server = await asyncio.start_unix_server(gw.handle, path=sock_path, limit=MAX_HEAD)
    os.chmod(sock_path, 0o600)
    async with server:
        await server.serve_forever()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--policy", required=True)
    ap.add_argument("--phase", default="runtime", choices=("runtime", "install"))
    ap.add_argument("--socket", required=True)
    a = ap.parse_args(argv)
    policy = load_policy(a.policy)
    try:
        asyncio.run(serve(policy, a.phase, a.socket))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
