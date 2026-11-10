"""In-process egress guard: a Python audit hook that checks every DNS lookup, connection, bind,
subprocess start and http.client request against the egress policy.

What it covers: all Python-level network and process APIs, whichever library calls them (requests,
httpx, aiohttp, litellm, openai, boto3, huggingface_hub, posthog, asyncio...). Once installed it cannot
be removed (``sys.addaudithook``).

What it cannot cover: native code that opens sockets or resolves names itself (Rust/C++ HTTP clients in
extension modules) and child processes that do not load this guard. For those, run the application
under ``python -m cognee_egress.isolate``, which adds an OS-level network boundary.

A connection to an IP address is allowed only if that address came from a lookup of an allowed host
name in this process (or the IP itself is listed), so resolving a host some other way and connecting
to its address is refused.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import sys
import threading
import time
from urllib.parse import urlsplit

from .policy import EgressPolicy, HostRule, Phase, PolicyError, is_loopback, load_policy

PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
SERVICE_PORTS = {"http": 80, "https": 443}
PY_CHILD_MODULES = ("cognee", "cognee_db_workers", "cognee_egress", "docmem")
HTTP_METHODS = (
    b"GET",
    b"HEAD",
    b"POST",
    b"PUT",
    b"PATCH",
    b"DELETE",
    b"OPTIONS",
    b"CONNECT",
    b"TRACE",
)


class EgressDenied(PermissionError):
    """A network or process operation was refused by the egress policy."""


class _Guard:
    def __init__(self, policy: EgressPolicy, phase: str):
        self.policy = policy
        self.phase_name = phase
        self.phase: Phase = policy.phase(phase)
        self.enforce = policy.mode == "enforce"
        self.resolved: dict[str, HostRule] = {}  # IP -> rule of the allowed host it came from
        self.seen: set[tuple] = set()
        self.lock = threading.Lock()
        self.log_file = None
        if policy.log:
            self.log_file = open(policy.log, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - process-lifetime log

    # ----------------------------------------------------------------------------------- decisions

    def decide(self, kind: str, dest: str, ok: bool, reason: str = "") -> None:
        key = (kind, dest, ok)
        if ok and key in self.seen:
            return
        self.seen.add(key)
        self.write(kind, dest, ok, reason)
        if not ok and self.enforce:
            raise EgressDenied(
                f"egress policy {self.policy.path} ({self.phase_name}) refuses {kind} {dest}"
                + (f": {reason}" if reason else "")
            )

    def write(self, kind: str, dest: str, ok: bool, reason: str) -> None:
        if self.log_file is None:
            return
        rec = {
            "ts": round(time.time(), 3),
            "pid": os.getpid(),
            "mode": self.policy.mode,
            "phase": self.phase_name,
            "decision": "allow" if ok else "deny",
            "kind": kind,
            "dest": dest,
            "reason": reason,
            "origin": _origin(),
        }
        try:
            with self.lock:
                self.log_file.write(json.dumps(rec) + "\n")
        except (OSError, ValueError):
            pass  # logging is best-effort; the decision above stands

    def host_ok(self, kind: str, host: str, port: int | None) -> HostRule | None:
        """Decide on a host name or IP literal. Returns the matching rule (None for loopback)."""
        if is_loopback(host):
            return None
        ip = _as_ip(host)
        rule = self.resolved.get(ip) if ip else None
        if rule is None:
            rule = self.phase.rule_for(ip or host)
        if rule is None:
            why = "address not obtained from an allowed host" if ip else "host not in policy"
            self.decide(kind, _hp(host, port), False, why)
            return None
        if not rule.allows_port(port):
            self.decide(kind, _hp(host, port), False, f"port not in {list(rule.ports)}")
            return rule
        self.decide(kind, _hp(host, port), True)
        return rule

    def method_ok(self, kind: str, host: str, port: int | None, method: str) -> None:
        if is_loopback(host):
            return
        ip = _as_ip(host)
        rule = (self.resolved.get(ip) if ip else None) or self.phase.rule_for(ip or host)
        if rule is not None and rule.download_only and method not in ("GET", "HEAD"):
            self.decide(kind, f"{method} {_hp(host, port)}", False, "host is download-only")

    # ------------------------------------------------------------------------------- audit events

    def on_getaddrinfo(self, args) -> None:
        host, port = args[0], args[1]
        if host is None:
            return
        if isinstance(host, bytes):
            host = host.decode("idna", "replace")
        if not host:
            return
        self.host_ok("dns", host, _port(port))

    def on_connect(self, args, kind: str = "connect") -> None:
        sock, address = args[0], args[1]
        if getattr(sock, "family", None) not in (socket.AF_INET, socket.AF_INET6):
            return  # Unix sockets and other local families
        if not isinstance(address, tuple) or len(address) < 2:
            return
        host = address[0].decode() if isinstance(address[0], bytes) else str(address[0])
        self.host_ok(kind, host, _port(address[1]))

    def on_bind(self, args) -> None:
        sock, address = args[0], args[1]
        if self.phase.listen == "any" or getattr(sock, "family", None) not in (
            socket.AF_INET,
            socket.AF_INET6,
        ):
            return
        host = str(address[0]) if isinstance(address, tuple) and address else ""
        if not is_loopback(host):
            self.decide(
                "listen",
                _hp(host or "0.0.0.0", address[1] if isinstance(address, tuple) else None),
                False,
                "only loopback addresses may be bound (runtime.listen)",
            )

    def on_process(self, program, argv) -> None:
        program = os.fsdecode(program) if program is not None else None
        argv = (
            [os.fsdecode(a) for a in argv]
            if isinstance(argv, (list, tuple))
            else [os.fsdecode(argv)]
        )
        if program is None:
            program = argv[0] if argv else ""
        name = os.path.basename(program)
        if _is_python(program) and _python_child_ok(argv):
            self.decide("exec", name, True)
            return
        ok = name in self.phase.subprocesses
        self.decide(
            "exec",
            " ".join([name, *argv[1:3]])[:200],
            ok,
            "" if ok else "executable not in subprocesses",
        )

    def on_http_send(self, args) -> None:
        conn, data = args[0], args[1]
        if not isinstance(data, (bytes, bytearray)):
            return
        head = bytes(data[:2048])
        sp = head.find(b" ")
        if sp <= 0 or head[:sp] not in HTTP_METHODS:
            return  # a body chunk, not a request line
        method = head[:sp].decode()
        target = head[sp + 1 :].split(b" ", 1)[0].decode("latin-1")
        if method == "CONNECT":  # tunnel through a proxy: check the real destination
            host, _, port = target.rpartition(":")
            self.host_ok("connect-via-proxy", host.strip("[]"), _port(port))
            return
        if "://" in target:  # absolute form: a request through a plain HTTP proxy
            u = urlsplit(target)
            if u.hostname:
                self.host_ok("http-via-proxy", u.hostname, u.port or SERVICE_PORTS.get(u.scheme))
                self.method_ok("http", u.hostname, u.port, method)
            return
        self.method_ok("http", str(getattr(conn, "host", "")), getattr(conn, "port", None), method)

    def on_urllib_request(self, args) -> None:
        url, data, method = args[0], args[1], args[3]
        u = urlsplit(url)
        if u.hostname:
            self.method_ok(
                "http",
                u.hostname,
                u.port,
                (method or ("POST" if data is not None else "GET")).upper(),
            )

    def hook(self, event: str, args) -> None:
        handler = self.handlers.get(event)
        if handler is not None:
            handler(args)

    @property
    def handlers(self):
        h = self.__dict__.get("_handlers")
        if h is None:
            h = self.__dict__["_handlers"] = {
                "socket.getaddrinfo": self.on_getaddrinfo,
                "socket.connect": self.on_connect,
                "socket.sendto": lambda a: self.on_connect(a, "sendto"),
                "socket.sendmsg": lambda a: a[1] is not None and self.on_connect(a, "sendto"),
                "socket.bind": self.on_bind,
                "subprocess.Popen": lambda a: self.on_process(a[0], a[1]),
                "os.system": lambda a: self.on_process("/bin/sh", ["sh", "-c", os.fsdecode(a[0])]),
                "os.exec": lambda a: self.on_process(a[0], a[1]),
                "os.posix_spawn": lambda a: self.on_process(a[0], a[1]),
                "os.spawn": lambda a: self.on_process(a[1], a[2]),
                "os.startfile": lambda a: self.on_process(a[0], [a[0]]),
                "http.client.send": self.on_http_send,
                "urllib.Request": self.on_urllib_request,
            }
        return h


# --------------------------------------------------------------------------------------- helpers


def _hp(host: str, port) -> str:
    return (
        f"[{host}]:{port}"
        if ":" in host and port is not None
        else (f"{host}:{port}" if port is not None else host)
    )


def _port(port) -> int | None:
    if isinstance(port, int):
        return port
    if isinstance(port, bytes):
        port = port.decode()
    if isinstance(port, str):
        return int(port) if port.isdigit() else SERVICE_PORTS.get(port.lower())
    return None


def _as_ip(host: str) -> str | None:
    try:
        ip = ipaddress.ip_address(host.strip("[]").split("%")[0])
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return str(ip)


def _is_python(program: str) -> bool:
    if not program:
        return False
    try:
        if os.path.realpath(program) == os.path.realpath(sys.executable):
            return True
    except OSError:
        pass
    return os.path.basename(program).startswith("python")


def _python_child_ok(argv: list[str]) -> bool:
    """A Python child may start only to run multiprocessing machinery or one of our own modules,
    which load this guard again from the inherited COGNEE_EGRESS_POLICY."""
    i = 1
    while i < len(argv) and argv[i].startswith("-") and argv[i] not in ("-c", "-m"):
        i += 1
    if i + 1 >= len(argv):
        return False
    flag, what = argv[i], argv[i + 1]
    if flag == "-c":
        return what.lstrip().startswith(("from multiprocessing", "import multiprocessing"))
    if flag == "-m":
        return what.split(".")[0] in PY_CHILD_MODULES
    return False


def _origin() -> str:
    """file:line of the innermost caller outside this package and the standard library."""
    stdlib = os.path.dirname(os.__file__)
    f = sys._getframe(1)
    here = os.path.dirname(__file__)
    while f is not None:
        fn = f.f_code.co_filename
        if not fn.startswith((here, stdlib, "<")):
            return f"{fn}:{f.f_lineno}"
        f = f.f_back
    return ""


# ---------------------------------------------------------------------------------- installation

_guard: _Guard | None = None
_real_getaddrinfo = socket.getaddrinfo


def _getaddrinfo(host, port, *args, **kwargs):
    infos = _real_getaddrinfo(host, port, *args, **kwargs)  # the audit hook checks the host here
    g = _guard
    if g is not None and host:
        h = host.decode("idna", "replace") if isinstance(host, bytes) else str(host)
        rule = g.phase.rule_for(_as_ip(h) or h)
        if rule is not None:
            for info in infos:
                ip = _as_ip(str(info[4][0]))
                if ip:
                    g.resolved[ip] = rule
    return infos


def _check_proxy_env(phase: Phase, policy: EgressPolicy) -> None:
    for var in PROXY_VARS:
        value = os.environ.get(var)
        if not value:
            continue
        host = urlsplit(value if "://" in value else f"http://{value}").hostname or ""
        if is_loopback(host):
            continue
        rule = phase.rule_for(host)
        if rule is None or rule.role != "proxy":
            raise PolicyError(
                f"{var}={value} sends traffic through a proxy that this process cannot see past. "
                f"Unset it, or list host {host!r} with role 'proxy' in {policy.path} if that proxy "
                f"enforces the same allowlist."
            )


# Settings forced while a policy is active. They switch off features whose network use is either
# pointless under a policy (telemetry, update checks, runtime installs) or invisible to this guard
# because it happens in native code (hf_xet's Rust downloader, git's own transport).
HARDENED_ENV = {
    "TELEMETRY_DISABLED": "1",  # cognee telemetry
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_XET": "1",  # downloads through Python HTTP, where the guard sees them
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",  # no model price list fetched from GitHub at import
    "LITELLM_TELEMETRY": "False",
    "RUNTIME__DLTHUB_TELEMETRY": "false",
    "ANONYMIZED_TELEMETRY": "False",
    "GLINER_AUTO_INSTALL": "false",  # no pip/uv at runtime
    "ENOLA_NO_UPDATE_CHECK": "1",
}
RUNTIME_ONLY_ENV = {
    "GIT_ALLOW_PROTOCOL": "file",  # git may read local repositories only; clones are an install step
}


def harden_environment(phase: str) -> None:
    """Force HARDENED_ENV (and, at runtime, RUNTIME_ONLY_ENV) into os.environ."""
    os.environ.update(HARDENED_ENV)
    if phase == "runtime":
        os.environ.update(RUNTIME_ONLY_ENV)


def install(policy: EgressPolicy, phase: str = "runtime") -> None:
    """Install the guard for this process. A second call with the same policy and phase is a no-op
    apart from re-applying the hardened environment (a .env file may have been loaded meanwhile)."""
    global _guard
    harden_environment(phase)
    if _guard is not None:
        if (_guard.policy.path, _guard.phase_name) != (policy.path, phase):
            raise PolicyError(
                f"egress guard already installed from {_guard.policy.path} ({_guard.phase_name}); "
                f"refusing to switch to {policy.path} ({phase})"
            )
        return
    _check_proxy_env(policy.phase(phase), policy)
    g = _Guard(policy, phase)
    socket.getaddrinfo = _getaddrinfo
    _guard = g
    sys.addaudithook(g.hook)


def install_from_env() -> EgressPolicy | None:
    """Install from COGNEE_EGRESS_POLICY / COGNEE_EGRESS_PHASE; no policy variable means no guard.
    Also called after a .env file is loaded: the variables must not change once a guard is active."""
    path = os.environ.get("COGNEE_EGRESS_POLICY", "").strip()
    phase = os.environ.get("COGNEE_EGRESS_PHASE", "runtime").strip() or "runtime"
    if not path:
        if _guard is not None:
            raise PolicyError("COGNEE_EGRESS_POLICY was unset after the egress guard was installed")
        return None
    policy = load_policy(path)
    install(policy, phase)
    return policy


def active_policy() -> EgressPolicy | None:
    return _guard.policy if _guard is not None else None


def check_url(url: str, role: str | None = None, method: str = "GET") -> None:
    """Fail fast if a configured endpoint is outside the policy. No guard installed: no check.

    Use at configuration time (LLM endpoint, decider URL, ...) so a wrong endpoint fails at startup
    with a clear message instead of at the first request."""
    g = _guard
    if g is None:
        return
    u = urlsplit(url)
    host = u.hostname
    if not host:
        raise EgressDenied(f"cannot check endpoint without a host: {url!r}")
    if is_loopback(host):
        return
    port = u.port or SERVICE_PORTS.get(u.scheme)
    rule = g.phase.rule_for(_as_ip(host) or host)
    if rule is None:
        g.decide(
            "endpoint", _hp(host, port), False, f"host not in policy (url {u.scheme}://{host})"
        )
        return
    if role is not None and rule.role != role:
        g.decide(
            "endpoint", _hp(host, port), False, f"listed for role {rule.role!r}, used as {role!r}"
        )
    if not rule.allows_port(port):
        g.decide("endpoint", _hp(host, port), False, f"port not in {list(rule.ports)}")
    g.method_ok("endpoint", host, port, method.upper())
