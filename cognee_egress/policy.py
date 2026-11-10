"""Egress policy: which destinations a process may reach, loaded from one JSON file.

The file is named by ``COGNEE_EGRESS_POLICY``. Example::

    {
      "mode": "enforce",                       # enforce | record
      "log": "/var/log/cognee/egress.jsonl",   # optional: one JSON line per decision
      "runtime": {
        "hosts": [
          {"host": "llm.corp.example",          "ports": [443], "role": "llm"},
          {"host": "decider.corp.example",      "ports": [443], "role": "decider"},
          {"host": ".artifactory.corp.example", "ports": [443], "role": "download",
           "download_only": true}
        ],
        "subprocesses": ["git"],              # executable basenames allowed to start
        "listen": "loopback"                   # loopback | any
      },
      "install": {
        "hosts": [{"host": "github.com", "ports": [443], "role": "download"}]
      }
    }

A host starting with "." matches that domain and its subdomains, and nothing else. Loopback and Unix
sockets are always allowed. ``runtime`` applies to the application, ``install`` only to the setup step
when it runs under the isolated runner (``python -m cognee_egress.isolate --phase install``).
"""

from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass, field
from pathlib import Path

MODES = ("enforce", "record")
LISTEN = ("loopback", "any")
PHASES = ("runtime", "install")


class PolicyError(ValueError):
    """The policy file is missing or malformed."""


@dataclass(frozen=True)
class HostRule:
    host: str  # exact host, ".suffix" for a domain, or an IP literal
    ports: tuple[int, ...]  # empty: any port
    role: str
    download_only: bool = False

    def matches(self, host: str) -> bool:
        host = host.lower().rstrip(".")
        if self.host.startswith("."):
            return host == self.host[1:] or host.endswith(self.host)
        return host == self.host

    def allows_port(self, port: int | None) -> bool:
        return not self.ports or port is None or port in self.ports


@dataclass(frozen=True)
class Phase:
    hosts: tuple[HostRule, ...]
    subprocesses: frozenset[str] = frozenset()
    listen: str = "loopback"

    def rule_for(self, host: str) -> HostRule | None:
        """The first rule whose host matches, or None."""
        for rule in self.hosts:
            if rule.matches(host):
                return rule
        return None


@dataclass(frozen=True)
class EgressPolicy:
    path: str
    mode: str
    log: str | None
    runtime: Phase
    install: Phase = field(default_factory=lambda: Phase(hosts=()))

    def phase(self, name: str) -> Phase:
        if name not in PHASES:
            raise PolicyError(f"unknown phase {name!r}; expected one of {PHASES}")
        return self.runtime if name == "runtime" else self.install


def is_loopback(host: str) -> bool:
    """True for 'localhost' and loopback IP literals."""
    host = host.lower().rstrip(".").strip("[]")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def _host_rule(raw: dict, where: str) -> HostRule:
    if not isinstance(raw, dict) or not isinstance(raw.get("host"), str) or not raw["host"].strip():
        raise PolicyError(f"{where}: each host entry needs a non-empty 'host' string")
    host = raw["host"].strip().lower().rstrip(".")
    if "*" in host or "/" in host or ":" in host.strip("[]") and not _is_ip(host):
        raise PolicyError(
            f"{where}: host {raw['host']!r} must be a bare host name, '.domain' or IP"
        )
    if host == ".":
        raise PolicyError(f"{where}: '.' would match every host")
    ports = raw.get("ports", [])
    if not isinstance(ports, list) or not all(isinstance(p, int) and 0 < p < 65536 for p in ports):
        raise PolicyError(f"{where}: 'ports' must be a list of port numbers")
    role = raw.get("role", "other")
    if not isinstance(role, str):
        raise PolicyError(f"{where}: 'role' must be a string")
    unknown = set(raw) - {"host", "ports", "role", "download_only"}
    if unknown:
        raise PolicyError(f"{where}: unknown keys {sorted(unknown)}")
    return HostRule(host.strip("[]"), tuple(ports), role, bool(raw.get("download_only", False)))


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def _phase(raw: dict | None, where: str) -> Phase:
    if raw is None:
        return Phase(hosts=())
    if not isinstance(raw, dict):
        raise PolicyError(f"{where} must be an object")
    unknown = set(raw) - {"hosts", "subprocesses", "listen"}
    if unknown:
        raise PolicyError(f"{where}: unknown keys {sorted(unknown)}")
    hosts = raw.get("hosts", [])
    if not isinstance(hosts, list):
        raise PolicyError(f"{where}.hosts must be a list")
    subs = raw.get("subprocesses", [])
    if not isinstance(subs, list) or not all(isinstance(s, str) and "/" not in s for s in subs):
        raise PolicyError(f"{where}.subprocesses must be a list of executable basenames")
    listen = raw.get("listen", "loopback")
    if listen not in LISTEN:
        raise PolicyError(f"{where}.listen must be one of {LISTEN}")
    rules = tuple(_host_rule(h, f"{where}.hosts[{i}]") for i, h in enumerate(hosts))
    return Phase(hosts=rules, subprocesses=frozenset(subs), listen=listen)


def load_policy(path: str | Path) -> EgressPolicy:
    """Read and validate a policy file. Raises PolicyError on any problem."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PolicyError(f"egress policy file not found: {p}") from None
    except json.JSONDecodeError as e:
        raise PolicyError(f"egress policy {p} is not valid JSON: {e}") from None
    if not isinstance(raw, dict):
        raise PolicyError(f"egress policy {p} must be a JSON object")
    unknown = set(raw) - {"mode", "log", "runtime", "install", "comment"}
    if unknown:
        raise PolicyError(f"egress policy {p}: unknown keys {sorted(unknown)}")
    mode = raw.get("mode", "enforce")
    if mode not in MODES:
        raise PolicyError(f"egress policy {p}: mode must be one of {MODES}")
    log = raw.get("log")
    if log is not None and not isinstance(log, str):
        raise PolicyError(f"egress policy {p}: log must be a path string")
    if "runtime" not in raw:
        raise PolicyError(f"egress policy {p}: a 'runtime' section is required")
    return EgressPolicy(
        path=str(p.resolve()),
        mode=mode,
        log=log,
        runtime=_phase(raw["runtime"], "runtime"),
        install=_phase(raw.get("install"), "install"),
    )
