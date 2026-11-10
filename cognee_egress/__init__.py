"""Egress control for cognee: one policy file decides which hosts, ports and programs the process may
use. Standard library only, so it can load before anything else (and in cognee_db_workers children).

Three levels, from lightest to strongest:

1. ``check_url(url, role)`` at configuration time: a configured endpoint outside the policy fails at
   startup.
2. The in-process guard (``install_from_env``), loaded by ``import cognee`` when
   ``COGNEE_EGRESS_POLICY`` is set: every Python-level DNS lookup, connection, bind, subprocess and
   http.client request is checked.
3. ``python -m cognee_egress.isolate``: the program runs in a network namespace with only loopback;
   its single way out is an allowlisting proxy, which also covers native code and child programs.

See docs/egress.md.
"""

from .guard import EgressDenied, active_policy, check_url, install, install_from_env
from .policy import EgressPolicy, PolicyError, load_policy

__all__ = [
    "EgressDenied",
    "EgressPolicy",
    "PolicyError",
    "active_policy",
    "check_url",
    "install",
    "install_from_env",
    "load_policy",
]
