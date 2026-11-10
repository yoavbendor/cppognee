# Egress control: running cognee on a corporate host

One JSON policy file lists the hosts, ports and programs cognee may use. Everything else is refused.
`cognee_egress` (standard library only) enforces it at three levels; pick the strongest your host
allows.

| level | how | stops | does not stop |
|---|---|---|---|
| 1. endpoint check | automatic when a policy is set | a misconfigured LLM, embedding or decider endpoint, including the silent OpenAI defaults: fails at startup | anything else |
| 2. in-process guard | `COGNEE_EGRESS_POLICY=policy.json` | every Python-level DNS lookup, connection, port bind, subprocess start and http.client request, whichever library makes it (litellm, openai, httpx, requests, aiohttp, boto3, huggingface_hub, telemetry...) | native code that opens its own sockets, and programs you allow to start |
| 3. isolated runner | `python -m cognee_egress.isolate --policy policy.json -- <command>` | everything: the command runs in a network namespace with only loopback; its one way out is an allowlisting HTTPS proxy | traffic to the hosts you allow |

Level 3 needs unprivileged user namespaces (`unshare -rn true && echo ok`), no root and no Docker.
It turns level 2 on inside as well. Use level 3 whenever the host allows it; level 2 alone is
protection against cognee's defaults and misconfiguration, not against a compromised dependency.

## The policy file

Start from `docs/egress.example.json`:

```json
{
  "mode": "enforce",
  "log": "egress.jsonl",
  "runtime": {
    "hosts": [
      {"host": "llm.corp.example", "ports": [443], "role": "llm"},
      {"host": ".mirror.corp.example", "ports": [443], "role": "download", "download_only": true}
    ],
    "subprocesses": [],
    "listen": "loopback"
  },
  "install": {"hosts": [{"host": "gitlab.corp.example", "ports": [443], "role": "download"}]}
}
```

- `host`: an exact name, `.domain` for that domain and its subdomains, or an IP address. No
  wildcards or URLs. Loopback (`localhost`, `127.0.0.1`, `::1`) and Unix sockets are always allowed.
- `role`: what the host is for. Configured endpoints must match: the LLM endpoint must be on an
  `llm` host, embeddings on `embedding`, docmem's decider on `decider`. Other roles (`download`,
  `auth`, `db`, `proxy`, ...) are free labels.
- `download_only`: only GET and HEAD to that host, checked for clients built on http.client
  (requests, urllib, urllib3). Under level 3 every connection is end-to-end TLS, which no proxy can
  look into without breaking it, so treat this as a second line: the first is to allow only mirrors
  you trust, such as an Artifactory remote repository in front of PyPI or Hugging Face.
- `subprocesses`: executable basenames cognee may start (for example `enola` for code graphs).
  `ldconfig` is always allowed: it only reads the local library cache (Python's `ctypes` uses it).
  Python children are allowed only for multiprocessing and cognee's own modules, which load the
  guard again. A program you list here runs outside level 2: list networking programs (`git`,
  `curl`, `uv`) only under level 3.
- `listen`: `loopback` (default) refuses binding a server to any other address.
- `mode`: `enforce` refuses; `record` allows everything and writes each decision to `log`. Use
  `record` once to learn what a workflow needs, then write the allowlist from the log.
- `runtime` is what cognee runs with. `install` is used only by
  `python -m cognee_egress.isolate --phase install -- <setup command>`: this is how hosts such as
  GitHub or GitLab are allowed while installing and never afterwards.

## What a policy also switches off

With a policy, these are forced on every start, after the `.env` file is loaded as well:
`TELEMETRY_DISABLED=1`, `DO_NOT_TRACK=1`, `HF_HUB_DISABLE_TELEMETRY=1`,
`HF_HUB_DISABLE_XET=1` (Hugging Face downloads go through Python, where the guard sees them),
`LITELLM_LOCAL_MODEL_COST_MAP=True` (no price list fetched from GitHub at import),
`LITELLM_TELEMETRY=False`, `RUNTIME__DLTHUB_TELEMETRY=false`, `ANONYMIZED_TELEMETRY=False`,
`GLINER_AUTO_INSTALL=false` (no pip at runtime), `ENOLA_NO_UPDATE_CHECK=1`, and at runtime
`GIT_ALLOW_PROTOCOL=file` (git may read local repositories only).

A policy cannot be replaced or removed after it is installed in a process, by a `.env` file or by
code, and an HTTP(S) proxy variable that points outside the machine is refused unless that proxy is
listed with role `proxy`, because a proxy hides the real destination from the guard.

## What still needs a decision per feature

| feature | what happens under a policy | to use it |
|---|---|---|
| keyless embeddings (fastembed) and Hugging Face tokenizers | model download from huggingface.co is refused | point `HF_ENDPOINT` at a mirror listed with role `download`, or pre-download the models during install |
| Ladybug JSON extension | its download (native code) is refused unless `extension.ladybugdb.com` is listed | install it during setup (`tools/work/setup.sh` does), or bundle the binaries |
| URL ingestion, web scraping, Tavily | refused unless the site is listed | list the internal sites you ingest from |
| code graphs | `enola` must be listed in `subprocesses`; git clones of URLs are refused at runtime | clone during install; ingest the local path |
| graph visualization page | the page loads d3 and fonts from public CDNs in your browser | not covered by this guard (it is your browser) |
| API server | binds 0.0.0.0 by default, refused with `listen: loopback` | bind it to 127.0.0.1 |

## Checking a policy

```sh
python -m cognee_egress.isolate --policy policy.json -- python -c "import cognee"
COGNEE_EGRESS_POLICY=policy.json pytest cognee/tests/unit
```

Every refusal raises `cognee_egress.EgressDenied` (a `PermissionError`) naming the policy file, the
destination and the reason, and is logged to `log` when one is set.
