<div align="center">
  <a href="https://github.com/topoteretes/cognee">
    <img src="assets/cognee-logo.svg" alt="Cognee Logo" width="260">
  </a>

  <br />

  <p>Cognee - The Free Open-Source AI Memory Platform for Agents</p>

  <p align="center">
  <a href="https://www.youtube.com/watch?v=8hmqS2Y5RVQ&t=13s">Demo</a>
  .
  <a href="https://docs.cognee.ai/">Docs</a>
  .
  <a href="https://cognee.ai">Learn More</a>
  ·
  <a href="https://discord.gg/NQPKmU5CCg">Join Discord</a>
  ·
  <a href="https://www.reddit.com/r/AIMemory/">Join r/AIMemory</a>
  .
  <a href="https://github.com/topoteretes/cognee-community">Community Plugins & Add-ons</a>
  </p>


  <p>
  <a href="https://GitHub.com/topoteretes/cognee/network/"><img src="https://img.shields.io/github/forks/topoteretes/cognee.svg?style=social&amp;label=Fork&amp;maxAge=2592000" alt="GitHub forks"></a>
  <a href="https://github.com/topoteretes/cognee"><img src="https://img.shields.io/github/stars/topoteretes/cognee.svg?style=social&amp;label=Star&amp;maxAge=2592000" alt="GitHub stars"></a>
  <a href="https://GitHub.com/topoteretes/cognee/commit/"><img src="https://badgen.net/github/commits/topoteretes/cognee" alt="GitHub commits"></a>
  <a href="https://github.com/topoteretes/cognee/tags/"><img src="https://badgen.net/github/tag/topoteretes/cognee" alt="GitHub tag"></a>
  <a href="https://pepy.tech/project/cognee"><img src="https://static.pepy.tech/badge/cognee" alt="Downloads"></a>
  <a href="https://github.com/topoteretes/cognee/blob/main/LICENSE"><img src="https://img.shields.io/github/license/topoteretes/cognee?colorA=00C586&amp;colorB=000000" alt="License"></a>
  <a href="https://github.com/topoteretes/cognee/graphs/contributors"><img src="https://img.shields.io/github/contributors/topoteretes/cognee?colorA=00C586&amp;colorB=000000" alt="Contributors"></a>
  <a href="https://github.com/sponsors/topoteretes"><img src="https://img.shields.io/badge/Sponsor-❤️-ff69b4.svg" alt="Sponsor"></a>
  </p>

<p>
  <a href="https://trendshift.io/repositories/13955" target="_blank" style="display:inline-block;">
    <img src="https://trendshift.io/api/badge/repositories/13955" alt="topoteretes%2Fcognee | Trendshift" width="250" height="55" />
  </a>
</p>

  <p>Cognee is a free open-source AI memory platform that gives AI agents persistent long-term memory across sessions. Turn documents, code, and conversations into a self-hosted knowledge graph your agents can search and reuse.</p>

  <p><strong>Runs locally for free — no API key required.</strong><br />
  We rely on free small models that use your CPU.</p>

  <p align="center">
  🌐 This README is also available in:<br />
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=de">Deutsch</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=es">Español</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=fr">Français</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=ja">日本語</a> |
  <a href="README_ko.md">한국어</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=pt">Português</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=ru">Русский</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=zh">中文</a>
  </p>

<p align="center">
  <img src="assets/cognee-demo.gif" alt="Cognee Demo" width="80%" />
</p>
</div>

📄 Read the research paper: [Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning](https://arxiv.org/abs/2505.24478) — Markovic et al., 2025

## When to use Cognee

- **Build a Company Brain.** Bring documentation, conversations, tickets, code, and agent work into shared memory. Help your team and agents connect a decision to the discussion and implementation behind it. [Explore Company Brain](https://www.cognee.ai/company-brain).
- **Give agents memory across runs.** Retain project context, past decisions, fixes, and learned rules. Distill useful session lessons into durable knowledge that another session can retrieve. [Connect your agent](#connect-your-agent).
- **Ground agents in your domain.** Structure memory around the entities and relationships your application needs, with custom data models and ontologies. [Explore ontologies](https://docs.cognee.ai/guides/ontology-support).

## Choose your starting point

| I want to… | Start here |
| --- | --- |
| Build memory without an LLM | [Local Python quickstart](#run-locally-without-an-llm) |
| Explore a prebuilt graph without downloading models | [Bundled demo](#explore-the-bundled-demo) |
| Generate answers with a local or hosted LLM | [Optional LLM setup](#optional-configure-the-llm) |
| Give an existing agent memory | [Plugins and MCP](#connect-your-agent) |
| Run Cognee on my infrastructure | [Deployment options](#deploy-cognee) |
| Use a managed service | [Cognee Cloud](https://docs.cognee.ai/cognee-cloud/overview) |
| Keep the graph in nanolance tables | [nanolance graph backend](#nanolance-graph-backend) |
| Run under a corporate egress policy | [Egress control](#egress-control) |
| Index a document folder by byte range | [docmem](apps/docmem/README.md) |

## What this fork adds

This repository is a fork of [topoteretes/cognee](https://github.com/topoteretes/cognee), based on 1.6.6. The sections below cover what it adds. The rest of this README describes upstream Cognee, which this fork does not change unless stated.

- **nanolance graph backend** (`cognee_nanograph`): an alternative graph store in nanolance's Lance tables. Opt-in and Python only. See [nanolance graph backend](#nanolance-graph-backend).
- **Egress control** (`cognee_egress`): one JSON policy that limits the hosts, ports and programs Cognee may use. See [Egress control](#egress-control).
- **docmem** (`apps/docmem`): a document index that keeps vectors and byte-range pointers into the files, never the passage text. See [apps/docmem/README.md](apps/docmem/README.md).
- **Core changes** these depend on: provider endpoints are checked against the egress policy at startup; a graph provider that is not built in uses the dataset handler of the same name; and a field marked `exclude=True` survives `copy_model`.

## Quickstart

Requires **Python 3.10–3.14**.

**1. Install Cognee** with **pip**, **uv**, or your preferred Python package manager. The `gliner` extra brings the local extraction model used when no LLM key is configured:

```bash
uv pip install "cognee[gliner]"
```

To run this fork's code instead of the PyPI release, install from a checkout of this repository: `uv pip install -e ".[gliner]"` in the repository root.

### Run locally without an LLM

**2. Build and query memory.** With no LLM key configured, Cognee extracts the graph with the local GLiNER model and embeds with a local embedding model; both download on first use.

Save this as `quickstart.py` and run `python quickstart.py` if you are feeling old school, or tell your LLM to do it:

```python
import asyncio

import cognee


async def main():
    # Extract a knowledge graph and embed the text with local models.
    await cognee.remember(
        "Marie Curie was born in Warsaw and worked at the University of Paris.",
        dataset_name="local_quickstart",
    )

    # Retrieve the matching source text; no LLM generates an answer.
    results = await cognee.recall(
        "Where was Marie Curie born?",
        datasets=["local_quickstart"],
    )
    for result in results:
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

The same workflow is available from the CLI:

```bash
cognee-cli remember "Marie Curie was born in Warsaw." -d local_quickstart
cognee-cli recall "Where was Marie Curie born?" -d local_quickstart
```

Text ingestion, retrieval, and session storage work without an LLM. LLM-dependent improvement stages skip automatically.

Generated answers and media processing that requires a vision or transcription model need additional LLM configuration.

The bundled GLiNER extractor is a demo of Cognee's small-model pipeline. For a production-ready version with higher accuracy and broader label coverage, [reach out to us](mailto:social@cognee.ai).

### Optional: Configure the LLM

**3. Add an LLM** to get generated answers instead of retrieved passages:

```python
import os

os.environ["LLM_API_KEY"] = "YOUR_OPENAI_API_KEY"
```
Alternatively, create a `.env` file using our [template](https://github.com/topoteretes/cognee/blob/main/.env.template).

Once a key is set, Cognee uses OpenAI for language models and embeddings, and processing and generated answers make provider calls. See [installation](https://docs.cognee.ai/getting-started/installation), [other providers](https://docs.cognee.ai/setup-configuration/llm-providers), or [local Ollama models](https://docs.cognee.ai/guides/local-ollama) for other setups.



### Explore the bundled demo

To explore a prebuilt graph without downloading extraction or embedding models:

```bash
cognee-cli demo
```

This command works with the base `pip install cognee` package. It loads bundled sample data and runs keyword search without an API key. Use the local quickstart above to build a graph from your own text.



## How Cognee works

Cognee builds connected memory from different sources. Text becomes entities, relationships, and searchable chunks; code becomes a graph of symbols and dependencies. Session distillation curates accepted lessons into permanent memory.

<p align="center">
  <img src="assets/remember.svg" alt="Text, code, and session guidance follow their ingestion paths into persistent Cognee memory" width="100%">
</p>

At query time, retrieval selects relevant graph, vector, or code context. Your application can inspect the retrieved evidence and use it to answer a question or continue an agent task.

<p align="center">
  <img src="assets/recall.svg" alt="Recall retrieves a document fact, a code symbol, and a learned release rule for an agent's next task" width="100%">
</p>

| Operation | What it does | Learn more |
| --- | --- | --- |
| `remember` | Store content or code in permanent memory, or in a session when a session ID is supplied. | [Store memory](https://docs.cognee.ai/core-concepts/main-operations/remember) |
| `recall` | Retrieve context and answers, using automatic routing or a chosen search strategy. | [Query memory](https://docs.cognee.ai/core-concepts/main-operations/recall) |
| `improve` | Enrich memory, apply feedback, and bridge session knowledge into the graph. | [Improve memory](https://docs.cognee.ai/core-concepts/main-operations/improve) |
| `forget` | Remove a specific item or dataset. | [Delete memory](https://docs.cognee.ai/core-concepts/main-operations/forget) |

Explore the [architecture](https://docs.cognee.ai/core-concepts/architecture) and [session lifecycle](https://docs.cognee.ai/core-concepts/sessions-and-caching).

## Connect your agent

Install the Claude Code plugin:

```bash
claude plugin marketplace add topoteretes/cognee-integrations
claude plugin install cognee-memory@cognee
```

Or install the Codex plugin. Enable hooks first, either with the CLI:

```bash
codex features enable hooks
```

or in `~/.codex/config.toml`:

```toml
[features]
hooks = true
```

Then add the marketplace and the plugin:

```bash
codex plugin marketplace add topoteretes/cognee-integrations --ref main
codex plugin add cognee@cognee
```

Follow the setup guide for [Claude Code](https://github.com/topoteretes/cognee-integrations/tree/main/integrations/claude-code) or [Codex](https://github.com/topoteretes/cognee-integrations/tree/main/integrations/codex) to configure local or remote memory.

| Interface | Start here |
| --- | --- |
| Claude Code memory plugin | [Install and configure the plugin](https://github.com/topoteretes/cognee-integrations/tree/main/integrations/claude-code) |
| Codex memory plugin | [Install and configure the plugin](https://github.com/topoteretes/cognee-integrations/tree/main/integrations/codex) |
| OpenClaw memory plugin | [Install `@cognee/cognee-openclaw`](https://www.npmjs.com/package/@cognee/cognee-openclaw) |
| Cursor, Cline, and other MCP clients | [Cognee MCP guide](https://docs.cognee.ai/cognee-mcp/mcp-overview) and [server README](cognee-mcp/README.md) |
| Python applications | [Python API reference](https://docs.cognee.ai/python-api) |
| TypeScript applications | [TypeScript SDK](https://docs.cognee.ai/typescript/getting-started) |
| Rust applications | [Cognee-RS](https://github.com/topoteretes/cognee-rs) |
| Applications using HTTP | [REST API reference](https://docs.cognee.ai/api-reference/introduction) |

Browse the [integrations repository](https://github.com/topoteretes/cognee-integrations) for agent frameworks, plugins, and source connectors. Each guide describes its setup and memory capture behavior.

To inspect a local installation in the UI:

```bash
cognee-cli -ui
```

The UI launcher requires Node.js/npm; Docker is needed for its MCP service. See [local UI setup](https://docs.cognee.ai/cognee-cli/overview).

## Explore examples

- [Build a small Company Brain from text, code, and session lessons](examples/cookbooks/company_brain/docs_code_conversations/company_brain_demo.py).
- [Build a Company Brain from your database, ticket export, and docs](examples/cookbooks/company_brain/company_qa/) (or a sample company), then browse it in the UI and query it from Claude Code or Codex.
- [Import memory from Mem0, Letta, Zep, or Graphiti](https://docs.cognee.ai/examples/migrate-memory-systems) using the COGX exchange format.
- [Run with local Ollama models](https://docs.cognee.ai/guides/local-ollama), including a local embedding model.
- [Visualize your knowledge graph](https://docs.cognee.ai/guides/graph-visualization) and inspect its connections.
- [Browse runnable examples](examples/README.md) for ingestion, sessions, feedback, and custom pipelines.
- [Run the prebuilt API with Docker Compose](docs/minimal-docker-compose.md) or use the [deployment templates](distributed/deploy/README.md).
- [Explore community adapters and add-ons](https://github.com/topoteretes/cognee-community).

<a id="run-with-docker"></a>

## Deploy Cognee

For a local API demo using a prebuilt image, follow the [minimal Docker Compose guide](docs/minimal-docker-compose.md). It includes a persistent-volume configuration and explains the single-user demo settings. Or run the image directly:

```bash
docker run --rm -it -p 8000:8000 \
  -e LLM_API_KEY="sk-..." \
  -e ENABLE_BACKEND_ACCESS_CONTROL=false \
  -v cognee_storage:/cognee-storage \
  cognee/cognee:main
```

`ENABLE_BACKEND_ACCESS_CONTROL=false` is the single-user/local posture — without it the API defaults to multi-tenant mode and every `/api/v1` call requires an authenticated user. To keep authentication on instead, set `DEFAULT_USER_PASSWORD` to make the default account loginable (see the [compose guide](docs/minimal-docker-compose.md)). Note that `--rm` discards container-local data on exit; the `-v cognee_storage:/cognee-storage` named volume keeps your memory across runs.

To run the API, UI, and MCP server from a source checkout, clone this repository, enter its directory, copy [`.env.template`](.env.template) to `.env`, and configure your providers. Then run:

```bash
docker compose --profile ui --profile mcp up
```

The default ports are API **8000**, UI **3000**, and MCP **8001**. For deployment beyond a local demo, configure authentication, persistent storage, and compatible backends using the [permissions guide](https://docs.cognee.ai/setup-configuration/permissions) and [deployment templates](distributed/deploy/README.md). [Cognee Cloud](https://docs.cognee.ai/cognee-cloud/overview) provides the managed option.

The `cognee/cognee` image ships with the GLiNER runtime baked in, so text ingestion and retrieval work in Docker without an LLM key; the local extraction and embedding models download on first use. Set `LLM_API_KEY` (as above) when you want generated answers.

The `cognee/cognee` image is built from upstream Cognee and does not include this fork's changes.

## nanolance graph backend

nanolance is a private project. Its repository, `yoavbendor/nanolance`, is not public, so this backend can be installed only by someone with access to that repository. Ladybug is the default graph store and needs nothing from nanolance. Skip this section unless you have access.

### What nanolance is

nanolance is a storage library for tables in [Lance](https://lancedb.github.io/lance/) format, a columnar, versioned file format. Its core is C++17, with Python bindings in `bindings/python` (the `nanolance` package). The bindings include a `nanolance.lancedb` module, which serves cognee's vector store from nanolance. docmem uses that module; the graph backend does not need it.

The build needs a C++17 compiler and three libraries, nanoarrow, zstd and CLI11, plus two private repositories, `nanom` and `nanos3reader`, which are fetched at pinned commits. Those private dependencies affect only building nanolance, not using the cognee adapter.

### What the adapter stores

`cognee_nanograph` implements cognee's graph interface on three tables in one directory:

| table | rows | notes |
|---|---|---|
| `nodes` | one per graph node: `id`, `name`, `type`, timestamps, `properties` (JSON, zstd-compressed), provenance columns | BTREE index on `id`, BITMAP on `type` |
| `edges` | one per edge: `source_id`, `target_id`, `relationship_name`, timestamps, `properties`, provenance columns, `edge_key` (`source\|relationship\|target`) | BTREE on `source_id`, `target_id`, `edge_key`; BITMAP on `relationship_name` |
| `metadata` | key/value pairs, such as the provenance marker | |

The provenance columns record which documents, dataset and pipeline runs own each node and edge. The adapter keeps them so that deletion can find the elements a document owns; the same columns exist in the Ladybug adapter.

Writes are upserts. Learned values that `improve()` writes (feedback weights, truth state) are kept when an element is added again, as in the Ladybug adapter.

### Concurrency

- Within one process, writes are serialized by an asyncio lock.
- Across processes on one machine, every write holds a file lock, `.write.lock`, in the graph directory. Each write re-reads the tables after taking the lock.
- Reads take no lock. Each operation opens the tables at their current version, so processes sharing a directory see each other's writes.
- Blocking Lance calls run in worker threads, so the event loop stays responsive.

The file lock is local to the directory. This README makes no claim about directories on network filesystems.

### What works, and what does not

- **Supported:** the graph operations cognee uses for ingestion, retrieval, `improve()` and `forget()`. The adapter is checked against the Ladybug adapter by a parity test suite (`cognee/tests/unit/graph_nanolance/test_nanolance_parity.py`) and by product journeys in CI.
- **Not supported: Cypher.** The adapter has no Cypher engine. Raw `query()` calls raise `NotImplementedError`, and so do the `CYPHER` and `NATURAL_LANGUAGE` search types. Ordinary `recall()` does not use Cypher unless `ALLOW_CYPHER_QUERY=true`, so it works; an explicit `CYPHER` search does not.
- **Local directories only.** A remote key (`GRAPH_DATABASE_KEY`) is refused. Remote graph servers are not supported.
- **Vectors are unchanged.** nanolance replaces only the graph. Vectors stay in the configured vector store, LanceDB by default.
- **Existing data does not move.** A new graph directory starts empty. This README does not describe copying an existing Ladybug graph into nanolance.
- **Not in the CLI or the Docker image.** Access is only through Python code that registers the backend, as shown below.

### Setup

1. Get access to `yoavbendor/nanolance` and check that `git ls-remote https://github.com/yoavbendor/nanolance` succeeds.
2. Install nanolance. This builds C++ code, so it needs a C++17 compiler:

   ```bash
   uv pip install "nanolance @ git+https://github.com/yoavbendor/nanolance@main#subdirectory=bindings/python"
   ```

3. Register the backend before Cognee builds a graph engine, then select it:

   ```python
   import cognee_nanograph

   cognee_nanograph.register()
   ```

   ```bash
   export GRAPH_DATABASE_PROVIDER=nanolance
   # Optional: the directory that holds the tables.
   export GRAPH_DATABASE_URL=/absolute/path/to/graph_dir
   ```

4. Without `GRAPH_DATABASE_URL`, the graph lives at `<system root>/databases/cognee_graph_nanolance`. With backend access control on (the default), each user and dataset gets its own directory, `<system root>/databases/<owner id>/<dataset id>.nanolance`.

`cognee-cli` and the API server do not call `register()`, so they cannot use this provider.

### Tests

- `pytest cognee/tests/unit/graph_nanolance` runs the adapter's unit, contract, parity and write-lock tests.
- [`nanolance_graph.yml`](.github/workflows/nanolance_graph.yml) builds nanolance in CI, runs those tests, and runs the journey suites on the nanolance graph with access control on and off.

## Egress control

A host may need Cognee to reach only approved endpoints. A policy file lists the hosts, ports and programs Cognee may use, and everything else is refused. Start from [`docs/egress.example.json`](docs/egress.example.json), replace its hosts with yours, and set:

```bash
export COGNEE_EGRESS_POLICY=/absolute/path/to/egress-policy.json
```

The policy works at three levels, from lightest to strongest:

1. **Endpoint check.** With a policy set, a configured LLM or embedding endpoint outside it fails at startup.
2. **In-process guard.** Every Python-level DNS lookup, connection and subprocess start is checked, whichever library makes it.
3. **Isolated runner.** `python -m cognee_egress.isolate --policy policy.json -- <command>` runs the command in a network namespace with only loopback. Its one way out is an allowlisting proxy, which also covers native code. This needs unprivileged user namespaces.

Once a process has installed a policy, it cannot be replaced or removed, and the policy forces telemetry off. Some features need a decision for each deployment, such as the Ladybug JSON extension, URL ingestion and code graphs. [`docs/egress.md`](docs/egress.md) covers the policy format, those decisions and the limits of each level.

## Run the Whole Memory Layer on Postgres

Graph memory traditionally means operating a stack — a graph database for relationships, a vector database for embeddings, Redis for sessions, and a relational database for metadata — all deployed, secured, and paid for before an agent remembers anything. Since cognee 1.0 you can run the entire memory layer on a single Postgres instance.

> **⚠️ Warning:** Using Postgres as a graph store is currently released as a demo feature. The production-ready version is available as a licensed product. Use the demo to keep relational metadata, PGVector, and graph state working together in one Postgres service.

<a id="benchmarks"></a>

## Benchmarks and research

The [BEAM evaluation](cognee/eval_framework/beam/REPORT.md) measures conversational memory using synthetic long-context conversations and an LLM judge. The reported runs use Cognee's memory components with benchmark-specific data formatting, prompts, and retrieval configuration.

| BEAM context | Reported score (0–1) | Scope |
| --- | --- | --- |
| 100K tokens | **0.79** | Fixed hybrid retrieval; four evaluation rounds over 20 questions from one held-out conversation. |
| 10M tokens | **0.67** | Exploratory result; question-type routing selected and scored on the same question set, averaged over five rounds. |

The two settings use different conversations, ingestion models, and retrieval-selection procedures. Read the [methodology, models, limitations, and reproduction instructions](cognee/eval_framework/beam/REPORT.md) before comparing these scores with other systems. The report also documents the remaining reproduction gap for the distributed 10M ingestion.

For the research behind Cognee's graph/LLM interface, see [Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning](https://arxiv.org/abs/2505.24478) (Markovic et al., 2025).

## Latest News

[![Watch Demo](https://img.youtube.com/vi/8hmqS2Y5RVQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=8hmqS2Y5RVQ&t=13s)

- **[v1.6.1 — Google Sync & Visualization](https://github.com/topoteretes/cognee/releases/tag/v1.6.1)** (September 24, 2026): Google Drive and Gmail OAuth connectors bundled in the SDK, `/visualize/json` streamed in chunks for large graphs, and the GLiNER installer moved off the event loop with CPU torch installed on first use.
- Document `external_metadata` is stamped on every chunk and surfaced in hybrid retrieval. **Breaking:** `dlt` is now a core dependency — make sure it is installed if you manage dependencies by hand.
- **[v1.6.0 — Keyless workflows & pipeline reliability](https://github.com/topoteretes/cognee/releases/tag/v1.6.0)** (September 18, 2026): build and search text memory with local models and no cloud LLM key; LLM-dependent improvement stages skip when no LLM is configured, and pipeline recovery preserves completed documents after crashes.


## Community & Support

### Contributing
We welcome contributions from the community! Your input helps make Cognee better for everyone. See [`CONTRIBUTING.md`](CONTRIBUTING.md) to get started.

### Code of Conduct

We're committed to fostering an inclusive and respectful community. Read our [Code of Conduct](https://github.com/topoteretes/cognee/blob/main/CODE_OF_CONDUCT.md) for guidelines.

## Research & Citation

We recently published a research paper on optimizing knowledge graphs for LLM reasoning:

```bibtex
@misc{markovic2025optimizinginterfaceknowledgegraphs,
      title={Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning},
      author={Vasilije Markovic and Lazar Obradovic and Laszlo Hajdu and Jovan Pavlovic},
      year={2025},
      eprint={2505.24478},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.24478},
}
```
