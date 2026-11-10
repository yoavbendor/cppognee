# docmem at work: setup and the decision-model evaluation

This is the short path to build docmem from a fresh clone and run the decision-model benchmark against
**Microsoft Decision-1** on Azure AI Foundry. The run produces one Markdown report to paste back for review.

## What you need

- git, [uv](https://docs.astral.sh/uv/getting-started/installation/), a C++17 compiler (gcc, clang or MSVC):
  nanolance is built from source during setup.
- Network access to PyPI and github.com (setup only).
- A Foundry resource with a **Microsoft-Decision-1** deployment, and its endpoint plus an API key (or Entra ID
  sign-in with `az login`).

No database, document corpus or embedding model is needed for this evaluation: the benchmark ships its item sets
in `examples/decision_bench/` (public text only: Wikipedia excerpts, the Airbus Apollo 11 model, nanos3reader).

## 1. Clone and set up (about 5 minutes)

```sh
git clone https://github.com/yoavbendor/cppognee
cd cppognee
bash apps/docmem/tools/work/setup.sh
```

This creates `apps/docmem/.venv` with cognee (this repository), nanolance (built from
`yoavbendor/nanolance@main`) and docmem, installs Ladybug's JSON extension, then checks the three import.
Variables, all optional: `DOCMEM_VENV` (venv path), `DOCMEM_PYTHON` (default 3.12), `NANOLANCE_GIT_URL`
(default the GitHub repository; set your internal mirror's URL), `NANOLANCE_REF` (default `main`),
`LADYBUG_EXTENSION_REPO` (default `https://extension.ladybugdb.com/`).

### On a corporate host: one egress policy for install and runtime

Copy `docs/egress.example.json` (repository root), put your internal hosts in it (package mirror, git
mirror, LLM, decider), and export it before setup and before every run:

```sh
export COGNEE_EGRESS_POLICY=$HOME/cognee-egress.json
NANOLANCE_GIT_URL=https://<your-gitlab>/<group>/nanolance bash apps/docmem/tools/work/setup.sh
apps/docmem/.venv/bin/python -m cognee_egress.isolate -- bash apps/docmem/tools/work/run_eval.sh
```

nanolance's C++ build clones five dependencies. Point them at your mirrors too, either all under one
group with `NANOLANCE_GIT_BASE=https://<your-gitlab>/<group>` (each is then `<group>/<owner>/<repo>.git`),
or one by one: `NANOLANCE_GIT_URL_NANOARROW` (apache/arrow-nanoarrow), `NANOLANCE_GIT_URL_ZSTD`
(facebook/zstd), `NANOLANCE_GIT_URL_CLI11` (CLIUtils/CLI11), `NANOLANCE_GIT_URL_NANOM` (yoavbendor/nanom),
`NANOLANCE_GIT_URL_NANOS3READER` (yoavbendor/nanos3reader). The build checks out pinned commits, so a
mirror must carry full history, not a shallow copy.

Setup then runs in a network namespace that can reach only the policy's `install` hosts; the evaluation
runs where only the `runtime` hosts are reachable, so the decider host must be listed there with role
`decider`. Without user namespaces (`unshare -rn true` fails), skip the `isolate` wrapper: the policy
still applies inside Python. Details and limits: `docs/egress.md`.

## 2. Point it at Decision-1

```sh
export DOCMEM_DECIDER_URL=https://<your-resource>.cognitiveservices.azure.com   # the Foundry resource endpoint
export DOCMEM_DECIDER_MODEL=<your Decision-1 deployment name>                   # deployment, not model, name
export DOCMEM_DECIDER_KEY=<api key>
# or, instead of a key, Entra ID:  export DOCMEM_DECIDER_AUTH=entra
#                                  apps/docmem/.venv/bin/python -m pip install azure-identity   (then az login)
```

The client calls `<URL>/providers/microsoft/v1/systemone`. A URL already ending in `/systemone` is used as it is.

Try one request first:

```sh
apps/docmem/.venv/bin/python -m docmem decide-check
```

It prints the parsed decision and the raw JSON response. Microsoft's documentation does not name the probability
fields; the client reads Kev's `answers[name]` shape (`noul`, `choice`, `probabilities`, `score`,
`confidence`) and fails with the field names it found if Decision-1 differs. The report below includes this
output either way, so a mismatch can be fixed from it.

## 3. Run the evaluation

```sh
QUICK=20 bash apps/docmem/tools/work/run_eval.sh    # first 20 items per task: a few minutes
bash apps/docmem/tools/work/run_eval.sh             # all 7 tasks, about 1,060 decisions
```

Outputs go to `./dbench_work/` (decisions are cached there, so a rerun only sends what is new), and the report to
`./docmem_work_report.md`. To run some tasks only: `TASKS="concept_a answerability" bash ...`.

## 4. Paste back

Paste the whole content of `docmem_work_report.md`. It holds versions, the decide-check response, the metrics
per task with timings, a few disagreement examples (benchmark text only) and the tail of the run log. Endpoint
hosts and anything that looks like a key or token are redacted, and no company data is involved.

## What is measured

Each task is a decision our knowledge-base pipeline makes today with rules or vector similarity; the report
compares Decision-1 with that baseline. Accuracy, Brier score (calibration; lower is better), AUC for yes/no
tasks, and **auto@5%**: the share of decisions Decision-1 could take on its own while staying at 95% precision.

| task | the decision |
|---|---|
| routing | which retrieval strategy a question needs |
| answerability | do the retrieved passages contain the answer |
| concept_a | which of 8 vector-ranked Wikipedia categories fits an engineering article |
| concept_b | which of 12 engineering fields an article belongs to |
| sysml_kind | the SysML element kind of a model element's text |
| alignment | does this passage describe this model element |
| maturity | concept / designed / implemented / verified |

The reference run (Kev-0.8B on a 4-core CPU) is in the report for comparison: it beat the baseline only on
concept_a (0.75 vs 0.63, 43% automatable at 5% error).

## Troubleshooting

- **nanolance fails to build:** install a C++17 compiler and CMake-compatible build tools; uv fetches CMake and
  Ninja from PyPI. Behind a proxy, set `HTTPS_PROXY` for uv and git.
- **401 / 403:** wrong key, or the key belongs to another resource. With Entra: run `az login` and check the
  role assignment (Cognitive Services User) on the resource.
- **404:** check the endpoint and the deployment name (`DOCMEM_DECIDER_MODEL`).
- **429:** rate limit; the client retries with Retry-After. Rerunning continues from the cache.
