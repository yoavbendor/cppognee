#!/usr/bin/env bash
# Run the decision benchmark against your decision-model endpoint and write a report to paste back.
#   export DOCMEM_DECIDER_URL=https://<resource>.cognitiveservices.azure.com   # Foundry resource endpoint
#   export DOCMEM_DECIDER_KEY=<api key>      # or: export DOCMEM_DECIDER_AUTH=entra  (needs: uv pip install azure-identity)
#   export DOCMEM_DECIDER_MODEL=<deployment name of Microsoft-Decision-1>
#   bash apps/docmem/tools/work/run_eval.sh            # all tasks (about 1,060 decisions)
#   QUICK=20 bash apps/docmem/tools/work/run_eval.sh   # first 20 items per task, to try it out
# Output: $DBENCH_SCRATCH (default ./dbench_work) and the paste-back report docmem_work_report.md.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docmem="$(cd "$here/../.." && pwd)"
venv="${DOCMEM_VENV:-$docmem/.venv}"
py="$venv/bin/python"; [ -x "$py" ] || py="$venv/Scripts/python.exe"
export DBENCH_SCRATCH="${DBENCH_SCRATCH:-$PWD/dbench_work}"
tasks="${TASKS:-routing answerability concept_a concept_b sysml_kind alignment maturity}"
limit="${QUICK:-0}"

: "${DOCMEM_DECIDER_URL:?set DOCMEM_DECIDER_URL (see the comments at the top of this script)}"
: "${DOCMEM_DECIDER_MODEL:?set DOCMEM_DECIDER_MODEL (your deployment name)}"
if [ -z "${DOCMEM_DECIDER_KEY:-}" ] && [ "${DOCMEM_DECIDER_AUTH:-}" != "entra" ]; then
  echo "set DOCMEM_DECIDER_KEY, or DOCMEM_DECIDER_AUTH=entra"; exit 1
fi
mkdir -p "$DBENCH_SCRATCH/dbench"
log="$DBENCH_SCRATCH/run.log"; : > "$log"

echo "== decide-check (one request; the raw response goes into the report)"
"$py" -m docmem decide-check > "$DBENCH_SCRATCH/decide_check.txt" 2>&1 || { cat "$DBENCH_SCRATCH/decide_check.txt"; echo "decide-check failed: see above"; exit 1; }
tail -20 "$DBENCH_SCRATCH/decide_check.txt"

for t in $tasks; do
  mkdir -p "$DBENCH_SCRATCH/dbench/$t"
  cp "$docmem/examples/decision_bench/$t/items.jsonl" "$DBENCH_SCRATCH/dbench/$t/items.jsonl"
  echo "== $t"
  args=(--task "$t" --run --score); [ "$limit" != "0" ] && args+=(--limit "$limit")
  if ! "$py" "$docmem/tools/decision_bench.py" "${args[@]}" >>"$log" 2>&1; then
    echo "   $t failed (details in $log); continuing"
  fi
done
"$py" "$docmem/tools/decision_bench.py" --summary >>"$log" 2>&1 || true
"$py" "$here/make_report.py" --bench "$DBENCH_SCRATCH" --out docmem_work_report.md
echo
echo "Report: $PWD/docmem_work_report.md  -- paste its content back (it holds no keys, no URL host, no company data)."
