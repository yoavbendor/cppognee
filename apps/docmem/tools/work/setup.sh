#!/usr/bin/env bash
# Set up docmem from a fresh clone of cppognee: one venv with cognee (this repo), nanolance (built from source)
# and docmem. Run from anywhere inside the clone:  bash apps/docmem/tools/work/setup.sh
# Needs: git, uv (https://docs.astral.sh/uv/), a C++17 compiler, and network access to your package
# index and to the nanolance git repository.
# Variables (all optional): DOCMEM_VENV, DOCMEM_PYTHON (3.12), NANOLANCE_GIT_URL (the GitHub repo; set your
# mirror's URL), NANOLANCE_REF (main), LADYBUG_EXTENSION_REPO (https://extension.ladybugdb.com/).
# With COGNEE_EGRESS_POLICY set, the whole setup runs isolated (no network except the policy's "install"
# hosts, through cognee_egress.isolate); see docs/egress.md.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docmem="$(cd "$here/../.." && pwd)"
repo="$(cd "$docmem/../.." && pwd)"
venv="${DOCMEM_VENV:-$docmem/.venv}"
python="${DOCMEM_PYTHON:-3.12}"
nanolance_url="${NANOLANCE_GIT_URL:-https://github.com/yoavbendor/nanolance}"
nanolance_ref="${NANOLANCE_REF:-main}"
ladybug_repo="${LADYBUG_EXTENSION_REPO:-https://extension.ladybugdb.com/}"

if [ -n "${COGNEE_EGRESS_POLICY:-}" ] && [ -z "${COGNEE_SETUP_ISOLATED:-}" ]; then
  echo "== isolated install: only the 'install' hosts of $COGNEE_EGRESS_POLICY are reachable"
  export COGNEE_SETUP_ISOLATED=1
  exec env PYTHONPATH="$repo" python3 -m cognee_egress.isolate --policy "$COGNEE_EGRESS_POLICY" \
    --phase install --without-guard -- bash "${BASH_SOURCE[0]}" "$@"
fi

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
command -v c++ >/dev/null || command -v g++ >/dev/null || command -v clang++ >/dev/null || command -v cl >/dev/null \
  || { echo "a C++17 compiler is required to build nanolance"; exit 1; }

echo "== venv: $venv (Python $python)"
uv venv --clear --python "$python" "$venv"
py="$venv/bin/python"; [ -x "$py" ] || py="$venv/Scripts/python.exe"

echo "== cognee (this repository, editable)"
uv pip install --python "$py" -e "$repo"
echo "== nanolance ($nanolance_url@$nanolance_ref, bindings/python; builds C++)"
uv pip install --python "$py" "nanolance @ git+${nanolance_url}@${nanolance_ref}#subdirectory=bindings/python"
echo "== docmem (editable)"
uv pip install --python "$py" -e "$docmem"

echo "== Ladybug JSON extension (installed now, so the runtime never downloads it)"
"$py" - "$ladybug_repo" <<'PY'
import os, sys, tempfile
import ladybug
conn = ladybug.Connection(ladybug.Database(os.path.join(tempfile.mkdtemp(), "db")))
conn.execute(f"INSTALL JSON FROM '{sys.argv[1]}';")
conn.execute("LOAD EXTENSION JSON;")
print("installed from", sys.argv[1])
PY

echo "== check"
"$py" -c "import cognee, nanolance, docmem; print('cognee', cognee.__version__ if hasattr(cognee,'__version__') else '?', '| nanolance', nanolance.__version__, '| docmem', docmem.__version__)"
"$py" -m docmem --help | head -3
echo
echo "Done. Next: set the decision-model variables and run"
echo "  bash $docmem/tools/work/run_eval.sh"
