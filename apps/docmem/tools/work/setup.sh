#!/usr/bin/env bash
# Set up docmem from a fresh clone of cppognee: one venv with cognee (this repo), nanolance and docmem.
# Run from anywhere inside the clone:  bash apps/docmem/tools/work/setup.sh
# Needs: git, uv (https://docs.astral.sh/uv/) and network access to your package index; to build nanolance
# from source also a C++17 compiler and access to the nanolance git repository.
# Variables (all optional): DOCMEM_VENV, DOCMEM_PYTHON (3.12), LADYBUG_EXTENSION_REPO
# (https://extension.ladybugdb.com/), and where nanolance comes from, NANOLANCE_INSTALL:
#   git (default)   build it from NANOLANCE_GIT_URL (the GitHub repo; set your mirror's URL) at NANOLANCE_REF (main)
#   index           `nanolance` from your package index (an Artifactory that carries it)
#   <path>.whl      that wheel file (a version your index does not carry yet)
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
nanolance_install="${NANOLANCE_INSTALL:-git}"
case "$nanolance_install" in
  git) nanolance_spec="nanolance @ git+${nanolance_url}@${nanolance_ref}#subdirectory=bindings/python"
       nanolance_from="$nanolance_url@$nanolance_ref, bindings/python; builds C++" ;;
  index) nanolance_spec="nanolance"; nanolance_from="your package index" ;;
  *.whl) [ -f "$nanolance_install" ] || { echo "NANOLANCE_INSTALL: no wheel at $nanolance_install"; exit 1; }
         nanolance_spec="$(cd "$(dirname "$nanolance_install")" && pwd)/$(basename "$nanolance_install")"
         nanolance_from="wheel $nanolance_spec" ;;
  *) echo "NANOLANCE_INSTALL must be git, index or a path to a .whl file, not '$nanolance_install'"; exit 1 ;;
esac
ladybug_repo="${LADYBUG_EXTENSION_REPO:-https://extension.ladybugdb.com/}"

if [ -n "${COGNEE_EGRESS_POLICY:-}" ] && [ -z "${COGNEE_SETUP_ISOLATED:-}" ]; then
  echo "== isolated install: only the 'install' hosts of $COGNEE_EGRESS_POLICY are reachable"
  export COGNEE_SETUP_ISOLATED=1
  exec env PYTHONPATH="$repo" python3 -m cognee_egress.isolate --policy "$COGNEE_EGRESS_POLICY" \
    --phase install --without-guard -- bash "${BASH_SOURCE[0]}" "$@"
fi

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
if [ "$nanolance_install" = git ]; then
  command -v c++ >/dev/null || command -v g++ >/dev/null || command -v clang++ >/dev/null || command -v cl >/dev/null \
    || { echo "a C++17 compiler is required to build nanolance (or set NANOLANCE_INSTALL=index or a .whl path)"; exit 1; }
fi

echo "== venv: $venv (Python $python)"
uv venv --clear --python "$python" "$venv"
py="$venv/bin/python"; [ -x "$py" ] || py="$venv/Scripts/python.exe"

echo "== cognee (this repository, editable)"
uv pip install --python "$py" -e "$repo"
echo "== nanolance ($nanolance_from)"
uv pip install --python "$py" "$nanolance_spec"
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
