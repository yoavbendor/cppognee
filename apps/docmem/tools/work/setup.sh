#!/usr/bin/env bash
# Set up docmem from a fresh clone of cppognee: one venv with cognee (this repo), nanolance (built from source)
# and docmem. Run from anywhere inside the clone:  bash apps/docmem/tools/work/setup.sh
# Needs: git, uv (https://docs.astral.sh/uv/), a C++17 compiler and network access to PyPI and GitHub.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docmem="$(cd "$here/../.." && pwd)"
repo="$(cd "$docmem/../.." && pwd)"
venv="${DOCMEM_VENV:-$docmem/.venv}"
python="${DOCMEM_PYTHON:-3.12}"
nanolance_ref="${NANOLANCE_REF:-main}"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
command -v c++ >/dev/null || command -v g++ >/dev/null || command -v clang++ >/dev/null || command -v cl >/dev/null \
  || { echo "a C++17 compiler is required to build nanolance"; exit 1; }

echo "== venv: $venv (Python $python)"
uv venv --python "$python" "$venv"
py="$venv/bin/python"; [ -x "$py" ] || py="$venv/Scripts/python.exe"

echo "== cognee (this repository, editable)"
uv pip install --python "$py" -e "$repo"
echo "== nanolance (yoavbendor/nanolance@$nanolance_ref, bindings/python; builds C++)"
uv pip install --python "$py" "nanolance @ git+https://github.com/yoavbendor/nanolance@${nanolance_ref}#subdirectory=bindings/python"
echo "== docmem (editable)"
uv pip install --python "$py" -e "$docmem"

echo "== check"
"$py" -c "import cognee, nanolance, docmem; print('cognee', cognee.__version__ if hasattr(cognee,'__version__') else '?', '| nanolance', nanolance.__version__, '| docmem', docmem.__version__)"
"$py" -m docmem --help | head -3
echo
echo "Done. Next: set the decision-model variables and run"
echo "  bash $docmem/tools/work/run_eval.sh"
