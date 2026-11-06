# Loaded by every Python process docmem starts (PYTHONPATH): cognee's worker processes then use nanolance too.
import os

if os.environ.get("DOCMEM_VECTOR_BACKEND", "nanolance") == "nanolance":
    import nanolance.lancedb

    nanolance.lancedb.install_as_lancedb()
