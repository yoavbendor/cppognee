"""docmem: index local documents into cognee (on nanolance) keeping only vectors and provenance
(file:// URI, byte offset, length, line, SHA-256), never the text; answer questions by reading the
cited byte ranges back from the files."""

import cognee_egress

cognee_egress.install_from_env()

__version__ = "0.1.0"
PROTOCOL = 1
