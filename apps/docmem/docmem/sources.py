"""Reading a passage back from its source: a file:// URI (docmem.store) or a Wikipedia revision.

A Wikipedia document's URI is https://en.wikipedia.org/w/index.php?oldid=N. Its passages are byte ranges of
the markdown that docmem.scan_wiki cuts from the revision's wikitext (extractor "wikiplain/1" or "wikiplain/2",
see docmem.extract); the wikitext is fetched with action=raw, checked against the document's SHA-256, extracted
again with the document's extractor, and the range is sliced out.
Fetched texts are kept in memory only (a small lru_cache), never on disk.

read_span returns (text, status) like docmem.store.read_span: "ok", "changed" (the source no longer matches the
indexed bytes) or "missing" (the revision is gone).

The page title is an argument, not looked up: the markdown starts with "# Title", and the title is part of the
document record the scan wrote (a lookup by revision would return the title as it is now, after any move).
"""

import hashlib
import re
import time
import urllib.error
import urllib.request
from functools import lru_cache

from . import store
from .extract import RENDERERS

USER_AGENT = "docmem/0.1 (research prototype)"
WIKI_OLDID = re.compile(r"^https://en\.wikipedia\.org/w/index\.php\?oldid=(\d+)$")
TRIES = 5


def fetch(url: str, headers: dict | None = None) -> bytes | None:
    """The body of url, or None on 404. 429 and 503 wait Retry-After (or 1, 2, 4, 8 s), up to TRIES
    attempts; connection errors are retried the same way."""
    for attempt in range(TRIES):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code not in (429, 503) or attempt == TRIES - 1:
                raise
            retry_after = e.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.isdigit() else 2**attempt
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == TRIES - 1:
                raise
            delay = 2**attempt
        time.sleep(delay)
    raise AssertionError("unreachable")


@lru_cache(maxsize=128)
def _markdown(oldid: int, title: str, doc_sha256: str | None, extractor: str):
    """(markdown bytes or None, status) for one revision: action=raw is byte-identical to the dump's text
    (checked on sampled articles), so a hash mismatch means the revision changed, not a format difference."""
    raw = fetch(f"https://en.wikipedia.org/w/index.php?oldid={oldid}&action=raw")
    if raw is None:
        return None, "missing"
    if doc_sha256 is not None and hashlib.sha256(raw).hexdigest() != doc_sha256:
        return None, "changed"
    md, _ = RENDERERS[extractor](title, raw.decode("utf-8"))
    return md.encode("utf-8"), "ok"


def read_span(
    uri: str,
    offset: int,
    length: int,
    sha256: str | None = None,
    extractor: str | None = None,
    doc_sha256: str | None = None,
    title: str | None = None,
):
    if uri.startswith("file://"):
        return store.read_span(uri, offset, length, sha256)
    m = WIKI_OLDID.match(uri)
    if not m:
        raise ValueError(f"no reader for {uri}")
    if extractor not in RENDERERS:
        raise ValueError(
            f"{uri}: extractor {extractor!r} is not one docmem reads (only {sorted(RENDERERS)})"
        )
    if title is None:
        raise ValueError(f"{uri}: a wikiplain passage needs the document's title")
    md, status = _markdown(int(m.group(1)), title, doc_sha256, extractor)
    if status != "ok":
        return "", status
    data = md[offset : offset + length]
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        return data.decode("utf-8", "replace"), "changed"
    return data.decode("utf-8", "replace"), "ok"
