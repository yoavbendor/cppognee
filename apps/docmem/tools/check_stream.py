"""Check a docmem protocol-1 stream: record order, and that every passage reads back from its source.

    python tools/check_stream.py STREAM.jsonl [--sample N] [--seed S]

Order: scan_start first; per document doc, then its section and passage records, then its link records, then end
(whose counts must match); scan_end last, listing the documents' uris in order. Each passage is then re-read with
docmem.sources.read_span (the document's extractor and sha256, the passage's sha256) and reported as ok,
changed or missing. A passage whose re-read text differs from the stream's "text" is counted separately.
Links: offsets must not decrease within a document; a category link has offset 0, length 0 and an empty anchor;
any other link's bytes at offset+length must decode to its anchor, re-read from the source (a file:// document
is read from disk, a Wikipedia revision through read_span), and reported as ok, changed, missing or mismatch.
--sample N checks N random documents (default: every document for file:// streams, 20 otherwise).
Exits 1 on any order error, any passage that is not ok, or any link that is not ok.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docmem.links import KINDS  # noqa: E402
from docmem.sources import read_span  # noqa: E402
from docmem.store import uri_path  # noqa: E402


def load(path: Path):
    recs = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    errors = []
    if not recs or recs[0].get("type") != "scan_start":
        errors.append("stream does not start with scan_start")
    if not recs or recs[-1].get("type") != "scan_end":
        errors.append("stream does not end with scan_end")
    docs, cur = [], None
    for i, r in enumerate(recs[1:-1], 2):
        t = r.get("type")
        if t == "doc":
            if cur is not None:
                errors.append(f"line {i}: doc before the end of {cur['rec']['uri']}")
            cur = {"rec": r, "sections": [], "passages": [], "links": [], "closed": False}
            docs.append(cur)
        elif t in ("section", "passage"):
            if cur is None or r.get("uri") != cur["rec"]["uri"]:
                errors.append(f"line {i}: {t} outside its document")
            elif cur["links"]:
                errors.append(f"line {i}: {t} after the links of {r['uri']}")
            else:
                cur["sections" if t == "section" else "passages"].append(r)
        elif t == "link":
            if cur is None or r.get("uri") != cur["rec"]["uri"]:
                errors.append(f"line {i}: link outside its document")
            else:
                cur["links"].append(r)
        elif t == "end":
            if cur is None or r.get("uri") != cur["rec"]["uri"]:
                errors.append(f"line {i}: end does not match the open document")
                continue
            if r.get("sections") != len(cur["sections"]) or r.get("passages") != len(
                cur["passages"]
            ):
                errors.append(f"line {i}: end counts differ from the records of {r['uri']}")
            if r.get("links") != len(cur["links"]):
                errors.append(f"line {i}: end links count differs from the link records of {r['uri']}")
            cur["closed"] = True
            cur = None
        else:
            errors.append(f"line {i}: unexpected record type {t!r}")
    if cur is not None:
        errors.append(f"document {cur['rec']['uri']} has no end record")
    last = recs[-1] if recs and recs[-1].get("type") == "scan_end" else {}
    if last.get("uris") != [d["rec"]["uri"] for d in docs]:
        errors.append("scan_end uris do not list the documents in order")
    for d in docs:
        offsets = [l.get("offset") for l in d["links"]]
        if offsets != sorted(offsets):
            errors.append(f"links of {d['rec']['uri']} are not sorted by offset")
    return docs, errors


def check_link(link: dict, doc: dict, source) -> str:
    """ok, mismatch, changed, missing or bad (the link record is malformed). source is the document's bytes for a
    file:// document (None when it changed), and is not used for a Wikipedia revision."""
    if link.get("kind") not in KINDS:
        return "bad"
    if link["kind"] == "category":
        return "ok" if (link["offset"], link["length"], link["anchor"]) == (0, 0, "") else "bad"
    offset, length, anchor = link.get("offset"), link.get("length"), link.get("anchor")
    if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length <= 0 or not anchor:
        return "bad"
    uri = doc["rec"]["uri"]
    if uri.startswith("file://"):
        if source is None:
            return "changed"
        if offset + length > len(source):
            return "mismatch"
        text = source[offset : offset + length].decode("utf-8", "replace")
    else:
        text, status = read_span(
            uri,
            offset,
            length,
            extractor=doc["rec"].get("extractor"),
            doc_sha256=doc["rec"].get("sha256"),
            title=doc["rec"].get("title") if doc["rec"].get("extractor") else None,
        )
        if status != "ok":
            return status
    return "ok" if text == anchor else "mismatch"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stream", type=Path)
    ap.add_argument("--sample", type=int, default=None, help="random documents to re-read")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    docs, errors = load(a.stream)
    print(
        f"{a.stream}: {len(docs)} documents, {sum(len(d['passages']) for d in docs)} passages, "
        f"{sum(len(d['links']) for d in docs)} links"
    )
    chosen = docs
    n = (
        a.sample
        if a.sample is not None
        else (len(docs) if docs and docs[0]["rec"]["uri"].startswith("file://") else 20)
    )
    if n < len(docs):
        chosen = random.Random(a.seed).sample(docs, n)
    status, text_differs, shown = Counter(), 0, 0
    link_status = Counter()
    for d in chosen:
        doc = d["rec"]
        source = None
        if d["links"] and doc["uri"].startswith("file://"):
            source = uri_path(doc["uri"]).read_bytes() if uri_path(doc["uri"]).exists() else None
            if source is not None and hashlib.sha256(source).hexdigest() != doc.get("sha256"):
                source = None
        for p in d["passages"]:
            text, st = read_span(
                doc["uri"],
                p["offset"],
                p["length"],
                sha256=p["sha256"],
                extractor=doc.get("extractor"),
                doc_sha256=doc.get("sha256"),
                title=doc.get("title") if doc.get("extractor") else None,
            )
            status[st] += 1
            if st == "ok" and text != p["text"]:
                text_differs += 1
            if st != "ok" and shown < 10:
                shown += 1
                print(f"  {st}: {doc['uri']} {p['offset']}+{p['length']}", file=sys.stderr)
        for l in d["links"]:
            st = check_link(l, d, source)
            link_status[st] += 1
            if st != "ok" and shown < 10:
                shown += 1
                print(f"  link {st}: {doc['uri']} {l.get('offset')}+{l.get('length')} {l.get('anchor')!r}",
                      file=sys.stderr)
    print(
        f"re-read {sum(status.values())} passages of {len(chosen)} documents: "
        f"ok {status['ok']}, changed {status['changed']}, missing {status['missing']}"
        f"; text differs from stream {text_differs}"
    )
    print(
        "links of those documents: "
        + ", ".join(f"{k} {v}" for k, v in sorted(link_status.items()))
        + ("" if link_status else "none")
    )
    for e in errors:
        print("ORDER ERROR:", e)
    if errors or status["ok"] != sum(status.values()) or any(k != "ok" for k in link_status):
        sys.exit(1)


if __name__ == "__main__":
    main()
