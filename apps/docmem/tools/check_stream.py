"""Check a docmem protocol-1 stream: record order, and that every passage reads back from its source.

    python tools/check_stream.py STREAM.jsonl [--sample N] [--seed S]

Order: scan_start first; per document doc, then its section and passage records, then end (whose counts
must match); scan_end last, listing the documents' uris in order. Each passage is then re-read with
docmem.sources.read_span (the document's extractor and sha256, the passage's sha256) and reported as ok,
changed or missing. A passage whose re-read text differs from the stream's "text" is counted separately.
--sample N checks N random documents (default: every document for file:// streams, 20 otherwise).
Exits 1 on any order error or any passage that is not ok.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docmem.sources import read_span


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
                errors.append(f"line {i}: doc before the end of {cur['uri']}")
            cur = {"rec": r, "sections": [], "passages": []}
            docs.append(cur)
        elif t in ("section", "passage"):
            if cur is None or r.get("uri") != cur["rec"]["uri"]:
                errors.append(f"line {i}: {t} outside its document")
            else:
                cur["sections" if t == "section" else "passages"].append(r)
        elif t == "end":
            if cur is None or r.get("uri") != cur["rec"]["uri"]:
                errors.append(f"line {i}: end does not match the open document")
                continue
            if r.get("sections") != len(cur["sections"]) or r.get("passages") != len(
                cur["passages"]
            ):
                errors.append(f"line {i}: end counts differ from the records of {r['uri']}")
            cur = None
        else:
            errors.append(f"line {i}: unexpected record type {t!r}")
    if cur is not None:
        errors.append(f"document {cur['rec']['uri']} has no end record")
    last = recs[-1] if recs and recs[-1].get("type") == "scan_end" else {}
    if last.get("uris") != [d["rec"]["uri"] for d in docs]:
        errors.append("scan_end uris do not list the documents in order")
    return docs, errors


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stream", type=Path)
    ap.add_argument("--sample", type=int, default=None, help="random documents to re-read")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    docs, errors = load(a.stream)
    print(f"{a.stream}: {len(docs)} documents, {sum(len(d['passages']) for d in docs)} passages")
    chosen = docs
    n = (
        a.sample
        if a.sample is not None
        else (len(docs) if docs and docs[0]["rec"]["uri"].startswith("file://") else 20)
    )
    if n < len(docs):
        chosen = random.Random(a.seed).sample(docs, n)
    status, text_differs, shown = Counter(), 0, 0
    for d in chosen:
        doc = d["rec"]
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
    print(
        f"re-read {sum(status.values())} passages of {len(chosen)} documents: "
        f"ok {status['ok']}, changed {status['changed']}, missing {status['missing']}"
        f"; text differs from stream {text_differs}"
    )
    for e in errors:
        print("ORDER ERROR:", e)
    if errors or status["ok"] != sum(status.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
