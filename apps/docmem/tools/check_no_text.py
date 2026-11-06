"""Check that an index holds no passage text: every file under the index's databases is searched for
40-character fragments of the source documents (every 60th position of each paragraph over 200 chars).

    python tools/check_no_text.py --db kb/ docs/*.md

Exit status 1 if any fragment is found. Derived text (summaries, entity names, link anchors, headings)
is expected in the index; long verbatim runs of body text are not.
"""
import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args()
    blob = b"".join(p.read_bytes() for p in (a.db / "system").rglob("*") if p.is_file())
    checked = found = 0
    for f in a.files:
        for para in f.read_text(errors="replace").split("\n\n"):
            if len(para) <= 200:
                continue
            for i in range(0, len(para) - 40, 60):
                checked += 1
                if para[i:i + 40].encode() in blob:
                    found += 1
                    if found <= 5:
                        print(f"found: {f.name}: {para[i:i + 40]!r}")
    print(f"{found} of {checked} fragments found in {len(blob)} bytes of index files")
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
