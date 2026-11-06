"""Size of a keyword index against the text it indexes, and what compressed layouts would take. Writes nothing.

    python tools/kw_size.py --db DB

Measured:
  text bytes     sum of the DocPassage byte lengths in the graph (the text the index covers)
  lance bytes    what the two kw tables occupy on disk (every version Lance keeps)
  postings       rows and entries (one entry per term occurrence in a passage)

Estimated, for the live postings compacted into one segment per term (the layout compact() writes):
  nums           per term, the sorted passage numbers in blocks of 128. Each block: its first number (4 bytes),
                 a bit width (1 byte), then the gaps (next - prev - 1) packed at the block's widest gap.
  tfs            per block of 128: a bit width (1 byte), then tf - 1 packed at the block's widest value.
  terms          a front-coded dictionary, restart every 16 terms (the first term of a restart is stored whole):
                 varint shared-prefix length, varint suffix length, suffix bytes. Plus per term a varint df and a
                 4-byte offset into the nums stream; per restart a 4-byte offset.
Varints take 1 byte below 128, 2 below 16384, else 3.
Each estimate is printed as a percentage of the text bytes. Raw means 4 bytes per num and 2 per tf, uncompressed.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BLOCK = 128
RESTART = 16


def varint(n: int) -> int:
    return 1 if n < 128 else 2 if n < 16384 else 3


def bits(x: int) -> int:
    return int(x).bit_length()


def nums_bytes(nums: np.ndarray) -> int:
    total = 0
    for s in range(0, len(nums), BLOCK):
        blk = nums[s:s + BLOCK]
        total += 4 + 1
        if len(blk) > 1:
            gaps = np.diff(blk) - 1
            total += -(-((len(blk) - 1) * bits(gaps.max())) // 8)
    return total


def tfs_bytes(tfs: np.ndarray) -> int:
    total = 0
    for s in range(0, len(tfs), BLOCK):
        blk = tfs[s:s + BLOCK] - 1
        total += 1 + -(-(len(blk) * bits(blk.max())) // 8)
    return total


def terms_bytes(terms: list[str], dfs: list[int]) -> int:
    total, prev = 0, b""
    for i, (t, df) in enumerate(sorted(zip(terms, dfs))):
        b = t.encode()
        if i % RESTART == 0:
            shared = 0
            total += 4  # restart offset
        else:
            shared = len(os.path.commonprefix([prev, b]))
        suffix = len(b) - shared
        total += varint(shared) + varint(suffix) + suffix + varint(df) + 4  # + df and the nums offset
        prev = b
    return total


async def text_bytes(db: Path) -> tuple[int, int]:
    from docmem import store

    store.configure(db)
    graph, _ = await store.engines()
    nodes, _ = await graph.get_filtered_graph_data([{"type": ["DocPassage"]}], include_edges=False)
    lengths = []
    for n in nodes:
        props = store.node_props({"id": n[0], **n[1]}) if isinstance(n, (tuple, list)) else store.node_props(n)
        lengths.append(int(props.get("length") or 0))
    return sum(lengths), len(lengths)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    a = ap.parse_args(argv)

    from docmem.keywords import KeywordIndex, PASSAGES, POSTINGS

    kw = KeywordIndex(a.db)
    text, n_text = asyncio.run(text_bytes(a.db))
    st = kw.stats()
    postings = kw.postings(live_only=True)
    entries = sum(len(n) for n, _ in postings.values())
    nums_total = sum(nums_bytes(n) for n, _ in postings.values())
    tfs_total = sum(tfs_bytes(f) for _, f in postings.values())
    terms_total = terms_bytes(list(postings), [len(n) for n, _ in postings.values()])
    encoded = nums_total + tfs_total + terms_total
    raw = 4 * entries + 2 * entries

    def pct(x):
        return f"{100 * x / text:6.2f}%" if text else "   n/a"

    lance = st["bytes"][PASSAGES] + st["bytes"][POSTINGS]
    print(f"passages in graph        {n_text:>12,}   live in keyword index {st['passages']:,}  "
          f"deleted rows {st['deleted_rows']:,}")
    print(f"text bytes indexed       {text:>12,}")
    print(f"kw lance bytes on disk   {lance:>12,}  {pct(lance)}   "
          f"(passages {st['bytes'][PASSAGES]:,}, postings {st['bytes'][POSTINGS]:,}; "
          f"versions {st['versions'][PASSAGES]}/{st['versions'][POSTINGS]})")
    print(f"postings rows / entries  {st['postings_rows']:>12,} / {st['postings_entries']:,}  "
          f"distinct terms {st['distinct_terms']:,}")
    print(f"raw nums+tfs (4+2 B/entry) {raw:>10,}  {pct(raw)}")
    print(f"est. nums (delta, bit-packed)  {nums_total:>10,}  {pct(nums_total)}")
    print(f"est. tfs (bit-packed)          {tfs_total:>10,}  {pct(tfs_total)}")
    print(f"est. term dictionary           {terms_total:>10,}  {pct(terms_total)}")
    print(f"est. total encoded             {encoded:>10,}  {pct(encoded)}")


if __name__ == "__main__":
    main()
