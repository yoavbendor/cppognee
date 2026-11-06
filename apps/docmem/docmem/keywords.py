"""Keyword index: BM25 over passage text, in Lance tables written with nanolance, beside the cognee index.

Passage text is never stored. The indexer hands each new passage's text to ``KeywordIndex.add`` once; its terms
and frequencies go into postings and the text is dropped. A search reads the postings rows of the query's terms
only (a BTree index on ``term`` serves the lookup) plus a three-column pass over the passage table.

Tables, in <db>/system/databases/docmem_keywords.lancedb:
  kw_passages  num (dense internal number), passage_id (the DocPassage id), length (tokens), deleted
  kw_postings  term, segment (one per flush), nums (sorted passage numbers), tfs (term frequencies)

Deletion sets ``deleted`` on the passage row (a copy-on-write update), so the postings keep the number until
``compact`` rewrites them without the deleted passages and drops those rows.

    docmem kwsearch --db DB "query"
"""
import argparse
import asyncio
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa

K1, B = 1.2, 0.75
PASSAGES, POSTINGS = "kw_passages", "kw_postings"
PASSAGE_SCHEMA = pa.schema([("num", pa.int32()), ("passage_id", pa.string()), ("length", pa.int32()),
                            ("deleted", pa.bool_())])
POSTING_SCHEMA = pa.schema([("term", pa.string()), ("segment", pa.int32()), ("nums", pa.list_(pa.int32())),
                            ("tfs", pa.list_(pa.uint16()))])
# letters and digits; an internal - ' ’ or . joins two runs into one token
_TOKEN = re.compile(r"[^\W_]+(?:[-'’.][^\W_]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase NFKC tokens: runs of letters and digits, joined by internal - ' or . ("REQ-1234", "o'neill",
    "3.5"). A trailing 's is dropped ("Carano's" -> "carano"). No stemming, no stop words."""
    out = []
    for tok in _TOKEN.findall(unicodedata.normalize("NFKC", text).lower()):
        if tok.endswith(("'s", "’s")):
            tok = tok[:-2]
        if tok:
            out.append(tok)
    return out


def _lit(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _index(table, column: str) -> None:
    """A BTree on ``column``: created once, then kept current with each append (optimize_indices), so a
    filter on it reads only the matching rows of indexed fragments."""
    ds = table._dataset()
    if any(i["name"] == f"{column}_idx" for i in ds.list_indices()):
        ds.optimize.optimize_indices()
    else:
        table.create_scalar_index(column, replace=True)


def _disk_bytes(table) -> int:
    return sum(p.stat().st_size for p in Path(table.uri).rglob("*") if p.is_file())


class KeywordIndex:
    def __init__(self, db_dir: Path, create: bool = False):
        import nanolance.lancedb as ldb

        self.path = Path(db_dir) / "system" / "databases" / "docmem_keywords.lancedb"
        if create:
            self.path.mkdir(parents=True, exist_ok=True)
        elif not self.path.exists():
            raise SystemExit(f"docmem: no keyword index at {self.path} (index with keywords first)")
        self._db = ldb.connect(str(self.path))
        for name, schema in ((PASSAGES, PASSAGE_SCHEMA), (POSTINGS, POSTING_SCHEMA)):
            if name not in self._db:
                if not create:
                    raise SystemExit(f"docmem: {self.path} has no {name} table")
                self._db.create_table(name, schema=schema)
        self._passages = self._db.open_table(PASSAGES)
        self._postings = self._db.open_table(POSTINGS)
        # write state, read from the tables on first use (see _load)
        self._live: dict[str, int] | None = None
        self._next_num = 0
        self._segment = 0
        self._pending: dict[str, tuple[int, int, Counter]] = {}  # passage_id -> (num, length, term counts)
        self._dels: list[int] = []  # committed passage numbers to mark deleted at the next flush

    def _load(self) -> None:
        if self._live is not None:
            return
        t = self._passages.query().limit(None).select(["num", "passage_id", "deleted"]).to_arrow()
        nums = t.column("num").to_pylist()
        self._live = {p: n for n, p, d in zip(nums, t.column("passage_id").to_pylist(),
                                              t.column("deleted").to_pylist()) if not d}
        self._next_num = max(nums) + 1 if nums else 0
        segs = self._postings.query().limit(None).select(["segment"]).to_arrow().column("segment").to_pylist()
        self._segment = max(segs) + 1 if segs else 0

    def add(self, passage_id: str, text: str) -> bool:
        """Buffer a passage's terms until flush. False (nothing added) when the passage is already indexed."""
        self._load()
        if passage_id in self._live:
            return False
        counts = Counter(tokenize(text))
        if counts and max(counts.values()) > 65535:
            raise ValueError(f"passage {passage_id}: a term occurs more than 65535 times")
        num = self._next_num
        self._next_num += 1
        self._live[passage_id] = num
        self._pending[passage_id] = (num, sum(counts.values()), counts)
        return True

    def delete(self, passage_ids: list[str]) -> None:
        """Forget passages: a buffered one is dropped, a flushed one is marked deleted at the next flush."""
        self._load()
        for pid in passage_ids:
            num = self._live.pop(pid, None)
            if num is None:
                continue
            if pid in self._pending:
                del self._pending[pid]
            else:
                self._dels.append(num)

    def flush(self) -> None:
        """Write the buffered passages (rows, then one postings segment), then apply the deletions."""
        self._load()
        if self._pending:
            pend = sorted(self._pending.values(), key=lambda v: v[0])
            self._passages.add(pa.Table.from_pylist(
                [{"num": num, "passage_id": pid, "length": length, "deleted": False}
                 for pid, (num, length, _) in self._pending.items()], schema=PASSAGE_SCHEMA))
            by_term: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
            for num, _, counts in pend:
                for term, tf in counts.items():
                    by_term[term][0].append(num)
                    by_term[term][1].append(tf)
            seg = self._segment
            self._segment += 1
            rows = [{"term": t, "segment": seg, "nums": n, "tfs": f} for t, (n, f) in by_term.items()]
            self._postings.add(pa.Table.from_pylist(rows, schema=POSTING_SCHEMA))
            _index(self._passages, "num")
            if rows:
                _index(self._postings, "term")
            self._pending.clear()
        if self._dels:
            self._passages.update(where=f"num IN ({', '.join(map(str, self._dels))})",
                                  values={"deleted": True})
            self._dels.clear()

    def _meta(self):
        """(num, length, deleted) of every passage row, sorted by num."""
        t = self._passages.query().limit(None).select(["num", "length", "deleted"]).to_arrow()
        num = t.column("num").to_numpy()
        order = np.argsort(num, kind="stable")
        return (num[order], t.column("length").to_numpy()[order].astype(np.float64),
                t.column("deleted").to_numpy()[order].astype(bool))

    def _postings_for(self, terms: list[str]):
        """(term, nums, tfs) for each query term, merged over its segments: only those terms' rows are read."""
        q = ", ".join(_lit(t) for t in terms)
        t = self._postings.query().where(f"term IN ({q})").limit(None).select(["term", "nums", "tfs"]).to_arrow()
        acc: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for term, nums, tfs in zip(t.column("term").to_pylist(), t.column("nums").to_pylist(),
                                   t.column("tfs").to_pylist()):
            acc[term][0].extend(nums)
            acc[term][1].extend(tfs)
        return [(term, np.asarray(n, np.int64), np.asarray(f, np.float64)) for term, (n, f) in acc.items()]

    def _ids(self, nums) -> dict[int, str]:
        q = ", ".join(str(int(n)) for n in nums)
        t = self._passages.query().where(f"num IN ({q})").limit(None).select(["num", "passage_id"]).to_arrow()
        return dict(zip(t.column("num").to_pylist(), t.column("passage_id").to_pylist()))

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """BM25 (k1=1.2, b=0.75) over the live passages: [(passage_id, score)], best first."""
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []
        num, length, deleted = self._meta()
        live = ~deleted
        n_live = int(live.sum())
        if n_live == 0:
            return []
        avgdl = float(length[live].mean())
        scores = np.zeros(len(num))
        for _, nums, tfs in self._postings_for(terms):
            pos = np.searchsorted(num, nums)
            if (pos >= len(num)).any() or (num[np.minimum(pos, len(num) - 1)] != nums).any():
                raise RuntimeError("keyword postings refer to a passage with no row (index is damaged)")
            keep = live[pos]
            pos, tf = pos[keep], tfs[keep]
            df = len(pos)
            if df == 0:
                continue
            idf = math.log(1 + (n_live - df + 0.5) / (df + 0.5))
            dl = length[pos]
            scores[pos] += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))
        hit = np.flatnonzero(scores > 0)
        top = hit[np.argsort(-scores[hit], kind="stable")][:k]
        ids = self._ids(num[top])
        return [(ids[int(num[i])], float(scores[i])) for i in top]

    def postings(self, live_only: bool = True) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """term -> (nums, tfs), merged over segments and sorted by num; deleted passages left out if live_only."""
        t = self._postings.query().limit(None).select(["term", "nums", "tfs"]).to_arrow()
        acc: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for term, nums, tfs in zip(t.column("term").to_pylist(), t.column("nums").to_pylist(),
                                   t.column("tfs").to_pylist()):
            acc[term][0].extend(nums)
            acc[term][1].extend(tfs)
        live_nums = None
        if live_only:
            num, _, deleted = self._meta()
            live_nums = num[~deleted]
        out = {}
        for term, (nums, tfs) in acc.items():
            n = np.asarray(nums, np.int64)
            f = np.asarray(tfs, np.int64)
            if live_nums is not None:
                keep = np.isin(n, live_nums)
                n, f = n[keep], f[keep]
            if n.size:
                order = np.argsort(n, kind="stable")
                out[term] = (n[order], f[order])
        return out

    def rare_terms(self, passage_ids: list[str], n: int = 10) -> list[tuple[str, float]]:
        """The n terms that best characterise these passages: tf * idf summed over them, idf from the live
        passages. Reads every postings row (nothing indexes terms by passage), so it is for expansion, not for
        the search path."""
        q = ", ".join(_lit(p) for p in passage_ids)
        ids = self._passages.query().where(f"passage_id IN ({q})").limit(None).select(["num"]).to_arrow()
        target = set(ids.column("num").to_pylist())
        num, _, deleted = self._meta()
        n_live = int((~deleted).sum())
        live_set = set(num[~deleted].tolist())
        t = self._postings.query().limit(None).select(["term", "nums", "tfs"]).to_arrow()
        scored = []
        for term, nums, tfs in zip(t.column("term").to_pylist(), t.column("nums").to_pylist(),
                                   t.column("tfs").to_pylist()):
            if target.isdisjoint(nums):
                continue
            df = sum(1 for x in nums if x in live_set)
            if df == 0:
                continue
            idf = math.log(1 + (n_live - df + 0.5) / (df + 0.5))
            scored.append((term, idf * sum(tf for x, tf in zip(nums, tfs) if x in target)))
        return sorted(scored, key=lambda s: -s[1])[:n]

    def bridge_terms(self, passage_ids: list[str], n: int = 8, max_df: int = 30) -> list[tuple[str, float]]:
        """Terms that can lead from these passages to others: they occur in these passages AND in at least
        one other passage, at most max_df passages in all (rarer is better: scored by idf alone), and are
        not numbers or shorter than 3 characters. Reads every postings row, like rare_terms."""
        q = ", ".join(_lit(p) for p in passage_ids)
        ids = self._passages.query().where(f"passage_id IN ({q})").limit(None).select(["num"]).to_arrow()
        target = set(ids.column("num").to_pylist())
        num, _, deleted = self._meta()
        n_live = int((~deleted).sum())
        live_set = set(num[~deleted].tolist())
        t = self._postings.query().limit(None).select(["term", "nums"]).to_arrow()
        acc: dict[str, set] = defaultdict(set)
        for term, nums in zip(t.column("term").to_pylist(), t.column("nums").to_pylist()):
            if len(term) >= 3 and not any(c.isdigit() for c in term):
                acc[term].update(x for x in nums if x in live_set)
        scored = []
        for term, ps in acc.items():
            inside = ps & target
            if inside and len(ps) > len(inside) and len(ps) <= max_df:
                scored.append((term, math.log(1 + (n_live - len(ps) + 0.5) / (len(ps) + 0.5))))
        return sorted(scored, key=lambda s: -s[1])[:n]

    def compact(self) -> None:
        """Rewrite postings as one segment per term, without deleted passages, and drop those passage rows."""
        self.flush()
        acc = self.postings(live_only=True)
        rows = [{"term": term, "segment": 0, "nums": n.tolist(), "tfs": f.tolist()} for term, (n, f) in acc.items()]
        self._postings = self._db.create_table(POSTINGS, data=pa.Table.from_pylist(rows, schema=POSTING_SCHEMA),
                                               mode="overwrite")
        self._passages.delete("deleted = true")
        self._segment = 1
        if rows:
            _index(self._postings, "term")

    def tidy(self) -> None:
        """Compact both tables' files and drop their old versions. The keyword tables are derived from
        the passages, so their version history is not kept (the graph and vector tables keep theirs)."""
        import datetime

        for t in (self._passages, self._postings):
            t.compact_files()
            t.cleanup_old_versions(datetime.timedelta(0))

    def stats(self) -> dict:
        num, _, deleted = self._meta()
        postings = self._postings.query().limit(None).select(["term", "nums"]).to_arrow()
        import pyarrow.compute as pc

        entries = pc.sum(pc.list_value_length(postings.column("nums"))).as_py() or 0
        return {"passages": int((~deleted).sum()), "deleted_rows": int(deleted.sum()),
                "distinct_terms": len(set(postings.column("term").to_pylist())),
                "postings_rows": postings.num_rows, "postings_entries": int(entries),
                "bytes": {PASSAGES: _disk_bytes(self._passages), POSTINGS: _disk_bytes(self._postings)},
                "versions": {PASSAGES: len(self._passages.list_versions()),
                             POSTINGS: len(self._postings.list_versions())}}


async def _show(hits: list[tuple[str, float]]) -> None:
    from . import store

    graph, _ = await store.engines()
    nodes = {n["id"]: store.node_props(n) for n in await graph.get_nodes([p for p, _ in hits])} if hits else {}
    if not hits:
        print("no hits")
    for pid, score in hits:
        p = nodes.get(pid, {})
        print(f"{score:8.3f}  {pid}  {p.get('path', '?')}  {p.get('name', '?')}  line {p.get('line', '?')}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docmem kwsearch",
                                 description="BM25 keyword search over an index's passages (no vectors, no LLM)")
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("query")
    a = ap.parse_args(argv)
    from . import store

    store.configure(a.db)
    hits = KeywordIndex(a.db).search(a.query, k=a.k)
    asyncio.run(_show(hits))


if __name__ == "__main__":
    main()
