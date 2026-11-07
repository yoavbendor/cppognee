"""Eval: run a question set against the index and score retrieval and answers.

Questions file, one JSON object per line:
  {"id": "q01", "question": "...", "answer": "reference answer",
   "sources": [{"file": "book.md", "heading": "Section heading"}],   # [] = not answerable from the corpus
   "must_include": ["1918", "Kabul"]}

Scores per question:
  retrieval  rank of the first hit in a source file whose heading path contains the source heading
             (hit@k, MRR); for multi-source questions also how many sources were found
  contains   the answer has every must_include string (case-insensitive)
  refusal    the answer starts with NOT FOUND (correct exactly for the unanswerable questions)
  judge      with --judge, the chat model grades the answer against the reference: correct, partial, wrong

    docmem eval --db ./kb --questions questions.jsonl --api-base ... --model ... --judge --out report/
"""
import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

from . import ask, store

JUDGE_SYSTEM = (
    "You grade an answer against a reference answer for the same question. Reply with JSON only: "
    '{"verdict": "correct" | "partial" | "wrong", "why": "<one short sentence>"}. "correct": it states '
    "the reference's facts (wording may differ, extra correct detail is fine); \"partial\": some of them, "
    "or hedged; \"wrong\": contradicts or misses them. If the reference says the corpus does not cover "
    "the question, an answer that says so is correct and any factual answer is wrong.")


def source_name(h) -> str:
    """What a question's "file" names: a local file's name, or a web source's title (top of its path)."""
    return store.uri_path(h["uri"]).name if h["uri"].startswith("file:") else h["path"].split(" › ")[0]


def heading_in(path: str, heading: str) -> bool:
    """The heading is one segment of the path (case-insensitive). A model item's path segments are its labels
    ("model › apollo11 › package P › part def ApolloLunarModule"), so its heading is e.g. "part def ApolloLunarModule"."""
    return heading.strip().lower() in [p.strip().lower() for p in path.split("›")]


def source_rank(hits, src) -> int | None:
    """The rank of the first hit (prose or model, aligned or retrieved) in the source's file whose heading path has
    the source's heading; None when no hit does. Hits are the whole context, not only the top k."""
    for h in hits:
        if source_name(h) == src["file"] and (not src.get("heading") or heading_in(h["path"], src["heading"])):
            return h["rank"]
    return None


async def run(a):
    chat = ask.open_db(a)
    qs = [json.loads(line) for line in open(a.questions, encoding="utf-8") if line.strip()]
    if a.ids:
        wanted = a.ids.split(",")
        missing = set(wanted) - {q["id"] for q in qs}
        if missing:
            raise SystemExit(f"docmem eval: no such question ids: {', '.join(sorted(missing))}")
        qs = [q for q in qs if q["id"] in set(wanted)]
    rows = []
    for q in qs:
        t = time.time()
        r = await ask.ask_one(q["question"], a, chat)
        hits = r["hits"]
        ranks = [source_rank(hits, s) for s in q.get("sources", [])]
        found = [x for x in ranks if x]
        row = {"id": q["id"], "kind": q.get("kind") or ("single" if len(ranks) == 1 else
                                                          "two-source" if ranks else "unanswerable"),
               "all_sources": bool(ranks) and len(found) == len(ranks),
               "question": q["question"], "answerable": bool(q.get("sources")),
               "first_rank": min(found) if found else None, "sources_found": f"{len(found)}/{len(ranks)}",
               "stale_hits": sum(h["status"] != "ok" for h in hits), "timing": r["timing"],
               "seconds": round(time.time() - t, 2), "searches": r["searches"], "calls": r["calls"],
               "linked": r["linked"], "pivot_concepts": r["pivot_concepts"],
               "aligned_found_by": dict(Counter(h["found_by"] for h in hits if h.get("found_by"))),
               "hits": [ask.label(h) for h in hits]}
        if chat:
            ans = r["answer"]
            row["answer"] = ans
            row["refused"] = ans.upper().startswith("NOT FOUND")
            row["contains"] = all(m.lower() in ans.lower() for m in q.get("must_include", [])) and \
                (not row["answerable"] or not row["refused"])
            if a.judge:
                v = chat.complete_json(JUDGE_SYSTEM, f"Question: {q['question']}\nReference: {q['answer']}\n"
                                                     f"Answer: {ans}")
                row["judge"], row["judge_why"] = v.get("verdict"), v.get("why")
        rows.append(row)
        status = f"rank {row['first_rank']}" if row["answerable"] else "unanswerable"
        extra = f"  judge={row.get('judge')}" if a.judge else (f"  contains={row.get('contains')}" if chat else "")
        hop = f"  searches={row['searches']}" if a.hops else ""
        print(f"{q['id']}: {status}{extra}{hop}  ({row['seconds']}s)", flush=True)
    return rows, (chat.usage if chat else None)


def summarize(rows, a, usage):
    ans = [r for r in rows if r["answerable"]]
    s = {"questions": len(rows), "answerable": len(ans), "k": a.k, "context": a.context, "rerank": a.rerank,
         "hops": a.hops, "links": a.links, "terms": a.terms, "concepts": a.concepts, "pivot": a.pivot}
    if a.hops:
        s.update({"hop_k": a.hop_k, "max_passages": a.max_passages})
    if a.with_model or a.with_prose:
        s.update({"with_model": str(a.with_model or ""), "with_prose": str(a.with_prose or ""), "aligned": a.aligned})
    if a.pivot:
        s.update({"pivot_scheme": a.pivot_scheme or "all", "pivot_concepts_per_question": a.pivot_concepts})
    for k in sorted({1, 3, a.k}):
        s[f"hit@{k}"] = round(sum(1 for r in ans if r["first_rank"] and r["first_rank"] <= k) / max(len(ans), 1), 3)
    s["mrr"] = round(sum(1 / r["first_rank"] for r in ans if r["first_rank"]) / max(len(ans), 1), 3)
    gathered = ("all sources gathered" if (a.hops or a.links or a.terms or a.split or a.concepts or a.pivot
                                           or a.with_model or a.with_prose)
                else f"all sources in top {a.k}")
    for kind in sorted({r["kind"] for r in ans}):
        rs = [r for r in ans if r["kind"] == kind]
        s[f"{kind}: {gathered}"] = f"{sum(r['all_sources'] for r in rs)}/{len(rs)}"
        if "answer" in rows[0]:
            s[f"{kind}: contains must_include"] = f"{sum(r['contains'] for r in rs)}/{len(rs)}"
    s["retrieval_ms_median"] = sorted(r["timing"]["total_ms"] for r in rows)[len(rows) // 2]
    s["llm_calls_per_question"] = round(sum(r["calls"] for r in rows) / max(len(rows), 1), 2)
    if "answer" in rows[0]:
        s["contains_must_include"] = round(sum(r["contains"] for r in ans) / max(len(ans), 1), 3)
        un = [r for r in rows if not r["answerable"]]
        s["refused_unanswerable"] = f"{sum(r['refused'] for r in un)}/{len(un)}"
        s["refused_answerable"] = f"{sum(r['refused'] for r in ans)}/{len(ans)}"
        s["seconds_median"] = sorted(r["seconds"] for r in rows)[len(rows) // 2]
        s["usage"] = usage
    if a.judge:
        for v in ("correct", "partial", "wrong"):
            s[f"judge_{v}"] = sum(r.get("judge") == v for r in rows)
    return s


def report_md(s, rows) -> str:
    out = ["# docmem eval", "", "| metric | value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in s.items()]
    out += ["", "| id | kind | rank | sources | judge | contains | answer |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        ans = (r.get("answer") or "").replace("\n", " ").replace("|", "\\|")[:160]
        out.append(f"| {r['id']} | {r['kind']} | {r['first_rank'] if r['answerable'] else '—'} | {r['sources_found']} | "
                   f"{r.get('judge', '')} | {r.get('contains', '')} | {ans} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docmem eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_common(ap)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--judge", action="store_true", help="grade answers with the chat model")
    ap.add_argument("--ids", help="comma-separated question ids to run, e.g. m01,w02 (default: all)")
    ap.add_argument("--out", type=Path, help="write results.jsonl, summary.json and report.md here")
    a = ap.parse_args(argv)
    rows, usage = asyncio.run(run(a))
    s = summarize(rows, a, usage)
    print(json.dumps(s, indent=1))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        (a.out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
        (a.out / "report.md").write_text(report_md(s, rows))
