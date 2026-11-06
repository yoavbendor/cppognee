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


def source_rank(hits, src) -> int | None:
    for h in hits:
        if source_name(h) == src["file"] and (
                not src.get("heading") or src["heading"].strip().lower() in
                [p.strip().lower() for p in h["path"].split("›")]):
            return h["rank"]
    return None


async def run(a):
    chat = ask.open_db(a)
    qs = [json.loads(line) for line in open(a.questions, encoding="utf-8") if line.strip()]
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
               "seconds": round(time.time() - t, 2),
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
        print(f"{q['id']}: {status}{extra}  ({row['seconds']}s)", flush=True)
    return rows, (chat.usage if chat else None)


def summarize(rows, a, usage):
    ans = [r for r in rows if r["answerable"]]
    s = {"questions": len(rows), "answerable": len(ans), "k": a.k, "context": a.context, "rerank": a.rerank}
    for k in sorted({1, 3, a.k}):
        s[f"hit@{k}"] = round(sum(1 for r in ans if r["first_rank"] and r["first_rank"] <= k) / max(len(ans), 1), 3)
    s["mrr"] = round(sum(1 / r["first_rank"] for r in ans if r["first_rank"]) / max(len(ans), 1), 3)
    for kind in sorted({r["kind"] for r in ans}):
        rs = [r for r in ans if r["kind"] == kind]
        s[f"{kind}: all sources in top {a.k}"] = f"{sum(r['all_sources'] for r in rs)}/{len(rs)}"
        if "answer" in rows[0]:
            s[f"{kind}: contains must_include"] = f"{sum(r['contains'] for r in rs)}/{len(rs)}"
    s["retrieval_ms_median"] = sorted(r["timing"]["total_ms"] for r in rows)[len(rows) // 2]
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
