"""Enrichment: while a document's text passes through the indexer (the one time it is read), the chat
model writes section summaries, entities, relations and topics from it. They are stored; the text is not.

One call covers a group of consecutive sections packed up to --enrich-words; a section longer than that
is split over several calls (one summary per part). Only the sections of a document that hold a newly
embedded passage are enriched, so an edit costs the calls for the sections it touched.

When a call fails twice (bad JSON, server error) or the run's budget (--enrich-max-calls,
--enrich-max-tokens) is spent, the section group is recorded as a gap (a DocGap node): the source is
not read again for it now; ``docmem refresh`` closes all of a document's gaps in one later read.
"""
import asyncio
import json
import os
import re

SYSTEM = (
    "You extract a compact knowledge record from document excerpts. Reply with one JSON object only, no "
    "prose, in this shape:\n"
    '{"summaries": [{"s": <section number>, "summary": "<one or two sentences>"}],\n'
    ' "entities": [{"name": "<canonical name>", "type": "<person|place|organization|event|work|concept|'
    'object|date|other>", "description": "<a few words, from the text>", "p": [<passage numbers>]}],\n'
    ' "relations": [{"source": "<entity name>", "relation": "<verb phrase, snake_case>", "target": '
    '"<entity name>", "p": <passage number>}],\n'
    ' "topics": ["<2 to 5 broad subjects of the document>"]}\n'
    "One summary per section number. Only entities that matter to the text (at most 8), only relations "
    "stated in it between those entities (at most 6). Descriptions of at most 6 words. Use only the "
    "excerpts. Keep the reply short: its length is the main cost.")

WORD = re.compile(r"\S+")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(s)).strip("_")[:60] or "related_to"


def groups(sections_of_passages, passages, words: int, wanted: set[int]):
    """Pack passages (indices into ``passages``) into call groups: consecutive sections, whole when they
    fit ``words``, and only sections containing an index in ``wanted``. Yields lists of
    (section_path, part, [passage indices])."""
    by_section = []
    for i, p in enumerate(passages):
        path = sections_of_passages[i]
        if by_section and by_section[-1][0] == path:
            by_section[-1][1].append(i)
        else:
            by_section.append((path, [i]))
    cur, cur_words = [], 0
    for path, idx in by_section:
        if not wanted.intersection(idx):
            continue
        part, piece, piece_words = 0, [], 0
        for i in idx:
            w = len(WORD.findall(passages[i]["text"]))
            if piece and piece_words + w > words:
                if cur:
                    yield cur
                    cur, cur_words = [], 0
                yield [(path, part, piece)]
                part, piece, piece_words = part + 1, [], 0
            piece.append(i)
            piece_words += w
        if cur and cur_words + piece_words > words:
            yield cur
            cur, cur_words = [], 0
        cur.append((path, part, piece))
        cur_words += piece_words
    if cur:
        yield cur


def prompt(title: str, group, passages) -> str:
    out, n = [f"Document: {title}"], 0
    for s, (path, part, idx) in enumerate(group, 1):
        out.append(f"\n## S{s}: {path}" + (f" (part {part + 1})" if part else ""))
        for i in idx:
            n += 1
            out.append(f"[P{n}] {passages[i]['text']}")
    return "\n".join(out)


def repair_json(text: str):
    """The longest complete prefix of a JSON object cut off mid-reply (max_tokens), closed: the entries
    written in full are kept, the one being written is dropped. None if nothing parses."""
    start = text.find("{")
    if start < 0:
        return None
    stack, in_str, esc, cuts = [], False, False, []
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if stack:
                stack.pop()
            cuts.append((i + 1, "".join(reversed(stack))))
            if not stack:
                break
    for end, closers in reversed(cuts):
        try:
            return json.loads(text[start:end] + closers)
        except json.JSONDecodeError:
            continue
    return None


class Budget:
    def __init__(self, max_calls=None, max_tokens=None):
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.calls = self.prompt_tokens = self.completion_tokens = self.failed = self.in_flight = 0
        self.salvaged = 0

    def spent(self) -> bool:
        """Whether another call would go over: calls in flight count against the call budget."""
        return (self.max_calls is not None and self.calls + self.in_flight >= self.max_calls) or (
            self.max_tokens is not None and self.prompt_tokens + self.completion_tokens >= self.max_tokens)

    def as_dict(self):
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "failed_groups": self.failed,
                "salvaged_replies": self.salvaged}


class Enricher:
    def __init__(self, api_base, api_key, model, concurrency=2, budget=None, json_mode=True,
                 max_tokens=500):
        from openai import AsyncOpenAI

        api_base = api_base or os.environ.get("DOCMEM_API_BASE")
        self.model = model or os.environ.get("DOCMEM_MODEL")
        if not api_base or not self.model:
            raise SystemExit("docmem: --enrich needs --api-base and --model (or DOCMEM_API_BASE, DOCMEM_MODEL)")
        self.client = AsyncOpenAI(base_url=api_base, api_key=api_key or os.environ.get("DOCMEM_API_KEY") or "none",
                                  timeout=600)
        self.sem = asyncio.Semaphore(concurrency)
        self.budget = budget or Budget()
        self.json_mode, self.max_tokens = json_mode, max_tokens
        self.version = f"enrich/1:{self.model}"

    async def call(self, user: str):
        kw = {"response_format": {"type": "json_object"}} if self.json_mode else {}
        self.budget.in_flight += 1
        try:
            r = await self.client.chat.completions.create(
                model=self.model, temperature=0, max_tokens=self.max_tokens,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], **kw)
        finally:
            self.budget.in_flight -= 1
        self.budget.calls += 1
        if r.usage:
            self.budget.prompt_tokens += r.usage.prompt_tokens or 0
            self.budget.completion_tokens += r.usage.completion_tokens or 0
        text = r.choices[0].message.content or ""
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            pass
        # cut off at max_tokens (or a stray character): keep the entries written in full
        obj = repair_json(text)
        if obj is None:
            raise ValueError(f"no JSON in reply ({r.choices[0].finish_reason}): {text[:120]!r}")
        self.budget.salvaged += 1
        return obj

    async def group(self, title, group, passages):
        """(record, None) or (None, reason)."""
        async with self.sem:
            for attempt in range(2):
                if self.budget.spent():
                    return None, "budget"
                try:
                    obj = await self.call(prompt(title, group, passages))
                    return parse(obj or {}, group), None
                except ValueError as e:  # unusable reply: at temperature 0 a retry repeats it
                    reason = f"{type(e).__name__}: {e}"[:200]
                    break
                except Exception as e:  # a server or network error: one retry, then a gap
                    reason = f"{type(e).__name__}: {e}"[:200]
            self.budget.failed += 1
            return None, reason


def parse(obj: dict, group):
    """Map the reply's section and passage numbers back to (section path, part) and passage indices."""
    flat = [i for _, _, idx in group for i in idx]

    def pas(ns):
        ns = ns if isinstance(ns, list) else [ns]
        return sorted({flat[int(n) - 1] for n in ns if str(n).strip().lstrip("P").isdigit()
                       and 1 <= int(str(n).strip().lstrip("P")) <= len(flat)
                       for n in [str(n).strip().lstrip("P")]})

    out = {"summaries": [], "entities": [], "relations": [], "topics": []}
    for s in obj.get("summaries") or []:
        try:
            k = int(str(s.get("s")).lstrip("S")) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= k < len(group) and str(s.get("summary", "")).strip():
            out["summaries"].append({"path": group[k][0], "part": group[k][1], "text": str(s["summary"]).strip()})
    for e in obj.get("entities") or []:
        if isinstance(e, dict) and str(e.get("name", "")).strip():
            out["entities"].append({"name": str(e["name"]).strip()[:200], "type": norm(e.get("type") or "other")[:40],
                                    "description": str(e.get("description") or "").strip()[:400],
                                    "passages": pas(e.get("p") or [])})
    for r in obj.get("relations") or []:
        if isinstance(r, dict) and r.get("source") and r.get("target"):
            out["relations"].append({"source": str(r["source"]).strip(), "relation": snake(r.get("relation") or ""),
                                     "target": str(r["target"]).strip(), "passages": pas(r.get("p") or [])})
    out["topics"] = [str(t).strip()[:80] for t in (obj.get("topics") or []) if str(t).strip()][:8]
    return out
