#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible local LLM server.

Measures, against --api-base:
  1. Prompt processing: a max_tokens=1 completion over the excerpt prompt.
  2. Full completion: a max_tokens=N summary of the same excerpt (temperature 0).
     Generation time = full wall time - prompt processing time.
  3. Structured output: one passage -> {summary, entities, relations} JSON, with
     response_format json_object and json_schema; reports whether each parsed.

Each request starts with a unique nonce so the server cannot reuse the KV cache of
an earlier request; without it the prompt-processing time would be too small.

Usage: bench_llm.py [--api-base http://127.0.0.1:8080/v1] [--model NAME]
"""

import argparse
import json
import time
import uuid

from openai import OpenAI

DEFAULT_CORPUS = (
    "/tmp/claude-0/-home-user/8df192dc-e0a5-50ec-ba29-4d1c2c18b24f/"
    "scratchpad/dm/corpus/book02-Amphetamine.md"
)

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "relation": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["source", "relation", "target"],
            },
        },
    },
    "required": ["summary", "entities", "relations"],
}

STRUCTURED_INSTRUCTIONS = (
    "Extract a small knowledge graph from the passage below. Return a JSON object with "
    "keys: summary (string), entities (list of objects with name, type, description), "
    "relations (list of objects with source, relation, target). Use entity names exactly "
    "as they appear in the passage.\n\nPassage:\n"
)


def summary_messages(excerpt: str, nonce: str) -> list[dict]:
    content = (
        f"[request {nonce}] Summarize the following text in exactly 2 sentences.\n\n"
        f"<text>\n{excerpt}\n</text>"
    )
    return [{"role": "user", "content": content}]


def timed_completion(client: OpenAI, model: str, messages: list[dict], max_tokens: int, **kw):
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=0, **kw
    )
    return resp, time.perf_counter() - start


def bench_speed(client: OpenAI, model: str, excerpt: str, max_tokens: int) -> dict:
    prefill_resp, prefill_s = timed_completion(
        client, model, summary_messages(excerpt, uuid.uuid4().hex), max_tokens=1
    )
    full_resp, full_s = timed_completion(
        client, model, summary_messages(excerpt, uuid.uuid4().hex), max_tokens=max_tokens
    )
    prompt_tokens = full_resp.usage.prompt_tokens
    completion_tokens = full_resp.usage.completion_tokens
    gen_s = full_s - prefill_s
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": full_resp.choices[0].finish_reason,
        "prefill_only_wall_s": prefill_s,
        "prefill_only_prompt_tokens": prefill_resp.usage.prompt_tokens,
        "full_wall_s": full_s,
        "prompt_tok_per_s": prompt_tokens / prefill_s,
        "gen_tok_per_s": completion_tokens / gen_s if gen_s > 0 else float("nan"),
        "overall_tok_per_s": (prompt_tokens + completion_tokens) / full_s,
        "summary": full_resp.choices[0].message.content,
    }


def bench_structured(client: OpenAI, model: str, passage: str, mode: str) -> dict:
    messages = [{"role": "user", "content": STRUCTURED_INSTRUCTIONS + passage}]
    if mode == "json_object":
        response_format = {"type": "json_object"}
    else:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "graph_extract", "schema": SCHEMA},
        }
    try:
        resp, wall_s = timed_completion(
            client, model, messages, max_tokens=1024, response_format=response_format
        )
    except Exception as exc:  # report unsupported modes instead of aborting the run
        return {"mode": mode, "error": f"{type(exc).__name__}: {exc}"}
    content = resp.choices[0].message.content or ""
    result = {
        "mode": mode,
        "wall_s": wall_s,
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "finish_reason": resp.choices[0].finish_reason,
        "raw": content,
    }
    try:
        parsed = json.loads(content)
        result["parsed"] = True
        result["top_level_keys"] = sorted(parsed) if isinstance(parsed, dict) else type(parsed).__name__
    except json.JSONDecodeError as exc:
        result["parsed"] = False
        result["parse_error"] = str(exc)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="gemma-4-E2B-it")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--chars", type=int, default=6000, help="excerpt length in characters")
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--passage-words", type=int, default=200)
    args = ap.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        text = f.read()
    excerpt = text[: args.chars]
    passage = " ".join(text[args.chars :].split()[: args.passage_words])

    client = OpenAI(base_url=args.api_base, api_key="not-needed")

    print(f"api_base={args.api_base} model={args.model}")
    print(f"excerpt: {len(excerpt)} chars from {args.corpus}")
    print("\n== speed ==")
    speed = bench_speed(client, args.model, excerpt, args.max_tokens)
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "finish_reason",
        "prefill_only_wall_s",
        "full_wall_s",
        "prompt_tok_per_s",
        "gen_tok_per_s",
        "overall_tok_per_s",
    ):
        value = speed[key]
        print(f"{key:24s} {value:.2f}" if isinstance(value, float) else f"{key:24s} {value}")
    print("summary:\n" + (speed["summary"] or "").strip())

    print(f"\n== structured output ({len(passage.split())} words) ==")
    for mode in ("json_object", "json_schema"):
        out = bench_structured(client, args.model, passage, mode)
        print(f"\n-- response_format={mode} --")
        if "error" in out:
            print(f"request failed: {out['error']}")
            continue
        print(
            f"wall_s={out['wall_s']:.2f} prompt_tokens={out['prompt_tokens']} "
            f"completion_tokens={out['completion_tokens']} finish_reason={out['finish_reason']}"
        )
        print(f"parsed_as_json={out['parsed']}", end="")
        if out["parsed"]:
            print(f" top_level_keys={out['top_level_keys']}")
        else:
            print(f" ({out['parse_error']})")
        print("raw output:")
        print(out["raw"].strip())


if __name__ == "__main__":
    main()
