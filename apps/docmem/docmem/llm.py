"""The OpenAI-compatible chat endpoint the user names: --api-base, --api-key, --model (or DOCMEM_API_BASE,
DOCMEM_API_KEY, DOCMEM_MODEL). Nothing here is written to disk."""
import json
import os
import re


class Chat:
    def __init__(self, api_base=None, api_key=None, model=None):
        from openai import OpenAI

        self.api_base = api_base or os.environ.get("DOCMEM_API_BASE")
        self.model = model or os.environ.get("DOCMEM_MODEL")
        if not self.api_base or not self.model:
            raise SystemExit("docmem: answering needs --api-base and --model (or DOCMEM_API_BASE, DOCMEM_MODEL)")
        self.client = OpenAI(base_url=self.api_base, api_key=api_key or os.environ.get("DOCMEM_API_KEY") or "none")
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        r = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        self.usage["calls"] += 1
        if r.usage:
            self.usage["prompt_tokens"] += r.usage.prompt_tokens or 0
            self.usage["completion_tokens"] += r.usage.completion_tokens or 0
        return (r.choices[0].message.content or "").strip()

    def complete_json(self, system: str, user: str, max_tokens: int = 400):
        text = self.complete(system, user, max_tokens)
        m = re.search(r"\{.*\}|\[.*\]", text, re.S)
        if not m:
            raise ValueError(f"the model did not answer with JSON: {text[:200]!r}")
        return json.loads(m.group(0))
