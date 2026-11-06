"""A stand-in OpenAI-compatible server for testing docmem without a real one: /v1/embeddings with
fastembed, and /v1/chat/completions that answers extractively (the excerpt sentence sharing most words
with the question), ranks excerpts by word overlap, and judges by word overlap. Not a model: it
only exercises the HTTP paths.

    python tools/mock_openai.py --port 8765
"""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORD = re.compile(r"[a-z0-9]{3,}")
_embedders = {}


def words(t):
    return set(WORD.findall(t.lower()))


def excerpt_blocks(text):
    return re.split(r"\n\n(?=\[\d+\] )", text.split("Excerpts:\n\n", 1)[-1].split("\n\nQuestion:")[0])


def chat(system, user):
    q = re.search(r"Question: (.*)", user)
    qw = words(q.group(1) if q else "")
    if system.startswith("You rank"):
        return json.dumps({"scores": [min(3, len(qw & words(b)) // 2) for b in excerpt_blocks(user)]})
    if system.startswith("You grade"):
        ref = re.search(r"Reference: (.*)", user).group(1)
        got = re.search(r"Answer: (.*)", user, re.S).group(1)
        share = len(words(ref) & words(got)) / max(len(words(ref)), 1)
        return json.dumps({"verdict": "correct" if share > 0.6 else "partial" if share > 0.3 else "wrong",
                           "why": f"word overlap {share:.2f}"})
    if system.startswith("You extract"):
        secs = re.split(r"\n## S\d+: ", user)[1:]
        caps = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+")
        ents, n, summaries = {}, 0, []
        for k, sec in enumerate(secs, 1):
            parts = re.split(r"\n\[P\d+\] ", "\n" + sec.split("\n", 1)[-1])[1:]
            summaries.append({"s": k, "summary": (re.split(r"(?<=[.!?])\s", parts[0]) if parts else [""])[0][:200]})
            for p in parts:
                n += 1
                for m in caps.findall(p)[:3]:
                    ents.setdefault(m, []).append(n)
        el = [{"name": e, "type": "other", "description": "", "p": ps} for e, ps in list(ents.items())[:12]]
        rel = [{"source": el[0]["name"], "relation": "related to", "target": el[1]["name"], "p": el[0]["p"][0]}] if len(el) > 1 else []
        return json.dumps({"summaries": summaries, "entities": el, "relations": rel, "topics": ["mock topic"]})
    best, score = None, 1
    for i, b in enumerate(excerpt_blocks(user), 1):
        for s in re.split(r"(?<=[.!?])\s+", b.split("\n", 1)[-1]):
            n = len(qw & words(s))
            if n > score:
                best, score = f"{s.strip()} [{i}]", n
    return best or "NOT FOUND: the excerpts do not mention it."


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            from fastembed import TextEmbedding

            m = body["model"]
            e = _embedders.setdefault(m, TextEmbedding(m))
            inp = body["input"] if isinstance(body["input"], list) else [body["input"]]
            data = [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(e.embed(inp))]
            out = {"object": "list", "data": data, "model": m, "usage": {"prompt_tokens": 0, "total_tokens": 0}}
        else:
            msgs = {m["role"]: m["content"] for m in body["messages"]}
            text = chat(msgs.get("system", ""), msgs.get("user", ""))
            out = {"id": "mock", "object": "chat.completion", "model": body["model"], "created": 0,
                   "choices": [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant", "content": text}}],
                   "usage": {"prompt_tokens": len(msgs.get("user", "")) // 4, "completion_tokens": len(text) // 4,
                             "total_tokens": 0}}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
