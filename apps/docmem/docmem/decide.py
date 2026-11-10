"""Decider: a client for System One-style decision models, with an on-disk cache of their answers.

Two servers take the same question shape:

* foundry: Microsoft Decision-1 on Azure AI Foundry, POST <URL>/providers/microsoft/v1/systemone.
  Its answer field names are not documented, so the parser tries known names and says what it found.
* kev: a Kev server or TypeSafe System One, POST <URL>/v1/systemone.

Configure with environment variables, or with the flags of `docmem decide-check`:

    DOCMEM_DECIDER_URL     base URL of the server
    DOCMEM_DECIDER_KEY     its key (kev: "local" when unset; foundry: the api-key)
    DOCMEM_DECIDER_MODEL   the model (foundry: the deployment name)
    DOCMEM_DECIDER_FLAVOR  foundry | kev (default: foundry when the URL names Azure, else kev)
    DOCMEM_DECIDER_AUTH    foundry only: key (default) | entra (a Microsoft Entra token, from azure-identity)

    from docmem.decide import Decider
    d = Decider.from_env(cache=Path("kb/decider_cache.jsonl"))
    d.decide(state, {"safe": {"type": "noul", "instructions": "Is the state safe?"}})["safe"].p_yes

The cache holds hashes and results only: the key is sha256 of model, state and question, and the state
text is never written. Errors never include the key.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path

import cognee_egress

FLAVORS = ("foundry", "kev")
AUTHS = ("key", "entra")
QUESTION_TYPES = ("noul", "choice", "score")
QUESTION_FIELDS = ("type", "instructions", "criteria")
AZURE_HINTS = ("azure", "cognitiveservices", "ai.azure.com")
ENTRA_SCOPE = "https://cognitiveservices.azure.com/.default"
MAX_ATTEMPTS = 4
# Foundry's answer fields are undocumented: these are the names tried, in order.
FOUNDRY_NOUL_KEYS = ("probability", "noul", "p", "value")
FOUNDRY_DIST_KEYS = ("probabilities", "distribution")
FOUNDRY_SCORE_KEYS = ("score", "mean")
_foundry_noted = False


class DeciderError(Exception):
    """A configuration, transport or server error. The message says what to fix; it never contains the key."""


class DeciderResponseError(DeciderError):
    """The server answered, but not with the fields this client reads."""


def default_flavor(url: str) -> str:
    low = url.lower()
    return "foundry" if any(h in low for h in AZURE_HINTS) else "kev"


@dataclass(frozen=True)
class Config:
    url: str
    model: str
    key: str | None
    flavor: str
    auth: str = "key"

    @classmethod
    def from_env(cls, env=None, url=None, key=None, model=None, flavor=None, auth=None) -> "Config":
        """Each argument, when given, overrides its DOCMEM_DECIDER_* variable."""
        env = os.environ if env is None else env

        def pick(value, name):
            return value if value is not None else (env.get(name) or None)

        url = pick(url, "DOCMEM_DECIDER_URL")
        if not url:
            raise DeciderError("set DOCMEM_DECIDER_URL (or --url)")
        model = pick(model, "DOCMEM_DECIDER_MODEL")
        if not model:
            raise DeciderError("set DOCMEM_DECIDER_MODEL (or --model)")
        flavor = pick(flavor, "DOCMEM_DECIDER_FLAVOR") or default_flavor(url)
        if flavor not in FLAVORS:
            raise DeciderError(f"DOCMEM_DECIDER_FLAVOR must be one of {', '.join(FLAVORS)}, not {flavor!r}")
        auth = pick(auth, "DOCMEM_DECIDER_AUTH") or "key"
        if auth not in AUTHS:
            raise DeciderError(f"DOCMEM_DECIDER_AUTH must be key or entra, not {auth!r}")
        if auth == "entra" and flavor != "foundry":
            raise DeciderError("DOCMEM_DECIDER_AUTH=entra applies to the foundry flavor only")
        key = pick(key, "DOCMEM_DECIDER_KEY")
        if key is None and flavor == "kev":
            key = "local"
        if key is None and auth == "key":
            raise DeciderError("set DOCMEM_DECIDER_KEY (or --key), or DOCMEM_DECIDER_AUTH=entra")
        return cls(url=url, model=model, key=key, flavor=flavor, auth=auth)

    @property
    def endpoint(self) -> str:
        base = self.url.rstrip("/")
        if base.endswith("/systemone"):
            return base
        return base + ("/providers/microsoft/v1/systemone" if self.flavor == "foundry" else "/v1/systemone")


@dataclass
class Decision:
    name: str
    type: str
    p_yes: float | None = None           # noul: the probability the answer is yes
    probabilities: dict | None = None    # choice and score: label -> probability
    score: float | None = None           # score: the expected score
    choice: str | None = None            # choice: the chosen label
    confidence: float | None = None
    raw: object = None                   # the server's answer for this question, as received

    def fields(self) -> dict:
        return {k: getattr(self, k) for k in ("type", "p_yes", "probabilities", "score", "choice", "confidence", "raw")}


def cache_key(model: str, state: str, question: dict) -> str:
    text = "\0".join([model, state, json.dumps(question, sort_keys=True, ensure_ascii=False)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(name: str, question: dict) -> dict:
    if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
        raise DeciderError(f"question {name!r}: type must be one of {', '.join(QUESTION_TYPES)}")
    unknown = set(question) - set(QUESTION_FIELDS)
    if unknown:
        raise DeciderError(f"question {name!r}: unknown fields {sorted(unknown)} (use {', '.join(QUESTION_FIELDS)})")
    return {k: question[k] for k in QUESTION_FIELDS if k in question}


def _retry_after(headers) -> float | None:
    value = headers.get("Retry-After") if headers else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return max(0.0, (parsedate_to_datetime(value).timestamp() - time.time()))


def _num(x, low=None, high=None) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if low is not None and not low <= x <= high:
        raise DeciderResponseError(f"{x} is outside [{low}, {high}]: the field read is not a probability")
    return float(x)


def _dist(probs, where: str) -> dict:
    if not isinstance(probs, dict) or not probs:
        raise DeciderResponseError(f"{where}: expected a non-empty label -> probability object")
    return {str(k): _num(v, 0, 1) for k, v in probs.items()}


def _field(obj, key: str, where: str):
    if not isinstance(obj, dict) or key not in obj:
        fields = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
        raise DeciderResponseError(f"{where} has no {key!r} field; fields: {fields}")
    return obj[key]


def _describe(obj) -> object:
    return sorted(obj) if isinstance(obj, dict) else type(obj).__name__


def _parse_answer(name: str, ans, qtype: str) -> Decision:
    """One question's answer in the "answers" shape both servers return: noul as {"noul": p} (Kev) or one of
    the names tried for Decision-1; choice as {"choice", "probabilities"}; score as {"score", "probabilities"}."""
    where = f"answers[{name!r}]"
    if qtype == "noul" and _num(ans, 0, 1) is not None:
        return Decision(name, qtype, p_yes=_num(ans, 0, 1), raw=ans)
    if not isinstance(ans, dict):
        raise DeciderResponseError(f"{where} is {type(ans).__name__}, not an object")
    conf = _num(ans.get("confidence"))
    if qtype == "noul":
        p = _num(ans.get("noul"), 0, 1)
        if p is None:
            p = next((_num(ans[k], 0, 1) for k in FOUNDRY_NOUL_KEYS if _num(ans.get(k)) is not None), None)
        if p is None:
            raise DeciderResponseError(f"{where} has no probability field (tried noul, {FOUNDRY_NOUL_KEYS}); "
                                       f"fields: {_describe(ans)}")
        return Decision(name, qtype, p_yes=p, confidence=conf, raw=ans)
    dist_key = next((k for k in FOUNDRY_DIST_KEYS if k in ans), None)
    if qtype == "choice":
        if dist_key is None:
            raise DeciderResponseError(f"{where} has no probabilities field; fields: {_describe(ans)}")
        probs = _dist(ans[dist_key], f"{where}[{dist_key!r}]")
        choice = ans.get("choice") if isinstance(ans.get("choice"), str) else max(probs, key=probs.get)
        return Decision(name, qtype, probabilities=probs, choice=choice, confidence=conf, raw=ans)
    s = next((_num(ans[k]) for k in FOUNDRY_SCORE_KEYS if _num(ans.get(k)) is not None), None)
    if s is None:
        raise DeciderResponseError(f"{where} has no score field (tried {FOUNDRY_SCORE_KEYS}); fields: {_describe(ans)}")
    probs = _dist(ans[dist_key], f"{where}[{dist_key!r}]") if dist_key else None
    return Decision(name, qtype, score=s, probabilities=probs, confidence=conf, raw=ans)


def _parse_kev_sections(name: str, resp: dict, qtype: str) -> Decision:
    """The older Kev form: nouls / choices / scores, each keyed by question name."""
    section = {"noul": "nouls", "choice": "choices", "score": "scores"}[qtype]
    if not isinstance(resp.get(section), dict) or name not in resp[section]:
        raise DeciderResponseError(f"response has no answers[{name!r}] or {section}[{name!r}]; "
                                   f"top-level keys: {sorted(resp)}")
    ans = resp[section][name]
    if qtype == "noul" and isinstance(ans, dict) and "noul" in ans:
        return _parse_answer(name, ans, qtype)
    if qtype != "noul" and isinstance(ans, dict):
        return _parse_answer(name, ans, qtype)
    raise DeciderResponseError(f"{section}[{name!r}] has no usable answer; fields: {_describe(ans)}")


def _parse_response(name: str, resp: dict, qtype: str, flavor: str) -> Decision:
    global _foundry_noted
    answers = resp.get("answers")
    if isinstance(answers, dict) and name in answers:
        if flavor == "foundry" and not _foundry_noted:
            _foundry_noted = True
            found = {n: _describe(a) for n, a in answers.items()}
            print(f"docmem decide: Decision-1 answer fields found: {found}. Their names are not documented; "
                  "the parser reads probability, distribution and score fields by the names in decide.py. "
                  "Pin the names there if these differ.", file=sys.stderr, flush=True)
        return _parse_answer(name, answers[name], qtype)
    return _parse_kev_sections(name, resp, qtype)


def _entra_token() -> str:
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError as e:
        raise DeciderError("DOCMEM_DECIDER_AUTH=entra needs azure-identity: pip install azure-identity") from e
    return DefaultAzureCredential().get_token(ENTRA_SCOPE).token


def _headers(cfg: Config) -> dict:
    if cfg.flavor == "kev":
        return {"Authorization": f"Bearer {cfg.key}"}
    if cfg.auth == "entra":
        return {"Authorization": f"Bearer {_entra_token()}"}
    return {"api-key": cfg.key}


def _read_cache(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rec = json.loads(line)
                out[rec["key"]] = rec["result"]
            except (ValueError, KeyError) as e:
                raise DeciderError(f"{path}:{i}: not a cache record ({e}); delete the file to start again") from None
    return out


class Decider:
    def __init__(self, cfg: Config, cache: Path | None = None, max_questions: int = 16, timeout: float = 120,
                 sleep=time.sleep):
        cognee_egress.check_url(cfg.endpoint, role="decider", method="POST")  # no-op without a policy
        self.cfg = cfg
        self.cache_path = cache
        self.max_questions = max_questions
        self.timeout = timeout
        self._sleep = sleep
        self._cache = _read_cache(cache) if cache else {}
        # requests sent, decisions answered from the cache, and seconds spent waiting for the server
        self.stats = {"requests": 0, "cache_hits": 0, "seconds": 0.0}

    @classmethod
    def from_env(cls, cache: Path | None = None, env=None, **flags) -> "Decider":
        return cls(Config.from_env(env, **flags), cache=cache)

    def ask(self, state: str, questions: dict[str, dict]) -> dict:
        """POST one request and return the server's JSON, as received. No cache."""
        body = {"model": self.cfg.model, "state": state,
                "questions": {n: _normalize(n, q) for n, q in questions.items()}}
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        for attempt in range(1, MAX_ATTEMPTS + 1):
            req = urllib.request.Request(self.cfg.endpoint, data=data, method="POST", headers={
                "Content-Type": "application/json", **_headers(self.cfg)})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
            except urllib.error.HTTPError as e:
                text = e.read()[:300].decode("utf-8", "replace")
                if (e.code == 429 or e.code >= 500) and attempt < MAX_ATTEMPTS:
                    self._sleep(_retry_after(e.headers) or 2 ** (attempt - 1))
                    continue
                raise DeciderError(f"{self.cfg.endpoint} answered {e.code}: {text}") from None
            except urllib.error.URLError as e:
                raise DeciderError(f"cannot reach {self.cfg.endpoint}: {e.reason}") from None
            try:
                return json.loads(raw)
            except ValueError:
                raise DeciderResponseError(f"{self.cfg.endpoint} did not answer with JSON") from None

    def parse(self, resp: dict, questions: dict[str, dict]) -> dict[str, Decision]:
        """Read each question's answer out of a response of this client's flavor."""
        return {n: _parse_response(n, resp, _normalize(n, q)["type"], self.cfg.flavor)
                for n, q in questions.items()}

    def decide(self, state: str, questions: dict[str, dict]) -> dict[str, Decision]:
        """Decisions for each named question, in the order given. Cached answers are not asked again;
        the rest go in requests of at most max_questions questions each."""
        norm = {n: _normalize(n, q) for n, q in questions.items()}
        keys = {n: cache_key(self.cfg.model, state, q) for n, q in norm.items()}
        out = {n: Decision(n, **self._cache[keys[n]]) for n in norm if keys[n] in self._cache}
        self.stats["cache_hits"] += len(out)
        todo = {n: norm[n] for n in norm if n not in out}
        names = list(todo)
        for i in range(0, len(names), self.max_questions):
            chunk = {n: todo[n] for n in names[i:i + self.max_questions]}
            t = time.perf_counter()
            resp = self.ask(state, chunk)
            self.stats["seconds"] += time.perf_counter() - t
            self.stats["requests"] += 1
            for n, d in self.parse(resp, chunk).items():
                out[n] = d
                self._store(keys[n], d)
        return {n: out[n] for n in questions}

    def _store(self, key: str, d: Decision) -> None:
        self._cache[key] = d.fields()
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "result": d.fields()}, ensure_ascii=False) + "\n")


CHECK_STATE = "The sky is clear and the sun is out."
CHECK_QUESTIONS = {
    "noul": {"type": "noul", "instructions": "Is the sky clear?"},
    "choice": {"type": "choice", "instructions": "Which weather does the state describe: sunny, cloudy or rainy?"},
    "score": {"type": "score", "instructions": "How sunny is the state, from 0 (dark) to 10 (blazing)?"},
}


def check_main(argv=None):
    ap = argparse.ArgumentParser(prog="docmem decide-check", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="DOCMEM_DECIDER_URL")
    ap.add_argument("--key", help="DOCMEM_DECIDER_KEY (prefer the variable: a flag shows in the process list)")
    ap.add_argument("--model", help="DOCMEM_DECIDER_MODEL")
    ap.add_argument("--flavor", choices=FLAVORS, help="DOCMEM_DECIDER_FLAVOR")
    ap.add_argument("--auth", choices=AUTHS, help="DOCMEM_DECIDER_AUTH (foundry)")
    ap.add_argument("--type", choices=QUESTION_TYPES, default="noul", help="the question to send (default noul)")
    a = ap.parse_args(argv)
    try:
        d = Decider.from_env(url=a.url, key=a.key, model=a.model, flavor=a.flavor, auth=a.auth)
        question = {a.type: CHECK_QUESTIONS[a.type]}
        resp = d.ask(CHECK_STATE, question)
        parsed = d.parse(resp, question)[a.type]
    except DeciderError as e:
        raise SystemExit(f"docmem decide-check: {e}")
    print(f"flavor {d.cfg.flavor}, endpoint {d.cfg.endpoint}, model {d.cfg.model}")
    print("parsed:", json.dumps({**parsed.fields(), "raw": "see below"}, ensure_ascii=False))
    print("raw response:")
    print(json.dumps(resp, ensure_ascii=False, indent=1))
