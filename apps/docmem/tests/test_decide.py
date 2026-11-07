"""Decider client against a fake HTTP server on localhost (no network, no real decision model).

Run from apps/docmem:  python -m pytest tests/test_decide.py
"""

import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from docmem import decide  # noqa: E402
from docmem.decide import Config, Decider, DeciderError, DeciderResponseError  # noqa: E402


class Fake:
    """Serves queued (status, headers, body) replies per path and records each request."""

    def __init__(self):
        self.replies = {}
        self.requests = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                fake.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                      "body": json.loads(body) if body else None})
                queue = fake.replies.get(self.path, [])
                status, headers, payload = queue.pop(0) if len(queue) > 1 else queue[0]
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reply(self, path, payload, status=200, headers=None):
        self.replies[path] = [(status, headers or {}, payload)]

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake():
    f = Fake()
    yield f
    f.close()


def no_sleep(seconds):
    no_sleep.calls.append(seconds)


no_sleep.calls = []


KEV_BODY = {
    "nouls": {"safe": {"noul": 0.8, "confidence": 0.9}},
    "choices": {"mood": {"choice": "calm", "probabilities": {"calm": 0.7, "tense": 0.3}, "confidence": 0.6}},
    "scores": {"risk": {"score": 3.5, "probabilities": {"1": 0.2, "5": 0.8}, "confidence": 0.5}},
}
Q = {
    "safe": {"type": "noul", "instructions": "Is it safe?"},
    "mood": {"type": "choice", "instructions": "Which mood?", "criteria": ["calm", "tense"]},
    "risk": {"type": "score", "instructions": "How risky?"},
}


def kev(fake, **kw):
    cfg = Config(url=fake.base, model="kev-model", key=kw.pop("key", "secret-key"), flavor="kev", **kw)
    return Decider(cfg, sleep=no_sleep)


# --- configuration -------------------------------------------------------------------------------

def test_flavor_defaults_from_url():
    assert decide.default_flavor("https://x.openai.azure.com/") == "foundry"
    assert decide.default_flavor("https://r.cognitiveservices.azure.com") == "foundry"
    assert decide.default_flavor("https://p.services.ai.azure.com") == "foundry"
    assert decide.default_flavor("http://localhost:8080") == "kev"


def test_config_from_env_and_flag_override():
    env = {"DOCMEM_DECIDER_URL": "http://localhost:8080", "DOCMEM_DECIDER_MODEL": "m", "DOCMEM_DECIDER_KEY": "k"}
    c = Config.from_env(env)
    assert (c.flavor, c.key, c.endpoint) == ("kev", "k", "http://localhost:8080/v1/systemone")
    c = Config.from_env(env, flavor="foundry", key="other")
    assert (c.flavor, c.key) == ("foundry", "other")


def test_kev_key_defaults_to_local():
    c = Config.from_env({"DOCMEM_DECIDER_URL": "http://h", "DOCMEM_DECIDER_MODEL": "m"})
    assert c.key == "local"


@pytest.mark.parametrize("url,expected", [
    ("https://r.cognitiveservices.azure.com", "https://r.cognitiveservices.azure.com/providers/microsoft/v1/systemone"),
    ("https://r.cognitiveservices.azure.com/providers/microsoft/v1/systemone",
     "https://r.cognitiveservices.azure.com/providers/microsoft/v1/systemone"),
])
def test_foundry_endpoint_appends_only_once(url, expected):
    assert Config.from_env({"DOCMEM_DECIDER_URL": url, "DOCMEM_DECIDER_MODEL": "d", "DOCMEM_DECIDER_KEY": "k"}).endpoint == expected


@pytest.mark.parametrize("env,message", [
    ({"DOCMEM_DECIDER_MODEL": "m"}, "DOCMEM_DECIDER_URL"),
    ({"DOCMEM_DECIDER_URL": "http://h"}, "DOCMEM_DECIDER_MODEL"),
    ({"DOCMEM_DECIDER_URL": "https://r.cognitiveservices.azure.com", "DOCMEM_DECIDER_MODEL": "d"}, "DOCMEM_DECIDER_KEY"),
    ({"DOCMEM_DECIDER_URL": "http://h", "DOCMEM_DECIDER_MODEL": "m", "DOCMEM_DECIDER_AUTH": "entra"}, "foundry flavor"),
    ({"DOCMEM_DECIDER_URL": "http://h", "DOCMEM_DECIDER_MODEL": "m", "DOCMEM_DECIDER_FLAVOR": "x"}, "FLAVOR"),
])
def test_config_errors_name_the_variable(env, message):
    with pytest.raises(DeciderError, match=message):
        Config.from_env(env)


def test_entra_without_azure_identity_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "azure.identity", None)  # import raises ImportError
    with pytest.raises(DeciderError, match="pip install azure-identity"):
        decide._entra_token()


def test_entra_token_is_sent_as_bearer(fake, monkeypatch):
    class Cred:
        def __init__(self):
            self.scopes = []

        def get_token(self, scope):
            self.scopes.append(scope)
            return types.SimpleNamespace(token="tok-123")

    cred = Cred()
    azure = types.ModuleType("azure")
    identity = types.ModuleType("azure.identity")
    identity.DefaultAzureCredential = lambda: cred
    monkeypatch.setitem(sys.modules, "azure", azure)
    monkeypatch.setitem(sys.modules, "azure.identity", identity)
    fake.reply("/providers/microsoft/v1/systemone", {"answers": {"safe": {"probability": 0.5}}})
    cfg = Config(url=fake.base, model="dep", key=None, flavor="foundry", auth="entra")
    Decider(cfg, sleep=no_sleep).decide("s", {"safe": {"type": "noul"}})
    assert fake.requests[0]["headers"]["authorization"] == "Bearer tok-123"
    assert cred.scopes == [decide.ENTRA_SCOPE]


# --- kev flavor -----------------------------------------------------------------------------------

def test_kev_request_and_parse(fake):
    fake.reply("/v1/systemone", KEV_BODY)
    out = kev(fake).decide("the state", Q)
    req = fake.requests[0]
    assert req["path"] == "/v1/systemone"
    assert req["headers"]["authorization"] == "Bearer secret-key"
    assert req["body"]["model"] == "kev-model" and req["body"]["state"] == "the state"
    assert req["body"]["questions"]["mood"] == {"type": "choice", "instructions": "Which mood?",
                                                "criteria": ["calm", "tense"]}
    assert out["safe"].type == "noul" and out["safe"].p_yes == 0.8 and out["safe"].confidence == 0.9
    assert out["mood"].choice == "calm" and out["mood"].probabilities == {"calm": 0.7, "tense": 0.3}
    assert out["risk"].score == 3.5 and out["risk"].probabilities == {"1": 0.2, "5": 0.8}
    assert list(out) == ["safe", "mood", "risk"]


def test_kev_missing_field_is_an_error_with_fields_listed(fake):
    fake.reply("/v1/systemone", {"nouls": {"safe": {"p": 0.8}}})
    with pytest.raises(DeciderResponseError, match="no usable answer"):
        kev(fake).decide("s", {"safe": {"type": "noul"}})


def test_kev_endpoint_accepts_full_systemone_url(fake):
    fake.reply("/v1/systemone", KEV_BODY)
    cfg = Config(url=fake.base + "/v1/systemone", model="m", key="k", flavor="kev")
    Decider(cfg, sleep=no_sleep).decide("s", {"safe": {"type": "noul"}})
    assert fake.requests[0]["path"] == "/v1/systemone"


# --- foundry flavor -------------------------------------------------------------------------------

FOUNDRY_PATH = "/providers/microsoft/v1/systemone"


def foundry(fake, **kw):
    cfg = Config(url=fake.base, model="dec-deploy", key="api-k", flavor="foundry", auth="key")
    return Decider(cfg, sleep=no_sleep, **kw)


def test_foundry_request_uses_api_key_header(fake):
    fake.reply(FOUNDRY_PATH, {"answers": {"safe": {"probability": 0.25}}})
    out = foundry(fake).decide("s", {"safe": {"type": "noul"}})
    assert fake.requests[0]["path"] == FOUNDRY_PATH
    assert fake.requests[0]["headers"]["api-key"] == "api-k"
    assert "authorization" not in fake.requests[0]["headers"]
    assert fake.requests[0]["body"]["model"] == "dec-deploy"
    assert out["safe"].p_yes == 0.25 and out["safe"].raw == {"probability": 0.25}


@pytest.mark.parametrize("field", ["probability", "noul", "p", "value"])
def test_foundry_noul_tolerates_field_names(fake, field):
    fake.reply(FOUNDRY_PATH, {"answers": {"safe": {field: 0.6, "other": "x"}}})
    assert foundry(fake).decide("s", {"safe": {"type": "noul"}})["safe"].p_yes == 0.6


def test_foundry_noul_bare_number(fake):
    fake.reply(FOUNDRY_PATH, {"answers": {"safe": 0.3}})
    assert foundry(fake).decide("s", {"safe": {"type": "noul"}})["safe"].p_yes == 0.3


def test_foundry_choice_and_score_tolerant(fake):
    fake.reply(FOUNDRY_PATH, {"answers": {
        "mood": {"distribution": {"calm": 0.1, "tense": 0.9}},
        "risk": {"mean": 7.0, "distribution": {"5": 0.5, "9": 0.5}},
    }})
    out = foundry(fake).decide("s", {"mood": {"type": "choice"}, "risk": {"type": "score"}})
    assert out["mood"].choice == "tense" and out["mood"].probabilities == {"calm": 0.1, "tense": 0.9}
    assert out["risk"].score == 7.0 and out["risk"].probabilities == {"5": 0.5, "9": 0.5}


def test_foundry_out_of_range_probability_fails_fast(fake):
    fake.reply(FOUNDRY_PATH, {"answers": {"safe": {"probability": 7.5}}})
    with pytest.raises(DeciderResponseError, match="outside"):
        foundry(fake).decide("s", {"safe": {"type": "noul"}})


def test_foundry_unknown_fields_error_lists_them(fake):
    fake.reply(FOUNDRY_PATH, {"answers": {"safe": {"verdict": "yes", "certainty": 0.9}}})
    with pytest.raises(DeciderResponseError, match="certainty"):
        foundry(fake).decide("s", {"safe": {"type": "noul"}})


def test_foundry_warns_once_naming_fields(fake, capsys):
    fake.reply(FOUNDRY_PATH, {"answers": {"safe": {"probability": 0.5, "extra": 1}}})
    decide._foundry_noted = False
    foundry(fake).decide("s", {"safe": {"type": "noul"}})
    foundry(fake).decide("t", {"safe": {"type": "noul"}})
    err = capsys.readouterr().err
    assert err.count("Decision-1 answer fields found") == 1 and "extra" in err


# --- retries, batching, cache ---------------------------------------------------------------------

def test_retries_429_with_retry_after_then_succeeds(fake):
    fake.replies[FOUNDRY_PATH] = [
        (429, {"Retry-After": "3"}, {"error": "slow"}),
        (503, {}, {"error": "busy"}),
        (200, {}, {"answers": {"safe": {"probability": 0.4}}}),
    ]
    d = foundry(fake)
    no_sleep.calls.clear()
    assert d.decide("s", {"safe": {"type": "noul"}})["safe"].p_yes == 0.4
    assert len(fake.requests) == 3
    assert no_sleep.calls == [3.0, 2.0]  # Retry-After, then the backoff for the second attempt


def test_retries_give_up_after_max_attempts(fake):
    fake.reply(FOUNDRY_PATH, {"error": "down"}, status=500)
    with pytest.raises(DeciderError, match="answered 500"):
        foundry(fake).decide("s", {"safe": {"type": "noul"}})
    assert len(fake.requests) == decide.MAX_ATTEMPTS


def test_client_error_is_not_retried_and_key_not_in_message(fake):
    fake.reply(FOUNDRY_PATH, {"error": "bad key"}, status=401)
    with pytest.raises(DeciderError) as e:
        foundry(fake).decide("s", {"safe": {"type": "noul"}})
    assert len(fake.requests) == 1 and "api-k" not in str(e.value)


def test_batches_questions_per_request(fake):
    names = [f"q{i}" for i in range(5)]
    fake.replies["/v1/systemone"] = [(200, {}, {"nouls": {n: {"noul": 0.5} for n in names[:2]}}),
                                     (200, {}, {"nouls": {n: {"noul": 0.5} for n in names[2:4]}}),
                                     (200, {}, {"nouls": {n: {"noul": 0.5} for n in names[4:]}})]
    d = Decider(Config(url=fake.base, model="m", key="k", flavor="kev"), max_questions=2, sleep=no_sleep)
    out = d.decide("s", {n: {"type": "noul"} for n in names})
    assert list(out) == names
    assert [len(r["body"]["questions"]) for r in fake.requests] == [2, 2, 1]


def test_cache_avoids_second_request_and_stores_no_state_text(fake, tmp_path):
    secret_state = "PATIENT-NAME-ZEBRA-7731 says the valve is open."
    fake.reply("/v1/systemone", KEV_BODY)
    cache = tmp_path / "decider_cache.jsonl"
    cfg = Config(url=fake.base, model="kev-model", key="k", flavor="kev")
    first = Decider(cfg, cache=cache, sleep=no_sleep).decide(secret_state, Q)
    assert len(fake.requests) == 1
    again = Decider(cfg, cache=cache, sleep=no_sleep).decide(secret_state, Q)
    assert len(fake.requests) == 1  # answered from the file
    assert again["safe"].p_yes == first["safe"].p_yes and again["mood"].choice == "calm"
    text = cache.read_text(encoding="utf-8")
    assert "ZEBRA" not in text and "valve" not in text
    assert len(text.splitlines()) == 3


def test_cache_key_depends_on_model_state_and_question(fake, tmp_path):
    fake.reply("/v1/systemone", KEV_BODY)
    cache = tmp_path / "c.jsonl"
    cfg = Config(url=fake.base, model="m1", key="k", flavor="kev")
    Decider(cfg, cache=cache, sleep=no_sleep).decide("s", {"safe": {"type": "noul"}})
    Decider(cfg, cache=cache, sleep=no_sleep).decide("t", {"safe": {"type": "noul"}})
    Decider(Config(url=fake.base, model="m2", key="k", flavor="kev"), cache=cache,
            sleep=no_sleep).decide("s", {"safe": {"type": "noul"}})
    assert len(fake.requests) == 3


def test_question_validation(fake):
    d = kev(fake)
    with pytest.raises(DeciderError, match="type must be"):
        d.decide("s", {"x": {"type": "maybe"}})
    with pytest.raises(DeciderError, match="unknown fields"):
        d.decide("s", {"x": {"type": "noul", "threshold": 0.5}})


def test_corrupt_cache_names_the_line(tmp_path):
    cache = tmp_path / "c.jsonl"
    cache.write_text('{"key": "a", "result": {}}\nnot json\n', encoding="utf-8")
    with pytest.raises(DeciderError, match=r"c\.jsonl:2"):
        Decider(Config(url="http://h", model="m", key="k", flavor="kev"), cache=cache)


# --- decide-check ---------------------------------------------------------------------------------

def test_decide_check_prints_parsed_and_raw(fake, capsys, monkeypatch):
    for k in ("DOCMEM_DECIDER_URL", "DOCMEM_DECIDER_MODEL", "DOCMEM_DECIDER_KEY", "DOCMEM_DECIDER_FLAVOR"):
        monkeypatch.delenv(k, raising=False)
    fake.reply("/v1/systemone", {"nouls": {"noul": {"noul": 0.91}}})
    decide.check_main(["--url", fake.base, "--key", "k", "--model", "m"])
    out = capsys.readouterr().out
    parsed = json.loads(out.split("parsed: ", 1)[1].split("\n", 1)[0])
    assert parsed["p_yes"] == 0.91 and parsed["type"] == "noul"
    raw = json.loads(out.split("raw response:\n", 1)[1])
    assert raw == {"nouls": {"noul": {"noul": 0.91}}}
    assert fake.requests[0]["body"]["questions"] == {"noul": {"type": "noul", "instructions": "Is the sky clear?"}}


def test_decide_check_config_error_exits(monkeypatch):
    for k in ("DOCMEM_DECIDER_URL", "DOCMEM_DECIDER_MODEL", "DOCMEM_DECIDER_KEY", "DOCMEM_DECIDER_FLAVOR"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit, match="DOCMEM_DECIDER_URL"):
        decide.check_main([])


KEV_ANSWERS_BODY = {"model": "kev-latest", "answers": {
    "escalate": {"type": "noul", "noul": 0.6681},
    "department": {"type": "choice", "choice": "shipping", "confidence": 0.4134,
                   "probabilities": {"returns": 0.201, "shipping": 0.5601, "billing": 0.0493, "other": 0.1897}},
    "frustration": {"type": "score", "score": 1.9661,
                    "legend": {"0": "calm", "1": "annoyed", "2": "angry", "3": "furious"},
                    "probabilities": {"0": 0.0971, "1": 0.2143, "2": 0.3141, "3": 0.3745}, "confidence": 0.0}},
    "usage": {"input_tokens": 386, "output_tokens": 206}, "latency_ms": 2222.9}


def test_kev_answers_shape_is_read_first(fake):
    fake.reply("/v1/systemone", KEV_ANSWERS_BODY)
    out = kev(fake).decide("s", {"escalate": {"type": "noul"}, "department": {"type": "choice"},
                                 "frustration": {"type": "score"}})
    assert out["escalate"].p_yes == 0.6681 and out["escalate"].raw == {"type": "noul", "noul": 0.6681}
    assert out["department"].choice == "shipping" and out["department"].confidence == 0.4134
    assert out["department"].probabilities["shipping"] == 0.5601
    assert out["frustration"].score == 1.9661 and out["frustration"].probabilities["3"] == 0.3745


def test_kev_answers_shape_also_serves_foundry_names(fake):
    fake.reply("/v1/systemone", {"answers": {"escalate": {"probability": 0.2}}})
    out = kev(fake).decide("s", {"escalate": {"type": "noul"}})
    assert out["escalate"].p_yes == 0.2
