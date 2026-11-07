"""--link-weights det: the pure weighting (no DB). Run from apps/docmem: python -m pytest tests/test_link_weights.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docmem.ask import LEAD_BONUS, OWN_BONUS, OWN_SUFFIX, weigh_links  # noqa: E402


def test_link_score_is_idf_times_support_times_lead():
    rec = weigh_links({"p": ["T"]}, {"p": 0.5}, {"T": 2.0}, {"T": 3}, {"p": True}, 0.5)
    assert rec["p"]["targets"][0]["contribution"] == round(2.0 * 3 * LEAD_BONUS, 4)
    assert rec["p"]["link_score"] == round(2.0 * 3 * LEAD_BONUS, 4)
    rec = weigh_links({"p": ["T"]}, {"p": 0.5}, {"T": 2.0}, {"T": 3}, {"p": False}, 0.5)
    assert rec["p"]["link_score"] == 6.0


def test_link_scores_sum_over_targets_and_own_article_bonus():
    cands = {"p": ["A", "B" + OWN_SUFFIX]}
    rec = weigh_links(cands, {"p": 0.0}, {"A": 1.0, "B": 1.0}, {"A": 1, "B": 2}, {"p": False}, 0.0)
    assert rec["p"]["link_score"] == round(1.0 + 1.0 * 2 * OWN_BONUS, 4)
    own = rec["p"]["targets"][1]
    assert own["own"] and own["target"] == "B" + OWN_SUFFIX and own["support"] == 2


def test_normalized_best_gets_one_and_lambda_mixes():
    cands = {"a": ["T"], "b": ["T"], "c": ["T"]}
    lead = {"a": False, "b": False, "c": False}
    sims = {"a": 0.9, "b": 0.5, "c": 0.1}
    support = {"T": 1}
    # a link-free candidate scores 0 links, so only the similarity counts
    cands["d"] = []
    rec = weigh_links({k: v for k, v in cands.items() if v}, sims, {"T": 1.0}, support, {**lead, "d": False},
                      0.5)
    assert rec["a"]["norm_link"] == 1.0 and rec["b"]["norm_link"] == 1.0
    assert rec["a"]["final"] == round(0.5 * 1.0 + 0.5 * 0.9, 4)


def test_lambda_zero_is_similarity_and_lambda_one_is_links():
    cands = {"x": ["T"], "y": ["U"]}
    sims = {"x": 0.2, "y": 0.9}
    idf, support, lead = {"T": 3.0, "U": 1.0}, {"T": 1, "U": 1}, {"x": False, "y": False}
    by_sim = weigh_links(cands, sims, idf, support, lead, 0.0)
    by_link = weigh_links(cands, sims, idf, support, lead, 1.0)
    assert max(by_sim, key=lambda p: by_sim[p]["final"]) == "y"
    assert max(by_link, key=lambda p: by_link[p]["final"]) == "x"


def test_no_links_anywhere_is_safe():
    rec = weigh_links({"p": ["T"]}, {"p": 0.3}, {"T": 1.0}, {"T": 1}, {"p": True}, 0.5)
    assert rec["p"]["norm_link"] == 1.0 and rec["p"]["final"] == round(0.5 * 1.0 + 0.5 * 0.3, 4)


# --link-weights decide: the decision model's choice of linked topic, with a fake decider (no network, no DB)

import asyncio  # noqa: E402

import pytest  # noqa: E402

from docmem import ask  # noqa: E402
from docmem.decide import Decision, DeciderError, DeciderResponseError  # noqa: E402


class FakeDecider:
    """Answers each choice with the probabilities of the first table key found in the state."""

    def __init__(self, table, error=None):
        self.table, self.error, self.calls = table, error, []
        self.stats = {"requests": 0, "cache_hits": 0, "seconds": 0.0}

    def decide(self, state, questions):
        (name, q), = questions.items()
        self.calls.append((state, q))
        if self.error:
            raise self.error
        key = next(k for k in self.table if k in state)
        probs = self.table[key]
        return {name: Decision(name, "choice", probabilities=probs, choice=max(probs, key=probs.get))}


@pytest.fixture
def offline(monkeypatch):
    """The store, link lookups and text reads replaced by fixed values."""

    async def engines():
        return None, None

    async def idf_of(graph, names, degree):
        return {n: 1.0 for n in names}

    async def hits_for(ids, dist):
        return [{"id": i, "path": f"P {i}", "uri": "u", "distance": dist.get(i)} for i in ids]

    async def distances(question, cands):
        return {pid: 0.5 for pid in cands}

    monkeypatch.setattr(ask.store, "engines", engines)
    monkeypatch.setattr(ask, "distances", distances)
    monkeypatch.setattr(ask, "link_idf", idf_of)
    monkeypatch.setattr(ask, "hits_for", hits_for)
    monkeypatch.setattr(ask, "read_hits", lambda hits, ctx, mb: [
        {**h, "text": f"{h['id']} text", "status": "ok"} for h in hits])
    return monkeypatch


def test_link_criteria_names_each_target_by_anchor_and_adds_none():
    crit = ask.link_criteria({"A": "the a link", "B": "B"})
    assert crit == {"A": "linked as: the a link", "B": "linked as: B",
                    "none": "none of these links helps answer the question"}
    # a target literally named "none" keeps its own option; the "none" option is renamed
    assert ask.link_criteria({"none": "x"})["_none"] == "none of these links helps answer the question"


def test_ask_links_states_question_passage_and_options(tmp_path):
    dec = FakeDecider({"Passage (Doc › Sec)": {"A": 0.7, "none": 0.3}})
    log = tmp_path / "decisions.jsonl"
    probs = ask.ask_links("why?", [{"id": "h1", "path": "Doc › Sec", "text": "body",
                                    "targets": {"A": "the a link"}}], dec, log)
    assert probs == {("h1", "A"): 0.7}
    state, q = dec.calls[0]
    assert state == "Question: why?\n\nPassage (Doc › Sec):\nbody"
    assert q == {"type": "choice", "instructions": ask.DECIDE_QUESTION,
                 "criteria": {"A": "linked as: the a link", "none": "none of these links helps answer the question"}}
    assert '"passage_id": "h1"' in log.read_text()


def test_weigh_decided_sums_reaching_pairs_and_mixes_with_similarity():
    cands = {"p1": ["A", "B" + ask.OWN_SUFFIX], "p2": ["A"], "p3": ["C"]}
    probs = {("h1", "A"): 0.6, ("h2", "A"): 0.3, ("h1", "B"): 0.1, ("h2", "C"): 0.5}
    rec = ask.weigh_decided(cands, probs, {"p1": 0.5, "p2": 0.5, "p3": 0.5}, 0.5)
    assert rec["p1"]["link_score"] == 1.0 and rec["p1"]["norm_link"] == 1.0
    assert rec["p2"]["link_score"] == 0.9 and rec["p2"]["norm_link"] == 0.9
    assert rec["p3"]["link_score"] == 0.5
    assert rec["p1"]["final"] == round(0.5 * 1.0 + 0.5 * 0.5, 4)
    assert {(t["target"], t["from"], t["p_target"]) for t in rec["p1"]["targets"]} == {
        ("A", "h1", 0.6), ("A", "h2", 0.3), ("B" + ask.OWN_SUFFIX, "h1", 0.1)}


def test_weigh_decided_lambda_zero_is_similarity():
    cands = {"x": ["A"], "y": ["C"]}
    rec = ask.weigh_decided(cands, {("h1", "A"): 0.9, ("h1", "C"): 0.1}, {"x": 0.2, "y": 0.9}, 0.0)
    assert max(rec, key=lambda p: rec[p]["final"]) == "y"


def test_expand_links_decide_ranks_by_the_decision_model(offline):
    async def link_neighbours(graph, ids):
        via = {"p1": ["A"], "p2": ["A", "B" + ask.OWN_SUFFIX], "p3": ["C"]}
        return via, {"A": 2, "B": 1, "C": 1}, {}

    async def outgoing_links(graph, ids):
        return {"h1": {"A": "a link", "C": "c"}, "h2": {"A": "alpha"}}

    async def distances(question, cands):
        return {"p1": 0.2, "p2": 0.4, "p3": 0.1}

    offline.setattr(ask, "link_neighbours", link_neighbours)
    offline.setattr(ask, "outgoing_links", outgoing_links)
    offline.setattr(ask, "distances", distances)
    dec = FakeDecider({"h1 text": {"A": 0.7, "C": 0.2, "none": 0.1}, "h2 text": {"A": 0.4, "none": 0.6}})
    hits = [{"id": "h1", "path": "D › S1"}, {"id": "h2", "path": "D › S2"}]
    out, linked = asyncio.run(ask.expand_links("q", hits, 2, 50, "decide", 0.5, dec, 6000, None))
    added = out[len(hits):]
    assert [h["id"] for h in added] == ["p1", "p2"]
    assert added[0]["weights"]["link_score"] == 1.1 and added[0]["weights"]["final"] == 0.9
    assert added[1]["weights"]["link_score"] == 1.1 and added[1]["weights"]["final"] == 0.8
    assert linked == [["A"], ["A", "B" + ask.OWN_SUFFIX]]
    assert len(dec.calls) == 2  # one choice question per hit with targets


def test_expand_links_decide_falls_back_to_det_when_unreachable(offline, capsys):
    async def link_neighbours(graph, ids):
        return {"p1": ["A"]}, {"A": 1}, {"A": 1}

    async def outgoing_links(graph, ids):
        return {"h1": {"A": "a"}}

    async def det_links(question, cands, degree, support, n, lam):
        return [{"id": "det", "weights": {}, "via": cands["p1"]}]

    offline.setattr(ask, "link_neighbours", link_neighbours)
    offline.setattr(ask, "outgoing_links", outgoing_links)
    offline.setattr(ask, "det_links", det_links)
    dec = FakeDecider({}, error=DeciderError("cannot reach http://127.0.0.1:1/v1/systemone"))
    out, _ = asyncio.run(ask.expand_links("q", [{"id": "h1", "path": "D"}], 1, 50, "decide", 0.5, dec))
    assert out[-1]["id"] == "det"
    assert "fell back to det" in capsys.readouterr().err


def test_expand_links_decide_does_not_hide_a_bad_response(offline):
    async def link_neighbours(graph, ids):
        return {"p1": ["A"]}, {"A": 1}, {"A": 1}

    async def outgoing_links(graph, ids):
        return {"h1": {"A": "a"}}

    offline.setattr(ask, "link_neighbours", link_neighbours)
    offline.setattr(ask, "outgoing_links", outgoing_links)
    dec = FakeDecider({}, error=DeciderResponseError("answers[link] has no probabilities field"))
    with pytest.raises(DeciderResponseError):
        asyncio.run(ask.expand_links("q", [{"id": "h1", "path": "D"}], 1, 50, "decide", 0.5, dec))


def test_decide_links_asks_only_the_40_highest_idf_targets(offline, capsys):
    async def distances(question, cands):
        return {"p1": 0.3}

    async def idf_of(graph, names, degree):
        return {n: float(i) for i, n in enumerate(sorted(names))}  # "t00" lowest ... "t44" highest

    offline.setattr(ask, "distances", distances)
    offline.setattr(ask, "link_idf", idf_of)
    targets = {f"t{i:02d}": "x" for i in range(45)}
    dec = FakeDecider({"h1 text": {"t44": 1.0}})
    added = asyncio.run(ask.decide_links("q", [{"id": "h1", "path": "D"}], {"p1": ["t44"]}, {"h1": targets}, {},
                                         1, 0.5, dec, 6000, None))
    crit = dec.calls[0][1]["criteria"]
    assert len(crit) == 41 and "t04" not in crit and "t05" in crit and "t44" in crit and "none" in crit
    assert "45 targets; asking about the 40" in capsys.readouterr().err
    assert added[0]["id"] == "p1"
