"""Improve: link the same entity across documents, group topics into categories, prune orphans.

    docmem improve --db ./kb --api-base https://host/v1 --model M [--max-calls N] [--distance 0.15] [--dry-run]

Works only on what enrichment stored (DocEntity, DocTopic, DocFile, DocPassage nodes and their edges).
Source documents and passage text are never read.

- Concepts: DocEntities with the same name (articles ignored) and kind form one unit with no LLM call.
  Units whose names embed within --distance are asked to the chat model in batches; confirmed pairs
  merge. Each merged group with two or more entities becomes a DocConcept, linked by "same_as" edges.
- Categories: when there are more than 12 topics without a category, the chat model groups them into
  at most 15 DocCategory nodes, linked by "broader" edges. Near-duplicate topic names are reported only.
- Orphans: DocEntity nodes with no "mentions" edge and no relation edge, and DocTopic nodes with no
  "topics" edge, are deleted.

Re-running changes nothing that is already stored: identity keys make the nodes and edges the same,
existing "same_as" edges count as merges, and topics that already have a category are not asked again.
"""
import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import store
from .enrich import norm
from .llm import Chat
from .store import log

PAIR_BATCH = 15
TOPIC_LLM_MIN = 12  # categories are asked for only above this many topics
TOPIC_BATCH = 80
MAX_CATEGORIES = 15
DESC_CHARS = 200
TOPICS = "DocTopic_name"

MERGE_SYSTEM = (
    "You decide whether pairs of entity names extracted from documents refer to the same real-world thing. "
    "Two entries are the same only when their names and descriptions show one thing (a spelling variant, an "
    "abbreviation, a translation, a shorter or longer form). Different things with similar names are not the same. "
    'Reply with one JSON object only: {"same": [<numbers of the pairs that are the same>]}. Use only the given text.')

CATEGORY_SYSTEM = (
    "You group topic names from a document collection into broad categories. Reply with one JSON object only: "
    '{"categories": {"<category name>": [<topic numbers>]}}. Every topic number goes into exactly one category. '
    "A category name is one to four words. Use at most " + str(MAX_CATEGORIES) + " categories in total, and reuse "
    "the categories already listed when a topic fits one.")


# ---- pure helpers (no database, no network) ----

def block_of(name: str) -> str:
    """Blocking key: the normalized name without a leading article ("The Court" and "court" match)."""
    k = norm(name)
    return k[4:].strip() if k.startswith("the ") else k


def canonical(values) -> str:
    """The most frequent value; ties go to the shortest, then the alphabetically first."""
    counts = Counter(v for v in values if v)
    return min(counts, key=lambda v: (-counts[v], len(v), v))


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self) -> list[list[int]]:
        """The sets, each sorted, ordered by their smallest member."""
        by_root = defaultdict(list)
        for i in range(len(self.parent)):
            by_root[self.find(i)].append(i)
        return list(by_root.values())


def units_of(entities: list[dict]) -> list[dict]:
    """Candidate units: entities sharing a block and a kind. Each unit is one node of the merge
    (its members need no LLM call to be one thing). Entities are dicts with id, name, kind, description."""
    by_key = defaultdict(list)
    for i, e in enumerate(entities):
        by_key[(block_of(e["name"]), norm(e["kind"]))].append(i)
    units = []
    for (block, kind), members in sorted(by_key.items()):
        units.append({
            "block": block, "kind": kind, "members": members,
            "name": canonical(entities[i]["name"] for i in members),
            "description": max((entities[i]["description"] for i in members), key=len),
        })
    return units


def batches(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def near_pairs(vectors, threshold: float) -> list[tuple[int, int, float]]:
    """(i, j, cosine distance) for every i < j whose cosine distance is below ``threshold``.
    Computed in row blocks so memory stays at one block of distances."""
    import numpy as np

    if len(vectors) < 2:
        return []
    x = np.asarray(vectors, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    out = []
    step = 1024
    for s in range(0, len(x), step):
        d = 1.0 - x[s:s + step] @ x.T
        rows, cols = np.nonzero(d < threshold)
        for r, c in zip(rows, cols):
            i = s + int(r)
            if i < int(c):
                out.append((i, int(c), float(d[r, c])))
    return out


def existing_links_pairs(units: list[dict], entities: list[dict], linked: dict) -> list[tuple[int, int]]:
    """Pairs of units whose members already share a concept (a same_as edge from an earlier run)."""
    unit_of = {entities[i]["id"]: ui for ui, u in enumerate(units) for i in u["members"]}
    by_concept = defaultdict(set)
    for eid, concept_ids in linked.items():
        if eid in unit_of:
            for cid in concept_ids:
                by_concept[cid].add(unit_of[eid])
    pairs = []
    for us in by_concept.values():
        us = sorted(us)
        pairs += [(us[0], u) for u in us[1:]]
    return pairs


def concepts_of(entities: list[dict], components: list[list[int]]) -> list[dict]:
    """One concept per component of two or more entities (component = entity indices)."""
    out = []
    for members in components:
        if len(members) < 2:
            continue
        es = [entities[i] for i in members]
        name = canonical(e["name"] for e in es)
        kind = canonical(norm(e["kind"]) for e in es)
        out.append({
            "name": name, "kind": kind,
            "aliases": sorted({e["name"] for e in es} - {name}),
            "n_docs": len({e["doc_uri"] for e in es}),
            "description": max((e["description"] for e in es), key=len),
            "key": store.key("concept", norm(name), kind),
            "members": [(e["id"], e["name"]) for e in es],
        })
    return out


def orphans(entity_ids: list[str], topic_ids: list[str], edges: list) -> tuple[list[str], list[str]]:
    """(entity ids with no incoming "mentions" and no relation edge, topic ids with no incoming "topics").
    A relation edge joins two DocEntity nodes and is not "same_as"."""
    ents = set(entity_ids)
    mentioned = {d for _, d, rel, _ in edges if rel == "mentions"}
    related = set()
    for s, d, rel, _ in edges:
        if rel != "same_as" and s in ents and d in ents:
            related.update((s, d))
    topic_linked = {d for _, d, rel, _ in edges if rel == "topics"}
    return ([e for e in entity_ids if e not in mentioned and e not in related],
            [t for t in topic_ids if t not in topic_linked])


def parse_same(reply, n: int) -> set[int]:
    """0-based indices of the pairs the reply calls the same. Accepts 3, "3", "P3" or "pair 3"."""
    raw = reply.get("same") if isinstance(reply, dict) else reply
    if not isinstance(raw, list):
        raise ValueError(f"no 'same' list in reply: {str(reply)[:200]!r}")
    out = set()
    for x in raw:
        m = re.search(r"\d+", str(x))
        if m and 1 <= int(m.group()) <= n:
            out.add(int(m.group()) - 1)
    return out


def parse_categories(reply, n: int, known: list[str], cap: int):
    """(assignment {0-based topic index: category}, known categories after, rejected names).
    A category beyond ``cap`` is rejected and its topics stay uncategorized."""
    cats = reply.get("categories") if isinstance(reply, dict) else None
    if not isinstance(cats, dict):
        raise ValueError(f"no 'categories' object in reply: {str(reply)[:200]!r}")
    known = list(known)
    by_norm = {norm(c): c for c in known}
    assign, rejected = {}, []
    for name, nums in cats.items():
        name = " ".join(str(name).split())[:60]
        if not name or not isinstance(nums, list):
            continue
        valid = [int(m.group()) - 1 for m in (re.search(r"\d+", str(x)) for x in nums)
                 if m and 1 <= int(m.group()) <= n]
        if not valid:
            continue
        key = norm(name)
        if key not in by_norm:
            if len(known) >= cap:
                rejected.append(name)
                continue
            known.append(name)
            by_norm[key] = name
        for i in valid:
            assign.setdefault(i, by_norm[key])
    return assign, known, rejected


def describe(unit: dict) -> str:
    return f"{unit['name']} ({unit['kind']}): {unit['description'][:DESC_CHARS]}"


def pair_prompt(pairs: list[tuple[dict, dict]]) -> str:
    lines = ["Pairs:"]
    for n, (a, b) in enumerate(pairs, 1):
        lines.append(f"{n}. A: {describe(a)}\n   B: {describe(b)}")
    return "\n".join(lines) + '\n\nReply {"same": [pair numbers]}.'


def category_prompt(names: list[str], known: list[str]) -> str:
    lines = ["Topics:"] + [f"{i}. {n}" for i, n in enumerate(names, 1)]
    lines.append("\nCategories so far: " + (json.dumps(known, ensure_ascii=False) if known else "none"))
    return "\n".join(lines) + '\n\nReply {"categories": {"<name>": [topic numbers]}}.'


def budget_spent(chat: Chat, max_calls: int | None) -> bool:
    return max_calls is not None and chat.usage["calls"] >= max_calls


# ---- DataPoint types (defined once cognee is importable) ----

_models = None


def models():
    """(DocConcept, DocCategory)."""
    global _models
    if _models is None:
        from cognee.low_level import DataPoint

        class DocConcept(DataPoint):
            """One real-world thing that several entity records describe (improve)."""
            name: str
            key: str
            kind: str
            description: str
            aliases: str = "[]"  # JSON list of the other names
            n_docs: int = 0
            metadata: dict = {"index_fields": ["name"], "identity_fields": ["key"]}

        class DocCategory(DataPoint):
            name: str
            key: str
            metadata: dict = {"index_fields": ["name"], "identity_fields": ["key"]}

        _models = (DocConcept, DocCategory)
    return _models


# ---- cognee side ----

async def load(graph, type_name: str) -> list[dict]:
    nodes, _ = await graph.get_filtered_graph_data([{"type": [type_name]}], include_edges=False)
    return [store.node_props({"id": n[0], **n[1]}) if isinstance(n, (tuple, list)) else store.node_props(n)
            for n in nodes]


async def embed(texts: list[str]) -> dict[str, list[float]]:
    from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine

    uniq = sorted(set(texts))
    vectors = await get_embedding_engine().embed_text(uniq)
    return dict(zip(uniq, vectors))


async def ask_pairs(chat: Chat, pairs: list[tuple[dict, dict]], max_calls: int | None, st: Counter):
    """(confirmed pair indices, number of pairs asked). Stops when the call budget is spent."""
    confirmed, asked = [], 0
    for batch in batches(list(range(len(pairs))), PAIR_BATCH):
        if budget_spent(chat, max_calls):
            break
        group = [pairs[i] for i in batch]
        asked += len(batch)
        try:
            reply = chat.complete_json(MERGE_SYSTEM, pair_prompt(group), 300)
            same = parse_same(reply, len(group))
        except ValueError as e:
            st["failed_replies"] += 1
            log(f"docmem improve: a pair reply was not usable, its pairs stay separate: {e}")
            continue
        confirmed += [batch[k] for k in sorted(same)]
    return confirmed, asked


async def write(nodes: list, edges: list) -> None:
    """One cognee pipeline run: the DataPoints and the custom edges, graph and vectors."""
    import cognee
    from cognee.modules.pipelines import Task
    from cognee.tasks.storage import add_data_points

    async def store_batch(items, ctx=None):
        await add_data_points(nodes, ctx=ctx, index_edges=False, custom_edges=edges or None)
        return []

    result = await cognee.run_custom_pipeline(tasks=[Task(store_batch, needs_llm=False)],
                                              data=[{"batch": 0}], dataset=store.DATASET)
    for run in (result or {}).values():
        status = str(getattr(run, "status", ""))
        if "Errored" in status or "ERRORED" in status:
            raise SystemExit(f"docmem improve: the cognee pipeline failed: {getattr(run, 'payload', run)}")


async def run(a) -> dict:
    import cognee

    t0 = time.time()
    chat = Chat(a.api_base, a.api_key, a.model)
    await cognee.run_migrations()
    graph, vector = await store.engines()
    DocConcept, DocCategory = models()
    st = Counter()

    entities = [{"id": str(n["id"]), "name": n.get("name") or "", "kind": norm(n.get("kind") or "other"),
                 "description": n.get("description") or "", "doc_uri": n.get("doc_uri") or ""}
                for n in await load(graph, "DocEntity")]
    _, edges = await graph.get_graph_data()
    ent_ids = {e["id"] for e in entities}
    linked = defaultdict(set)
    same_as_existing = set()
    for s, d, rel, _ in edges:
        if rel == "same_as" and s in ent_ids:
            linked[s].add(d)
            same_as_existing.add((s, d))
    existing_concepts = {str(n["id"]) for n in await load(graph, "DocConcept")}
    existing_categories = await load(graph, "DocCategory")

    # 1-2. concepts
    units = units_of(entities)
    blocks = Counter(u["block"] for u in units)  # units per name block
    uf = UnionFind(len(units))
    for x, y in existing_links_pairs(units, entities, linked):
        uf.union(x, y)
    candidates, asked, confirmed = [], 0, []
    if len(units) > 1:
        vec = await embed([u["name"] for u in units])
        near = near_pairs([vec[u["name"]] for u in units], a.distance)
        candidates = [(i, j) for i, j, _ in near if uf.find(i) != uf.find(j)]
        pairs = [(units[i], units[j]) for i, j in candidates]
        confirmed, asked = await ask_pairs(chat, pairs, a.max_calls, st)
        for k in confirmed:
            uf.union(*candidates[k])
    components = [sum((units[u]["members"] for u in g), []) for g in uf.groups()]
    concepts = concepts_of(entities, components)
    concept_nodes, same_as = [], []
    for c in concepts:
        node = DocConcept(name=c["name"], key=c["key"], kind=c["kind"], description=c["description"],
                          aliases=json.dumps(c["aliases"], ensure_ascii=False), n_docs=c["n_docs"])
        concept_nodes.append(node)
        cid = str(node.id)
        for eid, ename in c["members"]:
            if (eid, cid) not in same_as_existing:
                same_as.append((eid, cid, "same_as", {"edge_text": f"{ename} is {c['name']}"}))
    concept_ids = {str(n.id) for n in concept_nodes}

    # 3. topics
    topics = sorted(((str(n["id"]), n.get("name") or "") for n in await load(graph, "DocTopic")),
                    key=lambda t: (norm(t[1]), t[0]))
    topic_name = dict(topics)
    topic_merge = []
    if len(topics) > 1:
        tvec = await embed([name for _, name in topics])
        for i, j, d in near_pairs([tvec[name] for _, name in topics], a.distance):
            topic_merge.append([topics[i][1], topics[j][1], round(d, 4)])
    categorized = {s for s, _, rel, _ in edges if rel == "broader"}
    uncategorized = [t for t in topics if t[0] not in categorized]
    known = [c["name"] for c in existing_categories]
    cat_edges_by_topic, rejected = {}, []
    if len(topics) > TOPIC_LLM_MIN and uncategorized:
        for batch in batches(list(range(len(uncategorized))), TOPIC_BATCH):
            if budget_spent(chat, a.max_calls):
                st["topics_unasked"] += len(batch)
                continue
            names = [uncategorized[i][1] for i in batch]
            try:
                reply = chat.complete_json(CATEGORY_SYSTEM, category_prompt(names, known), 1200)
                assign, known, rej = parse_categories(reply, len(batch), known, MAX_CATEGORIES)
            except ValueError as e:
                st["failed_replies"] += 1
                log(f"docmem improve: a category reply was not usable, its topics stay uncategorized: {e}")
                continue
            rejected += rej
            for local, cat in assign.items():
                cat_edges_by_topic[uncategorized[batch[local]][0]] = cat
    used = {norm(c) for c in cat_edges_by_topic.values()}
    cat_nodes = {norm(name): DocCategory(name=name, key=store.key("category", norm(name)))
                 for name in known if norm(name) in used}
    existing_cat_ids = {str(c["id"]) for c in existing_categories}
    broader_existing = {(s, d) for s, d, rel, _ in edges if rel == "broader"}
    broader = []
    for tid, cat in cat_edges_by_topic.items():
        cnode = cat_nodes[norm(cat)]
        if (tid, str(cnode.id)) not in broader_existing:
            broader.append((tid, str(cnode.id), "broader", {"edge_text": f"{topic_name[tid]} broader {cat}"}))
    new_cats = [c for c in cat_nodes.values() if str(c.id) not in existing_cat_ids]

    # 4. orphans
    ent_orphans, topic_orphans = orphans(sorted(ent_ids), [t for t, _ in topics], edges)

    if not a.dry_run:
        nodes = concept_nodes + new_cats
        if nodes or same_as or broader:
            await write(nodes, same_as + broader)
        if ent_orphans or topic_orphans:
            await graph.delete_nodes(ent_orphans + topic_orphans)
            if topic_orphans and await vector.has_collection(TOPICS):
                await vector.delete_data_points(TOPICS, topic_orphans)

    stale = sorted(existing_concepts - concept_ids)
    return {
        "dry_run": a.dry_run,
        "entities": len(entities),
        "name_groups": len(blocks),
        "name_groups_multi_kind": sum(1 for n in blocks.values() if n > 1),
        "candidate_pairs": len(candidates),
        "pairs_asked": asked,
        "pairs_unasked": len(candidates) - asked,
        "pairs_confirmed": len(confirmed),
        "concepts": len(concept_nodes),
        "concepts_new": len([n for n in concept_nodes if str(n.id) not in existing_concepts]),
        "same_as_new": len(same_as),
        "concepts_stale": len(stale),
        "topics": len(topics),
        "topics_uncategorized_before": len(uncategorized),
        "topics_categorized_now": len(cat_edges_by_topic),
        "topics_unasked": st["topics_unasked"],
        "categories": len(known),
        "categories_new": len(new_cats),
        "categories_rejected": rejected,
        "topic_merge_candidates": topic_merge,
        "orphans_found": {"entities": len(ent_orphans), "topics": len(topic_orphans)},
        "orphans_pruned": {"entities": 0, "topics": 0} if a.dry_run else
        {"entities": len(ent_orphans), "topics": len(topic_orphans)},
        "llm": {**chat.usage, "failed_replies": st["failed_replies"]},
        "seconds": round(time.time() - t0, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docmem improve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--api-base", default=None, help="OpenAI-compatible base URL (or DOCMEM_API_BASE)")
    ap.add_argument("--api-key", default=None, help="API key (or DOCMEM_API_KEY); never stored")
    ap.add_argument("--model", default=None, help="chat model (or DOCMEM_MODEL)")
    ap.add_argument("--max-calls", type=int, default=None,
                    help="stop asking the chat model after this many calls (default: no limit)")
    ap.add_argument("--distance", type=float, default=0.15,
                    help="cosine distance below which names are candidates (default 0.15)")
    ap.add_argument("--dry-run", action="store_true", help="report what would change; write and delete nothing")
    a = ap.parse_args(argv)
    api_base = a.api_base or os.environ.get("DOCMEM_API_BASE")
    api_key = a.api_key or os.environ.get("DOCMEM_API_KEY")
    store.configure(a.db, api_base=api_base, api_key=api_key)
    report = asyncio.run(run(a))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
