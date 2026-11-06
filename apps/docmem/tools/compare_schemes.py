"""Compare concept schemes on an index without any LLM, against two references:

- Wikipedia categories (tools/wiki_categories' doc_categories.json): of document pairs that share a
  direct category, the share that also share a scheme concept, against random pairs (lift).
- Field labels (a TSV of field<TAB>title, as the seed selection writes): normalized mutual information
  between each document's top concept and its field.

    python tools/compare_schemes.py --db DB --schemes wikicat,anzsrc,openalex,isq \
        --categories doc_categories.json --fields selected_titles.tsv
"""
import argparse
import asyncio
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def nmi(xs, ys) -> float:
    n = len(xs)
    cx, cy, cxy = Counter(xs), Counter(ys), Counter(zip(xs, ys))
    mi = sum(c / n * math.log((c / n) / (cx[x] / n * cy[y] / n)) for (x, y), c in cxy.items())
    hx = -sum(c / n * math.log(c / n) for c in cx.values())
    hy = -sum(c / n * math.log(c / n) for c in cy.values())
    return 2 * mi / (hx + hy) if hx + hy else 0.0


async def doc_concepts(db: Path, scheme: str):
    """{doc title: [(score, concept label), ...] best first} for the scheme's doc-level edges."""
    from docmem import store

    graph, _ = await store.engines()
    rows = await graph.query(
        "MATCH (f:Node)-[r:EDGE]->(c:Node) WHERE f.type = 'DocFile' AND c.type = 'DocConcept' "
        "AND r.relationship_name = 'about_concept' RETURN f.id, c.id, c.name, c.properties, r.properties", {})
    titles = {n["id"]: store.node_props(n).get("title") for n in
              await graph.get_nodes(list({r[0] for r in rows}))}
    out = defaultdict(list)
    for fid, cid, cname, cprops, rprops in rows:
        cp = json.loads(cprops) if isinstance(cprops, str) else (cprops or {})
        if cp.get("scheme") != scheme:
            continue
        rp = json.loads(rprops) if isinstance(rprops, str) else (rprops or {})
        out[titles.get(fid)].append((float(rp.get("score", 0)), cname))
    return {t: sorted(v, reverse=True) for t, v in out.items() if t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--schemes", required=True)
    ap.add_argument("--categories", type=Path, required=True)
    ap.add_argument("--fields", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=20000)
    a = ap.parse_args()
    from docmem import store

    store.configure(a.db)
    cats = json.loads(a.categories.read_text())
    fields = {}
    for line in a.fields.read_text().splitlines():
        if "\t" in line:
            f, t = line.split("\t", 1)
            fields[t.strip()] = f.strip()
    by_cat = defaultdict(set)
    for t, cs in cats.items():
        for c in cs:
            by_cat[c].add(t)
    same = {tuple(sorted(p)) for docs in by_cat.values() if 2 <= len(docs) <= 50
            for p in itertools.combinations(sorted(docs), 2)}
    rng = random.Random(1)
    titles = sorted(cats)
    rand = {tuple(sorted(rng.sample(titles, 2))) for _ in range(a.pairs)}
    report = {"category_pairs": len(same), "random_pairs": len(rand)}

    async def go():
        import cognee

        await cognee.run_migrations()
        for scheme in a.schemes.split(","):
            dc = await doc_concepts(a.db, scheme)
            top3 = {t: {c for _, c in v[:3]} for t, v in dc.items()}

            def share(pairs):
                ok = [p for p in pairs if p[0] in top3 and p[1] in top3]
                return sum(1 for x, y in ok if top3[x] & top3[y]) / max(len(ok), 1)

            s_same, s_rand = share(same), share(rand)
            docs = [t for t in dc if t in fields]
            report[scheme] = {
                "docs": len(dc), "distinct_top_concepts": len({dc[t][0][1] for t in dc}),
                "mean_top_score": round(sum(v[0][0] for v in dc.values()) / max(len(dc), 1), 3),
                "share_top3_same_category": round(s_same, 3), "share_top3_random": round(s_rand, 3),
                "lift": round(s_same / s_rand, 2) if s_rand else None,
                "nmi_top_concept_vs_field": round(nmi([dc[t][0][1] for t in docs], [fields[t] for t in docs]), 3),
            }
        print(json.dumps(report, indent=1))

    asyncio.run(go())


if __name__ == "__main__":
    main()
