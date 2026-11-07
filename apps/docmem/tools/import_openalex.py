"""Import the OpenAlex topic hierarchy (CC0) as a SKOS Turtle file for `docmem ontology load`:
domain > field > subfield > topic, with each topic's description and keywords in its definition.

    OPENALEX_API_KEY=... python tools/import_openalex.py --out openalex_topics.ttl [--domains "Physical Sciences"]

The key is read from the environment only. --domains keeps the named domains (default: all four).
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = "https://api.openalex.org"


def get(path: str, params: dict, key: str) -> dict:
    q = urllib.parse.urlencode({**params, "api_key": key})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{API}/{path}?{q}", timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def all_of(path: str, key: str, select: str) -> list[dict]:
    out, cursor = [], "*"
    while cursor:
        d = get(path, {"per-page": 200, "cursor": cursor, "select": select}, key)
        out += d["results"]
        cursor = d["meta"].get("next_cursor")
    return out


def lit(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"@en'


def short(oid: str) -> str:
    return oid.rstrip("/").rsplit("/", 1)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default="", help="comma-separated domain names to keep (default: all)")
    a = ap.parse_args()
    key = os.environ.get("OPENALEX_API_KEY")
    if not key:
        raise SystemExit("set OPENALEX_API_KEY")
    keep = {d.strip().lower() for d in a.domains.split(",") if d.strip()}
    domains = all_of("domains", key, "id,display_name,description")
    fields = all_of("fields", key, "id,display_name,description,domain")
    subfields = all_of("subfields", key, "id,display_name,description,field,domain")
    topics = all_of("topics", key, "id,display_name,description,keywords,subfield,field,domain")
    ok = lambda dom: not keep or dom.lower() in keep  # noqa: E731
    lines = ["@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
             "@prefix dcterms: <http://purl.org/dc/terms/> .", "",
             "<https://openalex.org/topics> a skos:ConceptScheme ; dcterms:title \"OpenAlex topics\"@en ;",
             "  dcterms:license <https://creativecommons.org/publicdomain/zero/1.0/> ;",
             f"  dcterms:source <{API}> .", ""]

    def concept(uri, label, definition, broader=None):
        lines.append(f"<{uri}> a skos:Concept ; skos:prefLabel {lit(label)} ;")
        if broader:
            lines.append(f"  skos:broader <{broader}> ;")
        lines.append(f"  skos:definition {lit(definition)} ; skos:inScheme <https://openalex.org/topics> .")

    counts = {"domain": 0, "field": 0, "subfield": 0, "topic": 0}
    for d in domains:
        if ok(d["display_name"]):
            concept(d["id"], d["display_name"], d.get("description") or d["display_name"])
            counts["domain"] += 1
    for f in fields:
        if ok(f["domain"]["display_name"]):
            path = f"{f['domain']['display_name']} > {f['display_name']}"
            concept(f["id"], f["display_name"], f"{path}: {f['description']}" if f.get("description") else path,
                    f["domain"]["id"])
            counts["field"] += 1
    for s in subfields:
        if ok(s["domain"]["display_name"]):
            path = f"{s['field']['display_name']} > {s['display_name']}"
            concept(s["id"], s["display_name"], f"{path}: {s['description']}" if s.get("description") else path,
                    s["field"]["id"])
            counts["subfield"] += 1
    for t in sorted(topics, key=lambda t: short(t["id"])):
        if ok(t["domain"]["display_name"]):
            kws = "; ".join(k["display_name"] if isinstance(k, dict) else str(k) for k in t.get("keywords") or [])
            definition = (t.get("description") or t["display_name"]) + (f" Keywords: {kws}." if kws else "")
            concept(t["id"], t["display_name"], definition, t["subfield"]["id"])
            counts["topic"] += 1
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(json.dumps(counts), file=sys.stderr)


if __name__ == "__main__":
    main()
