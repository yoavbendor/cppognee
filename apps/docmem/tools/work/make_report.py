"""Collect a decision-benchmark run into one Markdown report to paste back for review.

    python make_report.py --bench ./dbench_work --out docmem_work_report.md

The report holds metrics, timings, the decide-check response and a few disagreement examples (public benchmark
text only). URL hosts and anything that looks like a key or token are redacted.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

URL = re.compile(r"https?://[^\s/\"']+")
SECRETISH = re.compile(r"(?i)(api[-_ ]?key|authorization|bearer|token)([\"':= ]+)([^\s\"',]{8,})")


def redact(text: str) -> str:
    text = URL.sub("https://<endpoint>", text)
    return SECRETISH.sub(lambda m: f"{m.group(1)}{m.group(2)}<redacted>", text)


def short(s, n=300):
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def git_rev(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "?"


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    from importlib import metadata

    for dist in ("docmem", "nanolance", "cognee"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = "not installed"
    out["repo_commit"] = git_rev(Path(__file__).resolve().parent)
    out["decider_flavor"] = os.environ.get("DOCMEM_DECIDER_FLAVOR", "(auto)")
    out["decider_auth"] = os.environ.get("DOCMEM_DECIDER_AUTH", "api-key")
    out["decider_model"] = os.environ.get("DOCMEM_DECIDER_MODEL", "?")
    return out


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return "—" if v is None else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    b = a.bench / "dbench"
    lines = ["# docmem decision benchmark: work run", ""]
    lines += ["## Environment", "", "```json", json.dumps(versions(), indent=1), "```", ""]
    dc = a.bench / "decide_check.txt"
    lines += ["## decide-check (one request, parsed + raw)", "", "```",
              redact(dc.read_text(errors="replace"))[-6000:] if dc.exists() else "(missing)", "```", ""]
    rows, details = [], []
    for tdir in sorted(p for p in b.iterdir() if p.is_dir() and not p.name.startswith("_")):
        rep, run = tdir / "report.json", tdir / "run.json"
        if not rep.exists():
            rows.append(f"| {tdir.name} | (no report) | | | | | | | |")
            continue
        r = json.loads(rep.read_text())
        m = r.get("metrics", {})
        d, base = m.get("decider", {}), m.get("baseline", {})
        rs = json.loads(run.read_text()) if run.exists() else {}
        n = m.get("n") or r.get("n")
        secs = rs.get("seconds") or rs.get("wall_seconds")
        per = (secs / n) if secs and n else None
        rows.append(f"| {tdir.name} | {fmt(n)} | {fmt(base.get('acc'))} | {fmt(d.get('acc'))} | {fmt(m.get('chance'))} "
                    f"| {fmt(d.get('brier'))} | {fmt(d.get('auto5'))} | {fmt(d.get('auc'))} | {fmt(per)} |")
        ex = r.get("disagreement_examples") or []
        details.append(f"### {tdir.name}\n\n- run: `{json.dumps(rs)}`\n- extra baselines: "
                       f"`{json.dumps(m.get('extra_baselines', {}))}`")
        for e in ex[:3]:
            details.append(f"- example `{short(e.get('id'), 80)}` (decider right: {e.get('decider_right')}): gold "
                           f"`{short(e.get('gold'), 60)}`, decider `{short(e.get('decider'), 120)}`, baseline "
                           f"`{short(e.get('baseline'), 60)}`; state: {redact(short(e.get('state'), 300))}")
        details.append("")
    lines += ["## Results", "",
              "| task | n | baseline acc | decider acc | chance | decider Brier | auto@5% | AUC | s/item |",
              "|---|---|---|---|---|---|---|---|---|", *rows, "",
              "Reference (Kev-0.8B on a 4-core CPU): routing 0.170, answerability 0.663, concept_a 0.746, "
              "concept_b 0.542, sysml_kind 0.564, alignment 0.425, maturity 0.394.", "", "## Details", "", *details]
    log = a.bench / "run.log"
    if log.exists():
        tail = [ln for ln in log.read_text(errors="replace").splitlines() if ln.strip()][-40:]
        lines += ["## Last lines of run.log", "", "```", redact("\n".join(tail)), "```", ""]
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
