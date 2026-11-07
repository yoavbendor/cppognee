"""Model checks in a model DB: nomograph-sysml's completeness checks on the model's export, kept as findings.
Model DBs only (never a prose DB). No LLM.

    docmem model-check --db MODELDB --model apollo11 [--nomograph nomograph-sysml]
    docmem model-gaps  --db MODELDB [--model apollo11] [--check NAME] [--under "Qualified::Prefix"]

``model-check`` exports the model as ``model-export`` does, runs ``nomograph-sysml check all --detail`` on that export
(in a temporary directory), and stores each finding as a ModelFinding node: name = the element's qualified name,
key = store.key("finding", check, element), with its check, severity, message, file, line and section_id (the DocSection
of the element, when the DB has it). Each finding with a section has an ``about`` edge to that section. Re-running
replaces the findings of the model: the old ones are deleted in the same pipeline run that writes the new ones.

``model-gaps`` lists the stored findings, optionally for one check and under one qualified-name prefix.

Severity is docmem's, since nomograph's findings carry none: ``error`` for DanglingReferences and MetamodelConformance
(a reference or a rule of the metamodel is broken), ``warning`` for the completeness checks (orphan, unverified and
unconnected elements, missing verification).
"""

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import store
from .model_export import export_db
from .ontology import _check_pipeline, nodes_of_type

NOMOGRAPH = "nomograph-sysml"
ERROR_CHECKS = frozenset({"DanglingReferences", "MetamodelConformance"})
_finding_type = None


@dataclass(frozen=True)
class Finding:
    check: str
    element: str  # the qualified name, as nomograph writes it
    file_path: str
    line: int  # 1-based
    message: str
    severity: str


def severity_of(check: str) -> str:
    return "error" if check in ERROR_CHECKS else "warning"


def parse_findings(doc: dict) -> list[Finding]:
    """The findings of ``nomograph-sysml check all --detail`` (its JSON): check_type, element, file_path, message and
    span (its start_line counts from 0)."""
    out = []
    for f in doc["findings"]:
        span = f.get("span") or {}
        out.append(
            Finding(
                check=f["check_type"],
                element=f.get("element") or "",
                file_path=f.get("file_path") or "",
                line=int(span.get("start_line", -1)) + 1,
                message=f["message"],
                severity=severity_of(f["check_type"]),
            )
        )
    return out


def select_gaps(rows: list[dict], check: str | None, under: str | None) -> list[dict]:
    """The stored findings for one check and under one qualified-name prefix (the element itself or any element below)."""
    return [
        r
        for r in rows
        if (check is None or r["check"] == check)
        and (under is None or r["name"] == under or r["name"].startswith(under + "::"))
    ]


def finding_type():
    """The ModelFinding DataPoint, defined once cognee is importable (as store.models())."""
    global _finding_type
    if _finding_type is None:
        from cognee.low_level import DataPoint

        class ModelFinding(DataPoint):
            name: str  # the element's qualified name
            key: str
            model: str
            check: str
            severity: str
            message: str
            file_path: str
            line: int
            section_id: str = ""  # the element's DocSection, when the DB has it
            metadata: dict = {"index_fields": [], "identity_fields": ["key"]}

        _finding_type = ModelFinding
    return _finding_type


def run_nomograph(binary: str, index: dict) -> dict:
    """``nomograph-sysml check all --detail`` on the index, as its JSON document."""
    exe = shutil.which(binary)
    if exe is None:
        raise SystemExit(
            f"docmem model-check: {binary} is not on PATH (install nomograph-sysml or pass --nomograph)"
        )
    with tempfile.TemporaryDirectory(prefix="docmem-check-") as tmp:
        path = Path(tmp) / "index.json"
        path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        proc = subprocess.run(
            [exe, "check", "all", "--detail", "--index", str(path)], capture_output=True, text=True
        )
    if proc.returncode != 0:
        raise SystemExit(
            f"docmem model-check: nomograph-sysml failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)


async def write_findings(model: str, findings: list[Finding], section_of: dict) -> dict:
    """Replace the model's ModelFinding nodes with these, in one cognee pipeline run."""
    import cognee
    from cognee.modules.pipelines import Task
    from cognee.tasks.storage import add_data_points

    ModelFinding = finding_type()
    graph, _ = await store.engines()
    old = [n["id"] for n in await nodes_of_type(graph, "ModelFinding") if n.get("model") == model]
    keyed = Counter(store.key("finding", f.check, f.element) for f in findings)
    shared = [f for f in findings if keyed[store.key("finding", f.check, f.element)] > 1]
    if shared:  # one node per key: two findings with one key would merge into one node
        raise SystemExit(
            f"docmem model-check: {len(shared)} findings share a key (check, element); "
            f"first: {shared[0].check} {shared[0].element}"
        )
    nodes, edges = [], []
    for f in findings:
        sid = section_of.get((model, f.element))
        node = ModelFinding(
            name=f.element,
            key=store.key("finding", f.check, f.element),
            model=model,
            check=f.check,
            severity=f.severity,
            message=f.message,
            file_path=f.file_path,
            line=f.line,
            section_id=str(sid) if sid else "",
        )
        nodes.append(node)
        if sid:
            edges.append(
                (str(node.id), str(sid), "about", {"edge_text": f"{f.check}: {f.element}"})
            )

    async def store_batch(items, ctx=None):
        if old:
            await graph.delete_nodes(old)
        if nodes:
            await add_data_points(nodes, ctx=ctx, graph_only=True, custom_edges=edges or None)
        return []

    result = await cognee.run_custom_pipeline(
        tasks=[Task(store_batch, needs_llm=False)], data=[{"batch": 0}], dataset=store.DATASET
    )
    _check_pipeline(result, "model-check")
    return {
        "stored": len(nodes),
        "replaced": len(old),
        "about_edges": len(edges),
        "without_section": len(nodes) - len(edges),
    }


async def check_async(model: str, binary: str) -> dict:
    graph, _ = await store.engines()
    export = await export_db(model, False, graph=graph)
    doc = run_nomograph(binary, export.index)
    findings = parse_findings(doc)
    stored = await write_findings(model, findings, export.section_of)
    return {
        "model": model,
        "nomograph_counts": doc["counts"],
        "findings_by_check": dict(Counter(f.check for f in findings)),
        **stored,
        "elements": export.report["elements"],
    }


def check_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="docmem model-check",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--db",
        type=Path,
        required=True,
        help="a model DB (scan-sysml, index and model-resolve first)",
    )
    ap.add_argument("--model", required=True, help="the model's name (scan-sysml --model-name)")
    ap.add_argument(
        "--nomograph", default=NOMOGRAPH, help="the nomograph-sysml binary (default: on PATH)"
    )
    a = ap.parse_args(argv)
    store.configure(a.db)
    summary = asyncio.run(check_async(a.model, a.nomograph))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


async def gaps_async(a) -> list[dict]:
    graph, _ = await store.engines()
    rows = [
        n
        for n in await nodes_of_type(graph, "ModelFinding")
        if a.model is None or n.get("model") == a.model
    ]
    return rows


def gaps_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="docmem model-gaps",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--model", default=None, help="only this model's findings")
    ap.add_argument("--check", default=None, help="only this check, e.g. UnverifiedRequirements")
    ap.add_argument(
        "--under",
        default=None,
        help="only elements under this qualified name, e.g. TechnicalRequirementsPackage",
    )
    a = ap.parse_args(argv)
    store.configure(a.db)
    rows = asyncio.run(gaps_async(a))
    picked = select_gaps(rows, a.check, a.under)
    per_check = Counter(r["check"] for r in select_gaps(rows, None, a.under))
    for check, n in sorted(per_check.items()):
        print(f"# {check}: {n}")
    for r in sorted(picked, key=lambda r: (r["check"], r["name"])):
        print(f"{r['check']:24s} {r['name']}  {r['file_path']}:{r['line']}  {r['message']}")
    print(f"# shown: {len(picked)}", file=sys.stderr)


if __name__ == "__main__":
    check_main()
