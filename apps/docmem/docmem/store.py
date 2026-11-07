"""The cognee side: configuration, the DataPoint types, and reads from cognee's graph and vector stores.

One database directory holds everything (``--db``): cognee's relational DB, its Ladybug graph and its
vector tables (Lance, written by nanolance), plus ``docmem.json``, which pins the embedding model the
index was built with (questions must be embedded with the same one). No API key is ever written there.

What is stored, per document: a DocFile node (URI, SHA-256, size, title), a DocSection node per
heading (heading path, byte offset and length, line) and a DocPassage node per passage (byte offset
and length, lines, SHA-256 of the bytes), with edges passage -> section -> file. The vector tables hold
one vector per passage (the embedding of "heading path + passage text") and per section (its heading
path). Passage text is in neither store: DocPassage marks its embedded field as not stored (cognee's
``store_index_text``) and excludes it from the graph node.
"""
import json
import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

NLSITE = Path(__file__).with_name("_nlsite")
PASSAGES = "DocPassage_embed_text"
SECTIONS = "DocSection_path"
DATASET = "docmem"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def embedding_env(spec: str, dims: int | None, api_base: str | None, api_key: str | None) -> dict:
    """Environment for cognee's embedding engine. spec: "fastembed:<model>" (local, no key) or
    "openai:<model>" (the OpenAI-compatible server at api_base)."""
    kind, _, model = spec.partition(":")
    if kind == "fastembed":
        if dims is None:
            from fastembed import TextEmbedding

            dims = next(m["dim"] for m in TextEmbedding.list_supported_models() if m["model"] == model)
        return {"EMBEDDING_PROVIDER": "fastembed", "EMBEDDING_MODEL": model, "EMBEDDING_DIMENSIONS": str(dims)}
    if kind == "openai":
        if not api_base:
            raise SystemExit("docmem: an openai: embedding model needs --api-base (or DOCMEM_API_BASE)")
        if dims is None:
            raise SystemExit("docmem: an openai: embedding model needs --embed-dims (e.g. 1536)")
        env = {"EMBEDDING_PROVIDER": "openai_compatible", "EMBEDDING_MODEL": model,
               "EMBEDDING_DIMENSIONS": str(dims), "EMBEDDING_ENDPOINT": api_base}
        if api_key:
            env["EMBEDDING_API_KEY"] = api_key
        return env
    raise SystemExit(f"docmem: unknown embedding model {spec!r}: use fastembed:<model> or openai:<model>")


def configure(db: Path, embed: str | None = None, embed_dims: int | None = None, api_base: str | None = None,
              api_key: str | None = None, create: bool = False) -> dict:
    """Point cognee at ``db`` and its pinned embedding model. Must run before cognee is imported."""
    if "cognee" in sys.modules:
        raise RuntimeError("docmem.store.configure must run before cognee is imported")
    db = db.resolve()
    cfg_path = db / "docmem.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if embed and (embed != cfg["embed"] or (embed_dims and embed_dims != cfg["dims"])):
            raise SystemExit(f"docmem: {db} was built with {cfg['embed']} ({cfg['dims']} dims); "
                             f"a different model needs a new --db")
    elif create:
        embed = embed or "fastembed:BAAI/bge-small-en-v1.5"
        env = embedding_env(embed, embed_dims, api_base, api_key)
        cfg = {"protocol": 1, "embed": embed, "dims": int(env["EMBEDDING_DIMENSIONS"]),
               "vector_backend": os.environ.get("DOCMEM_VECTOR_BACKEND", "nanolance")}
        db.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    else:
        raise SystemExit(f"docmem: no index at {db} (run docmem index first)")

    os.environ["DOCMEM_VECTOR_BACKEND"] = cfg["vector_backend"]
    os.environ.update(embedding_env(cfg["embed"], cfg["dims"], api_base, api_key))
    os.environ.update({
        "TELEMETRY_DISABLED": "1",
        "DATA_ROOT_DIRECTORY": str(db / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(db / "system"),
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "PYTHONPATH": os.pathsep.join(filter(None, [str(NLSITE), os.environ.get("PYTHONPATH")])),
    })
    os.environ.setdefault("LOG_LEVEL", "ERROR")
    if cfg["vector_backend"] == "nanolance":
        import nanolance.lancedb

        nanolance.lancedb.install_as_lancedb()
    return cfg


_models = None


def models():
    """The DataPoint types (DocFile, DocSection, DocPassage, DocEntity, DocSummary, DocTopic, DocGap,
    DocTarget),
    defined once cognee is importable."""
    global _models
    if _models is None:
        from cognee.low_level import DataPoint
        from pydantic import Field

        class DocTopic(DataPoint):
            name: str  # LLM-derived; one node per normalized name, shared by documents
            key: str
            metadata: dict = {"index_fields": ["name"], "identity_fields": ["key"]}

        class DocTarget(DataPoint):
            """What a link points at: a wiki title, a URL, a file, an ID (REQ-123) or a category. One
            node per target, shared by every passage that links to it; a document that IS the target
            has an "is_target" edge to it. No LLM involved."""
            name: str
            key: str
            kind: str
            metadata: dict = {"index_fields": [], "identity_fields": ["key"]}

        class DocFile(DataPoint):
            name: str  # the source URI (file:// or a pinned web revision)
            key: str
            title: str
            sha256: str
            size: int
            mtime: float
            url: str = ""  # the page to show a person (web sources)
            enriched_by: str = ""
            topics: list[DocTopic] = []
            metadata: dict = {"index_fields": ["title"], "identity_fields": ["key"]}

        class DocSection(DataPoint):
            name: str
            key: str
            path: str  # "Title › Chapter › Section"
            level: int
            offset: int
            length: int
            line: int
            doc: DocFile
            metadata: dict = {"index_fields": ["path"], "identity_fields": ["key"]}

        class DocEntity(DataPoint):
            """An entity as one document describes it (LLM-derived). ``improve`` links the same
            entity across documents; nothing merges them at ingest."""
            name: str
            key: str
            kind: str
            description: str
            doc_uri: str
            metadata: dict = {"index_fields": [], "identity_fields": ["key"]}

        class DocPassage(DataPoint):
            name: str
            key: str
            path: str
            offset: int
            length: int
            line: int
            end_line: int
            sha256: str
            section_offset: int  # the enclosing section's byte range, for --context section
            section_length: int
            # how to get the bytes back: offsets are into the file itself (extractor "") or into the
            # text the extractor makes from the source whose hash is doc_sha256 (a wiki revision)
            extractor: str = ""
            doc_sha256: str = ""
            # embedded, then dropped: not in the graph node (exclude) and not in the vector row
            embed_text: str = Field(default="", exclude=True)
            section: DocSection
            mentions: list[DocEntity] = []
            metadata: dict = {"index_fields": ["embed_text"], "identity_fields": ["key"],
                              "store_index_text": False}

        class DocSummary(DataPoint):
            name: str
            key: str
            text: str  # LLM-written, stored and embedded (searchable)
            part: int
            section: DocSection
            metadata: dict = {"index_fields": ["text"], "identity_fields": ["key"]}

        class DocGap(DataPoint):
            """Work the one read could not do (enrichment over budget or failed): closed on a later
            read of the source, all of a document's gaps at once."""
            name: str
            key: str
            kind: str
            reason: str
            path: str  # the section's heading path
            part: int
            section: DocSection
            metadata: dict = {"index_fields": [], "identity_fields": ["key"]}

        _models = (DocFile, DocSection, DocPassage, DocEntity, DocSummary, DocTopic, DocGap, DocTarget)
    return _models


def target(kind: str, name: str):
    """The DocTarget node for a link target (a wiki title, a URL, a file:// URI, an ID, a category)."""
    DocTarget = models()[7]
    return DocTarget(name=name, key=key("target", kind, name), kind=kind)


def self_target(doc: dict):
    """The DocTarget a document itself answers to: its wiki title, else its URI."""
    if str(doc.get("extractor", "")).startswith("wikiplain/"):
        return target("wiki", doc["title"])
    return target("file", doc["uri"])


def key(*parts) -> str:
    """Identity value: a hash, since cognee normalizes identity strings (case, punctuation)."""
    import hashlib

    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def uri_path(uri: str) -> Path:
    u = urlparse(uri)
    if u.scheme != "file":
        raise ValueError(f"not a file:// URI: {uri}")
    return Path(unquote(u.path))


def read_span(uri: str, offset: int, length: int, sha256: str | None = None):
    """(text, status): the bytes [offset, offset+length) of the file, and "ok", "changed" (the bytes'
    hash differs from the indexed one) or "missing"."""
    import hashlib

    try:
        with open(uri_path(uri), "rb") as f:
            f.seek(offset)
            data = f.read(length)
    except FileNotFoundError:
        return "", "missing"
    status = "ok" if sha256 is None or hashlib.sha256(data).hexdigest() == sha256 else "changed"
    return data.decode("utf-8", "replace"), status


async def engines():
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector import get_vector_engine_async

    return await get_graph_engine(), await get_vector_engine_async()


def node_props(node: dict) -> dict:
    """A node's fields as one dict (get_nodes returns them flat; filtered queries nest "properties")."""
    if "properties" not in node:
        return dict(node)
    props = node.get("properties") or {}
    if isinstance(props, str):
        props = json.loads(props)
    return {**props, "id": node.get("id"), "name": node.get("name"), "type": node.get("type")}


async def nodes_of(graph, type_name: str, uris: list[str]) -> list[dict]:
    """Every node of ``type_name`` whose name (URI) is in ``uris``."""
    nodes, _ = await graph.get_filtered_graph_data([{"type": [type_name], "name": uris}], include_edges=False)
    out = []
    for n in nodes:
        if isinstance(n, (tuple, list)):
            nid, props = n[0], n[1]
            out.append(node_props({"id": nid, **props}) if isinstance(props, dict) else {"id": nid})
        else:
            out.append(node_props(n))
    return out
