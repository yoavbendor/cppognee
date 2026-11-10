"""Graph adapter that keeps cognee's graph in two nanolance (Lance-format) tables.

``GRAPH_DATABASE_URL`` is the directory holding ``nodes.lance`` and ``edges.lance``.
Blocking Lance work runs in worker threads (``asyncio.to_thread``); writes are
serialized per adapter by an asyncio lock, and every operation opens the tables at
their current version, so several adapters on one directory see each other's writes.
"""

import asyncio
import json
import os
import threading
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from nanolance import lance

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import JSONEncoder
from cognee.shared.logging_utils import get_logger

from .tables import (
    EDGE_INDEXES,
    EDGE_SCHEMA,
    EDGES_TABLE,
    NODE_INDEXES,
    NODE_SCHEMA,
    NODES_TABLE,
    PROVENANCE_COLUMNS,
)

logger = get_logger()

# IN lists are sent to the Lance filter parser in slices of this size.
_IN_CHUNK = 500
_NODE_READ = ("id", "name", "type", "properties")
_EDGE_READ = ("source_id", "target_id", "relationship_name", "properties")
_CORE_NODE_KEYS = {"id", "name", "type"}
_ATTRIBUTE_FILTERS = {"id", "name", "type"}


def _quote(values: Iterable[str]) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def _chunks(values: list[str]) -> Iterable[list[str]]:
    for start in range(0, len(values), _IN_CHUNK):
        yield values[start : start + _IN_CHUNK]


def _unique(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(v) for v in values))


def _node_props(name: str, typ: str, properties: str | None) -> dict[str, Any]:
    data = {"name": name, "type": typ}
    if properties:
        data.update(json.loads(properties))
    return data


def _edge_props(properties: str | None) -> dict[str, Any]:
    return json.loads(properties) if properties else {}


def _component_sizes(node_ids: list[str], edges: list[tuple[str, str]]) -> list[int]:
    """Connected-component sizes (undirected), largest first. Isolated nodes count as size one."""
    parent: dict[str, str] = {node_id: node_id for node_id in node_ids}

    def find(item: str) -> str:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    for source_id, target_id in edges:
        parent.setdefault(source_id, source_id)
        parent.setdefault(target_id, target_id)
        root_a, root_b = find(source_id), find(target_id)
        if root_a != root_b:
            parent[root_b] = root_a

    sizes: dict[str, int] = defaultdict(int)
    for node_id in parent:
        sizes[find(node_id)] += 1
    return sorted(sizes.values(), reverse=True)


class NanolanceGraphAdapter(GraphDBInterface):
    """Graph-as-tables adapter on nanolance. Local directory only; no Cypher."""

    supports_cypher_queries = False

    def __init__(
        self,
        graph_database_url: str,
        graph_database_username: str = "",
        graph_database_password: str = "",
        graph_database_port: str = "",
        graph_database_key: str = "",
        database_name: str = "",
    ) -> None:
        """Bind to the graph directory in ``graph_database_url``.

        ``create_graph_engine`` passes the connection fields to every adapter registered
        in ``supported_databases``. Username, password, port and database name are
        accepted and ignored (a local directory has none); a remote key is refused.
        """
        if not graph_database_url:
            raise OSError(
                "Missing nanolance graph directory: set GRAPH_DATABASE_URL to an absolute "
                "directory path."
            )
        if graph_database_key:
            raise OSError(
                "Remote nanolance graphs are not supported; unset GRAPH_DATABASE_KEY to use a "
                "local directory."
            )
        self.database_path = os.path.abspath(graph_database_url)
        self._write_lock = asyncio.Lock()
        self._cache_lock = threading.Lock()
        # Key columns of the newest table version, per (table, column). Reused by
        # multi-id lookups until the next write creates a new version.
        self._key_cache: dict[tuple[str, str], tuple[int, pa.Array]] = {}
        self._initialized = False

    # ------------------------------------------------------------------ table access (threads)

    def _open(self, table: str):
        return lance.dataset(os.path.join(self.database_path, f"{table}.lance"))

    def _t_key_column(self, ds, table: str, column: str) -> pa.ChunkedArray:
        with self._cache_lock:
            hit = self._key_cache.get((table, column))
        if hit is not None and hit[0] == ds.version:
            return hit[1]
        # One Array, not a ChunkedArray: pyarrow segfaults in is_in/indices_nonzero on a
        # chunked column with zero chunks, which is what an empty table yields.
        values = ds.to_table(columns=[column])[column].combine_chunks()
        with self._cache_lock:
            self._key_cache[(table, column)] = (ds.version, values)
        return values

    def _t_take_by_key(self, table: str, column: str, keys: list[str], columns) -> pa.Table:
        ds = self._open(table)
        if not keys:
            return ds.schema.empty_table().select(list(columns))
        positions = pc.indices_nonzero(
            pc.is_in(self._t_key_column(ds, table, column), pa.array(keys))
        ).to_pylist()
        if not positions:
            return ds.schema.empty_table().select(list(columns))
        return ds.take(positions, columns=list(columns))

    def _t_rows_where(self, table: str, column: str, keys: list[str], columns) -> list[dict]:
        """Rows whose ``column`` is in ``keys``, read through the table's scalar index."""
        ds = self._open(table)
        rows: list[dict] = []
        for chunk in _chunks(keys):
            filter_sql = f"{column} IN ({_quote(chunk)})"
            rows.extend(ds.to_table(columns=list(columns), filter=filter_sql).to_pylist())
        return rows

    def _t_nodes_by_ids(self, node_ids: Iterable[str]) -> pa.Table:
        ids = _unique(node_ids)
        if len(ids) == 1:
            return self._open(NODES_TABLE).to_table(
                columns=list(_NODE_READ), filter=f"id = {_quote(ids)}"
            )
        return self._t_take_by_key(NODES_TABLE, "id", ids, _NODE_READ)

    def _t_edges_touching(self, node_ids: Iterable[str], edge_types: list[str] | None) -> pa.Table:
        ids = _unique(node_ids)
        ds = self._open(EDGES_TABLE)
        if not ids:
            return ds.schema.empty_table().select(list(_EDGE_READ))
        members = pa.array(ids)
        mask = pc.or_(
            pc.is_in(self._t_key_column(ds, EDGES_TABLE, "source_id"), members),
            pc.is_in(self._t_key_column(ds, EDGES_TABLE, "target_id"), members),
        )
        if edge_types:
            mask = pc.and_(
                mask,
                pc.is_in(
                    self._t_key_column(ds, EDGES_TABLE, "relationship_name"),
                    pa.array(list(edge_types)),
                ),
            )
        positions = pc.indices_nonzero(mask).to_pylist()
        if not positions:
            return ds.schema.empty_table().select(list(_EDGE_READ))
        return ds.take(positions, columns=list(_EDGE_READ))

    def _t_induced_edges(self, node_ids: Iterable[str]) -> pa.Table:
        """Edges whose source and target are both in ``node_ids``."""
        ids = _unique(node_ids)
        touching = self._t_edges_touching(ids, None)
        if touching.num_rows == 0:
            return touching
        members = pa.array(ids)
        keep = pc.and_(
            pc.is_in(touching["source_id"], members), pc.is_in(touching["target_id"], members)
        )
        return touching.filter(keep)

    @staticmethod
    def _node_tuples(table: pa.Table) -> list[tuple[str, dict[str, Any]]]:
        return [
            (node_id, _node_props(name, typ, props))
            for node_id, name, typ, props in zip(*(table[c].to_pylist() for c in _NODE_READ))
        ]

    @staticmethod
    def _node_dicts(table: pa.Table) -> list[dict[str, Any]]:
        return [
            {"id": node_id, **data} for node_id, data in NanolanceGraphAdapter._node_tuples(table)
        ]

    @staticmethod
    def _edge_tuples(table: pa.Table) -> list[tuple[str, str, str, dict[str, Any]]]:
        return [
            (source, target, rel, _edge_props(props))
            for source, target, rel, props in zip(*(table[c].to_pylist() for c in _EDGE_READ))
        ]

    # ------------------------------------------------------------------ table lifecycle

    def _t_create_tables(self) -> None:
        os.makedirs(self.database_path, exist_ok=True)
        for table, schema in ((NODES_TABLE, NODE_SCHEMA), (EDGES_TABLE, EDGE_SCHEMA)):
            path = os.path.join(self.database_path, f"{table}.lance")
            if not os.path.exists(path):
                lance.write_dataset(schema.empty_table(), path, mode="create")
        self._t_index(NODES_TABLE, NODE_INDEXES)
        self._t_index(EDGES_TABLE, EDGE_INDEXES)

    def _t_index(self, table: str, indexes: tuple[tuple[str, str], ...]) -> None:
        """Create missing scalar indexes, then fold fragments written since the last build.

        An index created on an empty table covers no fragments, and appends add fragments
        the index does not cover; ``optimize_indices`` folds them in. Filters stay correct
        meanwhile (uncovered fragments are scanned), only slower.
        """
        ds = self._open(table)
        covered = {field for index in ds.list_indices() for field in index["fields"]}
        for column, kind in indexes:
            if column not in covered:
                ds = ds.create_scalar_index(column, kind)
        if ds.list_indices():
            ds.optimize.optimize_indices()

    async def initialize(self) -> None:
        """Create the two tables and their indexes if they do not exist. Idempotent."""
        if self._initialized:
            return
        async with self._write_lock:
            if not self._initialized:
                await asyncio.to_thread(self._t_create_tables)
                self._initialized = True

    async def _write(self, operation, *args) -> None:
        async with self._write_lock:
            await asyncio.to_thread(operation, *args)

    # ------------------------------------------------------------------ writes (threads)

    def _t_merge_nodes(self, rows: list[dict]) -> None:
        existing = {
            row["id"]: row
            for row in self._t_rows_where(
                NODES_TABLE,
                "id",
                [r["id"] for r in rows],
                ("id", "created_at", *PROVENANCE_COLUMNS),
            )
        }
        now = datetime.now(timezone.utc)
        for row in rows:
            old = existing.get(row["id"])
            row["created_at"] = old["created_at"] if old else now
            for column in PROVENANCE_COLUMNS:
                row[column] = old[column] if old else None
        table = pa.Table.from_pylist(rows, schema=NODE_SCHEMA)
        self._open(NODES_TABLE).merge_insert(
            "id"
        ).when_matched_update_all().when_not_matched_insert_all().execute(table)
        self._t_index(NODES_TABLE, NODE_INDEXES)

    def _t_merge_edges(self, rows: list[dict]) -> None:
        endpoints = _unique([r["source_id"] for r in rows] + [r["target_id"] for r in rows])
        present = {row["id"] for row in self._t_rows_where(NODES_TABLE, "id", endpoints, ("id",))}
        rows = [r for r in rows if r["source_id"] in present and r["target_id"] in present]
        if not rows:
            return
        existing = {
            row["edge_key"]: row
            for row in self._t_rows_where(
                EDGES_TABLE,
                "edge_key",
                [r["edge_key"] for r in rows],
                ("edge_key", "created_at", *PROVENANCE_COLUMNS),
            )
        }
        now = datetime.now(timezone.utc)
        for row in rows:
            old = existing.get(row["edge_key"])
            row["created_at"] = old["created_at"] if old else now
            for column in PROVENANCE_COLUMNS:
                row[column] = old[column] if old else None
        table = pa.Table.from_pylist(rows, schema=EDGE_SCHEMA)
        self._open(EDGES_TABLE).merge_insert(
            "edge_key"
        ).when_matched_update_all().when_not_matched_insert_all().execute(table)
        self._t_index(EDGES_TABLE, EDGE_INDEXES)

    def _t_delete_nodes(self, node_ids: list[str]) -> None:
        # Edges first, so no edge is left pointing at a deleted node.
        for chunk in _chunks(node_ids):
            ids = _quote(chunk)
            self._open(EDGES_TABLE).delete(f"source_id IN ({ids}) OR target_id IN ({ids})")
        for chunk in _chunks(node_ids):
            self._open(NODES_TABLE).delete(f"id IN ({_quote(chunk)})")

    def _t_delete_graph(self) -> None:
        self._t_clear(NODES_TABLE, NODE_SCHEMA)
        self._t_clear(EDGES_TABLE, EDGE_SCHEMA)
        self._t_index(NODES_TABLE, NODE_INDEXES)
        self._t_index(EDGES_TABLE, EDGE_INDEXES)

    def _t_clear(self, table: str, schema: pa.Schema) -> None:
        lance.write_dataset(
            schema.empty_table(),
            os.path.join(self.database_path, f"{table}.lance"),
            mode="overwrite",
        )

    # ------------------------------------------------------------------ row builders

    @staticmethod
    def _node_row(node: DataPoint | tuple[str, dict] | Any) -> dict[str, Any]:
        if isinstance(node, tuple):
            props = {**(node[1] or {}), "id": node[0]}
        elif hasattr(node, "model_dump"):
            props = node.model_dump()
        else:
            props = dict(vars(node))
        extra = {k: v for k, v in props.items() if k not in _CORE_NODE_KEYS}
        return {
            "id": str(props.get("id", "")),
            "name": str(props.get("name", "")),
            "type": str(props.get("type", "")),
            "properties": json.dumps(extra, cls=JSONEncoder),
        }

    @staticmethod
    def _edge_row(edge: tuple) -> dict[str, Any]:
        source_id, target_id, relationship_name = str(edge[0]), str(edge[1]), str(edge[2])
        for part in (source_id, relationship_name, target_id):
            if "|" in part:
                raise ValueError(
                    f"nanolance edge keys are 'source|relationship|target'; {part!r} contains '|'"
                )
        props = edge[3] if len(edge) > 3 and edge[3] else {}
        return {
            "source_id": source_id,
            "target_id": target_id,
            "relationship_name": relationship_name,
            "properties": json.dumps(props, cls=JSONEncoder),
            "edge_key": f"{source_id}|{relationship_name}|{target_id}",
        }

    # ------------------------------------------------------------------ public API: writes

    async def query(self, query: str, params: dict | None = None) -> list[Any]:
        """Not supported: this backend has no Cypher engine."""
        raise NotImplementedError(
            "The nanolance graph backend does not support raw Cypher queries. "
            "Use the typed adapter methods (add_nodes, get_neighborhood, ...) or a "
            "Cypher-capable backend (Ladybug, Neo4j)."
        )

    async def is_empty(self) -> bool:
        await self.initialize()
        return await asyncio.to_thread(lambda: self._open(NODES_TABLE).count_rows() == 0)

    async def add_node(
        self, node: DataPoint | str, properties: dict[str, Any] | None = None
    ) -> None:
        if isinstance(node, str):
            await self.add_nodes([(node, {**(properties or {}), "id": node})])
        else:
            await self.add_nodes([node])

    async def add_nodes(
        self,
        nodes: list[tuple[str, dict]] | list[DataPoint],
        source_ref_key: str | None = None,
        pipeline_run_id: str | None = None,
    ) -> None:
        """Upsert nodes by id. Provenance arguments are accepted but not stored yet."""
        if not nodes:
            return
        rows = list({r["id"]: r for r in map(self._node_row, nodes)}.values())
        await self.initialize()
        await self._write(self._t_merge_nodes, rows)

    async def delete_node(self, node_id: str) -> None:
        await self.delete_nodes([node_id])

    async def delete_nodes(self, node_ids: list[str]) -> None:
        ids = _unique(node_ids)
        if not ids:
            return
        await self.initialize()
        await self._write(self._t_delete_nodes, ids)

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_name: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        await self.add_edges(
            [(str(source_id), str(target_id), relationship_name, properties or {})]
        )

    async def add_edges(
        self,
        edges: list[tuple[str, str, str, dict[str, Any] | None]] | list,
        source_ref_key: str | None = None,
        pipeline_run_id: str | None = None,
    ) -> None:
        """Upsert edges by (source, relationship, target). Edges with a missing endpoint are skipped."""
        if not edges:
            return
        rows = list({r["edge_key"]: r for r in map(self._edge_row, edges)}.values())
        await self.initialize()
        await self._write(self._t_merge_edges, rows)

    async def delete_graph(self) -> None:
        await self.initialize()
        await self._write(self._t_delete_graph)

    # ------------------------------------------------------------------ public API: reads

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        nodes = await self.get_nodes([node_id])
        return nodes[0] if nodes else None

    async def get_nodes(self, node_ids: list[str]) -> list[dict[str, Any]]:
        await self.initialize()
        return await asyncio.to_thread(lambda: self._node_dicts(self._t_nodes_by_ids(node_ids)))

    async def get_graph_data(self) -> tuple[list, list]:
        await self.initialize()

        def read() -> tuple[list, list]:
            nodes = self._open(NODES_TABLE).to_table(columns=list(_NODE_READ))
            if nodes.num_rows == 0:
                return [], []
            edges = self._open(EDGES_TABLE).to_table(columns=list(_EDGE_READ))
            return self._node_tuples(nodes), self._edge_tuples(edges)

        return await asyncio.to_thread(read)

    async def get_filtered_graph_data(
        self, attribute_filters: list[dict[str, list[str | int]]], *, include_edges: bool = True
    ) -> tuple[list, list]:
        clauses: list[str] = []
        for filter_dict in attribute_filters:
            for attr, values in filter_dict.items():
                if attr not in _ATTRIBUTE_FILTERS:
                    raise CogneeValidationError(
                        f"Invalid attribute filter key '{attr}'. Allowed: "
                        f"{', '.join(sorted(_ATTRIBUTE_FILTERS))}."
                    )
                if values:
                    clauses.append(f"{attr} IN ({_quote(str(v) for v in values)})")
        if not clauses:
            return [], []
        await self.initialize()

        def read() -> tuple[list, list]:
            nodes = self._open(NODES_TABLE).to_table(
                columns=list(_NODE_READ), filter=" AND ".join(clauses)
            )
            if nodes.num_rows == 0:
                return [], []
            node_tuples = self._node_tuples(nodes)
            if not include_edges:
                return node_tuples, []
            edges = self._t_induced_edges(nodes["id"].to_pylist())
            return node_tuples, self._edge_tuples(edges)

        return await asyncio.to_thread(read)

    async def get_nodeset_subgraph(
        self, node_type: type[Any], node_name: list[str], node_name_filter_operator: str = "OR"
    ) -> tuple[list, list]:
        """Nodes of ``node_type`` named in ``node_name``, their neighbours, and the induced edges.

        ``OR`` keeps every neighbour of any matched node; ``AND`` keeps only neighbours
        adjacent to all of them.
        """
        if not node_name:
            return [], []
        label = node_type.__name__
        await self.initialize()

        def read() -> tuple[list, list]:
            primary = _unique(
                self._open(NODES_TABLE)
                .to_table(
                    columns=["id"],
                    filter=f"type = {_quote([label])} AND name IN ({_quote(node_name)})",
                )["id"]
                .to_pylist()
            )
            if not primary:
                return [], []
            members = set(primary)
            links: dict[str, set[str]] = defaultdict(set)
            touching = self._t_edges_touching(primary, None)
            for source, target in zip(
                touching["source_id"].to_pylist(), touching["target_id"].to_pylist()
            ):
                if source in members:
                    links[target].add(source)
                if target in members:
                    links[source].add(target)
            if node_name_filter_operator == "OR":
                neighbours = set(links)
            else:
                neighbours = {n for n, matched in links.items() if len(matched) == len(members)}
            all_ids = members | neighbours
            nodes = self._t_nodes_by_ids(all_ids)
            return self._node_tuples(nodes), self._edge_tuples(self._t_induced_edges(all_ids))

        return await asyncio.to_thread(read)

    async def get_neighborhood(
        self,
        node_ids: list[str],
        depth: int = 1,
        edge_types: list[str] | None = None,
    ) -> tuple[list, list]:
        """Seeds plus every node within ``depth`` hops, and the edges among them.

        Breadth-first by frontier: each hop reads the edges touching the frontier once.
        With ``edge_types`` only those relationships are traversed.
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        seeds = _unique(node_ids)
        if not seeds:
            return [], []
        await self.initialize()

        def read() -> tuple[list, list]:
            seen = set(seeds)
            frontier = set(seeds)
            for _ in range(depth):
                if not frontier:
                    break
                hop = self._t_edges_touching(frontier, edge_types)
                next_frontier: set[str] = set()
                for source, target in zip(
                    hop["source_id"].to_pylist(), hop["target_id"].to_pylist()
                ):
                    for node_id in (source, target):
                        if node_id not in seen:
                            seen.add(node_id)
                            next_frontier.add(node_id)
                frontier = next_frontier
            nodes = self._t_nodes_by_ids(seen)
            if nodes.num_rows == 0:
                return [], []
            return self._node_tuples(nodes), self._edge_tuples(self._t_induced_edges(seen))

        return await asyncio.to_thread(read)

    async def get_connections(self, node_id: str) -> list[tuple[dict, dict, dict]]:
        """``(node, {"relationship_name": ..., **edge_props}, other)`` for each edge of ``node_id``."""
        nid = str(node_id)
        await self.initialize()

        def read() -> list[tuple[dict, dict, dict]]:
            touching = self._t_edges_touching([nid], None)
            if touching.num_rows == 0:
                return []
            edges = self._edge_tuples(touching)
            others = {t if s == nid else s for s, t, _, _ in edges}
            by_id = {n["id"]: n for n in self._node_dicts(self._t_nodes_by_ids(others | {nid}))}
            me = by_id.get(nid, {"id": nid})
            return [
                (
                    me,
                    {"relationship_name": rel, **props},
                    by_id.get(t if s == nid else s, {"id": t if s == nid else s}),
                )
                for s, t, rel, props in edges
            ]

        return await asyncio.to_thread(read)

    async def get_edges(self, node_id: str) -> list[tuple[dict, str, dict]]:
        """``(node, relationship_name, other)`` for each edge touching ``node_id``."""
        nid = str(node_id)
        await self.initialize()

        def read() -> list[tuple[dict, str, dict]]:
            touching = self._t_edges_touching([nid], None)
            if touching.num_rows == 0:
                return []
            edges = self._edge_tuples(touching)
            others = {t if s == nid else s for s, t, _, _ in edges}
            by_id = {n["id"]: n for n in self._node_dicts(self._t_nodes_by_ids(others | {nid}))}
            me = by_id.get(nid, {"id": nid})
            return [
                (me, rel, by_id.get(t if s == nid else s, {"id": t if s == nid else s}))
                for s, t, rel, _ in edges
            ]

        return await asyncio.to_thread(read)

    async def get_neighbors(self, node_id: str) -> list[dict[str, Any]]:
        nid = str(node_id)
        await self.initialize()

        def read() -> list[dict[str, Any]]:
            touching = self._t_edges_touching([nid], None)
            others = _unique(
                t if s == nid else s
                for s, t in zip(
                    touching["source_id"].to_pylist(), touching["target_id"].to_pylist()
                )
            )
            return self._node_dicts(self._t_nodes_by_ids(others))

        return await asyncio.to_thread(read)

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        return bool(await self.has_edges([(str(source_id), str(target_id), relationship_name)]))

    async def has_edges(self, edges: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
        """The input edges that exist, as ``(source, target, relationship)`` tuples."""
        candidates = [(str(e[0]), str(e[1]), str(e[2])) for e in edges]
        if not candidates:
            return []
        await self.initialize()

        def read() -> set[str]:
            keys = _unique(f"{s}|{r}|{t}" for s, t, r in candidates)
            return {
                row["edge_key"]
                for row in self._t_rows_where(EDGES_TABLE, "edge_key", keys, ("edge_key",))
            }

        present = await asyncio.to_thread(read)
        return [c for c in dict.fromkeys(candidates) if f"{c[0]}|{c[2]}|{c[1]}" in present]

    async def get_graph_counts(self) -> tuple[int, int]:
        await self.initialize()

        def read() -> tuple[int, int]:
            return (
                self._open(NODES_TABLE).count_rows(),
                self._open(EDGES_TABLE).count_rows(),
            )

        return await asyncio.to_thread(read)

    async def get_graph_metrics(self, include_optional: bool = False) -> dict[str, Any]:
        """Counts, mean degree, density and connected components. Optional diameter and path
        metrics are reported as -1 (not computed), as the Turso adapter does."""
        await self.initialize()

        def read() -> dict[str, Any]:
            node_ids = self._open(NODES_TABLE).to_table(columns=["id"])["id"].to_pylist()
            edges = self._open(EDGES_TABLE).to_table(columns=["source_id", "target_id"])
            sources = edges["source_id"].to_pylist()
            targets = edges["target_id"].to_pylist()
            num_nodes, num_edges = len(node_ids), len(sources)
            sizes = _component_sizes(node_ids, list(zip(sources, targets)))
            metrics: dict[str, Any] = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "mean_degree": (2 * num_edges) / num_nodes if num_nodes else None,
                "edge_density": num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
                "num_connected_components": len(sizes),
                "sizes_of_connected_components": sizes,
                "num_selfloops": -1,
                "diameter": -1,
                "avg_shortest_path_length": -1,
                "avg_clustering": -1,
            }
            if include_optional:
                metrics["num_selfloops"] = sum(s == t for s, t in zip(sources, targets))
            return metrics

        return await asyncio.to_thread(read)
