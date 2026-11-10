"""Graph adapter that keeps cognee's graph in two nanolance (Lance-format) tables.

``GRAPH_DATABASE_URL`` is the directory holding ``nodes.lance`` and ``edges.lance``.
Blocking Lance work runs in worker threads (``asyncio.to_thread``). Writes are serialized
per adapter by an asyncio lock and across processes by a file lock (``.write.lock`` in the
graph directory); every write reads the tables after taking the file lock, and every
operation opens the tables at their current version, so writers on one directory see each
other's writes. Reads take no lock.
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
from filelock import FileLock
from nanolance import lance

from cognee.base_config import get_base_config
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.graph.degree_seeds import EDGE_SAMPLE_ROWS
from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
    temporal_anchors_from_rows,
    timestamp_overlaps,
)
from cognee.infrastructure.databases.provenance import EdgeDeleteData, EdgeIdentity, NodeDeleteData
from cognee.infrastructure.databases.provenance.source_ref_state import (
    provenance_after_attach,
    provenance_after_remove,
)
from cognee.infrastructure.databases.provenance.source_refs import (
    get_dataset_id_from_source_ref_key,
    get_pipeline_run_id_from_source_run_ref,
    get_source_ref_key_from_source_run_ref,
)
from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import JSONEncoder
from cognee.shared.logging_utils import get_logger

from .tables import (
    EDGE_INDEXES,
    EDGE_SCHEMA,
    EDGES_TABLE,
    METADATA_SCHEMA,
    METADATA_TABLE,
    NODE_INDEXES,
    NODE_SCHEMA,
    NODES_TABLE,
    PROVENANCE_COLUMNS,
)

logger = get_logger()

# Cross-process lock file inside the graph directory; held around every write.
_WRITE_LOCK_FILE = ".write.lock"
# IN lists are sent to the Lance filter parser in slices of this size.
_IN_CHUNK = 500
_NODE_READ = ("id", "name", "type", "properties")
_EDGE_READ = ("source_id", "target_id", "relationship_name", "properties")
_NODE_COLUMNS = tuple(field.name for field in NODE_SCHEMA)
_EDGE_COLUMNS = tuple(field.name for field in EDGE_SCHEMA)
_CORE_NODE_KEYS = {"id", "name", "type"}
_ATTRIBUTE_FILTERS = {"id", "name", "type"}
_DEFAULT_FEEDBACK_WEIGHT = 0.5

# Properties improve() writes onto stored elements. A re-add of an existing element keeps
# these stored values instead of resetting them to the DataPoint defaults. Same keys as
# the Ladybug adapter.
LEARNED_NODE_PROPERTIES = ("feedback_weight", "truth_alignment", "truth_epoch")
LEARNED_EDGE_PROPERTIES = ("feedback_weight",)


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


def _props_blob(raw: str | None) -> dict[str, Any]:
    """Decode a properties JSON blob; anything that is not a JSON object yields ``{}``."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _with_learned(properties: str, stored: str | None, keys: tuple[str, ...]) -> str:
    """``properties`` with the learned ``keys`` of the ``stored`` blob carried over."""
    learned = _props_blob(stored)
    kept = {key: learned[key] for key in keys if key in learned}
    if not kept:
        return properties
    return json.dumps({**json.loads(properties), **kept}, cls=JSONEncoder)


def _edge_key(source_id: str, target_id: str, relationship_name: str) -> str:
    return f"{source_id}|{relationship_name}|{target_id}"


def _source_ref_map(source_ref_key, identities: list) -> dict:
    """Per-identity ref keys for a write. A str covers every identity; a dict must name each.

    The dict form raises KeyError for a missing identity on purpose: a partial mapping
    means the caller's ownership bookkeeping is broken, and an unstamped row would leak
    on deletion.
    """
    if source_ref_key is None:
        return {}
    if isinstance(source_ref_key, dict):
        return {identity: source_ref_key[identity] for identity in identities}
    return {identity: source_ref_key for identity in identities}


def _identity_columns(table: str) -> tuple[str, ...]:
    """Columns that identify a row's owner: the node id, or an edge's (source, target, relation)."""
    if table == NODES_TABLE:
        return ("id",)
    return ("source_id", "target_id", "relationship_name")


def _provenance_columns(row: dict | None) -> dict[str, list[str]]:
    """The four provenance lists of a stored row; a missing row or NULL column reads as []."""
    row = row or {}
    return {column: list(row.get(column) or []) for column in PROVENANCE_COLUMNS}


def _provenance_after_fold(row: dict | None, source_ref_key, pipeline_run_id) -> dict:
    """Provenance columns after a write stamped with ``source_ref_key`` (None: unchanged)."""
    current = _provenance_columns(row)
    if source_ref_key is None:
        return current
    return provenance_after_attach(
        current["source_ref_keys"],
        current["source_run_refs"],
        [source_ref_key],
        pipeline_run_id,
    )._asdict()


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
    # add_nodes / add_edges take a per-row source-ref mapping and stamp it in the same write.
    supports_per_row_source_refs = True

    def __init__(
        self,
        graph_database_url: str,
        graph_database_username: str = "",
        graph_database_password: str = "",
        graph_database_port: str = "",
        graph_database_key: str = "",
        database_name: str = "",
    ) -> None:
        """Bind to the graph directory in ``graph_database_url`` (default: under the system root).

        ``create_graph_engine`` passes the connection fields to every adapter registered
        in ``supported_databases``. Username, password, port and database name are
        accepted and ignored (a local directory has none); a remote key is refused.
        """
        if not graph_database_url:
            # Same default location as Ladybug's graph file: <system root>/databases/<name>.
            graph_database_url = os.path.join(
                get_base_config().system_root_directory, "databases", "cognee_graph_nanolance"
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
        tables = (
            (NODES_TABLE, NODE_SCHEMA),
            (EDGES_TABLE, EDGE_SCHEMA),
            (METADATA_TABLE, METADATA_SCHEMA),
        )
        for table, schema in tables:
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
                await asyncio.to_thread(self._t_locked, self._t_create_tables)
                self._initialized = True

    async def _write(self, operation, *args):
        """Run a blocking write under the adapter's write lock and return its result."""
        async with self._write_lock:
            return await asyncio.to_thread(self._t_locked, operation, *args)

    def _t_locked(self, operation, *args):
        """Run ``operation`` while holding the graph directory's cross-process file lock.

        Runs in a worker thread so the event loop never waits on the lock. The lock is taken
        before any table is read, and every read opens the newest table version, so the
        read-modify-write inside ``operation`` sees writes made by other processes.
        """
        os.makedirs(self.database_path, exist_ok=True)
        with FileLock(os.path.join(self.database_path, _WRITE_LOCK_FILE)):
            return operation(*args)

    # ------------------------------------------------------------------ writes (threads)

    def _t_full_rows(self, table: str, key: str, values: list[str], schema: pa.Schema) -> list:
        return self._t_rows_where(table, key, values, [field.name for field in schema])

    def _t_merge(self, table: str, key: str, schema: pa.Schema, rows: list[dict], indexes) -> None:
        """Upsert full rows by ``key`` (created_at and provenance already set by the caller)."""
        arrow = pa.Table.from_pylist(rows, schema=schema)
        self._open(table).merge_insert(
            key
        ).when_matched_update_all().when_not_matched_insert_all().execute(arrow)
        self._t_index(table, indexes)

    def _t_merge_nodes(self, rows: list[dict], refs: dict, pipeline_run_id: str | None) -> None:
        existing = {
            row["id"]: row
            for row in self._t_full_rows(NODES_TABLE, "id", [r["id"] for r in rows], NODE_SCHEMA)
        }
        now = datetime.now(timezone.utc)
        for row in rows:
            old = existing.get(row["id"])
            row["created_at"] = old["created_at"] if old else now
            row["updated_at"] = now
            row["properties"] = _with_learned(
                row["properties"],
                old["properties"] if old else None,
                LEARNED_NODE_PROPERTIES,
            )
            row.update(_provenance_after_fold(old, refs.get(row["id"]), pipeline_run_id))
        self._t_merge(NODES_TABLE, "id", NODE_SCHEMA, rows, NODE_INDEXES)

    def _t_merge_edges(self, rows: list[dict], refs: dict, pipeline_run_id: str | None) -> None:
        endpoints = _unique([r["source_id"] for r in rows] + [r["target_id"] for r in rows])
        present = {row["id"] for row in self._t_rows_where(NODES_TABLE, "id", endpoints, ("id",))}
        rows = [r for r in rows if r["source_id"] in present and r["target_id"] in present]
        if not rows:
            return
        existing = {
            row["edge_key"]: row
            for row in self._t_full_rows(
                EDGES_TABLE, "edge_key", [r["edge_key"] for r in rows], EDGE_SCHEMA
            )
        }
        now = datetime.now(timezone.utc)
        for row in rows:
            old = existing.get(row["edge_key"])
            row["created_at"] = old["created_at"] if old else now
            row["updated_at"] = now
            row["properties"] = _with_learned(
                row["properties"],
                old["properties"] if old else None,
                LEARNED_EDGE_PROPERTIES,
            )
            identity = (row["source_id"], row["target_id"], row["relationship_name"])
            row.update(_provenance_after_fold(old, refs.get(identity), pipeline_run_id))
        self._t_merge(EDGES_TABLE, "edge_key", EDGE_SCHEMA, rows, EDGE_INDEXES)

    def _t_apply_changes(
        self,
        table: str,
        key: str,
        schema: pa.Schema,
        indexes,
        rows: list[dict],
        change,
        *,
        touch_updated_at: bool,
    ) -> list[dict]:
        """Apply ``change(row)`` (column -> value, or None to skip) to stored rows and write them.

        Returns the rows that changed. Used by the feedback, tag and property updates, which
        Ladybug runs as SET on the matched rows only.
        """
        now = datetime.now(timezone.utc)
        changed = []
        for row in rows:
            updates = change(row)
            if updates is None:
                continue
            row.update(updates)
            if touch_updated_at:
                row["updated_at"] = now
            changed.append(row)
        if changed:
            self._t_merge(table, key, schema, changed, indexes)
        return changed

    def _t_transform_provenance(self, table, key, schema, indexes, values, transition) -> None:
        """Rewrite the provenance columns of the rows named by ``values``; others are skipped."""
        rows = self._t_full_rows(table, key, values, schema)
        changed = []
        for row in rows:
            row.update(transition(row))
            changed.append(row)
        if changed:
            self._t_merge(table, key, schema, changed, indexes)

    def _t_delete_edges(self, edge_keys: list[str]) -> None:
        for chunk in _chunks(edge_keys):
            self._open(EDGES_TABLE).delete(f"edge_key IN ({_quote(chunk)})")

    def _t_remove_belongs_to_set(self, tag_set: set[str], node_ids: list[str] | None) -> None:
        if node_ids is None:
            rows = self._open(NODES_TABLE).to_table(columns=["id", "properties"]).to_pylist()
        else:
            rows = self._t_rows_where(NODES_TABLE, "id", node_ids, ("id", "properties"))
        targets = []
        for row in rows:
            properties = _props_blob(row["properties"])
            current = properties.get("belongs_to_set")
            if not isinstance(current, list) or not any(tag in tag_set for tag in current):
                continue
            properties["belongs_to_set"] = [tag for tag in current if tag not in tag_set]
            targets.append({"id": row["id"], "properties": json.dumps(properties, cls=JSONEncoder)})
        if not targets:
            return
        rows = self._t_full_rows(NODES_TABLE, "id", [t["id"] for t in targets], NODE_SCHEMA)
        new_properties = {t["id"]: t["properties"] for t in targets}
        self._t_apply_changes(
            NODES_TABLE,
            "id",
            NODE_SCHEMA,
            NODE_INDEXES,
            rows,
            lambda row: {"properties": new_properties[row["id"]]},
            touch_updated_at=False,
        )

    def _t_set_node_feedback(self, weights: dict[str, float]) -> set[str]:
        rows = self._t_full_rows(NODES_TABLE, "id", list(weights), NODE_SCHEMA)

        def change(row: dict) -> dict | None:
            properties = _props_blob(row["properties"])
            properties["feedback_weight"] = float(weights[row["id"]])
            return {"properties": json.dumps(properties, cls=JSONEncoder)}

        changed = self._t_apply_changes(
            NODES_TABLE,
            "id",
            NODE_SCHEMA,
            NODE_INDEXES,
            rows,
            change,
            touch_updated_at=True,
        )
        return {row["id"] for row in changed}

    def _t_set_edge_feedback(self, weights: dict[str, float]) -> set[str]:
        """Set feedback on every edge whose ``edge_object_id`` property is in ``weights``.

        Scans the properties column (Ladybug does a json_extract scan for the same lookup);
        the matched rows are then read whole and rewritten.
        """
        scan = self._open(EDGES_TABLE).to_table(columns=["edge_key", "properties"]).to_pylist()
        matched = []
        for row in scan:
            object_id = _props_blob(row["properties"]).get("edge_object_id")
            if isinstance(object_id, str) and object_id in weights:
                matched.append(row["edge_key"])
        if not matched:
            return set()
        rows = self._t_full_rows(EDGES_TABLE, "edge_key", matched, EDGE_SCHEMA)
        returned: set[str] = set()

        def change(row: dict) -> dict | None:
            properties = _props_blob(row["properties"])
            object_id = properties["edge_object_id"]
            properties["feedback_weight"] = float(weights[object_id])
            returned.add(object_id)
            return {"properties": json.dumps(properties, cls=JSONEncoder)}

        self._t_apply_changes(
            EDGES_TABLE,
            "edge_key",
            EDGE_SCHEMA,
            EDGE_INDEXES,
            rows,
            change,
            touch_updated_at=True,
        )
        return returned

    def _t_set_node_truth(self, states: dict[str, dict[str, Any]]) -> set[str]:
        """Write truth_alignment, and truth_epoch when given, into the stored node blobs."""
        rows = self._t_full_rows(NODES_TABLE, "id", list(states), NODE_SCHEMA)

        def change(row: dict) -> dict | None:
            state = states[row["id"]]
            properties = _props_blob(row["properties"])
            properties["truth_alignment"] = list(state.get("truth_alignment") or [])
            if state.get("truth_epoch") is not None:
                properties["truth_epoch"] = int(state["truth_epoch"])
            return {"properties": json.dumps(properties, cls=JSONEncoder)}

        changed = self._t_apply_changes(
            NODES_TABLE,
            "id",
            NODE_SCHEMA,
            NODE_INDEXES,
            rows,
            change,
            touch_updated_at=True,
        )
        return {row["id"] for row in changed}

    def _t_update_node(self, node_id: str, values: dict[str, Any]) -> bool:
        """Merge ``values`` into one stored node blob; the native columns are not touched."""
        rows = self._t_full_rows(NODES_TABLE, "id", [node_id], NODE_SCHEMA)

        def change(row: dict) -> dict:
            properties = {**_props_blob(row["properties"]), **values}
            return {"properties": json.dumps(properties, cls=JSONEncoder)}

        changed = self._t_apply_changes(
            NODES_TABLE,
            "id",
            NODE_SCHEMA,
            NODE_INDEXES,
            rows,
            change,
            touch_updated_at=False,
        )
        return bool(changed)

    def _t_update_chunk_index(self, chunk_indexes: dict[str, int]) -> None:
        """Set chunk_index in the stored node blobs. Rows whose blob is not a JSON object are skipped."""
        rows = self._t_full_rows(NODES_TABLE, "id", list(chunk_indexes), NODE_SCHEMA)

        def change(row: dict) -> dict | None:
            try:
                properties = json.loads(row["properties"] or "")
            except json.JSONDecodeError:
                return None
            if not isinstance(properties, dict):
                return None
            properties["chunk_index"] = chunk_indexes[row["id"]]
            return {"properties": json.dumps(properties, cls=JSONEncoder)}

        self._t_apply_changes(
            NODES_TABLE,
            "id",
            NODE_SCHEMA,
            NODE_INDEXES,
            rows,
            change,
            touch_updated_at=False,
        )

    def _t_set_graph_metadata(self, metadata: dict[str, str]) -> None:
        rows = [{"key": key, "value": value} for key, value in metadata.items()]
        table = pa.Table.from_pylist(rows, schema=METADATA_SCHEMA)
        self._open(METADATA_TABLE).merge_insert(
            "key"
        ).when_matched_update_all().when_not_matched_insert_all().execute(table)

    def _t_get_graph_metadata(self) -> dict[str, str]:
        rows = self._open(METADATA_TABLE).to_table(columns=["key", "value"]).to_pylist()
        return {row["key"]: row["value"] for row in rows}

    def _t_timestamps_in_range(self, start: int | None, end: int | None) -> list[dict[str, Any]]:
        rows = (
            self._open(NODES_TABLE)
            .to_table(columns=["id", "properties"], filter="type = 'Timestamp'")
            .to_pylist()
        )
        result = []
        for row in rows:
            properties = _props_blob(row["properties"])
            node = {**properties, "id": row["id"]}
            if not timestamp_overlaps(node, start, end):
                continue
            time_at = int(properties["time_at"])
            until = properties.get("time_until")
            result.append(
                {
                    "id": row["id"],
                    "type": "Timestamp",
                    "timestamp_str": str(properties.get("timestamp_str") or ""),
                    "time_at": time_at,
                    "time_until": int(until) if until not in (None, "") else time_at + 1000,
                }
            )
        return sorted(result, key=lambda item: (item["time_at"], item["id"]))

    def _t_overlapping_timestamps(self, node_ids: list[str], start, end) -> set[str]:
        """The ids among ``node_ids`` that are Timestamp rows overlapping ``[start, end)``."""
        rows = self._t_rows_where(NODES_TABLE, "id", node_ids, ("id", "properties"))
        return {
            row["id"]
            for row in rows
            if timestamp_overlaps({**_props_blob(row["properties"]), "id": row["id"]}, start, end)
        }

    def _t_edge_triples(self, node_ids: Iterable[str], edge_types: list[str] | None) -> list:
        """``(source, target, relationship)`` of every edge touching ``node_ids``."""
        table = self._t_edges_touching(node_ids, edge_types)
        return list(
            zip(
                table["source_id"].to_pylist(),
                table["target_id"].to_pylist(),
                table["relationship_name"].to_pylist(),
            )
        )

    def _t_entity_type_names(self, entity_ids: list[str]) -> dict[str, str]:
        members = set(entity_ids)
        is_a = [(s, t) for s, t, _ in self._t_edge_triples(entity_ids, ["is_a"]) if s in members]
        names = {
            row["id"]: row["name"]
            for row in self._t_rows_where(
                NODES_TABLE, "id", _unique(t for _, t in is_a), ("id", "name", "type")
            )
            if row["type"] == "EntityType" and row["name"]
        }
        return {s: names[t] for s, t in is_a if t in names}

    def _t_top_degree_node_ids(self, top_k: int) -> list[str]:
        """Degree ranking over the first ``EDGE_SAMPLE_ROWS`` edges, then nodes to fill up.

        The same bounded sample Ladybug ranks (``cypher_degree_seeds``): ties go to the
        smaller id, and the fill takes any nodes not already ranked.
        """
        sample = self._open(EDGES_TABLE).head(EDGE_SAMPLE_ROWS, columns=["source_id", "target_id"])
        ends = pa.concat_arrays(
            [sample["source_id"].combine_chunks(), sample["target_id"].combine_chunks()]
        )
        counts = pc.value_counts(pc.drop_null(ends))
        ranked = pa.table({"id": counts.field("values"), "degree": counts.field("counts")}).sort_by(
            [("degree", "descending"), ("id", "ascending")]
        )
        seeds = ranked["id"].slice(0, top_k).to_pylist()
        if len(seeds) < top_k:
            # Any top_k nodes hold at most len(seeds) ranked ids, so this leaves enough.
            ids = self._open(NODES_TABLE).to_table(columns=["id"], limit=top_k)["id"].to_pylist()
            ranked_ids = set(seeds)
            fill = [i for i in ids if i not in ranked_ids]
            seeds.extend(fill[: top_k - len(seeds)])
        return seeds

    def _t_temporal_anchors(
        self, chunk_ids: list[str], entity_ids: list[str], start, end
    ) -> dict[str, set[str]]:
        chunks = _unique(chunk_ids)
        candidates = set(_unique([*chunks, *entity_ids]))
        if not candidates:
            return temporal_anchors_from_rows([], [], [])
        chunk_set = set(chunks)
        hops = [(s, t, r) for s, t, r in self._t_edge_triples(candidates, None) if s in candidates]
        contains = {(s, t) for s, t, r in hops if s in chunk_set and r == "contains"}
        reached = _unique(entity for _, entity in contains)
        reached_set = set(reached)
        onward = [(s, t) for s, t, _ in self._t_edge_triples(reached, None) if s in reached_set]
        targets = _unique([t for _, t, _ in hops] + [t for _, t in onward])
        types = {
            row["id"]: row["type"]
            for row in self._t_rows_where(NODES_TABLE, "id", targets, ("id", "type"))
        }
        in_window = self._t_overlapping_timestamps(
            [t for t in targets if types.get(t) == "Timestamp"], start, end
        )
        direct_rows = [(s, t) for s, t, _ in hops if t in in_window]
        stamps_by_entity: dict[str, set[str]] = defaultdict(set)
        for entity, stamp in onward:
            if stamp in in_window:
                stamps_by_entity[entity].add(stamp)
        via_rows = [
            (chunk, entity, stamp)
            for chunk, entity in contains
            if types.get(entity) == "Entity"
            for stamp in stamps_by_entity.get(entity, ())
        ]
        return temporal_anchors_from_rows(direct_rows, via_rows, chunks)

    def _t_triplets_page(self, offset: int, limit: int) -> list[dict[str, Any]]:
        """One page of edges ordered by (source id, target id, relationship), with their nodes.

        Only the three key columns are read for the ordering; the page's rows are then taken
        whole. Ladybug's ORDER BY ranks the same way, so offsets name the same triplets.
        """
        ds = self._open(EDGES_TABLE)
        keys = ds.to_table(columns=["source_id", "target_id", "relationship_name"])
        order = pc.sort_indices(
            keys,
            sort_keys=[
                ("source_id", "ascending"),
                ("target_id", "ascending"),
                ("relationship_name", "ascending"),
            ],
        )
        positions = order[offset : offset + limit].to_pylist()
        if not positions:
            return []
        edges = self._edge_tuples(ds.take(positions, columns=list(_EDGE_READ)))
        node_ids = _unique(part for source, target, _, _ in edges for part in (source, target))
        nodes = {node["id"]: node for node in self._node_dicts(self._t_nodes_by_ids(node_ids))}
        return [
            {
                "start_node": nodes[source],
                "relationship_properties": {**props, "relationship_name": relationship},
                "end_node": nodes[target],
            }
            for source, target, relationship, props in edges
            # An endpoint that is not in the nodes table drops the triplet, as in Ladybug's MATCH.
            if source in nodes and target in nodes
        ]

    def _t_ids_with_member(self, table: str, column: str, value: str, columns) -> list[dict]:
        """Rows whose list column ``column`` contains ``value`` (pushed down to Lance)."""
        return (
            self._open(table)
            .to_table(columns=list(columns), filter=f"array_has({column}, {_quote([value])})")
            .to_pylist()
        )

    def _t_delete_nodes(self, node_ids: list[str]) -> None:
        # Edges first, so no edge is left pointing at a deleted node.
        for chunk in _chunks(node_ids):
            ids = _quote(chunk)
            self._open(EDGES_TABLE).delete(f"source_id IN ({ids}) OR target_id IN ({ids})")
        for chunk in _chunks(node_ids):
            self._open(NODES_TABLE).delete(f"id IN ({_quote(chunk)})")

    def _t_delete_graph(self) -> None:
        # Metadata goes too: Ladybug's delete removes the whole database file, GraphMetadata included.
        self._t_clear(NODES_TABLE, NODE_SCHEMA)
        self._t_clear(EDGES_TABLE, EDGE_SCHEMA)
        self._t_clear(METADATA_TABLE, METADATA_SCHEMA)
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
        # Provenance lives in its own list columns, never in the JSON blob.
        extra = {
            k: v
            for k, v in props.items()
            if k not in _CORE_NODE_KEYS and k not in PROVENANCE_COLUMNS
        }
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
            "edge_key": _edge_key(source_id, target_id, relationship_name),
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
        source_ref_key: str | dict[str, str] | None = None,
        pipeline_run_id: str | None = None,
    ) -> None:
        """Upsert nodes by id, stamping provenance in the same write when ``source_ref_key`` is set.

        A str stamps every node with that ref; a dict maps node id -> ref (one entry per node).
        ``pipeline_run_id`` is recorded with the stamp, so the write is rollbackable by run.
        """
        if not nodes:
            return
        rows = list({r["id"]: r for r in map(self._node_row, nodes)}.values())
        refs = _source_ref_map(source_ref_key, [r["id"] for r in rows])
        await self.initialize()
        await self._write(self._t_merge_nodes, rows, refs, pipeline_run_id)

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
        source_ref_key: str | dict[tuple[str, str, str], str] | None = None,
        pipeline_run_id: str | None = None,
    ) -> None:
        """Upsert edges by (source, relationship, target), stamping provenance as add_nodes does.

        A dict maps (source_id, target_id, relationship_name) -> ref. Edges with a missing
        endpoint are skipped.
        """
        if not edges:
            return
        rows = list({r["edge_key"]: r for r in map(self._edge_row, edges)}.values())
        identities = [(r["source_id"], r["target_id"], r["relationship_name"]) for r in rows]
        refs = _source_ref_map(source_ref_key, identities)
        await self.initialize()
        await self._write(self._t_merge_edges, rows, refs, pipeline_run_id)

    async def delete_graph(self) -> None:
        await self.initialize()
        await self._write(self._t_delete_graph)

    async def close(self) -> None:
        """Nothing to release: every operation opens the tables at their current version."""
        return

    # ------------------------------------------------------------------ public API: provenance

    async def attach_node_source_refs(
        self,
        node_ids: list[str],
        source_ref_keys: list[str],
        pipeline_run_id: str | None = None,
    ) -> None:
        if not source_ref_keys or not node_ids:
            return
        add_keys = list(source_ref_keys)
        await self.initialize()
        await self._write(
            self._t_transform_provenance,
            NODES_TABLE,
            "id",
            NODE_SCHEMA,
            NODE_INDEXES,
            [str(n) for n in node_ids],
            lambda row: provenance_after_attach(
                row["source_ref_keys"] or [],
                row["source_run_refs"] or [],
                add_keys,
                pipeline_run_id,
            )._asdict(),
        )

    async def attach_edge_source_refs(
        self,
        edges: list[EdgeIdentity],
        source_ref_keys: list[str],
        pipeline_run_id: str | None = None,
    ) -> None:
        if not source_ref_keys or not edges:
            return
        add_keys = list(source_ref_keys)
        await self.initialize()
        await self._write(
            self._t_transform_provenance,
            EDGES_TABLE,
            "edge_key",
            EDGE_SCHEMA,
            EDGE_INDEXES,
            [_edge_key(e.source_id, e.target_id, e.relationship_name) for e in edges],
            lambda row: provenance_after_attach(
                row["source_ref_keys"] or [],
                row["source_run_refs"] or [],
                add_keys,
                pipeline_run_id,
            )._asdict(),
        )

    async def remove_node_source_refs(
        self, node_ids: list[str], source_ref_keys: list[str]
    ) -> None:
        if not source_ref_keys or not node_ids:
            return
        remove_keys = list(source_ref_keys)
        await self.initialize()
        await self._write(
            self._t_transform_provenance,
            NODES_TABLE,
            "id",
            NODE_SCHEMA,
            NODE_INDEXES,
            [str(n) for n in node_ids],
            lambda row: provenance_after_remove(
                row["source_ref_keys"] or [], row["source_run_refs"] or [], remove_keys
            )._asdict(),
        )

    async def remove_edge_source_refs(
        self, edges: list[EdgeIdentity], source_ref_keys: list[str]
    ) -> None:
        if not source_ref_keys or not edges:
            return
        remove_keys = list(source_ref_keys)
        await self.initialize()
        await self._write(
            self._t_transform_provenance,
            EDGES_TABLE,
            "edge_key",
            EDGE_SCHEMA,
            EDGE_INDEXES,
            [_edge_key(e.source_id, e.target_id, e.relationship_name) for e in edges],
            lambda row: provenance_after_remove(
                row["source_ref_keys"] or [], row["source_run_refs"] or [], remove_keys
            )._asdict(),
        )

    async def delete_edge_triples(self, edges: list[EdgeIdentity]) -> None:
        """Delete the matching edges only; their endpoint nodes stay."""
        if not edges:
            return
        keys = [
            _edge_key(str(e.source_id), str(e.target_id), str(e.relationship_name)) for e in edges
        ]
        await self.initialize()
        await self._write(self._t_delete_edges, keys)

    async def get_node_delete_data(self, node_ids: list[str]) -> dict[str, NodeDeleteData]:
        if not node_ids:
            return {}
        await self.initialize()

        def read() -> dict[str, NodeDeleteData]:
            rows = self._t_full_rows(NODES_TABLE, "id", [str(n) for n in node_ids], NODE_SCHEMA)
            result: dict[str, NodeDeleteData] = {}
            for row in rows:
                properties = _props_blob(row["properties"])
                # The flat payload get_node returns: core columns merged over the blob.
                properties.update({"id": row["id"], "name": row["name"], "type": row["type"]})
                metadata = properties.get("metadata") or {}
                indexed = (
                    list(metadata.get("index_fields") or []) if isinstance(metadata, dict) else []
                )
                prov = _provenance_columns(row)
                result[row["id"]] = NodeDeleteData(
                    node_id=row["id"],
                    node_type=row["type"] or "",
                    indexed_fields=indexed,
                    node_properties=properties,
                    source_ref_keys=prov["source_ref_keys"],
                    source_dataset_ids=prov["source_dataset_ids"],
                    source_run_ids=prov["source_run_ids"],
                    source_run_refs=prov["source_run_refs"],
                )
            return result

        return await asyncio.to_thread(read)

    async def get_edge_delete_data(
        self, edges: list[EdgeIdentity]
    ) -> dict[EdgeIdentity, EdgeDeleteData]:
        if not edges:
            return {}
        await self.initialize()
        # Lazy import: the modules layer imports this adapter through the graph engine.
        from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text

        def read() -> dict[EdgeIdentity, EdgeDeleteData]:
            keys = [_edge_key(e.source_id, e.target_id, e.relationship_name) for e in edges]
            rows = self._t_full_rows(EDGES_TABLE, "edge_key", keys, EDGE_SCHEMA)
            result: dict[EdgeIdentity, EdgeDeleteData] = {}
            for row in rows:
                edge = EdgeIdentity(
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    relationship_name=row["relationship_name"],
                )
                properties = _props_blob(row["properties"])
                prov = _provenance_columns(row)
                result[edge] = EdgeDeleteData(
                    edge=edge,
                    edge_text=get_edge_retrieval_text(
                        properties.get("edge_text"), edge.relationship_name
                    ),
                    edge_properties=properties,
                    source_ref_keys=prov["source_ref_keys"],
                    source_dataset_ids=prov["source_dataset_ids"],
                    source_run_ids=prov["source_run_ids"],
                    source_run_refs=prov["source_run_refs"],
                )
            return result

        return await asyncio.to_thread(read)

    async def find_nodes_by_source_ref(self, source_ref_key: str) -> list[str]:
        rows = await asyncio.to_thread(
            self._t_ids_with_member, NODES_TABLE, "source_ref_keys", source_ref_key, ("id",)
        )
        return [row["id"] for row in rows]

    async def find_edges_by_source_ref(self, source_ref_key: str) -> list[EdgeIdentity]:
        rows = await asyncio.to_thread(
            self._t_ids_with_member,
            EDGES_TABLE,
            "source_ref_keys",
            source_ref_key,
            ("source_id", "target_id", "relationship_name"),
        )
        return [
            EdgeIdentity(
                source_id=row["source_id"],
                target_id=row["target_id"],
                relationship_name=row["relationship_name"],
            )
            for row in rows
        ]

    async def find_node_source_refs_by_dataset(self, dataset_id: str) -> dict[str, list[str]]:
        return await asyncio.to_thread(
            self._find_refs_by_dataset, NODES_TABLE, str(dataset_id), lambda row: row["id"]
        )

    async def find_edge_source_refs_by_dataset(
        self, dataset_id: str
    ) -> dict[EdgeIdentity, list[str]]:
        return await asyncio.to_thread(
            self._find_refs_by_dataset,
            EDGES_TABLE,
            str(dataset_id),
            lambda row: EdgeIdentity(
                source_id=row["source_id"],
                target_id=row["target_id"],
                relationship_name=row["relationship_name"],
            ),
        )

    def _find_refs_by_dataset(self, table: str, dataset_id: str, owner) -> dict:
        """Owner -> the source refs of ``dataset_id`` it holds (Ladybug's CONTAINS filter)."""
        columns = ("source_ref_keys", *_identity_columns(table))
        rows = self._t_ids_with_member(table, "source_dataset_ids", dataset_id, columns)
        result: dict = {}
        for row in rows:
            owned = [
                key
                for key in (row["source_ref_keys"] or [])
                if str(get_dataset_id_from_source_ref_key(key)) == dataset_id
            ]
            if owned:
                result[owner(row)] = owned
        return result

    async def find_node_source_refs_by_pipeline_run(
        self, pipeline_run_id: str
    ) -> dict[str, list[str]]:
        return await asyncio.to_thread(
            self._find_refs_by_pipeline_run,
            NODES_TABLE,
            str(pipeline_run_id),
            lambda row: row["id"],
        )

    async def find_edge_source_refs_by_pipeline_run(
        self, pipeline_run_id: str
    ) -> dict[EdgeIdentity, list[str]]:
        return await asyncio.to_thread(
            self._find_refs_by_pipeline_run,
            EDGES_TABLE,
            str(pipeline_run_id),
            lambda row: EdgeIdentity(
                source_id=row["source_id"],
                target_id=row["target_id"],
                relationship_name=row["relationship_name"],
            ),
        )

    def _find_refs_by_pipeline_run(self, table: str, pipeline_run_id: str, owner) -> dict:
        """Owner -> the source refs that run ``pipeline_run_id`` attached to it."""
        columns = ("source_run_refs", *_identity_columns(table))
        rows = self._t_ids_with_member(table, "source_run_ids", pipeline_run_id, columns)
        result: dict = {}
        for row in rows:
            contributed = [
                get_source_ref_key_from_source_run_ref(ref)
                for ref in (row["source_run_refs"] or [])
                if str(get_pipeline_run_id_from_source_run_ref(ref)) == pipeline_run_id
            ]
            if contributed:
                result[owner(row)] = contributed
        return result

    async def set_graph_metadata(self, metadata: dict[str, str]) -> None:
        if not metadata:
            return
        normalized = {str(key): str(value) for key, value in metadata.items()}
        await self.initialize()
        await self._write(self._t_set_graph_metadata, normalized)

    async def get_graph_metadata(self) -> dict[str, str]:
        await self.initialize()
        return await asyncio.to_thread(self._t_get_graph_metadata)

    # ------------------------------------------------------------------ public API: tags & learned values

    async def remove_belongs_to_set_tags(
        self,
        tags: list[str],
        node_ids: list[str] | None = None,
    ) -> None:
        """Drop ``tags`` from ``belongs_to_set`` in the node properties (Ladybug's semantics)."""
        if not tags:
            return
        if node_ids is not None and not node_ids:
            return
        ids = None if node_ids is None else [str(n) for n in node_ids]
        await self.initialize()
        await self._write(self._t_remove_belongs_to_set, set(tags), ids)

    async def get_node_feedback_weights(self, node_ids: list[str]) -> dict[str, float]:
        valid = [n for n in node_ids if isinstance(n, str) and n]
        if not valid:
            return {}
        result: dict[str, float] = {}
        for node in await self.get_nodes(valid):
            try:
                result[node["id"]] = float(node.get("feedback_weight", _DEFAULT_FEEDBACK_WEIGHT))
            except (TypeError, ValueError):
                result[node["id"]] = _DEFAULT_FEEDBACK_WEIGHT
        return result

    async def set_node_feedback_weights(
        self, node_feedback_weights: dict[str, float]
    ) -> dict[str, bool]:
        if not node_feedback_weights:
            return {}
        valid = {n: w for n, w in node_feedback_weights.items() if isinstance(n, str) and n}
        updated: set[str] = set()
        if valid:
            await self.initialize()
            updated = await self._write(self._t_set_node_feedback, valid)
        return {nid: (nid in updated) for nid in node_feedback_weights}

    async def get_edge_feedback_weights(self, edge_object_ids: list[str]) -> dict[str, float]:
        wanted = {e for e in edge_object_ids if isinstance(e, str) and e}
        if not wanted:
            return {}
        await self.initialize()

        def read() -> dict[str, float]:
            result: dict[str, float] = {}
            for row in self._open(EDGES_TABLE).to_table(columns=["properties"]).to_pylist():
                properties = _props_blob(row["properties"])
                object_id = properties.get("edge_object_id")
                if not isinstance(object_id, str) or object_id not in wanted:
                    continue
                try:
                    result[object_id] = float(
                        properties.get("feedback_weight", _DEFAULT_FEEDBACK_WEIGHT)
                    )
                except (TypeError, ValueError):
                    result[object_id] = _DEFAULT_FEEDBACK_WEIGHT
            return result

        return await asyncio.to_thread(read)

    async def set_edge_feedback_weights(
        self, edge_feedback_weights: dict[str, float]
    ) -> dict[str, bool]:
        if not edge_feedback_weights:
            return {}
        valid = {e: w for e, w in edge_feedback_weights.items() if isinstance(e, str) and e}
        updated: set[str] = set()
        if valid:
            await self.initialize()
            updated = await self._write(self._t_set_edge_feedback, valid)
        return {eid: (eid in updated) for eid in edge_feedback_weights}

    async def get_node_truth_state(self, node_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Truth state of the found nodes; ``truth_epoch`` is None when unset or not an int."""
        valid = [n for n in node_ids if isinstance(n, str) and n]
        if not valid:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for node in await self.get_nodes(valid):
            alignment = node.get("truth_alignment", [])
            try:
                epoch = int(node["truth_epoch"]) if node.get("truth_epoch") is not None else None
            except (TypeError, ValueError):
                epoch = None
            result[node["id"]] = {
                "truth_alignment": list(alignment) if isinstance(alignment, (list, tuple)) else [],
                "truth_epoch": epoch,
            }
        return result

    async def set_node_truth_state(
        self, node_truth_state: dict[str, dict[str, Any]]
    ) -> dict[str, bool]:
        if not node_truth_state:
            return {}
        valid = {n: s for n, s in node_truth_state.items() if isinstance(n, str) and n}
        updated: set[str] = set()
        if valid:
            await self.initialize()
            updated = await self._write(self._t_set_node_truth, valid)
        return {nid: (nid in updated) for nid in node_truth_state}

    async def update_node(self, node_id: str, values: dict[str, Any]) -> bool:
        """Merge ``values`` into the node's properties. False for a missing node or no values."""
        if not isinstance(node_id, str) or not node_id or not values:
            return False
        await self.initialize()
        return await self._write(self._t_update_node, node_id, values)

    async def update_chunk_index(self, chunk_indexes: dict[str, int]) -> None:
        """Set ``chunk_index`` on the given nodes; every other property is carried over."""
        if not chunk_indexes:
            return
        await self.initialize()
        await self._write(self._t_update_chunk_index, chunk_indexes)

    async def get_timestamps_in_range(self, start: int | None, end: int | None) -> list[dict]:
        """Timestamp nodes whose ``[time_at, time_until)`` overlaps ``[start, end)``.

        Filters on the ``type`` column (indexed); the time fields live in the properties
        blob, so they are read and compared per Timestamp row.
        """
        await self.initialize()
        return await asyncio.to_thread(self._t_timestamps_in_range, start, end)

    # ------------------------------------------------------------------ public API: reads

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        nodes = await self.get_nodes([node_id])
        return nodes[0] if nodes else None

    async def has_node(self, node_id: str) -> bool:
        return bool(await self.get_nodes([str(node_id)]))

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

    async def get_top_degree_node_ids(self, top_k: int) -> list[str]:
        """Ids of up to ``top_k`` high-degree nodes, ranked over a bounded edge sample."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        await self.initialize()
        return await asyncio.to_thread(self._t_top_degree_node_ids, top_k)

    async def get_entity_type_names(self, entity_ids: list[str]) -> dict[str, str]:
        """The EntityType name each entity points to through one ``is_a`` edge."""
        members = _unique(entity_ids)
        if not members:
            return {}
        await self.initialize()
        return await asyncio.to_thread(self._t_entity_type_names, members)

    async def get_temporal_anchors(
        self,
        chunk_ids,
        entity_ids,
        start: int | None,
        end: int | None,
    ) -> dict[str, set[str]]:
        """Candidates attached to a Timestamp overlapping ``[start, end)``: directly,
        or (for chunks) through an entity they ``contains``. See the interface."""
        chunks = [str(node_id) for node_id in chunk_ids]
        entities = [str(node_id) for node_id in entity_ids]
        await self.initialize()
        return await asyncio.to_thread(self._t_temporal_anchors, chunks, entities, start, end)

    async def get_triplets_batch(self, offset: int, limit: int) -> list[dict[str, Any]]:
        """Triplets ``{start_node, relationship_properties, end_node}`` in Ladybug's order."""
        if offset < 0:
            raise ValueError(f"Offset must be non-negative, got {offset}")
        if limit < 0:
            raise ValueError(f"Limit must be non-negative, got {limit}")
        await self.initialize()
        return await asyncio.to_thread(self._t_triplets_page, offset, limit)
