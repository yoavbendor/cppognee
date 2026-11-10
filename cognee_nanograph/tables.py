"""Arrow schemas and index plan for the nanolance graph tables.

Columns match cognee's Ladybug/Turso graph columns. ``edges`` adds ``edge_key``
(``source|relationship|target``) so an edge is upserted and looked up by one column.
"""

import pyarrow as pa

NODES_TABLE = "nodes"
EDGES_TABLE = "edges"

PROVENANCE_COLUMNS = ("source_ref_keys", "source_dataset_ids", "source_run_ids", "source_run_refs")

# Properties are large JSON strings. zstd per value page lets a single row be read
# without decoding a big block (same setting the benchmark prototype measured).
_ZSTD = {"lance-encoding:compression": "zstd"}
_TS = pa.timestamp("us", tz="UTC")
_STR_LIST = pa.list_(pa.string())

NODE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("type", pa.string()),
        pa.field("created_at", _TS),
        pa.field("updated_at", _TS),
        pa.field("properties", pa.string(), metadata=_ZSTD),
        *(pa.field(column, _STR_LIST) for column in PROVENANCE_COLUMNS),
    ]
)

EDGE_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string()),
        pa.field("target_id", pa.string()),
        pa.field("relationship_name", pa.string()),
        pa.field("created_at", _TS),
        pa.field("updated_at", _TS),
        pa.field("properties", pa.string(), metadata=_ZSTD),
        *(pa.field(column, _STR_LIST) for column in PROVENANCE_COLUMNS),
        pa.field("edge_key", pa.string()),
    ]
)

# (column, index type) per table. BTREE for point and IN lookups, BITMAP for the
# low-cardinality type / relationship columns.
NODE_INDEXES = (("id", "BTREE"), ("type", "BITMAP"))
EDGE_INDEXES = (
    ("source_id", "BTREE"),
    ("target_id", "BTREE"),
    ("edge_key", "BTREE"),
    ("relationship_name", "BITMAP"),
)
