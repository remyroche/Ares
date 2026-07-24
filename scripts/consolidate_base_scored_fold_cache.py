#!/usr/bin/env python3
"""Atomically consolidate base OOS fold ledgers with bounded memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _schema_hash(schema: pa.Schema) -> str:
    return hashlib.sha256(str(schema).encode("utf-8")).hexdigest()


def _field_types_by_name(schema: pa.Schema) -> dict[str, str]:
    return {str(field.name): str(field.type) for field in schema}


def consolidate_fold_cache(
    cache_dir: Path,
    output_path: Path,
    *,
    expected_parts: int = 0,
    batch_rows: int = 25_000,
) -> dict[str, Any]:
    parts = sorted(Path(cache_dir).glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No scored fold parquet files found under {cache_dir}")
    if int(expected_parts) > 0 and len(parts) != int(expected_parts):
        raise RuntimeError(
            f"Expected {expected_parts} scored folds, found {len(parts)} under {cache_dir}"
        )

    sources: list[dict[str, Any]] = []
    reference_schema: pa.Schema | None = None
    expected_rows = 0
    for path in parts:
        parquet_file = pq.ParquetFile(path)
        schema = parquet_file.schema_arrow
        if reference_schema is None:
            reference_schema = schema
        elif _field_types_by_name(schema) != _field_types_by_name(reference_schema):
            raise RuntimeError(f"Scored fold schema mismatch: {path}")
        rows = int(parquet_file.metadata.num_rows)
        if rows <= 0:
            raise RuntimeError(f"Empty scored fold cache: {path}")
        expected_rows += rows
        sources.append(
            {
                "path": str(path),
                "rows": rows,
                "bytes": int(path.stat().st_size),
            }
        )

    assert reference_schema is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    tmp_path.unlink(missing_ok=True)
    writer = pq.ParquetWriter(
        tmp_path,
        reference_schema,
        compression="zstd",
        compression_level=5,
    )
    try:
        for source in sources:
            parquet_file = pq.ParquetFile(source["path"])
            for batch in parquet_file.iter_batches(
                batch_size=max(1, int(batch_rows)),
                columns=list(reference_schema.names),
            ):
                writer.write_batch(batch)
    finally:
        writer.close()

    consolidated = pq.ParquetFile(tmp_path)
    actual_rows = int(consolidated.metadata.num_rows)
    if actual_rows != expected_rows:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Consolidated row mismatch: expected {expected_rows}, got {actual_rows}"
        )
    if _field_types_by_name(consolidated.schema_arrow) != _field_types_by_name(
        reference_schema
    ):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Consolidated schema differs from scored fold schema")
    os.replace(tmp_path, output_path)
    return {
        "schema": "base_oos_scored_fold_consolidation_v1",
        "cache_dir": str(cache_dir),
        "output_path": str(output_path),
        "part_count": int(len(sources)),
        "row_count": actual_rows,
        "column_count": int(len(reference_schema)),
        "schema_hash": _schema_hash(reference_schema),
        "batch_rows": int(batch_rows),
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--expected-parts", type=int, default=0)
    parser.add_argument("--batch-rows", type=int, default=25_000)
    args = parser.parse_args()
    manifest = consolidate_fold_cache(
        args.cache_dir,
        args.output_path,
        expected_parts=args.expected_parts,
        batch_rows=args.batch_rows,
    )
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "sources"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
