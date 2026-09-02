#!/usr/bin/env python3
"""Append a current stateful strict-R3 input slice without loading its prefix.

The live stateful materializer emits only the new cross-section.  This helper
restores the immutable candidate/feature prefix for the next cycle by copying
Parquet record batches verbatim from the predecessor and then appending the
new slice.  It is deliberately a separate process so post-feature lineage
work cannot compete with the feature graph for memory.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROLES = {
    "candidate_population": "candidate_grid/target_free_candidate_population.parquet",
    "eligible_candidates": "candidate_grid/eligible_candidates.parquet",
    "candidate_rejections": "candidate_grid/candidate_rejection_audit.parquet",
    "features": "features/canonical120_features.parquet",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids(path: Path) -> set[str]:
    frame = pd.read_parquet(path, columns=["candidate_id"])
    raw = frame["candidate_id"]
    if raw.isna().any():
        raise ValueError(f"candidate identities include nulls: {path}")
    values = raw.astype(str)
    if values.duplicated().any():
        raise ValueError(f"candidate identities are invalid: {path}")
    return set(values)


def _tables(path: Path, *, schema: pa.Schema):
    """Yield one stable-schema batch at a time with bounded memory."""
    source = pq.ParquetFile(path)
    names = schema.names
    for batch in source.iter_batches(batch_size=256):
        yield pa.Table.from_batches([batch]).select(names).cast(schema, safe=True)


def _assert_exact_append_values(
    *, old: Path, current: Path, output: Path, schema: pa.Schema,
) -> int:
    """Prove every output field equals old batches followed by current batches.

    This is deliberately a value-level check, not a row-count or identity
    assertion.  It streams 256 rows at a time so it retains the prior
    append-only guarantee without re-materialising the historical 120-field
    panel in the hourly producer process.
    """
    expected = iter(itertools.chain(
        _tables(old, schema=schema), _tables(current, schema=schema),
    ))
    actual = iter(_tables(output, schema=schema))
    checked = 0
    expected_table = next(expected, None)
    actual_table = next(actual, None)
    comparison = 0
    while expected_table is not None or actual_table is not None:
        if expected_table is None or actual_table is None:
            raise AssertionError("appended output row count differs from its inputs")
        count = min(expected_table.num_rows, actual_table.num_rows)
        left = expected_table.slice(0, count)
        right = actual_table.slice(0, count)
        comparison += 1
        if not left.equals(right, check_metadata=False):
            raise AssertionError(f"appended output differs at comparison {comparison}")
        checked += int(count)
        expected_table = (
            expected_table.slice(count)
            if count < expected_table.num_rows else next(expected, None)
        )
        actual_table = (
            actual_table.slice(count)
            if count < actual_table.num_rows else next(actual, None)
        )
    return checked


def append_batches(*, old: Path, current: Path, destination: Path) -> dict[str, object]:
    old_file = pq.ParquetFile(old)
    current_file = pq.ParquetFile(current)
    old_names = old_file.schema_arrow.names
    if set(old_names) != set(current_file.schema_arrow.names):
        raise ValueError(f"field set differs for {destination}")
    old_ids = ids(old)
    current_ids = ids(current)
    overlap = old_ids & current_ids
    if overlap:
        raise ValueError(f"identity overlap for {destination}: {len(overlap)}")
    # Capture both source hashes before replacing ``destination``.  The
    # current slice and destination have the same path, so collecting this
    # evidence after the atomic replacement would only record the output.
    old_sha = sha(old)
    current_sha = sha(current)
    old_rows = old_file.metadata.num_rows
    current_rows = current_file.metadata.num_rows
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with pq.ParquetWriter(temporary, old_file.schema_arrow, compression="zstd") as writer:
            for source in (old_file, current_file):
                for batch in source.iter_batches(batch_size=4096):
                    table = pa.Table.from_batches([batch]).select(old_names)
                    # Candidate-source representations can differ only by
                    # lossless float32/float64 widening between historical
                    # and current appenders.  Cast to the predecessor schema
                    # so all historical columns retain one stable contract.
                    table = table.cast(old_file.schema_arrow, safe=True)
                    writer.write_table(table)
        exact_value_rows = _assert_exact_append_values(
            old=old,
            current=current,
            output=temporary,
            schema=old_file.schema_arrow,
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    output_file = pq.ParquetFile(destination)
    output_ids = ids(destination)
    expected_ids = old_ids | current_ids
    if output_file.metadata.num_rows != old_rows + current_rows:
        raise AssertionError(f"row count changed while appending {destination}")
    if exact_value_rows != old_rows + current_rows:
        raise AssertionError(f"value audit did not cover all rows in {destination}")
    if output_ids != expected_ids:
        raise AssertionError(f"identity set changed while appending {destination}")
    return {
        "previous_rows": len(old_ids),
        "current_rows": len(current_ids),
        "output_rows": len(old_ids) + len(current_ids),
        "identity_overlap": 0,
        "previous_sha256": old_sha,
        "current_sha256": current_sha,
        "output_sha256": sha(destination),
        "copy_mode": "arrow_record_batch_append",
        "output_row_count_verified": True,
        "output_identity_set_verified": True,
        "exact_value_comparison": True,
        "exact_value_rows_verified": int(exact_value_rows),
        # Every predecessor record batch is copied through Arrow with the
        # predecessor's stable schema; no prior value is recomputed or
        # transformed. These are the same invariants consumed by the
        # independent hourly chain audit.
        "changed_fields": [],
        "max_numeric_delta": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--current-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit: dict[str, object] = {"schema": "strict_r3_stateful_prefix_assembly_v1"}
    for role, relative in ROLES.items():
        old = args.previous_run / relative
        current = args.current_run / relative
        if not old.is_file() or not current.is_file():
            raise FileNotFoundError(f"missing {role} source")
        audit[role] = append_batches(old=old, current=current, destination=current)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"event": "complete", **audit}))


if __name__ == "__main__":
    main()
