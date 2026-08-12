#!/usr/bin/env python3
"""Partition a frozen strict-R3 target-free panel into immutable UTC months.

This is a storage-layout utility for causal replay.  It copies only immutable
candidate identity, decision time, side and the frozen long base contract; it
does not read outcome columns or create labels.  The resulting monthly store
lets a fitting/scoring process materialise only the selected calendar months
instead of decompressing an entire multi-month parquet row group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fields(contract: Path) -> list[str]:
    payload = json.loads(contract.read_text())
    fields = [str(value) for value in payload["base_fields_by_side"]["long"]]
    if len(fields) != 120 or len(set(fields)) != 120:
        raise ValueError("strict source store requires the frozen 120-field long contract")
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable monthly source store already exists: {args.out_dir}")
    fields = _fields(args.feature_contract)
    columns = ["candidate_id", "__decision_ts__", "__symbol__", "side_name", *fields]
    reader = pq.ParquetFile(args.source_panel)
    temporary = args.out_dir.with_name(f"{args.out_dir.name}.building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    counts: dict[str, int] = {}
    try:
        for row_group in range(reader.num_row_groups):
            table = reader.read_row_group(row_group, columns=columns)
            frame = table.to_pandas()
            frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
            frame = frame.loc[frame["side_name"].astype(str).str.lower().eq("long")].copy()
            month = frame["__decision_ts__"].dt.strftime("%Y-%m")
            for value, part in frame.groupby(month, sort=False):
                directory = temporary / f"month={value}"
                directory.mkdir(exist_ok=True)
                path = directory / f"part-{row_group:03d}.parquet"
                part.to_parquet(path, index=False, compression="zstd")
                counts[str(value)] = counts.get(str(value), 0) + int(len(part))
            del table, frame
            print(json.dumps({"event": "source_monthstore_row_group_complete", "row_group": row_group}), flush=True)
        manifest = {
            "schema": "strict_r3_targetfree_month_store_v1",
            "source_panel": str(args.source_panel),
            "source_panel_sha256": _sha(args.source_panel),
            "feature_contract": str(args.feature_contract),
            "feature_contract_sha256": _sha(args.feature_contract),
            "side": "long",
            "fields": fields,
            "months": dict(sorted(counts.items())),
            "rows": int(sum(counts.values())),
            "outcome_columns": [],
        }
        (temporary / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.rename(args.out_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
