#!/usr/bin/env python3
"""Materialize a side-only copy of a meta handoff and outcome ledger.

The output preserves the source contracts and row schema.  It exists to run a
true side-only model experiment without training or scoring the other side.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _filter(input_path: Path, output_path: Path, side: str) -> dict[str, int]:
    parquet = pq.ParquetFile(input_path)
    side_column = "side_name"
    if side_column not in parquet.schema.names:
        raise ValueError(f"{input_path} lacks required {side_column!r} column")
    writer: pq.ParquetWriter | None = None
    input_rows = 0
    output_rows = 0
    try:
        for batch in parquet.iter_batches(batch_size=100_000):
            input_rows += batch.num_rows
            mask = pc.equal(pc.utf8_lower(batch.column(batch.schema.get_field_index(side_column))), side)
            filtered = batch.filter(mask)
            if not filtered.num_rows:
                continue
            output_rows += filtered.num_rows
            table = pa.Table.from_batches([filtered])
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if output_rows == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"No {side!r} rows found in {input_path}")
    return {"input_rows": input_rows, "output_rows": output_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("long", "short"), required=True)
    args = parser.parse_args()

    source = args.source_dir
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    handoff = source / "train_meta_regime_handoff.parquet"
    ledger = source / "s52_trailing_regime_scored_ledger.parquet"
    if not handoff.is_file() or not ledger.is_file():
        raise FileNotFoundError("Source must contain the meta handoff and scored ledger parquets")
    result = {
        "side": args.side,
        "source_dir": str(source.resolve()),
        "handoff": _filter(handoff, output / handoff.name, args.side),
        "ledger": _filter(ledger, output / ledger.name, args.side),
    }
    contract = source / "train_meta_regime_handoff_contract.json"
    if contract.is_file():
        shutil.copy2(contract, output / contract.name)
        result["contract"] = str((output / contract.name).resolve())
    (output / "side_only_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
