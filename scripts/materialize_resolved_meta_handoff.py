#!/usr/bin/env python3
"""Add a causal forward-label resolution contract to an existing meta handoff."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import duckdb


def _quoted(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _path_contract(label_context_dir: Path) -> dict[str, Any]:
    summary = label_context_dir / "side_archetype_trailing_materialization_summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    contracts = {
        (
            int(row["path_fetch"]["path_len"]),
            str(row["path_fetch"]["path_timeframe"]),
        )
        for row in payload.get("datasets", [])
        if row.get("path_fetch", {}).get("path_len") is not None
        and row.get("path_fetch", {}).get("path_timeframe")
    }
    if len(contracts) != 1:
        raise ValueError(f"Expected one label path contract, found {sorted(contracts)}")
    path_len, timeframe = next(iter(contracts))
    seconds = float(duckdb.sql(f"SELECT epoch(INTERVAL {_quoted(timeframe)})").fetchone()[0])
    return {
        "schema": "forward_label_resolution_v1",
        "source": str(summary),
        "path_len": path_len,
        "path_timeframe": timeframe,
        "path_bar_seconds": seconds,
        "label_horizon_seconds": seconds * path_len,
        "resolution_column": "__label_path_end_ts__",
        "resolution_rule": "__first_path_ts__ + path_len * path_timeframe",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-context-dir", type=Path, required=True)
    args = parser.parse_args()

    source_handoff = args.source_dir / "train_meta_regime_handoff.parquet"
    source_ledger = args.source_dir / "s52_trailing_regime_scored_ledger.parquet"
    source_contract = args.source_dir / "train_meta_regime_handoff_contract.json"
    source_manifest = args.source_dir / "manifest.json"
    for path in (source_handoff, source_ledger, source_contract, source_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    contract = _path_contract(args.label_context_dir)
    horizon_seconds = int(contract["label_horizon_seconds"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_handoff = args.output_dir / source_handoff.name
    temporary_handoff = output_handoff.with_suffix(".parquet.tmp")
    if temporary_handoff.exists():
        temporary_handoff.unlink()

    h = _quoted(str(source_handoff.resolve()))
    ledger = _quoted(str(source_ledger.resolve()))
    output = _quoted(str(temporary_handoff.resolve()))
    query = f"""
        COPY (
            SELECT
                h.*,
                CAST(l.__first_path_ts__ AS TIMESTAMPTZ)
                    + INTERVAL {horizon_seconds} SECOND AS __label_path_end_ts__
            FROM read_parquet({h}) h
            INNER JOIN read_parquet({ledger}) l
              ON h.__ts__ = l.__ts__
             AND h.__symbol__ = l.__symbol__
             AND lower(h.side_name) = lower(l.side_name)
            WHERE l.__first_path_ts__ IS NOT NULL
        ) TO {output} (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    duckdb.sql(query)

    source_rows = int(duckdb.sql(f"SELECT count(*) FROM read_parquet({h})").fetchone()[0])
    output_ref = _quoted(str(temporary_handoff.resolve()))
    output_rows, missing_resolution = duckdb.sql(
        f"SELECT count(*), count(*) FILTER (WHERE __label_path_end_ts__ IS NULL) "
        f"FROM read_parquet({output_ref})"
    ).fetchone()
    if int(output_rows) != source_rows or int(missing_resolution) != 0:
        raise ValueError(
            f"Resolution join changed row coverage: source={source_rows}, "
            f"output={output_rows}, missing={missing_resolution}"
        )
    temporary_handoff.replace(output_handoff)

    output_ledger = args.output_dir / source_ledger.name
    if output_ledger.exists():
        output_ledger.unlink()
    try:
        os.link(source_ledger, output_ledger)
    except OSError:
        shutil.copy2(source_ledger, output_ledger)
    contract_payload = json.loads(source_contract.read_text(encoding="utf-8"))
    contract["rows"] = source_rows
    contract["missing_resolution_rows"] = 0
    contract_payload["label_resolution_contract"] = contract
    (args.output_dir / source_contract.name).write_text(
        json.dumps(contract_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest_payload["label_resolution_contract"] = contract
    manifest_payload["resolved_handoff_source"] = str(args.source_dir)
    (args.output_dir / source_manifest.name).write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": source_rows, "label_resolution_contract": contract}))


if __name__ == "__main__":
    main()
