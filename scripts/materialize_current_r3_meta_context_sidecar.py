#!/usr/bin/env python3
"""Materialise causal meta/context fields for the current long-only R3 surface.

The R3 base surface intentionally contains only the frozen base contract.  This
sidecar joins configured meta-family fields from the decision-time feature
store by ``symbol`` and exact ``decision_ts``.  It never reads target/path
columns from the feature source and keeps the row identity/order auditable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
META_FAMILIES = (
    "MODEL_REGIME_CONTEXT_META_FEATURE_KEYS",
    "RESIDUAL_META_FEATURE_KEYS",
    "CAUSAL_CONTINUOUS_REGIME_META_FEATURE_KEYS",
)


def _configured_meta_pool() -> list[str]:
    from extreme_price_movements.config import CFG

    result: list[str] = []
    seen: set[str] = set()
    for family in META_FAMILIES:
        for value in CFG.get(family, []):
            name = str(value)
            if name not in seen:
                result.append(name)
                seen.add(name)
    return result


def _source_files(root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in sorted(root.glob("symbol=*.parquet")):
        symbol = path.name[len("symbol=") : -len(".parquet")]
        output[symbol] = path
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coverage-gate", type=float, default=0.90)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not 0.0 < args.coverage_gate <= 1.0:
        raise ValueError("coverage gate must be in (0, 1]")

    surface = pd.read_parquet(
        args.surface, columns=["candidate_id", "decision_ts", "side_name"]
    )
    surface["decision_ts"] = pd.to_datetime(surface["decision_ts"], utc=True, errors="raise")
    surface = surface.loc[
        surface["side_name"].astype(str).str.lower().eq("long")
    ].copy()
    if surface.empty:
        raise ValueError("long-only surface is empty")
    if surface.candidate_id.duplicated().any():
        raise ValueError("surface candidate IDs must be unique")
    surface["symbol"] = surface.candidate_id.astype(str).str.split("|", n=1).str[0]

    source_files = _source_files(args.feature_store)
    if not source_files:
        raise FileNotFoundError(f"no symbol parquet files under {args.feature_store}")
    configured = _configured_meta_pool()
    if not configured:
        raise ValueError("configured meta feature pool is empty")

    args.out.mkdir(parents=True)
    writer: pq.ParquetWriter | None = None
    output_rows = 0
    chunks: list[pd.DataFrame] = []
    field_coverage: dict[str, int] = {field: 0 for field in configured}
    field_source_presence: dict[str, int] = {field: 0 for field in configured}
    missing_symbols: list[str] = []

    for symbol, group in surface.groupby("symbol", sort=True, observed=True):
        source_path = source_files.get(str(symbol))
        if source_path is None:
            missing_symbols.append(str(symbol))
            joined = group[["candidate_id", "decision_ts", "side_name"]].copy()
            for field in configured:
                joined[field] = np.nan
        else:
            source_schema = set(pq.read_schema(source_path).names)
            source_fields = [field for field in configured if field in source_schema]
            read_columns = [field for field in source_fields if field != "ts"]
            source = pd.read_parquet(source_path, columns=read_columns).reset_index()
            if "ts" not in source.columns:
                raise ValueError(f"feature source lacks ts index: {source_path}")
            source["ts"] = pd.to_datetime(source["ts"], utc=True, errors="raise")
            source = source.drop_duplicates("ts", keep="last").rename(columns={"ts": "decision_ts"})
            joined = group[["candidate_id", "decision_ts", "side_name"]].merge(
                source, on="decision_ts", how="left", sort=False, validate="one_to_one"
            )
            for field in configured:
                if field not in joined.columns:
                    joined[field] = np.nan
                else:
                    field_source_presence[field] += int(joined[field].notna().sum())
        for field in configured:
            joined[field] = pd.to_numeric(joined[field], errors="coerce").astype("float32")
            field_coverage[field] += int(joined[field].notna().sum())
        joined = joined[["candidate_id", "decision_ts", "side_name", *configured]]
        output_rows += len(joined)
        chunks.append(joined)
        if len(chunks) >= 8:
            block = pd.concat(chunks, ignore_index=True)
            table = pa.Table.from_pandas(block, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(args.out / "meta_context_sidecar.parquet", table.schema, compression="zstd")
            writer.write_table(table)
            chunks.clear()
    if chunks:
        block = pd.concat(chunks, ignore_index=True)
        table = pa.Table.from_pandas(block, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(args.out / "meta_context_sidecar.parquet", table.schema, compression="zstd")
        writer.write_table(table)
    if writer is not None:
        writer.close()

    coverage = {field: field_coverage[field] / output_rows for field in configured}
    admitted = [field for field in configured if coverage[field] >= args.coverage_gate]
    excluded = [field for field in configured if field not in admitted]
    if len(admitted) < 20:
        raise ValueError(f"fewer than 20 configured meta fields meet coverage gate: {len(admitted)}")
    manifest: dict[str, Any] = {
        "schema": "current_r3_long_meta_context_sidecar_v1",
        "status": "complete",
        "surface": str(args.surface),
        "feature_store": str(args.feature_store),
        "side_scope": ["long"],
        "rows": output_rows,
        "configured_meta_fields": configured,
        "coverage_gate": args.coverage_gate,
        "admitted_meta_fields": admitted,
        "excluded_meta_fields": excluded,
        "coverage": coverage,
        "source_presence_rows": field_source_presence,
        "missing_symbols": missing_symbols,
        "causal_join": "feature-store ts == decision_ts; exact timestamp; no outcome/path columns",
        "feature_families": list(META_FAMILIES),
        "outputs": ["meta_context_sidecar.parquet", "meta_context_coverage.parquet", "manifest.json"],
    }
    coverage_frame = pd.DataFrame(
        {
            "feature": configured,
            "coverage": [coverage[field] for field in configured],
            "admitted": [field in admitted for field in configured],
            "non_null_rows": [field_coverage[field] for field in configured],
            "source_presence_rows": [field_source_presence[field] for field in configured],
        }
    )
    coverage_frame.to_parquet(args.out / "meta_context_coverage.parquet", index=False, compression="zstd")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
