#!/usr/bin/env python3
"""Freeze the multi-era Stage-I reference population and causal store contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG
from extreme_price_movements.packb_static_point_feature_loader import (
    build_fresh_causal_feature_contract,
)
from extreme_price_movements.stage_i_feature_selection import (
    STAGE_I_ACTIVE_CONTRACTS,
    resolve_stage_i_feature_universe,
)
from extreme_price_movements.stage_i_production_data_adapter import (
    MonthlyReferencePartition,
    load_reference_ledgers,
)


def reference_partitions() -> list[MonthlyReferencePartition]:
    artifacts = ROOT / "data_perp/artifacts"
    parts: list[MonthlyReferencePartition] = []

    historical = artifacts / "stage_i_historical_tp6_sl4_h12_r3_20260803_v1/parts"
    for path in sorted(historical.glob("month=*/side=*.parquet")):
        parts.append(MonthlyReferencePartition(path, path.parent.name[6:], "historical_2022_2023"))

    surface = artifacts / "stage_i_surface_2024_2026_20260803_v1/source=full_universe_2024"
    for path in sorted(surface.glob("month=*/identity_labels")):
        parts.append(MonthlyReferencePartition(path, path.parent.name[6:], "surface_2024"))

    packb = artifacts / "stage_i_packb_tp6_sl4_h12_r3_20260803_v1/parts"
    for path in sorted(packb.glob("month=2025-*/side=*.parquet")):
        parts.append(MonthlyReferencePartition(path, path.parent.name[6:], "common30_2025_2026"))

    repaired = artifacts / "stage_i_common30_tp6_sl4_h12_r3_20260803_v1/parts"
    for path in sorted(repaired.glob("month=2026-0[1-4]/side=*.parquet")):
        parts.append(MonthlyReferencePartition(path, path.parent.name[6:], "common30_2025_2026"))
    for path in sorted(packb.glob("month=2026-0[5-7]/side=*.parquet")):
        parts.append(MonthlyReferencePartition(path, path.parent.name[6:], "common30_2025_2026"))
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-store", type=Path,
        default=ROOT / "data_perp/features/20260711_070000",
    )
    parser.add_argument("--coverage-sample-rows", type=int, default=40_000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable input contract: {args.output_dir}")

    partitions = reference_partitions()
    ledger = load_reference_ledgers(partitions)
    segments = {
        name: (
            group[["candidate_id", "__ts__", "__symbol__"]]
            .drop_duplicates(["__symbol__", "__ts__"], keep="first")
            .reset_index(drop=True)
        )
        for name, group in ledger.groupby("population_segment", observed=True, sort=True)
    }
    # One population-balanced scan is materially cheaper than the generic
    # profiler's all+per-segment scans.  Rows are spread through time inside
    # every segment; later selector/OOF chunks retain their own >=90% gates.
    quota = max(1, int(args.coverage_sample_rows) // len(segments))
    balanced: list[pd.DataFrame] = []
    for name in sorted(segments):
        frame = segments[name].sort_values(["__ts__", "candidate_id"], kind="stable")
        if len(frame) > quota:
            positions = np.linspace(0, len(frame) - 1, quota, dtype=np.int64)
            frame = frame.iloc[positions]
        balanced.append(frame)
    identity = pd.concat(balanced, ignore_index=True)
    universe, coverage, contract = build_fresh_causal_feature_contract(
        identity,
        feature_store_dir=args.feature_store,
        cfg=CFG,
        coverage_sample_rows=int(args.coverage_sample_rows),
        min_exact_key_coverage=0.99,
        min_non_null_feature_coverage=0.90,
        max_feature_columns=512,
        max_profile_feature_columns=512,
        reprofile_survivors=False,
        coverage_segments=None,
    )

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "frozen_feature_contract.json").write_text(
        json.dumps(contract.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    partition_frame = pd.DataFrame(
        [{"path": str(p.path), "source_month": p.source_month, "population": p.population}
         for p in partitions]
    )
    partition_frame.to_parquet(args.output_dir / "reference_partitions.parquet", index=False)
    population = (
        ledger.assign(year=ledger["__ts__"].dt.year)
        .groupby(["population_segment", "year", "side_name"], observed=True)
        .size().rename("rows").reset_index()
    )
    population.to_parquet(args.output_dir / "population_coverage.parquet", index=False)
    available = set(contract.feature_columns)
    layer_rows = []
    for item in STAGE_I_ACTIVE_CONTRACTS:
        declared = resolve_stage_i_feature_universe(
            CFG, layer=item.layer, side=item.side, head=item.head
        )
        present = [name for name in declared if name in available]
        layer_rows.append({
            "artifact_key": item.artifact_key,
            "layer": item.layer,
            "side": item.side,
            "head": item.head,
            "declared_features": len(declared),
            "store_admitted_features": len(present),
            "store_admitted_feature_names": json.dumps(present),
        })
    pd.DataFrame(layer_rows).to_parquet(
        args.output_dir / "layer_feature_availability.parquet", index=False
    )
    manifest = {
        "schema": "stage_i_production_input_contract_v1",
        "status": "complete",
        "rows": int(len(ledger)),
        "min_signal_ts": ledger["__ts__"].min().isoformat(),
        "max_signal_ts": ledger["__ts__"].max().isoformat(),
        "population_segments": sorted(segments),
        "feature_store": str(args.feature_store.resolve()),
        "causal_feature_universe": len(universe.feature_columns),
        "coverage_admitted_features": len(contract.feature_columns),
        "feature_contract_sha256": contract.feature_contract_sha256,
        "coverage_sampling": "single population-balanced time-spread scan; downstream chunk gates retained",
        "ranking_contract": "pooled global after common-bps mapping; never per timestamp",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
