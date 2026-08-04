#!/usr/bin/env python3
"""Materialise the bounded broad-R3 -> tail-base input contract.

The output is an immutable, p90-screened valid-label candidate substrate with
raw decision-time fields and row-independent frozen AE/GMM fields.  It creates
T1/T2 labels only; T3 needs separately materialised +4/+6/-4/-6 first-touch
minutes and is reported as unavailable rather than inferred from TP6/SL4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_production_data_adapter import MonthlyReferencePartition
from extreme_price_movements.tail_base_input_contract import (
    PooledP90SpreadMap,
    build_aegmm_projection_source_contract,
    load_frozen_feature_contract,
    materialize_tail_base_input_contract,
)


def _default_partitions() -> list[MonthlyReferencePartition]:
    """The complete multi-era R3 source used by the production input contract."""
    artifacts = ROOT / "data_perp/artifacts"
    values: list[MonthlyReferencePartition] = []
    for root, population in (
        (artifacts / "stage_i_historical_tp6_sl4_h12_r3_20260803_v1/parts", "historical_2022_2023"),
        (artifacts / "stage_i_packb_tp6_sl4_h12_r3_20260803_v1/parts", "common30_2025_2026"),
    ):
        for path in sorted(root.glob("month=*/side=*.parquet")):
            values.append(MonthlyReferencePartition(path, path.parent.name.removeprefix("month="), population))
    surface = artifacts / "stage_i_surface_2024_2026_20260803_v1/source=full_universe_2024"
    for path in sorted(surface.glob("month=*/identity_labels")):
        values.append(MonthlyReferencePartition(path, path.parent.name.removeprefix("month="), "surface_2024"))
    if not values:
        raise FileNotFoundError("no historical/surface/Pack-B R3 label partitions found")
    return sorted(values, key=lambda item: (item.source_month, str(item.path)))


def _partitions_from_parquet(path: Path) -> list[MonthlyReferencePartition]:
    frame = pd.read_parquet(path)
    required = {"path", "source_month", "population"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"reference partitions file missing {missing}")
    return [
        MonthlyReferencePartition(row.path, str(row.source_month), str(row.population))
        for row in frame.loc[:, ["path", "source_month", "population"]].itertuples(index=False)
    ]


def _selected_features(path: Path) -> list[str]:
    """Read the frozen per-side MDA selection, never a broad input universe."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_features")
    if not isinstance(selected, list) or not selected or not all(isinstance(name, str) for name in selected):
        raise ValueError(f"{path} has no immutable selected_features list")
    return sorted(set(selected))


def _raw_feature_list(path: Path) -> list[str]:
    """Load an explicit plain JSON list for a controlled research rerun."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or not all(isinstance(name, str) and name for name in value):
        raise ValueError("--raw-feature-list-json must contain one non-empty JSON string list")
    return sorted(set(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--p90-spread-map", type=Path, required=True,
                        help="CSV/Parquet with exactly __symbol__ and p90_spread_bps; average spread is rejected")
    parser.add_argument("--feature-contract", type=Path,
                        default=ROOT / "data_perp/artifacts/stage_i_production_inputs_20260803_v1/frozen_feature_contract.json")
    parser.add_argument("--feature-store", type=Path,
                        default=ROOT / "data_perp/features/20260711_070000")
    parser.add_argument("--aegmm-state", type=Path,
                        default=ROOT / "data_perp/artifacts/s59_s52_july_oosbase_fullmeta_v9tail95_mlp_hierev_20260716_v2/ae_gmm_state/ae_gmm_state.pkl")
    parser.add_argument("--reference-partitions", type=Path,
                        help="optional parquet path/source_month/population table; default is historical + 2024 surface + Pack-B R3")
    parser.add_argument("--long-r3-manifest", type=Path,
                        default=ROOT / "data_perp/artifacts/stage_i_base_selection_R3_tp6sl4_coverage90_20260804_v1/long/manifest.json")
    parser.add_argument("--short-r3-manifest", type=Path,
                        default=ROOT / "data_perp/artifacts/stage_i_base_selection_R3_tp6sl4_coverage90_20260804_v1/short/manifest.json")
    parser.add_argument("--raw-feature-list-json", type=Path,
                        help="optional plain JSON list; replaces both side selections before AE/GMM source-field union")
    parser.add_argument("--batch-rows", type=int, default=8_000)
    parser.add_argument("--p90-threshold-bps", type=float, default=90.0)
    parser.add_argument("--min-aegmm-source-overlap", type=float, default=0.50)
    args = parser.parse_args()
    partitions = _partitions_from_parquet(args.reference_partitions) if args.reference_partitions else _default_partitions()
    model_contract = load_frozen_feature_contract(args.feature_contract)
    state = pd.read_pickle(args.aegmm_state)
    # The output only loads the actual side-local R3 selections plus frozen
    # AE/GMM source fields that the *same* causal store contract admits.  It
    # never quietly widens to all 489 production fields.
    state_inputs = set(map(str, state.get("feature_columns", ()))) - {"side"}
    if args.raw_feature_list_json:
        requested = _raw_feature_list(args.raw_feature_list_json)
        model_side_features = {"long": requested, "short": requested}
    else:
        model_side_features = {
            "long": _selected_features(args.long_r3_manifest),
            "short": _selected_features(args.short_r3_manifest),
        }
    spread_map = PooledP90SpreadMap.from_path(args.p90_spread_map, threshold_bps=args.p90_threshold_bps)
    # Build a distinct causal source contract for the selected R3 model inputs
    # plus all frozen-state inputs that are schema-available.  This prevents
    # the 489-field generic production screen from needlessly truncating the
    # frozen latent projection, while leaving model features side-local.
    raw_contract, source_side_features, projection_audit = build_aegmm_projection_source_contract(
        partitions=partitions, p90_spread_map=spread_map, feature_store_dir=args.feature_store,
        side_model_features=model_side_features, aegmm_state=state,
    )
    manifest = materialize_tail_base_input_contract(
        partitions=partitions,
        raw_feature_contract=raw_contract,
        p90_spread_map=spread_map,
        aegmm_state=state,
        output_dir=args.output_dir,
        feature_store_dir=args.feature_store,
        batch_rows=args.batch_rows,
        side_raw_features=model_side_features,
        source_raw_features=sorted(set().union(*map(set, source_side_features.values()))),
        min_aegmm_source_overlap=args.min_aegmm_source_overlap,
    )
    manifest["model_feature_contract_sha256"] = model_contract.feature_contract_sha256
    manifest["aegmm_projection_source_discovery"] = projection_audit
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
