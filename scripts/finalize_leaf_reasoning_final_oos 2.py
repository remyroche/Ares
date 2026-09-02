#!/usr/bin/env python3
"""Fit the development-selected leaf-reasoning stack and seal final OOS.

This command is intentionally *not* an OOS replay.  It accepts only a frozen
development selection and pre-cutoff Parquet projections, writes native
LightGBM base/meta text models plus side value maps, and emits the contract
consumed later by ``run_leaf_reasoning_final_oos.py``.  There are no HPO,
feature-selection, clustering, policy, or final-panel arguments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.leaf_reasoning_finalizer import (  # noqa: E402
    DevelopmentFinalizationSelection,
    finalize_leaf_reasoning_final_oos,
    read_pre_cutoff_parquet,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-selection", required=True, type=Path, help="immutable leaf_reasoning_finalizer_selection_v1 JSON")
    parser.add_argument("--base-training-panel", required=True, type=Path, help="Parquet source for the frozen F0 base final fit")
    parser.add_argument("--meta-ledger", required=True, type=Path, help="Parquet source with strict same-side base OOF rows for final meta fit")
    parser.add_argument("--output-dir", required=True, type=Path, help="new immutable finalization directory")
    args = parser.parse_args()

    selection = DevelopmentFinalizationSelection.from_json_path(args.development_selection)
    base_columns = [
        "candidate_id", "side_name", "decision_ts", "label_available_ts", "gross_bps", "net_bps", "r3_class",
        "robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50", "f0_sample_weight",
        *[field for side in ("long", "short") for field in selection.base_features_by_side[side]],
    ]
    meta_columns = [
        "candidate_id", "side_name", "decision_ts", "label_available_ts", "base_expected_bps", "realized_net_bps",
        "base_same_side_strict_oof", "base_oof_fit_end_ts", "base_oof_generated_ts",
        *[field for side in ("long", "short") for field in selection.meta_features_by_side[side]],
    ]
    # Optional robust-clear flags are not guaranteed to be present when an
    # immutable precomputed F0 weight column is bound.  Read only columns that
    # physically exist; the finalizer then fail-closes unless one valid frozen
    # weighting representation is supplied.
    def existing(path: Path, fields: list[str]) -> list[str]:
        try:
            import pyarrow.parquet as pq
            names = set(pq.ParquetFile(path).schema_arrow.names)
        except Exception as exc:  # pragma: no cover - CLI/environment surface
            parser.error(f"cannot inspect parquet schema for {path}: {exc}")
        mandatory = set(fields).difference({"robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50", "f0_sample_weight"})
        missing = sorted(mandatory.difference(names))
        if missing:
            parser.error(f"{path} lacks required frozen fields: {missing[:16]}")
        return [field for field in dict.fromkeys(fields) if field in names]

    base = read_pre_cutoff_parquet(args.base_training_panel, columns=existing(args.base_training_panel, base_columns))
    meta = read_pre_cutoff_parquet(args.meta_ledger, columns=existing(args.meta_ledger, meta_columns))
    result = finalize_leaf_reasoning_final_oos(selection, base, meta, output_dir=args.output_dir)
    print(result.frozen_contract_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
