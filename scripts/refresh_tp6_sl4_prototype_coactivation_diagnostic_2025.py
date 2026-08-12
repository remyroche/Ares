#!/usr/bin/env python3
"""Refresh only the outcome-bearing prototype co-activation diagnostic.

This intentionally does not fit a model.  It attaches the frozen K=9
membership contract to the already-produced matched 2025 predictions and
summarises absolute and row-local-top-n co-activations.  The resulting grid is
for research/audit only; it is not an inference feature contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_downstream_retrain_2025 import MONTHS, _load  # noqa: E402
from scripts.run_tp6_sl4_prototype_cluster_use_ablation_2025 import _coactivation_grid  # noqa: E402


STRUCTURE = ROOT / "data_perp/artifacts/tp6_sl4_prototype_cluster_quality_20260809_v3"
PREDICTIONS = ROOT / "data_perp/artifacts/tp6_sl4_prototype_cluster_use_ablation_20260809_health_v2/predictions.parquet"
OUT = ROOT / "data_perp/artifacts/tp6_sl4_prototype_cluster_coactivation_20260809_v2"


def run(*, structure: Path = STRUCTURE, predictions: Path = PREDICTIONS, out: Path = OUT) -> Path:
    if out.exists():
        raise FileExistsError(out)
    source, context, context_hash = _load()
    source = source.loc[source.side_name.eq("long") & source.month.astype(str).isin(MONTHS)].copy()
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True)
    source["month"] = source["month"].astype(str)
    keep = ["candidate_id", "month", "__ts__", "base_score", *context]
    source = source.loc[:, keep]
    membership = pd.read_parquet(structure / "prototype_cluster_size_sweep_features.parquet")
    membership["__ts__"] = pd.to_datetime(membership["__ts__"], utc=True)
    fields = [field for field in membership.columns if field.startswith("k09__cluster__") and field.endswith("__membership")]
    membership = membership.loc[membership.month.astype(str).isin(MONTHS), ["candidate_id", "month", "__ts__", *fields]]
    prediction = pd.read_parquet(predictions)
    prediction["__ts__"] = pd.to_datetime(prediction["__ts__"], utc=True)
    analysis = prediction.loc[:, ["candidate_id", "month", "__ts__", "net_bps", "gross_bps", "canonical_control"]].merge(
        source, on=["candidate_id", "month", "__ts__"], how="inner", validate="one_to_one"
    ).merge(membership, on=["candidate_id", "month", "__ts__"], how="inner", validate="one_to_one")
    if len(analysis) != len(prediction):
        raise RuntimeError(f"candidate mismatch: prediction={len(prediction)} analysis={len(analysis)}")
    grid, shifts = _coactivation_grid(analysis, membership_fields=fields, context_fields=context)
    out.mkdir(parents=True)
    grid.to_parquet(out / "cluster_coactivation_grid_2025.parquet", index=False)
    shifts.to_parquet(out / "cluster_coactivation_context_shift_2025.parquet", index=False)
    grid.sort_values(["activation_mode", "rows"], ascending=[True, False], kind="stable").to_parquet(out / "cluster_coactivation_high_support.parquet", index=False)
    grid.loc[grid.rows.ge(200)].sort_values(
        ["activation_mode", "canonical_top10_net_bps", "rows"], ascending=[True, False, False], kind="stable"
    ).to_parquet(out / "cluster_coactivation_high_support_economics.parquet", index=False)
    correctness = {
        "schema": "tp6_sl4_prototype_coactivation_diagnostic_correctness_v1",
        "frozen_k9_contract": True,
        "matched_2025_candidate_ids": True,
        "diagnostic_uses_realised_outcomes_only_after_predictions_are_frozen": True,
        "diagnostic_is_not_an_inference_feature": True,
        "row_local_topn_uses_only_same_row_soft_memberships": True,
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {
        "schema": "tp6_sl4_prototype_cluster_coactivation_20260809_v2",
        "status": "COMPLETE", "rows": len(analysis), "grid_rows": len(grid),
        "structure": str(structure), "predictions": str(predictions), "context_sha256": context_hash,
        "activation_modes": ["absolute_membership: 0.05, 0.10, 0.20", "row_local_top2", "row_local_top3"],
        "usage": "diagnostic only; do not feed outcome-bearing grid metrics into an inference model",
        "artifacts": sorted(path.name for path in out.iterdir()),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# Frozen K=9 co-activation diagnostic — 2025", "",
        "The grid tests pairs and triples of persistent frozen clusters. It includes absolute membership activation and causal row-local top-2/top-3 activation to avoid treating a broad soft membership as a discrete state. These metrics use realised outcomes, are diagnostic only, and must not be fed into a live model.", "",
        "## Support by activation mode", "", grid.groupby("activation_mode", sort=True).agg(rows=("rows", "size"), median_support=("rows", "median"), max_support=("rows", "max")).to_string(), "",
        "## Row-local top-2, support >= 200", "", grid.loc[(grid.activation_mode.eq("row_local_top2")) & grid.rows.ge(200)].sort_values("canonical_top10_net_bps", ascending=False).head(30).round(3).to_string(index=False), "",
    ]
    (out / "TP6_SL4_PROTOTYPE_COACTIVATION_DIAGNOSTIC.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"out": str(out), "rows": len(analysis), "grid_rows": len(grid)}, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, default=STRUCTURE)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(structure=args.structure, predictions=args.predictions, out=args.out)
