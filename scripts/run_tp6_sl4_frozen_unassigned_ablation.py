#!/usr/bin/env python3
"""Ablate an explicit residual head for the unassigned path mass.

The frozen structural-family contract represents only part of native path
contribution mass.  This ablation gives the remaining mass an explicit soft
membership and learns one residual head for it, using the same causal/meta pool
and train-only CMI selection.  It is intentionally separate from the named
stable clusters so its economics can be rejected independently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.conditional_cluster_residual import conditional_mi_scores, materialize_cluster_features, soft_cluster_residual_target
from scripts.run_tp6_sl4_conditional_cluster_residual import _is_leak
from scripts.run_tp6_sl4_frozen_cluster_residual import (
    CONTEXT_CAP, _context_frame, _fit_cluster, _load, _tails,
)

DEFAULT_REPLAY = ROOT / "data_perp/artifacts/tp6_sl4_frozen_cluster_residual_20260812_v1"
DEFAULT_CONTRACT = DEFAULT_REPLAY / "frozen_cluster_contract.json"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_frozen_unassigned_ablation_20260812_v1"
UNASSIGNED_ID = "frozen_unassigned"
SEED = 20260813


def _append_unassigned(cluster_frame: pd.DataFrame) -> pd.DataFrame:
    out = cluster_frame.copy()
    mass = pd.to_numeric(out["cluster_path_unassigned_mass"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(float)
    prefix = f"cluster__{UNASSIGNED_ID}__"
    out[prefix + "membership"] = mass.astype("float32")
    out[prefix + "abs_contribution"] = mass.astype("float32")
    out[prefix + "signed_contribution"] = np.zeros(len(out), dtype="float32")
    out[prefix + "confidence_share"] = np.zeros(len(out), dtype="float32")
    out[prefix + "active"] = (mass > 1e-12).astype("float32")
    out[prefix + "direction"] = np.zeros(len(out), dtype="float32")
    return out


def run(*, replay_dir: Path = DEFAULT_REPLAY, contract_path: Path = DEFAULT_CONTRACT, out: Path = DEFAULT_OUT, base_path: Path | None = None, family_path: Path | None = None, meta_path: Path | None = None) -> Path:
    replay_dir, contract_path, out = map(Path, (replay_dir, contract_path, out))
    out.mkdir(parents=True, exist_ok=True)
    payload = json.loads(contract_path.read_text())
    development_end = pd.Timestamp(payload["development_end"])
    evaluation_start = pd.Timestamp(payload["evaluation_start"])
    evaluation_end = pd.Timestamp(payload["evaluation_end"]) if payload.get("evaluation_end") else None
    base_path = Path(base_path) if base_path else ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_base_20260811_v1.parquet"
    family_path = Path(family_path) if family_path else ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260811_extended_v1/meta_family_contribution_matrix.parquet"
    meta_path = Path(meta_path) if meta_path else ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_meta_pool_regime_20260811_v1.parquet"
    frame, family_fields, meta_fields = _load(base_path, family_path, meta_path, development_end=development_end)
    dev = frame.loc[frame.__ts__.lt(development_end) & frame.label_available_ts.lt(development_end)].copy().reset_index(drop=True)
    test_mask = frame.__ts__.ge(evaluation_start)
    if evaluation_end is not None:
        test_mask &= frame.__ts__.lt(evaluation_end)
    test = frame.loc[test_mask].copy().reset_index(drop=True)
    contracts = payload["clusters"]
    from extreme_price_movements.conditional_cluster_residual import ClusterContract
    frozen = [ClusterContract(
        cluster_id=str(row["cluster_id"]), family_fields=tuple(row["family_fields"]),
        family_indices=tuple(family_fields.index(f) for f in row["family_fields"]),
        centroid_distance=float(row["centroid_distance"]),
    ) for row in contracts]
    cluster_ids = [c.cluster_id for c in frozen]
    dev_cluster = _append_unassigned(materialize_cluster_features(dev, frozen, family_fields=family_fields))
    test_cluster = _append_unassigned(materialize_cluster_features(test, frozen, family_fields=family_fields))
    dev_ctx = _context_frame(dev, dev_cluster, cluster_ids, meta_fields)
    test_ctx = _context_frame(test, test_cluster, cluster_ids, meta_fields)
    membership_field = f"cluster__{UNASSIGNED_ID}__membership"
    membership = dev_cluster[membership_field].to_numpy(float)
    target = soft_cluster_residual_target(dev.net_bps.to_numpy(float) - dev.base_expected_bps.to_numpy(float), membership)
    cmi = conditional_mi_scores(dev_ctx, [f for f in meta_fields if f in dev_ctx.columns], target, membership)
    selected = cmi.head(CONTEXT_CAP).feature.tolist()
    common = [
        "base_expected_bps", "p_clear", "p_adverse", "p_weak", "base_raw", "base_score",
        membership_field, f"cluster__{UNASSIGNED_ID}__abs_contribution", f"cluster__{UNASSIGNED_ID}__active",
        "cluster_path_represented_mass", "cluster_path_unassigned_mass", "cluster_path_assignment_quality",
        "cluster_path_low_confidence_mass", "cluster_path_entropy", "cluster_path_top2_margin",
    ]
    fields = list(dict.fromkeys([f for f in common if f in dev_ctx.columns] + selected))
    correction = _fit_cluster(dev_ctx, test_ctx, target, 0.25 + 0.75 * membership, fields, SEED)
    existing = pd.read_parquet(replay_dir / "frozen_cluster_oos_predictions.parquet")
    result = test[["candidate_id", "__ts__", "month", "net_bps", "gross_bps", "base_expected_bps"]].copy()
    result = result.merge(existing[["candidate_id", "frozen_cluster_score"]], on="candidate_id", how="inner", validate="one_to_one")
    if len(result) != len(test):
        raise ValueError("unassigned ablation lost held candidates when joining frozen scores")
    result["unassigned_correction"] = np.clip(correction, -200.0, 200.0)
    result["base_score"] = result.base_expected_bps.to_numpy(float)
    result["frozen_cluster_plus_unassigned"] = result.frozen_cluster_score.to_numpy(float) + result.unassigned_correction.to_numpy(float)
    result.to_parquet(out / "frozen_unassigned_oos_predictions.parquet", index=False, compression="zstd")
    metrics = pd.DataFrame(_tails(result, "base_score") + _tails(result, "frozen_cluster_score") + _tails(result, "frozen_cluster_plus_unassigned"))
    metrics.to_parquet(out / "frozen_unassigned_metrics.parquet", index=False, compression="zstd")
    cmi.to_parquet(out / "unassigned_cmi_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "tp6_sl4_frozen_unassigned_ablation_v1", "status": "complete",
        "replay": str(replay_dir), "contract": str(contract_path),
        "development_end": str(development_end), "evaluation_start": str(evaluation_start),
        "evaluation_end": str(evaluation_end) if evaluation_end is not None else None,
        "meta_pool_count_before_selection": int(len(meta_fields)), "selected_count": int(len(selected)),
        "selected_fields": selected, "target": "unassigned_mass * (net_bps - base_expected_bps)",
        "target_like_fields_selected": [f for f in selected if _is_leak(f)],
        "held_outcomes_used_for_fit_or_selection": False, "global_ranking": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    correctness = {
        "schema": "tp6_sl4_frozen_unassigned_correctness_v1",
        "contract_frozen_before_evaluation": bool(development_end <= evaluation_start),
        "development_labels_matured": bool(dev.label_available_ts.lt(development_end).all()),
        "meta_pool_before_selection_count": int(len(meta_fields)),
        "target_like_fields_selected_count": int(sum(_is_leak(f) for f in selected)),
        "held_outcomes_used_for_fit_or_selection": False,
        "global_ranking_after_score_generation": True,
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    pooled = metrics.query("period == 'all'").sort_values(["tail", "net_bps_per_trade"], ascending=[True, False])
    lines = ["# TP6/SL4 explicit unassigned-path residual ablation", "", "The unassigned path mass is modeled as a separate soft residual head; it is not silently merged into a named cluster.", "", "| arm | tail | trades | gross bps/trade | net bps/trade | rank IC |", "|---|---:|---:|---:|---:|---:|"]
    for row in pooled.itertuples(index=False):
        lines.append(f"| {row.arm} | {row.tail:.3g} | {row.trades} | {row.gross_bps_per_trade:.2f} | {row.net_bps_per_trade:.2f} | {row.rank_ic:.4f} |")
    (out / "FROZEN_UNASSIGNED_REPORT.md").write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base", type=Path, default=None)
    parser.add_argument("--family", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    args = parser.parse_args()
    print(run(replay_dir=args.replay_dir, contract_path=args.contract, out=args.out, base_path=args.base, family_path=args.family, meta_path=args.meta))
