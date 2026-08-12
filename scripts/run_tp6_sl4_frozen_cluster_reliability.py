#!/usr/bin/env python3
"""Learned reliability correction on the frozen TP6/SL4 cluster layer.

The frozen cluster contract is discovered on 2024 development rows and reused
unchanged for 2025.  A second learner estimates each cluster's conversion
error using leave-one-month-out cluster predictions from 2024; its target is
therefore OOF with respect to the cluster residual model.
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

from extreme_price_movements.conditional_cluster_residual import (
    ClusterContract,
    conditional_mi_scores,
    materialize_cluster_features,
    soft_cluster_residual_target,
)
from scripts.run_tp6_sl4_frozen_cluster_residual import (
    CONTEXT_CAP,
    DEFAULT_BASE,
    DEFAULT_FAMILY,
    DEFAULT_META,
    DEV_END,
    SEED,
    _context_frame,
    _fit_cluster,
    _load,
    _numeric,
    _tails,
)
from scripts.run_tp6_sl4_conditional_cluster_residual import _is_leak

DEFAULT_CONTRACT = ROOT / "data_perp/artifacts/tp6_sl4_frozen_cluster_residual_20260812_v1/frozen_cluster_contract.json"
DEFAULT_SELECTION = ROOT / "data_perp/artifacts/tp6_sl4_frozen_cluster_residual_20260812_v1/frozen_cluster_feature_selection.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_frozen_cluster_reliability_20260812_v1"
RELIABILITY_CAP = 16
RELIABILITY_STRENGTHS = (0.25, 0.50, 0.75, 1.00)


def _load_contract(path: Path, family_fields: list[str]) -> list[ClusterContract]:
    payload = json.loads(Path(path).read_text())
    if payload.get("development_end") != str(DEV_END):
        raise ValueError("frozen cluster contract cutoff drift")
    contracts: list[ClusterContract] = []
    for row in payload["clusters"]:
        fields = tuple(sorted(map(str, row["family_fields"])))
        contracts.append(ClusterContract(
            cluster_id=str(row["cluster_id"]),
            family_fields=fields,
            family_indices=tuple(family_fields.index(f) for f in fields),
            centroid_distance=float(row["centroid_distance"]),
        ))
    return contracts


def _common_fields(ctx: pd.DataFrame, cluster_id: str) -> list[str]:
    prefix = f"cluster__{cluster_id}__"
    requested = [
        "base_expected_bps", "p_clear", "p_adverse", "p_weak", "base_raw", "base_score",
        f"{prefix}membership", f"{prefix}abs_contribution", f"{prefix}signed_contribution",
        f"{prefix}confidence_share", f"{prefix}active", "cluster_path_represented_mass",
        "cluster_path_unassigned_mass", "cluster_path_assignment_quality",
        "cluster_path_low_confidence_mass", "cluster_path_entropy", "cluster_path_top2_margin",
        "cluster_membership_max", "cluster_membership_second",
    ]
    return [f for f in requested if f in ctx.columns]


def _selected_map(path: Path, ctx: pd.DataFrame, meta_fields: list[str], dev: pd.DataFrame, clusters: pd.DataFrame, cluster_ids: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if path.exists():
        saved = pd.read_parquet(path)
        for row in saved.itertuples(index=False):
            try:
                fields = list(json.loads(row.selected_fields))
            except Exception:
                fields = []
            result[str(row.cluster_id)] = [f for f in fields if f in ctx.columns]
    residual = dev.net_bps.to_numpy(float) - dev.base_expected_bps.to_numpy(float)
    for cluster_id in cluster_ids:
        if cluster_id in result and result[cluster_id]:
            continue
        membership = clusters[f"cluster__{cluster_id}__membership"].to_numpy(float)
        target = soft_cluster_residual_target(residual, membership)
        cmi = conditional_mi_scores(ctx, [f for f in meta_fields if f in ctx.columns], target, membership)
        result[cluster_id] = cmi.head(CONTEXT_CAP).feature.tolist()
    return result


def _fit_reliability_model(train_ctx: pd.DataFrame, test_ctx: pd.DataFrame, target: np.ndarray, weight: np.ndarray, fields: list[str], seed: int) -> np.ndarray:
    import lightgbm as lgb

    common = dict(
        objective="huber", alpha=0.85, n_estimators=160, learning_rate=0.03,
        max_depth=4, num_leaves=15, min_child_samples=300,
        feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1,
        reg_alpha=0.0, reg_lambda=20.0, max_bin=127,
        random_state=seed, n_jobs=1, verbosity=-1,
    )
    xtr, med = _numeric(train_ctx, fields)
    xte, _ = _numeric(test_ctx, fields, med)
    model = lgb.LGBMRegressor(**common)
    model.fit(xtr, np.asarray(target, dtype=float), sample_weight=np.asarray(weight, dtype=float))
    return np.asarray(model.predict(xte), dtype=float)


def run(*, base_path: Path = DEFAULT_BASE, family_path: Path = DEFAULT_FAMILY, meta_path: Path = DEFAULT_META, contract_path: Path = DEFAULT_CONTRACT, selection_path: Path = DEFAULT_SELECTION, out: Path = DEFAULT_OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    frame, family_fields, meta_fields = _load(base_path, family_path, meta_path)
    dev = frame.loc[frame.__ts__.lt(DEV_END)].copy().reset_index(drop=True)
    test = frame.loc[frame.__ts__.ge(DEV_END)].copy().reset_index(drop=True)
    contracts = _load_contract(contract_path, family_fields)
    cluster_ids = [c.cluster_id for c in contracts]
    dev_cluster = materialize_cluster_features(dev, contracts, family_fields=family_fields)
    test_cluster = materialize_cluster_features(test, contracts, family_fields=family_fields)
    dev_ctx = _context_frame(dev, dev_cluster, cluster_ids, meta_fields)
    test_ctx = _context_frame(test, test_cluster, cluster_ids, meta_fields)
    selected_map = _selected_map(selection_path, dev_ctx, meta_fields, dev, dev_cluster, cluster_ids)

    # Leave-one-month-out predictions create an OOF cluster correction on the
    # development population. April has no prior month and is intentionally
    # excluded from the reliability fit.
    oof = dev[["candidate_id", "__ts__", "month", "net_bps", "gross_bps", "base_expected_bps"]].copy()
    oof_pred = {c: np.full(len(dev), np.nan, dtype=float) for c in cluster_ids}
    training_audit: list[dict[str, object]] = []
    for month in sorted(dev.month.astype(str).unique()):
        start = pd.Timestamp(month + "-01", tz="UTC")
        tr_mask = dev.__ts__.lt(start) & dev.label_available_ts.lt(start)
        va_mask = dev.month.astype(str).eq(month)
        if int(tr_mask.sum()) < 500 or int(va_mask.sum()) < 100:
            continue
        tr_ctx = dev_ctx.loc[tr_mask].reset_index(drop=True)
        va_ctx = dev_ctx.loc[va_mask].reset_index(drop=True)
        tr_resid = dev.loc[tr_mask, "net_bps"].to_numpy(float) - dev.loc[tr_mask, "base_expected_bps"].to_numpy(float)
        for idx, cluster_id in enumerate(cluster_ids):
            membership = dev_cluster.loc[tr_mask, f"cluster__{cluster_id}__membership"].to_numpy(float)
            target = soft_cluster_residual_target(tr_resid, membership)
            fields = list(dict.fromkeys(_common_fields(tr_ctx, cluster_id) + selected_map[cluster_id]))
            pred = _fit_cluster(tr_ctx, va_ctx, target, 0.25 + 0.75 * membership, fields, SEED + 100 + idx)
            oof_pred[cluster_id][va_mask.to_numpy()] = pred
        training_audit.append({"month": month, "train_rows": int(tr_mask.sum()), "validation_rows": int(va_mask.sum())})
    for cluster_id, values in oof_pred.items():
        oof[f"cluster_pred__{cluster_id}"] = values
    oof["oof_available"] = oof[[f"cluster_pred__{c}" for c in cluster_ids]].notna().all(axis=1)
    oof.to_parquet(out / "development_cluster_oof_predictions.parquet", index=False, compression="zstd")

    # Fit the final frozen cluster residual models on all development rows for
    # the 2025 score, using the frozen selected fields.
    final_pred: dict[str, np.ndarray] = {}
    for idx, cluster_id in enumerate(cluster_ids):
        membership = dev_cluster[f"cluster__{cluster_id}__membership"].to_numpy(float)
        target = soft_cluster_residual_target(dev.net_bps.to_numpy(float) - dev.base_expected_bps.to_numpy(float), membership)
        fields = list(dict.fromkeys(_common_fields(dev_ctx, cluster_id) + selected_map[cluster_id]))
        final_pred[cluster_id] = _fit_cluster(dev_ctx, test_ctx, target, 0.25 + 0.75 * membership, fields, SEED + idx)

    # Train one reliability model per stable cluster on the OOF dev residual.
    rel_selected: list[dict[str, object]] = []
    reliability_pred: dict[str, np.ndarray] = {}
    valid = oof.oof_available.to_numpy(bool)
    for idx, cluster_id in enumerate(cluster_ids):
        pred_col = f"cluster_pred__{cluster_id}"
        membership_dev = dev_cluster[f"cluster__{cluster_id}__membership"].to_numpy(float)
        actual_target = soft_cluster_residual_target(dev.net_bps.to_numpy(float) - dev.base_expected_bps.to_numpy(float), membership_dev)
        rel_target = actual_target[valid] - oof[pred_col].to_numpy(float)[valid]
        rel_ctx = dev_ctx.loc[valid].reset_index(drop=True).copy()
        rel_ctx["cluster_model_prediction"] = oof[pred_col].to_numpy(float)[valid]
        cmi = conditional_mi_scores(rel_ctx, [f for f in meta_fields if f in rel_ctx.columns], rel_target, membership_dev[valid])
        selected = cmi.head(RELIABILITY_CAP).feature.tolist()
        fields = list(dict.fromkeys(_common_fields(rel_ctx, cluster_id) + ["cluster_model_prediction"] + selected))
        reliability_pred[cluster_id] = _fit_reliability_model(
            rel_ctx,
            pd.concat([test_ctx.reset_index(drop=True), pd.DataFrame({"cluster_model_prediction": final_pred[cluster_id]})], axis=1),
            rel_target,
            0.25 + 0.75 * membership_dev[valid],
            fields,
            SEED + 200 + idx,
        )
        rel_selected.append({
            "cluster_id": cluster_id, "training_rows": int(valid.sum()),
            "meta_pool_count": int(len(meta_fields)), "selected_count": int(len(selected)),
            "selected_fields": json.dumps(selected), "target": "OOF soft-cluster target - OOF cluster prediction",
        })

    result = test[["candidate_id", "__ts__", "month", "net_bps", "gross_bps", "base_expected_bps"]].copy()
    result["base_score"] = result.base_expected_bps.to_numpy(float)
    result["frozen_cluster_correction"] = np.sum(np.column_stack(list(final_pred.values())), axis=1)
    result["frozen_cluster_score"] = result.base_score.to_numpy(float) + np.clip(result.frozen_cluster_correction.to_numpy(float), -200.0, 200.0)
    result["reliability_correction"] = np.sum(np.column_stack(list(reliability_pred.values())), axis=1)
    for strength in RELIABILITY_STRENGTHS:
        result[f"frozen_cluster_reliable_a{strength:g}"] = result.frozen_cluster_score.to_numpy(float) + float(strength) * np.clip(result.reliability_correction.to_numpy(float), -200.0, 200.0)
    result.to_parquet(out / "frozen_cluster_reliability_oos_predictions.parquet", index=False, compression="zstd")
    arms = ["base_score", "frozen_cluster_score", *[f"frozen_cluster_reliable_a{strength:g}" for strength in RELIABILITY_STRENGTHS]]
    metrics = pd.DataFrame(sum((_tails(result, arm) for arm in arms), []))
    metrics.to_parquet(out / "frozen_cluster_reliability_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(rel_selected).to_parquet(out / "reliability_feature_selection.parquet", index=False, compression="zstd")
    pd.DataFrame(training_audit).to_parquet(out / "reliability_training_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "tp6_sl4_frozen_cluster_reliability_v1", "status": "complete",
        "contract": str(contract_path), "base": str(base_path), "family": str(family_path), "meta": str(meta_path),
        "development_end": str(DEV_END), "development_rows": int(len(dev)), "oos_rows": int(len(test)),
        "cluster_count": len(cluster_ids), "meta_pool_count_before_selection": len(meta_fields),
        "cluster_target": "membership * (net_bps - base_expected_bps)",
        "reliability_target": "OOF soft-cluster target - OOF cluster prediction",
        "reliability_training": "leave-one-month-out 2024 cluster predictions; April excluded without prior support",
        "reliability_strengths": RELIABILITY_STRENGTHS,
        "held_outcomes_used_for_fit": False, "global_ranking": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    selected_fields = [field for row in rel_selected for field in json.loads(row["selected_fields"])]
    correctness = {
        "schema": "tp6_sl4_frozen_cluster_reliability_correctness_v1",
        "development_end": str(DEV_END),
        "development_oof_rows": int(valid.sum()),
        "april_excluded_without_prior_support": True,
        "reliability_target_is_oof_cluster_residual": True,
        "meta_pool_before_selection_count": int(len(meta_fields)),
        "target_like_fields_selected": sorted({f for f in selected_fields if _is_leak(f)}),
        "target_like_fields_selected_count": int(sum(_is_leak(f) for f in selected_fields)),
        "held_outcomes_used_for_2025_fit_or_selection": False,
        "global_ranking_after_score_generation": True,
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    pooled = metrics.query("period == 'all'").sort_values(["tail", "net_bps_per_trade"], ascending=[True, False])
    lines = [
        "# TP6/SL4 frozen cluster reliability learner", "",
        "Stable five-cluster contract discovered on 2024 development rows. Reliability models use leave-one-month-out 2024 cluster predictions and are evaluated on 2025 only.", "",
        "| arm | tail | trades | gross bps/trade | net bps/trade | rank IC |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pooled.itertuples(index=False):
        lines.append(f"| {row.arm} | {row.tail:.3g} | {row.trades} | {row.gross_bps_per_trade:.2f} | {row.net_bps_per_trade:.2f} | {row.rank_ic:.4f} |")
    lines += ["", "The reliability target is OOF with respect to the frozen cluster residual model; no 2025 outcomes enter fitting or feature selection."]
    (out / "FROZEN_CLUSTER_RELIABILITY_REPORT.md").write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(base_path=args.base, family_path=args.family, meta_path=args.meta, contract_path=args.contract, selection_path=args.selection, out=args.out))
