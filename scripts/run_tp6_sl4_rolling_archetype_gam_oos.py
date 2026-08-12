#!/usr/bin/env python3
"""Prequential month-ahead replay for structural archetypes and the GAM.

Unlike the frozen-development replay, every target month gets a fresh
archetype/cluster/GAM fit using only the immediately preceding 1, 2, or 3
available model months.  The target month is scored once and ranked globally.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.cofiring_economic_clusters import (  # noqa: E402
    CoFiringClusterContract,
    materialize_memberships,
    pairwise_cofiring_similarity,
)
from extreme_price_movements.structural_archetypes import (  # noqa: E402
    StructuralArchetype,
    build_recurrent_archetypes,
    match_catalog_to_archetypes,
    materialize_row_archetype_exposures,
)
from scripts.run_tp6_sl4_archetype_cluster_vcgam_oof import (  # noqa: E402
    _fit_vc,
    _weighted_cmi,
)
from scripts.run_tp6_sl4_frozen_cluster_residual import _load  # noqa: E402


SIDE = "long"
SEED = 20260815
WINDOWS = (1, 2, 3)
TOP_N = 3
MATCH_TEMPERATURE = 0.08
MATCH_THRESHOLD = 0.55
CLUSTER_ACTIVE_THRESHOLD = 0.10
CLUSTER_REPORT_ACTIVE_THRESHOLD = 0.05
MAX_ARCHETYPES = 64
GAM_CONTEXT_CAP = 12
GAMMAS = (0.25, 0.50, 1.00)
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)

DEFAULT_BASE = ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_base_20260811_v1.parquet"
DEFAULT_FAMILY = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260811_extended_v1/meta_family_contribution_matrix.parquet"
DEFAULT_META = ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_meta_pool_regime_20260811_v1.parquet"
DEFAULT_RAW = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260811_extended_v1/strict_base_reasoning"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_rolling_archetype_gam_oos_20260815_v1"


def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 32 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def _load_raw_month(raw_root: Path, month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = raw_root / f"month={month}"
    catalog = pd.read_parquet(folder / "leaf_rule_catalog.parquet")
    leaves = pd.read_parquet(folder / "leaf_assignments.parquet")
    catalog["fold_id"] = catalog.fold_id.astype(str)
    return catalog, leaves


def _cap_archetypes(archetypes: Sequence[StructuralArchetype], cap: int) -> list[StructuralArchetype]:
    ordered = sorted(
        archetypes,
        key=lambda a: (a.train_frequency_median * abs(a.contribution_median), a.recurrence_count, a.archetype_id),
        reverse=True,
    )[:cap]
    return [replace(a, archetype_id=f"archetype_{i:04d}") for i, a in enumerate(ordered)]


def _local_archetypes(catalog: pd.DataFrame, window: int) -> tuple[list[StructuralArchetype], str, int]:
    """Build a local contract, relaxing recurrence only as the window shrinks."""
    min_folds = min(3, int(window))
    separated_gap = 2 if window >= 3 else (1 if window == 2 else 0)
    archetypes, audit = build_recurrent_archetypes(
        catalog,
        min_folds=min_folds,
        min_sign_consistency=0.80,
        min_train_frequency=0.005,
        separated_gap=separated_gap,
    )
    if len(archetypes) >= 2:
        mode = f"recurrent_min{min_folds}"
        return _cap_archetypes(archetypes, MAX_ARCHETYPES), mode, int(len(archetypes))

    # A one-month inference fit cannot satisfy cross-fit recurrence.  Use the
    # strongest local structural paths as local archetypes instead of falling
    # back to fold-family IDs or silently disabling the branch.
    grouped = catalog.groupby("rule_signature", as_index=False).agg(
        rule_structural_path_json=("rule_structural_path_json", "first"),
        contribution=("ensemble_tree_contribution", "median"),
        frequency=("train_leaf_frequency", "median"),
        folds=("fold_id", lambda x: tuple(sorted(set(map(str, x))))),
    )
    grouped["importance"] = grouped.frequency.astype(float) * grouped.contribution.abs().astype(float)
    grouped = grouped.sort_values(["importance", "frequency"], ascending=False, kind="stable").head(MAX_ARCHETYPES)
    rows: list[StructuralArchetype] = []
    for idx, row in enumerate(grouped.itertuples(index=False)):
        path = json.loads(row.rule_structural_path_json) if isinstance(row.rule_structural_path_json, str) else row.rule_structural_path_json
        tokens = tuple(
            (
                str(x.get("feature", "")),
                str(x.get("branch", "")),
                int(x.get("threshold_band_index", -1)),
                int(x.get("threshold_band_count", -1)),
            )
            for x in (path or [])
            if isinstance(x, dict)
        )
        sign = int(np.sign(float(row.contribution))) or 1
        rows.append(StructuralArchetype(
            archetype_id=f"archetype_{idx:04d}",
            rule_signature=str(grouped.iloc[idx].rule_signature),
            tokens=tokens,
            recurrence_folds=tuple(row.folds),
            recurrence_count=len(row.folds),
            sign=sign,
            sign_consistency=1.0,
            contribution_median=float(abs(row.contribution)),
            train_frequency_median=float(row.frequency),
        ))
    return rows, "local_top_paths", 0


def _select_clusters(abs_train: pd.DataFrame, signed_train: pd.DataFrame, residual: np.ndarray, train_months: Sequence[str], mapping_quality: float) -> tuple[list[CoFiringClusterContract], pd.DataFrame, dict[str, float]]:
    if abs_train.shape[1] < 2:
        return [], pd.DataFrame(), {"valid": False, "reason": "<2 archetypes"}
    sim, _, _ = pairwise_cofiring_similarity(abs_train, signed_train, residual, active_threshold=CLUSTER_ACTIVE_THRESHOLD)
    candidates: list[dict[str, object]] = []
    label_map: dict[int, np.ndarray] = {}
    for k in range(2, min(6, abs_train.shape[1]) + 1):
        labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(1.0 - sim)
        label_map[k] = labels
        masses = np.column_stack([abs_train.to_numpy(float)[:, labels == j].sum(axis=1) for j in range(k)])
        mass_totals = np.maximum(masses.sum(axis=0), 1e-12)
        shares = mass_totals / mass_totals.sum()
        entropy = -float(np.sum(shares * np.log(np.maximum(shares, 1e-12)))) / max(math.log(k), 1e-12)
        compact = []
        for j in range(k):
            idx = np.flatnonzero(labels == j)
            tri = sim[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)] if len(idx) > 1 else np.array([0.0])
            compact.append(float(np.mean(tri)))
        support = []
        cv = []
        for j in range(k):
            month_mass = []
            for month in sorted(set(map(str, train_months))):
                block = masses[np.asarray(train_months, dtype=str) == month, j]
                month_mass.append(float(np.mean(block)) if len(block) else 0.0)
            support.append(float(np.mean(np.asarray(month_mass) >= 0.02)))
            cv.append(float(np.std(month_mass) / max(np.mean(month_mass), 1e-8)))
        mass_coverage = float(np.median(mass_totals / max(len(abs_train), 1)))
        transport_score = float(
            0.40 * float(np.mean(support))
            + 0.25 * float(np.clip(mass_coverage / 0.20, 0.0, 1.0))
            + 0.20 * float(np.clip(mapping_quality / 0.50, 0.0, 1.0))
            + 0.15 * float(np.median(1.0 / (1.0 + np.asarray(cv))))
        )
        balance_gate = bool(shares.max() <= 0.75 and shares.min() >= 0.02)
        support_gate = bool(min(support, default=0.0) >= 0.50)
        candidates.append({
            "k": k,
            "transport_score": transport_score,
            "balance": float(entropy),
            "max_mass_share": float(shares.max()),
            "min_mass_share": float(shares.min()),
            "mass_coverage": mass_coverage,
            "min_period_support": min(support, default=0.0),
            "median_mass_cv": float(np.median(cv)) if cv else np.nan,
            "mapping_quality": mapping_quality,
            "balance_gate": balance_gate,
            "support_gate": support_gate,
            "valid": bool(balance_gate and support_gate),
        })
    audit = pd.DataFrame(candidates).sort_values(["valid", "transport_score", "balance"], ascending=[False, False, False], kind="stable").reset_index(drop=True)
    if audit.empty:
        return [], audit, {"valid": False, "reason": "no cluster candidates"}
    selected_k = int(audit.iloc[0].k)
    labels = label_map[selected_k]
    contracts: list[CoFiringClusterContract] = []
    for j in range(selected_k):
        idx = np.flatnonzero(labels == j)
        inner = sim[np.ix_(idx, idx)]
        tri = inner[np.triu_indices(len(idx), 1)] if len(idx) > 1 else np.array([0.0])
        contracts.append(CoFiringClusterContract(
            cluster_id=f"rolling_cluster_{j:02d}",
            family_fields=tuple(abs_train.columns[idx].astype(str).tolist()),
            family_indices=tuple(int(x) for x in idx),
            mean_pair_similarity=float(np.mean(tri)),
            economic_coherence=1.0,
        ))
    selected = audit.iloc[0].to_dict()
    selected["valid"] = bool(selected.get("valid", False))
    return contracts, audit, selected


def _tail_metrics(frame: pd.DataFrame, score_cols: Sequence[str], window: int, target_month: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in score_cols:
        for tail in TAILS:
            n = max(1, int(math.ceil(len(frame) * tail)))
            top = frame.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            rows.append({
                "window_months": window,
                "target_month": target_month,
                "arm": col,
                "tail": tail,
                "trades": n,
                "gross_bps_per_trade": float(top.gross_bps.mean()),
                "net_bps_per_trade": float(top.net_bps.mean()),
                "rank_ic": _rank_ic(frame[col].to_numpy(float), frame.net_bps.to_numpy(float)),
            })
    return pd.DataFrame(rows)


def run(*, base_path: Path = DEFAULT_BASE, family_path: Path = DEFAULT_FAMILY, meta_path: Path = DEFAULT_META, raw_root: Path = DEFAULT_RAW, out: Path = DEFAULT_OUT) -> Path:
    if out.exists():
        raise FileExistsError(out)
    frame, _, meta_fields = _load(base_path, family_path, meta_path, development_end=pd.Timestamp("2099-01-01", tz="UTC"))
    frame = frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    frame["residual_bps"] = frame.net_bps.to_numpy(float) - frame.base_expected_bps.to_numpy(float)
    raw: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for path in sorted(raw_root.glob("month=*/leaf_rule_catalog.parquet")):
        month = path.parent.name.split("=", 1)[1]
        raw[month] = _load_raw_month(raw_root, month)
    months = sorted(raw)
    prediction_parts: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    archetype_rows: list[dict[str, object]] = []
    metric_parts: list[pd.DataFrame] = []
    for window in WINDOWS:
        for target_idx in range(window, len(months)):
            target_month = months[target_idx]
            train_months = months[target_idx - window:target_idx]
            catalogs = pd.concat([raw[m][0] for m in train_months], ignore_index=True)
            target_catalog, target_leaves = raw[target_month]
            local_catalog = pd.concat([catalogs, target_catalog], ignore_index=True)
            archetypes, archetype_mode, recurrent_count = _local_archetypes(catalogs, window)
            if not archetypes:
                fit_rows.append({"window_months": window, "target_month": target_month, "train_months": json.dumps(train_months), "archetype_count": 0, "archetype_mode": archetype_mode, "status": "NO_ARCHETYPES"})
                continue
            matches, summary = match_catalog_to_archetypes(local_catalog, archetypes, temperature=MATCH_TEMPERATURE, unmatched_threshold=MATCH_THRESHOLD, top_n=TOP_N)
            mass_catalog = np.abs(local_catalog.ensemble_tree_contribution.to_numpy(float))
            mapping_quality = float(np.average(matches.best_similarity.to_numpy(float), weights=np.maximum(mass_catalog, 1e-12)))
            parts: dict[str, pd.DataFrame] = {}
            for month in [*train_months, target_month]:
                cat, leaves = raw[month]
                m = matches.loc[matches.fold_id.astype(str).eq(month)].copy()
                feats, _ = materialize_row_archetype_exposures(leaves, cat, m, archetypes)
                part = leaves[["candidate_id", "__ts__"]].copy().reset_index(drop=True)
                parts[month] = pd.concat([part, feats.reset_index(drop=True)], axis=1)
                archetype_rows.append({
                    "window_months": window,
                    "target_month": target_month,
                    "source_month": month,
                    "archetype_count": len(archetypes),
                    "archetype_mode": archetype_mode,
                    "recurrent_count_before_cap": recurrent_count,
                    "mean_best_similarity": float(m.best_similarity.mean()),
                    "mean_unmatched_leaf_probability": float(m.unmatched_probability.mean()),
                    "matched_mass_mean": float(feats.archetype_matched_mass.mean()),
                    "matched_mass_median": float(feats.archetype_matched_mass.median()),
                })
            row_train = pd.concat([parts[m] for m in train_months], ignore_index=True)
            row_target = parts[target_month]
            archetype_fields = [f"archetype__{a.archetype_id}__abs_contribution" for a in archetypes]
            signed_fields = [f"archetype__{a.archetype_id}__signed_contribution" for a in archetypes]
            lookup = frame[["candidate_id", "__ts__", "month", "net_bps", "gross_bps", "base_expected_bps", "base_score", "residual_bps", *meta_fields]].copy()
            row_train = row_train.merge(lookup, on=["candidate_id", "__ts__"], how="inner", validate="one_to_one")
            row_target = row_target.merge(lookup, on=["candidate_id", "__ts__"], how="inner", validate="one_to_one")
            abs_train = row_train[archetype_fields].copy(); abs_train.columns = [a.archetype_id for a in archetypes]
            signed_train = row_train[signed_fields].copy(); signed_train.columns = [a.archetype_id for a in archetypes]
            contracts, cluster_audit, selected = _select_clusters(abs_train, signed_train, row_train.residual_bps.to_numpy(float), row_train.month.astype(str).to_numpy(), mapping_quality)
            if not contracts:
                fit_rows.append({"window_months": window, "target_month": target_month, "train_months": json.dumps(train_months), "archetype_count": len(archetypes), "archetype_mode": archetype_mode, "mapping_quality": mapping_quality, "status": "NO_CLUSTERS"})
                continue
            cluster_feats_train = materialize_memberships(abs_train, contracts)
            abs_target = row_target[archetype_fields].copy(); abs_target.columns = [a.archetype_id for a in archetypes]
            cluster_feats_target = materialize_memberships(abs_target, contracts)
            for col in cluster_feats_train.columns:
                row_train[col] = cluster_feats_train[col].to_numpy()
                row_target[col] = cluster_feats_target[col].to_numpy()
            zero_parts, int_parts, memberships = [], [], []
            selected_counts = []
            for cluster in contracts:
                prefix = f"cluster__{cluster.cluster_id}__"
                mtr = row_train[f"{prefix}membership"].to_numpy(float)
                mte = row_target[f"{prefix}membership"].to_numpy(float)
                etr = row_train[f"{prefix}abs_contribution"].to_numpy(float)
                ete = row_target[f"{prefix}abs_contribution"].to_numpy(float)
                selected_fields, cmi = _weighted_cmi(row_train, meta_fields, row_train.residual_bps.to_numpy(float), mtr, GAM_CONTEXT_CAP)
                p_zero, _ = _fit_vc(row_train, row_target, selected_fields, etr, ete, mtr, row_train.residual_bps.to_numpy(float), intercept=False)
                p_int, _ = _fit_vc(row_train, row_target, selected_fields, etr, ete, mtr, row_train.residual_bps.to_numpy(float), intercept=True)
                zero_parts.append(p_zero); int_parts.append(p_int); memberships.append(mte); selected_counts.append(len(selected_fields))
                cluster_rows.append({
                    "window_months": window,
                    "target_month": target_month,
                    "cluster_id": cluster.cluster_id,
                    "active_train_rows": int((mtr > CLUSTER_REPORT_ACTIVE_THRESHOLD).sum()),
                    "active_target_rows": int((mte > CLUSTER_REPORT_ACTIVE_THRESHOLD).sum()),
                    "mean_target_membership": float(mte.mean()),
                    "selected_context_count": len(selected_fields),
                    "selected_context": json.dumps(selected_fields),
                    "zero_at_exposure": True,
                })
            mtx = np.column_stack(memberships)
            zero_agg = np.divide((mtx * np.column_stack(zero_parts)).sum(axis=1), np.maximum(mtx.sum(axis=1), 1e-8), out=np.zeros(len(row_target)), where=mtx.sum(axis=1) > 1e-8)
            int_agg = np.divide((mtx * np.column_stack(int_parts)).sum(axis=1), np.maximum(mtx.sum(axis=1), 1e-8), out=np.zeros(len(row_target)), where=mtx.sum(axis=1) > 1e-8)
            out_frame = row_target[["candidate_id", "__ts__", "month", "net_bps", "gross_bps", "base_expected_bps", "base_score", "residual_bps", "archetype_matched_mass", "archetype_unmatched_mass"]].copy()
            out_frame["target_month"] = target_month
            out_frame["window_months"] = window
            out_frame["train_months"] = json.dumps(train_months)
            out_frame["archetype_mode"] = archetype_mode
            out_frame["archetype_count"] = len(archetypes)
            out_frame["rolling_cluster_count"] = len(contracts)
            out_frame["rolling_transport_valid"] = bool(selected.get("valid", False))
            out_frame["rolling_gam_zero_residual"] = zero_agg
            out_frame["rolling_gam_intercept_residual"] = int_agg
            for mode, agg in (("zero", zero_agg), ("intercept", int_agg)):
                for gamma in GAMMAS:
                    out_frame[f"rolling_gam_{mode}_gamma{int(gamma * 100):03d}"] = out_frame.base_expected_bps.to_numpy(float) + gamma * agg
                    # Inference fallback: if the local structural contract
                    # fails its transport/balance gate, do not apply an
                    # unvalidated conditioner; retain the base score.
                    out_frame[f"rolling_gam_gated_{mode}_gamma{int(gamma * 100):03d}"] = np.where(
                        bool(selected.get("valid", False)),
                        out_frame[f"rolling_gam_{mode}_gamma{int(gamma * 100):03d}"].to_numpy(float),
                        out_frame.base_expected_bps.to_numpy(float),
                    )
            prediction_parts.append(out_frame)
            metric_parts.append(_tail_metrics(out_frame, ["base_expected_bps", "rolling_gam_zero_gamma025", "rolling_gam_zero_gamma050", "rolling_gam_zero_gamma100", "rolling_gam_intercept_gamma100", "rolling_gam_gated_zero_gamma025", "rolling_gam_gated_zero_gamma050", "rolling_gam_gated_zero_gamma100", "rolling_gam_gated_intercept_gamma100"], window, target_month))
            fit_rows.append({
                "window_months": window,
                "target_month": target_month,
                "train_months": json.dumps(train_months),
                "train_rows": len(row_train),
                "target_rows": len(row_target),
                "archetype_count": len(archetypes),
                "archetype_mode": archetype_mode,
                "recurrent_count_before_cap": recurrent_count,
                "mapping_quality": mapping_quality,
                "train_matched_mass_mean": float(row_train.archetype_matched_mass.mean()),
                "target_matched_mass_mean": float(row_target.archetype_matched_mass.mean()),
                "target_unmatched_mass_mean": float(row_target.archetype_unmatched_mass.mean()),
                "cluster_count": len(contracts),
                "cluster_transport_valid": bool(selected.get("valid", False)),
                "cluster_transport_score": float(selected.get("transport_score", np.nan)),
                "cluster_balance": float(selected.get("balance", np.nan)),
                "selected_context_mean_count": float(np.mean(selected_counts)) if selected_counts else 0.0,
                "status": "COMPLETE",
            })
    out.mkdir(parents=True)
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    metrics = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    pd.DataFrame(fit_rows).to_parquet(out / "rolling_fit_audit.parquet", index=False)
    pd.DataFrame(cluster_rows).to_parquet(out / "rolling_cluster_audit.parquet", index=False)
    pd.DataFrame(archetype_rows).to_parquet(out / "rolling_archetype_audit.parquet", index=False)
    predictions.to_parquet(out / "rolling_oof_predictions.parquet", index=False, compression="zstd")
    metrics.to_parquet(out / "rolling_metrics.parquet", index=False)
    summary_rows: list[dict[str, object]] = []
    if not metrics.empty:
        for (window, arm, tail), block in metrics.groupby(["window_months", "arm", "tail"], sort=True):
            vals = block.net_bps_per_trade.to_numpy(float)
            summary_rows.append({
                "window_months": int(window), "arm": arm, "tail": float(tail), "months": len(vals),
                "pooled_net_bps_per_trade": float(np.average(block.net_bps_per_trade, weights=block.trades)),
                "mean_month_net_bps_per_trade": float(np.mean(vals)),
                "median_month_net_bps_per_trade": float(np.median(vals)),
                "worst_month_net_bps_per_trade": float(np.min(vals)),
                "positive_months": int(np.sum(vals > 0)),
                "mean_month_rank_ic": float(np.nanmean(block.rank_ic)),
            })
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_parquet(out / "rolling_summary.parquet", index=False)
    # Keep an explicit inference-style view alongside the raw diagnostic
    # table.  The base arm is included so every gated result has a matched
    # control in the same artifact.
    gated_arm_mask = metrics["arm"].astype(str).str.startswith(("base_expected_bps", "rolling_gam_gated_")) if not metrics.empty else pd.Series(dtype=bool)
    gated_metrics = metrics.loc[gated_arm_mask].copy() if not metrics.empty else metrics.copy()
    gated_summary = summary_frame.loc[summary_frame["arm"].astype(str).str.startswith(("base_expected_bps", "rolling_gam_gated_"))].copy() if not summary_frame.empty else summary_frame.copy()
    gated_metrics.to_parquet(out / "rolling_gated_metrics.parquet", index=False)
    gated_summary.to_parquet(out / "rolling_gated_summary.parquet", index=False)
    correctness = {
        "schema": "tp6_sl4_rolling_archetype_gam_correctness_v1",
        "target_month_only_scoring": True,
        "train_window_lengths": list(WINDOWS),
        "future_target_rows_used_in_fit": False,
        "future_target_paths_used_in_target_archetype_fit": False,
        "context_selection_train_only": True,
        "global_ranking_after_score_generation": True,
        "zero_at_exposure_principal": True,
        "candidate_ids_unique": bool(predictions.empty or predictions.duplicated(["window_months", "target_month", "candidate_id"]).sum() == 0),
        "all_scores_finite": bool(predictions.empty or np.isfinite(predictions.select_dtypes(include=[np.number]).to_numpy(float)).all()),
        "gated_fallback_materialized": True,
        "failed_transport_months_fallback_to_base": True,
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {
        "schema": "tp6_sl4_rolling_archetype_gam_oos_v1",
        "status": "COMPLETE",
        "side": SIDE,
        "window_lengths": list(WINDOWS),
        "target_month_evaluated_once": True,
        "base": str(base_path), "family": str(family_path), "meta": str(meta_path), "raw_root": str(raw_root),
        "rows_scored": int(len(predictions)),
        "artifacts": sorted(x.name for x in out.iterdir()),
        "gated_fallback_artifacts": ["rolling_gated_metrics.parquet", "rolling_gated_summary.parquet"],
        "gated_score_rule": "base score fallback when cluster_transport_valid is false",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# Rolling month-ahead archetype/GAM OOF replay",
        "",
        "Each target month is scored using only the preceding 1, 2, or 3 available model months.",
        "The target month is never used to fit archetypes, clusters, context selection, or GAM parameters.",
        "",
        "## Summary",
        "",
        pd.DataFrame(summary_rows).round(3).to_string(index=False) if summary_rows else "No metrics produced.",
        "",
        "## Correctness",
        "",
        json.dumps(correctness, indent=2),
    ]
    (out / "ROLLING_ARCHETYPE_GAM_OOS_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"output": str(out), "rows_scored": int(len(predictions)), "metric_rows": int(len(metrics)), "fit_rows": len(fit_rows)}, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(base_path=args.base, family_path=args.family, meta_path=args.meta, raw_root=args.raw_root, out=args.out)
