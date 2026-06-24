#!/usr/bin/env python3
"""J4/J5 contextual meta capacity and distillation ablation.

Keeps the meta-model contract fixed:

* unchanged `y_bin` label;
* binary log-loss objective for hard-label training;
* one probability output per head.

J4 tests deterministic LightGBM capacity/regularization configurations on the
frozen directional contextual feature arm for each head.  J5 then tests
controlled self-distillation variants only on the exact J4 selected
configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from scripts import run_one_head_contextual_meta_ablation as ctx
from scripts import run_top30_aware_contextual_meta_training_ablation as top30
from scripts.diagnose_meta_recent_failures import (
    _discover_heads,
    _downcast_numeric,
    _feature_store_union,
    _normalise_keys,
    _prepare_model_matrix,
    lgb,
)
from scripts.quantify_bad_regime_archetype_usefulness import _pick_realized_return


DEFAULT_SOURCE_DIR = Path("data_perp/reports/one_head_contextual_meta_ablation_directional_all_heads_20260623")


@dataclass(frozen=True)
class CapacitySpec:
    config_id: str
    regime: str
    max_depth: int
    num_leaves: int
    min_data_in_leaf: int
    min_sum_hessian_in_leaf: float
    min_gain_to_split: float
    lambda_l1: float
    lambda_l2: float
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    learning_rate: float
    n_estimators: int


def _json_default(value: Any) -> Any:
    return ctx._json_default(value)


def _capacity_ladder(max_configs: int) -> list[CapacitySpec]:
    specs = [
        CapacitySpec("conservative_01", "conservative", 2, 4, 1800, 1e-2, 0.020, 0.2, 5.0, 0.70, 0.85, 1, 0.030, 500),
        CapacitySpec("conservative_02", "conservative", 3, 6, 1600, 1e-2, 0.015, 0.1, 4.0, 0.75, 0.85, 1, 0.030, 500),
        CapacitySpec("conservative_03", "conservative", 3, 8, 1400, 5e-3, 0.020, 0.2, 6.0, 0.70, 0.80, 1, 0.025, 650),
        CapacitySpec("conservative_04", "conservative", 2, 4, 2200, 1e-2, 0.010, 0.0, 3.0, 0.85, 0.90, 1, 0.035, 450),
        CapacitySpec("moderate_01", "moderate", 3, 8, 1000, 1e-3, 0.000, 0.1, 1.0, 0.80, 0.85, 1, 0.035, 450),
        CapacitySpec("moderate_02", "moderate", 3, 12, 900, 1e-3, 0.005, 0.1, 2.0, 0.80, 0.85, 1, 0.030, 550),
        CapacitySpec("moderate_03", "moderate", 4, 12, 1000, 1e-3, 0.010, 0.2, 2.5, 0.75, 0.80, 1, 0.030, 600),
        CapacitySpec("moderate_04", "moderate", 3, 10, 1200, 5e-3, 0.010, 0.0, 2.0, 0.90, 0.90, 1, 0.040, 400),
        CapacitySpec("context_flexible_01", "context_flexible", 4, 16, 1200, 1e-3, 0.020, 0.2, 3.0, 0.80, 0.80, 1, 0.030, 650),
        CapacitySpec("context_flexible_02", "context_flexible", 4, 20, 1500, 5e-3, 0.030, 0.3, 4.0, 0.75, 0.80, 1, 0.025, 750),
        CapacitySpec("context_flexible_03", "context_flexible", 5, 24, 1800, 1e-2, 0.040, 0.4, 5.0, 0.70, 0.75, 1, 0.025, 800),
        CapacitySpec("context_flexible_04", "context_flexible", 4, 16, 1600, 1e-2, 0.050, 0.0, 6.0, 0.85, 0.85, 1, 0.035, 550),
    ]
    return specs[: max(0, int(max_configs))]


def _lgb_params(spec: CapacitySpec, seed: int) -> dict[str, Any]:
    return {
        "objective": "binary",
        "n_estimators": int(spec.n_estimators),
        "learning_rate": float(spec.learning_rate),
        "max_depth": int(spec.max_depth),
        "num_leaves": int(spec.num_leaves),
        "min_child_samples": int(spec.min_data_in_leaf),
        "min_sum_hessian_in_leaf": float(spec.min_sum_hessian_in_leaf),
        "min_gain_to_split": float(spec.min_gain_to_split),
        "reg_alpha": float(spec.lambda_l1),
        "reg_lambda": float(spec.lambda_l2),
        "feature_fraction": float(spec.feature_fraction),
        "bagging_fraction": float(spec.bagging_fraction),
        "bagging_freq": int(spec.bagging_freq),
        "random_state": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "data_random_seed": int(seed),
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": max(1, min(6, os.cpu_count() or 2)),
        "verbosity": -1,
    }


def _leaf_split_diagnostics(booster: Any, columns: list[str], context_cols: set[str]) -> dict[str, Any]:
    model = booster.dump_model()
    leaves: list[float] = []
    split_count = 0
    context_split_count = 0
    gain_total = 0.0
    context_gain = 0.0

    def visit(node: dict[str, Any]) -> None:
        nonlocal split_count, context_split_count, gain_total, context_gain
        if "split_feature" not in node and "split_index" not in node:
            leaves.append(float(node.get("leaf_count", np.nan)))
            return
        split_count += 1
        idx = int(node.get("split_feature", -1))
        gain = float(node.get("split_gain", 0.0) or 0.0)
        gain_total += max(gain, 0.0)
        name = columns[idx] if 0 <= idx < len(columns) else ""
        if name in context_cols:
            context_split_count += 1
            context_gain += max(gain, 0.0)
        if "left_child" in node:
            visit(node["left_child"])
        if "right_child" in node:
            visit(node["right_child"])

    for tree in model.get("tree_info", []):
        visit(tree.get("tree_structure", {}))
    arr = np.asarray(leaves, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return {
        "tree_count": int(len(model.get("tree_info", []))),
        "split_count": int(split_count),
        "context_split_count": int(context_split_count),
        "context_split_share": float(context_split_count / split_count) if split_count else 0.0,
        "context_gain_share": float(context_gain / gain_total) if gain_total > 0.0 else 0.0,
        "leaf_count_min": float(np.min(finite)) if finite.size else np.nan,
        "leaf_count_q10": float(np.quantile(finite, 0.10)) if finite.size else np.nan,
        "leaf_count_median": float(np.median(finite)) if finite.size else np.nan,
        "leaf_count_total": int(finite.size),
    }


def _fit_predict_capacity(
    *,
    x: pd.DataFrame,
    y: np.ndarray,
    timestamps: pd.Series,
    folds: list[Any],
    baseline_pred: np.ndarray,
    spec: CapacitySpec,
    seed: int,
    max_train_rows: int,
    context_cols: set[str],
    distillation_variant: str = "hard_label_only",
    teacher: np.ndarray | None = None,
    canonical: pd.DataFrame | None = None,
    distillation_lambda: float = 1.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if lgb is None:
        raise RuntimeError("lightgbm is required")
    x = x.replace([np.inf, -np.inf], np.nan)
    keep_cols = [c for c in x.columns if pd.to_numeric(x[c], errors="coerce").notna().mean() > 0.02]
    if not keep_cols:
        return np.full(len(y), np.nan, dtype=np.float32), [{"reason": "empty_matrix", "feature_count": 0}]
    x_prepared = _prepare_model_matrix(x.loc[:, keep_cols])
    y_float = np.maximum(y, 0).astype(np.float32, copy=False)
    soft_y = y_float
    lambda_i = np.zeros(len(y), dtype=np.float32)
    if distillation_variant != "hard_label_only":
        if teacher is None or canonical is None:
            raise RuntimeError("Distillation requires teacher scores and canonical context")
        lambda_i = ctx._reliability_weights(canonical, distillation_variant, lambda0=float(distillation_lambda))
        teacher_arr = np.clip(np.asarray(teacher, dtype=np.float32), 1e-4, 1.0 - 1e-4)
        soft_y = ((y_float + lambda_i * teacher_arr) / (1.0 + lambda_i)).astype(np.float32, copy=False)
    pred = np.full(len(y), np.nan, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for fold in folds:
        tr = np.asarray(fold.train_idx, dtype=np.int64)
        va = np.asarray(fold.valid_idx, dtype=np.int64)
        tr = tr[(y[tr] >= 0) & np.isfinite(soft_y[tr])]
        va = va[y[va] >= 0]
        if len(tr) < 200 or len(va) < 30 or len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            rows.append({"fold": int(fold.fold_id), "reason": "insufficient_rows_or_classes"})
            continue
        tr_fit = ctx._period_stratified_train_sample(
            timestamps=timestamps.reset_index(drop=True),
            y=np.maximum(y, 0),
            train_idx=tr,
            max_rows=int(max_train_rows),
            seed=int(seed + fold.fold_id * 31),
        )
        params = _lgb_params(spec, seed=int(seed + fold.fold_id * 1009))
        callbacks = [lgb.early_stopping(40, verbose=False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if distillation_variant == "hard_label_only":
                model = lgb.LGBMClassifier(**params)
                model.fit(
                    x_prepared.iloc[tr_fit],
                    y[tr_fit],
                    eval_set=[(x_prepared.iloc[va], y[va])],
                    eval_metric="binary_logloss",
                    callbacks=callbacks,
                )
                fold_pred = model.predict_proba(x_prepared.iloc[va])[:, 1]
            else:
                params["objective"] = "cross_entropy"
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    x_prepared.iloc[tr_fit],
                    soft_y[tr_fit],
                    eval_set=[(x_prepared.iloc[va], y_float[va])],
                    eval_metric="binary_logloss",
                    callbacks=callbacks,
                )
                fold_pred = model.predict(x_prepared.iloc[va])
        pred[va] = np.clip(fold_pred, 1e-6, 1.0 - 1e-6).astype(np.float32, copy=False)
        diag = _leaf_split_diagnostics(model.booster_, keep_cols, context_cols)
        if not np.isfinite(float(diag.get("leaf_count_min", np.nan))):
            fallback_leaf_count = float(len(tr_fit)) if int(diag.get("split_count", 0) or 0) == 0 else 0.0
            diag["leaf_count_min"] = fallback_leaf_count
            diag["leaf_count_q10"] = fallback_leaf_count
            diag["leaf_count_median"] = fallback_leaf_count
        rows.append(
            {
                "fold": int(fold.fold_id),
                "seed": int(seed),
                "config_id": spec.config_id,
                "regime": spec.regime,
                "distillation_variant": distillation_variant,
                "reason": "",
                "train_rows": int(len(tr_fit)),
                "valid_rows": int(len(va)),
                "feature_count": int(len(keep_cols)),
                "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
                "mean_distillation_lambda": float(np.nanmean(lambda_i[tr_fit])) if distillation_variant != "hard_label_only" else 0.0,
                **asdict(spec),
                **diag,
            }
        )
    return pred, rows


def _episode_stats(directional_episode: pd.DataFrame) -> pd.DataFrame:
    if directional_episode.empty:
        return pd.DataFrame()
    bad = directional_episode.loc[directional_episode["period_type"].astype(str).eq("bad_episode")].copy()
    if bad.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (head, arm, variant), group in bad.groupby(["head", "arm", "distillation_variant"], sort=True):
        vals = pd.to_numeric(group["delta_timestamp_weighted_hr_top30"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if vals.size == 0:
            continue
        rows.append(
            {
                "head": head,
                "arm": arm,
                "distillation_variant": variant,
                "episode_count": int(vals.size),
                "episode_median_delta_hr30": float(np.median(vals)),
                "episode_q25_delta_hr30": float(np.quantile(vals, 0.25)),
                "episode_positive_rate_delta_hr30": float(np.mean(vals > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _trial_table(summary: pd.DataFrame, directional: pd.DataFrame, directional_episode: pd.DataFrame, folds: pd.DataFrame, *, hr_tolerance: float, leaf_floor: float) -> pd.DataFrame:
    table = summary.merge(directional, on=["head", "arm", "distillation_variant"], how="left")
    ep = _episode_stats(directional_episode)
    if not ep.empty:
        table = table.merge(ep, on=["head", "arm", "distillation_variant"], how="left")
    leaf = (
        folds.groupby(["head", "arm", "distillation_variant"], as_index=False)
        .agg(
            leaf_count_min=("leaf_count_min", "min"),
            leaf_count_q10=("leaf_count_q10", "median"),
            context_split_count=("context_split_count", "sum"),
            context_split_share=("context_split_share", "mean"),
            context_gain_share=("context_gain_share", "mean"),
        )
    )
    if not leaf.empty:
        table = table.merge(leaf, on=["head", "arm", "distillation_variant"], how="left", suffixes=("", "_fold"))
    episode_count = pd.to_numeric(table.get("episode_count", pd.Series(0, index=table.index)), errors="coerce").fillna(0)
    episode_median = pd.to_numeric(table.get("episode_median_delta_hr30", pd.Series(np.nan, index=table.index)), errors="coerce")
    episode_q25 = pd.to_numeric(table.get("episode_q25_delta_hr30", pd.Series(np.nan, index=table.index)), errors="coerce")
    episode_rate = pd.to_numeric(table.get("episode_positive_rate_delta_hr30", pd.Series(np.nan, index=table.index)), errors="coerce")
    delta_hr30 = pd.to_numeric(table.get("delta_timestamp_weighted_hr_top30", pd.Series(np.nan, index=table.index)), errors="coerce")
    delta_hr10 = pd.to_numeric(table.get("delta_timestamp_weighted_hr_top10", pd.Series(np.nan, index=table.index)), errors="coerce")
    delta_hr20 = pd.to_numeric(table.get("delta_timestamp_weighted_hr_top20", pd.Series(np.nan, index=table.index)), errors="coerce")
    normal = pd.to_numeric(table.get("normal_period_delta_hr_top30", pd.Series(np.nan, index=table.index)), errors="coerce")
    ndcg = pd.to_numeric(table.get("delta_ndcg_top30", pd.Series(np.nan, index=table.index)), errors="coerce")
    leaf_min = pd.to_numeric(table.get("leaf_count_min", pd.Series(np.nan, index=table.index)), errors="coerce")
    table["passes_hard_constraints"] = (
        delta_hr10.fillna(-np.inf).ge(-float(hr_tolerance))
        & delta_hr20.fillna(-np.inf).ge(-float(hr_tolerance))
        & normal.fillna(0.0).ge(-float(hr_tolerance))
        & ndcg.fillna(-np.inf).ge(0.0)
        & leaf_min.fillna(float(leaf_floor)).ge(float(leaf_floor))
    )
    table["episode_selection_delta_hr30"] = episode_median.fillna(delta_hr30)
    table["episode_q25_selection_delta_hr30"] = episode_q25.fillna(delta_hr30)
    table["passes_episode_recurrence"] = episode_count.eq(0) | (episode_median.gt(0.0) & episode_rate.ge(0.75))
    table["trial_promoted"] = table["passes_hard_constraints"] & table["passes_episode_recurrence"] & delta_hr30.gt(0.0)
    return table


def _config_table(trials: pd.DataFrame, *, min_seed_pass_rate: float) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (head, config_id), group in trials.groupby(["head", "config_id"], sort=True):
        promoted = group["trial_promoted"].astype(bool)
        rows.append(
            {
                "head": head,
                "config_id": config_id,
                "regime": str(group["regime"].iloc[0]),
                "seed_count": int(group["seed"].nunique()),
                "seed_pass_rate": float(promoted.mean()),
                "median_episode_delta_hr30": float(pd.to_numeric(group["episode_selection_delta_hr30"], errors="coerce").median()),
                "q25_episode_delta_hr30": float(pd.to_numeric(group["episode_q25_selection_delta_hr30"], errors="coerce").median()),
                "median_delta_hr30": float(pd.to_numeric(group["delta_timestamp_weighted_hr_top30"], errors="coerce").median()),
                "median_delta_ndcg": float(pd.to_numeric(group["delta_ndcg_top30"], errors="coerce").median()),
                "median_net_correct": float(pd.to_numeric(group["net_correct_trades_gained"], errors="coerce").median()),
                "median_delta_hr10": float(pd.to_numeric(group["delta_timestamp_weighted_hr_top10"], errors="coerce").median()),
                "median_delta_hr20": float(pd.to_numeric(group["delta_timestamp_weighted_hr_top20"], errors="coerce").median()),
                "min_leaf_count_min": float(pd.to_numeric(group["leaf_count_min"], errors="coerce").min()),
                "context_split_share_mean": float(pd.to_numeric(group["context_split_share"], errors="coerce").mean()),
                "context_gain_share_mean": float(pd.to_numeric(group["context_gain_share"], errors="coerce").mean()),
                "config_promoted": bool(float(promoted.mean()) >= float(min_seed_pass_rate)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        [
            "head",
            "config_promoted",
            "median_episode_delta_hr30",
            "q25_episode_delta_hr30",
            "median_delta_hr30",
            "median_delta_ndcg",
            "median_net_correct",
        ],
        ascending=[True, False, False, False, False, False, False],
    )


def _select_j4_winners(configs: pd.DataFrame) -> pd.DataFrame:
    if configs.empty:
        return pd.DataFrame()
    return configs.groupby("head", group_keys=False).head(1).reset_index(drop=True)


def _freeze_decisions(freeze: pd.DataFrame, winners: pd.DataFrame, j5: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    winner_by_head = winners.set_index("head") if not winners.empty else pd.DataFrame()
    for _, frozen in freeze.iterrows():
        head = str(frozen["head"])
        selected_arm = str(frozen["selected_feature_arm"])
        if head not in winner_by_head.index:
            rows.append(
                {
                    "head": head,
                    "decision": "retain_selected_contextual_feature_arm_no_j4_winner",
                    "selected_contextual_feature_arm": selected_arm,
                    "selected_capacity_config": "",
                    "selected_distillation_variant": "hard_label_context_arm",
                    "promotion_status": "not_promoted",
                    "fresh_oos_status": "pending_later_labelled_interval",
                }
            )
            continue
        winner = winner_by_head.loc[head]
        promoted = bool(winner.get("config_promoted", False))
        if not promoted:
            rows.append(
                {
                    "head": head,
                    "decision": "retain_selected_contextual_feature_arm_j4_not_promoted",
                    "selected_contextual_feature_arm": selected_arm,
                    "selected_capacity_config": "",
                    "selected_distillation_variant": "hard_label_context_arm",
                    "promotion_status": "not_promoted",
                    "j4_best_config": winner.get("config_id", ""),
                    "j4_best_median_episode_delta_hr30": winner.get("median_episode_delta_hr30", np.nan),
                    "j4_best_seed_pass_rate": winner.get("seed_pass_rate", np.nan),
                    "fresh_oos_status": "pending_later_labelled_interval",
                }
            )
            continue
        head_j5 = j5.loc[j5["head"].astype(str).eq(head)].copy() if not j5.empty else pd.DataFrame()
        selected_variant = "hard_label_only"
        selected_j5_arm = ""
        if not head_j5.empty:
            candidates = head_j5.loc[head_j5.get("j5_beats_hard_label", pd.Series(False, index=head_j5.index)).astype(bool)]
            if not candidates.empty:
                candidates = candidates.sort_values(
                    [
                        "episode_selection_delta_hr30",
                        "delta_timestamp_weighted_hr_top30",
                        "delta_ndcg_top30",
                        "delta_timestamp_weighted_hr_top10",
                        "delta_timestamp_weighted_hr_top20",
                    ],
                    ascending=[False, False, False, False, False],
                )
                selected_variant = str(candidates.iloc[0]["distillation_variant"])
                selected_j5_arm = str(candidates.iloc[0]["arm"])
        rows.append(
            {
                "head": head,
                "decision": "promote_j4_capacity_pending_fresh_oos",
                "selected_contextual_feature_arm": selected_arm,
                "selected_capacity_config": winner.get("config_id", ""),
                "selected_distillation_variant": selected_variant,
                "selected_j5_arm": selected_j5_arm,
                "promotion_status": "development_promoted_pending_fresh_oos",
                "j4_best_median_episode_delta_hr30": winner.get("median_episode_delta_hr30", np.nan),
                "j4_best_seed_pass_rate": winner.get("seed_pass_rate", np.nan),
                "fresh_oos_status": "pending_later_labelled_interval",
            }
        )
    return pd.DataFrame(rows)


def _write_report(out_dir: Path, audit: dict[str, Any], configs: pd.DataFrame, j5: pd.DataFrame, freeze_decisions: pd.DataFrame) -> None:
    lines = [
        "# J4/J5 Contextual Meta Capacity Ablation",
        "",
        "J4 uses ordinary BCE weights and deterministic LightGBM capacity/regularization trials.",
        "J5 evaluates distillation variants only on the selected J4 configuration.",
        "",
        "## Requirement Audit",
        "",
        pd.DataFrame(audit.get("items", [])).to_markdown(index=False),
        "",
    ]
    if not configs.empty:
        cols = [
            "head",
            "config_id",
            "regime",
            "config_promoted",
            "seed_pass_rate",
            "median_episode_delta_hr30",
            "q25_episode_delta_hr30",
            "median_delta_hr30",
            "median_delta_ndcg",
            "median_net_correct",
            "median_delta_hr10",
            "median_delta_hr20",
            "min_leaf_count_min",
            "context_split_share_mean",
        ]
        lines.extend(["## J4 Config Summary", "", configs[[c for c in cols if c in configs.columns]].to_markdown(index=False, floatfmt=".6f"), ""])
    if not j5.empty:
        cols = [
            "head",
            "arm",
            "distillation_variant",
            "j5_beats_hard_label",
            "delta_timestamp_weighted_hr_top30",
            "episode_median_delta_hr30",
            "delta_ndcg_top30",
            "delta_timestamp_weighted_hr_top10",
            "delta_timestamp_weighted_hr_top20",
        ]
        lines.extend(["## J5 Distillation Summary", "", j5[[c for c in cols if c in j5.columns]].to_markdown(index=False, floatfmt=".6f"), ""])
    if not freeze_decisions.empty:
        lines.extend(["## Freeze Decisions", "", freeze_decisions.to_markdown(index=False, floatfmt=".6f"), ""])
    (out_dir / "j4_j5_contextual_meta_capacity_report.md").write_text("\n".join(lines))


def _audit_outputs(
    summary: pd.DataFrame,
    trials: pd.DataFrame,
    configs: pd.DataFrame,
    folds: pd.DataFrame,
    j5: pd.DataFrame,
    freeze_decisions: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    items = []
    items.append(
        {
            "requirement": "j4_capacity_trials_present",
            "status": "passed" if not summary.empty and not trials.empty and not configs.empty else "failed",
            "metrics": {"summary_rows": int(len(summary)), "trial_rows": int(len(trials)), "config_rows": int(len(configs))},
        }
    )
    items.append(
        {
            "requirement": "ordinary_bce_weights_for_j4",
            "status": "passed"
            if not summary.empty and set(summary.get("training_target", pd.Series(dtype=str)).astype(str)) == {"y_bin"} and not summary.get("sample_weight_used", pd.Series([True])).astype(bool).any()
            else "failed",
            "metrics": {"sample_weight_used_values": sorted(set(summary.get("sample_weight_used", pd.Series(dtype=bool)).astype(str)))},
        }
    )
    items.append(
        {
            "requirement": "leaf_and_context_split_diagnostics_persisted",
            "status": "passed"
            if not folds.empty and {"leaf_count_min", "context_split_count", "context_gain_share"} <= set(folds.columns)
            else "failed",
            "metrics": {"fold_rows": int(len(folds)), "columns": sorted(set(folds.columns) & {"leaf_count_min", "context_split_count", "context_gain_share"})},
        }
    )
    items.append(
        {
            "requirement": "j5_runs_only_on_j4_winners",
            "status": "passed"
            if bool(args.skip_j5)
            or not j5.empty
            or not configs.get("config_promoted", pd.Series(dtype=bool)).astype(bool).any()
            else "failed",
            "metrics": {
                "skip_j5": bool(args.skip_j5),
                "j5_rows": int(len(j5)),
                "promoted_j4_configs": int(configs.get("config_promoted", pd.Series(dtype=bool)).astype(bool).sum()) if not configs.empty else 0,
            },
        }
    )
    items.append(
        {
            "requirement": "freeze_decision_table_present",
            "status": "passed"
            if not freeze_decisions.empty
            and {"head", "decision", "selected_contextual_feature_arm", "selected_distillation_variant", "fresh_oos_status"} <= set(freeze_decisions.columns)
            else "failed",
            "metrics": {
                "rows": int(len(freeze_decisions)),
                "decisions": sorted(set(freeze_decisions.get("decision", pd.Series(dtype=str)).astype(str))) if not freeze_decisions.empty else [],
            },
        }
    )
    statuses = [item["status"] for item in items]
    return {"status": "passed" if all(s == "passed" for s in statuses) else "failed", "items": items}


def run(args: argparse.Namespace) -> Path:
    if lgb is None:
        raise RuntimeError("lightgbm is required")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_directional_dir)
    freeze = top30._load_frozen_feature_arms(source_dir)
    if args.only_head:
        freeze = freeze.loc[freeze["head"].astype(str).isin(set(str(x) for x in args.only_head))].reset_index(drop=True)
    freeze.to_csv(out_dir / "j4_j5_contextual_meta_feature_arm_freeze.csv", index=False)
    selected_by_head = dict(zip(freeze["head"].astype(str), freeze["selected_feature_arm"].astype(str)))
    meta_artifact_dir = Path(args.meta_artifact_dir)
    baseline_artifact_dir = Path(args.baseline_artifact_dir)
    feature_dir = Path(args.feature_dir)
    report_dir = Path(args.report_dir)
    transform_cache = Path(args.transform_cache) if args.transform_cache else None
    regime_context = Path(args.regime_context) if args.regime_context else None
    canonical_defs = ctx._load_canonical_definitions(Path(args.canonical_reduction))
    meta_state = joblib.load(meta_artifact_dir / "models" / "model_state_meta.pkl")
    meta_models = meta_state["bundle"]["meta_models"]
    heads = [h for h in _discover_heads(meta_artifact_dir, report_dir, meta_models) if h.head in selected_by_head]
    with (baseline_artifact_dir / "base_models_intermediate.pkl").open("rb") as fh:
        base_bundle = pickle.load(fh)
    symbol_columns = _feature_store_union(feature_dir)
    capacity_specs = _capacity_ladder(int(args.max_j4_configs))
    seeds = [int(x) for x in args.j4_seeds]

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    directional_rows: list[dict[str, Any]] = []
    directional_timestamp_rows: list[dict[str, Any]] = []
    directional_episode_rows: list[dict[str, Any]] = []
    j5_summary_rows: list[dict[str, Any]] = []
    head_cache: dict[str, dict[str, Any]] = {}

    for head in heads:
        print(f"[j4_j5_capacity] processing head={head.head}", flush=True)
        panel = _normalise_keys(pd.read_parquet(head.meta_oof_path))
        panel = _downcast_numeric(panel, exclude=["timestamp", "symbol"])
        if int(args.max_rows) > 0 and len(panel) > int(args.max_rows):
            keep = np.linspace(0, len(panel) - 1, int(args.max_rows)).round().astype(int)
            panel = panel.iloc[np.unique(keep)].sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
        race = meta_models[head.meta_key]
        current_x, raw = ctx._assemble_head_context(
            head=head,
            panel=panel,
            race=race,
            base_bundle=base_bundle,
            feature_dir=feature_dir,
            transform_cache=transform_cache,
            symbol_columns=symbol_columns,
            regime_context=regime_context,
            max_regime_columns=int(args.max_regime_columns),
        )
        y = ctx._meta_target(panel)
        baseline_pred = ctx._current_meta_score(panel)
        returns = _pick_realized_return(panel).to_numpy(dtype=np.float32, copy=False)
        folds = ctx._make_chrono_folds(panel["timestamp"], int(args.outer_folds), embargo_hours=int(args.embargo_hours))
        canonical, _diag = ctx._fold_canonical_features(
            raw,
            folds,
            canonical_defs,
            trailing_window=int(args.trailing_window),
            min_periods=int(args.min_periods),
            min_resolved_features=int(args.min_resolved_features),
        )
        canonical["leaf_occupancy_novelty"] = ctx._fit_leaf_occupancy_novelty(panel, folds).reset_index(drop=True)
        canonical = canonical.loc[:, list(ctx.CANONICAL_CONTEXT)]
        canonical.to_parquet(out_dir / f"{head.head}_j4_j5_fold_fitted_context.parquet", index=False)
        arms = ctx._arm_frames(panel, current_x.reset_index(drop=True), canonical)
        selected_arm = selected_by_head[head.head]
        x_selected = arms[selected_arm]
        context_cols = set(x_selected.columns) - set(current_x.columns)
        bad_episodes, _bad_meta = ctx._load_episode_registry(args.episode_registry, head=head.head, target_name=args.target_name)
        head_cache[head.head] = {
            "panel": panel,
            "x": x_selected,
            "y": y,
            "baseline_pred": baseline_pred,
            "returns": returns,
            "folds": folds,
            "canonical": canonical,
            "context_cols": context_cols,
            "bad_episodes": bad_episodes,
            "selected_arm": selected_arm,
        }
        for spec in capacity_specs:
            for seed in seeds:
                arm = f"J4_{spec.config_id}_seed{seed}"
                pred, rows = _fit_predict_capacity(
                    x=x_selected,
                    y=y,
                    timestamps=panel["timestamp"].reset_index(drop=True),
                    folds=folds,
                    baseline_pred=baseline_pred,
                    spec=spec,
                    seed=int(seed),
                    max_train_rows=int(args.max_train_rows),
                    context_cols=context_cols,
                )
                for row in rows:
                    fold_rows.append(
                        {
                            **row,
                            "head": head.head,
                            "arm": arm,
                            "distillation_variant": "j4_hard_label_capacity",
                            "selected_feature_arm": selected_arm,
                            "sample_weight_used": False,
                        }
                    )
                summary = ctx._overall_metrics(
                    head=head.head,
                    arm=arm,
                    variant="j4_hard_label_capacity",
                    y=y,
                    pred=pred,
                    baseline_pred=baseline_pred,
                    returns=returns,
                )
                summary.update(
                    {
                        "training_target": "y_bin",
                        "sample_weight_used": False,
                        "single_output_score": True,
                        "config_id": spec.config_id,
                        "seed": int(seed),
                        "selected_feature_arm": selected_arm,
                        **asdict(spec),
                    }
                )
                summary_rows.append(summary)
                ts_metrics = ctx._directional_timestamp_metrics(
                    head=head.head,
                    arm=arm,
                    variant="j4_hard_label_capacity",
                    panel=panel,
                    y=y,
                    pred=pred,
                    baseline_pred=baseline_pred,
                    returns=returns,
                    bad_episodes=bad_episodes,
                    rank_threshold=float(args.rank_threshold),
                    min_timestamp_rows=int(args.directional_min_timestamp_rows),
                )
                if not ts_metrics.empty:
                    directional_timestamp_rows.extend(ts_metrics.to_dict(orient="records"))
                    directional_rows.extend(ctx._directional_aggregate(ts_metrics).to_dict(orient="records"))
                    ep = ctx._directional_episode_metrics(ts_metrics, bad_episodes)
                    if not ep.empty:
                        directional_episode_rows.extend(ep.to_dict(orient="records"))

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)
    directional_df = pd.DataFrame(directional_rows)
    directional_timestamp_df = pd.DataFrame(directional_timestamp_rows)
    directional_episode_df = pd.DataFrame(directional_episode_rows)
    trial_df = _trial_table(summary_df, directional_df, directional_episode_df, fold_df, hr_tolerance=float(args.directional_hr_tolerance), leaf_floor=float(args.leaf_count_floor))
    config_df = _config_table(trial_df, min_seed_pass_rate=float(args.min_seed_pass_rate))
    winners = _select_j4_winners(config_df)

    j5_winners = winners.loc[winners.get("config_promoted", pd.Series(False, index=winners.index)).astype(bool)].copy() if not winners.empty else pd.DataFrame()
    if bool(args.j5_on_unpromoted_winner) and not winners.empty:
        j5_winners = winners.copy()
    if not bool(args.skip_j5) and not j5_winners.empty:
        for _, winner in j5_winners.iterrows():
            head_name = str(winner["head"])
            cache = head_cache[head_name]
            spec = next(s for s in capacity_specs if s.config_id == str(winner["config_id"]))
            seed = int(args.j5_seed)
            for variant in ctx.DISTILLATION_VARIANTS:
                arm = f"J5_{spec.config_id}_{variant}"
                pred, rows = _fit_predict_capacity(
                    x=cache["x"],
                    y=cache["y"],
                    timestamps=cache["panel"]["timestamp"].reset_index(drop=True),
                    folds=cache["folds"],
                    baseline_pred=cache["baseline_pred"],
                    spec=spec,
                    seed=seed,
                    max_train_rows=int(args.max_train_rows),
                    context_cols=cache["context_cols"],
                    distillation_variant=variant,
                    teacher=cache["baseline_pred"],
                    canonical=cache["canonical"],
                    distillation_lambda=float(args.distillation_lambda),
                )
                for row in rows:
                    fold_rows.append({"head": head_name, "arm": arm, "selected_feature_arm": cache["selected_arm"], "sample_weight_used": False, **row})
                summary = ctx._overall_metrics(
                    head=head_name,
                    arm=arm,
                    variant=variant,
                    y=cache["y"],
                    pred=pred,
                    baseline_pred=cache["baseline_pred"],
                    returns=cache["returns"],
                )
                summary.update({"training_target": "y_bin", "sample_weight_used": False, "config_id": spec.config_id, "seed": seed})
                j5_summary_rows.append(summary)
                ts_metrics = ctx._directional_timestamp_metrics(
                    head=head_name,
                    arm=arm,
                    variant=variant,
                    panel=cache["panel"],
                    y=cache["y"],
                    pred=pred,
                    baseline_pred=cache["baseline_pred"],
                    returns=cache["returns"],
                    bad_episodes=cache["bad_episodes"],
                    rank_threshold=float(args.rank_threshold),
                    min_timestamp_rows=int(args.directional_min_timestamp_rows),
                )
                if not ts_metrics.empty:
                    directional_timestamp_rows.extend(ts_metrics.to_dict(orient="records"))
                    directional_rows.extend(ctx._directional_aggregate(ts_metrics).to_dict(orient="records"))
                    ep = ctx._directional_episode_metrics(ts_metrics, cache["bad_episodes"])
                    if not ep.empty:
                        directional_episode_rows.extend(ep.to_dict(orient="records"))

    all_summary_df = pd.concat([summary_df, pd.DataFrame(j5_summary_rows)], ignore_index=True) if j5_summary_rows else summary_df
    fold_df = pd.DataFrame(fold_rows)
    directional_df = pd.DataFrame(directional_rows)
    directional_timestamp_df = pd.DataFrame(directional_timestamp_rows)
    directional_episode_df = pd.DataFrame(directional_episode_rows)
    all_trial_df = _trial_table(all_summary_df, directional_df, directional_episode_df, fold_df, hr_tolerance=float(args.directional_hr_tolerance), leaf_floor=float(args.leaf_count_floor))
    j5_df = all_trial_df.loc[all_trial_df["arm"].astype(str).str.startswith("J5_")].copy()
    if not j5_df.empty:
        hard = j5_df.loc[j5_df["distillation_variant"].astype(str).eq("hard_label_only")].set_index("head")
        j5_df["j5_beats_hard_label"] = False
        for idx, row in j5_df.iterrows():
            head_name = str(row["head"])
            if head_name not in hard.index or row["distillation_variant"] == "hard_label_only":
                continue
            base = hard.loc[head_name]
            j5_df.loc[idx, "j5_beats_hard_label"] = bool(
                float(row.get("episode_selection_delta_hr30", -np.inf)) >= float(base.get("episode_selection_delta_hr30", -np.inf))
                and float(row.get("delta_timestamp_weighted_hr_top30", -np.inf)) >= float(base.get("delta_timestamp_weighted_hr_top30", -np.inf))
                and float(row.get("delta_ndcg_top30", -np.inf)) >= float(base.get("delta_ndcg_top30", -np.inf))
                and float(row.get("delta_timestamp_weighted_hr_top10", -np.inf)) >= float(base.get("delta_timestamp_weighted_hr_top10", -np.inf))
                and float(row.get("delta_timestamp_weighted_hr_top20", -np.inf)) >= float(base.get("delta_timestamp_weighted_hr_top20", -np.inf))
            )

    freeze_decision_df = _freeze_decisions(freeze, winners, j5_df)
    audit = _audit_outputs(all_summary_df, all_trial_df, config_df, fold_df, j5_df, freeze_decision_df, args)
    freeze.to_csv(out_dir / "j4_j5_contextual_meta_feature_arm_freeze.csv", index=False)
    all_summary_df.to_csv(out_dir / "j4_j5_contextual_meta_summary.csv", index=False)
    fold_df.to_csv(out_dir / "j4_j5_contextual_meta_fold_leaf_diagnostics.csv", index=False)
    directional_df.to_csv(out_dir / "j4_j5_contextual_meta_directional_metrics.csv", index=False)
    directional_timestamp_df.to_csv(out_dir / "j4_j5_contextual_meta_directional_timestamp_metrics.csv", index=False)
    directional_episode_df.to_csv(out_dir / "j4_j5_contextual_meta_directional_episode_metrics.csv", index=False)
    all_trial_df.to_csv(out_dir / "j4_j5_contextual_meta_trial_table.csv", index=False)
    config_df.to_csv(out_dir / "j4_contextual_meta_config_summary.csv", index=False)
    winners.to_csv(out_dir / "j4_contextual_meta_winners.csv", index=False)
    j5_df.to_csv(out_dir / "j5_contextual_meta_distillation_summary.csv", index=False)
    freeze_decision_df.to_csv(out_dir / "j4_j5_contextual_meta_freeze_decisions.csv", index=False)
    (out_dir / "j4_j5_contextual_meta_requirement_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, default=_json_default))
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=_json_default))
    _write_report(out_dir, audit, config_df, j5_df, freeze_decision_df)
    print(f"[j4_j5_capacity] wrote results to {out_dir}", flush=True)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directional-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--meta-artifact-dir", default="data_perp/artifacts/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--baseline-artifact-dir", default="data_perp/artifacts/20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--report-dir", default="data_perp/reports/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--feature-dir", default="data_perp/features/20260605_070000")
    parser.add_argument("--transform-cache", default="data_perp/reports/performance_regime_break_transform_cache/generated_transforms_single_3f7f9c53eaaa98ce632760a976691f24.parquet")
    parser.add_argument("--canonical-reduction", default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_usefulness_multitarget_clean_contract_v1/canonical_archetype_reduction.csv")
    parser.add_argument("--regime-context", default="")
    parser.add_argument("--output-dir", default="data_perp/reports/j4_j5_contextual_meta_capacity_ablation_20260623")
    parser.add_argument("--episode-registry", default=str(ctx.DEFAULT_EPISODE_REGISTRY))
    parser.add_argument("--target-name", default="y_bin")
    parser.add_argument("--rank-threshold", type=float, default=0.70)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--embargo-hours", type=int, default=24)
    parser.add_argument("--trailing-window", type=int, default=24 * 28)
    parser.add_argument("--min-periods", type=int, default=24 * 7)
    parser.add_argument("--min-resolved-features", type=int, default=2)
    parser.add_argument("--max-regime-columns", type=int, default=80)
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-j4-configs", type=int, default=12)
    parser.add_argument("--j4-seeds", nargs="+", type=int, default=[29, 31, 37])
    parser.add_argument("--min-seed-pass-rate", type=float, default=2.0 / 3.0)
    parser.add_argument("--leaf-count-floor", type=float, default=50.0)
    parser.add_argument("--directional-min-timestamp-rows", type=int, default=5)
    parser.add_argument("--directional-hr-tolerance", type=float, default=0.001)
    parser.add_argument("--skip-j5", action="store_true")
    parser.add_argument("--j5-on-unpromoted-winner", action="store_true")
    parser.add_argument("--j5-seed", type=int, default=41)
    parser.add_argument("--distillation-lambda", type=float, default=1.0)
    parser.add_argument("--only-head", nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
