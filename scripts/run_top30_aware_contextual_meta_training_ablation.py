#!/usr/bin/env python3
"""Top-30-aware contextual meta training ablation.

This experiment keeps the existing meta contract:

* unchanged `y_bin` label;
* one binary LightGBM meta head;
* binary log-loss objective;
* one probability output.

Only row weights and model-selection metrics change.  Hard examples are defined
from cross-fitted reference ranks inside each outer training fold, so validation
rows never influence their own training weights.
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
class WeightSpec:
    arm: str
    description: str
    timestamp_weight: bool
    tail_weight: bool = False
    swap_weight: bool = False
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    tau: float = 0.05

    @property
    def needs_reference_rank(self) -> bool:
        return bool(self.tail_weight or self.swap_weight)


def _json_default(value: Any) -> Any:
    return ctx._json_default(value)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -50.0, 50.0)))


def _timestamp_reference_rank(timestamps: pd.Series, score: np.ndarray) -> np.ndarray:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    score_arr = np.asarray(score, dtype=np.float64)
    ranks = np.full(len(score_arr), np.nan, dtype=np.float64)
    frame = pd.DataFrame({"timestamp": ts, "score": score_arr})
    for _, idx in frame.groupby("timestamp", sort=False).groups.items():
        ids = np.asarray(list(idx), dtype=np.int64)
        vals = score_arr[ids]
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        valid_ids = ids[finite]
        if len(valid_ids) == 1:
            ranks[valid_ids] = 1.0
            continue
        local_rank = pd.Series(vals[finite]).rank(method="first").to_numpy(dtype=np.float64)
        ranks[valid_ids] = (local_rank - 1.0) / max(float(len(valid_ids) - 1), 1.0)
    return np.clip(ranks, 0.0, 1.0)


def _normalise_equal_timestamp_mass(raw_weight: np.ndarray, timestamps: pd.Series) -> np.ndarray:
    raw = np.asarray(raw_weight, dtype=np.float64)
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 1.0)
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    weight = np.zeros(len(raw), dtype=np.float64)
    frame = pd.DataFrame({"timestamp": ts, "raw": raw})
    for _, idx in frame.groupby("timestamp", sort=False).groups.items():
        ids = np.asarray(list(idx), dtype=np.int64)
        denom = float(np.sum(raw[ids]))
        if denom <= 0.0 or not np.isfinite(denom):
            weight[ids] = 1.0 / max(len(ids), 1)
        else:
            weight[ids] = raw[ids] / denom
    total = float(np.sum(weight))
    if total > 0.0 and np.isfinite(total):
        weight *= float(len(weight)) / total
    return np.clip(weight, 1e-6, 100.0).astype(np.float32, copy=False)


def _build_top30_weights(
    *,
    timestamps: pd.Series,
    y: np.ndarray,
    ref_score: np.ndarray,
    spec: WeightSpec,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not spec.timestamp_weight and not spec.tail_weight and not spec.swap_weight:
        return None, {"weight_mode": "current_unweighted"}
    labels = np.asarray(y, dtype=np.int8)
    ranks = _timestamp_reference_rank(timestamps, ref_score)
    raw = np.ones(len(labels), dtype=np.float64)
    if spec.tail_weight:
        raw *= 1.0 + float(spec.alpha) * _sigmoid((ranks - 0.70) / max(float(spec.tau), 1e-6))
    if spec.swap_weight:
        false_positive_top = (labels == 0) & (ranks >= 0.70)
        missed_positive_boundary = (labels == 1) & (ranks >= 0.50) & (ranks < 0.70)
        raw[false_positive_top] *= 1.0 + float(spec.beta)
        raw[missed_positive_boundary] *= 1.0 + float(spec.gamma)
    if spec.timestamp_weight:
        weight = _normalise_equal_timestamp_mass(raw, timestamps)
    else:
        weight = raw.astype(np.float32, copy=False)
        weight *= float(len(weight)) / max(float(np.sum(weight)), 1e-12)
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    timestamp_mass = pd.Series(weight).groupby(ts).sum()
    diag = {
        "weight_mode": spec.arm,
        "weight_mean": float(np.nanmean(weight)),
        "weight_std": float(np.nanstd(weight)),
        "weight_max": float(np.nanmax(weight)),
        "reference_rank_mean": float(np.nanmean(ranks)),
        "reference_rank_top30_share": float(np.nanmean(ranks >= 0.70)),
        "swap_false_positive_top30_rows": int(np.sum((labels == 0) & (ranks >= 0.70))),
        "swap_missed_positive_boundary_rows": int(np.sum((labels == 1) & (ranks >= 0.50) & (ranks < 0.70))),
        "timestamp_mass_min": float(timestamp_mass.min()) if len(timestamp_mass) else np.nan,
        "timestamp_mass_max": float(timestamp_mass.max()) if len(timestamp_mass) else np.nan,
        "timestamp_mass_cv": float(timestamp_mass.std() / timestamp_mass.mean())
        if len(timestamp_mass) and timestamp_mass.mean() > 0
        else np.nan,
    }
    return weight, diag


def _lgbm_params(seed: int, *, max_depth: int = 3, min_child_fraction: float = 0.025) -> dict[str, Any]:
    return {
        "objective": "binary",
        "n_estimators": 350,
        "learning_rate": 0.035,
        "max_depth": int(max_depth),
        "num_leaves": max(4, min(16, 2 ** int(max_depth))),
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": int(seed),
        "n_jobs": max(1, min(6, os.cpu_count() or 2)),
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "data_random_seed": int(seed),
        "min_child_fraction": float(min_child_fraction),
    }


def _fit_fold(
    *,
    x_prepared: pd.DataFrame,
    y: np.ndarray,
    timestamps: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    sample_weight: np.ndarray | None,
    max_train_rows: int,
    seed: int,
    max_depth: int,
    min_child_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if lgb is None:
        raise RuntimeError("lightgbm is required")
    tr = np.asarray(train_idx, dtype=np.int64)
    va = np.asarray(valid_idx, dtype=np.int64)
    tr = tr[y[tr] >= 0]
    va = va[y[va] >= 0]
    if len(tr) < 200 or len(va) < 30 or len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
        return np.full(len(va), np.nan, dtype=np.float32), {
            "reason": "insufficient_rows_or_classes",
            "train_rows": int(len(tr)),
            "valid_rows": int(len(va)),
        }
    tr_fit = ctx._period_stratified_train_sample(
        timestamps=timestamps.reset_index(drop=True),
        y=np.maximum(y, 0),
        train_idx=tr,
        max_rows=int(max_train_rows),
        seed=int(seed),
    )
    fit_weight = None
    if sample_weight is not None:
        fit_weight = np.asarray(sample_weight, dtype=np.float32)[tr_fit]
        fit_weight = np.where(np.isfinite(fit_weight) & (fit_weight > 0.0), fit_weight, 1.0)
    params = _lgbm_params(seed, max_depth=max_depth, min_child_fraction=min_child_fraction)
    min_child = max(50, int(math.ceil(float(params.pop("min_child_fraction")) * len(tr_fit))))
    clf = lgb.LGBMClassifier(**params, min_child_samples=min_child)
    callbacks = [lgb.early_stopping(40, verbose=False)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(
            x_prepared.iloc[tr_fit],
            y[tr_fit],
            sample_weight=fit_weight,
            eval_set=[(x_prepared.iloc[va], y[va])],
            eval_metric="binary_logloss",
            callbacks=callbacks,
        )
    pred = clf.predict_proba(x_prepared.iloc[va])[:, 1].astype(np.float32, copy=False)
    return pred, {
        "reason": "",
        "train_rows": int(len(tr_fit)),
        "valid_rows": int(len(va)),
        "best_iteration": int(getattr(clf, "best_iteration_", 0) or 0),
        "min_child_samples": int(min_child),
        "weight_mean": float(np.nanmean(fit_weight)) if fit_weight is not None else 1.0,
        "weight_max": float(np.nanmax(fit_weight)) if fit_weight is not None else 1.0,
        "max_depth": int(max_depth),
        "feature_count": int(x_prepared.shape[1]),
    }


def _inner_oof_reference_scores(
    *,
    x_prepared: pd.DataFrame,
    y: np.ndarray,
    timestamps: pd.Series,
    outer_train_idx: np.ndarray,
    baseline_pred: np.ndarray,
    inner_folds: int,
    embargo_hours: int,
    max_train_rows: int,
    seed: int,
    max_depth: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_idx = np.asarray(outer_train_idx, dtype=np.int64)
    ref = np.asarray(baseline_pred, dtype=np.float32)[train_idx].copy()
    if len(train_idx) < 500 or int(inner_folds) < 2:
        return ref, {"reference_source": "baseline_fallback", "inner_valid_rows": 0}
    inner = ctx._make_chrono_folds(timestamps.iloc[train_idx].reset_index(drop=True), int(inner_folds), embargo_hours=int(embargo_hours))
    inner_pred = np.full(len(train_idx), np.nan, dtype=np.float32)
    valid_rows = 0
    for fold in inner:
        tr_abs = train_idx[np.asarray(fold.train_idx, dtype=np.int64)]
        va_rel = np.asarray(fold.valid_idx, dtype=np.int64)
        va_abs = train_idx[va_rel]
        pred, _info = _fit_fold(
            x_prepared=x_prepared,
            y=y,
            timestamps=timestamps,
            train_idx=tr_abs,
            valid_idx=va_abs,
            sample_weight=None,
            max_train_rows=int(max_train_rows),
            seed=int(seed + fold.fold_id * 1009),
            max_depth=int(max_depth),
            min_child_fraction=0.025,
        )
        if len(pred) == len(va_abs):
            inner_pred[va_rel] = pred
            valid_rows += int(np.isfinite(pred).sum())
    mask = np.isfinite(inner_pred)
    ref[mask] = inner_pred[mask]
    return ref, {
        "reference_source": "inner_oof_contextual_with_baseline_fallback",
        "inner_valid_rows": int(valid_rows),
        "inner_coverage": float(np.mean(mask)) if len(mask) else 0.0,
    }


def _fit_predict_weighted_classifier(
    *,
    x: pd.DataFrame,
    y: np.ndarray,
    timestamps: pd.Series,
    folds: list[Any],
    baseline_pred: np.ndarray,
    spec: WeightSpec,
    seed: int,
    max_train_rows: int,
    inner_folds: int,
    embargo_hours: int,
    max_depth: int = 3,
    min_child_fraction: float = 0.025,
    reference_cache: dict[tuple[int, int, int, int, int], tuple[np.ndarray, dict[str, Any]]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    x = x.replace([np.inf, -np.inf], np.nan)
    keep_cols = [c for c in x.columns if pd.to_numeric(x[c], errors="coerce").notna().mean() > 0.02]
    if not keep_cols:
        return np.full(len(y), np.nan, dtype=np.float32), [{"reason": "empty_matrix", "feature_count": 0}], []
    x_prepared = _prepare_model_matrix(x.loc[:, keep_cols])
    pred = np.full(len(y), np.nan, dtype=np.float32)
    fit_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for fold in folds:
        tr = np.asarray(fold.train_idx, dtype=np.int64)
        va = np.asarray(fold.valid_idx, dtype=np.int64)
        tr = tr[y[tr] >= 0]
        va = va[y[va] >= 0]
        ref_diag: dict[str, Any] = {"reference_source": "not_required", "inner_valid_rows": 0, "inner_coverage": np.nan}
        ref_scores = np.asarray(baseline_pred, dtype=np.float32)[tr]
        if spec.needs_reference_rank:
            cache_key = (
                int(fold.fold_id),
                int(max_depth),
                int(inner_folds),
                int(embargo_hours),
                int(max_train_rows),
            )
            if reference_cache is not None and cache_key in reference_cache:
                cached_score, cached_diag = reference_cache[cache_key]
                ref_scores = cached_score.copy()
                ref_diag = {**cached_diag, "reference_cache_hit": True}
            else:
                ref_scores, ref_diag = _inner_oof_reference_scores(
                    x_prepared=x_prepared,
                    y=y,
                    timestamps=timestamps,
                    outer_train_idx=tr,
                    baseline_pred=baseline_pred,
                    inner_folds=int(inner_folds),
                    embargo_hours=int(embargo_hours),
                    max_train_rows=int(max_train_rows),
                    seed=int(seed + fold.fold_id * 10_007),
                    max_depth=int(max_depth),
                )
                ref_diag = {**ref_diag, "reference_cache_hit": False}
                if reference_cache is not None:
                    reference_cache[cache_key] = (ref_scores.copy(), ref_diag.copy())
        train_weight = np.full(len(y), np.nan, dtype=np.float32)
        local_weight, weight_diag = _build_top30_weights(
            timestamps=timestamps.iloc[tr].reset_index(drop=True),
            y=y[tr],
            ref_score=ref_scores,
            spec=spec,
        )
        if local_weight is not None:
            train_weight[tr] = local_weight
        fold_pred, fit_info = _fit_fold(
            x_prepared=x_prepared,
            y=y,
            timestamps=timestamps,
            train_idx=tr,
            valid_idx=va,
            sample_weight=train_weight if local_weight is not None else None,
            max_train_rows=int(max_train_rows),
            seed=int(seed + fold.fold_id * 917),
            max_depth=int(max_depth),
            min_child_fraction=float(min_child_fraction),
        )
        if len(fold_pred) == len(va):
            pred[va] = fold_pred
        fit_rows.append(
            {
                "fold": int(fold.fold_id),
                "arm": spec.arm,
                "description": spec.description,
                **asdict(spec),
                **fit_info,
                **ref_diag,
            }
        )
        weight_rows.append(
            {
                "fold": int(fold.fold_id),
                "arm": spec.arm,
                "description": spec.description,
                "train_rows": int(len(tr)),
                **asdict(spec),
                **weight_diag,
                **ref_diag,
            }
        )
    return pred, fit_rows, weight_rows


def _specs_from_args(args: argparse.Namespace) -> list[WeightSpec]:
    specs = [
        WeightSpec("J0_current_contextual_bce", "selected contextual arm with current unweighted BCE", False),
        WeightSpec("J1_timestamp_balanced_bce", "equal timestamp mass BCE", True),
    ]
    alphas = [float(x) for x in args.alpha_grid]
    betas = [float(x) for x in args.beta_grid]
    gammas = [float(x) for x in args.gamma_grid]
    taus = [float(x) for x in args.tau_grid]
    j2 = [
        WeightSpec(
            f"J2_top_tail_a{alpha:g}_tau{tau:g}",
            "timestamp-balanced BCE plus smooth top-tail emphasis",
            True,
            tail_weight=True,
            alpha=alpha,
            tau=tau,
        )
        for alpha in alphas
        for tau in taus
    ]
    specs.extend(j2[: max(0, int(args.max_j2_configs))])
    full_j3 = [
        WeightSpec(
            f"J3_swap_a{alpha:g}_b{beta:g}_g{gamma:g}_tau{tau:g}",
            "J2 plus swap-critical hard-example emphasis",
            True,
            tail_weight=True,
            swap_weight=True,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            tau=tau,
        )
        for alpha in alphas
        for beta in betas
        for gamma in gammas
        for tau in taus
    ]
    preferred = WeightSpec(
        "J3_swap_a1_b2_g2_tau0.05",
        "J2 plus swap-critical hard-example emphasis",
        True,
        tail_weight=True,
        swap_weight=True,
        alpha=1.0,
        beta=2.0,
        gamma=2.0,
        tau=0.05,
    )
    selected: list[WeightSpec] = []
    max_j3 = max(0, int(args.max_j3_configs))
    if max_j3 > 0 and preferred in full_j3:
        selected.append(preferred)
    for spec in full_j3:
        if len(selected) >= max_j3:
            break
        if spec not in selected:
            selected.append(spec)
    specs.extend(selected)
    return specs


def _load_frozen_feature_arms(source_dir: Path) -> pd.DataFrame:
    promotion_path = source_dir / "one_head_contextual_meta_promotion_table.csv"
    if not promotion_path.exists():
        raise FileNotFoundError(f"Missing persisted directional promotion table: {promotion_path}")
    promotion = pd.read_csv(promotion_path)
    rows: list[dict[str, Any]] = []
    for head, group in promotion.groupby("head", sort=True):
        candidates = group.loc[
            group["arm"].astype(str).isin(set(ctx.FEATURE_ARMS))
            & group["arm"].astype(str).ne(ctx.ARM_A)
        ].copy()
        if candidates.empty:
            continue
        for col in (
            "delta_timestamp_weighted_hr_top30",
            "directional_episode_median_delta_timestamp_weighted_hr_top30",
            "delta_ndcg_top30",
            "delta_pairwise_concordance_top30",
            "delta_timestamp_weighted_hr_70_80",
            "delta_timestamp_weighted_hr_top10",
            "delta_timestamp_weighted_hr_top20",
            "delta_log_loss_improvement",
        ):
            candidates[col] = pd.to_numeric(candidates.get(col), errors="coerce")
        candidates["episode_rank"] = candidates["directional_episode_median_delta_timestamp_weighted_hr_top30"].fillna(
            candidates["delta_timestamp_weighted_hr_top30"]
        )
        selected = candidates.sort_values(
            [
                "delta_timestamp_weighted_hr_top30",
                "episode_rank",
                "delta_ndcg_top30",
                "delta_pairwise_concordance_top30",
                "delta_timestamp_weighted_hr_70_80",
                "delta_timestamp_weighted_hr_top20",
                "delta_timestamp_weighted_hr_top10",
                "delta_log_loss_improvement",
            ],
            ascending=[False] * 8,
        ).iloc[0]
        rows.append(
            {
                "head": head,
                "selected_feature_arm": str(selected["arm"]),
                "selection_source": str(promotion_path),
                "selection_objective": "persisted_A_to_E_timestamp_HR30_episode_NDCG_boundary_safeguards",
                "delta_timestamp_weighted_hr_top30": selected.get("delta_timestamp_weighted_hr_top30", np.nan),
                "episode_rank_delta_hr_top30": selected.get("episode_rank", np.nan),
                "delta_ndcg_top30": selected.get("delta_ndcg_top30", np.nan),
                "delta_pairwise_concordance_top30": selected.get("delta_pairwise_concordance_top30", np.nan),
                "delta_boundary_hr_70_80": selected.get("delta_timestamp_weighted_hr_70_80", np.nan),
                "delta_hr_top10": selected.get("delta_timestamp_weighted_hr_top10", np.nan),
                "delta_hr_top20": selected.get("delta_timestamp_weighted_hr_top20", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _directional_ci(directional_episode: pd.DataFrame, seed: int) -> pd.DataFrame:
    return ctx._directional_episode_block_confidence_intervals(directional_episode, seed=int(seed), bootstrap_rounds=1000)


def _promotion_table(summary: pd.DataFrame, directional: pd.DataFrame, directional_ci: pd.DataFrame, *, hr_tolerance: float) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    table = summary.merge(directional, on=["head", "arm", "distillation_variant"], how="left")
    if not directional_ci.empty:
        ci = directional_ci.loc[
            directional_ci["metric"].astype(str).isin(
                ["delta_timestamp_weighted_hr_top30", "delta_ndcg_top30"]
            )
        ].copy()
        pivots = []
        for value_col, prefix in (
            ("median", "episode_median"),
            ("positive_episode_rate", "episode_positive_rate"),
            ("episode_count", "episode_count"),
        ):
            pivot = ci.pivot_table(
                index=["head", "arm", "distillation_variant"],
                columns="metric",
                values=value_col,
                aggfunc="first",
            )
            pivot.columns = [f"{prefix}_{col}" for col in pivot.columns]
            pivots.append(pivot.reset_index())
        if pivots:
            wide = pivots[0]
            for pivot in pivots[1:]:
                wide = wide.merge(pivot, on=["head", "arm", "distillation_variant"], how="outer")
            table = table.merge(wide, on=["head", "arm", "distillation_variant"], how="left")
    episode_count = pd.to_numeric(table.get("episode_count_delta_timestamp_weighted_hr_top30", pd.Series(0, index=table.index)), errors="coerce").fillna(0)
    episode_median = pd.to_numeric(
        table.get("episode_median_delta_timestamp_weighted_hr_top30", pd.Series(np.nan, index=table.index)), errors="coerce"
    )
    episode_rate = pd.to_numeric(
        table.get("episode_positive_rate_delta_timestamp_weighted_hr_top30", pd.Series(np.nan, index=table.index)), errors="coerce"
    )
    delta_hr30 = pd.to_numeric(table.get("delta_timestamp_weighted_hr_top30", pd.Series(np.nan, index=table.index)), errors="coerce")
    delta_hr10 = pd.to_numeric(table.get("delta_timestamp_weighted_hr_top10", pd.Series(np.nan, index=table.index)), errors="coerce")
    delta_ndcg = pd.to_numeric(table.get("delta_ndcg_top30", pd.Series(np.nan, index=table.index)), errors="coerce")
    normal_delta = pd.to_numeric(table.get("normal_period_delta_hr_top30", pd.Series(np.nan, index=table.index)), errors="coerce")
    table["passes_directional_training_gate"] = (
        delta_hr30.fillna(-np.inf).gt(0.0)
        & delta_ndcg.fillna(-np.inf).gt(0.0)
        & delta_hr10.fillna(-np.inf).ge(-float(hr_tolerance))
        & normal_delta.fillna(0.0).ge(-float(hr_tolerance))
        & (episode_count.eq(0) | (episode_median.fillna(-np.inf).gt(0.0) & episode_rate.fillna(0.0).ge(0.75)))
    )
    table["selection_rank_score"] = (
        episode_median.fillna(delta_hr30).fillna(0.0) * 1_000_000.0
        + delta_hr30.fillna(0.0) * 100_000.0
        + delta_ndcg.fillna(0.0) * 10_000.0
    )
    return table.sort_values(
        ["head", "passes_directional_training_gate", "selection_rank_score"],
        ascending=[True, False, False],
    )


def _write_report(out_dir: Path, freeze: pd.DataFrame, promotion: pd.DataFrame, audit: dict[str, Any]) -> None:
    lines = [
        "# Top-30-Aware Contextual Meta Training Ablation",
        "",
        "The experiment keeps one meta head, the unchanged `y_bin` label, binary log-loss, and one probability output.",
        "Only row weighting and directional model selection change.",
        "",
        "## Requirement Audit",
        "",
        pd.DataFrame(audit.get("items", [])).to_markdown(index=False),
        "",
        "## Frozen Feature Arms",
        "",
        freeze.to_markdown(index=False, floatfmt=".6f") if not freeze.empty else "_No frozen arms._",
        "",
    ]
    if not promotion.empty:
        cols = [
            "head",
            "arm",
            "passes_directional_training_gate",
            "timestamp_weighted_hr_top30",
            "delta_timestamp_weighted_hr_top30",
            "episode_median_delta_timestamp_weighted_hr_top30",
            "episode_positive_rate_delta_timestamp_weighted_hr_top30",
            "ndcg_top30",
            "delta_ndcg_top30",
            "delta_timestamp_weighted_hr_top10",
            "delta_timestamp_weighted_hr_top20",
            "normal_period_delta_hr_top30",
            "bad_period_delta_hr_top30",
            "top30_jaccard",
            "net_correct_trades_gained",
            "delta_log_loss_improvement",
        ]
        lines.extend(["## Promotion Table", "", promotion[[c for c in cols if c in promotion.columns]].to_markdown(index=False, floatfmt=".6f"), ""])
    (out_dir / "top30_aware_contextual_meta_training_report.md").write_text("\n".join(lines))


def _audit_outputs(
    *,
    freeze: pd.DataFrame,
    summary: pd.DataFrame,
    directional: pd.DataFrame,
    timestamp_metrics: pd.DataFrame,
    weights: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    items = []
    items.append(
        {
            "requirement": "frozen_contextual_feature_arm_per_head",
            "status": "passed" if not freeze.empty and {"head", "selected_feature_arm"} <= set(freeze.columns) else "failed",
            "metrics": {"rows": int(len(freeze)), "heads": sorted(set(freeze.get("head", pd.Series(dtype=str)).astype(str)))},
        }
    )
    items.append(
        {
            "requirement": "single_head_unchanged_label_contract",
            "status": "passed"
            if not summary.empty
            and set(summary.get("training_target", pd.Series(dtype=str)).dropna().astype(str)) == {"y_bin"}
            and not summary.get("forbidden_targets_used", pd.Series([True])).astype(bool).any()
            else "failed",
            "metrics": {"rows": int(len(summary)), "arms": sorted(set(summary.get("arm", pd.Series(dtype=str)).astype(str)))},
        }
    )
    items.append(
        {
            "requirement": "top30_timestamp_local_metrics_present",
            "status": "passed"
            if not directional.empty
            and not timestamp_metrics.empty
            and {"timestamp_weighted_hr_top30", "delta_timestamp_weighted_hr_top30", "ndcg_top30", "delta_ndcg_top30"} <= set(directional.columns)
            else "failed",
            "metrics": {"directional_rows": int(len(directional)), "timestamp_rows": int(len(timestamp_metrics))},
        }
    )
    weighted = weights.loc[weights["arm"].astype(str).ne("J0_current_contextual_bce")] if not weights.empty else pd.DataFrame()
    mass_cv = pd.to_numeric(weighted.get("timestamp_mass_cv", pd.Series(dtype=float)), errors="coerce")
    items.append(
        {
            "requirement": "timestamp_equal_mass_weights",
            "status": "passed" if not weighted.empty and mass_cv.dropna().le(1e-5).all() else "failed",
            "metrics": {
                "weighted_rows": int(len(weighted)),
                "max_timestamp_mass_cv": float(mass_cv.max()) if not mass_cv.dropna().empty else np.nan,
            },
        }
    )
    items.append(
        {
            "requirement": "cross_fitted_reference_for_tail_and_swap_weights",
            "status": "passed"
            if weights.empty
            or weights.loc[weights["tail_weight"].astype(bool) | weights["swap_weight"].astype(bool), "reference_source"]
            .astype(str)
            .str.contains("inner_oof|baseline_fallback")
            .all()
            else "failed",
            "metrics": {
                "reference_sources": sorted(set(weights.get("reference_source", pd.Series(dtype=str)).astype(str))) if not weights.empty else [],
                "inner_folds": int(args.inner_folds),
            },
        }
    )
    statuses = [item["status"] for item in items]
    return {
        "status": "passed" if all(status == "passed" for status in statuses) else "failed",
        "items": items,
        "outcomes": {
            "summary_rows": int(len(summary)),
            "directional_rows": int(len(directional)),
            "timestamp_metric_rows": int(len(timestamp_metrics)),
            "weight_rows": int(len(weights)),
        },
    }


def run(args: argparse.Namespace) -> Path:
    if ctx.lgb is None:
        raise RuntimeError("lightgbm is required")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_directional_dir)
    freeze = _load_frozen_feature_arms(source_dir)
    if args.only_head:
        freeze = freeze.loc[freeze["head"].astype(str).isin(set(str(x) for x in args.only_head))].reset_index(drop=True)
    freeze.to_csv(out_dir / "top30_aware_contextual_meta_feature_arm_freeze.csv", index=False)
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
    heads = _discover_heads(meta_artifact_dir, report_dir, meta_models)
    wanted = set(selected_by_head)
    heads = [h for h in heads if h.head in wanted]
    with (baseline_artifact_dir / "base_models_intermediate.pkl").open("rb") as fh:
        base_bundle = pickle.load(fh)
    symbol_columns = _feature_store_union(feature_dir)
    specs = _specs_from_args(args)

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    directional_rows: list[dict[str, Any]] = []
    directional_timestamp_rows: list[dict[str, Any]] = []
    directional_episode_rows: list[dict[str, Any]] = []

    for head in heads:
        print(f"[top30_context_train] processing head={head.head}", flush=True)
        panel = _normalise_keys(pd.read_parquet(head.meta_oof_path))
        panel = _downcast_numeric(panel, exclude=["timestamp", "symbol"])
        if int(args.max_rows) > 0 and len(panel) > int(args.max_rows):
            keep = np.linspace(0, len(panel) - 1, int(args.max_rows)).round().astype(int)
            panel = panel.iloc[np.unique(keep)].sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
            print(f"[top30_context_train] sampled head={head.head} rows={len(panel)}", flush=True)
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
        canonical, _ctx_diag = ctx._fold_canonical_features(
            raw,
            folds,
            canonical_defs,
            trailing_window=int(args.trailing_window),
            min_periods=int(args.min_periods),
            min_resolved_features=int(args.min_resolved_features),
        )
        canonical["leaf_occupancy_novelty"] = ctx._fit_leaf_occupancy_novelty(panel, folds).reset_index(drop=True)
        canonical = canonical.loc[:, list(ctx.CANONICAL_CONTEXT)]
        canonical.to_parquet(out_dir / f"{head.head}_top30_fold_fitted_context.parquet", index=False)
        arms = ctx._arm_frames(panel, current_x.reset_index(drop=True), canonical)
        selected_arm = selected_by_head[head.head]
        if selected_arm not in arms or arms[selected_arm] is None:
            raise RuntimeError(f"Selected arm {selected_arm} is unavailable for {head.head}")
        bad_episodes, _bad_meta = ctx._load_episode_registry(args.episode_registry, head=head.head, target_name=args.target_name)
        x_selected = arms[selected_arm]
        reference_cache: dict[tuple[int, int, int, int, int], tuple[np.ndarray, dict[str, Any]]] = {}
        for spec_i, spec in enumerate(specs, start=1):
            pred, fit_info, w_info = _fit_predict_weighted_classifier(
                x=x_selected,
                y=y,
                timestamps=panel["timestamp"].reset_index(drop=True),
                folds=folds,
                baseline_pred=baseline_pred,
                spec=spec,
                seed=int(args.seed + spec_i * 1009),
                max_train_rows=int(args.max_train_rows),
                inner_folds=int(args.inner_folds),
                embargo_hours=int(args.embargo_hours),
                max_depth=int(args.max_depth),
                min_child_fraction=float(args.min_child_fraction),
                reference_cache=reference_cache,
            )
            for row in fit_info:
                fold_rows.append({"head": head.head, "selected_feature_arm": selected_arm, "training_target": "y_bin", **row})
            for row in w_info:
                weight_rows.append({"head": head.head, "selected_feature_arm": selected_arm, **row})
            summary = ctx._overall_metrics(
                head=head.head,
                arm=spec.arm,
                variant="top30_aware_hard_label_bce",
                y=y,
                pred=pred,
                baseline_pred=baseline_pred,
                returns=returns,
            )
            summary.update(
                {
                    "selected_feature_arm": selected_arm,
                    "training_target": "y_bin",
                    "single_output_score": True,
                    "binary_logloss_objective": True,
                    "forbidden_targets_used": False,
                    **asdict(spec),
                }
            )
            summary_rows.append(summary)
            ts_metrics = ctx._directional_timestamp_metrics(
                head=head.head,
                arm=spec.arm,
                variant="top30_aware_hard_label_bce",
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
                agg = ctx._directional_aggregate(ts_metrics)
                if not agg.empty:
                    directional_rows.extend(agg.to_dict(orient="records"))
                episode = ctx._directional_episode_metrics(ts_metrics, bad_episodes)
                if not episode.empty:
                    directional_episode_rows.extend(episode.to_dict(orient="records"))

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)
    weights_df = pd.DataFrame(weight_rows)
    directional_df = pd.DataFrame(directional_rows)
    directional_timestamp_df = pd.DataFrame(directional_timestamp_rows)
    directional_episode_df = pd.DataFrame(directional_episode_rows)
    directional_ci_df = _directional_ci(directional_episode_df, seed=int(args.seed) + 55000)
    promotion_df = _promotion_table(summary_df, directional_df, directional_ci_df, hr_tolerance=float(args.directional_hr_tolerance))
    audit = _audit_outputs(
        freeze=freeze,
        summary=summary_df,
        directional=directional_df,
        timestamp_metrics=directional_timestamp_df,
        weights=weights_df,
        args=args,
    )

    summary_df.to_csv(out_dir / "top30_aware_contextual_meta_training_summary.csv", index=False)
    fold_df.to_csv(out_dir / "top30_aware_contextual_meta_training_fold_metrics.csv", index=False)
    weights_df.to_csv(out_dir / "top30_aware_contextual_meta_training_weight_diagnostics.csv", index=False)
    directional_df.to_csv(out_dir / "top30_aware_contextual_meta_training_directional_metrics.csv", index=False)
    directional_timestamp_df.to_csv(out_dir / "top30_aware_contextual_meta_training_directional_timestamp_metrics.csv", index=False)
    directional_episode_df.to_csv(out_dir / "top30_aware_contextual_meta_training_directional_episode_metrics.csv", index=False)
    directional_ci_df.to_csv(out_dir / "top30_aware_contextual_meta_training_directional_episode_confidence_intervals.csv", index=False)
    promotion_df.to_csv(out_dir / "top30_aware_contextual_meta_training_promotion_table.csv", index=False)
    (out_dir / "top30_aware_contextual_meta_training_requirement_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=_json_default)
    )
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=_json_default))
    _write_report(out_dir, freeze, promotion_df, audit)
    print(f"[top30_context_train] wrote results to {out_dir}", flush=True)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directional-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--meta-artifact-dir", default="data_perp/artifacts/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--baseline-artifact-dir", default="data_perp/artifacts/20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--report-dir", default="data_perp/reports/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--feature-dir", default="data_perp/features/20260605_070000")
    parser.add_argument(
        "--transform-cache",
        default="data_perp/reports/performance_regime_break_transform_cache/generated_transforms_single_3f7f9c53eaaa98ce632760a976691f24.parquet",
    )
    parser.add_argument(
        "--canonical-reduction",
        default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_usefulness_multitarget_clean_contract_v1/canonical_archetype_reduction.csv",
    )
    parser.add_argument("--regime-context", default="")
    parser.add_argument("--output-dir", default="data_perp/reports/top30_aware_contextual_meta_training_ablation_20260623")
    parser.add_argument("--episode-registry", default=str(ctx.DEFAULT_EPISODE_REGISTRY))
    parser.add_argument("--target-name", default="y_bin")
    parser.add_argument("--rank-threshold", type=float, default=0.70)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--embargo-hours", type=int, default=24)
    parser.add_argument("--trailing-window", type=int, default=24 * 28)
    parser.add_argument("--min-periods", type=int, default=24 * 7)
    parser.add_argument("--min-resolved-features", type=int, default=2)
    parser.add_argument("--max-regime-columns", type=int, default=80)
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-child-fraction", type=float, default=0.025)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--beta-grid", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--gamma-grid", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--tau-grid", nargs="+", type=float, default=[0.03, 0.05, 0.10])
    parser.add_argument("--max-j2-configs", type=int, default=9)
    parser.add_argument("--max-j3-configs", type=int, default=12)
    parser.add_argument("--directional-min-timestamp-rows", type=int, default=5)
    parser.add_argument("--directional-hr-tolerance", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--only-head", nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
