#!/usr/bin/env python3
"""Two-head month-forward model smoke for materialized first-touch labels.

This is a pre-production label-quality diagnostic. It trains cheap support and
utility proxies on prior months only, then ranks utility inside rows passing a
support-score gate. It does not run production base/meta training, Optuna, or
policy geometry optimisation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_first_touch_label_training_smoke import (  # noqa: E402
    DEFAULT_LABELS_DIR,
    _decile_monotonicity,
    _first_touch_eval_metrics,
    _fit_predict,
    _parse_csv,
    _parse_int_csv,
    _safe_numeric,
    _table,
    _target_from_frame,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_label_weighted_proxy_ablation import (  # noqa: E402
    WEIGHT_ARMS,
    _effective_sample_size,
    _weight_series,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/first_touch_two_head_training_smoke_v1")
DEFAULT_SUPPORT_TARGETS = ("clean_exec", "exec_guarded_policy")
DEFAULT_UTILITY_TARGET_MODE = "policy_soft"
DEFAULT_UTILITY_WEIGHT_ARMS = ("W8_combined_conservative", "W0_base")
DEFAULT_SUPPORT_WEIGHT_ARMS = ("W0_base", "W7_timestamp_balanced")
DEFAULT_SUPPORT_GATE_FRACS = (0.03, 0.05, 0.10, 0.20, 0.30)
DEFAULT_TOP_FRACS = (0.01, 0.03, 0.05)
DEFAULT_SCORE_RULES = ("utility_inside_support", "blend_70u_30s")


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _rank_pct(values: Any) -> pd.Series:
    return _safe_numeric(values).rank(method="average", pct=True).fillna(0.5).clip(0.0, 1.0)


def _final_score(*, utility_pred: pd.Series, support_pred: pd.Series, score_rule: str) -> pd.Series:
    utility = _safe_numeric(utility_pred).reset_index(drop=True)
    support = _safe_numeric(support_pred).reset_index(drop=True)
    if score_rule == "utility_inside_support":
        return utility
    if score_rule == "blend_70u_30s":
        return (0.70 * _rank_pct(utility)) + (0.30 * _rank_pct(support))
    raise ValueError(f"Unknown score rule: {score_rule}")


def _select_indices(
    *,
    score: pd.Series,
    support_pred: pd.Series,
    support_gate_frac: float,
    top_frac: float,
    total_rows: int,
) -> tuple[np.ndarray, pd.Series]:
    support_idx = _rank_top_indices(support_pred, float(support_gate_frac))
    gate_mask = pd.Series(False, index=np.arange(total_rows))
    if len(support_idx):
        gate_mask.iloc[support_idx] = True
    gated_score = _safe_numeric(score).reset_index(drop=True).where(gate_mask)
    valid = gated_score.notna().to_numpy()
    if not bool(valid.any()):
        return np.array([], dtype=np.int64), gate_mask
    valid_idx = np.flatnonzero(valid)
    k = max(1, int(math.ceil(float(top_frac) * float(total_rows))))
    k = min(k, len(valid_idx))
    order = np.argsort(-gated_score.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    return valid_idx[order[:k]].astype(np.int64, copy=False), gate_mask


def _selected_metrics(
    *,
    valid: pd.DataFrame,
    metrics: pd.DataFrame,
    utility_target: pd.DataFrame,
    support_target: pd.DataFrame,
    utility_pred: pd.Series,
    support_pred: pd.Series,
    score: pd.Series,
    selected_idx: np.ndarray,
    gate_mask: pd.Series,
    period: str,
    utility_target_mode: str,
    support_target_mode: str,
    utility_weight_arm: str,
    support_weight_arm: str,
    support_gate_frac: float,
    top_frac: float,
    score_rule: str,
) -> dict[str, Any]:
    selected = valid.iloc[selected_idx].copy() if len(selected_idx) else valid.iloc[:0].copy()
    selected_metrics = metrics.iloc[selected_idx].copy() if len(selected_idx) else metrics.iloc[:0].copy()
    selected_utility = (
        utility_target.iloc[selected_idx].copy() if len(selected_idx) else utility_target.iloc[:0].copy()
    )
    selected_support = (
        support_target.iloc[selected_idx].copy() if len(selected_idx) else support_target.iloc[:0].copy()
    )
    u_sel = _safe_numeric(selected_metrics["u_policy_net"])
    ft_sel = _safe_numeric(selected_metrics["first_touch_net"])
    period_u = _safe_numeric(metrics["u_policy_net"])
    period_ft = _safe_numeric(metrics["first_touch_net"])
    support_gate_rows = int(pd.Series(gate_mask).fillna(False).astype(bool).sum())
    top_symbol_share = (
        float(selected["__symbol__"].astype(str).value_counts(normalize=True).iloc[0])
        if len(selected)
        else float("nan")
    )
    return {
        "period": str(period),
        "utility_target_mode": str(utility_target_mode),
        "support_target_mode": str(support_target_mode),
        "utility_weight_arm": str(utility_weight_arm),
        "support_weight_arm": str(support_weight_arm),
        "score_rule": str(score_rule),
        "support_gate_frac": float(support_gate_frac),
        "top_frac": float(top_frac),
        "rows": int(len(valid)),
        "support_gate_rows": support_gate_rows,
        "support_gate_rate": support_gate_rows / float(len(valid)) if len(valid) else 0.0,
        "selected_rows": int(len(selected_idx)),
        "selected_rate": len(selected_idx) / float(len(valid)) if len(valid) else 0.0,
        "mean_u": _safe_mean(u_sel),
        "hit_u": _safe_mean(u_sel > 0.0),
        "q10_u": _safe_quantile(u_sel, 0.10),
        "period_mean_u": _safe_mean(period_u),
        "delta_mean_u_vs_period": _safe_mean(u_sel) - _safe_mean(period_u),
        "mean_first_touch_net": _safe_mean(ft_sel),
        "hit_first_touch_net": _safe_mean(ft_sel > 0.0),
        "q10_first_touch_net": _safe_quantile(ft_sel, 0.10),
        "period_mean_first_touch_net": _safe_mean(period_ft),
        "delta_first_touch_net_vs_period": _safe_mean(ft_sel) - _safe_mean(period_ft),
        "utility_target_top_soft_mean": _safe_mean(selected_utility["target_soft"]),
        "utility_target_top_hard_rate": _safe_mean(selected_utility["target_hard"]),
        "support_target_top_soft_mean": _safe_mean(selected_support["target_soft"]),
        "support_target_top_hard_rate": _safe_mean(selected_support["target_hard"]),
        "utility_ic_first_touch_net": _spearman(utility_pred, metrics["first_touch_net"]),
        "utility_ic_utility_target": _spearman(utility_pred, utility_target["target_soft"]),
        "support_ic_clean_exec": _spearman(support_pred, metrics["clean_first_touch_exec"]),
        "support_ic_support_target": _spearman(support_pred, support_target["target_soft"]),
        "score_ic_first_touch_net": _spearman(score, metrics["first_touch_net"]),
        "score_ic_u": _spearman(score, metrics["u_policy_net"]),
        "decile_monotonicity_first_touch_net": _decile_monotonicity(score, metrics["first_touch_net"]),
        "clean_first_touch_exec_rate": _safe_mean(selected_metrics["clean_first_touch_exec"]),
        "first_touch_hit_rate": _safe_mean(selected_metrics["first_touch_hit"]),
        "first_touch_stop_rate": _safe_mean(selected_metrics["first_touch_stop"]),
        "first_touch_timeout_rate": _safe_mean(selected_metrics["first_touch_timeout"]),
        "first_touch_same_bar_rate": _safe_mean(selected_metrics["first_touch_same_bar"]),
        "first_touch_bad_mae_to_sl_rate": _safe_mean(selected_metrics["first_touch_mae_to_sl"] >= 1.0),
        "p90_first_touch_mae_to_sl": _safe_quantile(selected_metrics["first_touch_mae_to_sl"], 0.90),
        "p90_first_touch_bar": _safe_quantile(selected_metrics["first_touch_bar"], 0.90),
        "p90_full_path_mae_to_sl": _safe_quantile(selected_metrics["first_touch_full_path_mae_to_sl"], 0.90),
        "bad_mae_1r_rate": _safe_mean(selected_metrics["mae_norm"] >= 1.0),
        "wide_barrier_25bps_rate": _safe_mean(selected_metrics["barrier"] >= 0.025),
        "top_symbol_share": top_symbol_share,
    }


def _weekly_rows(
    *,
    valid: pd.DataFrame,
    metrics: pd.DataFrame,
    selected_idx: np.ndarray,
    base_row: dict[str, Any],
) -> list[dict[str, Any]]:
    if not len(selected_idx):
        return []
    selected = valid.iloc[selected_idx].reset_index(drop=True)
    selected_metrics = metrics.iloc[selected_idx].reset_index(drop=True)
    weeks = selected["__ts__"].dt.to_period("W-SUN").astype(str)
    rows: list[dict[str, Any]] = []
    for week, ids in pd.Series(np.arange(len(selected))).groupby(weeks, dropna=False):
        pos = ids.to_numpy(dtype=np.int64)
        week_metrics = selected_metrics.iloc[pos]
        week_frame = selected.iloc[pos]
        row = {
            key: base_row[key]
            for key in [
                "period",
                "utility_target_mode",
                "support_target_mode",
                "utility_weight_arm",
                "support_weight_arm",
                "score_rule",
                "support_gate_frac",
                "top_frac",
            ]
        }
        row.update(
            {
                "week": str(week),
                "week_selected_rows": int(len(pos)),
                "mean_first_touch_net": _safe_mean(week_metrics["first_touch_net"]),
                "hit_first_touch_net": _safe_mean(week_metrics["first_touch_net"] > 0.0),
                "q10_first_touch_net": _safe_quantile(week_metrics["first_touch_net"], 0.10),
                "mean_u": _safe_mean(week_metrics["u_policy_net"]),
                "hit_u": _safe_mean(week_metrics["u_policy_net"] > 0.0),
                "clean_first_touch_exec_rate": _safe_mean(week_metrics["clean_first_touch_exec"]),
                "first_touch_hit_rate": _safe_mean(week_metrics["first_touch_hit"]),
                "first_touch_timeout_rate": _safe_mean(week_metrics["first_touch_timeout"]),
                "first_touch_bad_mae_to_sl_rate": _safe_mean(week_metrics["first_touch_mae_to_sl"] >= 1.0),
                "p90_first_touch_mae_to_sl": _safe_quantile(week_metrics["first_touch_mae_to_sl"], 0.90),
                "p90_full_path_mae_to_sl": _safe_quantile(week_metrics["first_touch_full_path_mae_to_sl"], 0.90),
                "top_symbol_share": float(week_frame["__symbol__"].astype(str).value_counts(normalize=True).iloc[0])
                if len(week_frame)
                else float("nan"),
            }
        )
        rows.append(row)
    return rows


def _fit_seed_ensemble(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    seeds: list[int],
) -> tuple[pd.Series, float]:
    preds = [
        _fit_predict(
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
            x_valid=x_valid,
            seed=seed,
        )
        for seed in seeds
    ]
    matrix = np.vstack(preds) if preds else np.empty((0, len(x_valid)), dtype=np.float32)
    pred = np.mean(matrix, axis=0).astype(np.float32) if len(matrix) else np.full(len(x_valid), np.nan)
    seed_std = float(np.mean(np.std(matrix, axis=0))) if len(matrix) > 1 else 0.0
    return pd.Series(pred).reset_index(drop=True), seed_std


def _run_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    utility_target: pd.DataFrame,
    utility_target_mode: str,
    support_targets: dict[str, pd.DataFrame],
    features: list[str],
    month: str,
    utility_weight_arms: list[str],
    support_weight_arms: list[str],
    support_gate_fracs: list[float],
    top_fracs: list[float],
    score_rules: list[str],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < month
    valid_mask = month_period == month
    if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
        return [], [], [{"period": month, "skipped": True, "train_rows": int(train_mask.sum()), "valid_rows": int(valid_mask.sum())}]

    x = frame[features].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    med = x.loc[train_mask].median(numeric_only=True)
    x = x.fillna(med).fillna(0.0).astype(np.float32, copy=False)
    train = frame.loc[train_mask].copy()
    train_metrics = metrics.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    valid_utility_target = utility_target.loc[valid_mask].copy().reset_index(drop=True)

    diagnostics: list[dict[str, Any]] = []
    utility_preds: dict[str, pd.Series] = {}
    for weight_arm in utility_weight_arms:
        train_target = utility_target.loc[train_mask].copy()
        weights = _weight_series(frame=train, metrics=train_metrics, target=train_target, arm=weight_arm)
        pred, seed_std = _fit_seed_ensemble(
            x_train=x.loc[train_mask],
            y_train=train_target["target_soft"],
            w_train=weights,
            x_valid=x.loc[valid_mask],
            seeds=seeds,
        )
        utility_preds[weight_arm] = pred
        diagnostics.append(
            {
                "period": str(month),
                "head": "utility",
                "target_mode": str(utility_target_mode),
                "weight_arm": str(weight_arm),
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "target_train_mean": _safe_mean(train_target["target_soft"]),
                "target_valid_mean": _safe_mean(valid_utility_target["target_soft"]),
                "weight_mean": _safe_mean(weights),
                "weight_p90": _safe_quantile(weights, 0.90),
                "weight_effective_frac": _effective_sample_size(weights) / float(len(weights)) if len(weights) else float("nan"),
                "seed_std_mean": seed_std,
            }
        )

    support_preds: dict[tuple[str, str], pd.Series] = {}
    valid_support_targets: dict[str, pd.DataFrame] = {}
    for support_mode, support_target in support_targets.items():
        valid_support_targets[support_mode] = support_target.loc[valid_mask].copy().reset_index(drop=True)
        for weight_arm in support_weight_arms:
            train_target = support_target.loc[train_mask].copy()
            weights = _weight_series(frame=train, metrics=train_metrics, target=train_target, arm=weight_arm)
            pred, seed_std = _fit_seed_ensemble(
                x_train=x.loc[train_mask],
                y_train=train_target["target_soft"],
                w_train=weights,
                x_valid=x.loc[valid_mask],
                seeds=seeds,
            )
            support_preds[(support_mode, weight_arm)] = pred
            diagnostics.append(
                {
                    "period": str(month),
                    "head": "support",
                    "target_mode": str(support_mode),
                    "weight_arm": str(weight_arm),
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": int(valid_mask.sum()),
                    "target_train_mean": _safe_mean(train_target["target_soft"]),
                    "target_valid_mean": _safe_mean(valid_support_targets[support_mode]["target_soft"]),
                    "weight_mean": _safe_mean(weights),
                    "weight_p90": _safe_quantile(weights, 0.90),
                    "weight_effective_frac": _effective_sample_size(weights) / float(len(weights)) if len(weights) else float("nan"),
                    "seed_std_mean": seed_std,
                }
            )

    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    for utility_weight_arm, utility_pred in utility_preds.items():
        for (support_mode, support_weight_arm), support_pred in support_preds.items():
            support_target = valid_support_targets[support_mode]
            for score_rule in score_rules:
                score = _final_score(
                    utility_pred=utility_pred,
                    support_pred=support_pred,
                    score_rule=score_rule,
                )
                for support_gate_frac in support_gate_fracs:
                    for top_frac in top_fracs:
                        selected_idx, gate_mask = _select_indices(
                            score=score,
                            support_pred=support_pred,
                            support_gate_frac=support_gate_frac,
                            top_frac=top_frac,
                            total_rows=len(valid),
                        )
                        row = _selected_metrics(
                            valid=valid,
                            metrics=valid_metrics,
                            utility_target=valid_utility_target,
                            support_target=support_target,
                            utility_pred=utility_pred,
                            support_pred=support_pred,
                            score=score,
                            selected_idx=selected_idx,
                            gate_mask=gate_mask,
                            period=month,
                            utility_target_mode=utility_target_mode,
                            support_target_mode=support_mode,
                            utility_weight_arm=utility_weight_arm,
                            support_weight_arm=support_weight_arm,
                            support_gate_frac=support_gate_frac,
                            top_frac=top_frac,
                            score_rule=score_rule,
                        )
                        monthly_rows.append(row)
                        weekly_rows.extend(
                            _weekly_rows(
                                valid=valid,
                                metrics=valid_metrics,
                                selected_idx=selected_idx,
                                base_row=row,
                            )
                        )
    return monthly_rows, weekly_rows, diagnostics


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    rows: list[dict[str, Any]] = []
    keys = [
        "utility_target_mode",
        "support_target_mode",
        "utility_weight_arm",
        "support_weight_arm",
        "score_rule",
        "support_gate_frac",
        "top_frac",
    ]
    for key, group in monthly.groupby(keys, observed=True, dropna=False):
        mean_u = _safe_numeric(group["mean_u"])
        mean_ft = _safe_numeric(group["mean_first_touch_net"])
        row = {name: value for name, value in zip(keys, key, strict=True)}
        row.update(
            {
                "months": int(group["period"].nunique()),
                "positive_months": int((mean_u > 0.0).sum()),
                "positive_first_touch_months": int((mean_ft > 0.0).sum()),
                "mean_u": _safe_mean(mean_u),
                "worst_month_mean_u": _safe_quantile(mean_u, 0.0),
                "hit_u": _safe_mean(group["hit_u"]),
                "q10_u": _safe_mean(group["q10_u"]),
                "delta_mean_u_vs_period": _safe_mean(group["delta_mean_u_vs_period"]),
                "mean_first_touch_net": _safe_mean(mean_ft),
                "worst_month_first_touch_net": _safe_quantile(mean_ft, 0.0),
                "hit_first_touch_net": _safe_mean(group["hit_first_touch_net"]),
                "q10_first_touch_net": _safe_mean(group["q10_first_touch_net"]),
                "delta_first_touch_net_vs_period": _safe_mean(group["delta_first_touch_net_vs_period"]),
                "utility_ic_first_touch_net": _safe_mean(group["utility_ic_first_touch_net"]),
                "support_ic_clean_exec": _safe_mean(group["support_ic_clean_exec"]),
                "score_ic_first_touch_net": _safe_mean(group["score_ic_first_touch_net"]),
                "decile_monotonicity_first_touch_net": _safe_mean(group["decile_monotonicity_first_touch_net"]),
                "clean_first_touch_exec_rate": _safe_mean(group["clean_first_touch_exec_rate"]),
                "first_touch_hit_rate": _safe_mean(group["first_touch_hit_rate"]),
                "first_touch_timeout_rate": _safe_mean(group["first_touch_timeout_rate"]),
                "first_touch_bad_mae_to_sl_rate": _safe_mean(group["first_touch_bad_mae_to_sl_rate"]),
                "p90_first_touch_mae_to_sl": _safe_mean(group["p90_first_touch_mae_to_sl"]),
                "p90_first_touch_bar": _safe_mean(group["p90_first_touch_bar"]),
                "p90_full_path_mae_to_sl": _safe_mean(group["p90_full_path_mae_to_sl"]),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "support_gate_rate": _safe_mean(group["support_gate_rate"]),
                "selected_rate": _safe_mean(group["selected_rate"]),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
                "mean_selected_rows": _safe_mean(group["selected_rows"]),
                "min_selected_rows": int(_safe_numeric(group["selected_rows"]).min()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["positive_first_touch_months", "mean_first_touch_net", "worst_month_first_touch_net"],
        ascending=[False, False, False],
    )


def _write_markdown(
    *,
    path: Path,
    aggregate: pd.DataFrame,
    monthly: pd.DataFrame,
    outputs: dict[str, Path],
) -> None:
    cols = [
        "support_target_mode",
        "utility_weight_arm",
        "support_weight_arm",
        "score_rule",
        "support_gate_frac",
        "top_frac",
        "months",
        "positive_first_touch_months",
        "mean_first_touch_net",
        "worst_month_first_touch_net",
        "hit_first_touch_net",
        "clean_first_touch_exec_rate",
        "first_touch_timeout_rate",
        "first_touch_bad_mae_to_sl_rate",
        "p90_first_touch_mae_to_sl",
        "support_ic_clean_exec",
        "utility_ic_first_touch_net",
        "score_ic_first_touch_net",
        "mean_selected_rows",
    ]
    clean = aggregate[
        (_safe_numeric(aggregate["positive_first_touch_months"]) >= 3)
        & (_safe_numeric(aggregate["clean_first_touch_exec_rate"]) >= 0.70)
        & (_safe_numeric(aggregate["first_touch_timeout_rate"]) <= 0.05)
        & (_safe_numeric(aggregate["first_touch_bad_mae_to_sl_rate"]) <= 0.20)
    ].copy()
    balanced = aggregate[
        (_safe_numeric(aggregate["positive_first_touch_months"]) >= 3)
        & (_safe_numeric(aggregate["clean_first_touch_exec_rate"]) >= 0.60)
        & (_safe_numeric(aggregate["first_touch_timeout_rate"]) <= 0.15)
        & (_safe_numeric(aggregate["first_touch_bad_mae_to_sl_rate"]) <= 0.30)
    ].copy()
    lines = [
        "# First-Touch Two-Head Training Smoke",
        "",
        "Scope: cheap month-forward support-gate plus utility-rank smoke. This is not production training.",
        "",
        "## Clean executable candidates",
        "",
        _table(clean.sort_values(["mean_first_touch_net", "worst_month_first_touch_net"], ascending=[False, False]), cols, limit=30),
        "",
        "## Balanced candidates",
        "",
        _table(balanced.sort_values(["mean_first_touch_net", "worst_month_first_touch_net"], ascending=[False, False]), cols, limit=30),
        "",
        "## Top aggregate rows",
        "",
        _table(aggregate, cols, limit=40),
        "",
        "## Monthly detail for top 10 aggregate rows",
        "",
    ]
    if not aggregate.empty and not monthly.empty:
        key_cols = [
            "support_target_mode",
            "utility_weight_arm",
            "support_weight_arm",
            "score_rule",
            "support_gate_frac",
            "top_frac",
        ]
        top_keys = aggregate.head(10)[key_cols].drop_duplicates()
        detail = monthly.merge(top_keys, on=key_cols, how="inner")
        detail_cols = key_cols + [
            "period",
            "mean_first_touch_net",
            "hit_first_touch_net",
            "clean_first_touch_exec_rate",
            "first_touch_timeout_rate",
            "first_touch_bad_mae_to_sl_rate",
            "p90_first_touch_mae_to_sl",
            "selected_rows",
        ]
        lines.extend([_table(detail.sort_values(key_cols + ["period"]), detail_cols, limit=120), ""])
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Monthly: `{outputs['monthly']}`",
            f"- Weekly: `{outputs['weekly']}`",
            f"- Aggregate: `{outputs['aggregate']}`",
            f"- Diagnostics: `{outputs['diagnostics']}`",
            f"- Manifest: `{outputs['manifest']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    utility_target_mode: str,
    utility_weight_arms: list[str],
    support_weight_arms: list[str],
    support_target_modes: list[str],
    support_gate_fracs: list[float],
    top_fracs: list[float],
    score_rules: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        new_cols = [col for col in feature_matrix.columns if col not in frame.columns]
        if new_cols:
            frame = pd.concat(
                [
                    frame.reset_index(drop=True),
                    feature_matrix.loc[:, new_cols].reset_index(drop=True).astype(np.float32, copy=False),
                ],
                axis=1,
            ).copy()
    metrics = _first_touch_eval_metrics(frame, _path_metrics(frame))
    utility_target = _target_from_frame(frame, metrics, target_mode=utility_target_mode)
    support_targets = {
        mode: _target_from_frame(frame, metrics, target_mode=mode)
        for mode in support_target_modes
    }
    features = _feature_columns(frame)
    missing_weights = sorted((set(utility_weight_arms) | set(support_weight_arms)) - set(WEIGHT_ARMS))
    if missing_weights:
        raise ValueError(f"Unknown weight arms: {missing_weights}")
    months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())

    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for month in months[1:]:
        rows, weeks, diagnostics = _run_month(
            frame=frame,
            metrics=metrics,
            utility_target=utility_target,
            utility_target_mode=utility_target_mode,
            support_targets=support_targets,
            features=features,
            month=month,
            utility_weight_arms=utility_weight_arms,
            support_weight_arms=support_weight_arms,
            support_gate_fracs=support_gate_fracs,
            top_fracs=top_fracs,
            score_rules=score_rules,
            seeds=seeds,
        )
        monthly_rows.extend(rows)
        weekly_rows.extend(weeks)
        diagnostic_rows.extend(diagnostics)

    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    aggregate = _aggregate(monthly)
    diagnostics = pd.DataFrame(diagnostic_rows)
    outputs = {
        "monthly": output_dir / "first_touch_two_head_training_smoke_monthly.csv",
        "weekly": output_dir / "first_touch_two_head_training_smoke_weekly.csv",
        "aggregate": output_dir / "first_touch_two_head_training_smoke_aggregate.csv",
        "diagnostics": output_dir / "first_touch_two_head_training_smoke_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
        "markdown": output_dir / "first_touch_two_head_training_smoke.md",
    }
    monthly.to_csv(outputs["monthly"], index=False)
    weekly.to_csv(outputs["weekly"], index=False)
    aggregate.to_csv(outputs["aggregate"], index=False)
    diagnostics.to_csv(outputs["diagnostics"], index=False)
    manifest = {
        "scope": "cheap_month_forward_first_touch_two_head_smoke_not_full_policy_training",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_count": int(len(features)),
        "feature_store": feature_store_report,
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "utility_target_mode": str(utility_target_mode),
        "support_target_modes": list(support_target_modes),
        "utility_weight_arms": list(utility_weight_arms),
        "support_weight_arms": list(support_weight_arms),
        "support_gate_fracs": [float(v) for v in support_gate_fracs],
        "top_fracs": [float(v) for v in top_fracs],
        "score_rules": list(score_rules),
        "seeds": [int(seed) for seed in seeds],
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    _write_markdown(
        path=outputs["markdown"],
        aggregate=aggregate,
        monthly=monthly,
        outputs=outputs,
    )
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--utility-target-mode", default=DEFAULT_UTILITY_TARGET_MODE)
    parser.add_argument("--utility-weight-arms", default=",".join(DEFAULT_UTILITY_WEIGHT_ARMS))
    parser.add_argument("--support-weight-arms", default=",".join(DEFAULT_SUPPORT_WEIGHT_ARMS))
    parser.add_argument("--support-target-modes", default=",".join(DEFAULT_SUPPORT_TARGETS))
    parser.add_argument("--support-gate-fracs", default=",".join(str(v) for v in DEFAULT_SUPPORT_GATE_FRACS))
    parser.add_argument("--top-fracs", default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--score-rules", default=",".join(DEFAULT_SCORE_RULES))
    parser.add_argument("--seeds", default="42,7301,999")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_smoke(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        utility_target_mode=str(args.utility_target_mode),
        utility_weight_arms=_parse_csv(args.utility_weight_arms, DEFAULT_UTILITY_WEIGHT_ARMS),
        support_weight_arms=_parse_csv(args.support_weight_arms, DEFAULT_SUPPORT_WEIGHT_ARMS),
        support_target_modes=_parse_csv(args.support_target_modes, DEFAULT_SUPPORT_TARGETS),
        support_gate_fracs=_parse_float_csv(args.support_gate_fracs, DEFAULT_SUPPORT_GATE_FRACS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        score_rules=_parse_csv(args.score_rules, DEFAULT_SCORE_RULES),
        seeds=_parse_int_csv(args.seeds, (42, 7301, 999)),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
