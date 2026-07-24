#!/usr/bin/env python3
"""Cheap month-forward model smoke for materialized first-touch labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    _sigmoid,
    _spearman,
)
from scripts.run_label_weighted_proxy_ablation import (  # noqa: E402
    WEIGHT_ARMS,
    _effective_sample_size,
    _weight_series,
)


DEFAULT_LABELS_DIR = Path(
    "data_perp/artifacts/20260702_094500_first_touch_c0_fast6_s10_policy_net_labels/labels"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/first_touch_label_training_smoke_v1")
TOP_FRACS = (0.30, 0.10, 0.05, 0.03, 0.01)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _target_from_frame(frame: pd.DataFrame, metrics: pd.DataFrame, *, target_mode: str) -> pd.DataFrame:
    mode = str(target_mode)
    if mode == "p90_trailing_blend":
        required = ("__p90_trailing_target_soft__", "__p90_trailing_target_hard__")
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(
                "p90_trailing_blend requires the keyed target sidecar; "
                f"missing={missing}"
            )
        return pd.DataFrame(
            {
                "target_soft": _safe_numeric(frame[required[0]]).fillna(0.0).clip(0.0, 1.0).astype(np.float32),
                "target_hard": _safe_numeric(frame[required[1]]).fillna(0.0).clip(0.0, 1.0).astype(np.float32),
            },
            index=frame.index,
        )
    if mode == "side_continuous_geometry_v1":
        from extreme_price_movements.base_side_target_contract import (
            build_promoted_side_target,
        )

        return build_promoted_side_target(frame)
    if mode.startswith("column:"):
        parts = mode.split(":", 2)
        if len(parts) != 3 or not parts[1] or not parts[2]:
            raise ValueError("column target mode must be column:<soft_col>:<hard_col>")
        soft_col, hard_col = parts[1], parts[2]
        missing = [col for col in (soft_col, hard_col) if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing column target mode columns: {missing}")
        return pd.DataFrame(
            {
                "target_soft": _safe_numeric(frame[soft_col]).fillna(0.0).clip(0.0, 1.0).astype(np.float32),
                "target_hard": _safe_numeric(frame[hard_col]).fillna(0.0).clip(0.0, 1.0).astype(np.float32),
            },
            index=frame.index,
        )
    u = _safe_numeric(frame["__u_policy_net__"]).fillna(-0.02)
    policy_soft = (
        _safe_numeric(frame["__first_touch_policy_soft__"]).clip(0.0, 1.0)
        if "__first_touch_policy_soft__" in frame.columns
        else pd.Series(_sigmoid(u / 0.004), index=frame.index).clip(0.0, 1.0)
    )
    target_soft = (
        _safe_numeric(frame["__first_touch_target_soft__"]).clip(0.0, 1.0)
        if "__first_touch_target_soft__" in frame.columns
        else policy_soft
    )
    clean_envelope = (
        _safe_numeric(metrics["first_touch_hit"]).fillna(0.0).clip(0.0, 1.0)
        * (1.0 - _safe_numeric(metrics["first_touch_stop"]).fillna(1.0).clip(0.0, 1.0))
        * (1.0 - _safe_numeric(metrics["first_touch_timeout"]).fillna(1.0).clip(0.0, 1.0))
        * (1.0 - _safe_numeric(metrics["first_touch_same_bar"]).fillna(1.0).clip(0.0, 1.0))
        * pd.Series(
            _sigmoid((1.25 - _safe_numeric(metrics["first_touch_mae_to_sl"]).fillna(10.0)) / 0.35),
            index=frame.index,
        )
        * (
            0.50
            + 0.50
            * pd.Series(
                _sigmoid((12.0 - _safe_numeric(metrics["first_touch_bar"]).fillna(36.0)) / 4.0),
                index=frame.index,
            )
        )
    ).clip(0.0, 1.0)
    clean_exec_soft = (
        pd.Series(
            _sigmoid((_safe_numeric(metrics["first_touch_net"]).fillna(-0.02) - 0.0010) / 0.0060),
            index=frame.index,
        )
        * clean_envelope
    ).clip(0.0, 1.0)
    time_decay = pd.Series(
        _sigmoid((10.0 - _safe_numeric(metrics["first_touch_bar"]).fillna(36.0)) / 4.0),
        index=frame.index,
    ).clip(0.0, 1.0)
    tight_mae = pd.Series(
        _sigmoid((0.85 - _safe_numeric(metrics["first_touch_mae_to_sl"]).fillna(10.0)) / 0.25),
        index=frame.index,
    ).clip(0.0, 1.0)
    fast_clean_exec_soft = (clean_exec_soft * (0.20 + 0.80 * time_decay)).clip(0.0, 1.0)
    tight_clean_exec_soft = (clean_exec_soft * (0.20 + 0.80 * tight_mae)).clip(0.0, 1.0)
    fast_tight_clean_exec_soft = (
        clean_exec_soft * (0.15 + 0.85 * time_decay) * (0.15 + 0.85 * tight_mae)
    ).clip(0.0, 1.0)
    time_decay_policy = (policy_soft * (0.25 + 0.75 * time_decay)).clip(0.0, 1.0)
    if mode == "policy_soft":
        soft = policy_soft
    elif mode == "target_soft":
        soft = target_soft
    elif mode == "exec_guarded_policy":
        soft = (policy_soft * (0.15 + 0.85 * clean_envelope)).clip(0.0, 1.0)
    elif mode == "clean_exec":
        soft = clean_exec_soft
    elif mode == "fast_clean_exec":
        soft = fast_clean_exec_soft
    elif mode == "tight_clean_exec":
        soft = tight_clean_exec_soft
    elif mode == "fast_tight_clean_exec":
        soft = fast_tight_clean_exec_soft
    elif mode == "time_decay_policy":
        soft = time_decay_policy
    else:
        raise ValueError(f"Unknown target mode: {target_mode}")
    clean_hard = _safe_numeric(metrics["clean_first_touch_exec"])
    if mode == "clean_exec":
        hard = clean_hard
    elif mode == "fast_clean_exec":
        hard = clean_hard * (_safe_numeric(metrics["first_touch_bar"]).fillna(36.0) <= 12.0).astype(float)
    elif mode == "tight_clean_exec":
        hard = clean_hard * (_safe_numeric(metrics["first_touch_mae_to_sl"]).fillna(10.0) <= 0.85).astype(float)
    elif mode == "fast_tight_clean_exec":
        hard = _safe_numeric(metrics["clean_first_touch_exec"])
        hard = hard * (_safe_numeric(metrics["first_touch_bar"]).fillna(36.0) <= 12.0).astype(float)
        hard = hard * (_safe_numeric(metrics["first_touch_mae_to_sl"]).fillna(10.0) <= 0.85).astype(float)
    elif "__first_touch_hit__" in frame.columns:
        hard = _safe_numeric(frame["__first_touch_hit__"])
    else:
        hard = (u > 0.0).astype(float)
    return pd.DataFrame(
        {
            "target_soft": soft.fillna(0.0).astype(np.float32),
            "target_hard": hard.fillna(0.0).clip(0.0, 1.0).astype(np.float32),
        },
        index=frame.index,
    )


def _decile_monotonicity(pred: pd.Series, utility: pd.Series) -> float:
    frame = pd.DataFrame({"pred": _safe_numeric(pred), "u": _safe_numeric(utility)}).dropna()
    if len(frame) < 20:
        return float("nan")
    try:
        frame["decile"] = pd.qcut(
            frame["pred"].rank(method="first"),
            10,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return float("nan")
    by_decile = frame.groupby("decile", observed=True)["u"].mean()
    if len(by_decile) < 3:
        return float("nan")
    return _spearman(pd.Series(by_decile.index, dtype=float), by_decile.reset_index(drop=True))


def _first_touch_eval_metrics(frame: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    out["first_touch_net"] = _safe_numeric(
        frame.get("__first_touch_capture_net__", out["u_policy_net"])
    ).fillna(out["u_policy_net"])
    out["first_touch_hit"] = _safe_numeric(frame.get("__first_touch_hit__", out["u_policy_net"] > 0.0)).fillna(0.0).clip(
        0.0,
        1.0,
    )
    out["first_touch_stop"] = _safe_numeric(frame.get("__first_touch_stop__", 0.0)).fillna(0.0).clip(0.0, 1.0)
    out["first_touch_timeout"] = _safe_numeric(
        frame.get("__first_touch_timeout__", out["is_timeout"].astype(float))
    ).fillna(0.0).clip(0.0, 1.0)
    out["first_touch_same_bar"] = _safe_numeric(frame.get("__first_touch_same_bar_both__", 0.0)).fillna(0.0).clip(
        0.0,
        1.0,
    )
    out["first_touch_bar"] = _safe_numeric(frame.get("__first_touch_bar__", out["bars_policy"])).fillna(36.0).clip(
        lower=0.0
    )
    out["first_touch_mae_to_sl"] = _safe_numeric(frame.get("__first_touch_mae_to_sl__", out["mae_norm"])).fillna(
        10.0
    ).clip(lower=0.0)
    out["first_touch_mfe_to_tp"] = _safe_numeric(frame.get("__first_touch_mfe_to_tp__", out["mfe_norm"])).fillna(
        0.0
    ).clip(lower=0.0)
    out["first_touch_net_positive"] = _safe_numeric(
        frame.get("__first_touch_net_positive__", out["first_touch_net"] > 0.0)
    ).fillna(0.0).clip(0.0, 1.0)
    out["first_touch_valid_path"] = _safe_numeric(frame.get("__first_touch_valid_path__", 1.0)).fillna(1.0).clip(
        0.0,
        1.0,
    )
    out["first_touch_full_path_mae_to_sl"] = _safe_numeric(
        frame.get("__first_touch_full_path_mae_to_sl__", out["mae_norm"])
    ).fillna(10.0).clip(lower=0.0)
    out["clean_first_touch_exec"] = (
        (out["first_touch_net"] > 0.0)
        & (out["first_touch_hit"] > 0.5)
        & (out["first_touch_stop"] <= 0.5)
        & (out["first_touch_timeout"] <= 0.5)
        & (out["first_touch_same_bar"] <= 0.5)
        & (out["first_touch_mae_to_sl"] <= 1.0)
    ).astype(float)
    return out


def _selection_metrics(
    *,
    valid: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    pred: pd.Series,
    month: str,
    weight_arm: str,
    top_frac: float,
) -> dict[str, Any]:
    idx = _rank_top_indices(pred, top_frac)
    selected = valid.iloc[idx].copy() if len(idx) else valid.iloc[:0].copy()
    selected_metrics = metrics.iloc[idx].copy() if len(idx) else metrics.iloc[:0].copy()
    selected_target = target.iloc[idx].copy() if len(idx) else target.iloc[:0].copy()
    u_sel = _safe_numeric(selected_metrics["u_policy_net"])
    period_u = _safe_numeric(metrics["u_policy_net"])
    ft_sel = _safe_numeric(selected_metrics["first_touch_net"])
    period_ft = _safe_numeric(metrics["first_touch_net"])
    top_symbol_share = (
        float(selected["__symbol__"].astype(str).value_counts(normalize=True).iloc[0])
        if len(selected)
        else float("nan")
    )
    return {
        "weight_arm": str(weight_arm),
        "period": str(month),
        "top_frac": float(top_frac),
        "rows": int(len(valid)),
        "selected_rows": int(len(selected)),
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
        "target_top_soft_mean": _safe_mean(selected_target["target_soft"]),
        "target_top_hard_rate": _safe_mean(selected_target["target_hard"]),
        "score_ic_u": _spearman(pred, metrics["u_policy_net"]),
        "score_ic_first_touch_net": _spearman(pred, metrics["first_touch_net"]),
        "score_ic_label": _spearman(pred, target["target_soft"]),
        "decile_monotonicity_u": _decile_monotonicity(pred, metrics["u_policy_net"]),
        "decile_monotonicity_first_touch_net": _decile_monotonicity(pred, metrics["first_touch_net"]),
        "bad_mae_1r_rate": _safe_mean(selected_metrics["mae_norm"] >= 1.0),
        "wide_barrier_25bps_rate": _safe_mean(selected_metrics["barrier"] >= 0.025),
        "clean_first_touch_exec_rate": _safe_mean(selected_metrics["clean_first_touch_exec"]),
        "first_touch_hit_rate": _safe_mean(selected_metrics["first_touch_hit"]),
        "first_touch_stop_rate": _safe_mean(selected_metrics["first_touch_stop"]),
        "first_touch_timeout_rate": _safe_mean(selected_metrics["first_touch_timeout"]),
        "first_touch_same_bar_rate": _safe_mean(selected_metrics["first_touch_same_bar"]),
        "first_touch_valid_path_rate": _safe_mean(selected_metrics["first_touch_valid_path"]),
        "first_touch_bad_mae_to_sl_rate": _safe_mean(selected_metrics["first_touch_mae_to_sl"] >= 1.0),
        "p90_first_touch_mae_to_sl": _safe_quantile(selected_metrics["first_touch_mae_to_sl"], 0.90),
        "p90_first_touch_bar": _safe_quantile(selected_metrics["first_touch_bar"], 0.90),
        "p90_full_path_mae_to_sl": _safe_quantile(selected_metrics["first_touch_full_path_mae_to_sl"], 0.90),
        "top_symbol_share": top_symbol_share,
    }


def _fit_predict(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=180,
        max_depth=8,
        min_samples_leaf=32,
        max_features="sqrt",
        random_state=int(seed),
        n_jobs=2,
    )
    model.fit(
        x_train,
        _safe_numeric(y_train).fillna(0.0).to_numpy(dtype=np.float32),
        sample_weight=_safe_numeric(w_train).fillna(1.0).to_numpy(dtype=np.float32),
    )
    return model.predict(x_valid).astype(np.float32)


def _run_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    month: str,
    weight_arms: list[str],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < month
    valid_mask = month_period == month
    if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
        return [], [{"period": month, "skipped": True, "train_rows": int(train_mask.sum()), "valid_rows": int(valid_mask.sum())}]

    x = frame[features].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    med = x.loc[train_mask].median(numeric_only=True)
    x = x.fillna(med).fillna(0.0).astype(np.float32, copy=False)
    train = frame.loc[train_mask].copy()
    train_metrics = metrics.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    valid_target = target.loc[valid_mask].copy().reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for weight_arm in weight_arms:
        train_target = target.loc[train_mask].copy()
        weights = _weight_series(
            frame=train,
            metrics=train_metrics,
            target=train_target,
            arm=weight_arm,
        )
        preds = [
            _fit_predict(
                x_train=x.loc[train_mask],
                y_train=train_target["target_soft"],
                w_train=weights,
                x_valid=x.loc[valid_mask],
                seed=seed,
            )
            for seed in seeds
        ]
        pred = pd.Series(np.mean(np.vstack(preds), axis=0).astype(np.float32))
        seed_std = float(np.mean(np.std(np.vstack(preds), axis=0))) if len(preds) > 1 else 0.0
        diagnostics.append(
            {
                "period": str(month),
                "weight_arm": str(weight_arm),
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "target_train_mean": _safe_mean(train_target["target_soft"]),
                "target_valid_mean": _safe_mean(valid_target["target_soft"]),
                "weight_mean": _safe_mean(weights),
                "weight_p90": _safe_quantile(weights, 0.90),
                "weight_effective_frac": _effective_sample_size(weights) / float(len(weights)) if len(weights) else float("nan"),
                "seed_std_mean": seed_std,
            }
        )
        for top_frac in TOP_FRACS:
            rows.append(
                _selection_metrics(
                    valid=valid,
                    metrics=valid_metrics,
                    target=valid_target,
                    pred=pred,
                    month=month,
                    weight_arm=weight_arm,
                    top_frac=top_frac,
                )
            )
    return rows, diagnostics


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if monthly.empty:
        return monthly
    for (weight_arm, top_frac), group in monthly.groupby(["weight_arm", "top_frac"], observed=True, dropna=False):
        mean_u = _safe_numeric(group["mean_u"])
        rows.append(
            {
                "weight_arm": str(weight_arm),
                "top_frac": float(top_frac),
                "months": int(group["period"].nunique()),
                "positive_months": int((mean_u > 0.0).sum()),
                "mean_u": _safe_mean(mean_u),
                "worst_month_mean_u": _safe_quantile(mean_u, 0.0),
                "hit_u": _safe_mean(group["hit_u"]),
                "q10_u": _safe_mean(group["q10_u"]),
                "delta_mean_u_vs_period": _safe_mean(group["delta_mean_u_vs_period"]),
                "positive_first_touch_months": int((_safe_numeric(group["mean_first_touch_net"]) > 0.0).sum()),
                "mean_first_touch_net": _safe_mean(group["mean_first_touch_net"]),
                "worst_month_first_touch_net": _safe_quantile(group["mean_first_touch_net"], 0.0),
                "hit_first_touch_net": _safe_mean(group["hit_first_touch_net"]),
                "q10_first_touch_net": _safe_mean(group["q10_first_touch_net"]),
                "delta_first_touch_net_vs_period": _safe_mean(group["delta_first_touch_net_vs_period"]),
                "score_ic_u": _safe_mean(group["score_ic_u"]),
                "score_ic_first_touch_net": _safe_mean(group["score_ic_first_touch_net"]),
                "score_ic_label": _safe_mean(group["score_ic_label"]),
                "decile_monotonicity_u": _safe_mean(group["decile_monotonicity_u"]),
                "decile_monotonicity_first_touch_net": _safe_mean(group["decile_monotonicity_first_touch_net"]),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "clean_first_touch_exec_rate": _safe_mean(group["clean_first_touch_exec_rate"]),
                "first_touch_hit_rate": _safe_mean(group["first_touch_hit_rate"]),
                "first_touch_stop_rate": _safe_mean(group["first_touch_stop_rate"]),
                "first_touch_timeout_rate": _safe_mean(group["first_touch_timeout_rate"]),
                "first_touch_same_bar_rate": _safe_mean(group["first_touch_same_bar_rate"]),
                "first_touch_bad_mae_to_sl_rate": _safe_mean(group["first_touch_bad_mae_to_sl_rate"]),
                "p90_first_touch_mae_to_sl": _safe_mean(group["p90_first_touch_mae_to_sl"]),
                "p90_first_touch_bar": _safe_mean(group["p90_first_touch_bar"]),
                "p90_full_path_mae_to_sl": _safe_mean(group["p90_full_path_mae_to_sl"]),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
                "mean_selected_rows": _safe_mean(group["selected_rows"]),
                "min_selected_rows": int(_safe_numeric(group["selected_rows"]).min()),
            }
        )
    return pd.DataFrame(rows).sort_values(["top_frac", "mean_u", "worst_month_mean_u"], ascending=[True, False, False])


def _table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[c for c in cols if c in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
    return view.to_markdown(index=False)


def run_smoke(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    weight_arms: list[str],
    seeds: list[int],
    target_mode: str,
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
    target = _target_from_frame(frame, metrics, target_mode=target_mode)
    features = _feature_columns(frame)
    missing_weights = sorted(set(weight_arms) - set(WEIGHT_ARMS))
    if missing_weights:
        raise ValueError(f"Unknown weight arms: {missing_weights}")
    months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())

    monthly_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for month in months[1:]:
        rows, diagnostics = _run_month(
            frame=frame,
            metrics=metrics,
            target=target,
            features=features,
            month=month,
            weight_arms=weight_arms,
            seeds=seeds,
        )
        monthly_rows.extend(rows)
        diagnostic_rows.extend(diagnostics)

    monthly = pd.DataFrame(monthly_rows)
    aggregate = _aggregate(monthly)
    diagnostics = pd.DataFrame(diagnostic_rows)
    paths = {
        "monthly": output_dir / "first_touch_training_smoke_monthly.csv",
        "aggregate": output_dir / "first_touch_training_smoke_aggregate.csv",
        "diagnostics": output_dir / "first_touch_training_smoke_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
        "markdown": output_dir / "first_touch_label_training_smoke.md",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    manifest = {
        "scope": "cheap_month_forward_model_smoke_not_full_policy_training",
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
        "weight_arms": list(weight_arms),
        "seeds": [int(seed) for seed in seeds],
        "target_mode": str(target_mode),
        "top_fracs": list(TOP_FRACS),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    cols = [
        "weight_arm",
        "top_frac",
        "months",
        "positive_months",
        "mean_u",
        "worst_month_mean_u",
        "hit_u",
        "q10_u",
        "delta_mean_u_vs_period",
        "positive_first_touch_months",
        "mean_first_touch_net",
        "worst_month_first_touch_net",
        "hit_first_touch_net",
        "q10_first_touch_net",
        "delta_first_touch_net_vs_period",
        "score_ic_u",
        "score_ic_first_touch_net",
        "score_ic_label",
        "decile_monotonicity_u",
        "decile_monotonicity_first_touch_net",
        "clean_first_touch_exec_rate",
        "first_touch_hit_rate",
        "first_touch_timeout_rate",
        "first_touch_bad_mae_to_sl_rate",
        "p90_first_touch_mae_to_sl",
        "p90_full_path_mae_to_sl",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
        "top_symbol_share",
        "mean_selected_rows",
        "min_selected_rows",
    ]
    lines = [
        "# First-Touch Label Training Smoke",
        "",
        "Scope: cheap month-forward ExtraTrees smoke. This is not full production training or final policy OOS.",
        "",
        f"Target mode: `{target_mode}`",
        "",
    ]
    for frac in TOP_FRACS:
        subset = aggregate[aggregate["top_frac"].eq(frac)].sort_values(
            ["mean_u", "worst_month_mean_u"],
            ascending=[False, False],
        )
        lines.extend([f"## Top {frac:.0%}", "", _table(subset, cols, limit=25), ""])
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Monthly: `{paths['monthly']}`",
            f"- Aggregate: `{paths['aggregate']}`",
            f"- Diagnostics: `{paths['diagnostics']}`",
            f"- Manifest: `{paths['manifest']}`",
        ]
    )
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_int_csv(value: str | None, default: tuple[int, ...]) -> list[int]:
    if value is None or not str(value).strip():
        return list(default)
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--weight-arms", default="W0_base,W7_timestamp_balanced,W8_combined_conservative")
    parser.add_argument("--seeds", default="42,7301,999")
    parser.add_argument(
        "--target-mode",
        choices=[
            "policy_soft",
            "target_soft",
            "exec_guarded_policy",
            "clean_exec",
            "fast_clean_exec",
            "tight_clean_exec",
            "fast_tight_clean_exec",
            "time_decay_policy",
        ],
        default="policy_soft",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_smoke(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        weight_arms=_parse_csv(args.weight_arms, ("W0_base", "W7_timestamp_balanced", "W8_combined_conservative")),
        seeds=_parse_int_csv(args.seeds, (42, 7301, 999)),
        target_mode=str(args.target_mode),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
