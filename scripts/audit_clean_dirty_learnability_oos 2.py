#!/usr/bin/env python3
"""Month-forward clean-vs-dirty positive-path learnability audit.

This is a diagnostic, not a selector. It asks whether pre-entry features can
separate profitable clean paths from profitable dirty paths under stable OOS
month-forward splits. If this fails, downstream selector blending is the wrong
repair and the label/TBM design needs to change.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _feature_columns,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
)
from scripts.run_label_feature_store_model_smoke import (  # noqa: E402
    DEFAULT_AE_GMM_STATE_FEATURE_MAX_ITER,
    DEFAULT_AE_GMM_STATE_FEATURE_MAX_TRAIN_ROWS,
    _append_fold_ae_gmm_state_features,
    _apply_evaluation_utility_column,
    _apply_spread_symbol_universe,
)
from extreme_price_movements.economic_target_optimizer import (  # noqa: E402
    EconomicTargetSpec,
    append_economic_target_columns,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/clean_dirty_learnability_oos_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_LABEL_VARIANTS = (
    "positive_clean_1r",
    "positive_clean_075r",
    "positive_fast_clean_1r",
    "positive_exec_admissible",
    "positive_exec_margin_stable",
    "positive_econ_sideaware",
    "positive_econ_side_resolution",
    "positive_econ_sideaware_exec_resolution",
    "positive_econ_sideaware_short_decisive",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or str(value).strip() == "":
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y.loc[valid].astype(int), s.loc[valid]))


def _safe_ap(y_true: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(average_precision_score(y.loc[valid].astype(int), s.loc[valid]))


def _safe_brier(y_true: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    s = pd.to_numeric(score, errors="coerce").clip(0.0, 1.0)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(brier_score_loss(y.loc[valid].astype(int), s.loc[valid]))


def _topk_precision_metrics(
    y_true: pd.Series,
    score: pd.Series,
    *,
    prefix: str = "",
    fractions: tuple[float, ...] = (0.30, 0.20, 0.10, 0.05),
) -> dict[str, float | int]:
    y = pd.to_numeric(y_true, errors="coerce").reset_index(drop=True)
    s = pd.to_numeric(score, errors="coerce").reset_index(drop=True)
    valid = y.notna() & s.notna()
    y = y.loc[valid].astype(float).reset_index(drop=True)
    s = s.loc[valid].astype(float).reset_index(drop=True)
    out: dict[str, float | int] = {}
    base_rate = _safe_mean(y)
    if len(y) == 0:
        for frac in fractions:
            tag = f"{int(round(frac * 100)):02d}"
            out[f"{prefix}top{tag}_rows"] = 0
            out[f"{prefix}top{tag}_clean_rate"] = float("nan")
            out[f"{prefix}top{tag}_clean_lift"] = float("nan")
            out[f"{prefix}top{tag}_rank_weighted_clean_rate"] = float("nan")
            out[f"{prefix}top{tag}_score_weighted_clean_rate"] = float("nan")
        return out
    order = np.argsort(-s.to_numpy(dtype=np.float64), kind="mergesort")
    for frac in fractions:
        tag = f"{int(round(frac * 100)):02d}"
        top_n = max(1, int(math.ceil(float(frac) * len(y))))
        idx = order[:top_n]
        top_y = y.iloc[idx].reset_index(drop=True)
        top_s = s.iloc[idx].reset_index(drop=True)
        rank_weights = pd.Series(np.arange(top_n, 0, -1, dtype=np.float64))
        shifted_score_weights = top_s - float(top_s.min()) + 1e-9
        if not np.isfinite(shifted_score_weights).all() or float(shifted_score_weights.sum()) <= 1e-12:
            shifted_score_weights = rank_weights
        clean_rate = _safe_mean(top_y)
        out[f"{prefix}top{tag}_rows"] = int(top_n)
        out[f"{prefix}top{tag}_clean_rate"] = clean_rate
        out[f"{prefix}top{tag}_clean_lift"] = (
            clean_rate / base_rate if base_rate and math.isfinite(base_rate) else float("nan")
        )
        out[f"{prefix}top{tag}_rank_weighted_clean_rate"] = float(
            np.average(top_y.to_numpy(dtype=np.float64), weights=rank_weights.to_numpy(dtype=np.float64))
        )
        out[f"{prefix}top{tag}_score_weighted_clean_rate"] = float(
            np.average(
                top_y.to_numpy(dtype=np.float64),
                weights=shifted_score_weights.to_numpy(dtype=np.float64),
            )
        )
    return out


def _posterior_regime(frame: pd.DataFrame, side: pd.Series) -> pd.Series:
    side_s = pd.to_numeric(side.reset_index(drop=True), errors="coerce").fillna(1.0)
    out = pd.Series("unknown", index=frame.reset_index(drop=True).index, dtype=object)
    global_cols = [c for c in frame.columns if str(c).startswith("gmm_cluster_posterior_")]
    long_cols = [c for c in frame.columns if str(c).startswith("long_gmm_cluster_posterior_")]
    short_cols = [c for c in frame.columns if str(c).startswith("short_gmm_cluster_posterior_")]
    for side_name, mask, cols in (
        ("long", side_s.ge(0.0).to_numpy(dtype=bool), long_cols or global_cols),
        ("short", side_s.lt(0.0).to_numpy(dtype=bool), short_cols or global_cols),
    ):
        if not cols or not bool(mask.any()):
            continue
        values = frame.reset_index(drop=True).loc[mask, cols].apply(pd.to_numeric, errors="coerce")
        valid_rows = values.notna().any(axis=1).to_numpy(dtype=bool)
        if not bool(valid_rows.any()):
            continue
        codes = np.argmax(values.fillna(-np.inf).to_numpy(dtype=np.float32), axis=1)
        idx = np.flatnonzero(mask)
        out.iloc[idx[valid_rows]] = [f"{side_name}_regime_{int(code)}" for code in codes[valid_rows]]
    return out


def _spread_bucket(frame: pd.DataFrame) -> pd.Series:
    if "median_spread_bps" not in frame.columns:
        return pd.Series("spread_unknown", index=frame.index, dtype=object)
    values = pd.to_numeric(frame["median_spread_bps"], errors="coerce")
    if int(values.notna().sum()) < 50:
        return pd.Series("spread_unknown", index=frame.index, dtype=object)
    try:
        return pd.qcut(values.rank(method="average"), q=5, labels=[f"spread_q{i}" for i in range(5)]).astype(str)
    except ValueError:
        return pd.Series("spread_unknown", index=frame.index, dtype=object)


def _numeric_quantile_bucket(frame: pd.DataFrame, column: str, prefix: str, q: int = 5) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(f"{prefix}_unknown", index=frame.index, dtype=object)
    values = pd.to_numeric(frame[column], errors="coerce")
    if int(values.notna().sum()) < 50:
        return pd.Series(f"{prefix}_unknown", index=frame.index, dtype=object)
    try:
        return pd.qcut(
            values.rank(method="average"),
            q=q,
            labels=[f"{prefix}_q{i}" for i in range(q)],
        ).astype(str)
    except ValueError:
        return pd.Series(f"{prefix}_unknown", index=frame.index, dtype=object)


def _categorical_bucket(frame: pd.DataFrame, column: str, prefix: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(f"{prefix}_unknown", index=frame.index, dtype=object)
    values = pd.to_numeric(frame[column], errors="coerce")
    out = pd.Series(f"{prefix}_unknown", index=frame.index, dtype=object)
    valid = values.notna()
    out.loc[valid] = [f"{prefix}_{int(value)}" for value in values.loc[valid].astype(int)]
    return out


def _slice_rows(
    *,
    month: str,
    label_variant: str,
    split: str,
    valid: pd.DataFrame,
    y_valid: pd.Series,
    score: pd.Series,
    min_slice_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    side = pd.to_numeric(valid["side"], errors="coerce").fillna(1.0)
    slices: list[tuple[str, str, pd.Series]] = [
        ("all", "all", pd.Series(True, index=valid.index)),
        ("side", "long", side.ge(0.0)),
        ("side", "short", side.lt(0.0)),
    ]
    spread = _spread_bucket(valid)
    for value in sorted(spread.dropna().unique()):
        slices.append(("spread_bucket", str(value), spread.eq(value)))
    barrier = _numeric_quantile_bucket(valid, "__barrier_pct__", "barrier")
    for value in sorted(barrier.dropna().unique()):
        slices.append(("barrier_bucket", str(value), barrier.eq(value)))
    holding_column = "__bars_policy__" if "__bars_policy__" in valid.columns else "__bars_to_mfe__"
    holding = _numeric_quantile_bucket(valid, holding_column, "holding")
    for value in sorted(holding.dropna().unique()):
        slices.append(("holding_bucket", str(value), holding.eq(value)))
    geometry = _categorical_bucket(valid, "__econ_sideaware_execres_geometry_bucket__", "geometry")
    for value in sorted(geometry.dropna().unique()):
        slices.append(("exec_geometry", str(value), geometry.eq(value)))
    reason = _categorical_bucket(valid, "__econ_sideaware_execres_reason_code__", "reason")
    for value in sorted(reason.dropna().unique()):
        slices.append(("exec_reason", str(value), reason.eq(value)))
    regime = _posterior_regime(valid, side)
    for value in sorted(regime.dropna().unique()):
        slices.append(("ae_gmm_regime", str(value), regime.eq(value)))
    for bucket_name, bucket_values in (
        ("spread_bucket", spread),
        ("barrier_bucket", barrier),
        ("holding_bucket", holding),
        ("exec_geometry", geometry),
        ("exec_reason", reason),
    ):
        for side_name, side_mask in (("long", side.ge(0.0)), ("short", side.lt(0.0))):
            for value in sorted(bucket_values.dropna().unique()):
                slices.append(
                    (
                        f"side_x_{bucket_name}",
                        f"{side_name}|{value}",
                        side_mask & bucket_values.eq(value),
                    )
                )
    for slice_type, slice_value, mask in slices:
        mask = mask.fillna(False).astype(bool)
        if int(mask.sum()) < int(min_slice_rows):
            continue
        y_local = y_valid.loc[mask].reset_index(drop=True)
        s_local = score.loc[mask].reset_index(drop=True)
        clean_count = int(y_local.sum())
        dirty_count = int((1 - y_local).sum())
        score_gap = (
            _safe_mean(s_local.loc[y_local.astype(bool)])
            - _safe_mean(s_local.loc[~y_local.astype(bool)])
        )
        top_metrics = _topk_precision_metrics(y_local, s_local)
        row = (
            {
                "month": month,
                "label_variant": label_variant,
                "split": split,
                "slice_type": slice_type,
                "slice_value": slice_value,
                "rows": int(mask.sum()),
                "clean_rows": clean_count,
                "dirty_rows": dirty_count,
                "clean_rate": float(clean_count / max(clean_count + dirty_count, 1)),
                "roc_auc": _safe_auc(y_local, s_local),
                "average_precision": _safe_ap(y_local, s_local),
                "brier": _safe_brier(y_local, s_local),
                "score_gap_clean_minus_dirty": score_gap,
            }
        )
        row.update(top_metrics)
        rows.append(row)
    return rows


def _fit_predict_clean_head(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    seed: int,
    model_kind: str,
    lgbm_hpo: bool,
) -> tuple[pd.Series, dict[str, Any]]:
    y = pd.to_numeric(y_train.reset_index(drop=True), errors="coerce").fillna(0).astype(int)
    if int(y.sum()) < 100 or int((1 - y).sum()) < 100:
        return pd.Series(np.nan, index=x_valid.reset_index(drop=True).index, dtype=np.float32), {
            "model_status": "insufficient_class_rows",
            "train_clean_rows": int(y.sum()),
            "train_dirty_rows": int((1 - y).sum()),
        }
    kind = str(model_kind or "extratrees").strip().lower()
    if kind in {"lgbm", "lightgbm"}:
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("LightGBM model_kind requested but lightgbm is unavailable") from exc
        base_params: dict[str, Any] = {
            "objective": "binary",
            "n_estimators": 260,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "min_child_samples": 80,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "class_weight": "balanced",
            "random_state": int(seed),
            "n_jobs": 2,
            "verbose": -1,
        }
        selected_params = dict(base_params)
        hpo_diag: dict[str, Any] = {
            "hpo_enabled": bool(lgbm_hpo),
            "hpo_status": "disabled",
            "hpo_trials": 0,
        }
        if bool(lgbm_hpo) and len(y) >= 2500:
            split = int(math.floor(0.75 * len(y)))
            split = min(max(split, 1000), len(y) - 500)
            y_inner_train = y.iloc[:split].reset_index(drop=True)
            y_inner_valid = y.iloc[split:].reset_index(drop=True)
            if (
                int(y_inner_train.sum()) >= 100
                and int((1 - y_inner_train).sum()) >= 100
                and int(y_inner_valid.sum()) >= 50
                and int((1 - y_inner_valid).sum()) >= 50
            ):
                grid: list[dict[str, Any]] = [
                    {},
                    {"num_leaves": 15, "min_child_samples": 60, "learning_rate": 0.035, "reg_lambda": 1.5},
                    {"num_leaves": 31, "min_child_samples": 40, "learning_rate": 0.025, "reg_lambda": 1.0},
                    {"num_leaves": 31, "min_child_samples": 120, "learning_rate": 0.040, "reg_lambda": 2.0},
                    {"num_leaves": 63, "min_child_samples": 100, "learning_rate": 0.025, "reg_alpha": 0.10, "reg_lambda": 2.5},
                    {"num_leaves": 15, "min_child_samples": 160, "learning_rate": 0.050, "colsample_bytree": 0.65},
                ]
                best_score = float("-inf")
                best_trial: dict[str, Any] | None = None
                for trial_id, overrides in enumerate(grid):
                    params = dict(base_params)
                    params.update(overrides)
                    params["random_state"] = int(seed) + 1000 + int(trial_id)
                    model_trial = LGBMClassifier(**params)
                    model_trial.fit(
                        x_train.iloc[:split].reset_index(drop=True),
                        y_inner_train,
                    )
                    trial_pred = pd.Series(
                        model_trial.predict_proba(x_train.iloc[split:].reset_index(drop=True))[:, 1],
                        dtype=np.float32,
                    )
                    auc = _safe_auc(y_inner_valid, trial_pred)
                    gap = (
                        _safe_mean(trial_pred.loc[y_inner_valid.astype(bool)])
                        - _safe_mean(trial_pred.loc[~y_inner_valid.astype(bool)])
                    )
                    ap = _safe_ap(y_inner_valid, trial_pred)
                    if not math.isfinite(auc) or not math.isfinite(gap):
                        score = float("-inf")
                    else:
                        score = float(auc) + 2.0 * max(float(gap), -0.25)
                    if score > best_score:
                        best_score = score
                        best_trial = {
                            "trial_id": int(trial_id),
                            "score": float(score),
                            "auc": float(auc),
                            "gap": float(gap),
                            "average_precision": float(ap),
                            "params": params,
                        }
                if best_trial is not None and math.isfinite(float(best_trial["score"])):
                    selected_params = dict(best_trial["params"])
                    selected_params["random_state"] = int(seed)
                    hpo_diag.update(
                        {
                            "hpo_status": "ok",
                            "hpo_trials": len(grid),
                            "hpo_inner_train_rows": int(split),
                            "hpo_inner_valid_rows": int(len(y) - split),
                            "hpo_best_trial": int(best_trial["trial_id"]),
                            "hpo_best_score": float(best_trial["score"]),
                            "hpo_best_auc": float(best_trial["auc"]),
                            "hpo_best_gap": float(best_trial["gap"]),
                            "hpo_best_average_precision": float(best_trial["average_precision"]),
                            "hpo_selected_num_leaves": int(selected_params.get("num_leaves", -1)),
                            "hpo_selected_min_child_samples": int(selected_params.get("min_child_samples", -1)),
                            "hpo_selected_learning_rate": float(selected_params.get("learning_rate", float("nan"))),
                            "hpo_selected_reg_alpha": float(selected_params.get("reg_alpha", float("nan"))),
                            "hpo_selected_reg_lambda": float(selected_params.get("reg_lambda", float("nan"))),
                            "hpo_selected_colsample_bytree": float(selected_params.get("colsample_bytree", float("nan"))),
                        }
                    )
                else:
                    hpo_diag["hpo_status"] = "no_finite_trial"
            else:
                hpo_diag.update(
                    {
                        "hpo_status": "insufficient_inner_class_rows",
                        "hpo_inner_train_rows": int(split),
                        "hpo_inner_valid_rows": int(len(y) - split),
                    }
                )
        elif bool(lgbm_hpo):
            hpo_diag["hpo_status"] = "insufficient_rows"
        model = LGBMClassifier(**selected_params)
    else:
        hpo_diag = {"hpo_enabled": False, "hpo_status": "not_lgbm", "hpo_trials": 0}
        model = ExtraTreesClassifier(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=50,
            max_features="sqrt",
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=2,
        )
    model.fit(x_train.reset_index(drop=True), y)
    pred = model.predict_proba(x_valid.reset_index(drop=True))[:, 1]
    importances = pd.Series(model.feature_importances_, index=x_train.columns).sort_values(ascending=False)
    return pd.Series(pred.astype(np.float32), index=x_valid.reset_index(drop=True).index), {
        "model_status": "ok",
        "model_kind": kind,
        "train_clean_rows": int(y.sum()),
        "train_dirty_rows": int((1 - y).sum()),
        "top_features": ",".join(importances.head(20).index.astype(str).tolist()),
        "top_feature_importance": float(importances.iloc[0]) if len(importances) else float("nan"),
        **hpo_diag,
    }


def _select_univariate_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    top_k: int,
) -> tuple[list[str], dict[str, Any]]:
    cols = [str(c) for c in x_train.columns]
    if int(top_k) <= 0 or int(top_k) >= len(cols):
        return cols, {
            "feature_select_enabled": False,
            "feature_select_status": "disabled",
            "feature_select_input_count": int(len(cols)),
            "feature_select_selected_count": int(len(cols)),
        }
    y = pd.to_numeric(y_train.reset_index(drop=True), errors="coerce").fillna(0).astype(int)
    if int(y.sum()) < 50 or int((1 - y).sum()) < 50:
        return cols, {
            "feature_select_enabled": True,
            "feature_select_status": "insufficient_class_rows_keep_all",
            "feature_select_input_count": int(len(cols)),
            "feature_select_selected_count": int(len(cols)),
        }
    scores: list[tuple[float, str, float]] = []
    for col in cols:
        s = pd.to_numeric(x_train[col], errors="coerce")
        valid = s.notna() & np.isfinite(s.to_numpy(dtype=np.float64, copy=False))
        if int(valid.sum()) < 100 or s.loc[valid].nunique(dropna=True) < 2:
            continue
        auc = _safe_auc(y.loc[valid].reset_index(drop=True), s.loc[valid].reset_index(drop=True))
        if not math.isfinite(auc):
            continue
        scores.append((abs(float(auc) - 0.5), col, float(auc)))
    if not scores:
        return cols, {
            "feature_select_enabled": True,
            "feature_select_status": "no_scored_features_keep_all",
            "feature_select_input_count": int(len(cols)),
            "feature_select_selected_count": int(len(cols)),
        }
    scores.sort(key=lambda item: item[0], reverse=True)
    selected = [col for _score, col, _auc in scores[: int(top_k)]]
    return selected, {
        "feature_select_enabled": True,
        "feature_select_status": "ok",
        "feature_select_input_count": int(len(cols)),
        "feature_select_selected_count": int(len(selected)),
        "feature_select_top_features": ",".join(selected[:20]),
        "feature_select_top_abs_auc_edge": float(scores[0][0]),
        "feature_select_top_auc": float(scores[0][2]),
    }


def _side_values(frame: pd.DataFrame) -> pd.Series:
    if "side" in frame.columns:
        raw = pd.to_numeric(frame["side"], errors="coerce")
    elif "__side__" in frame.columns:
        raw = pd.to_numeric(frame["__side__"], errors="coerce")
    else:
        raw = pd.Series(1.0, index=frame.index, dtype=np.float64)
    return pd.Series(np.where(raw.fillna(1.0) < 0.0, -1, 1), index=frame.index)


def _fit_predict_clean_head_by_side(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    train_side: pd.Series,
    x_valid: pd.DataFrame,
    valid_side: pd.Series,
    seed: int,
    model_kind: str,
    lgbm_hpo: bool,
    side_feature_select_top_k: int,
) -> tuple[pd.Series, dict[str, Any]]:
    pred = pd.Series(np.nan, index=x_valid.reset_index(drop=True).index, dtype=np.float32)
    diag: dict[str, Any] = {
        "fit_by_side": True,
        "model_status": "side_split_no_successful_side",
    }
    ok_sides = 0
    for offset, (side_name, side_sign) in enumerate((("long", 1), ("short", -1))):
        tr_mask = train_side.reset_index(drop=True).eq(side_sign)
        va_mask = valid_side.reset_index(drop=True).eq(side_sign)
        diag[f"{side_name}_train_rows"] = int(tr_mask.sum())
        diag[f"{side_name}_valid_rows"] = int(va_mask.sum())
        if int(tr_mask.sum()) < 200 or int(va_mask.sum()) < 100:
            diag[f"{side_name}_model_status"] = "skipped_insufficient_side_rows"
            continue
        x_train_side = x_train.loc[tr_mask].reset_index(drop=True)
        x_valid_side = x_valid.loc[va_mask].reset_index(drop=True)
        y_train_side = y_train.loc[tr_mask].reset_index(drop=True)
        selected_features, fs_diag = _select_univariate_features(
            x_train_side,
            y_train_side,
            top_k=int(side_feature_select_top_k),
        )
        score_side, side_diag = _fit_predict_clean_head(
            x_train=x_train_side[selected_features].reset_index(drop=True),
            y_train=y_train_side,
            x_valid=x_valid_side[selected_features].reset_index(drop=True),
            seed=int(seed) + offset + 1,
            model_kind=model_kind,
            lgbm_hpo=lgbm_hpo,
        )
        diag[f"{side_name}_model_status"] = side_diag.get("model_status")
        diag[f"{side_name}_train_clean_rows"] = side_diag.get("train_clean_rows")
        diag[f"{side_name}_train_dirty_rows"] = side_diag.get("train_dirty_rows")
        diag[f"{side_name}_top_features"] = side_diag.get("top_features")
        diag[f"{side_name}_top_feature_importance"] = side_diag.get("top_feature_importance")
        for key, value in fs_diag.items():
            diag[f"{side_name}_{key}"] = value
        for key, value in side_diag.items():
            if str(key).startswith("hpo_"):
                diag[f"{side_name}_{key}"] = value
        if side_diag.get("model_status") == "ok":
            pred.loc[va_mask.to_numpy(dtype=bool)] = score_side.to_numpy(copy=False)
            ok_sides += 1
    if ok_sides:
        diag["model_status"] = "ok"
    return pred, diag


def _apply_short_regime_abstention(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    x_train_model: pd.DataFrame,
    x_valid_model: pd.DataFrame,
    train_pop: np.ndarray,
    valid_pop: np.ndarray,
    train_label: pd.Series,
    min_rows: int,
    min_clean_rate: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_pop_s = pd.Series(np.asarray(train_pop, dtype=bool), index=train_frame.reset_index(drop=True).index)
    valid_pop_s = pd.Series(np.asarray(valid_pop, dtype=bool), index=valid_frame.reset_index(drop=True).index)
    train_side = _side_values(train_frame).reset_index(drop=True)
    valid_side = _side_values(valid_frame).reset_index(drop=True)
    train_regime = _posterior_regime(x_train_model.reset_index(drop=True), train_side)
    valid_regime = _posterior_regime(x_valid_model.reset_index(drop=True), valid_side)
    label = pd.to_numeric(train_label.reset_index(drop=True), errors="coerce").fillna(0).astype(int)

    short_train_pop = train_pop_s & train_side.lt(0)
    short_valid_pop = valid_pop_s & valid_side.lt(0)
    rows: list[dict[str, Any]] = []
    for regime, idx in train_regime.loc[short_train_pop].groupby(train_regime.loc[short_train_pop], dropna=False).groups.items():
        regime_s = str(regime)
        local_idx = list(idx)
        if regime_s == "unknown" or len(local_idx) == 0:
            continue
        clean_rate = float(label.iloc[local_idx].mean())
        rows.append(
            {
                "regime": regime_s,
                "rows": int(len(local_idx)),
                "clean_rows": int(label.iloc[local_idx].sum()),
                "clean_rate": clean_rate,
            }
        )
    stats = pd.DataFrame(rows)
    if stats.empty:
        return train_pop_s.to_numpy(dtype=bool), valid_pop_s.to_numpy(dtype=bool), {
            "short_regime_abstain_status": "no_regime_stats_keep_all",
            "short_regime_abstain_enabled": True,
            "short_regime_abstain_allowed": "",
            "short_regime_abstain_train_short_before": int(short_train_pop.sum()),
            "short_regime_abstain_train_short_after": int(short_train_pop.sum()),
            "short_regime_abstain_valid_short_before": int(short_valid_pop.sum()),
            "short_regime_abstain_valid_short_after": int(short_valid_pop.sum()),
        }
    allowed = stats.loc[
        stats["rows"].ge(int(min_rows))
        & stats["clean_rate"].ge(float(min_clean_rate)),
        "regime",
    ].astype(str).tolist()
    if not allowed:
        return train_pop_s.to_numpy(dtype=bool), valid_pop_s.to_numpy(dtype=bool), {
            "short_regime_abstain_status": "no_allowed_regime_keep_all",
            "short_regime_abstain_enabled": True,
            "short_regime_abstain_allowed": "",
            "short_regime_abstain_min_rows": int(min_rows),
            "short_regime_abstain_min_clean_rate": float(min_clean_rate),
            "short_regime_abstain_train_short_before": int(short_train_pop.sum()),
            "short_regime_abstain_train_short_after": int(short_train_pop.sum()),
            "short_regime_abstain_valid_short_before": int(short_valid_pop.sum()),
            "short_regime_abstain_valid_short_after": int(short_valid_pop.sum()),
            "short_regime_abstain_stats": stats.to_dict(orient="records"),
        }

    train_keep_short = train_regime.astype(str).isin(allowed)
    valid_keep_short = valid_regime.astype(str).isin(allowed)
    train_pop_out = train_pop_s & (train_side.ge(0) | train_keep_short)
    valid_pop_out = valid_pop_s & (valid_side.ge(0) | valid_keep_short)
    train_short_after = int((train_pop_out & train_side.lt(0)).sum())
    valid_short_after = int((valid_pop_out & valid_side.lt(0)).sum())
    if train_short_after < 200 or valid_short_after < 100:
        return train_pop_s.to_numpy(dtype=bool), valid_pop_s.to_numpy(dtype=bool), {
            "short_regime_abstain_status": "filtered_too_sparse_keep_all",
            "short_regime_abstain_enabled": True,
            "short_regime_abstain_allowed": ",".join(allowed),
            "short_regime_abstain_min_rows": int(min_rows),
            "short_regime_abstain_min_clean_rate": float(min_clean_rate),
            "short_regime_abstain_train_short_before": int(short_train_pop.sum()),
            "short_regime_abstain_train_short_after": train_short_after,
            "short_regime_abstain_valid_short_before": int(short_valid_pop.sum()),
            "short_regime_abstain_valid_short_after": valid_short_after,
            "short_regime_abstain_stats": stats.to_dict(orient="records"),
        }
    return train_pop_out.to_numpy(dtype=bool), valid_pop_out.to_numpy(dtype=bool), {
        "short_regime_abstain_status": "ok",
        "short_regime_abstain_enabled": True,
        "short_regime_abstain_allowed": ",".join(allowed),
        "short_regime_abstain_min_rows": int(min_rows),
        "short_regime_abstain_min_clean_rate": float(min_clean_rate),
        "short_regime_abstain_train_short_before": int(short_train_pop.sum()),
        "short_regime_abstain_train_short_after": train_short_after,
        "short_regime_abstain_valid_short_before": int(short_valid_pop.sum()),
        "short_regime_abstain_valid_short_after": valid_short_after,
        "short_regime_abstain_stats": stats.to_dict(orient="records"),
    }


def _positive_clean_variant(
    metrics: pd.DataFrame,
    *,
    variant: str,
    frame: pd.DataFrame | None = None,
) -> tuple[np.ndarray, pd.Series, dict[str, Any]]:
    index = metrics.index
    if variant in {
        "positive_econ_sideaware",
        "positive_econ_side_resolution",
        "positive_econ_sideaware_exec_resolution",
        "positive_econ_sideaware_short_decisive",
    }:
        if frame is None:
            raise ValueError(f"{variant} requires the source frame")
        target_frame, summary = append_economic_target_columns(
            frame.reset_index(drop=True),
            EconomicTargetSpec(
                name="oos_sideaware_learnability_probe",
                utility_source="policy_net",
                cost=0.001,
                margin=0.0005,
                sl_buffer=0.1,
                vol_source="barrier",
                temperature=0.75,
            ),
            copy=True,
        )
        if variant in {"positive_econ_side_resolution", "positive_econ_sideaware_exec_resolution"}:
            if variant == "positive_econ_sideaware_exec_resolution":
                net_col = "__u_econ_sideaware_execres_net__"
                clean_col = "__econ_sideaware_execres_clean__"
                hard_col = "__y_econ_sideaware_execres_bin__"
            else:
                net_col = "__u_econ_side_resolution_net__"
                clean_col = "__econ_side_resolution_clean__"
                hard_col = "__y_econ_side_resolution_bin__"
            net = pd.to_numeric(
                target_frame[net_col],
                errors="coerce",
            ).fillna(0.0)
            clean = pd.to_numeric(
                target_frame[clean_col],
                errors="coerce",
            ).fillna(0.0).gt(0.5)
            hard = pd.to_numeric(
                target_frame[hard_col],
                errors="coerce",
            ).fillna(0.0).gt(0.5)
        else:
            net = pd.to_numeric(target_frame["__u_econ_sideaware_net__"], errors="coerce").fillna(0.0)
            clean = pd.to_numeric(target_frame["__econ_sideaware_clean__"], errors="coerce").fillna(0.0).gt(0.5)
            hard = pd.to_numeric(target_frame["__y_econ_sideaware_bin__"], errors="coerce").fillna(0.0).gt(0.5)
        positive = net.gt(0.0)
        if variant == "positive_econ_sideaware_short_decisive":
            metric_local = metrics.reset_index(drop=True)
            side = pd.to_numeric(metric_local["side"], errors="coerce").fillna(1.0)
            mae_norm = pd.to_numeric(metric_local["mae_norm"], errors="coerce").fillna(99.0)
            mfe_norm = pd.to_numeric(metric_local["mfe_norm"], errors="coerce").fillna(0.0)
            bars_policy = pd.to_numeric(metric_local["bars_policy"], errors="coerce").fillna(999.0)
            bars_to_mfe = pd.to_numeric(metric_local["bars_to_mfe"], errors="coerce").fillna(999.0)
            timeout = pd.to_numeric(metric_local["is_timeout"], errors="coerce").fillna(1.0).gt(0.5)
            mfe_mae = (mfe_norm / mae_norm.clip(lower=0.25)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            is_short = side.lt(0.0)
            short_decisive = (
                is_short
                & net.gt(0.0015)
                & (~timeout)
                & bars_policy.ge(4.0)
                & bars_policy.le(10.0)
                & bars_to_mfe.le(10.0)
                & mae_norm.le(0.50)
                & mfe_norm.ge(1.75)
                & mfe_mae.ge(2.00)
            )
            hard = hard.where(~is_short, short_decisive).astype(bool)
            clean = clean.where(~is_short, short_decisive).astype(bool)
        y = hard.astype(int)
        description = (
            "side-aware executable economic target: strict long high-barrier/late-path "
            "demotion and short 4-14 bar clean-path preservation"
        )
        if variant == "positive_econ_side_resolution":
            description = (
                "side-resolution executable target: long positives require fast non-timeout "
                "resolution; short positives require tighter adverse-excursion cleanliness"
            )
        if variant == "positive_econ_sideaware_exec_resolution":
            description = (
                "side-aware executable-resolution target: long positives require fast "
                "non-timeout resolution; short positives preserve 4-12 bar clean "
                "opportunities while demoting dirty adverse-excursion paths"
            )
        if variant == "positive_econ_sideaware_short_decisive":
            description = (
                "side-aware executable target with decisive short clean labels: short "
                "positives require stronger net edge, low MAE, high MFE/MFE-MAE, and 4-10 bar resolution"
            )
        return positive.to_numpy(dtype=bool), y.reindex(index).astype(int), {
            "label_variant": variant,
            "label_description": description,
            "positive_rows": int(positive.sum()),
            "clean_rows": int(hard.sum()),
            "dirty_rows": int((positive & ~hard).sum()),
            "clean_rate": float(hard.sum() / max(int(positive.sum()), 1)),
            "path_clean_rows": int(clean.sum()),
            "path_clean_rate": float(clean.sum() / max(int(positive.sum()), 1)),
            "sideaware_soft_mean": summary.get("sideaware_soft_mean"),
            "sideaware_long_hard_rate": summary.get("sideaware_long_hard_rate"),
            "sideaware_short_hard_rate": summary.get("sideaware_short_hard_rate"),
            "side_resolution_soft_mean": summary.get("side_resolution_soft_mean"),
            "side_resolution_long_hard_rate": summary.get("side_resolution_long_hard_rate"),
            "side_resolution_short_hard_rate": summary.get("side_resolution_short_hard_rate"),
        }
    u = pd.to_numeric(metrics["u_policy_net"], errors="coerce").fillna(0.0)
    mae_norm = pd.to_numeric(metrics["mae_norm"], errors="coerce").fillna(10.0)
    mfe_norm = pd.to_numeric(metrics["mfe_norm"], errors="coerce").fillna(0.0)
    bars_to_mfe = pd.to_numeric(metrics["bars_to_mfe"], errors="coerce").fillna(10_000.0)
    timeout = pd.to_numeric(metrics["is_timeout"], errors="coerce").fillna(1.0).gt(0.5)
    mfe_mae = (mfe_norm / mae_norm.clip(lower=0.25)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    positive = u.gt(0.0)
    if variant == "positive_clean_1r":
        clean = positive & mae_norm.lt(1.0) & (~timeout)
        description = "current positive utility clean path: bad-MAE < 1R and no timeout"
    elif variant == "positive_clean_075r":
        clean = positive & mae_norm.lt(0.75) & (~timeout)
        description = "stricter positive clean path: bad-MAE < 0.75R and no timeout"
    elif variant == "positive_fast_clean_1r":
        clean = (
            positive
            & mae_norm.lt(1.0)
            & (~timeout)
            & mfe_norm.ge(1.0)
            & bars_to_mfe.le(12.0)
        )
        description = "positive clean path requiring favorable excursion >= 1R within 12 bars"
    elif variant == "positive_exec_admissible":
        clean = (
            u.gt(0.0005)
            & mae_norm.le(0.85)
            & (~timeout)
            & mfe_norm.ge(1.0)
            & mfe_mae.ge(1.25)
            & bars_to_mfe.le(14.0)
        )
        description = "execution-admissible path label with margin, speed, MFE/MAE, and no timeout"
    elif variant == "positive_exec_margin_stable":
        barrier = pd.to_numeric(metrics.get("barrier", pd.Series(0.02, index=index)), errors="coerce").fillna(0.02)
        exec_margin = (
            u
            - 0.0040 * (mae_norm - 0.65).clip(lower=0.0)
            - 0.0050 * mae_norm.ge(1.0).astype(float)
            - 0.0060 * timeout.astype(float)
            - 0.0010 * np.log1p(bars_to_mfe.clip(lower=0.0))
            - 0.75 * (barrier - 0.020).clip(lower=0.0)
            + 0.0015 * (mfe_mae - 1.25).clip(lower=0.0, upper=2.0)
        )
        clean = (
            positive
            & exec_margin.gt(0.0005)
            & mae_norm.le(0.85)
            & (~timeout)
            & mfe_norm.ge(1.0)
            & mfe_mae.ge(1.25)
            & bars_to_mfe.le(14.0)
        )
        description = "redesigned executable-margin-stable label with explicit path, speed, and barrier penalties"
    else:
        raise ValueError(f"Unknown label variant: {variant}")
    population = positive.to_numpy(dtype=bool)
    y = clean.reindex(index).astype(int)
    return population, y, {
        "label_variant": variant,
        "label_description": description,
        "positive_rows": int(positive.sum()),
        "clean_rows": int(clean.sum()),
        "dirty_rows": int((positive & ~clean).sum()),
        "clean_rate": float(clean.sum() / max(int(positive.sum()), 1)),
    }


def run_audit(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    output_dir: Path,
    months: list[str],
    label_variants: list[str],
    evaluation_utility_column: str | None,
    max_feature_store_features: int | None,
    spread_baseline_path: Path | None,
    spread_rank_column: str,
    target_symbol_count: int | None,
    max_spread_bps: float | None,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
    fit_by_side: bool,
    model_kind: str,
    lgbm_hpo: bool,
    side_feature_select_top_k: int,
    require_side_slice_pass: bool,
    side_slice_min_auc: float,
    short_regime_abstain: bool,
    short_regime_abstain_min_rows: int,
    short_regime_abstain_min_clean_rate: float,
    min_train_rows: int,
    min_valid_rows: int,
    min_slice_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    frame, symbol_universe_filter, symbol_universe = _apply_spread_symbol_universe(
        frame,
        spread_baseline_path=spread_baseline_path,
        spread_rank_column=spread_rank_column,
        target_symbol_count=target_symbol_count,
        max_spread_bps=max_spread_bps,
    )
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        frame = pd.concat(
            [frame.reset_index(drop=True), feature_matrix.astype(np.float32, copy=False).reset_index(drop=True)],
            axis=1,
            copy=False,
        )
    metrics = _path_metrics(frame)
    utility_source = _apply_evaluation_utility_column(frame, metrics, evaluation_utility_column)
    features = _feature_columns(frame)
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for month in months:
        train_mask = month_period < month
        valid_mask = month_period.eq(month)
        if int(train_mask.sum()) < int(min_train_rows) or int(valid_mask.sum()) < int(min_valid_rows):
            fold_rows.append(
                {
                    "month": month,
                    "model_status": "skipped_insufficient_rows",
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": int(valid_mask.sum()),
                }
            )
            continue
        x_train = frame.loc[train_mask, features].copy().reset_index(drop=True)
        x_valid = frame.loc[valid_mask, features].copy().reset_index(drop=True)
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy().reset_index(drop=True)
        train_metrics = metrics.loc[train_mask].copy().reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        x_train, x_valid, ae_features, ae_diag = _append_fold_ae_gmm_state_features(
            x_train=x_train,
            x_valid=x_valid,
            train_frame=train,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            enabled=True,
            max_train_rows=int(ae_gmm_state_feature_max_train_rows),
            gmm_max_train_rows=int(ae_gmm_state_feature_max_train_rows),
            ae_max_iter=int(ae_gmm_state_feature_max_iter),
            random_state=70001 + sum((i + 1) * ord(ch) for i, ch in enumerate(str(month))),
        )
        model_features = list(dict.fromkeys(features + list(ae_features)))
        x_train_model = x_train[model_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        x_valid_model = x_valid[model_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

        for label_variant in label_variants:
            train_pop, train_label, train_label_diag = _positive_clean_variant(
                train_metrics,
                variant=label_variant,
                frame=train,
            )
            valid_pop, valid_label, valid_label_diag = _positive_clean_variant(
                valid_metrics,
                variant=label_variant,
                frame=valid,
            )
            abstain_diag: dict[str, Any] = {
                "short_regime_abstain_enabled": bool(short_regime_abstain),
                "short_regime_abstain_status": "disabled",
            }
            if bool(short_regime_abstain):
                train_pop, valid_pop, abstain_diag = _apply_short_regime_abstention(
                    train_frame=train,
                    valid_frame=valid,
                    x_train_model=x_train_model,
                    x_valid_model=x_valid_model,
                    train_pop=train_pop,
                    valid_pop=valid_pop,
                    train_label=train_label,
                    min_rows=int(short_regime_abstain_min_rows),
                    min_clean_rate=float(short_regime_abstain_min_clean_rate),
                )
            y_train = train_label.iloc[train_pop].reset_index(drop=True)
            y_valid = valid_label.iloc[valid_pop].reset_index(drop=True)
            if int(train_pop.sum()) < int(min_train_rows) or int(valid_pop.sum()) < int(min_valid_rows):
                fold_rows.append(
                    {
                        "month": month,
                        "label_variant": label_variant,
                        "model_status": "skipped_insufficient_positive_utility_rows",
                        "train_rows": int(train_pop.sum()),
                        "valid_rows": int(valid_pop.sum()),
                        **{f"train_{k}": v for k, v in train_label_diag.items() if k not in {"label_variant"}},
                        **{f"valid_{k}": v for k, v in valid_label_diag.items() if k not in {"label_variant"}},
                        **abstain_diag,
                        **ae_diag,
                    }
                )
                continue
            x_train_pop = x_train_model.iloc[train_pop].reset_index(drop=True)
            x_valid_pop = x_valid_model.iloc[valid_pop].reset_index(drop=True)
            if fit_by_side:
                score, model_diag = _fit_predict_clean_head_by_side(
                    x_train=x_train_pop,
                    y_train=y_train,
                    train_side=_side_values(train.iloc[train_pop]).reset_index(drop=True),
                    x_valid=x_valid_pop,
                    valid_side=_side_values(valid.iloc[valid_pop]).reset_index(drop=True),
                    seed=42000 + len(fold_rows),
                    model_kind=model_kind,
                    lgbm_hpo=bool(lgbm_hpo),
                    side_feature_select_top_k=int(side_feature_select_top_k),
                )
            else:
                score, model_diag = _fit_predict_clean_head(
                    x_train=x_train_pop,
                    y_train=y_train,
                    x_valid=x_valid_pop,
                    seed=42000 + len(fold_rows),
                    model_kind=model_kind,
                    lgbm_hpo=bool(lgbm_hpo),
                )
            valid_pop_frame = pd.concat(
                [
                    valid.iloc[valid_pop].reset_index(drop=True),
                    x_valid_model.loc[valid_pop, ae_features].reset_index(drop=True),
                ],
                axis=1,
                copy=False,
            )
            if label_variant in {
                "positive_econ_side_resolution",
                "positive_econ_sideaware_exec_resolution",
            }:
                try:
                    valid_target_context, _context_summary = append_economic_target_columns(
                        valid.reset_index(drop=True),
                        EconomicTargetSpec(
                            name="oos_sideaware_learnability_probe_context",
                            utility_source="policy_net",
                            cost=0.001,
                            margin=0.0005,
                            sl_buffer=0.1,
                            vol_source="barrier",
                            temperature=0.75,
                        ),
                        copy=True,
                    )
                    context_cols = [
                        "__econ_sideaware_execres_reason_code__",
                        "__econ_sideaware_execres_geometry_bucket__",
                        "__econ_sideaware_execres_dirty_positive__",
                    ]
                    for col in context_cols:
                        if col in valid_target_context.columns:
                            valid_pop_frame[col] = (
                                valid_target_context.loc[valid_pop, col]
                                .reset_index(drop=True)
                                .to_numpy(copy=False)
                            )
                except Exception:
                    pass
            fold_topk_metrics = _topk_precision_metrics(y_valid, score)
            fold_summary = {
                "month": month,
                "label_variant": label_variant,
                "model_status": model_diag.get("model_status"),
                "population": "positive_utility_rows",
                "label_description": valid_label_diag["label_description"],
                "model_feature_count": int(len(model_features)),
                "ae_gmm_feature_count": int(len(ae_features)),
                "train_rows": int(train_pop.sum()),
                "valid_rows": int(valid_pop.sum()),
                "train_clean_rows": int(y_train.sum()),
                "train_dirty_rows": int((1 - y_train).sum()),
                "valid_clean_rows": int(y_valid.sum()),
                "valid_dirty_rows": int((1 - y_valid).sum()),
                "valid_clean_rate": _safe_mean(y_valid),
                "roc_auc": _safe_auc(y_valid, score),
                "average_precision": _safe_ap(y_valid, score),
                "brier": _safe_brier(y_valid, score),
                "score_gap_clean_minus_dirty": (
                    _safe_mean(score.loc[y_valid.astype(bool)])
                    - _safe_mean(score.loc[~y_valid.astype(bool)])
                ),
                **fold_topk_metrics,
                **model_diag,
                **abstain_diag,
                **ae_diag,
            }
            fold_rows.append(fold_summary)
            rows.extend(
                _slice_rows(
                    month=month,
                    label_variant=label_variant,
                    split="month_forward_oos",
                    valid=valid_pop_frame.reset_index(drop=True),
                    y_valid=y_valid.reset_index(drop=True),
                    score=score.reset_index(drop=True),
                    min_slice_rows=min_slice_rows,
                )
            )

    fold_df = pd.DataFrame(fold_rows)
    slice_df = pd.DataFrame(rows)
    paths = {
        "folds": output_dir / "clean_dirty_learnability_folds.csv",
        "slices": output_dir / "clean_dirty_learnability_slices.csv",
        "variant_summary": output_dir / "clean_dirty_learnability_variant_summary.csv",
        "manifest": output_dir / "manifest.json",
        "markdown": output_dir / "clean_dirty_learnability_oos.md",
    }
    fold_df.to_csv(paths["folds"], index=False)
    slice_df.to_csv(paths["slices"], index=False)
    if not fold_df.empty and {"label_variant", "roc_auc", "score_gap_clean_minus_dirty"}.issubset(fold_df.columns):
        scored_folds = fold_df.copy()
        scored_folds["fold_pass"] = (
            pd.to_numeric(scored_folds["roc_auc"], errors="coerce").ge(0.55)
            & pd.to_numeric(scored_folds["score_gap_clean_minus_dirty"], errors="coerce").gt(0.0)
        )
        summary_rows = []
        for label_variant, group in scored_folds.groupby("label_variant", dropna=False):
            def _group_numeric(column: str) -> pd.Series:
                if column not in group.columns:
                    return pd.Series(dtype=float)
                return pd.to_numeric(group[column], errors="coerce")

            side_local = (
                slice_df[
                    slice_df.get("label_variant", pd.Series(dtype=object)).astype(str).eq(str(label_variant))
                    & slice_df.get("slice_type", pd.Series(dtype=object)).astype(str).eq("side")
                ].copy()
                if not slice_df.empty and {"label_variant", "slice_type"}.issubset(slice_df.columns)
                else pd.DataFrame()
            )
            if not side_local.empty:
                side_auc = pd.to_numeric(side_local["roc_auc"], errors="coerce")
                side_gap = pd.to_numeric(side_local["score_gap_clean_minus_dirty"], errors="coerce")
                short_auc = pd.to_numeric(
                    side_local.loc[side_local["slice_value"].astype(str).eq("short"), "roc_auc"],
                    errors="coerce",
                )
                long_auc = pd.to_numeric(
                    side_local.loc[side_local["slice_value"].astype(str).eq("long"), "roc_auc"],
                    errors="coerce",
                )
                side_slices_total = int(side_auc.notna().sum())
                side_slices_pass = int((side_auc.ge(float(side_slice_min_auc)) & side_gap.gt(0.0)).sum())
                all_side_slices_pass = bool(
                    side_slices_total > 0
                    and side_slices_pass == side_slices_total
                )
                min_side_auc = float(side_auc.min())
                min_side_gap = float(side_gap.min())
                min_short_auc = float(short_auc.min()) if short_auc.notna().any() else float("nan")
                min_long_auc = float(long_auc.min()) if long_auc.notna().any() else float("nan")
            else:
                side_slices_total = 0
                side_slices_pass = 0
                all_side_slices_pass = False
                min_side_auc = float("nan")
                min_side_gap = float("nan")
                min_short_auc = float("nan")
                min_long_auc = float("nan")
            summary_rows.append(
                {
                    "label_variant": label_variant,
                    "folds_passed": int(group["fold_pass"].sum()),
                    "folds_total": int(group["fold_pass"].notna().sum()),
                    "all_folds_pass": bool(group["fold_pass"].all()) if len(group) else False,
                    "mean_auc": _safe_mean(pd.to_numeric(group["roc_auc"], errors="coerce")),
                    "min_auc": float(pd.to_numeric(group["roc_auc"], errors="coerce").min()),
                    "mean_score_gap": _safe_mean(pd.to_numeric(group["score_gap_clean_minus_dirty"], errors="coerce")),
                    "min_score_gap": float(pd.to_numeric(group["score_gap_clean_minus_dirty"], errors="coerce").min()),
                    "mean_clean_rate": _safe_mean(pd.to_numeric(group["valid_clean_rate"], errors="coerce")),
                    "mean_top30_clean_rate": _safe_mean(_group_numeric("top30_clean_rate")),
                    "min_top30_clean_rate": float(_group_numeric("top30_clean_rate").min()),
                    "mean_top20_clean_rate": _safe_mean(_group_numeric("top20_clean_rate")),
                    "min_top20_clean_rate": float(_group_numeric("top20_clean_rate").min()),
                    "mean_top10_clean_rate": _safe_mean(_group_numeric("top10_clean_rate")),
                    "min_top10_clean_rate": float(_group_numeric("top10_clean_rate").min()),
                    "mean_top05_clean_rate": _safe_mean(_group_numeric("top05_clean_rate")),
                    "min_top05_clean_rate": float(_group_numeric("top05_clean_rate").min()),
                    "mean_top10_clean_lift": _safe_mean(_group_numeric("top10_clean_lift")),
                    "mean_top10_rank_weighted_clean_rate": _safe_mean(
                        _group_numeric("top10_rank_weighted_clean_rate")
                    ),
                    "mean_top10_score_weighted_clean_rate": _safe_mean(
                        _group_numeric("top10_score_weighted_clean_rate")
                    ),
                    "side_slices_passed": side_slices_pass,
                    "side_slices_total": side_slices_total,
                    "all_side_slices_pass": all_side_slices_pass,
                    "min_side_auc": min_side_auc,
                    "min_side_gap": min_side_gap,
                    "min_short_auc": min_short_auc,
                    "min_long_auc": min_long_auc,
                }
            )
        variant_summary = pd.DataFrame(summary_rows).sort_values(
            ["all_folds_pass", "folds_passed", "mean_top10_clean_rate", "mean_top20_clean_rate", "mean_auc"],
            ascending=[False, False, False, False, False],
        )
    else:
        scored_folds = fold_df.copy()
        variant_summary = pd.DataFrame()
    variant_summary.to_csv(paths["variant_summary"], index=False)
    passing_variants = (
        variant_summary.loc[
            variant_summary["all_folds_pass"].astype(bool)
            & (
                variant_summary["all_side_slices_pass"].astype(bool)
                if bool(require_side_slice_pass) and "all_side_slices_pass" in variant_summary.columns
                else True
            ),
            "label_variant",
        ].astype(str).tolist()
        if not variant_summary.empty and "all_folds_pass" in variant_summary.columns
        else []
    )
    status = "pass" if passing_variants else "fail"
    recommendation = (
        f"promote_label_redesign_for_veto_head:{passing_variants[0]}"
        if status == "pass"
        else "redesign_labels_or_tbm_before_more_selector_blends"
    )
    model_feature_counts = (
        pd.to_numeric(fold_df.get("model_feature_count", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    ae_gmm_feature_counts = (
        pd.to_numeric(fold_df.get("ae_gmm_feature_count", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    manifest = {
        "scope": "clean_vs_dirty_positive_path_month_forward_learnability",
        "labels_path": str(labels_path),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "months": months,
        "label_variants": label_variants,
        "population": "positive_utility_rows",
        "utility_source": utility_source,
        "feature_count": int(len(features)),
        "model_feature_count_min": int(min(model_feature_counts)) if model_feature_counts else None,
        "model_feature_count_max": int(max(model_feature_counts)) if model_feature_counts else None,
        "ae_gmm_feature_count_min": int(min(ae_gmm_feature_counts)) if ae_gmm_feature_counts else None,
        "ae_gmm_feature_count_max": int(max(ae_gmm_feature_counts)) if ae_gmm_feature_counts else None,
        "ae_gmm_state_feature_max_train_rows": int(ae_gmm_state_feature_max_train_rows),
        "ae_gmm_state_feature_max_iter": int(ae_gmm_state_feature_max_iter),
        "fit_by_side": bool(fit_by_side),
        "model_kind": str(model_kind),
        "lgbm_hpo": bool(lgbm_hpo),
        "side_feature_select_top_k": int(side_feature_select_top_k),
        "require_side_slice_pass": bool(require_side_slice_pass),
        "side_slice_min_auc": float(side_slice_min_auc),
        "short_regime_abstain": bool(short_regime_abstain),
        "short_regime_abstain_min_rows": int(short_regime_abstain_min_rows),
        "short_regime_abstain_min_clean_rate": float(short_regime_abstain_min_clean_rate),
        "feature_store": feature_store_report,
        "symbol_universe_filter": symbol_universe_filter,
        "passing_label_variants": passing_variants,
        "status": status,
        "recommendation": recommendation,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_markdown(paths["markdown"], manifest, fold_df, slice_df, variant_summary)
    if symbol_universe is not None and not symbol_universe.empty:
        symbol_universe.to_csv(output_dir / "symbol_universe.csv", index=False)
    return manifest


def _write_markdown(
    path: Path,
    manifest: dict[str, Any],
    folds: pd.DataFrame,
    slices: pd.DataFrame,
    variant_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Clean-vs-Dirty Positive Path Learnability",
        "",
        f"Status: `{manifest['status']}`",
        f"Recommendation: `{manifest['recommendation']}`",
        "",
        "Population: positive-utility rows only. Label is clean path (`bad_MAE < 1R` and no timeout) versus dirty-positive path.",
        "Validation: month-forward OOS; AE/GMM state features are fit on prior rows and transformed on the validation month.",
        "",
        "## Variant Summary",
        "",
    ]
    if variant_summary.empty:
        lines.append("No variant summary produced.")
    else:
        cols = [
            "label_variant",
            "folds_passed",
            "folds_total",
            "all_folds_pass",
            "mean_clean_rate",
            "mean_top30_clean_rate",
            "mean_top20_clean_rate",
            "mean_top10_clean_rate",
            "min_top10_clean_rate",
            "mean_top10_clean_lift",
            "mean_top10_rank_weighted_clean_rate",
            "mean_top10_score_weighted_clean_rate",
            "mean_auc",
            "min_auc",
            "mean_score_gap",
            "min_score_gap",
            "side_slices_passed",
            "side_slices_total",
            "all_side_slices_pass",
            "min_side_auc",
            "min_short_auc",
        ]
        lines.append(variant_summary[[col for col in cols if col in variant_summary.columns]].to_markdown(index=False))
    lines.extend([
        "",
        "## Fold Metrics",
        "",
    ])
    if folds.empty:
        lines.append("No fold rows produced.")
    else:
        display_cols = [
            "month",
            "label_variant",
            "model_status",
            "train_rows",
            "valid_rows",
            "valid_clean_rate",
            "top30_clean_rate",
            "top20_clean_rate",
            "top10_clean_rate",
            "top05_clean_rate",
            "top10_clean_lift",
            "top10_rank_weighted_clean_rate",
            "top10_score_weighted_clean_rate",
            "roc_auc",
            "average_precision",
            "score_gap_clean_minus_dirty",
            "brier",
            "top_features",
        ]
        cols = [col for col in display_cols if col in folds.columns]
        lines.append(folds[cols].to_markdown(index=False))
    lines.extend(["", "## Slice Metrics", ""])
    if slices.empty:
        lines.append("No slice rows produced.")
    else:
        cols = [
            "month",
            "label_variant",
            "slice_type",
            "slice_value",
            "rows",
            "clean_rate",
            "top30_clean_rate",
            "top20_clean_rate",
            "top10_clean_rate",
            "top10_clean_lift",
            "top10_rank_weighted_clean_rate",
            "top10_score_weighted_clean_rate",
            "roc_auc",
            "score_gap_clean_minus_dirty",
        ]
        lines.append(slices[[col for col in cols if col in slices.columns]].to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", type=lambda value: _parse_csv(value, DEFAULT_MONTHS), default=",".join(DEFAULT_MONTHS))
    parser.add_argument(
        "--label-variants",
        type=lambda value: _parse_csv(value, DEFAULT_LABEL_VARIANTS),
        default=",".join(DEFAULT_LABEL_VARIANTS),
    )
    parser.add_argument("--evaluation-utility-column", default=None)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--spread-baseline-path", type=Path, default=None)
    parser.add_argument("--spread-rank-column", default="p75_spread_bps")
    parser.add_argument("--target-symbol-count", type=int, default=None)
    parser.add_argument("--max-spread-bps", type=float, default=None)
    parser.add_argument("--ae-gmm-state-feature-max-train-rows", type=int, default=DEFAULT_AE_GMM_STATE_FEATURE_MAX_TRAIN_ROWS)
    parser.add_argument("--ae-gmm-state-feature-max-iter", type=int, default=DEFAULT_AE_GMM_STATE_FEATURE_MAX_ITER)
    parser.add_argument("--fit-by-side", action="store_true")
    parser.add_argument("--model-kind", choices=["extratrees", "lgbm"], default="extratrees")
    parser.add_argument("--lgbm-hpo", action="store_true")
    parser.add_argument("--side-feature-select-top-k", type=int, default=0)
    parser.add_argument("--require-side-slice-pass", action="store_true")
    parser.add_argument("--side-slice-min-auc", type=float, default=0.55)
    parser.add_argument("--short-regime-abstain", action="store_true")
    parser.add_argument("--short-regime-abstain-min-rows", type=int, default=150)
    parser.add_argument("--short-regime-abstain-min-clean-rate", type=float, default=0.12)
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--min-valid-rows", type=int, default=300)
    parser.add_argument("--min-slice-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_audit(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        output_dir=args.output_dir,
        months=list(args.months),
        label_variants=list(args.label_variants),
        evaluation_utility_column=args.evaluation_utility_column,
        max_feature_store_features=args.max_feature_store_features,
        spread_baseline_path=args.spread_baseline_path,
        spread_rank_column=args.spread_rank_column,
        target_symbol_count=args.target_symbol_count,
        max_spread_bps=args.max_spread_bps,
        ae_gmm_state_feature_max_train_rows=args.ae_gmm_state_feature_max_train_rows,
        ae_gmm_state_feature_max_iter=args.ae_gmm_state_feature_max_iter,
        fit_by_side=bool(args.fit_by_side),
        model_kind=str(args.model_kind),
        lgbm_hpo=bool(args.lgbm_hpo),
        side_feature_select_top_k=int(args.side_feature_select_top_k),
        require_side_slice_pass=bool(args.require_side_slice_pass),
        side_slice_min_auc=float(args.side_slice_min_auc),
        short_regime_abstain=bool(args.short_regime_abstain),
        short_regime_abstain_min_rows=int(args.short_regime_abstain_min_rows),
        short_regime_abstain_min_clean_rate=float(args.short_regime_abstain_min_clean_rate),
        min_train_rows=args.min_train_rows,
        min_valid_rows=args.min_valid_rows,
        min_slice_rows=args.min_slice_rows,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    if manifest["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
