#!/usr/bin/env python3
"""Train a shadow direct accepted-frontier suppression learner.

The no-backfill threshold controller failed because full replay deltas were
driven by path/capacity effects while direct accepted-trade suppression was
not recurrent.  This script trains a small, audited shadow model on the direct
accepted-frontier ledger:

    baseline-accepted near-frontier row -> direct defensive utility if suppressed

It is deliberately non-executing.  It does not change scores, ranks, auction
order, policy thresholds, or production activation state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")


DEFAULT_LEDGER = Path(
    "data_perp/reports/market_state_direct_suppression_ledger_globalrank_no_backfill_combined_20260627_v4_with_jun26_partial_strategy_diagnostics"
    "/direct_accepted_frontier_training_ledger.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_direct_suppression_controller_training"
)

ARTIFACT_CONTRACT = "direct_accepted_frontier_suppression_controller_training_v1"

NUMERIC_FEATURES = [
    "rank_score",
    "rank_minus_base_threshold",
    "frontier_distance",
    "required_threshold_raise_to_suppress",
    "base_threshold",
    "risk_severity",
    "prediction_coverage",
    "state_ood_score_mean",
    "state_ood_score_max",
    "state_ood_share",
    "state_low_input_coverage_share",
    "mean_pred_utility",
    "mean_pred_lcb",
    "mean_pred_full_sl",
    "mean_pred_timeout",
    "frontier_candidate_count",
    "accepted_frontier_candidate_count",
    "controller_has_forecast",
    "controller_is_overlay",
    "pred_lcb_gap",
    "pred_risk_sum",
]
CATEGORICAL_FEATURES = ["head", "side", "strategy_id"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + frame.to_csv(index=False) + "```"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _safe_num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def prepare_training_frame(ledger: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp",
        "controller_arm",
        "decision_key",
        "direct_defensive_utility",
        "direct_suppression_profitable",
        "required_threshold_raise_to_suppress",
    }
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise KeyError(f"direct suppression ledger is missing columns: {missing}")
    frame = ledger.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.loc[frame["timestamp"].notna()].copy()
    frame["controller_arm"] = frame["controller_arm"].astype(str)
    frame["decision_key"] = frame["decision_key"].astype(str)
    frame["controller_has_forecast"] = frame["controller_arm"].str.contains(
        "forecast",
        case=False,
        regex=False,
    ).astype(float)
    frame["controller_is_overlay"] = frame["controller_arm"].str.contains(
        "post_selection_overlay",
        case=False,
        regex=False,
    ).astype(float)
    frame["pred_lcb_gap"] = _safe_num(frame, "mean_pred_utility").fillna(0.0) - _safe_num(
        frame,
        "mean_pred_lcb",
    ).fillna(0.0)
    frame["pred_risk_sum"] = _safe_num(frame, "mean_pred_full_sl").fillna(0.0) + _safe_num(
        frame,
        "mean_pred_timeout",
    ).fillna(0.0)
    frame["y_profit"] = frame["direct_suppression_profitable"].astype(bool).astype(int)
    frame["y_utility"] = _safe_num(frame, "direct_defensive_utility").fillna(0.0)
    frame["sample_weight"] = _safe_num(frame, "frontier_sample_weight").fillna(1.0).clip(
        lower=0.05,
        upper=10.0,
    )
    return frame.sort_values(["timestamp", "controller_arm", "decision_key"]).reset_index(drop=True)


def fit_feature_spec(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = [col for col in NUMERIC_FEATURES if col in frame.columns]
    categories: dict[str, list[str]] = {}
    for col in CATEGORICAL_FEATURES:
        if col in frame.columns:
            categories[col] = sorted(frame[col].astype(str).fillna("missing").unique().tolist())
    medians = {
        col: float(_safe_num(frame, col).replace([np.inf, -np.inf], np.nan).median())
        for col in numeric
    }
    medians = {col: (0.0 if not np.isfinite(val) else val) for col, val in medians.items()}
    feature_columns = list(numeric)
    for col, values in categories.items():
        feature_columns.extend([f"{col}__{value}" for value in values])
    return {
        "numeric": numeric,
        "categorical": categories,
        "medians": medians,
        "feature_columns": feature_columns,
    }


def transform_features(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    numeric = list(spec.get("numeric") or [])
    medians = dict(spec.get("medians") or {})
    if numeric:
        num = frame.reindex(columns=numeric).apply(pd.to_numeric, errors="coerce")
        num = num.replace([np.inf, -np.inf], np.nan)
        for col in numeric:
            num[col] = num[col].fillna(float(medians.get(col, 0.0)))
        parts.append(num.astype("float64"))
    for col, values in (spec.get("categorical") or {}).items():
        raw = frame[col].astype(str).fillna("missing") if col in frame.columns else pd.Series("missing", index=frame.index)
        cat = pd.DataFrame(index=frame.index)
        for value in values:
            cat[f"{col}__{value}"] = raw.eq(str(value)).astype(float)
        parts.append(cat)
    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)
    for col in spec.get("feature_columns") or []:
        if col not in out.columns:
            out[col] = 0.0
    return out[list(spec.get("feature_columns") or [])].astype("float64")


def chronological_fold_plan(frame: pd.DataFrame) -> list[dict[str, Any]]:
    folds = sorted(int(f) for f in pd.Series(frame["fold"]).dropna().unique())
    if len(folds) < 2:
        return []
    plan: list[dict[str, Any]] = []
    for fold in folds[1:]:
        train_folds = [f for f in folds if f < fold]
        train_idx = frame.index[frame["fold"].isin(train_folds)].to_numpy()
        valid_idx = frame.index[frame["fold"].eq(fold)].to_numpy()
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue
        plan.append(
            {
                "valid_fold": int(fold),
                "train_folds": train_folds,
                "train_index": train_idx,
                "valid_index": valid_idx,
                "train_timestamps": int(frame.loc[train_idx, "timestamp"].nunique()),
                "valid_timestamps": int(frame.loc[valid_idx, "timestamp"].nunique()),
            }
        )
    return plan


def _make_lgbm_classifier(train_rows: int) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="binary",
        n_estimators=60,
        learning_rate=0.04,
        max_depth=2,
        num_leaves=4,
        min_child_samples=max(2, int(train_rows * 0.10)),
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        random_state=20260627,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
        n_jobs=1,
    )


def _make_lgbm_regressor(train_rows: int) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=80,
        learning_rate=0.04,
        max_depth=2,
        num_leaves=4,
        min_child_samples=max(2, int(train_rows * 0.10)),
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        random_state=20260628,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
        n_jobs=1,
    )


def _constant_predictions(
    y_profit: pd.Series,
    y_utility: pd.Series,
    valid_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full(valid_count, float(y_profit.mean()), dtype=float),
        np.full(valid_count, float(y_utility.mean()), dtype=float),
    )


def fit_oof_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    plan = chronological_fold_plan(frame)
    pred = frame[
        [
            "timestamp",
            "controller_arm",
            "decision_key",
            "head",
            "side",
            "strategy_id",
            "rank_score",
            "base_threshold",
            "required_threshold_raise_to_suppress",
            "loss_avoided_if_suppressed",
            "winner_pnl_sacrificed_if_suppressed",
            "direct_defensive_utility",
            "direct_suppression_profitable",
            "direct_suppression_full_sl",
            "direct_suppression_timeout",
            "sample_weight",
        ]
    ].copy()
    pred["prediction_available"] = False
    pred["pred_suppression_profit_prob"] = np.nan
    pred["pred_direct_utility"] = np.nan
    pred["valid_fold"] = np.nan
    pred["model_mode"] = ""
    fold_reports: list[dict[str, Any]] = []
    final_spec = fit_feature_spec(frame)
    for item in plan:
        train = frame.loc[item["train_index"]].copy()
        valid = frame.loc[item["valid_index"]].copy()
        spec = fit_feature_spec(train)
        X_train = transform_features(train, spec)
        X_valid = transform_features(valid, spec)
        y_profit = train["y_profit"].astype(int)
        y_utility = train["y_utility"].astype(float)
        weights = train["sample_weight"].astype(float).to_numpy()
        mode = "lgbm"
        if len(train) < 12 or y_profit.nunique(dropna=True) < 2:
            prob, util = _constant_predictions(y_profit, y_utility, len(valid))
            mode = "constant_prior"
        else:
            clf = _make_lgbm_classifier(len(train))
            reg = _make_lgbm_regressor(len(train))
            clf.fit(X_train, y_profit.to_numpy(dtype=np.int8), sample_weight=weights)
            reg.fit(X_train, y_utility.to_numpy(dtype=float), sample_weight=weights)
            proba = clf.predict_proba(X_valid)
            prob = np.asarray(proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba, dtype=float)
            util = np.asarray(reg.predict(X_valid), dtype=float)
        pred.loc[valid.index, "prediction_available"] = True
        pred.loc[valid.index, "pred_suppression_profit_prob"] = prob
        pred.loc[valid.index, "pred_direct_utility"] = util
        pred.loc[valid.index, "valid_fold"] = int(item["valid_fold"])
        pred.loc[valid.index, "model_mode"] = mode
        fold_reports.append(
            {
                "valid_fold": int(item["valid_fold"]),
                "train_folds": item["train_folds"],
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "train_timestamps": item["train_timestamps"],
                "valid_timestamps": item["valid_timestamps"],
                "train_unique_decision_keys": int(train["decision_key"].nunique()),
                "valid_unique_decision_keys": int(valid["decision_key"].nunique()),
                "train_positive_rate": float(y_profit.mean()),
                "valid_positive_rate": float(valid["y_profit"].mean()),
                "model_mode": mode,
                "feature_count": int(len(spec.get("feature_columns") or [])),
            }
        )
    return pred, fold_reports, final_spec


def _binary_auc(y: pd.Series, score: pd.Series) -> float | None:
    y = pd.Series(y).astype(int)
    score = pd.to_numeric(score, errors="coerce")
    mask = y.notna() & score.notna()
    y = y.loc[mask]
    score = score.loc[mask]
    if y.nunique(dropna=True) < 2 or len(y) < 3:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, score))
    except Exception:
        return None


def _average_precision(y: pd.Series, score: pd.Series) -> float | None:
    y = pd.Series(y).astype(int)
    score = pd.to_numeric(score, errors="coerce")
    mask = y.notna() & score.notna()
    y = y.loc[mask]
    score = score.loc[mask]
    if y.nunique(dropna=True) < 2 or len(y) < 3:
        return None
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y, score))
    except Exception:
        return None


def oof_diagnostics(pred: pd.DataFrame) -> dict[str, Any]:
    available = pred.loc[pred["prediction_available"].astype(bool)].copy()
    y = available["direct_suppression_profitable"].astype(bool).astype(int)
    prob = pd.to_numeric(available["pred_suppression_profit_prob"], errors="coerce")
    util = pd.to_numeric(available["pred_direct_utility"], errors="coerce")
    target_util = pd.to_numeric(available["direct_defensive_utility"], errors="coerce")
    corr = util.corr(target_util, method="spearman") if len(available) >= 3 else np.nan
    return {
        "oof_rows": int(len(available)),
        "oof_folds": sorted(int(f) for f in available["valid_fold"].dropna().unique()),
        "oof_unique_decision_keys": int(available["decision_key"].nunique()) if len(available) else 0,
        "oof_positive_rate": float(y.mean()) if len(y) else None,
        "prob_auc": _binary_auc(y, prob),
        "prob_average_precision": _average_precision(y, prob),
        "utility_spearman": float(corr) if np.isfinite(corr) else None,
        "pred_prob_std": float(prob.std(ddof=0)) if len(prob.dropna()) else None,
        "pred_utility_std": float(util.std(ddof=0)) if len(util.dropna()) else None,
    }


def evaluate_policy_grid(
    pred: pd.DataFrame,
    *,
    max_delta: float = 0.08,
    min_suppressed_rows: int = 2,
    min_suppressed_folds: int = 2,
    policy_scopes: tuple[str, ...] = ("controller_arm",),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    available = pred.loc[pred["prediction_available"].astype(bool)].copy()
    if available.empty:
        return pd.DataFrame(), {"selected_arm": None, "reason": "no_oof_predictions"}
    probability_grid = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    utility_grid = [-0.005, 0.0, 0.005, 0.010]
    scope_columns = {
        "controller_arm": ["controller_arm"],
        "controller_arm_head": ["controller_arm", "head"],
        "controller_arm_strategy": ["controller_arm", "strategy_id"],
        "controller_arm_head_strategy": ["controller_arm", "head", "strategy_id"],
    }
    rows: list[dict[str, Any]] = []
    for scope in policy_scopes:
        group_cols = scope_columns.get(str(scope))
        if group_cols is None:
            raise ValueError(f"unsupported policy scope: {scope!r}")
        for group_key, arm_frame in available.groupby(group_cols, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = {
                col: str(value) for col, value in zip(group_cols, group_key)
            }
            arm = group_values["controller_arm"]
            target_head = group_values.get("head")
            target_strategy_id = group_values.get("strategy_id")
            policy_group = "|".join(f"{col}={group_values[col]}" for col in group_cols)
            for prob_cut in probability_grid:
                for util_cut in utility_grid:
                    eligible = (
                        pd.to_numeric(arm_frame["pred_suppression_profit_prob"], errors="coerce").ge(prob_cut)
                        & pd.to_numeric(arm_frame["pred_direct_utility"], errors="coerce").ge(util_cut)
                        & pd.to_numeric(
                            arm_frame["required_threshold_raise_to_suppress"],
                            errors="coerce",
                        ).le(float(max_delta))
                    )
                    selected = arm_frame.loc[eligible].copy()
                    loss = float(
                        pd.to_numeric(
                            selected["loss_avoided_if_suppressed"],
                            errors="coerce",
                        ).fillna(0.0).sum()
                    )
                    winner = float(
                        pd.to_numeric(
                            selected["winner_pnl_sacrificed_if_suppressed"],
                            errors="coerce",
                        ).fillna(0.0).sum()
                    )
                    utility = float(
                        pd.to_numeric(selected["direct_defensive_utility"], errors="coerce").fillna(0.0).sum()
                    )
                    fold_utility = (
                        selected.groupby("valid_fold")["direct_defensive_utility"].sum()
                        if not selected.empty
                        else pd.Series(dtype=float)
                    )
                    all_valid_folds = sorted(
                        int(f) for f in arm_frame["valid_fold"].dropna().unique().tolist()
                    )
                    fold_utility_all = pd.Series(0.0, index=all_valid_folds, dtype=float)
                    if len(fold_utility):
                        fold_utility_all.loc[
                            [int(f) for f in fold_utility.index.tolist()]
                        ] = fold_utility.to_numpy(dtype=float)
                    positive_fold_share = (
                        float((fold_utility_all > 0.0).mean())
                        if len(fold_utility_all)
                        else 0.0
                    )
                    suppressed_folds = int(selected["valid_fold"].nunique()) if len(selected) else 0
                    valid_fold_count = int(arm_frame["valid_fold"].nunique())
                    suppression_fold_share = (
                        float(suppressed_folds / valid_fold_count)
                        if valid_fold_count
                        else 0.0
                    )
                    rows.append(
                        {
                            "policy_scope": str(scope),
                            "policy_group": policy_group,
                            "controller_arm": str(arm),
                            "target_head": target_head,
                            "target_strategy_id": target_strategy_id,
                            "probability_cutoff": float(prob_cut),
                            "utility_cutoff": float(util_cut),
                            "max_delta": float(max_delta),
                            "suppressed_rows": int(len(selected)),
                            "suppressed_unique_decision_keys": int(selected["decision_key"].nunique()) if len(selected) else 0,
                            "suppressed_head_count": int(selected["head"].nunique()) if len(selected) else 0,
                            "loss_avoided": loss,
                            "winner_pnl_sacrificed": winner,
                            "defensive_success": utility,
                            "positive_fold_share": positive_fold_share,
                            "valid_fold_count": valid_fold_count,
                            "suppressed_folds": suppressed_folds,
                            "suppression_fold_share": suppression_fold_share,
                            "mean_pred_prob": float(selected["pred_suppression_profit_prob"].mean()) if len(selected) else np.nan,
                            "mean_pred_utility": float(selected["pred_direct_utility"].mean()) if len(selected) else np.nan,
                        }
                    )
    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid, {"selected_arm": None, "reason": "empty_policy_grid"}
    grid["passes_diagnostic_gate"] = (
        grid["suppressed_rows"].ge(int(min_suppressed_rows))
        & grid["suppressed_folds"].ge(int(min_suppressed_folds))
        & grid["defensive_success"].gt(0.0)
        & grid["loss_avoided"].gt(grid["winner_pnl_sacrificed"])
        & grid["positive_fold_share"].ge(0.50)
    )
    grid["selection_score"] = (
        grid["defensive_success"]
        + 0.25 * grid["loss_avoided"]
        - 0.50 * grid["winner_pnl_sacrificed"]
        + 0.01 * grid["positive_fold_share"]
    )
    candidates = grid.loc[grid["passes_diagnostic_gate"]].sort_values(
        ["selection_score", "defensive_success", "suppressed_rows"],
        ascending=[False, False, False],
    )
    if candidates.empty:
        return grid.sort_values("selection_score", ascending=False), {
            "selected_arm": None,
            "reason": "no_policy_grid_row_passed_diagnostic_gate",
            "best_attempt": _json_safe(
                grid.sort_values("selection_score", ascending=False).head(1).to_dict("records")[0]
            ),
        }
    selected = candidates.iloc[0].to_dict()
    return grid.sort_values("selection_score", ascending=False), {
        "selected_arm": selected["controller_arm"],
        "selected_policy_scope": selected.get("policy_scope"),
        "selected_policy_group": selected.get("policy_group"),
        "selected_target_head": selected.get("target_head"),
        "selected_target_strategy_id": selected.get("target_strategy_id"),
        "reason": "diagnostic_shadow_policy_selected",
        "selected_policy": _json_safe(selected),
        "promotion_allowed": False,
        "promotion_note": "diagnostic selection only; requires later shadow scoring and full replay gates",
    }


def fit_final_models(frame: pd.DataFrame, output_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import joblib

    X = transform_features(frame, spec)
    y_profit = frame["y_profit"].astype(int)
    y_utility = frame["y_utility"].astype(float)
    weights = frame["sample_weight"].astype(float).to_numpy()
    if len(frame) < 12 or y_profit.nunique(dropna=True) < 2:
        payload = {
            "model_mode": "constant_prior",
            "prior_profit_prob": float(y_profit.mean()) if len(y_profit) else 0.0,
            "prior_direct_utility": float(y_utility.mean()) if len(y_utility) else 0.0,
            "feature_spec": spec,
        }
    else:
        clf = _make_lgbm_classifier(len(frame))
        reg = _make_lgbm_regressor(len(frame))
        clf.fit(X, y_profit.to_numpy(dtype=np.int8), sample_weight=weights)
        reg.fit(X, y_utility.to_numpy(dtype=float), sample_weight=weights)
        payload = {
            "model_mode": "lgbm",
            "classifier": clf,
            "regressor": reg,
            "feature_spec": spec,
        }
    model_path = output_dir / "direct_suppression_shadow_models.joblib"
    joblib.dump(payload, model_path)
    return {
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "model_mode": payload["model_mode"],
    }


def feature_importance(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    if len(frame) < 12 or frame["y_profit"].nunique(dropna=True) < 2:
        return pd.DataFrame(columns=["feature", "classifier_importance", "regressor_importance"])
    clf = _make_lgbm_classifier(len(frame))
    reg = _make_lgbm_regressor(len(frame))
    X = transform_features(frame, spec)
    weights = frame["sample_weight"].astype(float).to_numpy()
    clf.fit(X, frame["y_profit"].astype(int).to_numpy(dtype=np.int8), sample_weight=weights)
    reg.fit(X, frame["y_utility"].astype(float).to_numpy(dtype=float), sample_weight=weights)
    return pd.DataFrame(
        {
            "feature": list(X.columns),
            "classifier_importance": getattr(clf, "feature_importances_", np.zeros(X.shape[1])),
            "regressor_importance": getattr(reg, "feature_importances_", np.zeros(X.shape[1])),
        }
    ).sort_values(["classifier_importance", "regressor_importance"], ascending=False)


def _render_report(summary: dict[str, Any], grid: pd.DataFrame, fold_report: pd.DataFrame) -> str:
    lines = [
        "# Direct Suppression Controller Training",
        "",
        "This is a shadow-only diagnostic learner for accepted-frontier threshold raises.",
        "It does not activate a controller or change T1 scores/ranks/auction ordering.",
        "",
        "## Summary",
        "",
        f"- Ledger rows: `{summary.get('ledger_rows')}`",
        f"- Unique decision keys: `{summary.get('unique_decision_keys')}`",
        f"- OOF rows: `{summary.get('oof', {}).get('oof_rows')}`",
        f"- OOF probability AUC: `{summary.get('oof', {}).get('prob_auc')}`",
        f"- OOF average precision: `{summary.get('oof', {}).get('prob_average_precision')}`",
        f"- OOF utility Spearman: `{summary.get('oof', {}).get('utility_spearman')}`",
        f"- Selected shadow policy: `{summary.get('selection', {}).get('selected_arm')}`",
        f"- Selection reason: `{summary.get('selection', {}).get('reason')}`",
        f"- Minimum suppressed rows: `{summary.get('policy_grid', {}).get('min_suppressed_rows')}`",
        f"- Minimum suppressed folds: `{summary.get('policy_grid', {}).get('min_suppressed_folds')}`",
        f"- Policy scopes: `{summary.get('policy_grid', {}).get('policy_scopes')}`",
        "",
        "## Fold Support",
        "",
        _markdown_table(fold_report) if not fold_report.empty else "_No fold report._",
        "",
        "## Top Policy Grid Rows",
        "",
        _markdown_table(grid.head(12)) if not grid.empty else "_No grid rows._",
        "",
        "## Contract",
        "",
        "- Uses grouped chronological folds from the ledger fold field.",
        "- Uses only prediction-time schedule/context fields plus rank distance.",
        "- Requires threshold raises only; rows needing a raise above max_delta are not suppressible.",
        "- Any selected policy is shadow-only and requires later full replay validation.",
    ]
    return "\n".join(lines) + "\n"


def run_training(
    ledger_path: Path,
    output_dir: Path,
    *,
    max_delta: float = 0.08,
    min_suppressed_rows: int = 2,
    min_suppressed_folds: int = 2,
    policy_scopes: tuple[str, ...] = ("controller_arm",),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = prepare_training_frame(_read_frame(ledger_path))
    oof, fold_reports, spec = fit_oof_predictions(ledger)
    grid, selection = evaluate_policy_grid(
        oof,
        max_delta=float(max_delta),
        min_suppressed_rows=int(min_suppressed_rows),
        min_suppressed_folds=int(min_suppressed_folds),
        policy_scopes=tuple(policy_scopes),
    )
    fi = feature_importance(ledger, spec)
    model_info = fit_final_models(ledger, output_dir, spec)
    fold_report = pd.DataFrame(fold_reports)
    oof_diag = oof_diagnostics(oof)
    summary = {
        "generated_by": "train_market_state_direct_suppression_controller",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_contract": ARTIFACT_CONTRACT,
        "ledger_path": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
        "ledger_rows": int(len(ledger)),
        "unique_decision_keys": int(ledger["decision_key"].nunique()),
        "timestamp_count": int(ledger["timestamp"].nunique()),
        "controller_arm_count": int(ledger["controller_arm"].nunique()),
        "active_heads": sorted(ledger["head"].astype(str).unique().tolist()),
        "feature_columns": spec.get("feature_columns") or [],
        "feature_count": int(len(spec.get("feature_columns") or [])),
        "chronological_validation": {
            "fold_count": int(len(fold_reports)),
            "folds": fold_reports,
            "first_fold_without_prior_training_is_not_scored": True,
        },
        "oof": oof_diag,
        "policy_grid": {
            "max_delta": float(max_delta),
            "min_suppressed_rows": int(min_suppressed_rows),
            "min_suppressed_folds": int(min_suppressed_folds),
            "policy_scopes": list(policy_scopes),
            "row_count": int(len(grid)),
        },
        "selection": selection,
        "model": model_info,
        "promotion_allowed": False,
        "promotion_status": "shadow_only_training_artifact",
        "interpretation": (
            "This model is a diagnostic controller-training surface. Promotion still requires "
            "later-window shadow scoring, positive direct suppression recurrence, and full replay gates."
        ),
    }
    paths = {
        "oof_predictions": output_dir / "direct_suppression_oof_predictions.parquet",
        "policy_grid": output_dir / "direct_suppression_policy_grid.csv",
        "fold_report": output_dir / "direct_suppression_fold_report.csv",
        "feature_importance": output_dir / "direct_suppression_feature_importance.csv",
        "summary": output_dir / "direct_suppression_training_summary.json",
        "report": output_dir / "direct_suppression_training_report.md",
        "feature_spec": output_dir / "direct_suppression_feature_spec.json",
    }
    oof.to_parquet(paths["oof_predictions"], index=False)
    grid.to_csv(paths["policy_grid"], index=False)
    fold_report.to_csv(paths["fold_report"], index=False)
    fi.to_csv(paths["feature_importance"], index=False)
    paths["feature_spec"].write_text(json.dumps(_json_safe(spec), indent=2) + "\n", encoding="utf-8")
    summary["outputs"] = {key: str(path) for key, path in paths.items()}
    summary["output_hashes"] = {
        key: _sha256(path)
        for key, path in paths.items()
        if key not in {"summary", "report"} and path.exists()
    }
    paths["summary"].write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["report"].write_text(_render_report(summary, grid, fold_report), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-delta", type=float, default=0.08)
    parser.add_argument("--min-suppressed-rows", type=int, default=2)
    parser.add_argument("--min-suppressed-folds", type=int, default=2)
    parser.add_argument(
        "--policy-scopes",
        default="controller_arm,controller_arm_head,controller_arm_strategy,controller_arm_head_strategy",
        help=(
            "Comma-separated policy scopes: controller_arm, controller_arm_head, "
            "controller_arm_strategy, controller_arm_head_strategy."
        ),
    )
    args = parser.parse_args()
    policy_scopes = tuple(
        scope.strip()
        for scope in str(args.policy_scopes).split(",")
        if scope.strip()
    )
    summary = run_training(
        args.ledger,
        args.output_dir,
        max_delta=float(args.max_delta),
        min_suppressed_rows=int(args.min_suppressed_rows),
        min_suppressed_folds=int(args.min_suppressed_folds),
        policy_scopes=policy_scopes,
    )
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
