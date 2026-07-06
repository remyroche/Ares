#!/usr/bin/env python3
"""Month-forward meta-context clean-path filter audit.

This report tests whether regime features exported by
``report_regime_source_interaction_audit.py`` improve a train_meta-style path
filter.  Model inputs come from the meta pre-feature export only; labels and
path outcomes are read from the scored ledger for evaluation.
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
from pandas.errors import EmptyDataError
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None
    LGBMRegressor = None
    _LIGHTGBM_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_REGIME_AUDIT_DIR = Path(
    "data_perp/reports/ae_gmm_archetype_validation_status_20260704/"
    "meta_prefeature_regime_source_interaction_audit_v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_REGIME_AUDIT_DIR / "meta_regime_context_filter_oos_v1"
TOP_FRACTIONS = (0.30, 0.20, 0.10, 0.05)
KEY_COLUMNS = ("timestamp", "symbol", "side_name", "month")
LABEL_COLUMNS = (
    "u",
    "ev_after_cost",
    "clean_exec",
    "dirty_positive",
    "bad_mae",
    "timeout",
    "mfe_before_mae_1r",
    "mae_before_mfe_1r",
    "source_tag",
    "source_family",
    "gross_u",
)
EXECUTION_PROXY_REGIME_COLUMNS = (
    "candidate_exec_move_speed_bin",
    "candidate_archetype_side_exec_move_speed_bin",
    "candidate_exec_signal_to_spread_bin",
    "candidate_archetype_side_exec_signal_to_spread_bin",
    "candidate_exec_slow_resolution_risk_bin",
    "candidate_archetype_side_exec_slow_resolution_risk_bin",
    "candidate_exec_adverse_path_pressure_bin",
    "candidate_archetype_side_exec_adverse_path_pressure_bin",
    "candidate_exec_opportunity_pressure_bin",
    "candidate_archetype_side_exec_opportunity_pressure_bin",
)
EXECUTION_PROXY_NUMERIC_COLUMNS = (
    "ctx_exec_spread_bps_proxy",
    "ctx_exec_liquidity_rank_proxy",
    "ctx_exec_spread_pressure_proxy",
    "ctx_exec_volatility_rank_proxy",
    "ctx_exec_move_speed_proxy",
    "ctx_exec_signal_to_spread_proxy",
    "ctx_exec_aegmm_uncertainty_proxy",
    "ctx_exec_model_risk_pressure_proxy",
    "ctx_exec_adverse_path_pressure_proxy",
    "ctx_exec_slow_resolution_risk_proxy",
    "ctx_exec_opportunity_pressure_proxy",
)
PRIOR_GROUP_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "side_source_aegmm_distance",
        ("side_name", "source_family", "candidate_archetype_side_aegmm_distance_bin"),
    ),
    (
        "side_source_aegmm_entropy",
        ("side_name", "source_family", "candidate_archetype_side_aegmm_entropy_bin"),
    ),
    (
        "side_source_liquidity",
        ("side_name", "source_family", "candidate_archetype_side_liquidity_bin"),
    ),
    (
        "side_source_volatility",
        ("side_name", "source_family", "candidate_archetype_side_volatility_bin"),
    ),
    (
        "side_source_activity_liquidity",
        ("side_name", "source_family", "candidate_archetype_side_activity_liquidity_bin"),
    ),
    (
        "side_source_directional_vol_imbalance",
        ("side_name", "source_family", "candidate_archetype_side_directional_vol_imbalance_bin"),
    ),
    (
        "side_source_market_dispersion",
        ("side_name", "source_family", "candidate_archetype_side_market_dispersion_bin"),
    ),
    (
        "side_source_volatility_shape",
        ("side_name", "source_family", "candidate_volatility_shape_bin"),
    ),
    (
        "side_source_volatility_zscore",
        ("side_name", "source_family", "candidate_volatility_zscore_bin"),
    ),
    (
        "side_source_reconstruction",
        ("side_name", "source_family", "candidate_reconstruction_bin"),
    ),
    (
        "side_aegmm_distance",
        ("side_name", "candidate_archetype_side_aegmm_distance_bin"),
    ),
    (
        "side_aegmm_entropy",
        ("side_name", "candidate_archetype_side_aegmm_entropy_bin"),
    ),
    (
        "side_liquidity",
        ("side_name", "candidate_archetype_side_liquidity_bin"),
    ),
    (
        "side_volatility",
        ("side_name", "candidate_archetype_side_volatility_bin"),
    ),
    (
        "side_activity_liquidity",
        ("side_name", "candidate_archetype_side_activity_liquidity_bin"),
    ),
    (
        "side_directional_vol_imbalance",
        ("side_name", "candidate_archetype_side_directional_vol_imbalance_bin"),
    ),
    (
        "side_market_dispersion",
        ("side_name", "candidate_archetype_side_market_dispersion_bin"),
    ),
    (
        "side_volatility_shape",
        ("side_name", "candidate_volatility_shape_bin"),
    ),
    (
        "side_volatility_zscore",
        ("side_name", "candidate_volatility_zscore_bin"),
    ),
    (
        "side_reconstruction",
        ("side_name", "candidate_reconstruction_bin"),
    ),
    (
        "side_source_exec_move_speed",
        ("side_name", "source_family", "candidate_archetype_side_exec_move_speed_bin"),
    ),
    (
        "side_source_exec_signal_to_spread",
        ("side_name", "source_family", "candidate_archetype_side_exec_signal_to_spread_bin"),
    ),
    (
        "side_source_exec_slow_resolution_risk",
        ("side_name", "source_family", "candidate_archetype_side_exec_slow_resolution_risk_bin"),
    ),
    (
        "side_source_exec_adverse_path_pressure",
        ("side_name", "source_family", "candidate_archetype_side_exec_adverse_path_pressure_bin"),
    ),
    (
        "side_source_exec_opportunity_pressure",
        ("side_name", "source_family", "candidate_archetype_side_exec_opportunity_pressure_bin"),
    ),
    (
        "side_exec_move_speed",
        ("side_name", "candidate_archetype_side_exec_move_speed_bin"),
    ),
    (
        "side_exec_signal_to_spread",
        ("side_name", "candidate_archetype_side_exec_signal_to_spread_bin"),
    ),
    (
        "side_exec_slow_resolution_risk",
        ("side_name", "candidate_archetype_side_exec_slow_resolution_risk_bin"),
    ),
    (
        "side_exec_adverse_path_pressure",
        ("side_name", "candidate_archetype_side_exec_adverse_path_pressure_bin"),
    ),
    (
        "side_exec_opportunity_pressure",
        ("side_name", "candidate_archetype_side_exec_opportunity_pressure_bin"),
    ),
)

REGIME_SLICE_COLUMNS = (
    "side_name",
    "source_family",
    "source_tag",
    "candidate_liquidity_bin",
    "candidate_activity_liquidity_bin",
    "candidate_volatility_bin",
    "candidate_volatility_zscore_bin",
    "candidate_directional_vol_imbalance_bin",
    "candidate_market_dispersion_bin",
    "candidate_aegmm_entropy_bin",
    "candidate_aegmm_distance_bin",
    "candidate_reconstruction_bin",
    "candidate_archetype_side_aegmm_entropy_bin",
    "candidate_archetype_side_aegmm_distance_bin",
    "candidate_archetype_side_liquidity_bin",
    "candidate_archetype_side_volatility_bin",
    "candidate_archetype_side_activity_liquidity_bin",
    "candidate_archetype_side_directional_vol_imbalance_bin",
    "candidate_archetype_side_market_dispersion_bin",
    "candidate_volatility_shape_bin",
    *EXECUTION_PROXY_REGIME_COLUMNS,
)
MODEL_RISK_CAP_PRIORS = {
    "side_source_liquidity",
    "side_liquidity",
    "side_source_volatility_shape",
    "side_volatility_shape",
    "side_market_dispersion",
    "side_aegmm_entropy",
    "side_source_exec_move_speed",
    "side_exec_move_speed",
    "side_source_exec_signal_to_spread",
    "side_exec_signal_to_spread",
    "side_source_exec_slow_resolution_risk",
    "side_exec_slow_resolution_risk",
    "side_source_exec_adverse_path_pressure",
    "side_exec_adverse_path_pressure",
    "side_source_exec_opportunity_pressure",
    "side_exec_opportunity_pressure",
}
MODEL_RISK_CAPS = (
    (0.50, 0.70),
    (0.45, 0.65),
    (0.40, 0.60),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _safe_num(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_mean(values: Any) -> float:
    arr = _safe_num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _rate(values: Any) -> float:
    arr = _safe_num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.clip(0.0, 1.0).mean()) if len(arr) else float("nan")


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = _safe_num(y_true)
    s = _safe_num(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y.loc[valid].astype(int), s.loc[valid]))


def _safe_ap(y_true: pd.Series, score: pd.Series) -> float:
    y = _safe_num(y_true)
    s = _safe_num(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(average_precision_score(y.loc[valid].astype(int), s.loc[valid]))


def _safe_brier(y_true: pd.Series, score: pd.Series) -> float:
    y = _safe_num(y_true)
    s = _safe_num(score).clip(0.0, 1.0)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(brier_score_loss(y.loc[valid].astype(int), s.loc[valid]))


def _numeric_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.astype(np.float32)


def _fit_transform_features(train: pd.DataFrame, valid: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_values = _numeric_feature_frame(train, columns)
    valid_values = _numeric_feature_frame(valid, columns)
    medians = train_values.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return train_values.fillna(medians).fillna(0.0), valid_values.fillna(medians).fillna(0.0)


def _balanced_weights(y: pd.Series) -> np.ndarray:
    target = _safe_num(y).fillna(0.0).astype(int)
    pos = int(target.sum())
    neg = int(len(target) - pos)
    weights = np.ones(len(target), dtype=np.float32)
    if pos > 0 and neg > 0:
        weights[target.to_numpy(dtype=bool)] = neg / max(pos, 1)
    weights = np.clip(weights, 0.25, 8.0)
    return weights / max(float(weights.mean()), 1e-12)


def _fit_classifier(train_x: pd.DataFrame, train_y: pd.Series, seed: int) -> Any:
    y = _safe_num(train_y).fillna(0.0).astype(int)
    if y.nunique(dropna=True) < 2:
        return None
    weights = _balanced_weights(y)
    if _LIGHTGBM_AVAILABLE and LGBMClassifier is not None:
        model = LGBMClassifier(
            objective="binary",
            n_estimators=220,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=80,
            subsample=0.90,
            colsample_bytree=0.80,
            reg_alpha=0.05,
            reg_lambda=6.0,
            random_state=int(seed),
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(train_x, y, sample_weight=weights)
        return model
    model = ExtraTreesClassifier(
        n_estimators=240,
        max_depth=7,
        min_samples_leaf=40,
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=2,
    )
    model.fit(train_x, y)
    return model


def _regression_weights(train_y: pd.Series, train_frame: pd.DataFrame | None = None) -> np.ndarray:
    y = _safe_num(train_y).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = np.ones(len(y), dtype=np.float32)
    if len(y):
        positive = y.gt(0.0).to_numpy(dtype=bool)
        weights[positive] *= 1.75
        high_tail = y.ge(float(y.quantile(0.80))).to_numpy(dtype=bool)
        weights[high_tail] *= 1.35
    if train_frame is not None and len(train_frame) == len(weights):
        if "clean_exec" in train_frame.columns:
            weights *= np.where(_safe_num(train_frame["clean_exec"]).fillna(0.0).gt(0.0), 1.35, 1.0)
        if "bad_mae" in train_frame.columns:
            weights *= np.where(_safe_num(train_frame["bad_mae"]).fillna(0.0).gt(0.0), 0.85, 1.0)
        if "timeout" in train_frame.columns:
            weights *= np.where(_safe_num(train_frame["timeout"]).fillna(0.0).gt(0.0), 0.90, 1.0)
    weights = np.clip(weights, 0.25, 6.0)
    return weights / max(float(weights.mean()), 1e-12)


def _fit_regressor(train_x: pd.DataFrame, train_y: pd.Series, train_frame: pd.DataFrame, seed: int) -> Any:
    y = _safe_num(train_y).replace([np.inf, -np.inf], np.nan)
    valid = y.notna()
    if int(valid.sum()) < 100 or float(y.loc[valid].std()) <= 1e-12:
        return None
    train_x_valid = train_x.loc[valid]
    y_valid = y.loc[valid].astype(np.float32)
    frame_valid = train_frame.loc[valid] if len(train_frame) == len(train_x) else train_frame
    weights = _regression_weights(y_valid, frame_valid)
    if _LIGHTGBM_AVAILABLE and LGBMRegressor is not None:
        model = LGBMRegressor(
            objective="regression",
            n_estimators=260,
            learning_rate=0.030,
            num_leaves=31,
            min_child_samples=90,
            subsample=0.90,
            colsample_bytree=0.80,
            reg_alpha=0.05,
            reg_lambda=8.0,
            random_state=int(seed),
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(train_x_valid, y_valid, sample_weight=weights)
        return model
    model = ExtraTreesRegressor(
        n_estimators=260,
        max_depth=8,
        min_samples_leaf=45,
        random_state=int(seed),
        n_jobs=2,
    )
    model.fit(train_x_valid, y_valid, sample_weight=weights)
    return model


def _predict_classifier(model: Any, valid_x: pd.DataFrame) -> pd.Series:
    if model is None:
        return pd.Series(np.nan, index=valid_x.index, dtype=np.float32)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(valid_x)
        if np.asarray(proba).ndim == 2 and proba.shape[1] >= 2:
            return pd.Series(proba[:, 1], index=valid_x.index, dtype=np.float32)
    return pd.Series(model.predict(valid_x), index=valid_x.index, dtype=np.float32)


def _predict_regressor(model: Any, valid_x: pd.DataFrame) -> pd.Series:
    if model is None:
        return pd.Series(np.nan, index=valid_x.index, dtype=np.float32)
    return pd.Series(np.asarray(model.predict(valid_x), dtype=np.float32), index=valid_x.index)


def _safe_spearman(y_true: pd.Series, score: pd.Series) -> float:
    y = _safe_num(y_true)
    s = _safe_num(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2 or s.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(y.loc[valid].rank(method="average").corr(s.loc[valid].rank(method="average")))


def _feature_importance(model: Any, feature_cols: list[str]) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame(columns=["feature", "importance"])
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=np.float64)
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    if len(imp) != len(feature_cols):
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": feature_cols, "importance": imp})


def _top_metrics(frame: pd.DataFrame, score_col: str, frac: float) -> dict[str, Any]:
    valid = frame[pd.to_numeric(frame[score_col], errors="coerce").notna()].copy()
    tag = f"top{int(round(frac * 100)):02d}"
    if valid.empty:
        return {
            f"{tag}_rows": 0,
            f"{tag}_ev": float("nan"),
            f"{tag}_clean_precision": float("nan"),
            f"{tag}_ev_weighted_clean_precision": float("nan"),
            f"{tag}_bad_mae": float("nan"),
            f"{tag}_timeout": float("nan"),
            f"{tag}_dirty_positive": float("nan"),
            f"{tag}_mfe_before_mae_1r": float("nan"),
            f"{tag}_mae_before_mfe_1r": float("nan"),
            f"{tag}_long_share": float("nan"),
            f"{tag}_short_share": float("nan"),
        }
    top_n = max(1, int(math.ceil(float(frac) * len(valid))))
    selected = valid.sort_values(score_col, ascending=False, kind="mergesort").head(top_n)
    ev = _safe_num(selected["ev_after_cost"])
    positive_ev_weight = ev.clip(lower=0.0)
    if float(positive_ev_weight.sum()) > 1e-12:
        ev_weighted_clean = float(
            np.average(_safe_num(selected["clean_exec"]).fillna(0.0), weights=positive_ev_weight)
        )
    else:
        ev_weighted_clean = float("nan")
    side = selected["side_name"].astype(str)
    return {
        f"{tag}_rows": int(len(selected)),
        f"{tag}_ev": _safe_mean(ev),
        f"{tag}_clean_precision": _rate(selected["clean_exec"]),
        f"{tag}_ev_weighted_clean_precision": ev_weighted_clean,
        f"{tag}_bad_mae": _rate(selected["bad_mae"]),
        f"{tag}_timeout": _rate(selected["timeout"]),
        f"{tag}_dirty_positive": _rate(selected["dirty_positive"]),
        f"{tag}_mfe_before_mae_1r": _rate(selected["mfe_before_mae_1r"]),
        f"{tag}_mae_before_mfe_1r": _rate(selected["mae_before_mfe_1r"]),
        f"{tag}_long_share": float(side.eq("long").mean()) if len(side) else float("nan"),
        f"{tag}_short_share": float(side.eq("short").mean()) if len(side) else float("nan"),
    }


def _selector_metrics(
    frame: pd.DataFrame,
    *,
    feature_set: str,
    model_target: str,
    selector: str,
    score_col: str,
    test_month: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "feature_set": feature_set,
        "model_target": model_target,
        "selector": selector,
        "test_month": test_month,
        "score_col": score_col,
        "rows": int(len(frame)),
        "scorable_rows": int(pd.to_numeric(frame[score_col], errors="coerce").notna().sum()),
        "scorable_rate": float(pd.to_numeric(frame[score_col], errors="coerce").notna().mean()) if len(frame) else float("nan"),
        "base_clean_rate": _rate(frame["clean_exec"]),
        "base_bad_mae": _rate(frame["bad_mae"]),
        "base_timeout": _rate(frame["timeout"]),
        "auc_clean_exec": _safe_auc(frame["clean_exec"], frame[score_col]),
        "ap_clean_exec": _safe_ap(frame["clean_exec"], frame[score_col]),
        "brier_clean_exec": _safe_brier(frame["clean_exec"], frame[score_col]),
        "spearman_ev_after_cost": _safe_spearman(frame.get("ev_after_cost", pd.Series(dtype=float)), frame[score_col]),
        "spearman_gross_u": _safe_spearman(frame.get("gross_u", pd.Series(dtype=float)), frame[score_col])
        if "gross_u" in frame.columns
        else float("nan"),
    }
    for frac in TOP_FRACTIONS:
        row.update(_top_metrics(frame, score_col, frac))
    return row


def _summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fold_metrics.groupby(["feature_set", "model_target", "selector"], dropna=False)
        .agg(
            folds=("test_month", "nunique"),
            mean_scorable_rows=("scorable_rows", "mean"),
            min_scorable_rows=("scorable_rows", "min"),
            mean_scorable_rate=("scorable_rate", "mean"),
            mean_top30_ev=("top30_ev", "mean"),
            worst_month_top30_ev=("top30_ev", "min"),
            mean_top30_clean_precision=("top30_clean_precision", "mean"),
            mean_top30_ev_weighted_clean_precision=("top30_ev_weighted_clean_precision", "mean"),
            mean_top30_bad_mae=("top30_bad_mae", "mean"),
            mean_top30_timeout=("top30_timeout", "mean"),
            mean_top20_ev=("top20_ev", "mean"),
            worst_month_top20_ev=("top20_ev", "min"),
            mean_top20_clean_precision=("top20_clean_precision", "mean"),
            mean_top20_ev_weighted_clean_precision=("top20_ev_weighted_clean_precision", "mean"),
            mean_top20_bad_mae=("top20_bad_mae", "mean"),
            mean_top20_timeout=("top20_timeout", "mean"),
            mean_top10_ev=("top10_ev", "mean"),
            worst_month_top10_ev=("top10_ev", "min"),
            mean_top10_clean_precision=("top10_clean_precision", "mean"),
            mean_top10_ev_weighted_clean_precision=("top10_ev_weighted_clean_precision", "mean"),
            mean_top10_bad_mae=("top10_bad_mae", "mean"),
            max_month_top10_bad_mae=("top10_bad_mae", "max"),
            mean_top10_timeout=("top10_timeout", "mean"),
            max_month_top10_timeout=("top10_timeout", "max"),
            min_top10_rows=("top10_rows", "min"),
            mean_top10_mfe_before_mae_1r=("top10_mfe_before_mae_1r", "mean"),
            mean_top10_mae_before_mfe_1r=("top10_mae_before_mfe_1r", "mean"),
            mean_auc_clean_exec=("auc_clean_exec", "mean"),
            mean_ap_clean_exec=("ap_clean_exec", "mean"),
            mean_spearman_ev_after_cost=("spearman_ev_after_cost", "mean"),
            mean_spearman_gross_u=("spearman_gross_u", "mean"),
            feature_count=("feature_count", "max"),
        )
        .reset_index()
    )
    summary["mean_top10_path_order_edge"] = (
        pd.to_numeric(summary["mean_top10_mfe_before_mae_1r"], errors="coerce")
        - pd.to_numeric(summary["mean_top10_mae_before_mfe_1r"], errors="coerce")
    )
    baseline_key = ("meta_prefeature_no_regime_candidates", "blend_clean_contrast", "blend_65_clean_35_contrast")
    baseline = summary.set_index(["feature_set", "model_target", "selector"])
    if baseline_key in baseline.index:
        base = baseline.loc[baseline_key]
        for metric in (
            "mean_top30_clean_precision",
            "mean_top20_clean_precision",
            "mean_top10_clean_precision",
            "mean_top10_ev",
            "mean_top10_bad_mae",
            "mean_top10_timeout",
            "mean_top10_mfe_before_mae_1r",
            "mean_top10_mae_before_mfe_1r",
            "mean_top10_path_order_edge",
        ):
            summary[f"delta_{metric}_vs_no_regime_blend"] = pd.to_numeric(summary[metric], errors="coerce") - float(base[metric])
    no_regime = baseline[baseline.index.get_level_values("feature_set") == "meta_prefeature_no_regime_candidates"]
    for metric in (
        "mean_top10_clean_precision",
        "mean_top10_ev",
        "mean_top10_bad_mae",
        "mean_top10_timeout",
        "mean_top10_mfe_before_mae_1r",
        "mean_top10_mae_before_mfe_1r",
        "mean_top10_path_order_edge",
    ):
        deltas = []
        for row in summary.itertuples(index=False):
            key = ("meta_prefeature_no_regime_candidates", row.model_target, row.selector)
            if key in no_regime.index:
                deltas.append(float(getattr(row, metric)) - float(no_regime.loc[key, metric]))
            else:
                deltas.append(float("nan"))
        summary[f"delta_{metric}_vs_no_regime_same_selector"] = deltas
    summary["gate3_candidate_status"] = np.where(
        (summary["mean_top10_ev"] > 0.0)
        & (summary["worst_month_top10_ev"] > 0.0)
        & (summary["mean_top10_bad_mae"] <= 0.50)
        & (summary["max_month_top10_bad_mae"] <= 0.50)
        & (summary["max_month_top10_timeout"] <= 0.12)
        & (summary["mean_top10_timeout"] <= 0.12)
        & (summary["min_top10_rows"] >= 10),
        "local_path_filter_pass",
        "fail_or_diagnostic",
    )
    return summary.sort_values(
        ["gate3_candidate_status", "mean_top10_bad_mae", "mean_top10_clean_precision", "mean_top10_ev"],
        ascending=[True, True, False, False],
    )


def _collapse_executable_keys(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    key_cols = [col for col in ("timestamp", "symbol", "side_name") if col in frame.columns]
    if len(key_cols) < 3:
        return frame.copy()
    valid = frame[pd.to_numeric(frame[score_col], errors="coerce").notna()].copy()
    if valid.empty:
        return valid
    return valid.sort_values(score_col, ascending=False, kind="mergesort").drop_duplicates(key_cols, keep="first")


def _executable_fold_metrics_from_predictions(fold_metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if fold_metrics.empty or predictions.empty:
        return pd.DataFrame()
    required = {"timestamp", "symbol", "side_name", "test_month"}
    if not required.issubset(set(predictions.columns)):
        return pd.DataFrame()
    predictions_by_month = {
        str(month): group.copy()
        for month, group in predictions.groupby(predictions["test_month"].astype(str), dropna=False)
    }
    for spec in fold_metrics.drop_duplicates(["feature_set", "model_target", "selector", "test_month"]).itertuples(index=False):
        score_col = str(getattr(spec, "score_col", ""))
        if not score_col or score_col not in predictions.columns:
            continue
        month = str(getattr(spec, "test_month"))
        month_frame = predictions_by_month.get(month)
        if month_frame is None:
            continue
        if month_frame.empty:
            continue
        source_rows = int(pd.to_numeric(month_frame[score_col], errors="coerce").notna().sum())
        executable_frame = _collapse_executable_keys(month_frame, score_col)
        row = _selector_metrics(
            executable_frame,
            feature_set=str(getattr(spec, "feature_set")),
            model_target=str(getattr(spec, "model_target")),
            selector=str(getattr(spec, "selector")),
            score_col=score_col,
            test_month=month,
        )
        row["source_scorable_rows"] = source_rows
        row["executable_key_rows"] = int(len(executable_frame))
        row["duplicate_source_row_rate"] = float(1.0 - (len(executable_frame) / source_rows)) if source_rows else float("nan")
        row["train_rows"] = int(getattr(spec, "train_rows", 0))
        row["valid_rows"] = int(getattr(spec, "valid_rows", 0))
        row["feature_count"] = int(getattr(spec, "feature_count", 0))
        row["positive_rate_train"] = float(getattr(spec, "positive_rate_train", np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def _slice_metrics(
    frame: pd.DataFrame,
    *,
    feature_set: str,
    model_target: str,
    selector: str,
    score_col: str,
    test_month: str,
    group_col: str,
    frac: float,
    min_rows: int,
) -> list[dict[str, Any]]:
    valid = frame[pd.to_numeric(frame[score_col], errors="coerce").notna()].copy()
    if valid.empty or group_col not in valid.columns:
        return []
    top_n = max(1, int(math.ceil(float(frac) * len(valid))))
    selected = valid.sort_values(score_col, ascending=False, kind="mergesort").head(top_n)
    rows: list[dict[str, Any]] = []
    for group_value, group in selected.groupby(group_col, dropna=False):
        if len(group) < int(min_rows):
            continue
        row = {
            "feature_set": feature_set,
            "model_target": model_target,
            "selector": selector,
            "test_month": test_month,
            "slice_col": group_col,
            "slice_value": str(group_value),
            "top_frac": float(frac),
            "selected_rows": int(len(group)),
            "selected_ev": _safe_mean(group["ev_after_cost"]),
            "selected_clean_precision": _rate(group["clean_exec"]),
            "selected_bad_mae": _rate(group["bad_mae"]),
            "selected_timeout": _rate(group["timeout"]),
            "selected_dirty_positive": _rate(group["dirty_positive"]),
            "selected_mfe_before_mae_1r": _rate(group["mfe_before_mae_1r"]),
            "selected_mae_before_mfe_1r": _rate(group["mae_before_mfe_1r"]),
        }
        rows.append(row)
    return rows


def _rank_pct(values: pd.Series) -> pd.Series:
    numeric = _safe_num(values).replace([np.inf, -np.inf], np.nan)
    out = pd.Series(0.5, index=values.index, dtype=np.float32)
    valid = numeric.notna()
    if int(valid.sum()) <= 1 or int(numeric.loc[valid].nunique(dropna=True)) <= 1:
        return out
    out.loc[valid] = numeric.loc[valid].rank(method="average", pct=True).astype(np.float32)
    return out


def _frozen_bucket_prior_score(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    group_cols: tuple[str, ...],
    shrinkage_k: float = 150.0,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Build a train-only path-quality prior and transform validation rows.

    The score is deliberately fixed, not optimized on validation:

    clean + MFE-first + EV rank - bad-MAE - timeout - MAE-first.
    """
    present = [col for col in group_cols if col in train.columns and col in valid.columns]
    score = pd.Series(0.0, index=valid.index, dtype=np.float32)
    if len(present) != len(group_cols) or len(present) == 0:
        return score, pd.DataFrame(), pd.DataFrame(index=valid.index)
    train_work = train.loc[:, present].copy()
    valid_work = valid.loc[:, present].copy()
    for col in present:
        train_work[col] = train_work[col].astype(str).fillna("missing")
        valid_work[col] = valid_work[col].astype(str).fillna("missing")
    metrics = {
        "clean_prior": "clean_exec",
        "bad_mae_prior": "bad_mae",
        "timeout_prior": "timeout",
        "mfe_first_prior": "mfe_before_mae_1r",
        "mae_first_prior": "mae_before_mfe_1r",
        "ev_prior": "ev_after_cost",
    }
    global_values = {
        name: _safe_mean(train[source_col])
        for name, source_col in metrics.items()
        if source_col in train.columns
    }
    if "ev_prior" in global_values:
        global_values["ev_rank_prior"] = 0.5
    agg_source = train_work.copy()
    for source_col in set(metrics.values()):
        if source_col in train.columns:
            agg_source[source_col] = _safe_num(train[source_col]).values
    grouped = agg_source.groupby(present, dropna=False)
    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: str(value) for col, value in zip(present, key)}
        row["prior_support"] = int(len(group))
        n = float(len(group))
        weight = n / (n + float(shrinkage_k))
        row["prior_shrinkage_weight"] = float(weight)
        for name, source_col in metrics.items():
            if source_col not in group.columns:
                continue
            local = _safe_mean(group[source_col])
            parent = float(global_values.get(name, 0.0))
            row[name] = float(weight * local + (1.0 - weight) * parent)
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return score, table, pd.DataFrame(index=valid.index)
    if "ev_prior" in table.columns:
        table["ev_rank_prior"] = _rank_pct(table["ev_prior"])
    else:
        table["ev_rank_prior"] = 0.5
    for col in (
        "clean_prior",
        "bad_mae_prior",
        "timeout_prior",
        "mfe_first_prior",
        "mae_first_prior",
        "ev_rank_prior",
    ):
        if col not in table.columns:
            table[col] = float(global_values.get(col, 0.5))
    table["bucket_prior_score"] = (
        table["clean_prior"]
        + 0.50 * table["mfe_first_prior"]
        + 0.25 * table["ev_rank_prior"]
        - 0.60 * table["bad_mae_prior"]
        - 0.25 * table["timeout_prior"]
        - 0.25 * table["mae_first_prior"]
    )
    lookup_cols = present + [
        "bucket_prior_score",
        "prior_support",
        "prior_shrinkage_weight",
        "clean_prior",
        "bad_mae_prior",
        "timeout_prior",
        "mfe_first_prior",
        "mae_first_prior",
        "ev_prior",
    ]
    lookup = table[[col for col in lookup_cols if col in table.columns]].copy()
    merged = valid_work.merge(lookup, on=present, how="left")
    if merged["bucket_prior_score"].isna().any():
        fallback = (
            float(global_values.get("clean_prior", 0.0))
            + 0.50 * float(global_values.get("mfe_first_prior", 0.0))
            + 0.25 * 0.5
            - 0.60 * float(global_values.get("bad_mae_prior", 0.0))
            - 0.25 * float(global_values.get("timeout_prior", 0.0))
            - 0.25 * float(global_values.get("mae_first_prior", 0.0))
        )
        merged["bucket_prior_score"] = merged["bucket_prior_score"].fillna(fallback)
    fallback_values = {
        "prior_support": 0,
        "prior_shrinkage_weight": 0.0,
        "clean_prior": float(global_values.get("clean_prior", 0.0)),
        "bad_mae_prior": float(global_values.get("bad_mae_prior", 0.0)),
        "timeout_prior": float(global_values.get("timeout_prior", 0.0)),
        "mfe_first_prior": float(global_values.get("mfe_first_prior", 0.0)),
        "mae_first_prior": float(global_values.get("mae_first_prior", 0.0)),
        "ev_prior": float(global_values.get("ev_prior", 0.0)),
    }
    for col, value in fallback_values.items():
        if col not in merged.columns:
            merged[col] = value
        else:
            merged[col] = merged[col].fillna(value)
    score = pd.Series(merged["bucket_prior_score"].to_numpy(np.float32), index=valid.index)
    row_priors = pd.DataFrame(
        {
            col: merged[col].to_numpy()
            for col in (
                "bucket_prior_score",
                "prior_support",
                "prior_shrinkage_weight",
                "clean_prior",
                "bad_mae_prior",
                "timeout_prior",
                "mfe_first_prior",
                "mae_first_prior",
                "ev_prior",
            )
            if col in merged.columns
        },
        index=valid.index,
    )
    table["group_cols"] = "|".join(present)
    return score, table, row_priors


def _month_values(frame: pd.DataFrame) -> list[str]:
    months = sorted(str(m) for m in frame["month"].dropna().astype(str).unique())
    return months


def _build_feature_sets(schema: pd.DataFrame, integration: pd.DataFrame | None) -> dict[str, list[str]]:
    numeric = schema[schema["numeric"].astype(bool)].copy()
    safe = numeric[numeric["export_role"].astype(str).eq("meta_prefeature_inference_safe")].copy()
    safe_cols = [str(c) for c in safe["column"]]
    regime_cols = [
        str(c)
        for c in safe.loc[safe["feature_family"].astype(str).eq("regime_candidate_feature"), "column"]
    ]
    no_regime_cols = [c for c in safe_cols if c not in set(regime_cols)]
    prefit_cols = [
        str(c)
        for c in safe.loc[
            safe["feature_family"].astype(str).isin(
                {"prefit_score_or_rank", "prefit_path_risk_score", "key_or_split"}
            ),
            "column",
        ]
    ]
    promoted_cols: list[str] = []
    if integration is not None and len(integration):
        promoted = integration[integration["integration_status"].astype(str).eq("promoted_for_meta_hpo")]
        for value in promoted.get("exported_feature_columns", pd.Series(dtype=object)).dropna().astype(str):
            for col in value.split("|"):
                col = col.strip()
                if col in safe_cols:
                    promoted_cols.append(col)
    promoted_cols = sorted(set(promoted_cols))
    raw_sets = {
        "prefit_scores_risk_only": sorted(set(prefit_cols)),
        "meta_prefeature_no_regime_candidates": sorted(set(no_regime_cols)),
        "meta_prefeature_plus_promoted_regimes": sorted(set(no_regime_cols + promoted_cols)),
        "meta_prefeature_plus_all_regime_candidates": sorted(set(safe_cols)),
        "regime_candidate_codes_only": sorted(set(regime_cols + [c for c in ("side", "ctx_side") if c in safe_cols])),
    }
    feature_sets: dict[str, list[str]] = {}
    seen: dict[tuple[str, ...], str] = {}
    for name, cols in raw_sets.items():
        key = tuple(cols)
        if key in seen:
            continue
        seen[key] = name
        feature_sets[name] = cols
    return feature_sets


def _load_inputs(regime_audit_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    features_path = regime_audit_dir / "meta_regime_feature_export.parquet"
    ledger_path = regime_audit_dir / "regime_scored_base_ledger.parquet"
    schema_path = regime_audit_dir / "meta_regime_feature_export_schema.csv"
    integration_path = regime_audit_dir / "meta_regime_integration_plan.csv"
    features = pd.read_parquet(features_path)
    ledger = pd.read_parquet(ledger_path)
    schema = pd.read_csv(schema_path)
    integration = None
    if integration_path.exists():
        try:
            integration = pd.read_csv(integration_path)
        except EmptyDataError:
            integration = None
    report: dict[str, Any] = {
        "features_path": str(features_path),
        "ledger_path": str(ledger_path),
        "schema_path": str(schema_path),
        "integration_path": str(integration_path) if integration_path.exists() else None,
        "feature_rows": int(len(features)),
        "ledger_rows": int(len(ledger)),
        "feature_columns": int(features.shape[1]),
        "schema_rows": int(len(schema)),
    }
    if len(features) != len(ledger):
        raise ValueError(f"feature/ledger row mismatch: {len(features)} vs {len(ledger)}")
    key_match: dict[str, float] = {}
    for col in ("timestamp", "symbol", "side_name"):
        if col in features.columns and col in ledger.columns:
            left = features[col].astype(str).reset_index(drop=True)
            right = ledger[col].astype(str).reset_index(drop=True)
            key_match[col] = float(left.eq(right).mean())
    report["positional_key_match_rate"] = key_match
    if any(rate < 0.999 for rate in key_match.values()):
        raise ValueError(f"feature/ledger positional key mismatch: {key_match}")
    return features, ledger, schema, integration, report


def run_report(*, regime_audit_dir: Path, output_dir: Path, min_slice_rows: int = 10) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features, ledger, schema, integration, input_report = _load_inputs(regime_audit_dir)
    feature_sets = _build_feature_sets(schema, integration)
    data = features.copy()
    for col in KEY_COLUMNS:
        if col in ledger.columns and col not in data.columns:
            data[col] = ledger[col].values
    for col in LABEL_COLUMNS:
        if col in ledger.columns:
            data[col] = ledger[col].values
    required = {"month", "clean_exec", "dirty_positive", "bad_mae", "timeout", "ev_after_cost", "side_name"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing required columns after label join: {missing}")
    months = _month_values(data)
    fold_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    prior_rows: list[pd.DataFrame] = []
    feature_importance_rows: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    model_specs = (
        "clean_exec",
        "clean_vs_dirty_positive",
        "bad_mae",
        "timeout",
        "mfe_before_mae_1r",
        "mae_before_mfe_1r",
    )
    regression_specs = tuple(col for col in ("ev_after_cost", "gross_u") if col in data.columns)
    bad_schema_cols = [
        col
        for col in schema["column"].astype(str)
        if col in {"row_pos", "promotion_decision", "promotion_score", "gate3_status"}
        or col.startswith("s22_bucket")
    ]

    for test_month in months[1:]:
        train_mask = data["month"].astype(str).lt(str(test_month))
        valid_mask = data["month"].astype(str).eq(str(test_month))
        train_all = data.loc[train_mask].copy()
        valid_all = data.loc[valid_mask].copy()
        if len(train_all) < 500 or len(valid_all) < 100:
            continue
        print(
            json.dumps(
                {
                    "event": "meta_regime_context_filter_fold_start",
                    "test_month": str(test_month),
                    "train_rows": int(len(train_all)),
                    "valid_rows": int(len(valid_all)),
                    "feature_sets": int(len(feature_sets)),
                    "model_specs": len(model_specs),
                    "regression_specs": len(regression_specs),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        valid_scores = valid_all.loc[:, list(KEY_COLUMNS)].copy()
        valid_score_extra: dict[str, np.ndarray] = {}
        valid_scores["source_tag"] = valid_all.get("source_tag", pd.Series("unknown", index=valid_all.index)).astype(str)
        valid_scores["source_family"] = valid_all.get("source_family", pd.Series("unknown", index=valid_all.index)).astype(str)
        for col in LABEL_COLUMNS:
            if col in valid_all.columns:
                valid_scores[col] = valid_all[col].values
        for col in REGIME_SLICE_COLUMNS:
            if col in valid_all.columns:
                valid_scores[col] = valid_all[col].astype(str).values
        for col in EXECUTION_PROXY_NUMERIC_COLUMNS:
            if col in valid_all.columns:
                valid_scores[col] = _safe_num(valid_all[col]).astype(np.float32).values
        prior_scores: dict[str, pd.Series] = {}
        prior_row_frames: dict[str, pd.DataFrame] = {}
        for prior_name, group_cols in PRIOR_GROUP_SPECS:
            prior_score, prior_table, prior_row_frame = _frozen_bucket_prior_score(
                train_all,
                valid_all,
                group_cols=group_cols,
            )
            if prior_table.empty:
                continue
            prior_scores[prior_name] = prior_score
            prior_row_frames[prior_name] = prior_row_frame
            valid_score_extra[f"score_bucket_prior_{prior_name}"] = prior_score.to_numpy(np.float32)
            prior_table = prior_table.copy()
            prior_table["prior_name"] = prior_name
            prior_table["test_month"] = test_month
            prior_rows.append(prior_table)
            eval_frame = valid_all.copy()
            eval_frame["__score__"] = prior_score.values
            fold_rows.append(
                {
                    **_selector_metrics(
                        eval_frame,
                        feature_set="frozen_bucket_prior",
                        model_target="bucket_prior",
                        selector=prior_name,
                        score_col="__score__",
                        test_month=test_month,
                    ),
                    "train_rows": int(len(train_all)),
                    "valid_rows": int(len(valid_all)),
                    "feature_count": int(len(group_cols)),
                    "positive_rate_train": _rate(train_all["clean_exec"]),
                }
            )
        for feature_set, feature_cols_raw in feature_sets.items():
            feature_cols = [col for col in feature_cols_raw if col in data.columns and col not in bad_schema_cols]
            if not feature_cols:
                continue
            print(
                json.dumps(
                    {
                        "event": "meta_regime_context_filter_feature_set_start",
                        "test_month": str(test_month),
                        "feature_set": str(feature_set),
                        "feature_count": int(len(feature_cols)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            train_x_all, valid_x = _fit_transform_features(train_all, valid_all, feature_cols)
            scores_for_blend: dict[str, pd.Series] = {}
            scores_for_value: dict[str, pd.Series] = {}
            for model_target in model_specs:
                if model_target == "clean_exec":
                    train_model = train_all
                    train_x = train_x_all
                    train_y = _safe_num(train_model["clean_exec"]).fillna(0.0).astype(int)
                elif model_target == "clean_vs_dirty_positive":
                    pos_region = (
                        _safe_num(train_all["clean_exec"]).fillna(0.0).gt(0.0)
                        | _safe_num(train_all["dirty_positive"]).fillna(0.0).gt(0.0)
                    )
                    train_model = train_all.loc[pos_region].copy()
                    if len(train_model) < 100:
                        continue
                    train_x = train_x_all.loc[pos_region]
                    train_y = _safe_num(train_model["clean_exec"]).fillna(0.0).astype(int)
                elif model_target in {"mfe_before_mae_1r", "mae_before_mfe_1r"}:
                    observed_target = _safe_num(train_all[model_target]).replace([np.inf, -np.inf], np.nan).notna()
                    train_model = train_all.loc[observed_target].copy()
                    if len(train_model) < 100:
                        continue
                    train_x = train_x_all.loc[observed_target]
                    train_y = _safe_num(train_model[model_target]).fillna(0.0).astype(int)
                else:
                    train_model = train_all
                    train_x = train_x_all
                    train_y = _safe_num(train_model[model_target]).fillna(0.0).astype(int)
                model = _fit_classifier(train_x, train_y, seed=137 + len(feature_set) + len(test_month) + len(model_target))
                score = _predict_classifier(model, valid_x)
                score_col = f"score_{feature_set}_{model_target}"
                valid_score_extra[score_col] = score.to_numpy(np.float32)
                scores_for_blend[model_target] = score
                for row in (
                    _feature_importance(model, feature_cols)
                    .sort_values("importance", ascending=False)
                    .head(50)
                    .to_dict("records")
                ):
                    row.update(
                        {
                            "feature_set": feature_set,
                            "model_target": model_target,
                            "test_month": test_month,
                            "train_rows": int(len(train_model)),
                            "valid_rows": int(len(valid_all)),
                        }
                    )
                    feature_importance_rows.append(pd.DataFrame([row]))
                if model_target in {"clean_exec", "clean_vs_dirty_positive"}:
                    eval_frame = valid_all.copy()
                    eval_frame["__score__"] = score.values
                    fold_rows.append(
                        {
                            **_selector_metrics(
                                eval_frame,
                                feature_set=feature_set,
                                model_target=model_target,
                                selector=model_target,
                                score_col="__score__",
                                test_month=test_month,
                            ),
                            "train_rows": int(len(train_model)),
                            "valid_rows": int(len(valid_all)),
                            "feature_count": int(len(feature_cols)),
                            "positive_rate_train": _rate(train_y),
                        }
                    )
                    for group_col in REGIME_SLICE_COLUMNS:
                        slice_rows.extend(
                            _slice_metrics(
                                eval_frame,
                                feature_set=feature_set,
                                model_target=model_target,
                                selector=model_target,
                                score_col="__score__",
                                test_month=test_month,
                                group_col=group_col,
                                frac=0.20,
                                min_rows=min_slice_rows,
                            )
                    )
            if "clean_exec" in scores_for_blend and "clean_vs_dirty_positive" in scores_for_blend:
                blend = 0.65 * scores_for_blend["clean_exec"] + 0.35 * scores_for_blend["clean_vs_dirty_positive"]
                selector_scores: list[tuple[str, str, pd.Series]] = [
                    ("blend_clean_contrast", "blend_65_clean_35_contrast", blend),
                ]
                for regression_target in regression_specs:
                    train_y_reg = _safe_num(train_all[regression_target]).replace([np.inf, -np.inf], np.nan)
                    model = _fit_regressor(
                        train_x_all,
                        train_y_reg,
                        train_all,
                        seed=911 + len(feature_set) + len(test_month) + len(regression_target),
                    )
                    score = _predict_regressor(model, valid_x)
                    score_col = f"score_{feature_set}_{regression_target}_regressor"
                    valid_score_extra[score_col] = score.to_numpy(np.float32)
                    scores_for_value[regression_target] = score
                    for row in (
                        _feature_importance(model, feature_cols)
                        .sort_values("importance", ascending=False)
                        .head(50)
                        .to_dict("records")
                    ):
                        row.update(
                            {
                                "feature_set": feature_set,
                                "model_target": f"{regression_target}_regression",
                                "test_month": test_month,
                                "train_rows": int(train_y_reg.notna().sum()),
                                "valid_rows": int(len(valid_all)),
                            }
                        )
                        feature_importance_rows.append(pd.DataFrame([row]))
                    eval_frame = valid_all.copy()
                    eval_frame["__score__"] = score.values
                    fold_rows.append(
                        {
                            **_selector_metrics(
                                eval_frame,
                                feature_set=feature_set,
                                model_target=f"{regression_target}_regression",
                                selector=f"{regression_target}_regressor",
                                score_col="__score__",
                                test_month=test_month,
                            ),
                            "train_rows": int(train_y_reg.notna().sum()),
                            "valid_rows": int(len(valid_all)),
                            "feature_count": int(len(feature_cols)),
                            "positive_rate_train": _rate(train_y_reg.gt(0.0)),
                        }
                    )
                    for group_col in REGIME_SLICE_COLUMNS:
                        slice_rows.extend(
                            _slice_metrics(
                                eval_frame,
                                feature_set=feature_set,
                                model_target=f"{regression_target}_regression",
                                selector=f"{regression_target}_regressor",
                                score_col="__score__",
                                test_month=test_month,
                                group_col=group_col,
                                frac=0.20,
                                min_rows=min_slice_rows,
                            )
                        )
                if "bad_mae" in scores_for_blend and "timeout" in scores_for_blend:
                    bad = scores_for_blend["bad_mae"]
                    timeout = scores_for_blend["timeout"]
                    selector_scores.extend(
                        [
                            (
                                "path_penalized_blend",
                                "blend_minus_bad35_timeout15",
                                blend - 0.35 * bad - 0.15 * timeout,
                            ),
                            (
                                "path_penalized_blend",
                                "blend_minus_bad50_timeout20",
                                blend - 0.50 * bad - 0.20 * timeout,
                            ),
                            (
                                "path_penalized_blend",
                                "blend_minus_bad75_timeout25",
                                blend - 0.75 * bad - 0.25 * timeout,
                            ),
                            (
                                "path_penalized_clean",
                                "clean_minus_bad50_timeout20",
                                scores_for_blend["clean_exec"] - 0.50 * bad - 0.20 * timeout,
                            ),
                        ]
                    )
                if (
                    "mfe_before_mae_1r" in scores_for_blend
                    and "mae_before_mfe_1r" in scores_for_blend
                    and "bad_mae" in scores_for_blend
                    and "timeout" in scores_for_blend
                ):
                    bad = scores_for_blend["bad_mae"]
                    timeout = scores_for_blend["timeout"]
                    mfe_first = scores_for_blend["mfe_before_mae_1r"]
                    mae_first = scores_for_blend["mae_before_mfe_1r"]
                    path_order_score = (
                        0.35 * _rank_pct(blend)
                        + 0.25 * _rank_pct(scores_for_blend["clean_vs_dirty_positive"])
                        + 0.20 * _rank_pct(mfe_first)
                        + 0.10 * _rank_pct(scores_for_blend["clean_exec"])
                        - 0.30 * _rank_pct(mae_first)
                        - 0.25 * _rank_pct(bad)
                        - 0.10 * _rank_pct(timeout)
                    )
                    selector_scores.extend(
                        [
                            (
                                "path_order_blend",
                                "blend_pathorder_mfe20_minus_mae30_bad25_timeout10",
                                path_order_score,
                            ),
                            (
                                "path_order_clean_contrast",
                                "contrast_clean_pathorder_mfe25_minus_mae35_bad25_timeout10",
                                0.35 * _rank_pct(scores_for_blend["clean_vs_dirty_positive"])
                                + 0.20 * _rank_pct(scores_for_blend["clean_exec"])
                                + 0.25 * _rank_pct(mfe_first)
                                - 0.35 * _rank_pct(mae_first)
                                - 0.25 * _rank_pct(bad)
                                - 0.10 * _rank_pct(timeout),
                            ),
                        ]
                    )
                if "ev_after_cost" in scores_for_value:
                    ev_rank = _rank_pct(scores_for_value["ev_after_cost"])
                    selector_scores.append(
                        (
                            "execution_value_regression",
                            "ev_after_cost_ranker",
                            ev_rank,
                        )
                    )
                    if "gross_u" in scores_for_value:
                        selector_scores.append(
                            (
                                "execution_value_regression",
                                "gross_ev_rank50_50",
                                0.50 * _rank_pct(scores_for_value["gross_u"]) + 0.50 * ev_rank,
                            )
                        )
                    if "bad_mae" in scores_for_blend and "timeout" in scores_for_blend:
                        selector_scores.append(
                            (
                                "execution_value_path_adjusted",
                                "ev_rank_minus_bad25_timeout10",
                                ev_rank
                                - 0.25 * _rank_pct(scores_for_blend["bad_mae"])
                                - 0.10 * _rank_pct(scores_for_blend["timeout"]),
                            )
                        )
                        selector_scores.append(
                            (
                                "execution_value_path_adjusted",
                                "ev_rank_clean20_contrast20_minus_bad25_timeout10",
                                0.60 * ev_rank
                                + 0.20 * _rank_pct(scores_for_blend["clean_exec"])
                                + 0.20 * _rank_pct(scores_for_blend["clean_vs_dirty_positive"])
                                - 0.25 * _rank_pct(scores_for_blend["bad_mae"])
                                - 0.10 * _rank_pct(scores_for_blend["timeout"]),
                            )
                        )
                for prior_name, prior_score in prior_scores.items():
                    ranked_prior = _rank_pct(prior_score)
                    prior_frame = prior_row_frames.get(prior_name, pd.DataFrame(index=valid_all.index))
                    selector_scores.extend(
                        [
                            (
                                "bucket_prior_adjusted_blend",
                                f"blend_rank75_prior25_{prior_name}",
                                0.75 * _rank_pct(blend) + 0.25 * ranked_prior,
                            ),
                            (
                                "bucket_prior_adjusted_clean",
                                f"clean_rank75_prior25_{prior_name}",
                                0.75 * _rank_pct(scores_for_blend["clean_exec"]) + 0.25 * ranked_prior,
                            ),
                        ]
                    )
                    if "bad_mae" in scores_for_blend and "timeout" in scores_for_blend:
                        path_score = (
                            blend
                            - 0.50 * scores_for_blend["bad_mae"]
                            - 0.20 * scores_for_blend["timeout"]
                        )
                        contrast_path_score = (
                            0.50 * _rank_pct(scores_for_blend["clean_vs_dirty_positive"])
                            + 0.25 * _rank_pct(scores_for_blend["clean_exec"])
                            + 0.25 * ranked_prior
                            - 0.30 * _rank_pct(scores_for_blend["bad_mae"])
                            - 0.15 * _rank_pct(scores_for_blend["timeout"])
                        )
                        if (
                            "mfe_before_mae_1r" in scores_for_blend
                            and "mae_before_mfe_1r" in scores_for_blend
                        ):
                            regime_path_order_score = (
                                0.30 * _rank_pct(scores_for_blend["clean_vs_dirty_positive"])
                                + 0.15 * _rank_pct(scores_for_blend["clean_exec"])
                                + 0.20 * ranked_prior
                                + 0.20 * _rank_pct(scores_for_blend["mfe_before_mae_1r"])
                                - 0.30 * _rank_pct(scores_for_blend["mae_before_mfe_1r"])
                                - 0.25 * _rank_pct(scores_for_blend["bad_mae"])
                                - 0.10 * _rank_pct(scores_for_blend["timeout"])
                            )
                            selector_scores.append(
                                (
                                    "bucket_prior_path_order",
                                    f"pathorder_contrast30_clean15_prior20_mfe20_minus_mae30_bad25_timeout10_{prior_name}",
                                    regime_path_order_score,
                                )
                            )
                        selector_scores.append(
                            (
                                "bucket_prior_adjusted_path",
                                f"path_rank75_prior25_{prior_name}",
                                0.75 * _rank_pct(path_score) + 0.25 * ranked_prior,
                            )
                        )
                        selector_scores.append(
                            (
                                "bucket_prior_contrast_path",
                                f"contrast_path_rank50_clean25_prior25_{prior_name}",
                                contrast_path_score,
                            )
                        )
                        if "ev_after_cost" in scores_for_value:
                            ev_rank = _rank_pct(scores_for_value["ev_after_cost"])
                            execution_regime_score = (
                                0.55 * ev_rank
                                + 0.20 * _rank_pct(scores_for_blend["clean_vs_dirty_positive"])
                                + 0.15 * ranked_prior
                                + 0.10 * _rank_pct(scores_for_blend["clean_exec"])
                                - 0.25 * _rank_pct(scores_for_blend["bad_mae"])
                                - 0.10 * _rank_pct(scores_for_blend["timeout"])
                            )
                            selector_scores.append(
                                (
                                    "execution_value_regime_path",
                                    f"ev_rank55_contrast20_prior15_clean10_minus_bad25_timeout10_{prior_name}",
                                    execution_regime_score,
                                )
                            )
                        if prior_name in MODEL_RISK_CAP_PRIORS:
                            bad_rank = _rank_pct(scores_for_blend["bad_mae"])
                            timeout_rank = _rank_pct(scores_for_blend["timeout"])
                            for bad_rank_cap, timeout_rank_cap in MODEL_RISK_CAPS:
                                eligible_model_risk = bad_rank.le(float(bad_rank_cap)) & timeout_rank.le(
                                    float(timeout_rank_cap)
                                )
                                selector_scores.append(
                                    (
                                        "bucket_prior_modelrisk_contrast_path",
                                        (
                                            f"contrast_path_rank50_clean25_prior25_{prior_name}"
                                            f"_modelrisk_bad{int(round(bad_rank_cap * 100)):02d}"
                                            f"_timeout{int(round(timeout_rank_cap * 100)):02d}"
                                        ),
                                        contrast_path_score.where(eligible_model_risk, np.nan),
                                    )
                                )
                                if (
                                    "mfe_before_mae_1r" in scores_for_blend
                                    and "mae_before_mfe_1r" in scores_for_blend
                                ):
                                    mae_first_rank = _rank_pct(scores_for_blend["mae_before_mfe_1r"])
                                    mfe_first_rank = _rank_pct(scores_for_blend["mfe_before_mae_1r"])
                                    path_order_model_risk = (
                                        0.30 * _rank_pct(scores_for_blend["clean_vs_dirty_positive"])
                                        + 0.15 * _rank_pct(scores_for_blend["clean_exec"])
                                        + 0.20 * ranked_prior
                                        + 0.20 * mfe_first_rank
                                        - 0.30 * mae_first_rank
                                        - 0.25 * bad_rank
                                        - 0.10 * timeout_rank
                                    )
                                    selector_scores.append(
                                        (
                                            "bucket_prior_modelrisk_path_order",
                                            (
                                                f"pathorder_contrast30_clean15_prior20_mfe20_{prior_name}"
                                                f"_modelrisk_bad{int(round(bad_rank_cap * 100)):02d}"
                                                f"_timeout{int(round(timeout_rank_cap * 100)):02d}"
                                            ),
                                            path_order_model_risk.where(eligible_model_risk, np.nan),
                                        )
                                    )
                                if "ev_after_cost" in scores_for_value:
                                    selector_scores.append(
                                        (
                                            "execution_value_regime_modelrisk",
                                            (
                                                f"ev_rank55_contrast20_prior15_clean10_{prior_name}"
                                                f"_modelrisk_bad{int(round(bad_rank_cap * 100)):02d}"
                                                f"_timeout{int(round(timeout_rank_cap * 100)):02d}"
                                            ),
                                            execution_regime_score.where(eligible_model_risk, np.nan),
                                        )
                                    )
                        if {"bad_mae_prior", "timeout_prior", "prior_support"}.issubset(prior_frame.columns):
                            base_guarded_path = 0.75 * _rank_pct(path_score) + 0.25 * ranked_prior
                            base_guarded_contrast_path = contrast_path_score
                            base_guarded_path_order = (
                                0.30 * _rank_pct(scores_for_blend["clean_vs_dirty_positive"])
                                + 0.15 * _rank_pct(scores_for_blend["clean_exec"])
                                + 0.20 * ranked_prior
                                + 0.20 * _rank_pct(scores_for_blend["mfe_before_mae_1r"])
                                - 0.30 * _rank_pct(scores_for_blend["mae_before_mfe_1r"])
                                - 0.25 * _rank_pct(scores_for_blend["bad_mae"])
                                - 0.10 * _rank_pct(scores_for_blend["timeout"])
                                if (
                                    "mfe_before_mae_1r" in scores_for_blend
                                    and "mae_before_mfe_1r" in scores_for_blend
                                )
                                else None
                            )
                            base_guarded_execution_value = (
                                0.55 * _rank_pct(scores_for_value["ev_after_cost"])
                                + 0.20 * _rank_pct(scores_for_blend["clean_vs_dirty_positive"])
                                + 0.15 * ranked_prior
                                + 0.10 * _rank_pct(scores_for_blend["clean_exec"])
                                - 0.25 * _rank_pct(scores_for_blend["bad_mae"])
                                - 0.10 * _rank_pct(scores_for_blend["timeout"])
                                if "ev_after_cost" in scores_for_value
                                else None
                            )
                            for bad_cap, timeout_cap, min_support in (
                                (0.58, 0.14, 20),
                                (0.55, 0.12, 20),
                                (0.52, 0.12, 20),
                                (0.50, 0.12, 20),
                                (0.48, 0.10, 30),
                                (0.45, 0.10, 30),
                                (0.42, 0.08, 30),
                            ):
                                eligible = (
                                    _safe_num(prior_frame["bad_mae_prior"]).le(float(bad_cap))
                                    & _safe_num(prior_frame["timeout_prior"]).le(float(timeout_cap))
                                    & _safe_num(prior_frame["prior_support"]).ge(float(min_support))
                                )
                                selector_scores.append(
                                    (
                                        "bucket_prior_guarded_path",
                                        (
                                            f"path_rank75_prior25_{prior_name}"
                                            f"_guard_bad{int(round(bad_cap * 100)):02d}"
                                            f"_timeout{int(round(timeout_cap * 100)):02d}"
                                            f"_n{int(min_support)}"
                                        ),
                                        base_guarded_path.where(eligible, np.nan),
                                    )
                                )
                                selector_scores.append(
                                    (
                                        "bucket_prior_guarded_contrast_path",
                                        (
                                            f"contrast_path_rank50_clean25_prior25_{prior_name}"
                                            f"_guard_bad{int(round(bad_cap * 100)):02d}"
                                            f"_timeout{int(round(timeout_cap * 100)):02d}"
                                            f"_n{int(min_support)}"
                                        ),
                                        base_guarded_contrast_path.where(eligible, np.nan),
                                    )
                                )
                                if base_guarded_path_order is not None:
                                    selector_scores.append(
                                        (
                                            "bucket_prior_guarded_path_order",
                                            (
                                                f"pathorder_contrast30_clean15_prior20_mfe20_{prior_name}"
                                                f"_guard_bad{int(round(bad_cap * 100)):02d}"
                                                f"_timeout{int(round(timeout_cap * 100)):02d}"
                                                f"_n{int(min_support)}"
                                            ),
                                            base_guarded_path_order.where(eligible, np.nan),
                                        )
                                    )
                                if base_guarded_execution_value is not None:
                                    selector_scores.append(
                                        (
                                            "execution_value_regime_guarded",
                                            (
                                                f"ev_rank55_contrast20_prior15_clean10_{prior_name}"
                                                f"_guard_bad{int(round(bad_cap * 100)):02d}"
                                                f"_timeout{int(round(timeout_cap * 100)):02d}"
                                                f"_n{int(min_support)}"
                                            ),
                                            base_guarded_execution_value.where(eligible, np.nan),
                                        )
                                    )
                for model_target_name, selector_name, selector_score in selector_scores:
                    score_col = f"score_{feature_set}_{selector_name}"
                    valid_score_extra[score_col] = selector_score.to_numpy(np.float32)
                    eval_frame = valid_all.copy()
                    eval_frame["__score__"] = selector_score.values
                    fold_rows.append(
                        {
                            **_selector_metrics(
                                eval_frame,
                                feature_set=feature_set,
                                model_target=model_target_name,
                                selector=selector_name,
                                score_col="__score__",
                                test_month=test_month,
                            ),
                            "train_rows": int(len(train_all)),
                            "valid_rows": int(len(valid_all)),
                            "feature_count": int(len(feature_cols)),
                            "positive_rate_train": _rate(train_all["clean_exec"]),
                        }
                    )
                    for group_col in REGIME_SLICE_COLUMNS:
                        slice_rows.extend(
                            _slice_metrics(
                                eval_frame,
                                feature_set=feature_set,
                                model_target=model_target_name,
                                selector=selector_name,
                                score_col="__score__",
                                test_month=test_month,
                                group_col=group_col,
                                frac=0.20,
                                min_rows=min_slice_rows,
                            )
                        )
                print(
                    json.dumps(
                        {
                            "event": "meta_regime_context_filter_feature_set_done",
                            "test_month": str(test_month),
                            "feature_set": str(feature_set),
                            "selector_scores": int(len(selector_scores)),
                            "fold_metric_rows_so_far": int(len(fold_rows)),
                            "slice_metric_rows_so_far": int(len(slice_rows)),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if valid_score_extra:
            valid_scores = pd.concat(
                [valid_scores.reset_index(drop=True), pd.DataFrame(valid_score_extra).reset_index(drop=True)],
                axis=1,
            )
        prediction_frames.append(valid_scores)

    fold_metrics = pd.DataFrame(fold_rows)
    if fold_metrics.empty:
        raise RuntimeError("No fold metrics were produced.")
    summary = _summarize_fold_metrics(fold_metrics)

    slice_metrics = pd.DataFrame(slice_rows)
    prior_tables = (
        pd.concat(prior_rows, ignore_index=True, sort=False)
        if prior_rows
        else pd.DataFrame(columns=["prior_name", "test_month", "group_cols", "bucket_prior_score"])
    )
    feature_importance = (
        pd.concat(feature_importance_rows, ignore_index=True)
        if feature_importance_rows
        else pd.DataFrame(columns=["feature", "importance", "feature_set", "model_target", "test_month"])
    )
    if len(feature_importance):
        feature_importance_summary = (
            feature_importance.groupby(["feature_set", "model_target", "feature"], dropna=False)
            .agg(mean_importance=("importance", "mean"), folds=("test_month", "nunique"))
            .reset_index()
            .sort_values(["feature_set", "model_target", "mean_importance"], ascending=[True, True, False])
        )
    else:
        feature_importance_summary = feature_importance.copy()
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    executable_fold_metrics = _executable_fold_metrics_from_predictions(fold_metrics, predictions)
    executable_summary = (
        _summarize_fold_metrics(executable_fold_metrics)
        if not executable_fold_metrics.empty
        else pd.DataFrame(columns=summary.columns)
    )

    outputs = {
        "fold_metrics": output_dir / "meta_regime_context_filter_fold_metrics.csv",
        "summary": output_dir / "meta_regime_context_filter_summary.csv",
        "executable_fold_metrics": output_dir / "meta_regime_context_filter_executable_fold_metrics.csv",
        "executable_summary": output_dir / "meta_regime_context_filter_executable_summary.csv",
        "slice_metrics": output_dir / "meta_regime_context_filter_slice_metrics.csv",
        "bucket_priors": output_dir / "meta_regime_context_filter_bucket_priors.csv",
        "feature_importance": output_dir / "meta_regime_context_filter_feature_importance.csv",
        "predictions": output_dir / "meta_regime_context_filter_oos_predictions.parquet",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "meta_regime_context_filter_oos.md",
    }
    fold_metrics.to_csv(outputs["fold_metrics"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    executable_fold_metrics.to_csv(outputs["executable_fold_metrics"], index=False)
    executable_summary.to_csv(outputs["executable_summary"], index=False)
    slice_metrics.to_csv(outputs["slice_metrics"], index=False)
    prior_tables.to_csv(outputs["bucket_priors"], index=False)
    feature_importance_summary.to_csv(outputs["feature_importance"], index=False)
    predictions.to_parquet(outputs["predictions"], index=False)

    manifest = {
        "scope": "meta_regime_context_filter_month_forward_oos",
        "regime_audit_dir": str(regime_audit_dir),
        "output_dir": str(output_dir),
        "input_report": input_report,
        "model_backend": "lightgbm" if _LIGHTGBM_AVAILABLE else "extra_trees_fallback",
        "months": months,
        "fold_months": sorted(fold_metrics["test_month"].astype(str).unique()),
        "feature_sets": {name: len(cols) for name, cols in feature_sets.items()},
        "bad_schema_cols_excluded": bad_schema_cols,
        "summary_rows": int(len(summary)),
        "fold_metric_rows": int(len(fold_metrics)),
        "executable_summary_rows": int(len(executable_summary)),
        "executable_fold_metric_rows": int(len(executable_fold_metrics)),
        "executable_gate3_pass_rows": int(
            executable_summary["gate3_candidate_status"].eq("local_path_filter_pass").sum()
        )
        if "gate3_candidate_status" in executable_summary.columns
        else 0,
        "slice_metric_rows": int(len(slice_metrics)),
        "bucket_prior_rows": int(len(prior_tables)),
        "prediction_rows": int(len(predictions)),
        "leakage_contract": (
            "Inputs are meta_regime_feature_export inference-safe columns; labels/outcomes from scored ledger "
            "are used only as training targets and evaluation metrics. Each validation month is scored by "
            "models trained only on earlier months."
        ),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    with open(outputs["manifest"], "w", encoding="utf-8") as fh:
        json.dump(_json_safe(manifest), fh, indent=2, sort_keys=True)

    lines = [
        "# Meta Regime Context Filter OOS",
        "",
        f"Backend: `{manifest['model_backend']}`",
        f"Fold months: `{', '.join(manifest['fold_months'])}`",
        "",
        "## Top Summary",
        "",
    ]
    display_cols = [
        "feature_set",
        "model_target",
        "selector",
        "mean_top10_ev",
        "worst_month_top10_ev",
        "mean_top10_clean_precision",
        "mean_top10_bad_mae",
        "mean_top10_timeout",
        "mean_top10_mfe_before_mae_1r",
        "mean_top10_mae_before_mfe_1r",
        "mean_top10_path_order_edge",
        "delta_mean_top10_clean_precision_vs_no_regime_blend",
        "delta_mean_top10_bad_mae_vs_no_regime_blend",
        "gate3_candidate_status",
    ]
    existing_display = [c for c in display_cols if c in summary.columns]
    lines.append(summary[existing_display].head(15).to_markdown(index=False))
    if not executable_summary.empty:
        lines.extend(
            [
                "",
                "## Executable-Key Summary",
                "",
            ]
        )
        executable_display = [c for c in display_cols if c in executable_summary.columns]
        lines.append(executable_summary[executable_display].head(15).to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Contract",
            "",
            manifest["leakage_contract"],
            "",
            "## Outputs",
            "",
        ]
    )
    for key, path in outputs.items():
        lines.append(f"- `{key}`: `{path}`")
    outputs["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime-audit-dir", type=Path, default=DEFAULT_REGIME_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-slice-rows", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_report(
        regime_audit_dir=args.regime_audit_dir,
        output_dir=args.output_dir,
        min_slice_rows=int(args.min_slice_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
