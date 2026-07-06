#!/usr/bin/env python3
"""Month-forward replay-aligned meta filter diagnostic.

This report answers a narrow question after a proxy/meta handoff has been
replayed with simple_policy execution costs:

Can decision-time meta context and scenario constants learn replay executable
net / gross-minus-friction well enough to select positive top-k rows?

Inputs are the joined replay/context rows produced by
``report_meta_handoff_replay_regime_friction.py``.  Replay outcomes are used
only as training labels and validation metrics.  Each validation month is
scored by models trained only on earlier months.
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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

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


DEFAULT_JOINED_REPLAY = Path(
    "data_perp/reports/ae_gmm_archetype_validation_status_20260704/"
    "meta_prefeature_regime_source_interaction_audit_v1/"
    "meta_regime_context_filter_oos_v5_path_order/"
    "meta_regime_handoff_candidates_v5_path_order/"
    "execution_replay_all_exec_keys_cost1pct_h9_h10_h12_v1/"
    "regime_friction_v1/meta_handoff_replay_regime_friction_candidates.parquet"
)
DEFAULT_OUT_DIR = DEFAULT_JOINED_REPLAY.parent / "replay_aligned_filter_oos_v1"

TOP_FRACTIONS = (0.30, 0.20, 0.10, 0.05)
MIN_TOP10_ROWS_FOR_REPLAY_PASS = 10
ROUND_TRIP_COST_FLOOR = 0.01
KEY_COLUMNS = ("timestamp", "symbol", "side_name", "scenario", "month")
CONTEXT_CATEGORICAL = (
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
    "policy_overlay",
)
CONTEXT_NUMERIC = (
    "meta_regime_score",
    "score_rank_pct_by_month",
    "rank_pct",
    "calibrated_score",
    "barrier_pct",
    "path_len",
    "horizon_hours",
    "barrier_multiplier",
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
REPLAY_DIAGNOSTIC_NUMERIC = (
    "expected_friction_bps",
    "entry_reanchor_bps",
    "spread_cost_bps",
    "expected_spread_bps",
    "delay_window_range_bps",
    "delay_max_adverse_bps",
    "delay_max_favorable_bps",
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


def _num(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _mean(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _rate(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.clip(0.0, 1.0).mean()) if len(arr) else float("nan")


def _safe_spearman(y_true: pd.Series, score: pd.Series) -> float:
    y = _num(y_true)
    s = _num(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2 or s.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(y.loc[valid].rank(method="average").corr(s.loc[valid].rank(method="average")))


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = _num(y_true)
    s = _num(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y.loc[valid].astype(int), s.loc[valid]))


def _safe_ap(y_true: pd.Series, score: pd.Series) -> float:
    y = _num(y_true)
    s = _num(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(average_precision_score(y.loc[valid].astype(int), s.loc[valid]))


def _rank_pct(values: pd.Series) -> pd.Series:
    numeric = _num(values).replace([np.inf, -np.inf], np.nan)
    out = pd.Series(0.5, index=values.index, dtype=np.float32)
    valid = numeric.notna()
    if int(valid.sum()) <= 1 or int(numeric.loc[valid].nunique(dropna=True)) <= 1:
        return out
    out.loc[valid] = numeric.loc[valid].rank(method="average", pct=True).astype(np.float32)
    return out


def _prepare_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "month" not in out.columns or out["month"].isna().all():
        out["month"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.to_period("M").astype(str)
    out["month"] = out["month"].astype(str)
    exit_reason = out.get("simple_policy_exit_reason", pd.Series("", index=out.index)).astype(str)
    out["replay_full_sl"] = exit_reason.eq("full_sl").astype(float)
    out["replay_timeout"] = exit_reason.eq("timeout").astype(float)
    out["replay_trailing"] = exit_reason.eq("trailing").astype(float)
    out["replay_clean_net_positive"] = (
        _num(out.get("net_return")).gt(0.0)
        & out["replay_full_sl"].eq(0.0)
        & out["replay_timeout"].eq(0.0)
    ).astype(float)
    out["replay_bad_path"] = (out["replay_full_sl"].gt(0.0) | out["replay_timeout"].gt(0.0)).astype(float)
    out["net_return"] = _num(out.get("net_return"))
    out["gross_return"] = _num(out.get("gross_return"))
    expected_friction = _num(out.get("expected_friction_bps")) / 10000.0
    out["expected_friction_return"] = expected_friction
    out["gross_minus_friction_return"] = out["gross_return"] - expected_friction
    out["gross_minus_1pct_return"] = out["gross_return"] - ROUND_TRIP_COST_FLOOR
    out["executable_margin_return"] = out["gross_return"] - np.maximum(
        expected_friction.fillna(ROUND_TRIP_COST_FLOOR).to_numpy(dtype=np.float64),
        ROUND_TRIP_COST_FLOOR,
    )
    out["timeout_adjusted_net_return"] = (
        out["net_return"]
        - 0.50 * ROUND_TRIP_COST_FLOOR * out["replay_timeout"]
        - 0.25 * ROUND_TRIP_COST_FLOOR * out["replay_full_sl"]
    )
    holding = _num(out.get("holding_bars"))
    horizon = _num(out.get("path_len")).replace(0.0, np.nan)
    out["holding_frac_of_horizon"] = holding / horizon
    out["replay_positive_margin"] = out["executable_margin_return"].gt(0.0).astype(float)
    out["replay_fast_clean_positive"] = (
        out["net_return"].gt(0.0)
        & out["replay_full_sl"].eq(0.0)
        & out["replay_timeout"].eq(0.0)
        & out["holding_frac_of_horizon"].le(0.75)
    ).astype(float)
    out["replay_slow_or_loss"] = (
        out["net_return"].le(0.0)
        | out["replay_timeout"].gt(0.0)
        | out["holding_frac_of_horizon"].gt(0.90)
    ).astype(float)
    return out


def _feature_frame(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    categorical_cols: tuple[str, ...],
    numeric_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cat_cols = [col for col in categorical_cols if col in train.columns and col in valid.columns]
    num_cols = [col for col in numeric_cols if col in train.columns and col in valid.columns]
    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    if num_cols:
        train_num = train.loc[:, num_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid_num = valid.loc[:, num_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        medians = train_num.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        train_parts.append(train_num.fillna(medians).fillna(0.0).astype(np.float32))
        valid_parts.append(valid_num.fillna(medians).fillna(0.0).astype(np.float32))
    if cat_cols:
        train_cat = pd.get_dummies(train.loc[:, cat_cols].astype(str).fillna("missing"), dummy_na=False)
        valid_cat = pd.get_dummies(valid.loc[:, cat_cols].astype(str).fillna("missing"), dummy_na=False)
        valid_cat = valid_cat.reindex(columns=train_cat.columns, fill_value=0)
        train_parts.append(train_cat.astype(np.float32))
        valid_parts.append(valid_cat.astype(np.float32))
    if not train_parts:
        raise ValueError("No feature columns available.")
    train_x = pd.concat(train_parts, axis=1)
    valid_x = pd.concat(valid_parts, axis=1).reindex(columns=train_x.columns, fill_value=0.0)
    return train_x, valid_x, list(train_x.columns)


def _regression_weights(y: pd.Series, frame: pd.DataFrame) -> np.ndarray:
    target = _num(y).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = np.ones(len(target), dtype=np.float32)
    weights *= np.where(target.gt(0.0), 2.0, 1.0)
    if len(target) and target.notna().any():
        weights *= np.where(target.ge(float(target.quantile(0.80))), 1.5, 1.0)
    if "replay_bad_path" in frame.columns:
        weights *= np.where(_num(frame["replay_bad_path"]).fillna(0.0).gt(0.0), 1.25, 1.0)
    weights = np.clip(weights, 0.25, 6.0)
    return weights / max(float(weights.mean()), 1e-12)


def _fit_regressor(train_x: pd.DataFrame, y: pd.Series, frame: pd.DataFrame, seed: int) -> Any:
    target = _num(y).replace([np.inf, -np.inf], np.nan)
    valid = target.notna()
    if int(valid.sum()) < 50 or float(target.loc[valid].std()) <= 1e-12:
        return None
    weights = _regression_weights(target.loc[valid], frame.loc[valid])
    if _LIGHTGBM_AVAILABLE and LGBMRegressor is not None:
        model = LGBMRegressor(
            objective="regression",
            n_estimators=220,
            learning_rate=0.035,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.90,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=8.0,
            random_state=int(seed),
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(train_x.loc[valid], target.loc[valid].astype(np.float32), sample_weight=weights)
        return model
    model = ExtraTreesRegressor(
        n_estimators=260,
        max_depth=6,
        min_samples_leaf=15,
        random_state=int(seed),
        n_jobs=2,
    )
    model.fit(train_x.loc[valid], target.loc[valid].astype(np.float32), sample_weight=weights)
    return model


def _balanced_weights(y: pd.Series) -> np.ndarray:
    target = _num(y).fillna(0.0).astype(int)
    pos = int(target.sum())
    neg = int(len(target) - pos)
    weights = np.ones(len(target), dtype=np.float32)
    if pos > 0 and neg > 0:
        weights[target.to_numpy(dtype=bool)] = neg / max(pos, 1)
    weights = np.clip(weights, 0.25, 8.0)
    return weights / max(float(weights.mean()), 1e-12)


def _fit_classifier(train_x: pd.DataFrame, y: pd.Series, seed: int) -> Any:
    target = _num(y).fillna(0.0).astype(int)
    if target.nunique(dropna=True) < 2:
        return None
    weights = _balanced_weights(target)
    if _LIGHTGBM_AVAILABLE and LGBMClassifier is not None:
        model = LGBMClassifier(
            objective="binary",
            n_estimators=180,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.90,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=6.0,
            random_state=int(seed),
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(train_x, target, sample_weight=weights)
        return model
    model = ExtraTreesClassifier(
        n_estimators=240,
        max_depth=6,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=2,
    )
    model.fit(train_x, target)
    return model


def _predict(model: Any, valid_x: pd.DataFrame, *, classifier: bool = False) -> pd.Series:
    if model is None:
        return pd.Series(np.nan, index=valid_x.index, dtype=np.float32)
    if classifier and hasattr(model, "predict_proba"):
        proba = model.predict_proba(valid_x)
        if np.asarray(proba).ndim == 2 and proba.shape[1] >= 2:
            return pd.Series(proba[:, 1], index=valid_x.index, dtype=np.float32)
    return pd.Series(np.asarray(model.predict(valid_x), dtype=np.float32), index=valid_x.index)


def _top_metrics(frame: pd.DataFrame, score_col: str, frac: float) -> dict[str, Any]:
    valid = frame[_num(frame[score_col]).notna()].copy()
    tag = f"top{int(round(frac * 100)):02d}"
    if valid.empty:
        return {
            f"{tag}_rows": 0,
            f"{tag}_net": float("nan"),
            f"{tag}_oracle_net": float("nan"),
            f"{tag}_oracle_recall": float("nan"),
            f"{tag}_gross": float("nan"),
            f"{tag}_gross_minus_friction": float("nan"),
            f"{tag}_gross_minus_1pct": float("nan"),
            f"{tag}_executable_margin": float("nan"),
            f"{tag}_timeout_adjusted_net": float("nan"),
            f"{tag}_hit_net": float("nan"),
            f"{tag}_clean_net_positive": float("nan"),
            f"{tag}_positive_margin": float("nan"),
            f"{tag}_fast_clean_positive": float("nan"),
            f"{tag}_full_sl": float("nan"),
            f"{tag}_timeout": float("nan"),
            f"{tag}_mean_holding_frac": float("nan"),
            f"{tag}_short_share": float("nan"),
        }
    n = max(1, int(math.ceil(len(valid) * float(frac))))
    selected = valid.sort_values(score_col, ascending=False, kind="mergesort").head(n)
    oracle = valid.sort_values("net_return", ascending=False, kind="mergesort").head(n)
    oracle_index = set(oracle.index.tolist())
    selected_index = set(selected.index.tolist())
    return {
        f"{tag}_rows": int(len(selected)),
        f"{tag}_net": _mean(selected["net_return"]),
        f"{tag}_oracle_net": _mean(oracle["net_return"]),
        f"{tag}_oracle_recall": float(len(selected_index & oracle_index) / max(len(oracle_index), 1)),
        f"{tag}_gross": _mean(selected["gross_return"]),
        f"{tag}_gross_minus_friction": _mean(selected["gross_minus_friction_return"]),
        f"{tag}_gross_minus_1pct": _mean(selected["gross_minus_1pct_return"]),
        f"{tag}_executable_margin": _mean(selected["executable_margin_return"]),
        f"{tag}_timeout_adjusted_net": _mean(selected["timeout_adjusted_net_return"]),
        f"{tag}_hit_net": _rate(_num(selected["net_return"]).gt(0.0)),
        f"{tag}_clean_net_positive": _rate(selected["replay_clean_net_positive"]),
        f"{tag}_positive_margin": _rate(selected["replay_positive_margin"]),
        f"{tag}_fast_clean_positive": _rate(selected["replay_fast_clean_positive"]),
        f"{tag}_full_sl": _rate(selected["replay_full_sl"]),
        f"{tag}_timeout": _rate(selected["replay_timeout"]),
        f"{tag}_mean_holding_frac": _mean(selected["holding_frac_of_horizon"]),
        f"{tag}_short_share": float(selected["side_name"].astype(str).eq("short").mean())
        if "side_name" in selected.columns and len(selected)
        else float("nan"),
    }


def _selector_metrics(
    frame: pd.DataFrame,
    *,
    feature_set: str,
    target: str,
    selector: str,
    score_col: str,
    test_month: str,
    collapse_candidate: bool,
) -> dict[str, Any]:
    work = frame.copy()
    if collapse_candidate and "handoff_candidate_id" in work.columns:
        work = (
            work[_num(work[score_col]).notna()]
            .sort_values(score_col, ascending=False, kind="mergesort")
            .drop_duplicates(["handoff_candidate_id"], keep="first")
        )
    row: dict[str, Any] = {
        "feature_set": feature_set,
        "target": target,
        "selector": selector,
        "test_month": test_month,
        "collapse_candidate": bool(collapse_candidate),
        "rows": int(len(work)),
        "scorable_rows": int(_num(work[score_col]).notna().sum()) if score_col in work.columns else 0,
        "base_net": _mean(work["net_return"]) if "net_return" in work.columns else float("nan"),
        "base_gross_minus_friction": _mean(work["gross_minus_friction_return"])
        if "gross_minus_friction_return" in work.columns
        else float("nan"),
        "base_executable_margin": _mean(work["executable_margin_return"])
        if "executable_margin_return" in work.columns
        else float("nan"),
        "spearman_net": _safe_spearman(work.get("net_return", pd.Series(dtype=float)), work[score_col]),
        "spearman_gross_minus_friction": _safe_spearman(
            work.get("gross_minus_friction_return", pd.Series(dtype=float)),
            work[score_col],
        ),
        "spearman_executable_margin": _safe_spearman(
            work.get("executable_margin_return", pd.Series(dtype=float)),
            work[score_col],
        ),
        "auc_clean_net_positive": _safe_auc(work.get("replay_clean_net_positive", pd.Series(dtype=float)), work[score_col]),
        "ap_clean_net_positive": _safe_ap(work.get("replay_clean_net_positive", pd.Series(dtype=float)), work[score_col]),
        "auc_positive_margin": _safe_auc(work.get("replay_positive_margin", pd.Series(dtype=float)), work[score_col]),
        "ap_positive_margin": _safe_ap(work.get("replay_positive_margin", pd.Series(dtype=float)), work[score_col]),
    }
    for frac in TOP_FRACTIONS:
        row.update(_top_metrics(work, score_col, frac))
    return row


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    summary = (
        metrics.groupby(["feature_set", "target", "selector", "collapse_candidate"], dropna=False)
        .agg(
            folds=("test_month", "nunique"),
            mean_scorable_rows=("scorable_rows", "mean"),
            min_scorable_rows=("scorable_rows", "min"),
            mean_top30_net=("top30_net", "mean"),
            mean_top30_oracle_net=("top30_oracle_net", "mean"),
            mean_top30_oracle_recall=("top30_oracle_recall", "mean"),
            worst_top30_net=("top30_net", "min"),
            mean_top30_gross_minus_friction=("top30_gross_minus_friction", "mean"),
            mean_top30_executable_margin=("top30_executable_margin", "mean"),
            mean_top30_full_sl=("top30_full_sl", "mean"),
            mean_top30_timeout=("top30_timeout", "mean"),
            mean_top20_net=("top20_net", "mean"),
            mean_top20_oracle_net=("top20_oracle_net", "mean"),
            mean_top20_oracle_recall=("top20_oracle_recall", "mean"),
            worst_top20_net=("top20_net", "min"),
            mean_top20_gross_minus_friction=("top20_gross_minus_friction", "mean"),
            mean_top20_executable_margin=("top20_executable_margin", "mean"),
            mean_top20_full_sl=("top20_full_sl", "mean"),
            mean_top20_timeout=("top20_timeout", "mean"),
            mean_top10_net=("top10_net", "mean"),
            mean_top10_oracle_net=("top10_oracle_net", "mean"),
            mean_top10_oracle_recall=("top10_oracle_recall", "mean"),
            worst_top10_net=("top10_net", "min"),
            mean_top10_gross_minus_friction=("top10_gross_minus_friction", "mean"),
            mean_top10_gross_minus_1pct=("top10_gross_minus_1pct", "mean"),
            mean_top10_executable_margin=("top10_executable_margin", "mean"),
            mean_top10_timeout_adjusted_net=("top10_timeout_adjusted_net", "mean"),
            mean_top10_clean_net_positive=("top10_clean_net_positive", "mean"),
            mean_top10_positive_margin=("top10_positive_margin", "mean"),
            mean_top10_fast_clean_positive=("top10_fast_clean_positive", "mean"),
            mean_top10_full_sl=("top10_full_sl", "mean"),
            mean_top10_timeout=("top10_timeout", "mean"),
            mean_top10_holding_frac=("top10_mean_holding_frac", "mean"),
            min_top10_rows=("top10_rows", "min"),
            mean_spearman_net=("spearman_net", "mean"),
            mean_spearman_gross_minus_friction=("spearman_gross_minus_friction", "mean"),
            mean_spearman_executable_margin=("spearman_executable_margin", "mean"),
            mean_auc_clean_net_positive=("auc_clean_net_positive", "mean"),
            mean_ap_clean_net_positive=("ap_clean_net_positive", "mean"),
            mean_auc_positive_margin=("auc_positive_margin", "mean"),
            mean_ap_positive_margin=("ap_positive_margin", "mean"),
        )
        .reset_index()
    )
    summary["replay_aligned_status"] = np.where(
        (summary["mean_top10_net"] > 0.0)
        & (summary["worst_top10_net"] > 0.0)
        & (summary["mean_top10_executable_margin"] > 0.0)
        & (summary["mean_top10_full_sl"] <= 0.20)
        & (summary["mean_top10_timeout"] <= 0.35)
        & (summary["min_top10_rows"] >= MIN_TOP10_ROWS_FOR_REPLAY_PASS),
        "replay_filter_pass",
        "fail_or_diagnostic",
    )
    return summary.sort_values(
        ["replay_aligned_status", "mean_top10_net", "worst_top10_net"],
        ascending=[True, False, False],
    )


def _feature_importance(model: Any, feature_cols: list[str]) -> pd.DataFrame:
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    imp = np.asarray(model.feature_importances_, dtype=np.float64)
    if len(imp) != len(feature_cols):
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": feature_cols, "importance": imp})


def run_report(*, joined_replay: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _prepare_labels(pd.read_parquet(joined_replay))
    months = sorted(str(m) for m in data["month"].dropna().unique())
    if len(months) < 2:
        raise ValueError(f"Need at least two months for month-forward OOS, got {months}")

    feature_sets = {
        "context_only": (CONTEXT_CATEGORICAL, CONTEXT_NUMERIC),
        "context_plus_replay_friction_diagnostics": (
            CONTEXT_CATEGORICAL,
            CONTEXT_NUMERIC + REPLAY_DIAGNOSTIC_NUMERIC,
        ),
    }
    fold_rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    regression_targets = (
        "net_return",
        "gross_minus_friction_return",
        "gross_minus_1pct_return",
        "executable_margin_return",
        "timeout_adjusted_net_return",
        "gross_return",
    )
    classification_targets = (
        "replay_clean_net_positive",
        "replay_positive_margin",
        "replay_fast_clean_positive",
        "replay_bad_path",
        "replay_slow_or_loss",
    )

    for test_month in months[1:]:
        train = data[data["month"].astype(str).lt(str(test_month))].copy()
        valid = data[data["month"].astype(str).eq(str(test_month))].copy()
        if len(train) < 50 or len(valid) < 20:
            continue
        print(
            json.dumps(
                {
                    "event": "replay_aligned_fold_start",
                    "test_month": str(test_month),
                    "train_rows": int(len(train)),
                    "valid_rows": int(len(valid)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        valid_scores = valid.loc[:, [col for col in KEY_COLUMNS if col in valid.columns]].copy()
        for col in (
            "handoff_candidate_id",
            "source_family",
            "source_tag",
            "candidate_volatility_shape_bin",
            "candidate_aegmm_entropy_bin",
            "candidate_liquidity_bin",
            "candidate_activity_liquidity_bin",
            "candidate_market_dispersion_bin",
            "net_return",
            "gross_return",
            "gross_minus_friction_return",
            "gross_minus_1pct_return",
            "executable_margin_return",
            "timeout_adjusted_net_return",
            "replay_clean_net_positive",
            "replay_positive_margin",
            "replay_fast_clean_positive",
            "replay_bad_path",
            "replay_slow_or_loss",
            "replay_full_sl",
            "replay_timeout",
            "holding_frac_of_horizon",
        ):
            if col in valid.columns:
                valid_scores[col] = valid[col].values
        extra_scores: dict[str, np.ndarray] = {}
        for feature_set, (cat_cols, num_cols) in feature_sets.items():
            train_x, valid_x, feature_cols = _feature_frame(
                train,
                valid,
                categorical_cols=cat_cols,
                numeric_cols=num_cols,
            )
            score_bank: dict[str, pd.Series] = {}
            for target in regression_targets:
                model = _fit_regressor(train_x, train[target], train, seed=1000 + len(feature_set) + len(target))
                score = _predict(model, valid_x)
                score_bank[target] = score
                score_col = f"score_{feature_set}_{target}"
                extra_scores[score_col] = score.to_numpy(np.float32)
                eval_frame = valid.copy()
                eval_frame["__score__"] = score.values
                for collapse in (False, True):
                    fold_rows.append(
                        _selector_metrics(
                            eval_frame,
                            feature_set=feature_set,
                            target=target,
                            selector=f"{target}_regressor",
                            score_col="__score__",
                            test_month=str(test_month),
                            collapse_candidate=collapse,
                        )
                    )
                imp = _feature_importance(model, feature_cols).sort_values("importance", ascending=False).head(60)
                if not imp.empty:
                    imp["feature_set"] = feature_set
                    imp["target"] = target
                    imp["test_month"] = str(test_month)
                    importance_frames.append(imp)
            for target in classification_targets:
                model = _fit_classifier(train_x, train[target], seed=2000 + len(feature_set) + len(target))
                score = _predict(model, valid_x, classifier=True)
                score_bank[target] = score
                score_col = f"score_{feature_set}_{target}"
                extra_scores[score_col] = score.to_numpy(np.float32)
                eval_frame = valid.copy()
                inverted = target in {"replay_bad_path", "replay_slow_or_loss"}
                eval_frame["__score__"] = score.values if not inverted else -score.values
                selector = f"{target}_classifier" if not inverted else f"minus_{target}_classifier"
                for collapse in (False, True):
                    fold_rows.append(
                        _selector_metrics(
                            eval_frame,
                            feature_set=feature_set,
                            target=target,
                            selector=selector,
                            score_col="__score__",
                            test_month=str(test_month),
                            collapse_candidate=collapse,
                        )
                    )
                imp = _feature_importance(model, feature_cols).sort_values("importance", ascending=False).head(60)
                if not imp.empty:
                    imp["feature_set"] = feature_set
                    imp["target"] = target
                    imp["test_month"] = str(test_month)
                    importance_frames.append(imp)

            if {
                "net_return",
                "gross_minus_friction_return",
                "replay_clean_net_positive",
                "replay_bad_path",
            }.issubset(score_bank):
                blend = (
                    0.45 * _rank_pct(score_bank["net_return"])
                    + 0.25 * _rank_pct(score_bank["gross_minus_friction_return"])
                    + 0.20 * _rank_pct(score_bank["replay_clean_net_positive"])
                    - 0.25 * _rank_pct(score_bank["replay_bad_path"])
                )
                score_col = f"score_{feature_set}_replay_net_margin_clean_minus_bad"
                extra_scores[score_col] = blend.to_numpy(np.float32)
                eval_frame = valid.copy()
                eval_frame["__score__"] = blend.values
                for collapse in (False, True):
                    fold_rows.append(
                        _selector_metrics(
                            eval_frame,
                            feature_set=feature_set,
                            target="replay_aligned_blend",
                            selector="net45_margin25_clean20_minus_bad25",
                            score_col="__score__",
                            test_month=str(test_month),
                            collapse_candidate=collapse,
                        )
                    )
            if {
                "executable_margin_return",
                "gross_minus_1pct_return",
                "timeout_adjusted_net_return",
                "replay_positive_margin",
                "replay_fast_clean_positive",
                "replay_bad_path",
                "replay_slow_or_loss",
            }.issubset(score_bank):
                executable_blend = (
                    0.40 * _rank_pct(score_bank["executable_margin_return"])
                    + 0.20 * _rank_pct(score_bank["gross_minus_1pct_return"])
                    + 0.20 * _rank_pct(score_bank["timeout_adjusted_net_return"])
                    + 0.20 * _rank_pct(score_bank["replay_positive_margin"])
                    + 0.15 * _rank_pct(score_bank["replay_fast_clean_positive"])
                    - 0.25 * _rank_pct(score_bank["replay_bad_path"])
                    - 0.20 * _rank_pct(score_bank["replay_slow_or_loss"])
                )
                score_col = f"score_{feature_set}_executable_margin_blend"
                extra_scores[score_col] = executable_blend.to_numpy(np.float32)
                eval_frame = valid.copy()
                eval_frame["__score__"] = executable_blend.values
                for collapse in (False, True):
                    fold_rows.append(
                        _selector_metrics(
                            eval_frame,
                            feature_set=feature_set,
                            target="executable_margin_blend",
                            selector="margin40_1pct20_timeout20_pos20_fast15_minus_bad25_slow20",
                            score_col="__score__",
                            test_month=str(test_month),
                            collapse_candidate=collapse,
                        )
                    )
                conservative_blend = (
                    0.50 * _rank_pct(score_bank["executable_margin_return"])
                    + 0.25 * _rank_pct(score_bank["replay_positive_margin"])
                    - 0.35 * _rank_pct(score_bank["replay_bad_path"])
                    - 0.30 * _rank_pct(score_bank["replay_slow_or_loss"])
                )
                score_col = f"score_{feature_set}_conservative_executable_margin_blend"
                extra_scores[score_col] = conservative_blend.to_numpy(np.float32)
                eval_frame = valid.copy()
                eval_frame["__score__"] = conservative_blend.values
                for collapse in (False, True):
                    fold_rows.append(
                        _selector_metrics(
                            eval_frame,
                            feature_set=feature_set,
                            target="conservative_executable_margin_blend",
                            selector="margin50_pos25_minus_bad35_slow30",
                            score_col="__score__",
                            test_month=str(test_month),
                            collapse_candidate=collapse,
                        )
                    )
        if extra_scores:
            valid_scores = pd.concat(
                [valid_scores.reset_index(drop=True), pd.DataFrame(extra_scores).reset_index(drop=True)],
                axis=1,
            )
        prediction_frames.append(valid_scores)

    fold_metrics = pd.DataFrame(fold_rows)
    if fold_metrics.empty:
        raise RuntimeError("No fold metrics were produced.")
    summary = _summarize(fold_metrics)
    feature_importance = (
        pd.concat(importance_frames, ignore_index=True, sort=False) if importance_frames else pd.DataFrame()
    )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    )

    outputs = {
        "summary": out_dir / "meta_replay_aligned_filter_summary.csv",
        "fold_metrics": out_dir / "meta_replay_aligned_filter_fold_metrics.csv",
        "feature_importance": out_dir / "meta_replay_aligned_filter_feature_importance.csv",
        "predictions": out_dir / "meta_replay_aligned_filter_predictions.parquet",
        "manifest": out_dir / "manifest.json",
        "report": out_dir / "meta_replay_aligned_filter_report.md",
    }
    summary.to_csv(outputs["summary"], index=False)
    fold_metrics.to_csv(outputs["fold_metrics"], index=False)
    feature_importance.to_csv(outputs["feature_importance"], index=False)
    predictions.to_parquet(outputs["predictions"], index=False)

    manifest = {
        "generated_by": "report_meta_replay_aligned_filter_oos",
        "joined_replay": str(joined_replay),
        "out_dir": str(out_dir),
        "rows": int(len(data)),
        "months": months,
        "fold_months": sorted(fold_metrics["test_month"].astype(str).unique().tolist()),
        "feature_sets": {name: {"categorical": list(cats), "numeric": list(nums)} for name, (cats, nums) in feature_sets.items()},
        "model_backend": "lightgbm" if _LIGHTGBM_AVAILABLE else "extra_trees",
        "leakage_contract": (
            "Joined replay rows provide labels/outcomes. Feature set context_only uses selected-row meta context "
            "and scenario constants only. The friction diagnostic feature set additionally includes replay-derived "
            "friction diagnostics and is not decision-time-safe unless those fields are materialized pre-entry. "
            "Each validation month is scored by models trained only on earlier months."
        ),
        "summary_rows": int(len(summary)),
        "fold_metric_rows": int(len(fold_metrics)),
        "prediction_rows": int(len(predictions)),
        "min_top10_rows_for_replay_pass": int(MIN_TOP10_ROWS_FOR_REPLAY_PASS),
        "round_trip_cost_floor": float(ROUND_TRIP_COST_FLOOR),
        "pass_rows": int(summary["replay_aligned_status"].eq("replay_filter_pass").sum())
        if "replay_aligned_status" in summary.columns
        else 0,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")

    display_cols = [
        "feature_set",
        "target",
        "selector",
        "collapse_candidate",
        "mean_top10_net",
        "mean_top10_oracle_net",
        "mean_top10_oracle_recall",
        "worst_top10_net",
        "mean_top10_gross_minus_friction",
        "mean_top10_executable_margin",
        "mean_top10_timeout_adjusted_net",
        "mean_top10_clean_net_positive",
        "mean_top10_positive_margin",
        "mean_top10_fast_clean_positive",
        "mean_top10_full_sl",
        "mean_top10_timeout",
        "mean_top10_holding_frac",
        "mean_spearman_net",
        "mean_spearman_executable_margin",
        "min_top10_rows",
        "replay_aligned_status",
    ]
    existing = [col for col in display_cols if col in summary.columns]
    lines = [
        "# Meta Replay-Aligned Filter OOS",
        "",
        f"Backend: `{manifest['model_backend']}`",
        f"Fold months: `{', '.join(manifest['fold_months'])}`",
        "",
        "## Top Summary",
        "",
        summary[existing].head(25).to_markdown(index=False),
        "",
        "## Leakage Contract",
        "",
        manifest["leakage_contract"],
        "",
    ]
    outputs["report"].write_text("\n".join(lines), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-replay", type=Path, default=DEFAULT_JOINED_REPLAY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_report(joined_replay=args.joined_replay, out_dir=args.out_dir)
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
