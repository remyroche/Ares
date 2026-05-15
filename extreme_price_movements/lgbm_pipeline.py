from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

try:
    from .ridge_on_lgbm import _compute_weight_distillation
except Exception:  # pragma: no cover - standalone fallback
    def _compute_weight_distillation(
        y_true: np.ndarray,
        pred: np.ndarray,
        prev_pred: np.ndarray | None = None,
        *,
        is_classifier: bool = True,
        include_false_positive_focus: bool = False,
    ) -> np.ndarray:
        del prev_pred, include_false_positive_focus
        y = np.asarray(y_true, dtype=np.float32)
        p = np.asarray(pred, dtype=np.float32)
        if len(y) == 0:
            return np.ones(0, dtype=np.float32)
        if is_classifier:
            yb = (y >= 0.5).astype(np.float32)
            err = np.abs(yb - np.clip(p, 0.0, 1.0))
        else:
            scale = float(np.nanpercentile(np.abs(y - np.nanmedian(y)), 75.0) + 1e-6)
            err = np.abs(y - p) / scale
        rank = pd.Series(np.nan_to_num(err, nan=0.0)).rank(pct=True).to_numpy(dtype=np.float32)
        return np.clip(0.75 + 1.50 * rank, 0.25, 4.0).astype(np.float32)

try:
    from .utils import tprint
except Exception:  # pragma: no cover - standalone fallback
    def tprint(message: str) -> None:
        print(message, flush=True)


LGBM_CV_SPLITS = int(os.environ.get("EPM_LGBM_CV_SPLITS", "3"))
LGBM_RACE_MAX_ROWS = int(os.environ.get("EPM_LGBM_RACE_MAX_ROWS", "120000"))
LGBM_RACE_EVAL_FRACTION = float(os.environ.get("EPM_LGBM_RACE_EVAL_FRACTION", "0.3333333333"))
LGBM_MIN_FEATURES = int(os.environ.get("EPM_LGBM_MIN_FEATURES", "40"))
LGBM_SELECTED_FEATURES_MIN = int(os.environ.get("EPM_LGBM_SELECTED_FEATURES_MIN", "100"))
LGBM_SELECTED_FEATURES_MAX = int(os.environ.get("EPM_LGBM_SELECTED_FEATURES_MAX", "350"))
LGBM_MAX_ROUNDS = int(os.environ.get("EPM_LGBM_MAX_ROUNDS", "10"))
LGBM_ROW_SUBSAMPLE_FRAC = float(os.environ.get("EPM_LGBM_ROW_SUBSAMPLE_FRAC", "1.0"))
LGBM_HPO_MAX_ROWS = int(os.environ.get("EPM_LGBM_HPO_MAX_ROWS", "10000"))
LGBM_HPO_TRIALS = int(os.environ.get("EPM_LGBM_HPO_TRIALS", "200"))
LGBM_HPO_EARLY_STOP_PATIENCE = int(os.environ.get("EPM_LGBM_HPO_EARLY_STOP_PATIENCE", "50"))
LGBM_FINAL_MODEL_COUNT = int(os.environ.get("EPM_LGBM_FINAL_MODEL_COUNT", "3"))
LGBM_OOF_DISTILLATION_PASSES = int(os.environ.get("EPM_LGBM_OOF_DISTILLATION_PASSES", "1"))
LGBM_DIRECTION_STABILITY_MIN = float(os.environ.get("EPM_LGBM_DIRECTION_STABILITY_MIN", "0.75"))
LGBM_POSITIVE_PERM_RATE_MIN = float(os.environ.get("EPM_LGBM_POSITIVE_PERM_RATE_MIN", "0.50"))
LGBM_LOW_PRESENCE_RATE = float(os.environ.get("EPM_LGBM_LOW_PRESENCE_RATE", "0.20"))
LGBM_REDUNDANCY_CORR_THRESHOLD = float(os.environ.get("EPM_LGBM_REDUNDANCY_CORR_THRESHOLD", "0.90"))
LGBM_REDUNDANCY_PENALTY_START = float(os.environ.get("EPM_LGBM_REDUNDANCY_PENALTY_START", "0.85"))
LGBM_UNIVARIATE_MONOTONICITY_MIN = float(os.environ.get("EPM_LGBM_UNIVARIATE_MONOTONICITY_MIN", "0.95"))
LGBM_PERMUTATION_REPEATS = int(os.environ.get("EPM_LGBM_PERMUTATION_REPEATS", "2"))
LGBM_PERMUTATION_EPS = float(os.environ.get("EPM_LGBM_PERMUTATION_EPS", "1e-5"))
LGBM_PERMUTATION_MAX_FEATURES = int(os.environ.get("EPM_LGBM_PERMUTATION_MAX_FEATURES", "0"))
LGBM_FINAL_FIT_MAX_ROWS = int(os.environ.get("EPM_LGBM_FINAL_FIT_MAX_ROWS", "200000"))
LGBM_HPO_LEARNING_RATE = float(os.environ.get("EPM_LGBM_HPO_LEARNING_RATE", "0.02"))
LGBM_FINAL_LEARNING_RATE = float(os.environ.get("EPM_LGBM_FINAL_LEARNING_RATE", "0.02"))

LGBM_CV_SPLITS = max(2, int(LGBM_CV_SPLITS))
LGBM_RACE_EVAL_FRACTION = float(np.clip(LGBM_RACE_EVAL_FRACTION, 0.10, 0.50))
LGBM_ROW_SUBSAMPLE_FRAC = float(np.clip(LGBM_ROW_SUBSAMPLE_FRAC, 0.01, 1.0))
LGBM_FINAL_MODEL_COUNT = max(1, int(LGBM_FINAL_MODEL_COUNT))
LGBM_OOF_DISTILLATION_PASSES = max(0, int(LGBM_OOF_DISTILLATION_PASSES))


@dataclass
class FeatureSelectionResult:
    feature_names: list[str]
    selected_features: list[str]
    history: list[dict[str, Any]]
    stats: pd.DataFrame
    oof_pred: np.ndarray
    metrics: dict[str, Any]
    stage_indices: dict[str, np.ndarray]


@dataclass
class LGBMStabilityModel:
    mode: str = "classifier"
    models: list[Any] = field(default_factory=list)
    selected_features: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    pruning_history: list[dict[str, Any]] = field(default_factory=list)
    oof_probs: Optional[np.ndarray] = None
    best_params: dict[str, Any] = field(default_factory=dict)

    def _frame(self, X: Any) -> pd.DataFrame:
        X_df = _frame(X)
        for col in self.selected_features:
            if col not in X_df.columns:
                X_df[col] = 0.0
        return X_df.reindex(columns=self.selected_features, fill_value=0.0)

    def predict(self, X: Any) -> np.ndarray:
        X_df = self._frame(X)
        if not self.models:
            fill = 0.5 if self.mode == "classifier" else 0.0
            return np.full(len(X_df), fill, dtype=np.float32)
        preds = [_predict_lgbm_raw(model, X_df, self.mode) for model in self.models]
        out = np.mean(np.vstack(preds), axis=0).astype(np.float32)
        if self.mode == "classifier":
            out = np.clip(out, 1e-5, 1.0 - 1e-5)
        return out.astype(np.float32)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = self.predict(X)
        if self.mode != "classifier":
            return np.column_stack([p, p]).astype(np.float32)
        return np.column_stack([1.0 - p, p]).astype(np.float32)


def _frame(X: Any) -> pd.DataFrame:
    X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    X_df = X_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    X_df.columns = [str(c) for c in X_df.columns]
    return X_df


def _looks_classifier_target(y: np.ndarray) -> bool:
    yy = np.asarray(y)
    finite = yy[np.isfinite(yy)]
    if len(finite) == 0:
        return True
    unique = np.unique(finite)
    return bool(len(unique) <= 20 and np.all(np.isclose(unique, np.round(unique))))


def _coerce_target(y: np.ndarray, classifier: bool) -> np.ndarray:
    if classifier:
        return np.asarray(y >= 0.5, dtype=np.int8)
    return np.asarray(y, dtype=np.float32)


def _as_returns(y: np.ndarray, returns: Any = None) -> np.ndarray:
    if returns is None:
        return np.asarray(y, dtype=np.float32)
    arr = np.asarray(returns, dtype=np.float32)
    if len(arr) != len(y):
        raise ValueError("returns must have the same length as y")
    return arr


def _rank01(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    out = np.zeros(len(vals), dtype=np.float32)
    if int(np.sum(finite)) <= 1:
        return out
    finite_vals = vals[finite]
    span = float(np.nanmax(finite_vals) - np.nanmin(finite_vals))
    if span <= 1e-12:
        return out
    out[finite] = pd.Series(finite_vals).rank(pct=True).to_numpy(dtype=np.float32)
    return out


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    m = np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(m)) < 8:
        return 0.0
    if float(np.nanstd(aa[m])) <= 1e-12 or float(np.nanstd(bb[m])) <= 1e-12:
        return 0.0
    val = spearmanr(aa[m], bb[m]).correlation
    return float(val) if val is not None and np.isfinite(val) else 0.0


def _top_idx(order: np.ndarray, frac: float, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    k = max(1, int(np.ceil(float(frac) * n)))
    return np.asarray(order[-k:], dtype=np.int64)


def _unit_interval(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _normalize_precision(precision: float, baseline: float) -> float:
    if not (np.isfinite(precision) and np.isfinite(baseline)):
        return 0.0
    return _unit_interval((float(precision) - float(baseline)) / max(1.0 - float(baseline), 1e-6))


def _normalize_return(value: float, scale: float) -> float:
    if not np.isfinite(value):
        return 0.0
    scale = max(float(scale), 1e-6)
    return _unit_interval(0.5 + 0.5 * np.tanh(float(value) / scale))


def _ndcg_at_k(y_true: np.ndarray, pred: np.ndarray, k: int) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    if len(y) <= 1:
        return 0.0
    k = max(1, min(int(k), len(y)))
    order = np.argsort(p)[-k:][::-1]
    ideal = np.argsort(y)[-k:][::-1]
    gains = np.maximum(y, 0.0)
    dcg = float(np.sum(gains[order] / np.log2(np.arange(2, k + 2))))
    idcg = float(np.sum(gains[ideal] / np.log2(np.arange(2, k + 2))))
    if idcg <= 1e-12:
        return 0.0
    return float(np.clip(dcg / idcg, 0.0, 1.0))


def _bucket_monotonicity_score(y_win: np.ndarray, order: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_win, dtype=np.float64)
    if len(y) == 0:
        return {
            "rank_bucket_monotonicity": 0.0,
            "rank_bucket_monotonicity_violation": 1.0,
            "rank_bucket_win_rate_top10": 0.0,
            "rank_bucket_win_rate_10_15": 0.0,
            "rank_bucket_win_rate_15_20": 0.0,
            "rank_bucket_win_rate_20_25": 0.0,
            "rank_bucket_win_rate_25_30": 0.0,
        }
    sorted_y = y[np.asarray(order, dtype=np.int64)[::-1]]
    n = len(sorted_y)
    bounds = np.asarray(
        [
            0,
            max(1, int(np.ceil(0.10 * n))),
            max(1, int(np.ceil(0.15 * n))),
            max(1, int(np.ceil(0.20 * n))),
            max(1, int(np.ceil(0.25 * n))),
            max(1, int(np.ceil(0.30 * n))),
        ],
        dtype=np.int64,
    )
    bounds = np.maximum.accumulate(np.clip(bounds, 0, n))
    sums = np.concatenate([[0.0], np.cumsum(sorted_y, dtype=np.float64)])
    bucket_n = np.maximum(bounds[1:] - bounds[:-1], 1)
    rates = (sums[bounds[1:]] - sums[bounds[:-1]]) / bucket_n
    diffs = rates[1:] - rates[:-1]
    violation = float(np.mean(np.maximum(diffs, 0.0)))
    return {
        "rank_bucket_monotonicity": float(np.clip(1.0 - violation, 0.0, 1.0)),
        "rank_bucket_monotonicity_violation": violation,
        "rank_bucket_win_rate_top10": float(rates[0]),
        "rank_bucket_win_rate_10_15": float(rates[1]),
        "rank_bucket_win_rate_15_20": float(rates[2]),
        "rank_bucket_win_rate_20_25": float(rates[3]),
        "rank_bucket_win_rate_25_30": float(rates[4]),
    }


def _grouped_top_stability(
    y: np.ndarray,
    pred: np.ndarray,
    classifier: bool,
    groups: Any = None,
    frac: float = 0.20,
    min_groups: int = 3,
    min_group_n: int = 20,
) -> dict[str, float]:
    if groups is None:
        return {"stability": 0.0, "n_groups": 0.0, "group_mean": 0.0, "group_std": 0.0}
    yy = np.asarray(y, dtype=np.float64)
    pp = np.asarray(pred, dtype=np.float64)
    gg = np.asarray(groups, dtype=object)
    n = min(len(yy), len(pp), len(gg))
    yy = yy[:n]
    pp = pp[:n]
    gg = gg[:n]
    m = np.isfinite(yy) & np.isfinite(pp)
    yy = yy[m]
    pp = pp[m]
    gg = gg[m]
    vals: list[float] = []
    for group in pd.unique(pd.Series(gg)):
        mask = gg == group
        if int(np.sum(mask)) < int(min_group_n):
            continue
        yg = yy[mask]
        pg = pp[mask]
        k = max(1, int(np.ceil(float(frac) * len(pg))))
        if k >= len(pg):
            continue
        top = np.argsort(pg)[-k:]
        if classifier:
            yb = (yg >= 0.5).astype(np.int8)
            base = float(np.mean(yb))
            if base <= 1e-6:
                continue
            vals.append(float(np.mean(yb[top]) / base))
        else:
            denom = float(np.mean(np.abs(yg))) + 1e-6
            vals.append(float(np.mean(yg[top]) / denom))
    if len(vals) < int(min_groups):
        return {
            "stability": 0.0,
            "n_groups": float(len(vals)),
            "group_mean": float(np.mean(vals)) if vals else 0.0,
            "group_std": float(np.std(vals)) if vals else 0.0,
        }
    arr = np.asarray(vals, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv_v = std_v / (abs(mean_v) + 1e-6)
    return {
        "stability": float(np.clip(1.0 / (1.0 + cv_v), 0.0, 1.0)),
        "n_groups": float(len(arr)),
        "group_mean": mean_v,
        "group_std": std_v,
    }


def _stability_group_bundle(n: int, timestamps: Any = None, assets: Any = None) -> dict[str, np.ndarray] | None:
    if n <= 0:
        return None
    out: dict[str, np.ndarray] = {}
    if timestamps is not None and len(np.asarray(timestamps)) == n:
        ts = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
        week = pd.Series(ts).dt.tz_localize(None).dt.to_period("W").astype(str).to_numpy(dtype=object)
        month = pd.Series(ts).dt.tz_localize(None).dt.to_period("M").astype(str).to_numpy(dtype=object)
        out["week"] = np.where(pd.isna(week), "__unknown_week__", week).astype(str)
        out["month"] = np.where(pd.isna(month), "__unknown_month__", month).astype(str)
    if assets is not None and len(np.asarray(assets)) == n:
        asset_arr = np.asarray(assets).astype(str)
        counts = pd.Series(asset_arr).value_counts()
        common = set(counts[counts >= 20].index.astype(str))
        out["asset"] = np.asarray([a if a in common else "__rare_asset__" for a in asset_arr], dtype=object).astype(str)
    if not out:
        return None
    if "week" not in out:
        out["week"] = out.get("asset", np.asarray(["__all__"] * n, dtype=str))
    return out


def _groups_take(groups: Any, idx: Any) -> Any:
    if groups is None:
        return None
    if isinstance(groups, dict):
        return {k: np.asarray(v, dtype=object)[idx] for k, v in groups.items()}
    return np.asarray(groups, dtype=object)[idx]


def _groups_primary(groups: Any) -> Any:
    if isinstance(groups, dict):
        return groups.get("week")
    return groups


def _metric_pack(
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    classifier: bool,
    groups: Any = None,
    returns: Any = None,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    ret = _as_returns(y, returns).astype(np.float64)
    m = np.isfinite(y) & np.isfinite(p) & np.isfinite(ret)
    if isinstance(groups, dict):
        grp = {k: np.asarray(v, dtype=object)[m] for k, v in groups.items() if len(np.asarray(v)) == len(m)}
    else:
        grp = np.asarray(groups, dtype=object)[m] if groups is not None and len(np.asarray(groups)) == len(m) else None
    y = y[m]
    p = p[m]
    ret = ret[m]
    if len(y) < 8:
        return {
            "J_meta": 0.0,
            "J_final": 0.0,
            "J_Score": 0.0,
            "precision10_norm": 0.0,
            "precision20_norm": 0.0,
            "ndcg_at_10": 0.0,
            "ndcg_at_20": 0.0,
            "rank_bucket_monotonicity": 0.0,
            "stability20": 0.0,
            "lift20": 1.0,
        }
    order = np.argsort(p)
    top10 = _top_idx(order, 0.10, len(y))
    top20 = _top_idx(order, 0.20, len(y))
    top30 = _top_idx(order, 0.30, len(y))
    if classifier:
        y_win = (y >= 0.5).astype(np.float64)
        baseline = float(np.mean(y_win))
        precision10 = float(np.mean(y_win[top10])) if len(top10) else 0.0
        precision20 = float(np.mean(y_win[top20])) if len(top20) else 0.0
        precision30 = float(np.mean(y_win[top30])) if len(top30) else 0.0
        lift10 = precision10 / max(baseline, 1e-6)
        lift20 = precision20 / max(baseline, 1e-6)
        lift30 = precision30 / max(baseline, 1e-6)
        auc = float(roc_auc_score(y_win, p)) if len(np.unique(y_win)) > 1 else 0.5
        pr_auc = float(average_precision_score(y_win, p)) if len(np.unique(y_win)) > 1 else baseline
        brier = float(brier_score_loss(y_win, np.clip(p, 1e-6, 1.0 - 1e-6)))
    else:
        y_win = (y > 0.0).astype(np.float64)
        baseline = float(np.mean(y_win))
        precision10 = float(np.mean(y_win[top10])) if len(top10) else 0.0
        precision20 = float(np.mean(y_win[top20])) if len(top20) else 0.0
        precision30 = float(np.mean(y_win[top30])) if len(top30) else 0.0
        denom = float(np.mean(np.abs(y))) + 1e-6
        lift10 = float(np.mean(y[top10]) / denom) if len(top10) else 0.0
        lift20 = float(np.mean(y[top20]) / denom) if len(top20) else 0.0
        lift30 = float(np.mean(y[top30]) / denom) if len(top30) else 0.0
        auc = max(0.0, _safe_spearman(y, p))
        pr_auc = auc
        brier = float(np.mean((p - y) ** 2))
    ret_scale = float(np.nanpercentile(np.abs(ret), 75.0) + 1e-6)
    mean_ret10 = float(np.mean(ret[top10])) if len(top10) else 0.0
    mean_ret20 = float(np.mean(ret[top20])) if len(top20) else 0.0
    norm_ret10 = _normalize_return(mean_ret10, ret_scale)
    norm_ret20 = _normalize_return(mean_ret20, ret_scale)
    precision10_norm = _normalize_precision(precision10, baseline)
    precision20_norm = _normalize_precision(precision20, baseline)
    ndcg10 = _ndcg_at_k(ret, p, k=10)
    ndcg20 = _ndcg_at_k(ret, p, k=20)
    mono = _bucket_monotonicity_score(y_win, order)
    stability = _grouped_top_stability(y, p, classifier, groups=_groups_primary(grp), frac=0.20)
    stability20 = float(stability["stability"])
    if stability20 <= 0.0:
        top_vals = y_win[top20] if len(top20) else np.asarray([], dtype=np.float64)
        stability20 = float(1.0 / (1.0 + np.std(top_vals))) if len(top_vals) else 0.0
    net_return_blend = 0.60 * norm_ret10 + 0.40 * norm_ret20
    precision_blend = 0.60 * precision10_norm + 0.40 * precision20_norm
    ndcg_blend = 0.60 * ndcg10 + 0.40 * ndcg20
    j_meta = float(
        0.35 * net_return_blend
        + 0.25 * precision_blend
        + 0.15 * ndcg_blend
        + 0.15 * float(mono["rank_bucket_monotonicity"])
        + 0.10 * stability20
    )
    out = {
        "J_meta": j_meta,
        "J_final": j_meta,
        "J_Score": j_meta,
        "net_return_blend": float(net_return_blend),
        "normalized_net_mean_ret10": float(norm_ret10),
        "normalized_net_mean_ret20": float(norm_ret20),
        "mean_ret10": float(mean_ret10),
        "mean_ret20": float(mean_ret20),
        "precision_blend": float(precision_blend),
        "precision10": float(precision10),
        "precision20": float(precision20),
        "precision30": float(precision30),
        "precision10_norm": float(precision10_norm),
        "precision20_norm": float(precision20_norm),
        "NDCG_blend": float(ndcg_blend),
        "ndcg_at_10": float(ndcg10),
        "ndcg_at_20": float(ndcg20),
        "lift10": float(lift10),
        "lift20": float(lift20),
        "lift30": float(lift30),
        "baseline_win_rate": float(baseline),
        "stability20": float(stability20),
        "stability20_n_groups": float(stability.get("n_groups", 0.0)),
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "brier": float(brier),
        "oof_std": float(np.std(p)),
    }
    out.update(mono)
    return out


def _aggregate_j(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not fold_metrics:
        return {"J_final": -999.0, "J_mean": -999.0, "J_std": 0.0, "J_se": 0.0, "J_median": -999.0, "J_iqr": 0.0, "J_robust": -999.0}
    vals = np.asarray([float(m.get("J_meta", m.get("J_final", np.nan))) for m in fold_metrics], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"J_final": -999.0, "J_mean": -999.0, "J_std": 0.0, "J_se": 0.0, "J_median": -999.0, "J_iqr": 0.0, "J_robust": -999.0}
    q25, q50, q75 = np.percentile(vals, [25, 50, 75])
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    se = float(std / np.sqrt(max(len(vals), 1)))
    robust = float(q50 - 0.50 * (q75 - q25))
    means: dict[str, float] = {}
    for key in sorted(set().union(*(m.keys() for m in fold_metrics))):
        arr = np.asarray([float(m.get(key, np.nan)) for m in fold_metrics], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            means[key] = float(np.mean(arr))
    means.update({"J_final": robust, "J_mean": float(np.mean(vals)), "J_std": std, "J_se": se, "J_median": float(q50), "J_iqr": float(q75 - q25), "J_robust": robust})
    return means


def _stratified_subsample_indices(y: np.ndarray, max_n: int, random_state: int, classifier: bool) -> np.ndarray:
    n = len(y)
    if n <= max_n:
        return np.arange(n, dtype=np.int32)
    rng = np.random.default_rng(random_state)
    if classifier:
        strata = np.asarray(y >= 0.5, dtype=np.int8)
    else:
        ranks = pd.Series(np.asarray(y, dtype=np.float32)).rank(pct=True).to_numpy()
        strata = np.clip((ranks * 5).astype(np.int32), 0, 4)
    out: list[np.ndarray] = []
    for s in np.unique(strata):
        ids = np.where(strata == s)[0]
        take = max(1, int(round(max_n * len(ids) / n)))
        take = min(take, len(ids))
        out.append(rng.choice(ids, size=take, replace=False))
    idx = np.sort(np.concatenate(out).astype(np.int32))
    if len(idx) > max_n:
        idx = np.sort(rng.choice(idx, size=max_n, replace=False).astype(np.int32))
    return idx


def _stage_partition_indices(y: np.ndarray, *, timestamps: Any = None, assets: Any = None, random_state: int) -> dict[str, np.ndarray]:
    y_arr = np.asarray(y)
    n = len(y_arr)
    if n == 0:
        empty = np.array([], dtype=np.int32)
        return {"lgbm_select": empty, "hpo": empty, "fit_oof": empty}
    classifier = _looks_classifier_target(y_arr)
    if classifier:
        y_bucket = np.asarray(y_arr >= 0.5, dtype=np.int8).astype(str)
    else:
        ranks = pd.Series(np.asarray(y_arr, dtype=np.float32)).rank(pct=True).to_numpy()
        y_bucket = np.clip((ranks * 5).astype(np.int32), 0, 4).astype(str)
    if assets is not None and len(np.asarray(assets)) == n:
        asset_arr = np.asarray(assets).astype(str)
        counts = pd.Series(asset_arr).value_counts()
        common = set(counts[counts >= 20].index.astype(str))
        asset_bucket = np.asarray([a if a in common else "__rare_asset__" for a in asset_arr], dtype=object)
    else:
        asset_bucket = np.asarray(["__all_assets__"] * n, dtype=object)
    if timestamps is not None and len(np.asarray(timestamps)) == n:
        ts = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
        if bool(pd.Series(ts).notna().any()):
            week = pd.Series(ts).dt.tz_localize(None).dt.to_period("W").astype(str).to_numpy()
            week_rank = pd.Series(week).rank(method="dense").to_numpy(dtype=np.int32)
        else:
            week_rank = np.arange(n, dtype=np.int32)
    else:
        week_rank = np.arange(n, dtype=np.int32)
    strata = np.asarray([f"{yb}|{ab}" for yb, ab in zip(y_bucket, asset_bucket)], dtype=object)
    rng = np.random.default_rng(random_state)
    pattern = np.asarray(["lgbm_select"] * 7 + ["hpo"] * 2 + ["fit_oof"] * 11)
    out: dict[str, list[int]] = {"lgbm_select": [], "hpo": [], "fit_oof": []}
    for stratum in np.unique(strata):
        ids = np.where(strata == stratum)[0]
        jitter = rng.random(len(ids)) * 1e-6
        order = np.lexsort((jitter, np.arange(len(ids)) % 997, week_rank[ids]))
        ordered = ids[order]
        offset = int(rng.integers(0, len(pattern)))
        labels = pattern[(np.arange(len(ordered)) + offset) % len(pattern)]
        for key in out:
            out[key].extend(ordered[labels == key].tolist())
    result = {key: np.asarray(sorted(vals), dtype=np.int32) for key, vals in out.items()}
    assigned = np.concatenate([v for v in result.values() if len(v)])
    missing = np.setdiff1d(np.arange(n, dtype=np.int32), assigned, assume_unique=False)
    if len(missing):
        result["fit_oof"] = np.asarray(sorted(np.concatenate([result["fit_oof"], missing]).tolist()), dtype=np.int32)
    tprint(
        "LGBM stage split: "
        f"select={len(result['lgbm_select'])}/{n}, hpo={len(result['hpo'])}/{n}, "
        f"fit_oof={len(result['fit_oof'])}/{n}."
    )
    return result


def _subsample_stage_indices(stage_indices: dict[str, np.ndarray], y: np.ndarray, *, max_fraction: float, random_state: int, classifier: bool) -> dict[str, np.ndarray]:
    frac = float(np.clip(float(max_fraction), 0.01, 1.0))
    if frac >= 0.999:
        return stage_indices
    n = len(y)
    cap = max(1, int(np.ceil(frac * max(n, 1))))
    out = dict(stage_indices)
    for offset, stage_key in enumerate(("lgbm_select", "hpo", "fit_oof"), start=1):
        idx = np.asarray(out.get(stage_key, []), dtype=np.int32)
        if len(idx) <= cap:
            continue
        keep_local = _stratified_subsample_indices(np.asarray(y, dtype=np.float32)[idx], max_n=cap, random_state=int(random_state) + offset * 10007, classifier=classifier)
        out[stage_key] = np.sort(idx[keep_local].astype(np.int32))
    return out


def _cap_stage_and_move_unused_to_fit_oof(stage_indices: dict[str, np.ndarray], y: np.ndarray, *, stage_key: str, cap: int, random_state: int, classifier: bool) -> dict[str, np.ndarray]:
    if cap <= 0:
        return stage_indices
    idx = np.asarray(stage_indices.get(stage_key, []), dtype=np.int32)
    if len(idx) <= cap:
        return stage_indices
    keep_local = _stratified_subsample_indices(np.asarray(y, dtype=np.float32)[idx], max_n=int(cap), random_state=int(random_state), classifier=classifier)
    keep = np.sort(idx[keep_local].astype(np.int32))
    unused = np.setdiff1d(idx, keep, assume_unique=False).astype(np.int32)
    out = dict(stage_indices)
    out[stage_key] = keep
    out["fit_oof"] = np.asarray(sorted(np.unique(np.concatenate([out.get("fit_oof", np.array([], dtype=np.int32)), unused])).tolist()), dtype=np.int32)
    return out


def _splitter(y: np.ndarray, classifier: bool, random_state: int, n_splits: int = LGBM_CV_SPLITS) -> Any:
    y_split = np.asarray(y >= 0.5, dtype=np.int8) if classifier else np.asarray(y, dtype=np.float32)
    if classifier and len(np.unique(y_split)) > 1 and np.min(np.bincount(y_split, minlength=2)) >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), y_split
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state), y_split


def _direction_score_for_feature(x: np.ndarray, y: np.ndarray, *, classifier: bool, groups: Any = None, returns: Any = None) -> tuple[float, float, int, float]:
    x_arr = np.asarray(x, dtype=np.float32)
    if float(np.nanstd(x_arr)) <= 1e-12:
        return 0.0, 0.0, 0, 0.0
    j_pos = _metric_pack(y, x_arr, classifier=classifier, groups=groups, returns=returns)["J_meta"]
    j_neg = _metric_pack(y, -x_arr, classifier=classifier, groups=groups, returns=returns)["J_meta"]
    direction = 1 if j_pos >= j_neg else -1
    margin = abs(float(j_pos) - float(j_neg))
    return float(j_pos), float(j_neg), int(direction), float(margin)


def _weighted_direction_stability(directions: np.ndarray, margins: np.ndarray) -> float:
    d = np.asarray(directions, dtype=np.float64)
    w = np.asarray(margins, dtype=np.float64)
    m = np.isfinite(d) & np.isfinite(w) & (w > 0)
    if int(np.sum(m)) == 0 or float(np.sum(w[m])) <= 1e-12:
        return 0.0
    return float(abs(np.sum(d[m] * w[m]) / np.sum(w[m])))


def _univariate_directional_filter(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    classifier: bool,
    groups: Any = None,
    returns: Any = None,
    random_state: int,
) -> tuple[list[str], pd.DataFrame]:
    names = list(X.columns)
    splitter, y_split = _splitter(y, classifier, random_state, n_splits=LGBM_CV_SPLITS)
    records: list[dict[str, Any]] = []
    for fi, name in enumerate(names):
        x = X[name].to_numpy(dtype=np.float32)
        if float(np.nanstd(x)) <= 1e-12:
            records.append({"feature": name, "passed": False, "univariate_j": 0.0, "direction_stability": 0.0})
            continue
        j_pos_vals: list[float] = []
        j_neg_vals: list[float] = []
        p20_norm_vals: list[float] = []
        lift20_vals: list[float] = []
        mono_vals: list[float] = []
        dirs: list[int] = []
        margins: list[float] = []
        for _tr, va in splitter.split(np.zeros(len(y_split)), y_split):
            x_va = x[va]
            y_va = y[va]
            grp_va = _groups_take(groups, va)
            ret_va = _as_returns(y, returns)[va]
            j_pos, j_neg, direction, margin = _direction_score_for_feature(x_va, y_va, classifier=classifier, groups=grp_va, returns=ret_va)
            pred = x_va if direction >= 0 else -x_va
            metrics = _metric_pack(y_va, pred, classifier=classifier, groups=grp_va, returns=ret_va)
            j_pos_vals.append(j_pos)
            j_neg_vals.append(j_neg)
            p20_norm_vals.append(float(metrics.get("precision20_norm", 0.0)))
            lift20_vals.append(float(metrics.get("lift20", 1.0)))
            mono_vals.append(float(metrics.get("rank_bucket_monotonicity", 0.0)))
            dirs.append(direction)
            margins.append(margin)
        j_pos_med = float(np.median(j_pos_vals)) if j_pos_vals else 0.0
        j_neg_med = float(np.median(j_neg_vals)) if j_neg_vals else 0.0
        direction = 1 if j_pos_med >= j_neg_med else -1
        direction_stability = _weighted_direction_stability(np.asarray(dirs), np.asarray(margins))
        univariate_j = max(j_pos_med, j_neg_med)
        precision_pass = float(np.median(p20_norm_vals)) > 0.0 if p20_norm_vals else False
        lift_pass = float(np.median(lift20_vals)) > 1.0 if lift20_vals else False
        mono_pass = float(np.median(mono_vals)) >= LGBM_UNIVARIATE_MONOTONICITY_MIN if mono_vals else False
        passed = bool((precision_pass or lift_pass or mono_pass) and direction_stability >= LGBM_DIRECTION_STABILITY_MIN)
        records.append(
            {
                "feature": name,
                "feature_index": int(fi),
                "passed": passed,
                "univariate_j": float(univariate_j),
                "J_pos_median": j_pos_med,
                "J_neg_median": j_neg_med,
                "direction": int(direction),
                "direction_stability": float(direction_stability),
                "direction_margin_median": float(np.median(margins)) if margins else 0.0,
                "precision20_norm_median": float(np.median(p20_norm_vals)) if p20_norm_vals else 0.0,
                "lift20_median": float(np.median(lift20_vals)) if lift20_vals else 1.0,
                "monotonicity_median": float(np.median(mono_vals)) if mono_vals else 0.0,
                "pass_precision": bool(precision_pass),
                "pass_lift": bool(lift_pass),
                "pass_monotonicity": bool(mono_pass),
            }
        )
    stats = pd.DataFrame(records)
    selected = stats.loc[stats["passed"].astype(bool), "feature"].astype(str).tolist()
    if len(selected) < min(LGBM_MIN_FEATURES, len(names)):
        rescue = stats.sort_values("univariate_j", ascending=False)["feature"].astype(str).head(min(LGBM_MIN_FEATURES, len(names))).tolist()
        selected = sorted(set(selected).union(rescue), key=lambda c: names.index(c))
    tprint(f"LGBM univariate filter: {len(names)} -> {len(selected)} features.")
    return selected, stats


def _redundancy_cluster_filter(
    X: pd.DataFrame,
    features: list[str],
    score_map: dict[str, float],
    *,
    random_state: int,
    corr_threshold: float = LGBM_REDUNDANCY_CORR_THRESHOLD,
) -> list[str]:
    if len(features) <= 2:
        return list(features)
    rng = np.random.default_rng(random_state)
    sub_n = min(len(X), 5000)
    sub = rng.choice(len(X), size=sub_n, replace=False) if len(X) > sub_n else np.arange(len(X))
    arr = X.iloc[sub][features].to_numpy(dtype=np.float32)
    ranks = pd.DataFrame(arr).rank(pct=True).to_numpy(dtype=np.float32)
    corr = np.abs(np.corrcoef(ranks, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if corr.shape[0] != len(features):
        return list(features)
    dist = 1.0 - np.clip(corr, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    if not np.any(np.isfinite(condensed)):
        return list(features)
    z = linkage(condensed, method="average")
    labels = fcluster(z, t=1.0 - float(corr_threshold), criterion="distance")
    keep: list[str] = []
    for lab in sorted(set(labels)):
        members = [features[i] for i in np.where(labels == lab)[0]]
        members_sorted = sorted(members, key=lambda f: float(score_map.get(f, 0.0)), reverse=True)
        keep_n = min(3, max(1, int(np.ceil(0.25 * len(members_sorted)))))
        keep.extend(members_sorted[:keep_n])
    keep_ordered = [f for f in features if f in set(keep)]
    tprint(f"LGBM redundancy clusters: {len(features)} -> {len(keep_ordered)} features.")
    return keep_ordered


def _base_lgbm_params(seed: int, *, classifier: bool, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "binary" if classifier else "regression",
        "n_estimators": 800,
        "learning_rate": 0.02,
        "max_depth": 4,
        "num_leaves": 16,
        "min_child_samples": 300,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.60,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "min_split_gain": 0.01,
        "random_state": int(seed),
        "n_jobs": 1,
        "verbosity": -1,
    }
    if overrides:
        params.update(overrides)
    depth = int(params.get("max_depth", 4))
    if "num_leaves" not in params or params.get("num_leaves") is None:
        params["num_leaves"] = int(2 ** depth)
    params["num_leaves"] = int(min(int(params["num_leaves"]), 2 ** max(depth, 1)))
    return params


def _make_lgbm_model(params: dict[str, Any], classifier: bool) -> Any:
    import lightgbm as lgb

    if classifier:
        return lgb.LGBMClassifier(**params)
    return lgb.LGBMRegressor(**params)


def _fit_lgbm_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    *,
    classifier: bool,
    params: dict[str, Any],
    X_valid: pd.DataFrame | None = None,
    y_valid: np.ndarray | None = None,
    early_stopping_rounds: int | None = None,
) -> Any:
    import lightgbm as lgb

    model = _make_lgbm_model(dict(params), classifier)
    callbacks = []
    eval_set = None
    if X_valid is not None and y_valid is not None and early_stopping_rounds and len(y_valid) > 10:
        callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))
        eval_set = [(X_valid, y_valid)]
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float32)
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set
        fit_kwargs["callbacks"] = callbacks
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def _predict_lgbm_raw(model: Any, X: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "classifier" and hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X), dtype=np.float64)
        if p.ndim == 2 and p.shape[1] > 1:
            return np.clip(p[:, 1], 1e-6, 1.0 - 1e-6).astype(np.float32)
        return np.clip(p.reshape(-1), 1e-6, 1.0 - 1e-6).astype(np.float32)
    return np.asarray(model.predict(X), dtype=np.float32).reshape(-1)


def _false_positive_avoidance_weight(
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    classifier: bool,
    top_frac: float = 0.20,
    fp_upweight: float = 1.60,
    top_positive_upweight: float = 1.25,
    max_weight: float = 4.0,
) -> np.ndarray:
    if not classifier:
        return np.ones(len(pred), dtype=np.float32)
    yb = np.asarray(y_true, dtype=np.float32)
    pp = np.nan_to_num(np.asarray(pred, dtype=np.float32), nan=-np.inf)
    if len(pp) == 0:
        return np.ones(0, dtype=np.float32)
    rank_pct = pd.Series(pp).rank(method="average", pct=True).to_numpy(dtype=np.float32)
    top_mask = rank_pct >= 1.0 - float(np.clip(top_frac, 0.001, 0.95))
    support_mask = rank_pct >= 1.0 - float(np.clip(1.5 * top_frac, top_frac, 0.95))
    w = np.ones(len(pp), dtype=np.float32)
    w[(yb < 0.5) & top_mask] = fp_upweight
    w[(yb >= 0.5) & support_mask] = np.maximum(w[(yb >= 0.5) & support_mask], top_positive_upweight)
    return np.clip(w, 0.25, max_weight).astype(np.float32)


def _normalize_weights(weights: np.ndarray, *, min_weight: float = 0.25, max_weight: float = 4.0) -> tuple[np.ndarray, float]:
    w = np.nan_to_num(np.asarray(weights, dtype=np.float32), nan=1.0, posinf=max_weight, neginf=min_weight)
    w = np.clip(w, min_weight, max_weight)
    w = w / max(float(np.mean(w)), 1e-6)
    ess = float((w.sum() ** 2) / max(np.sum(w**2), 1e-6))
    return w.astype(np.float32), ess


def _drop_fraction(n_features: int) -> float:
    n = int(n_features)
    if n > 150:
        return 0.25
    if n > 100:
        return 0.20
    if n > 70:
        return 0.15
    return 0.05


def _importance_rank_scores(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    out = np.zeros(len(vals), dtype=np.float32)
    positive = np.isfinite(vals) & (vals > 0.0)
    if int(np.sum(positive)) <= 1:
        out[positive] = 1.0
        return out
    ranks = pd.Series(vals[positive]).rank(pct=True).to_numpy(dtype=np.float32)
    out[positive] = ranks
    return out


def _feature_importances(model: Any, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    gain = np.zeros(n_features, dtype=np.float32)
    split = np.zeros(n_features, dtype=np.float32)
    try:
        booster = getattr(model, "booster_", None)
        if booster is not None:
            gain_v = np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float32)
            split_v = np.asarray(booster.feature_importance(importance_type="split"), dtype=np.float32)
        else:
            gain_v = np.asarray(model.feature_importances_, dtype=np.float32)
            split_v = gain_v.copy()
        gain[: min(n_features, len(gain_v))] = gain_v[:n_features]
        split[: min(n_features, len(split_v))] = split_v[:n_features]
    except Exception:
        pass
    return gain, split


def _permutation_delta_j(
    model: Any,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    base_pred: np.ndarray,
    classifier: bool,
    groups_valid: Any,
    returns_valid: Any,
    rng: np.random.Generator,
    feature_indices: np.ndarray,
) -> np.ndarray:
    n_features = X_valid.shape[1]
    out = np.zeros(n_features, dtype=np.float32)
    base_j = float(_metric_pack(y_valid, base_pred, classifier=classifier, groups=groups_valid, returns=returns_valid)["J_meta"])
    if not np.isfinite(base_j):
        return out
    Xp = X_valid.copy()
    for j in feature_indices:
        vals = Xp.iloc[:, int(j)].to_numpy(copy=True)
        deltas: list[float] = []
        for _ in range(max(1, LGBM_PERMUTATION_REPEATS)):
            Xp.iloc[:, int(j)] = rng.permutation(vals)
            pred_perm = _predict_lgbm_raw(model, Xp, "classifier" if classifier else "regressor")
            perm_j = float(_metric_pack(y_valid, pred_perm, classifier=classifier, groups=groups_valid, returns=returns_valid)["J_meta"])
            deltas.append(base_j - perm_j)
        Xp.iloc[:, int(j)] = vals
        out[int(j)] = float(np.median(deltas)) if deltas else 0.0
    return out


def _redundancy_penalty(X: pd.DataFrame, features: list[str], quality: np.ndarray, *, random_state: int) -> np.ndarray:
    p = len(features)
    out = np.zeros(p, dtype=np.float32)
    if p <= 1:
        return out
    rng = np.random.default_rng(random_state)
    sub_n = min(len(X), 5000)
    sub = rng.choice(len(X), size=sub_n, replace=False) if len(X) > sub_n else np.arange(len(X))
    ranks = pd.DataFrame(X.iloc[sub][features].to_numpy(dtype=np.float32)).rank(pct=True).to_numpy(dtype=np.float32)
    corr = np.abs(np.corrcoef(ranks, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(np.asarray(quality, dtype=np.float32))[::-1]
    seen: list[int] = []
    for idx in order:
        if seen:
            max_corr = float(np.max(corr[int(idx), np.asarray(seen, dtype=np.int32)]))
            out[int(idx)] = max(0.0, (max_corr - LGBM_REDUNDANCY_PENALTY_START) / max(1.0 - LGBM_REDUNDANCY_PENALTY_START, 1e-6))
        seen.append(int(idx))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _lgbm_stability_selection_pass(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    features: list[str],
    *,
    classifier: bool,
    groups: Any = None,
    returns: Any = None,
    random_state: int,
    seeds: list[int] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    if seeds is None:
        seeds = [int(random_state)]
    Xf = X[features].reset_index(drop=True)
    y_arr = np.asarray(y)
    ret_arr = _as_returns(y_arr, returns)
    p = len(features)
    configs = []
    for depth in (4, 5):
        for l2 in (5.0, 15.0):
            configs.append({"max_depth": depth, "num_leaves": 2 ** depth, "reg_lambda": l2})
    n_fits = 0
    used_count = np.zeros(p, dtype=np.float32)
    top_used_count = np.zeros(p, dtype=np.float32)
    gain_rank_sum = np.zeros(p, dtype=np.float32)
    split_rank_sum = np.zeros(p, dtype=np.float32)
    perm_values: list[np.ndarray] = []
    direction_values: list[np.ndarray] = []
    margin_values: list[np.ndarray] = []
    fit_scores: list[float] = []
    fold_metrics_all: list[dict[str, float]] = []
    best_oof = np.full(len(y_arr), np.nan, dtype=np.float32)
    best_score = -np.inf
    base_weight = np.asarray(sample_weight, dtype=np.float32)
    current_weight = base_weight.copy()
    prev_oof: np.ndarray | None = None
    splitter, y_split = _splitter(y_arr, classifier, random_state, n_splits=3)
    rng_perm = np.random.default_rng(random_state + 107)
    all_fit_usage: list[np.ndarray] = []
    all_fit_gain_rank: list[np.ndarray] = []
    all_fit_split_rank: list[np.ndarray] = []
    all_fit_perm: list[np.ndarray] = []
    all_fit_direction: list[np.ndarray] = []
    all_fit_margin: list[np.ndarray] = []
    for seed in seeds:
        for cfg_i, cfg in enumerate(configs, start=1):
            cfg_oof = np.full(len(y_arr), np.nan, dtype=np.float32)
            cfg_metrics: list[dict[str, float]] = []
            for fold_i, (tr, va) in enumerate(splitter.split(np.zeros(len(y_split)), y_split), start=1):
                params = _base_lgbm_params(int(seed) + cfg_i * 1000 + fold_i, classifier=classifier, overrides=cfg)
                model = _fit_lgbm_model(
                    Xf.iloc[tr].reset_index(drop=True),
                    y_arr[tr],
                    current_weight[tr],
                    classifier=classifier,
                    params=params,
                )
                pred = _predict_lgbm_raw(model, Xf.iloc[va].reset_index(drop=True), "classifier" if classifier else "regressor")
                cfg_oof[va] = pred
                fold_groups = _groups_take(groups, va)
                fold_returns = ret_arr[va]
                metrics = _metric_pack(y_arr[va], pred, classifier=classifier, groups=fold_groups, returns=fold_returns)
                cfg_metrics.append(metrics)
                fit_scores.append(float(metrics.get("J_meta", metrics.get("J_final", np.nan))))
                gain, split = _feature_importances(model, p)
                used = (split > 0).astype(np.float32)
                all_fit_usage.append(used)
                all_fit_gain_rank.append(_importance_rank_scores(gain))
                all_fit_split_rank.append(_importance_rank_scores(split))
                if LGBM_PERMUTATION_MAX_FEATURES > 0 and p > LGBM_PERMUTATION_MAX_FEATURES:
                    quality = _importance_rank_scores(gain) + _importance_rank_scores(split)
                    candidate_idx = np.argsort(quality)[-LGBM_PERMUTATION_MAX_FEATURES:]
                else:
                    candidate_idx = np.arange(p, dtype=np.int32)
                perm = _permutation_delta_j(
                    model,
                    Xf.iloc[va].reset_index(drop=True),
                    y_arr[va],
                    base_pred=pred,
                    classifier=classifier,
                    groups_valid=fold_groups,
                    returns_valid=fold_returns,
                    rng=rng_perm,
                    feature_indices=np.asarray(candidate_idx, dtype=np.int32),
                )
                all_fit_perm.append(perm)
                d_vec = np.zeros(p, dtype=np.float32)
                m_vec = np.zeros(p, dtype=np.float32)
                Xva = Xf.iloc[va].reset_index(drop=True)
                for j in range(p):
                    _jp, _jn, direction, margin = _direction_score_for_feature(
                        Xva.iloc[:, j].to_numpy(dtype=np.float32),
                        y_arr[va],
                        classifier=classifier,
                        groups=fold_groups,
                        returns=fold_returns,
                    )
                    d_vec[j] = float(direction)
                    m_vec[j] = float(margin)
                all_fit_direction.append(d_vec)
                all_fit_margin.append(m_vec)
                n_fits += 1
            agg = _aggregate_j(cfg_metrics)
            cfg_score = float(agg.get("J_final", -np.inf))
            fold_metrics_all.extend(cfg_metrics)
            if cfg_score > best_score:
                best_score = cfg_score
                best_oof = cfg_oof.copy()
            distill = _compute_weight_distillation(y_arr, np.nan_to_num(cfg_oof, nan=float(np.mean(y_arr))), prev_oof, is_classifier=classifier, include_false_positive_focus=False)
            fp_weight = _false_positive_avoidance_weight(y_arr, np.nan_to_num(cfg_oof, nan=float(np.mean(y_arr))), classifier=classifier, top_frac=0.20)
            current_weight, ess = _normalize_weights(base_weight * distill * fp_weight)
            prev_oof = np.nan_to_num(cfg_oof, nan=float(np.mean(y_arr))).astype(np.float32)
            tprint(f"LGBM stability grid seed={seed} config={cfg_i}/{len(configs)} score={cfg_score:.4f} ess={ess:.1f}")
    if n_fits == 0:
        raise RuntimeError("No LGBM stability fits completed")
    usage = np.vstack(all_fit_usage).astype(np.float32)
    gain_rank = np.vstack(all_fit_gain_rank).astype(np.float32)
    split_rank = np.vstack(all_fit_split_rank).astype(np.float32)
    perm_mat = np.vstack(all_fit_perm).astype(np.float32)
    dirs = np.vstack(all_fit_direction).astype(np.float32)
    margins = np.vstack(all_fit_margin).astype(np.float32)
    fit_scores_arr = np.asarray(fit_scores, dtype=np.float32)
    top_threshold = float(np.nanmedian(fit_scores_arr)) if len(fit_scores_arr) else -np.inf
    top_mask = np.isfinite(fit_scores_arr) & (fit_scores_arr >= top_threshold)
    presence_rate = np.mean(usage > 0.0, axis=0).astype(np.float32)
    top_count = np.sum(usage[top_mask] > 0.0, axis=0).astype(np.float32) if np.any(top_mask) else np.zeros(p, dtype=np.float32)
    gain_rank_score = np.mean(gain_rank, axis=0).astype(np.float32)
    split_rank_score = np.mean(split_rank, axis=0).astype(np.float32)
    median_perm = np.median(perm_mat, axis=0).astype(np.float32)
    positive_perm_rate = np.mean(perm_mat > LGBM_PERMUTATION_EPS, axis=0).astype(np.float32)
    direction_stability = np.asarray([_weighted_direction_stability(dirs[:, j], margins[:, j]) for j in range(p)], dtype=np.float32)
    direction = np.asarray([1 if np.sum(dirs[:, j] * margins[:, j]) >= 0 else -1 for j in range(p)], dtype=np.int8)
    norm_perm = _rank01(np.maximum(median_perm, 0.0))
    prelim_quality = 0.50 * norm_perm + 0.25 * positive_perm_rate + 0.15 * presence_rate + 0.10 * direction_stability
    redundancy = _redundancy_penalty(Xf, features, prelim_quality, random_state=random_state + 677)
    feature_score = (
        0.40 * norm_perm
        + 0.20 * positive_perm_rate
        + 0.15 * presence_rate
        + 0.10 * direction_stability
        + 0.075 * gain_rank_score
        + 0.075 * split_rank_score
        - 0.10 * redundancy
    ).astype(np.float32)
    hard_drop = (direction_stability < LGBM_DIRECTION_STABILITY_MIN) | (positive_perm_rate < LGBM_POSITIVE_PERM_RATE_MIN) | ((median_perm < -LGBM_PERMUTATION_EPS) & (presence_rate < LGBM_LOW_PRESENCE_RATE))
    rescue = (top_count >= max(2.0, 0.25 * max(float(np.sum(top_mask)), 1.0))) & (gain_rank_score >= 0.75) & (split_rank_score >= 0.75)
    hard_drop = hard_drop & ~rescue
    feature_score = np.where(hard_drop, -1.0, feature_score).astype(np.float32)
    stats = pd.DataFrame(
        {
            "feature": features,
            "feature_score": feature_score,
            "normalized_permutation_delta_J": norm_perm,
            "median_permutation_delta_J": median_perm,
            "positive_permutation_rate": positive_perm_rate,
            "presence_rate": presence_rate,
            "direction": direction,
            "direction_stability": direction_stability,
            "gain_rank_score": gain_rank_score,
            "split_rank_score": split_rank_score,
            "selected_in_top_model_count": top_count,
            "redundancy_penalty": redundancy,
            "hard_drop": hard_drop.astype(bool),
            "rescue": rescue.astype(bool),
        }
    )
    agg_all = _aggregate_j(fold_metrics_all)
    return stats, np.nan_to_num(best_oof, nan=float(np.mean(y_arr))).astype(np.float32), agg_all


def _select_smallest_within_one_se(history: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [h for h in history if h.get("active_features")]
    if not valid:
        return {}
    best = max(valid, key=lambda h: float(h.get("J_final", -np.inf)))
    best_score = float(best.get("J_final", -np.inf))
    one_se = float(best.get("J_se", 0.0))
    floor = best_score - max(one_se, 0.0)
    close = [h for h in valid if float(h.get("J_final", -np.inf)) >= floor]
    if not close:
        close = [best]
    chosen = min(close, key=lambda h: (int(h.get("n_features", 10**9)), -float(h.get("J_final", -np.inf))))
    out = dict(chosen)
    out["selection_best_J"] = best_score
    out["selection_one_se"] = one_se
    out["selection_floor"] = floor
    out["selection_policy"] = "smallest_within_one_se"
    return out


def _iterative_feature_prune(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    initial_features: list[str],
    *,
    classifier: bool,
    groups: Any = None,
    returns: Any = None,
    random_state: int,
) -> tuple[list[str], list[dict[str, Any]], pd.DataFrame, np.ndarray, dict[str, Any]]:
    active = list(initial_features)
    history: list[dict[str, Any]] = []
    last_stats = pd.DataFrame()
    last_oof = np.full(len(y), float(np.mean(y)), dtype=np.float32)
    last_metrics: dict[str, Any] = {}
    for round_id in range(1, LGBM_MAX_ROUNDS + 1):
        if len(active) <= LGBM_MIN_FEATURES:
            break
        tprint(f"LGBM prune round {round_id}: evaluating {len(active)} features.")
        stats, oof, metrics = _lgbm_stability_selection_pass(
            X,
            y,
            sample_weight,
            active,
            classifier=classifier,
            groups=groups,
            returns=returns,
            random_state=random_state + round_id * 1009,
            seeds=[random_state],
        )
        rec = {
            "round": int(round_id),
            "n_features": int(len(active)),
            "active_features": list(active),
            "J_final": float(metrics.get("J_final", metrics.get("J_meta", -999.0))),
            "J_mean": float(metrics.get("J_mean", metrics.get("J_final", -999.0))),
            "J_std": float(metrics.get("J_std", 0.0)),
            "J_se": float(metrics.get("J_se", 0.0)),
            "J_median": float(metrics.get("J_median", metrics.get("J_final", -999.0))),
            "J_iqr": float(metrics.get("J_iqr", 0.0)),
            "J_robust": float(metrics.get("J_robust", metrics.get("J_final", -999.0))),
            "lift20": float(metrics.get("lift20", np.nan)),
            "precision20_norm": float(metrics.get("precision20_norm", np.nan)),
            "rank_bucket_monotonicity": float(metrics.get("rank_bucket_monotonicity", np.nan)),
            "ndcg_at_20": float(metrics.get("ndcg_at_20", np.nan)),
        }
        history.append(rec)
        last_stats = stats.copy()
        last_oof = oof.copy()
        last_metrics = dict(metrics)
        hard_kept = stats.loc[~stats["hard_drop"].astype(bool)].copy()
        if len(hard_kept) < max(LGBM_MIN_FEATURES, 1):
            hard_kept = stats.sort_values("feature_score", ascending=False).head(max(LGBM_MIN_FEATURES, min(len(stats), len(active)))).copy()
        drop_frac = _drop_fraction(len(active))
        keep_n = max(LGBM_MIN_FEATURES, int(np.ceil(len(active) * (1.0 - drop_frac))))
        keep_n = min(keep_n, len(hard_kept), len(active))
        next_active = hard_kept.sort_values("feature_score", ascending=False)["feature"].astype(str).head(keep_n).tolist()
        next_active = [f for f in active if f in set(next_active)]
        tprint(
            f"LGBM prune round {round_id}: J={rec['J_final']:.4f}, "
            f"SE={rec['J_se']:.4f}, {len(active)} -> {len(next_active)}."
        )
        if len(next_active) >= len(active) or len(next_active) <= LGBM_MIN_FEATURES:
            active = next_active
            break
        active = next_active
        gc.collect()
    chosen = _select_smallest_within_one_se(history)
    selected = list(chosen.get("active_features", active)) if chosen else active
    return selected, history, last_stats, last_oof, last_metrics


def _cross_val_oof_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    features: list[str],
    *,
    classifier: bool,
    params: dict[str, Any],
    groups: Any = None,
    returns: Any = None,
    random_state: int,
    n_splits: int = LGBM_CV_SPLITS,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    Xf = X[features].reset_index(drop=True)
    y_arr = np.asarray(y)
    ret_arr = _as_returns(y_arr, returns)
    splitter, y_split = _splitter(y_arr, classifier, random_state, n_splits=n_splits)
    oof = np.full(len(y_arr), np.nan, dtype=np.float32)
    metrics: list[dict[str, float]] = []
    for fold_i, (tr, va) in enumerate(splitter.split(np.zeros(len(y_split)), y_split), start=1):
        fold_params = dict(params)
        fold_params["random_state"] = int(random_state + fold_i * 1009)
        model = _fit_lgbm_model(
            Xf.iloc[tr].reset_index(drop=True),
            y_arr[tr],
            sample_weight[tr],
            classifier=classifier,
            params=fold_params,
            X_valid=Xf.iloc[va].reset_index(drop=True),
            y_valid=y_arr[va],
            early_stopping_rounds=50,
        )
        pred = _predict_lgbm_raw(model, Xf.iloc[va].reset_index(drop=True), "classifier" if classifier else "regressor")
        oof[va] = pred
        metrics.append(_metric_pack(y_arr[va], pred, classifier=classifier, groups=_groups_take(groups, va), returns=ret_arr[va]))
    fill = float(np.nanmean(oof)) if np.isfinite(oof).any() else float(np.mean(y_arr))
    return np.nan_to_num(oof, nan=fill).astype(np.float32), metrics


def _oof_distilled_sample_weights_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    base_weight: np.ndarray,
    features: list[str],
    *,
    classifier: bool,
    params: dict[str, Any],
    groups: Any = None,
    returns: Any = None,
    random_state: int,
    passes: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    base, _ = _normalize_weights(base_weight)
    current = base.copy()
    prev_oof: np.ndarray | None = None
    last_oof = np.full(len(y), float(np.mean(y)), dtype=np.float32)
    for pass_i in range(1, max(1, int(passes)) + 1):
        start = time.perf_counter()
        last_oof, _fold_metrics = _cross_val_oof_lgbm(
            X,
            y,
            current,
            features,
            classifier=classifier,
            params=params,
            groups=groups,
            returns=returns,
            random_state=random_state + pass_i * 7919,
        )
        distill = _compute_weight_distillation(y, last_oof, prev_oof, is_classifier=classifier, include_false_positive_focus=False)
        fp_weight = _false_positive_avoidance_weight(y, last_oof, classifier=classifier, top_frac=0.20)
        current, ess = _normalize_weights(base * distill * fp_weight)
        prev_oof = last_oof.copy()
        tprint(
            f"LGBM OOF distilled weights {label} pass {pass_i}/{max(1, int(passes))} "
            f"in {time.perf_counter() - start:.1f}s, ess={ess:.1f}."
        )
    return current.astype(np.float32), last_oof.astype(np.float32)


def _default_hpo_params(seed: int, classifier: bool) -> dict[str, Any]:
    return _base_lgbm_params(
        seed,
        classifier=classifier,
        overrides={
            "n_estimators": 1200,
            "learning_rate": LGBM_HPO_LEARNING_RATE,
            "max_depth": 4,
            "num_leaves": 16,
            "min_child_samples": 300,
            "min_child_weight": 40.0,
            "min_split_gain": 0.01,
            "reg_alpha": 1.0,
            "reg_lambda": 8.0,
            "subsample": 0.75,
            "colsample_bytree": 0.70,
        },
    )


def _run_lgbm_hpo(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    features: list[str],
    *,
    classifier: bool,
    groups: Any = None,
    returns: Any = None,
    random_state: int,
    max_trials: int = LGBM_HPO_TRIALS,
    patience: int = LGBM_HPO_EARLY_STOP_PATIENCE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.trial import TrialState
    except Exception as exc:
        tprint(f"LGBM HPO skipped, Optuna unavailable ({exc}).")
        params = _default_hpo_params(random_state, classifier)
        return params, {"hpo_available": False, "hpo_best_value": np.nan}
    y_arr = np.asarray(y)
    if len(y_arr) > LGBM_HPO_MAX_ROWS > 0:
        idx = _stratified_subsample_indices(y_arr, LGBM_HPO_MAX_ROWS, random_state + 71, classifier)
    else:
        idx = np.arange(len(y_arr), dtype=np.int32)
    X_sub = X.iloc[idx][features].reset_index(drop=True)
    y_sub = y_arr[idx]
    sw_sub = sample_weight[idx]
    ret_sub = _as_returns(y_arr, returns)[idx]
    groups_sub = _groups_take(groups, idx)
    splitter, y_split = _splitter(y_sub, classifier, random_state + 83, n_splits=3)
    best_seen = {"value": -np.inf, "trial": -1}

    def objective(trial: Any) -> float:
        depth = trial.suggest_int("max_depth", 3, 6)
        params = _base_lgbm_params(
            random_state + trial.number * 101,
            classifier=classifier,
            overrides={
                "n_estimators": 1600,
                "learning_rate": LGBM_HPO_LEARNING_RATE,
                "max_depth": depth,
                "num_leaves": int(2 ** depth),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 20.0, log=True),
                "min_child_samples": max(2, int(trial.suggest_float("min_child_samples_pct", 0.01, 0.05) * len(y_sub))),
                "min_child_weight": trial.suggest_float("min_child_weight", 20.0, 70.0),
                "min_split_gain": trial.suggest_categorical("min_split_gain", [0.0001, 0.01]),
                "subsample": trial.suggest_float("subsample", 0.60, 0.80),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.80),
            },
        )
        fold_metrics: list[dict[str, float]] = []
        for step, (tr, va) in enumerate(splitter.split(np.zeros(len(y_split)), y_split)):
            model = _fit_lgbm_model(
                X_sub.iloc[tr].reset_index(drop=True),
                y_sub[tr],
                sw_sub[tr],
                classifier=classifier,
                params=params,
                X_valid=X_sub.iloc[va].reset_index(drop=True),
                y_valid=y_sub[va],
                early_stopping_rounds=75,
            )
            pred = _predict_lgbm_raw(model, X_sub.iloc[va].reset_index(drop=True), "classifier" if classifier else "regressor")
            fold_metrics.append(_metric_pack(y_sub[va], pred, classifier=classifier, groups=_groups_take(groups_sub, va), returns=ret_sub[va]))
            agg_step = _aggregate_j(fold_metrics)
            value_step = float(agg_step.get("J_final", -999.0))
            trial.report(value_step, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        agg = _aggregate_j(fold_metrics)
        for key, value in agg.items():
            try:
                trial.set_user_attr(key, float(value))
            except Exception:
                pass
        return float(agg.get("J_final", -999.0))

    def early_stop_callback(study: Any, trial: Any) -> None:
        if trial.state != TrialState.COMPLETE or trial.value is None:
            return
        if float(trial.value) > float(best_seen["value"]):
            best_seen["value"] = float(trial.value)
            best_seen["trial"] = int(trial.number)
        elif int(trial.number) - int(best_seen["trial"]) >= int(patience):
            study.stop()

    study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1, interval_steps=1))
    study.optimize(objective, n_trials=max(0, int(max_trials)), callbacks=[early_stop_callback], n_jobs=1, show_progress_bar=False)
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    if not complete:
        params = _default_hpo_params(random_state, classifier)
        return params, {"hpo_available": True, "hpo_completed_trials": 0, "hpo_best_value": np.nan}
    best = study.best_trial
    depth = int(best.params.get("max_depth", 4))
    best_params = _base_lgbm_params(
        random_state + 191,
        classifier=classifier,
        overrides={
            "n_estimators": 1600,
            "learning_rate": LGBM_FINAL_LEARNING_RATE,
            "max_depth": depth,
            "num_leaves": int(2 ** depth),
            "reg_alpha": float(best.params.get("reg_alpha", 1.0)),
            "reg_lambda": float(best.params.get("reg_lambda", 8.0)),
            "min_child_samples": max(2, int(float(best.params.get("min_child_samples_pct", 0.03)) * max(1, len(y)))),
            "min_child_weight": float(best.params.get("min_child_weight", 40.0)),
            "min_split_gain": float(best.params.get("min_split_gain", 0.01)),
            "subsample": float(best.params.get("subsample", 0.75)),
            "colsample_bytree": float(best.params.get("colsample_bytree", 0.70)),
        },
    )
    attrs = dict(best.user_attrs)
    attrs.update({"hpo_available": True, "hpo_completed_trials": int(len(complete)), "hpo_best_trial": int(best.number), "hpo_best_value": float(best.value), "hpo_best_params": dict(best_params)})
    tprint(f"LGBM HPO complete: best_trial={best.number}, value={float(best.value):.4f}, params={json.dumps(best_params, sort_keys=True)}")
    return best_params, attrs


def train_lgbm_stability_candidate(
    X: Any,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 42,
    mode: str = "classifier",
    timestamps: Any = None,
    assets: Any = None,
    returns: Any = None,
) -> Optional[dict[str, Any]]:
    tprint("LGBM stability candidate training started.")
    t0 = time.perf_counter()
    classifier = mode == "classifier"
    X_df = _frame(X)
    y_arr = _coerce_target(y, classifier)
    ret_arr = _as_returns(y_arr, returns)
    n = len(y_arr)
    if n < 200 or X_df.shape[1] < 2:
        tprint("LGBM stability candidate skipped: not enough rows or features.")
        return None
    sw = np.ones(n, dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
    sw, _ = _normalize_weights(sw)
    stage_indices = _stage_partition_indices(y_arr, timestamps=timestamps, assets=assets, random_state=random_state + 701)
    stage_indices = _subsample_stage_indices(stage_indices, y_arr, max_fraction=LGBM_ROW_SUBSAMPLE_FRAC, random_state=random_state + 3701, classifier=classifier)
    stage_indices = _cap_stage_and_move_unused_to_fit_oof(stage_indices, y_arr, stage_key="lgbm_select", cap=LGBM_RACE_MAX_ROWS, random_state=random_state + 1701, classifier=classifier)
    stage_indices = _cap_stage_and_move_unused_to_fit_oof(stage_indices, y_arr, stage_key="hpo", cap=LGBM_HPO_MAX_ROWS, random_state=random_state + 2701, classifier=classifier)
    race_idx = np.asarray(stage_indices["lgbm_select"], dtype=np.int32)
    if len(race_idx) < 200:
        race_idx = _stratified_subsample_indices(y_arr, max_n=min(LGBM_RACE_MAX_ROWS, n), random_state=random_state + 701, classifier=classifier)
        stage_indices["lgbm_select"] = race_idx
    X_race = X_df.iloc[race_idx].reset_index(drop=True)
    y_race = y_arr[race_idx]
    sw_race = sw[race_idx]
    ret_race = ret_arr[race_idx]
    race_groups = _stability_group_bundle(
        len(race_idx),
        timestamps=(np.asarray(timestamps)[race_idx] if timestamps is not None and len(np.asarray(timestamps)) == n else None),
        assets=(np.asarray(assets)[race_idx] if assets is not None and len(np.asarray(assets)) == n else None),
    )
    local_idx = np.arange(len(y_race), dtype=np.int32)
    split_strata = y_race if classifier else np.clip((pd.Series(y_race).rank(pct=True).to_numpy() * 5).astype(np.int32), 0, 4)
    select_local, eval_local = train_test_split(local_idx, test_size=LGBM_RACE_EVAL_FRACTION, stratify=split_strata if classifier and len(np.unique(split_strata)) > 1 else None, random_state=random_state + 1701)
    select_local = np.asarray(select_local, dtype=np.int32)
    eval_local = np.asarray(eval_local, dtype=np.int32)
    X_select = X_race.iloc[select_local].reset_index(drop=True)
    y_select = y_race[select_local]
    sw_select = sw_race[select_local]
    ret_select = ret_race[select_local]
    select_groups = _groups_take(race_groups, select_local)
    X_eval = X_race.iloc[eval_local].reset_index(drop=True)
    y_eval = y_race[eval_local]
    ret_eval = ret_race[eval_local]
    eval_groups = _groups_take(race_groups, eval_local)
    tprint(f"LGBM candidate split: select={len(y_select)}, eval={len(y_eval)}, features={X_select.shape[1]}.")
    uni_features, uni_stats = _univariate_directional_filter(X_select, y_select, classifier=classifier, groups=select_groups, returns=ret_select, random_state=random_state + 101)
    score_map = dict(zip(uni_stats["feature"].astype(str), uni_stats["univariate_j"].astype(float)))
    cluster_features = _redundancy_cluster_filter(X_select, uni_features, score_map, random_state=random_state + 211)
    selected_features, history, feature_stats, prune_oof, prune_metrics = _iterative_feature_prune(
        X_select,
        y_select,
        sw_select,
        cluster_features,
        classifier=classifier,
        groups=select_groups,
        returns=ret_select,
        random_state=random_state + 307,
    )
    if not selected_features:
        tprint("LGBM candidate rejected: no selected features.")
        return None
    base_params = _default_hpo_params(random_state + 401, classifier)
    final_weights, _ = _oof_distilled_sample_weights_lgbm(
        X_select,
        y_select,
        sw_select,
        selected_features,
        classifier=classifier,
        params=base_params,
        groups=select_groups,
        returns=ret_select,
        random_state=random_state + 409,
        passes=max(1, LGBM_OOF_DISTILLATION_PASSES),
        label="candidate",
    )
    eval_preds: list[np.ndarray] = []
    for i, cfg in enumerate(({"max_depth": 4, "reg_lambda": 5.0}, {"max_depth": 4, "reg_lambda": 15.0}, {"max_depth": 5, "reg_lambda": 5.0}, {"max_depth": 5, "reg_lambda": 15.0}), start=1):
        params = _base_lgbm_params(random_state + 500 + i, classifier=classifier, overrides=cfg)
        model = _fit_lgbm_model(X_select[selected_features], y_select, final_weights, classifier=classifier, params=params)
        eval_preds.append(_predict_lgbm_raw(model, X_eval[selected_features], mode))
    eval_pred = np.mean(np.vstack(eval_preds), axis=0).astype(np.float32)
    metrics = _metric_pack(y_eval, eval_pred, classifier=classifier, groups=eval_groups, returns=ret_eval)
    metrics.update(_aggregate_j([metrics]))
    metrics["feature_count"] = int(len(selected_features))
    metrics["n_univariate_features"] = int(len(uni_features))
    metrics["n_cluster_features"] = int(len(cluster_features))
    metrics["feature_pruning_rounds_completed"] = int(len(history))
    metrics["candidate_elapsed_sec"] = float(time.perf_counter() - t0)
    oof_full = np.full(n, np.nan, dtype=np.float32)
    oof_race = np.full(len(y_race), np.nan, dtype=np.float32)
    oof_race[eval_local] = eval_pred
    oof_full[race_idx] = oof_race
    fill = float(np.mean(y_arr))
    oof_for_fit = np.nan_to_num(oof_full, nan=fill).astype(np.float32)
    tprint(f"LGBM candidate done: J={metrics.get('J_final', 0.0):.4f}, features={len(selected_features)}, elapsed={metrics['candidate_elapsed_sec']:.1f}s.")
    return {
        "model": None,
        "metrics": metrics,
        "oof_probs": oof_full,
        "oof_for_full_fit": oof_for_fit,
        "selected_feature_names": list(selected_features),
        "selected_features_from_cv": np.asarray([X_df.columns.get_loc(c) for c in selected_features if c in X_df.columns], dtype=np.int32),
        "pruning_history": history,
        "univariate_stats": uni_stats,
        "feature_stats": feature_stats,
        "stage_indices": {k: np.asarray(v, dtype=np.int32) for k, v in stage_indices.items()},
        "full_fit_needed": True,
        "mode": mode,
    }


def fit_lgbm_stability_full_model(
    X: Any,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    selected_features_from_cv: np.ndarray | None = None,
    random_state: int = 42,
    mode: str = "classifier",
    oof_probs: Optional[np.ndarray] = None,
    metrics: Optional[dict[str, Any]] = None,
    pruning_history: Optional[list[dict[str, Any]]] = None,
    selected_feature_names: Optional[list[str]] = None,
    stage_indices: Optional[dict[str, np.ndarray]] = None,
    timestamps: Any = None,
    assets: Any = None,
    returns: Any = None,
    hpo_trials_override: int | None = None,
    hpo_patience_override: int | None = None,
) -> Optional[LGBMStabilityModel]:
    classifier = mode == "classifier"
    X_df = _frame(X)
    y_arr = _coerce_target(y, classifier)
    ret_arr = _as_returns(y_arr, returns)
    n = len(y_arr)
    if selected_feature_names:
        selected_features = [str(c) for c in selected_feature_names]
    else:
        idx = np.asarray(selected_features_from_cv if selected_features_from_cv is not None else [], dtype=np.int32)
        idx = idx[(idx >= 0) & (idx < X_df.shape[1])]
        selected_features = [str(X_df.columns[i]) for i in idx]
    if not selected_features:
        return None
    for col in selected_features:
        if col not in X_df.columns:
            X_df[col] = 0.0
    sw = np.ones(n, dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
    sw, _ = _normalize_weights(sw)
    if stage_indices is None:
        all_idx = np.arange(n, dtype=np.int32)
        stage_indices = {"lgbm_select": all_idx, "hpo": all_idx, "fit_oof": all_idx}
    hpo_idx = np.asarray(stage_indices.get("hpo", np.arange(n)), dtype=np.int32)
    fit_idx = np.asarray(stage_indices.get("fit_oof", np.arange(n)), dtype=np.int32)
    hpo_idx = hpo_idx[(hpo_idx >= 0) & (hpo_idx < n)]
    fit_idx = fit_idx[(fit_idx >= 0) & (fit_idx < n)]
    if len(hpo_idx) == 0:
        hpo_idx = np.arange(n, dtype=np.int32)
    if len(fit_idx) == 0:
        fit_idx = np.arange(n, dtype=np.int32)
    if LGBM_FINAL_FIT_MAX_ROWS > 0 and len(fit_idx) > LGBM_FINAL_FIT_MAX_ROWS:
        local = _stratified_subsample_indices(y_arr[fit_idx], LGBM_FINAL_FIT_MAX_ROWS, random_state + 40711, classifier)
        fit_idx = np.sort(fit_idx[local].astype(np.int32))
    stability_groups = _stability_group_bundle(n, timestamps=timestamps, assets=assets)
    hpo_groups = _groups_take(stability_groups, hpo_idx)
    best_params, hpo_metrics = _run_lgbm_hpo(
        X_df.iloc[hpo_idx].reset_index(drop=True),
        y_arr[hpo_idx],
        sw[hpo_idx],
        selected_features,
        classifier=classifier,
        groups=hpo_groups,
        returns=ret_arr[hpo_idx],
        random_state=random_state + 131,
        max_trials=LGBM_HPO_TRIALS if hpo_trials_override is None else int(hpo_trials_override),
        patience=LGBM_HPO_EARLY_STOP_PATIENCE if hpo_patience_override is None else int(hpo_patience_override),
    )
    if LGBM_OOF_DISTILLATION_PASSES > 0:
        final_weights, pre_final_oof = _oof_distilled_sample_weights_lgbm(
            X_df,
            y_arr,
            sw,
            selected_features,
            classifier=classifier,
            params=best_params,
            groups=stability_groups,
            returns=ret_arr,
            random_state=random_state + 33107,
            passes=LGBM_OOF_DISTILLATION_PASSES,
            label="final",
        )
    else:
        final_weights = sw.copy()
        pre_final_oof = np.asarray(oof_probs if oof_probs is not None else np.full(n, float(np.mean(y_arr))), dtype=np.float32)
    model = LGBMStabilityModel(mode=mode)
    model.selected_features = list(selected_features)
    model.best_params = dict(best_params)
    X_fit = X_df.iloc[fit_idx][selected_features].reset_index(drop=True)
    y_fit = y_arr[fit_idx]
    w_fit = final_weights[fit_idx]
    for i in range(LGBM_FINAL_MODEL_COUNT):
        params_i = dict(best_params)
        params_i["random_state"] = int(random_state + 7001 + i * 101)
        fitted = _fit_lgbm_model(X_fit, y_fit, w_fit, classifier=classifier, params=params_i)
        model.models.append(fitted)
        tprint(f"LGBM final model {i + 1}/{LGBM_FINAL_MODEL_COUNT} fitted on {len(y_fit)} rows.")
    final_oof, final_fold_metrics = _cross_val_oof_lgbm(
        X_df,
        y_arr,
        final_weights,
        selected_features,
        classifier=classifier,
        params=best_params,
        groups=stability_groups,
        returns=ret_arr,
        random_state=random_state + 11701,
    )
    model.oof_probs = final_oof.astype(np.float32)
    final_metrics = _metric_pack(y_arr, final_oof, classifier=classifier, groups=stability_groups, returns=ret_arr)
    final_metrics.update(_aggregate_j(final_fold_metrics))
    model.metrics = dict(metrics or {})
    model.metrics.update(hpo_metrics)
    model.metrics.update(final_metrics)
    model.metrics["feature_count"] = int(len(selected_features))
    model.metrics["final_fit_train_rows"] = int(len(fit_idx))
    model.metrics["final_fit_train_rows_total"] = int(n)
    model.metrics["final_model_count"] = int(LGBM_FINAL_MODEL_COUNT)
    model.metrics["best_params"] = dict(best_params)
    if pre_final_oof is not None and len(pre_final_oof) == n:
        pre_metrics = _metric_pack(y_arr, pre_final_oof, classifier=classifier, groups=stability_groups, returns=ret_arr)
        for key, value in pre_metrics.items():
            model.metrics[f"pre_final_distill_{key}"] = value
            if key in final_metrics:
                model.metrics[f"distill_delta_{key}"] = float(final_metrics[key]) - float(value)
    model.pruning_history = list(pruning_history or [])
    tprint(f"LGBM full fit done: J={model.metrics.get('J_final', 0.0):.4f}, features={len(selected_features)}.")
    return model


def train_lgbm_stability_pipeline(
    X: Any,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 42,
    mode: str = "classifier",
    timestamps: Any = None,
    assets: Any = None,
    returns: Any = None,
    hpo_trials_override: int | None = None,
    hpo_patience_override: int | None = None,
) -> Optional[LGBMStabilityModel]:
    candidate = train_lgbm_stability_candidate(
        X,
        y,
        sample_weight=sample_weight,
        random_state=random_state,
        mode=mode,
        timestamps=timestamps,
        assets=assets,
        returns=returns,
    )
    if candidate is None:
        return None
    return fit_lgbm_stability_full_model(
        X,
        y,
        sample_weight,
        selected_features_from_cv=candidate.get("selected_features_from_cv"),
        random_state=random_state,
        mode=mode,
        oof_probs=candidate.get("oof_probs"),
        metrics=candidate.get("metrics"),
        pruning_history=candidate.get("pruning_history"),
        selected_feature_names=candidate.get("selected_feature_names"),
        stage_indices=candidate.get("stage_indices"),
        timestamps=timestamps,
        assets=assets,
        returns=returns,
        hpo_trials_override=hpo_trials_override,
        hpo_patience_override=hpo_patience_override,
    )


__all__ = [
    "LGBMStabilityModel",
    "FeatureSelectionResult",
    "train_lgbm_stability_candidate",
    "fit_lgbm_stability_full_model",
    "train_lgbm_stability_pipeline",
]
