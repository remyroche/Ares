#!/usr/bin/env python3
"""Walk-forward AE/GMM cluster policy smoke over feature-store label candidates."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMRanker

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - optional comparator
    LGBMRanker = None
    _LIGHTGBM_AVAILABLE = False

try:
    from numba import njit as _numba_njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional acceleration
    _NUMBA_AVAILABLE = False

    def _numba_njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    fit_ae_gmm_state,
    transform_ae_gmm_features,
)
from scripts.run_label_economic_proxy_ablation import _label_targets  # noqa: E402
from scripts.run_label_feature_store_model_smoke import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_LABELS_DIR,
    _add_delta_fields,
    _apply_evaluation_utility_column,
    _baseline_row,
    _fixed_artifact_targets,
    _fit_predict,
    _month_model_frame,
    _parse_csv,
    _parse_float_csv,
    _parse_int_csv,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    _decile_diagnostics,
    _feature_columns,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics as _base_selection_metrics,
    _spearman,
)
from scripts.run_label_weighted_proxy_ablation import _weight_series  # noqa: E402


DEFAULT_SELECTION_DIR = Path(
    "data_perp/reports/conditional_gmm_feature_selection_20260702_lowcost_strict_econ_target_wide_sidebalanced_hpo"
)
DEFAULT_FEATURE_LIST_CSV = DEFAULT_SELECTION_DIR / "conditional_gmm_training_feature_list.csv"
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced")

_POLICY_BLOCK = np.int8(0)
_POLICY_THROTTLE = np.int8(1)
_POLICY_HIGH_THRESHOLD = np.int8(2)
_POLICY_NORMAL = np.int8(3)

DEFAULT_VIABILITY_THRESHOLDS: dict[str, float] = {
    "min_mean_u": 0.0,
    "min_positive_month_rate": 2.0 / 3.0,
    "min_worst_month_mean_u": -0.0010,
    "max_bad_mae_1r_rate": 0.50,
    "max_timeout_rate": 0.12,
    "min_weekly_mean_u_q10": -0.0030,
    "max_month_bad_mae_1r_rate": 0.60,
    "max_month_timeout_rate": 0.18,
    "min_worst_month_weekly_mean_u_q10": -0.0040,
    "min_month_final_oracle_recall": 0.005,
    "max_side_share": 0.70,
    "min_selected_rows": 25.0,
    "min_utility_monotonic_rate": 0.60,
    "min_bad_mae_improves_rate": 0.50,
    "min_calibration_groups": 2.0,
    "min_score_ic_u": 0.0,
    "min_stage_a_oracle_recall": 0.50,
    "min_final_oracle_recall": 0.02,
}

CALIBRATION_SELECTOR_BY_POLICY = {
    "raw_model_score": "S0_raw_score",
    "side_calibrated_score": "S2_side_calibrated_score",
    "side_calibrated_soft_risk_score": "S3_side_calibrated_soft_risk_score",
    "side_calibrated_soft_risk_bad_mae_cap_score": "S4_risk_cap_score",
    "side_calibrated_soft_risk_bad_mae_cap_threshold_select_score": "S5_risk_cap_threshold_score",
    "lgbm_ranker_side_calibrated_risk_cap_score": "S6_lgbm_ranker_side_calibrated_risk_cap_score",
    "s7a_lgbm_ranker_no_prefilter_score": "S7a_lgbm_ranker_no_prefilter_score",
    "s7b_lgbm_ranker_relaxed_risk_cap_score": "S7b_lgbm_ranker_relaxed_risk_cap_score",
    "s7c_side_specific_ranker_risk_cap_score": "S7c_side_specific_ranker_risk_cap_score",
    "s7d_oracle_enriched_ranker_risk_cap_score": "S7d_oracle_enriched_ranker_risk_cap_score",
    "s7_two_stage_candidate_rerank_score": "S7_two_stage_candidate_rerank_score",
    "s8d_oracle_enriched_ranker_tight_bad_mae_score": "S8d_oracle_enriched_ranker_tight_bad_mae_score",
    "s8_two_stage_tight_bad_mae_score": "S8_two_stage_tight_bad_mae_score",
    "s9d_oracle_enriched_ranker_calibrated_risk_cap_score": (
        "S9d_oracle_enriched_ranker_calibrated_risk_cap_score"
    ),
    "s9_two_stage_calibrated_risk_cap_score": "S9_two_stage_calibrated_risk_cap_score",
    "s10_recall_soft_calibrated_risk_score": "S10_recall_soft_calibrated_risk_score",
    "s10_recall_soft_calibrated_risk_loose_cap_score": (
        "S10_recall_soft_calibrated_risk_loose_cap_score"
    ),
    "s11_recall_tail_balanced_score": "S11_recall_tail_balanced_score",
    "s12_path_quality_ranker_score": "S12_path_quality_ranker_score",
    "s12_path_quality_ranker_soft_risk_score": "S12_path_quality_ranker_soft_risk_score",
    "s13_constrained_path_quality_score": "S12_path_quality_ranker_score",
    "s13_constrained_path_quality_soft_risk_score": "S12_path_quality_ranker_soft_risk_score",
    "s14_path_quality_risk_trim_score": "S12_path_quality_ranker_score",
    "s14_path_quality_soft_risk_trim_score": "S12_path_quality_ranker_soft_risk_score",
    "s15_side_path_quality_ranker_score": "S15_side_path_quality_ranker_score",
    "s15_side_path_quality_risk_trim_score": "S15_side_path_quality_ranker_score",
    "s16_discovery_path_quality_blend_score": "S16_discovery_path_quality_blend_score",
    "s16_discovery_path_quality_risk_trim_score": "S16_discovery_path_quality_blend_score",
}

FINAL_STAGE_BY_POLICY = {
    "lgbm_ranker_side_calibrated_risk_cap_score": "final_S6",
    "s7a_lgbm_ranker_no_prefilter_score": "final_S7a",
    "s7b_lgbm_ranker_relaxed_risk_cap_score": "final_S7b",
    "s7c_side_specific_ranker_risk_cap_score": "final_S7c",
    "s7d_oracle_enriched_ranker_risk_cap_score": "final_S7d",
    "s7_two_stage_candidate_rerank_score": "final_S7_two_stage",
    "s8d_oracle_enriched_ranker_tight_bad_mae_score": "final_S8d",
    "s8_two_stage_tight_bad_mae_score": "final_S8_two_stage",
    "s9d_oracle_enriched_ranker_calibrated_risk_cap_score": "final_S9d",
    "s9_two_stage_calibrated_risk_cap_score": "final_S9_two_stage",
    "s10_recall_soft_calibrated_risk_score": "final_S10_soft",
    "s10_recall_soft_calibrated_risk_loose_cap_score": "final_S10_loose_cap",
    "s11_recall_tail_balanced_score": "final_S11_tail_balanced",
    "s12_path_quality_ranker_score": "final_S12_path_quality",
    "s12_path_quality_ranker_soft_risk_score": "final_S12_path_quality_soft_risk",
    "s13_constrained_path_quality_score": "final_S13_constrained_path_quality",
    "s13_constrained_path_quality_soft_risk_score": "final_S13_constrained_path_quality_soft_risk",
    "s14_path_quality_risk_trim_score": "final_S14_path_quality_risk_trim",
    "s14_path_quality_soft_risk_trim_score": "final_S14_path_quality_soft_risk_trim",
    "s15_side_path_quality_ranker_score": "final_S15_side_path_quality",
    "s15_side_path_quality_risk_trim_score": "final_S15_side_path_quality_risk_trim",
    "s16_discovery_path_quality_blend_score": "final_S16_discovery_path_quality_blend",
    "s16_discovery_path_quality_risk_trim_score": "final_S16_discovery_path_quality_risk_trim",
}

PIPELINE_GATE_PLAN = [
    {
        "step": 1,
        "name": "gmm_feature_archetype_quality",
        "goal": (
            "Stable clusters with interpretable feature-target contrast, both sides represented, "
            "and live-predictable AE/GMM state features."
        ),
        "advance_when": (
            "GMM HPO state is enabled OOS, cluster-side coverage is adequate, and candidate "
            "discovery recall passes before final ranking."
        ),
    },
    {
        "step": 2,
        "name": "label_learnability",
        "goal": "Higher model score maps to higher net utility and lower bad-MAE by side/month quantile.",
        "advance_when": (
            "Utility quantile monotonicity and bad-MAE improvement pass on OOS calibration groups."
        ),
    },
    {
        "step": 3,
        "name": "economic_viability",
        "goal": "Active labels are positive net, path-risk controlled, side-aware, cost-robust, and stable.",
        "advance_when": (
            "The viability matrix marks at least one selector active across top buckets without "
            "tail-risk or recall exceptions."
        ),
    },
    {
        "step": 4,
        "name": "train_base_to_train_meta_readiness",
        "goal": (
            "Promote viable archetype labels/features into train_base -> train_meta and validate "
            "profitability under simple_policy_optimiser-style exits."
        ),
        "advance_when": (
            "A viable label improves OOS/policy metrics under frozen thresholds and does not rely "
            "on post-selection or same-period tuning."
        ),
    },
]


def _strict_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _strict_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_strict_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _cluster_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if str(c).startswith("gmm_prob_")]


def _cluster_labels(gmm_features: pd.DataFrame) -> np.ndarray:
    prob_cols = _cluster_columns(gmm_features)
    if not prob_cols:
        return np.full(len(gmm_features), -1, dtype=np.int32)
    return gmm_features[prob_cols].to_numpy(dtype=np.float32, copy=False).argmax(axis=1).astype(np.int32)


def _side_counts(side: pd.Series) -> tuple[int, int]:
    long_count = int((side > 0).sum())
    short_count = int((side < 0).sum())
    return long_count, short_count


@_numba_njit(cache=True)
def _cluster_core_stats_numba(
    clusters: np.ndarray,
    u: np.ndarray,
    side: np.ndarray,
    mae_norm: np.ndarray,
    timeout: np.ndarray,
    max_cluster: int,
) -> tuple[np.ndarray, ...]:
    n_clusters = max_cluster + 1
    rows = np.zeros(n_clusters, dtype=np.int64)
    finite_u = np.zeros(n_clusters, dtype=np.int64)
    sum_u = np.zeros(n_clusters, dtype=np.float64)
    hit_u = np.zeros(n_clusters, dtype=np.int64)
    long_rows = np.zeros(n_clusters, dtype=np.int64)
    short_rows = np.zeros(n_clusters, dtype=np.int64)
    long_sum_u = np.zeros(n_clusters, dtype=np.float64)
    short_sum_u = np.zeros(n_clusters, dtype=np.float64)
    long_finite_u = np.zeros(n_clusters, dtype=np.int64)
    short_finite_u = np.zeros(n_clusters, dtype=np.int64)
    bad_mae = np.zeros(n_clusters, dtype=np.int64)
    finite_mae = np.zeros(n_clusters, dtype=np.int64)
    timeout_count = np.zeros(n_clusters, dtype=np.int64)
    finite_timeout = np.zeros(n_clusters, dtype=np.int64)
    for i in range(clusters.shape[0]):
        c = int(clusters[i])
        if c < 0 or c >= n_clusters:
            continue
        rows[c] += 1
        ui = float(u[i])
        if np.isfinite(ui):
            finite_u[c] += 1
            sum_u[c] += ui
            if ui > 0.0:
                hit_u[c] += 1
        if side[i] < 0.0:
            short_rows[c] += 1
            if np.isfinite(ui):
                short_sum_u[c] += ui
                short_finite_u[c] += 1
        else:
            long_rows[c] += 1
            if np.isfinite(ui):
                long_sum_u[c] += ui
                long_finite_u[c] += 1
        mi = float(mae_norm[i])
        if np.isfinite(mi):
            finite_mae[c] += 1
            if mi >= 1.0:
                bad_mae[c] += 1
        ti = float(timeout[i])
        if np.isfinite(ti):
            finite_timeout[c] += 1
            if ti > 0.5:
                timeout_count[c] += 1
    return (
        rows,
        finite_u,
        sum_u,
        hit_u,
        long_rows,
        short_rows,
        long_sum_u,
        short_sum_u,
        long_finite_u,
        short_finite_u,
        bad_mae,
        finite_mae,
        timeout_count,
        finite_timeout,
    )


@_numba_njit(cache=True)
def _apply_cluster_policy_score_numba(
    raw: np.ndarray,
    clusters: np.ndarray,
    action_codes: np.ndarray,
    adjustments: np.ndarray,
    high_threshold: float,
    throttle_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    adjusted = np.empty(raw.shape[0], dtype=np.float32)
    eligible = np.zeros(raw.shape[0], dtype=np.bool_)
    for i in range(raw.shape[0]):
        adjusted[i] = np.nan
        score = float(raw[i])
        if not np.isfinite(score):
            continue
        c = int(clusters[i])
        if c < 0 or c >= action_codes.shape[0]:
            continue
        action = action_codes[c]
        threshold = np.inf
        if action == _POLICY_NORMAL:
            threshold = -np.inf
        elif action == _POLICY_HIGH_THRESHOLD:
            threshold = high_threshold
        elif action == _POLICY_THROTTLE:
            threshold = throttle_threshold
        if score >= threshold and action != _POLICY_BLOCK:
            adjusted[i] = np.float32(score + 0.01 * float(adjustments[c]))
            eligible[i] = True
    return adjusted, eligible


@_numba_njit(cache=True)
def _apply_cluster_side_policy_score_numba(
    raw: np.ndarray,
    clusters: np.ndarray,
    side: np.ndarray,
    action_codes: np.ndarray,
    adjustments: np.ndarray,
    high_threshold: float,
    throttle_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    adjusted = np.empty(raw.shape[0], dtype=np.float32)
    eligible = np.zeros(raw.shape[0], dtype=np.bool_)
    for i in range(raw.shape[0]):
        adjusted[i] = np.nan
        score = float(raw[i])
        if not np.isfinite(score):
            continue
        c = int(clusters[i])
        side_idx = 1 if side[i] < 0.0 else 0
        if c < 0 or c >= action_codes.shape[0]:
            continue
        action = action_codes[c, side_idx]
        threshold = np.inf
        if action == _POLICY_NORMAL:
            threshold = -np.inf
        elif action == _POLICY_HIGH_THRESHOLD:
            threshold = high_threshold
        elif action == _POLICY_THROTTLE:
            threshold = throttle_threshold
        if score >= threshold and action != _POLICY_BLOCK:
            adjusted[i] = np.float32(score + 0.01 * float(adjustments[c, side_idx]))
            eligible[i] = True
    return adjusted, eligible


def _rank_top_indices_fast(score: Any, frac: float) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(score).reset_index(drop=True), errors="coerce").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    valid = np.isfinite(arr)
    if not bool(valid.any()):
        return np.array([], dtype=np.int64)
    valid_idx = np.flatnonzero(valid)
    k = max(1, int(math.ceil(float(frac) * len(valid_idx))))
    if k >= len(valid_idx):
        order = np.argsort(-arr[valid_idx], kind="mergesort")
        return valid_idx[order].astype(np.int64, copy=False)
    part = np.argpartition(-arr[valid_idx], kth=k - 1)[:k]
    chosen = valid_idx[part]
    order = np.argsort(-arr[chosen], kind="mergesort")
    return chosen[order].astype(np.int64, copy=False)


def _selection_indices(score: Any, frac: float, selection_mode: str = "top_frac") -> np.ndarray:
    arr = pd.to_numeric(pd.Series(score).reset_index(drop=True), errors="coerce").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    valid_idx = np.flatnonzero(np.isfinite(arr))
    if str(selection_mode) == "all_finite":
        if len(valid_idx) == 0:
            return np.array([], dtype=np.int64)
        order = np.argsort(-arr[valid_idx], kind="mergesort")
        return valid_idx[order].astype(np.int64, copy=False)
    return _rank_top_indices_fast(pd.Series(arr), frac)


def _score_percentiles_for_indices(score: Any, idx: np.ndarray) -> dict[str, float]:
    arr = pd.to_numeric(pd.Series(score).reset_index(drop=True), errors="coerce")
    if len(idx) == 0:
        return {
            "oracle_top_score_percentile_mean": float("nan"),
            "oracle_top_score_percentile_q10": float("nan"),
            "oracle_top_score_percentile_min": float("nan"),
            "oracle_top_score_finite_frac": float("nan"),
        }
    finite = arr.notna() & np.isfinite(arr.to_numpy(dtype=np.float64, copy=False))
    if int(finite.sum()) == 0:
        return {
            "oracle_top_score_percentile_mean": float("nan"),
            "oracle_top_score_percentile_q10": float("nan"),
            "oracle_top_score_percentile_min": float("nan"),
            "oracle_top_score_finite_frac": 0.0,
        }
    ranks = arr[finite].rank(method="average", pct=True)
    percentiles = pd.Series(np.nan, index=arr.index, dtype=np.float32)
    percentiles.loc[ranks.index] = ranks.astype(np.float32)
    selected = percentiles.iloc[np.asarray(idx, dtype=np.int64)]
    finite_selected = selected.dropna()
    return {
        "oracle_top_score_percentile_mean": _safe_mean(finite_selected),
        "oracle_top_score_percentile_q10": _safe_quantile(finite_selected, 0.10),
        "oracle_top_score_percentile_min": float(finite_selected.min()) if len(finite_selected) else float("nan"),
        "oracle_top_score_finite_frac": float(len(finite_selected) / max(len(idx), 1)),
    }


def _oracle_recall_summary(
    *,
    score: Any,
    oracle_score: Any,
    top_frac: float,
    selection_mode: str = "top_frac",
) -> dict[str, float]:
    oracle_idx = _selection_indices(oracle_score, top_frac, "top_frac")
    selected_idx = _selection_indices(score, top_frac, selection_mode)
    if len(oracle_idx) == 0:
        overlap_rows = 0
        recall = float("nan")
    else:
        overlap_rows = int(np.intersect1d(oracle_idx, selected_idx, assume_unique=False).size)
        recall = float(overlap_rows / len(oracle_idx))
    precision = float(overlap_rows / len(selected_idx)) if len(selected_idx) else float("nan")
    out = {
        "oracle_top_rows": int(len(oracle_idx)),
        "model_selected_rows_for_recall": int(len(selected_idx)),
        "oracle_overlap_rows": int(overlap_rows),
        "oracle_recall_at_model_top_k": recall,
        "oracle_precision_at_model_top_k": precision,
    }
    out.update(_score_percentiles_for_indices(score, oracle_idx))
    return out


def _top_n_from_mask(arr: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return np.array([], dtype=np.int64)
    n = min(int(n), len(idx))
    if n >= len(idx):
        order = np.argsort(-arr[idx], kind="mergesort")
        return idx[order].astype(np.int64, copy=False)
    part = np.argpartition(-arr[idx], kth=n - 1)[:n]
    chosen = idx[part]
    order = np.argsort(-arr[chosen], kind="mergesort")
    return chosen[order].astype(np.int64, copy=False)


def _side_balanced_top_score(score: pd.Series, side: pd.Series, frac: float) -> pd.Series:
    raw = pd.to_numeric(score.reset_index(drop=True), errors="coerce").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    side_arr = pd.to_numeric(side.reset_index(drop=True), errors="coerce").fillna(1.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    valid = np.isfinite(raw)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 0:
        return pd.Series(np.nan, index=pd.RangeIndex(len(raw)), dtype=np.float32)
    k = max(1, int(math.ceil(float(frac) * len(valid_idx))))
    long_mask = valid & (side_arr > 0.0)
    short_mask = valid & (side_arr < 0.0)
    long_count = int(long_mask.sum())
    short_count = int(short_mask.sum())
    if k < 2 or long_count == 0 or short_count == 0:
        return pd.Series(raw, index=pd.RangeIndex(len(raw)), dtype=np.float32)

    long_quota = min(long_count, k // 2)
    short_quota = min(short_count, k - long_quota)
    if short_quota < k - long_quota:
        long_quota = min(long_count, k - short_quota)
    if long_quota == 0 and long_count > 0 and k > 1:
        long_quota = 1
        short_quota = min(short_count, k - 1)
    if short_quota == 0 and short_count > 0 and k > 1:
        short_quota = 1
        long_quota = min(long_count, k - 1)

    selected = np.concatenate(
        [
            _top_n_from_mask(raw, long_mask, long_quota),
            _top_n_from_mask(raw, short_mask, short_quota),
        ]
    )
    if len(selected) < k:
        selected_mask = np.zeros(raw.shape[0], dtype=bool)
        selected_mask[selected] = True
        selected = np.concatenate(
            [
                selected,
                _top_n_from_mask(raw, valid & ~selected_mask, k - len(selected)),
            ]
        )
    if len(selected) == 0:
        return pd.Series(raw, index=pd.RangeIndex(len(raw)), dtype=np.float32)

    finite_raw = raw[valid]
    low = float(np.nanmin(finite_raw))
    high = float(np.nanmax(finite_raw))
    spread = max(float(high - low), 1.0)
    balanced = np.full(raw.shape[0], np.float32(low - spread - 1.0), dtype=np.float32)
    balanced[~valid] = np.nan
    order = selected[np.argsort(-raw[selected], kind="mergesort")]
    balanced[order] = (
        np.float32(high + spread + 1.0)
        - np.arange(len(order), dtype=np.float32) * np.float32(1e-6)
    )
    return pd.Series(balanced, index=pd.RangeIndex(len(raw)), dtype=np.float32)


def _side_capped_top_score(
    score: pd.Series,
    side: pd.Series,
    frac: float,
    *,
    max_side_share: float,
) -> pd.Series:
    raw = pd.to_numeric(score.reset_index(drop=True), errors="coerce").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    side_arr = pd.to_numeric(side.reset_index(drop=True), errors="coerce").fillna(1.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    valid = np.isfinite(raw)
    valid_idx = np.flatnonzero(valid)
    out = np.full(raw.shape[0], np.nan, dtype=np.float32)
    if len(valid_idx) == 0:
        return pd.Series(out, index=pd.RangeIndex(len(raw)), dtype=np.float32)
    k = max(1, int(math.ceil(float(frac) * len(valid_idx))))
    max_share = float(np.clip(max_side_share, 0.50, 1.00))
    max_per_side = max(1, int(math.floor(max_share * k)))
    long_selected = 0
    short_selected = 0
    selected: list[int] = []
    for idx in valid_idx[np.argsort(-raw[valid_idx], kind="mergesort")]:
        is_short = bool(side_arr[idx] < 0.0)
        if is_short:
            if short_selected >= max_per_side:
                continue
            short_selected += 1
        else:
            if long_selected >= max_per_side:
                continue
            long_selected += 1
        selected.append(int(idx))
        if len(selected) >= k:
            break
    if selected:
        chosen = np.asarray(selected, dtype=np.int64)
        out[chosen] = raw[chosen]
    return pd.Series(out, index=pd.RangeIndex(len(raw)), dtype=np.float32)


def _enforce_actual_selected_side_share(
    selected: list[int],
    raw: np.ndarray,
    side_arr: np.ndarray,
    *,
    max_share: float,
) -> list[int]:
    selected = [int(i) for i in selected]
    max_share = float(np.clip(max_share, 0.50, 1.00))
    while len(selected) > 1:
        chosen = np.asarray(selected, dtype=np.int64)
        long_idx = chosen[side_arr[chosen] > 0.0]
        short_idx = chosen[side_arr[chosen] < 0.0]
        if len(long_idx) == 0 or len(short_idx) == 0:
            break
        long_share = len(long_idx) / len(chosen)
        short_share = len(short_idx) / len(chosen)
        if long_share <= max_share and short_share <= max_share:
            break
        drop_pool = long_idx if long_share > max_share else short_idx
        drop_idx = int(drop_pool[np.argmin(raw[drop_pool])])
        selected.remove(drop_idx)
    return selected


def _risk_constrained_backfill_top_score(
    score: pd.Series,
    side: pd.Series,
    risk_predictions: pd.DataFrame,
    frac: float,
    *,
    max_side_share: float,
    primary_max_pred_bad_mae: float,
    primary_max_pred_timeout: float,
    primary_max_pred_lower_tail: float,
    backfill_max_pred_bad_mae: float,
    backfill_max_pred_timeout: float,
    backfill_max_pred_lower_tail: float,
    max_backfill_share: float,
) -> pd.Series:
    raw = pd.to_numeric(score.reset_index(drop=True), errors="coerce").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    side_arr = pd.to_numeric(side.reset_index(drop=True), errors="coerce").fillna(1.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    valid = np.isfinite(raw)
    valid_idx = np.flatnonzero(valid)
    out = np.full(raw.shape[0], np.nan, dtype=np.float32)
    if len(risk_predictions) != len(raw):
        raise ValueError("risk_predictions length must match score length")
    if len(valid_idx) == 0:
        return pd.Series(out, index=pd.RangeIndex(len(raw)), dtype=np.float32)

    k = max(1, int(math.ceil(float(frac) * len(valid_idx))))
    max_share = float(np.clip(max_side_share, 0.50, 1.00))
    max_per_side = max(1, int(math.floor(max_share * k)))
    backfill_limit = max(0, int(math.ceil(float(np.clip(max_backfill_share, 0.0, 1.0)) * k)))

    primary_mask = valid & _risk_pass_mask(
        risk_predictions,
        max_pred_bad_mae=float(primary_max_pred_bad_mae),
        max_pred_timeout=float(primary_max_pred_timeout),
        max_pred_lower_tail=float(primary_max_pred_lower_tail),
    )
    backfill_mask = valid & _risk_pass_mask(
        risk_predictions,
        max_pred_bad_mae=float(backfill_max_pred_bad_mae),
        max_pred_timeout=float(backfill_max_pred_timeout),
        max_pred_lower_tail=float(backfill_max_pred_lower_tail),
    )

    selected: list[int] = []
    selected_mask = np.zeros(raw.shape[0], dtype=bool)
    long_selected = 0
    short_selected = 0

    def _try_add(idx: int) -> bool:
        nonlocal long_selected, short_selected
        if selected_mask[idx]:
            return False
        is_short = bool(side_arr[idx] < 0.0)
        if is_short:
            if short_selected >= max_per_side:
                return False
            short_selected += 1
        else:
            if long_selected >= max_per_side:
                return False
            long_selected += 1
        selected_mask[idx] = True
        selected.append(int(idx))
        return True

    ordered = valid_idx[np.argsort(-raw[valid_idx], kind="mergesort")]
    for idx in ordered:
        if len(selected) >= k:
            break
        if primary_mask[idx]:
            _try_add(int(idx))

    backfilled = 0
    for idx in ordered:
        if len(selected) >= k or backfilled >= backfill_limit:
            break
        if backfill_mask[idx] and not primary_mask[idx] and _try_add(int(idx)):
            backfilled += 1

    selected = _enforce_actual_selected_side_share(
        selected,
        raw,
        side_arr,
        max_share=max_share,
    )

    if selected:
        chosen = np.asarray(selected, dtype=np.int64)
        out[chosen] = raw[chosen]
    return pd.Series(out, index=pd.RangeIndex(len(raw)), dtype=np.float32)


def _risk_trimmed_top_score(
    score: pd.Series,
    side: pd.Series,
    risk_predictions: pd.DataFrame,
    frac: float,
    *,
    max_side_share: float,
    trim_share: float,
    protect_top_score_share: float,
    bad_mae_weight: float,
    timeout_weight: float,
    lower_tail_weight: float,
) -> pd.Series:
    base = _side_capped_top_score(
        score,
        side,
        frac,
        max_side_share=float(max_side_share),
    )
    raw = pd.to_numeric(score.reset_index(drop=True), errors="coerce").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    side_arr = pd.to_numeric(side.reset_index(drop=True), errors="coerce").fillna(1.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    if len(risk_predictions) != len(raw):
        raise ValueError("risk_predictions length must match score length")
    selected = [int(i) for i in np.flatnonzero(pd.to_numeric(base, errors="coerce").notna().to_numpy())]
    out = np.full(raw.shape[0], np.nan, dtype=np.float32)
    if not selected:
        return pd.Series(out, index=pd.RangeIndex(len(raw)), dtype=np.float32)

    bad_mae = pd.to_numeric(risk_predictions.get("pred_bad_mae"), errors="coerce").fillna(1.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    timeout = pd.to_numeric(risk_predictions.get("pred_timeout"), errors="coerce").fillna(1.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    lower_tail = pd.to_numeric(risk_predictions.get("pred_lower_tail"), errors="coerce").fillna(1.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    risk_burden = (
        np.float32(bad_mae_weight) * np.clip(bad_mae, 0.0, 1.0)
        + np.float32(timeout_weight) * np.clip(timeout, 0.0, 1.0)
        + np.float32(lower_tail_weight) * np.clip(lower_tail, 0.0, 1.0)
    ).astype(np.float32)
    selected_arr = np.asarray(selected, dtype=np.int64)
    score_order = selected_arr[np.argsort(-raw[selected_arr], kind="mergesort")]
    protect_count = int(math.floor(float(np.clip(protect_top_score_share, 0.0, 1.0)) * len(score_order)))
    protected = set(int(i) for i in score_order[:protect_count])
    trim_candidates = np.asarray([i for i in selected if i not in protected], dtype=np.int64)
    trim_count = min(
        len(trim_candidates),
        int(math.ceil(float(np.clip(trim_share, 0.0, 1.0)) * len(selected))),
    )
    if trim_count > 0 and len(trim_candidates) > 0:
        # Highest predicted risk leaves first; ties drop weaker score first.
        order = np.lexsort((raw[trim_candidates], -risk_burden[trim_candidates]))
        drop = set(int(i) for i in trim_candidates[order[:trim_count]])
        selected = [i for i in selected if i not in drop]

    selected = _enforce_actual_selected_side_share(
        selected,
        raw,
        side_arr,
        max_share=float(max_side_share),
    )
    if selected:
        chosen = np.asarray(selected, dtype=np.int64)
        out[chosen] = raw[chosen]
    return pd.Series(out, index=pd.RangeIndex(len(raw)), dtype=np.float32)


def _side_exposure_capped_score(
    score: pd.Series,
    side: pd.Series,
    *,
    max_side_share: float,
) -> pd.Series:
    raw = pd.to_numeric(score.reset_index(drop=True), errors="coerce").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    side_arr = pd.to_numeric(side.reset_index(drop=True), errors="coerce").fillna(1.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    selected = np.isfinite(raw)
    out = np.full(raw.shape[0], np.nan, dtype=np.float32)
    if not bool(selected.any()):
        return pd.Series(out, index=pd.RangeIndex(len(raw)), dtype=np.float32)
    max_share = float(np.clip(max_side_share, 0.50, 1.00))
    for _ in range(raw.shape[0]):
        idx = np.flatnonzero(selected)
        if len(idx) <= 1:
            break
        long_idx = idx[side_arr[idx] > 0.0]
        short_idx = idx[side_arr[idx] < 0.0]
        if len(long_idx) == 0 or len(short_idx) == 0:
            break
        long_share = len(long_idx) / len(idx)
        short_share = len(short_idx) / len(idx)
        if long_share <= max_share and short_share <= max_share:
            break
        drop_pool = long_idx if long_share > max_share else short_idx
        drop_idx = int(drop_pool[np.argmin(raw[drop_pool])])
        selected[drop_idx] = False
    out[selected] = raw[selected]
    return pd.Series(out, index=pd.RangeIndex(len(raw)), dtype=np.float32)


def _append_model_state_features(
    base: pd.DataFrame,
    state_features: pd.DataFrame,
) -> pd.DataFrame:
    if state_features.empty:
        return base
    aligned = state_features.reindex(base.index).astype(np.float32, copy=False)
    overlap = [c for c in aligned.columns if c in base.columns]
    if overlap:
        aligned = aligned.drop(columns=overlap)
    if aligned.empty:
        return base
    return pd.concat([base, aligned], axis=1, copy=False).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(
        np.float32,
        copy=False,
    )


def _risk_targets(
    train_metrics: pd.DataFrame,
) -> dict[str, pd.Series]:
    u = pd.to_numeric(train_metrics["u_policy_net"], errors="coerce")
    finite_u = u[np.isfinite(u)]
    lower_tail_cutoff = float(finite_u.quantile(0.10)) if len(finite_u) else -0.01
    return {
        "bad_mae": (pd.to_numeric(train_metrics["mae_norm"], errors="coerce").fillna(0.0) >= 1.0).astype(np.float32),
        "timeout": pd.Series(train_metrics["is_timeout"]).astype(float).fillna(0.0).clip(0.0, 1.0).astype(np.float32),
        "lower_tail": (u.fillna(lower_tail_cutoff) <= lower_tail_cutoff).astype(np.float32),
    }


def _mean_seed_predictions(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    seeds: list[int],
) -> np.ndarray:
    seed_preds = [
        _fit_predict(
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
            x_valid=x_valid,
            seed=seed,
        )
        for seed in seeds
    ]
    return np.mean(np.vstack(seed_preds), axis=0).astype(np.float32)


def _fit_risk_predictions(
    *,
    x_train: pd.DataFrame,
    train_metrics: pd.DataFrame,
    x_valid: pd.DataFrame,
    seeds: list[int],
) -> pd.DataFrame:
    risk_train = _risk_targets(train_metrics)
    weights = pd.Series(1.0, index=train_metrics.index, dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    for name, y_train in risk_train.items():
        pred = _mean_seed_predictions(
            x_train=x_train,
            y_train=y_train,
            w_train=weights,
            x_valid=x_valid,
            seeds=seeds,
        )
        out[f"pred_{name}"] = np.clip(pred, 0.0, 1.0).astype(np.float32)
    return pd.DataFrame(out, index=pd.RangeIndex(len(x_valid)))


def _fit_side_calibrated_risk_predictions(
    *,
    train_predictions: pd.DataFrame,
    train_metrics: pd.DataFrame,
    valid_predictions: pd.DataFrame,
    valid_side: pd.Series,
    n_bins: int,
    min_bin_rows: int,
) -> pd.DataFrame:
    risk_train = _risk_targets(train_metrics.reset_index(drop=True))
    train_side = pd.to_numeric(train_metrics["side"].reset_index(drop=True), errors="coerce").fillna(1.0)
    valid_side_ser = pd.to_numeric(valid_side.reset_index(drop=True), errors="coerce").fillna(1.0)
    out: dict[str, np.ndarray] = {}
    for target_name, train_target in risk_train.items():
        col = f"pred_{target_name}"
        train_raw = pd.to_numeric(
            train_predictions.get(col, pd.Series(np.nan, index=train_metrics.index)).reset_index(drop=True),
            errors="coerce",
        )
        valid_raw = pd.to_numeric(
            valid_predictions.get(col, pd.Series(np.nan, index=pd.RangeIndex(len(valid_predictions)))).reset_index(
                drop=True
            ),
            errors="coerce",
        )
        calibrated = np.full(len(valid_raw), np.nan, dtype=np.float32)
        global_rate = _safe_mean(train_target)
        fallback = float(np.clip(global_rate if math.isfinite(global_rate) else 1.0, 0.0, 1.0))
        for _, side_mask_train, side_mask_valid in (
            ("long", train_side > 0.0, valid_side_ser > 0.0),
            ("short", train_side < 0.0, valid_side_ser < 0.0),
        ):
            train_mask = side_mask_train & train_raw.notna() & train_target.notna()
            valid_mask = side_mask_valid & valid_raw.notna()
            valid_idx = np.flatnonzero(valid_mask.to_numpy())
            if int(train_mask.sum()) < max(20, int(min_bin_rows)) or len(valid_idx) == 0:
                calibrated[valid_idx] = np.float32(fallback)
                continue
            local_score = train_raw[train_mask]
            local_target = train_target[train_mask]
            q = max(
                2,
                min(
                    int(n_bins),
                    int(local_score.nunique(dropna=True)),
                    int(len(local_score) // max(1, int(min_bin_rows))),
                ),
            )
            if q < 2:
                calibrated[valid_idx] = np.float32(_safe_mean(local_target))
                continue
            try:
                train_bins, edges = pd.qcut(local_score, q=q, labels=False, retbins=True, duplicates="drop")
            except ValueError:
                calibrated[valid_idx] = np.float32(_safe_mean(local_target))
                continue
            if len(edges) < 3:
                calibrated[valid_idx] = np.float32(_safe_mean(local_target))
                continue
            bin_rates = local_target.groupby(train_bins, dropna=True).mean()
            side_rate = float(local_target.mean())
            valid_bins = np.searchsorted(
                edges[1:-1],
                valid_raw.iloc[valid_idx].to_numpy(dtype=np.float32),
                side="right",
            )
            valid_bins = np.clip(valid_bins, 0, len(edges) - 2)
            mapped = np.asarray(
                [float(bin_rates.get(int(bin_id), side_rate)) for bin_id in valid_bins],
                dtype=np.float32,
            )
            calibrated[valid_idx] = np.clip(mapped, 0.0, 1.0)
        calibrated[~np.isfinite(calibrated)] = np.float32(fallback)
        out[col] = np.clip(calibrated, 0.0, 1.0).astype(np.float32)
    return pd.DataFrame(out, index=pd.RangeIndex(len(valid_predictions)))


def _cluster_side_adjustment_vector(
    *,
    clusters: np.ndarray,
    side: pd.Series,
    side_policy: pd.DataFrame,
) -> np.ndarray:
    out = np.zeros(len(clusters), dtype=np.float32)
    if side_policy.empty:
        return out
    side_arr = pd.to_numeric(side.reset_index(drop=True), errors="coerce").fillna(1.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    policy = side_policy.set_index(["cluster", "side_name"])
    for i, cluster in enumerate(np.asarray(clusters, dtype=np.int32)):
        side_name = "short" if side_arr[i] < 0.0 else "long"
        key = (int(cluster), side_name)
        if key not in policy.index:
            continue
        out[i] = np.float32(float(policy.loc[key].get("cluster_side_policy_adjustment", 0.0) or 0.0))
    return out


def _risk_adjusted_score(
    utility_score: pd.Series,
    risk_predictions: pd.DataFrame,
    *,
    bad_mae_lambda: float,
    timeout_lambda: float,
    lower_tail_lambda: float,
    cluster_adjustment: np.ndarray | None = None,
    cluster_adjustment_lambda: float = 0.0,
) -> pd.Series:
    raw = pd.to_numeric(utility_score.reset_index(drop=True), errors="coerce").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    bad_mae = pd.to_numeric(risk_predictions.get("pred_bad_mae"), errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    timeout = pd.to_numeric(risk_predictions.get("pred_timeout"), errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    lower_tail = pd.to_numeric(risk_predictions.get("pred_lower_tail"), errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    score = (
        raw
        - np.float32(bad_mae_lambda) * bad_mae
        - np.float32(timeout_lambda) * timeout
        - np.float32(lower_tail_lambda) * lower_tail
    )
    if cluster_adjustment is not None and float(cluster_adjustment_lambda) != 0.0:
        score = score + np.float32(cluster_adjustment_lambda) * np.asarray(cluster_adjustment, dtype=np.float32)
    score[~np.isfinite(raw)] = np.nan
    return pd.Series(score.astype(np.float32), index=pd.RangeIndex(len(raw)), dtype=np.float32)


def _fit_side_calibrated_score(
    *,
    train_score: np.ndarray,
    train_metrics: pd.DataFrame,
    valid_score: pd.Series,
    valid_side: pd.Series,
    n_bins: int,
    min_bin_rows: int,
) -> pd.Series:
    train_raw = pd.Series(train_score, dtype=np.float32).reset_index(drop=True)
    train_u = pd.to_numeric(train_metrics["u_policy_net"].reset_index(drop=True), errors="coerce")
    train_side = pd.to_numeric(train_metrics["side"].reset_index(drop=True), errors="coerce").fillna(1.0)
    valid_raw = pd.to_numeric(valid_score.reset_index(drop=True), errors="coerce").astype(np.float32)
    valid_side_ser = pd.to_numeric(valid_side.reset_index(drop=True), errors="coerce").fillna(1.0)
    out = np.full(len(valid_raw), np.nan, dtype=np.float32)
    global_mean = _safe_mean(train_u)
    fallback = float(global_mean) if math.isfinite(float(global_mean)) else 0.0
    for side_name, side_mask_train, side_mask_valid in (
        ("long", train_side > 0.0, valid_side_ser > 0.0),
        ("short", train_side < 0.0, valid_side_ser < 0.0),
    ):
        train_mask = side_mask_train & train_raw.notna() & train_u.notna()
        valid_mask = side_mask_valid & valid_raw.notna()
        if int(train_mask.sum()) < max(20, int(min_bin_rows)) or not bool(valid_mask.any()):
            out[np.flatnonzero(valid_mask.to_numpy())] = np.float32(fallback)
            continue
        local_score = train_raw[train_mask]
        local_u = train_u[train_mask]
        q = max(2, min(int(n_bins), int(local_score.nunique(dropna=True)), int(len(local_score) // max(1, min_bin_rows))))
        if q < 2:
            out[np.flatnonzero(valid_mask.to_numpy())] = np.float32(_safe_mean(local_u))
            continue
        try:
            train_bins, edges = pd.qcut(local_score, q=q, labels=False, retbins=True, duplicates="drop")
        except ValueError:
            out[np.flatnonzero(valid_mask.to_numpy())] = np.float32(_safe_mean(local_u))
            continue
        if len(edges) < 3:
            out[np.flatnonzero(valid_mask.to_numpy())] = np.float32(_safe_mean(local_u))
            continue
        bin_means = local_u.groupby(train_bins, dropna=True).mean()
        side_mean = float(local_u.mean())
        valid_idx = np.flatnonzero(valid_mask.to_numpy())
        valid_bins = np.searchsorted(edges[1:-1], valid_raw.iloc[valid_idx].to_numpy(dtype=np.float32), side="right")
        valid_bins = np.clip(valid_bins, 0, len(edges) - 2)
        mapped = np.asarray([float(bin_means.get(int(bin_id), side_mean)) for bin_id in valid_bins], dtype=np.float32)
        out[valid_idx] = mapped
    return pd.Series(out, index=pd.RangeIndex(len(valid_raw)), dtype=np.float32)


def _robust_upper_tail_score(train_values: pd.Series, valid_values: pd.Series) -> np.ndarray:
    train = pd.to_numeric(train_values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    valid = pd.to_numeric(valid_values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(train) < 20:
        return np.zeros(len(valid), dtype=np.float32)
    lo = float(train.quantile(0.50))
    hi = float(train.quantile(0.90))
    denom = max(hi - lo, 1e-6)
    return np.clip((valid.to_numpy(dtype=np.float32) - lo) / denom, 0.0, 1.0).astype(np.float32)


def _gmm_state_risk_penalty(
    train_gmm: pd.DataFrame,
    valid_gmm: pd.DataFrame,
    *,
    entropy_lambda: float,
    recon_lambda: float,
    mahal_lambda: float,
) -> np.ndarray:
    penalty = np.zeros(len(valid_gmm), dtype=np.float32)
    if "cluster_entropy_norm" in valid_gmm.columns:
        entropy = pd.to_numeric(valid_gmm["cluster_entropy_norm"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        penalty += np.float32(entropy_lambda) * entropy.to_numpy(dtype=np.float32)
    if "dae_reconstruction_error" in train_gmm.columns and "dae_reconstruction_error" in valid_gmm.columns:
        penalty += np.float32(recon_lambda) * _robust_upper_tail_score(
            train_gmm["dae_reconstruction_error"],
            valid_gmm["dae_reconstruction_error"],
        )
    if "expected_mahalanobis" in train_gmm.columns and "expected_mahalanobis" in valid_gmm.columns:
        penalty += np.float32(mahal_lambda) * _robust_upper_tail_score(
            train_gmm["expected_mahalanobis"],
            valid_gmm["expected_mahalanobis"],
        )
    return penalty.astype(np.float32)


def _calibrated_soft_risk_score(
    calibrated_score: pd.Series,
    risk_predictions: pd.DataFrame,
    gmm_penalty: np.ndarray,
    *,
    bad_mae_lambda: float,
    timeout_lambda: float,
    lower_tail_lambda: float,
) -> pd.Series:
    base = pd.to_numeric(calibrated_score.reset_index(drop=True), errors="coerce").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    bad_mae = pd.to_numeric(risk_predictions.get("pred_bad_mae"), errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    timeout = pd.to_numeric(risk_predictions.get("pred_timeout"), errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    lower_tail = pd.to_numeric(risk_predictions.get("pred_lower_tail"), errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    score = (
        base
        - np.float32(bad_mae_lambda) * bad_mae
        - np.float32(timeout_lambda) * timeout
        - np.float32(lower_tail_lambda) * lower_tail
        - np.asarray(gmm_penalty, dtype=np.float32)
    )
    score[~np.isfinite(base)] = np.nan
    return pd.Series(score.astype(np.float32), index=pd.RangeIndex(len(score)), dtype=np.float32)


def _recall_preserving_calibrated_risk_score(
    *,
    base_score: pd.Series,
    discovery_score: pd.Series,
    calibrated_risk_predictions: pd.DataFrame,
    candidate_mask: np.ndarray,
    bad_mae_lambda: float,
    timeout_lambda: float,
    lower_tail_lambda: float,
    discovery_lambda: float,
) -> pd.Series:
    base = pd.to_numeric(base_score.reset_index(drop=True), errors="coerce").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    discovery = pd.to_numeric(discovery_score.reset_index(drop=True), errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    bad_mae = pd.to_numeric(calibrated_risk_predictions.get("pred_bad_mae"), errors="coerce").fillna(1.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    timeout = pd.to_numeric(calibrated_risk_predictions.get("pred_timeout"), errors="coerce").fillna(1.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    lower_tail = pd.to_numeric(
        calibrated_risk_predictions.get("pred_lower_tail"),
        errors="coerce",
    ).fillna(1.0).to_numpy(dtype=np.float32, copy=False)
    mask = np.asarray(candidate_mask, dtype=bool)
    if len(mask) != len(base):
        raise ValueError("candidate_mask length must match base score length")
    score = (
        base
        + np.float32(discovery_lambda) * np.clip(discovery, 0.0, 1.0)
        - np.float32(bad_mae_lambda) * np.clip(bad_mae, 0.0, 1.0)
        - np.float32(timeout_lambda) * np.clip(timeout, 0.0, 1.0)
        - np.float32(lower_tail_lambda) * np.clip(lower_tail, 0.0, 1.0)
    ).astype(np.float32)
    score[(~mask) | (~np.isfinite(base))] = np.nan
    return pd.Series(score, index=pd.RangeIndex(len(score)), dtype=np.float32)


def _threshold_score(score: pd.Series, threshold: float) -> pd.Series:
    out = pd.to_numeric(score.reset_index(drop=True), errors="coerce").astype(np.float32)
    return out.where(out >= float(threshold))


def _risk_capped_score(
    score: pd.Series,
    risk_predictions: pd.DataFrame,
    *,
    max_pred_bad_mae: float,
    max_pred_timeout: float,
    max_pred_lower_tail: float,
) -> pd.Series:
    out = pd.to_numeric(score.reset_index(drop=True), errors="coerce").astype(np.float32)
    mask = out.notna()
    for col, threshold in (
        ("pred_bad_mae", max_pred_bad_mae),
        ("pred_timeout", max_pred_timeout),
        ("pred_lower_tail", max_pred_lower_tail),
    ):
        if col not in risk_predictions.columns or not math.isfinite(float(threshold)):
            continue
        values = pd.to_numeric(risk_predictions[col], errors="coerce").fillna(1.0)
        mask &= values.reset_index(drop=True) <= float(threshold)
    return out.where(mask)


def _risk_pass_mask(
    risk_predictions: pd.DataFrame,
    *,
    max_pred_bad_mae: float,
    max_pred_timeout: float,
    max_pred_lower_tail: float,
) -> np.ndarray:
    mask = np.ones(len(risk_predictions), dtype=bool)
    for col, threshold in (
        ("pred_bad_mae", max_pred_bad_mae),
        ("pred_timeout", max_pred_timeout),
        ("pred_lower_tail", max_pred_lower_tail),
    ):
        if col not in risk_predictions.columns or not math.isfinite(float(threshold)):
            continue
        values = pd.to_numeric(risk_predictions[col], errors="coerce").fillna(1.0).to_numpy(
            dtype=np.float32,
            copy=False,
        )
        mask &= values <= np.float32(threshold)
    return mask


def _mask_score(score: pd.Series, mask: np.ndarray) -> pd.Series:
    out = pd.to_numeric(score.reset_index(drop=True), errors="coerce").astype(np.float32)
    mask_arr = np.asarray(mask, dtype=bool)
    if len(mask_arr) != len(out):
        raise ValueError("mask length must match score length")
    return out.where(mask_arr)


def _per_timestamp_top_mask(
    frame: pd.DataFrame,
    score: pd.Series,
    *,
    top_n: int | None = None,
    top_frac: float | None = None,
) -> np.ndarray:
    arr = pd.to_numeric(score.reset_index(drop=True), errors="coerce").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    ts = pd.to_datetime(frame["__ts__"].reset_index(drop=True), errors="coerce")
    out = np.zeros(len(arr), dtype=bool)
    valid = np.isfinite(arr) & ts.notna().to_numpy()
    if not bool(valid.any()):
        return out
    positions = pd.Series(np.arange(len(arr), dtype=np.int64))
    for _, ids in positions[valid].groupby(ts[valid], sort=False):
        idx = ids.to_numpy(dtype=np.int64)
        if len(idx) == 0:
            continue
        if top_n is not None:
            k = min(int(top_n), len(idx))
        elif top_frac is not None:
            k = max(1, int(math.ceil(float(top_frac) * len(idx))))
        else:
            k = len(idx)
        if k <= 0:
            continue
        order = idx[np.argsort(-arr[idx], kind="mergesort")[:k]]
        out[order] = True
    return out


def _max_percentile_score(scores: list[pd.Series]) -> pd.Series:
    parts: list[pd.Series] = []
    for score in scores:
        raw = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
        if raw.notna().any():
            parts.append(raw.rank(method="average", pct=True).astype(np.float32))
    if not parts:
        return pd.Series(np.nan, index=pd.RangeIndex(0), dtype=np.float32)
    return pd.concat(parts, axis=1).max(axis=1).astype(np.float32)


def _ranker_relevance(
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    *,
    mode: str = "utility_quintile",
    oracle_alpha: float = 1.0,
    bad_mae_beta: float = 0.75,
    timeout_gamma: float = 0.50,
) -> np.ndarray:
    ts = pd.to_datetime(train_frame["__ts__"], errors="coerce")
    u = pd.to_numeric(train_metrics["u_policy_net"], errors="coerce").fillna(0.0)
    if str(mode) == "oracle_enriched":
        group_mean = u.groupby(ts, sort=False).transform("mean")
        group_std = u.groupby(ts, sort=False).transform("std").replace(0.0, np.nan).fillna(1.0)
        realized_edge_z = ((u - group_mean) / group_std).clip(-3.0, 3.0)
        pct_for_oracle = u.groupby(ts, sort=False).rank(method="average", pct=True)
        oracle_top = (pct_for_oracle >= 0.90).astype(np.float32)
        bad_mae = (pd.to_numeric(train_metrics["mae_norm"], errors="coerce").fillna(0.0) >= 1.0).astype(np.float32)
        timeout = pd.Series(train_metrics["is_timeout"]).astype(float).fillna(0.0).clip(0.0, 1.0).astype(np.float32)
        blended = (
            realized_edge_z.astype(np.float32)
            + np.float32(oracle_alpha) * oracle_top
            - np.float32(bad_mae_beta) * bad_mae
            - np.float32(timeout_gamma) * timeout
        )
        pct = blended.groupby(ts, sort=False).rank(method="average", pct=True)
    elif str(mode) == "path_quality":
        group_mean = u.groupby(ts, sort=False).transform("mean")
        group_std = u.groupby(ts, sort=False).transform("std").replace(0.0, np.nan).fillna(1.0)
        realized_edge_z = ((u - group_mean) / group_std).clip(-3.0, 3.0)
        pct_for_oracle = u.groupby(ts, sort=False).rank(method="average", pct=True)
        oracle_top = (pct_for_oracle >= 0.90).astype(np.float32)
        bad_mae = (pd.to_numeric(train_metrics["mae_norm"], errors="coerce").fillna(0.0) >= 1.0).astype(np.float32)
        timeout = pd.Series(train_metrics["is_timeout"]).astype(float).fillna(0.0).clip(0.0, 1.0).astype(np.float32)
        finite_u = u[np.isfinite(u)]
        lower_tail_cutoff = float(finite_u.quantile(0.10)) if len(finite_u) else -0.01
        lower_tail = (u <= lower_tail_cutoff).astype(np.float32)
        clean_path = ((bad_mae <= 0.0) & (timeout <= 0.0) & (lower_tail <= 0.0)).astype(np.float32)
        blended = (
            realized_edge_z.astype(np.float32)
            + np.float32(0.80) * oracle_top
            + np.float32(0.75) * clean_path
            + np.float32(0.25) * (u > 0.0).astype(np.float32)
            - np.float32(1.75) * bad_mae
            - np.float32(0.75) * timeout
            - np.float32(0.75) * lower_tail
        )
        pct = blended.groupby(ts, sort=False).rank(method="average", pct=True)
    else:
        pct = u.groupby(ts, sort=False).rank(method="average", pct=True)
    relevance = np.floor(pct.fillna(0.0).to_numpy(dtype=np.float32) * 5.0).astype(np.int32)
    return np.clip(relevance, 0, 4)


def _timestamp_groups(train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.to_datetime(train_frame["__ts__"], errors="coerce").astype("int64").to_numpy()
    order = np.argsort(ts, kind="mergesort")
    sorted_ts = ts[order]
    if len(sorted_ts) == 0:
        return order, np.asarray([], dtype=np.int32)
    _, counts = np.unique(sorted_ts, return_counts=True)
    return order.astype(np.int64, copy=False), counts.astype(np.int32, copy=False)


def _fit_lgbm_ranker_predictions(
    *,
    x_train: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    seeds: list[int],
    relevance_mode: str = "utility_quintile",
) -> tuple[np.ndarray, np.ndarray, str]:
    if not _LIGHTGBM_AVAILABLE or LGBMRanker is None:
        return (
            np.full(len(x_train), np.nan, dtype=np.float32),
            np.full(len(x_valid), np.nan, dtype=np.float32),
            "lightgbm_unavailable",
        )
    order, group = _timestamp_groups(train_frame.reset_index(drop=True))
    if len(group) == 0 or len(order) != len(x_train):
        return (
            np.full(len(x_train), np.nan, dtype=np.float32),
            np.full(len(x_valid), np.nan, dtype=np.float32),
            "empty_or_invalid_rank_groups",
        )
    y = _ranker_relevance(
        train_frame.reset_index(drop=True),
        train_metrics.reset_index(drop=True),
        mode=str(relevance_mode),
    )
    weights = pd.to_numeric(w_train.reset_index(drop=True), errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    x_train_sorted = x_train.reset_index(drop=True).iloc[order]
    y_sorted = y[order]
    weights_sorted = weights[order]
    train_preds: list[np.ndarray] = []
    valid_preds: list[np.ndarray] = []
    for seed in seeds:
        model = LGBMRanker(
            objective="lambdarank",
            n_estimators=80,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=int(seed),
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(
            x_train_sorted,
            y_sorted,
            group=group,
            sample_weight=weights_sorted,
        )
        train_preds.append(model.predict(x_train.reset_index(drop=True)).astype(np.float32))
        valid_preds.append(model.predict(x_valid.reset_index(drop=True)).astype(np.float32))
    return (
        np.mean(np.vstack(train_preds), axis=0).astype(np.float32),
        np.mean(np.vstack(valid_preds), axis=0).astype(np.float32),
        "ok",
    )


def _fit_side_lgbm_ranker_predictions(
    *,
    x_train: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    seeds: list[int],
    relevance_mode: str = "utility_quintile",
) -> tuple[np.ndarray, np.ndarray, str]:
    train_side = pd.to_numeric(train_metrics["side"], errors="coerce").fillna(1.0).reset_index(drop=True)
    valid_side = pd.to_numeric(valid_metrics["side"], errors="coerce").fillna(1.0).reset_index(drop=True)
    train_out = np.full(len(x_train), np.nan, dtype=np.float32)
    valid_out = np.full(len(x_valid), np.nan, dtype=np.float32)
    statuses: list[str] = []
    for side_name, train_mask, valid_mask in (
        ("long", train_side > 0.0, valid_side > 0.0),
        ("short", train_side < 0.0, valid_side < 0.0),
    ):
        train_idx = np.flatnonzero(train_mask.to_numpy())
        valid_idx = np.flatnonzero(valid_mask.to_numpy())
        if len(train_idx) < 500 or len(valid_idx) == 0:
            statuses.append(f"{side_name}:insufficient_rows")
            continue
        local_train_pred, local_valid_pred, status = _fit_lgbm_ranker_predictions(
            x_train=x_train.reset_index(drop=True).iloc[train_idx].reset_index(drop=True),
            train_frame=train_frame.reset_index(drop=True).iloc[train_idx].reset_index(drop=True),
            train_metrics=train_metrics.reset_index(drop=True).iloc[train_idx].reset_index(drop=True),
            w_train=w_train.reset_index(drop=True).iloc[train_idx].reset_index(drop=True),
            x_valid=x_valid.reset_index(drop=True).iloc[valid_idx].reset_index(drop=True),
            seeds=seeds,
            relevance_mode=relevance_mode,
        )
        statuses.append(f"{side_name}:{status}")
        train_out[train_idx] = local_train_pred
        valid_out[valid_idx] = local_valid_pred
    return train_out, valid_out, ";".join(statuses)


def _selected_prediction_summary(
    score: pd.Series,
    top_frac: float,
    predictions: pd.DataFrame,
    *,
    selection_mode: str = "top_frac",
) -> dict[str, float]:
    idx = _selection_indices(score, top_frac, selection_mode)
    if len(idx) == 0 or predictions.empty:
        return {f"{col}_top_mean": float("nan") for col in predictions.columns}
    selected = predictions.iloc[idx]
    return {
        f"{col}_top_mean": _safe_mean(selected[col])
        for col in selected.columns
    }


def _weekly_utility_tail(timestamps: pd.Series, utility: pd.Series) -> dict[str, float]:
    ts = pd.to_datetime(timestamps, errors="coerce")
    u = pd.to_numeric(utility, errors="coerce").to_numpy(dtype=np.float32, copy=False)
    valid = ts.notna().to_numpy() & np.isfinite(u)
    if not bool(valid.any()):
        return {
            "selected_week_count": 0,
            "weekly_mean_u_q10": float("nan"),
            "worst_weekly_mean_u": float("nan"),
            "positive_week_rate": float("nan"),
        }
    weeks = ts[valid].dt.to_period("W-SUN").astype(str)
    weekly_mean = pd.Series(u[valid], index=weeks).groupby(level=0, sort=False).mean()
    weekly_values = weekly_mean.to_numpy(dtype=np.float32, copy=False)
    return {
        "selected_week_count": int(len(weekly_values)),
        "weekly_mean_u_q10": _safe_quantile(pd.Series(weekly_values), 0.10),
        "worst_weekly_mean_u": float(np.min(weekly_values)) if len(weekly_values) else float("nan"),
        "positive_week_rate": float(np.mean(weekly_values > 0.0)) if len(weekly_values) else float("nan"),
    }


def _selected_weekly_utility_tail(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    score: pd.Series,
    top_frac: float,
    selection_mode: str = "top_frac",
) -> dict[str, float]:
    idx = _selection_indices(score, top_frac, selection_mode)
    if len(idx) == 0 or "__ts__" not in frame.columns or "u_policy_net" not in metrics.columns:
        return {
            "selected_week_count": 0,
            "weekly_mean_u_q10": float("nan"),
            "worst_weekly_mean_u": float("nan"),
            "positive_week_rate": float("nan"),
        }
    return _weekly_utility_tail(frame.iloc[idx]["__ts__"], metrics.iloc[idx]["u_policy_net"])


def _selection_metrics(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    period: str,
    top_frac: float,
    selection_mode: str = "top_frac",
) -> dict[str, Any]:
    effective_top_frac = 1.0 if str(selection_mode) == "all_finite" else float(top_frac)
    row = _base_selection_metrics(
        frame=frame,
        metrics=metrics,
        target=target,
        score=score,
        arm=arm,
        selector=selector,
        period=period,
        top_frac=effective_top_frac,
    )
    row["top_frac"] = float(top_frac)
    row["selection_mode"] = str(selection_mode)
    row.update(
        _selected_weekly_utility_tail(
            frame=frame,
            metrics=metrics,
            score=score,
            top_frac=top_frac,
            selection_mode=selection_mode,
        )
    )
    return row


def _oracle_diagnostic_rows(
    *,
    month: str,
    label_arm: str,
    weight_arm: str,
    top_frac: float,
    valid: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    clusters: np.ndarray,
    scores: dict[str, pd.Series],
    min_group_rows: int,
    selection_modes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    side = pd.to_numeric(metrics["side"], errors="coerce").fillna(1.0).reset_index(drop=True)
    group_frame = pd.DataFrame(
        {
            "all": "all",
            "side": np.where(side.to_numpy(dtype=np.float32) < 0.0, "short", "long"),
            "cluster": [f"cluster_{int(c)}" for c in np.asarray(clusters, dtype=np.int32)],
            "cluster_side": [
                f"cluster_{int(c)}_{'short' if s < 0.0 else 'long'}"
                for c, s in zip(np.asarray(clusters, dtype=np.int32), side.to_numpy(dtype=np.float32), strict=False)
            ],
            "symbol": valid["__symbol__"].reset_index(drop=True).astype(str).to_numpy(),
        }
    )
    metrics_local = metrics.reset_index(drop=True)
    valid_local = valid.reset_index(drop=True)
    target_local = target.reset_index(drop=True)
    score_map = {
        "oracle_utility": pd.to_numeric(metrics_local["u_policy_net"], errors="coerce").astype(np.float32),
        **{name: pd.to_numeric(score.reset_index(drop=True), errors="coerce").astype(np.float32) for name, score in scores.items()},
    }
    selection_modes = selection_modes or {}
    rows: list[dict[str, Any]] = []
    for dimension in ("all", "side", "cluster", "cluster_side", "symbol"):
        for group_value, group_idx_raw in group_frame.groupby(dimension, sort=False).indices.items():
            group_idx = np.asarray(group_idx_raw, dtype=np.int64)
            if len(group_idx) < int(min_group_rows):
                continue
            group_metrics = metrics_local.iloc[group_idx].reset_index(drop=True)
            group_valid = valid_local.iloc[group_idx].reset_index(drop=True)
            group_target = target_local.iloc[group_idx].reset_index(drop=True)
            baseline_mean = _safe_mean(group_metrics["u_policy_net"])
            baseline_hit = _safe_mean(group_metrics["u_policy_net"] > 0.0)
            random_rows = max(1, int(math.ceil(float(top_frac) * len(group_idx))))
            rows.append(
                {
                    "period": month,
                    "label_arm": label_arm,
                    "weight_arm": weight_arm,
                    "top_frac": float(top_frac),
                    "dimension": dimension,
                    "group": str(group_value),
                    "selector": "random_expected",
                    "rows": int(len(group_idx)),
                    "selected_rows": int(random_rows),
                    "mean_u": baseline_mean,
                    "hit_u": baseline_hit,
                    "q10_u": _safe_quantile(group_metrics["u_policy_net"], 0.10),
                    "selected_long_share": _safe_mean(pd.to_numeric(group_metrics["side"], errors="coerce") > 0.0),
                    "selected_short_share": _safe_mean(pd.to_numeric(group_metrics["side"], errors="coerce") < 0.0),
                    "bad_mae_1r_rate": _safe_mean(group_metrics["mae_norm"] >= 1.0),
                    "timeout_rate": _safe_mean(group_metrics["is_timeout"].astype(float) > 0.5),
                    "score_finite_frac": 1.0,
                    "no_trade_rate": 0.0,
                    **_weekly_utility_tail(group_valid["__ts__"], group_metrics["u_policy_net"]),
                }
            )
            for selector, score in score_map.items():
                group_score = score.iloc[group_idx].reset_index(drop=True)
                selection_mode = "top_frac" if selector == "oracle_utility" else selection_modes.get(selector, "top_frac")
                row = _selection_metrics(
                    frame=group_valid,
                    metrics=group_metrics,
                    target=group_target,
                    score=group_score,
                    arm=f"{label_arm}::{weight_arm}::{selector}",
                    selector=selector,
                    period=month,
                    top_frac=float(top_frac),
                    selection_mode=selection_mode,
                )
                recall = _oracle_recall_summary(
                    score=group_score,
                    oracle_score=group_metrics["u_policy_net"],
                    top_frac=float(top_frac),
                    selection_mode=selection_mode,
                )
                row.update(
                    {
                        "label_arm": label_arm,
                        "weight_arm": weight_arm,
                        "dimension": dimension,
                        "group": str(group_value),
                        "score_finite_frac": float(pd.to_numeric(group_score, errors="coerce").notna().mean()),
                        "no_trade_rate": float(1.0 - pd.to_numeric(group_score, errors="coerce").notna().mean()),
                        **recall,
                    }
                )
                rows.append(row)
    return rows


def _score_quantile_side_rows(
    *,
    month: str,
    label_arm: str,
    weight_arm: str,
    selector: str,
    score: pd.Series,
    metrics: pd.DataFrame,
    n_bins: int,
    min_bin_rows: int,
) -> list[dict[str, Any]]:
    local_score = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
    local_metrics = metrics.reset_index(drop=True)
    side = pd.to_numeric(local_metrics["side"], errors="coerce").fillna(1.0)
    rows: list[dict[str, Any]] = []
    for side_name, side_mask in (("long", side > 0.0), ("short", side < 0.0)):
        finite_mask = side_mask & local_score.notna()
        if int(finite_mask.sum()) < max(2, int(min_bin_rows)):
            continue
        side_score = local_score[finite_mask]
        q = max(2, min(int(n_bins), int(side_score.nunique(dropna=True)), int(len(side_score) // max(1, min_bin_rows))))
        if q < 2:
            continue
        try:
            bins = pd.qcut(side_score, q=q, labels=False, duplicates="drop")
        except ValueError:
            continue
        if bins.isna().all():
            continue
        side_metrics = local_metrics.loc[finite_mask].reset_index(drop=True)
        side_bins = pd.Series(bins.to_numpy(dtype=np.float32), index=side_metrics.index)
        for bin_id in sorted(int(v) for v in side_bins.dropna().unique()):
            mask = side_bins.eq(float(bin_id))
            if not bool(mask.any()):
                continue
            bin_metrics = side_metrics.loc[mask]
            bin_score = side_score.reset_index(drop=True).loc[mask]
            rows.append(
                {
                    "period": month,
                    "label_arm": label_arm,
                    "weight_arm": weight_arm,
                    "selector": selector,
                    "side_name": side_name,
                    "score_quantile": int(bin_id + 1),
                    "score_quantiles": int(q),
                    "rows": int(mask.sum()),
                    "mean_score": _safe_mean(bin_score),
                    "mean_u": _safe_mean(bin_metrics["u_policy_net"]),
                    "hit_u": _safe_mean(bin_metrics["u_policy_net"] > 0.0),
                    "q10_u": _safe_quantile(bin_metrics["u_policy_net"], 0.10),
                    "bad_mae_1r_rate": _safe_mean(bin_metrics["mae_norm"] >= 1.0),
                    "timeout_rate": _safe_mean(bin_metrics["is_timeout"].astype(float) > 0.5),
                }
            )
    return rows


def _calibration_viability_rows(
    calibration_diagnostics: pd.DataFrame,
    *,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    thresholds = {**DEFAULT_VIABILITY_THRESHOLDS, **(thresholds or {})}
    columns = [
        "label_arm",
        "weight_arm",
        "calibration_selector",
        "calibration_group_count",
        "utility_quantile_spearman_mean",
        "utility_top_bottom_u_mean",
        "utility_monotonic_rate",
        "bad_mae_quantile_spearman_mean",
        "bad_mae_bottom_top_improvement_mean",
        "bad_mae_improves_rate",
        "learnability_pass",
        "bad_mae_calibration_pass",
    ]
    if calibration_diagnostics.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    group_cols = ["label_arm", "weight_arm", "selector"]
    for (label_arm, weight_arm, selector), selector_rows in calibration_diagnostics.groupby(
        group_cols,
        sort=False,
        dropna=False,
        observed=True,
    ):
        group_stats: list[dict[str, float]] = []
        for _, local in selector_rows.groupby(["period", "side_name"], sort=False, dropna=False, observed=True):
            local = local.sort_values("score_quantile")
            if len(local) < 2:
                continue
            q = pd.to_numeric(local["score_quantile"], errors="coerce")
            mean_u = pd.to_numeric(local["mean_u"], errors="coerce")
            bad_mae = pd.to_numeric(local["bad_mae_1r_rate"], errors="coerce")
            if q.nunique(dropna=True) < 2:
                continue
            first = local.iloc[0]
            last = local.iloc[-1]
            utility_delta = float(last["mean_u"] - first["mean_u"])
            bad_mae_improvement = float(first["bad_mae_1r_rate"] - last["bad_mae_1r_rate"])
            group_stats.append(
                {
                    "utility_spearman": _spearman(q, mean_u),
                    "bad_mae_spearman": _spearman(q, bad_mae),
                    "utility_delta": utility_delta,
                    "bad_mae_improvement": bad_mae_improvement,
                }
            )
        group_frame = pd.DataFrame(group_stats)
        group_count = int(len(group_frame))
        if group_frame.empty:
            utility_rate = float("nan")
            bad_mae_rate = float("nan")
            utility_delta_mean = float("nan")
            bad_mae_improvement_mean = float("nan")
            utility_spearman_mean = float("nan")
            bad_mae_spearman_mean = float("nan")
        else:
            utility_spearman = pd.to_numeric(group_frame["utility_spearman"], errors="coerce")
            bad_mae_spearman = pd.to_numeric(group_frame["bad_mae_spearman"], errors="coerce")
            utility_ok = (
                pd.to_numeric(group_frame["utility_delta"], errors="coerce") > 0.0
            ) & (utility_spearman.isna() | (utility_spearman > 0.0))
            bad_mae_ok = (
                pd.to_numeric(group_frame["bad_mae_improvement"], errors="coerce") >= 0.0
            ) & (bad_mae_spearman.isna() | (bad_mae_spearman <= 0.0))
            utility_rate = _safe_mean(utility_ok.astype(float))
            bad_mae_rate = _safe_mean(bad_mae_ok.astype(float))
            utility_delta_mean = _safe_mean(group_frame["utility_delta"])
            bad_mae_improvement_mean = _safe_mean(group_frame["bad_mae_improvement"])
            utility_spearman_mean = _safe_mean(group_frame["utility_spearman"])
            bad_mae_spearman_mean = _safe_mean(group_frame["bad_mae_spearman"])
        learnability_pass = bool(
            group_count >= int(thresholds["min_calibration_groups"])
            and math.isfinite(utility_rate)
            and utility_rate >= float(thresholds["min_utility_monotonic_rate"])
            and math.isfinite(utility_delta_mean)
            and utility_delta_mean > 0.0
        )
        bad_mae_calibration_pass = bool(
            group_count >= int(thresholds["min_calibration_groups"])
            and math.isfinite(bad_mae_rate)
            and bad_mae_rate >= float(thresholds["min_bad_mae_improves_rate"])
            and math.isfinite(bad_mae_improvement_mean)
            and bad_mae_improvement_mean >= 0.0
        )
        rows.append(
            {
                "label_arm": str(label_arm),
                "weight_arm": str(weight_arm),
                "calibration_selector": str(selector),
                "calibration_group_count": group_count,
                "utility_quantile_spearman_mean": utility_spearman_mean,
                "utility_top_bottom_u_mean": utility_delta_mean,
                "utility_monotonic_rate": utility_rate,
                "bad_mae_quantile_spearman_mean": bad_mae_spearman_mean,
                "bad_mae_bottom_top_improvement_mean": bad_mae_improvement_mean,
                "bad_mae_improves_rate": bad_mae_rate,
                "learnability_pass": learnability_pass,
                "bad_mae_calibration_pass": bad_mae_calibration_pass,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _stage_gate_summary_rows(stage_gate_diagnostics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "label_arm",
        "weight_arm",
        "top_frac",
        "stage",
        "stage_rows_mean",
        "stage_oracle_recall_mean",
        "stage_oracle_recall_min",
        "stage_june_oracle_recall",
        "stage_bad_mae_1r_rate_mean",
        "stage_bad_mae_1r_rate_max",
        "stage_timeout_rate_mean",
        "stage_timeout_rate_max",
        "stage_mean_u_mean",
    ]
    if stage_gate_diagnostics.empty:
        return pd.DataFrame(columns=columns)
    local = stage_gate_diagnostics[stage_gate_diagnostics["side_name"].eq("all")].copy()
    if local.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        local.groupby(["label_arm", "weight_arm", "top_frac", "stage"], sort=False, dropna=False, observed=True)
        .agg(
            stage_rows_mean=("rows", "mean"),
            stage_oracle_recall_mean=("oracle_recall", "mean"),
            stage_oracle_recall_min=("oracle_recall", "min"),
            stage_bad_mae_1r_rate_mean=("bad_mae_1r_rate", "mean"),
            stage_bad_mae_1r_rate_max=("bad_mae_1r_rate", "max"),
            stage_timeout_rate_mean=("timeout_rate", "mean"),
            stage_timeout_rate_max=("timeout_rate", "max"),
            stage_mean_u_mean=("mean_u", "mean"),
        )
        .reset_index()
    )
    june = local[local["period"].eq("2026-06")]
    if not june.empty:
        june_recall = (
            june.groupby(["label_arm", "weight_arm", "top_frac", "stage"], sort=False, dropna=False, observed=True)[
                "oracle_recall"
            ]
            .mean()
            .rename("stage_june_oracle_recall")
            .reset_index()
        )
        grouped = grouped.merge(june_recall, on=["label_arm", "weight_arm", "top_frac", "stage"], how="left")
    else:
        grouped["stage_june_oracle_recall"] = np.nan
    return grouped[columns]


def _first_failed_gate(gates: list[tuple[str, bool | None]]) -> str:
    for name, passed in gates:
        if passed is False:
            return name
    return "pass"


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return bool(value)


def _build_label_viability_matrix(
    *,
    aggregate: pd.DataFrame,
    calibration_diagnostics: pd.DataFrame,
    stage_gate_diagnostics: pd.DataFrame,
    evaluation_utility_source: str,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    thresholds = {**DEFAULT_VIABILITY_THRESHOLDS, **(thresholds or {})}
    calibration = _calibration_viability_rows(calibration_diagnostics, thresholds=thresholds)
    stage_summary = _stage_gate_summary_rows(stage_gate_diagnostics)
    rows: list[dict[str, Any]] = []
    if aggregate.empty:
        return pd.DataFrame()
    calibration_index = (
        calibration.set_index(["label_arm", "weight_arm", "calibration_selector"])
        if not calibration.empty
        else pd.DataFrame()
    )
    stage_index = (
        stage_summary.set_index(["label_arm", "weight_arm", "top_frac", "stage"])
        if not stage_summary.empty
        else pd.DataFrame()
    )
    net_utility_source = "net" in str(evaluation_utility_source).lower()
    for item in aggregate.to_dict(orient="records"):
        selector = str(item.get("cluster_policy", ""))
        label_arm = str(item.get("label_arm", ""))
        weight_arm = str(item.get("weight_arm", ""))
        top_frac = float(item.get("top_frac", float("nan")))
        months = int(item.get("months", 0) or 0)
        positive_months = int(item.get("positive_months", 0) or 0)
        positive_month_rate = positive_months / max(months, 1)
        mean_u = float(item.get("mean_u", float("nan")))
        worst_month_mean_u = float(item.get("worst_month_mean_u", float("nan")))
        bad_mae_rate = float(item.get("bad_mae_1r_rate", float("nan")))
        timeout_rate = float(item.get("timeout_rate", float("nan")))
        weekly_q10 = float(item.get("weekly_mean_u_q10", float("nan")))
        max_month_bad_mae_rate = float(item.get("max_month_bad_mae_1r_rate", float("nan")))
        max_month_timeout_rate = float(item.get("max_month_timeout_rate", float("nan")))
        worst_month_weekly_q10 = float(item.get("worst_month_weekly_mean_u_q10", float("nan")))
        selected_rows = float(item.get("selected_rows", float("nan")))
        selected_long_share = float(item.get("selected_long_share", float("nan")))
        selected_short_share = float(item.get("selected_short_share", float("nan")))
        score_ic_u = float(item.get("score_ic_u", float("nan")))
        fallback_final_recall = float(item.get("oracle_recall_at_model_top_k", float("nan")))

        calibration_selector = CALIBRATION_SELECTOR_BY_POLICY.get(selector, "")
        cal_row: dict[str, Any] = {}
        if calibration_selector and not calibration_index.empty:
            key = (label_arm, weight_arm, calibration_selector)
            if key in calibration_index.index:
                cal_raw = calibration_index.loc[key]
                if isinstance(cal_raw, pd.DataFrame):
                    cal_raw = cal_raw.iloc[0]
                cal_row = cal_raw.to_dict()

        stage_a_row: dict[str, Any] = {}
        final_stage_row: dict[str, Any] = {}
        if not stage_index.empty:
            stage_a_key = (label_arm, weight_arm, top_frac, "stageA_candidate_union")
            if stage_a_key in stage_index.index:
                raw_stage_a = stage_index.loc[stage_a_key]
                if isinstance(raw_stage_a, pd.DataFrame):
                    raw_stage_a = raw_stage_a.iloc[0]
                stage_a_row = raw_stage_a.to_dict()
            final_stage = FINAL_STAGE_BY_POLICY.get(selector, "")
            if final_stage:
                final_key = (label_arm, weight_arm, top_frac, final_stage)
                if final_key in stage_index.index:
                    raw_final = stage_index.loc[final_key]
                    if isinstance(raw_final, pd.DataFrame):
                        raw_final = raw_final.iloc[0]
                    final_stage_row = raw_final.to_dict()

        stage_a_recall = float(stage_a_row.get("stage_oracle_recall_mean", float("nan")))
        stage_a_june_recall = float(stage_a_row.get("stage_june_oracle_recall", float("nan")))
        final_recall = float(final_stage_row.get("stage_oracle_recall_mean", fallback_final_recall))
        final_recall_min = float(final_stage_row.get("stage_oracle_recall_min", float("nan")))
        final_june_recall = float(final_stage_row.get("stage_june_oracle_recall", float("nan")))

        learnability_pass = _bool_or_none(cal_row.get("learnability_pass"))
        if learnability_pass is None:
            learnability_pass = bool(math.isfinite(score_ic_u) and score_ic_u > float(thresholds["min_score_ic_u"]))
        bad_mae_calibration_pass = _bool_or_none(cal_row.get("bad_mae_calibration_pass"))
        if bad_mae_calibration_pass is None:
            bad_mae_calibration_pass = False
        economic_utility_pass = bool(
            math.isfinite(mean_u)
            and mean_u > float(thresholds["min_mean_u"])
            and positive_month_rate >= float(thresholds["min_positive_month_rate"])
        )
        tail_risk_pass = bool(
            math.isfinite(bad_mae_rate)
            and bad_mae_rate <= float(thresholds["max_bad_mae_1r_rate"])
            and math.isfinite(timeout_rate)
            and timeout_rate <= float(thresholds["max_timeout_rate"])
            and (
                not math.isfinite(weekly_q10)
                or weekly_q10 >= float(thresholds["min_weekly_mean_u_q10"])
            )
        )
        monthly_tail_risk_pass = None
        if (
            math.isfinite(max_month_bad_mae_rate)
            or math.isfinite(max_month_timeout_rate)
            or math.isfinite(worst_month_weekly_q10)
        ):
            monthly_tail_risk_pass = bool(
                math.isfinite(max_month_bad_mae_rate)
                and max_month_bad_mae_rate <= float(thresholds["max_month_bad_mae_1r_rate"])
                and math.isfinite(max_month_timeout_rate)
                and max_month_timeout_rate <= float(thresholds["max_month_timeout_rate"])
                and (
                    not math.isfinite(worst_month_weekly_q10)
                    or worst_month_weekly_q10 >= float(thresholds["min_worst_month_weekly_mean_u_q10"])
                )
            )
        cost_robust_pass = bool(net_utility_source and math.isfinite(mean_u) and mean_u > 0.0)
        temporal_stability_pass = bool(
            positive_month_rate >= float(thresholds["min_positive_month_rate"])
            and math.isfinite(worst_month_mean_u)
            and worst_month_mean_u >= float(thresholds["min_worst_month_mean_u"])
        )
        max_side_share = max(selected_long_share, selected_short_share)
        side_exposure_pass = bool(
            math.isfinite(max_side_share)
            and max_side_share <= float(thresholds["max_side_share"]) + 1e-9
        )
        minimum_exposure_pass = bool(
            math.isfinite(selected_rows)
            and selected_rows >= float(thresholds["min_selected_rows"])
        )
        candidate_discovery_pass = None
        if math.isfinite(stage_a_recall):
            candidate_discovery_pass = bool(stage_a_recall >= float(thresholds["min_stage_a_oracle_recall"]))
        final_oracle_recall_pass = bool(
            math.isfinite(final_recall)
            and final_recall >= float(thresholds["min_final_oracle_recall"])
        )
        final_oracle_recall_stability_pass = None
        if math.isfinite(final_recall_min):
            final_oracle_recall_stability_pass = bool(
                final_recall_min >= float(thresholds["min_month_final_oracle_recall"])
            )
        hard_gates = [
            ("candidate_discovery", candidate_discovery_pass),
            ("learnability", learnability_pass),
            ("economic_utility", economic_utility_pass),
            ("tail_risk", tail_risk_pass),
            ("monthly_tail_risk", monthly_tail_risk_pass),
            ("bad_mae_calibration", bad_mae_calibration_pass),
            ("cost_robustness", cost_robust_pass),
            ("temporal_stability", temporal_stability_pass),
            ("side_exposure", side_exposure_pass),
            ("minimum_exposure", minimum_exposure_pass),
            ("final_oracle_recall", final_oracle_recall_pass),
            ("final_oracle_recall_stability", final_oracle_recall_stability_pass),
        ]
        scored_gates = [passed for _, passed in hard_gates if passed is not None]
        label_viability_score = 100.0 * float(sum(bool(v) for v in scored_gates)) / max(len(scored_gates), 1)
        active_label = bool(scored_gates and all(bool(v) for v in scored_gates))
        rows.append(
            {
                **item,
                "evaluation_utility_source": str(evaluation_utility_source),
                "calibration_selector": calibration_selector,
                "positive_month_rate": positive_month_rate,
                "max_selected_side_share": max_side_share,
                "stage_a_oracle_recall_mean": stage_a_recall,
                "stage_a_june_oracle_recall": stage_a_june_recall,
                "final_stage_oracle_recall_mean": final_recall,
                "final_stage_oracle_recall_min": final_recall_min,
                "final_stage_june_oracle_recall": final_june_recall,
                "max_month_bad_mae_1r_rate": max_month_bad_mae_rate,
                "max_month_timeout_rate": max_month_timeout_rate,
                "worst_month_weekly_mean_u_q10": worst_month_weekly_q10,
                "calibration_group_count": int(cal_row.get("calibration_group_count", 0) or 0),
                "utility_quantile_spearman_mean": float(
                    cal_row.get("utility_quantile_spearman_mean", float("nan"))
                ),
                "utility_top_bottom_u_mean": float(cal_row.get("utility_top_bottom_u_mean", float("nan"))),
                "utility_monotonic_rate": float(cal_row.get("utility_monotonic_rate", float("nan"))),
                "bad_mae_quantile_spearman_mean": float(
                    cal_row.get("bad_mae_quantile_spearman_mean", float("nan"))
                ),
                "bad_mae_bottom_top_improvement_mean": float(
                    cal_row.get("bad_mae_bottom_top_improvement_mean", float("nan"))
                ),
                "bad_mae_improves_rate": float(cal_row.get("bad_mae_improves_rate", float("nan"))),
                "candidate_discovery_pass": candidate_discovery_pass,
                "learnability_pass": learnability_pass,
                "economic_utility_pass": economic_utility_pass,
                "tail_risk_pass": tail_risk_pass,
                "monthly_tail_risk_pass": monthly_tail_risk_pass,
                "bad_mae_calibration_pass": bad_mae_calibration_pass,
                "cost_robust_pass": cost_robust_pass,
                "temporal_stability_pass": temporal_stability_pass,
                "side_exposure_pass": side_exposure_pass,
                "minimum_exposure_pass": minimum_exposure_pass,
                "final_oracle_recall_pass": final_oracle_recall_pass,
                "final_oracle_recall_stability_pass": final_oracle_recall_stability_pass,
                "label_viability_score": label_viability_score,
                "active_label": active_label,
                "first_failed_gate": _first_failed_gate(hard_gates),
            }
        )
    return pd.DataFrame(rows)


def _build_train_meta_readiness_matrix(
    viability_matrix: pd.DataFrame,
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    evaluation_utility_source: str,
) -> pd.DataFrame:
    columns = [
        "readiness_status",
        "is_final_promotion_ready",
        "label_arm",
        "weight_arm",
        "cluster_policy",
        "top_frac",
        "score_column",
        "calibration_selector",
        "evaluation_utility_source",
        "mean_u",
        "worst_month_mean_u",
        "bad_mae_1r_rate",
        "max_month_bad_mae_1r_rate",
        "timeout_rate",
        "max_month_timeout_rate",
        "final_stage_oracle_recall_mean",
        "final_stage_oracle_recall_min",
        "selected_rows",
        "max_selected_side_share",
        "label_viability_score",
        "labels_path",
        "feature_dir",
        "feature_list_csv",
        "required_next_checks",
    ]
    if viability_matrix.empty or "active_label" not in viability_matrix.columns:
        return pd.DataFrame(columns=columns)
    active = viability_matrix[viability_matrix["active_label"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    next_checks = (
        "train_base_oof_learnability;"
        "train_meta_oos_profitability;"
        "simple_policy_optimiser_exit_policy;"
        "frozen_threshold_replay;"
        "leakage_and_feature_parity_audit"
    )
    for item in active.to_dict(orient="records"):
        selector = str(item.get("cluster_policy", ""))
        rows.append(
            {
                "readiness_status": "candidate_for_train_base_meta_smoke",
                "is_final_promotion_ready": False,
                "label_arm": str(item.get("label_arm", "")),
                "weight_arm": str(item.get("weight_arm", "")),
                "cluster_policy": selector,
                "top_frac": float(item.get("top_frac", float("nan"))),
                "score_column": selector,
                "calibration_selector": str(item.get("calibration_selector", "")),
                "evaluation_utility_source": str(evaluation_utility_source),
                "mean_u": float(item.get("mean_u", float("nan"))),
                "worst_month_mean_u": float(item.get("worst_month_mean_u", float("nan"))),
                "bad_mae_1r_rate": float(item.get("bad_mae_1r_rate", float("nan"))),
                "max_month_bad_mae_1r_rate": float(item.get("max_month_bad_mae_1r_rate", float("nan"))),
                "timeout_rate": float(item.get("timeout_rate", float("nan"))),
                "max_month_timeout_rate": float(item.get("max_month_timeout_rate", float("nan"))),
                "final_stage_oracle_recall_mean": float(
                    item.get("final_stage_oracle_recall_mean", float("nan"))
                ),
                "final_stage_oracle_recall_min": float(
                    item.get("final_stage_oracle_recall_min", float("nan"))
                ),
                "selected_rows": float(item.get("selected_rows", float("nan"))),
                "max_selected_side_share": float(item.get("max_selected_side_share", float("nan"))),
                "label_viability_score": float(item.get("label_viability_score", float("nan"))),
                "labels_path": str(labels_path),
                "feature_dir": str(feature_dir),
                "feature_list_csv": str(feature_list_csv),
                "required_next_checks": next_checks,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _stage_gate_diagnostic_rows(
    *,
    month: str,
    label_arm: str,
    weight_arm: str,
    top_frac: float,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    stage_masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    metrics_local = metrics.reset_index(drop=True)
    side = pd.to_numeric(metrics_local["side"], errors="coerce").fillna(1.0)
    oracle_idx = _selection_indices(metrics_local["u_policy_net"], top_frac, "top_frac")
    oracle_mask = np.zeros(len(metrics_local), dtype=bool)
    oracle_mask[oracle_idx] = True
    rows: list[dict[str, Any]] = []
    for stage, raw_mask in stage_masks.items():
        mask = np.asarray(raw_mask, dtype=bool)
        if len(mask) != len(metrics_local):
            raise ValueError(f"stage mask length mismatch for {stage}")
        for group_name, group_mask in (
            ("all", np.ones(len(mask), dtype=bool)),
            ("long", (side > 0.0).to_numpy()),
            ("short", (side < 0.0).to_numpy()),
        ):
            local = mask & group_mask
            local_oracle = oracle_mask & group_mask
            local_metrics = metrics_local.loc[local]
            oracle_rows = int(local_oracle.sum())
            rows.append(
                {
                    "period": month,
                    "label_arm": label_arm,
                    "weight_arm": weight_arm,
                    "top_frac": float(top_frac),
                    "stage": stage,
                    "side_name": group_name,
                    "rows": int(local.sum()),
                    "row_share": float(local.mean()) if len(local) else float("nan"),
                    "oracle_rows": oracle_rows,
                    "oracle_hit_rows": int((local & local_oracle).sum()),
                    "oracle_recall": float((local & local_oracle).sum() / oracle_rows)
                    if oracle_rows
                    else float("nan"),
                    "mean_u": _safe_mean(local_metrics["u_policy_net"]),
                    "q10_u": _safe_quantile(local_metrics["u_policy_net"], 0.10),
                    "bad_mae_1r_rate": _safe_mean(local_metrics["mae_norm"] >= 1.0),
                    "timeout_rate": _safe_mean(local_metrics["is_timeout"].astype(float) > 0.5),
                    "lower_tail_rate": _safe_mean(
                        local_metrics["u_policy_net"]
                        <= _safe_quantile(metrics_local["u_policy_net"], 0.10)
                    ),
                }
            )
    return rows


def _cluster_stats(
    *,
    clusters: np.ndarray,
    metrics: pd.DataFrame,
    prefix: str,
    min_side_rows: int,
) -> pd.DataFrame:
    clusters_i = np.asarray(clusters, dtype=np.int32)
    if len(clusters_i) == 0:
        return pd.DataFrame()
    max_cluster = int(np.nanmax(clusters_i))
    u = pd.to_numeric(metrics["u_policy_net"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    side = pd.to_numeric(metrics["side"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32, copy=False)
    mae_norm = pd.to_numeric(metrics["mae_norm"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    timeout = pd.Series(metrics["is_timeout"]).astype(float).to_numpy(dtype=np.float32, copy=False)
    (
        rows_arr,
        finite_u,
        sum_u,
        hit_u,
        long_rows_arr,
        short_rows_arr,
        long_sum_u,
        short_sum_u,
        long_finite_u,
        short_finite_u,
        bad_mae,
        finite_mae,
        timeout_count,
        finite_timeout,
    ) = _cluster_core_stats_numba(clusters_i, u, side, mae_norm, timeout, max_cluster)
    rows: list[dict[str, Any]] = []
    for cluster in range(max_cluster + 1):
        row_count = int(rows_arr[cluster])
        if row_count <= 0:
            continue
        mask_np = clusters_i == cluster
        mean_u = float(sum_u[cluster] / finite_u[cluster]) if finite_u[cluster] else float("nan")
        hit = float(hit_u[cluster] / finite_u[cluster]) if finite_u[cluster] else float("nan")
        long_count = int(long_rows_arr[cluster])
        short_count = int(short_rows_arr[cluster])
        long_mean = (
            float(long_sum_u[cluster] / long_finite_u[cluster])
            if long_finite_u[cluster]
            else float("nan")
        )
        short_mean = (
            float(short_sum_u[cluster] / short_finite_u[cluster])
            if short_finite_u[cluster]
            else float("nan")
        )
        q10 = float(np.nanquantile(u[mask_np], 0.10)) if np.isfinite(u[mask_np]).any() else float("nan")
        min_side_mean = np.nanmin([long_mean, short_mean])
        cancellation = bool(
            math.isfinite(long_mean)
            and math.isfinite(short_mean)
            and (
                (long_mean > 0.0 and short_mean < -0.00025)
                or (short_mean > 0.0 and long_mean < -0.00025)
                or (mean_u > 0.0 and math.isfinite(min_side_mean) and min_side_mean < -0.0005)
            )
        )
        rows.append(
            {
                "cluster": int(cluster),
                f"{prefix}_rows": row_count,
                f"{prefix}_share": float(row_count / max(len(clusters_i), 1)),
                f"{prefix}_mean_u": mean_u,
                f"{prefix}_hit_u": hit,
                f"{prefix}_q10_u": q10,
                f"{prefix}_long_rows": long_count,
                f"{prefix}_short_rows": short_count,
                f"{prefix}_long_share": float(long_count / max(row_count, 1)),
                f"{prefix}_short_share": float(short_count / max(row_count, 1)),
                f"{prefix}_long_mean_u": long_mean,
                f"{prefix}_short_mean_u": short_mean,
                f"{prefix}_min_side_mean_u": float(min_side_mean) if math.isfinite(min_side_mean) else float("nan"),
                f"{prefix}_side_coverage_ok": bool(long_count >= min_side_rows and short_count >= min_side_rows),
                f"{prefix}_side_cancellation_flag": cancellation,
                f"{prefix}_bad_mae_1r_rate": (
                    float(bad_mae[cluster] / finite_mae[cluster])
                    if finite_mae[cluster]
                    else float("nan")
                ),
                f"{prefix}_timeout_rate": (
                    float(timeout_count[cluster] / finite_timeout[cluster])
                    if finite_timeout[cluster]
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _cluster_side_stats(
    *,
    clusters: np.ndarray,
    metrics: pd.DataFrame,
    prefix: str,
    min_side_rows: int,
) -> pd.DataFrame:
    clusters_i = np.asarray(clusters, dtype=np.int32)
    if len(clusters_i) == 0:
        return pd.DataFrame()
    max_cluster = int(np.nanmax(clusters_i))
    u = pd.to_numeric(metrics["u_policy_net"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    side = pd.to_numeric(metrics["side"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32, copy=False)
    mae_norm = pd.to_numeric(metrics["mae_norm"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    timeout = pd.Series(metrics["is_timeout"]).astype(float).to_numpy(dtype=np.float32, copy=False)
    total_rows = max(len(clusters_i), 1)
    rows: list[dict[str, Any]] = []
    for cluster in range(max_cluster + 1):
        cluster_mask = clusters_i == cluster
        if not bool(cluster_mask.any()):
            continue
        for side_name, side_value, side_mask in (
            ("long", 1, side > 0.0),
            ("short", -1, side < 0.0),
        ):
            mask = cluster_mask & side_mask
            row_count = int(mask.sum())
            if row_count <= 0:
                continue
            u_local = u[mask]
            finite_u = np.isfinite(u_local)
            mae_local = mae_norm[mask]
            finite_mae = np.isfinite(mae_local)
            timeout_local = timeout[mask]
            finite_timeout = np.isfinite(timeout_local)
            rows.append(
                {
                    "cluster": int(cluster),
                    "side_name": side_name,
                    "side": int(side_value),
                    f"{prefix}_side_rows": row_count,
                    f"{prefix}_side_share": float(row_count / total_rows),
                    f"{prefix}_side_mean_u": float(np.mean(u_local[finite_u])) if bool(finite_u.any()) else float("nan"),
                    f"{prefix}_side_hit_u": float(np.mean(u_local[finite_u] > 0.0)) if bool(finite_u.any()) else float("nan"),
                    f"{prefix}_side_q10_u": float(np.quantile(u_local[finite_u], 0.10)) if bool(finite_u.any()) else float("nan"),
                    f"{prefix}_side_coverage_ok": bool(row_count >= int(min_side_rows)),
                    f"{prefix}_side_bad_mae_1r_rate": (
                        float(np.mean(mae_local[finite_mae] >= 1.0))
                        if bool(finite_mae.any())
                        else float("nan")
                    ),
                    f"{prefix}_side_timeout_rate": (
                        float(np.mean(timeout_local[finite_timeout] > 0.5))
                        if bool(finite_timeout.any())
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _assign_cluster_policy(
    train_stats: pd.DataFrame,
    *,
    min_allow_mean_u: float,
    min_allow_hit_u: float,
    block_mean_u: float,
    min_side_mean_u: float,
) -> pd.DataFrame:
    if train_stats.empty:
        return pd.DataFrame(columns=["cluster", "cluster_policy_action"])
    rows: list[dict[str, Any]] = []
    for row in train_stats.to_dict(orient="records"):
        mean_u = float(row.get("train_mean_u", float("nan")))
        hit_u = float(row.get("train_hit_u", float("nan")))
        q10_u = float(row.get("train_q10_u", float("nan")))
        min_side = float(row.get("train_min_side_mean_u", float("nan")))
        side_ok = bool(row.get("train_side_coverage_ok", False))
        cancellation = bool(row.get("train_side_cancellation_flag", False))
        if not side_ok:
            action = "block"
            reason = "insufficient_long_short_coverage"
            multiplier = 0.0
        elif math.isfinite(mean_u) and mean_u <= float(block_mean_u):
            action = "block"
            reason = "negative_train_cluster_utility"
            multiplier = 0.0
        elif cancellation and math.isfinite(min_side) and min_side < float(min_side_mean_u):
            action = "throttle"
            reason = "long_short_cancellation"
            multiplier = 0.25
        elif (
            math.isfinite(mean_u)
            and mean_u >= float(min_allow_mean_u)
            and math.isfinite(hit_u)
            and hit_u >= float(min_allow_hit_u)
            and (not math.isfinite(min_side) or min_side >= float(min_side_mean_u))
        ):
            action = "allow_normal"
            reason = "positive_prior_cluster"
            multiplier = 1.0
        elif math.isfinite(mean_u) and mean_u > 0.0:
            action = "allow_high_threshold"
            reason = "weak_positive_prior_cluster"
            multiplier = 0.50
        else:
            action = "throttle"
            reason = "weak_or_unproven_prior_cluster"
            multiplier = 0.25
        out = dict(row)
        out.update(
            {
                "cluster_policy_action": action,
                "cluster_policy_reason": reason,
                "cluster_threshold_multiplier": float(multiplier),
                "cluster_policy_adjustment": float(np.clip(mean_u / 0.005, -1.0, 1.0))
                if math.isfinite(mean_u)
                else 0.0,
                "train_q10_u_for_policy": q10_u,
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def _assign_cluster_side_policy(
    train_side_stats: pd.DataFrame,
    cluster_policy: pd.DataFrame,
    *,
    min_allow_mean_u: float,
    min_allow_hit_u: float,
    block_mean_u: float,
    min_side_rows: int,
) -> pd.DataFrame:
    if train_side_stats.empty:
        return pd.DataFrame(columns=["cluster", "side_name", "cluster_side_policy_action"])
    cluster_policy_index = cluster_policy.set_index("cluster") if not cluster_policy.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for row in train_side_stats.to_dict(orient="records"):
        cluster = int(row.get("cluster", -1))
        side_name = str(row.get("side_name", ""))
        side_rows = int(row.get("train_side_rows", 0) or 0)
        mean_u = float(row.get("train_side_mean_u", float("nan")))
        hit_u = float(row.get("train_side_hit_u", float("nan")))
        q10_u = float(row.get("train_side_q10_u", float("nan")))
        bad_mae = float(row.get("train_side_bad_mae_1r_rate", float("nan")))
        parent_action = ""
        parent_reason = ""
        if not cluster_policy_index.empty and cluster in cluster_policy_index.index:
            parent_action = str(cluster_policy_index.loc[cluster].get("cluster_policy_action", ""))
            parent_reason = str(cluster_policy_index.loc[cluster].get("cluster_policy_reason", ""))

        if side_rows < int(min_side_rows):
            action = "block"
            reason = "insufficient_cluster_side_rows"
            multiplier = 0.0
        elif math.isfinite(mean_u) and mean_u <= float(block_mean_u):
            action = "block"
            reason = "negative_cluster_side_utility"
            multiplier = 0.0
        elif (
            math.isfinite(mean_u)
            and mean_u >= float(min_allow_mean_u)
            and math.isfinite(hit_u)
            and hit_u >= float(min_allow_hit_u)
        ):
            action = "allow_normal"
            reason = "positive_cluster_side"
            multiplier = 1.0
        elif math.isfinite(mean_u) and mean_u > 0.0:
            action = "allow_high_threshold"
            reason = "weak_positive_cluster_side"
            multiplier = 0.50
        else:
            action = "throttle"
            reason = "weak_or_unproven_cluster_side"
            multiplier = 0.25

        # Keep very path-risk-heavy sides from being treated as normal even when mean utility is positive.
        if action == "allow_normal" and math.isfinite(bad_mae) and bad_mae >= 0.75:
            action = "allow_high_threshold"
            reason = f"{reason}_high_bad_mae"
            multiplier = min(multiplier, 0.50)

        out = dict(row)
        out.update(
            {
                "cluster_side_policy_action": action,
                "cluster_side_policy_reason": reason,
                "cluster_side_threshold_multiplier": float(multiplier),
                "cluster_side_policy_adjustment": (
                    float(np.clip(mean_u / 0.005, -1.0, 1.0)) if math.isfinite(mean_u) else 0.0
                ),
                "train_side_q10_u_for_policy": q10_u,
                "parent_cluster_policy_action": parent_action,
                "parent_cluster_policy_reason": parent_reason,
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def _apply_cluster_policy_score(
    score: pd.Series,
    clusters: np.ndarray,
    policy: pd.DataFrame,
    *,
    top_frac: float,
) -> tuple[pd.Series, pd.Series]:
    raw = pd.to_numeric(score.reset_index(drop=True), errors="coerce").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    actions = policy.set_index("cluster").to_dict(orient="index") if not policy.empty else {}
    finite = raw[np.isfinite(raw)]
    index = pd.RangeIndex(len(raw))
    if finite.size == 0:
        return pd.Series(np.nan, index=index, dtype=np.float32), pd.Series(False, index=index)
    thresholds = {
        "allow_normal": float("-inf"),
        "allow_high_threshold": float(np.quantile(finite, max(0.0, 1.0 - float(top_frac) * 0.50))),
        "throttle": float(np.quantile(finite, max(0.0, 1.0 - float(top_frac) * 0.25))),
        "block": float("inf"),
    }
    max_cluster = max(
        int(np.max(clusters)) if len(clusters) else -1,
        max((int(c) for c in actions.keys()), default=-1),
    )
    action_codes = np.zeros(max_cluster + 1, dtype=np.int8)
    adjustments = np.zeros(max_cluster + 1, dtype=np.float32)
    action_map = {
        "block": _POLICY_BLOCK,
        "throttle": _POLICY_THROTTLE,
        "allow_high_threshold": _POLICY_HIGH_THRESHOLD,
        "allow_normal": _POLICY_NORMAL,
    }
    for cluster, item in actions.items():
        c = int(cluster)
        if c < 0 or c >= len(action_codes):
            continue
        item = actions.get(int(cluster), {})
        action = str(item.get("cluster_policy_action", "block"))
        action_codes[c] = action_map.get(action, _POLICY_BLOCK)
        adjustments[c] = np.float32(float(item.get("cluster_policy_adjustment", 0.0) or 0.0))
    adjusted, eligible = _apply_cluster_policy_score_numba(
        raw,
        np.asarray(clusters, dtype=np.int32),
        action_codes,
        adjustments,
        float(thresholds["allow_high_threshold"]),
        float(thresholds["throttle"]),
    )
    return pd.Series(adjusted, index=index, dtype=np.float32), pd.Series(eligible, index=index)


def _apply_cluster_side_policy_score(
    score: pd.Series,
    clusters: np.ndarray,
    side: pd.Series,
    side_policy: pd.DataFrame,
    *,
    top_frac: float,
) -> tuple[pd.Series, pd.Series]:
    raw = pd.to_numeric(score.reset_index(drop=True), errors="coerce").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    side_arr = pd.to_numeric(side.reset_index(drop=True), errors="coerce").fillna(1.0).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    finite = raw[np.isfinite(raw)]
    index = pd.RangeIndex(len(raw))
    if finite.size == 0:
        return pd.Series(np.nan, index=index, dtype=np.float32), pd.Series(False, index=index)
    thresholds = {
        "allow_normal": float("-inf"),
        "allow_high_threshold": float(np.quantile(finite, max(0.0, 1.0 - float(top_frac) * 0.50))),
        "throttle": float(np.quantile(finite, max(0.0, 1.0 - float(top_frac) * 0.25))),
        "block": float("inf"),
    }
    max_cluster = int(np.max(clusters)) if len(clusters) else -1
    if not side_policy.empty and "cluster" in side_policy.columns:
        max_cluster = max(max_cluster, int(pd.to_numeric(side_policy["cluster"], errors="coerce").max()))
    action_codes = np.zeros((max_cluster + 1, 2), dtype=np.int8)
    adjustments = np.zeros((max_cluster + 1, 2), dtype=np.float32)
    action_map = {
        "block": _POLICY_BLOCK,
        "throttle": _POLICY_THROTTLE,
        "allow_high_threshold": _POLICY_HIGH_THRESHOLD,
        "allow_normal": _POLICY_NORMAL,
    }
    if not side_policy.empty:
        for item in side_policy.to_dict(orient="records"):
            c = int(item.get("cluster", -1))
            side_idx = 1 if int(item.get("side", 1) or 1) < 0 else 0
            if c < 0 or c >= len(action_codes):
                continue
            action = str(item.get("cluster_side_policy_action", "block"))
            action_codes[c, side_idx] = action_map.get(action, _POLICY_BLOCK)
            adjustments[c, side_idx] = np.float32(float(item.get("cluster_side_policy_adjustment", 0.0) or 0.0))
    adjusted, eligible = _apply_cluster_side_policy_score_numba(
        raw,
        np.asarray(clusters, dtype=np.int32),
        side_arr,
        action_codes,
        adjustments,
        float(thresholds["allow_high_threshold"]),
        float(thresholds["throttle"]),
    )
    return pd.Series(adjusted, index=index, dtype=np.float32), pd.Series(eligible, index=index)


def _selected_cluster_rows(
    *,
    month: str,
    label_arm: str,
    weight_arm: str,
    top_frac: float,
    selector: str,
    score: pd.Series,
    clusters: np.ndarray,
    metrics: pd.DataFrame,
    policy: pd.DataFrame,
    side_policy: pd.DataFrame | None = None,
    selection_mode: str = "top_frac",
) -> list[dict[str, Any]]:
    idx = _selection_indices(score, top_frac, selection_mode)
    if len(idx) == 0:
        return []
    selected_clusters = clusters[idx]
    selected_metrics = metrics.iloc[idx].reset_index(drop=True)
    selected_policy = policy.set_index("cluster") if not policy.empty else pd.DataFrame()
    selected_side_policy = (
        side_policy.set_index(["cluster", "side_name"])
        if side_policy is not None and not side_policy.empty
        else pd.DataFrame()
    )
    rows: list[dict[str, Any]] = []
    for cluster in sorted(set(int(v) for v in selected_clusters)):
        local_idx = np.flatnonzero(selected_clusters == cluster)
        local_metrics = selected_metrics.iloc[local_idx]
        action = ""
        reason = ""
        if not selected_policy.empty and cluster in selected_policy.index:
            action = str(selected_policy.loc[cluster].get("cluster_policy_action", ""))
            reason = str(selected_policy.loc[cluster].get("cluster_policy_reason", ""))
        side = pd.to_numeric(local_metrics["side"], errors="coerce").fillna(1.0)
        long_rows, short_rows = _side_counts(side)
        long_action = ""
        long_reason = ""
        short_action = ""
        short_reason = ""
        if not selected_side_policy.empty:
            if (cluster, "long") in selected_side_policy.index:
                long_action = str(selected_side_policy.loc[(cluster, "long")].get("cluster_side_policy_action", ""))
                long_reason = str(selected_side_policy.loc[(cluster, "long")].get("cluster_side_policy_reason", ""))
            if (cluster, "short") in selected_side_policy.index:
                short_action = str(selected_side_policy.loc[(cluster, "short")].get("cluster_side_policy_action", ""))
                short_reason = str(selected_side_policy.loc[(cluster, "short")].get("cluster_side_policy_reason", ""))
        rows.append(
            {
                "period": month,
                "label_arm": label_arm,
                "weight_arm": weight_arm,
                "top_frac": float(top_frac),
                "selector": selector,
                "cluster": int(cluster),
                "cluster_policy_action": action,
                "cluster_policy_reason": reason,
                "cluster_side_policy_long_action": long_action,
                "cluster_side_policy_long_reason": long_reason,
                "cluster_side_policy_short_action": short_action,
                "cluster_side_policy_short_reason": short_reason,
                "selected_rows": int(len(local_idx)),
                "selected_share": float(len(local_idx) / max(len(idx), 1)),
                "selected_mean_u": _safe_mean(local_metrics["u_policy_net"]),
                "selected_hit_u": _safe_mean(local_metrics["u_policy_net"] > 0.0),
                "selected_q10_u": _safe_quantile(local_metrics["u_policy_net"], 0.10),
                "selected_long_rows": long_rows,
                "selected_short_rows": short_rows,
                "selected_long_mean_u": _safe_mean(local_metrics.loc[side > 0, "u_policy_net"]),
                "selected_short_mean_u": _safe_mean(local_metrics.loc[side < 0, "u_policy_net"]),
                "selected_bad_mae_1r_rate": _safe_mean(local_metrics["mae_norm"] >= 1.0),
                "selected_timeout_rate": _safe_mean(local_metrics["is_timeout"].astype(float) > 0.5),
            }
        )
    return rows


def _downcast_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy(deep=False)
    for col in out.columns:
        if col == "side":
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(1.0).astype(np.int8)
        elif pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(bool, copy=False)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32, copy=False)
    return out


def _june_diagnosis(cluster_selection: pd.DataFrame, monthly: pd.DataFrame) -> dict[str, Any]:
    june = cluster_selection[cluster_selection["period"].eq("2026-06")].copy()
    monthly_june = monthly[monthly["period"].eq("2026-06")].copy()
    out: dict[str, Any] = {"enabled": bool(not june.empty), "period": "2026-06"}
    if june.empty:
        out["reason"] = "missing_june_cluster_rows"
        return out
    raw = june[june["selector"].eq("raw_model_score")]
    gated = june[june["selector"].eq("cluster_policy_score")]
    raw_selected = float(pd.to_numeric(raw["selected_rows"], errors="coerce").sum())
    bad_actions = {"block", "throttle"}
    weak_actions = {"block", "throttle", "allow_high_threshold"}
    raw_bad_rows = float(pd.to_numeric(raw[raw["cluster_policy_action"].isin(bad_actions)]["selected_rows"], errors="coerce").sum())
    raw_weak_rows = float(pd.to_numeric(raw[raw["cluster_policy_action"].isin(weak_actions)]["selected_rows"], errors="coerce").sum())
    raw_good = raw[raw["cluster_policy_action"].eq("allow_normal")]
    raw_good_rows = float(pd.to_numeric(raw_good["selected_rows"], errors="coerce").sum())
    raw_good_mean = (
        float(np.average(raw_good["selected_mean_u"], weights=raw_good["selected_rows"]))
        if raw_good_rows > 0
        else float("nan")
    )
    raw_bad_share = raw_bad_rows / max(raw_selected, 1.0)
    raw_weak_share = raw_weak_rows / max(raw_selected, 1.0)
    raw_good_share = raw_good_rows / max(raw_selected, 1.0)
    raw_month = monthly_june[monthly_june["selector"].eq("raw_model_score")]
    raw_side_balanced_month = monthly_june[monthly_june["selector"].eq("raw_side_balanced_score")]
    risk_month = monthly_june[monthly_june["selector"].eq("risk_adjusted_score")]
    risk_side_balanced_month = monthly_june[monthly_june["selector"].eq("risk_adjusted_side_balanced_score")]
    side_calibrated_month = monthly_june[monthly_june["selector"].eq("side_calibrated_score")]
    side_calibrated_balanced_month = monthly_june[
        monthly_june["selector"].eq("side_calibrated_side_balanced_score")
    ]
    soft_risk_month = monthly_june[monthly_june["selector"].eq("side_calibrated_soft_risk_score")]
    soft_risk_balanced_month = monthly_june[
        monthly_june["selector"].eq("side_calibrated_soft_risk_side_balanced_score")
    ]
    threshold_month = monthly_june[monthly_june["selector"].eq("side_calibrated_soft_risk_threshold_score")]
    threshold_balanced_month = monthly_june[
        monthly_june["selector"].eq("side_calibrated_soft_risk_threshold_side_balanced_score")
    ]
    risk_cap_month = monthly_june[
        monthly_june["selector"].eq("side_calibrated_soft_risk_bad_mae_cap_score")
    ]
    risk_cap_side_capped_month = monthly_june[
        monthly_june["selector"].eq("side_calibrated_soft_risk_bad_mae_cap_side_capped_score")
    ]
    threshold_select_month = monthly_june[
        monthly_june["selector"].eq("side_calibrated_soft_risk_bad_mae_cap_threshold_select_score")
    ]
    ranker_month = monthly_june[
        monthly_june["selector"].eq("lgbm_ranker_side_calibrated_risk_cap_score")
    ]
    s7a_month = monthly_june[monthly_june["selector"].eq("s7a_lgbm_ranker_no_prefilter_score")]
    s7b_month = monthly_june[monthly_june["selector"].eq("s7b_lgbm_ranker_relaxed_risk_cap_score")]
    s7c_month = monthly_june[monthly_june["selector"].eq("s7c_side_specific_ranker_risk_cap_score")]
    s7d_month = monthly_june[monthly_june["selector"].eq("s7d_oracle_enriched_ranker_risk_cap_score")]
    s7_two_stage_month = monthly_june[monthly_june["selector"].eq("s7_two_stage_candidate_rerank_score")]
    s8d_month = monthly_june[monthly_june["selector"].eq("s8d_oracle_enriched_ranker_tight_bad_mae_score")]
    s8_two_stage_month = monthly_june[monthly_june["selector"].eq("s8_two_stage_tight_bad_mae_score")]
    s9d_month = monthly_june[monthly_june["selector"].eq("s9d_oracle_enriched_ranker_calibrated_risk_cap_score")]
    s9_two_stage_month = monthly_june[monthly_june["selector"].eq("s9_two_stage_calibrated_risk_cap_score")]
    s10_soft_month = monthly_june[monthly_june["selector"].eq("s10_recall_soft_calibrated_risk_score")]
    s10_loose_month = monthly_june[
        monthly_june["selector"].eq("s10_recall_soft_calibrated_risk_loose_cap_score")
    ]
    s11_month = monthly_june[monthly_june["selector"].eq("s11_recall_tail_balanced_score")]
    s12_month = monthly_june[monthly_june["selector"].eq("s12_path_quality_ranker_score")]
    s12_soft_month = monthly_june[monthly_june["selector"].eq("s12_path_quality_ranker_soft_risk_score")]
    s13_month = monthly_june[monthly_june["selector"].eq("s13_constrained_path_quality_score")]
    s13_soft_month = monthly_june[
        monthly_june["selector"].eq("s13_constrained_path_quality_soft_risk_score")
    ]
    s14_month = monthly_june[monthly_june["selector"].eq("s14_path_quality_risk_trim_score")]
    s14_soft_month = monthly_june[monthly_june["selector"].eq("s14_path_quality_soft_risk_trim_score")]
    s15_month = monthly_june[monthly_june["selector"].eq("s15_side_path_quality_ranker_score")]
    s15_trim_month = monthly_june[monthly_june["selector"].eq("s15_side_path_quality_risk_trim_score")]
    s16_month = monthly_june[monthly_june["selector"].eq("s16_discovery_path_quality_blend_score")]
    s16_trim_month = monthly_june[
        monthly_june["selector"].eq("s16_discovery_path_quality_risk_trim_score")
    ]
    gated_month = monthly_june[monthly_june["selector"].eq("cluster_policy_score")]
    side_gated_month = monthly_june[monthly_june["selector"].eq("cluster_side_policy_score")]
    side_gated_balanced_month = monthly_june[
        monthly_june["selector"].eq("cluster_side_policy_side_balanced_score")
    ]
    cluster_side_risk_month = monthly_june[monthly_june["selector"].eq("cluster_side_risk_adjusted_score")]
    cluster_side_risk_balanced_month = monthly_june[
        monthly_june["selector"].eq("cluster_side_risk_adjusted_side_balanced_score")
    ]
    raw_mean = _safe_mean(raw_month["mean_u"]) if not raw_month.empty else float("nan")
    raw_side_balanced_mean = (
        _safe_mean(raw_side_balanced_month["mean_u"]) if not raw_side_balanced_month.empty else float("nan")
    )
    risk_mean = _safe_mean(risk_month["mean_u"]) if not risk_month.empty else float("nan")
    risk_side_balanced_mean = (
        _safe_mean(risk_side_balanced_month["mean_u"]) if not risk_side_balanced_month.empty else float("nan")
    )
    side_calibrated_mean = (
        _safe_mean(side_calibrated_month["mean_u"]) if not side_calibrated_month.empty else float("nan")
    )
    side_calibrated_balanced_mean = (
        _safe_mean(side_calibrated_balanced_month["mean_u"])
        if not side_calibrated_balanced_month.empty
        else float("nan")
    )
    soft_risk_mean = _safe_mean(soft_risk_month["mean_u"]) if not soft_risk_month.empty else float("nan")
    soft_risk_balanced_mean = (
        _safe_mean(soft_risk_balanced_month["mean_u"]) if not soft_risk_balanced_month.empty else float("nan")
    )
    threshold_mean = _safe_mean(threshold_month["mean_u"]) if not threshold_month.empty else float("nan")
    threshold_balanced_mean = (
        _safe_mean(threshold_balanced_month["mean_u"]) if not threshold_balanced_month.empty else float("nan")
    )
    risk_cap_mean = _safe_mean(risk_cap_month["mean_u"]) if not risk_cap_month.empty else float("nan")
    risk_cap_side_capped_mean = (
        _safe_mean(risk_cap_side_capped_month["mean_u"]) if not risk_cap_side_capped_month.empty else float("nan")
    )
    threshold_select_mean = (
        _safe_mean(threshold_select_month["mean_u"]) if not threshold_select_month.empty else float("nan")
    )
    ranker_mean = _safe_mean(ranker_month["mean_u"]) if not ranker_month.empty else float("nan")
    s7a_mean = _safe_mean(s7a_month["mean_u"]) if not s7a_month.empty else float("nan")
    s7b_mean = _safe_mean(s7b_month["mean_u"]) if not s7b_month.empty else float("nan")
    s7c_mean = _safe_mean(s7c_month["mean_u"]) if not s7c_month.empty else float("nan")
    s7d_mean = _safe_mean(s7d_month["mean_u"]) if not s7d_month.empty else float("nan")
    s7_two_stage_mean = (
        _safe_mean(s7_two_stage_month["mean_u"]) if not s7_two_stage_month.empty else float("nan")
    )
    s8d_mean = _safe_mean(s8d_month["mean_u"]) if not s8d_month.empty else float("nan")
    s8_two_stage_mean = (
        _safe_mean(s8_two_stage_month["mean_u"]) if not s8_two_stage_month.empty else float("nan")
    )
    s9d_mean = _safe_mean(s9d_month["mean_u"]) if not s9d_month.empty else float("nan")
    s9_two_stage_mean = (
        _safe_mean(s9_two_stage_month["mean_u"]) if not s9_two_stage_month.empty else float("nan")
    )
    s10_soft_mean = _safe_mean(s10_soft_month["mean_u"]) if not s10_soft_month.empty else float("nan")
    s10_loose_mean = _safe_mean(s10_loose_month["mean_u"]) if not s10_loose_month.empty else float("nan")
    s11_mean = _safe_mean(s11_month["mean_u"]) if not s11_month.empty else float("nan")
    s12_mean = _safe_mean(s12_month["mean_u"]) if not s12_month.empty else float("nan")
    s12_soft_mean = _safe_mean(s12_soft_month["mean_u"]) if not s12_soft_month.empty else float("nan")
    s13_mean = _safe_mean(s13_month["mean_u"]) if not s13_month.empty else float("nan")
    s13_soft_mean = _safe_mean(s13_soft_month["mean_u"]) if not s13_soft_month.empty else float("nan")
    s14_mean = _safe_mean(s14_month["mean_u"]) if not s14_month.empty else float("nan")
    s14_soft_mean = _safe_mean(s14_soft_month["mean_u"]) if not s14_soft_month.empty else float("nan")
    s15_mean = _safe_mean(s15_month["mean_u"]) if not s15_month.empty else float("nan")
    s15_trim_mean = _safe_mean(s15_trim_month["mean_u"]) if not s15_trim_month.empty else float("nan")
    s16_mean = _safe_mean(s16_month["mean_u"]) if not s16_month.empty else float("nan")
    s16_trim_mean = _safe_mean(s16_trim_month["mean_u"]) if not s16_trim_month.empty else float("nan")
    gated_mean = _safe_mean(gated_month["mean_u"]) if not gated_month.empty else float("nan")
    side_gated_mean = _safe_mean(side_gated_month["mean_u"]) if not side_gated_month.empty else float("nan")
    side_gated_balanced_mean = (
        _safe_mean(side_gated_balanced_month["mean_u"])
        if not side_gated_balanced_month.empty
        else float("nan")
    )
    cluster_side_risk_mean = (
        _safe_mean(cluster_side_risk_month["mean_u"]) if not cluster_side_risk_month.empty else float("nan")
    )
    cluster_side_risk_balanced_mean = (
        _safe_mean(cluster_side_risk_balanced_month["mean_u"])
        if not cluster_side_risk_balanced_month.empty
        else float("nan")
    )
    if raw_bad_share >= 0.50 and (not math.isfinite(raw_good_mean) or raw_good_mean >= 0.0):
        diagnosis = "wrong_cluster_selection_policy_gating_likely"
    elif raw_good_share >= 0.50 and math.isfinite(raw_good_mean) and raw_good_mean < 0.0:
        diagnosis = "prior_good_clusters_decayed_deeper_issue"
    elif (
        math.isfinite(s16_trim_mean)
        and math.isfinite(raw_mean)
        and s16_trim_mean > raw_mean + 0.0005
    ):
        diagnosis = "s16_discovery_path_quality_risk_trim_materially_improves_june"
    elif (
        math.isfinite(s16_mean)
        and math.isfinite(raw_mean)
        and s16_mean > raw_mean + 0.0005
    ):
        diagnosis = "s16_discovery_path_quality_blend_materially_improves_june"
    elif (
        math.isfinite(s15_trim_mean)
        and math.isfinite(raw_mean)
        and s15_trim_mean > raw_mean + 0.0005
    ):
        diagnosis = "s15_side_path_quality_risk_trim_materially_improves_june"
    elif (
        math.isfinite(s15_mean)
        and math.isfinite(raw_mean)
        and s15_mean > raw_mean + 0.0005
    ):
        diagnosis = "s15_side_path_quality_ranker_materially_improves_june"
    elif (
        math.isfinite(s14_mean)
        and math.isfinite(raw_mean)
        and s14_mean > raw_mean + 0.0005
    ):
        diagnosis = "s14_path_quality_risk_trim_materially_improves_june"
    elif (
        math.isfinite(s14_soft_mean)
        and math.isfinite(raw_mean)
        and s14_soft_mean > raw_mean + 0.0005
    ):
        diagnosis = "s14_path_quality_soft_risk_trim_materially_improves_june"
    elif (
        math.isfinite(s13_mean)
        and math.isfinite(raw_mean)
        and s13_mean > raw_mean + 0.0005
    ):
        diagnosis = "s13_constrained_path_quality_materially_improves_june"
    elif (
        math.isfinite(s13_soft_mean)
        and math.isfinite(raw_mean)
        and s13_soft_mean > raw_mean + 0.0005
    ):
        diagnosis = "s13_constrained_path_quality_soft_risk_materially_improves_june"
    elif (
        math.isfinite(s12_soft_mean)
        and math.isfinite(raw_mean)
        and s12_soft_mean > raw_mean + 0.0005
    ):
        diagnosis = "s12_path_quality_soft_risk_materially_improves_june"
    elif (
        math.isfinite(s12_mean)
        and math.isfinite(raw_mean)
        and s12_mean > raw_mean + 0.0005
    ):
        diagnosis = "s12_path_quality_ranker_materially_improves_june"
    elif (
        math.isfinite(s11_mean)
        and math.isfinite(raw_mean)
        and s11_mean > raw_mean + 0.0005
    ):
        diagnosis = "s11_recall_tail_balanced_materially_improves_june"
    elif (
        math.isfinite(s10_loose_mean)
        and math.isfinite(raw_mean)
        and s10_loose_mean > raw_mean + 0.0005
    ):
        diagnosis = "s10_recall_soft_calibrated_risk_loose_cap_materially_improves_june"
    elif (
        math.isfinite(s10_soft_mean)
        and math.isfinite(raw_mean)
        and s10_soft_mean > raw_mean + 0.0005
    ):
        diagnosis = "s10_recall_soft_calibrated_risk_materially_improves_june"
    elif (
        math.isfinite(s9_two_stage_mean)
        and math.isfinite(raw_mean)
        and s9_two_stage_mean > raw_mean + 0.0005
    ):
        diagnosis = "s9_two_stage_calibrated_risk_materially_improves_june"
    elif (
        math.isfinite(s9d_mean)
        and math.isfinite(raw_mean)
        and s9d_mean > raw_mean + 0.0005
    ):
        diagnosis = "s9_oracle_enriched_calibrated_risk_materially_improves_june"
    elif (
        math.isfinite(s8_two_stage_mean)
        and math.isfinite(raw_mean)
        and s8_two_stage_mean > raw_mean + 0.0005
    ):
        diagnosis = "s8_two_stage_tight_tail_risk_materially_improves_june"
    elif (
        math.isfinite(s8d_mean)
        and math.isfinite(raw_mean)
        and s8d_mean > raw_mean + 0.0005
    ):
        diagnosis = "s8_oracle_enriched_tight_tail_risk_materially_improves_june"
    elif (
        math.isfinite(s7_two_stage_mean)
        and math.isfinite(raw_mean)
        and s7_two_stage_mean > raw_mean + 0.0005
    ):
        diagnosis = "s7_two_stage_candidate_rerank_materially_improves_june"
    elif (
        math.isfinite(s7c_mean)
        and math.isfinite(raw_mean)
        and s7c_mean > raw_mean + 0.0005
    ):
        diagnosis = "s7_side_specific_ranker_materially_improves_june"
    elif (
        math.isfinite(s7d_mean)
        and math.isfinite(raw_mean)
        and s7d_mean > raw_mean + 0.0005
    ):
        diagnosis = "s7_oracle_enriched_ranker_materially_improves_june"
    elif (
        math.isfinite(s7b_mean)
        and math.isfinite(raw_mean)
        and s7b_mean > raw_mean + 0.0005
    ):
        diagnosis = "s7_relaxed_risk_ranker_materially_improves_june"
    elif (
        math.isfinite(s7a_mean)
        and math.isfinite(raw_mean)
        and s7a_mean > raw_mean + 0.0005
    ):
        diagnosis = "s7_no_prefilter_ranker_materially_improves_june"
    elif (
        math.isfinite(threshold_select_mean)
        and math.isfinite(raw_mean)
        and threshold_select_mean > raw_mean + 0.0005
    ):
        diagnosis = "risk_cap_threshold_select_materially_improves_june"
    elif (
        math.isfinite(ranker_mean)
        and math.isfinite(raw_mean)
        and ranker_mean > raw_mean + 0.0005
    ):
        diagnosis = "ranker_risk_cap_materially_improves_june"
    elif (
        math.isfinite(risk_cap_side_capped_mean)
        and math.isfinite(raw_mean)
        and risk_cap_side_capped_mean > raw_mean + 0.0005
    ):
        diagnosis = "risk_cap_side_cap_materially_improves_june"
    elif (
        math.isfinite(threshold_balanced_mean)
        and math.isfinite(raw_mean)
        and threshold_balanced_mean > raw_mean + 0.0005
    ):
        diagnosis = "thresholded_soft_risk_side_calibration_materially_improves_june"
    elif (
        math.isfinite(soft_risk_balanced_mean)
        and math.isfinite(raw_mean)
        and soft_risk_balanced_mean > raw_mean + 0.0005
    ):
        diagnosis = "soft_risk_side_calibration_materially_improves_june"
    elif (
        math.isfinite(side_calibrated_balanced_mean)
        and math.isfinite(raw_mean)
        and side_calibrated_balanced_mean > raw_mean + 0.0005
    ):
        diagnosis = "side_calibration_materially_improves_june"
    elif (
        math.isfinite(risk_side_balanced_mean)
        and math.isfinite(raw_mean)
        and risk_side_balanced_mean > raw_mean + 0.0005
    ):
        diagnosis = "risk_adjusted_side_balance_materially_improves_june"
    elif math.isfinite(side_gated_mean) and math.isfinite(raw_mean) and side_gated_mean > raw_mean + 0.0005:
        diagnosis = "cluster_side_policy_materially_improves_june"
    elif (
        math.isfinite(raw_side_balanced_mean)
        and math.isfinite(raw_mean)
        and raw_side_balanced_mean > raw_mean + 0.0005
    ):
        diagnosis = "side_balance_materially_improves_june_but_may_not_solve"
    elif math.isfinite(gated_mean) and math.isfinite(raw_mean) and gated_mean > raw_mean + 0.0005:
        diagnosis = "cluster_policy_materially_improves_june"
    else:
        diagnosis = "mixed_or_inconclusive"
    out.update(
        {
            "raw_selected_rows": raw_selected,
            "raw_bad_or_throttled_share": raw_bad_share,
            "raw_weak_or_restricted_share": raw_weak_share,
            "raw_allow_normal_share": raw_good_share,
            "raw_allow_normal_selected_mean_u": raw_good_mean,
            "raw_month_mean_u": raw_mean,
            "raw_side_balanced_month_mean_u": raw_side_balanced_mean,
            "raw_side_balanced_delta_mean_u": raw_side_balanced_mean - raw_mean
            if math.isfinite(raw_side_balanced_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "risk_adjusted_month_mean_u": risk_mean,
            "risk_adjusted_delta_mean_u": risk_mean - raw_mean
            if math.isfinite(risk_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "risk_adjusted_side_balanced_month_mean_u": risk_side_balanced_mean,
            "risk_adjusted_side_balanced_delta_mean_u": risk_side_balanced_mean - raw_mean
            if math.isfinite(risk_side_balanced_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "side_calibrated_month_mean_u": side_calibrated_mean,
            "side_calibrated_delta_mean_u": side_calibrated_mean - raw_mean
            if math.isfinite(side_calibrated_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "side_calibrated_side_balanced_month_mean_u": side_calibrated_balanced_mean,
            "side_calibrated_side_balanced_delta_mean_u": side_calibrated_balanced_mean - raw_mean
            if math.isfinite(side_calibrated_balanced_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "side_calibrated_soft_risk_month_mean_u": soft_risk_mean,
            "side_calibrated_soft_risk_delta_mean_u": soft_risk_mean - raw_mean
            if math.isfinite(soft_risk_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "side_calibrated_soft_risk_side_balanced_month_mean_u": soft_risk_balanced_mean,
            "side_calibrated_soft_risk_side_balanced_delta_mean_u": soft_risk_balanced_mean - raw_mean
            if math.isfinite(soft_risk_balanced_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "side_calibrated_soft_risk_threshold_month_mean_u": threshold_mean,
            "side_calibrated_soft_risk_threshold_delta_mean_u": threshold_mean - raw_mean
            if math.isfinite(threshold_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "side_calibrated_soft_risk_threshold_side_balanced_month_mean_u": threshold_balanced_mean,
            "side_calibrated_soft_risk_threshold_side_balanced_delta_mean_u": threshold_balanced_mean - raw_mean
            if math.isfinite(threshold_balanced_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "side_calibrated_soft_risk_bad_mae_cap_month_mean_u": risk_cap_mean,
            "side_calibrated_soft_risk_bad_mae_cap_delta_mean_u": risk_cap_mean - raw_mean
            if math.isfinite(risk_cap_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "side_calibrated_soft_risk_bad_mae_cap_side_capped_month_mean_u": risk_cap_side_capped_mean,
            "side_calibrated_soft_risk_bad_mae_cap_side_capped_delta_mean_u": risk_cap_side_capped_mean - raw_mean
            if math.isfinite(risk_cap_side_capped_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "side_calibrated_soft_risk_bad_mae_cap_threshold_select_month_mean_u": threshold_select_mean,
            "side_calibrated_soft_risk_bad_mae_cap_threshold_select_delta_mean_u": threshold_select_mean - raw_mean
            if math.isfinite(threshold_select_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "lgbm_ranker_side_calibrated_risk_cap_month_mean_u": ranker_mean,
            "lgbm_ranker_side_calibrated_risk_cap_delta_mean_u": ranker_mean - raw_mean
            if math.isfinite(ranker_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s7a_lgbm_ranker_no_prefilter_month_mean_u": s7a_mean,
            "s7a_lgbm_ranker_no_prefilter_delta_mean_u": s7a_mean - raw_mean
            if math.isfinite(s7a_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s7b_lgbm_ranker_relaxed_risk_cap_month_mean_u": s7b_mean,
            "s7b_lgbm_ranker_relaxed_risk_cap_delta_mean_u": s7b_mean - raw_mean
            if math.isfinite(s7b_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s7c_side_specific_ranker_risk_cap_month_mean_u": s7c_mean,
            "s7c_side_specific_ranker_risk_cap_delta_mean_u": s7c_mean - raw_mean
            if math.isfinite(s7c_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s7d_oracle_enriched_ranker_risk_cap_month_mean_u": s7d_mean,
            "s7d_oracle_enriched_ranker_risk_cap_delta_mean_u": s7d_mean - raw_mean
            if math.isfinite(s7d_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s7_two_stage_candidate_rerank_month_mean_u": s7_two_stage_mean,
            "s7_two_stage_candidate_rerank_delta_mean_u": s7_two_stage_mean - raw_mean
            if math.isfinite(s7_two_stage_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s8d_oracle_enriched_ranker_tight_bad_mae_month_mean_u": s8d_mean,
            "s8d_oracle_enriched_ranker_tight_bad_mae_delta_mean_u": s8d_mean - raw_mean
            if math.isfinite(s8d_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s8_two_stage_tight_bad_mae_month_mean_u": s8_two_stage_mean,
            "s8_two_stage_tight_bad_mae_delta_mean_u": s8_two_stage_mean - raw_mean
            if math.isfinite(s8_two_stage_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s9d_oracle_enriched_ranker_calibrated_risk_cap_month_mean_u": s9d_mean,
            "s9d_oracle_enriched_ranker_calibrated_risk_cap_delta_mean_u": s9d_mean - raw_mean
            if math.isfinite(s9d_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s9_two_stage_calibrated_risk_cap_month_mean_u": s9_two_stage_mean,
            "s9_two_stage_calibrated_risk_cap_delta_mean_u": s9_two_stage_mean - raw_mean
            if math.isfinite(s9_two_stage_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s10_recall_soft_calibrated_risk_month_mean_u": s10_soft_mean,
            "s10_recall_soft_calibrated_risk_delta_mean_u": s10_soft_mean - raw_mean
            if math.isfinite(s10_soft_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s10_recall_soft_calibrated_risk_loose_cap_month_mean_u": s10_loose_mean,
            "s10_recall_soft_calibrated_risk_loose_cap_delta_mean_u": s10_loose_mean - raw_mean
            if math.isfinite(s10_loose_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s11_recall_tail_balanced_month_mean_u": s11_mean,
            "s11_recall_tail_balanced_delta_mean_u": s11_mean - raw_mean
            if math.isfinite(s11_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s12_path_quality_ranker_month_mean_u": s12_mean,
            "s12_path_quality_ranker_delta_mean_u": s12_mean - raw_mean
            if math.isfinite(s12_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s12_path_quality_ranker_soft_risk_month_mean_u": s12_soft_mean,
            "s12_path_quality_ranker_soft_risk_delta_mean_u": s12_soft_mean - raw_mean
            if math.isfinite(s12_soft_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s13_constrained_path_quality_month_mean_u": s13_mean,
            "s13_constrained_path_quality_delta_mean_u": s13_mean - raw_mean
            if math.isfinite(s13_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s13_constrained_path_quality_soft_risk_month_mean_u": s13_soft_mean,
            "s13_constrained_path_quality_soft_risk_delta_mean_u": s13_soft_mean - raw_mean
            if math.isfinite(s13_soft_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s14_path_quality_risk_trim_month_mean_u": s14_mean,
            "s14_path_quality_risk_trim_delta_mean_u": s14_mean - raw_mean
            if math.isfinite(s14_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s14_path_quality_soft_risk_trim_month_mean_u": s14_soft_mean,
            "s14_path_quality_soft_risk_trim_delta_mean_u": s14_soft_mean - raw_mean
            if math.isfinite(s14_soft_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s15_side_path_quality_ranker_month_mean_u": s15_mean,
            "s15_side_path_quality_ranker_delta_mean_u": s15_mean - raw_mean
            if math.isfinite(s15_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s15_side_path_quality_risk_trim_month_mean_u": s15_trim_mean,
            "s15_side_path_quality_risk_trim_delta_mean_u": s15_trim_mean - raw_mean
            if math.isfinite(s15_trim_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s16_discovery_path_quality_blend_month_mean_u": s16_mean,
            "s16_discovery_path_quality_blend_delta_mean_u": s16_mean - raw_mean
            if math.isfinite(s16_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "s16_discovery_path_quality_risk_trim_month_mean_u": s16_trim_mean,
            "s16_discovery_path_quality_risk_trim_delta_mean_u": s16_trim_mean - raw_mean
            if math.isfinite(s16_trim_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "gated_month_mean_u": gated_mean,
            "gated_delta_mean_u": gated_mean - raw_mean
            if math.isfinite(gated_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "cluster_side_policy_month_mean_u": side_gated_mean,
            "cluster_side_policy_delta_mean_u": side_gated_mean - raw_mean
            if math.isfinite(side_gated_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "cluster_side_policy_side_balanced_month_mean_u": side_gated_balanced_mean,
            "cluster_side_policy_side_balanced_delta_mean_u": side_gated_balanced_mean - raw_mean
            if math.isfinite(side_gated_balanced_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "cluster_side_risk_adjusted_month_mean_u": cluster_side_risk_mean,
            "cluster_side_risk_adjusted_delta_mean_u": cluster_side_risk_mean - raw_mean
            if math.isfinite(cluster_side_risk_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "cluster_side_risk_adjusted_side_balanced_month_mean_u": cluster_side_risk_balanced_mean,
            "cluster_side_risk_adjusted_side_balanced_delta_mean_u": cluster_side_risk_balanced_mean - raw_mean
            if math.isfinite(cluster_side_risk_balanced_mean) and math.isfinite(raw_mean)
            else float("nan"),
            "diagnosis": diagnosis,
        }
    )
    return out


def run_cluster_policy_smoke(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    evaluation_utility_column: str | None,
    max_feature_store_features: int | None,
    label_arms: list[str],
    weight_arms: list[str],
    seeds: list[int],
    top_fracs: list[float],
    train_lookback_months: int | None,
    cluster_candidates: str | None,
    reg_covar_candidates: str | None,
    smooth_lambda_candidates: str | None,
    ae_max_iter: int,
    max_train_rows: int,
    min_side_cluster_rows: int,
    min_side_cluster_frac: float,
    min_allow_mean_u: float,
    min_allow_hit_u: float,
    block_mean_u: float,
    min_side_mean_u: float,
    bad_mae_lambda: float,
    timeout_lambda: float,
    lower_tail_lambda: float,
    cluster_adjustment_lambda: float,
    calibration_bins: int,
    calibration_min_bin_rows: int,
    calibrated_bad_mae_lambda: float,
    calibrated_timeout_lambda: float,
    calibrated_lower_tail_lambda: float,
    gmm_entropy_lambda: float,
    gmm_recon_lambda: float,
    gmm_mahal_lambda: float,
    min_calibrated_score: float,
    max_pred_bad_mae: float,
    max_pred_timeout: float,
    max_pred_lower_tail: float,
    s7_relaxed_max_pred_bad_mae: float,
    s7_relaxed_max_pred_timeout: float,
    s7_relaxed_max_pred_lower_tail: float,
    s7_candidate_top_n: int,
    s7_candidate_alt_top_n: int,
    s7_candidate_top_frac: float,
    s8_tight_max_pred_bad_mae: float,
    s8_tight_max_pred_timeout: float,
    s8_tight_max_pred_lower_tail: float,
    s9_calibrated_max_pred_bad_mae: float,
    s9_calibrated_max_pred_timeout: float,
    s9_calibrated_max_pred_lower_tail: float,
    s10_bad_mae_lambda: float,
    s10_timeout_lambda: float,
    s10_lower_tail_lambda: float,
    s10_discovery_lambda: float,
    s10_loose_max_pred_bad_mae: float,
    s10_loose_max_pred_timeout: float,
    s10_loose_max_pred_lower_tail: float,
    s11_bad_mae_lambda: float,
    s11_timeout_lambda: float,
    s11_lower_tail_lambda: float,
    s11_discovery_lambda: float,
    s12_bad_mae_lambda: float,
    s12_timeout_lambda: float,
    s12_lower_tail_lambda: float,
    s12_discovery_lambda: float,
    s13_primary_max_pred_bad_mae: float,
    s13_primary_max_pred_timeout: float,
    s13_primary_max_pred_lower_tail: float,
    s13_backfill_max_pred_bad_mae: float,
    s13_backfill_max_pred_timeout: float,
    s13_backfill_max_pred_lower_tail: float,
    s13_max_backfill_share: float,
    s14_trim_share: float,
    s14_protect_top_score_share: float,
    s14_bad_mae_weight: float,
    s14_timeout_weight: float,
    s14_lower_tail_weight: float,
    side_max_share: float,
    enable_ranker: bool,
    oracle_min_group_rows: int,
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
        feature_matrix = feature_matrix.astype(np.float32, copy=False)
        overlapping_feature_cols = [c for c in feature_matrix.columns if c in frame.columns]
        if overlapping_feature_cols:
            frame = frame.drop(columns=overlapping_feature_cols)
        frame = pd.concat([frame, feature_matrix], axis=1, copy=False)

    metrics = _path_metrics(frame)
    evaluation_utility_source = _apply_evaluation_utility_column(frame, metrics, evaluation_utility_column)
    metrics = _downcast_metrics(metrics)
    features = _feature_columns(frame)
    targets = _label_targets(frame, metrics)
    targets.update(_fixed_artifact_targets(frame, metrics))
    missing_labels = sorted(set(label_arms) - set(targets))
    if missing_labels:
        raise ValueError(f"Unknown label arms: {missing_labels}")

    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(month_period.dropna().unique())
    monthly_rows: list[dict[str, Any]] = []
    cluster_policy_rows: list[dict[str, Any]] = []
    cluster_side_policy_rows: list[dict[str, Any]] = []
    cluster_diag_rows: list[dict[str, Any]] = []
    cluster_selection_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    stage_gate_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []

    for month in months[1:]:
        train_mask = month_period < month
        if train_lookback_months is not None and int(train_lookback_months) > 0:
            prior_months = sorted(month_period[train_mask].dropna().unique())
            keep_months = set(prior_months[-int(train_lookback_months) :])
            train_mask = train_mask & month_period.isin(keep_months)
        valid_mask = month_period == month
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
            continue
        x_train, x_valid = _month_model_frame(
            frame,
            train_mask=train_mask,
            valid_mask=valid_mask,
            features=features,
        )
        train = frame.loc[train_mask].copy(deep=False)
        valid = frame.loc[valid_mask].copy(deep=False).reset_index(drop=True)
        train_metrics = metrics.loc[train_mask].copy(deep=False)
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        baseline = _baseline_row(valid_metrics)

        for label_arm in label_arms:
            target_train = targets[label_arm].loc[train_mask].copy()
            target_valid = targets[label_arm].loc[valid_mask].copy().reset_index(drop=True)
            econ_targets = {
                "returns": pd.to_numeric(train_metrics["u_policy_net"], errors="coerce").to_numpy(dtype=np.float32),
                "target": pd.to_numeric(target_train["target_soft"], errors="coerce").to_numpy(dtype=np.float32),
                "side": pd.to_numeric(train_metrics["side"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32),
            }
            gmm_state = fit_ae_gmm_state(
                x_train,
                economic_targets=econ_targets,
                random_state=913 + int(pd.Period(month).month),
                max_train_rows=int(max_train_rows),
                ae_max_iter=int(ae_max_iter),
                cluster_candidates=cluster_candidates,
                reg_covar_candidates=reg_covar_candidates,
                smooth_lambda_candidates=smooth_lambda_candidates,
                require_both_sides=True,
                min_side_cluster_frac=float(min_side_cluster_frac),
                min_side_cluster_rows=int(min_side_cluster_rows),
            )
            train_gmm = transform_ae_gmm_features(x_train, gmm_state, index=x_train.index)
            valid_gmm = transform_ae_gmm_features(x_valid, gmm_state, index=x_valid.index)
            x_train_model = _append_model_state_features(x_train, train_gmm)
            x_valid_model = _append_model_state_features(x_valid, valid_gmm)
            train_clusters = _cluster_labels(train_gmm)
            valid_clusters = _cluster_labels(valid_gmm)
            train_stats = _cluster_stats(
                clusters=train_clusters,
                metrics=train_metrics.reset_index(drop=True),
                prefix="train",
                min_side_rows=int(min_side_cluster_rows),
            )
            train_side_stats = _cluster_side_stats(
                clusters=train_clusters,
                metrics=train_metrics.reset_index(drop=True),
                prefix="train",
                min_side_rows=int(min_side_cluster_rows),
            )
            valid_stats = _cluster_stats(
                clusters=valid_clusters,
                metrics=valid_metrics,
                prefix="valid_all",
                min_side_rows=int(min_side_cluster_rows),
            )
            policy = _assign_cluster_policy(
                train_stats,
                min_allow_mean_u=float(min_allow_mean_u),
                min_allow_hit_u=float(min_allow_hit_u),
                block_mean_u=float(block_mean_u),
                min_side_mean_u=float(min_side_mean_u),
            )
            if not policy.empty:
                policy.insert(0, "period", month)
                policy.insert(1, "label_arm", label_arm)
                cluster_policy_rows.extend(policy.to_dict(orient="records"))
            side_policy = _assign_cluster_side_policy(
                train_side_stats,
                policy,
                min_allow_mean_u=float(min_allow_mean_u),
                min_allow_hit_u=float(min_allow_hit_u),
                block_mean_u=float(block_mean_u),
                min_side_rows=int(min_side_cluster_rows),
            )
            if not side_policy.empty:
                side_policy.insert(0, "period", month)
                side_policy.insert(1, "label_arm", label_arm)
                cluster_side_policy_rows.extend(side_policy.to_dict(orient="records"))
            state_rows.append(
                {
                    "period": month,
                    "label_arm": label_arm,
                    "gmm_enabled": bool(gmm_state.get("enabled", False)),
                    "gmm_reason": str(gmm_state.get("reason", "")),
                    "selected_config": json.dumps(
                        _strict_json_safe(gmm_state.get("selected_config", {})),
                        sort_keys=True,
                    ),
                    "hpo_report_count": int(gmm_state.get("hpo_report_count", len(gmm_state.get("hpo_reports", [])))),
                    "hpo_grid": json.dumps(_strict_json_safe(gmm_state.get("hpo_grid", {})), sort_keys=True),
                }
            )
            policy_for_merge = policy.drop(columns=["period", "label_arm"], errors="ignore")
            merged_cluster = (
                policy_for_merge.merge(valid_stats, on="cluster", how="outer")
                if not policy_for_merge.empty
                else valid_stats
            )

            for weight_arm in weight_arms:
                weights = _weight_series(
                    frame=train,
                    metrics=train_metrics,
                    target=target_train,
                    arm=weight_arm,
                )
                raw_pred = _mean_seed_predictions(
                    x_train=x_train_model,
                    y_train=target_train["target_soft"],
                    w_train=weights,
                    x_valid=x_valid_model,
                    seeds=seeds,
                )
                train_raw_pred = _mean_seed_predictions(
                    x_train=x_train_model,
                    y_train=target_train["target_soft"],
                    w_train=weights,
                    x_valid=x_train_model,
                    seeds=seeds,
                )
                raw_score = pd.Series(raw_pred.astype(np.float32), index=valid.index)
                risk_predictions = _fit_risk_predictions(
                    x_train=x_train_model,
                    train_metrics=train_metrics.reset_index(drop=True),
                    x_valid=x_valid_model,
                    seeds=seeds,
                )
                train_risk_predictions = _fit_risk_predictions(
                    x_train=x_train_model,
                    train_metrics=train_metrics.reset_index(drop=True),
                    x_valid=x_train_model,
                    seeds=seeds,
                )
                calibrated_risk_predictions = _fit_side_calibrated_risk_predictions(
                    train_predictions=train_risk_predictions,
                    train_metrics=train_metrics.reset_index(drop=True),
                    valid_predictions=risk_predictions,
                    valid_side=valid_metrics["side"],
                    n_bins=int(calibration_bins),
                    min_bin_rows=int(calibration_min_bin_rows),
                )
                side_calibrated_score = _fit_side_calibrated_score(
                    train_score=train_raw_pred,
                    train_metrics=train_metrics.reset_index(drop=True),
                    valid_score=raw_score,
                    valid_side=valid_metrics["side"],
                    n_bins=int(calibration_bins),
                    min_bin_rows=int(calibration_min_bin_rows),
                )
                gmm_penalty = _gmm_state_risk_penalty(
                    train_gmm,
                    valid_gmm,
                    entropy_lambda=float(gmm_entropy_lambda),
                    recon_lambda=float(gmm_recon_lambda),
                    mahal_lambda=float(gmm_mahal_lambda),
                )
                side_calibrated_soft_risk_score = _calibrated_soft_risk_score(
                    side_calibrated_score,
                    risk_predictions,
                    gmm_penalty,
                    bad_mae_lambda=float(calibrated_bad_mae_lambda),
                    timeout_lambda=float(calibrated_timeout_lambda),
                    lower_tail_lambda=float(calibrated_lower_tail_lambda),
                )
                side_calibrated_soft_risk_threshold_score = _threshold_score(
                    side_calibrated_soft_risk_score,
                    float(min_calibrated_score),
                )
                side_calibrated_soft_risk_bad_mae_cap_score = _risk_capped_score(
                    side_calibrated_soft_risk_score,
                    risk_predictions,
                    max_pred_bad_mae=float(max_pred_bad_mae),
                    max_pred_timeout=float(max_pred_timeout),
                    max_pred_lower_tail=float(max_pred_lower_tail),
                )
                side_calibrated_soft_risk_bad_mae_cap_threshold_score = _threshold_score(
                    side_calibrated_soft_risk_bad_mae_cap_score,
                    float(min_calibrated_score),
                )
                side_calibrated_soft_risk_bad_mae_cap_threshold_side_capped_score = _side_exposure_capped_score(
                    side_calibrated_soft_risk_bad_mae_cap_threshold_score,
                    valid_metrics["side"],
                    max_side_share=float(side_max_share),
                )
                if bool(enable_ranker):
                    train_ranker_pred, valid_ranker_pred, ranker_status = _fit_lgbm_ranker_predictions(
                        x_train=x_train_model,
                        train_frame=train,
                        train_metrics=train_metrics,
                        w_train=weights,
                        x_valid=x_valid_model,
                        seeds=seeds,
                    )
                else:
                    train_ranker_pred = np.full(len(x_train_model), np.nan, dtype=np.float32)
                    valid_ranker_pred = np.full(len(x_valid_model), np.nan, dtype=np.float32)
                    ranker_status = "disabled"
                ranker_raw_score = pd.Series(valid_ranker_pred.astype(np.float32), index=valid.index)
                ranker_side_calibrated_score = _fit_side_calibrated_score(
                    train_score=train_ranker_pred,
                    train_metrics=train_metrics.reset_index(drop=True),
                    valid_score=ranker_raw_score,
                    valid_side=valid_metrics["side"],
                    n_bins=int(calibration_bins),
                    min_bin_rows=int(calibration_min_bin_rows),
                )
                ranker_side_calibrated_risk_cap_score = _risk_capped_score(
                    ranker_side_calibrated_score,
                    risk_predictions,
                    max_pred_bad_mae=float(max_pred_bad_mae),
                    max_pred_timeout=float(max_pred_timeout),
                    max_pred_lower_tail=float(max_pred_lower_tail),
                )
                if bool(enable_ranker):
                    (
                        train_side_ranker_pred,
                        valid_side_ranker_pred,
                        side_ranker_status,
                    ) = _fit_side_lgbm_ranker_predictions(
                        x_train=x_train_model,
                        train_frame=train,
                        train_metrics=train_metrics,
                        w_train=weights,
                        x_valid=x_valid_model,
                        valid_metrics=valid_metrics,
                        seeds=seeds,
                    )
                    (
                        train_enriched_ranker_pred,
                        valid_enriched_ranker_pred,
                        enriched_ranker_status,
                    ) = _fit_lgbm_ranker_predictions(
                        x_train=x_train_model,
                        train_frame=train,
                        train_metrics=train_metrics,
                        w_train=weights,
                        x_valid=x_valid_model,
                        seeds=seeds,
                        relevance_mode="oracle_enriched",
                    )
                    (
                        train_path_quality_ranker_pred,
                        valid_path_quality_ranker_pred,
                        path_quality_ranker_status,
                    ) = _fit_lgbm_ranker_predictions(
                        x_train=x_train_model,
                        train_frame=train,
                        train_metrics=train_metrics,
                        w_train=weights,
                        x_valid=x_valid_model,
                        seeds=seeds,
                        relevance_mode="path_quality",
                    )
                    (
                        train_side_path_quality_ranker_pred,
                        valid_side_path_quality_ranker_pred,
                        side_path_quality_ranker_status,
                    ) = _fit_side_lgbm_ranker_predictions(
                        x_train=x_train_model,
                        train_frame=train,
                        train_metrics=train_metrics,
                        w_train=weights,
                        x_valid=x_valid_model,
                        valid_metrics=valid_metrics,
                        seeds=seeds,
                        relevance_mode="path_quality",
                    )
                else:
                    train_side_ranker_pred = np.full(len(x_train_model), np.nan, dtype=np.float32)
                    valid_side_ranker_pred = np.full(len(x_valid_model), np.nan, dtype=np.float32)
                    side_ranker_status = "disabled"
                    train_enriched_ranker_pred = np.full(len(x_train_model), np.nan, dtype=np.float32)
                    valid_enriched_ranker_pred = np.full(len(x_valid_model), np.nan, dtype=np.float32)
                    enriched_ranker_status = "disabled"
                    train_path_quality_ranker_pred = np.full(len(x_train_model), np.nan, dtype=np.float32)
                    valid_path_quality_ranker_pred = np.full(len(x_valid_model), np.nan, dtype=np.float32)
                    path_quality_ranker_status = "disabled"
                    train_side_path_quality_ranker_pred = np.full(len(x_train_model), np.nan, dtype=np.float32)
                    valid_side_path_quality_ranker_pred = np.full(len(x_valid_model), np.nan, dtype=np.float32)
                    side_path_quality_ranker_status = "disabled"
                side_ranker_raw_score = pd.Series(valid_side_ranker_pred.astype(np.float32), index=valid.index)
                side_ranker_side_calibrated_score = _fit_side_calibrated_score(
                    train_score=train_side_ranker_pred,
                    train_metrics=train_metrics.reset_index(drop=True),
                    valid_score=side_ranker_raw_score,
                    valid_side=valid_metrics["side"],
                    n_bins=int(calibration_bins),
                    min_bin_rows=int(calibration_min_bin_rows),
                )
                side_ranker_risk_cap_score = _risk_capped_score(
                    side_ranker_side_calibrated_score,
                    risk_predictions,
                    max_pred_bad_mae=float(max_pred_bad_mae),
                    max_pred_timeout=float(max_pred_timeout),
                    max_pred_lower_tail=float(max_pred_lower_tail),
                )
                enriched_ranker_raw_score = pd.Series(valid_enriched_ranker_pred.astype(np.float32), index=valid.index)
                enriched_ranker_side_calibrated_score = _fit_side_calibrated_score(
                    train_score=train_enriched_ranker_pred,
                    train_metrics=train_metrics.reset_index(drop=True),
                    valid_score=enriched_ranker_raw_score,
                    valid_side=valid_metrics["side"],
                    n_bins=int(calibration_bins),
                    min_bin_rows=int(calibration_min_bin_rows),
                )
                enriched_ranker_risk_cap_score = _risk_capped_score(
                    enriched_ranker_side_calibrated_score,
                    risk_predictions,
                    max_pred_bad_mae=float(max_pred_bad_mae),
                    max_pred_timeout=float(max_pred_timeout),
                    max_pred_lower_tail=float(max_pred_lower_tail),
                )
                path_quality_ranker_raw_score = pd.Series(
                    valid_path_quality_ranker_pred.astype(np.float32),
                    index=valid.index,
                )
                path_quality_ranker_side_calibrated_score = _fit_side_calibrated_score(
                    train_score=train_path_quality_ranker_pred,
                    train_metrics=train_metrics.reset_index(drop=True),
                    valid_score=path_quality_ranker_raw_score,
                    valid_side=valid_metrics["side"],
                    n_bins=int(calibration_bins),
                    min_bin_rows=int(calibration_min_bin_rows),
                )
                side_path_quality_ranker_raw_score = pd.Series(
                    valid_side_path_quality_ranker_pred.astype(np.float32),
                    index=valid.index,
                )
                side_path_quality_ranker_side_calibrated_score = _fit_side_calibrated_score(
                    train_score=train_side_path_quality_ranker_pred,
                    train_metrics=train_metrics.reset_index(drop=True),
                    valid_score=side_path_quality_ranker_raw_score,
                    valid_side=valid_metrics["side"],
                    n_bins=int(calibration_bins),
                    min_bin_rows=int(calibration_min_bin_rows),
                )
                ranker_relaxed_risk_cap_score = _risk_capped_score(
                    ranker_side_calibrated_score,
                    risk_predictions,
                    max_pred_bad_mae=float(s7_relaxed_max_pred_bad_mae),
                    max_pred_timeout=float(s7_relaxed_max_pred_timeout),
                    max_pred_lower_tail=float(s7_relaxed_max_pred_lower_tail),
                )
                strict_risk_mask = _risk_pass_mask(
                    risk_predictions,
                    max_pred_bad_mae=float(max_pred_bad_mae),
                    max_pred_timeout=float(max_pred_timeout),
                    max_pred_lower_tail=float(max_pred_lower_tail),
                )
                relaxed_risk_mask = _risk_pass_mask(
                    risk_predictions,
                    max_pred_bad_mae=float(s7_relaxed_max_pred_bad_mae),
                    max_pred_timeout=float(s7_relaxed_max_pred_timeout),
                    max_pred_lower_tail=float(s7_relaxed_max_pred_lower_tail),
                )
                tight_tail_risk_mask = _risk_pass_mask(
                    risk_predictions,
                    max_pred_bad_mae=float(s8_tight_max_pred_bad_mae),
                    max_pred_timeout=float(s8_tight_max_pred_timeout),
                    max_pred_lower_tail=float(s8_tight_max_pred_lower_tail),
                )
                enriched_ranker_tight_bad_mae_score = _risk_capped_score(
                    enriched_ranker_side_calibrated_score,
                    risk_predictions,
                    max_pred_bad_mae=float(s8_tight_max_pred_bad_mae),
                    max_pred_timeout=float(s8_tight_max_pred_timeout),
                    max_pred_lower_tail=float(s8_tight_max_pred_lower_tail),
                )
                calibrated_risk_mask = _risk_pass_mask(
                    calibrated_risk_predictions,
                    max_pred_bad_mae=float(s9_calibrated_max_pred_bad_mae),
                    max_pred_timeout=float(s9_calibrated_max_pred_timeout),
                    max_pred_lower_tail=float(s9_calibrated_max_pred_lower_tail),
                )
                enriched_ranker_calibrated_risk_cap_score = _risk_capped_score(
                    enriched_ranker_side_calibrated_score,
                    calibrated_risk_predictions,
                    max_pred_bad_mae=float(s9_calibrated_max_pred_bad_mae),
                    max_pred_timeout=float(s9_calibrated_max_pred_timeout),
                    max_pred_lower_tail=float(s9_calibrated_max_pred_lower_tail),
                )
                discovery_score = _max_percentile_score(
                    [
                        raw_score,
                        side_calibrated_score,
                        side_calibrated_soft_risk_score,
                        ranker_raw_score,
                        ranker_side_calibrated_score,
                        side_ranker_raw_score,
                        enriched_ranker_raw_score,
                    ]
                )
                stage_a_top_primary_pre_risk_mask = _per_timestamp_top_mask(
                    valid,
                    discovery_score,
                    top_n=int(s7_candidate_top_n),
                )
                stage_a_top_alt_pre_risk_mask = _per_timestamp_top_mask(
                    valid,
                    discovery_score,
                    top_n=int(s7_candidate_alt_top_n),
                )
                stage_a_top_frac_pre_risk_mask = _per_timestamp_top_mask(
                    valid,
                    discovery_score,
                    top_frac=float(s7_candidate_top_frac),
                )
                stage_a_candidate_pre_risk_mask = (
                    stage_a_top_primary_pre_risk_mask
                    | stage_a_top_alt_pre_risk_mask
                    | stage_a_top_frac_pre_risk_mask
                )
                stage_a_top_primary_mask = stage_a_top_primary_pre_risk_mask & relaxed_risk_mask
                stage_a_top_alt_mask = stage_a_top_alt_pre_risk_mask & relaxed_risk_mask
                stage_a_top_frac_mask = stage_a_top_frac_pre_risk_mask & relaxed_risk_mask
                stage_a_candidate_mask = (
                    stage_a_top_primary_mask
                    | stage_a_top_alt_mask
                    | stage_a_top_frac_mask
                ) & relaxed_risk_mask
                two_stage_candidate_score = _mask_score(
                    ranker_side_calibrated_score,
                    stage_a_candidate_mask & strict_risk_mask,
                )
                s8_two_stage_tight_bad_mae_candidate_score = _mask_score(
                    enriched_ranker_side_calibrated_score,
                    stage_a_candidate_mask & tight_tail_risk_mask,
                )
                s9_two_stage_calibrated_risk_candidate_score = _mask_score(
                    enriched_ranker_side_calibrated_score,
                    stage_a_candidate_mask & calibrated_risk_mask,
                )
                s10_recall_soft_calibrated_risk_score = _recall_preserving_calibrated_risk_score(
                    base_score=enriched_ranker_side_calibrated_score,
                    discovery_score=discovery_score,
                    calibrated_risk_predictions=calibrated_risk_predictions,
                    candidate_mask=stage_a_candidate_mask,
                    bad_mae_lambda=float(s10_bad_mae_lambda),
                    timeout_lambda=float(s10_timeout_lambda),
                    lower_tail_lambda=float(s10_lower_tail_lambda),
                    discovery_lambda=float(s10_discovery_lambda),
                )
                s10_loose_calibrated_risk_mask = _risk_pass_mask(
                    calibrated_risk_predictions,
                    max_pred_bad_mae=float(s10_loose_max_pred_bad_mae),
                    max_pred_timeout=float(s10_loose_max_pred_timeout),
                    max_pred_lower_tail=float(s10_loose_max_pred_lower_tail),
                )
                s10_recall_soft_calibrated_risk_loose_cap_score = _mask_score(
                    s10_recall_soft_calibrated_risk_score,
                    s10_loose_calibrated_risk_mask,
                )
                s11_recall_tail_balanced_score = _recall_preserving_calibrated_risk_score(
                    base_score=enriched_ranker_side_calibrated_score,
                    discovery_score=discovery_score,
                    calibrated_risk_predictions=calibrated_risk_predictions,
                    candidate_mask=stage_a_candidate_mask,
                    bad_mae_lambda=float(s11_bad_mae_lambda),
                    timeout_lambda=float(s11_timeout_lambda),
                    lower_tail_lambda=float(s11_lower_tail_lambda),
                    discovery_lambda=float(s11_discovery_lambda),
                )
                s12_path_quality_ranker_score = _mask_score(
                    path_quality_ranker_side_calibrated_score,
                    stage_a_candidate_mask,
                )
                s12_path_quality_ranker_soft_risk_score = _recall_preserving_calibrated_risk_score(
                    base_score=path_quality_ranker_side_calibrated_score,
                    discovery_score=discovery_score,
                    calibrated_risk_predictions=calibrated_risk_predictions,
                    candidate_mask=stage_a_candidate_mask,
                    bad_mae_lambda=float(s12_bad_mae_lambda),
                    timeout_lambda=float(s12_timeout_lambda),
                    lower_tail_lambda=float(s12_lower_tail_lambda),
                    discovery_lambda=float(s12_discovery_lambda),
                )
                s15_side_path_quality_ranker_score = _mask_score(
                    side_path_quality_ranker_side_calibrated_score,
                    stage_a_candidate_mask,
                )
                s16_discovery_path_quality_blend_score = _mask_score(
                    _max_percentile_score(
                        [
                            discovery_score,
                            path_quality_ranker_side_calibrated_score,
                            side_path_quality_ranker_side_calibrated_score,
                        ]
                    ),
                    stage_a_candidate_mask,
                )
                for calibration_selector, calibration_score in (
                    ("S0_raw_score", raw_score),
                    ("S2_side_calibrated_score", side_calibrated_score),
                    ("S3_side_calibrated_soft_risk_score", side_calibrated_soft_risk_score),
                    ("S4_risk_cap_score", side_calibrated_soft_risk_bad_mae_cap_score),
                    (
                        "S5_risk_cap_threshold_score",
                        side_calibrated_soft_risk_bad_mae_cap_threshold_score,
                    ),
                    (
                        "S6_lgbm_ranker_side_calibrated_risk_cap_score",
                        ranker_side_calibrated_risk_cap_score,
                    ),
                    (
                        "S7a_lgbm_ranker_no_prefilter_score",
                        ranker_side_calibrated_score,
                    ),
                    (
                        "S7b_lgbm_ranker_relaxed_risk_cap_score",
                        ranker_relaxed_risk_cap_score,
                    ),
                    (
                        "S7c_side_specific_ranker_risk_cap_score",
                        side_ranker_risk_cap_score,
                    ),
                    (
                        "S7d_oracle_enriched_ranker_risk_cap_score",
                        enriched_ranker_risk_cap_score,
                    ),
                    (
                        "S7_two_stage_candidate_rerank_score",
                        two_stage_candidate_score,
                    ),
                    (
                        "S8d_oracle_enriched_ranker_tight_bad_mae_score",
                        enriched_ranker_tight_bad_mae_score,
                    ),
                    (
                        "S8_two_stage_tight_bad_mae_score",
                        s8_two_stage_tight_bad_mae_candidate_score,
                    ),
                    (
                        "S9d_oracle_enriched_ranker_calibrated_risk_cap_score",
                        enriched_ranker_calibrated_risk_cap_score,
                    ),
                    (
                        "S9_two_stage_calibrated_risk_cap_score",
                        s9_two_stage_calibrated_risk_candidate_score,
                    ),
                    (
                        "S10_recall_soft_calibrated_risk_score",
                        s10_recall_soft_calibrated_risk_score,
                    ),
                    (
                        "S10_recall_soft_calibrated_risk_loose_cap_score",
                        s10_recall_soft_calibrated_risk_loose_cap_score,
                    ),
                    (
                        "S11_recall_tail_balanced_score",
                        s11_recall_tail_balanced_score,
                    ),
                    (
                        "S12_path_quality_ranker_score",
                        s12_path_quality_ranker_score,
                    ),
                    (
                        "S12_path_quality_ranker_soft_risk_score",
                        s12_path_quality_ranker_soft_risk_score,
                    ),
                    (
                        "S15_side_path_quality_ranker_score",
                        s15_side_path_quality_ranker_score,
                    ),
                    (
                        "S16_discovery_path_quality_blend_score",
                        s16_discovery_path_quality_blend_score,
                    ),
                ):
                    calibration_rows.extend(
                        _score_quantile_side_rows(
                            month=month,
                            label_arm=label_arm,
                            weight_arm=weight_arm,
                            selector=calibration_selector,
                            score=calibration_score,
                            metrics=valid_metrics,
                            n_bins=int(calibration_bins),
                            min_bin_rows=int(calibration_min_bin_rows),
                        )
                    )
                cluster_side_adjustment = _cluster_side_adjustment_vector(
                    clusters=valid_clusters,
                    side=valid_metrics["side"],
                    side_policy=side_policy,
                )
                risk_score = _risk_adjusted_score(
                    raw_score,
                    risk_predictions,
                    bad_mae_lambda=float(bad_mae_lambda),
                    timeout_lambda=float(timeout_lambda),
                    lower_tail_lambda=float(lower_tail_lambda),
                )
                cluster_side_risk_score = _risk_adjusted_score(
                    raw_score,
                    risk_predictions,
                    bad_mae_lambda=float(bad_mae_lambda),
                    timeout_lambda=float(timeout_lambda),
                    lower_tail_lambda=float(lower_tail_lambda),
                    cluster_adjustment=cluster_side_adjustment,
                    cluster_adjustment_lambda=float(cluster_adjustment_lambda),
                )
                for top_frac in top_fracs:
                    gated_score, eligible = _apply_cluster_policy_score(
                        raw_score,
                        valid_clusters,
                        policy,
                        top_frac=float(top_frac),
                    )
                    side_gated_score, side_eligible = _apply_cluster_side_policy_score(
                        raw_score,
                        valid_clusters,
                        valid_metrics["side"],
                        side_policy,
                        top_frac=float(top_frac),
                    )
                    raw_side_balanced_score = _side_balanced_top_score(
                        raw_score,
                        valid_metrics["side"],
                        float(top_frac),
                    )
                    gated_side_balanced_score = _side_balanced_top_score(
                        gated_score,
                        valid_metrics["side"],
                        float(top_frac),
                    )
                    side_gated_side_balanced_score = _side_balanced_top_score(
                        side_gated_score,
                        valid_metrics["side"],
                        float(top_frac),
                    )
                    risk_side_balanced_score = _side_balanced_top_score(
                        risk_score,
                        valid_metrics["side"],
                        float(top_frac),
                    )
                    side_calibrated_side_balanced_score = _side_balanced_top_score(
                        side_calibrated_score,
                        valid_metrics["side"],
                        float(top_frac),
                    )
                    side_calibrated_soft_risk_side_balanced_score = _side_balanced_top_score(
                        side_calibrated_soft_risk_score,
                        valid_metrics["side"],
                        float(top_frac),
                    )
                    side_calibrated_soft_risk_threshold_side_balanced_score = _side_balanced_top_score(
                        side_calibrated_soft_risk_threshold_score,
                        valid_metrics["side"],
                        float(top_frac),
                    )
                    side_calibrated_soft_risk_bad_mae_cap_side_capped_score = _side_capped_top_score(
                        side_calibrated_soft_risk_bad_mae_cap_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    ranker_side_calibrated_risk_cap_side_capped_score = _side_capped_top_score(
                        ranker_side_calibrated_risk_cap_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s7a_ranker_no_prefilter_side_capped_score = _side_capped_top_score(
                        ranker_side_calibrated_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s7b_ranker_relaxed_risk_cap_side_capped_score = _side_capped_top_score(
                        ranker_relaxed_risk_cap_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s7c_side_ranker_risk_cap_side_capped_score = _side_capped_top_score(
                        side_ranker_risk_cap_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s7d_enriched_ranker_risk_cap_side_capped_score = _side_capped_top_score(
                        enriched_ranker_risk_cap_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s7_two_stage_candidate_rerank_score = _side_capped_top_score(
                        two_stage_candidate_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s8d_enriched_ranker_tight_bad_mae_side_capped_score = _side_capped_top_score(
                        enriched_ranker_tight_bad_mae_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s8_two_stage_tight_bad_mae_score = _side_capped_top_score(
                        s8_two_stage_tight_bad_mae_candidate_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s9d_enriched_ranker_calibrated_risk_side_capped_score = _side_capped_top_score(
                        enriched_ranker_calibrated_risk_cap_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s9_two_stage_calibrated_risk_cap_score = _side_capped_top_score(
                        s9_two_stage_calibrated_risk_candidate_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s10_recall_soft_calibrated_risk_side_capped_score = _side_capped_top_score(
                        s10_recall_soft_calibrated_risk_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s10_recall_soft_calibrated_risk_loose_cap_side_capped_score = _side_capped_top_score(
                        s10_recall_soft_calibrated_risk_loose_cap_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s11_recall_tail_balanced_side_capped_score = _side_capped_top_score(
                        s11_recall_tail_balanced_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s12_path_quality_ranker_side_capped_score = _side_capped_top_score(
                        s12_path_quality_ranker_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s12_path_quality_ranker_soft_risk_side_capped_score = _side_capped_top_score(
                        s12_path_quality_ranker_soft_risk_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s13_constrained_path_quality_score = _risk_constrained_backfill_top_score(
                        s12_path_quality_ranker_score,
                        valid_metrics["side"],
                        calibrated_risk_predictions,
                        float(top_frac),
                        max_side_share=float(side_max_share),
                        primary_max_pred_bad_mae=float(s13_primary_max_pred_bad_mae),
                        primary_max_pred_timeout=float(s13_primary_max_pred_timeout),
                        primary_max_pred_lower_tail=float(s13_primary_max_pred_lower_tail),
                        backfill_max_pred_bad_mae=float(s13_backfill_max_pred_bad_mae),
                        backfill_max_pred_timeout=float(s13_backfill_max_pred_timeout),
                        backfill_max_pred_lower_tail=float(s13_backfill_max_pred_lower_tail),
                        max_backfill_share=float(s13_max_backfill_share),
                    )
                    s13_constrained_path_quality_soft_risk_score = _risk_constrained_backfill_top_score(
                        s12_path_quality_ranker_soft_risk_score,
                        valid_metrics["side"],
                        calibrated_risk_predictions,
                        float(top_frac),
                        max_side_share=float(side_max_share),
                        primary_max_pred_bad_mae=float(s13_primary_max_pred_bad_mae),
                        primary_max_pred_timeout=float(s13_primary_max_pred_timeout),
                        primary_max_pred_lower_tail=float(s13_primary_max_pred_lower_tail),
                        backfill_max_pred_bad_mae=float(s13_backfill_max_pred_bad_mae),
                        backfill_max_pred_timeout=float(s13_backfill_max_pred_timeout),
                        backfill_max_pred_lower_tail=float(s13_backfill_max_pred_lower_tail),
                        max_backfill_share=float(s13_max_backfill_share),
                    )
                    s14_path_quality_risk_trim_score = _risk_trimmed_top_score(
                        s12_path_quality_ranker_score,
                        valid_metrics["side"],
                        calibrated_risk_predictions,
                        float(top_frac),
                        max_side_share=float(side_max_share),
                        trim_share=float(s14_trim_share),
                        protect_top_score_share=float(s14_protect_top_score_share),
                        bad_mae_weight=float(s14_bad_mae_weight),
                        timeout_weight=float(s14_timeout_weight),
                        lower_tail_weight=float(s14_lower_tail_weight),
                    )
                    s14_path_quality_soft_risk_trim_score = _risk_trimmed_top_score(
                        s12_path_quality_ranker_soft_risk_score,
                        valid_metrics["side"],
                        calibrated_risk_predictions,
                        float(top_frac),
                        max_side_share=float(side_max_share),
                        trim_share=float(s14_trim_share),
                        protect_top_score_share=float(s14_protect_top_score_share),
                        bad_mae_weight=float(s14_bad_mae_weight),
                        timeout_weight=float(s14_timeout_weight),
                        lower_tail_weight=float(s14_lower_tail_weight),
                    )
                    s15_side_path_quality_ranker_side_capped_score = _side_capped_top_score(
                        s15_side_path_quality_ranker_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s15_side_path_quality_risk_trim_score = _risk_trimmed_top_score(
                        s15_side_path_quality_ranker_score,
                        valid_metrics["side"],
                        calibrated_risk_predictions,
                        float(top_frac),
                        max_side_share=float(side_max_share),
                        trim_share=float(s14_trim_share),
                        protect_top_score_share=float(s14_protect_top_score_share),
                        bad_mae_weight=float(s14_bad_mae_weight),
                        timeout_weight=float(s14_timeout_weight),
                        lower_tail_weight=float(s14_lower_tail_weight),
                    )
                    s16_discovery_path_quality_blend_side_capped_score = _side_capped_top_score(
                        s16_discovery_path_quality_blend_score,
                        valid_metrics["side"],
                        float(top_frac),
                        max_side_share=float(side_max_share),
                    )
                    s16_discovery_path_quality_risk_trim_score = _risk_trimmed_top_score(
                        s16_discovery_path_quality_blend_score,
                        valid_metrics["side"],
                        calibrated_risk_predictions,
                        float(top_frac),
                        max_side_share=float(side_max_share),
                        trim_share=float(s14_trim_share),
                        protect_top_score_share=float(s14_protect_top_score_share),
                        bad_mae_weight=float(s14_bad_mae_weight),
                        timeout_weight=float(s14_timeout_weight),
                        lower_tail_weight=float(s14_lower_tail_weight),
                    )
                    oracle_scores = {
                        "S0_raw_score": raw_score,
                        "S1_raw_side_balanced_score": raw_side_balanced_score,
                        "S2_side_calibrated_score": side_calibrated_score,
                        "S3_side_calibrated_soft_risk_score": side_calibrated_soft_risk_score,
                        "S4_side_calibrated_soft_risk_threshold_score": (
                            side_calibrated_soft_risk_threshold_score
                        ),
                        "S4b_side_calibrated_soft_risk_threshold_side_balanced_score": (
                            side_calibrated_soft_risk_threshold_side_balanced_score
                        ),
                        "S4_risk_cap_score": side_calibrated_soft_risk_bad_mae_cap_score,
                        "S4c_risk_cap_side_capped_score": (
                            side_calibrated_soft_risk_bad_mae_cap_side_capped_score
                        ),
                        "S5_risk_cap_threshold_select_score": (
                            side_calibrated_soft_risk_bad_mae_cap_threshold_side_capped_score
                        ),
                        "S6_lgbm_ranker_side_calibrated_risk_cap_score": (
                            ranker_side_calibrated_risk_cap_side_capped_score
                        ),
                        "S7a_lgbm_ranker_no_prefilter_score": (
                            s7a_ranker_no_prefilter_side_capped_score
                        ),
                        "S7b_lgbm_ranker_relaxed_risk_cap_score": (
                            s7b_ranker_relaxed_risk_cap_side_capped_score
                        ),
                        "S7c_side_specific_ranker_risk_cap_score": (
                            s7c_side_ranker_risk_cap_side_capped_score
                        ),
                        "S7d_oracle_enriched_ranker_risk_cap_score": (
                            s7d_enriched_ranker_risk_cap_side_capped_score
                        ),
                        "S7_two_stage_candidate_rerank_score": (
                            s7_two_stage_candidate_rerank_score
                        ),
                        "S8d_oracle_enriched_ranker_tight_bad_mae_score": (
                            s8d_enriched_ranker_tight_bad_mae_side_capped_score
                        ),
                        "S8_two_stage_tight_bad_mae_score": (
                            s8_two_stage_tight_bad_mae_score
                        ),
                        "S9d_oracle_enriched_ranker_calibrated_risk_cap_score": (
                            s9d_enriched_ranker_calibrated_risk_side_capped_score
                        ),
                        "S9_two_stage_calibrated_risk_cap_score": (
                            s9_two_stage_calibrated_risk_cap_score
                        ),
                        "S10_recall_soft_calibrated_risk_score": (
                            s10_recall_soft_calibrated_risk_side_capped_score
                        ),
                        "S10_recall_soft_calibrated_risk_loose_cap_score": (
                            s10_recall_soft_calibrated_risk_loose_cap_side_capped_score
                        ),
                        "S11_recall_tail_balanced_score": (
                            s11_recall_tail_balanced_side_capped_score
                        ),
                        "S12_path_quality_ranker_score": (
                            s12_path_quality_ranker_side_capped_score
                        ),
                        "S12_path_quality_ranker_soft_risk_score": (
                            s12_path_quality_ranker_soft_risk_side_capped_score
                        ),
                        "S13_constrained_path_quality_score": (
                            s13_constrained_path_quality_score
                        ),
                        "S13_constrained_path_quality_soft_risk_score": (
                            s13_constrained_path_quality_soft_risk_score
                        ),
                        "S14_path_quality_risk_trim_score": (
                            s14_path_quality_risk_trim_score
                        ),
                        "S14_path_quality_soft_risk_trim_score": (
                            s14_path_quality_soft_risk_trim_score
                        ),
                        "S15_side_path_quality_ranker_score": (
                            s15_side_path_quality_ranker_side_capped_score
                        ),
                        "S15_side_path_quality_risk_trim_score": (
                            s15_side_path_quality_risk_trim_score
                        ),
                        "S16_discovery_path_quality_blend_score": (
                            s16_discovery_path_quality_blend_side_capped_score
                        ),
                        "S16_discovery_path_quality_risk_trim_score": (
                            s16_discovery_path_quality_risk_trim_score
                        ),
                    }
                    selection_modes = {
                        "S4c_risk_cap_side_capped_score": "all_finite",
                        "S5_risk_cap_threshold_select_score": "all_finite",
                        "S6_lgbm_ranker_side_calibrated_risk_cap_score": "all_finite",
                        "S7a_lgbm_ranker_no_prefilter_score": "all_finite",
                        "S7b_lgbm_ranker_relaxed_risk_cap_score": "all_finite",
                        "S7c_side_specific_ranker_risk_cap_score": "all_finite",
                        "S7d_oracle_enriched_ranker_risk_cap_score": "all_finite",
                        "S7_two_stage_candidate_rerank_score": "all_finite",
                        "S8d_oracle_enriched_ranker_tight_bad_mae_score": "all_finite",
                        "S8_two_stage_tight_bad_mae_score": "all_finite",
                        "S9d_oracle_enriched_ranker_calibrated_risk_cap_score": "all_finite",
                        "S9_two_stage_calibrated_risk_cap_score": "all_finite",
                        "S10_recall_soft_calibrated_risk_score": "all_finite",
                        "S10_recall_soft_calibrated_risk_loose_cap_score": "all_finite",
                        "S11_recall_tail_balanced_score": "all_finite",
                        "S12_path_quality_ranker_score": "all_finite",
                        "S12_path_quality_ranker_soft_risk_score": "all_finite",
                        "S13_constrained_path_quality_score": "all_finite",
                        "S13_constrained_path_quality_soft_risk_score": "all_finite",
                        "S14_path_quality_risk_trim_score": "all_finite",
                        "S14_path_quality_soft_risk_trim_score": "all_finite",
                        "S15_side_path_quality_ranker_score": "all_finite",
                        "S15_side_path_quality_risk_trim_score": "all_finite",
                        "S16_discovery_path_quality_blend_score": "all_finite",
                        "S16_discovery_path_quality_risk_trim_score": "all_finite",
                    }
                    oracle_rows.extend(
                        _oracle_diagnostic_rows(
                            month=month,
                            label_arm=label_arm,
                            weight_arm=weight_arm,
                            top_frac=float(top_frac),
                            valid=valid,
                            metrics=valid_metrics,
                            target=target_valid,
                            clusters=valid_clusters,
                            scores=oracle_scores,
                            selection_modes=selection_modes,
                            min_group_rows=int(oracle_min_group_rows),
                        )
                    )
                    stage_gate_rows.extend(
                        _stage_gate_diagnostic_rows(
                            month=month,
                            label_arm=label_arm,
                            weight_arm=weight_arm,
                            top_frac=float(top_frac),
                            frame=valid,
                            metrics=valid_metrics,
                            stage_masks={
                                "raw_all": np.ones(len(valid_metrics), dtype=bool),
                                "finite_raw_score": pd.to_numeric(raw_score, errors="coerce").notna().to_numpy(),
                                "strict_risk_cap_pass": strict_risk_mask,
                                "relaxed_risk_cap_pass": relaxed_risk_mask,
                                f"stageA_top{int(s7_candidate_alt_top_n)}_discovery_pre_risk": (
                                    stage_a_top_alt_pre_risk_mask
                                ),
                                f"stageA_top{int(s7_candidate_top_n)}_discovery_pre_risk": (
                                    stage_a_top_primary_pre_risk_mask
                                ),
                                f"stageA_top{float(s7_candidate_top_frac):.2f}_frac_discovery_pre_risk": (
                                    stage_a_top_frac_pre_risk_mask
                                ),
                                "stageA_candidate_union_pre_risk": stage_a_candidate_pre_risk_mask,
                                f"stageA_top{int(s7_candidate_alt_top_n)}_discovery": stage_a_top_alt_mask,
                                f"stageA_top{int(s7_candidate_top_n)}_discovery": stage_a_top_primary_mask,
                                f"stageA_top{float(s7_candidate_top_frac):.2f}_frac_discovery": stage_a_top_frac_mask,
                                "stageA_candidate_union_relaxed_risk": stage_a_candidate_mask,
                                "stageA_candidate_union": stage_a_candidate_mask,
                                "ranker_top20": _per_timestamp_top_mask(
                                    valid,
                                    ranker_side_calibrated_score,
                                    top_n=20,
                                ),
                                "ranker_top10": _per_timestamp_top_mask(
                                    valid,
                                    ranker_side_calibrated_score,
                                    top_n=10,
                                ),
                                "ranker_top3": _per_timestamp_top_mask(
                                    valid,
                                    ranker_side_calibrated_score,
                                    top_n=3,
                                ),
                                "final_S6": pd.to_numeric(
                                    ranker_side_calibrated_risk_cap_side_capped_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S7a": pd.to_numeric(
                                    s7a_ranker_no_prefilter_side_capped_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S7b": pd.to_numeric(
                                    s7b_ranker_relaxed_risk_cap_side_capped_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S7c": pd.to_numeric(
                                    s7c_side_ranker_risk_cap_side_capped_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S7d": pd.to_numeric(
                                    s7d_enriched_ranker_risk_cap_side_capped_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S7_two_stage": pd.to_numeric(
                                    s7_two_stage_candidate_rerank_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S8d": pd.to_numeric(
                                    s8d_enriched_ranker_tight_bad_mae_side_capped_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S8_two_stage": pd.to_numeric(
                                    s8_two_stage_tight_bad_mae_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S9d": pd.to_numeric(
                                    s9d_enriched_ranker_calibrated_risk_side_capped_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S9_two_stage": pd.to_numeric(
                                    s9_two_stage_calibrated_risk_cap_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S10_soft": pd.to_numeric(
                                    s10_recall_soft_calibrated_risk_side_capped_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S10_loose_cap": pd.to_numeric(
                                    s10_recall_soft_calibrated_risk_loose_cap_side_capped_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S11_tail_balanced": pd.to_numeric(
                                    s11_recall_tail_balanced_side_capped_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S12_path_quality": pd.to_numeric(
                                    s12_path_quality_ranker_side_capped_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S12_path_quality_soft_risk": pd.to_numeric(
                                    s12_path_quality_ranker_soft_risk_side_capped_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S13_constrained_path_quality": pd.to_numeric(
                                    s13_constrained_path_quality_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S13_constrained_path_quality_soft_risk": pd.to_numeric(
                                    s13_constrained_path_quality_soft_risk_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S14_path_quality_risk_trim": pd.to_numeric(
                                    s14_path_quality_risk_trim_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S14_path_quality_soft_risk_trim": pd.to_numeric(
                                    s14_path_quality_soft_risk_trim_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S15_side_path_quality": pd.to_numeric(
                                    s15_side_path_quality_ranker_side_capped_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S15_side_path_quality_risk_trim": pd.to_numeric(
                                    s15_side_path_quality_risk_trim_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S16_discovery_path_quality_blend": pd.to_numeric(
                                    s16_discovery_path_quality_blend_side_capped_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                                "final_S16_discovery_path_quality_risk_trim": pd.to_numeric(
                                    s16_discovery_path_quality_risk_trim_score,
                                    errors="coerce",
                                ).notna().to_numpy(),
                            },
                        )
                    )
                    cluster_side_risk_side_balanced_score = _side_balanced_top_score(
                        cluster_side_risk_score,
                        valid_metrics["side"],
                        float(top_frac),
                    )
                    eligible_by_selector = {
                        "cluster_policy_score": eligible,
                        "cluster_policy_side_balanced_score": eligible,
                        "cluster_side_policy_score": side_eligible,
                        "cluster_side_policy_side_balanced_score": side_eligible,
                    }
                    selector_modes = {
                        "side_calibrated_soft_risk_bad_mae_cap_side_capped_score": "all_finite",
                        "side_calibrated_soft_risk_bad_mae_cap_threshold_select_score": "all_finite",
                        "lgbm_ranker_side_calibrated_risk_cap_score": "all_finite",
                        "s7a_lgbm_ranker_no_prefilter_score": "all_finite",
                        "s7b_lgbm_ranker_relaxed_risk_cap_score": "all_finite",
                        "s7c_side_specific_ranker_risk_cap_score": "all_finite",
                        "s7d_oracle_enriched_ranker_risk_cap_score": "all_finite",
                        "s7_two_stage_candidate_rerank_score": "all_finite",
                        "s8d_oracle_enriched_ranker_tight_bad_mae_score": "all_finite",
                        "s8_two_stage_tight_bad_mae_score": "all_finite",
                        "s9d_oracle_enriched_ranker_calibrated_risk_cap_score": "all_finite",
                        "s9_two_stage_calibrated_risk_cap_score": "all_finite",
                        "s10_recall_soft_calibrated_risk_score": "all_finite",
                        "s10_recall_soft_calibrated_risk_loose_cap_score": "all_finite",
                        "s11_recall_tail_balanced_score": "all_finite",
                        "s12_path_quality_ranker_score": "all_finite",
                        "s12_path_quality_ranker_soft_risk_score": "all_finite",
                        "s13_constrained_path_quality_score": "all_finite",
                        "s13_constrained_path_quality_soft_risk_score": "all_finite",
                        "s14_path_quality_risk_trim_score": "all_finite",
                        "s14_path_quality_soft_risk_trim_score": "all_finite",
                        "s15_side_path_quality_ranker_score": "all_finite",
                        "s15_side_path_quality_risk_trim_score": "all_finite",
                        "s16_discovery_path_quality_blend_score": "all_finite",
                        "s16_discovery_path_quality_risk_trim_score": "all_finite",
                    }
                    selection_by_selector: dict[str, list[dict[str, Any]]] = {}
                    for selector, score in (
                        ("raw_model_score", raw_score),
                        ("raw_side_balanced_score", raw_side_balanced_score),
                        ("risk_adjusted_score", risk_score),
                        ("risk_adjusted_side_balanced_score", risk_side_balanced_score),
                        ("side_calibrated_score", side_calibrated_score),
                        ("side_calibrated_side_balanced_score", side_calibrated_side_balanced_score),
                        ("side_calibrated_soft_risk_score", side_calibrated_soft_risk_score),
                        (
                            "side_calibrated_soft_risk_side_balanced_score",
                            side_calibrated_soft_risk_side_balanced_score,
                        ),
                        (
                            "side_calibrated_soft_risk_threshold_score",
                            side_calibrated_soft_risk_threshold_score,
                        ),
                        (
                            "side_calibrated_soft_risk_threshold_side_balanced_score",
                            side_calibrated_soft_risk_threshold_side_balanced_score,
                        ),
                        (
                            "side_calibrated_soft_risk_bad_mae_cap_score",
                            side_calibrated_soft_risk_bad_mae_cap_score,
                        ),
                        (
                            "side_calibrated_soft_risk_bad_mae_cap_side_capped_score",
                            side_calibrated_soft_risk_bad_mae_cap_side_capped_score,
                        ),
                        (
                            "side_calibrated_soft_risk_bad_mae_cap_threshold_select_score",
                            side_calibrated_soft_risk_bad_mae_cap_threshold_side_capped_score,
                        ),
                        (
                            "lgbm_ranker_side_calibrated_risk_cap_score",
                            ranker_side_calibrated_risk_cap_side_capped_score,
                        ),
                        (
                            "s7a_lgbm_ranker_no_prefilter_score",
                            s7a_ranker_no_prefilter_side_capped_score,
                        ),
                        (
                            "s7b_lgbm_ranker_relaxed_risk_cap_score",
                            s7b_ranker_relaxed_risk_cap_side_capped_score,
                        ),
                        (
                            "s7c_side_specific_ranker_risk_cap_score",
                            s7c_side_ranker_risk_cap_side_capped_score,
                        ),
                        (
                            "s7d_oracle_enriched_ranker_risk_cap_score",
                            s7d_enriched_ranker_risk_cap_side_capped_score,
                        ),
                        (
                            "s7_two_stage_candidate_rerank_score",
                            s7_two_stage_candidate_rerank_score,
                        ),
                        (
                            "s8d_oracle_enriched_ranker_tight_bad_mae_score",
                            s8d_enriched_ranker_tight_bad_mae_side_capped_score,
                        ),
                        (
                            "s8_two_stage_tight_bad_mae_score",
                            s8_two_stage_tight_bad_mae_score,
                        ),
                        (
                            "s9d_oracle_enriched_ranker_calibrated_risk_cap_score",
                            s9d_enriched_ranker_calibrated_risk_side_capped_score,
                        ),
                        (
                            "s9_two_stage_calibrated_risk_cap_score",
                            s9_two_stage_calibrated_risk_cap_score,
                        ),
                        (
                            "s10_recall_soft_calibrated_risk_score",
                            s10_recall_soft_calibrated_risk_side_capped_score,
                        ),
                        (
                            "s10_recall_soft_calibrated_risk_loose_cap_score",
                            s10_recall_soft_calibrated_risk_loose_cap_side_capped_score,
                        ),
                        (
                            "s11_recall_tail_balanced_score",
                            s11_recall_tail_balanced_side_capped_score,
                        ),
                        (
                            "s12_path_quality_ranker_score",
                            s12_path_quality_ranker_side_capped_score,
                        ),
                        (
                            "s12_path_quality_ranker_soft_risk_score",
                            s12_path_quality_ranker_soft_risk_side_capped_score,
                        ),
                        (
                            "s13_constrained_path_quality_score",
                            s13_constrained_path_quality_score,
                        ),
                        (
                            "s13_constrained_path_quality_soft_risk_score",
                            s13_constrained_path_quality_soft_risk_score,
                        ),
                        (
                            "s14_path_quality_risk_trim_score",
                            s14_path_quality_risk_trim_score,
                        ),
                        (
                            "s14_path_quality_soft_risk_trim_score",
                            s14_path_quality_soft_risk_trim_score,
                        ),
                        (
                            "s15_side_path_quality_ranker_score",
                            s15_side_path_quality_ranker_side_capped_score,
                        ),
                        (
                            "s15_side_path_quality_risk_trim_score",
                            s15_side_path_quality_risk_trim_score,
                        ),
                        (
                            "s16_discovery_path_quality_blend_score",
                            s16_discovery_path_quality_blend_side_capped_score,
                        ),
                        (
                            "s16_discovery_path_quality_risk_trim_score",
                            s16_discovery_path_quality_risk_trim_score,
                        ),
                        ("cluster_policy_score", gated_score),
                        ("cluster_policy_side_balanced_score", gated_side_balanced_score),
                        ("cluster_side_policy_score", side_gated_score),
                        ("cluster_side_policy_side_balanced_score", side_gated_side_balanced_score),
                        ("cluster_side_risk_adjusted_score", cluster_side_risk_score),
                        (
                            "cluster_side_risk_adjusted_side_balanced_score",
                            cluster_side_risk_side_balanced_score,
                        ),
                    ):
                        row = _selection_metrics(
                            frame=valid,
                            metrics=valid_metrics,
                            target=target_valid,
                            score=score,
                            arm=f"{label_arm}::{weight_arm}::{selector}",
                            selector=selector,
                            period=month,
                            top_frac=float(top_frac),
                            selection_mode=selector_modes.get(selector, "top_frac"),
                        )
                        _add_delta_fields(row, baseline)
                        row.update(
                            {
                                "label_arm": label_arm,
                                "weight_arm": weight_arm,
                                "cluster_policy": selector,
                                "side_balance_enforced": bool(selector.endswith("side_balanced_score")),
                                "side_cap_enforced": bool(
                                    "side_capped" in selector
                                    or selector.endswith("threshold_select_score")
                                    or selector.startswith("lgbm_ranker_")
                                    or selector.startswith("s7")
                                    or selector.startswith("s8")
                                    or selector.startswith("s9")
                                    or selector.startswith("s10")
                                    or selector.startswith("s11")
                                    or selector.startswith("s12")
                                    or selector.startswith("s13")
                                    or selector.startswith("s14")
                                    or selector.startswith("s15")
                                    or selector.startswith("s16")
                                ),
                                "model_feature_count": int(x_train_model.shape[1]),
                                "base_feature_count": int(len(features)),
                                "ae_gmm_model_feature_count": int(max(0, x_train_model.shape[1] - len(features))),
                                "bad_mae_lambda": float(bad_mae_lambda),
                                "timeout_lambda": float(timeout_lambda),
                                "lower_tail_lambda": float(lower_tail_lambda),
                                "cluster_adjustment_lambda": float(cluster_adjustment_lambda),
                                "calibrated_bad_mae_lambda": float(calibrated_bad_mae_lambda),
                                "calibrated_timeout_lambda": float(calibrated_timeout_lambda),
                                "calibrated_lower_tail_lambda": float(calibrated_lower_tail_lambda),
                                "gmm_entropy_lambda": float(gmm_entropy_lambda),
                                "gmm_recon_lambda": float(gmm_recon_lambda),
                                "gmm_mahal_lambda": float(gmm_mahal_lambda),
                                "min_calibrated_score": float(min_calibrated_score),
                                "max_pred_bad_mae": float(max_pred_bad_mae),
                                "max_pred_timeout": float(max_pred_timeout),
                                "max_pred_lower_tail": float(max_pred_lower_tail),
                                "side_max_share": float(side_max_share),
                                "ranker_status": ranker_status,
                                "side_ranker_status": side_ranker_status,
                                "enriched_ranker_status": enriched_ranker_status,
                                "path_quality_ranker_status": path_quality_ranker_status,
                                "side_path_quality_ranker_status": side_path_quality_ranker_status,
                                "score_finite_rows": int(pd.to_numeric(score, errors="coerce").notna().sum()),
                                "score_finite_frac": float(pd.to_numeric(score, errors="coerce").notna().mean()),
                                "no_trade_rate": float(1.0 - pd.to_numeric(score, errors="coerce").notna().mean()),
                                "score_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
                                "score_ic_label": _spearman(score, target_valid["target_soft"]),
                                **_selected_prediction_summary(
                                    score,
                                    float(top_frac),
                                    risk_predictions,
                                    selection_mode=selector_modes.get(selector, "top_frac"),
                                ),
                                **_oracle_recall_summary(
                                    score=score,
                                    oracle_score=valid_metrics["u_policy_net"],
                                    top_frac=float(top_frac),
                                    selection_mode=selector_modes.get(selector, "top_frac"),
                                ),
                                **_decile_diagnostics(score, valid_metrics["u_policy_net"]),
                            }
                        )
                        if selector in eligible_by_selector:
                            local_eligible = eligible_by_selector[selector]
                            row["cluster_policy_eligible_rows"] = int(local_eligible.sum())
                            row["cluster_policy_eligible_frac"] = float(local_eligible.mean())
                        monthly_rows.append(row)
                        selection_by_selector[selector] = _selected_cluster_rows(
                            month=month,
                            label_arm=label_arm,
                            weight_arm=weight_arm,
                            top_frac=float(top_frac),
                            selector=selector,
                            score=score,
                            clusters=valid_clusters,
                            metrics=valid_metrics,
                            policy=policy,
                            side_policy=side_policy,
                            selection_mode=selector_modes.get(selector, "top_frac"),
                        )
                        cluster_selection_rows.extend(selection_by_selector[selector])
                    raw_sel = pd.DataFrame(selection_by_selector.get("raw_model_score", []))
                    gated_sel = pd.DataFrame(selection_by_selector.get("cluster_policy_score", []))
                    cluster_diag = merged_cluster.copy()
                    cluster_diag.insert(0, "period", month)
                    cluster_diag.insert(1, "label_arm", label_arm)
                    cluster_diag.insert(2, "weight_arm", weight_arm)
                    cluster_diag.insert(3, "top_frac", float(top_frac))
                    for prefix, sel in (("raw_selected", raw_sel), ("gated_selected", gated_sel)):
                        if sel.empty:
                            continue
                        renamed = sel[
                            [
                                "cluster",
                                "selected_rows",
                                "selected_share",
                                "selected_mean_u",
                                "selected_hit_u",
                                "selected_long_rows",
                                "selected_short_rows",
                                "selected_long_mean_u",
                                "selected_short_mean_u",
                                "selected_bad_mae_1r_rate",
                                "selected_timeout_rate",
                            ]
                        ].rename(columns={c: f"{prefix}_{c}" for c in sel.columns if c != "cluster"})
                        cluster_diag = cluster_diag.merge(renamed, on="cluster", how="left")
                    cluster_diag_rows.extend(cluster_diag.to_dict(orient="records"))

    monthly = pd.DataFrame(monthly_rows)
    cluster_policy = pd.DataFrame(cluster_policy_rows)
    cluster_side_policy = pd.DataFrame(cluster_side_policy_rows)
    cluster_diag = pd.DataFrame(cluster_diag_rows)
    cluster_selection = pd.DataFrame(cluster_selection_rows)
    oracle_diagnostics = pd.DataFrame(oracle_rows)
    calibration_diagnostics = pd.DataFrame(calibration_rows)
    stage_gate_diagnostics = pd.DataFrame(stage_gate_rows)
    states = pd.DataFrame(state_rows)
    aggregate = (
        monthly.groupby(["arm", "label_arm", "weight_arm", "cluster_policy", "top_frac"], dropna=False, observed=True)
        .agg(
            months=("period", "nunique"),
            positive_months=("mean_u", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            mean_u=("mean_u", "mean"),
            worst_month_mean_u=("mean_u", "min"),
            hit_u=("hit_u", "mean"),
            q10_u=("q10_u", "mean"),
            weekly_mean_u_q10=("weekly_mean_u_q10", "mean"),
            worst_month_weekly_mean_u_q10=("weekly_mean_u_q10", "min"),
            worst_weekly_mean_u=("worst_weekly_mean_u", "min"),
            selected_week_count=("selected_week_count", "mean"),
            positive_week_rate=("positive_week_rate", "mean"),
            selected_rows=("selected_rows", "mean"),
            selected_long_share=("selected_long_share", "mean"),
            selected_short_share=("selected_short_share", "mean"),
            no_trade_rate=("no_trade_rate", "mean"),
            score_finite_frac=("score_finite_frac", "mean"),
            score_ic_u=("score_ic_u", "mean"),
            oracle_recall_at_model_top_k=("oracle_recall_at_model_top_k", "mean"),
            oracle_precision_at_model_top_k=("oracle_precision_at_model_top_k", "mean"),
            oracle_top_score_percentile_mean=("oracle_top_score_percentile_mean", "mean"),
            oracle_top_score_percentile_q10=("oracle_top_score_percentile_q10", "mean"),
            bad_mae_1r_rate=("bad_mae_1r_rate", "mean"),
            max_month_bad_mae_1r_rate=("bad_mae_1r_rate", "max"),
            timeout_rate=("timeout_rate", "mean"),
            max_month_timeout_rate=("timeout_rate", "max"),
        )
        .reset_index()
        if not monthly.empty
        else pd.DataFrame()
    )
    june_diagnosis = _june_diagnosis(cluster_selection, monthly)
    viability_matrix = _build_label_viability_matrix(
        aggregate=aggregate,
        calibration_diagnostics=calibration_diagnostics,
        stage_gate_diagnostics=stage_gate_diagnostics,
        evaluation_utility_source=str(evaluation_utility_source),
    )
    train_meta_readiness = _build_train_meta_readiness_matrix(
        viability_matrix,
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        evaluation_utility_source=str(evaluation_utility_source),
    )

    paths = {
        "monthly": output_dir / "gmm_cluster_policy_monthly.csv",
        "aggregate": output_dir / "gmm_cluster_policy_aggregate.csv",
        "label_viability_matrix": output_dir / "gmm_label_viability_matrix.csv",
        "train_meta_readiness": output_dir / "gmm_train_meta_readiness.csv",
        "cluster_policy": output_dir / "gmm_cluster_policy_table.csv",
        "cluster_side_policy": output_dir / "gmm_cluster_side_policy_table.csv",
        "cluster_diagnostics": output_dir / "gmm_cluster_policy_cluster_diagnostics.csv",
        "cluster_selection": output_dir / "gmm_cluster_policy_selected_cluster_attribution.csv",
        "oracle_diagnostics": output_dir / "gmm_oracle_vs_model_diagnostics.csv",
        "score_quantile_side_calibration": output_dir / "gmm_score_quantile_side_calibration.csv",
        "s7_stage_gate_diagnostics": output_dir / "gmm_s7_stage_gate_diagnostics.csv",
        "gmm_states": output_dir / "gmm_cluster_policy_gmm_states.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    viability_matrix.to_csv(paths["label_viability_matrix"], index=False)
    train_meta_readiness.to_csv(paths["train_meta_readiness"], index=False)
    cluster_policy.to_csv(paths["cluster_policy"], index=False)
    cluster_side_policy.to_csv(paths["cluster_side_policy"], index=False)
    cluster_diag.to_csv(paths["cluster_diagnostics"], index=False)
    cluster_selection.to_csv(paths["cluster_selection"], index=False)
    oracle_diagnostics.to_csv(paths["oracle_diagnostics"], index=False)
    calibration_diagnostics.to_csv(paths["score_quantile_side_calibration"], index=False)
    stage_gate_diagnostics.to_csv(paths["s7_stage_gate_diagnostics"], index=False)
    states.to_csv(paths["gmm_states"], index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "walk_forward_cluster_policy_smoke_not_production_policy",
        "labels_path": str(labels_path),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_count": int(len(features)),
        "feature_store": feature_store_report,
        "evaluation_utility_source": evaluation_utility_source,
        "label_arms": label_arms,
        "weight_arms": weight_arms,
        "top_fracs": [float(v) for v in top_fracs],
        "seeds": [int(v) for v in seeds],
        "train_lookback_months": train_lookback_months,
        "gmm_design": {
            "fit_scope": "prior_months_only_per_validation_month",
            "economic_targets": ["u_policy_net", "target_soft"],
            "side_balance": "required_per_cluster_during_hpo_when_side_available",
            "policy_layer": "cluster_and_cluster_x_side_smoke_policies",
            "cluster_candidates": cluster_candidates,
            "reg_covar_candidates": reg_covar_candidates,
            "smooth_lambda_candidates": smooth_lambda_candidates,
            "ae_max_iter": int(ae_max_iter),
            "max_train_rows": int(max_train_rows),
            "min_side_cluster_rows": int(min_side_cluster_rows),
            "min_side_cluster_frac": float(min_side_cluster_frac),
            "ranker_model_features": "feature_store_plus_prior_fit_ae_gmm_state_features",
            "risk_heads": ["bad_mae", "timeout", "lower_tail"],
            "lgbm_ranker": {
                "enabled": bool(enable_ranker),
                "available": bool(_LIGHTGBM_AVAILABLE),
                "group": "timestamp",
                "relevance": [
                    "within_timestamp_u_policy_net_quintile",
                    "oracle_enriched",
                    "path_quality",
                ],
            },
        },
        "pipeline_gate_plan": PIPELINE_GATE_PLAN,
        "label_viability_matrix": {
            "enabled": True,
            "output": str(paths["label_viability_matrix"]),
            "thresholds": DEFAULT_VIABILITY_THRESHOLDS,
            "active_label_rule": (
                "all hard gates pass: candidate discovery when stage-A diagnostics exist, "
                "learnability, economic utility, aggregate and monthly tail risk, bad-MAE calibration, "
                "cost robustness, temporal stability, side exposure, minimum exposure, final oracle "
                "recall, and final oracle recall stability"
            ),
            "active_rows": int(viability_matrix["active_label"].sum())
            if "active_label" in viability_matrix.columns
            else 0,
            "first_failed_gate_counts": (
                viability_matrix["first_failed_gate"].value_counts(dropna=False).to_dict()
                if "first_failed_gate" in viability_matrix.columns
                else {}
            ),
        },
        "train_base_meta_readiness": {
            "enabled": True,
            "output": str(paths["train_meta_readiness"]),
            "candidate_rows": int(len(train_meta_readiness)),
            "readiness_statuses": (
                train_meta_readiness["readiness_status"].value_counts(dropna=False).to_dict()
                if "readiness_status" in train_meta_readiness.columns
                else {}
            ),
            "is_final_promotion_ready": False,
            "required_next_checks": [
                "train_base_oof_learnability",
                "train_meta_oos_profitability",
                "simple_policy_optimiser_exit_policy",
                "frozen_threshold_replay",
                "leakage_and_feature_parity_audit",
            ],
        },
        "cluster_policy_thresholds": {
            "min_allow_mean_u": float(min_allow_mean_u),
            "min_allow_hit_u": float(min_allow_hit_u),
            "block_mean_u": float(block_mean_u),
            "min_side_mean_u": float(min_side_mean_u),
        },
        "risk_adjusted_score": {
            "formula": (
                "utility_score - bad_mae_lambda*pred_bad_mae "
                "- timeout_lambda*pred_timeout - lower_tail_lambda*pred_lower_tail "
                "+ cluster_adjustment_lambda*cluster_side_policy_adjustment"
            ),
            "bad_mae_lambda": float(bad_mae_lambda),
            "timeout_lambda": float(timeout_lambda),
            "lower_tail_lambda": float(lower_tail_lambda),
            "cluster_adjustment_lambda": float(cluster_adjustment_lambda),
        },
        "side_calibrated_soft_risk_score": {
            "formula": (
                "side_calibrated_expected_utility "
                "- calibrated_bad_mae_lambda*pred_bad_mae "
                "- calibrated_timeout_lambda*pred_timeout "
                "- calibrated_lower_tail_lambda*pred_lower_tail "
                "- gmm_entropy_lambda*cluster_entropy_norm "
                "- gmm_recon_lambda*reconstruction_tail_score "
                "- gmm_mahal_lambda*mahalanobis_tail_score"
            ),
            "calibration_bins": int(calibration_bins),
            "calibration_min_bin_rows": int(calibration_min_bin_rows),
            "calibrated_bad_mae_lambda": float(calibrated_bad_mae_lambda),
            "calibrated_timeout_lambda": float(calibrated_timeout_lambda),
            "calibrated_lower_tail_lambda": float(calibrated_lower_tail_lambda),
            "gmm_entropy_lambda": float(gmm_entropy_lambda),
            "gmm_recon_lambda": float(gmm_recon_lambda),
            "gmm_mahal_lambda": float(gmm_mahal_lambda),
            "min_calibrated_score": float(min_calibrated_score),
        },
        "risk_constraints": {
            "formula": (
                "hard fail-closed eligibility: pred_bad_mae <= max_pred_bad_mae, "
                "pred_timeout <= max_pred_timeout, pred_lower_tail <= max_pred_lower_tail"
            ),
            "max_pred_bad_mae": float(max_pred_bad_mae),
            "max_pred_timeout": float(max_pred_timeout),
            "max_pred_lower_tail": float(max_pred_lower_tail),
            "side_max_share": float(side_max_share),
            "threshold_selection": "S5 selects all finite threshold-passing rows after risk and side caps",
        },
        "s7_two_stage_policy": {
            "enabled": True,
            "stage_a": {
                "candidate_score": "max_percentile(raw, side_calibrated, soft_risk, ranker, side_ranker, oracle_enriched_ranker)",
                "pre_risk_candidate_diagnostics": [
                    f"stageA_top{int(s7_candidate_alt_top_n)}_discovery_pre_risk",
                    f"stageA_top{int(s7_candidate_top_n)}_discovery_pre_risk",
                    f"stageA_top{float(s7_candidate_top_frac):.2f}_frac_discovery_pre_risk",
                    "stageA_candidate_union_pre_risk",
                ],
                "admitted_candidate_stage": "stageA_candidate_union",
                "relaxed_max_pred_bad_mae": float(s7_relaxed_max_pred_bad_mae),
                "relaxed_max_pred_timeout": float(s7_relaxed_max_pred_timeout),
                "relaxed_max_pred_lower_tail": float(s7_relaxed_max_pred_lower_tail),
                "candidate_top_n": int(s7_candidate_top_n),
                "candidate_alt_top_n": int(s7_candidate_alt_top_n),
                "candidate_top_frac": float(s7_candidate_top_frac),
            },
            "stage_b": {
                "reranker": "LGBMRanker score with side calibration, strict risk cap, side exposure cap",
                "final_selectors": [
                    "S7a_lgbm_ranker_no_prefilter_score",
                    "S7b_lgbm_ranker_relaxed_risk_cap_score",
                    "S7c_side_specific_ranker_risk_cap_score",
                    "S7d_oracle_enriched_ranker_risk_cap_score",
                    "S7_two_stage_candidate_rerank_score",
                ],
            },
            "stage_gate_output": str(paths["s7_stage_gate_diagnostics"]),
        },
        "s8_tail_risk_repair_policy": {
            "enabled": True,
            "intent": (
                "first-failing-gate repair: tighten final bad-MAE/timeout/lower-tail caps "
                "around the oracle-enriched ranker and two-stage candidate pool"
            ),
            "tight_max_pred_bad_mae": float(s8_tight_max_pred_bad_mae),
            "tight_max_pred_timeout": float(s8_tight_max_pred_timeout),
            "tight_max_pred_lower_tail": float(s8_tight_max_pred_lower_tail),
            "selectors": [
                "S8d_oracle_enriched_ranker_tight_bad_mae_score",
                "S8_two_stage_tight_bad_mae_score",
            ],
        },
        "s9_calibrated_risk_policy": {
            "enabled": True,
            "intent": (
                "risk-head calibration repair: map raw risk predictions to side-specific train-window "
                "realized rates before hard bad-MAE/timeout/lower-tail caps"
            ),
            "calibrated_max_pred_bad_mae": float(s9_calibrated_max_pred_bad_mae),
            "calibrated_max_pred_timeout": float(s9_calibrated_max_pred_timeout),
            "calibrated_max_pred_lower_tail": float(s9_calibrated_max_pred_lower_tail),
            "selectors": [
                "S9d_oracle_enriched_ranker_calibrated_risk_cap_score",
                "S9_two_stage_calibrated_risk_cap_score",
            ],
        },
        "s10_recall_preserving_calibrated_risk_policy": {
            "enabled": True,
            "intent": (
                "final-oracle-recall repair: keep the high-recall Stage-A candidate pool finite "
                "and rank it with oracle-enriched expected utility plus discovery bonus minus "
                "side-calibrated risk penalties"
            ),
            "bad_mae_lambda": float(s10_bad_mae_lambda),
            "timeout_lambda": float(s10_timeout_lambda),
            "lower_tail_lambda": float(s10_lower_tail_lambda),
            "discovery_lambda": float(s10_discovery_lambda),
            "loose_max_pred_bad_mae": float(s10_loose_max_pred_bad_mae),
            "loose_max_pred_timeout": float(s10_loose_max_pred_timeout),
            "loose_max_pred_lower_tail": float(s10_loose_max_pred_lower_tail),
            "selectors": [
                "S10_recall_soft_calibrated_risk_score",
                "S10_recall_soft_calibrated_risk_loose_cap_score",
            ],
        },
        "s11_tail_balanced_recall_policy": {
            "enabled": True,
            "intent": (
                "narrow repair after S10: keep Stage-A recall pressure but increase calibrated "
                "bad-MAE/timeout/lower-tail penalties to try to pass the tail-risk gate"
            ),
            "bad_mae_lambda": float(s11_bad_mae_lambda),
            "timeout_lambda": float(s11_timeout_lambda),
            "lower_tail_lambda": float(s11_lower_tail_lambda),
            "discovery_lambda": float(s11_discovery_lambda),
            "selectors": ["S11_recall_tail_balanced_score"],
        },
        "s12_path_quality_ranker_policy": {
            "enabled": True,
            "intent": (
                "path-quality repair: train a timestamp-grouped ranker with direct bad-MAE, "
                "timeout, and lower-tail penalties, then rerank the same high-recall Stage-A pool"
            ),
            "relevance_mode": "path_quality",
            "bad_mae_lambda": float(s12_bad_mae_lambda),
            "timeout_lambda": float(s12_timeout_lambda),
            "lower_tail_lambda": float(s12_lower_tail_lambda),
            "discovery_lambda": float(s12_discovery_lambda),
            "selectors": [
                "S12_path_quality_ranker_score",
                "S12_path_quality_ranker_soft_risk_score",
            ],
        },
        "s13_constrained_path_quality_policy": {
            "enabled": True,
            "intent": (
                "tail-risk repair after S12: sort the high-recall path-quality pool by S12 score, "
                "admit calibrated low-risk rows first, then allow only limited relaxed-risk "
                "backfill so oracle recall and exposure cannot be preserved by unrestricted bad paths"
            ),
            "primary_max_pred_bad_mae": float(s13_primary_max_pred_bad_mae),
            "primary_max_pred_timeout": float(s13_primary_max_pred_timeout),
            "primary_max_pred_lower_tail": float(s13_primary_max_pred_lower_tail),
            "backfill_max_pred_bad_mae": float(s13_backfill_max_pred_bad_mae),
            "backfill_max_pred_timeout": float(s13_backfill_max_pred_timeout),
            "backfill_max_pred_lower_tail": float(s13_backfill_max_pred_lower_tail),
            "max_backfill_share": float(s13_max_backfill_share),
            "calibration_evidence": {
                "S13_constrained_path_quality_score": "S12_path_quality_ranker_score",
                "S13_constrained_path_quality_soft_risk_score": (
                    "S12_path_quality_ranker_soft_risk_score"
                ),
            },
            "selectors": [
                "S13_constrained_path_quality_score",
                "S13_constrained_path_quality_soft_risk_score",
            ],
        },
        "s14_path_quality_risk_trim_policy": {
            "enabled": True,
            "intent": (
                "gentler tail-risk repair after S13: keep the S12 path-quality top bucket, "
                "protect the highest score rows, then trim only the worst calibrated path-risk "
                "tail from the lower-ranked selected rows"
            ),
            "trim_share": float(s14_trim_share),
            "protect_top_score_share": float(s14_protect_top_score_share),
            "bad_mae_weight": float(s14_bad_mae_weight),
            "timeout_weight": float(s14_timeout_weight),
            "lower_tail_weight": float(s14_lower_tail_weight),
            "calibration_evidence": {
                "S14_path_quality_risk_trim_score": "S12_path_quality_ranker_score",
                "S14_path_quality_soft_risk_trim_score": "S12_path_quality_ranker_soft_risk_score",
            },
            "selectors": [
                "S14_path_quality_risk_trim_score",
                "S14_path_quality_soft_risk_trim_score",
            ],
        },
        "s15_side_path_quality_policy": {
            "enabled": True,
            "intent": (
                "side-ranking repair after S14: train separate long and short path-quality "
                "rankers on the same prior-only folds, rerank the high-recall Stage-A pool, "
                "then compare direct side-capped and S14-style risk-trimmed final selections"
            ),
            "relevance_mode": "path_quality",
            "trim_share": float(s14_trim_share),
            "protect_top_score_share": float(s14_protect_top_score_share),
            "bad_mae_weight": float(s14_bad_mae_weight),
            "timeout_weight": float(s14_timeout_weight),
            "lower_tail_weight": float(s14_lower_tail_weight),
            "calibration_evidence": {
                "S15_side_path_quality_risk_trim_score": "S15_side_path_quality_ranker_score"
            },
            "selectors": [
                "S15_side_path_quality_ranker_score",
                "S15_side_path_quality_risk_trim_score",
            ],
        },
        "s16_discovery_path_quality_blend_policy": {
            "enabled": True,
            "intent": (
                "long-oracle recall repair after S15: blend the high-recall Stage-A discovery "
                "percentile with global and side-specific path-quality rankers, then compare "
                "direct side-capped and S14-style risk-trimmed final selections"
            ),
            "score": "max_percentile(discovery_score, path_quality_ranker, side_path_quality_ranker) inside Stage-A",
            "trim_share": float(s14_trim_share),
            "protect_top_score_share": float(s14_protect_top_score_share),
            "bad_mae_weight": float(s14_bad_mae_weight),
            "timeout_weight": float(s14_timeout_weight),
            "lower_tail_weight": float(s14_lower_tail_weight),
            "calibration_evidence": {
                "S16_discovery_path_quality_risk_trim_score": "S16_discovery_path_quality_blend_score"
            },
            "selectors": [
                "S16_discovery_path_quality_blend_score",
                "S16_discovery_path_quality_risk_trim_score",
            ],
        },
        "oracle_diagnostics": {
            "enabled": True,
            "dimensions": ["all", "side", "cluster", "cluster_side", "symbol"],
            "recall_metrics": [
                "oracle_recall_at_model_top_k",
                "oracle_precision_at_model_top_k",
                "oracle_top_score_percentile_mean",
                "oracle_top_score_percentile_q10",
            ],
            "selectors": [
                "oracle_utility",
                "random_expected",
                "S0_raw_score",
                "S1_raw_side_balanced_score",
                "S2_side_calibrated_score",
                "S3_side_calibrated_soft_risk_score",
                "S4_side_calibrated_soft_risk_threshold_score",
                "S4b_side_calibrated_soft_risk_threshold_side_balanced_score",
                "S4_risk_cap_score",
                "S4c_risk_cap_side_capped_score",
                "S5_risk_cap_threshold_select_score",
                "S6_lgbm_ranker_side_calibrated_risk_cap_score",
                "S7a_lgbm_ranker_no_prefilter_score",
                "S7b_lgbm_ranker_relaxed_risk_cap_score",
                "S7c_side_specific_ranker_risk_cap_score",
                "S7d_oracle_enriched_ranker_risk_cap_score",
                "S7_two_stage_candidate_rerank_score",
                "S8d_oracle_enriched_ranker_tight_bad_mae_score",
                "S8_two_stage_tight_bad_mae_score",
                "S9d_oracle_enriched_ranker_calibrated_risk_cap_score",
                "S9_two_stage_calibrated_risk_cap_score",
                "S10_recall_soft_calibrated_risk_score",
                "S10_recall_soft_calibrated_risk_loose_cap_score",
                "S11_recall_tail_balanced_score",
                "S12_path_quality_ranker_score",
                "S12_path_quality_ranker_soft_risk_score",
                "S13_constrained_path_quality_score",
                "S13_constrained_path_quality_soft_risk_score",
                "S14_path_quality_risk_trim_score",
                "S14_path_quality_soft_risk_trim_score",
                "S15_side_path_quality_ranker_score",
                "S15_side_path_quality_risk_trim_score",
                "S16_discovery_path_quality_blend_score",
                "S16_discovery_path_quality_risk_trim_score",
            ],
            "min_group_rows": int(oracle_min_group_rows),
        },
        "score_quantile_side_calibration": {
            "enabled": True,
            "output": str(paths["score_quantile_side_calibration"]),
            "selectors": [
                "S0_raw_score",
                "S2_side_calibrated_score",
                "S3_side_calibrated_soft_risk_score",
                "S4_risk_cap_score",
                "S5_risk_cap_threshold_score",
                "S6_lgbm_ranker_side_calibrated_risk_cap_score",
                "S7a_lgbm_ranker_no_prefilter_score",
                "S7b_lgbm_ranker_relaxed_risk_cap_score",
                "S7c_side_specific_ranker_risk_cap_score",
                "S7d_oracle_enriched_ranker_risk_cap_score",
                "S7_two_stage_candidate_rerank_score",
                "S8d_oracle_enriched_ranker_tight_bad_mae_score",
                "S8_two_stage_tight_bad_mae_score",
                "S9d_oracle_enriched_ranker_calibrated_risk_cap_score",
                "S9_two_stage_calibrated_risk_cap_score",
                "S10_recall_soft_calibrated_risk_score",
                "S10_recall_soft_calibrated_risk_loose_cap_score",
                "S11_recall_tail_balanced_score",
                "S12_path_quality_ranker_score",
                "S12_path_quality_ranker_soft_risk_score",
                "S13_constrained_path_quality_score: inherits S12 calibration evidence",
                "S13_constrained_path_quality_soft_risk_score: inherits S12 soft-risk calibration evidence",
                "S14_path_quality_risk_trim_score: inherits S12 calibration evidence",
                "S14_path_quality_soft_risk_trim_score: inherits S12 soft-risk calibration evidence",
                "S15_side_path_quality_ranker_score",
                "S15_side_path_quality_risk_trim_score: inherits S15 calibration evidence",
                "S16_discovery_path_quality_blend_score",
                "S16_discovery_path_quality_risk_trim_score: inherits S16 calibration evidence",
            ],
            "dimensions": ["period", "side_name", "score_quantile"],
        },
        "s7_stage_gate_diagnostics": {
            "enabled": True,
            "output": str(paths["s7_stage_gate_diagnostics"]),
            "stages": [
                "raw_all",
                "finite_raw_score",
                "strict_risk_cap_pass",
                "relaxed_risk_cap_pass",
                f"stageA_top{int(s7_candidate_alt_top_n)}_discovery_pre_risk",
                f"stageA_top{int(s7_candidate_top_n)}_discovery_pre_risk",
                f"stageA_top{float(s7_candidate_top_frac):.2f}_frac_discovery_pre_risk",
                "stageA_candidate_union_pre_risk",
                "stageA_candidate_union_relaxed_risk",
                "stageA_candidate_union",
                "ranker_top20",
                "ranker_top10",
                "ranker_top3",
                "final_S6",
                "final_S7a",
                "final_S7b",
                "final_S7c",
                "final_S7d",
                "final_S7_two_stage",
                "final_S8d",
                "final_S8_two_stage",
                "final_S9d",
                "final_S9_two_stage",
                "final_S10_soft",
                "final_S10_loose_cap",
                "final_S11_tail_balanced",
                "final_S12_path_quality",
                "final_S12_path_quality_soft_risk",
                "final_S13_constrained_path_quality",
                "final_S13_constrained_path_quality_soft_risk",
                "final_S14_path_quality_risk_trim",
                "final_S14_path_quality_soft_risk_trim",
                "final_S15_side_path_quality",
                "final_S15_side_path_quality_risk_trim",
                "final_S16_discovery_path_quality_blend",
                "final_S16_discovery_path_quality_risk_trim",
            ],
        },
        "june_diagnosis": june_diagnosis,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_strict_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--evaluation-utility-column", default="__u_econ_net__")
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--label-arms", default="OPTIMIZED_ECONOMIC_TARGET")
    parser.add_argument("--weight-arms", default="W0_base")
    parser.add_argument("--seeds", default="17,29")
    parser.add_argument("--top-fracs", default="0.05,0.03,0.01")
    parser.add_argument("--train-lookback-months", type=int, default=2)
    parser.add_argument("--cluster-candidates", default="2,3,4,5,6")
    parser.add_argument("--reg-covar-candidates", default="1e-4,3e-4,1e-3")
    parser.add_argument("--smooth-lambda-candidates", default="0.5,0.8,0.925")
    parser.add_argument("--ae-max-iter", type=int, default=8)
    parser.add_argument("--max-train-rows", type=int, default=4000)
    parser.add_argument("--min-side-cluster-rows", type=int, default=10)
    parser.add_argument("--min-side-cluster-frac", type=float, default=0.02)
    parser.add_argument("--min-allow-mean-u", type=float, default=0.00025)
    parser.add_argument("--min-allow-hit-u", type=float, default=0.40)
    parser.add_argument("--block-mean-u", type=float, default=-0.00025)
    parser.add_argument("--min-side-mean-u", type=float, default=-0.00050)
    parser.add_argument("--bad-mae-lambda", type=float, default=0.10)
    parser.add_argument("--timeout-lambda", type=float, default=0.08)
    parser.add_argument("--lower-tail-lambda", type=float, default=0.12)
    parser.add_argument("--cluster-adjustment-lambda", type=float, default=0.01)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--calibration-min-bin-rows", type=int, default=250)
    parser.add_argument("--calibrated-bad-mae-lambda", type=float, default=0.0010)
    parser.add_argument("--calibrated-timeout-lambda", type=float, default=0.0008)
    parser.add_argument("--calibrated-lower-tail-lambda", type=float, default=0.0012)
    parser.add_argument("--gmm-entropy-lambda", type=float, default=0.00035)
    parser.add_argument("--gmm-recon-lambda", type=float, default=0.00035)
    parser.add_argument("--gmm-mahal-lambda", type=float, default=0.00035)
    parser.add_argument("--min-calibrated-score", type=float, default=0.0)
    parser.add_argument("--max-pred-bad-mae", type=float, default=0.65)
    parser.add_argument("--max-pred-timeout", type=float, default=0.10)
    parser.add_argument("--max-pred-lower-tail", type=float, default=0.20)
    parser.add_argument("--s7-relaxed-max-pred-bad-mae", type=float, default=0.85)
    parser.add_argument("--s7-relaxed-max-pred-timeout", type=float, default=0.25)
    parser.add_argument("--s7-relaxed-max-pred-lower-tail", type=float, default=0.45)
    parser.add_argument("--s7-candidate-top-n", type=int, default=50)
    parser.add_argument("--s7-candidate-alt-top-n", type=int, default=20)
    parser.add_argument("--s7-candidate-top-frac", type=float, default=0.10)
    parser.add_argument("--s8-tight-max-pred-bad-mae", type=float, default=0.55)
    parser.add_argument("--s8-tight-max-pred-timeout", type=float, default=0.10)
    parser.add_argument("--s8-tight-max-pred-lower-tail", type=float, default=0.20)
    parser.add_argument("--s9-calibrated-max-pred-bad-mae", type=float, default=0.50)
    parser.add_argument("--s9-calibrated-max-pred-timeout", type=float, default=0.12)
    parser.add_argument("--s9-calibrated-max-pred-lower-tail", type=float, default=0.20)
    parser.add_argument("--s10-bad-mae-lambda", type=float, default=0.0040)
    parser.add_argument("--s10-timeout-lambda", type=float, default=0.0020)
    parser.add_argument("--s10-lower-tail-lambda", type=float, default=0.0030)
    parser.add_argument("--s10-discovery-lambda", type=float, default=0.0020)
    parser.add_argument("--s10-loose-max-pred-bad-mae", type=float, default=0.65)
    parser.add_argument("--s10-loose-max-pred-timeout", type=float, default=0.20)
    parser.add_argument("--s10-loose-max-pred-lower-tail", type=float, default=0.35)
    parser.add_argument("--s11-bad-mae-lambda", type=float, default=0.0065)
    parser.add_argument("--s11-timeout-lambda", type=float, default=0.0025)
    parser.add_argument("--s11-lower-tail-lambda", type=float, default=0.0035)
    parser.add_argument("--s11-discovery-lambda", type=float, default=0.0020)
    parser.add_argument("--s12-bad-mae-lambda", type=float, default=0.0035)
    parser.add_argument("--s12-timeout-lambda", type=float, default=0.0020)
    parser.add_argument("--s12-lower-tail-lambda", type=float, default=0.0030)
    parser.add_argument("--s12-discovery-lambda", type=float, default=0.0015)
    parser.add_argument("--s13-primary-max-pred-bad-mae", type=float, default=0.50)
    parser.add_argument("--s13-primary-max-pred-timeout", type=float, default=0.12)
    parser.add_argument("--s13-primary-max-pred-lower-tail", type=float, default=0.20)
    parser.add_argument("--s13-backfill-max-pred-bad-mae", type=float, default=0.62)
    parser.add_argument("--s13-backfill-max-pred-timeout", type=float, default=0.18)
    parser.add_argument("--s13-backfill-max-pred-lower-tail", type=float, default=0.35)
    parser.add_argument("--s13-max-backfill-share", type=float, default=0.30)
    parser.add_argument("--s14-trim-share", type=float, default=0.12)
    parser.add_argument("--s14-protect-top-score-share", type=float, default=0.55)
    parser.add_argument("--s14-bad-mae-weight", type=float, default=1.00)
    parser.add_argument("--s14-timeout-weight", type=float, default=0.75)
    parser.add_argument("--s14-lower-tail-weight", type=float, default=0.75)
    parser.add_argument("--side-max-share", type=float, default=0.70)
    parser.add_argument("--disable-ranker", action="store_true")
    parser.add_argument("--oracle-min-group-rows", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_cluster_policy_smoke(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        evaluation_utility_column=args.evaluation_utility_column,
        max_feature_store_features=args.max_feature_store_features,
        label_arms=_parse_csv(args.label_arms, ("OPTIMIZED_ECONOMIC_TARGET",)),
        weight_arms=_parse_csv(args.weight_arms, ("W0_base",)),
        seeds=_parse_int_csv(args.seeds, (17, 29)),
        top_fracs=_parse_float_csv(args.top_fracs, (0.05, 0.03, 0.01)),
        train_lookback_months=args.train_lookback_months,
        cluster_candidates=args.cluster_candidates,
        reg_covar_candidates=args.reg_covar_candidates,
        smooth_lambda_candidates=args.smooth_lambda_candidates,
        ae_max_iter=int(args.ae_max_iter),
        max_train_rows=int(args.max_train_rows),
        min_side_cluster_rows=int(args.min_side_cluster_rows),
        min_side_cluster_frac=float(args.min_side_cluster_frac),
        min_allow_mean_u=float(args.min_allow_mean_u),
        min_allow_hit_u=float(args.min_allow_hit_u),
        block_mean_u=float(args.block_mean_u),
        min_side_mean_u=float(args.min_side_mean_u),
        bad_mae_lambda=float(args.bad_mae_lambda),
        timeout_lambda=float(args.timeout_lambda),
        lower_tail_lambda=float(args.lower_tail_lambda),
        cluster_adjustment_lambda=float(args.cluster_adjustment_lambda),
        calibration_bins=int(args.calibration_bins),
        calibration_min_bin_rows=int(args.calibration_min_bin_rows),
        calibrated_bad_mae_lambda=float(args.calibrated_bad_mae_lambda),
        calibrated_timeout_lambda=float(args.calibrated_timeout_lambda),
        calibrated_lower_tail_lambda=float(args.calibrated_lower_tail_lambda),
        gmm_entropy_lambda=float(args.gmm_entropy_lambda),
        gmm_recon_lambda=float(args.gmm_recon_lambda),
        gmm_mahal_lambda=float(args.gmm_mahal_lambda),
        min_calibrated_score=float(args.min_calibrated_score),
        max_pred_bad_mae=float(args.max_pred_bad_mae),
        max_pred_timeout=float(args.max_pred_timeout),
        max_pred_lower_tail=float(args.max_pred_lower_tail),
        s7_relaxed_max_pred_bad_mae=float(args.s7_relaxed_max_pred_bad_mae),
        s7_relaxed_max_pred_timeout=float(args.s7_relaxed_max_pred_timeout),
        s7_relaxed_max_pred_lower_tail=float(args.s7_relaxed_max_pred_lower_tail),
        s7_candidate_top_n=int(args.s7_candidate_top_n),
        s7_candidate_alt_top_n=int(args.s7_candidate_alt_top_n),
        s7_candidate_top_frac=float(args.s7_candidate_top_frac),
        s8_tight_max_pred_bad_mae=float(args.s8_tight_max_pred_bad_mae),
        s8_tight_max_pred_timeout=float(args.s8_tight_max_pred_timeout),
        s8_tight_max_pred_lower_tail=float(args.s8_tight_max_pred_lower_tail),
        s9_calibrated_max_pred_bad_mae=float(args.s9_calibrated_max_pred_bad_mae),
        s9_calibrated_max_pred_timeout=float(args.s9_calibrated_max_pred_timeout),
        s9_calibrated_max_pred_lower_tail=float(args.s9_calibrated_max_pred_lower_tail),
        s10_bad_mae_lambda=float(args.s10_bad_mae_lambda),
        s10_timeout_lambda=float(args.s10_timeout_lambda),
        s10_lower_tail_lambda=float(args.s10_lower_tail_lambda),
        s10_discovery_lambda=float(args.s10_discovery_lambda),
        s10_loose_max_pred_bad_mae=float(args.s10_loose_max_pred_bad_mae),
        s10_loose_max_pred_timeout=float(args.s10_loose_max_pred_timeout),
        s10_loose_max_pred_lower_tail=float(args.s10_loose_max_pred_lower_tail),
        s11_bad_mae_lambda=float(args.s11_bad_mae_lambda),
        s11_timeout_lambda=float(args.s11_timeout_lambda),
        s11_lower_tail_lambda=float(args.s11_lower_tail_lambda),
        s11_discovery_lambda=float(args.s11_discovery_lambda),
        s12_bad_mae_lambda=float(args.s12_bad_mae_lambda),
        s12_timeout_lambda=float(args.s12_timeout_lambda),
        s12_lower_tail_lambda=float(args.s12_lower_tail_lambda),
        s12_discovery_lambda=float(args.s12_discovery_lambda),
        s13_primary_max_pred_bad_mae=float(args.s13_primary_max_pred_bad_mae),
        s13_primary_max_pred_timeout=float(args.s13_primary_max_pred_timeout),
        s13_primary_max_pred_lower_tail=float(args.s13_primary_max_pred_lower_tail),
        s13_backfill_max_pred_bad_mae=float(args.s13_backfill_max_pred_bad_mae),
        s13_backfill_max_pred_timeout=float(args.s13_backfill_max_pred_timeout),
        s13_backfill_max_pred_lower_tail=float(args.s13_backfill_max_pred_lower_tail),
        s13_max_backfill_share=float(args.s13_max_backfill_share),
        s14_trim_share=float(args.s14_trim_share),
        s14_protect_top_score_share=float(args.s14_protect_top_score_share),
        s14_bad_mae_weight=float(args.s14_bad_mae_weight),
        s14_timeout_weight=float(args.s14_timeout_weight),
        s14_lower_tail_weight=float(args.s14_lower_tail_weight),
        side_max_share=float(args.side_max_share),
        enable_ranker=not bool(args.disable_ranker),
        oracle_min_group_rows=int(args.oracle_min_group_rows),
    )
    print(json.dumps(_strict_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
