#!/usr/bin/env python3
"""Cheap feature-store model smoke for soft-label/sample-weight candidates.

This is still a pre-training screen: it does not run the production LightGBM
pipeline, Optuna, or policy geometry. It uses the existing label ledger, joins
the candidate feature-store columns, trains a small month-forward tree model,
and evaluates whether predicted top buckets are profitable after policy costs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

try:
    from lightgbm import LGBMClassifier, LGBMRanker

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - optional smoke dependency
    LGBMClassifier = None
    LGBMRanker = None
    _LIGHTGBM_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    AE_GMM_FEATURE_COLUMNS,
    fit_ae_gmm_state,
    load_ae_gmm_state_artifact,
    save_ae_gmm_state_artifact,
    transform_ae_gmm_features,
)
from extreme_price_movements.economic_target_optimizer import (  # noqa: E402
    EconomicTargetSpec,
    append_economic_target_columns,
)
from scripts.run_label_economic_proxy_ablation import LABEL_ARMS, _label_targets
from scripts.run_label_quality_proxy_diagnostics import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    TOP_FRACS,
    _decile_diagnostics,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _make_targets,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _spearman,
)
from scripts.run_label_weighted_proxy_ablation import WEIGHT_ARMS, _effective_sample_size, _weight_series
from scripts.run_label_weighted_proxy_ablation import PROXY_METHODS, _weighted_proxy_score
from scripts.run_soft_label_rounda_topk_proxy_diagnostics import (
    STRICT_LABEL_ARMS as STRICT_ROUNDA_LABEL_ARMS,
    _strict_rounda_targets,
)


def _average_rank_1d(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    n = int(len(values))
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    boundaries = np.flatnonzero(sorted_vals[1:] != sorted_vals[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, n]
    avg_ranks = 0.5 * (starts.astype(np.float64) + stops.astype(np.float64) - 1.0) + 1.0
    ranks_sorted = np.repeat(avg_ranks, stops - starts)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _spearman(x: Any, y: Any) -> float:
    x_arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if x_arr.shape[0] != y_arr.shape[0]:
        n = min(int(x_arr.shape[0]), int(y_arr.shape[0]))
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 5:
        return float("nan")
    x_valid = x_arr[mask]
    y_valid = y_arr[mask]
    if np.unique(x_valid).size < 2 or np.unique(y_valid).size < 2:
        return float("nan")
    xr = _average_rank_1d(x_valid)
    yr = _average_rank_1d(y_valid)
    xr = xr - float(np.mean(xr))
    yr = yr - float(np.mean(yr))
    denom = float(np.sqrt(np.sum(xr * xr) * np.sum(yr * yr)))
    if not math.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(np.sum(xr * yr) / denom)


def _finite_quantile_np(values: Any, q: float) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, float(q)))


def _rank_pct_np(values: Any, fill: float = 0.5) -> pd.Series:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    out = np.full(arr.shape[0], float(fill), dtype=np.float32)
    valid = np.isfinite(arr)
    if int(valid.sum()) >= 1:
        ranks = _average_rank_1d(arr[valid]) / float(valid.sum())
        out[valid] = ranks.astype(np.float32, copy=False)
    return pd.Series(out)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_feature_store_model_smoke_v1")
DEFAULT_TOP_FRACS = tuple(float(v) for v in TOP_FRACS)
DISCOVERY_CONTEXT_KEYWORDS = (
    "gmm",
    "cluster",
    "archetype",
    "posterior",
    "reconstruction",
    "latent",
    "state_spectral",
    "bars_in_high_vol_state",
    "ae_gmm_oof_available",
)
MAX_DISCOVERY_CONTEXT_COLUMNS = 128
DISCOVERY_CONTEXT_BUCKET_LIMITS = {
    "market_state": 24,
    "global_ae_gmm": 40,
    "long_ae_gmm": 32,
    "short_ae_gmm": 32,
}
DEFAULT_AE_GMM_STATE_FEATURE_MAX_TRAIN_ROWS = 15000
DEFAULT_AE_GMM_STATE_FEATURE_GMM_MAX_TRAIN_ROWS = 100000
DEFAULT_AE_GMM_STATE_FEATURE_MAX_ITER = 32
AE_GMM_SMOKE_FEATURE_POLICY = os.environ.get(
    "EPM_AE_GMM_SMOKE_FEATURE_POLICY",
    "all",
).strip().lower()
_AE_GMM_SIDE_CONTEXT_MODE_RAW = os.environ.get(
    "EPM_AE_GMM_SIDE_CONTEXT_MODE",
    "off",
).strip().lower()
if _AE_GMM_SIDE_CONTEXT_MODE_RAW in {"short_asset", "short_boll", "short_asset_short_boll"}:
    raise ValueError(
        "EPM_AE_GMM_SIDE_CONTEXT_MODE must use global long/short side context; "
        "set it to 'long_short' instead of legacy short-only context names."
    )
if _AE_GMM_SIDE_CONTEXT_MODE_RAW in {"1", "true", "yes", "y", "on", "long_short", "long_short_context"}:
    AE_GMM_SIDE_CONTEXT_MODE = "long_short"
elif _AE_GMM_SIDE_CONTEXT_MODE_RAW in {"", "0", "false", "no", "n", "off", "none"}:
    AE_GMM_SIDE_CONTEXT_MODE = "off"
else:
    raise ValueError(
        "Unsupported EPM_AE_GMM_SIDE_CONTEXT_MODE="
        f"{_AE_GMM_SIDE_CONTEXT_MODE_RAW!r}; expected 'off' or 'long_short'."
    )
AE_GMM_CROSSFIT_TRAIN_FEATURES = os.environ.get(
    "EPM_AE_GMM_CROSSFIT_TRAIN_FEATURES",
    "1",
).strip().lower() not in {"0", "false", "no", "off", "n"}
SOURCE_BUCKET_QUALITY_FEATURES = (
    "median_spread_bps",
    "log_quote_volume",
    "state_spectral_top3_reconstruction_error",
    "bars_in_high_vol_state_log_norm",
    "state_spectral_eig_gap_1_2",
    "q_iqr__bars_in_high_vol_state_log_norm",
    "q_tail_width__bars_in_high_vol_state_log_norm",
    "gmm_posterior_max",
    "gmm_posterior_margin",
    "gmm_entropy",
    "cluster_entropy_norm",
    "cluster_speed",
    "cluster_acceleration",
    "mahalanobis_distance",
    "min_mahalanobis",
    "expected_mahalanobis",
    "time_since_cluster_change",
    "rolling_cluster_stability",
    "cluster_flip_count_20",
    "long_gmm_posterior_max",
    "long_gmm_posterior_margin",
    "long_gmm_entropy",
    "long_cluster_entropy_norm",
    "long_cluster_speed",
    "long_cluster_acceleration",
    "long_mahalanobis_distance",
    "long_min_mahalanobis",
    "long_expected_mahalanobis",
    "long_time_since_cluster_change",
    "long_rolling_cluster_stability",
    "long_cluster_flip_count_20",
    "short_gmm_posterior_max",
    "short_gmm_posterior_margin",
    "short_gmm_entropy",
    "short_cluster_entropy_norm",
    "short_cluster_speed",
    "short_cluster_acceleration",
    "short_mahalanobis_distance",
    "short_min_mahalanobis",
    "short_expected_mahalanobis",
    "short_time_since_cluster_change",
    "short_rolling_cluster_stability",
    "short_cluster_flip_count_20",
)
DEFAULT_LABEL_ARMS = (
    "S10_policy_net_soft",
    "S14_policy_net_path_blend",
    "S16_tail_utility_soft",
    "S20_tail_risk_adjusted_soft",
    "S21_tail_margin_ts_rank",
    "S24_policy_tail_s14_lean",
    "S25_tail_fast_risk_soft",
    "S26_broad_policy_path_fast",
    "S27_tail_rank_risk_balanced",
    "S28_lowbarrier_broad_policy",
    "S29_lowbarrier_s24_blend",
    "S30_lowbarrier_tail_risk",
    "S31_clean_tail_economic",
    "S32_econ_limited_broad_policy",
    "S33_clean_margin_ts_rank",
    "S34_exec_guard_broad_policy",
    "S35_exec_margin_soft",
    "S36_exec_margin_ts_rank",
    "S37_lowdrawdown_tail_rank",
    "S38_conditional_clean_utility",
    "S39_conditional_clean_ts_rank",
    "S40_dirty_capped_broad_policy",
    "S41_lowmae_timeout_safe_tail",
    "S42_lowbarrier_lowmae_tail",
    "S43_lowbarrier_dirty_capped_broad",
    "S44_clean_masked_lowmae_rank",
    "S45_strict_clean_tail_rank",
    "S46_badmae_contrast_margin",
    "S47_dirty_capped_s41",
    "S52_timeout_barrier_cap_policy_soft",
    "S53_timeout_barrier_cap_path_blend",
    "S54_timeout_barrier_cap_clean_tail",
    "S55_timeout_barrier_cap_exec_guard",
    "S56_timeout_tpnet_cap_policy_soft",
    "S57_timeout_tpnet_cap_path_blend",
    "S58_timeout_tpnet_cap_clean_tail",
    "S59_timeout_tpnet_cap_exec_guard",
)
DEFAULT_WEIGHT_ARMS = (
    "W0_base",
    "W7_timestamp_balanced",
    "W11_tail_clean_utility",
    "W12_tail_timestamp_balanced",
    "W13_lowbarrier_timestamp",
)
FIXED_ARTIFACT_LABEL_ARMS = (
    "STAGE15_quiet_mid_clean_utility",
    "OPTIMIZED_ECONOMIC_TARGET",
    "OPTIMIZED_ECONOMIC_PATH_SAFE_TARGET",
    "OPTIMIZED_ECONOMIC_BAD_MAE_CONTRAST_TARGET",
    "OPTIMIZED_ECONOMIC_CLEAN_RANK_TARGET",
    "OPTIMIZED_ECONOMIC_TIMEOUT_SAFE_TARGET",
    "OPTIMIZED_ECONOMIC_STRICT_PATH_FIRST_TARGET",
    "OPTIMIZED_ECONOMIC_CLEAN_UTILITY_RANK_TARGET",
    "OPTIMIZED_ECONOMIC_PATH_FIRST_CLEAN_RELEVANCE_TARGET",
    "OPTIMIZED_ECONOMIC_S24_BROAD_PATH_FIRST_SOURCE_TARGET",
    "OPTIMIZED_ECONOMIC_TIMEOUT_AWARE_CLEAN_SOURCE_TARGET",
    "OPTIMIZED_ECONOMIC_EXEC_MARGIN_STABLE_TARGET",
    "OPTIMIZED_ECONOMIC_SIDEAWARE_TARGET",
    "OPTIMIZED_ECONOMIC_SIDE_RESOLUTION_TARGET",
    "OPTIMIZED_ECONOMIC_SIDEAWARE_EXEC_RESOLUTION_TARGET",
)

def _bounded_sigmoid(values: Any) -> pd.Series:
    arr = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return pd.Series(1.0 / (1.0 + np.exp(-arr)))


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    lowered = str(value).strip().lower()
    if lowered == "all":
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_int_csv(value: str | None, default: tuple[int, ...]) -> list[int]:
    if value is None or not str(value).strip():
        return list(default)
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _apply_spread_symbol_universe(
    frame: pd.DataFrame,
    *,
    spread_baseline_path: Path | None,
    spread_rank_column: str,
    target_symbol_count: int | None,
    max_spread_bps: float | None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if spread_baseline_path is None and target_symbol_count is None and max_spread_bps is None:
        return frame, {"enabled": False}, pd.DataFrame()
    if spread_baseline_path is None:
        raise ValueError("--spread-baseline-path is required for spread universe filtering")
    if "__symbol__" not in frame.columns:
        raise ValueError("Cannot apply spread universe filter: label frame is missing __symbol__")
    spread = pd.read_csv(spread_baseline_path)
    if "symbol" not in spread.columns:
        raise ValueError(f"Spread baseline is missing required column 'symbol': {spread_baseline_path}")
    if spread_rank_column not in spread.columns:
        raise ValueError(
            f"Spread baseline is missing rank column {spread_rank_column!r}: {spread_baseline_path}"
        )
    present_symbols = pd.Series(frame["__symbol__"].dropna().astype(str).unique(), name="symbol")
    universe = present_symbols.to_frame().merge(spread, on="symbol", how="left")
    universe["_spread_rank_value"] = pd.to_numeric(
        universe[spread_rank_column],
        errors="coerce",
    )
    universe = universe.sort_values(
        ["_spread_rank_value", "symbol"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)
    universe["available_rank_by_spread"] = np.arange(1, len(universe) + 1, dtype=np.int32)
    selected_mask = universe["_spread_rank_value"].notna()
    if max_spread_bps is not None:
        selected_mask &= universe["_spread_rank_value"].le(float(max_spread_bps))
    if target_symbol_count is not None and int(target_symbol_count) > 0:
        finite_order = universe.index[selected_mask].to_numpy(dtype=np.int64)
        keep = set(int(idx) for idx in finite_order[: int(target_symbol_count)])
        selected_mask = universe.index.to_series().isin(keep).to_numpy(dtype=bool)
    universe["selected"] = selected_mask.astype(bool)
    universe["exclusion_reason"] = ""
    universe.loc[universe["_spread_rank_value"].isna(), "exclusion_reason"] = "missing_spread"
    universe.loc[
        universe["_spread_rank_value"].notna() & ~universe["selected"],
        "exclusion_reason",
    ] = "spread_rank_excluded"
    selected_symbols = set(universe.loc[universe["selected"], "symbol"].astype(str))
    filtered = (
        frame.loc[frame["__symbol__"].astype(str).isin(selected_symbols)]
        .copy()
        .reset_index(drop=True)
    )
    report = {
        "enabled": True,
        "spread_baseline_path": str(spread_baseline_path),
        "spread_rank_column": str(spread_rank_column),
        "target_symbol_count": int(target_symbol_count) if target_symbol_count is not None else None,
        "max_spread_bps": float(max_spread_bps) if max_spread_bps is not None else None,
        "input_rows": int(len(frame)),
        "input_symbols": int(len(present_symbols)),
        "selected_rows": int(len(filtered)),
        "selected_symbols": int(len(selected_symbols)),
        "excluded_symbols": int(len(universe) - len(selected_symbols)),
        "selected_spread_max_bps": _safe_mean([universe.loc[universe["selected"], "_spread_rank_value"].max()]),
        "excluded_missing_spread_symbols": int(universe["_spread_rank_value"].isna().sum()),
    }
    return filtered, report, universe


def _baseline_row(valid_metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "period_baseline_mean_u": _safe_mean(valid_metrics["u_policy_net"]),
        "period_baseline_hit_u": _safe_mean(valid_metrics["u_policy_net"] > 0.0),
        "period_baseline_q10_u": _safe_quantile(valid_metrics["u_policy_net"], 0.10),
    }


def _add_delta_fields(row: dict[str, Any], baseline: dict[str, float]) -> None:
    mean_u = float(row["mean_u"])
    hit_u = float(row["hit_u"])
    q10_u = float(row["q10_u"])
    row.update(baseline)
    row["delta_mean_u_vs_period"] = (
        mean_u - baseline["period_baseline_mean_u"]
        if math.isfinite(mean_u) and math.isfinite(baseline["period_baseline_mean_u"])
        else float("nan")
    )
    row["delta_hit_u_vs_period"] = (
        hit_u - baseline["period_baseline_hit_u"]
        if math.isfinite(hit_u) and math.isfinite(baseline["period_baseline_hit_u"])
        else float("nan")
    )
    row["delta_q10_u_vs_period"] = (
        q10_u - baseline["period_baseline_q10_u"]
        if math.isfinite(q10_u) and math.isfinite(baseline["period_baseline_q10_u"])
        else float("nan")
    )


def _fixed_artifact_targets(frame: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    targets: dict[str, pd.DataFrame] = {}
    if "__y_econ_sideaware_soft__" not in frame.columns:
        try:
            sideaware_frame, _sideaware_summary = append_economic_target_columns(
                frame.reset_index(drop=True),
                EconomicTargetSpec(
                    name="label_feature_store_sideaware",
                    utility_source="policy_net",
                    cost=0.001,
                    margin=0.0005,
                    sl_buffer=0.1,
                    vol_source="barrier",
                    temperature=0.75,
                ),
                copy=True,
            )
            sideaware_frame.index = frame.index
        except Exception:
            sideaware_frame = pd.DataFrame(index=frame.index)
    else:
        sideaware_frame = frame
    if "__y_econ_sideaware_soft__" in sideaware_frame.columns:
        soft = pd.to_numeric(
            sideaware_frame["__y_econ_sideaware_soft__"],
            errors="coerce",
        ).clip(0.0, 1.0)
        if "__y_econ_sideaware_bin__" in sideaware_frame.columns:
            hard = pd.to_numeric(
                sideaware_frame["__y_econ_sideaware_bin__"],
                errors="coerce",
            ).fillna(0.0)
        elif "__econ_sideaware_clean__" in sideaware_frame.columns:
            hard = pd.to_numeric(
                sideaware_frame["__econ_sideaware_clean__"],
                errors="coerce",
            ).fillna(0.0)
        else:
            hard = (soft >= 0.50).astype(float)
        targets["OPTIMIZED_ECONOMIC_SIDEAWARE_TARGET"] = pd.DataFrame(
            {
                "target_soft": soft.fillna(0.0).astype(np.float32),
                "target_hard": hard.clip(0.0, 1.0).astype(np.float32),
            },
            index=frame.index,
        )
    if "__y_econ_side_resolution_soft__" in sideaware_frame.columns:
        soft = pd.to_numeric(
            sideaware_frame["__y_econ_side_resolution_soft__"],
            errors="coerce",
        ).clip(0.0, 1.0)
        if "__y_econ_side_resolution_bin__" in sideaware_frame.columns:
            hard = pd.to_numeric(
                sideaware_frame["__y_econ_side_resolution_bin__"],
                errors="coerce",
            ).fillna(0.0)
        elif "__econ_side_resolution_clean__" in sideaware_frame.columns:
            hard = pd.to_numeric(
                sideaware_frame["__econ_side_resolution_clean__"],
                errors="coerce",
            ).fillna(0.0)
        else:
            hard = (soft >= 0.50).astype(float)
        targets["OPTIMIZED_ECONOMIC_SIDE_RESOLUTION_TARGET"] = pd.DataFrame(
            {
                "target_soft": soft.fillna(0.0).astype(np.float32),
                "target_hard": hard.clip(0.0, 1.0).astype(np.float32),
            },
            index=frame.index,
        )
    if "__y_econ_sideaware_execres_soft__" in sideaware_frame.columns:
        soft = pd.to_numeric(
            sideaware_frame["__y_econ_sideaware_execres_soft__"],
            errors="coerce",
        ).clip(0.0, 1.0)
        if "__y_econ_sideaware_execres_bin__" in sideaware_frame.columns:
            hard = pd.to_numeric(
                sideaware_frame["__y_econ_sideaware_execres_bin__"],
                errors="coerce",
            ).fillna(0.0)
        elif "__econ_sideaware_execres_clean__" in sideaware_frame.columns:
            hard = pd.to_numeric(
                sideaware_frame["__econ_sideaware_execres_clean__"],
                errors="coerce",
            ).fillna(0.0)
        else:
            hard = (soft >= 0.50).astype(float)
        targets["OPTIMIZED_ECONOMIC_SIDEAWARE_EXEC_RESOLUTION_TARGET"] = pd.DataFrame(
            {
                "target_soft": soft.fillna(0.0).astype(np.float32),
                "target_hard": hard.clip(0.0, 1.0).astype(np.float32),
            },
            index=frame.index,
        )
    if "__stage15_target_soft__" in frame.columns:
        soft = pd.to_numeric(frame["__stage15_target_soft__"], errors="coerce").clip(0.0, 1.0)
        if "__stage15_target_hard__" in frame.columns:
            hard = pd.to_numeric(frame["__stage15_target_hard__"], errors="coerce").fillna(0.0)
        else:
            hard = (soft >= 0.50).astype(float)
        targets["STAGE15_quiet_mid_clean_utility"] = pd.DataFrame(
            {
                "target_soft": soft.fillna(0.0).astype(np.float32),
                "target_hard": hard.clip(0.0, 1.0).astype(np.float32),
            },
            index=frame.index,
        )
    if "__y_econ_soft__" in frame.columns:
        soft = pd.to_numeric(frame["__y_econ_soft__"], errors="coerce").clip(0.0, 1.0)
        if "__y_econ_bin__" in frame.columns:
            hard = pd.to_numeric(frame["__y_econ_bin__"], errors="coerce").fillna(0.0)
        elif "__y_econ_pos__" in frame.columns:
            hard = pd.to_numeric(frame["__y_econ_pos__"], errors="coerce").fillna(0.0)
        elif "__u_econ_net__" in frame.columns:
            hard = (pd.to_numeric(frame["__u_econ_net__"], errors="coerce") > 0.0).astype(float)
        else:
            hard = (soft >= 0.50).astype(float)
        targets["OPTIMIZED_ECONOMIC_TARGET"] = pd.DataFrame(
            {
                "target_soft": soft.fillna(0.0).astype(np.float32),
                "target_hard": hard.clip(0.0, 1.0).astype(np.float32),
            },
            index=frame.index,
        )
        u = pd.to_numeric(frame.get("__u_econ_net__", metrics["u_policy_net"]), errors="coerce")
        mae_norm = pd.to_numeric(metrics["mae_norm"], errors="coerce").fillna(10.0)
        mfe_norm = pd.to_numeric(metrics["mfe_norm"], errors="coerce").fillna(0.0)
        timeout = metrics["is_timeout"].astype(float).fillna(1.0)
        clean_gate = (
            _bounded_sigmoid((1.00 - mae_norm) / 0.20).reindex(frame.index).fillna(0.0)
            * _bounded_sigmoid((0.50 - timeout) / 0.10).reindex(frame.index).fillna(0.0)
            * _bounded_sigmoid(u.fillna(-0.02) / 0.006).reindex(frame.index).fillna(0.0)
        ).clip(0.0, 1.0)
        path_safe_soft = (soft.fillna(0.0) * clean_gate).clip(0.0, 1.0)
        clean_hard = ((u > 0.0) & (mae_norm < 1.0) & (timeout <= 0.0)).astype(float)
        targets["OPTIMIZED_ECONOMIC_PATH_SAFE_TARGET"] = pd.DataFrame(
            {
                "target_soft": path_safe_soft.astype(np.float32),
                "target_hard": clean_hard.astype(np.float32),
            },
            index=frame.index,
        )

        contrast_raw = (
            u.fillna(-0.02) / 0.006
            + 0.30 * mfe_norm.clip(upper=4.0)
            - 1.35 * (mae_norm - 0.65).clip(lower=0.0)
            - 0.65 * timeout
        )
        contrast_soft = _bounded_sigmoid(contrast_raw).reindex(frame.index).fillna(0.0)
        contrast_soft = (0.65 * contrast_soft + 0.35 * path_safe_soft).clip(0.0, 1.0)
        targets["OPTIMIZED_ECONOMIC_BAD_MAE_CONTRAST_TARGET"] = pd.DataFrame(
            {
                "target_soft": contrast_soft.astype(np.float32),
                "target_hard": clean_hard.astype(np.float32),
            },
            index=frame.index,
        )

        if "__ts__" in frame.columns:
            clean_rank = contrast_soft.groupby(frame["__ts__"], dropna=False).rank(
                method="average",
                pct=True,
            )
            clean_rank = clean_rank.fillna(contrast_soft.rank(method="average", pct=True))
        else:
            clean_rank = contrast_soft.rank(method="average", pct=True)
        clean_rank_soft = (0.55 * contrast_soft + 0.45 * clean_rank).clip(0.0, 1.0)
        targets["OPTIMIZED_ECONOMIC_CLEAN_RANK_TARGET"] = pd.DataFrame(
            {
                "target_soft": clean_rank_soft.astype(np.float32),
                "target_hard": (clean_rank_soft >= 0.70).astype(np.float32),
            },
            index=frame.index,
        )
        timeout_safe_raw = (
            u.fillna(-0.02) / 0.006
            + 0.20 * mfe_norm.clip(upper=4.0)
            - 1.20 * (mae_norm - 0.60).clip(lower=0.0)
            - 2.50 * timeout
        )
        timeout_safe_soft = _bounded_sigmoid(timeout_safe_raw).reindex(frame.index).fillna(0.0)
        timeout_safe_soft = (0.70 * timeout_safe_soft + 0.30 * path_safe_soft).clip(0.0, 1.0)
        timeout_safe_soft = timeout_safe_soft.where(timeout <= 0.0, timeout_safe_soft * 0.10)
        targets["OPTIMIZED_ECONOMIC_TIMEOUT_SAFE_TARGET"] = pd.DataFrame(
            {
                "target_soft": timeout_safe_soft.astype(np.float32),
                "target_hard": clean_hard.astype(np.float32),
            },
            index=frame.index,
        )

        strict_clean_gate = (
            _bounded_sigmoid((0.75 - mae_norm) / 0.12).reindex(frame.index).fillna(0.0)
            * _bounded_sigmoid((0.50 - timeout) / 0.10).reindex(frame.index).fillna(0.0)
            * _bounded_sigmoid(u.fillna(-0.02) / 0.004).reindex(frame.index).fillna(0.0)
        ).clip(0.0, 1.0)
        strict_path_raw = (
            u.fillna(-0.02) / 0.004
            + 0.20 * (mfe_norm - 0.75).clip(lower=0.0, upper=3.0)
            - 2.75 * (mae_norm - 0.45).clip(lower=0.0)
            - 3.00 * (mae_norm >= 1.0).astype(float)
            - 2.50 * timeout
        )
        strict_path_soft = _bounded_sigmoid(strict_path_raw).reindex(frame.index).fillna(0.0)
        strict_path_soft = (0.70 * strict_path_soft + 0.30 * strict_clean_gate).clip(0.0, 1.0)
        strict_path_soft = strict_path_soft.where(
            (mae_norm < 1.0) & (timeout <= 0.0),
            strict_path_soft * 0.05,
        )
        strict_clean_hard = ((u > 0.0) & (mae_norm < 0.75) & (timeout <= 0.0)).astype(float)
        targets["OPTIMIZED_ECONOMIC_STRICT_PATH_FIRST_TARGET"] = pd.DataFrame(
            {
                "target_soft": strict_path_soft.astype(np.float32),
                "target_hard": strict_clean_hard.astype(np.float32),
            },
            index=frame.index,
        )

        clean_utility = u.where((u > 0.0) & (mae_norm < 0.75) & (timeout <= 0.0))
        if "__ts__" in frame.columns:
            clean_utility_rank = clean_utility.groupby(frame["__ts__"], dropna=False).rank(
                method="average",
                pct=True,
            )
            fallback_rank = clean_utility.rank(method="average", pct=True)
            clean_utility_rank = clean_utility_rank.fillna(fallback_rank).fillna(0.0)
        else:
            clean_utility_rank = clean_utility.rank(method="average", pct=True).fillna(0.0)
        clean_utility_rank_soft = (
            0.60 * clean_utility_rank
            + 0.25 * strict_path_soft
            + 0.15 * strict_clean_gate
        ).clip(0.0, 1.0)
        clean_utility_rank_soft = clean_utility_rank_soft.where(
            (mae_norm < 1.0) & (timeout <= 0.0),
            clean_utility_rank_soft * 0.05,
        )
        targets["OPTIMIZED_ECONOMIC_CLEAN_UTILITY_RANK_TARGET"] = pd.DataFrame(
            {
                "target_soft": clean_utility_rank_soft.astype(np.float32),
                "target_hard": strict_clean_hard.astype(np.float32),
            },
            index=frame.index,
        )

        side = pd.to_numeric(
            metrics.get("side", pd.Series(1.0, index=frame.index)),
            errors="coerce",
        ).reindex(frame.index).fillna(1.0)
        side_key = side.where(side < 0.0, 1.0).where(side >= 0.0, -1.0)
        if "__ts__" in frame.columns:
            group_keys = [frame["__ts__"], side_key]
            ts_side_u_rank = u.groupby(group_keys, dropna=False).rank(
                method="average",
                pct=True,
            )
        else:
            ts_side_u_rank = u.rank(method="average", pct=True)
        ts_side_u_rank = ts_side_u_rank.fillna(0.0)
        bad_mae = mae_norm >= 1.0
        timed_out = timeout > 0.5
        full_stop = pd.Series(False, index=frame.index)
        for stop_col in ("__full_stop_loss__", "full_stop_loss", "full_sl", "replay_full_sl"):
            if stop_col in frame.columns:
                full_stop = pd.to_numeric(frame[stop_col], errors="coerce").fillna(0.0).gt(0.5)
                break
        clean_path = (u > 0.0) & (~bad_mae) & (~timed_out) & (~full_stop)
        dirty_positive = (u > 0.0) & (bad_mae | timed_out | full_stop)
        clean_high_u = clean_path & ts_side_u_rank.ge(0.80)
        clean_oracle = clean_path & ts_side_u_rank.ge(0.90)
        relevance = pd.Series(0.0, index=frame.index, dtype=np.float32)
        relevance.loc[dirty_positive] = 1.0
        relevance.loc[clean_path] = 2.0
        relevance.loc[clean_high_u] = 3.0
        relevance.loc[clean_oracle] = 4.0
        targets["OPTIMIZED_ECONOMIC_PATH_FIRST_CLEAN_RELEVANCE_TARGET"] = pd.DataFrame(
            {
                "target_soft": (relevance / 4.0).clip(0.0, 1.0).astype(np.float32),
                "target_hard": relevance.ge(2.0).astype(np.float32),
            },
            index=frame.index,
        )

        # S24 keeps the source broad enough for Gate 3 recall while teaching the
        # model that dirty-positive paths are weak, non-final candidates.
        clean_rank_u = u.where(clean_path).groupby(group_keys, dropna=False).rank(
            method="average",
            pct=True,
        )
        clean_rank_u = clean_rank_u.fillna(0.0)
        dirty_rank_u = u.where(dirty_positive).groupby(group_keys, dropna=False).rank(
            method="average",
            pct=True,
        )
        dirty_rank_u = dirty_rank_u.fillna(0.0)
        broad_path_soft = pd.Series(0.0, index=frame.index, dtype=np.float32)
        broad_path_soft.loc[dirty_positive] = (
            0.08 + 0.18 * dirty_rank_u.loc[dirty_positive]
        ).astype(np.float32)
        broad_path_soft.loc[clean_path] = (
            0.42 + 0.46 * clean_rank_u.loc[clean_path]
        ).astype(np.float32)
        broad_path_soft.loc[clean_path & ts_side_u_rank.ge(0.90)] = (
            0.90 + 0.10 * ts_side_u_rank.loc[clean_path & ts_side_u_rank.ge(0.90)]
        ).astype(np.float32)
        broad_path_soft = broad_path_soft.clip(0.0, 1.0).fillna(0.0)
        broad_path_hard = (
            clean_path
            & (
                ts_side_u_rank.ge(0.75)
                | clean_rank_u.ge(0.70)
            )
        ).astype(np.float32)
        targets["OPTIMIZED_ECONOMIC_S24_BROAD_PATH_FIRST_SOURCE_TARGET"] = pd.DataFrame(
            {
                "target_soft": broad_path_soft.astype(np.float32),
                "target_hard": broad_path_hard.astype(np.float32),
            },
            index=frame.index,
        )

        # S31 explicitly separates clean positives from profitable-but-slow
        # timeout positives. Timeout positives remain source candidates, but
        # they are ranked below clean non-timeout paths by construction.
        timeout_clean_positive = (u > 0.0) & timed_out & (~bad_mae) & (~full_stop)
        fast_clean_path = clean_path & mae_norm.lt(0.75) & mfe_norm.ge(1.0)
        timeout_clean_rank = u.where(timeout_clean_positive).groupby(group_keys, dropna=False).rank(
            method="average",
            pct=True,
        ).fillna(0.0)
        timeout_aware_soft = pd.Series(0.0, index=frame.index, dtype=np.float32)
        timeout_aware_soft.loc[timeout_clean_positive] = (
            0.04 + 0.10 * timeout_clean_rank.loc[timeout_clean_positive]
        ).astype(np.float32)
        timeout_aware_soft.loc[clean_path] = (
            0.46 + 0.34 * clean_rank_u.loc[clean_path]
        ).astype(np.float32)
        timeout_aware_soft.loc[fast_clean_path] = (
            0.72 + 0.20 * clean_rank_u.loc[fast_clean_path]
        ).astype(np.float32)
        timeout_aware_soft.loc[fast_clean_path & ts_side_u_rank.ge(0.90)] = (
            0.92 + 0.08 * ts_side_u_rank.loc[fast_clean_path & ts_side_u_rank.ge(0.90)]
        ).astype(np.float32)
        timeout_aware_soft = timeout_aware_soft.clip(0.0, 1.0).fillna(0.0)
        timeout_aware_hard = (
            fast_clean_path
            & (
                ts_side_u_rank.ge(0.72)
                | clean_rank_u.ge(0.68)
            )
        ).astype(np.float32)
        targets["OPTIMIZED_ECONOMIC_TIMEOUT_AWARE_CLEAN_SOURCE_TARGET"] = pd.DataFrame(
            {
                "target_soft": timeout_aware_soft.astype(np.float32),
                "target_hard": timeout_aware_hard.astype(np.float32),
            },
            index=frame.index,
        )

        mfe_mae = (mfe_norm / mae_norm.clip(lower=0.25)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bars_to_mfe = pd.to_numeric(metrics.get("bars_to_mfe", pd.Series(0.0, index=frame.index)), errors="coerce").reindex(frame.index).fillna(0.0)
        barrier = pd.to_numeric(metrics.get("barrier", pd.Series(0.02, index=frame.index)), errors="coerce").reindex(frame.index).fillna(0.02)
        exec_margin = (
            u.fillna(-0.02)
            - 0.0040 * (mae_norm - 0.65).clip(lower=0.0)
            - 0.0050 * bad_mae.astype(float)
            - 0.0060 * timed_out.astype(float)
            - 0.0010 * np.log1p(bars_to_mfe.clip(lower=0.0))
            - 0.75 * (barrier - 0.020).clip(lower=0.0)
            + 0.0015 * (mfe_mae - 1.25).clip(lower=0.0, upper=2.0)
        )
        exec_admissible = (
            u.gt(0.0005)
            & mae_norm.le(0.85)
            & (~timed_out)
            & (~full_stop)
            & mfe_norm.ge(1.0)
            & mfe_mae.ge(1.25)
            & bars_to_mfe.le(14.0)
        ).fillna(False)
        exec_strict = (
            exec_admissible
            & u.gt(0.0020)
            & mae_norm.le(0.75)
            & bars_to_mfe.le(10.0)
            & mfe_mae.ge(1.40)
        ).fillna(False)
        exec_rank = exec_margin.where(exec_admissible).groupby(group_keys, dropna=False).rank(
            method="average",
            pct=True,
        ).fillna(0.0)
        exec_soft = pd.Series(0.0, index=frame.index, dtype=np.float32)
        exec_soft.loc[dirty_positive] = 0.02
        exec_margin_sigmoid = pd.Series(
            _bounded_sigmoid(exec_margin.loc[exec_admissible] / 0.003).to_numpy(dtype=np.float32),
            index=exec_margin.loc[exec_admissible].index,
        )
        exec_soft.loc[exec_admissible] = (
            0.35
            + 0.40 * exec_rank.loc[exec_admissible]
            + 0.25 * exec_margin_sigmoid.reindex(exec_admissible.loc[exec_admissible].index).fillna(0.0)
        ).astype(np.float32)
        exec_soft.loc[exec_strict] = np.maximum(
            exec_soft.loc[exec_strict].to_numpy(dtype=np.float32),
            (0.75 + 0.25 * exec_rank.loc[exec_strict]).to_numpy(dtype=np.float32),
        )
        exec_soft = exec_soft.clip(0.0, 1.0).fillna(0.0)
        targets["OPTIMIZED_ECONOMIC_EXEC_MARGIN_STABLE_TARGET"] = pd.DataFrame(
            {
                "target_soft": exec_soft.astype(np.float32),
                "target_hard": (exec_admissible & exec_rank.ge(0.55)).astype(np.float32),
            },
            index=frame.index,
        )
    return targets


def _apply_evaluation_utility_column(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    column: str | None,
) -> str:
    if column is None or not str(column).strip():
        return str(metrics.attrs.get("utility_source", "u_policy_net"))
    source = str(column).strip()
    if source not in frame.columns:
        raise ValueError(f"Evaluation utility column not found: {source}")
    utility = pd.to_numeric(frame[source], errors="coerce").reindex(frame.index)
    if not bool(utility.notna().any()):
        raise ValueError(f"Evaluation utility column has no finite values: {source}")
    metrics["u_policy_net"] = utility.astype(np.float32)
    metrics.attrs["utility_source"] = source
    return source


def _dcg(gains: np.ndarray) -> float:
    if gains.size == 0:
        return float("nan")
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
    return float(np.sum(gains.astype(np.float64, copy=False) * discounts))


def _empty_timestamp_ranking_metrics() -> dict[str, Any]:
    out: dict[str, Any] = {
        "ts_rank_timestamp_count": 0,
        "ts_rank_top30_rows": 0,
        "ts_rank_ndcg30_u": float("nan"),
        "ts_rank_ndcg30_opportunity_rate": float("nan"),
        "ts_rank_top30_bad_mae_1r_rate": float("nan"),
        "ts_rank_top30_wide_barrier_25bps_rate": float("nan"),
        "ts_rank_top30_timeout_rate": float("nan"),
        "ts_rank_week_count": 0,
    }
    for k in (10, 20, 30):
        out[f"ts_rank_hr{k}_u"] = float("nan")
        out[f"ts_rank_target_hr{k}"] = float("nan")
        out[f"ts_rank_mean_u{k}"] = float("nan")
        out[f"ts_rank_q05_u{k}"] = float("nan")
    for q in (5, 10, 25, 50, 75):
        out[f"ts_rank_week_hr30_q{q:02d}"] = float("nan")
    return out


def _timestamp_ranking_metrics(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
) -> dict[str, Any]:
    """Evaluate live-like top-k ranking inside each timestamp.

    The economic hit is net policy utility > 0. NDCG uses positive net utility
    as gain, so a ranking can only score well by putting profitable rows above
    unprofitable or low-upside rows at the same timestamp.
    """
    if "__ts__" not in frame.columns:
        return _empty_timestamp_ranking_metrics()

    score_s = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
    ts = pd.to_datetime(frame["__ts__"].reset_index(drop=True), errors="coerce")
    u = pd.to_numeric(metrics["u_policy_net"].reset_index(drop=True), errors="coerce")
    hard_default = pd.Series(np.zeros(len(frame), dtype=np.float32))
    target_hard = pd.to_numeric(
        target.get("target_hard", hard_default).reset_index(drop=True),
        errors="coerce",
    ).fillna(0.0)
    mae_norm = pd.to_numeric(metrics["mae_norm"].reset_index(drop=True), errors="coerce")
    barrier = pd.to_numeric(metrics["barrier"].reset_index(drop=True), errors="coerce")
    timeout = metrics["is_timeout"].reset_index(drop=True).astype(float)

    valid_mask = ts.notna() & score_s.notna() & u.notna()
    if not bool(valid_mask.any()):
        return _empty_timestamp_ranking_metrics()

    ts_valid = ts.loc[valid_mask].to_numpy(dtype="datetime64[ns]")
    ts_ns = ts_valid.astype("datetime64[ns]").astype(np.int64, copy=False)
    score_arr = score_s.loc[valid_mask].to_numpy(dtype=np.float32, copy=False)
    u_arr = u.loc[valid_mask].to_numpy(dtype=np.float32, copy=False)
    target_arr = target_hard.loc[valid_mask].to_numpy(dtype=np.float32, copy=False)
    mae_arr = mae_norm.loc[valid_mask].to_numpy(dtype=np.float32, copy=False)
    barrier_arr = barrier.loc[valid_mask].to_numpy(dtype=np.float32, copy=False)
    timeout_arr = timeout.loc[valid_mask].to_numpy(dtype=np.float32, copy=False)

    order_ts = np.argsort(ts_ns, kind="mergesort")
    ts_ordered = ts_ns[order_ts]
    boundaries = np.flatnonzero(np.diff(ts_ordered)) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(order_ts)]

    ts_values: list[np.datetime64] = []
    rows30_values: list[int] = []
    ndcg30_values: list[float] = []
    opportunity_values: list[float] = []
    top30_bad_values: list[float] = []
    top30_wide_values: list[float] = []
    top30_timeout_values: list[float] = []
    hr_values: dict[int, list[float]] = {10: [], 20: [], 30: []}
    target_hr_values: dict[int, list[float]] = {10: [], 20: [], 30: []}
    mean_u_values: dict[int, list[float]] = {10: [], 20: [], 30: []}
    q05_u_values: dict[int, list[float]] = {10: [], 20: [], 30: []}

    def _q05_small(values: np.ndarray) -> float:
        if values.size == 0:
            return float("nan")
        finite = np.asarray(values, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return float("nan")
        if finite.size == 1:
            return float(finite[0])
        sorted_vals = np.sort(finite.astype(np.float64, copy=False))
        pos = 0.05 * float(sorted_vals.size - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(sorted_vals[lo])
        weight = pos - float(lo)
        return float((1.0 - weight) * sorted_vals[lo] + weight * sorted_vals[hi])

    for start, stop in zip(starts, stops):
        group_idx = order_ts[start:stop]
        if group_idx.size == 0:
            continue
        ranked_idx = group_idx[
            np.argsort(-score_arr[group_idx], kind="mergesort")
        ]
        gains_all = np.maximum(u_arr[group_idx].astype(np.float64, copy=False), 0.0)
        ideal = np.sort(gains_all)[::-1][: min(30, gains_all.size)]
        ideal_dcg = _dcg(ideal)
        top30_idx = ranked_idx[: min(30, ranked_idx.size)]
        top30_u = u_arr[top30_idx]
        pred_dcg = _dcg(np.maximum(top30_u.astype(np.float64, copy=False), 0.0))

        ts_values.append(ts_valid[group_idx[0]])
        rows30_values.append(int(top30_idx.size))
        ndcg30_values.append(
            pred_dcg / ideal_dcg if math.isfinite(ideal_dcg) and ideal_dcg > 0.0 else 0.0
        )
        opportunity_values.append(float(np.any(gains_all > 0.0)))
        top30_bad_values.append(float(np.mean(mae_arr[top30_idx] >= 1.0)))
        top30_wide_values.append(float(np.mean(barrier_arr[top30_idx] > 0.025)))
        top30_timeout_values.append(float(np.mean(timeout_arr[top30_idx] > 0.5)))

        for k in (10, 20, 30):
            top_idx = ranked_idx[: min(k, ranked_idx.size)]
            top_u = u_arr[top_idx]
            hr_values[k].append(float(np.mean(top_u > 0.0)))
            target_hr_values[k].append(float(np.mean(target_arr[top_idx] > 0.5)))
            mean_u_values[k].append(float(np.mean(top_u)))
            q05_u_values[k].append(_q05_small(top_u))

    if not ts_values:
        return _empty_timestamp_ranking_metrics()

    def _nanmean_np(values: list[float] | np.ndarray) -> float:
        arr = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(arr)
        return float(arr[finite].mean()) if finite.any() else float("nan")

    out = _empty_timestamp_ranking_metrics()
    out["ts_rank_timestamp_count"] = int(len(ts_values))
    out["ts_rank_top30_rows"] = int(np.sum(np.asarray(rows30_values, dtype=np.int64)))
    out["ts_rank_ndcg30_u"] = _nanmean_np(ndcg30_values)
    out["ts_rank_ndcg30_opportunity_rate"] = _nanmean_np(opportunity_values)
    out["ts_rank_top30_bad_mae_1r_rate"] = _nanmean_np(top30_bad_values)
    out["ts_rank_top30_wide_barrier_25bps_rate"] = _nanmean_np(top30_wide_values)
    out["ts_rank_top30_timeout_rate"] = _nanmean_np(top30_timeout_values)
    for k in (10, 20, 30):
        out[f"ts_rank_hr{k}_u"] = _nanmean_np(hr_values[k])
        out[f"ts_rank_target_hr{k}"] = _nanmean_np(target_hr_values[k])
        out[f"ts_rank_mean_u{k}"] = _nanmean_np(mean_u_values[k])
        out[f"ts_rank_q05_u{k}"] = _nanmean_np(q05_u_values[k])

    week_frame = pd.DataFrame(
        {
            "week": pd.to_datetime(np.asarray(ts_values), errors="coerce")
            .to_period("W-SUN")
            .astype(str),
            "hr30": np.asarray(hr_values[30], dtype=np.float32),
        }
    )
    weekly_hr30 = week_frame.groupby("week", dropna=False, observed=True)["hr30"].mean()
    out["ts_rank_week_count"] = int(weekly_hr30.shape[0])
    for q in (5, 10, 25, 50, 75):
        out[f"ts_rank_week_hr30_q{q:02d}"] = _safe_quantile(weekly_hr30, q / 100.0)
    return out


def _month_model_frame(
    frame: pd.DataFrame,
    *,
    train_mask: pd.Series,
    valid_mask: pd.Series,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_values = frame.loc[train_mask, features].to_numpy(dtype=np.float32, copy=True)
    valid_values = frame.loc[valid_mask, features].to_numpy(dtype=np.float32, copy=True)
    train_values[~np.isfinite(train_values)] = np.nan
    valid_values[~np.isfinite(valid_values)] = np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(train_values, axis=0).astype(np.float32, copy=False)
    med[~np.isfinite(med)] = 0.0
    train_missing = ~np.isfinite(train_values)
    if bool(train_missing.any()):
        _row, col = np.where(train_missing)
        train_values[train_missing] = med[col]
    valid_missing = ~np.isfinite(valid_values)
    if bool(valid_missing.any()):
        _row, col = np.where(valid_missing)
        valid_values[valid_missing] = med[col]
    return (
        pd.DataFrame(train_values, columns=features),
        pd.DataFrame(valid_values, columns=features),
    )


def _fold_ae_gmm_economic_targets(
    train_metrics: pd.DataFrame,
    train_frame: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    u = pd.to_numeric(train_metrics["u_policy_net"], errors="coerce").fillna(0.0)
    bad_mae = pd.to_numeric(train_metrics["mae_norm"], errors="coerce").fillna(10.0).ge(1.0)
    timeout = pd.to_numeric(train_metrics["is_timeout"], errors="coerce").fillna(1.0).gt(0.5)
    side = pd.to_numeric(
        train_metrics.get("side", pd.Series(1.0, index=train_metrics.index)),
        errors="coerce",
    ).fillna(1.0)
    clean_positive = u.gt(0.0) & (~bad_mae) & (~timeout)
    dirty_positive = u.gt(0.0) & (bad_mae | timeout)
    out = {
        "returns": u.to_numpy(dtype=np.float32, copy=False),
        "bad_mae_1r": bad_mae.astype(np.float32).to_numpy(copy=False),
        "timeout": timeout.astype(np.float32).to_numpy(copy=False),
        "clean_positive": clean_positive.astype(np.float32).to_numpy(copy=False),
        "dirty_positive": dirty_positive.astype(np.float32).to_numpy(copy=False),
        "side": side.to_numpy(dtype=np.float32, copy=False),
    }
    time_source = train_frame if train_frame is not None else train_metrics
    ts_col = next((col for col in ("__ts__", "timestamp", "ts") if col in time_source.columns), None)
    if ts_col is not None:
        ts = pd.to_datetime(time_source[ts_col], errors="coerce")
        if bool(ts.notna().any()):
            periods = ts.dt.to_period("M").astype(str)
            codes, _uniques = pd.factorize(periods, sort=True)
            time_bucket = codes.astype(np.float32)
            time_bucket[~ts.notna().to_numpy(dtype=bool)] = np.nan
            out["time_bucket"] = time_bucket
    return out


def _ae_gmm_smoke_feature_policy_columns(columns: list[str]) -> list[str]:
    policy = str(AE_GMM_SMOKE_FEATURE_POLICY or "all").strip().lower()
    hard_cluster = {"gmm_cluster_id", "cluster_t"}
    if policy in {"all", "legacy", "with_cluster_id"}:
        return list(columns)
    if policy in {"continuous", "continuous_only", "continuous_no_cluster_id", "no_cluster_id"}:
        return [str(col) for col in columns if str(col) not in hard_cluster]
    if policy in {"soft_distance_transition", "soft_distance_transition_no_cluster_id", "soft"}:
        prefixes = (
            "gmm_prob_",
            "gmm_cluster_posterior_",
            "gmm_dist_center_",
            "gmm_mahal_",
        )
        exact = {
            "gmm_posterior_max",
            "gmm_posterior_margin",
            "gmm_posterior_delta_1",
            "gmm_posterior_accel_1",
            "gmm_entropy",
            "cluster_entropy",
            "cluster_entropy_norm",
            "cluster_entropy_delta_1",
            "cluster_entropy_accel_1",
            "mahalanobis_distance",
            "min_mahalanobis",
            "min_mahalanobis_delta_1",
            "expected_mahalanobis",
            "expected_mahalanobis_delta_1",
            "expected_mahalanobis_accel_1",
            "cluster_speed",
            "cluster_acceleration",
            "time_since_cluster_change",
            "rolling_cluster_stability",
            "cluster_flip_count_20",
            "AE_reconstruction_error",
            "ae_reconstruction_error",
            "dae_reconstruction_error",
            "dae_reconstruction_error_zscore",
            "dae_reconstruction_error_delta_1",
            "dae_reconstruction_error_accel_1",
            "latent_mahalanobis_drift",
            "latent_speed",
            "latent_acceleration",
        }
        return [
            str(col)
            for col in columns
            if str(col) not in hard_cluster
            and (str(col) in exact or any(str(col).startswith(prefix) for prefix in prefixes))
        ]
    return [str(col) for col in columns if str(col) not in hard_cluster]


def _side_context_enabled() -> bool:
    return str(AE_GMM_SIDE_CONTEXT_MODE or "off").strip().lower() == "long_short"


def _prefixed_side_context(frame: pd.DataFrame, side_name: str) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [f"{side_name}_{col}" for col in out.columns]
    return out


def _chronological_inner_oof_splits(
    *,
    train_frame: pd.DataFrame,
    n_rows: int,
    min_train_rows: int = 500,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_rows <= int(min_train_rows):
        return []
    if "__ts__" in train_frame.columns:
        ts = pd.to_datetime(train_frame["__ts__"].reset_index(drop=True), errors="coerce")
        months = ts.dt.to_period("M").astype(str)
        unique_months = [m for m in sorted(months.dropna().unique()) if str(m) != "NaT"]
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for month in unique_months[1:]:
            train_mask = months < month
            valid_mask = months == month
            if int(train_mask.sum()) >= int(min_train_rows) and int(valid_mask.sum()) > 0:
                splits.append(
                    (
                        np.flatnonzero(train_mask.to_numpy(dtype=bool)),
                        np.flatnonzero(valid_mask.to_numpy(dtype=bool)),
                    )
                )
        if splits:
            return splits
    fold_count = min(5, max(2, n_rows // max(int(min_train_rows), 1)))
    boundaries = np.linspace(0, n_rows, fold_count + 1, dtype=np.int64)
    splits = []
    for i in range(1, len(boundaries) - 1):
        train_end = int(boundaries[i])
        valid_start = int(boundaries[i])
        valid_end = int(boundaries[i + 1])
        if train_end >= int(min_train_rows) and valid_end > valid_start:
            splits.append((np.arange(0, train_end, dtype=np.int64), np.arange(valid_start, valid_end, dtype=np.int64)))
    return splits


def _fit_ae_gmm_state_for_rows(
    *,
    x_base: pd.DataFrame,
    metrics: pd.DataFrame,
    train_frame: pd.DataFrame,
    row_positions: np.ndarray,
    random_state: int,
    max_train_rows: int,
    gmm_max_train_rows: int,
    ae_max_iter: int,
    require_both_sides: bool,
) -> dict[str, Any]:
    pos = np.asarray(row_positions, dtype=np.int64)
    return fit_ae_gmm_state(
        x_base.reset_index(drop=True).iloc[pos].reset_index(drop=True),
        economic_targets=_fold_ae_gmm_economic_targets(
            metrics.reset_index(drop=True).iloc[pos].reset_index(drop=True),
            train_frame=train_frame.reset_index(drop=True).iloc[pos].reset_index(drop=True),
        ),
        random_state=int(random_state),
        max_train_rows=int(max_train_rows),
        gmm_max_train_rows=int(gmm_max_train_rows),
        ae_max_iter=int(ae_max_iter),
        require_both_sides=bool(require_both_sides),
        min_side_cluster_frac=0.02,
        min_side_cluster_rows=10,
    )


def _persist_ae_gmm_state_artifact(
    *,
    state: dict[str, Any],
    artifact_dir: Path | None,
    artifact_name: str,
    scope: str,
    train_rows: int,
    valid_rows: int,
    input_feature_count: int,
) -> dict[str, Any]:
    if artifact_dir is None:
        return {}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(artifact_name or "fold")).strip("_") or "fold"
    safe_scope = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scope or "global")).strip("_") or "global"
    stem = f"{safe_name}__{safe_scope}"
    state_path = artifact_dir / f"{stem}_state.pkl"
    manifest_path = artifact_dir / f"{stem}_manifest.json"
    selected = dict(state.get("selected_config", {}) or {})
    save_ae_gmm_state_artifact(
        state,
        state_path,
        manifest_path=manifest_path,
        extra_manifest={
            "scope": str(scope),
            "train_rows": int(train_rows),
            "valid_rows": int(valid_rows),
            "input_feature_count": int(input_feature_count),
            "selected_config": selected,
            "oos_transform_contract": "state_fit_on_fold_train_rows_only_then_frozen_transform_on_validation_rows",
            "materialized_transform_rules": {
                "emitted_feature_subset": list(state.get("_emitted_feature_subset", []) or []),
                "emitted_feature_subset_count": int(len(state.get("_emitted_feature_subset", []) or [])),
                "side_context_mode": str(state.get("_side_context_mode", "off")),
                "chunk_rows": state.get("_transform_chunk_rows"),
                "train_missing_value_policy": "median_from_train_matrix_then_zero_fill_remaining",
                "validation_missing_value_policy": "same_train_median_then_zero_fill_remaining",
                "feature_policy": str(AE_GMM_SMOKE_FEATURE_POLICY or "all"),
            },
        },
    )
    return {
        f"ae_gmm_{safe_scope}_state_path": str(state_path),
        f"ae_gmm_{safe_scope}_manifest_path": str(manifest_path),
    }


def _transform_ae_gmm_features_selected_chunked(
    x_base: pd.DataFrame,
    state: dict[str, Any],
    *,
    index: Any,
    columns: list[str],
    chunk_rows: int | None = None,
) -> pd.DataFrame:
    selected_columns = [str(col) for col in columns if str(col).strip()]
    if not selected_columns:
        return pd.DataFrame(index=index)
    n_rows = int(len(x_base))
    if n_rows == 0:
        return pd.DataFrame(index=index, columns=selected_columns, dtype=np.float32)
    chunk = int(
        chunk_rows
        if chunk_rows is not None
        else os.environ.get("EPM_AE_GMM_TRANSFORM_CHUNK_ROWS", "200000")
    )
    chunk = max(10_000, int(chunk))
    out = np.zeros((n_rows, len(selected_columns)), dtype=np.float32)
    idx_values = x_base.index if index is None else index
    for start in range(0, n_rows, chunk):
        end = min(start + chunk, n_rows)
        idx_slice = idx_values[start:end] if hasattr(idx_values, "__getitem__") else x_base.index[start:end]
        values = transform_ae_gmm_features(
            x_base.iloc[start:end],
            state,
            index=idx_slice,
        ).reindex(columns=selected_columns, fill_value=0.0)
        out[start:end, :] = values.to_numpy(dtype=np.float32, copy=False)
    return pd.DataFrame(out, index=idx_values, columns=selected_columns)


def _crossfit_ae_gmm_features(
    *,
    x_base: pd.DataFrame,
    metrics: pd.DataFrame,
    train_frame: pd.DataFrame,
    generated_features: list[str],
    random_state: int,
    max_train_rows: int,
    gmm_max_train_rows: int,
    ae_max_iter: int,
    require_both_sides: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = pd.DataFrame(0.0, index=x_base.index, columns=generated_features, dtype=np.float32)
    available = pd.Series(0.0, index=x_base.index, dtype=np.float32)
    splits = _chronological_inner_oof_splits(
        train_frame=train_frame,
        n_rows=len(x_base),
        min_train_rows=max(500, min(2_000, int(max_train_rows) if int(max_train_rows) > 0 else 500)),
    )
    transformed_rows = 0
    fitted_folds = 0
    failed_folds = 0
    for fold_i, (inner_train, inner_valid) in enumerate(splits):
        if inner_train.size <= 0 or inner_valid.size <= 0:
            continue
        state = _fit_ae_gmm_state_for_rows(
            x_base=x_base,
            metrics=metrics,
            train_frame=train_frame,
            row_positions=inner_train,
            random_state=int(random_state + 10_000 + fold_i * 101),
            max_train_rows=int(max_train_rows),
            gmm_max_train_rows=int(gmm_max_train_rows),
            ae_max_iter=int(ae_max_iter),
            require_both_sides=bool(require_both_sides),
        )
        if not bool(state.get("enabled", False)):
            failed_folds += 1
            continue
        values = transform_ae_gmm_features(
            x_base.reset_index(drop=True).iloc[inner_valid].reset_index(drop=True),
            state,
            index=x_base.index[inner_valid],
        ).reindex(columns=generated_features, fill_value=0.0)
        out.loc[x_base.index[inner_valid], generated_features] = values.to_numpy(
            dtype=np.float32,
            copy=False,
        )
        available.loc[x_base.index[inner_valid]] = 1.0
        fitted_folds += 1
        transformed_rows += int(inner_valid.size)
    if "ae_gmm_oof_available" in generated_features:
        out.loc[:, "ae_gmm_oof_available"] = available.to_numpy(dtype=np.float32, copy=False)
    return out, {
        "crossfit_enabled": bool(AE_GMM_CROSSFIT_TRAIN_FEATURES),
        "crossfit_split_count": int(len(splits)),
        "crossfit_fitted_folds": int(fitted_folds),
        "crossfit_failed_folds": int(failed_folds),
        "crossfit_transformed_rows": int(transformed_rows),
        "crossfit_uncovered_rows": int(max(len(x_base) - transformed_rows, 0)),
        "crossfit_coverage": float(transformed_rows / max(len(x_base), 1)),
    }


def _append_fold_ae_gmm_state_features(
    *,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    enabled: bool,
    max_train_rows: int,
    gmm_max_train_rows: int,
    ae_max_iter: int,
    random_state: int,
    state_artifact_dir: Path | None = None,
    state_artifact_name: str = "",
    fixed_state_path: Path | None = None,
    output_feature_subset: list[str] | None = None,
    input_feature_cols: Sequence[str] | None = None,
    fit_x_base: pd.DataFrame | None = None,
    fit_train_frame: pd.DataFrame | None = None,
    fit_train_metrics: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    if not enabled:
        return x_train, x_valid, [], {
            "ae_gmm_state_features_enabled": False,
            "ae_gmm_state_feature_status": "disabled",
            "ae_gmm_state_feature_count": 0,
        }
    excluded_generated = set(str(v) for v in AE_GMM_FEATURE_COLUMNS)
    if input_feature_cols is not None:
        requested_inputs = [str(col) for col in input_feature_cols if str(col).strip()]
        base_features = [
            col
            for col in dict.fromkeys(requested_inputs)
            if col in x_train.columns and col not in excluded_generated
        ]
    else:
        base_features = [
            str(col)
            for col in x_train.columns
            if str(col) not in excluded_generated
        ]
    if len(base_features) < 2 or len(x_train) < 500:
        return x_train, x_valid, [], {
            "ae_gmm_state_features_enabled": False,
            "ae_gmm_state_feature_status": "insufficient_rows_or_features",
            "ae_gmm_state_feature_count": 0,
            "ae_gmm_state_input_feature_count": int(len(base_features)),
            "ae_gmm_state_input_feature_policy": "explicit_override" if input_feature_cols is not None else "all_non_generated",
        }
    x_train_base = x_train.reindex(columns=base_features).astype(np.float32, copy=False)
    x_valid_base = x_valid.reindex(columns=base_features).astype(np.float32, copy=False)
    fit_x_base_local = (
        fit_x_base.reindex(columns=base_features).astype(np.float32, copy=False)
        if fit_x_base is not None
        else x_train_base
    )
    train_metrics_local = train_metrics.reset_index(drop=True)
    valid_metrics_local = valid_metrics.reset_index(drop=True)
    train_frame_local = train_frame.reset_index(drop=True)
    fit_train_metrics_local = (
        fit_train_metrics.reset_index(drop=True) if fit_train_metrics is not None else train_metrics_local
    )
    fit_train_frame_local = (
        fit_train_frame.reset_index(drop=True) if fit_train_frame is not None else train_frame_local
    )
    all_train_positions = np.arange(len(fit_x_base_local), dtype=np.int64)
    if fixed_state_path is not None:
        state = load_ae_gmm_state_artifact(fixed_state_path)
        state_source = "loaded_fixed_state_artifact"
    else:
        state = _fit_ae_gmm_state_for_rows(
            x_base=fit_x_base_local,
            metrics=fit_train_metrics_local,
            train_frame=fit_train_frame_local,
            row_positions=all_train_positions,
            random_state=int(random_state),
            max_train_rows=int(max_train_rows),
            gmm_max_train_rows=int(gmm_max_train_rows),
            ae_max_iter=int(ae_max_iter),
            require_both_sides=True,
        )
        state_source = "fit_on_outer_train_fold"
    if not bool(state.get("enabled", False)):
        persisted_disabled = _persist_ae_gmm_state_artifact(
            state=state,
            artifact_dir=state_artifact_dir,
            artifact_name=state_artifact_name,
            scope="global_disabled",
            train_rows=len(fit_x_base_local),
            valid_rows=len(x_valid_base),
            input_feature_count=len(base_features),
        )
        return x_train, x_valid, [], {
            "ae_gmm_state_features_enabled": False,
            "ae_gmm_state_feature_status": str(state.get("reason", "state_disabled")),
            "ae_gmm_state_feature_count": 0,
            "ae_gmm_state_input_feature_count": int(len(base_features)),
            "ae_gmm_state_input_feature_policy": "explicit_override" if input_feature_cols is not None else "all_non_generated",
            "ae_gmm_state_hpo_report_count": int(state.get("hpo_report_count", 0) or 0),
            **persisted_disabled,
        }
    valid_generated = transform_ae_gmm_features(x_valid_base, state, index=x_valid.index)
    all_generated_features = [str(col) for col in valid_generated.columns]
    generated_features = _ae_gmm_smoke_feature_policy_columns(all_generated_features)
    generated_features = list(dict.fromkeys([*generated_features, "ae_gmm_oof_available"]))
    output_subset = set(str(c) for c in (output_feature_subset or []) if str(c).strip())
    if output_subset:
        generated_features = [col for col in generated_features if col in output_subset]
    state["_emitted_feature_subset"] = list(generated_features)
    state["_side_context_mode"] = str(AE_GMM_SIDE_CONTEXT_MODE or "off")
    state["_transform_chunk_rows"] = int(os.environ.get("EPM_AE_GMM_TRANSFORM_CHUNK_ROWS", "200000"))
    persisted_artifacts: dict[str, Any] = _persist_ae_gmm_state_artifact(
        state=state,
        artifact_dir=state_artifact_dir,
        artifact_name=state_artifact_name,
        scope="global",
        train_rows=len(fit_x_base_local),
        valid_rows=len(x_valid_base),
        input_feature_count=len(base_features),
    )
    valid_generated["ae_gmm_oof_available"] = np.float32(1.0)
    if bool(AE_GMM_CROSSFIT_TRAIN_FEATURES) and fixed_state_path is None:
        train_generated, crossfit_diag = _crossfit_ae_gmm_features(
            x_base=x_train_base,
            metrics=train_metrics_local,
            train_frame=train_frame_local,
            generated_features=generated_features,
            random_state=int(random_state),
            max_train_rows=int(max_train_rows),
            gmm_max_train_rows=int(gmm_max_train_rows),
            ae_max_iter=int(ae_max_iter),
            require_both_sides=True,
        )
    else:
        train_generated = _transform_ae_gmm_features_selected_chunked(
            x_train_base,
            state,
            index=x_train.index,
            columns=generated_features,
        )
        crossfit_diag = {
            "crossfit_enabled": False,
            "crossfit_split_count": 0,
            "crossfit_fitted_folds": 0,
            "crossfit_failed_folds": 0,
            "crossfit_transformed_rows": int(len(x_train_base)),
            "crossfit_uncovered_rows": 0,
            "crossfit_coverage": 1.0,
        }
        if "ae_gmm_oof_available" in generated_features:
            train_generated["ae_gmm_oof_available"] = np.float32(1.0)
    valid_generated = valid_generated.reindex(columns=generated_features, fill_value=0.0)
    side_context_reports: list[dict[str, Any]] = []
    side_feature_frames_train: list[pd.DataFrame] = []
    side_feature_frames_valid: list[pd.DataFrame] = []
    side_generated_features: list[str] = []
    if _side_context_enabled():
        if fixed_state_path is not None:
            side_context_reports.append(
                {
                    "status": "skipped_for_fixed_global_state",
                    "reason": "side-specific AE/GMM states require explicit side-state artifacts",
                    "feature_count": 0,
                }
            )
        else:
            side_train = pd.to_numeric(
                train_metrics_local.get("side", pd.Series(1.0, index=train_metrics_local.index)),
                errors="coerce",
            ).fillna(1.0)
            side_valid = pd.to_numeric(
                valid_metrics_local.get("side", pd.Series(1.0, index=valid_metrics_local.index)),
                errors="coerce",
            ).fillna(1.0)
            for side_name, side_mask in (
                ("long", side_train.ge(0.0).to_numpy(dtype=bool)),
                ("short", side_train.lt(0.0).to_numpy(dtype=bool)),
            ):
                valid_side_mask = (
                    side_valid.ge(0.0).to_numpy(dtype=bool)
                    if side_name == "long"
                    else side_valid.lt(0.0).to_numpy(dtype=bool)
                )
                if int(side_mask.sum()) < 250:
                    side_context_reports.append(
                        {
                            "side": side_name,
                            "status": "insufficient_train_rows",
                            "train_rows": int(side_mask.sum()),
                            "feature_count": 0,
                        }
                    )
                    continue
                side_state = fit_ae_gmm_state(
                    x_train_base.reset_index(drop=True).iloc[side_mask].reset_index(drop=True),
                    economic_targets=_fold_ae_gmm_economic_targets(
                        train_metrics_local.iloc[side_mask].reset_index(drop=True),
                        train_frame=train_frame_local.iloc[side_mask].reset_index(drop=True),
                    ),
                    random_state=int(random_state + (70_000 if side_name == "long" else 80_000)),
                    max_train_rows=max(200, int(max_train_rows // 2)) if int(max_train_rows) > 0 else 0,
                    gmm_max_train_rows=max(500, int(gmm_max_train_rows // 2)) if int(gmm_max_train_rows) > 0 else 0,
                    ae_max_iter=int(ae_max_iter),
                    require_both_sides=False,
                )
                if not bool(side_state.get("enabled", False)):
                    side_disabled_persist = _persist_ae_gmm_state_artifact(
                        state=side_state,
                        artifact_dir=state_artifact_dir,
                        artifact_name=state_artifact_name,
                        scope=f"side_{side_name}_disabled",
                        train_rows=int(side_mask.sum()),
                        valid_rows=int(valid_side_mask.sum()),
                        input_feature_count=len(base_features),
                    )
                    side_context_reports.append(
                        {
                            "side": side_name,
                            "status": str(side_state.get("reason", "state_disabled")),
                            "train_rows": int(side_mask.sum()),
                            "feature_count": 0,
                            **side_disabled_persist,
                        }
                    )
                    continue
                side_train_generated = pd.DataFrame(
                    0.0,
                    index=x_train.index,
                    columns=generated_features,
                    dtype=np.float32,
                )
                side_train_available = pd.Series(0.0, index=x_train.index, dtype=np.float32)
                side_valid_generated = pd.DataFrame(
                    0.0,
                    index=x_valid.index,
                    columns=generated_features,
                    dtype=np.float32,
                )
                side_valid_available = pd.Series(0.0, index=x_valid.index, dtype=np.float32)
                side_crossfit_rows = 0
                side_crossfit_folds = 0
                side_crossfit_failed = 0
                if bool(AE_GMM_CROSSFIT_TRAIN_FEATURES):
                    inner_splits = _chronological_inner_oof_splits(
                        train_frame=train_frame_local,
                        n_rows=len(x_train_base),
                        min_train_rows=max(500, min(2_000, int(max_train_rows) if int(max_train_rows) > 0 else 500)),
                    )
                    for fold_i, (inner_train, inner_valid) in enumerate(inner_splits):
                        side_inner_train = inner_train[side_mask[inner_train]]
                        side_inner_valid = inner_valid[side_mask[inner_valid]]
                        if int(side_inner_train.size) < 250 or int(side_inner_valid.size) <= 0:
                            continue
                        side_inner_state = _fit_ae_gmm_state_for_rows(
                            x_base=x_train_base,
                            metrics=train_metrics_local,
                            train_frame=train_frame_local,
                            row_positions=side_inner_train,
                            random_state=int(
                                random_state
                                + (170_000 if side_name == "long" else 180_000)
                                + fold_i * 101
                            ),
                            max_train_rows=max(200, int(max_train_rows // 2)) if int(max_train_rows) > 0 else 0,
                            gmm_max_train_rows=max(500, int(gmm_max_train_rows // 2)) if int(gmm_max_train_rows) > 0 else 0,
                            ae_max_iter=int(ae_max_iter),
                            require_both_sides=False,
                        )
                        if not bool(side_inner_state.get("enabled", False)):
                            side_crossfit_failed += 1
                            continue
                        side_train_values = transform_ae_gmm_features(
                            x_train_base.reset_index(drop=True).iloc[side_inner_valid].reset_index(drop=True),
                            side_inner_state,
                            index=x_train.index[side_inner_valid],
                        ).reindex(columns=generated_features, fill_value=0.0)
                        side_train_generated.loc[
                            x_train.index[side_inner_valid],
                            generated_features,
                        ] = side_train_values.to_numpy(dtype=np.float32, copy=False)
                        side_train_available.loc[x_train.index[side_inner_valid]] = 1.0
                        side_crossfit_rows += int(side_inner_valid.size)
                        side_crossfit_folds += 1
                else:
                    side_train_values = _transform_ae_gmm_features_selected_chunked(
                        x_train_base.reset_index(drop=True).iloc[side_mask].reset_index(drop=True),
                        side_state,
                        index=x_train.index[side_mask],
                        columns=generated_features,
                    )
                    side_train_generated.loc[x_train.index[side_mask], generated_features] = side_train_values.to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )
                    side_train_available.loc[x_train.index[side_mask]] = 1.0
                    side_crossfit_rows = int(side_mask.sum())
                if bool(valid_side_mask.any()):
                    side_valid_values = transform_ae_gmm_features(
                        x_valid_base.reset_index(drop=True).iloc[valid_side_mask].reset_index(drop=True),
                        side_state,
                        index=x_valid.index[valid_side_mask],
                    ).reindex(columns=generated_features, fill_value=0.0)
                    side_valid_generated.loc[x_valid.index[valid_side_mask], generated_features] = side_valid_values.to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )
                    side_valid_available.loc[x_valid.index[valid_side_mask]] = 1.0
                if "ae_gmm_oof_available" in generated_features:
                    side_train_generated.loc[:, "ae_gmm_oof_available"] = side_train_available.to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )
                    side_valid_generated.loc[:, "ae_gmm_oof_available"] = side_valid_available.to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )
                side_train_prefixed = _prefixed_side_context(side_train_generated, side_name)
                side_valid_prefixed = _prefixed_side_context(side_valid_generated, side_name)
                if output_subset:
                    side_keep = [col for col in side_train_prefixed.columns if str(col) in output_subset]
                    side_train_prefixed = side_train_prefixed.reindex(columns=side_keep, fill_value=0.0)
                    side_valid_prefixed = side_valid_prefixed.reindex(columns=side_keep, fill_value=0.0)
                side_feature_frames_train.append(side_train_prefixed)
                side_feature_frames_valid.append(side_valid_prefixed)
                side_generated_features.extend(str(col) for col in side_train_prefixed.columns)
                side_state["_emitted_feature_subset"] = [str(col) for col in side_train_prefixed.columns]
                side_state["_side_context_mode"] = f"side_{side_name}"
                side_state["_transform_chunk_rows"] = int(os.environ.get("EPM_AE_GMM_TRANSFORM_CHUNK_ROWS", "200000"))
                side_persisted = _persist_ae_gmm_state_artifact(
                    state=side_state,
                    artifact_dir=state_artifact_dir,
                    artifact_name=state_artifact_name,
                    scope=f"side_{side_name}",
                    train_rows=int(side_mask.sum()),
                    valid_rows=int(valid_side_mask.sum()),
                    input_feature_count=len(base_features),
                )
                side_selected = dict(side_state.get("selected_config", {}) or {})
                side_context_reports.append(
                    {
                        "side": side_name,
                        "status": "ok",
                        "train_rows": int(side_mask.sum()),
                        "valid_rows": int(valid_side_mask.sum()),
                        "feature_count": int(side_train_prefixed.shape[1]),
                        "train_crossfit_rows": int(side_crossfit_rows),
                        "train_crossfit_uncovered_rows": int(max(int(side_mask.sum()) - side_crossfit_rows, 0)),
                        "train_crossfit_coverage": float(side_crossfit_rows / max(int(side_mask.sum()), 1)),
                        "train_crossfit_folds": int(side_crossfit_folds),
                        "train_crossfit_failed_folds": int(side_crossfit_failed),
                        "n_components": int(side_state.get("gmm_n_components", 0) or 0),
                        "path_cleanliness_score": float(
                            side_selected.get("path_cleanliness_score", float("nan"))
                        ),
                        "temporal_concentration_score": float(
                            side_selected.get("temporal_concentration_score", float("nan"))
                        ),
                        **side_persisted,
                    }
                )
    if output_subset:
        raw_keep = [col for col in output_feature_subset or [] if str(col) in x_train.columns]
        x_train_emit = x_train.reindex(columns=raw_keep, fill_value=0.0)
        x_valid_emit = x_valid.reindex(columns=raw_keep, fill_value=0.0)
    else:
        x_train_emit = x_train
        x_valid_emit = x_valid
    train_concat = [x_train_emit, train_generated, *side_feature_frames_train]
    valid_concat = [x_valid_emit, valid_generated, *side_feature_frames_valid]
    x_train_out = pd.concat(train_concat, axis=1, copy=False)
    x_valid_out = pd.concat(valid_concat, axis=1, copy=False)
    emitted_features = list(generated_features) + list(side_generated_features)
    selected_config = dict(state.get("selected_config", {}) or {})
    return (
        x_train_out.astype(np.float32, copy=False),
        x_valid_out.astype(np.float32, copy=False),
        emitted_features,
        {
            "ae_gmm_state_features_enabled": True,
            "ae_gmm_state_feature_status": "ok",
            "ae_gmm_state_feature_count": int(len(emitted_features)),
            "ae_gmm_state_feature_policy": str(AE_GMM_SMOKE_FEATURE_POLICY or "all"),
            "ae_gmm_state_all_feature_count": int(len(all_generated_features)),
            "ae_gmm_state_train_feature_scope": "inner_chronological_oof"
            if bool(AE_GMM_CROSSFIT_TRAIN_FEATURES) and fixed_state_path is None
            else "loaded_fixed_state_artifact"
            if fixed_state_path is not None
            else "outer_train_in_sample",
            "ae_gmm_state_validation_feature_scope": "frozen_outer_train_artifact",
            "ae_gmm_state_source": state_source,
            "ae_gmm_state_fit_scope": "explicit_fit_frame" if fit_x_base is not None else "outer_train_fold",
            "ae_gmm_state_crossfit_enabled": bool(AE_GMM_CROSSFIT_TRAIN_FEATURES),
            "ae_gmm_state_crossfit_split_count": int(crossfit_diag.get("crossfit_split_count", 0)),
            "ae_gmm_state_crossfit_fitted_folds": int(crossfit_diag.get("crossfit_fitted_folds", 0)),
            "ae_gmm_state_crossfit_failed_folds": int(crossfit_diag.get("crossfit_failed_folds", 0)),
            "ae_gmm_state_crossfit_transformed_rows": int(crossfit_diag.get("crossfit_transformed_rows", 0)),
            "ae_gmm_state_crossfit_uncovered_rows": int(crossfit_diag.get("crossfit_uncovered_rows", 0)),
            "ae_gmm_state_crossfit_coverage": float(crossfit_diag.get("crossfit_coverage", float("nan"))),
            "ae_gmm_side_context_mode": str(AE_GMM_SIDE_CONTEXT_MODE or "off"),
            "ae_gmm_side_context_enabled": bool(_side_context_enabled()),
            "ae_gmm_side_context_feature_count": int(len(side_generated_features)),
            "ae_gmm_side_context_report": json.dumps(_json_safe(side_context_reports)),
            "ae_gmm_state_input_feature_count": int(len(base_features)),
            "ae_gmm_state_input_feature_policy": "explicit_override" if input_feature_cols is not None else "all_non_generated",
            "ae_gmm_state_hpo_report_count": int(state.get("hpo_report_count", 0) or 0),
            "ae_gmm_state_train_rows_available": int(state.get("train_rows_available", len(fit_x_base_local)) or 0),
            "ae_gmm_state_ae_fit_rows": int(state.get("ae_fit_rows", 0) or 0),
            "ae_gmm_state_gmm_fit_rows": int(state.get("gmm_fit_rows", 0) or 0),
            "ae_gmm_state_ae_max_train_rows": int(state.get("ae_max_train_rows", max_train_rows) or 0),
            "ae_gmm_state_gmm_max_train_rows": int(state.get("gmm_max_train_rows", gmm_max_train_rows) or 0),
            "ae_gmm_state_sample_policy": str(state.get("sample_policy", "")),
            "ae_gmm_state_n_components": int(state.get("gmm_n_components", 0) or 0),
            "ae_gmm_state_reg_covar": float(state.get("gmm_reg_covar", float("nan"))),
            "ae_gmm_state_smooth_lambda": float(state.get("smooth_lambda", float("nan"))),
            "ae_gmm_state_economic_regime_separation": float(
                selected_config.get("economic_regime_separation", float("nan"))
            ),
            "ae_gmm_state_target_signature_score": float(
                selected_config.get("target_signature_score", float("nan"))
            ),
            "ae_gmm_state_path_aware_hpo": bool(
                selected_config.get("path_aware_hpo", False)
            ),
            "ae_gmm_state_path_cleanliness_score": float(
                selected_config.get("path_cleanliness_score", float("nan"))
            ),
            "ae_gmm_state_clean_positive_contrast": float(
                selected_config.get("clean_positive_contrast", float("nan"))
            ),
            "ae_gmm_state_bad_mae_contrast": float(
                selected_config.get("bad_mae_contrast", float("nan"))
            ),
            "ae_gmm_state_timeout_contrast": float(
                selected_config.get("timeout_contrast", float("nan"))
            ),
            "ae_gmm_state_temporal_concentration_hpo": bool(
                selected_config.get("temporal_concentration_hpo", False)
            ),
            "ae_gmm_state_temporal_concentration_score": float(
                selected_config.get("temporal_concentration_score", float("nan"))
            ),
            "ae_gmm_state_max_cluster_time_bucket_share": float(
                selected_config.get("max_cluster_time_bucket_share", float("nan"))
            ),
            "ae_gmm_state_temporal_stability_score": float(
                selected_config.get("temporal_stability_score", float("nan"))
            ),
            "ae_gmm_state_switch_rate": float(selected_config.get("switch_rate", float("nan"))),
            "ae_gmm_state_side_balance_score": float(
                selected_config.get("side_balance_score", float("nan"))
            ),
            "ae_gmm_state_min_occupancy": float(selected_config.get("min_occupancy", float("nan"))),
            "ae_gmm_state_max_occupancy": float(selected_config.get("max_occupancy", float("nan"))),
            "ae_gmm_state_artifact_dir": str(state_artifact_dir) if state_artifact_dir is not None else None,
            "ae_gmm_frozen_replay_contract": (
                "global and side AE/GMM states fit on the outer train fold are persisted; "
                "validation/OOS rows are transformed with those frozen train-fitted states"
            ),
            **persisted_artifacts,
        },
    )


def _fit_predict(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=96,
        max_depth=8,
        min_samples_leaf=40,
        max_features="sqrt",
        random_state=int(seed),
        n_jobs=2,
    )
    model.fit(
        x_train,
        pd.to_numeric(y_train, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        sample_weight=pd.to_numeric(w_train, errors="coerce").fillna(1.0).to_numpy(dtype=np.float32),
    )
    return model.predict(x_valid).astype(np.float32)


def _timestamp_groups(train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.to_datetime(train_frame["__ts__"], errors="coerce").astype("int64").to_numpy()
    order = np.argsort(ts, kind="mergesort")
    if not len(order):
        return order.astype(np.int64), np.asarray([], dtype=np.int32)
    _, counts = np.unique(ts[order], return_counts=True)
    return order.astype(np.int64, copy=False), counts.astype(np.int32, copy=False)


def _ranker_relevance(
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    target: pd.DataFrame,
    *,
    mode: str,
) -> np.ndarray:
    ts = pd.to_datetime(train_frame["__ts__"].reset_index(drop=True), errors="coerce")
    metrics = train_metrics.reset_index(drop=True)
    target_local = target.reset_index(drop=True)
    u = pd.to_numeric(metrics["u_policy_net"], errors="coerce").fillna(0.0)
    pct_u = u.groupby(ts, sort=False).rank(method="average", pct=True).fillna(0.0)
    if str(mode) == "path_quality":
        group_mean = u.groupby(ts, sort=False).transform("mean")
        group_std = u.groupby(ts, sort=False).transform("std").replace(0.0, np.nan).fillna(1.0)
        realized_edge_z = ((u - group_mean) / group_std).clip(-3.0, 3.0)
        bad_mae = pd.to_numeric(metrics["mae_norm"], errors="coerce").fillna(10.0).ge(1.0)
        timeout = metrics["is_timeout"].astype(float).fillna(1.0).gt(0.5)
        target_soft = pd.to_numeric(target_local["target_soft"], errors="coerce").fillna(0.0)
        clean_path = (
            (u > 0.0)
            & (~bad_mae)
            & (~timeout)
        ).astype(np.float32)
        blended = (
            realized_edge_z.astype(np.float32)
            + 0.65 * pct_u.astype(np.float32)
            + 0.60 * target_soft.astype(np.float32)
            + 0.75 * clean_path
            - 1.50 * bad_mae.astype(np.float32)
            - 0.75 * timeout.astype(np.float32)
        )
        pct = blended.groupby(ts, sort=False).rank(method="average", pct=True).fillna(0.0)
    elif str(mode) == "oracle_enriched":
        group_mean = u.groupby(ts, sort=False).transform("mean")
        group_std = u.groupby(ts, sort=False).transform("std").replace(0.0, np.nan).fillna(1.0)
        realized_edge_z = ((u - group_mean) / group_std).clip(-3.0, 3.0)
        oracle_top = pct_u.ge(0.90).astype(np.float32)
        bad_mae = pd.to_numeric(metrics["mae_norm"], errors="coerce").fillna(10.0).ge(1.0)
        timeout = metrics["is_timeout"].astype(float).fillna(1.0).gt(0.5)
        blended = (
            realized_edge_z.astype(np.float32)
            + 1.00 * oracle_top
            - 0.90 * bad_mae.astype(np.float32)
            - 0.55 * timeout.astype(np.float32)
        )
        pct = blended.groupby(ts, sort=False).rank(method="average", pct=True).fillna(0.0)
    elif str(mode) == "clean_oracle":
        bad_mae = pd.to_numeric(metrics["mae_norm"], errors="coerce").fillna(10.0).ge(1.0)
        timeout = metrics["is_timeout"].astype(float).fillna(1.0).gt(0.5)
        clean_path = (u > 0.0) & (~bad_mae) & (~timeout)
        target_soft = pd.to_numeric(target_local["target_soft"], errors="coerce").fillna(0.0)
        clean_u = u.where(clean_path)
        clean_rank = clean_u.groupby(ts, sort=False).rank(method="average", pct=True).fillna(0.0)
        blended = (
            3.00 * clean_rank.astype(np.float32)
            + 0.75 * clean_path.astype(np.float32)
            + 0.35 * target_soft.astype(np.float32)
            - 1.25 * bad_mae.astype(np.float32)
            - 0.75 * timeout.astype(np.float32)
        )
        pct = blended.groupby(ts, sort=False).rank(method="average", pct=True).fillna(0.0)
    elif str(mode) in {
        "path_first_clean",
        "path_first_clean_dirty_zero",
        "s24_broad_path_first_source",
        "s24_broad_path_first_dirty_zero",
        "s30_side_asymmetric_path_first_source",
        "s30_side_asymmetric_path_first_dirty_zero",
        "timeout_aware_clean_source",
    }:
        bad_mae = pd.to_numeric(metrics["mae_norm"], errors="coerce").fillna(10.0).ge(1.0)
        timeout = metrics["is_timeout"].astype(float).fillna(1.0).gt(0.5)
        mae_norm = pd.to_numeric(metrics["mae_norm"], errors="coerce").fillna(10.0)
        full_stop = pd.Series(False, index=metrics.index)
        for stop_col in ("full_stop_loss", "full_sl", "replay_full_sl"):
            if stop_col in metrics.columns:
                full_stop = pd.to_numeric(metrics[stop_col], errors="coerce").fillna(0.0).gt(0.5)
                break
        side = pd.to_numeric(
            metrics.get("side", pd.Series(1.0, index=metrics.index)),
            errors="coerce",
        ).fillna(1.0)
        side_key = side.where(side < 0.0, 1.0).where(side >= 0.0, -1.0)
        ts_side_pct_u = u.groupby([ts, side_key], sort=False).rank(
            method="average",
            pct=True,
        ).fillna(0.0)
        clean_path = (u > 0.0) & (~bad_mae) & (~timeout) & (~full_stop)
        dirty_positive = (u > 0.0) & (bad_mae | timeout | full_stop)
        relevance_s = pd.Series(0, index=u.index, dtype=np.int32)
        if str(mode) == "timeout_aware_clean_source":
            timeout_clean_positive = (u > 0.0) & timeout & (~bad_mae) & (~full_stop)
            fast_clean = clean_path & mae_norm.lt(0.75)
            relevance_s.loc[timeout_clean_positive] = 1
            relevance_s.loc[clean_path] = 2
            relevance_s.loc[fast_clean & ts_side_pct_u.ge(0.68)] = 3
            relevance_s.loc[fast_clean & ts_side_pct_u.ge(0.88)] = 4
            return np.clip(relevance_s.to_numpy(dtype=np.int32, copy=False), 0, 4)
        if str(mode) in {"path_first_clean", "s24_broad_path_first_source"}:
            relevance_s.loc[dirty_positive] = 1
        if str(mode).startswith("s30_"):
            long_side = side.ge(0.0)
            short_side = ~long_side
            long_clean = clean_path & long_side & mae_norm.lt(0.75)
            short_clean = clean_path & short_side
            if str(mode) == "s30_side_asymmetric_path_first_source":
                relevance_s.loc[dirty_positive & short_side] = 1
                relevance_s.loc[dirty_positive & long_side & mae_norm.lt(1.15) & (~timeout)] = 1
            relevance_s.loc[short_clean] = 2
            relevance_s.loc[long_clean] = 2
            relevance_s.loc[short_clean & ts_side_pct_u.ge(0.70)] = 3
            relevance_s.loc[short_clean & ts_side_pct_u.ge(0.88)] = 4
            relevance_s.loc[long_clean & ts_side_pct_u.ge(0.78)] = 3
            relevance_s.loc[long_clean & ts_side_pct_u.ge(0.92)] = 4
            return np.clip(relevance_s.to_numpy(dtype=np.int32, copy=False), 0, 4)
        relevance_s.loc[clean_path] = 2
        if str(mode).startswith("s24_"):
            relevance_s.loc[clean_path & ts_side_pct_u.ge(0.70)] = 3
            relevance_s.loc[clean_path & ts_side_pct_u.ge(0.88)] = 4
        else:
            relevance_s.loc[clean_path & ts_side_pct_u.ge(0.80)] = 3
            relevance_s.loc[clean_path & ts_side_pct_u.ge(0.90)] = 4
        return np.clip(relevance_s.to_numpy(dtype=np.int32, copy=False), 0, 4)
    else:
        pct = pct_u
    relevance = np.floor(pct.to_numpy(dtype=np.float32, copy=False) * 5.0).astype(np.int32)
    return np.clip(relevance, 0, 4)


def _fit_lgbm_ranker_prediction(
    *,
    x_train: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    target_train: pd.DataFrame,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    seeds: list[int],
    relevance_mode: str,
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
        target_train.reset_index(drop=True),
        mode=str(relevance_mode),
    )
    if int(pd.Series(y).nunique(dropna=True)) < 2:
        return (
            np.full(len(x_train), np.nan, dtype=np.float32),
            np.full(len(x_valid), np.nan, dtype=np.float32),
            "constant_relevance",
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
            n_estimators=48,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
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


def _fit_side_lgbm_ranker_prediction(
    *,
    x_train: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    target_train: pd.DataFrame,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    seeds: list[int],
    relevance_mode: str,
    min_train_rows: int = 500,
) -> tuple[np.ndarray, np.ndarray, str]:
    train_side = pd.to_numeric(
        train_metrics.get("side", pd.Series(1.0, index=train_metrics.index)),
        errors="coerce",
    ).reset_index(drop=True).fillna(1.0)
    valid_side = pd.to_numeric(
        valid_metrics.get("side", pd.Series(1.0, index=valid_metrics.index)),
        errors="coerce",
    ).reset_index(drop=True).fillna(1.0)
    train_out = np.full(len(x_train), np.nan, dtype=np.float32)
    valid_out = np.full(len(x_valid), np.nan, dtype=np.float32)
    statuses: list[str] = []
    for side_name, side_value in (("long", 1.0), ("short", -1.0)):
        if side_value > 0.0:
            train_mask = train_side.ge(0.0).to_numpy(dtype=bool)
            valid_mask = valid_side.ge(0.0).to_numpy(dtype=bool)
        else:
            train_mask = train_side.lt(0.0).to_numpy(dtype=bool)
            valid_mask = valid_side.lt(0.0).to_numpy(dtype=bool)
        train_idx = np.flatnonzero(train_mask)
        valid_idx = np.flatnonzero(valid_mask)
        if len(train_idx) < int(min_train_rows) or not len(valid_idx):
            statuses.append(f"{side_name}:insufficient_rows")
            continue
        train_pred, valid_pred, status = _fit_lgbm_ranker_prediction(
            x_train=x_train.reset_index(drop=True).iloc[train_idx].reset_index(drop=True),
            train_frame=train_frame.reset_index(drop=True).iloc[train_idx].reset_index(drop=True),
            train_metrics=train_metrics.reset_index(drop=True).iloc[train_idx].reset_index(drop=True),
            target_train=target_train.reset_index(drop=True).iloc[train_idx].reset_index(drop=True),
            w_train=w_train.reset_index(drop=True).iloc[train_idx].reset_index(drop=True),
            x_valid=x_valid.reset_index(drop=True).iloc[valid_idx].reset_index(drop=True),
            seeds=seeds,
            relevance_mode=relevance_mode,
        )
        statuses.append(f"{side_name}:{status}")
        if status == "ok":
            train_out[train_idx] = train_pred
            valid_out[valid_idx] = valid_pred
    return (
        train_out,
        valid_out,
        ";".join(statuses) if statuses else "not_run",
    )


def _side_sign_calibrated_ranker_score(
    *,
    train_pred: np.ndarray,
    valid_pred: np.ndarray,
    train_metrics: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    train_relevance: np.ndarray,
) -> tuple[pd.Series, dict[str, Any]]:
    train_side = pd.to_numeric(
        train_metrics.get("side", pd.Series(1.0, index=train_metrics.index)),
        errors="coerce",
    ).reset_index(drop=True).fillna(1.0)
    valid_side = pd.to_numeric(
        valid_metrics.get("side", pd.Series(1.0, index=valid_metrics.index)),
        errors="coerce",
    ).reset_index(drop=True).fillna(1.0)
    train_score = pd.Series(train_pred, dtype=np.float32)
    valid_score = pd.Series(valid_pred, dtype=np.float32)
    relevance = pd.Series(train_relevance, dtype=np.float32)
    out = valid_score.copy()
    diag: dict[str, Any] = {}
    for side_name, side_value in (("long", 1.0), ("short", -1.0)):
        train_mask = train_side.ge(0.0) if side_value > 0.0 else train_side.lt(0.0)
        valid_mask = valid_side.ge(0.0) if side_value > 0.0 else valid_side.lt(0.0)
        side_ic = _spearman(train_score.loc[train_mask], relevance.loc[train_mask])
        sign = -1.0 if math.isfinite(side_ic) and side_ic < 0.0 else 1.0
        out.loc[valid_mask] = sign * valid_score.loc[valid_mask]
        diag[f"s44_{side_name}_train_relevance_ic"] = float(side_ic)
        diag[f"s44_{side_name}_score_sign"] = float(sign)
        diag[f"s44_{side_name}_train_rows"] = int(train_mask.sum())
        diag[f"s44_{side_name}_valid_rows"] = int(valid_mask.sum())
    return out.astype(np.float32), diag


def _score_from_selected_indices(
    *,
    base_score: pd.Series,
    selected_idx: np.ndarray,
) -> pd.Series:
    """Build a ranking score that makes _selection_metrics choose selected_idx.

    _selection_metrics owns the common metric computation and always selects a
    top fraction from finite scores. For constrained selectors, assign the
    constrained rows the highest scores while leaving all rows finite so the
    selected count remains comparable with the unconstrained top-frac selector.
    """
    score = pd.to_numeric(base_score.reset_index(drop=True), errors="coerce").fillna(-1.0e9)
    adjusted = pd.Series(-1.0e6 + score.rank(method="first", pct=True).to_numpy(), index=score.index)
    if len(selected_idx):
        selected = np.asarray(selected_idx, dtype=np.int64)
        selected_order = np.argsort(-score.iloc[selected].to_numpy(dtype=np.float64), kind="mergesort")
        ordered = selected[selected_order]
        adjusted.iloc[ordered] = np.arange(len(ordered), 0, -1, dtype=np.float64)
    return adjusted


def _side_capped_score(
    *,
    score: pd.Series,
    side: pd.Series,
    top_frac: float,
    max_side_share: float,
) -> tuple[pd.Series, dict[str, Any]]:
    score_s = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
    side_s = pd.to_numeric(side.reset_index(drop=True), errors="coerce").fillna(1.0)
    valid_idx = np.flatnonzero(score_s.notna().to_numpy())
    if not len(valid_idx):
        return score_s, {
            "side_cap_enabled": True,
            "side_cap_filled_rows": 0,
            "side_cap_target_rows": 0,
            "side_cap_max_side_share": float(max_side_share),
        }
    target_rows = max(1, int(math.ceil(float(top_frac) * len(valid_idx))))
    max_side_rows = max(1, int(math.floor(float(max_side_share) * target_rows)))
    counts = {1: 0, -1: 0}
    selected: list[int] = []
    order = valid_idx[
        np.argsort(-score_s.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    ]
    for idx in order:
        side_key = -1 if float(side_s.iloc[int(idx)]) < 0.0 else 1
        if counts[side_key] >= max_side_rows:
            continue
        selected.append(int(idx))
        counts[side_key] += 1
        if len(selected) >= target_rows:
            break
    if len(selected) < target_rows:
        chosen = set(selected)
        for idx in order:
            idx_int = int(idx)
            if idx_int in chosen:
                continue
            selected.append(idx_int)
            side_key = -1 if float(side_s.iloc[idx_int]) < 0.0 else 1
            counts[side_key] += 1
            if len(selected) >= target_rows:
                break
    adjusted = _score_from_selected_indices(
        base_score=score_s,
        selected_idx=np.asarray(selected, dtype=np.int64),
    )
    max_actual_share = (
        max(counts.values()) / float(len(selected)) if selected else float("nan")
    )
    return adjusted, {
        "side_cap_enabled": True,
        "side_cap_filled_rows": int(len(selected)),
        "side_cap_target_rows": int(target_rows),
        "side_cap_max_side_share": float(max_side_share),
        "side_cap_actual_max_side_share": float(max_actual_share)
        if math.isfinite(max_actual_share)
        else float("nan"),
        "side_cap_long_rows": int(counts[1]),
        "side_cap_short_rows": int(counts[-1]),
    }


def _constrained_top_indices(
    *,
    score: pd.Series,
    side: pd.Series,
    eligible: pd.Series,
    top_frac: float,
    max_side_share: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    score_s = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
    side_s = pd.to_numeric(side.reset_index(drop=True), errors="coerce").fillna(1.0)
    eligible_s = eligible.reset_index(drop=True).fillna(False).astype(bool)
    finite = score_s.notna()
    base_valid_idx = np.flatnonzero(finite.to_numpy())
    target_rows = max(1, int(math.ceil(float(top_frac) * len(base_valid_idx)))) if len(base_valid_idx) else 0
    eligible_idx = np.flatnonzero((finite & eligible_s).to_numpy())
    if target_rows <= 0 or not len(eligible_idx):
        return np.array([], dtype=np.int64), {
            "hard_risk_cap_enabled": True,
            "hard_risk_cap_target_rows": int(target_rows),
            "hard_risk_cap_eligible_rows": int(len(eligible_idx)),
            "hard_risk_cap_selected_rows": 0,
            "hard_risk_cap_no_trade_rate": 1.0,
        }
    max_side_rows = max(1, int(math.floor(float(max_side_share) * target_rows)))
    counts = {1: 0, -1: 0}
    selected: list[int] = []
    order = eligible_idx[
        np.argsort(-score_s.iloc[eligible_idx].to_numpy(dtype=np.float64), kind="mergesort")
    ]
    for idx in order:
        side_key = -1 if float(side_s.iloc[int(idx)]) < 0.0 else 1
        if counts[side_key] >= max_side_rows:
            continue
        selected.append(int(idx))
        counts[side_key] += 1
        if len(selected) >= target_rows:
            break
    while selected:
        current_max_side = max(counts, key=counts.get)
        current_share = counts[current_max_side] / float(len(selected))
        minority_side = -current_max_side
        minority_count = counts[minority_side]
        if current_share <= float(max_side_share) or minority_count <= 0:
            break
        majority_positions = [
            pos
            for pos, idx in enumerate(selected)
            if (-1 if float(side_s.iloc[int(idx)]) < 0.0 else 1) == current_max_side
        ]
        if not majority_positions:
            break
        worst_pos = min(
            majority_positions,
            key=lambda pos: float(score_s.iloc[int(selected[pos])]),
        )
        removed = selected.pop(worst_pos)
        removed_side = -1 if float(side_s.iloc[int(removed)]) < 0.0 else 1
        counts[removed_side] -= 1
    max_actual_share = (
        max(counts.values()) / float(len(selected)) if selected else float("nan")
    )
    no_trade_rate = 1.0 - (float(len(selected)) / float(target_rows))
    return np.asarray(selected, dtype=np.int64), {
        "hard_risk_cap_enabled": True,
        "hard_risk_cap_target_rows": int(target_rows),
        "hard_risk_cap_eligible_rows": int(len(eligible_idx)),
        "hard_risk_cap_selected_rows": int(len(selected)),
        "hard_risk_cap_no_trade_rate": float(max(0.0, min(1.0, no_trade_rate))),
        "side_cap_enabled": True,
        "side_cap_max_side_share": float(max_side_share),
        "side_cap_actual_max_side_share": float(max_actual_share)
        if math.isfinite(max_actual_share)
        else float("nan"),
        "side_cap_long_rows": int(counts[1]),
        "side_cap_short_rows": int(counts[-1]),
    }


def _budgeted_top_indices(
    *,
    score: pd.Series,
    side: pd.Series,
    eligible: pd.Series,
    bad_risk: pd.Series,
    timeout_risk: pd.Series,
    top_frac: float,
    max_side_share: float,
    bad_risk_budget: float,
    timeout_risk_budget: float,
    budget_mode: str,
    min_fill_ratio: float = 0.90,
) -> tuple[np.ndarray, dict[str, Any]]:
    score_s = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
    side_s = pd.to_numeric(side.reset_index(drop=True), errors="coerce").fillna(1.0)
    eligible_s = eligible.reset_index(drop=True).fillna(False).astype(bool)
    bad_s = (
        pd.to_numeric(bad_risk.reset_index(drop=True), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.0, upper=1.0)
    )
    timeout_s = (
        pd.to_numeric(timeout_risk.reset_index(drop=True), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.0, upper=1.0)
    )
    finite = score_s.notna()
    base_valid_idx = np.flatnonzero(finite.to_numpy())
    target_rows = max(1, int(math.ceil(float(top_frac) * len(base_valid_idx)))) if len(base_valid_idx) else 0
    eligible_idx = np.flatnonzero((finite & eligible_s).to_numpy())
    if target_rows <= 0 or not len(eligible_idx):
        return np.array([], dtype=np.int64), {
            "budgeted_allocator_enabled": True,
            "budgeted_allocator_mode": str(budget_mode),
            "budgeted_allocator_target_rows": int(target_rows),
            "budgeted_allocator_eligible_rows": int(len(eligible_idx)),
            "budgeted_allocator_selected_rows": 0,
            "budgeted_allocator_no_trade_rate": 1.0,
            "budgeted_allocator_budget_fallback_rows": 0,
            "budgeted_bad_mae_budget": float(bad_risk_budget),
            "budgeted_timeout_budget": float(timeout_risk_budget),
        }
    max_side_rows = max(1, int(math.floor(float(max_side_share) * target_rows)))
    min_fill_rows = max(1, int(math.ceil(float(min_fill_ratio) * target_rows)))
    order = eligible_idx[
        np.argsort(-score_s.iloc[eligible_idx].to_numpy(dtype=np.float64), kind="mergesort")
    ]
    counts = {1: 0, -1: 0}
    bad_sums = {1: 0.0, -1: 0.0}
    timeout_sums = {1: 0.0, -1: 0.0}
    selected: list[int] = []
    selected_set: set[int] = set()

    def _side_key(idx: int) -> int:
        return -1 if float(side_s.iloc[int(idx)]) < 0.0 else 1

    def _within_side_cap(idx: int) -> bool:
        return counts[_side_key(idx)] < max_side_rows

    def _within_budget(idx: int) -> bool:
        side_key = _side_key(idx)
        bad_val = float(bad_s.iloc[int(idx)])
        timeout_val = float(timeout_s.iloc[int(idx)])
        if str(budget_mode) == "side":
            denom = float(counts[side_key] + 1)
            next_bad = (bad_sums[side_key] + bad_val) / denom
            next_timeout = (timeout_sums[side_key] + timeout_val) / denom
        else:
            denom = float(len(selected) + 1)
            next_bad = (sum(bad_sums.values()) + bad_val) / denom
            next_timeout = (sum(timeout_sums.values()) + timeout_val) / denom
        return next_bad <= float(bad_risk_budget) and next_timeout <= float(timeout_risk_budget)

    def _add(idx: int) -> None:
        idx_int = int(idx)
        side_key = _side_key(idx_int)
        selected.append(idx_int)
        selected_set.add(idx_int)
        counts[side_key] += 1
        bad_sums[side_key] += float(bad_s.iloc[idx_int])
        timeout_sums[side_key] += float(timeout_s.iloc[idx_int])

    for idx in order:
        idx_int = int(idx)
        if idx_int in selected_set or not _within_side_cap(idx_int) or not _within_budget(idx_int):
            continue
        _add(idx_int)
        if len(selected) >= target_rows:
            break

    budget_fallback_rows = 0
    if len(selected) < min_fill_rows:
        for idx in order:
            idx_int = int(idx)
            if idx_int in selected_set or not _within_side_cap(idx_int):
                continue
            _add(idx_int)
            budget_fallback_rows += 1
            if len(selected) >= min_fill_rows:
                break

    if len(selected) < target_rows:
        for idx in order:
            idx_int = int(idx)
            if idx_int in selected_set:
                continue
            _add(idx_int)
            budget_fallback_rows += 1
            if len(selected) >= target_rows:
                break

    selected_arr = np.asarray(selected, dtype=np.int64)
    if len(selected_arr):
        selected_bad = bad_s.iloc[selected_arr].to_numpy(dtype=np.float64)
        selected_timeout = timeout_s.iloc[selected_arr].to_numpy(dtype=np.float64)
        selected_sides = np.asarray([_side_key(int(idx)) for idx in selected_arr], dtype=np.int8)
        long_mask = selected_sides > 0
        short_mask = selected_sides < 0
        bad_mean = float(np.mean(selected_bad))
        timeout_mean = float(np.mean(selected_timeout))
        long_bad_mean = float(np.mean(selected_bad[long_mask])) if bool(long_mask.any()) else float("nan")
        short_bad_mean = float(np.mean(selected_bad[short_mask])) if bool(short_mask.any()) else float("nan")
        long_timeout_mean = (
            float(np.mean(selected_timeout[long_mask])) if bool(long_mask.any()) else float("nan")
        )
        short_timeout_mean = (
            float(np.mean(selected_timeout[short_mask])) if bool(short_mask.any()) else float("nan")
        )
    else:
        bad_mean = timeout_mean = long_bad_mean = short_bad_mean = float("nan")
        long_timeout_mean = short_timeout_mean = float("nan")
    max_actual_share = (
        max(counts.values()) / float(len(selected)) if selected else float("nan")
    )
    no_trade_rate = 1.0 - (float(len(selected)) / float(target_rows))
    return selected_arr, {
        "budgeted_allocator_enabled": True,
        "budgeted_allocator_mode": str(budget_mode),
        "budgeted_allocator_target_rows": int(target_rows),
        "budgeted_allocator_eligible_rows": int(len(eligible_idx)),
        "budgeted_allocator_selected_rows": int(len(selected)),
        "budgeted_allocator_no_trade_rate": float(max(0.0, min(1.0, no_trade_rate))),
        "budgeted_allocator_budget_fallback_rows": int(budget_fallback_rows),
        "budgeted_allocator_min_fill_ratio": float(min_fill_ratio),
        "budgeted_bad_mae_budget": float(bad_risk_budget),
        "budgeted_timeout_budget": float(timeout_risk_budget),
        "budgeted_bad_mae_pred_mean": bad_mean,
        "budgeted_timeout_pred_mean": timeout_mean,
        "budgeted_long_bad_mae_pred_mean": long_bad_mean,
        "budgeted_short_bad_mae_pred_mean": short_bad_mean,
        "budgeted_long_timeout_pred_mean": long_timeout_mean,
        "budgeted_short_timeout_pred_mean": short_timeout_mean,
        "side_cap_enabled": True,
        "side_cap_max_side_share": float(max_side_share),
        "side_cap_actual_max_side_share": float(max_actual_share)
        if math.isfinite(max_actual_share)
        else float("nan"),
        "side_cap_long_rows": int(counts[1]),
        "side_cap_short_rows": int(counts[-1]),
    }


def _per_timestamp_top_mask(
    frame: pd.DataFrame,
    score: pd.Series,
    *,
    top_n: int | None = None,
    top_frac: float | None = None,
) -> np.ndarray:
    score_arr = pd.to_numeric(score.reset_index(drop=True), errors="coerce").to_numpy(
        dtype=np.float32,
        copy=False,
    )
    ts = pd.to_datetime(frame["__ts__"].reset_index(drop=True), errors="coerce")
    out = np.zeros(len(score_arr), dtype=bool)
    valid = np.isfinite(score_arr) & ts.notna().to_numpy()
    if not bool(valid.any()):
        return out
    positions = pd.Series(np.arange(len(score_arr), dtype=np.int64))
    for _, ids in positions[valid].groupby(ts[valid], sort=False):
        idx = ids.to_numpy(dtype=np.int64)
        if not len(idx):
            continue
        if top_n is not None:
            k = min(int(top_n), len(idx))
        elif top_frac is not None:
            k = max(1, int(math.ceil(float(top_frac) * len(idx))))
        else:
            k = len(idx)
        if k <= 0:
            continue
        order = idx[np.argsort(-score_arr[idx], kind="mergesort")[:k]]
        out[order] = True
    return out


def _max_percentile_score(scores: list[pd.Series], length: int) -> pd.Series:
    parts: list[pd.Series] = []
    for score in scores:
        raw = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
        if raw.notna().any():
            parts.append(raw.rank(method="average", pct=True).astype(np.float32))
    if not parts:
        return pd.Series(np.nan, index=pd.RangeIndex(length), dtype=np.float32)
    return pd.concat(parts, axis=1).max(axis=1).astype(np.float32)


def _timestamp_rank_percentile(
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    ascending: bool,
) -> pd.Series:
    values_s = pd.to_numeric(values.reset_index(drop=True), errors="coerce")
    ts = pd.to_datetime(frame["__ts__"].reset_index(drop=True), errors="coerce")
    out = pd.Series(np.nan, index=values_s.index, dtype=np.float32)
    valid = values_s.notna() & ts.notna()
    if not bool(valid.any()):
        return out
    ranked = values_s[valid].groupby(ts[valid], sort=False).rank(
        method="average",
        pct=True,
        ascending=ascending,
    )
    out.loc[valid] = ranked.astype(np.float32)
    return out


def _augment_s42_source_features(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_aug = x_train.copy()
    valid_aug = x_valid.copy()
    base_candidates = (
        "median_spread_bps",
        "log_quote_volume",
    )
    state_candidates = (
        "gmm_posterior_max",
        "gmm_posterior_margin",
        "gmm_entropy",
        "mahalanobis_distance",
        "cluster_speed",
        "cluster_acceleration",
        "AE_reconstruction_error",
        "ae_reconstruction_error",
        "dae_reconstruction_error",
        "dae_reconstruction_error_zscore",
        "dae_reconstruction_error_delta_1",
        "dae_reconstruction_error_accel_1",
        "latent_mahalanobis_drift",
        "latent_speed",
        "latent_acceleration",
        "long_gmm_posterior_max",
        "long_gmm_entropy",
        "long_mahalanobis_distance",
        "long_cluster_speed",
        "long_cluster_acceleration",
        "long_AE_reconstruction_error",
        "long_ae_reconstruction_error",
        "long_dae_reconstruction_error",
        "long_dae_reconstruction_error_zscore",
        "long_dae_reconstruction_error_delta_1",
        "long_dae_reconstruction_error_accel_1",
        "long_latent_mahalanobis_drift",
        "long_latent_speed",
        "long_latent_acceleration",
        "short_gmm_posterior_max",
        "short_gmm_entropy",
        "short_mahalanobis_distance",
        "short_cluster_speed",
        "short_cluster_acceleration",
        "short_AE_reconstruction_error",
        "short_ae_reconstruction_error",
        "short_dae_reconstruction_error",
        "short_dae_reconstruction_error_zscore",
        "short_dae_reconstruction_error_delta_1",
        "short_dae_reconstruction_error_accel_1",
        "short_latent_mahalanobis_drift",
        "short_latent_speed",
        "short_latent_acceleration",
    )
    added: list[str] = []

    def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
        return (
            pd.to_numeric(frame[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )

    for base in base_candidates:
        if base not in train_aug.columns or base not in valid_aug.columns:
            continue
        train_base = _numeric_column(train_aug, base)
        valid_base = _numeric_column(valid_aug, base)
        for state in state_candidates:
            if state not in train_aug.columns or state not in valid_aug.columns:
                continue
            name = f"s42_{base}_x_{state}"
            train_aug[name] = (train_base * _numeric_column(train_aug, state)).astype(
                np.float32
            )
            valid_aug[name] = (valid_base * _numeric_column(valid_aug, state)).astype(
                np.float32
            )
            added.append(name)
    diag = {
        "s42_interaction_feature_count": int(len(added)),
        "s42_interaction_features": ",".join(added[:80]),
        "s42_interaction_features_truncated": int(max(len(added) - 80, 0)),
    }
    return train_aug, valid_aug, diag


def _recent_train_indices(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    lookback_days: int,
    min_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_ts = pd.to_datetime(train_frame["__ts__"].reset_index(drop=True), errors="coerce")
    valid_ts = pd.to_datetime(valid_frame["__ts__"].reset_index(drop=True), errors="coerce")
    valid_min = valid_ts.min()
    all_idx = np.arange(len(train_frame), dtype=np.int64)
    if pd.isna(valid_min) or int(lookback_days) <= 0:
        return all_idx, {
            "recent_train_window_status": "full_train_invalid_window",
            "recent_train_window_days": int(lookback_days),
            "recent_train_rows": int(len(all_idx)),
            "recent_train_full_rows": int(len(all_idx)),
        }
    start = valid_min - pd.Timedelta(days=int(lookback_days))
    recent_mask = train_ts.ge(start) & train_ts.lt(valid_min)
    idx = np.flatnonzero(recent_mask.fillna(False).to_numpy(dtype=bool))
    status = "recent_train_ok"
    if int(len(idx)) < int(min_rows):
        idx = all_idx
        status = "recent_train_fallback_full_train"
    return idx.astype(np.int64, copy=False), {
        "recent_train_window_status": status,
        "recent_train_window_days": int(lookback_days),
        "recent_train_rows": int(len(idx)),
        "recent_train_full_rows": int(len(all_idx)),
    }


def _mask_from_indices(length: int, selected_idx: np.ndarray | None) -> np.ndarray:
    mask = np.zeros(int(length), dtype=bool)
    if selected_idx is None:
        return mask
    idx = np.asarray(selected_idx, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < int(length))]
    mask[idx] = True
    return mask


def _oracle_recall_stats(
    *,
    metrics: pd.DataFrame,
    mask: np.ndarray,
    top_frac: float,
    prefix: str,
) -> dict[str, Any]:
    metrics_local = metrics.reset_index(drop=True)
    mask_arr = np.asarray(mask, dtype=bool)
    if len(mask_arr) != len(metrics_local):
        raise ValueError(f"{prefix} mask length must match metrics length")
    oracle_idx = _rank_top_indices(metrics_local["u_policy_net"], top_frac)
    oracle_mask = np.zeros(len(metrics_local), dtype=bool)
    oracle_mask[oracle_idx] = True
    side = pd.to_numeric(metrics_local["side"], errors="coerce").fillna(1.0)
    selected_metrics = metrics_local.loc[mask_arr]
    lower_tail_cutoff = _safe_quantile(metrics_local["u_policy_net"], 0.10)

    out: dict[str, Any] = {
        f"{prefix}_rows": int(mask_arr.sum()),
        f"{prefix}_row_share": float(mask_arr.mean()) if len(mask_arr) else float("nan"),
        f"{prefix}_oracle_rows": int(oracle_mask.sum()),
        f"{prefix}_oracle_hit_rows": int((mask_arr & oracle_mask).sum()),
        f"{prefix}_oracle_recall": (
            float((mask_arr & oracle_mask).sum() / max(int(oracle_mask.sum()), 1))
            if int(oracle_mask.sum())
            else float("nan")
        ),
        f"{prefix}_mean_u": _safe_mean(selected_metrics["u_policy_net"]),
        f"{prefix}_q10_u": _safe_quantile(selected_metrics["u_policy_net"], 0.10),
        f"{prefix}_bad_mae_1r_rate": _safe_mean(selected_metrics["mae_norm"] >= 1.0),
        f"{prefix}_timeout_rate": _safe_mean(selected_metrics["is_timeout"].astype(float) > 0.5),
        f"{prefix}_lower_tail_rate": _safe_mean(
            selected_metrics["u_policy_net"] <= lower_tail_cutoff
        ),
    }
    for side_name, side_mask in (
        ("long", (side > 0.0).to_numpy()),
        ("short", (side < 0.0).to_numpy()),
    ):
        local_oracle = oracle_mask & side_mask
        out[f"{prefix}_{side_name}_oracle_rows"] = int(local_oracle.sum())
        out[f"{prefix}_{side_name}_oracle_hit_rows"] = int((mask_arr & local_oracle).sum())
        out[f"{prefix}_{side_name}_oracle_recall"] = (
            float((mask_arr & local_oracle).sum() / max(int(local_oracle.sum()), 1))
            if int(local_oracle.sum())
            else float("nan")
        )
    return out


def _clean_dirty_selected_diagnostics(
    *,
    metrics: pd.DataFrame,
    score: pd.Series,
    selector: str,
    month: str,
    top_frac: float,
    selected_idx: np.ndarray | None,
    base_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    metrics_local = metrics.reset_index(drop=True)
    score_s = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
    idx = (
        _rank_top_indices(score_s, top_frac)
        if selected_idx is None
        else np.asarray(selected_idx, dtype=np.int64)
    )
    idx = idx[(idx >= 0) & (idx < len(metrics_local))]
    selected_mask = np.zeros(len(metrics_local), dtype=bool)
    selected_mask[idx] = True

    u = pd.to_numeric(metrics_local["u_policy_net"], errors="coerce").fillna(0.0)
    bad_mae = pd.to_numeric(metrics_local["mae_norm"], errors="coerce").fillna(10.0).ge(1.0)
    timeout = metrics_local["is_timeout"].astype(float).fillna(1.0).gt(0.5)
    positive_u = u.gt(0.0)
    clean_positive = positive_u & (~bad_mae) & (~timeout)
    dirty_positive = positive_u & (bad_mae | timeout)
    oracle_mask = np.zeros(len(metrics_local), dtype=bool)
    oracle_mask[_rank_top_indices(u, top_frac)] = True
    clean_oracle = oracle_mask & clean_positive.to_numpy(dtype=bool)
    side = pd.to_numeric(
        metrics_local.get("side", pd.Series(1.0, index=metrics_local.index)),
        errors="coerce",
    ).fillna(1.0)
    base = dict(base_fields or {})
    rows: list[dict[str, Any]] = []
    side_masks = (
        ("all", np.ones(len(metrics_local), dtype=bool)),
        ("long", side.gt(0.0).to_numpy(dtype=bool)),
        ("short", side.lt(0.0).to_numpy(dtype=bool)),
    )
    for side_name, side_mask in side_masks:
        local_selected = selected_mask & side_mask
        selected_rows = int(local_selected.sum())
        selected_metrics = metrics_local.loc[local_selected]
        selected_positive = local_selected & positive_u.to_numpy(dtype=bool)
        selected_clean = local_selected & clean_positive.to_numpy(dtype=bool)
        selected_dirty = local_selected & dirty_positive.to_numpy(dtype=bool)
        local_oracle = oracle_mask & side_mask
        local_clean_oracle = clean_oracle & side_mask
        clean_score = score_s.loc[selected_clean]
        dirty_score = score_s.loc[selected_dirty]
        mean_clean_score = _safe_mean(clean_score)
        mean_dirty_score = _safe_mean(dirty_score)
        rows.append(
            {
                **base,
                "selector": selector,
                "month": month,
                "side": side_name,
                "top_frac": float(top_frac),
                "selected_rows": selected_rows,
                "mean_u": _safe_mean(selected_metrics["u_policy_net"]),
                "clean_positive_rate": _safe_mean(clean_positive.loc[local_selected]),
                "dirty_positive_rate": _safe_mean(dirty_positive.loc[local_selected]),
                "bad_mae_1r_rate": _safe_mean(selected_metrics["mae_norm"] >= 1.0),
                "timeout_rate": _safe_mean(
                    selected_metrics["is_timeout"].astype(float) > 0.5
                ),
                "oracle_recall": (
                    float((local_selected & local_oracle).sum() / max(int(local_oracle.sum()), 1))
                    if int(local_oracle.sum())
                    else float("nan")
                ),
                "clean_oracle_recall": (
                    float(
                        (local_selected & local_clean_oracle).sum()
                        / max(int(local_clean_oracle.sum()), 1)
                    )
                    if int(local_clean_oracle.sum())
                    else float("nan")
                ),
                "dirty_positive_share_of_positive_u": (
                    float(selected_dirty.sum() / max(int(selected_positive.sum()), 1))
                    if int(selected_positive.sum())
                    else float("nan")
                ),
                "mean_rank_score_clean_positive": mean_clean_score,
                "mean_rank_score_dirty_positive": mean_dirty_score,
                "score_gap_clean_minus_dirty": (
                    mean_clean_score - mean_dirty_score
                    if math.isfinite(mean_clean_score) and math.isfinite(mean_dirty_score)
                    else float("nan")
                ),
            }
        )
    return rows


def _discovery_context_bucket(feature: str) -> str:
    lower = str(feature).lower()
    if lower.startswith("long_"):
        return "long_ae_gmm"
    if lower.startswith("short_"):
        return "short_ae_gmm"
    if any(token in lower for token in ("gmm", "cluster", "archetype", "posterior", "mahalanobis", "reconstruction", "latent")):
        return "global_ae_gmm"
    return "market_state"


def _discovery_context_scores(frame: pd.DataFrame, features: list[str]) -> dict[str, pd.Series]:
    bucketed: dict[str, list[tuple[str, pd.Series]]] = {
        "market_state": [],
        "global_ae_gmm": [],
        "long_ae_gmm": [],
        "short_ae_gmm": [],
    }
    for feature in features:
        if feature not in frame.columns:
            continue
        lower = str(feature).lower()
        bucket = _discovery_context_bucket(str(feature))
        is_priority_context = "ae_gmm_oof_available" in lower
        if (
            bucket != "market_state"
            and not is_priority_context
            and not any(keyword in lower for keyword in DISCOVERY_CONTEXT_KEYWORDS)
        ):
            continue
        series = pd.to_numeric(frame[feature], errors="coerce")
        if not bool(series.notna().any()):
            continue
        limit = int(DISCOVERY_CONTEXT_BUCKET_LIMITS.get(bucket, 0))
        item = (str(feature), series.astype(np.float32))
        if is_priority_context:
            bucketed[bucket].insert(0, item)
            if len(bucketed[bucket]) > limit:
                bucketed[bucket] = bucketed[bucket][:limit]
        elif len(bucketed[bucket]) < limit:
            bucketed[bucket].append(item)
    out: dict[str, pd.Series] = {}
    for bucket in ("market_state", "global_ae_gmm", "long_ae_gmm", "short_ae_gmm"):
        for feature, series in bucketed[bucket]:
            out[f"ctx_{feature}"] = series
            if len(out) >= MAX_DISCOVERY_CONTEXT_COLUMNS:
                return out
    return out


def _bucket_codes_from_train(
    train_values: pd.Series,
    valid_values: pd.Series,
    *,
    bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    train_num = pd.to_numeric(train_values, errors="coerce").to_numpy(dtype=np.float64)
    valid_num = pd.to_numeric(valid_values, errors="coerce").to_numpy(dtype=np.float64)
    finite_train = train_num[np.isfinite(train_num)]
    train_codes = np.full(len(train_num), -1, dtype=np.int16)
    valid_codes = np.full(len(valid_num), -1, dtype=np.int16)
    if len(finite_train) < max(bins * 10, 50):
        return train_codes, valid_codes
    quantiles = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)[1:-1]
    edges = np.unique(np.nanquantile(finite_train, quantiles))
    if len(edges) == 0:
        train_codes[np.isfinite(train_num)] = 0
        valid_codes[np.isfinite(valid_num)] = 0
        return train_codes, valid_codes
    train_codes[np.isfinite(train_num)] = np.searchsorted(
        edges,
        train_num[np.isfinite(train_num)],
        side="right",
    ).astype(np.int16)
    valid_codes[np.isfinite(valid_num)] = np.searchsorted(
        edges,
        valid_num[np.isfinite(valid_num)],
        side="right",
    ).astype(np.int16)
    return train_codes, valid_codes


def _prior_bucket_quality_overlay(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    train_metrics: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    features: list[str],
    min_bucket_rows: int = 80,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    available = [feature for feature in SOURCE_BUCKET_QUALITY_FEATURES if feature in features and feature in train.columns and feature in valid.columns]
    index = valid.reset_index(drop=True).index
    if not available:
        zeros = pd.Series(np.zeros(len(valid), dtype=np.float32), index=index)
        return zeros, zeros, zeros, zeros, {
            "s22_bucket_quality_feature_count": 0,
            "s22_bucket_quality_features": "",
            "s22_bucket_quality_status": "no_features",
        }

    train_m = train_metrics.reset_index(drop=True)
    valid_m = valid_metrics.reset_index(drop=True)
    u = pd.to_numeric(train_m["u_policy_net"], errors="coerce").fillna(0.0)
    bad = pd.to_numeric(train_m["mae_norm"], errors="coerce").fillna(10.0).ge(1.0)
    timeout = pd.to_numeric(train_m["is_timeout"], errors="coerce").fillna(1.0).gt(0.5)
    clean = u.gt(0.0) & (~bad) & (~timeout)
    dirty = u.gt(0.0) & (bad | timeout)
    train_side = np.where(
        pd.to_numeric(train_m["side"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32) < 0.0,
        -1,
        1,
    )
    valid_side = np.where(
        pd.to_numeric(valid_m["side"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32) < 0.0,
        -1,
        1,
    )
    train_local = pd.DataFrame(
        {
            "u": u.to_numpy(dtype=np.float32, copy=False),
            "bad": bad.to_numpy(dtype=np.float32, copy=False),
            "timeout": timeout.to_numpy(dtype=np.float32, copy=False),
            "clean": clean.to_numpy(dtype=np.float32, copy=False),
            "dirty": dirty.to_numpy(dtype=np.float32, copy=False),
            "side_key": train_side,
        }
    )
    feature_quality: list[np.ndarray] = []
    relaxed_counts = np.zeros(len(valid), dtype=np.float32)
    strict_counts = np.zeros(len(valid), dtype=np.float32)
    used_features: list[str] = []
    for feature in available:
        train_codes, valid_codes = _bucket_codes_from_train(train[feature], valid[feature])
        if np.max(train_codes) < 0 or np.max(valid_codes) < 0:
            continue
        local = train_local.copy()
        local["bucket"] = train_codes
        local = local[local["bucket"] >= 0]
        if local.empty:
            continue
        by_bucket_side = local.groupby(["bucket", "side_key"], sort=False).agg(
            rows=("u", "size"),
            mean_u=("u", "mean"),
            bad=("bad", "mean"),
            timeout=("timeout", "mean"),
            clean=("clean", "mean"),
            dirty=("dirty", "mean"),
        )
        by_bucket = local.groupby(["bucket"], sort=False).agg(
            rows=("u", "size"),
            mean_u=("u", "mean"),
            bad=("bad", "mean"),
            timeout=("timeout", "mean"),
            clean=("clean", "mean"),
            dirty=("dirty", "mean"),
        )
        values = np.full(len(valid), 0.0, dtype=np.float32)
        relaxed = np.zeros(len(valid), dtype=bool)
        strict = np.zeros(len(valid), dtype=bool)
        for i, bucket in enumerate(valid_codes):
            if bucket < 0:
                continue
            stats = None
            key = (int(bucket), int(valid_side[i]))
            if key in by_bucket_side.index and int(by_bucket_side.loc[key, "rows"]) >= min_bucket_rows:
                stats = by_bucket_side.loc[key]
            elif int(bucket) in by_bucket.index and int(by_bucket.loc[int(bucket), "rows"]) >= min_bucket_rows:
                stats = by_bucket.loc[int(bucket)]
            if stats is None:
                continue
            rows = float(stats["rows"])
            mean_u = float(stats["mean_u"])
            bad_rate = float(stats["bad"])
            timeout_rate = float(stats["timeout"])
            clean_rate = float(stats["clean"])
            dirty_rate = float(stats["dirty"])
            mean_u_score = float(np.clip(mean_u / 0.004, -1.0, 1.0))
            clean_gap = clean_rate - dirty_rate
            row_conf = float(np.clip(np.log1p(rows) / np.log1p(1000.0), 0.0, 1.0))
            quality = (
                0.40 * mean_u_score
                + 0.35 * clean_gap
                - 0.20 * max(bad_rate - 0.50, 0.0)
                - 0.20 * max(timeout_rate - 0.12, 0.0)
                + 0.10 * row_conf
            )
            values[i] = np.float32(quality)
            relaxed[i] = (
                rows >= min_bucket_rows
                and mean_u > 0.0
                and bad_rate <= 0.70
                and timeout_rate <= 0.20
                and clean_rate >= dirty_rate
            )
            strict[i] = (
                rows >= min_bucket_rows
                and mean_u > 0.0
                and bad_rate <= 0.65
                and timeout_rate <= 0.15
                and clean_rate > dirty_rate
            )
        feature_quality.append(values)
        relaxed_counts += relaxed.astype(np.float32)
        strict_counts += strict.astype(np.float32)
        used_features.append(feature)
    if not feature_quality:
        zeros = pd.Series(np.zeros(len(valid), dtype=np.float32), index=index)
        return zeros, zeros, zeros, zeros, {
            "s22_bucket_quality_feature_count": 0,
            "s22_bucket_quality_features": "",
            "s22_bucket_quality_status": "no_usable_buckets",
        }
    quality_arr = np.nanmean(np.vstack(feature_quality), axis=0).astype(np.float32)
    quality = pd.Series(quality_arr, index=index)
    quality_rank = quality.rank(method="average", pct=True).fillna(0.0).astype(np.float32)
    relaxed_count_s = pd.Series(relaxed_counts, index=index, dtype=np.float32)
    strict_count_s = pd.Series(strict_counts, index=index, dtype=np.float32)
    return quality, quality_rank, relaxed_count_s, strict_count_s, {
        "s22_bucket_quality_feature_count": int(len(used_features)),
        "s22_bucket_quality_features": ",".join(used_features),
        "s22_bucket_quality_min_bucket_rows": int(min_bucket_rows),
        "s22_bucket_quality_status": "ok",
        "s22_bucket_relaxed_pass_rate": float((relaxed_counts > 0).mean()),
        "s22_bucket_strict_pass_rate": float((strict_counts > 0).mean()),
    }


def _side_spread_aegmm_bucket_quality(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    train_metrics: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    min_bucket_rows: int = 120,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    index = valid.reset_index(drop=True).index
    if "median_spread_bps" not in train.columns or "median_spread_bps" not in valid.columns:
        zeros = pd.Series(np.zeros(len(valid), dtype=np.float32), index=index)
        return zeros, zeros, {
            "s46_bucket_quality_status": "missing_median_spread_bps",
            "s46_bucket_quality_min_bucket_rows": int(min_bucket_rows),
        }

    def _posterior_codes(frame: pd.DataFrame, metrics: pd.DataFrame) -> np.ndarray:
        side = pd.to_numeric(
            metrics.get("side", pd.Series(1.0, index=metrics.index)),
            errors="coerce",
        ).reset_index(drop=True).fillna(1.0)
        codes = np.full(len(frame), -1, dtype=np.int16)
        global_cols = [
            col for col in frame.columns if re.match(r"^gmm_cluster_posterior_\d+$", str(col))
        ]
        long_cols = [
            col for col in frame.columns if re.match(r"^long_gmm_cluster_posterior_\d+$", str(col))
        ]
        short_cols = [
            col for col in frame.columns if re.match(r"^short_gmm_cluster_posterior_\d+$", str(col))
        ]
        for mask, cols in (
            (side.ge(0.0).to_numpy(dtype=bool), long_cols or global_cols),
            (side.lt(0.0).to_numpy(dtype=bool), short_cols or global_cols),
        ):
            if not cols or not bool(mask.any()):
                continue
            values = frame.loc[mask, cols].apply(pd.to_numeric, errors="coerce")
            if values.shape[1] <= 0:
                continue
            valid_rows = values.notna().any(axis=1).to_numpy(dtype=bool)
            if not bool(valid_rows.any()):
                continue
            local_codes = np.argmax(
                values.fillna(-np.inf).to_numpy(dtype=np.float32),
                axis=1,
            ).astype(np.int16)
            target_idx = np.flatnonzero(mask)
            codes[target_idx[valid_rows]] = local_codes[valid_rows]
        return codes

    train_m = train_metrics.reset_index(drop=True)
    valid_m = valid_metrics.reset_index(drop=True)
    u = pd.to_numeric(train_m["u_policy_net"], errors="coerce").fillna(0.0)
    bad = pd.to_numeric(train_m["mae_norm"], errors="coerce").fillna(10.0).ge(1.0)
    timeout = pd.to_numeric(train_m["is_timeout"], errors="coerce").fillna(1.0).gt(0.5)
    clean = u.gt(0.0) & (~bad) & (~timeout)
    dirty = u.gt(0.0) & (bad | timeout)
    train_side = np.where(
        pd.to_numeric(train_m["side"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32) < 0.0,
        -1,
        1,
    )
    valid_side = np.where(
        pd.to_numeric(valid_m["side"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32) < 0.0,
        -1,
        1,
    )
    train_spread, valid_spread = _bucket_codes_from_train(
        train["median_spread_bps"],
        valid["median_spread_bps"],
        bins=5,
    )
    train_regime = _posterior_codes(train.reset_index(drop=True), train_m)
    valid_regime = _posterior_codes(valid.reset_index(drop=True), valid_m)
    local = pd.DataFrame(
        {
            "u": u.to_numpy(dtype=np.float32, copy=False),
            "bad": bad.to_numpy(dtype=np.float32, copy=False),
            "timeout": timeout.to_numpy(dtype=np.float32, copy=False),
            "clean": clean.to_numpy(dtype=np.float32, copy=False),
            "dirty": dirty.to_numpy(dtype=np.float32, copy=False),
            "side_key": train_side,
            "spread_bucket": train_spread,
            "regime_bucket": train_regime,
        }
    )
    local = local[(local["spread_bucket"] >= 0) & (local["regime_bucket"] >= 0)]
    if local.empty:
        zeros = pd.Series(np.zeros(len(valid), dtype=np.float32), index=index)
        return zeros, zeros, {
            "s46_bucket_quality_status": "no_train_buckets",
            "s46_bucket_quality_min_bucket_rows": int(min_bucket_rows),
        }

    def _agg(group_cols: list[str]) -> pd.DataFrame:
        return local.groupby(group_cols, sort=False).agg(
            rows=("u", "size"),
            mean_u=("u", "mean"),
            bad=("bad", "mean"),
            timeout=("timeout", "mean"),
            clean=("clean", "mean"),
            dirty=("dirty", "mean"),
        )

    tables = {
        "exact": _agg(["side_key", "spread_bucket", "regime_bucket"]),
        "side_regime": _agg(["side_key", "regime_bucket"]),
        "side_spread": _agg(["side_key", "spread_bucket"]),
        "side": _agg(["side_key"]),
    }
    values = np.zeros(len(valid), dtype=np.float32)
    fallback_counts = {"exact": 0, "side_regime": 0, "side_spread": 0, "side": 0, "missing": 0}
    for i in range(len(valid)):
        side_key = int(valid_side[i])
        spread_bucket = int(valid_spread[i])
        regime_bucket = int(valid_regime[i])
        candidates = (
            ("exact", (side_key, spread_bucket, regime_bucket)),
            ("side_regime", (side_key, regime_bucket)),
            ("side_spread", (side_key, spread_bucket)),
            ("side", (side_key,)),
        )
        stats = None
        used = "missing"
        for level, key in candidates:
            table = tables[level]
            if key in table.index and int(table.loc[key, "rows"]) >= int(min_bucket_rows):
                stats = table.loc[key]
                used = level
                break
        fallback_counts[used] += 1
        if stats is None:
            continue
        rows = float(stats["rows"])
        mean_u = float(stats["mean_u"])
        bad_rate = float(stats["bad"])
        timeout_rate = float(stats["timeout"])
        clean_rate = float(stats["clean"])
        dirty_rate = float(stats["dirty"])
        mean_u_score = float(np.clip(mean_u / 0.004, -1.0, 1.0))
        clean_gap = clean_rate - dirty_rate
        row_conf = float(np.clip(np.log1p(rows) / np.log1p(2500.0), 0.0, 1.0))
        values[i] = np.float32(
            0.42 * mean_u_score
            + 0.42 * clean_gap
            - 0.22 * max(bad_rate - 0.50, 0.0)
            - 0.18 * max(timeout_rate - 0.12, 0.0)
            + 0.08 * row_conf
        )
    quality = pd.Series(values, index=index, dtype=np.float32)
    quality_rank = quality.rank(method="average", pct=True).fillna(0.0).astype(np.float32)
    return quality, quality_rank, {
        "s46_bucket_quality_status": "ok",
        "s46_bucket_quality_min_bucket_rows": int(min_bucket_rows),
        "s46_exact_match_rate": float(fallback_counts["exact"] / max(len(valid), 1)),
        "s46_side_regime_match_rate": float(fallback_counts["side_regime"] / max(len(valid), 1)),
        "s46_side_spread_match_rate": float(fallback_counts["side_spread"] / max(len(valid), 1)),
        "s46_side_fallback_rate": float(fallback_counts["side"] / max(len(valid), 1)),
        "s46_missing_rate": float(fallback_counts["missing"] / max(len(valid), 1)),
        "s46_train_regime_coverage": float((train_regime >= 0).mean()) if len(train_regime) else 0.0,
        "s46_valid_regime_coverage": float((valid_regime >= 0).mean()) if len(valid_regime) else 0.0,
    }


def _timestamp_side_rank_percentile(
    frame: pd.DataFrame,
    values: pd.Series,
    side: pd.Series,
) -> pd.Series:
    values_s = pd.to_numeric(values.reset_index(drop=True), errors="coerce")
    ts = pd.to_datetime(frame["__ts__"].reset_index(drop=True), errors="coerce")
    side_s = pd.to_numeric(side.reset_index(drop=True), errors="coerce").fillna(1.0)
    side_key = side_s.where(side_s < 0.0, 1.0).where(side_s >= 0.0, -1.0)
    out = pd.Series(np.nan, index=values_s.index, dtype=np.float32)
    valid = values_s.notna() & ts.notna()
    if not bool(valid.any()):
        return out
    ranked = values_s[valid].groupby([ts[valid], side_key[valid]], sort=False).rank(
        method="average",
        pct=True,
        ascending=True,
    )
    out.loc[valid] = ranked.astype(np.float32)
    return out


def _candidate_ledger_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    score: pd.Series,
    selector: str,
    month: str,
    top_frac: float,
    selected_idx: np.ndarray | None,
    base_fields: dict[str, Any],
    extra_scores: dict[str, pd.Series],
) -> list[dict[str, Any]]:
    frame_local = frame.reset_index(drop=True)
    metrics_local = metrics.reset_index(drop=True)
    score_s = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
    idx = (
        _rank_top_indices(score_s, top_frac)
        if selected_idx is None
        else np.asarray(selected_idx, dtype=np.int64)
    )
    idx = idx[(idx >= 0) & (idx < len(metrics_local))]
    if not len(idx):
        return []
    u = pd.to_numeric(metrics_local["u_policy_net"], errors="coerce").fillna(0.0)
    bad_mae = pd.to_numeric(metrics_local["mae_norm"], errors="coerce").fillna(10.0).ge(1.0)
    timeout = metrics_local["is_timeout"].astype(float).fillna(1.0).gt(0.5)
    clean_positive = u.gt(0.0) & (~bad_mae) & (~timeout)
    dirty_positive = u.gt(0.0) & (bad_mae | timeout)
    oracle_mask = np.zeros(len(metrics_local), dtype=bool)
    oracle_mask[_rank_top_indices(u, top_frac)] = True
    clean_oracle = oracle_mask & clean_positive.to_numpy(dtype=bool)
    side = pd.to_numeric(
        metrics_local.get("side", pd.Series(1.0, index=metrics_local.index)),
        errors="coerce",
    ).fillna(1.0)
    score_rank = score_s.rank(method="average", pct=True).astype(np.float32)
    score_ts_rank = _timestamp_rank_percentile(frame_local, score_s, ascending=True)
    score_ts_side_rank = _timestamp_side_rank_percentile(frame_local, score_s, side)
    order = idx[np.argsort(-score_s.iloc[idx].to_numpy(dtype=np.float64), kind="mergesort")]
    side_arr = side.to_numpy(dtype=np.float32, copy=False)
    score_arr = score_s.to_numpy(dtype=np.float32, copy=False)
    score_rank_arr = score_rank.to_numpy(dtype=np.float32, copy=False)
    score_ts_rank_arr = score_ts_rank.to_numpy(dtype=np.float32, copy=False)
    score_ts_side_rank_arr = score_ts_side_rank.to_numpy(dtype=np.float32, copy=False)
    n_selected = int(len(order))
    def _metric_arr(name: str) -> np.ndarray:
        if name not in metrics_local.columns:
            return np.full(len(metrics_local), np.nan, dtype=np.float32)
        return pd.to_numeric(metrics_local[name], errors="coerce").to_numpy(
            dtype=np.float32,
            copy=False,
        )

    data: dict[str, Any] = {
        **base_fields,
        "period": month,
        "selector_variant": selector,
        "top_frac": float(top_frac),
        "row_pos": order.astype(np.int64, copy=False),
        "selected_rank": np.arange(1, n_selected + 1, dtype=np.int32),
        "selected_count": n_selected,
        "timestamp": frame_local["__ts__"].iloc[order].to_numpy()
        if "__ts__" in frame_local.columns
        else np.full(n_selected, pd.NaT, dtype=object),
        "symbol": frame_local["__symbol__"].iloc[order].to_numpy()
        if "__symbol__" in frame_local.columns
        else np.full(n_selected, "", dtype=object),
        "side": side_arr[order],
        "selector_score": score_arr[order],
        "selector_rank_pct": score_rank_arr[order],
        "selector_ts_rank_pct": score_ts_rank_arr[order],
        "selector_ts_side_rank_pct": score_ts_side_rank_arr[order],
        "u_policy_net": _metric_arr("u_policy_net")[order],
        "ret_net": _metric_arr("ret_net")[order],
        "mae_norm": _metric_arr("mae_norm")[order],
        "mfe_norm": _metric_arr("mfe_norm")[order],
        "barrier": _metric_arr("barrier")[order],
        "is_timeout": _metric_arr("is_timeout")[order],
        "bad_mae_1r": bad_mae.to_numpy(dtype=bool, copy=False)[order],
        "clean_positive": clean_positive.to_numpy(dtype=bool, copy=False)[order],
        "dirty_positive": dirty_positive.to_numpy(dtype=bool, copy=False)[order],
        "oracle_top": oracle_mask[order],
        "clean_oracle_top": clean_oracle[order],
        "oracle_rows_total": int(oracle_mask.sum()),
        "clean_oracle_rows_total": int(clean_oracle.sum()),
    }
    extra_local = {
        name: pd.to_numeric(series.reset_index(drop=True), errors="coerce").to_numpy(
            dtype=np.float32,
            copy=False,
        )
        for name, series in extra_scores.items()
    }
    for name, values in extra_local.items():
        if len(values) >= len(metrics_local):
            data[name] = values[order]
        else:
            filled = np.full(n_selected, np.nan, dtype=np.float32)
            valid_pos = order < len(values)
            if bool(valid_pos.any()):
                filled[valid_pos] = values[order[valid_pos]]
            data[name] = filled
    return pd.DataFrame(data).to_dict("records")


def _strip_side_cap_suffix(selector: str) -> str:
    text = str(selector)
    for suffix in ("_side_cap_70", "_side_cap_60", "_side_cap_75", "_side_cap_80"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _fit_risk_prediction(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    seeds: list[int],
) -> np.ndarray:
    uniform_weight = pd.Series(np.ones(len(y_train), dtype=np.float32), index=y_train.index)
    seed_preds = [
        _fit_predict(
            x_train=x_train,
            y_train=pd.to_numeric(y_train, errors="coerce").fillna(0.0).clip(0.0, 1.0),
            w_train=uniform_weight,
            x_valid=x_valid,
            seed=seed + 10_000,
        )
        for seed in seeds
    ]
    return np.clip(np.mean(np.vstack(seed_preds), axis=0), 0.0, 1.0).astype(np.float32)


def _fit_lgbm_binary_risk_prediction(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    seeds: list[int],
    sample_weight: pd.Series | None = None,
) -> tuple[np.ndarray, str]:
    if not _LIGHTGBM_AVAILABLE or LGBMClassifier is None:
        return np.full(len(x_valid), np.nan, dtype=np.float32), "lightgbm_unavailable"
    y = pd.to_numeric(y_train.reset_index(drop=True), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if int((y > 0.5).sum()) <= 0 or int((y <= 0.5).sum()) <= 0:
        return np.full(len(x_valid), float(y.mean()), dtype=np.float32), "single_class"
    weights = (
        pd.to_numeric(sample_weight.reset_index(drop=True), errors="coerce")
        if sample_weight is not None
        else pd.Series(np.ones(len(y), dtype=np.float32))
    )
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.05, upper=20.0)
    preds: list[np.ndarray] = []
    for seed in seeds:
        model = LGBMClassifier(
            objective="binary",
            n_estimators=96,
            learning_rate=0.045,
            num_leaves=31,
            max_depth=6,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_alpha=0.05,
            reg_lambda=1.50,
            random_state=int(seed),
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(
            x_train,
            y.to_numpy(dtype=np.float32),
            sample_weight=weights.to_numpy(dtype=np.float32),
        )
        pred = model.predict_proba(x_valid)[:, 1].astype(np.float32)
        preds.append(np.clip(pred, 0.0, 1.0))
    return np.clip(np.mean(np.vstack(preds), axis=0), 0.0, 1.0).astype(np.float32), "ok"


def _fit_lgbm_conditional_binary_prediction(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    train_mask: pd.Series,
    x_valid: pd.DataFrame,
    seeds: list[int],
    sample_weight: pd.Series | None = None,
    min_train_rows: int = 500,
) -> tuple[np.ndarray, str]:
    mask = train_mask.reset_index(drop=True).fillna(False).astype(bool)
    mask_arr = mask.to_numpy(dtype=bool)
    if int(mask.sum()) < int(min_train_rows):
        return np.full(len(x_valid), np.nan, dtype=np.float32), "insufficient_conditional_rows"
    conditional_weight = (
        sample_weight.reset_index(drop=True).iloc[mask_arr]
        if sample_weight is not None
        else None
    )
    pred, status = _fit_lgbm_binary_risk_prediction(
        x_train=x_train.reset_index(drop=True).iloc[mask_arr].reset_index(drop=True),
        y_train=y_train.reset_index(drop=True).iloc[mask_arr],
        x_valid=x_valid,
        seeds=seeds,
        sample_weight=conditional_weight.reset_index(drop=True)
        if conditional_weight is not None
        else None,
    )
    return pred, f"conditional_{status}"


def _fit_side_lgbm_conditional_binary_prediction(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    train_mask: pd.Series,
    train_side: pd.Series,
    x_valid: pd.DataFrame,
    valid_side: pd.Series,
    seeds: list[int],
    sample_weight: pd.Series | None = None,
    min_train_rows: int = 500,
    min_side_train_rows: int = 300,
) -> tuple[np.ndarray, str]:
    """Fit conditional binary heads separately by side with a global fallback."""
    global_pred, global_status = _fit_lgbm_conditional_binary_prediction(
        x_train=x_train,
        y_train=y_train,
        train_mask=train_mask,
        x_valid=x_valid,
        seeds=[seed + 210_000 for seed in seeds],
        sample_weight=sample_weight,
        min_train_rows=min_train_rows,
    )
    out = global_pred.copy()
    train_side_s = pd.to_numeric(train_side.reset_index(drop=True), errors="coerce").fillna(1.0)
    valid_side_s = pd.to_numeric(valid_side.reset_index(drop=True), errors="coerce").fillna(1.0)
    base_mask = train_mask.reset_index(drop=True).fillna(False).astype(bool)
    statuses = [f"global:{global_status}"]
    for side_name, side_value, seed_offset in (
        ("short", -1.0, 220_000),
        ("long", 1.0, 230_000),
    ):
        if side_value < 0.0:
            side_train_mask = base_mask & train_side_s.lt(0.0)
            side_valid_mask = valid_side_s.lt(0.0).to_numpy(dtype=bool)
        else:
            side_train_mask = base_mask & train_side_s.ge(0.0)
            side_valid_mask = valid_side_s.ge(0.0).to_numpy(dtype=bool)
        if int(side_train_mask.sum()) < int(min_side_train_rows) or not bool(side_valid_mask.any()):
            statuses.append(f"{side_name}:insufficient_rows")
            continue
        pred, status = _fit_lgbm_conditional_binary_prediction(
            x_train=x_train,
            y_train=y_train,
            train_mask=side_train_mask,
            x_valid=x_valid.reset_index(drop=True).iloc[side_valid_mask].reset_index(drop=True),
            seeds=[seed + seed_offset for seed in seeds],
            sample_weight=sample_weight,
            min_train_rows=min_side_train_rows,
        )
        statuses.append(f"{side_name}:{status}")
        if status.endswith("_ok"):
            out[side_valid_mask] = pred
    return np.clip(out, 0.0, 1.0).astype(np.float32), ";".join(statuses)


def _path_risk_sample_weight(train_metrics: pd.DataFrame) -> pd.Series:
    metrics = train_metrics.reset_index(drop=True)
    u = pd.to_numeric(metrics["u_policy_net"], errors="coerce").fillna(0.0)
    positive_rank = u.clip(lower=0.0).rank(method="average", pct=True).fillna(0.0)
    mae = pd.to_numeric(metrics["mae_norm"], errors="coerce").fillna(0.0)
    timeout = metrics["is_timeout"].astype(float).fillna(0.0)
    weight = (
        1.0
        + 1.50 * u.gt(0.0).astype(float)
        + 1.50 * positive_rank
        + 0.75 * mae.ge(1.0).astype(float)
        + 0.50 * timeout.gt(0.0).astype(float)
    )
    return weight.astype(np.float32)


def _fit_side_risk_prediction(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    train_side: pd.Series,
    x_valid: pd.DataFrame,
    valid_side: pd.Series,
    seeds: list[int],
    min_side_train_rows: int = 500,
) -> np.ndarray:
    global_pred = _fit_risk_prediction(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        seeds=[seed + 20_000 for seed in seeds],
    )
    out = global_pred.copy()
    train_side_s = pd.to_numeric(train_side.reset_index(drop=True), errors="coerce").fillna(1.0)
    valid_side_s = pd.to_numeric(valid_side.reset_index(drop=True), errors="coerce").fillna(1.0)
    y_train_s = pd.to_numeric(y_train.reset_index(drop=True), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    for side_value in (-1, 1):
        train_mask = (train_side_s < 0.0).to_numpy() if side_value < 0 else (train_side_s >= 0.0).to_numpy()
        valid_mask = (valid_side_s < 0.0).to_numpy() if side_value < 0 else (valid_side_s >= 0.0).to_numpy()
        if int(train_mask.sum()) < int(min_side_train_rows) or not bool(valid_mask.any()):
            continue
        out[valid_mask] = _fit_risk_prediction(
            x_train=x_train.iloc[train_mask],
            y_train=y_train_s.iloc[train_mask],
            x_valid=x_valid.iloc[valid_mask],
            seeds=[seed + (30_000 if side_value < 0 else 40_000) for seed in seeds],
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _fit_feature_gap_risk_score(
    *,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    train_metrics: pd.DataFrame,
    top_k: int = 12,
) -> tuple[pd.Series, dict[str, Any]]:
    """Build an interpretable train-only feature risk score.

    This is deliberately simple: for each month-forward fold, it finds features
    whose train-period values separate clean positive rows from risky rows, then
    averages valid-period percentile ranks in the risky direction. No valid
    labels are used to choose features or thresholds.
    """
    train_x = x_train.reset_index(drop=True)
    valid_x = x_valid.reset_index(drop=True)
    metrics = train_metrics.reset_index(drop=True)
    clean = (
        (pd.to_numeric(metrics["u_policy_net"], errors="coerce") > 0.0)
        & (pd.to_numeric(metrics["mae_norm"], errors="coerce") < 1.0)
        & (metrics["is_timeout"].astype(float) <= 0.0)
    )
    risky = ~clean
    bad_mae = pd.to_numeric(metrics["mae_norm"], errors="coerce").ge(1.0)
    timeout = metrics["is_timeout"].astype(float).gt(0.0)
    rows: list[dict[str, Any]] = []
    for feature in train_x.columns:
        values = pd.to_numeric(train_x[feature], errors="coerce")
        clean_values = values[clean]
        risky_values = values[risky]
        if int(clean_values.notna().sum()) < 100 or int(risky_values.notna().sum()) < 100:
            continue
        clean_median = _finite_quantile_np(clean_values, 0.50)
        risky_median = _finite_quantile_np(risky_values, 0.50)
        pooled = pd.concat([clean_values, risky_values], ignore_index=True)
        iqr = _finite_quantile_np(pooled, 0.75) - _finite_quantile_np(pooled, 0.25)
        if not math.isfinite(iqr) or abs(iqr) <= 1.0e-12:
            continue
        robust_delta = (risky_median - clean_median) / iqr
        clean_ic = _spearman(values, clean.astype(float))
        bad_ic = _spearman(values, bad_mae.astype(float))
        timeout_ic = _spearman(values, timeout.astype(float))
        if not math.isfinite(robust_delta):
            continue
        score = (
            abs(robust_delta)
            + 2.0 * abs(clean_ic if math.isfinite(clean_ic) else 0.0)
            + abs(bad_ic if math.isfinite(bad_ic) else 0.0)
            + abs(timeout_ic if math.isfinite(timeout_ic) else 0.0)
        )
        rows.append(
            {
                "feature": feature,
                "risky_minus_clean_robust_delta": robust_delta,
                "clean_ic": clean_ic,
                "bad_mae_ic": bad_ic,
                "timeout_ic": timeout_ic,
                "score": score,
                "risk_direction": 1.0 if robust_delta >= 0.0 else -1.0,
            }
        )
    feature_table = pd.DataFrame(rows)
    if feature_table.empty:
        return pd.Series(np.zeros(len(valid_x), dtype=np.float32)), {
            "feature_gap_risk_feature_count": 0,
            "feature_gap_risk_features": "",
        }
    feature_table = feature_table.sort_values("score", ascending=False).head(int(top_k)).copy()
    parts: list[pd.Series] = []
    for _, row in feature_table.iterrows():
        feature = str(row["feature"])
        ranks = _rank_pct_np(valid_x[feature])
        if float(row["risk_direction"]) < 0.0:
            ranks = 1.0 - ranks
        parts.append(ranks.fillna(0.5))
    risk = pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(0.0, index=valid_x.index)
    return risk.astype(np.float32), {
        "feature_gap_risk_feature_count": int(len(feature_table)),
        "feature_gap_risk_features": ",".join(feature_table["feature"].astype(str).tolist()),
        "feature_gap_risk_top_score": float(feature_table["score"].iloc[0]),
        "feature_gap_risk_mean_score": float(feature_table["score"].mean()),
    }


def _fit_clean_dirty_positive_risk_score(
    *,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    train_metrics: pd.DataFrame,
    top_k: int = 12,
    min_class_rows: int = 100,
) -> tuple[pd.Series, dict[str, Any]]:
    """Train-only score for dirty profitable paths.

    The current blocked variants often select rows with positive utility but
    bad path quality. This contrast avoids generic losers and only compares
    positive-utility clean rows against positive-utility dirty rows in the
    training window.
    """
    train_x = x_train.reset_index(drop=True)
    valid_x = x_valid.reset_index(drop=True)
    metrics = train_metrics.reset_index(drop=True)
    positive = pd.to_numeric(metrics["u_policy_net"], errors="coerce").gt(0.0)
    bad_mae = pd.to_numeric(metrics["mae_norm"], errors="coerce").ge(1.0)
    timeout = metrics["is_timeout"].astype(float).gt(0.0)
    clean = positive & (~bad_mae) & (~timeout)
    dirty = positive & (bad_mae | timeout)
    clean_rows = int(clean.sum())
    dirty_rows = int(dirty.sum())
    if clean_rows < int(min_class_rows) or dirty_rows < int(min_class_rows):
        return pd.Series(np.zeros(len(valid_x), dtype=np.float32)), {
            "clean_dirty_positive_risk_feature_count": 0,
            "clean_dirty_positive_risk_features": "",
            "clean_dirty_positive_train_clean_rows": clean_rows,
            "clean_dirty_positive_train_dirty_rows": dirty_rows,
        }

    rows: list[dict[str, Any]] = []
    for feature in train_x.columns:
        values = pd.to_numeric(train_x[feature], errors="coerce")
        clean_values = values[clean]
        dirty_values = values[dirty]
        if int(clean_values.notna().sum()) < int(min_class_rows) or int(dirty_values.notna().sum()) < int(min_class_rows):
            continue
        clean_median = _finite_quantile_np(clean_values, 0.50)
        dirty_median = _finite_quantile_np(dirty_values, 0.50)
        pooled = pd.concat([clean_values, dirty_values], ignore_index=True)
        iqr = _finite_quantile_np(pooled, 0.75) - _finite_quantile_np(pooled, 0.25)
        if not math.isfinite(iqr) or abs(iqr) <= 1.0e-12:
            continue
        robust_delta = (dirty_median - clean_median) / iqr
        if not math.isfinite(robust_delta):
            continue
        clean_ic = _spearman(values[positive], clean[positive].astype(float))
        dirty_ic = _spearman(values[positive], dirty[positive].astype(float))
        bad_ic = _spearman(values, bad_mae.astype(float))
        timeout_ic = _spearman(values, timeout.astype(float))
        score = (
            abs(robust_delta)
            + 2.0 * abs(clean_ic if math.isfinite(clean_ic) else 0.0)
            + abs(dirty_ic if math.isfinite(dirty_ic) else 0.0)
            + 0.50 * abs(bad_ic if math.isfinite(bad_ic) else 0.0)
            + 0.50 * abs(timeout_ic if math.isfinite(timeout_ic) else 0.0)
        )
        rows.append(
            {
                "feature": feature,
                "dirty_minus_clean_positive_robust_delta": robust_delta,
                "clean_positive_ic": clean_ic,
                "dirty_positive_ic": dirty_ic,
                "bad_mae_ic": bad_ic,
                "timeout_ic": timeout_ic,
                "score": score,
                "risk_direction": 1.0 if robust_delta >= 0.0 else -1.0,
            }
        )
    feature_table = pd.DataFrame(rows)
    if feature_table.empty:
        return pd.Series(np.zeros(len(valid_x), dtype=np.float32)), {
            "clean_dirty_positive_risk_feature_count": 0,
            "clean_dirty_positive_risk_features": "",
            "clean_dirty_positive_train_clean_rows": clean_rows,
            "clean_dirty_positive_train_dirty_rows": dirty_rows,
        }
    feature_table = feature_table.sort_values("score", ascending=False).head(int(top_k)).copy()
    parts: list[pd.Series] = []
    for _, row in feature_table.iterrows():
        feature = str(row["feature"])
        ranks = _rank_pct_np(valid_x[feature])
        if float(row["risk_direction"]) < 0.0:
            ranks = 1.0 - ranks
        parts.append(ranks.fillna(0.5))
    risk = pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(0.0, index=valid_x.index)
    return risk.astype(np.float32), {
        "clean_dirty_positive_risk_feature_count": int(len(feature_table)),
        "clean_dirty_positive_risk_features": ",".join(feature_table["feature"].astype(str).tolist()),
        "clean_dirty_positive_risk_top_score": float(feature_table["score"].iloc[0]),
        "clean_dirty_positive_risk_mean_score": float(feature_table["score"].mean()),
        "clean_dirty_positive_train_clean_rows": clean_rows,
        "clean_dirty_positive_train_dirty_rows": dirty_rows,
    }


def _run_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    features: list[str],
    month: str,
    label_arms: list[str],
    weight_arms: list[str],
    seeds: list[int],
    model_feature_selector: str,
    model_feature_tail_frac: float,
    top_fracs: list[float],
    train_lookback_months: int | None,
    include_risk_selector_variants: bool,
    side_cap_max_share: float,
    candidate_ledger_selector_names: set[str],
    candidate_ledger_only: bool,
    candidate_ledger_fast_mode: bool,
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_gmm_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < month
    if train_lookback_months is not None and int(train_lookback_months) > 0:
        prior_months = sorted(month_period[train_mask].dropna().unique())
        keep_months = set(prior_months[-int(train_lookback_months) :])
        train_mask = train_mask & month_period.isin(keep_months)
    valid_mask = month_period == month
    if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
        return [], [
            {
                "period": month,
                "skipped": True,
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
            }
        ], [], []

    x_train, x_valid = _month_model_frame(
        frame,
        train_mask=train_mask,
        valid_mask=valid_mask,
        features=features,
    )
    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    train_metrics = metrics.loc[train_mask].copy()
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    x_train, x_valid, ae_gmm_state_features, ae_gmm_state_diag = _append_fold_ae_gmm_state_features(
        x_train=x_train,
        x_valid=x_valid,
        train_frame=train,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        enabled=bool(include_ae_gmm_state_features),
        max_train_rows=int(ae_gmm_state_feature_max_train_rows),
        gmm_max_train_rows=int(ae_gmm_state_feature_gmm_max_train_rows),
        ae_max_iter=int(ae_gmm_state_feature_max_iter),
        random_state=90221 + sum((i + 1) * ord(ch) for i, ch in enumerate(str(month))),
    )
    fold_features = list(dict.fromkeys(list(features) + list(ae_gmm_state_features)))
    valid_context = pd.concat(
        [
            valid.reset_index(drop=True),
            x_valid.reindex(columns=ae_gmm_state_features).reset_index(drop=True),
        ],
        axis=1,
        copy=False,
    )
    baseline = _baseline_row(valid_metrics)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    clean_dirty_diagnostics: list[dict[str, Any]] = []
    candidate_ledger: list[dict[str, Any]] = []
    requested_ledger_selectors = set(candidate_ledger_selector_names)
    requested_ledger_base_selectors = {
        _strip_side_cap_suffix(name) for name in requested_ledger_selectors
    }
    fast_ledger_mode = bool(candidate_ledger_fast_mode and requested_ledger_selectors)
    needs_oracle_rankers = (not fast_ledger_mode) or any(
        ("oracle" in name or name.startswith("s10_lgbm_three_ranker"))
        for name in requested_ledger_base_selectors
    )
    needs_feature_gap_score = (not fast_ledger_mode) or any(
        "feature_gap" in name for name in requested_ledger_base_selectors
    )
    needs_clean_dirty_score = (not fast_ledger_mode) or any(
        (
            "s14_" in name
            or "s20_" in name
            or "s22_" in name
            or "s50_" in name
            or "s42_" in name
            or "s43_" in name
            or "s44_" in name
            or "s45_" in name
            or "s46_" in name
            or "s47_" in name
            or "exec_clean" in name
            or "clean_dirty_penalty" in name
            or "clean_dirty_contrast" in name
        )
        for name in requested_ledger_base_selectors
    )

    for label_arm in label_arms:
        target_train = targets[label_arm].loc[train_mask].copy()
        target_valid = targets[label_arm].loc[valid_mask].copy().reset_index(drop=True)
        for weight_arm in weight_arms:
            weights = _weight_series(
                frame=train,
                metrics=train_metrics,
                target=target_train,
                arm=weight_arm,
            )
            model_features = list(fold_features)
            feature_selector_diag: dict[str, Any] = {
                "model_feature_selector": model_feature_selector,
                "model_feature_selector_fallback": "",
                "model_features": ",".join(model_features),
                **ae_gmm_state_diag,
            }
            if model_feature_selector != "all":
                _score_unused, selector_diag = _weighted_proxy_score(
                    train,
                    frame.loc[valid_mask].copy(),
                    features,
                    target_train["target_soft"],
                    weights,
                    method=model_feature_selector,
                    tail_frac=model_feature_tail_frac,
                )
                selected = [
                    feature
                    for feature in selector_diag.get("proxy_features", [])
                    if feature in x_train.columns
                ]
                if selected:
                    model_features = list(dict.fromkeys(selected + list(ae_gmm_state_features)))
                    feature_selector_diag.update(
                        {
                            "model_features": ",".join(model_features),
                            "model_feature_proxy_top_abs_ic": selector_diag.get("proxy_top_abs_ic"),
                            "model_feature_proxy_mean_top_abs_ic": selector_diag.get(
                                "proxy_mean_top_abs_ic"
                            ),
                            "model_feature_proxy_mean_tail_gain_abs": selector_diag.get(
                                "proxy_mean_tail_gain_abs"
                            ),
                            "model_feature_stability_fallback": selector_diag.get(
                                "stability_fallback", ""
                            ),
                        }
                    )
                else:
                    feature_selector_diag["model_feature_selector_fallback"] = "empty_selected_features"

            x_train_model = x_train[model_features]
            x_valid_model = x_valid[model_features]
            seed_preds = [
                _fit_predict(
                    x_train=x_train_model,
                    y_train=target_train["target_soft"],
                    w_train=weights,
                    x_valid=x_valid_model,
                    seed=seed,
                )
                for seed in seeds
            ]
            pred_matrix = np.vstack(seed_preds)
            pred = np.mean(pred_matrix, axis=0).astype(np.float32)
            pred_seed_std = (
                np.std(pred_matrix, axis=0).astype(np.float32)
                if len(seed_preds) > 1
                else np.zeros_like(pred, dtype=np.float32)
            )
            score = pd.Series(pred, index=valid.index)
            bad_mae_pred = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            timeout_pred = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            side_bad_mae_pred = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            side_timeout_pred = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            clean_path_pred = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            feature_gap_risk = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            feature_gap_diag: dict[str, Any] = {}
            clean_dirty_positive_risk = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            clean_dirty_positive_diag: dict[str, Any] = {}
            lgbm_bad_mae_pred = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            lgbm_timeout_pred = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            lgbm_clean_path_pred = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            lgbm_dirty_positive_bad_mae_pred = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_positive_clean_path_pred = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_side_dirty_positive_bad_mae_pred = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_side_positive_clean_path_pred = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_bad_mae_ts_pct = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            lgbm_timeout_ts_pct = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            lgbm_clean_path_ts_pct = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            lgbm_dirty_positive_bad_mae_ts_pct = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_positive_clean_path_ts_pct = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_side_dirty_positive_bad_mae_ts_pct = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_side_positive_clean_path_ts_pct = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_bad_mae_status = "not_run"
            lgbm_timeout_status = "not_run"
            lgbm_clean_path_status = "not_run"
            lgbm_dirty_positive_bad_mae_status = "not_run"
            lgbm_positive_clean_path_status = "not_run"
            lgbm_side_dirty_positive_bad_mae_status = "not_run"
            lgbm_side_positive_clean_path_status = "not_run"
            lgbm_ranker_score = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            lgbm_path_ranker_score = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            lgbm_oracle_ranker_score = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            lgbm_clean_oracle_ranker_score = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            lgbm_path_first_ranker_score = pd.Series(np.nan, index=valid.index, dtype=np.float32)
            lgbm_path_first_dirty_zero_ranker_score = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_s24_broad_path_first_ranker_score = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_s24_broad_path_first_dirty_zero_ranker_score = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_s28_side_s24_ranker_score = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_s28_side_s24_dirty_zero_ranker_score = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_s30_side_asym_ranker_score = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_s30_side_asym_dirty_zero_ranker_score = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_s42_side_interaction_dirty_zero_ranker_score = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_s44_side_interaction_sign_calibrated_ranker_score = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_s45_side_interaction_roll45_dirty_zero_ranker_score = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_timeout_aware_clean_ranker_score = pd.Series(
                np.nan,
                index=valid.index,
                dtype=np.float32,
            )
            lgbm_ranker_status = "not_run"
            lgbm_path_ranker_status = "not_run"
            lgbm_oracle_ranker_status = "not_run"
            lgbm_clean_oracle_ranker_status = "not_run"
            lgbm_path_first_ranker_status = "not_run"
            lgbm_path_first_dirty_zero_ranker_status = "not_run"
            lgbm_s24_broad_path_first_ranker_status = "not_run"
            lgbm_s24_broad_path_first_dirty_zero_ranker_status = "not_run"
            lgbm_s28_side_s24_ranker_status = "not_run"
            lgbm_s28_side_s24_dirty_zero_ranker_status = "not_run"
            lgbm_s30_side_asym_ranker_status = "not_run"
            lgbm_s30_side_asym_dirty_zero_ranker_status = "not_run"
            lgbm_s42_side_interaction_dirty_zero_ranker_status = "not_run"
            s42_interaction_diag: dict[str, Any] = {
                "s42_interaction_feature_count": 0,
                "s42_interaction_features": "",
                "s42_interaction_features_truncated": 0,
            }
            s44_sign_calibration_diag: dict[str, Any] = {
                "s44_long_train_relevance_ic": float("nan"),
                "s44_short_train_relevance_ic": float("nan"),
                "s44_long_score_sign": float("nan"),
                "s44_short_score_sign": float("nan"),
            }
            lgbm_s45_side_interaction_roll45_dirty_zero_ranker_status = "not_run"
            s45_recent_train_diag: dict[str, Any] = {
                "s45_recent_train_window_status": "not_run",
                "s45_recent_train_window_days": 45,
                "s45_recent_train_rows": 0,
                "s45_recent_train_full_rows": int(len(train)),
            }
            lgbm_timeout_aware_clean_ranker_status = "not_run"
            s22_bucket_quality_score = pd.Series(
                np.zeros(len(valid), dtype=np.float32),
                index=valid.index,
            )
            s22_bucket_quality_rank_pct = pd.Series(
                np.zeros(len(valid), dtype=np.float32),
                index=valid.index,
            )
            s22_bucket_relaxed_pass_count = pd.Series(
                np.zeros(len(valid), dtype=np.float32),
                index=valid.index,
            )
            s22_bucket_strict_pass_count = pd.Series(
                np.zeros(len(valid), dtype=np.float32),
                index=valid.index,
            )
            s22_bucket_quality_diag: dict[str, Any] = {
                "s22_bucket_quality_status": "not_run",
            }
            s46_bucket_quality_score = pd.Series(
                np.zeros(len(valid), dtype=np.float32),
                index=valid.index,
            )
            s46_bucket_quality_rank_pct = pd.Series(
                np.zeros(len(valid), dtype=np.float32),
                index=valid.index,
            )
            s46_bucket_quality_diag: dict[str, Any] = {
                "s46_bucket_quality_status": "not_run",
            }
            include_path_first_selectors = (
                str(label_arm)
                in {
                    "OPTIMIZED_ECONOMIC_PATH_FIRST_CLEAN_RELEVANCE_TARGET",
                    "OPTIMIZED_ECONOMIC_S24_BROAD_PATH_FIRST_SOURCE_TARGET",
                }
            )
            include_s24_path_first_selectors = (
                str(label_arm) == "OPTIMIZED_ECONOMIC_S24_BROAD_PATH_FIRST_SOURCE_TARGET"
            )
            include_timeout_aware_selectors = (
                str(label_arm) == "OPTIMIZED_ECONOMIC_TIMEOUT_AWARE_CLEAN_SOURCE_TARGET"
            )
            include_s42_source_selectors = (not fast_ledger_mode) or any(
                (
                    "s42_" in name
                    or "s43_" in name
                    or "s44_" in name
                    or "s45_" in name
                    or "s46_" in name
                )
                for name in requested_ledger_base_selectors
            )
            if include_risk_selector_variants:
                bad_mae_train_target = (train_metrics["mae_norm"].reset_index(drop=True) >= 1.0).astype(float)
                timeout_train_target = train_metrics["is_timeout"].reset_index(drop=True).astype(float)
                clean_path_train_target = (
                    (pd.to_numeric(train_metrics["u_policy_net"], errors="coerce").reset_index(drop=True) > 0.0)
                    & (pd.to_numeric(train_metrics["mae_norm"], errors="coerce").reset_index(drop=True) < 1.0)
                    & (
                        train_metrics["is_timeout"]
                        .reset_index(drop=True)
                        .astype(float)
                        .le(0.0)
                    )
                ).astype(float)
                bad_mae_pred = pd.Series(
                    _fit_risk_prediction(
                        x_train=x_train_model,
                        y_train=bad_mae_train_target,
                        x_valid=x_valid_model,
                        seeds=seeds,
                    ),
                    index=valid.index,
                )
                timeout_pred = pd.Series(
                    _fit_risk_prediction(
                        x_train=x_train_model,
                        y_train=timeout_train_target,
                        x_valid=x_valid_model,
                        seeds=seeds,
                    ),
                    index=valid.index,
                )
                side_bad_mae_pred = pd.Series(
                    _fit_side_risk_prediction(
                        x_train=x_train_model,
                        y_train=bad_mae_train_target,
                        train_side=train_metrics["side"],
                        x_valid=x_valid_model,
                        valid_side=valid_metrics["side"],
                        seeds=seeds,
                    ),
                    index=valid.index,
                )
                side_timeout_pred = pd.Series(
                    _fit_side_risk_prediction(
                        x_train=x_train_model,
                        y_train=timeout_train_target,
                        train_side=train_metrics["side"],
                        x_valid=x_valid_model,
                        valid_side=valid_metrics["side"],
                        seeds=seeds,
                    ),
                    index=valid.index,
                )
                clean_path_pred = pd.Series(
                    _fit_risk_prediction(
                        x_train=x_train_model,
                        y_train=clean_path_train_target,
                        x_valid=x_valid_model,
                        seeds=[seed + 50_000 for seed in seeds],
                    ),
                    index=valid.index,
                )
                risk_head_weight = _path_risk_sample_weight(train_metrics.reset_index(drop=True))
                lgbm_bad_mae_values, lgbm_bad_mae_status = _fit_lgbm_binary_risk_prediction(
                    x_train=x_train_model,
                    y_train=bad_mae_train_target,
                    x_valid=x_valid_model,
                    seeds=[seed + 90_000 for seed in seeds],
                    sample_weight=risk_head_weight,
                )
                lgbm_bad_mae_pred = pd.Series(lgbm_bad_mae_values, index=valid.index)
                lgbm_timeout_values, lgbm_timeout_status = _fit_lgbm_binary_risk_prediction(
                    x_train=x_train_model,
                    y_train=timeout_train_target,
                    x_valid=x_valid_model,
                    seeds=[seed + 100_000 for seed in seeds],
                    sample_weight=risk_head_weight,
                )
                lgbm_timeout_pred = pd.Series(lgbm_timeout_values, index=valid.index)
                clean_path_weight = risk_head_weight + 2.0 * clean_path_train_target.astype(np.float32)
                lgbm_clean_path_values, lgbm_clean_path_status = _fit_lgbm_binary_risk_prediction(
                    x_train=x_train_model,
                    y_train=clean_path_train_target,
                    x_valid=x_valid_model,
                    seeds=[seed + 110_000 for seed in seeds],
                    sample_weight=clean_path_weight,
                )
                lgbm_clean_path_pred = pd.Series(lgbm_clean_path_values, index=valid.index)
                positive_train_mask = (
                    pd.to_numeric(
                        train_metrics["u_policy_net"].reset_index(drop=True),
                        errors="coerce",
                    )
                    > 0.0
                )
                (
                    lgbm_dirty_positive_bad_mae_values,
                    lgbm_dirty_positive_bad_mae_status,
                ) = _fit_lgbm_conditional_binary_prediction(
                    x_train=x_train_model,
                    y_train=bad_mae_train_target,
                    train_mask=positive_train_mask,
                    x_valid=x_valid_model,
                    seeds=[seed + 120_000 for seed in seeds],
                    sample_weight=risk_head_weight,
                    min_train_rows=500,
                )
                lgbm_dirty_positive_bad_mae_pred = pd.Series(
                    lgbm_dirty_positive_bad_mae_values,
                    index=valid.index,
                )
                (
                    lgbm_positive_clean_path_values,
                    lgbm_positive_clean_path_status,
                ) = _fit_lgbm_conditional_binary_prediction(
                    x_train=x_train_model,
                    y_train=clean_path_train_target,
                    train_mask=positive_train_mask,
                    x_valid=x_valid_model,
                    seeds=[seed + 125_000 for seed in seeds],
                    sample_weight=clean_path_weight,
                    min_train_rows=500,
                )
                lgbm_positive_clean_path_pred = pd.Series(
                    lgbm_positive_clean_path_values,
                    index=valid.index,
                )
                (
                    lgbm_side_dirty_positive_bad_mae_values,
                    lgbm_side_dirty_positive_bad_mae_status,
                ) = _fit_side_lgbm_conditional_binary_prediction(
                    x_train=x_train_model,
                    y_train=bad_mae_train_target,
                    train_mask=positive_train_mask,
                    train_side=train_metrics["side"],
                    x_valid=x_valid_model,
                    valid_side=valid_metrics["side"],
                    seeds=[seed + 126_000 for seed in seeds],
                    sample_weight=risk_head_weight,
                    min_train_rows=500,
                    min_side_train_rows=300,
                )
                lgbm_side_dirty_positive_bad_mae_pred = pd.Series(
                    lgbm_side_dirty_positive_bad_mae_values,
                    index=valid.index,
                )
                (
                    lgbm_side_positive_clean_path_values,
                    lgbm_side_positive_clean_path_status,
                ) = _fit_side_lgbm_conditional_binary_prediction(
                    x_train=x_train_model,
                    y_train=clean_path_train_target,
                    train_mask=positive_train_mask,
                    train_side=train_metrics["side"],
                    x_valid=x_valid_model,
                    valid_side=valid_metrics["side"],
                    seeds=[seed + 127_000 for seed in seeds],
                    sample_weight=clean_path_weight,
                    min_train_rows=500,
                    min_side_train_rows=300,
                )
                lgbm_side_positive_clean_path_pred = pd.Series(
                    lgbm_side_positive_clean_path_values,
                    index=valid.index,
                )
                if needs_feature_gap_score:
                    feature_gap_values, feature_gap_diag = _fit_feature_gap_risk_score(
                        x_train=x_train_model,
                        x_valid=x_valid_model,
                        train_metrics=train_metrics.reset_index(drop=True),
                        top_k=12,
                    )
                    feature_gap_risk = pd.Series(
                        feature_gap_values.to_numpy(dtype=np.float32),
                        index=valid.index,
                    )
                else:
                    feature_gap_risk = pd.Series(
                        np.zeros(len(valid), dtype=np.float32),
                        index=valid.index,
                    )
                    feature_gap_diag = {
                        "feature_gap_risk_feature_count": 0,
                        "feature_gap_risk_features": "",
                        "feature_gap_risk_skipped": "candidate_ledger_fast_mode",
                    }
                if needs_clean_dirty_score:
                    clean_dirty_values, clean_dirty_positive_diag = _fit_clean_dirty_positive_risk_score(
                        x_train=x_train_model,
                        x_valid=x_valid_model,
                        train_metrics=train_metrics.reset_index(drop=True),
                        top_k=12,
                    )
                    clean_dirty_positive_risk = pd.Series(
                        clean_dirty_values.to_numpy(dtype=np.float32),
                        index=valid.index,
                    )
                else:
                    clean_dirty_positive_risk = pd.Series(
                        np.zeros(len(valid), dtype=np.float32),
                        index=valid.index,
                    )
                    clean_dirty_positive_diag = {
                        "clean_dirty_positive_risk_feature_count": 0,
                        "clean_dirty_positive_risk_features": "",
                        "clean_dirty_positive_risk_skipped": "candidate_ledger_fast_mode",
                    }
                _train_ranker_pred, valid_ranker_pred, lgbm_ranker_status = _fit_lgbm_ranker_prediction(
                    x_train=x_train_model,
                    train_frame=train.reset_index(drop=True),
                    train_metrics=train_metrics.reset_index(drop=True),
                    target_train=target_train.reset_index(drop=True),
                    w_train=weights,
                    x_valid=x_valid_model,
                    seeds=seeds,
                    relevance_mode="utility_quintile",
                )
                lgbm_ranker_score = pd.Series(valid_ranker_pred, index=valid.index)
                (
                    _train_path_ranker_pred,
                    valid_path_ranker_pred,
                    lgbm_path_ranker_status,
                ) = _fit_lgbm_ranker_prediction(
                    x_train=x_train_model,
                    train_frame=train.reset_index(drop=True),
                    train_metrics=train_metrics.reset_index(drop=True),
                    target_train=target_train.reset_index(drop=True),
                    w_train=weights,
                    x_valid=x_valid_model,
                    seeds=[seed + 60_000 for seed in seeds],
                    relevance_mode="path_quality",
                )
                lgbm_path_ranker_score = pd.Series(valid_path_ranker_pred, index=valid.index)
                if needs_oracle_rankers:
                    (
                        _train_oracle_ranker_pred,
                        valid_oracle_ranker_pred,
                        lgbm_oracle_ranker_status,
                    ) = _fit_lgbm_ranker_prediction(
                        x_train=x_train_model,
                        train_frame=train.reset_index(drop=True),
                        train_metrics=train_metrics.reset_index(drop=True),
                        target_train=target_train.reset_index(drop=True),
                        w_train=weights,
                        x_valid=x_valid_model,
                        seeds=[seed + 70_000 for seed in seeds],
                        relevance_mode="oracle_enriched",
                    )
                    lgbm_oracle_ranker_score = pd.Series(
                        valid_oracle_ranker_pred,
                        index=valid.index,
                    )
                    (
                        _train_clean_oracle_ranker_pred,
                        valid_clean_oracle_ranker_pred,
                        lgbm_clean_oracle_ranker_status,
                    ) = _fit_lgbm_ranker_prediction(
                        x_train=x_train_model,
                        train_frame=train.reset_index(drop=True),
                        train_metrics=train_metrics.reset_index(drop=True),
                        target_train=target_train.reset_index(drop=True),
                        w_train=weights,
                        x_valid=x_valid_model,
                        seeds=[seed + 80_000 for seed in seeds],
                        relevance_mode="clean_oracle",
                    )
                    lgbm_clean_oracle_ranker_score = pd.Series(
                        valid_clean_oracle_ranker_pred,
                        index=valid.index,
                    )
                else:
                    lgbm_oracle_ranker_status = "skipped_candidate_ledger_fast_mode"
                    lgbm_clean_oracle_ranker_status = "skipped_candidate_ledger_fast_mode"
                if include_path_first_selectors:
                    (
                        _train_path_first_ranker_pred,
                        valid_path_first_ranker_pred,
                        lgbm_path_first_ranker_status,
                    ) = _fit_lgbm_ranker_prediction(
                        x_train=x_train_model,
                        train_frame=train.reset_index(drop=True),
                        train_metrics=train_metrics.reset_index(drop=True),
                        target_train=target_train.reset_index(drop=True),
                        w_train=weights,
                        x_valid=x_valid_model,
                        seeds=[seed + 130_000 for seed in seeds],
                        relevance_mode="path_first_clean",
                    )
                    lgbm_path_first_ranker_score = pd.Series(
                        valid_path_first_ranker_pred,
                        index=valid.index,
                    )
                    (
                        _train_path_first_dirty_zero_ranker_pred,
                        valid_path_first_dirty_zero_ranker_pred,
                        lgbm_path_first_dirty_zero_ranker_status,
                    ) = _fit_lgbm_ranker_prediction(
                        x_train=x_train_model,
                        train_frame=train.reset_index(drop=True),
                        train_metrics=train_metrics.reset_index(drop=True),
                        target_train=target_train.reset_index(drop=True),
                        w_train=weights,
                        x_valid=x_valid_model,
                        seeds=[seed + 140_000 for seed in seeds],
                        relevance_mode="path_first_clean_dirty_zero",
                    )
                    lgbm_path_first_dirty_zero_ranker_score = pd.Series(
                        valid_path_first_dirty_zero_ranker_pred,
                        index=valid.index,
                    )
                if include_s24_path_first_selectors:
                    (
                        _train_s24_broad_path_first_ranker_pred,
                        valid_s24_broad_path_first_ranker_pred,
                        lgbm_s24_broad_path_first_ranker_status,
                    ) = _fit_lgbm_ranker_prediction(
                        x_train=x_train_model,
                        train_frame=train.reset_index(drop=True),
                        train_metrics=train_metrics.reset_index(drop=True),
                        target_train=target_train.reset_index(drop=True),
                        w_train=weights,
                        x_valid=x_valid_model,
                        seeds=[seed + 150_000 for seed in seeds],
                        relevance_mode="s24_broad_path_first_source",
                    )
                    lgbm_s24_broad_path_first_ranker_score = pd.Series(
                        valid_s24_broad_path_first_ranker_pred,
                        index=valid.index,
                    )
                    (
                        _train_s24_broad_path_first_dirty_zero_ranker_pred,
                        valid_s24_broad_path_first_dirty_zero_ranker_pred,
                        lgbm_s24_broad_path_first_dirty_zero_ranker_status,
                    ) = _fit_lgbm_ranker_prediction(
                        x_train=x_train_model,
                        train_frame=train.reset_index(drop=True),
                        train_metrics=train_metrics.reset_index(drop=True),
                        target_train=target_train.reset_index(drop=True),
                        w_train=weights,
                        x_valid=x_valid_model,
                        seeds=[seed + 160_000 for seed in seeds],
                        relevance_mode="s24_broad_path_first_dirty_zero",
                    )
                    lgbm_s24_broad_path_first_dirty_zero_ranker_score = pd.Series(
                        valid_s24_broad_path_first_dirty_zero_ranker_pred,
                        index=valid.index,
                    )
                    (
                        _train_s28_side_s24_ranker_pred,
                        valid_s28_side_s24_ranker_pred,
                        lgbm_s28_side_s24_ranker_status,
                    ) = _fit_side_lgbm_ranker_prediction(
                        x_train=x_train_model,
                        train_frame=train.reset_index(drop=True),
                        train_metrics=train_metrics.reset_index(drop=True),
                        target_train=target_train.reset_index(drop=True),
                        w_train=weights,
                        x_valid=x_valid_model,
                        valid_metrics=valid_metrics.reset_index(drop=True),
                        seeds=[seed + 170_000 for seed in seeds],
                        relevance_mode="s24_broad_path_first_source",
                    )
                    lgbm_s28_side_s24_ranker_score = pd.Series(
                        valid_s28_side_s24_ranker_pred,
                        index=valid.index,
                    )
                    (
                        _train_s28_side_s24_dirty_zero_ranker_pred,
                        valid_s28_side_s24_dirty_zero_ranker_pred,
                        lgbm_s28_side_s24_dirty_zero_ranker_status,
                    ) = _fit_side_lgbm_ranker_prediction(
                        x_train=x_train_model,
                        train_frame=train.reset_index(drop=True),
                        train_metrics=train_metrics.reset_index(drop=True),
                        target_train=target_train.reset_index(drop=True),
                        w_train=weights,
                        x_valid=x_valid_model,
                        valid_metrics=valid_metrics.reset_index(drop=True),
                        seeds=[seed + 180_000 for seed in seeds],
                        relevance_mode="s24_broad_path_first_dirty_zero",
                    )
                    lgbm_s28_side_s24_dirty_zero_ranker_score = pd.Series(
                        valid_s28_side_s24_dirty_zero_ranker_pred,
                        index=valid.index,
                    )
                    (
                        _train_s30_side_asym_ranker_pred,
                        valid_s30_side_asym_ranker_pred,
                        lgbm_s30_side_asym_ranker_status,
                    ) = _fit_side_lgbm_ranker_prediction(
                        x_train=x_train_model,
                        train_frame=train.reset_index(drop=True),
                        train_metrics=train_metrics.reset_index(drop=True),
                        target_train=target_train.reset_index(drop=True),
                        w_train=weights,
                        x_valid=x_valid_model,
                        valid_metrics=valid_metrics.reset_index(drop=True),
                        seeds=[seed + 190_000 for seed in seeds],
                        relevance_mode="s30_side_asymmetric_path_first_source",
                    )
                    lgbm_s30_side_asym_ranker_score = pd.Series(
                        valid_s30_side_asym_ranker_pred,
                        index=valid.index,
                    )
                    (
                        _train_s30_side_asym_dirty_zero_ranker_pred,
                        valid_s30_side_asym_dirty_zero_ranker_pred,
                        lgbm_s30_side_asym_dirty_zero_ranker_status,
                    ) = _fit_side_lgbm_ranker_prediction(
                        x_train=x_train_model,
                        train_frame=train.reset_index(drop=True),
                        train_metrics=train_metrics.reset_index(drop=True),
                        target_train=target_train.reset_index(drop=True),
                        w_train=weights,
                        x_valid=x_valid_model,
                        valid_metrics=valid_metrics.reset_index(drop=True),
                        seeds=[seed + 200_000 for seed in seeds],
                        relevance_mode="s30_side_asymmetric_path_first_dirty_zero",
                    )
                    lgbm_s30_side_asym_dirty_zero_ranker_score = pd.Series(
                        valid_s30_side_asym_dirty_zero_ranker_pred,
                        index=valid.index,
                    )
                if include_s42_source_selectors:
                    (
                        x_train_s42,
                        x_valid_s42,
                        s42_interaction_diag,
                    ) = _augment_s42_source_features(x_train_model, x_valid_model)
                    (
                        train_s42_side_interaction_dirty_zero_ranker_pred,
                        valid_s42_side_interaction_dirty_zero_ranker_pred,
                        lgbm_s42_side_interaction_dirty_zero_ranker_status,
                    ) = _fit_side_lgbm_ranker_prediction(
                        x_train=x_train_s42,
                        train_frame=train.reset_index(drop=True),
                        train_metrics=train_metrics.reset_index(drop=True),
                        target_train=target_train.reset_index(drop=True),
                        w_train=weights,
                        x_valid=x_valid_s42,
                        valid_metrics=valid_metrics.reset_index(drop=True),
                        seeds=[seed + 240_000 for seed in seeds],
                        relevance_mode="s30_side_asymmetric_path_first_dirty_zero",
                    )
                    lgbm_s42_side_interaction_dirty_zero_ranker_score = pd.Series(
                        valid_s42_side_interaction_dirty_zero_ranker_pred,
                        index=valid.index,
                    )
                    s44_train_relevance = _ranker_relevance(
                        train_frame=train.reset_index(drop=True),
                        train_metrics=train_metrics.reset_index(drop=True),
                        target=target_train.reset_index(drop=True),
                        mode="s30_side_asymmetric_path_first_dirty_zero",
                    )
                    (
                        s44_calibrated_score,
                        s44_sign_calibration_diag,
                    ) = _side_sign_calibrated_ranker_score(
                        train_pred=train_s42_side_interaction_dirty_zero_ranker_pred,
                        valid_pred=valid_s42_side_interaction_dirty_zero_ranker_pred,
                        train_metrics=train_metrics.reset_index(drop=True),
                        valid_metrics=valid_metrics.reset_index(drop=True),
                        train_relevance=s44_train_relevance,
                    )
                    lgbm_s44_side_interaction_sign_calibrated_ranker_score = pd.Series(
                        s44_calibrated_score.to_numpy(dtype=np.float32, copy=False),
                        index=valid.index,
                    )
                    s45_recent_idx, s45_recent_base_diag = _recent_train_indices(
                        train_frame=train.reset_index(drop=True),
                        valid_frame=valid.reset_index(drop=True),
                        lookback_days=45,
                        min_rows=2000,
                    )
                    s45_recent_train_diag = {
                        f"s45_{key}": value
                        for key, value in s45_recent_base_diag.items()
                    }
                    (
                        _train_s45_side_interaction_roll45_dirty_zero_ranker_pred,
                        valid_s45_side_interaction_roll45_dirty_zero_ranker_pred,
                        lgbm_s45_side_interaction_roll45_dirty_zero_ranker_status,
                    ) = _fit_side_lgbm_ranker_prediction(
                        x_train=x_train_s42.reset_index(drop=True)
                        .iloc[s45_recent_idx]
                        .reset_index(drop=True),
                        train_frame=train.reset_index(drop=True)
                        .iloc[s45_recent_idx]
                        .reset_index(drop=True),
                        train_metrics=train_metrics.reset_index(drop=True)
                        .iloc[s45_recent_idx]
                        .reset_index(drop=True),
                        target_train=target_train.reset_index(drop=True)
                        .iloc[s45_recent_idx]
                        .reset_index(drop=True),
                        w_train=weights.reset_index(drop=True)
                        .iloc[s45_recent_idx]
                        .reset_index(drop=True),
                        x_valid=x_valid_s42,
                        valid_metrics=valid_metrics.reset_index(drop=True),
                        seeds=[seed + 250_000 for seed in seeds],
                        relevance_mode="s30_side_asymmetric_path_first_dirty_zero",
                    )
                    lgbm_s45_side_interaction_roll45_dirty_zero_ranker_score = pd.Series(
                        valid_s45_side_interaction_roll45_dirty_zero_ranker_pred,
                        index=valid.index,
                    )
                if include_timeout_aware_selectors:
                    (
                        _train_timeout_aware_clean_ranker_pred,
                        valid_timeout_aware_clean_ranker_pred,
                        lgbm_timeout_aware_clean_ranker_status,
                    ) = _fit_lgbm_ranker_prediction(
                        x_train=x_train_model,
                        train_frame=train.reset_index(drop=True),
                        train_metrics=train_metrics.reset_index(drop=True),
                        target_train=target_train.reset_index(drop=True),
                        w_train=weights,
                        x_valid=x_valid_model,
                        seeds=[seed + 210_000 for seed in seeds],
                        relevance_mode="timeout_aware_clean_source",
                    )
                    lgbm_timeout_aware_clean_ranker_score = pd.Series(
                        valid_timeout_aware_clean_ranker_pred,
                        index=valid.index,
                    )
            base_arm = (
                f"{label_arm}::{weight_arm}"
                if model_feature_selector == "all"
                else f"{label_arm}::{weight_arm}::{model_feature_selector}"
            )
            selector_prefix = (
                "feature_store_model_seed_ensemble_smoke_oos"
                if len(seeds) > 1
                else "feature_store_model_smoke_oos"
            )
            base_variants: list[tuple[str, pd.Series, dict[str, Any], np.ndarray | None]] = [
                ("raw_utility", score, {}, None),
            ]
            if include_risk_selector_variants:
                risk_penalty = (score - 0.35 * bad_mae_pred - 0.10 * timeout_pred).astype(np.float32)
                strong_risk_penalty = (
                    score - 0.55 * bad_mae_pred - 0.15 * timeout_pred
                ).astype(np.float32)
                side_risk_penalty = (
                    score - 0.55 * side_bad_mae_pred - 0.15 * side_timeout_pred
                ).astype(np.float32)
                feature_gap_penalty = (
                    score - 0.55 * bad_mae_pred - 0.15 * timeout_pred - 0.20 * feature_gap_risk
                ).astype(np.float32)
                lgbm_ranker_risk_score = (
                    pd.to_numeric(lgbm_ranker_score, errors="coerce").rank(method="average", pct=True)
                    - 0.45 * bad_mae_pred
                    - 0.15 * timeout_pred
                ).astype(np.float32)
                lgbm_path_ranker_risk_score = (
                    pd.to_numeric(lgbm_path_ranker_score, errors="coerce").rank(method="average", pct=True)
                    - 0.55 * bad_mae_pred
                    - 0.20 * timeout_pred
                    + 0.20 * clean_path_pred
                ).astype(np.float32)
                lgbm_oracle_ranker_risk_score = (
                    pd.to_numeric(lgbm_oracle_ranker_score, errors="coerce").rank(method="average", pct=True)
                    - 0.50 * bad_mae_pred
                    - 0.18 * timeout_pred
                ).astype(np.float32)
                lgbm_clean_oracle_ranker_risk_score = (
                    pd.to_numeric(
                        lgbm_clean_oracle_ranker_score,
                        errors="coerce",
                    ).rank(method="average", pct=True)
                    - 0.35 * bad_mae_pred
                    - 0.15 * timeout_pred
                    + 0.10 * clean_path_pred
                ).astype(np.float32)
                lgbm_path_first_ranker_pct = pd.to_numeric(
                    lgbm_path_first_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True).astype(np.float32)
                lgbm_path_first_dirty_zero_ranker_pct = pd.to_numeric(
                    lgbm_path_first_dirty_zero_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True).astype(np.float32)
                lgbm_s24_broad_path_first_ranker_pct = pd.to_numeric(
                    lgbm_s24_broad_path_first_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True).astype(np.float32)
                lgbm_s24_broad_path_first_dirty_zero_ranker_pct = pd.to_numeric(
                    lgbm_s24_broad_path_first_dirty_zero_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True).astype(np.float32)
                lgbm_s28_side_s24_ranker_pct = pd.to_numeric(
                    lgbm_s28_side_s24_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True).astype(np.float32)
                lgbm_s28_side_s24_dirty_zero_ranker_pct = pd.to_numeric(
                    lgbm_s28_side_s24_dirty_zero_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True).astype(np.float32)
                lgbm_s30_side_asym_ranker_pct = pd.to_numeric(
                    lgbm_s30_side_asym_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True).astype(np.float32)
                lgbm_s30_side_asym_dirty_zero_ranker_pct = pd.to_numeric(
                    lgbm_s30_side_asym_dirty_zero_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True).astype(np.float32)
                lgbm_s42_side_interaction_dirty_zero_ranker_pct = pd.to_numeric(
                    lgbm_s42_side_interaction_dirty_zero_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True).astype(np.float32)
                lgbm_s44_side_interaction_sign_calibrated_ranker_pct = pd.to_numeric(
                    lgbm_s44_side_interaction_sign_calibrated_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True).astype(np.float32)
                lgbm_s45_side_interaction_roll45_dirty_zero_ranker_pct = pd.to_numeric(
                    lgbm_s45_side_interaction_roll45_dirty_zero_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True).astype(np.float32)
                lgbm_timeout_aware_clean_ranker_pct = pd.to_numeric(
                    lgbm_timeout_aware_clean_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True).astype(np.float32)
                lgbm_utility_pct = pd.to_numeric(
                    lgbm_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True)
                lgbm_path_pct = pd.to_numeric(
                    lgbm_path_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True)
                lgbm_oracle_pct = pd.to_numeric(
                    lgbm_oracle_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True)
                lgbm_clean_oracle_pct = pd.to_numeric(
                    lgbm_clean_oracle_ranker_score,
                    errors="coerce",
                ).rank(method="average", pct=True)
                lgbm_utility_path_blend_75 = (
                    0.75 * lgbm_utility_pct
                    + 0.25 * lgbm_path_pct
                    - 0.28 * bad_mae_pred
                    - 0.12 * timeout_pred
                ).astype(np.float32)
                lgbm_utility_path_blend_60 = (
                    0.60 * lgbm_utility_pct
                    + 0.40 * lgbm_path_pct
                    - 0.32 * bad_mae_pred
                    - 0.14 * timeout_pred
                ).astype(np.float32)
                lgbm_utility_path_blend_50 = (
                    0.50 * lgbm_utility_pct
                    + 0.50 * lgbm_path_pct
                    - 0.35 * bad_mae_pred
                    - 0.15 * timeout_pred
                ).astype(np.float32)
                lgbm_three_ranker_blend = (
                    0.55 * lgbm_utility_pct
                    + 0.25 * lgbm_path_pct
                    + 0.20 * lgbm_oracle_pct
                    - 0.30 * bad_mae_pred
                    - 0.14 * timeout_pred
                ).astype(np.float32)
                lgbm_utility_clean_oracle_blend = (
                    0.55 * lgbm_utility_pct
                    + 0.45 * lgbm_clean_oracle_pct
                    - 0.30 * bad_mae_pred
                    - 0.12 * timeout_pred
                ).astype(np.float32)
                lgbm_clean_path_oracle_blend = (
                    0.35 * lgbm_utility_pct
                    + 0.35 * lgbm_clean_oracle_pct
                    + 0.30 * lgbm_path_pct
                    - 0.36 * bad_mae_pred
                    - 0.14 * timeout_pred
                    + 0.10 * clean_path_pred
                ).astype(np.float32)
                lgbm_utility_clean_dirty_penalty = (
                    lgbm_ranker_risk_score
                    - 0.30 * clean_dirty_positive_risk
                ).astype(np.float32)
                lgbm_utility_clean_dirty_strong_penalty = (
                    lgbm_ranker_risk_score
                    - 0.45 * clean_dirty_positive_risk
                ).astype(np.float32)
                lgbm_path_clean_dirty_penalty = (
                    lgbm_path_ranker_risk_score
                    - 0.30 * clean_dirty_positive_risk
                ).astype(np.float32)
                lgbm_utility_lgbm_risk_score = (
                    pd.to_numeric(lgbm_ranker_score, errors="coerce").rank(method="average", pct=True)
                    - 0.55 * lgbm_bad_mae_pred
                    - 0.18 * lgbm_timeout_pred
                ).astype(np.float32)
                lgbm_utility_blended_risk_score = (
                    pd.to_numeric(lgbm_ranker_score, errors="coerce").rank(method="average", pct=True)
                    - 0.30 * bad_mae_pred
                    - 0.35 * lgbm_bad_mae_pred
                    - 0.10 * timeout_pred
                    - 0.12 * lgbm_timeout_pred
                ).astype(np.float32)
                lgbm_path_lgbm_risk_score = (
                    pd.to_numeric(lgbm_path_ranker_score, errors="coerce").rank(method="average", pct=True)
                    - 0.60 * lgbm_bad_mae_pred
                    - 0.20 * lgbm_timeout_pred
                    + 0.15 * clean_path_pred
                ).astype(np.float32)
                lgbm_utility_ts_pct = _timestamp_rank_percentile(
                    valid,
                    lgbm_ranker_score,
                    ascending=True,
                )
                lgbm_path_ts_pct = _timestamp_rank_percentile(
                    valid,
                    lgbm_path_ranker_score,
                    ascending=True,
                )
                lgbm_bad_mae_ts_pct = _timestamp_rank_percentile(
                    valid,
                    lgbm_bad_mae_pred,
                    ascending=True,
                )
                lgbm_timeout_ts_pct = _timestamp_rank_percentile(
                    valid,
                    lgbm_timeout_pred,
                    ascending=True,
                )
                lgbm_clean_path_ts_pct = _timestamp_rank_percentile(
                    valid,
                    lgbm_clean_path_pred,
                    ascending=True,
                )
                lgbm_dirty_positive_bad_mae_ts_pct = _timestamp_rank_percentile(
                    valid,
                    lgbm_dirty_positive_bad_mae_pred,
                    ascending=True,
                )
                lgbm_positive_clean_path_ts_pct = _timestamp_rank_percentile(
                    valid,
                    lgbm_positive_clean_path_pred,
                    ascending=True,
                )
                lgbm_side_dirty_positive_bad_mae_ts_pct = _timestamp_rank_percentile(
                    valid,
                    lgbm_side_dirty_positive_bad_mae_pred,
                    ascending=True,
                )
                lgbm_side_positive_clean_path_ts_pct = _timestamp_rank_percentile(
                    valid,
                    lgbm_side_positive_clean_path_pred,
                    ascending=True,
                )
                lgbm_clean_surplus_score = (
                    0.50 * lgbm_utility_ts_pct
                    + 0.25 * lgbm_path_ts_pct
                    + 0.45 * lgbm_clean_path_ts_pct
                    - 0.30 * lgbm_bad_mae_ts_pct
                    - 0.20 * lgbm_timeout_ts_pct
                ).astype(np.float32)
                lgbm_clean_path_utility_score = (
                    0.65 * lgbm_utility_ts_pct
                    + 0.50 * lgbm_clean_path_ts_pct
                    - 0.35 * lgbm_bad_mae_ts_pct
                    - 0.15 * lgbm_timeout_ts_pct
                ).astype(np.float32)
                lgbm_path_clean_surplus_score = (
                    0.40 * lgbm_path_ts_pct
                    + 0.35 * lgbm_utility_ts_pct
                    + 0.50 * lgbm_clean_path_ts_pct
                    - 0.35 * lgbm_bad_mae_ts_pct
                    - 0.25 * lgbm_timeout_ts_pct
                ).astype(np.float32)
                lgbm_dirty_positive_aware_score = (
                    0.50 * lgbm_utility_ts_pct
                    + 0.30 * lgbm_path_ts_pct
                    + 0.35 * lgbm_clean_path_ts_pct
                    - 0.60 * lgbm_dirty_positive_bad_mae_ts_pct
                    - 0.20 * lgbm_timeout_ts_pct
                ).astype(np.float32)
                lgbm_path_dirty_positive_aware_score = (
                    0.45 * lgbm_path_ts_pct
                    + 0.35 * lgbm_utility_ts_pct
                    + 0.30 * lgbm_clean_path_ts_pct
                    - 0.65 * lgbm_dirty_positive_bad_mae_ts_pct
                    - 0.20 * lgbm_timeout_ts_pct
                ).astype(np.float32)
                lgbm_exec_clean_score = (
                    0.35 * lgbm_path_ts_pct
                    + 0.25 * lgbm_utility_ts_pct
                    + 0.55 * lgbm_clean_path_ts_pct
                    - 0.45 * lgbm_dirty_positive_bad_mae_ts_pct
                    - 0.30 * lgbm_bad_mae_ts_pct
                    - 0.35 * lgbm_timeout_ts_pct
                ).astype(np.float32)
                lgbm_exec_clean_strict_score = (
                    0.40 * lgbm_path_ts_pct
                    + 0.20 * lgbm_utility_ts_pct
                    + 0.65 * lgbm_clean_path_ts_pct
                    - 0.60 * lgbm_dirty_positive_bad_mae_ts_pct
                    - 0.35 * lgbm_bad_mae_ts_pct
                    - 0.45 * lgbm_timeout_ts_pct
                ).astype(np.float32)
                lgbm_exec_clean_contrast_score = (
                    lgbm_exec_clean_score
                    - 0.45 * clean_dirty_positive_risk
                ).astype(np.float32)
                lgbm_exec_clean_strict_contrast_score = (
                    lgbm_exec_clean_strict_score
                    - 0.60 * clean_dirty_positive_risk
                ).astype(np.float32)
                lgbm_positive_clean_exec_score = (
                    0.35 * lgbm_path_ts_pct
                    + 0.20 * lgbm_utility_ts_pct
                    + 0.70 * lgbm_positive_clean_path_ts_pct
                    - 0.45 * lgbm_dirty_positive_bad_mae_ts_pct
                    - 0.25 * lgbm_bad_mae_ts_pct
                    - 0.35 * lgbm_timeout_ts_pct
                ).astype(np.float32)
                lgbm_positive_clean_exec_strict_score = (
                    0.35 * lgbm_path_ts_pct
                    + 0.15 * lgbm_utility_ts_pct
                    + 0.85 * lgbm_positive_clean_path_ts_pct
                    - 0.55 * lgbm_dirty_positive_bad_mae_ts_pct
                    - 0.30 * lgbm_bad_mae_ts_pct
                    - 0.45 * lgbm_timeout_ts_pct
                ).astype(np.float32)
                (
                    s22_bucket_quality_score,
                    s22_bucket_quality_rank_pct,
                    s22_bucket_relaxed_pass_count,
                    s22_bucket_strict_pass_count,
                    s22_bucket_quality_diag,
                ) = _prior_bucket_quality_overlay(
                    train=x_train_model.reset_index(drop=True),
                    valid=x_valid_model.reset_index(drop=True),
                    train_metrics=train_metrics.reset_index(drop=True),
                    valid_metrics=valid_metrics.reset_index(drop=True),
                    features=model_features,
                    min_bucket_rows=80,
                )
                (
                    s46_bucket_quality_score,
                    s46_bucket_quality_rank_pct,
                    s46_bucket_quality_diag,
                ) = _side_spread_aegmm_bucket_quality(
                    train=x_train_model.reset_index(drop=True),
                    valid=x_valid_model.reset_index(drop=True),
                    train_metrics=train_metrics.reset_index(drop=True),
                    valid_metrics=valid_metrics.reset_index(drop=True),
                    min_bucket_rows=120,
                )
                s22_bucket_relaxed_pass_rank = _timestamp_rank_percentile(
                    valid,
                    s22_bucket_relaxed_pass_count,
                    ascending=True,
                )
                s22_bucket_strict_pass_rank = _timestamp_rank_percentile(
                    valid,
                    s22_bucket_strict_pass_count,
                    ascending=True,
                )
                lgbm_positive_clean_bucket_score = (
                    0.75 * lgbm_positive_clean_exec_score
                    + 0.25 * s22_bucket_quality_rank_pct
                    + 0.05 * s22_bucket_relaxed_pass_rank
                ).astype(np.float32)
                lgbm_positive_clean_bucket_strict_score = (
                    0.70 * lgbm_positive_clean_exec_strict_score
                    + 0.30 * s22_bucket_quality_rank_pct
                    + 0.05 * s22_bucket_strict_pass_rank
                ).astype(np.float32)
                s7_candidate_score = _max_percentile_score(
                    [
                        score,
                        risk_penalty,
                        strong_risk_penalty,
                        side_risk_penalty,
                        clean_path_pred,
                        lgbm_clean_path_pred,
                        lgbm_ranker_score,
                        lgbm_path_ranker_score,
                        lgbm_oracle_ranker_score,
                        lgbm_clean_oracle_ranker_score,
                        lgbm_path_first_ranker_score,
                        lgbm_path_first_dirty_zero_ranker_score,
                        lgbm_s24_broad_path_first_ranker_score,
                        lgbm_s24_broad_path_first_dirty_zero_ranker_score,
                        lgbm_s28_side_s24_ranker_score,
                        lgbm_s28_side_s24_dirty_zero_ranker_score,
                        lgbm_s30_side_asym_ranker_score,
                        lgbm_s30_side_asym_dirty_zero_ranker_score,
                        lgbm_s42_side_interaction_dirty_zero_ranker_score,
                        lgbm_s44_side_interaction_sign_calibrated_ranker_score,
                        lgbm_s45_side_interaction_roll45_dirty_zero_ranker_score,
                        s46_bucket_quality_score,
                        lgbm_side_positive_clean_path_pred,
                    ],
                    len(valid),
                )
                s7_clean_path_enriched_score = _max_percentile_score(
                    [
                        strong_risk_penalty,
                        clean_path_pred,
                        lgbm_clean_path_pred,
                        lgbm_path_ranker_risk_score,
                        lgbm_oracle_ranker_risk_score,
                    ],
                    len(valid),
                )
                base_variants.extend(
                    [
                        (
                            "bad_mae_timeout_penalty",
                            risk_penalty,
                            {"bad_mae_penalty_lambda": 0.35, "timeout_penalty_lambda": 0.10},
                            None,
                        ),
                        (
                            "strong_bad_mae_timeout_penalty",
                            strong_risk_penalty,
                            {"bad_mae_penalty_lambda": 0.55, "timeout_penalty_lambda": 0.15},
                            None,
                        ),
                        (
                            "side_specific_bad_mae_timeout_penalty",
                            side_risk_penalty,
                            {
                                "bad_mae_penalty_lambda": 0.55,
                                "timeout_penalty_lambda": 0.15,
                                "risk_head": "side_specific",
                            },
                            None,
                        ),
                        (
                            "clean_path_probability",
                            clean_path_pred,
                            {"clean_path_head": True},
                            None,
                        ),
                        (
                            "feature_gap_risk_penalty",
                            feature_gap_penalty,
                            {
                                "bad_mae_penalty_lambda": 0.55,
                                "timeout_penalty_lambda": 0.15,
                                "feature_gap_risk_penalty_lambda": 0.20,
                                **feature_gap_diag,
                            },
                            None,
                        ),
                    ]
                )
            for top_frac in top_fracs:
                variants: list[tuple[str, pd.Series, dict[str, Any], np.ndarray | None]] = (
                    [] if candidate_ledger_only else list(base_variants)
                )
                if include_risk_selector_variants:
                    stage_a_top20_pre_risk_mask = _per_timestamp_top_mask(
                        valid,
                        s7_candidate_score,
                        top_n=20,
                    )
                    stage_a_top50_pre_risk_mask = _per_timestamp_top_mask(
                        valid,
                        s7_candidate_score,
                        top_n=50,
                    )
                    stage_a_top10pct_pre_risk_mask = _per_timestamp_top_mask(
                        valid,
                        s7_candidate_score,
                        top_frac=0.10,
                    )
                    stage_a_candidate_pre_risk_mask = (
                        stage_a_top20_pre_risk_mask
                        | stage_a_top50_pre_risk_mask
                        | stage_a_top10pct_pre_risk_mask
                    )
                    relaxed_risk_mask = (
                        pd.to_numeric(bad_mae_pred, errors="coerce").le(0.70)
                        & pd.to_numeric(timeout_pred, errors="coerce").le(0.30)
                    ).fillna(False).to_numpy(dtype=bool)
                    relaxed_side_risk_mask = (
                        pd.to_numeric(side_bad_mae_pred, errors="coerce").le(0.70)
                        & pd.to_numeric(side_timeout_pred, errors="coerce").le(0.30)
                    ).fillna(False).to_numpy(dtype=bool)
                    strict_risk_mask = (
                        pd.to_numeric(bad_mae_pred, errors="coerce").le(0.57)
                        & pd.to_numeric(timeout_pred, errors="coerce").le(0.12)
                    ).fillna(False).to_numpy(dtype=bool)
                    stage_a_candidate_mask = stage_a_candidate_pre_risk_mask & relaxed_risk_mask
                    stage_a_side_candidate_mask = (
                        stage_a_candidate_pre_risk_mask & relaxed_side_risk_mask
                    )
                    s7_stage_diag: dict[str, Any] = {
                        "s7_two_stage_enabled": True,
                        "s7_stage_a_top_n_primary": 20,
                        "s7_stage_a_top_n_alt": 50,
                        "s7_stage_a_top_frac": 0.10,
                        "s7_relaxed_pred_bad_mae_cap": 0.70,
                        "s7_relaxed_pred_timeout_cap": 0.30,
                        "s7_strict_pred_bad_mae_cap": 0.57,
                        "s7_strict_pred_timeout_cap": 0.12,
                    }
                    for prefix, mask in (
                        ("stageA_top20_pre_risk", stage_a_top20_pre_risk_mask),
                        ("stageA_top50_pre_risk", stage_a_top50_pre_risk_mask),
                        ("stageA_top10pct_pre_risk", stage_a_top10pct_pre_risk_mask),
                        ("stageA_candidate_pre_risk", stage_a_candidate_pre_risk_mask),
                        ("stageA_candidate", stage_a_candidate_mask),
                    ):
                        s7_stage_diag.update(
                            _oracle_recall_stats(
                                metrics=valid_metrics,
                                mask=mask,
                                top_frac=float(top_frac),
                                prefix=prefix,
                            )
                        )
                    s7_specs: list[tuple[str, pd.Series, np.ndarray, dict[str, Any]]] = [
                        (
                            "s7a_no_prefilter_rerank",
                            strong_risk_penalty,
                            pd.to_numeric(strong_risk_penalty, errors="coerce")
                            .notna()
                            .to_numpy(dtype=bool),
                            {
                                "s7_ablation": "ranker_without_hard_prefilter",
                                "s7_stage_b_risk_cap": "none",
                            },
                        ),
                        (
                            "s7b_high_recall_risk_cap_rerank",
                            strong_risk_penalty,
                            relaxed_risk_mask,
                            {
                                "s7_ablation": "high_recall_risk_caps",
                                "pred_bad_mae_cap": 0.70,
                                "pred_timeout_cap": 0.30,
                            },
                        ),
                        (
                            "s7c_side_specific_high_recall_risk_cap_rerank",
                            side_risk_penalty,
                            relaxed_side_risk_mask,
                            {
                                "s7_ablation": "side_specific_risk_heads",
                                "risk_head": "side_specific",
                                "pred_bad_mae_cap": 0.70,
                                "pred_timeout_cap": 0.30,
                            },
                        ),
                        (
                            "s7_two_stage_candidate_rerank",
                            strong_risk_penalty,
                            stage_a_candidate_mask,
                            {
                                "s7_ablation": "two_stage_candidate_discovery_then_rerank",
                                "pred_bad_mae_cap": 0.70,
                                "pred_timeout_cap": 0.30,
                            },
                        ),
                        (
                            "s7_two_stage_strict_risk_candidate_rerank",
                            strong_risk_penalty,
                            stage_a_candidate_pre_risk_mask & strict_risk_mask,
                            {
                                "s7_ablation": "two_stage_candidate_discovery_then_strict_risk_rerank",
                                "pred_bad_mae_cap": 0.57,
                                "pred_timeout_cap": 0.12,
                            },
                        ),
                        (
                            "s7d_clean_path_enriched_candidate_rerank",
                            s7_clean_path_enriched_score,
                            stage_a_side_candidate_mask,
                            {
                                "s7_ablation": "oracle_enriched_proxy_clean_path_head",
                                "risk_head": "side_specific",
                                "pred_bad_mae_cap": 0.70,
                                "pred_timeout_cap": 0.30,
                            },
                        ),
                        (
                            "s8_lgbm_utility_ranker_stageA_rerank",
                            lgbm_ranker_risk_score,
                            stage_a_candidate_mask,
                            {
                                "s7_ablation": "lgbm_ranker_stage_b_utility_relevance",
                                "ranker_type": "lgbm_lambdarank",
                                "ranker_relevance": "utility_quintile",
                                "ranker_status": lgbm_ranker_status,
                                "pred_bad_mae_cap": 0.70,
                                "pred_timeout_cap": 0.30,
                            },
                        ),
                        (
                            "s8_lgbm_path_quality_ranker_stageA_rerank",
                            lgbm_path_ranker_risk_score,
                            stage_a_candidate_mask,
                            {
                                "s7_ablation": "lgbm_ranker_stage_b_path_quality_relevance",
                                "ranker_type": "lgbm_lambdarank",
                                "ranker_relevance": "path_quality",
                                "ranker_status": lgbm_path_ranker_status,
                                "pred_bad_mae_cap": 0.70,
                                "pred_timeout_cap": 0.30,
                            },
                        ),
                        (
                            "s8_lgbm_oracle_enriched_ranker_stageA_rerank",
                            lgbm_oracle_ranker_risk_score,
                            stage_a_candidate_mask,
                            {
                                "s7_ablation": "lgbm_ranker_stage_b_oracle_enriched_relevance",
                                "ranker_type": "lgbm_lambdarank",
                                "ranker_relevance": "oracle_enriched",
                                "ranker_status": lgbm_oracle_ranker_status,
                                "pred_bad_mae_cap": 0.70,
                                "pred_timeout_cap": 0.30,
                            },
                        ),
                        (
                            "s8_lgbm_path_quality_ranker_strict_stageA_rerank",
                            lgbm_path_ranker_risk_score,
                            stage_a_candidate_pre_risk_mask & strict_risk_mask,
                            {
                                "s7_ablation": "lgbm_ranker_stage_b_path_quality_strict_risk",
                                "ranker_type": "lgbm_lambdarank",
                                "ranker_relevance": "path_quality",
                                "ranker_status": lgbm_path_ranker_status,
                                "pred_bad_mae_cap": 0.57,
                                "pred_timeout_cap": 0.12,
                            },
                        ),
                        (
                            "s13_lgbm_clean_oracle_ranker_stageA_rerank",
                            lgbm_clean_oracle_ranker_risk_score,
                            stage_a_candidate_mask,
                            {
                                "s7_ablation": "lgbm_ranker_stage_b_clean_oracle_relevance",
                                "ranker_type": "lgbm_lambdarank",
                                "ranker_relevance": "clean_oracle",
                                "ranker_status": lgbm_clean_oracle_ranker_status,
                                "pred_bad_mae_cap": 0.70,
                                "pred_timeout_cap": 0.30,
                            },
                        ),
                        (
                            "s13_lgbm_clean_oracle_ranker_strict_stageA_rerank",
                            lgbm_clean_oracle_ranker_risk_score,
                            stage_a_candidate_pre_risk_mask & strict_risk_mask,
                            {
                                "s7_ablation": (
                                    "lgbm_ranker_stage_b_clean_oracle_relevance_strict_risk"
                                ),
                                "ranker_type": "lgbm_lambdarank",
                                "ranker_relevance": "clean_oracle",
                                "ranker_status": lgbm_clean_oracle_ranker_status,
                                "pred_bad_mae_cap": 0.57,
                                "pred_timeout_cap": 0.12,
                            },
                        ),
                        (
                            "s13_lgbm_utility_clean_oracle_blend_stageA_rerank",
                            lgbm_utility_clean_oracle_blend,
                            stage_a_candidate_mask,
                            {
                                "s7_ablation": "lgbm_utility_clean_oracle_blend",
                                "ranker_type": "lgbm_lambdarank_blend",
                                "ranker_status": (
                                    f"utility:{lgbm_ranker_status};"
                                    f"clean_oracle:{lgbm_clean_oracle_ranker_status}"
                                ),
                                "utility_ranker_weight": 0.55,
                                "clean_oracle_ranker_weight": 0.45,
                                "pred_bad_mae_cap": 0.70,
                                "pred_timeout_cap": 0.30,
                            },
                        ),
                        (
                            "s13_lgbm_clean_path_oracle_blend_stageA_rerank",
                            lgbm_clean_path_oracle_blend,
                            stage_a_candidate_mask,
                            {
                                "s7_ablation": "lgbm_clean_path_oracle_blend",
                                "ranker_type": "lgbm_lambdarank_blend",
                                "ranker_status": (
                                    f"utility:{lgbm_ranker_status};"
                                    f"path:{lgbm_path_ranker_status};"
                                    f"clean_oracle:{lgbm_clean_oracle_ranker_status}"
                                ),
                                "utility_ranker_weight": 0.35,
                                "path_ranker_weight": 0.30,
                                "clean_oracle_ranker_weight": 0.35,
                                "pred_bad_mae_cap": 0.70,
                                "pred_timeout_cap": 0.30,
                            },
                        ),
                    ]
                    for (
                        clean_dirty_name,
                        clean_dirty_score,
                        clean_dirty_lambda,
                        ranker_relevance,
                    ) in (
                        (
                            "s14_lgbm_utility_clean_dirty_penalty_stageA_rerank",
                            lgbm_utility_clean_dirty_penalty,
                            0.30,
                            "utility_quintile",
                        ),
                        (
                            "s14_lgbm_utility_clean_dirty_strong_penalty_stageA_rerank",
                            lgbm_utility_clean_dirty_strong_penalty,
                            0.45,
                            "utility_quintile",
                        ),
                        (
                            "s14_lgbm_path_clean_dirty_penalty_stageA_rerank",
                            lgbm_path_clean_dirty_penalty,
                            0.30,
                            "path_quality",
                        ),
                    ):
                        base_clean_dirty_diag = {
                            "s7_ablation": "lgbm_ranker_clean_dirty_positive_risk_penalty",
                            "ranker_type": "lgbm_lambdarank",
                            "ranker_relevance": ranker_relevance,
                            "ranker_status": (
                                lgbm_path_ranker_status
                                if ranker_relevance == "path_quality"
                                else lgbm_ranker_status
                            ),
                            "pred_bad_mae_cap": 0.70,
                            "pred_timeout_cap": 0.30,
                            "clean_dirty_positive_risk_penalty_lambda": clean_dirty_lambda,
                            **clean_dirty_positive_diag,
                        }
                        s7_specs.append(
                            (
                                clean_dirty_name,
                                clean_dirty_score,
                                stage_a_candidate_mask,
                                base_clean_dirty_diag,
                            )
                        )
                        for final_frac in (0.015, 0.020):
                            s7_specs.append(
                                (
                                    clean_dirty_name.replace(
                                        "_stageA_",
                                        f"_stageA_final_frac_{int(round(final_frac * 1000)):03d}_",
                                    ),
                                    clean_dirty_score,
                                    stage_a_candidate_mask,
                                    {
                                        **base_clean_dirty_diag,
                                        "selection_top_frac": float(final_frac),
                                    },
                                )
                            )
                        for clean_dirty_cap in (0.45, 0.50, 0.55, 0.60):
                            clean_dirty_mask = (
                                stage_a_candidate_mask
                                & pd.to_numeric(clean_dirty_positive_risk, errors="coerce")
                                .le(float(clean_dirty_cap))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                            )
                            s7_specs.append(
                                (
                                    clean_dirty_name.replace(
                                        "_stageA_",
                                        (
                                            "_clean_dirty_cap_"
                                            f"{int(round(clean_dirty_cap * 100)):02d}_stageA_"
                                        ),
                                    ),
                                    clean_dirty_score,
                                    clean_dirty_mask,
                                    {
                                        **base_clean_dirty_diag,
                                        "clean_dirty_positive_risk_cap": float(clean_dirty_cap),
                                    },
                                )
                            )
                    for path_pct_min in (0.50, 0.60, 0.70):
                        path_agreement_mask = (
                            stage_a_candidate_mask
                            & pd.to_numeric(lgbm_path_pct, errors="coerce")
                            .ge(float(path_pct_min))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        s7_specs.append(
                            (
                                (
                                    "s15_lgbm_utility_path_agreement"
                                    f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                    "_stageA_rerank"
                                ),
                                lgbm_ranker_risk_score,
                                path_agreement_mask,
                                {
                                    "s7_ablation": "lgbm_utility_ranker_path_quality_agreement",
                                    "ranker_type": "lgbm_lambdarank_agreement",
                                    "ranker_status": (
                                        f"utility:{lgbm_ranker_status};"
                                        f"path:{lgbm_path_ranker_status}"
                                    ),
                                    "ranker_relevance": "utility_quintile",
                                    "path_ranker_min_percentile": float(path_pct_min),
                                    "pred_bad_mae_cap": 0.70,
                                    "pred_timeout_cap": 0.30,
                                },
                            )
                        )
                        for timeout_cap in (0.15, 0.20):
                            path_timeout_mask = (
                                path_agreement_mask
                                & pd.to_numeric(timeout_pred, errors="coerce")
                                .le(float(timeout_cap))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                            )
                            s7_specs.append(
                                (
                                    (
                                        "s15_lgbm_utility_path_agreement"
                                        f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                        f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                        "_stageA_rerank"
                                    ),
                                    lgbm_ranker_risk_score,
                                    path_timeout_mask,
                                    {
                                        "s7_ablation": (
                                            "lgbm_utility_ranker_path_quality_agreement_timeout_cap"
                                        ),
                                        "ranker_type": "lgbm_lambdarank_agreement",
                                        "ranker_status": (
                                            f"utility:{lgbm_ranker_status};"
                                            f"path:{lgbm_path_ranker_status}"
                                        ),
                                        "ranker_relevance": "utility_quintile",
                                        "path_ranker_min_percentile": float(path_pct_min),
                                        "pred_bad_mae_cap": 0.70,
                                        "pred_timeout_cap": float(timeout_cap),
                                    },
                                )
                            )
                            s34_opportunity_clean_score = (
                                lgbm_ranker_risk_score
                                + 0.04 * lgbm_path_pct
                                + 0.03 * lgbm_clean_path_ts_pct
                                + 0.03 * lgbm_side_positive_clean_path_ts_pct
                                - 0.04 * lgbm_dirty_positive_bad_mae_ts_pct
                                - 0.03 * lgbm_side_dirty_positive_bad_mae_ts_pct
                            ).astype(np.float32)
                            s7_specs.append(
                                (
                                    (
                                        "s34_lgbm_opportunity_preserving_dirty_clean"
                                        f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                        f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                        "_stageA_rerank"
                                    ),
                                    s34_opportunity_clean_score,
                                    path_timeout_mask,
                                    {
                                        "s7_ablation": (
                                            "s34_opportunity_preserving_dirty_clean_source"
                                        ),
                                        "ranker_type": (
                                            "lgbm_utility_ranker_with_light_dirty_clean_tiebreak"
                                        ),
                                        "ranker_status": (
                                            f"utility:{lgbm_ranker_status};"
                                            f"path:{lgbm_path_ranker_status};"
                                            f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                            f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                            f"side_clean:{lgbm_side_positive_clean_path_status}"
                                        ),
                                        "ranker_relevance": (
                                            "utility_quintile_opportunity_preserving_dirty_clean"
                                        ),
                                        "path_ranker_min_percentile": float(path_pct_min),
                                        "pred_timeout_cap": float(timeout_cap),
                                        "base_score": "s15_lgbm_ranker_risk_score",
                                        "path_ranker_tiebreak_weight": 0.04,
                                        "dirty_positive_bad_mae_ts_penalty_lambda": 0.04,
                                        "side_dirty_positive_bad_mae_ts_penalty_lambda": 0.03,
                                    },
                                    )
                                )
                            s35_ultralight_score = (
                                lgbm_ranker_risk_score
                                + 0.010 * lgbm_path_pct
                                + 0.008 * lgbm_clean_path_ts_pct
                                + 0.006 * lgbm_side_positive_clean_path_ts_pct
                                - 0.012 * lgbm_dirty_positive_bad_mae_ts_pct
                                - 0.008 * lgbm_side_dirty_positive_bad_mae_ts_pct
                                - 0.006 * lgbm_bad_mae_ts_pct
                            ).astype(np.float32)
                            s7_specs.append(
                                (
                                    (
                                        "s35_lgbm_s15_ultralight_dirty_clean"
                                        f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                        f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                        "_stageA_rerank"
                                    ),
                                    s35_ultralight_score,
                                    path_timeout_mask,
                                    {
                                        "s7_ablation": (
                                            "s35_s15_ultralight_dirty_clean_source"
                                        ),
                                        "ranker_type": (
                                            "lgbm_utility_ranker_with_ultralight_dirty_clean_tiebreak"
                                        ),
                                        "ranker_status": (
                                            f"utility:{lgbm_ranker_status};"
                                            f"path:{lgbm_path_ranker_status};"
                                            f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                            f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                            f"side_clean:{lgbm_side_positive_clean_path_status}"
                                        ),
                                        "ranker_relevance": (
                                            "utility_quintile_s15_ultralight_dirty_clean"
                                        ),
                                        "path_ranker_min_percentile": float(path_pct_min),
                                        "pred_timeout_cap": float(timeout_cap),
                                        "base_score": "s15_lgbm_ranker_risk_score",
                                        "path_ranker_tiebreak_weight": 0.010,
                                        "clean_path_tiebreak_weight": 0.008,
                                        "dirty_positive_bad_mae_ts_penalty_lambda": 0.012,
                                        "side_dirty_positive_bad_mae_ts_penalty_lambda": 0.008,
                                    },
                                )
                            )
                            s36_micro_score = (
                                lgbm_ranker_risk_score
                                + 0.004 * lgbm_path_pct
                                + 0.003 * lgbm_clean_path_ts_pct
                                + 0.002 * lgbm_side_positive_clean_path_ts_pct
                                - 0.005 * lgbm_dirty_positive_bad_mae_ts_pct
                                - 0.003 * lgbm_side_dirty_positive_bad_mae_ts_pct
                                - 0.002 * lgbm_bad_mae_ts_pct
                            ).astype(np.float32)
                            s7_specs.append(
                                (
                                    (
                                        "s36_lgbm_s15_micro_dirty_clean"
                                        f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                        f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                        "_stageA_rerank"
                                    ),
                                    s36_micro_score,
                                    path_timeout_mask,
                                    {
                                        "s7_ablation": (
                                            "s36_s15_micro_dirty_clean_source"
                                        ),
                                        "ranker_type": (
                                            "lgbm_utility_ranker_with_micro_dirty_clean_tiebreak"
                                        ),
                                        "ranker_status": (
                                            f"utility:{lgbm_ranker_status};"
                                            f"path:{lgbm_path_ranker_status};"
                                            f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                            f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                            f"side_clean:{lgbm_side_positive_clean_path_status}"
                                        ),
                                        "ranker_relevance": (
                                            "utility_quintile_s15_micro_dirty_clean"
                                        ),
                                        "path_ranker_min_percentile": float(path_pct_min),
                                        "pred_timeout_cap": float(timeout_cap),
                                        "base_score": "s15_lgbm_ranker_risk_score",
                                        "path_ranker_tiebreak_weight": 0.004,
                                        "clean_path_tiebreak_weight": 0.003,
                                        "dirty_positive_bad_mae_ts_penalty_lambda": 0.005,
                                        "side_dirty_positive_bad_mae_ts_penalty_lambda": 0.003,
                                    },
                                )
                            )
                            s37_local_score = (
                                lgbm_ranker_risk_score
                                + 0.004 * lgbm_path_pct
                                + 0.003 * lgbm_clean_path_ts_pct
                                + 0.002 * lgbm_side_positive_clean_path_ts_pct
                                + 0.006 * s22_bucket_quality_rank_pct
                                + 0.002 * s22_bucket_relaxed_pass_rank
                                - 0.005 * lgbm_dirty_positive_bad_mae_ts_pct
                                - 0.003 * lgbm_side_dirty_positive_bad_mae_ts_pct
                                - 0.002 * lgbm_bad_mae_ts_pct
                            ).astype(np.float32)
                            s7_specs.append(
                                (
                                    (
                                        "s37_lgbm_s15_micro_local_bucket_dirty_clean"
                                        f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                        f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                        "_stageA_rerank"
                                    ),
                                    s37_local_score,
                                    path_timeout_mask,
                                    {
                                        "s7_ablation": (
                                            "s37_s15_micro_local_bucket_dirty_clean_source"
                                        ),
                                        "ranker_type": (
                                            "lgbm_utility_ranker_with_micro_dirty_clean_local_bucket_tiebreak"
                                        ),
                                        "ranker_status": (
                                            f"utility:{lgbm_ranker_status};"
                                            f"path:{lgbm_path_ranker_status};"
                                            f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                            f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                            f"side_clean:{lgbm_side_positive_clean_path_status};"
                                            f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                        ),
                                        "ranker_relevance": (
                                            "utility_quintile_s15_micro_local_bucket_dirty_clean"
                                        ),
                                        "path_ranker_min_percentile": float(path_pct_min),
                                        "pred_timeout_cap": float(timeout_cap),
                                        "base_score": "s15_lgbm_ranker_risk_score",
                                        "path_ranker_tiebreak_weight": 0.004,
                                        "clean_path_tiebreak_weight": 0.003,
                                        "local_bucket_quality_tiebreak_weight": 0.006,
                                        "local_bucket_relaxed_pass_tiebreak_weight": 0.002,
                                        "dirty_positive_bad_mae_ts_penalty_lambda": 0.005,
                                        "side_dirty_positive_bad_mae_ts_penalty_lambda": 0.003,
                                        **s22_bucket_quality_diag,
                                    },
                                )
                            )
                            for (
                                bucket_q_min,
                                relaxed_min,
                                strict_min,
                                bucket_suffix,
                            ) in (
                                (0.45, 1.0, 0.0, "q45_relaxed01"),
                                (0.50, 1.0, 0.0, "q50_relaxed01"),
                                (0.55, 1.0, 0.0, "q55_relaxed01"),
                                (0.50, 0.0, 1.0, "q50_strict01"),
                            ):
                                s38_bucket_mask = (
                                    path_timeout_mask
                                    & pd.to_numeric(
                                        s22_bucket_quality_rank_pct,
                                        errors="coerce",
                                    )
                                    .ge(float(bucket_q_min))
                                    .fillna(False)
                                    .to_numpy(dtype=bool)
                                )
                                if relaxed_min > 0.0:
                                    s38_bucket_mask = s38_bucket_mask & pd.to_numeric(
                                        s22_bucket_relaxed_pass_count,
                                        errors="coerce",
                                    ).ge(float(relaxed_min)).fillna(False).to_numpy(dtype=bool)
                                if strict_min > 0.0:
                                    s38_bucket_mask = s38_bucket_mask & pd.to_numeric(
                                        s22_bucket_strict_pass_count,
                                        errors="coerce",
                                    ).ge(float(strict_min)).fillna(False).to_numpy(dtype=bool)
                                s7_specs.append(
                                    (
                                        (
                                            "s38_lgbm_s15_micro_local_bucket_abstain"
                                            f"_{bucket_suffix}"
                                            f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                            f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                            "_stageA_rerank"
                                        ),
                                        s37_local_score,
                                        s38_bucket_mask,
                                        {
                                            "s7_ablation": (
                                                "s38_s15_micro_local_bucket_abstention_source"
                                            ),
                                            "ranker_type": (
                                                "lgbm_utility_ranker_with_prior_bucket_abstention"
                                            ),
                                            "ranker_status": (
                                                f"utility:{lgbm_ranker_status};"
                                                f"path:{lgbm_path_ranker_status};"
                                                f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                                f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                f"side_clean:{lgbm_side_positive_clean_path_status};"
                                                f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                            ),
                                            "ranker_relevance": (
                                                "utility_quintile_s15_micro_local_bucket_abstention"
                                            ),
                                            "path_ranker_min_percentile": float(path_pct_min),
                                            "pred_timeout_cap": float(timeout_cap),
                                            "base_score": "s37_lgbm_s15_micro_local_bucket_dirty_clean",
                                            "s22_bucket_quality_rank_pct_min": float(bucket_q_min),
                                            "s22_bucket_relaxed_pass_count_min": float(relaxed_min),
                                            "s22_bucket_strict_pass_count_min": float(strict_min),
                                            "local_bucket_abstention": True,
                                            **s22_bucket_quality_diag,
                                        },
                                    )
                                )
                            side_values_for_bucket = pd.to_numeric(
                                valid_metrics["side"],
                                errors="coerce",
                            ).fillna(1.0).to_numpy(dtype=np.float32)
                            for (
                                bucket_q_min,
                                relaxed_min,
                                strict_min,
                                min_side_candidates,
                                bucket_suffix,
                            ) in (
                                (0.45, 1.0, 0.0, 60, "q45_relaxed01_sidefallback60"),
                                (0.50, 1.0, 0.0, 60, "q50_relaxed01_sidefallback60"),
                                (0.50, 0.0, 1.0, 60, "q50_strict01_sidefallback60"),
                            ):
                                raw_bucket_mask = (
                                    path_timeout_mask
                                    & pd.to_numeric(
                                        s22_bucket_quality_rank_pct,
                                        errors="coerce",
                                    )
                                    .ge(float(bucket_q_min))
                                    .fillna(False)
                                    .to_numpy(dtype=bool)
                                )
                                if relaxed_min > 0.0:
                                    raw_bucket_mask = raw_bucket_mask & pd.to_numeric(
                                        s22_bucket_relaxed_pass_count,
                                        errors="coerce",
                                    ).ge(float(relaxed_min)).fillna(False).to_numpy(dtype=bool)
                                if strict_min > 0.0:
                                    raw_bucket_mask = raw_bucket_mask & pd.to_numeric(
                                        s22_bucket_strict_pass_count,
                                        errors="coerce",
                                    ).ge(float(strict_min)).fillna(False).to_numpy(dtype=bool)
                                s39_bucket_mask = np.zeros(len(valid), dtype=bool)
                                fallback_sides: list[str] = []
                                for side_value, side_name in ((1.0, "long"), (-1.0, "short")):
                                    side_mask = (
                                        side_values_for_bucket > 0.0
                                        if side_value > 0.0
                                        else side_values_for_bucket < 0.0
                                    )
                                    side_bucket_mask = raw_bucket_mask & side_mask
                                    if int(side_bucket_mask.sum()) >= int(min_side_candidates):
                                        s39_bucket_mask |= side_bucket_mask
                                    else:
                                        fallback_sides.append(side_name)
                                        s39_bucket_mask |= path_timeout_mask & side_mask
                                s7_specs.append(
                                    (
                                        (
                                            "s39_lgbm_s15_micro_local_bucket_abstain"
                                            f"_{bucket_suffix}"
                                            f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                            f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                            "_stageA_rerank"
                                        ),
                                        s37_local_score,
                                        s39_bucket_mask,
                                        {
                                            "s7_ablation": (
                                                "s39_s15_micro_local_bucket_abstention_side_fallback_source"
                                            ),
                                            "ranker_type": (
                                                "lgbm_utility_ranker_with_prior_bucket_abstention_side_fallback"
                                            ),
                                            "ranker_status": (
                                                f"utility:{lgbm_ranker_status};"
                                                f"path:{lgbm_path_ranker_status};"
                                                f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                                f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                f"side_clean:{lgbm_side_positive_clean_path_status};"
                                                f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                            ),
                                            "ranker_relevance": (
                                                "utility_quintile_s15_micro_local_bucket_abstention_side_fallback"
                                            ),
                                            "path_ranker_min_percentile": float(path_pct_min),
                                            "pred_timeout_cap": float(timeout_cap),
                                            "base_score": "s37_lgbm_s15_micro_local_bucket_dirty_clean",
                                            "s22_bucket_quality_rank_pct_min": float(bucket_q_min),
                                            "s22_bucket_relaxed_pass_count_min": float(relaxed_min),
                                            "s22_bucket_strict_pass_count_min": float(strict_min),
                                            "s39_min_side_bucket_candidates": int(min_side_candidates),
                                            "s39_bucket_fallback_sides": ",".join(fallback_sides),
                                            "local_bucket_abstention": True,
                                            "local_bucket_side_fallback": True,
                                            **s22_bucket_quality_diag,
                                        },
                                    )
                                )
                                s42_source_score = (
                                    0.62
                                    * pd.to_numeric(
                                        lgbm_s42_side_interaction_dirty_zero_ranker_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.14
                                    * pd.to_numeric(
                                        s22_bucket_quality_rank_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.08
                                    * pd.to_numeric(
                                        lgbm_clean_path_ts_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.08
                                    * pd.to_numeric(
                                        lgbm_side_positive_clean_path_ts_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    - 0.10
                                    * pd.to_numeric(
                                        lgbm_bad_mae_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                    - 0.08
                                    * pd.to_numeric(
                                        lgbm_timeout_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                    - 0.10
                                    * pd.to_numeric(
                                        lgbm_side_dirty_positive_bad_mae_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                ).astype(np.float32)
                                s7_specs.append(
                                    (
                                        (
                                            "s42_lgbm_side_spread_aegmm_dirtyzero_ranker"
                                            f"_{bucket_suffix}"
                                            f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                            f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                            "_stageA_rerank"
                                        ),
                                        s42_source_score,
                                        s39_bucket_mask,
                                        {
                                            "s7_ablation": (
                                                "s42_side_spread_aegmm_dirtyzero_source_ranker"
                                            ),
                                            "ranker_type": (
                                                "side_lgbm_ranker_with_spread_aegmm_interactions"
                                            ),
                                            "ranker_status": (
                                                f"s42:{lgbm_s42_side_interaction_dirty_zero_ranker_status};"
                                                f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                                f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                f"side_clean:{lgbm_side_positive_clean_path_status};"
                                                f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                            ),
                                            "ranker_relevance": (
                                                "s30_side_asymmetric_path_first_dirty_zero"
                                            ),
                                            "path_ranker_min_percentile": float(path_pct_min),
                                            "pred_timeout_cap": float(timeout_cap),
                                            "base_score": (
                                                "s42_side_spread_aegmm_dirtyzero_ranker"
                                            ),
                                            "s22_bucket_quality_rank_pct_min": float(bucket_q_min),
                                            "s22_bucket_relaxed_pass_count_min": float(relaxed_min),
                                            "s22_bucket_strict_pass_count_min": float(strict_min),
                                            "s39_min_side_bucket_candidates": int(
                                                min_side_candidates
                                            ),
                                            "s39_bucket_fallback_sides": ",".join(
                                                fallback_sides
                                            ),
                                            "s42_ranker_weight": 0.62,
                                            "s42_bucket_quality_weight": 0.14,
                                            "s42_clean_path_ts_weight": 0.08,
                                            "s42_side_clean_path_ts_weight": 0.08,
                                            "s42_bad_mae_ts_penalty": 0.10,
                                            "s42_timeout_ts_penalty": 0.08,
                                            "s42_side_dirty_ts_penalty": 0.10,
                                            "local_bucket_abstention": True,
                                            "local_bucket_side_fallback": True,
                                            **s42_interaction_diag,
                                            **s22_bucket_quality_diag,
                                        },
                                    )
                                )
                                s43_inverted_source_score = (
                                    0.62
                                    * (
                                        1.0
                                        - pd.to_numeric(
                                            lgbm_s42_side_interaction_dirty_zero_ranker_pct,
                                            errors="coerce",
                                        ).fillna(0.5)
                                    )
                                    + 0.14
                                    * pd.to_numeric(
                                        s22_bucket_quality_rank_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.08
                                    * pd.to_numeric(
                                        lgbm_clean_path_ts_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.08
                                    * pd.to_numeric(
                                        lgbm_side_positive_clean_path_ts_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    - 0.10
                                    * pd.to_numeric(
                                        lgbm_bad_mae_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                    - 0.08
                                    * pd.to_numeric(
                                        lgbm_timeout_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                    - 0.10
                                    * pd.to_numeric(
                                        lgbm_side_dirty_positive_bad_mae_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                ).astype(np.float32)
                                s7_specs.append(
                                    (
                                        (
                                            "s43_lgbm_side_spread_aegmm_dirtyzero_inverted_ranker"
                                            f"_{bucket_suffix}"
                                            f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                            f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                            "_stageA_rerank"
                                        ),
                                        s43_inverted_source_score,
                                        s39_bucket_mask,
                                        {
                                            "s7_ablation": (
                                                "s43_side_spread_aegmm_dirtyzero_inverted_source_ranker"
                                            ),
                                            "ranker_type": (
                                                "side_lgbm_ranker_with_spread_aegmm_interactions_inverted"
                                            ),
                                            "ranker_status": (
                                                f"s42:{lgbm_s42_side_interaction_dirty_zero_ranker_status};"
                                                f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                                f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                f"side_clean:{lgbm_side_positive_clean_path_status};"
                                                f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                            ),
                                            "ranker_relevance": (
                                                "inverted_s30_side_asymmetric_path_first_dirty_zero"
                                            ),
                                            "path_ranker_min_percentile": float(path_pct_min),
                                            "pred_timeout_cap": float(timeout_cap),
                                            "base_score": (
                                                "s43_side_spread_aegmm_dirtyzero_inverted_ranker"
                                            ),
                                            "s22_bucket_quality_rank_pct_min": float(bucket_q_min),
                                            "s22_bucket_relaxed_pass_count_min": float(relaxed_min),
                                            "s22_bucket_strict_pass_count_min": float(strict_min),
                                            "s39_min_side_bucket_candidates": int(
                                                min_side_candidates
                                            ),
                                            "s39_bucket_fallback_sides": ",".join(
                                                fallback_sides
                                            ),
                                            "s43_inverted_s42_ranker": True,
                                            "s43_ranker_weight": 0.62,
                                            "s43_bucket_quality_weight": 0.14,
                                            "s43_clean_path_ts_weight": 0.08,
                                            "s43_side_clean_path_ts_weight": 0.08,
                                            "s43_bad_mae_ts_penalty": 0.10,
                                            "s43_timeout_ts_penalty": 0.08,
                                            "s43_side_dirty_ts_penalty": 0.10,
                                            "local_bucket_abstention": True,
                                            "local_bucket_side_fallback": True,
                                            **s42_interaction_diag,
                                            **s22_bucket_quality_diag,
                                        },
                                    )
                                )
                                s44_sign_calibrated_source_score = (
                                    0.62
                                    * pd.to_numeric(
                                        lgbm_s44_side_interaction_sign_calibrated_ranker_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.14
                                    * pd.to_numeric(
                                        s22_bucket_quality_rank_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.08
                                    * pd.to_numeric(
                                        lgbm_clean_path_ts_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.08
                                    * pd.to_numeric(
                                        lgbm_side_positive_clean_path_ts_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    - 0.10
                                    * pd.to_numeric(
                                        lgbm_bad_mae_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                    - 0.08
                                    * pd.to_numeric(
                                        lgbm_timeout_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                    - 0.10
                                    * pd.to_numeric(
                                        lgbm_side_dirty_positive_bad_mae_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                ).astype(np.float32)
                                s7_specs.append(
                                    (
                                        (
                                            "s44_lgbm_side_spread_aegmm_dirtyzero_signcal_ranker"
                                            f"_{bucket_suffix}"
                                            f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                            f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                            "_stageA_rerank"
                                        ),
                                        s44_sign_calibrated_source_score,
                                        s39_bucket_mask,
                                        {
                                            "s7_ablation": (
                                                "s44_side_spread_aegmm_dirtyzero_sign_calibrated_source_ranker"
                                            ),
                                            "ranker_type": (
                                                "side_lgbm_ranker_with_train_side_sign_calibration"
                                            ),
                                            "ranker_status": (
                                                f"s42:{lgbm_s42_side_interaction_dirty_zero_ranker_status};"
                                                f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                                f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                f"side_clean:{lgbm_side_positive_clean_path_status};"
                                                f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                            ),
                                            "ranker_relevance": (
                                                "train_side_sign_calibrated_s30_side_asymmetric_path_first_dirty_zero"
                                            ),
                                            "path_ranker_min_percentile": float(path_pct_min),
                                            "pred_timeout_cap": float(timeout_cap),
                                            "base_score": (
                                                "s44_side_spread_aegmm_dirtyzero_signcal_ranker"
                                            ),
                                            "s22_bucket_quality_rank_pct_min": float(bucket_q_min),
                                            "s22_bucket_relaxed_pass_count_min": float(relaxed_min),
                                            "s22_bucket_strict_pass_count_min": float(strict_min),
                                            "s39_min_side_bucket_candidates": int(
                                                min_side_candidates
                                            ),
                                            "s39_bucket_fallback_sides": ",".join(
                                                fallback_sides
                                            ),
                                            "s44_ranker_weight": 0.62,
                                            "s44_bucket_quality_weight": 0.14,
                                            "s44_clean_path_ts_weight": 0.08,
                                            "s44_side_clean_path_ts_weight": 0.08,
                                            "s44_bad_mae_ts_penalty": 0.10,
                                            "s44_timeout_ts_penalty": 0.08,
                                            "s44_side_dirty_ts_penalty": 0.10,
                                            "local_bucket_abstention": True,
                                            "local_bucket_side_fallback": True,
                                            **s44_sign_calibration_diag,
                                            **s42_interaction_diag,
                                            **s22_bucket_quality_diag,
                                        },
                                    )
                                )
                                s45_roll45_source_score = (
                                    0.62
                                    * pd.to_numeric(
                                        lgbm_s45_side_interaction_roll45_dirty_zero_ranker_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.14
                                    * pd.to_numeric(
                                        s22_bucket_quality_rank_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.08
                                    * pd.to_numeric(
                                        lgbm_clean_path_ts_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.08
                                    * pd.to_numeric(
                                        lgbm_side_positive_clean_path_ts_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    - 0.10
                                    * pd.to_numeric(
                                        lgbm_bad_mae_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                    - 0.08
                                    * pd.to_numeric(
                                        lgbm_timeout_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                    - 0.10
                                    * pd.to_numeric(
                                        lgbm_side_dirty_positive_bad_mae_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                ).astype(np.float32)
                                s7_specs.append(
                                    (
                                        (
                                            "s45_lgbm_side_spread_aegmm_dirtyzero_roll45_ranker"
                                            f"_{bucket_suffix}"
                                            f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                            f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                            "_stageA_rerank"
                                        ),
                                        s45_roll45_source_score,
                                        s39_bucket_mask,
                                        {
                                            "s7_ablation": (
                                                "s45_side_spread_aegmm_dirtyzero_rolling45_source_ranker"
                                            ),
                                            "ranker_type": (
                                                "side_lgbm_ranker_with_spread_aegmm_interactions_rolling45"
                                            ),
                                            "ranker_status": (
                                                f"s45:{lgbm_s45_side_interaction_roll45_dirty_zero_ranker_status};"
                                                f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                                f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                f"side_clean:{lgbm_side_positive_clean_path_status};"
                                                f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                            ),
                                            "ranker_relevance": (
                                                "rolling45_s30_side_asymmetric_path_first_dirty_zero"
                                            ),
                                            "path_ranker_min_percentile": float(path_pct_min),
                                            "pred_timeout_cap": float(timeout_cap),
                                            "base_score": (
                                                "s45_side_spread_aegmm_dirtyzero_roll45_ranker"
                                            ),
                                            "s22_bucket_quality_rank_pct_min": float(bucket_q_min),
                                            "s22_bucket_relaxed_pass_count_min": float(relaxed_min),
                                            "s22_bucket_strict_pass_count_min": float(strict_min),
                                            "s39_min_side_bucket_candidates": int(
                                                min_side_candidates
                                            ),
                                            "s39_bucket_fallback_sides": ",".join(
                                                fallback_sides
                                            ),
                                            "s45_ranker_weight": 0.62,
                                            "s45_bucket_quality_weight": 0.14,
                                            "s45_clean_path_ts_weight": 0.08,
                                            "s45_side_clean_path_ts_weight": 0.08,
                                            "s45_bad_mae_ts_penalty": 0.10,
                                            "s45_timeout_ts_penalty": 0.08,
                                            "s45_side_dirty_ts_penalty": 0.10,
                                            "local_bucket_abstention": True,
                                            "local_bucket_side_fallback": True,
                                            **s45_recent_train_diag,
                                            **s42_interaction_diag,
                                            **s22_bucket_quality_diag,
                                        },
                                    )
                                )
                                s46_local_state_score = (
                                    0.44
                                    * pd.to_numeric(
                                        lgbm_s42_side_interaction_dirty_zero_ranker_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.30
                                    * pd.to_numeric(
                                        s46_bucket_quality_rank_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.08
                                    * pd.to_numeric(
                                        s22_bucket_quality_rank_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.08
                                    * pd.to_numeric(
                                        lgbm_clean_path_ts_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.08
                                    * pd.to_numeric(
                                        lgbm_side_positive_clean_path_ts_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    - 0.09
                                    * pd.to_numeric(
                                        lgbm_bad_mae_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                    - 0.08
                                    * pd.to_numeric(
                                        lgbm_timeout_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                    - 0.09
                                    * pd.to_numeric(
                                        lgbm_side_dirty_positive_bad_mae_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                ).astype(np.float32)
                                s7_specs.append(
                                    (
                                        (
                                            "s46_lgbm_side_spread_aegmm_local_quality_ranker"
                                            f"_{bucket_suffix}"
                                            f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                            f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                            "_stageA_rerank"
                                        ),
                                        s46_local_state_score,
                                        s39_bucket_mask,
                                        {
                                            "s7_ablation": (
                                                "s46_side_spread_aegmm_local_quality_source_ranker"
                                            ),
                                            "ranker_type": (
                                                "s42_ranker_with_train_side_spread_aegmm_bucket_quality"
                                            ),
                                            "ranker_status": (
                                                f"s42:{lgbm_s42_side_interaction_dirty_zero_ranker_status};"
                                                f"s46_bucket:{s46_bucket_quality_diag.get('s46_bucket_quality_status', 'unknown')};"
                                                f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                                f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                f"side_clean:{lgbm_side_positive_clean_path_status};"
                                                f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                            ),
                                            "ranker_relevance": (
                                                "s42_dirty_zero_ranker_plus_train_side_spread_aegmm_bucket_quality"
                                            ),
                                            "path_ranker_min_percentile": float(path_pct_min),
                                            "pred_timeout_cap": float(timeout_cap),
                                            "base_score": (
                                                "s46_side_spread_aegmm_local_quality_ranker"
                                            ),
                                            "s22_bucket_quality_rank_pct_min": float(bucket_q_min),
                                            "s22_bucket_relaxed_pass_count_min": float(relaxed_min),
                                            "s22_bucket_strict_pass_count_min": float(strict_min),
                                            "s39_min_side_bucket_candidates": int(
                                                min_side_candidates
                                            ),
                                            "s39_bucket_fallback_sides": ",".join(
                                                fallback_sides
                                            ),
                                            "s46_ranker_weight": 0.44,
                                            "s46_local_quality_weight": 0.30,
                                            "s46_univariate_bucket_weight": 0.08,
                                            "s46_clean_path_ts_weight": 0.08,
                                            "s46_side_clean_path_ts_weight": 0.08,
                                            "s46_bad_mae_ts_penalty": 0.09,
                                            "s46_timeout_ts_penalty": 0.08,
                                            "s46_side_dirty_ts_penalty": 0.09,
                                            "local_bucket_abstention": True,
                                            "local_bucket_side_fallback": True,
                                            **s46_bucket_quality_diag,
                                            **s42_interaction_diag,
                                            **s22_bucket_quality_diag,
                                        },
                                    )
                                )
                                s47_s15_local_quality_score = (
                                    0.82
                                    * pd.to_numeric(
                                        s37_local_score,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.12
                                    * pd.to_numeric(
                                        s46_bucket_quality_rank_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.03
                                    * pd.to_numeric(
                                        lgbm_clean_path_ts_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    + 0.03
                                    * pd.to_numeric(
                                        lgbm_side_positive_clean_path_ts_pct,
                                        errors="coerce",
                                    ).fillna(0.0)
                                    - 0.04
                                    * pd.to_numeric(
                                        lgbm_bad_mae_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                    - 0.03
                                    * pd.to_numeric(
                                        lgbm_side_dirty_positive_bad_mae_ts_pct,
                                        errors="coerce",
                                    ).fillna(1.0)
                                ).astype(np.float32)
                                s7_specs.append(
                                    (
                                        (
                                            "s47_lgbm_s15_local_quality_tiebreak"
                                            f"_{bucket_suffix}"
                                            f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                            f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                            "_stageA_rerank"
                                        ),
                                        s47_s15_local_quality_score,
                                        s39_bucket_mask,
                                        {
                                            "s7_ablation": (
                                                "s47_s15_local_quality_tiebreak_source"
                                            ),
                                            "ranker_type": (
                                                "s15_micro_local_score_with_train_side_spread_aegmm_quality_tiebreak"
                                            ),
                                            "ranker_status": (
                                                f"utility:{lgbm_ranker_status};"
                                                f"path:{lgbm_path_ranker_status};"
                                                f"s46_bucket:{s46_bucket_quality_diag.get('s46_bucket_quality_status', 'unknown')};"
                                                f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                                f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                f"side_clean:{lgbm_side_positive_clean_path_status};"
                                                f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                            ),
                                            "ranker_relevance": (
                                                "s15_micro_score_plus_train_side_spread_aegmm_bucket_quality"
                                            ),
                                            "path_ranker_min_percentile": float(path_pct_min),
                                            "pred_timeout_cap": float(timeout_cap),
                                            "base_score": "s37_lgbm_s15_micro_local_bucket_dirty_clean",
                                            "s22_bucket_quality_rank_pct_min": float(bucket_q_min),
                                            "s22_bucket_relaxed_pass_count_min": float(relaxed_min),
                                            "s22_bucket_strict_pass_count_min": float(strict_min),
                                            "s39_min_side_bucket_candidates": int(
                                                min_side_candidates
                                            ),
                                            "s39_bucket_fallback_sides": ",".join(
                                                fallback_sides
                                            ),
                                            "s47_s15_local_score_weight": 0.82,
                                            "s47_local_quality_weight": 0.12,
                                            "s47_clean_path_ts_weight": 0.03,
                                            "s47_side_clean_path_ts_weight": 0.03,
                                            "s47_bad_mae_ts_penalty": 0.04,
                                            "s47_side_dirty_ts_penalty": 0.03,
                                            "local_bucket_abstention": True,
                                            "local_bucket_side_fallback": True,
                                            **s46_bucket_quality_diag,
                                            **s22_bucket_quality_diag,
                                        },
                                    )
                                )
                                for (
                                    long_timeout_ts_cap,
                                    short_timeout_ts_cap,
                                    long_lgbm_timeout_cap,
                                    short_lgbm_timeout_cap,
                                    timeout_min_side_candidates,
                                    timeout_suffix,
                                ) in (
                                    (
                                        0.75,
                                        0.98,
                                        0.18,
                                        0.30,
                                        60,
                                        "long_ts75_lgbm18_short_ts98_lgbm30",
                                    ),
                                    (
                                        0.70,
                                        0.95,
                                        0.16,
                                        0.28,
                                        60,
                                        "long_ts70_lgbm16_short_ts95_lgbm28",
                                    ),
                                    (
                                        0.65,
                                        0.90,
                                        0.14,
                                        0.24,
                                        60,
                                        "long_ts65_lgbm14_short_ts90_lgbm24",
                                    ),
                                ):
                                    s40_timeout_mask = np.zeros(len(valid), dtype=bool)
                                    s40_bucket_fallback_sides: list[str] = []
                                    s40_timeout_fallback_sides: list[str] = []
                                    for side_value, side_name in ((1.0, "long"), (-1.0, "short")):
                                        side_mask = (
                                            side_values_for_bucket > 0.0
                                            if side_value > 0.0
                                            else side_values_for_bucket < 0.0
                                        )
                                        side_bucket_mask = raw_bucket_mask & side_mask
                                        if int(side_bucket_mask.sum()) >= int(min_side_candidates):
                                            side_source_mask = side_bucket_mask
                                        else:
                                            s40_bucket_fallback_sides.append(side_name)
                                            side_source_mask = path_timeout_mask & side_mask

                                        timeout_ts_cap = (
                                            float(long_timeout_ts_cap)
                                            if side_value > 0.0
                                            else float(short_timeout_ts_cap)
                                        )
                                        lgbm_timeout_cap = (
                                            float(long_lgbm_timeout_cap)
                                            if side_value > 0.0
                                            else float(short_lgbm_timeout_cap)
                                        )
                                        side_timeout_mask = (
                                            side_source_mask
                                            & pd.to_numeric(
                                                lgbm_timeout_ts_pct,
                                                errors="coerce",
                                            )
                                            .le(timeout_ts_cap)
                                            .fillna(False)
                                            .to_numpy(dtype=bool)
                                            & pd.to_numeric(
                                                lgbm_timeout_pred,
                                                errors="coerce",
                                            )
                                            .le(lgbm_timeout_cap)
                                            .fillna(False)
                                            .to_numpy(dtype=bool)
                                        )
                                        if int(side_timeout_mask.sum()) >= int(
                                            timeout_min_side_candidates
                                        ):
                                            s40_timeout_mask |= side_timeout_mask
                                        else:
                                            s40_timeout_fallback_sides.append(side_name)
                                            s40_timeout_mask |= side_source_mask

                                    s40_timeout_score = (
                                        s37_local_score
                                        - 0.004 * lgbm_timeout_ts_pct
                                        - 0.002 * lgbm_timeout_pred
                                    ).astype(np.float32)
                                    s7_specs.append(
                                        (
                                            (
                                                "s40_lgbm_s15_micro_local_bucket_timeout_abstain"
                                                f"_{bucket_suffix}"
                                                f"_{timeout_suffix}"
                                                f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                                f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                                "_stageA_rerank"
                                            ),
                                            s40_timeout_score,
                                            s40_timeout_mask,
                                            {
                                                "s7_ablation": (
                                                    "s40_s15_micro_local_bucket_timeout_abstention_side_fallback_source"
                                                ),
                                                "ranker_type": (
                                                    "lgbm_utility_ranker_with_prior_bucket_and_timeout_abstention_side_fallback"
                                                ),
                                                "ranker_status": (
                                                    f"utility:{lgbm_ranker_status};"
                                                    f"path:{lgbm_path_ranker_status};"
                                                    f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                                    f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                    f"side_clean:{lgbm_side_positive_clean_path_status};"
                                                    f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                                ),
                                                "ranker_relevance": (
                                                    "utility_quintile_s15_micro_local_bucket_timeout_abstention_side_fallback"
                                                ),
                                                "path_ranker_min_percentile": float(path_pct_min),
                                                "pred_timeout_cap": float(timeout_cap),
                                                "base_score": (
                                                    "s37_lgbm_s15_micro_local_bucket_dirty_clean"
                                                ),
                                                "s22_bucket_quality_rank_pct_min": float(
                                                    bucket_q_min
                                                ),
                                                "s22_bucket_relaxed_pass_count_min": float(
                                                    relaxed_min
                                                ),
                                                "s22_bucket_strict_pass_count_min": float(
                                                    strict_min
                                                ),
                                                "s39_min_side_bucket_candidates": int(
                                                    min_side_candidates
                                                ),
                                                "s40_min_side_timeout_candidates": int(
                                                    timeout_min_side_candidates
                                                ),
                                                "s40_long_timeout_ts_pct_cap": float(
                                                    long_timeout_ts_cap
                                                ),
                                                "s40_short_timeout_ts_pct_cap": float(
                                                    short_timeout_ts_cap
                                                ),
                                                "s40_long_lgbm_timeout_pred_cap": float(
                                                    long_lgbm_timeout_cap
                                                ),
                                                "s40_short_lgbm_timeout_pred_cap": float(
                                                    short_lgbm_timeout_cap
                                                ),
                                                "s40_bucket_fallback_sides": ",".join(
                                                    s40_bucket_fallback_sides
                                                ),
                                                "s40_timeout_fallback_sides": ",".join(
                                                    s40_timeout_fallback_sides
                                                ),
                                                "local_bucket_abstention": True,
                                                "local_bucket_side_fallback": True,
                                                "local_timeout_abstention": True,
                                                "local_timeout_side_fallback": True,
                                                **s22_bucket_quality_diag,
                                            },
                                        )
                                    )
                                    for (
                                        dirty_ts_cap,
                                        bad_ts_cap,
                                        clean_ts_min,
                                        clean_min_side_candidates,
                                        clean_suffix,
                                    ) in (
                                        (0.75, 0.70, 0.20, 40, "dirty75_bad70_clean20_min40"),
                                        (0.65, 0.65, 0.25, 40, "dirty65_bad65_clean25_min40"),
                                        (0.55, 0.60, 0.30, 40, "dirty55_bad60_clean30_min40"),
                                    ):
                                        s41_clean_mask = np.zeros(len(valid), dtype=bool)
                                        s41_bucket_fallback_sides: list[str] = []
                                        s41_timeout_fallback_sides: list[str] = []
                                        s41_clean_fallback_sides: list[str] = []
                                        for side_value, side_name in ((1.0, "long"), (-1.0, "short")):
                                            side_mask = (
                                                side_values_for_bucket > 0.0
                                                if side_value > 0.0
                                                else side_values_for_bucket < 0.0
                                            )
                                            side_bucket_mask = raw_bucket_mask & side_mask
                                            if int(side_bucket_mask.sum()) >= int(min_side_candidates):
                                                side_source_mask = side_bucket_mask
                                            else:
                                                s41_bucket_fallback_sides.append(side_name)
                                                side_source_mask = path_timeout_mask & side_mask

                                            timeout_ts_cap = (
                                                float(long_timeout_ts_cap)
                                                if side_value > 0.0
                                                else float(short_timeout_ts_cap)
                                            )
                                            lgbm_timeout_cap = (
                                                float(long_lgbm_timeout_cap)
                                                if side_value > 0.0
                                                else float(short_lgbm_timeout_cap)
                                            )
                                            side_timeout_mask = (
                                                side_source_mask
                                                & pd.to_numeric(
                                                    lgbm_timeout_ts_pct,
                                                    errors="coerce",
                                                )
                                                .le(timeout_ts_cap)
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                                & pd.to_numeric(
                                                    lgbm_timeout_pred,
                                                    errors="coerce",
                                                )
                                                .le(lgbm_timeout_cap)
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                            )
                                            if int(side_timeout_mask.sum()) < int(
                                                timeout_min_side_candidates
                                            ):
                                                s41_timeout_fallback_sides.append(side_name)
                                                side_timeout_mask = side_source_mask

                                            side_path_clean_mask = (
                                                side_timeout_mask
                                                & pd.to_numeric(
                                                    lgbm_side_dirty_positive_bad_mae_ts_pct,
                                                    errors="coerce",
                                                )
                                                .le(float(dirty_ts_cap))
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                                & pd.to_numeric(
                                                    lgbm_bad_mae_ts_pct,
                                                    errors="coerce",
                                                )
                                                .le(float(bad_ts_cap))
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                                & pd.to_numeric(
                                                    lgbm_clean_path_ts_pct,
                                                    errors="coerce",
                                                )
                                                .ge(float(clean_ts_min))
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                            )
                                            if int(side_path_clean_mask.sum()) >= int(
                                                clean_min_side_candidates
                                            ):
                                                s41_clean_mask |= side_path_clean_mask
                                            else:
                                                s41_clean_fallback_sides.append(side_name)
                                                s41_clean_mask |= side_timeout_mask

                                        s41_clean_score = (
                                            s40_timeout_score
                                            + 0.003 * lgbm_clean_path_ts_pct
                                            - 0.004 * lgbm_side_dirty_positive_bad_mae_ts_pct
                                            - 0.003 * lgbm_bad_mae_ts_pct
                                        ).astype(np.float32)
                                        s7_specs.append(
                                            (
                                                (
                                                    "s41_lgbm_s15_micro_local_bucket_timeout_dirty_abstain"
                                                    f"_{bucket_suffix}"
                                                    f"_{timeout_suffix}"
                                                    f"_{clean_suffix}"
                                                    f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                                    f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                                    "_stageA_rerank"
                                                ),
                                                s41_clean_score,
                                                s41_clean_mask,
                                                {
                                                    "s7_ablation": (
                                                        "s41_s15_micro_local_bucket_timeout_dirty_abstention_side_fallback_source"
                                                    ),
                                                    "ranker_type": (
                                                        "lgbm_utility_ranker_with_prior_bucket_timeout_and_dirty_abstention_side_fallback"
                                                    ),
                                                    "ranker_status": (
                                                        f"utility:{lgbm_ranker_status};"
                                                        f"path:{lgbm_path_ranker_status};"
                                                        f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                                        f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                        f"side_clean:{lgbm_side_positive_clean_path_status};"
                                                        f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                                    ),
                                                    "ranker_relevance": (
                                                        "utility_quintile_s15_micro_local_bucket_timeout_dirty_abstention_side_fallback"
                                                    ),
                                                    "path_ranker_min_percentile": float(
                                                        path_pct_min
                                                    ),
                                                    "pred_timeout_cap": float(timeout_cap),
                                                    "base_score": (
                                                        "s40_lgbm_s15_micro_local_bucket_timeout_abstain"
                                                    ),
                                                    "s22_bucket_quality_rank_pct_min": float(
                                                        bucket_q_min
                                                    ),
                                                    "s22_bucket_relaxed_pass_count_min": float(
                                                        relaxed_min
                                                    ),
                                                    "s22_bucket_strict_pass_count_min": float(
                                                        strict_min
                                                    ),
                                                    "s39_min_side_bucket_candidates": int(
                                                        min_side_candidates
                                                    ),
                                                    "s40_min_side_timeout_candidates": int(
                                                        timeout_min_side_candidates
                                                    ),
                                                    "s41_min_side_clean_candidates": int(
                                                        clean_min_side_candidates
                                                    ),
                                                    "s40_long_timeout_ts_pct_cap": float(
                                                        long_timeout_ts_cap
                                                    ),
                                                    "s40_short_timeout_ts_pct_cap": float(
                                                        short_timeout_ts_cap
                                                    ),
                                                    "s40_long_lgbm_timeout_pred_cap": float(
                                                        long_lgbm_timeout_cap
                                                    ),
                                                    "s40_short_lgbm_timeout_pred_cap": float(
                                                        short_lgbm_timeout_cap
                                                    ),
                                                    "s41_side_dirty_positive_ts_pct_cap": float(
                                                        dirty_ts_cap
                                                    ),
                                                    "s41_bad_mae_ts_pct_cap": float(bad_ts_cap),
                                                    "s41_clean_path_ts_pct_min": float(
                                                        clean_ts_min
                                                    ),
                                                    "s41_bucket_fallback_sides": ",".join(
                                                        s41_bucket_fallback_sides
                                                    ),
                                                    "s41_timeout_fallback_sides": ",".join(
                                                        s41_timeout_fallback_sides
                                                    ),
                                                    "s41_clean_fallback_sides": ",".join(
                                                        s41_clean_fallback_sides
                                                    ),
                                                    "local_bucket_abstention": True,
                                                    "local_bucket_side_fallback": True,
                                                    "local_timeout_abstention": True,
                                                    "local_timeout_side_fallback": True,
                                                    "local_dirty_abstention": True,
                                                    "local_dirty_side_fallback": True,
                                                    **s22_bucket_quality_diag,
                                                },
                                            )
                                        )
                                    for (
                                        long_clean_ts_min,
                                        short_clean_ts_min,
                                        long_dirty_ts_cap,
                                        short_dirty_ts_cap,
                                        long_bad_ts_cap,
                                        short_bad_ts_cap,
                                        long_timeout_pred_cap,
                                        short_timeout_pred_cap,
                                        clean_suffix,
                                    ) in (
                                        (
                                            0.65,
                                            0.60,
                                            0.45,
                                            0.48,
                                            0.55,
                                            0.55,
                                            0.16,
                                            0.24,
                                            "balanced_clean65_60_dirty45_48_bad55",
                                        ),
                                        (
                                            0.70,
                                            0.65,
                                            0.40,
                                            0.45,
                                            0.50,
                                            0.50,
                                            0.14,
                                            0.22,
                                            "strict_clean70_65_dirty40_45_bad50",
                                        ),
                                        (
                                            0.62,
                                            0.68,
                                            0.48,
                                            0.40,
                                            0.55,
                                            0.48,
                                            0.16,
                                            0.22,
                                            "short_strict_clean62_68_dirty48_40_bad55_48",
                                        ),
                                    ):
                                        s48_mask = np.zeros(len(valid), dtype=bool)
                                        for side_value in (1.0, -1.0):
                                            side_mask = (
                                                side_values_for_bucket > 0.0
                                                if side_value > 0.0
                                                else side_values_for_bucket < 0.0
                                            )
                                            clean_ts_min = (
                                                float(long_clean_ts_min)
                                                if side_value > 0.0
                                                else float(short_clean_ts_min)
                                            )
                                            dirty_ts_cap_local = (
                                                float(long_dirty_ts_cap)
                                                if side_value > 0.0
                                                else float(short_dirty_ts_cap)
                                            )
                                            bad_ts_cap_local = (
                                                float(long_bad_ts_cap)
                                                if side_value > 0.0
                                                else float(short_bad_ts_cap)
                                            )
                                            timeout_pred_cap_local = (
                                                float(long_timeout_pred_cap)
                                                if side_value > 0.0
                                                else float(short_timeout_pred_cap)
                                            )
                                            timeout_ts_cap_local = (
                                                float(long_timeout_ts_cap)
                                                if side_value > 0.0
                                                else float(short_timeout_ts_cap)
                                            )
                                            s48_mask |= (
                                                raw_bucket_mask
                                                & side_mask
                                                & pd.to_numeric(
                                                    lgbm_timeout_ts_pct,
                                                    errors="coerce",
                                                )
                                                .le(timeout_ts_cap_local)
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                                & pd.to_numeric(
                                                    lgbm_timeout_pred,
                                                    errors="coerce",
                                                )
                                                .le(timeout_pred_cap_local)
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                                & pd.to_numeric(
                                                    lgbm_side_positive_clean_path_ts_pct,
                                                    errors="coerce",
                                                )
                                                .ge(clean_ts_min)
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                                & pd.to_numeric(
                                                    lgbm_side_dirty_positive_bad_mae_ts_pct,
                                                    errors="coerce",
                                                )
                                                .le(dirty_ts_cap_local)
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                                & pd.to_numeric(
                                                    lgbm_bad_mae_ts_pct,
                                                    errors="coerce",
                                                )
                                                .le(bad_ts_cap_local)
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                            )

                                        s48_score = (
                                            s40_timeout_score
                                            + 0.010 * lgbm_side_positive_clean_path_ts_pct
                                            + 0.006 * lgbm_clean_path_ts_pct
                                            - 0.014 * lgbm_side_dirty_positive_bad_mae_ts_pct
                                            - 0.010 * lgbm_bad_mae_ts_pct
                                            - 0.004 * lgbm_timeout_ts_pct
                                        ).astype(np.float32)
                                        s7_specs.append(
                                            (
                                                (
                                                    "s48_lgbm_s15_topk_clean_veto_nofallback"
                                                    f"_{bucket_suffix}"
                                                    f"_{timeout_suffix}"
                                                    f"_{clean_suffix}"
                                                    f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                                    f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                                    "_stageA_rerank"
                                                ),
                                                s48_score,
                                                s48_mask,
                                                {
                                                    "s7_ablation": (
                                                        "s48_topk_clean_path_veto_no_side_fallback"
                                                    ),
                                                    "ranker_type": (
                                                        "s40_timeout_source_with_hard_topk_clean_veto"
                                                    ),
                                                    "ranker_status": (
                                                        f"utility:{lgbm_ranker_status};"
                                                        f"path:{lgbm_path_ranker_status};"
                                                        f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                        f"side_clean:{lgbm_side_positive_clean_path_status};"
                                                        f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                                    ),
                                                    "ranker_relevance": (
                                                        "utility_quintile_s15_topk_clean_veto"
                                                    ),
                                                    "path_ranker_min_percentile": float(
                                                        path_pct_min
                                                    ),
                                                    "pred_timeout_cap": float(timeout_cap),
                                                    "base_score": (
                                                        "s40_lgbm_s15_micro_local_bucket_timeout_abstain"
                                                    ),
                                                    "s22_bucket_quality_rank_pct_min": float(
                                                        bucket_q_min
                                                    ),
                                                    "s22_bucket_relaxed_pass_count_min": float(
                                                        relaxed_min
                                                    ),
                                                    "s22_bucket_strict_pass_count_min": float(
                                                        strict_min
                                                    ),
                                                    "s48_no_bucket_side_fallback": True,
                                                    "s48_no_timeout_side_fallback": True,
                                                    "s48_no_clean_side_fallback": True,
                                                    "s48_long_positive_clean_ts_pct_min": float(
                                                        long_clean_ts_min
                                                    ),
                                                    "s48_short_positive_clean_ts_pct_min": float(
                                                        short_clean_ts_min
                                                    ),
                                                    "s48_long_dirty_positive_ts_pct_max": float(
                                                        long_dirty_ts_cap
                                                    ),
                                                    "s48_short_dirty_positive_ts_pct_max": float(
                                                        short_dirty_ts_cap
                                                    ),
                                                    "s48_long_bad_mae_ts_pct_max": float(
                                                        long_bad_ts_cap
                                                    ),
                                                    "s48_short_bad_mae_ts_pct_max": float(
                                                        short_bad_ts_cap
                                                    ),
                                                    "s48_long_timeout_pred_cap": float(
                                                        long_timeout_pred_cap
                                                    ),
                                                    "s48_short_timeout_pred_cap": float(
                                                        short_timeout_pred_cap
                                                    ),
                                                    "local_bucket_abstention": True,
                                                    "local_timeout_abstention": True,
                                                    "local_dirty_abstention": True,
                                                    "local_clean_path_veto": True,
                                                    **s22_bucket_quality_diag,
                                                },
                                            )
                                        )
                                        s49_mask = np.zeros(len(valid), dtype=bool)
                                        s49_bucket_fallback_sides: list[str] = []
                                        s49_timeout_fallback_sides: list[str] = []
                                        for side_value, side_name in ((1.0, "long"), (-1.0, "short")):
                                            side_mask = (
                                                side_values_for_bucket > 0.0
                                                if side_value > 0.0
                                                else side_values_for_bucket < 0.0
                                            )
                                            side_bucket_mask = raw_bucket_mask & side_mask
                                            if int(side_bucket_mask.sum()) >= int(min_side_candidates):
                                                side_source_mask = side_bucket_mask
                                            else:
                                                s49_bucket_fallback_sides.append(side_name)
                                                side_source_mask = path_timeout_mask & side_mask

                                            clean_ts_min = (
                                                float(long_clean_ts_min)
                                                if side_value > 0.0
                                                else float(short_clean_ts_min)
                                            )
                                            dirty_ts_cap_local = (
                                                float(long_dirty_ts_cap)
                                                if side_value > 0.0
                                                else float(short_dirty_ts_cap)
                                            )
                                            bad_ts_cap_local = (
                                                float(long_bad_ts_cap)
                                                if side_value > 0.0
                                                else float(short_bad_ts_cap)
                                            )
                                            timeout_pred_cap_local = (
                                                float(long_timeout_pred_cap)
                                                if side_value > 0.0
                                                else float(short_timeout_pred_cap)
                                            )
                                            timeout_ts_cap_local = (
                                                float(long_timeout_ts_cap)
                                                if side_value > 0.0
                                                else float(short_timeout_ts_cap)
                                            )
                                            side_timeout_mask = (
                                                side_source_mask
                                                & pd.to_numeric(
                                                    lgbm_timeout_ts_pct,
                                                    errors="coerce",
                                                )
                                                .le(timeout_ts_cap_local)
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                                & pd.to_numeric(
                                                    lgbm_timeout_pred,
                                                    errors="coerce",
                                                )
                                                .le(timeout_pred_cap_local)
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                            )
                                            if int(side_timeout_mask.sum()) < int(
                                                timeout_min_side_candidates
                                            ):
                                                s49_timeout_fallback_sides.append(side_name)
                                                side_timeout_mask = side_source_mask

                                            s49_mask |= (
                                                side_timeout_mask
                                                & pd.to_numeric(
                                                    lgbm_side_positive_clean_path_ts_pct,
                                                    errors="coerce",
                                                )
                                                .ge(clean_ts_min)
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                                & pd.to_numeric(
                                                    lgbm_side_dirty_positive_bad_mae_ts_pct,
                                                    errors="coerce",
                                                )
                                                .le(dirty_ts_cap_local)
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                                & pd.to_numeric(
                                                    lgbm_bad_mae_ts_pct,
                                                    errors="coerce",
                                                )
                                                .le(bad_ts_cap_local)
                                                .fillna(False)
                                                .to_numpy(dtype=bool)
                                            )

                                        s7_specs.append(
                                            (
                                                (
                                                    "s49_lgbm_s15_topk_clean_veto_no_clean_fallback"
                                                    f"_{bucket_suffix}"
                                                    f"_{timeout_suffix}"
                                                    f"_{clean_suffix}"
                                                    f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                                    f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                                    "_stageA_rerank"
                                                ),
                                                s48_score,
                                                s49_mask,
                                                {
                                                    "s7_ablation": (
                                                        "s49_topk_clean_path_veto_bucket_timeout_side_fallback_only"
                                                    ),
                                                    "ranker_type": (
                                                        "s40_timeout_source_with_hard_topk_clean_veto_no_clean_fallback"
                                                    ),
                                                    "ranker_status": (
                                                        f"utility:{lgbm_ranker_status};"
                                                        f"path:{lgbm_path_ranker_status};"
                                                        f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                        f"side_clean:{lgbm_side_positive_clean_path_status};"
                                                        f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                                    ),
                                                    "ranker_relevance": (
                                                        "utility_quintile_s15_topk_clean_veto_no_clean_fallback"
                                                    ),
                                                    "path_ranker_min_percentile": float(
                                                        path_pct_min
                                                    ),
                                                    "pred_timeout_cap": float(timeout_cap),
                                                    "base_score": (
                                                        "s40_lgbm_s15_micro_local_bucket_timeout_abstain"
                                                    ),
                                                    "s22_bucket_quality_rank_pct_min": float(
                                                        bucket_q_min
                                                    ),
                                                    "s22_bucket_relaxed_pass_count_min": float(
                                                        relaxed_min
                                                    ),
                                                    "s22_bucket_strict_pass_count_min": float(
                                                        strict_min
                                                    ),
                                                    "s49_bucket_fallback_sides": ",".join(
                                                        s49_bucket_fallback_sides
                                                    ),
                                                    "s49_timeout_fallback_sides": ",".join(
                                                        s49_timeout_fallback_sides
                                                    ),
                                                    "s49_clean_fallback_sides": "",
                                                    "s49_no_clean_side_fallback": True,
                                                    "s49_long_positive_clean_ts_pct_min": float(
                                                        long_clean_ts_min
                                                    ),
                                                    "s49_short_positive_clean_ts_pct_min": float(
                                                        short_clean_ts_min
                                                    ),
                                                    "s49_long_dirty_positive_ts_pct_max": float(
                                                        long_dirty_ts_cap
                                                    ),
                                                    "s49_short_dirty_positive_ts_pct_max": float(
                                                        short_dirty_ts_cap
                                                    ),
                                                    "s49_long_bad_mae_ts_pct_max": float(
                                                        long_bad_ts_cap
                                                    ),
                                                    "s49_short_bad_mae_ts_pct_max": float(
                                                        short_bad_ts_cap
                                                    ),
                                                    "s49_long_timeout_pred_cap": float(
                                                        long_timeout_pred_cap
                                                    ),
                                                    "s49_short_timeout_pred_cap": float(
                                                        short_timeout_pred_cap
                                                    ),
                                                    "local_bucket_abstention": True,
                                                    "local_bucket_side_fallback": True,
                                                    "local_timeout_abstention": True,
                                                    "local_timeout_side_fallback": True,
                                                    "local_dirty_abstention": True,
                                                    "local_clean_path_veto": True,
                                                    **s22_bucket_quality_diag,
                                                },
                                            )
                                        )
                                    s50_scores: list[tuple[str, pd.Series, dict[str, Any]]] = [
                                        (
                                            "clean_rerank",
                                            (
                                                s40_timeout_score
                                                + 0.012 * lgbm_side_positive_clean_path_ts_pct
                                                + 0.006 * lgbm_clean_path_ts_pct
                                                - 0.002 * lgbm_timeout_ts_pct
                                            ).astype(np.float32),
                                            {
                                                "s50_utility_score_weight": 1.0,
                                                "s50_side_clean_path_ts_weight": 0.012,
                                                "s50_clean_path_ts_weight": 0.006,
                                                "s50_timeout_ts_penalty": 0.002,
                                            },
                                        ),
                                        (
                                            "utility_clean_blend",
                                            (
                                                s40_timeout_score
                                                + 0.010 * lgbm_side_positive_clean_path_ts_pct
                                                + 0.006 * lgbm_clean_path_ts_pct
                                                - 0.008 * lgbm_bad_mae_ts_pct
                                                - 0.004 * lgbm_timeout_ts_pct
                                            ).astype(np.float32),
                                            {
                                                "s50_utility_score_weight": 1.0,
                                                "s50_side_clean_path_ts_weight": 0.010,
                                                "s50_clean_path_ts_weight": 0.006,
                                                "s50_bad_mae_ts_penalty": 0.008,
                                                "s50_timeout_ts_penalty": 0.004,
                                            },
                                        ),
                                        (
                                            "utility_clean_dirty_blend",
                                            (
                                                s40_timeout_score
                                                + 0.012 * lgbm_side_positive_clean_path_ts_pct
                                                + 0.006 * lgbm_clean_path_ts_pct
                                                - 0.014 * lgbm_side_dirty_positive_bad_mae_ts_pct
                                                - 0.010 * lgbm_bad_mae_ts_pct
                                                - 0.004 * lgbm_timeout_ts_pct
                                            ).astype(np.float32),
                                            {
                                                "s50_utility_score_weight": 1.0,
                                                "s50_side_clean_path_ts_weight": 0.012,
                                                "s50_clean_path_ts_weight": 0.006,
                                                "s50_side_dirty_positive_ts_penalty": 0.014,
                                                "s50_bad_mae_ts_penalty": 0.010,
                                                "s50_timeout_ts_penalty": 0.004,
                                            },
                                        ),
                                    ]
                                    for s50_score_suffix, s50_score, s50_score_diag in s50_scores:
                                        s50_name = (
                                            "s50_lgbm_budgeted_clean_allocator"
                                            f"_{bucket_suffix}"
                                            f"_{timeout_suffix}"
                                            f"_{s50_score_suffix}"
                                            f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                            f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                            "_stageA_rerank"
                                        )
                                        s50_full_name = (
                                            f"{s50_name}_side_cap_{int(round(side_cap_max_share * 100)):02d}"
                                        )
                                        if not (
                                            candidate_ledger_only
                                            and s50_full_name not in requested_ledger_selectors
                                        ):
                                            selected_idx, budget_diag = _constrained_top_indices(
                                                score=s50_score,
                                                side=valid_metrics["side"],
                                                eligible=pd.Series(
                                                    s40_timeout_mask,
                                                    index=valid.index,
                                                ),
                                                top_frac=float(top_frac),
                                                max_side_share=float(side_cap_max_share),
                                            )
                                            final_mask = _mask_from_indices(len(valid), selected_idx)
                                            final_diag = _oracle_recall_stats(
                                                metrics=valid_metrics,
                                                mask=final_mask,
                                                top_frac=float(top_frac),
                                                prefix="final",
                                            )
                                            hard_cap_score = _score_from_selected_indices(
                                                base_score=s50_score,
                                                selected_idx=selected_idx,
                                            )
                                            variants.append(
                                                (
                                                    s50_full_name,
                                                    hard_cap_score,
                                                    {
                                                        **s7_stage_diag,
                                                        "s7_ablation": (
                                                            "s50_budgeted_clean_allocator_opportunity_pool_rerank"
                                                        ),
                                                        "ranker_type": (
                                                            "s40_timeout_source_clean_path_allocator"
                                                        ),
                                                        "ranker_status": (
                                                            f"utility:{lgbm_ranker_status};"
                                                            f"path:{lgbm_path_ranker_status};"
                                                            f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                            f"side_clean:{lgbm_side_positive_clean_path_status};"
                                                            f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                                        ),
                                                        "ranker_relevance": (
                                                            "utility_plus_topk_clean_path_allocator"
                                                        ),
                                                        "path_ranker_min_percentile": float(
                                                            path_pct_min
                                                        ),
                                                        "pred_timeout_cap": float(timeout_cap),
                                                        "base_score": (
                                                            "s40_lgbm_s15_micro_local_bucket_timeout_abstain"
                                                        ),
                                                        "s50_allocator_stage": "rerank_only",
                                                        "s50_candidate_source": "s40_timeout_mask",
                                                        "s50_budgeted_allocator": False,
                                                        **s50_score_diag,
                                                        **budget_diag,
                                                        **final_diag,
                                                        **s22_bucket_quality_diag,
                                                    },
                                                    selected_idx,
                                                )
                                            )
                                        for s50_bad_budget in (0.54, 0.53, 0.52, 0.50):
                                            for s50_timeout_budget in (0.12, 0.10):
                                                for s50_budget_mode in ("global", "side"):
                                                    s50_name = (
                                                        "s50_lgbm_budgeted_clean_allocator"
                                                        f"_{bucket_suffix}"
                                                        f"_{timeout_suffix}"
                                                        f"_{s50_score_suffix}"
                                                        f"_{s50_budget_mode}"
                                                        f"_bad_budget_{int(round(s50_bad_budget * 100)):02d}"
                                                        f"_timeout_budget_{int(round(s50_timeout_budget * 100)):02d}"
                                                        f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                                        f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                                        "_stageA_rerank"
                                                    )
                                                    s50_full_name = (
                                                        f"{s50_name}_side_cap_{int(round(side_cap_max_share * 100)):02d}"
                                                    )
                                                    if (
                                                        candidate_ledger_only
                                                        and s50_full_name
                                                        not in requested_ledger_selectors
                                                    ):
                                                        continue
                                                    selected_idx, budget_diag = _budgeted_top_indices(
                                                        score=s50_score,
                                                        side=valid_metrics["side"],
                                                        eligible=pd.Series(
                                                            s40_timeout_mask,
                                                            index=valid.index,
                                                        ),
                                                        bad_risk=side_bad_mae_pred,
                                                        timeout_risk=lgbm_timeout_pred,
                                                        top_frac=float(top_frac),
                                                        max_side_share=float(side_cap_max_share),
                                                        bad_risk_budget=float(s50_bad_budget),
                                                        timeout_risk_budget=float(
                                                            s50_timeout_budget
                                                        ),
                                                        budget_mode=s50_budget_mode,
                                                        min_fill_ratio=0.90,
                                                    )
                                                    final_mask = _mask_from_indices(
                                                        len(valid),
                                                        selected_idx,
                                                    )
                                                    final_diag = _oracle_recall_stats(
                                                        metrics=valid_metrics,
                                                        mask=final_mask,
                                                        top_frac=float(top_frac),
                                                        prefix="final",
                                                    )
                                                    hard_cap_score = _score_from_selected_indices(
                                                        base_score=s50_score,
                                                        selected_idx=selected_idx,
                                                    )
                                                    variants.append(
                                                        (
                                                            s50_full_name,
                                                            hard_cap_score,
                                                            {
                                                                **s7_stage_diag,
                                                                "s7_ablation": (
                                                                    "s50_budgeted_clean_allocator_opportunity_pool"
                                                                ),
                                                                "ranker_type": (
                                                                    "s40_timeout_source_budgeted_clean_path_allocator"
                                                                ),
                                                                "ranker_status": (
                                                                    f"utility:{lgbm_ranker_status};"
                                                                    f"path:{lgbm_path_ranker_status};"
                                                                    f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                                    f"side_clean:{lgbm_side_positive_clean_path_status};"
                                                                    f"bucket:{s22_bucket_quality_diag.get('s22_bucket_quality_status', 'unknown')}"
                                                                ),
                                                                "ranker_relevance": (
                                                                    "utility_plus_topk_clean_path_with_predicted_risk_budget"
                                                                ),
                                                                "path_ranker_min_percentile": float(
                                                                    path_pct_min
                                                                ),
                                                                "pred_timeout_cap": float(
                                                                    timeout_cap
                                                                ),
                                                                "base_score": (
                                                                    "s40_lgbm_s15_micro_local_bucket_timeout_abstain"
                                                                ),
                                                                "s50_allocator_stage": (
                                                                    "budgeted_risk_allocation"
                                                                ),
                                                                "s50_candidate_source": (
                                                                    "s40_timeout_mask"
                                                                ),
                                                                "s50_budgeted_allocator": True,
                                                                "s50_bad_risk_source": (
                                                                    "side_bad_mae_pred"
                                                                ),
                                                                "s50_timeout_risk_source": (
                                                                    "lgbm_timeout_pred"
                                                                ),
                                                                **s50_score_diag,
                                                                **budget_diag,
                                                                **final_diag,
                                                                **s22_bucket_quality_diag,
                                                            },
                                                            selected_idx,
                                                        )
                                                    )
                            for dirty_ts_cap in (0.90, 0.85):
                                s34_dirty_mask = (
                                    path_timeout_mask
                                    & pd.to_numeric(
                                        lgbm_dirty_positive_bad_mae_ts_pct,
                                        errors="coerce",
                                    )
                                    .le(float(dirty_ts_cap))
                                    .fillna(False)
                                    .to_numpy(dtype=bool)
                                )
                                s7_specs.append(
                                    (
                                        (
                                            "s34_lgbm_opportunity_preserving_dirty_clean"
                                            f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                            f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                            f"_dirty_ts_cap_{int(round(dirty_ts_cap * 100)):02d}"
                                            "_stageA_rerank"
                                        ),
                                        s34_opportunity_clean_score,
                                        s34_dirty_mask,
                                        {
                                            "s7_ablation": (
                                                "s34_opportunity_preserving_dirty_clean_source_cap"
                                            ),
                                            "ranker_type": (
                                                "lgbm_utility_ranker_with_light_dirty_clean_tiebreak"
                                            ),
                                            "ranker_status": (
                                                f"utility:{lgbm_ranker_status};"
                                                f"path:{lgbm_path_ranker_status};"
                                                f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                                f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                f"side_clean:{lgbm_side_positive_clean_path_status}"
                                            ),
                                            "ranker_relevance": (
                                                "utility_quintile_opportunity_preserving_dirty_clean"
                                            ),
                                            "path_ranker_min_percentile": float(path_pct_min),
                                            "pred_timeout_cap": float(timeout_cap),
                                            "dirty_positive_bad_mae_ts_pct_max": float(
                                                dirty_ts_cap
                                            ),
                                            "base_score": "s15_lgbm_ranker_risk_score",
                                            "path_ranker_tiebreak_weight": 0.04,
                                            "dirty_positive_bad_mae_ts_penalty_lambda": 0.04,
                                            "side_dirty_positive_bad_mae_ts_penalty_lambda": 0.03,
                                        },
                                    )
                                )
                            s33_dirty_clean_score = (
                                0.58 * lgbm_ranker_risk_score
                                + 0.18 * lgbm_path_pct
                                + 0.14 * lgbm_side_positive_clean_path_ts_pct
                                + 0.08 * lgbm_clean_path_ts_pct
                                - 0.30 * lgbm_bad_mae_ts_pct
                                - 0.28 * lgbm_dirty_positive_bad_mae_ts_pct
                                - 0.18 * lgbm_side_dirty_positive_bad_mae_ts_pct
                            ).astype(np.float32)
                            s7_specs.append(
                                (
                                    (
                                        "s33_lgbm_utility_path_dirty_clean"
                                        f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                        f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                        "_stageA_rerank"
                                    ),
                                    s33_dirty_clean_score,
                                    path_timeout_mask,
                                    {
                                        "s7_ablation": (
                                            "s33_lgbm_utility_path_dirty_clean_source"
                                        ),
                                        "ranker_type": "lgbm_lambdarank_dirty_clean_blend",
                                        "ranker_status": (
                                            f"utility:{lgbm_ranker_status};"
                                            f"path:{lgbm_path_ranker_status};"
                                            f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                            f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                            f"side_clean:{lgbm_side_positive_clean_path_status}"
                                        ),
                                        "ranker_relevance": "utility_quintile_dirty_clean_penalty",
                                        "path_ranker_min_percentile": float(path_pct_min),
                                        "pred_timeout_cap": float(timeout_cap),
                                        "dirty_positive_bad_mae_ts_penalty_lambda": 0.28,
                                        "side_dirty_positive_bad_mae_ts_penalty_lambda": 0.18,
                                    },
                                )
                            )
                            for bad_cap in (0.58, 0.54, 0.50):
                                s33_bad_mask = (
                                    path_timeout_mask
                                    & pd.to_numeric(lgbm_bad_mae_pred, errors="coerce")
                                    .le(float(bad_cap))
                                    .fillna(False)
                                    .to_numpy(dtype=bool)
                                )
                                s7_specs.append(
                                    (
                                        (
                                            "s33_lgbm_utility_path_dirty_clean"
                                            f"_path_pct_min_{int(round(path_pct_min * 100)):02d}"
                                            f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                            f"_bad_cap_{int(round(bad_cap * 100)):02d}"
                                            "_stageA_rerank"
                                        ),
                                        s33_dirty_clean_score,
                                        s33_bad_mask,
                                        {
                                            "s7_ablation": (
                                                "s33_lgbm_utility_path_dirty_clean_source_bad_cap"
                                            ),
                                            "ranker_type": (
                                                "lgbm_lambdarank_dirty_clean_blend"
                                            ),
                                            "ranker_status": (
                                                f"utility:{lgbm_ranker_status};"
                                                f"path:{lgbm_path_ranker_status};"
                                                f"dirty:{lgbm_dirty_positive_bad_mae_status};"
                                                f"side_dirty:{lgbm_side_dirty_positive_bad_mae_status};"
                                                f"side_clean:{lgbm_side_positive_clean_path_status}"
                                            ),
                                            "ranker_relevance": (
                                                "utility_quintile_dirty_clean_penalty"
                                            ),
                                            "path_ranker_min_percentile": float(path_pct_min),
                                            "pred_timeout_cap": float(timeout_cap),
                                            "pred_bad_mae_cap": float(bad_cap),
                                            "dirty_positive_bad_mae_ts_penalty_lambda": 0.28,
                                            "side_dirty_positive_bad_mae_ts_penalty_lambda": 0.18,
                                        },
                                    )
                                )
                    for (
                        s16_name,
                        s16_score,
                        ranker_relevance,
                        risk_lambda_bad,
                        risk_lambda_timeout,
                    ) in (
                        (
                            "s16_lgbm_utility_lgbm_risk_stageA_rerank",
                            lgbm_utility_lgbm_risk_score,
                            "utility_quintile",
                            0.55,
                            0.18,
                        ),
                        (
                            "s16_lgbm_utility_blended_risk_stageA_rerank",
                            lgbm_utility_blended_risk_score,
                            "utility_quintile",
                            0.65,
                            0.22,
                        ),
                        (
                            "s16_lgbm_path_lgbm_risk_stageA_rerank",
                            lgbm_path_lgbm_risk_score,
                            "path_quality",
                            0.60,
                            0.20,
                        ),
                    ):
                        base_s16_diag = {
                            "s7_ablation": "lgbm_ranker_lgbm_path_risk_head",
                            "ranker_type": "lgbm_lambdarank",
                            "ranker_relevance": ranker_relevance,
                            "ranker_status": (
                                lgbm_path_ranker_status
                                if ranker_relevance == "path_quality"
                                else lgbm_ranker_status
                            ),
                            "risk_head": "lgbm_binary_path_risk",
                            "lgbm_bad_mae_status": lgbm_bad_mae_status,
                            "lgbm_timeout_status": lgbm_timeout_status,
                            "lgbm_bad_mae_penalty_lambda": float(risk_lambda_bad),
                            "lgbm_timeout_penalty_lambda": float(risk_lambda_timeout),
                            "pred_bad_mae_cap": 0.70,
                            "pred_timeout_cap": 0.30,
                        }
                        s7_specs.append(
                            (
                                s16_name,
                                s16_score,
                                stage_a_candidate_mask,
                                base_s16_diag,
                            )
                        )
                        for final_frac in (0.015, 0.020):
                            s7_specs.append(
                                (
                                    s16_name.replace(
                                        "_stageA_",
                                        f"_stageA_final_frac_{int(round(final_frac * 1000)):03d}_",
                                    ),
                                    s16_score,
                                    stage_a_candidate_mask,
                                    {**base_s16_diag, "selection_top_frac": float(final_frac)},
                                )
                            )
                        for bad_cap, timeout_cap in (
                            (0.60, 0.20),
                            (0.55, 0.18),
                            (0.50, 0.15),
                            (0.45, 0.15),
                            (0.42, 0.15),
                            (0.40, 0.15),
                            (0.38, 0.15),
                        ):
                            lgbm_risk_mask = (
                                stage_a_candidate_mask
                                & pd.to_numeric(lgbm_bad_mae_pred, errors="coerce")
                                .le(float(bad_cap))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                                & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                                .le(float(timeout_cap))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                            )
                            s7_specs.append(
                                (
                                    s16_name.replace(
                                        "_stageA_",
                                        (
                                            f"_lgbm_bad_cap_{int(round(bad_cap * 100)):02d}"
                                            f"_timeout_cap_{int(round(timeout_cap * 100)):02d}"
                                            "_stageA_"
                                        ),
                                    ),
                                    s16_score,
                                    lgbm_risk_mask,
                                    {
                                        **base_s16_diag,
                                        "lgbm_pred_bad_mae_cap": float(bad_cap),
                                        "lgbm_pred_timeout_cap": float(timeout_cap),
                                    },
                                )
                            )
                    for (
                        s17_name,
                        s17_score,
                        ranker_relevance,
                        clean_weight,
                    ) in (
                        (
                            "s17_lgbm_clean_surplus_ts_gate_stageA_rerank",
                            lgbm_clean_surplus_score,
                            "utility_path_clean_surplus",
                            0.45,
                        ),
                        (
                            "s17_lgbm_clean_path_utility_ts_gate_stageA_rerank",
                            lgbm_clean_path_utility_score,
                            "utility_clean_path_surplus",
                            0.50,
                        ),
                        (
                            "s17_lgbm_path_clean_surplus_ts_gate_stageA_rerank",
                            lgbm_path_clean_surplus_score,
                            "path_clean_surplus",
                            0.50,
                        ),
                    ):
                        base_s17_diag = {
                            "s7_ablation": "timestamp_local_lgbm_clean_path_risk_gate",
                            "ranker_type": "lgbm_lambdarank_timestamp_percentile_blend",
                            "ranker_relevance": ranker_relevance,
                            "ranker_status": (
                                f"utility:{lgbm_ranker_status};"
                                f"path:{lgbm_path_ranker_status}"
                            ),
                            "risk_head": "lgbm_binary_path_risk_and_clean_path",
                            "lgbm_bad_mae_status": lgbm_bad_mae_status,
                            "lgbm_timeout_status": lgbm_timeout_status,
                            "lgbm_clean_path_status": lgbm_clean_path_status,
                            "timestamp_percentile_gate": True,
                            "clean_path_score_weight": float(clean_weight),
                        }
                        for (
                            clean_ts_min,
                            bad_ts_max,
                            timeout_ts_max,
                            raw_bad_cap,
                            raw_timeout_cap,
                        ) in (
                            (0.50, 0.65, 0.70, 0.70, 0.30),
                            (0.55, 0.60, 0.65, 0.55, 0.18),
                            (0.60, 0.55, 0.60, 0.50, 0.15),
                            (0.65, 0.50, 0.60, 0.45, 0.15),
                        ):
                            s17_mask = (
                                stage_a_candidate_pre_risk_mask
                                & pd.to_numeric(lgbm_clean_path_ts_pct, errors="coerce")
                                .ge(float(clean_ts_min))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                                & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                                .le(float(bad_ts_max))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                                & pd.to_numeric(lgbm_timeout_ts_pct, errors="coerce")
                                .le(float(timeout_ts_max))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                                & pd.to_numeric(lgbm_bad_mae_pred, errors="coerce")
                                .le(float(raw_bad_cap))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                                & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                                .le(float(raw_timeout_cap))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                            )
                            gate_name = (
                                f"_clean_ts_min_{int(round(clean_ts_min * 100)):02d}"
                                f"_bad_ts_max_{int(round(bad_ts_max * 100)):02d}"
                                f"_timeout_ts_max_{int(round(timeout_ts_max * 100)):02d}"
                                f"_raw_bad_cap_{int(round(raw_bad_cap * 100)):02d}"
                                f"_raw_timeout_cap_{int(round(raw_timeout_cap * 100)):02d}"
                            )
                            s7_specs.append(
                                (
                                    s17_name.replace("_stageA_", f"{gate_name}_stageA_"),
                                    s17_score,
                                    s17_mask,
                                    {
                                        **base_s17_diag,
                                        "lgbm_clean_path_ts_pct_min": float(clean_ts_min),
                                        "lgbm_bad_mae_ts_pct_max": float(bad_ts_max),
                                        "lgbm_timeout_ts_pct_max": float(timeout_ts_max),
                                        "lgbm_pred_bad_mae_cap": float(raw_bad_cap),
                                        "lgbm_pred_timeout_cap": float(raw_timeout_cap),
                                    },
                                )
                            )
                            for final_frac in (0.015, 0.020):
                                s7_specs.append(
                                    (
                                        s17_name.replace(
                                            "_stageA_",
                                            (
                                                f"{gate_name}"
                                                f"_final_frac_{int(round(final_frac * 1000)):03d}"
                                                "_stageA_"
                                            ),
                                        ),
                                        s17_score,
                                        s17_mask,
                                        {
                                            **base_s17_diag,
                                            "lgbm_clean_path_ts_pct_min": float(clean_ts_min),
                                            "lgbm_bad_mae_ts_pct_max": float(bad_ts_max),
                                            "lgbm_timeout_ts_pct_max": float(timeout_ts_max),
                                            "lgbm_pred_bad_mae_cap": float(raw_bad_cap),
                                            "lgbm_pred_timeout_cap": float(raw_timeout_cap),
                                            "selection_top_frac": float(final_frac),
                                        },
                                    )
                                )
                    for (
                        s18_name,
                        s18_score,
                        ranker_relevance,
                        dirty_lambda,
                    ) in (
                        (
                            "s18_lgbm_dirty_positive_aware_ts_gate_stageA_rerank",
                            lgbm_dirty_positive_aware_score,
                            "utility_path_dirty_positive_aware",
                            0.60,
                        ),
                        (
                            "s18_lgbm_path_dirty_positive_aware_ts_gate_stageA_rerank",
                            lgbm_path_dirty_positive_aware_score,
                            "path_dirty_positive_aware",
                            0.65,
                        ),
                    ):
                        base_s18_diag = {
                            "s7_ablation": "timestamp_local_lgbm_dirty_positive_risk_gate",
                            "ranker_type": "lgbm_lambdarank_timestamp_percentile_blend",
                            "ranker_relevance": ranker_relevance,
                            "ranker_status": (
                                f"utility:{lgbm_ranker_status};"
                                f"path:{lgbm_path_ranker_status}"
                            ),
                            "risk_head": "lgbm_conditional_dirty_positive_bad_mae",
                            "lgbm_dirty_positive_bad_mae_status": (
                                lgbm_dirty_positive_bad_mae_status
                            ),
                            "lgbm_timeout_status": lgbm_timeout_status,
                            "lgbm_clean_path_status": lgbm_clean_path_status,
                            "timestamp_percentile_gate": True,
                            "dirty_positive_bad_mae_ts_penalty_lambda": float(dirty_lambda),
                        }
                        for (
                            clean_ts_min,
                            dirty_ts_max,
                            timeout_ts_max,
                            raw_dirty_cap,
                            raw_timeout_cap,
                        ) in (
                            (0.50, 0.55, 0.70, 0.60, 0.30),
                            (0.55, 0.50, 0.65, 0.55, 0.20),
                            (0.60, 0.45, 0.60, 0.50, 0.15),
                        ):
                            s18_mask = (
                                stage_a_candidate_pre_risk_mask
                                & pd.to_numeric(lgbm_clean_path_ts_pct, errors="coerce")
                                .ge(float(clean_ts_min))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                                & pd.to_numeric(
                                    lgbm_dirty_positive_bad_mae_ts_pct,
                                    errors="coerce",
                                )
                                .le(float(dirty_ts_max))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                                & pd.to_numeric(lgbm_timeout_ts_pct, errors="coerce")
                                .le(float(timeout_ts_max))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                                & pd.to_numeric(
                                    lgbm_dirty_positive_bad_mae_pred,
                                    errors="coerce",
                                )
                                .le(float(raw_dirty_cap))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                                & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                                .le(float(raw_timeout_cap))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                            )
                            gate_name = (
                                f"_clean_ts_min_{int(round(clean_ts_min * 100)):02d}"
                                f"_dirty_ts_max_{int(round(dirty_ts_max * 100)):02d}"
                                f"_timeout_ts_max_{int(round(timeout_ts_max * 100)):02d}"
                                f"_raw_dirty_cap_{int(round(raw_dirty_cap * 100)):02d}"
                                f"_raw_timeout_cap_{int(round(raw_timeout_cap * 100)):02d}"
                            )
                            s7_specs.append(
                                (
                                    s18_name.replace("_stageA_", f"{gate_name}_stageA_"),
                                    s18_score,
                                    s18_mask,
                                    {
                                        **base_s18_diag,
                                        "lgbm_clean_path_ts_pct_min": float(clean_ts_min),
                                        "lgbm_dirty_positive_bad_mae_ts_pct_max": float(
                                            dirty_ts_max
                                        ),
                                        "lgbm_timeout_ts_pct_max": float(timeout_ts_max),
                                        "lgbm_dirty_positive_bad_mae_cap": float(raw_dirty_cap),
                                        "lgbm_pred_timeout_cap": float(raw_timeout_cap),
                                    },
                                )
                            )
                            for final_frac in (0.015, 0.020):
                                s7_specs.append(
                                    (
                                        s18_name.replace(
                                            "_stageA_",
                                            (
                                                f"{gate_name}"
                                                f"_final_frac_{int(round(final_frac * 1000)):03d}"
                                                "_stageA_"
                                            ),
                                        ),
                                        s18_score,
                                        s18_mask,
                                        {
                                            **base_s18_diag,
                                            "lgbm_clean_path_ts_pct_min": float(clean_ts_min),
                                            "lgbm_dirty_positive_bad_mae_ts_pct_max": float(
                                                dirty_ts_max
                                            ),
                                            "lgbm_timeout_ts_pct_max": float(timeout_ts_max),
                                            "lgbm_dirty_positive_bad_mae_cap": float(
                                                raw_dirty_cap
                                            ),
                                            "lgbm_pred_timeout_cap": float(raw_timeout_cap),
                                            "selection_top_frac": float(final_frac),
                                        },
                                    )
                                )
                    for (
                        clean_ts_min,
                        dirty_ts_max,
                        bad_ts_max,
                        timeout_ts_max,
                        raw_dirty_cap,
                        raw_bad_cap,
                        raw_timeout_cap,
                    ) in (
                        (0.55, 0.45, 0.55, 0.55, 0.50, 0.55, 0.18),
                        (0.60, 0.40, 0.50, 0.50, 0.45, 0.50, 0.15),
                        (0.65, 0.35, 0.45, 0.45, 0.40, 0.47, 0.12),
                    ):
                        s20_mask = (
                            stage_a_candidate_pre_risk_mask
                            & pd.to_numeric(lgbm_clean_path_ts_pct, errors="coerce")
                            .ge(float(clean_ts_min))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_dirty_positive_bad_mae_ts_pct,
                                errors="coerce",
                            )
                            .le(float(dirty_ts_max))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(float(bad_ts_max))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_ts_pct, errors="coerce")
                            .le(float(timeout_ts_max))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_dirty_positive_bad_mae_pred,
                                errors="coerce",
                            )
                            .le(float(raw_dirty_cap))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_pred, errors="coerce")
                            .le(float(raw_bad_cap))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                            .le(float(raw_timeout_cap))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        gate_name = (
                            f"_clean_ts_min_{int(round(clean_ts_min * 100)):02d}"
                            f"_dirty_ts_max_{int(round(dirty_ts_max * 100)):02d}"
                            f"_bad_ts_max_{int(round(bad_ts_max * 100)):02d}"
                            f"_timeout_ts_max_{int(round(timeout_ts_max * 100)):02d}"
                            f"_raw_dirty_cap_{int(round(raw_dirty_cap * 100)):02d}"
                            f"_raw_bad_cap_{int(round(raw_bad_cap * 100)):02d}"
                            f"_raw_timeout_cap_{int(round(raw_timeout_cap * 100)):02d}"
                        )
                        base_s20_diag = {
                            "s7_ablation": "executable_path_clean_source_gate",
                            "ranker_type": "lgbm_lambdarank_timestamp_percentile_blend",
                            "ranker_relevance": "path_exec_clean",
                            "ranker_status": (
                                f"utility:{lgbm_ranker_status};"
                                f"path:{lgbm_path_ranker_status}"
                            ),
                            "risk_head": "lgbm_exec_clean_bad_dirty_timeout",
                            "lgbm_clean_path_status": lgbm_clean_path_status,
                            "lgbm_dirty_positive_bad_mae_status": (
                                lgbm_dirty_positive_bad_mae_status
                            ),
                            "lgbm_bad_mae_status": lgbm_bad_mae_status,
                            "lgbm_timeout_status": lgbm_timeout_status,
                            "timestamp_percentile_gate": True,
                            "lgbm_clean_path_ts_pct_min": float(clean_ts_min),
                            "lgbm_dirty_positive_bad_mae_ts_pct_max": float(dirty_ts_max),
                            "lgbm_bad_mae_ts_pct_max": float(bad_ts_max),
                            "lgbm_timeout_ts_pct_max": float(timeout_ts_max),
                            "lgbm_dirty_positive_bad_mae_cap": float(raw_dirty_cap),
                            "lgbm_pred_bad_mae_cap": float(raw_bad_cap),
                            "lgbm_pred_timeout_cap": float(raw_timeout_cap),
                            "clean_dirty_positive_risk_enabled": True,
                            **clean_dirty_positive_diag,
                        }
                        for s20_name, s20_score in (
                            (
                                "s20_lgbm_exec_clean_ts_gate_stageA_rerank",
                                lgbm_exec_clean_score,
                            ),
                            (
                                "s20_lgbm_exec_clean_strict_ts_gate_stageA_rerank",
                                lgbm_exec_clean_strict_score,
                            ),
                            (
                                "s20_lgbm_exec_clean_contrast_ts_gate_stageA_rerank",
                                lgbm_exec_clean_contrast_score,
                            ),
                            (
                                "s20_lgbm_exec_clean_strict_contrast_ts_gate_stageA_rerank",
                                lgbm_exec_clean_strict_contrast_score,
                            ),
                        ):
                            s7_specs.append(
                                (
                                    s20_name.replace("_stageA_", f"{gate_name}_stageA_"),
                                    s20_score,
                                    s20_mask,
                                    base_s20_diag,
                                )
                            )
                            for final_frac in (0.015, 0.020, 0.030):
                                s7_specs.append(
                                    (
                                        s20_name.replace(
                                            "_stageA_",
                                            (
                                                f"{gate_name}"
                                                f"_final_frac_{int(round(final_frac * 1000)):03d}"
                                                "_stageA_"
                                            ),
                                        ),
                                        s20_score,
                                        s20_mask,
                                        {
                                            **base_s20_diag,
                                            "selection_top_frac": float(final_frac),
                                        },
                                    )
                                )
                            if "contrast" in s20_name:
                                for clean_dirty_cap in (0.45, 0.50, 0.55):
                                    clean_dirty_mask = (
                                        s20_mask
                                        & pd.to_numeric(
                                            clean_dirty_positive_risk,
                                            errors="coerce",
                                        )
                                        .le(float(clean_dirty_cap))
                                        .fillna(False)
                                        .to_numpy(dtype=bool)
                                    )
                                    s7_specs.append(
                                        (
                                            s20_name.replace(
                                                "_stageA_",
                                                (
                                                    f"{gate_name}"
                                                    "_clean_dirty_cap_"
                                                    f"{int(round(clean_dirty_cap * 100)):02d}"
                                                    "_stageA_"
                                                ),
                                            ),
                                            s20_score,
                                            clean_dirty_mask,
                                            {
                                                **base_s20_diag,
                                                "clean_dirty_positive_risk_cap": float(
                                                    clean_dirty_cap
                                                ),
                                            },
                                        )
                                    )
                                    for final_frac in (0.020, 0.030):
                                        s7_specs.append(
                                            (
                                                s20_name.replace(
                                                    "_stageA_",
                                                    (
                                                        f"{gate_name}"
                                                        "_clean_dirty_cap_"
                                                        f"{int(round(clean_dirty_cap * 100)):02d}"
                                                        f"_final_frac_{int(round(final_frac * 1000)):03d}"
                                                        "_stageA_"
                                                    ),
                                                ),
                                                s20_score,
                                                clean_dirty_mask,
                                                {
                                                    **base_s20_diag,
                                                    "clean_dirty_positive_risk_cap": float(
                                                        clean_dirty_cap
                                                    ),
                                                    "selection_top_frac": float(final_frac),
                                                },
                                            )
                                        )
                    for (
                        clean_ts_min,
                        raw_clean_min,
                        dirty_ts_max,
                        bad_ts_max,
                        timeout_ts_max,
                        raw_dirty_cap,
                        raw_bad_cap,
                        raw_timeout_cap,
                    ) in (
                        (0.55, 0.25, 0.45, 0.55, 0.55, 0.50, 0.55, 0.18),
                        (0.60, 0.30, 0.40, 0.50, 0.50, 0.45, 0.50, 0.15),
                        (0.65, 0.35, 0.35, 0.45, 0.45, 0.40, 0.47, 0.12),
                    ):
                        s21_mask = (
                            stage_a_candidate_pre_risk_mask
                            & pd.to_numeric(
                                lgbm_positive_clean_path_ts_pct,
                                errors="coerce",
                            )
                            .ge(float(clean_ts_min))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_positive_clean_path_pred,
                                errors="coerce",
                            )
                            .ge(float(raw_clean_min))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_dirty_positive_bad_mae_ts_pct,
                                errors="coerce",
                            )
                            .le(float(dirty_ts_max))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(float(bad_ts_max))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_ts_pct, errors="coerce")
                            .le(float(timeout_ts_max))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_dirty_positive_bad_mae_pred,
                                errors="coerce",
                            )
                            .le(float(raw_dirty_cap))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_pred, errors="coerce")
                            .le(float(raw_bad_cap))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                            .le(float(raw_timeout_cap))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        gate_name = (
                            f"_pos_clean_ts_min_{int(round(clean_ts_min * 100)):02d}"
                            f"_raw_clean_min_{int(round(raw_clean_min * 100)):02d}"
                            f"_dirty_ts_max_{int(round(dirty_ts_max * 100)):02d}"
                            f"_bad_ts_max_{int(round(bad_ts_max * 100)):02d}"
                            f"_timeout_ts_max_{int(round(timeout_ts_max * 100)):02d}"
                            f"_raw_dirty_cap_{int(round(raw_dirty_cap * 100)):02d}"
                            f"_raw_bad_cap_{int(round(raw_bad_cap * 100)):02d}"
                            f"_raw_timeout_cap_{int(round(raw_timeout_cap * 100)):02d}"
                        )
                        base_s21_diag = {
                            "s7_ablation": "supervised_positive_clean_source_gate",
                            "ranker_type": "lgbm_lambdarank_timestamp_percentile_blend",
                            "ranker_relevance": "positive_clean_exec",
                            "ranker_status": (
                                f"utility:{lgbm_ranker_status};"
                                f"path:{lgbm_path_ranker_status}"
                            ),
                            "risk_head": "lgbm_positive_clean_vs_dirty",
                            "lgbm_positive_clean_path_status": (
                                lgbm_positive_clean_path_status
                            ),
                            "lgbm_dirty_positive_bad_mae_status": (
                                lgbm_dirty_positive_bad_mae_status
                            ),
                            "lgbm_bad_mae_status": lgbm_bad_mae_status,
                            "lgbm_timeout_status": lgbm_timeout_status,
                            "timestamp_percentile_gate": True,
                            "lgbm_positive_clean_path_ts_pct_min": float(clean_ts_min),
                            "lgbm_positive_clean_path_pred_min": float(raw_clean_min),
                            "lgbm_dirty_positive_bad_mae_ts_pct_max": float(dirty_ts_max),
                            "lgbm_bad_mae_ts_pct_max": float(bad_ts_max),
                            "lgbm_timeout_ts_pct_max": float(timeout_ts_max),
                            "lgbm_dirty_positive_bad_mae_cap": float(raw_dirty_cap),
                            "lgbm_pred_bad_mae_cap": float(raw_bad_cap),
                            "lgbm_pred_timeout_cap": float(raw_timeout_cap),
                        }
                        for s21_name, s21_score in (
                            (
                                "s21_lgbm_positive_clean_exec_ts_gate_stageA_rerank",
                                lgbm_positive_clean_exec_score,
                            ),
                            (
                                "s21_lgbm_positive_clean_exec_strict_ts_gate_stageA_rerank",
                                lgbm_positive_clean_exec_strict_score,
                            ),
                        ):
                            s7_specs.append(
                                (
                                    s21_name.replace("_stageA_", f"{gate_name}_stageA_"),
                                    s21_score,
                                    s21_mask,
                                    base_s21_diag,
                                )
                            )
                            for final_frac in (0.020, 0.030):
                                s7_specs.append(
                                    (
                                        s21_name.replace(
                                            "_stageA_",
                                            (
                                                f"{gate_name}"
                                                f"_final_frac_{int(round(final_frac * 1000)):03d}"
                                                "_stageA_"
                                            ),
                                        ),
                                        s21_score,
                                        s21_mask,
                                        {
                                            **base_s21_diag,
                                            "selection_top_frac": float(final_frac),
                                        },
                                    )
                                )
                        for (
                            bucket_name,
                            bucket_score,
                            min_quality_pct,
                            min_relaxed_count,
                            min_strict_count,
                        ) in (
                            (
                                "s22_lgbm_positive_clean_bucket_relaxed_stageA_rerank",
                                lgbm_positive_clean_bucket_score,
                                0.45,
                                2.0,
                                0.0,
                            ),
                            (
                                "s22_lgbm_positive_clean_bucket_consensus_stageA_rerank",
                                lgbm_positive_clean_bucket_score,
                                0.50,
                                2.0,
                                1.0,
                            ),
                            (
                                "s22_lgbm_positive_clean_bucket_strict_stageA_rerank",
                                lgbm_positive_clean_bucket_strict_score,
                                0.55,
                                0.0,
                                1.0,
                            ),
                        ):
                            bucket_mask = (
                                s21_mask
                                & pd.to_numeric(
                                    s22_bucket_quality_rank_pct,
                                    errors="coerce",
                                )
                                .ge(float(min_quality_pct))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                            )
                            if min_relaxed_count > 0.0:
                                bucket_mask = bucket_mask & pd.to_numeric(
                                    s22_bucket_relaxed_pass_count,
                                    errors="coerce",
                                ).ge(float(min_relaxed_count)).fillna(False).to_numpy(dtype=bool)
                            if min_strict_count > 0.0:
                                bucket_mask = bucket_mask & pd.to_numeric(
                                    s22_bucket_strict_pass_count,
                                    errors="coerce",
                                ).ge(float(min_strict_count)).fillna(False).to_numpy(dtype=bool)
                            bucket_gate_name = (
                                f"{gate_name}"
                                f"_bucket_q_min_{int(round(min_quality_pct * 100)):02d}"
                                f"_relaxed_min_{int(round(min_relaxed_count)):02d}"
                                f"_strict_min_{int(round(min_strict_count)):02d}"
                            )
                            base_s22_diag = {
                                **base_s21_diag,
                                **s22_bucket_quality_diag,
                                "s7_ablation": "prior_bucket_quality_abstention",
                                "prior_bucket_quality_overlay": True,
                                "s22_bucket_quality_rank_pct_min": float(min_quality_pct),
                                "s22_bucket_relaxed_pass_count_min": float(min_relaxed_count),
                                "s22_bucket_strict_pass_count_min": float(min_strict_count),
                            }
                            s7_specs.append(
                                (
                                    bucket_name.replace("_stageA_", f"{bucket_gate_name}_stageA_"),
                                    bucket_score,
                                    bucket_mask,
                                    base_s22_diag,
                                )
                            )
                            for final_frac in (0.020, 0.030):
                                s7_specs.append(
                                    (
                                        bucket_name.replace(
                                            "_stageA_",
                                            (
                                                f"{bucket_gate_name}"
                                                f"_final_frac_{int(round(final_frac * 1000)):03d}"
                                                "_stageA_"
                                            ),
                                        ),
                                        bucket_score,
                                        bucket_mask,
                                        {
                                            **base_s22_diag,
                                            "selection_top_frac": float(final_frac),
                                        },
                                    )
                                )
                            bootstrap_bucket_mask = (
                                stage_a_candidate_pre_risk_mask
                                & pd.to_numeric(
                                    s22_bucket_quality_rank_pct,
                                    errors="coerce",
                                )
                                .ge(float(min_quality_pct))
                                .fillna(False)
                                .to_numpy(dtype=bool)
                            )
                            if min_relaxed_count > 0.0:
                                bootstrap_bucket_mask = bootstrap_bucket_mask & pd.to_numeric(
                                    s22_bucket_relaxed_pass_count,
                                    errors="coerce",
                                ).ge(float(min_relaxed_count)).fillna(False).to_numpy(dtype=bool)
                            if min_strict_count > 0.0:
                                bootstrap_bucket_mask = bootstrap_bucket_mask & pd.to_numeric(
                                    s22_bucket_strict_pass_count,
                                    errors="coerce",
                                ).ge(float(min_strict_count)).fillna(False).to_numpy(dtype=bool)
                            bootstrap_diag = {
                                **base_s22_diag,
                                **s22_bucket_quality_diag,
                                "s7_ablation": (
                                    "prior_bucket_quality_abstention_bootstrap"
                                ),
                                "prior_bucket_quality_overlay": True,
                                "s22_bucket_quality_rank_pct_min": float(min_quality_pct),
                                "s22_bucket_relaxed_pass_count_min": float(min_relaxed_count),
                                "s22_bucket_strict_pass_count_min": float(min_strict_count),
                                "s22_bootstrap_overlay": True,
                                "s22_bootstrap_base_mask": "stage_a_candidate_pre_risk",
                            }
                            for final_frac in (0.020, 0.030):
                                s7_specs.append(
                                    (
                                        bucket_name.replace(
                                            "_stageA_",
                                            (
                                                f"{bucket_gate_name}"
                                                "_bootstrap_final_frac_"
                                                f"{int(round(final_frac * 1000)):03d}"
                                                "_stageA_"
                                            ),
                                        ),
                                        bucket_score,
                                        bootstrap_bucket_mask,
                                        {
                                            **bootstrap_diag,
                                            "selection_top_frac": float(final_frac),
                                        },
                                    )
                                )
                    if include_path_first_selectors:
                        base_s19_diag = {
                            "s7_ablation": "path_first_clean_relevance_stageA_rerank",
                            "ranker_type": "lgbm_lambdarank",
                            "ranker_relevance": "path_first_clean",
                            "ranker_status": lgbm_path_first_ranker_status,
                            "stage_a_candidate_pool": "pre_risk_high_recall",
                            "s19_path_first_relevance_target": True,
                        }
                        s7_specs.extend(
                            [
                                (
                                    "s19_path_first_ranker_stageA_rerank",
                                    lgbm_path_first_ranker_pct,
                                    stage_a_candidate_pre_risk_mask,
                                    base_s19_diag,
                                ),
                                (
                                    "s19_path_first_ranker_final_frac_020",
                                    lgbm_path_first_ranker_pct,
                                    stage_a_candidate_pre_risk_mask,
                                    {
                                        **base_s19_diag,
                                        "selection_top_frac": 0.020,
                                    },
                                ),
                                (
                                    "s19_path_first_ranker_strict_dirty_zero",
                                    lgbm_path_first_dirty_zero_ranker_pct,
                                    stage_a_candidate_pre_risk_mask,
                                    {
                                        **base_s19_diag,
                                        "ranker_relevance": (
                                            "path_first_clean_dirty_zero"
                                        ),
                                        "ranker_status": (
                                            lgbm_path_first_dirty_zero_ranker_status
                                        ),
                                        "dirty_positive_relevance": 0,
                                    },
                                ),
                            ]
                        )
                    if include_timeout_aware_selectors:
                        timeout_aware_score = (
                            0.62 * lgbm_timeout_aware_clean_ranker_pct
                            + 0.14 * lgbm_utility_ts_pct
                            + 0.18 * lgbm_clean_path_ts_pct
                            - 0.18 * lgbm_bad_mae_ts_pct
                            - 0.24 * lgbm_timeout_ts_pct
                            - 0.10 * lgbm_timeout_pred
                        ).astype(np.float32)
                        timeout_aware_strict_score = (
                            0.58 * lgbm_timeout_aware_clean_ranker_pct
                            + 0.12 * lgbm_utility_ts_pct
                            + 0.22 * lgbm_clean_path_ts_pct
                            - 0.22 * lgbm_bad_mae_ts_pct
                            - 0.30 * lgbm_timeout_ts_pct
                            - 0.14 * lgbm_timeout_pred
                            - 0.08 * lgbm_bad_mae_pred
                        ).astype(np.float32)
                        clean_oracle_timeout_recall_score = (
                            0.38 * lgbm_clean_oracle_pct
                            + 0.32 * lgbm_timeout_aware_clean_ranker_pct
                            + 0.12 * lgbm_utility_ts_pct
                            + 0.10 * lgbm_clean_path_ts_pct
                            + 0.08 * lgbm_side_positive_clean_path_ts_pct
                            - 0.16 * lgbm_bad_mae_ts_pct
                            - 0.22 * lgbm_timeout_ts_pct
                            - 0.08 * lgbm_dirty_positive_bad_mae_ts_pct
                        ).astype(np.float32)
                        clean_oracle_timeout_recall_strict_score = (
                            0.44 * lgbm_clean_oracle_pct
                            + 0.28 * lgbm_timeout_aware_clean_ranker_pct
                            + 0.10 * lgbm_utility_ts_pct
                            + 0.12 * lgbm_clean_path_ts_pct
                            + 0.08 * lgbm_side_positive_clean_path_ts_pct
                            - 0.22 * lgbm_bad_mae_ts_pct
                            - 0.28 * lgbm_timeout_ts_pct
                            - 0.12 * lgbm_dirty_positive_bad_mae_ts_pct
                        ).astype(np.float32)
                        timeout_aware_mask = (
                            stage_a_candidate_pre_risk_mask
                            & pd.to_numeric(lgbm_timeout_ts_pct, errors="coerce")
                            .le(0.58)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                            .le(0.22)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        timeout_aware_strict_mask = (
                            timeout_aware_mask
                            & pd.to_numeric(lgbm_clean_path_ts_pct, errors="coerce")
                            .ge(0.48)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(0.58)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_ts_pct, errors="coerce")
                            .le(0.48)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                            .le(0.18)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        clean_oracle_timeout_recall_mask = (
                            stage_a_candidate_pre_risk_mask
                            & (
                                pd.to_numeric(lgbm_clean_oracle_pct, errors="coerce")
                                .ge(0.38)
                                .fillna(False)
                                .to_numpy(dtype=bool)
                                | pd.to_numeric(
                                    lgbm_timeout_aware_clean_ranker_pct,
                                    errors="coerce",
                                )
                                .ge(0.45)
                                .fillna(False)
                                .to_numpy(dtype=bool)
                                | pd.to_numeric(
                                    lgbm_positive_clean_path_ts_pct,
                                    errors="coerce",
                                )
                                .ge(0.55)
                                .fillna(False)
                                .to_numpy(dtype=bool)
                            )
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(0.70)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_ts_pct, errors="coerce")
                            .le(0.72)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                            .le(0.28)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        clean_oracle_timeout_recall_strict_mask = (
                            clean_oracle_timeout_recall_mask
                            & pd.to_numeric(lgbm_clean_oracle_pct, errors="coerce")
                            .ge(0.45)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(0.62)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_ts_pct, errors="coerce")
                            .le(0.62)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                            .le(0.24)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        base_s31_diag = {
                            "s7_ablation": "s31_timeout_aware_clean_source_stageA_rerank",
                            "ranker_type": "lgbm_lambdarank",
                            "ranker_relevance": "timeout_aware_clean_source",
                            "ranker_status": lgbm_timeout_aware_clean_ranker_status,
                            "stage_a_candidate_pool": "pre_risk_high_recall",
                            "s31_timeout_aware_clean_source_target": True,
                            "bad_mae_penalty_lambda": 0.18,
                            "timeout_penalty_lambda": 0.34,
                            "lgbm_timeout_ts_pct_max": 0.58,
                            "lgbm_pred_timeout_cap": 0.22,
                        }
                        for s31_name, s31_score, s31_mask, s31_diag in (
                            (
                                "s31_timeout_aware_clean_source_stageA_rerank",
                                timeout_aware_score,
                                timeout_aware_mask,
                                base_s31_diag,
                            ),
                            (
                                "s31_timeout_aware_clean_source_strict_stageA_rerank",
                                timeout_aware_strict_score,
                                timeout_aware_strict_mask,
                                {
                                    **base_s31_diag,
                                    "timeout_aware_strict_gate": True,
                                    "bad_mae_penalty_lambda": 0.30,
                                    "timeout_penalty_lambda": 0.44,
                                    "lgbm_clean_path_ts_pct_min": 0.48,
                                    "lgbm_bad_mae_ts_pct_max": 0.58,
                                    "lgbm_timeout_ts_pct_max": 0.48,
                                    "lgbm_pred_timeout_cap": 0.18,
                                },
                            ),
                        ):
                            s7_specs.append((s31_name, s31_score, s31_mask, s31_diag))
                            for final_frac in (0.015, 0.020, 0.030, 0.050):
                                s7_specs.append(
                                    (
                                        s31_name.replace(
                                            "_stageA_",
                                            (
                                                f"_final_frac_"
                                                f"{int(round(final_frac * 1000)):03d}"
                                                "_stageA_"
                                            ),
                                        ),
                                        s31_score,
                                        s31_mask,
                                        {
                                            **s31_diag,
                                            "selection_top_frac": float(final_frac),
                                        },
                                    )
                                )
                        base_s32_diag = {
                            "s7_ablation": (
                                "s32_clean_oracle_timeout_recall_source_stageA_rerank"
                            ),
                            "ranker_type": "lgbm_lambdarank_blend",
                            "ranker_relevance": (
                                "clean_oracle_plus_timeout_aware_clean_source"
                            ),
                            "ranker_status": (
                                f"clean_oracle:{lgbm_clean_oracle_ranker_status};"
                                f"timeout_aware:{lgbm_timeout_aware_clean_ranker_status}"
                            ),
                            "stage_a_candidate_pool": "pre_risk_high_recall",
                            "s32_clean_oracle_timeout_recall_source": True,
                            "clean_oracle_ranker_weight": 0.38,
                            "timeout_aware_ranker_weight": 0.32,
                            "lgbm_clean_oracle_pct_min": 0.38,
                            "lgbm_timeout_aware_ranker_pct_min": 0.45,
                            "lgbm_positive_clean_path_ts_pct_min": 0.55,
                            "lgbm_bad_mae_ts_pct_max": 0.70,
                            "lgbm_timeout_ts_pct_max": 0.72,
                            "lgbm_pred_timeout_cap": 0.28,
                        }
                        for s32_name, s32_score, s32_mask, s32_diag in (
                            (
                                "s32_clean_oracle_timeout_recall_source_stageA_rerank",
                                clean_oracle_timeout_recall_score,
                                clean_oracle_timeout_recall_mask,
                                base_s32_diag,
                            ),
                            (
                                (
                                    "s32_clean_oracle_timeout_recall_source_strict"
                                    "_stageA_rerank"
                                ),
                                clean_oracle_timeout_recall_strict_score,
                                clean_oracle_timeout_recall_strict_mask,
                                {
                                    **base_s32_diag,
                                    "s32_strict_gate": True,
                                    "clean_oracle_ranker_weight": 0.44,
                                    "timeout_aware_ranker_weight": 0.28,
                                    "lgbm_clean_oracle_pct_min": 0.45,
                                    "lgbm_bad_mae_ts_pct_max": 0.62,
                                    "lgbm_timeout_ts_pct_max": 0.62,
                                    "lgbm_pred_timeout_cap": 0.24,
                                },
                            ),
                        ):
                            s7_specs.append((s32_name, s32_score, s32_mask, s32_diag))
                            for final_frac in (0.020, 0.030, 0.050, 0.080):
                                s7_specs.append(
                                    (
                                        s32_name.replace(
                                            "_stageA_",
                                            (
                                                f"_final_frac_"
                                                f"{int(round(final_frac * 1000)):03d}"
                                                "_stageA_"
                                            ),
                                        ),
                                        s32_score,
                                        s32_mask,
                                        {
                                            **s32_diag,
                                            "selection_top_frac": float(final_frac),
                                        },
                                    )
                                )
                    if include_s24_path_first_selectors:
                        s24_source_score = (
                            0.70 * lgbm_s24_broad_path_first_ranker_pct
                            + 0.15 * lgbm_utility_ts_pct
                            + 0.15 * lgbm_clean_path_ts_pct
                            - 0.16 * lgbm_bad_mae_ts_pct
                            - 0.08 * lgbm_timeout_ts_pct
                        ).astype(np.float32)
                        s24_dirty_zero_score = (
                            0.75 * lgbm_s24_broad_path_first_dirty_zero_ranker_pct
                            + 0.15 * lgbm_utility_ts_pct
                            + 0.10 * lgbm_clean_path_ts_pct
                            - 0.20 * lgbm_bad_mae_ts_pct
                            - 0.10 * lgbm_timeout_ts_pct
                        ).astype(np.float32)
                        base_s24_diag = {
                            "s7_ablation": "s24_broad_path_first_source_stageA_rerank",
                            "ranker_type": "lgbm_lambdarank",
                            "ranker_relevance": "s24_broad_path_first_source",
                            "ranker_status": lgbm_s24_broad_path_first_ranker_status,
                            "stage_a_candidate_pool": "pre_risk_high_recall",
                            "s24_broad_path_first_source_target": True,
                            "bad_mae_penalty_lambda": 0.16,
                            "timeout_penalty_lambda": 0.08,
                        }
                        s7_specs.append(
                            (
                                "s24_broad_path_first_source_stageA_rerank",
                                s24_source_score,
                                stage_a_candidate_pre_risk_mask,
                                base_s24_diag,
                            )
                        )
                        for final_frac in (0.020, 0.030, 0.050):
                            s7_specs.append(
                                (
                                    (
                                        "s24_broad_path_first_source_final_frac_"
                                        f"{int(round(final_frac * 1000)):03d}"
                                    ),
                                    s24_source_score,
                                    stage_a_candidate_pre_risk_mask,
                                    {
                                        **base_s24_diag,
                                        "selection_top_frac": float(final_frac),
                                    },
                                )
                            )
                        strict_s24_diag = {
                            **base_s24_diag,
                            "ranker_relevance": "s24_broad_path_first_dirty_zero",
                            "ranker_status": (
                                lgbm_s24_broad_path_first_dirty_zero_ranker_status
                            ),
                            "dirty_positive_relevance": 0,
                            "bad_mae_penalty_lambda": 0.20,
                            "timeout_penalty_lambda": 0.10,
                        }
                        for final_frac in (0.030, 0.050):
                            s7_specs.append(
                                (
                                    (
                                        "s24_broad_path_first_dirty_zero_final_frac_"
                                        f"{int(round(final_frac * 1000)):03d}"
                                    ),
                                    s24_dirty_zero_score,
                                    stage_a_candidate_pre_risk_mask,
                                    {
                                        **strict_s24_diag,
                                        "selection_top_frac": float(final_frac),
                                    },
                                )
                            )
                        s27_inverted_s24_pct = (
                            1.0 - lgbm_s24_broad_path_first_ranker_pct
                        ).astype(np.float32)
                        s27_inverted_s24_dirty_zero_pct = (
                            1.0 - lgbm_s24_broad_path_first_dirty_zero_ranker_pct
                        ).astype(np.float32)
                        s27_path_clean_score = (
                            0.42 * lgbm_positive_clean_exec_strict_score
                            + 0.30 * s27_inverted_s24_pct
                            + 0.18 * s22_bucket_quality_rank_pct
                            + 0.10 * lgbm_clean_path_ts_pct
                            - 0.22 * lgbm_bad_mae_ts_pct
                            - 0.12 * lgbm_timeout_ts_pct
                        ).astype(np.float32)
                        s27_path_clean_strict_score = (
                            0.38 * lgbm_positive_clean_exec_strict_score
                            + 0.35 * s27_inverted_s24_dirty_zero_pct
                            + 0.18 * s22_bucket_quality_rank_pct
                            + 0.12 * lgbm_clean_path_ts_pct
                            - 0.26 * lgbm_bad_mae_ts_pct
                            - 0.14 * lgbm_timeout_ts_pct
                        ).astype(np.float32)
                        s27_base_mask = (
                            stage_a_candidate_pre_risk_mask
                            & pd.to_numeric(
                                lgbm_positive_clean_path_ts_pct,
                                errors="coerce",
                            )
                            .ge(0.55)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_dirty_positive_bad_mae_ts_pct,
                                errors="coerce",
                            )
                            .le(0.55)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(0.60)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_ts_pct, errors="coerce")
                            .le(0.60)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                            .le(0.18)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        s27_strict_mask = (
                            s27_base_mask
                            & pd.to_numeric(lgbm_bad_mae_pred, errors="coerce")
                            .le(0.55)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        base_s27_diag = {
                            "s7_ablation": (
                                "s27_s24_inverted_path_first_source_repair"
                            ),
                            "ranker_type": "lgbm_lambdarank_inverted_s24_blend",
                            "ranker_relevance": "s24_broad_path_first_source_inverted",
                            "ranker_status": lgbm_s24_broad_path_first_ranker_status,
                            "stage_a_candidate_pool": "pre_risk_high_recall",
                            "s24_broad_path_first_source_target": True,
                            "s27_inverted_s24_ranker": True,
                            "lgbm_positive_clean_path_ts_pct_min": 0.55,
                            "lgbm_dirty_positive_bad_mae_ts_pct_max": 0.55,
                            "lgbm_bad_mae_ts_pct_max": 0.60,
                            "lgbm_timeout_ts_pct_max": 0.60,
                            "lgbm_pred_timeout_cap": 0.18,
                        }
                        for s27_name, s27_score, s27_mask, s27_diag in (
                            (
                                "s27_inverted_s24_positive_clean_stageA_rerank",
                                s27_path_clean_score,
                                s27_base_mask,
                                base_s27_diag,
                            ),
                            (
                                "s27_inverted_s24_positive_clean_strict_stageA_rerank",
                                s27_path_clean_strict_score,
                                s27_strict_mask,
                                {
                                    **base_s27_diag,
                                    "ranker_relevance": (
                                        "s24_broad_path_first_dirty_zero_inverted"
                                    ),
                                    "ranker_status": (
                                        lgbm_s24_broad_path_first_dirty_zero_ranker_status
                                    ),
                                    "lgbm_pred_bad_mae_cap": 0.55,
                                },
                            ),
                        ):
                            s7_specs.append(
                                (
                                    s27_name,
                                    s27_score,
                                    s27_mask,
                                    s27_diag,
                                )
                            )
                            for final_frac in (0.020, 0.030, 0.050):
                                s7_specs.append(
                                    (
                                        s27_name.replace(
                                            "_stageA_",
                                            (
                                                f"_final_frac_"
                                                f"{int(round(final_frac * 1000)):03d}"
                                                "_stageA_"
                                            ),
                                        ),
                                        s27_score,
                                        s27_mask,
                                        {
                                            **s27_diag,
                                            "selection_top_frac": float(final_frac),
                                        },
                                    )
                                )
                        s28_side_path_score = (
                            0.46 * lgbm_s28_side_s24_ranker_pct
                            + 0.24 * lgbm_positive_clean_exec_strict_score
                            + 0.15 * s22_bucket_quality_rank_pct
                            + 0.15 * lgbm_clean_path_ts_pct
                            - 0.24 * lgbm_bad_mae_ts_pct
                            - 0.12 * lgbm_timeout_ts_pct
                        ).astype(np.float32)
                        s28_side_path_strict_score = (
                            0.50 * lgbm_s28_side_s24_dirty_zero_ranker_pct
                            + 0.22 * lgbm_positive_clean_exec_strict_score
                            + 0.14 * s22_bucket_quality_rank_pct
                            + 0.16 * lgbm_clean_path_ts_pct
                            - 0.28 * lgbm_bad_mae_ts_pct
                            - 0.14 * lgbm_timeout_ts_pct
                        ).astype(np.float32)
                        s28_base_mask = (
                            stage_a_candidate_pre_risk_mask
                            & pd.to_numeric(
                                lgbm_positive_clean_path_ts_pct,
                                errors="coerce",
                            )
                            .ge(0.55)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_dirty_positive_bad_mae_ts_pct,
                                errors="coerce",
                            )
                            .le(0.55)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(0.60)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_ts_pct, errors="coerce")
                            .le(0.60)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                            .le(0.18)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        s28_strict_mask = (
                            s28_base_mask
                            & pd.to_numeric(lgbm_bad_mae_pred, errors="coerce")
                            .le(0.55)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        base_s28_diag = {
                            "s7_ablation": "s28_side_specific_s24_path_first_ranker",
                            "ranker_type": "side_specific_lgbm_lambdarank",
                            "ranker_relevance": "s24_broad_path_first_source",
                            "ranker_status": lgbm_s28_side_s24_ranker_status,
                            "stage_a_candidate_pool": "pre_risk_high_recall",
                            "s24_broad_path_first_source_target": True,
                            "s28_side_specific_ranker": True,
                            "lgbm_positive_clean_path_ts_pct_min": 0.55,
                            "lgbm_dirty_positive_bad_mae_ts_pct_max": 0.55,
                            "lgbm_bad_mae_ts_pct_max": 0.60,
                            "lgbm_timeout_ts_pct_max": 0.60,
                            "lgbm_pred_timeout_cap": 0.18,
                        }
                        for s28_name, s28_score, s28_mask, s28_diag in (
                            (
                                "s28_side_s24_positive_clean_stageA_rerank",
                                s28_side_path_score,
                                s28_base_mask,
                                base_s28_diag,
                            ),
                            (
                                "s28_side_s24_positive_clean_strict_stageA_rerank",
                                s28_side_path_strict_score,
                                s28_strict_mask,
                                {
                                    **base_s28_diag,
                                    "ranker_relevance": (
                                        "s24_broad_path_first_dirty_zero"
                                    ),
                                    "ranker_status": (
                                        lgbm_s28_side_s24_dirty_zero_ranker_status
                                    ),
                                    "lgbm_pred_bad_mae_cap": 0.55,
                                },
                            ),
                        ):
                            s7_specs.append(
                                (
                                    s28_name,
                                    s28_score,
                                    s28_mask,
                                    s28_diag,
                                )
                            )
                            for final_frac in (0.020, 0.030, 0.050):
                                s7_specs.append(
                                    (
                                        s28_name.replace(
                                            "_stageA_",
                                            (
                                                f"_final_frac_"
                                                f"{int(round(final_frac * 1000)):03d}"
                                                "_stageA_"
                                            ),
                                        ),
                                        s28_score,
                                        s28_mask,
                                        {
                                            **s28_diag,
                                            "selection_top_frac": float(final_frac),
                                        },
                                    )
                                )
                        side_s29 = pd.to_numeric(
                            valid_metrics["side"].reset_index(drop=True),
                            errors="coerce",
                        ).fillna(1.0)
                        long_s29 = side_s29.ge(0.0).to_numpy(dtype=bool)
                        short_s29 = side_s29.lt(0.0).to_numpy(dtype=bool)
                        s29_side_clean_score = (
                            0.28 * s28_side_path_strict_score
                            + 0.38 * lgbm_side_positive_clean_path_ts_pct
                            + 0.18 * lgbm_positive_clean_exec_strict_score
                            + 0.12 * s22_bucket_quality_rank_pct
                            + 0.08 * lgbm_clean_path_ts_pct
                            - 0.34 * lgbm_side_dirty_positive_bad_mae_ts_pct
                            - 0.22 * lgbm_bad_mae_ts_pct
                            - 0.12 * lgbm_timeout_ts_pct
                        ).astype(np.float32)
                        s29_side_clean_strict_score = (
                            0.24 * s28_side_path_strict_score
                            + 0.46 * lgbm_side_positive_clean_path_ts_pct
                            + 0.16 * lgbm_positive_clean_exec_strict_score
                            + 0.10 * s22_bucket_quality_rank_pct
                            + 0.06 * lgbm_clean_path_ts_pct
                            - 0.44 * lgbm_side_dirty_positive_bad_mae_ts_pct
                            - 0.28 * lgbm_bad_mae_ts_pct
                            - 0.15 * lgbm_timeout_ts_pct
                        ).astype(np.float32)
                        s29_long_mask = (
                            long_s29
                            & pd.to_numeric(
                                lgbm_side_positive_clean_path_ts_pct,
                                errors="coerce",
                            )
                            .ge(0.65)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_side_positive_clean_path_pred,
                                errors="coerce",
                            )
                            .ge(0.30)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_side_dirty_positive_bad_mae_ts_pct,
                                errors="coerce",
                            )
                            .le(0.45)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_side_dirty_positive_bad_mae_pred,
                                errors="coerce",
                            )
                            .le(0.52)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(0.55)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_ts_pct, errors="coerce")
                            .le(0.60)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                            .le(0.18)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        s29_short_mask = (
                            short_s29
                            & pd.to_numeric(
                                lgbm_side_positive_clean_path_ts_pct,
                                errors="coerce",
                            )
                            .ge(0.55)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_side_dirty_positive_bad_mae_ts_pct,
                                errors="coerce",
                            )
                            .le(0.55)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(0.60)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_ts_pct, errors="coerce")
                            .le(0.60)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                            .le(0.20)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        s29_base_mask = stage_a_candidate_pre_risk_mask & (
                            s29_long_mask | s29_short_mask
                        )
                        s29_long_strict_mask = (
                            s29_long_mask
                            & pd.to_numeric(
                                lgbm_side_positive_clean_path_ts_pct,
                                errors="coerce",
                            )
                            .ge(0.70)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_side_dirty_positive_bad_mae_ts_pct,
                                errors="coerce",
                            )
                            .le(0.40)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(0.50)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        s29_short_strict_mask = (
                            s29_short_mask
                            & pd.to_numeric(
                                lgbm_side_positive_clean_path_ts_pct,
                                errors="coerce",
                            )
                            .ge(0.60)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_side_dirty_positive_bad_mae_ts_pct,
                                errors="coerce",
                            )
                            .le(0.50)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(0.55)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        s29_strict_mask = stage_a_candidate_pre_risk_mask & (
                            s29_long_strict_mask | s29_short_strict_mask
                        )
                        base_s29_diag = {
                            "s7_ablation": (
                                "s29_side_conditional_positive_clean_contrast"
                            ),
                            "ranker_type": (
                                "side_specific_lgbm_lambdarank_plus_side_conditional_heads"
                            ),
                            "ranker_relevance": "s24_broad_path_first_dirty_zero",
                            "ranker_status": (
                                lgbm_s28_side_s24_dirty_zero_ranker_status
                            ),
                            "stage_a_candidate_pool": "pre_risk_high_recall",
                            "s24_broad_path_first_source_target": True,
                            "s29_side_conditional_heads": True,
                            "lgbm_side_positive_clean_path_status": (
                                lgbm_side_positive_clean_path_status
                            ),
                            "lgbm_side_dirty_positive_bad_mae_status": (
                                lgbm_side_dirty_positive_bad_mae_status
                            ),
                            "s29_long_positive_clean_ts_pct_min": 0.65,
                            "s29_long_dirty_positive_ts_pct_max": 0.45,
                            "s29_long_pred_dirty_positive_bad_mae_cap": 0.52,
                            "s29_short_positive_clean_ts_pct_min": 0.55,
                            "s29_short_dirty_positive_ts_pct_max": 0.55,
                            "lgbm_timeout_pred_long_cap": 0.18,
                            "lgbm_timeout_pred_short_cap": 0.20,
                        }
                        for s29_name, s29_score, s29_mask, s29_diag in (
                            (
                                "s29_side_clean_contrast_stageA_rerank",
                                s29_side_clean_score,
                                s29_base_mask,
                                base_s29_diag,
                            ),
                            (
                                "s29_side_clean_contrast_strict_stageA_rerank",
                                s29_side_clean_strict_score,
                                s29_strict_mask,
                                {
                                    **base_s29_diag,
                                    "s29_strict_side_conditional_gate": True,
                                    "s29_long_positive_clean_ts_pct_min": 0.70,
                                    "s29_long_dirty_positive_ts_pct_max": 0.40,
                                    "s29_short_positive_clean_ts_pct_min": 0.60,
                                    "s29_short_dirty_positive_ts_pct_max": 0.50,
                                },
                            ),
                        ):
                            s7_specs.append(
                                (
                                    s29_name,
                                    s29_score,
                                    s29_mask,
                                    s29_diag,
                                )
                            )
                            for final_frac in (0.020, 0.030, 0.050):
                                s7_specs.append(
                                    (
                                        s29_name.replace(
                                            "_stageA_",
                                            (
                                                f"_final_frac_"
                                                f"{int(round(final_frac * 1000)):03d}"
                                                "_stageA_"
                                            ),
                                        ),
                                        s29_score,
                                        s29_mask,
                                        {
                                            **s29_diag,
                                            "selection_top_frac": float(final_frac),
                                        },
                                    )
                                )
                        s30_side_asym_score = (
                            0.52 * lgbm_s30_side_asym_ranker_pct
                            + 0.16 * lgbm_s30_side_asym_dirty_zero_ranker_pct
                            + 0.18 * lgbm_side_positive_clean_path_ts_pct
                            + 0.10 * s22_bucket_quality_rank_pct
                            + 0.08 * lgbm_clean_path_ts_pct
                            - 0.26 * lgbm_side_dirty_positive_bad_mae_ts_pct
                            - 0.18 * lgbm_bad_mae_ts_pct
                            - 0.10 * lgbm_timeout_ts_pct
                        ).astype(np.float32)
                        s30_side_asym_strict_score = (
                            0.46 * lgbm_s30_side_asym_dirty_zero_ranker_pct
                            + 0.22 * lgbm_s30_side_asym_ranker_pct
                            + 0.20 * lgbm_side_positive_clean_path_ts_pct
                            + 0.08 * s22_bucket_quality_rank_pct
                            + 0.08 * lgbm_clean_path_ts_pct
                            - 0.34 * lgbm_side_dirty_positive_bad_mae_ts_pct
                            - 0.23 * lgbm_bad_mae_ts_pct
                            - 0.12 * lgbm_timeout_ts_pct
                        ).astype(np.float32)
                        s30_long_mask = (
                            long_s29
                            & pd.to_numeric(
                                lgbm_s30_side_asym_ranker_pct,
                                errors="coerce",
                            )
                            .ge(0.55)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_side_positive_clean_path_ts_pct,
                                errors="coerce",
                            )
                            .ge(0.62)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_side_dirty_positive_bad_mae_ts_pct,
                                errors="coerce",
                            )
                            .le(0.48)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(0.58)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_ts_pct, errors="coerce")
                            .le(0.62)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                            .le(0.20)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        s30_short_mask = (
                            short_s29
                            & pd.to_numeric(
                                lgbm_s30_side_asym_ranker_pct,
                                errors="coerce",
                            )
                            .ge(0.52)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_side_positive_clean_path_ts_pct,
                                errors="coerce",
                            )
                            .ge(0.52)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_side_dirty_positive_bad_mae_ts_pct,
                                errors="coerce",
                            )
                            .le(0.58)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(0.62)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_ts_pct, errors="coerce")
                            .le(0.65)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_timeout_pred, errors="coerce")
                            .le(0.22)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        s30_base_mask = stage_a_candidate_pre_risk_mask & (
                            s30_long_mask | s30_short_mask
                        )
                        s30_long_strict_mask = (
                            s30_long_mask
                            & pd.to_numeric(
                                lgbm_s30_side_asym_dirty_zero_ranker_pct,
                                errors="coerce",
                            )
                            .ge(0.58)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_side_positive_clean_path_ts_pct,
                                errors="coerce",
                            )
                            .ge(0.68)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_side_dirty_positive_bad_mae_ts_pct,
                                errors="coerce",
                            )
                            .le(0.42)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(0.52)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        s30_short_strict_mask = (
                            s30_short_mask
                            & pd.to_numeric(
                                lgbm_s30_side_asym_dirty_zero_ranker_pct,
                                errors="coerce",
                            )
                            .ge(0.55)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(
                                lgbm_side_dirty_positive_bad_mae_ts_pct,
                                errors="coerce",
                            )
                            .le(0.52)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(lgbm_bad_mae_ts_pct, errors="coerce")
                            .le(0.58)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        s30_strict_mask = stage_a_candidate_pre_risk_mask & (
                            s30_long_strict_mask | s30_short_strict_mask
                        )
                        base_s30_diag = {
                            "s7_ablation": "s30_side_asymmetric_path_first_source_objective",
                            "ranker_type": "side_specific_lgbm_lambdarank",
                            "ranker_relevance": "s30_side_asymmetric_path_first_source",
                            "ranker_status": lgbm_s30_side_asym_ranker_status,
                            "stage_a_candidate_pool": "pre_risk_high_recall",
                            "s30_side_asymmetric_source_target": True,
                            "s30_long_positive_clean_ts_pct_min": 0.62,
                            "s30_long_dirty_positive_ts_pct_max": 0.48,
                            "s30_short_positive_clean_ts_pct_min": 0.52,
                            "s30_short_dirty_positive_ts_pct_max": 0.58,
                        }
                        for s30_name, s30_score, s30_mask, s30_diag in (
                            (
                                "s30_side_asym_path_first_stageA_rerank",
                                s30_side_asym_score,
                                s30_base_mask,
                                base_s30_diag,
                            ),
                            (
                                "s30_side_asym_path_first_strict_stageA_rerank",
                                s30_side_asym_strict_score,
                                s30_strict_mask,
                                {
                                    **base_s30_diag,
                                    "ranker_relevance": (
                                        "s30_side_asymmetric_path_first_dirty_zero"
                                    ),
                                    "ranker_status": (
                                        lgbm_s30_side_asym_dirty_zero_ranker_status
                                    ),
                                    "s30_strict_side_asymmetric_gate": True,
                                    "s30_long_positive_clean_ts_pct_min": 0.68,
                                    "s30_long_dirty_positive_ts_pct_max": 0.42,
                                    "s30_short_dirty_positive_ts_pct_max": 0.52,
                                },
                            ),
                        ):
                            s7_specs.append((s30_name, s30_score, s30_mask, s30_diag))
                            for final_frac in (0.020, 0.030, 0.050):
                                s7_specs.append(
                                    (
                                        s30_name.replace(
                                            "_stageA_",
                                            (
                                                f"_final_frac_"
                                                f"{int(round(final_frac * 1000)):03d}"
                                                "_stageA_"
                                            ),
                                        ),
                                        s30_score,
                                        s30_mask,
                                        {
                                            **s30_diag,
                                            "selection_top_frac": float(final_frac),
                                        },
                                    )
                                )
                    for max_bad_mae_pred in (0.60, 0.57, 0.55, 0.53, 0.52, 0.50):
                        s9_utility_cap_mask = (
                            stage_a_candidate_pre_risk_mask
                            & pd.to_numeric(bad_mae_pred, errors="coerce")
                            .le(float(max_bad_mae_pred))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(timeout_pred, errors="coerce")
                            .le(0.12)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        s7_specs.append(
                            (
                                (
                                    "s9_lgbm_utility_ranker"
                                    f"_pred_bad_mae_cap_{int(round(max_bad_mae_pred * 100)):02d}"
                                    "_timeout_cap_12_stageA_rerank"
                                ),
                                lgbm_ranker_risk_score,
                                s9_utility_cap_mask,
                                {
                                    "s7_ablation": "lgbm_utility_ranker_with_hard_path_risk_cap",
                                    "ranker_type": "lgbm_lambdarank",
                                    "ranker_relevance": "utility_quintile",
                                    "ranker_status": lgbm_ranker_status,
                                    "pred_bad_mae_cap": float(max_bad_mae_pred),
                                    "pred_timeout_cap": 0.12,
                                },
                            )
                        )
                    for blend_name, blend_score, blend_diag in (
                        (
                            "s10_lgbm_utility_path_blend_75_25_stageA_rerank",
                            lgbm_utility_path_blend_75,
                            {"utility_ranker_weight": 0.75, "path_ranker_weight": 0.25},
                        ),
                        (
                            "s10_lgbm_utility_path_blend_60_40_stageA_rerank",
                            lgbm_utility_path_blend_60,
                            {"utility_ranker_weight": 0.60, "path_ranker_weight": 0.40},
                        ),
                        (
                            "s10_lgbm_utility_path_blend_50_50_stageA_rerank",
                            lgbm_utility_path_blend_50,
                            {"utility_ranker_weight": 0.50, "path_ranker_weight": 0.50},
                        ),
                        (
                            "s10_lgbm_three_ranker_blend_stageA_rerank",
                            lgbm_three_ranker_blend,
                            {
                                "utility_ranker_weight": 0.55,
                                "path_ranker_weight": 0.25,
                                "oracle_ranker_weight": 0.20,
                            },
                        ),
                    ):
                        s7_specs.append(
                            (
                                blend_name,
                                blend_score,
                                stage_a_candidate_mask,
                                {
                                    "s7_ablation": "lgbm_ranker_utility_path_blend",
                                    "ranker_type": "lgbm_lambdarank_blend",
                                    "ranker_status": (
                                        f"utility:{lgbm_ranker_status};"
                                        f"path:{lgbm_path_ranker_status};"
                                        f"oracle:{lgbm_oracle_ranker_status}"
                                    ),
                                    "pred_bad_mae_cap": 0.70,
                                    "pred_timeout_cap": 0.30,
                                    **blend_diag,
                                },
                            )
                        )
                        s7_specs.append(
                            (
                                blend_name.replace("_stageA_", "_strict_stageA_"),
                                blend_score,
                                stage_a_candidate_pre_risk_mask & strict_risk_mask,
                                {
                                    "s7_ablation": "lgbm_ranker_utility_path_blend_strict_risk",
                                    "ranker_type": "lgbm_lambdarank_blend",
                                    "ranker_status": (
                                        f"utility:{lgbm_ranker_status};"
                                        f"path:{lgbm_path_ranker_status};"
                                        f"oracle:{lgbm_oracle_ranker_status}"
                                    ),
                                    "pred_bad_mae_cap": 0.57,
                                    "pred_timeout_cap": 0.12,
                                    **blend_diag,
                                },
                            )
                        )
                    for final_frac in (0.010, 0.015, 0.020):
                        s7_specs.append(
                            (
                                (
                                    "s11_lgbm_utility_ranker_stageA"
                                    f"_final_frac_{int(round(final_frac * 1000)):03d}_rerank"
                                ),
                                lgbm_ranker_risk_score,
                                stage_a_candidate_mask,
                                {
                                    "s7_ablation": "lgbm_utility_ranker_reduced_final_exposure",
                                    "ranker_type": "lgbm_lambdarank",
                                    "ranker_relevance": "utility_quintile",
                                    "ranker_status": lgbm_ranker_status,
                                    "pred_bad_mae_cap": 0.70,
                                    "pred_timeout_cap": 0.30,
                                    "selection_top_frac": float(final_frac),
                                },
                            )
                        )
                        s7_specs.append(
                            (
                                (
                                    "s11_lgbm_utility_path_blend_60_40_stageA"
                                    f"_final_frac_{int(round(final_frac * 1000)):03d}_rerank"
                                ),
                                lgbm_utility_path_blend_60,
                                stage_a_candidate_mask,
                                {
                                    "s7_ablation": "lgbm_utility_path_blend_reduced_final_exposure",
                                    "ranker_type": "lgbm_lambdarank_blend",
                                    "ranker_status": (
                                        f"utility:{lgbm_ranker_status};"
                                        f"path:{lgbm_path_ranker_status}"
                                    ),
                                    "pred_bad_mae_cap": 0.70,
                                    "pred_timeout_cap": 0.30,
                                    "selection_top_frac": float(final_frac),
                                    "utility_ranker_weight": 0.60,
                                    "path_ranker_weight": 0.40,
                                },
                            )
                        )
                        s7_specs.append(
                            (
                                (
                                    "s13_lgbm_utility_clean_oracle_blend_stageA"
                                    f"_final_frac_{int(round(final_frac * 1000)):03d}"
                                    "_rerank"
                                ),
                                lgbm_utility_clean_oracle_blend,
                                stage_a_candidate_mask,
                                {
                                    "s7_ablation": (
                                        "lgbm_utility_clean_oracle_blend_reduced_final_exposure"
                                    ),
                                    "ranker_type": "lgbm_lambdarank_blend",
                                    "ranker_status": (
                                        f"utility:{lgbm_ranker_status};"
                                        f"clean_oracle:{lgbm_clean_oracle_ranker_status}"
                                    ),
                                    "selection_top_frac": float(final_frac),
                                    "utility_ranker_weight": 0.55,
                                    "clean_oracle_ranker_weight": 0.45,
                                    "pred_bad_mae_cap": 0.70,
                                    "pred_timeout_cap": 0.30,
                                },
                            )
                        )
                        s7_specs.append(
                            (
                                (
                                    "s13_lgbm_clean_path_oracle_blend_stageA"
                                    f"_final_frac_{int(round(final_frac * 1000)):03d}"
                                    "_rerank"
                                ),
                                lgbm_clean_path_oracle_blend,
                                stage_a_candidate_mask,
                                {
                                    "s7_ablation": (
                                        "lgbm_clean_path_oracle_blend_reduced_final_exposure"
                                    ),
                                    "ranker_type": "lgbm_lambdarank_blend",
                                    "ranker_status": (
                                        f"utility:{lgbm_ranker_status};"
                                        f"path:{lgbm_path_ranker_status};"
                                        f"clean_oracle:{lgbm_clean_oracle_ranker_status}"
                                    ),
                                    "selection_top_frac": float(final_frac),
                                    "utility_ranker_weight": 0.35,
                                    "path_ranker_weight": 0.30,
                                    "clean_oracle_ranker_weight": 0.35,
                                    "pred_bad_mae_cap": 0.70,
                                    "pred_timeout_cap": 0.30,
                                },
                            )
                        )
                    for min_clean_path_pred in (0.20, 0.25, 0.30):
                        clean_path_admission_mask = (
                            stage_a_candidate_pre_risk_mask
                            & pd.to_numeric(clean_path_pred, errors="coerce")
                            .ge(float(min_clean_path_pred))
                            .fillna(False)
                            .to_numpy(dtype=bool)
                            & pd.to_numeric(timeout_pred, errors="coerce")
                            .le(0.20)
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                        for final_frac in (0.020, 0.030):
                            s7_specs.append(
                                (
                                    (
                                        "s12_lgbm_utility_ranker_clean_path"
                                        f"_min_{int(round(min_clean_path_pred * 100)):02d}"
                                        f"_final_frac_{int(round(final_frac * 1000)):03d}"
                                        "_stageA_rerank"
                                    ),
                                    lgbm_ranker_risk_score,
                                    clean_path_admission_mask,
                                    {
                                        "s7_ablation": "lgbm_utility_ranker_clean_path_admission",
                                        "ranker_type": "lgbm_lambdarank",
                                        "ranker_relevance": "utility_quintile",
                                        "ranker_status": lgbm_ranker_status,
                                        "min_clean_path_pred": float(min_clean_path_pred),
                                        "pred_timeout_cap": 0.20,
                                        "selection_top_frac": float(final_frac),
                                    },
                                )
                            )
                            s7_specs.append(
                                (
                                    (
                                        "s12_lgbm_utility_path_blend_clean_path"
                                        f"_min_{int(round(min_clean_path_pred * 100)):02d}"
                                        f"_final_frac_{int(round(final_frac * 1000)):03d}"
                                        "_stageA_rerank"
                                    ),
                                    lgbm_utility_path_blend_60,
                                    clean_path_admission_mask,
                                    {
                                        "s7_ablation": (
                                            "lgbm_utility_path_blend_clean_path_admission"
                                        ),
                                        "ranker_type": "lgbm_lambdarank_blend",
                                        "ranker_status": (
                                            f"utility:{lgbm_ranker_status};"
                                            f"path:{lgbm_path_ranker_status}"
                                        ),
                                        "min_clean_path_pred": float(min_clean_path_pred),
                                        "pred_timeout_cap": 0.20,
                                        "selection_top_frac": float(final_frac),
                                        "utility_ranker_weight": 0.60,
                                        "path_ranker_weight": 0.40,
                                    },
                                )
                            )
                    for s7_name, s7_score, s7_eligible_mask, s7_diag in s7_specs:
                        full_s7_name = (
                            f"{s7_name}_side_cap_{int(round(side_cap_max_share * 100)):02d}"
                        )
                        if candidate_ledger_only and full_s7_name not in requested_ledger_selectors:
                            continue
                        selector_top_frac = float(s7_diag.get("selection_top_frac", top_frac))
                        selected_idx, hard_cap_diag = _constrained_top_indices(
                            score=s7_score,
                            side=valid_metrics["side"],
                            eligible=pd.Series(s7_eligible_mask, index=valid.index),
                            top_frac=selector_top_frac,
                            max_side_share=float(side_cap_max_share),
                        )
                        final_mask = _mask_from_indices(len(valid), selected_idx)
                        final_diag = _oracle_recall_stats(
                            metrics=valid_metrics,
                            mask=final_mask,
                            top_frac=float(top_frac),
                            prefix="final",
                        )
                        hard_cap_score = _score_from_selected_indices(
                            base_score=s7_score,
                            selected_idx=selected_idx,
                        )
                        variants.append(
                            (
                                full_s7_name,
                                hard_cap_score,
                                {
                                    **s7_stage_diag,
                                    **s7_diag,
                                    **hard_cap_diag,
                                    **final_diag,
                                },
                                selected_idx,
                            )
                        )
                    if candidate_ledger_only:
                        pass
                    for variant_name, variant_score, variant_diag, _selected_idx in base_variants:
                        capped_score, cap_diag = _side_capped_score(
                            score=variant_score,
                            side=valid_metrics["side"],
                            top_frac=float(top_frac),
                            max_side_share=float(side_cap_max_share),
                        )
                        variants.append(
                            (
                                f"{variant_name}_side_cap_{int(round(side_cap_max_share * 100)):02d}",
                                capped_score,
                                {**variant_diag, **cap_diag},
                                None,
                            )
                        )
                    for max_bad_mae_pred in (0.60, 0.57, 0.55, 0.53, 0.52, 0.50):
                        eligible = (
                            pd.to_numeric(bad_mae_pred, errors="coerce") <= float(max_bad_mae_pred)
                        ) & (pd.to_numeric(timeout_pred, errors="coerce") <= 0.12)
                        selected_idx, hard_cap_diag = _constrained_top_indices(
                            score=strong_risk_penalty,
                            side=valid_metrics["side"],
                            eligible=eligible,
                            top_frac=float(top_frac),
                            max_side_share=float(side_cap_max_share),
                        )
                        hard_cap_score = _score_from_selected_indices(
                            base_score=strong_risk_penalty,
                            selected_idx=selected_idx,
                        )
                        variants.append(
                            (
                                (
                                    "strong_bad_mae_timeout_penalty"
                                    f"_pred_bad_mae_cap_{int(round(max_bad_mae_pred * 100)):02d}"
                                    f"_side_cap_{int(round(side_cap_max_share * 100)):02d}"
                                ),
                                hard_cap_score,
                                {
                                    "bad_mae_penalty_lambda": 0.55,
                                    "timeout_penalty_lambda": 0.15,
                                    "pred_bad_mae_cap": float(max_bad_mae_pred),
                                    "pred_timeout_cap": 0.12,
                                    **hard_cap_diag,
                                },
                                selected_idx,
                            )
                        )
                    for max_bad_mae_pred in (0.60, 0.55, 0.53, 0.52, 0.50):
                        eligible = (
                            pd.to_numeric(side_bad_mae_pred, errors="coerce")
                            <= float(max_bad_mae_pred)
                        ) & (pd.to_numeric(side_timeout_pred, errors="coerce") <= 0.12)
                        selected_idx, hard_cap_diag = _constrained_top_indices(
                            score=side_risk_penalty,
                            side=valid_metrics["side"],
                            eligible=eligible,
                            top_frac=float(top_frac),
                            max_side_share=float(side_cap_max_share),
                        )
                        hard_cap_score = _score_from_selected_indices(
                            base_score=side_risk_penalty,
                            selected_idx=selected_idx,
                        )
                        variants.append(
                            (
                                (
                                    "side_specific_bad_mae_timeout_penalty"
                                    f"_pred_bad_mae_cap_{int(round(max_bad_mae_pred * 100)):02d}"
                                    f"_side_cap_{int(round(side_cap_max_share * 100)):02d}"
                                ),
                                hard_cap_score,
                                {
                                    "bad_mae_penalty_lambda": 0.55,
                                    "timeout_penalty_lambda": 0.15,
                                    "pred_bad_mae_cap": float(max_bad_mae_pred),
                                    "pred_timeout_cap": 0.12,
                                    "risk_head": "side_specific",
                                    **hard_cap_diag,
                                },
                                selected_idx,
                            )
                        )
                    for min_clean_path_pred in (0.35, 0.40, 0.45, 0.50):
                        eligible = (
                            pd.to_numeric(clean_path_pred, errors="coerce")
                            >= float(min_clean_path_pred)
                        )
                        selected_idx, hard_cap_diag = _constrained_top_indices(
                            score=clean_path_pred,
                            side=valid_metrics["side"],
                            eligible=eligible,
                            top_frac=float(top_frac),
                            max_side_share=float(side_cap_max_share),
                        )
                        hard_cap_score = _score_from_selected_indices(
                            base_score=clean_path_pred,
                            selected_idx=selected_idx,
                        )
                        variants.append(
                            (
                                (
                                    "clean_path_probability"
                                    f"_min_{int(round(min_clean_path_pred * 100)):02d}"
                                    f"_side_cap_{int(round(side_cap_max_share * 100)):02d}"
                                ),
                                hard_cap_score,
                                {
                                    "clean_path_head": True,
                                    "min_clean_path_pred": float(min_clean_path_pred),
                                    **hard_cap_diag,
                                },
                                selected_idx,
                            )
                        )
                    for feature_gap_cap in (0.45, 0.50, 0.55, 0.60):
                        eligible = (
                            pd.to_numeric(bad_mae_pred, errors="coerce").le(0.57)
                            & pd.to_numeric(timeout_pred, errors="coerce").le(0.12)
                            & pd.to_numeric(feature_gap_risk, errors="coerce").le(float(feature_gap_cap))
                        )
                        selected_idx, hard_cap_diag = _constrained_top_indices(
                            score=feature_gap_penalty,
                            side=valid_metrics["side"],
                            eligible=eligible,
                            top_frac=float(top_frac),
                            max_side_share=float(side_cap_max_share),
                        )
                        hard_cap_score = _score_from_selected_indices(
                            base_score=feature_gap_penalty,
                            selected_idx=selected_idx,
                        )
                        variants.append(
                            (
                                (
                                    "feature_gap_risk_veto"
                                    f"_cap_{int(round(feature_gap_cap * 100)):02d}"
                                    "_pred_bad_mae_cap_57"
                                    f"_side_cap_{int(round(side_cap_max_share * 100)):02d}"
                                ),
                                hard_cap_score,
                                {
                                    "bad_mae_penalty_lambda": 0.55,
                                    "timeout_penalty_lambda": 0.15,
                                    "feature_gap_risk_penalty_lambda": 0.20,
                                    "feature_gap_risk_cap": float(feature_gap_cap),
                                    "pred_bad_mae_cap": 0.57,
                                    "pred_timeout_cap": 0.12,
                                    **feature_gap_diag,
                                    **hard_cap_diag,
                                },
                                selected_idx,
                            )
                        )
                for variant_name, variant_score, variant_diag, selected_idx in variants:
                    decile = _decile_diagnostics(variant_score, valid_metrics["u_policy_net"])
                    ts_rank = _timestamp_ranking_metrics(
                        frame=valid,
                        metrics=valid_metrics,
                        target=target_valid,
                        score=variant_score,
                    )
                    row = _selection_metrics(
                        frame=valid,
                        metrics=valid_metrics,
                        target=target_valid,
                        score=variant_score,
                        arm=base_arm if variant_name == "raw_utility" else f"{base_arm}::{variant_name}",
                        selector=f"{selector_prefix}:{variant_name}",
                        period=month,
                        top_frac=top_frac,
                        selected_idx=selected_idx,
                    )
                    _add_delta_fields(row, baseline)
                    selected_for_risk = (
                        _rank_top_indices(variant_score, top_frac)
                        if selected_idx is None
                        else np.asarray(selected_idx, dtype=np.int64)
                    )
                    row.update(
                        {
                            "label_arm": label_arm,
                            "weight_arm": weight_arm,
                            "selector_variant": variant_name,
                            "model_feature_selector": model_feature_selector,
                            "model_feature_count": int(len(model_features)),
                            "model_features": ",".join(model_features),
                            **feature_selector_diag,
                            **variant_diag,
                            "pred_bad_mae_mean": _safe_mean(bad_mae_pred),
                            "pred_timeout_mean": _safe_mean(timeout_pred),
                            "side_pred_bad_mae_mean": _safe_mean(side_bad_mae_pred),
                            "side_pred_timeout_mean": _safe_mean(side_timeout_pred),
                            "clean_path_pred_mean": _safe_mean(clean_path_pred),
                            "feature_gap_risk_mean": _safe_mean(feature_gap_risk),
                            "clean_dirty_positive_risk_mean": _safe_mean(
                                clean_dirty_positive_risk
                            ),
                            "lgbm_bad_mae_pred_mean": _safe_mean(lgbm_bad_mae_pred),
                            "lgbm_timeout_pred_mean": _safe_mean(lgbm_timeout_pred),
                            "lgbm_clean_path_pred_mean": _safe_mean(lgbm_clean_path_pred),
                            "lgbm_dirty_positive_bad_mae_pred_mean": _safe_mean(
                                lgbm_dirty_positive_bad_mae_pred
                            ),
                            "lgbm_positive_clean_path_pred_mean": _safe_mean(
                                lgbm_positive_clean_path_pred
                            ),
                            "lgbm_side_dirty_positive_bad_mae_pred_mean": _safe_mean(
                                lgbm_side_dirty_positive_bad_mae_pred
                            ),
                            "lgbm_side_positive_clean_path_pred_mean": _safe_mean(
                                lgbm_side_positive_clean_path_pred
                            ),
                            "s22_bucket_quality_score_mean": _safe_mean(
                                s22_bucket_quality_score
                            ),
                            "s22_bucket_quality_rank_pct_mean": _safe_mean(
                                s22_bucket_quality_rank_pct
                            ),
                            "s22_bucket_relaxed_pass_count_mean": _safe_mean(
                                s22_bucket_relaxed_pass_count
                            ),
                            "s22_bucket_strict_pass_count_mean": _safe_mean(
                                s22_bucket_strict_pass_count
                            ),
                            **s22_bucket_quality_diag,
                            "lgbm_bad_mae_status": lgbm_bad_mae_status,
                            "lgbm_timeout_status": lgbm_timeout_status,
                            "lgbm_clean_path_status": lgbm_clean_path_status,
                            "lgbm_dirty_positive_bad_mae_status": (
                                lgbm_dirty_positive_bad_mae_status
                            ),
                            "lgbm_positive_clean_path_status": (
                                lgbm_positive_clean_path_status
                            ),
                            "lgbm_side_dirty_positive_bad_mae_status": (
                                lgbm_side_dirty_positive_bad_mae_status
                            ),
                            "lgbm_side_positive_clean_path_status": (
                                lgbm_side_positive_clean_path_status
                            ),
                            "lgbm_ranker_status": lgbm_ranker_status,
                            "lgbm_path_ranker_status": lgbm_path_ranker_status,
                            "lgbm_oracle_ranker_status": lgbm_oracle_ranker_status,
                            "lgbm_clean_oracle_ranker_status": lgbm_clean_oracle_ranker_status,
                            "lgbm_path_first_ranker_status": lgbm_path_first_ranker_status,
                            "lgbm_path_first_dirty_zero_ranker_status": (
                                lgbm_path_first_dirty_zero_ranker_status
                            ),
                            "lgbm_s24_broad_path_first_ranker_status": (
                                lgbm_s24_broad_path_first_ranker_status
                            ),
                            "lgbm_s24_broad_path_first_dirty_zero_ranker_status": (
                                lgbm_s24_broad_path_first_dirty_zero_ranker_status
                            ),
                            "lgbm_s28_side_s24_ranker_status": (
                                lgbm_s28_side_s24_ranker_status
                            ),
                            "lgbm_s28_side_s24_dirty_zero_ranker_status": (
                                lgbm_s28_side_s24_dirty_zero_ranker_status
                            ),
                            "lgbm_s30_side_asym_ranker_status": (
                                lgbm_s30_side_asym_ranker_status
                            ),
                            "lgbm_s30_side_asym_dirty_zero_ranker_status": (
                                lgbm_s30_side_asym_dirty_zero_ranker_status
                            ),
                            "lgbm_timeout_aware_clean_ranker_status": (
                                lgbm_timeout_aware_clean_ranker_status
                            ),
                            "lgbm_ranker_score_mean": _safe_mean(lgbm_ranker_score),
                            "lgbm_path_ranker_score_mean": _safe_mean(lgbm_path_ranker_score),
                            "lgbm_oracle_ranker_score_mean": _safe_mean(lgbm_oracle_ranker_score),
                            "lgbm_clean_oracle_ranker_score_mean": _safe_mean(
                                lgbm_clean_oracle_ranker_score
                            ),
                            "lgbm_path_first_ranker_score_mean": _safe_mean(
                                lgbm_path_first_ranker_score
                            ),
                            "lgbm_path_first_dirty_zero_ranker_score_mean": _safe_mean(
                                lgbm_path_first_dirty_zero_ranker_score
                            ),
                            "lgbm_s24_broad_path_first_ranker_score_mean": _safe_mean(
                                lgbm_s24_broad_path_first_ranker_score
                            ),
                            "lgbm_s24_broad_path_first_dirty_zero_ranker_score_mean": (
                                _safe_mean(
                                    lgbm_s24_broad_path_first_dirty_zero_ranker_score
                                )
                            ),
                            "lgbm_s28_side_s24_ranker_score_mean": _safe_mean(
                                lgbm_s28_side_s24_ranker_score
                            ),
                            "lgbm_s28_side_s24_dirty_zero_ranker_score_mean": (
                                _safe_mean(lgbm_s28_side_s24_dirty_zero_ranker_score)
                            ),
                            "lgbm_s30_side_asym_ranker_score_mean": _safe_mean(
                                lgbm_s30_side_asym_ranker_score
                            ),
                            "lgbm_s30_side_asym_dirty_zero_ranker_score_mean": (
                                _safe_mean(lgbm_s30_side_asym_dirty_zero_ranker_score)
                            ),
                            "lgbm_timeout_aware_clean_ranker_score_mean": (
                                _safe_mean(lgbm_timeout_aware_clean_ranker_score)
                            ),
                            "selected_feature_gap_risk_mean": _safe_mean(
                                feature_gap_risk.iloc[selected_for_risk]
                            ),
                            "selected_clean_dirty_positive_risk_mean": _safe_mean(
                                clean_dirty_positive_risk.iloc[selected_for_risk]
                            ),
                            "selected_lgbm_bad_mae_pred_mean": _safe_mean(
                                lgbm_bad_mae_pred.iloc[selected_for_risk]
                            ),
                            "selected_lgbm_timeout_pred_mean": _safe_mean(
                                lgbm_timeout_pred.iloc[selected_for_risk]
                            ),
                            "selected_lgbm_clean_path_pred_mean": _safe_mean(
                                lgbm_clean_path_pred.iloc[selected_for_risk]
                            ),
                            "selected_lgbm_dirty_positive_bad_mae_pred_mean": _safe_mean(
                                lgbm_dirty_positive_bad_mae_pred.iloc[selected_for_risk]
                            ),
                            "selected_lgbm_positive_clean_path_pred_mean": _safe_mean(
                                lgbm_positive_clean_path_pred.iloc[selected_for_risk]
                            ),
                            "selected_lgbm_side_dirty_positive_bad_mae_pred_mean": _safe_mean(
                                lgbm_side_dirty_positive_bad_mae_pred.iloc[selected_for_risk]
                            ),
                            "selected_lgbm_side_positive_clean_path_pred_mean": _safe_mean(
                                lgbm_side_positive_clean_path_pred.iloc[selected_for_risk]
                            ),
                            "selected_s22_bucket_quality_score_mean": _safe_mean(
                                s22_bucket_quality_score.iloc[selected_for_risk]
                            ),
                            "selected_s22_bucket_quality_rank_pct_mean": _safe_mean(
                                s22_bucket_quality_rank_pct.iloc[selected_for_risk]
                            ),
                            "selected_s22_bucket_relaxed_pass_count_mean": _safe_mean(
                                s22_bucket_relaxed_pass_count.iloc[selected_for_risk]
                            ),
                            "selected_s22_bucket_strict_pass_count_mean": _safe_mean(
                                s22_bucket_strict_pass_count.iloc[selected_for_risk]
                            ),
                            "selected_lgbm_bad_mae_ts_pct_mean": _safe_mean(
                                lgbm_bad_mae_ts_pct.iloc[selected_for_risk]
                            ),
                            "selected_lgbm_timeout_ts_pct_mean": _safe_mean(
                                lgbm_timeout_ts_pct.iloc[selected_for_risk]
                            ),
                            "selected_lgbm_clean_path_ts_pct_mean": _safe_mean(
                                lgbm_clean_path_ts_pct.iloc[selected_for_risk]
                            ),
                            "selected_lgbm_dirty_positive_bad_mae_ts_pct_mean": _safe_mean(
                                lgbm_dirty_positive_bad_mae_ts_pct.iloc[selected_for_risk]
                            ),
                            "selected_lgbm_positive_clean_path_ts_pct_mean": _safe_mean(
                                lgbm_positive_clean_path_ts_pct.iloc[selected_for_risk]
                            ),
                            "selected_lgbm_side_dirty_positive_bad_mae_ts_pct_mean": _safe_mean(
                                lgbm_side_dirty_positive_bad_mae_ts_pct.iloc[selected_for_risk]
                            ),
                            "selected_lgbm_side_positive_clean_path_ts_pct_mean": _safe_mean(
                                lgbm_side_positive_clean_path_ts_pct.iloc[selected_for_risk]
                            ),
                            "selected_clean_path_pred_mean": _safe_mean(
                                clean_path_pred.iloc[selected_for_risk]
                            ),
                            "clean_positive_rate": _safe_mean(
                                (
                                    (valid_metrics["u_policy_net"].iloc[selected_for_risk] > 0.0)
                                    & (
                                        valid_metrics["mae_norm"].iloc[selected_for_risk]
                                        < 1.0
                                    )
                                    & (
                                        valid_metrics["is_timeout"]
                                        .iloc[selected_for_risk]
                                        .astype(float)
                                        <= 0.0
                                    )
                                )
                            ),
                            "dirty_positive_rate": _safe_mean(
                                (
                                    (valid_metrics["u_policy_net"].iloc[selected_for_risk] > 0.0)
                                    & (
                                        (
                                            valid_metrics["mae_norm"].iloc[selected_for_risk]
                                            >= 1.0
                                        )
                                        | (
                                            valid_metrics["is_timeout"]
                                            .iloc[selected_for_risk]
                                            .astype(float)
                                            > 0.5
                                        )
                                    )
                                )
                            ),
                            "selected_pred_bad_mae_mean": _safe_mean(
                                (
                                    side_bad_mae_pred
                                    if str(variant_diag.get("risk_head", "")) == "side_specific"
                                    else bad_mae_pred
                                ).iloc[selected_for_risk]
                            ),
                            "selected_pred_timeout_mean": _safe_mean(
                                (
                                    side_timeout_pred
                                    if str(variant_diag.get("risk_head", "")) == "side_specific"
                                    else timeout_pred
                                ).iloc[selected_for_risk]
                            ),
                            "score_ic_u": _spearman(variant_score, valid_metrics["u_policy_net"]),
                            "score_ic_label": _spearman(variant_score, target_valid["target_soft"]),
                            "score_ic_bad_mae": _spearman(
                                variant_score,
                                (valid_metrics["mae_norm"] >= 1.0).astype(float),
                            ),
                            **decile,
                            **ts_rank,
                        }
                    )
                    rows.append(row)
                    clean_dirty_diagnostics.extend(
                        _clean_dirty_selected_diagnostics(
                            metrics=valid_metrics,
                            score=variant_score,
                            selector=variant_name,
                            month=month,
                            top_frac=top_frac,
                            selected_idx=selected_idx,
                            base_fields={
                                "label_arm": label_arm,
                                "weight_arm": weight_arm,
                                "arm": row["arm"],
                                "model_feature_selector": model_feature_selector,
                            },
                        )
                    )
                    if variant_name in candidate_ledger_selector_names:
                        candidate_ledger.extend(
                            _candidate_ledger_rows(
                                frame=valid,
                                metrics=valid_metrics,
                                score=variant_score,
                                selector=variant_name,
                                month=month,
                                top_frac=top_frac,
                                selected_idx=selected_idx,
                                base_fields={
                                    "label_arm": label_arm,
                                    "weight_arm": weight_arm,
                                    "arm": row["arm"],
                                    "model_feature_selector": model_feature_selector,
                                },
                                extra_scores={
                                    **_discovery_context_scores(valid_context, model_features),
                                    "base_model_score": score,
                                    "bad_mae_pred": bad_mae_pred,
                                    "timeout_pred": timeout_pred,
                                    "side_bad_mae_pred": side_bad_mae_pred,
                                    "side_timeout_pred": side_timeout_pred,
                                    "clean_path_pred": clean_path_pred,
                                    "feature_gap_risk": feature_gap_risk,
                                    "clean_dirty_positive_risk": clean_dirty_positive_risk,
                                    "lgbm_bad_mae_pred": lgbm_bad_mae_pred,
                                    "lgbm_timeout_pred": lgbm_timeout_pred,
                                    "lgbm_clean_path_pred": lgbm_clean_path_pred,
                                    "lgbm_dirty_positive_bad_mae_pred": (
                                        lgbm_dirty_positive_bad_mae_pred
                                    ),
                                    "lgbm_positive_clean_path_pred": (
                                        lgbm_positive_clean_path_pred
                                    ),
                                    "lgbm_side_dirty_positive_bad_mae_pred": (
                                        lgbm_side_dirty_positive_bad_mae_pred
                                    ),
                                    "lgbm_side_positive_clean_path_pred": (
                                        lgbm_side_positive_clean_path_pred
                                    ),
                                    "s22_bucket_quality_score": s22_bucket_quality_score,
                                    "s22_bucket_quality_rank_pct": (
                                        s22_bucket_quality_rank_pct
                                    ),
                                    "s22_bucket_relaxed_pass_count": (
                                        s22_bucket_relaxed_pass_count
                                    ),
                                    "s22_bucket_strict_pass_count": (
                                        s22_bucket_strict_pass_count
                                    ),
                                    "s46_bucket_quality_score": s46_bucket_quality_score,
                                    "s46_bucket_quality_rank_pct": (
                                        s46_bucket_quality_rank_pct
                                    ),
                                    "lgbm_bad_mae_ts_pct": lgbm_bad_mae_ts_pct,
                                    "lgbm_timeout_ts_pct": lgbm_timeout_ts_pct,
                                    "lgbm_clean_path_ts_pct": lgbm_clean_path_ts_pct,
                                    "lgbm_dirty_positive_bad_mae_ts_pct": (
                                        lgbm_dirty_positive_bad_mae_ts_pct
                                    ),
                                    "lgbm_positive_clean_path_ts_pct": (
                                        lgbm_positive_clean_path_ts_pct
                                    ),
                                    "lgbm_side_dirty_positive_bad_mae_ts_pct": (
                                        lgbm_side_dirty_positive_bad_mae_ts_pct
                                    ),
                                    "lgbm_side_positive_clean_path_ts_pct": (
                                        lgbm_side_positive_clean_path_ts_pct
                                    ),
                                    "lgbm_ranker_score": lgbm_ranker_score,
                                    "lgbm_path_ranker_score": lgbm_path_ranker_score,
                                    "lgbm_oracle_ranker_score": lgbm_oracle_ranker_score,
                                    "lgbm_clean_oracle_ranker_score": (
                                        lgbm_clean_oracle_ranker_score
                                    ),
                                    "lgbm_path_first_ranker_score": (
                                        lgbm_path_first_ranker_score
                                    ),
                                    "lgbm_path_first_dirty_zero_ranker_score": (
                                        lgbm_path_first_dirty_zero_ranker_score
                                    ),
                                    "lgbm_s24_broad_path_first_ranker_score": (
                                        lgbm_s24_broad_path_first_ranker_score
                                    ),
                                    "lgbm_s24_broad_path_first_dirty_zero_ranker_score": (
                                        lgbm_s24_broad_path_first_dirty_zero_ranker_score
                                    ),
                                    "lgbm_s28_side_s24_ranker_score": (
                                        lgbm_s28_side_s24_ranker_score
                                    ),
                                    "lgbm_s28_side_s24_dirty_zero_ranker_score": (
                                        lgbm_s28_side_s24_dirty_zero_ranker_score
                                    ),
                                    "lgbm_timeout_aware_clean_ranker_score": (
                                        lgbm_timeout_aware_clean_ranker_score
                                    ),
                                    "lgbm_s30_side_asym_ranker_score": (
                                        lgbm_s30_side_asym_ranker_score
                                    ),
                                    "lgbm_s30_side_asym_dirty_zero_ranker_score": (
                                        lgbm_s30_side_asym_dirty_zero_ranker_score
                                    ),
                                    "lgbm_s42_side_interaction_dirty_zero_ranker_score": (
                                        lgbm_s42_side_interaction_dirty_zero_ranker_score
                                    ),
                                    "lgbm_s44_side_interaction_sign_calibrated_ranker_score": (
                                        lgbm_s44_side_interaction_sign_calibrated_ranker_score
                                    ),
                                    "lgbm_s45_side_interaction_roll45_dirty_zero_ranker_score": (
                                        lgbm_s45_side_interaction_roll45_dirty_zero_ranker_score
                                    ),
                                },
                            )
                        )
            diagnostics.append(
                {
                    "period": month,
                    "label_arm": label_arm,
                    "weight_arm": weight_arm,
                    "model_feature_selector": model_feature_selector,
                    "model_feature_count": int(len(model_features)),
                    "model_features": ",".join(model_features),
                    **feature_selector_diag,
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": int(valid_mask.sum()),
                    "target_train_mean": _safe_mean(target_train["target_soft"]),
                    "target_train_std": float(
                        pd.to_numeric(target_train["target_soft"], errors="coerce").std(ddof=0)
                    ),
                    "target_train_hard_rate": _safe_mean(target_train["target_hard"]),
                    "weight_mean": _safe_mean(weights),
                    "weight_p90": _safe_quantile(weights, 0.90),
                    "weight_p99": _safe_quantile(weights, 0.99),
                    "weight_effective_n": _effective_sample_size(weights),
                    "weight_effective_frac": _effective_sample_size(weights) / float(len(weights))
                    if len(weights)
                    else float("nan"),
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "seed_count": int(len(seeds)),
                    "prediction_seed_std_mean": float(np.mean(pred_seed_std)),
                    "prediction_seed_std_p90": float(np.percentile(pred_seed_std, 90)),
                    "include_risk_selector_variants": bool(include_risk_selector_variants),
                    "pred_bad_mae_mean": _safe_mean(bad_mae_pred),
                    "pred_bad_mae_ic": _spearman(
                        bad_mae_pred,
                        (valid_metrics["mae_norm"] >= 1.0).astype(float),
                    ),
                    "pred_timeout_mean": _safe_mean(timeout_pred),
                    "pred_timeout_ic": _spearman(
                        timeout_pred,
                        valid_metrics["is_timeout"].astype(float),
                    ),
                    "side_pred_bad_mae_mean": _safe_mean(side_bad_mae_pred),
                    "side_pred_bad_mae_ic": _spearman(
                        side_bad_mae_pred,
                        (valid_metrics["mae_norm"] >= 1.0).astype(float),
                    ),
                    "side_pred_timeout_mean": _safe_mean(side_timeout_pred),
                    "side_pred_timeout_ic": _spearman(
                        side_timeout_pred,
                        valid_metrics["is_timeout"].astype(float),
                    ),
                    "clean_path_pred_mean": _safe_mean(clean_path_pred),
                    "clean_path_pred_ic": _spearman(
                        clean_path_pred,
                        (
                            (valid_metrics["u_policy_net"] > 0.0)
                            & (valid_metrics["mae_norm"] < 1.0)
                            & (valid_metrics["is_timeout"].astype(float) <= 0.0)
                        ).astype(float),
                    ),
                    "feature_gap_risk_mean": _safe_mean(feature_gap_risk),
                    "clean_dirty_positive_risk_mean": _safe_mean(
                        clean_dirty_positive_risk
                    ),
                    "lgbm_bad_mae_pred_mean": _safe_mean(lgbm_bad_mae_pred),
                    "lgbm_bad_mae_pred_ic": _spearman(
                        lgbm_bad_mae_pred,
                        (valid_metrics["mae_norm"] >= 1.0).astype(float),
                    ),
                    "lgbm_timeout_pred_mean": _safe_mean(lgbm_timeout_pred),
                    "lgbm_timeout_pred_ic": _spearman(
                        lgbm_timeout_pred,
                        valid_metrics["is_timeout"].astype(float),
                    ),
                    "lgbm_clean_path_pred_mean": _safe_mean(lgbm_clean_path_pred),
                    "lgbm_clean_path_pred_ic": _spearman(
                        lgbm_clean_path_pred,
                        (
                            (valid_metrics["u_policy_net"] > 0.0)
                            & (valid_metrics["mae_norm"] < 1.0)
                            & (valid_metrics["is_timeout"].astype(float) <= 0.0)
                        ).astype(float),
                    ),
                    "lgbm_dirty_positive_bad_mae_pred_mean": _safe_mean(
                        lgbm_dirty_positive_bad_mae_pred
                    ),
                    "lgbm_dirty_positive_bad_mae_pred_ic": _spearman(
                        lgbm_dirty_positive_bad_mae_pred,
                        (
                            (valid_metrics["u_policy_net"] > 0.0)
                            & (valid_metrics["mae_norm"] >= 1.0)
                        ).astype(float),
                    ),
                    "lgbm_side_dirty_positive_bad_mae_pred_mean": _safe_mean(
                        lgbm_side_dirty_positive_bad_mae_pred
                    ),
                    "lgbm_side_dirty_positive_bad_mae_pred_ic": _spearman(
                        lgbm_side_dirty_positive_bad_mae_pred,
                        (
                            (valid_metrics["u_policy_net"] > 0.0)
                            & (valid_metrics["mae_norm"] >= 1.0)
                        ).astype(float),
                    ),
                    "lgbm_side_positive_clean_path_pred_mean": _safe_mean(
                        lgbm_side_positive_clean_path_pred
                    ),
                    "lgbm_side_positive_clean_path_pred_ic": _spearman(
                        lgbm_side_positive_clean_path_pred,
                        (
                            (valid_metrics["u_policy_net"] > 0.0)
                            & (valid_metrics["mae_norm"] < 1.0)
                            & (valid_metrics["is_timeout"].astype(float) <= 0.0)
                        ).astype(float),
                    ),
                    "lgbm_bad_mae_status": lgbm_bad_mae_status,
                    "lgbm_timeout_status": lgbm_timeout_status,
                    "lgbm_clean_path_status": lgbm_clean_path_status,
                    "lgbm_dirty_positive_bad_mae_status": (
                        lgbm_dirty_positive_bad_mae_status
                    ),
                    "lgbm_positive_clean_path_status": (
                        lgbm_positive_clean_path_status
                    ),
                    "lgbm_side_dirty_positive_bad_mae_status": (
                        lgbm_side_dirty_positive_bad_mae_status
                    ),
                    "lgbm_side_positive_clean_path_status": (
                        lgbm_side_positive_clean_path_status
                    ),
                    "lgbm_ranker_status": lgbm_ranker_status,
                    "lgbm_path_ranker_status": lgbm_path_ranker_status,
                    "lgbm_oracle_ranker_status": lgbm_oracle_ranker_status,
                    "lgbm_clean_oracle_ranker_status": lgbm_clean_oracle_ranker_status,
                    "lgbm_path_first_ranker_status": lgbm_path_first_ranker_status,
                    "lgbm_path_first_dirty_zero_ranker_status": (
                        lgbm_path_first_dirty_zero_ranker_status
                    ),
                    "lgbm_s24_broad_path_first_ranker_status": (
                        lgbm_s24_broad_path_first_ranker_status
                    ),
                    "lgbm_s24_broad_path_first_dirty_zero_ranker_status": (
                        lgbm_s24_broad_path_first_dirty_zero_ranker_status
                    ),
                    "lgbm_s28_side_s24_ranker_status": (
                        lgbm_s28_side_s24_ranker_status
                    ),
                    "lgbm_s28_side_s24_dirty_zero_ranker_status": (
                        lgbm_s28_side_s24_dirty_zero_ranker_status
                    ),
                    "lgbm_s30_side_asym_ranker_status": (
                        lgbm_s30_side_asym_ranker_status
                    ),
                    "lgbm_s30_side_asym_dirty_zero_ranker_status": (
                        lgbm_s30_side_asym_dirty_zero_ranker_status
                    ),
                    "lgbm_s42_side_interaction_dirty_zero_ranker_status": (
                        lgbm_s42_side_interaction_dirty_zero_ranker_status
                    ),
                    "lgbm_ranker_score_mean": _safe_mean(lgbm_ranker_score),
                    "lgbm_ranker_score_ic": _spearman(
                        lgbm_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    "lgbm_path_ranker_score_mean": _safe_mean(lgbm_path_ranker_score),
                    "lgbm_path_ranker_score_ic": _spearman(
                        lgbm_path_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    "lgbm_oracle_ranker_score_mean": _safe_mean(lgbm_oracle_ranker_score),
                    "lgbm_oracle_ranker_score_ic": _spearman(
                        lgbm_oracle_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    "lgbm_clean_oracle_ranker_score_mean": _safe_mean(
                        lgbm_clean_oracle_ranker_score
                    ),
                    "lgbm_clean_oracle_ranker_score_ic": _spearman(
                        lgbm_clean_oracle_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    "lgbm_path_first_ranker_score_mean": _safe_mean(
                        lgbm_path_first_ranker_score
                    ),
                    "lgbm_path_first_ranker_score_ic": _spearman(
                        lgbm_path_first_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    "lgbm_path_first_dirty_zero_ranker_score_mean": _safe_mean(
                        lgbm_path_first_dirty_zero_ranker_score
                    ),
                    "lgbm_path_first_dirty_zero_ranker_score_ic": _spearman(
                        lgbm_path_first_dirty_zero_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    "lgbm_s24_broad_path_first_ranker_score_mean": _safe_mean(
                        lgbm_s24_broad_path_first_ranker_score
                    ),
                    "lgbm_s24_broad_path_first_ranker_score_ic": _spearman(
                        lgbm_s24_broad_path_first_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    "lgbm_s24_broad_path_first_dirty_zero_ranker_score_mean": (
                        _safe_mean(lgbm_s24_broad_path_first_dirty_zero_ranker_score)
                    ),
                    "lgbm_s24_broad_path_first_dirty_zero_ranker_score_ic": _spearman(
                        lgbm_s24_broad_path_first_dirty_zero_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    "lgbm_s28_side_s24_ranker_score_mean": _safe_mean(
                        lgbm_s28_side_s24_ranker_score
                    ),
                    "lgbm_s28_side_s24_ranker_score_ic": _spearman(
                        lgbm_s28_side_s24_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    "lgbm_s28_side_s24_dirty_zero_ranker_score_mean": (
                        _safe_mean(lgbm_s28_side_s24_dirty_zero_ranker_score)
                    ),
                    "lgbm_s28_side_s24_dirty_zero_ranker_score_ic": _spearman(
                        lgbm_s28_side_s24_dirty_zero_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    "lgbm_s30_side_asym_ranker_score_mean": _safe_mean(
                        lgbm_s30_side_asym_ranker_score
                    ),
                    "lgbm_s30_side_asym_ranker_score_ic": _spearman(
                        lgbm_s30_side_asym_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    "lgbm_s30_side_asym_dirty_zero_ranker_score_mean": (
                        _safe_mean(lgbm_s30_side_asym_dirty_zero_ranker_score)
                    ),
                    "lgbm_s30_side_asym_dirty_zero_ranker_score_ic": _spearman(
                        lgbm_s30_side_asym_dirty_zero_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    "lgbm_s42_side_interaction_dirty_zero_ranker_score_mean": (
                        _safe_mean(lgbm_s42_side_interaction_dirty_zero_ranker_score)
                    ),
                    "lgbm_s42_side_interaction_dirty_zero_ranker_score_ic": _spearman(
                        lgbm_s42_side_interaction_dirty_zero_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    **s42_interaction_diag,
                    "lgbm_s44_side_interaction_sign_calibrated_ranker_score_mean": (
                        _safe_mean(lgbm_s44_side_interaction_sign_calibrated_ranker_score)
                    ),
                    "lgbm_s44_side_interaction_sign_calibrated_ranker_score_ic": _spearman(
                        lgbm_s44_side_interaction_sign_calibrated_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    **s44_sign_calibration_diag,
                    "lgbm_s45_side_interaction_roll45_dirty_zero_ranker_status": (
                        lgbm_s45_side_interaction_roll45_dirty_zero_ranker_status
                    ),
                    "lgbm_s45_side_interaction_roll45_dirty_zero_ranker_score_mean": (
                        _safe_mean(lgbm_s45_side_interaction_roll45_dirty_zero_ranker_score)
                    ),
                    "lgbm_s45_side_interaction_roll45_dirty_zero_ranker_score_ic": _spearman(
                        lgbm_s45_side_interaction_roll45_dirty_zero_ranker_score,
                        valid_metrics["u_policy_net"],
                    ),
                    **s45_recent_train_diag,
                    "s46_bucket_quality_score_mean": _safe_mean(s46_bucket_quality_score),
                    "s46_bucket_quality_rank_pct_mean": _safe_mean(
                        s46_bucket_quality_rank_pct
                    ),
                    "s46_bucket_quality_score_ic": _spearman(
                        s46_bucket_quality_score,
                        valid_metrics["u_policy_net"],
                    ),
                    **s46_bucket_quality_diag,
                    "feature_gap_risk_ic_bad_mae": _spearman(
                        feature_gap_risk,
                        (valid_metrics["mae_norm"] >= 1.0).astype(float),
                    ),
                    "feature_gap_risk_ic_timeout": _spearman(
                        feature_gap_risk,
                        valid_metrics["is_timeout"].astype(float),
                    ),
                    "clean_dirty_positive_risk_ic_bad_mae": _spearman(
                        clean_dirty_positive_risk,
                        (valid_metrics["mae_norm"] >= 1.0).astype(float),
                    ),
                    "clean_dirty_positive_risk_ic_timeout": _spearman(
                        clean_dirty_positive_risk,
                        valid_metrics["is_timeout"].astype(float),
                    ),
                    **feature_gap_diag,
                    **clean_dirty_positive_diag,
                    "score_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
                    "score_ic_label": _spearman(score, target_valid["target_soft"]),
                    **_decile_diagnostics(score, valid_metrics["u_policy_net"]),
                    **_timestamp_ranking_metrics(
                        frame=valid,
                        metrics=valid_metrics,
                        target=target_valid,
                        score=score,
                    ),
                }
            )
    return rows, diagnostics, clean_dirty_diagnostics, candidate_ledger


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    rows: list[dict[str, Any]] = []
    groups = monthly.groupby(
        ["arm", "label_arm", "weight_arm", "selector_variant", "model_feature_selector", "top_frac"],
        dropna=False,
        observed=True,
    )
    for key, group in groups:
        arm, label_arm, weight_arm, selector_variant, model_feature_selector, top_frac = key
        mean_u = pd.to_numeric(group["mean_u"], errors="coerce")
        selected_rows = pd.to_numeric(group["selected_rows"], errors="coerce")
        q10 = pd.to_numeric(group["q10_u"], errors="coerce")
        score_ic_u = pd.to_numeric(group["score_ic_u"], errors="coerce")
        selected_long_share = pd.to_numeric(group["selected_long_share"], errors="coerce")
        selected_short_share = pd.to_numeric(group["selected_short_share"], errors="coerce")
        active_months = int((selected_rows > 0).sum())
        no_trade_months = int(group["period"].nunique()) - active_months
        selected_max_side_share = pd.concat(
            [selected_long_share, selected_short_share],
            axis=1,
        ).max(axis=1)
        worst_month = float(mean_u.min()) if len(mean_u.dropna()) else float("nan")
        positive_months = int((mean_u > 0.0).sum())
        rows.append(
            {
                "arm": arm,
                "label_arm": label_arm,
                "weight_arm": weight_arm,
                "selector_variant": selector_variant,
                "model_feature_selector": model_feature_selector,
                "top_frac": float(top_frac),
                "months": int(group["period"].nunique()),
                "active_selected_months": active_months,
                "no_trade_months": no_trade_months,
                "no_trade_month_share": (
                    no_trade_months / float(group["period"].nunique())
                    if int(group["period"].nunique())
                    else float("nan")
                ),
                "positive_months": positive_months,
                "mean_u": _safe_mean(mean_u),
                "worst_month_mean_u": worst_month,
                "hit_u": _safe_mean(group["hit_u"]),
                "q10_u": _safe_mean(q10),
                "delta_mean_u_vs_period": _safe_mean(group["delta_mean_u_vs_period"]),
                "delta_hit_u_vs_period": _safe_mean(group["delta_hit_u_vs_period"]),
                "delta_q10_u_vs_period": _safe_mean(group["delta_q10_u_vs_period"]),
                "score_ic_u": _safe_mean(score_ic_u),
                "score_ic_label": _safe_mean(group["score_ic_label"]),
                "decile_spearman_u": _safe_mean(group["decile_spearman_u"]),
                "decile_violations_u": _safe_mean(group["decile_violations_u"]),
                "top_bottom_decile_spread_u": _safe_mean(group["top_bottom_decile_spread_u"]),
                "ts_rank_hr10_u": _safe_mean(group["ts_rank_hr10_u"]),
                "ts_rank_hr20_u": _safe_mean(group["ts_rank_hr20_u"]),
                "ts_rank_hr30_u": _safe_mean(group["ts_rank_hr30_u"]),
                "ts_rank_target_hr30": _safe_mean(group["ts_rank_target_hr30"]),
                "ts_rank_mean_u30": _safe_mean(group["ts_rank_mean_u30"]),
                "ts_rank_q05_u30": _safe_mean(group["ts_rank_q05_u30"]),
                "ts_rank_ndcg30_u": _safe_mean(group["ts_rank_ndcg30_u"]),
                "ts_rank_ndcg30_opportunity_rate": _safe_mean(
                    group["ts_rank_ndcg30_opportunity_rate"]
                ),
                "ts_rank_week_hr30_q05": _safe_mean(group["ts_rank_week_hr30_q05"]),
                "ts_rank_week_hr30_q10": _safe_mean(group["ts_rank_week_hr30_q10"]),
                "ts_rank_week_hr30_q25": _safe_mean(group["ts_rank_week_hr30_q25"]),
                "ts_rank_week_hr30_q50": _safe_mean(group["ts_rank_week_hr30_q50"]),
                "ts_rank_week_hr30_q75": _safe_mean(group["ts_rank_week_hr30_q75"]),
                "ts_rank_top30_bad_mae_1r_rate": _safe_mean(
                    group["ts_rank_top30_bad_mae_1r_rate"]
                ),
                "ts_rank_top30_wide_barrier_25bps_rate": _safe_mean(
                    group["ts_rank_top30_wide_barrier_25bps_rate"]
                ),
                "ts_rank_top30_timeout_rate": _safe_mean(group["ts_rank_top30_timeout_rate"]),
                "mean_ts_rank_top30_rows": _safe_mean(group["ts_rank_top30_rows"]),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "clean_positive_rate": _safe_mean(group["clean_positive_rate"])
                if "clean_positive_rate" in group.columns
                else float("nan"),
                "dirty_positive_rate": _safe_mean(group["dirty_positive_rate"])
                if "dirty_positive_rate" in group.columns
                else float("nan"),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "wide_barrier_35bps_rate": _safe_mean(group["wide_barrier_35bps_rate"]),
                "timeout_rate": _safe_mean(group["timeout_rate"]),
                "selected_long_share": _safe_mean(selected_long_share),
                "selected_short_share": _safe_mean(selected_short_share),
                "max_selected_side_share": _safe_mean(selected_max_side_share),
                "worst_month_selected_side_share": _safe_quantile(selected_max_side_share, 1.0),
                "selected_pred_bad_mae_mean": _safe_mean(group["selected_pred_bad_mae_mean"]),
                "selected_pred_timeout_mean": _safe_mean(group["selected_pred_timeout_mean"]),
                "pred_bad_mae_mean": _safe_mean(group["pred_bad_mae_mean"]),
                "pred_timeout_mean": _safe_mean(group["pred_timeout_mean"]),
                "lgbm_bad_mae_pred_mean": _safe_mean(group["lgbm_bad_mae_pred_mean"])
                if "lgbm_bad_mae_pred_mean" in group.columns
                else float("nan"),
                "lgbm_timeout_pred_mean": _safe_mean(group["lgbm_timeout_pred_mean"])
                if "lgbm_timeout_pred_mean" in group.columns
                else float("nan"),
                "lgbm_clean_path_pred_mean": _safe_mean(group["lgbm_clean_path_pred_mean"])
                if "lgbm_clean_path_pred_mean" in group.columns
                else float("nan"),
                "lgbm_dirty_positive_bad_mae_pred_mean": _safe_mean(
                    group["lgbm_dirty_positive_bad_mae_pred_mean"]
                )
                if "lgbm_dirty_positive_bad_mae_pred_mean" in group.columns
                else float("nan"),
                "lgbm_side_dirty_positive_bad_mae_pred_mean": _safe_mean(
                    group["lgbm_side_dirty_positive_bad_mae_pred_mean"]
                )
                if "lgbm_side_dirty_positive_bad_mae_pred_mean" in group.columns
                else float("nan"),
                "lgbm_side_positive_clean_path_pred_mean": _safe_mean(
                    group["lgbm_side_positive_clean_path_pred_mean"]
                )
                if "lgbm_side_positive_clean_path_pred_mean" in group.columns
                else float("nan"),
                "selected_lgbm_bad_mae_pred_mean": _safe_mean(
                    group["selected_lgbm_bad_mae_pred_mean"]
                )
                if "selected_lgbm_bad_mae_pred_mean" in group.columns
                else float("nan"),
                "selected_lgbm_timeout_pred_mean": _safe_mean(
                    group["selected_lgbm_timeout_pred_mean"]
                )
                if "selected_lgbm_timeout_pred_mean" in group.columns
                else float("nan"),
                "selected_lgbm_clean_path_pred_mean": _safe_mean(
                    group["selected_lgbm_clean_path_pred_mean"]
                )
                if "selected_lgbm_clean_path_pred_mean" in group.columns
                else float("nan"),
                "selected_lgbm_dirty_positive_bad_mae_pred_mean": _safe_mean(
                    group["selected_lgbm_dirty_positive_bad_mae_pred_mean"]
                )
                if "selected_lgbm_dirty_positive_bad_mae_pred_mean" in group.columns
                else float("nan"),
                "selected_lgbm_side_dirty_positive_bad_mae_pred_mean": _safe_mean(
                    group["selected_lgbm_side_dirty_positive_bad_mae_pred_mean"]
                )
                if "selected_lgbm_side_dirty_positive_bad_mae_pred_mean" in group.columns
                else float("nan"),
                "selected_lgbm_side_positive_clean_path_pred_mean": _safe_mean(
                    group["selected_lgbm_side_positive_clean_path_pred_mean"]
                )
                if "selected_lgbm_side_positive_clean_path_pred_mean" in group.columns
                else float("nan"),
                "selected_lgbm_bad_mae_ts_pct_mean": _safe_mean(
                    group["selected_lgbm_bad_mae_ts_pct_mean"]
                )
                if "selected_lgbm_bad_mae_ts_pct_mean" in group.columns
                else float("nan"),
                "selected_lgbm_timeout_ts_pct_mean": _safe_mean(
                    group["selected_lgbm_timeout_ts_pct_mean"]
                )
                if "selected_lgbm_timeout_ts_pct_mean" in group.columns
                else float("nan"),
                "selected_lgbm_clean_path_ts_pct_mean": _safe_mean(
                    group["selected_lgbm_clean_path_ts_pct_mean"]
                )
                if "selected_lgbm_clean_path_ts_pct_mean" in group.columns
                else float("nan"),
                "selected_lgbm_dirty_positive_bad_mae_ts_pct_mean": _safe_mean(
                    group["selected_lgbm_dirty_positive_bad_mae_ts_pct_mean"]
                )
                if "selected_lgbm_dirty_positive_bad_mae_ts_pct_mean" in group.columns
                else float("nan"),
                "selected_lgbm_side_dirty_positive_bad_mae_ts_pct_mean": _safe_mean(
                    group["selected_lgbm_side_dirty_positive_bad_mae_ts_pct_mean"]
                )
                if "selected_lgbm_side_dirty_positive_bad_mae_ts_pct_mean" in group.columns
                else float("nan"),
                "selected_lgbm_side_positive_clean_path_ts_pct_mean": _safe_mean(
                    group["selected_lgbm_side_positive_clean_path_ts_pct_mean"]
                )
                if "selected_lgbm_side_positive_clean_path_ts_pct_mean" in group.columns
                else float("nan"),
                "feature_gap_risk_mean": _safe_mean(group["feature_gap_risk_mean"])
                if "feature_gap_risk_mean" in group.columns
                else float("nan"),
                "selected_feature_gap_risk_mean": _safe_mean(
                    group["selected_feature_gap_risk_mean"]
                )
                if "selected_feature_gap_risk_mean" in group.columns
                else float("nan"),
                "clean_dirty_positive_risk_mean": _safe_mean(
                    group["clean_dirty_positive_risk_mean"]
                )
                if "clean_dirty_positive_risk_mean" in group.columns
                else float("nan"),
                "selected_clean_dirty_positive_risk_mean": _safe_mean(
                    group["selected_clean_dirty_positive_risk_mean"]
                )
                if "selected_clean_dirty_positive_risk_mean" in group.columns
                else float("nan"),
                "s22_bucket_quality_score_mean": _safe_mean(
                    group["s22_bucket_quality_score_mean"]
                )
                if "s22_bucket_quality_score_mean" in group.columns
                else float("nan"),
                "s22_bucket_quality_rank_pct_mean": _safe_mean(
                    group["s22_bucket_quality_rank_pct_mean"]
                )
                if "s22_bucket_quality_rank_pct_mean" in group.columns
                else float("nan"),
                "s22_bucket_relaxed_pass_count_mean": _safe_mean(
                    group["s22_bucket_relaxed_pass_count_mean"]
                )
                if "s22_bucket_relaxed_pass_count_mean" in group.columns
                else float("nan"),
                "s22_bucket_strict_pass_count_mean": _safe_mean(
                    group["s22_bucket_strict_pass_count_mean"]
                )
                if "s22_bucket_strict_pass_count_mean" in group.columns
                else float("nan"),
                "selected_s22_bucket_quality_score_mean": _safe_mean(
                    group["selected_s22_bucket_quality_score_mean"]
                )
                if "selected_s22_bucket_quality_score_mean" in group.columns
                else float("nan"),
                "selected_s22_bucket_quality_rank_pct_mean": _safe_mean(
                    group["selected_s22_bucket_quality_rank_pct_mean"]
                )
                if "selected_s22_bucket_quality_rank_pct_mean" in group.columns
                else float("nan"),
                "selected_s22_bucket_relaxed_pass_count_mean": _safe_mean(
                    group["selected_s22_bucket_relaxed_pass_count_mean"]
                )
                if "selected_s22_bucket_relaxed_pass_count_mean" in group.columns
                else float("nan"),
                "selected_s22_bucket_strict_pass_count_mean": _safe_mean(
                    group["selected_s22_bucket_strict_pass_count_mean"]
                )
                if "selected_s22_bucket_strict_pass_count_mean" in group.columns
                else float("nan"),
                "hard_risk_cap_no_trade_rate": _safe_mean(group["hard_risk_cap_no_trade_rate"])
                if "hard_risk_cap_no_trade_rate" in group.columns
                else float("nan"),
                "stageA_candidate_rows": _safe_mean(group["stageA_candidate_rows"])
                if "stageA_candidate_rows" in group.columns
                else float("nan"),
                "stageA_candidate_row_share": _safe_mean(group["stageA_candidate_row_share"])
                if "stageA_candidate_row_share" in group.columns
                else float("nan"),
                "stageA_candidate_oracle_recall": _safe_mean(
                    group["stageA_candidate_oracle_recall"]
                )
                if "stageA_candidate_oracle_recall" in group.columns
                else float("nan"),
                "stageA_candidate_long_oracle_recall": _safe_mean(
                    group["stageA_candidate_long_oracle_recall"]
                )
                if "stageA_candidate_long_oracle_recall" in group.columns
                else float("nan"),
                "stageA_candidate_short_oracle_recall": _safe_mean(
                    group["stageA_candidate_short_oracle_recall"]
                )
                if "stageA_candidate_short_oracle_recall" in group.columns
                else float("nan"),
                "stageA_candidate_bad_mae_1r_rate": _safe_mean(
                    group["stageA_candidate_bad_mae_1r_rate"]
                )
                if "stageA_candidate_bad_mae_1r_rate" in group.columns
                else float("nan"),
                "stageA_candidate_timeout_rate": _safe_mean(
                    group["stageA_candidate_timeout_rate"]
                )
                if "stageA_candidate_timeout_rate" in group.columns
                else float("nan"),
                "final_rows": _safe_mean(group["final_rows"])
                if "final_rows" in group.columns
                else float("nan"),
                "final_oracle_recall": _safe_mean(group["final_oracle_recall"])
                if "final_oracle_recall" in group.columns
                else float("nan"),
                "final_long_oracle_recall": _safe_mean(group["final_long_oracle_recall"])
                if "final_long_oracle_recall" in group.columns
                else float("nan"),
                "final_short_oracle_recall": _safe_mean(group["final_short_oracle_recall"])
                if "final_short_oracle_recall" in group.columns
                else float("nan"),
                "final_bad_mae_1r_rate": _safe_mean(group["final_bad_mae_1r_rate"])
                if "final_bad_mae_1r_rate" in group.columns
                else float("nan"),
                "final_timeout_rate": _safe_mean(group["final_timeout_rate"])
                if "final_timeout_rate" in group.columns
                else float("nan"),
                "feature_gap_risk_features": str(group["feature_gap_risk_features"].dropna().iloc[-1])
                if "feature_gap_risk_features" in group.columns
                and group["feature_gap_risk_features"].dropna().size
                else "",
                "clean_dirty_positive_risk_features": str(
                    group["clean_dirty_positive_risk_features"].dropna().iloc[-1]
                )
                if "clean_dirty_positive_risk_features" in group.columns
                and group["clean_dirty_positive_risk_features"].dropna().size
                else "",
                "s22_bucket_quality_features": str(
                    group["s22_bucket_quality_features"].dropna().iloc[-1]
                )
                if "s22_bucket_quality_features" in group.columns
                and group["s22_bucket_quality_features"].dropna().size
                else "",
                "score_ic_bad_mae": _safe_mean(group["score_ic_bad_mae"]),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
                "mean_selected_rows": _safe_mean(selected_rows),
                "min_selected_rows": int(selected_rows.min()) if len(selected_rows.dropna()) else 0,
                "mean_model_feature_count": _safe_mean(group["model_feature_count"]),
                "model_features": str(group["model_features"].dropna().iloc[0])
                if group["model_features"].dropna().size
                else "",
                "decision": (
                    "promote_to_full_walkforward_test"
                    if positive_months >= 3
                    and _safe_mean(mean_u) > 0.0
                    and math.isfinite(worst_month)
                    and worst_month > 0.0
                    and _safe_mean(score_ic_u) > 0.0
                    and _safe_mean(group["wide_barrier_25bps_rate"]) <= 0.02
                    else "reject_or_rework"
                ),
            }
        )
    aggregate = pd.DataFrame(rows)
    if not aggregate.empty:
        aggregate["plan_rank_score"] = (
            1.00 * pd.to_numeric(aggregate["ts_rank_hr30_u"], errors="coerce")
            + 0.50 * pd.to_numeric(aggregate["ts_rank_ndcg30_u"], errors="coerce")
            + 0.35 * pd.to_numeric(aggregate["ts_rank_hr20_u"], errors="coerce")
            + 0.20 * pd.to_numeric(aggregate["ts_rank_hr10_u"], errors="coerce")
            + 0.25 * pd.to_numeric(aggregate["ts_rank_week_hr30_q25"], errors="coerce")
            + 0.15 * pd.to_numeric(aggregate["ts_rank_week_hr30_q10"], errors="coerce")
        )
        aggregate["plan_rank_score_penalized"] = (
            aggregate["plan_rank_score"]
            - 0.25 * pd.to_numeric(aggregate["ts_rank_top30_bad_mae_1r_rate"], errors="coerce")
            - 0.15 * pd.to_numeric(aggregate["ts_rank_top30_timeout_rate"], errors="coerce")
            - 0.50 * pd.to_numeric(aggregate["wide_barrier_25bps_rate"], errors="coerce")
        )
    return aggregate.sort_values(
        ["top_frac", "mean_u", "worst_month_mean_u"],
        ascending=[True, False, False],
    )


def _write_markdown(output_dir: Path, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "label_feature_store_model_smoke.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "decision",
        "model_feature_selector",
        "selector_variant",
        "arm",
        "months",
        "positive_months",
        "mean_u",
        "worst_month_mean_u",
        "hit_u",
        "q10_u",
        "delta_mean_u_vs_period",
        "score_ic_u",
        "decile_spearman_u",
        "ts_rank_hr10_u",
        "ts_rank_hr20_u",
        "ts_rank_hr30_u",
        "ts_rank_ndcg30_u",
        "ts_rank_week_hr30_q25",
        "ts_rank_top30_timeout_rate",
        "plan_rank_score_penalized",
        "stageA_candidate_oracle_recall",
        "final_oracle_recall",
        "hard_risk_cap_no_trade_rate",
        "bad_mae_1r_rate",
        "clean_positive_rate",
        "dirty_positive_rate",
        "wide_barrier_25bps_rate",
        "wide_barrier_35bps_rate",
        "timeout_rate",
        "max_selected_side_share",
        "selected_pred_bad_mae_mean",
        "selected_lgbm_bad_mae_pred_mean",
        "selected_lgbm_clean_path_pred_mean",
        "selected_lgbm_dirty_positive_bad_mae_pred_mean",
        "selected_lgbm_bad_mae_ts_pct_mean",
        "selected_lgbm_clean_path_ts_pct_mean",
        "selected_lgbm_dirty_positive_bad_mae_ts_pct_mean",
        "selected_clean_dirty_positive_risk_mean",
        "mean_selected_rows",
        "min_selected_rows",
        "mean_model_feature_count",
        "top_symbol_share",
    ]
    lines = [
        "# Label Feature-Store Model Smoke",
        "",
        "Scope: cheap month-forward tree model over the joined feature-store features. This is not production LightGBM training or a clean final OOS claim.",
        "",
        f"Rows: `{manifest['rows']}`",
        f"Symbols: `{manifest['symbols']}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Periods: `{manifest['timestamp_min']}` to `{manifest['timestamp_max']}`",
        "",
    ]
    if "decision" in aggregate.columns:
        promoted = aggregate[aggregate["decision"].eq("promote_to_full_walkforward_test")]
    else:
        promoted = aggregate.iloc[:0].copy()
    lines.extend(["## Promotion Candidates", "", table(promoted, cols, limit=50), ""])
    for frac in manifest["top_fracs"]:
        if "top_frac" in aggregate.columns:
            subset = aggregate[aggregate["top_frac"].eq(frac)].copy()
            sort_cols = [col for col in ("mean_u", "worst_month_mean_u") if col in subset.columns]
            if sort_cols:
                subset = subset.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        else:
            subset = aggregate.iloc[:0].copy()
        frac_label = f"{frac:.1%}" if float(frac) < 0.01 else f"{frac:.0%}"
        lines.extend([f"## Top {frac_label}", "", table(subset, cols, limit=30), ""])
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Monthly: `{manifest['outputs']['monthly']}`",
            f"- Aggregate: `{manifest['outputs']['aggregate']}`",
            f"- Diagnostics: `{manifest['outputs']['diagnostics']}`",
            f"- Clean/dirty selected: `{manifest['outputs']['clean_dirty_selected']}`",
            f"- Candidate ledger: `{manifest['outputs']['candidate_ledger']}`",
            f"- Manifest: `{manifest['outputs']['manifest']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_smoke(
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
    model_feature_selector: str,
    model_feature_tail_frac: float,
    top_fracs: list[float],
    train_lookback_months: int | None,
    include_risk_selector_variants: bool = False,
    side_cap_max_share: float = 0.70,
    candidate_ledger_selector_names: list[str] | None = None,
    candidate_ledger_only: bool = False,
    candidate_ledger_fast_mode: bool = False,
    spread_baseline_path: Path | None = None,
    spread_rank_column: str = "p75_spread_bps",
    target_symbol_count: int | None = None,
    max_spread_bps: float | None = None,
    include_ae_gmm_state_features: bool = True,
    ae_gmm_state_feature_max_train_rows: int = DEFAULT_AE_GMM_STATE_FEATURE_MAX_TRAIN_ROWS,
    ae_gmm_state_feature_gmm_max_train_rows: int = DEFAULT_AE_GMM_STATE_FEATURE_GMM_MAX_TRAIN_ROWS,
    ae_gmm_state_feature_max_iter: int = DEFAULT_AE_GMM_STATE_FEATURE_MAX_ITER,
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
        feature_matrix = feature_matrix.astype(np.float32, copy=False).reset_index(drop=True)
        frame = pd.concat([frame.reset_index(drop=True), feature_matrix], axis=1, copy=False)

    metrics = _path_metrics(frame)
    evaluation_utility_source = _apply_evaluation_utility_column(frame, metrics, evaluation_utility_column)
    features = _feature_columns(frame)
    base_targets = _make_targets(frame, metrics)
    targets = _label_targets(frame, metrics)
    strict_rounda_base = {**base_targets, **targets}
    if {"S3_path_quality", "S8_timestamp_rank_path"}.issubset(strict_rounda_base):
        targets.update(
            _strict_rounda_targets(
                frame=frame,
                metrics=metrics,
                base_targets=strict_rounda_base,
            )
        )
    targets.update(_fixed_artifact_targets(frame, metrics))
    available_labels = set(targets)
    available_weights = set(WEIGHT_ARMS)
    if not label_arms:
        label_arms = (
            list(LABEL_ARMS)
            + [arm for arm in STRICT_ROUNDA_LABEL_ARMS if arm in available_labels]
            + [arm for arm in FIXED_ARTIFACT_LABEL_ARMS if arm in available_labels]
        )
    missing_labels = sorted(set(label_arms) - available_labels)
    missing_weights = sorted(set(weight_arms) - available_weights)
    allowed_feature_selectors = {"all", *PROXY_METHODS}
    if model_feature_selector not in allowed_feature_selectors:
        raise ValueError(f"Unknown model feature selector: {model_feature_selector}")
    if missing_labels:
        raise ValueError(f"Unknown label arms: {missing_labels}")
    if missing_weights:
        raise ValueError(f"Unknown weight arms: {missing_weights}")

    months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())
    monthly_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    clean_dirty_rows: list[dict[str, Any]] = []
    candidate_ledger_rows: list[dict[str, Any]] = []
    candidate_ledger_names = set(candidate_ledger_selector_names or [])
    for month in months[1:]:
        rows, diagnostics, clean_dirty, candidate_ledger = _run_month(
            frame=frame,
            metrics=metrics,
            targets=targets,
            features=features,
            month=month,
            label_arms=label_arms,
            weight_arms=weight_arms,
            seeds=seeds,
            model_feature_selector=model_feature_selector,
            model_feature_tail_frac=model_feature_tail_frac,
            top_fracs=top_fracs,
            train_lookback_months=train_lookback_months,
            include_risk_selector_variants=include_risk_selector_variants,
            side_cap_max_share=side_cap_max_share,
            candidate_ledger_selector_names=candidate_ledger_names,
            candidate_ledger_only=bool(candidate_ledger_only),
            candidate_ledger_fast_mode=bool(candidate_ledger_fast_mode),
            include_ae_gmm_state_features=bool(include_ae_gmm_state_features),
            ae_gmm_state_feature_max_train_rows=int(ae_gmm_state_feature_max_train_rows),
            ae_gmm_state_feature_gmm_max_train_rows=int(ae_gmm_state_feature_gmm_max_train_rows),
            ae_gmm_state_feature_max_iter=int(ae_gmm_state_feature_max_iter),
        )
        monthly_rows.extend(rows)
        diagnostic_rows.extend(diagnostics)
        clean_dirty_rows.extend(clean_dirty)
        candidate_ledger_rows.extend(candidate_ledger)

    monthly = pd.DataFrame(monthly_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    clean_dirty_diagnostics = pd.DataFrame(clean_dirty_rows)
    candidate_ledger = pd.DataFrame(candidate_ledger_rows)
    aggregate = _aggregate(monthly)

    paths = {
        "monthly": output_dir / "label_feature_store_model_smoke_monthly.csv",
        "aggregate": output_dir / "label_feature_store_model_smoke_aggregate.csv",
        "diagnostics": output_dir / "label_feature_store_model_smoke_diagnostics.csv",
        "clean_dirty_selected": output_dir
        / "label_feature_store_model_smoke_clean_dirty_selected.csv",
        "candidate_ledger": output_dir / "label_feature_store_model_smoke_candidate_ledger.csv",
        "symbol_universe": output_dir / "label_feature_store_model_smoke_symbol_universe.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    clean_dirty_diagnostics.to_csv(paths["clean_dirty_selected"], index=False)
    candidate_ledger.to_csv(paths["candidate_ledger"], index=False)
    symbol_universe.to_csv(paths["symbol_universe"], index=False)

    ae_gmm_feature_names = (
        _ae_gmm_smoke_feature_policy_columns(list(AE_GMM_FEATURE_COLUMNS))
        if bool(include_ae_gmm_state_features)
        else []
    )
    if bool(include_ae_gmm_state_features):
        ae_gmm_feature_names = list(dict.fromkeys([*ae_gmm_feature_names, "ae_gmm_oof_available"]))
    if bool(include_ae_gmm_state_features) and _side_context_enabled():
        ae_gmm_feature_names = list(ae_gmm_feature_names) + [
            f"{side_name}_{feature}"
            for side_name in ("long", "short")
            for feature in ae_gmm_feature_names
        ]
    ae_gmm_generated_feature_count = int(len(ae_gmm_feature_names))
    if bool(include_ae_gmm_state_features) and "ae_gmm_state_feature_count" in diagnostics.columns:
        observed = pd.to_numeric(
            diagnostics["ae_gmm_state_feature_count"],
            errors="coerce",
        ).dropna()
        if not observed.empty:
            ae_gmm_generated_feature_count = int(observed.max())

    manifest = {
        "scope": "cheap_feature_store_model_smoke_not_full_policy_training",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_count": int(len(features)),
        "raw_feature_count": int(len(features)),
        "ae_gmm_state_features": {
            "enabled": bool(include_ae_gmm_state_features),
            "generated_feature_count": ae_gmm_generated_feature_count,
            "feature_policy": str(AE_GMM_SMOKE_FEATURE_POLICY or "all"),
            "side_context_mode": str(AE_GMM_SIDE_CONTEXT_MODE or "off"),
            "side_context_enabled": bool(_side_context_enabled()),
            "train_feature_scope": "inner_chronological_oof"
            if bool(AE_GMM_CROSSFIT_TRAIN_FEATURES)
            else "outer_train_in_sample",
            "validation_feature_scope": "frozen_outer_train_artifact",
            "crossfit_train_features": bool(AE_GMM_CROSSFIT_TRAIN_FEATURES),
            "feature_names": ae_gmm_feature_names,
            "max_train_rows": int(ae_gmm_state_feature_max_train_rows),
            "gmm_max_train_rows": int(ae_gmm_state_feature_gmm_max_train_rows),
            "ae_max_iter": int(ae_gmm_state_feature_max_iter),
            "fit_scope": "prior_month_fold_only",
            "validation_transform": "pre_entry_features_plus_prior_fold_state",
        },
        "features": features,
        "label_arms": label_arms,
        "weight_arms": weight_arms,
        "top_fracs": [float(v) for v in top_fracs],
        "feature_store": feature_store_report,
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "symbol_universe_filter": symbol_universe_filter,
        "evaluation_utility_source": evaluation_utility_source,
        "max_feature_store_features": max_feature_store_features,
        "model": {
            "type": "ExtraTreesRegressor",
            "n_estimators": 96,
            "max_depth": 8,
            "min_samples_leaf": 40,
            "max_features": "sqrt",
            "seeds": [int(seed) for seed in seeds],
            "seed_count": int(len(seeds)),
            "feature_selector": model_feature_selector,
            "feature_selector_tail_frac": float(model_feature_tail_frac),
            "train_lookback_months": int(train_lookback_months)
            if train_lookback_months is not None
            else None,
            "include_risk_selector_variants": bool(include_risk_selector_variants),
            "side_cap_max_share": float(side_cap_max_share),
            "candidate_ledger_selector_names": sorted(candidate_ledger_names),
            "candidate_ledger_only": bool(candidate_ledger_only),
            "candidate_ledger_fast_mode": bool(candidate_ledger_fast_mode),
            "include_ae_gmm_state_features": bool(include_ae_gmm_state_features),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir, aggregate, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument(
        "--evaluation-utility-column",
        type=str,
        default=None,
        help="Optional label-frame column to use for mean_u/hit_u/ranking diagnostics.",
    )
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument(
        "--label-arms",
        type=str,
        default=",".join(DEFAULT_LABEL_ARMS),
        help="Comma-separated label arms, or 'all'.",
    )
    parser.add_argument(
        "--weight-arms",
        type=str,
        default=",".join(DEFAULT_WEIGHT_ARMS),
        help="Comma-separated weight arms.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to average. Overrides --seed when set.",
    )
    parser.add_argument(
        "--model-feature-selector",
        choices=("all", *PROXY_METHODS),
        default="all",
        help="Optional prior-month feature selector applied separately per label/weight candidate.",
    )
    parser.add_argument("--model-feature-tail-frac", type=float, default=0.01)
    parser.add_argument(
        "--train-lookback-months",
        type=int,
        default=None,
        help="Optional number of most recent prior months to train each smoke model on.",
    )
    parser.add_argument(
        "--top-fracs",
        type=str,
        default=",".join(str(v) for v in DEFAULT_TOP_FRACS),
        help="Comma-separated selection fractions to evaluate.",
    )
    parser.add_argument(
        "--include-risk-selector-variants",
        action="store_true",
        help="Also evaluate OOS bad-MAE/timeout penalty and side-capped selector variants.",
    )
    parser.add_argument(
        "--side-cap-max-share",
        type=float,
        default=0.70,
        help="Maximum selected share for either side in side-capped selector variants.",
    )
    parser.add_argument(
        "--candidate-ledger-selector-names",
        type=str,
        default="",
        help="Comma-separated exact selector variants for row-level candidate-ledger export.",
    )
    parser.add_argument(
        "--candidate-ledger-only",
        action="store_true",
        help="Only materialize requested candidate-ledger selector rows where possible.",
    )
    parser.add_argument(
        "--candidate-ledger-fast-mode",
        action="store_true",
        help="Skip auxiliary scores/rankers not needed by requested candidate-ledger selectors.",
    )
    parser.add_argument(
        "--spread-baseline-path",
        type=Path,
        default=None,
        help="Optional per-symbol spread baseline used to filter the available symbol universe.",
    )
    parser.add_argument(
        "--spread-rank-column",
        type=str,
        default="p75_spread_bps",
        help="Spread baseline column used for ranking symbols when filtering the universe.",
    )
    parser.add_argument(
        "--target-symbol-count",
        type=int,
        default=None,
        help="Keep the lowest-spread N symbols from the available label universe.",
    )
    parser.add_argument(
        "--max-spread-bps",
        type=float,
        default=None,
        help="Optional absolute spread cap in bps applied before target-symbol-count.",
    )
    parser.add_argument(
        "--disable-ae-gmm-state-features",
        action="store_true",
        help="Disable fold-local generated AE/GMM state features in the smoke model.",
    )
    parser.add_argument(
        "--ae-gmm-state-feature-max-train-rows",
        type=int,
        default=DEFAULT_AE_GMM_STATE_FEATURE_MAX_TRAIN_ROWS,
        help="Maximum prior-fold rows used to fit the denoising AE in the generated AE/GMM state transform.",
    )
    parser.add_argument(
        "--ae-gmm-state-feature-gmm-max-train-rows",
        type=int,
        default=DEFAULT_AE_GMM_STATE_FEATURE_GMM_MAX_TRAIN_ROWS,
        help="Maximum prior-fold latent rows used to fit/HPO the GMM after the AE is frozen.",
    )
    parser.add_argument(
        "--ae-gmm-state-feature-max-iter",
        type=int,
        default=DEFAULT_AE_GMM_STATE_FEATURE_MAX_ITER,
        help="Maximum denoising-AE iterations for generated AE/GMM state features.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_smoke(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        evaluation_utility_column=args.evaluation_utility_column,
        max_feature_store_features=args.max_feature_store_features,
        label_arms=_parse_csv(args.label_arms, DEFAULT_LABEL_ARMS),
        weight_arms=_parse_csv(args.weight_arms, DEFAULT_WEIGHT_ARMS),
        seeds=_parse_int_csv(args.seeds, (args.seed,)),
        model_feature_selector=str(args.model_feature_selector),
        model_feature_tail_frac=float(args.model_feature_tail_frac),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        train_lookback_months=args.train_lookback_months,
        include_risk_selector_variants=bool(args.include_risk_selector_variants),
        side_cap_max_share=float(args.side_cap_max_share),
        candidate_ledger_selector_names=_parse_csv(
            args.candidate_ledger_selector_names,
            (),
        ),
        candidate_ledger_only=bool(args.candidate_ledger_only),
        candidate_ledger_fast_mode=bool(args.candidate_ledger_fast_mode),
        spread_baseline_path=args.spread_baseline_path,
        spread_rank_column=str(args.spread_rank_column),
        target_symbol_count=args.target_symbol_count,
        max_spread_bps=args.max_spread_bps,
        include_ae_gmm_state_features=not bool(args.disable_ae_gmm_state_features),
        ae_gmm_state_feature_max_train_rows=int(args.ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_gmm_max_train_rows=int(args.ae_gmm_state_feature_gmm_max_train_rows),
        ae_gmm_state_feature_max_iter=int(args.ae_gmm_state_feature_max_iter),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
