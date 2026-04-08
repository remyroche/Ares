from __future__ import annotations

import collections
import glob
import hashlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import pickle
import re
import time
import traceback
from dataclasses import dataclass, field, replace
from math import sqrt
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats
from lightgbm import LGBMRegressor
from numba import njit, prange
from sklearn.metrics import average_precision_score, roc_auc_score

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "3")

from extreme_price_movements.config import (
    CFG,
    CONTINUOUS_LOCATION_COLS,
    CONTINUOUS_TRIGGER_COLS,
    LOC_CONTINUOUS_FAMILY_MAP,
    RIDGE_FEATURE_COLS,
    RIDGE_FEATURE_META,
    TEST_FEATURE_KEYS,
    TIME_FEATURE_KEYS,
)
from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    load_features_selected,
    to_panel,
)
from extreme_price_movements.hpo_lgbm_regime_miner import (
    PREFERRED_SUPPORT_MAX,
    PREFERRED_SUPPORT_MIN,
    SUPPORT_MAX,
    SUPPORT_MIN,
    TARGET_SUPPORT,
    run_short_hpo_for_target_horizon,
)
from extreme_price_movements.intraday_crypto_library import (
    INTRADAY_TRIGGER_COLUMNS,
    LOCATION_FILTER_COLUMNS,
)
from extreme_price_movements.periods_symbols_management import (
    EventSchema,
    SlicePlanner,
    SlicePlannerConfig,
)
from extreme_price_movements.universe import get_training_universe
from extreme_price_movements.utils import tprint

LOGGER = logging.getLogger(__name__)

# =============================================================================
# TRIAD TARGET CONFIGURATION
# =============================================================================

# Default horizons for triad target training (in bars)
TRIAD_DEFAULT_HORIZONS: List[int] = [5, 10]

# Default triad target names
TRIAD_DEFAULT_TARGET_NAMES: List[str] = ["returns_target", "atr_norm_returns_target"]

# Per-target configuration for triad targets
TRIAD_TARGET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "target_eff": {
        "huber_alpha": 1.0,
        "learning_rate": 0.03,
        "min_support_pct": 0.05,
        "ic_hurdle": 0.02,
        "description": "Efficiency: direct vs actual path ratio",
    },
    "target_vame": {
        "huber_alpha": 0.5,
        "learning_rate": 0.04,
        "min_support_pct": 0.06,
        "ic_hurdle": 0.025,
        "description": "Volume-adjusted momentum efficiency",
    },
    "returns_target": {
        "huber_alpha": 1.0,
        "learning_rate": 0.03,
        "min_support_pct": 0.05,
        "ic_hurdle": 0.02,
        "description": "Log forward returns",
    },
    "atr_norm_returns_target": {
        "huber_alpha": 1.0,
        "learning_rate": 0.03,
        "min_support_pct": 0.05,
        "ic_hurdle": 0.02,
        "description": "ATR-normalized log forward returns",
    },
}

# Horizon-specific configuration multipliers
HORIZON_CONFIGS: Dict[int, Dict[str, Any]] = {
    5: {"min_data_in_leaf_multiplier": 0.8, "description": "Very short-term"},
    10: {"min_data_in_leaf_multiplier": 1.0, "description": "Short-term"},
}

MINER_TARGET_RESIDUALIZATION_COLUMNS: Tuple[str, ...] = (
    "ema50_ema200_spread_continuous",
    "atr_change_rate_ts_continuous",
    "bars_in_high_vol_state_log_norm",
    "volatility_of_volatility_48",
    "trend_strength_percentile",
    "volatility_autocorr_48",
)

MINER_TARGET_RESIDUALIZATION_ALIAS_MAP: Dict[str, Tuple[str, ...]] = {
    "ema50_ema200_spread_continuous": (
        "ema50_ema200_spread_continuous",
        "ema50_ema200_spread_atr",
    ),
    "ema50_ema200_spread_atr": (
        "ema50_ema200_spread_continuous",
        "ema50_ema200_spread_atr",
    ),
    "atr_change_rate_ts_continuous": (
        "atr_change_rate_ts_continuous",
        "atr_change_rate_ts",
        "atr_change_rate",
    ),
    "atr_change_rate_ts": (
        "atr_change_rate_ts_continuous",
        "atr_change_rate_ts",
        "atr_change_rate",
    ),
    "atr_change_rate": (
        "atr_change_rate_ts_continuous",
        "atr_change_rate_ts",
        "atr_change_rate",
    ),
    "bars_in_high_vol_state_log_norm": (
        "bars_in_high_vol_state_log_norm",
    ),
    "volatility_of_volatility_48": (
        "volatility_of_volatility_48",
    ),
    "trend_strength_percentile": (
        "trend_strength_percentile",
    ),
    "volatility_autocorr_48": (
        "volatility_autocorr_48",
    ),
}

MINER_NUISANCE_REGIME_SOURCE_NAMES: Set[str] = {
    "ema20_gt_ema50",
    "ema50_gt_ema200",
    "price_lt_ema200",
    "ema50_ema200_spread_continuous",
    "ema50_ema200_spread_atr",
    "atr_change_rate_ts_continuous",
    "atr_change_rate_ts",
    "atr_change_rate",
    "bars_in_high_vol_state_log_norm",
    "volatility_of_volatility_48",
    "trend_strength_percentile",
    "volatility_autocorr_48",
}

MINER_CONTINUOUS_PASSTHROUGH_SOURCE_NAMES: Set[str] = {
    "atr_compression_ratio",
}

MINER_OPTIONAL_LOCATION_NUISANCE_PREFIXES: Tuple[str, ...] = (
    "dist_ema20_atr",
    "dist_ema200_atr",
)

SCORER_REGISTRY_COLUMNS: List[str] = [
    "canonical_key",
    "trigger",
    "location",
    "regime",
    "mean_net_ret",
    "directional_mean_ret",
    "std_net_ret",
    "mean_within_fold_std",
    "mean_support_pct",
    "std_support_pct",
    "presence_freq",
    "presence_freq_units",
    "sign_consistency",
    "min_support_actual",
    "mean_uplift",
    "mean_baseline_ret",
    "mean_oos_ic",
    "p25_oos_ic",
    "p50_oos_ic",
    "p75_oos_ic",
    "mean_delta_ic",
    "positive_ic_fraction",
    "mean_ic",
    "ic_tstat",
    "ic_sign_consistency",
    "decile_spread_sharpe",
    "mask_ic_uplift",
    "regression_beta",
    "regression_slope_fit",
    "learnability_step_c_score",
    "trade_path_quality_score",
    "quality_stability_score",
    "path_smoothness_term",
    "path_survivability_term",
    "path_stability_term",
    "path_realized_profit_consistency_term",
    "path_trajectory_smoothness_term",
    "full_quality_score",
    "composite_score",
    "required_hurdle",
    "hurdle_excess",
    "n_folds",
    "discovery_count",
    "n_instances",
    "display_arity",
    "structural_depth",
    "pipeline_stage",
    "parent_context_key",
    "side",
    "rule_type",
    "accepted",
    "rejection_reason",
]



class TargetNaNReason:
    HORIZON_EXCEEDED = "horizon_exceeded"
    BARRIER_UNRESOLVED = "barrier_unresolved"
    AMBIGUOUS_BAR = "ambiguous_bar"
    OUTSIDE_SUPPORT_MASK = "outside_support_mask"
    NEUTRAL_FILTERED = "neutral_filtered"
    CURRENT_CLOSE_MISSING = "current_close_missing"
    ATR_MISSING = "atr_missing"
    TRANSFORMED_TARGET_NONFINITE = "transformed_target_nonfinite"
    SYMBOL_ALIGNMENT_MISSING = "symbol_alignment_missing"
    FUTURE_CLOSE_MISSING = "future_close_missing"
    OTHER_TARGET_NAN = "other_target_nan"

def generate_fwd_ret_with_reasons(panel: pd.DataFrame, fwd_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates forward returns and a reason code array for Target NaNs."""
    fwd_ret_wide = panel["close"].pct_change(fwd_hours, fill_method=None).shift(-fwd_hours)

    reasons_wide = pd.DataFrame("", index=fwd_ret_wide.index, columns=fwd_ret_wide.columns)
    current_close_missing = panel["close"].isna()
    future_close_missing = panel["close"].shift(-fwd_hours).isna()
    reasons_wide[current_close_missing] = TargetNaNReason.CURRENT_CLOSE_MISSING
    reasons_wide[future_close_missing] = TargetNaNReason.FUTURE_CLOSE_MISSING
    reasons_wide[np.isnan(fwd_ret_wide) & (reasons_wide == "")] = TargetNaNReason.TRANSFORMED_TARGET_NONFINITE
    if fwd_hours > 0 and len(reasons_wide) >= fwd_hours:
        reasons_wide.iloc[-fwd_hours:] = TargetNaNReason.HORIZON_EXCEEDED

    return fwd_ret_wide, reasons_wide


def _with_expected_columns(
    df: Optional[pd.DataFrame], expected_columns: Sequence[str]
) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=list(expected_columns))
    work = df.copy()
    for column in expected_columns:
        if column not in work.columns:
            work[column] = pd.Series([np.nan] * len(work), index=work.index)
    ordered = list(expected_columns) + [
        column for column in work.columns if column not in expected_columns
    ]
    return work.loc[:, ordered]


def atomic_to_csv(
    df: Optional[pd.DataFrame],
    output_path: Path,
    expected_columns: Optional[Sequence[str]] = None,
    index: bool = False,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared = (
        _with_expected_columns(df, expected_columns)
        if expected_columns is not None
        else (pd.DataFrame() if df is None else df.copy())
    )
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        prepared.to_csv(tmp_file, index=index)
    os.replace(tmp_path, output_path)
    return prepared


def _drop_duplicate_columns(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    work = df.copy()
    if work.columns.has_duplicates:
        work = work.loc[:, ~work.columns.duplicated()].copy()
    return work


def _build_row_funnel_df(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["stage", "rows", "symbols", "fraction_of_prev"])


def build_run_output_dir(
    cfg: Dict[str, Any],
    target_name: Optional[str] = None,
    horizon: Optional[int] = None,
    side: Optional[str] = None,
    stage: Optional[str] = None,
) -> Path:
    """
    Build output directory path for run artifacts.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Configuration dictionary containing 'output_dir' and 'timestamped_run_outputs'
    target_name : Optional[str]
        Target name for triad mode (e.g., 'target_eff', 'target_vame')
    horizon : Optional[int]
        Horizon in bars for triad mode
    side : Optional[str]
        Side ('long' or 'short') for stage-specific paths
    stage : Optional[str]
        Optional stage subdirectory name

    Returns
    -------
    Path
        Output directory path

    Notes
    -----
    When target_name and horizon are provided, creates a triad-style path:
        base_dir/h{horizon}/{target_name}/{side}/{stage}
    Otherwise uses the legacy path structure.
    """
    base_output_dir = Path(cfg.get("output_dir", "./lgbm_outputs"))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    cached_run_base = cfg.get("_resolved_run_base_dir")
    if cached_run_base:
        run_base = Path(cached_run_base)
    else:
        # Determine if we should use timestamped run directory
        if not bool(cfg.get("timestamped_run_outputs", True)):
            run_base = base_output_dir
        else:
            # Use a more precise timestamp for the run directory to reduce collisions
            timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S_%f")
            run_base = base_output_dir / f"run_{timestamp}"
            suffix = 1
            while run_base.exists():
                suffix += 1
                run_base = base_output_dir / f"run_{timestamp}_{suffix:02d}"
        cfg["_resolved_run_base_dir"] = str(run_base)

    # Build path based on triad vs legacy mode
    if target_name is not None and horizon is not None:
        # Triad mode: base_dir/h{horizon}/{target_name}/...
        run_output_dir = run_base / f"h{horizon}" / target_name
        if side is not None:
            run_output_dir = run_output_dir / side
        if stage is not None:
            run_output_dir = run_output_dir / stage
    else:
        # Legacy mode
        run_output_dir = run_base

    # Only create directory if this is the root call (no side/stage specified)
    if side is None and stage is None:
        run_output_dir.mkdir(parents=True, exist_ok=True)

    return run_output_dir


def resolve_stage_a_step1_dir(
    step1_base_dir: Union[str, Path],
    target_name: str,
    horizon: int,
    side: str,
) -> Path:
    base = Path(step1_base_dir)
    if (base / "candidate_rule_registry.csv").exists():
        return base
    return base / f"h{horizon}" / target_name / side / "stage_a_context"


def save_stage_a_step1_checkpoint(
    output_dir: Path,
    candidate_registry: pd.DataFrame,
    cheap_gate_rows: Dict[Tuple[str, int], List[Tuple[float, str]]],
    cheap_gate_result: Dict[Tuple[Tuple[str, int], str], Tuple[bool, str]],
    bucket_cheap_ranks: Dict[Tuple[str, int], Dict[str, float]],
    stage_a_matrices: Dict[Tuple[str, int], Dict[str, Any]],
) -> Path:
    checkpoint_path = output_dir / "stage_a_step1_checkpoint.pkl"
    payload = {
        "candidate_registry": candidate_registry,
        "cheap_gate_rows": cheap_gate_rows,
        "cheap_gate_result": cheap_gate_result,
        "bucket_cheap_ranks": bucket_cheap_ranks,
        "stage_a_matrices": stage_a_matrices,
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    post_dedup_rows: List[Dict[str, Any]] = []
    ordered_survivor_keys: List[str] = []
    cheap_rank_map: Dict[str, float] = {}
    bucket_side_map: Dict[str, str] = {}
    bucket_horizon_map: Dict[str, int] = {}
    for (side, source_horizon), entries in cheap_gate_rows.items():
        for cheap_rank, canonical_key in entries:
            ordered_survivor_keys.append(str(canonical_key))
            cheap_rank_map[str(canonical_key)] = float(cheap_rank)
            bucket_side_map[str(canonical_key)] = str(side)
            bucket_horizon_map[str(canonical_key)] = int(source_horizon)
            post_dedup_rows.append(
                {
                    "side": side,
                    "source_horizon": source_horizon,
                    "cheap_rank": cheap_rank,
                    "canonical_key": canonical_key,
                }
            )
    post_dedup_source_df = candidate_registry.copy()
    if not post_dedup_source_df.empty:
        post_dedup_source_df["canonical_key"] = post_dedup_source_df[
            "canonical_key"
        ].astype(str)
        post_dedup_source_df = post_dedup_source_df.drop_duplicates(
            subset=["canonical_key"], keep="first"
        )
        post_dedup_source_df = post_dedup_source_df.set_index("canonical_key", drop=False)
        survivor_records: List[Dict[str, Any]] = []
        for canonical_key in ordered_survivor_keys:
            if canonical_key not in post_dedup_source_df.index:
                continue
            row_dict = post_dedup_source_df.loc[canonical_key].to_dict()
            row_dict["cheap_rank"] = float(cheap_rank_map.get(canonical_key, -np.inf))
            row_dict["step1_bucket_side"] = bucket_side_map.get(canonical_key, "unknown")
            row_dict["step1_bucket_horizon"] = int(
                bucket_horizon_map.get(canonical_key, -1)
            )
            survivor_records.append(row_dict)
        post_dedup_df = pd.DataFrame(survivor_records)
    else:
        post_dedup_df = pd.DataFrame()
    atomic_to_csv(
        post_dedup_df,
        output_dir / "step1_post_dedup_registry.csv",
        expected_columns=list(post_dedup_df.columns) if not post_dedup_df.empty else None,
    )
    atomic_to_csv(
        pd.DataFrame(post_dedup_rows),
        output_dir / "step1_post_dedup_keys.csv",
        expected_columns=["side", "source_horizon", "cheap_rank", "canonical_key"],
    )

    gate_rows: List[Dict[str, Any]] = []
    for (bucket_key, canonical_key), (rejected, reason) in cheap_gate_result.items():
        gate_rows.append(
            {
                "side": bucket_key[0],
                "source_horizon": bucket_key[1],
                "canonical_key": canonical_key,
                "rejected": bool(rejected),
                "reason": reason,
            }
        )
    atomic_to_csv(
        pd.DataFrame(gate_rows),
        output_dir / "step1_gate_registry.csv",
        expected_columns=[
            "side",
            "source_horizon",
            "canonical_key",
            "rejected",
            "reason",
        ],
    )

    with open(output_dir / "stage_a_step1_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "candidate_rules": int(len(candidate_registry)),
                "post_dedup_rules": int(sum(len(v) for v in cheap_gate_rows.values())),
                "buckets": int(len(cheap_gate_rows)),
                "checkpoint_path": str(checkpoint_path),
            },
            f,
            indent=2,
        )
    return checkpoint_path


def load_stage_a_step1_checkpoint(step1_dir: Union[str, Path]) -> Dict[str, Any]:
    checkpoint_path = Path(step1_dir) / "stage_a_step1_checkpoint.pkl"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing step1 checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def _make_slice_selection_key(target_name: str, horizon: int, side: str) -> str:
    return f"{target_name}|{int(horizon)}|{side}"


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    # Check for constant arrays
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(scipy.stats.spearmanr(x, y).correlation)


def _safe_tanh_scale(x: float, scale: float) -> float:
    if not np.isfinite(x):
        return np.nan
    scale = max(float(scale), 1e-9)
    return float(np.tanh(x / scale))


def _resolve_miner_nuisance_feature_arrays(
    feature_dict: Dict[str, np.ndarray], cfg: Dict[str, Any]
) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    requested_columns = tuple(
        cfg.get(
            "miner_target_residualization_columns",
            MINER_TARGET_RESIDUALIZATION_COLUMNS,
        )
    )
    resolved: Dict[str, str] = {}
    arrays: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for requested_name in requested_columns:
        requested_name = str(requested_name)
        candidates = MINER_TARGET_RESIDUALIZATION_ALIAS_MAP.get(
            requested_name, (requested_name,)
        )
        resolved_name = next((name for name in candidates if name in feature_dict), None)
        if resolved_name is None:
            missing.append(requested_name)
            continue
        resolved[requested_name] = str(resolved_name)
        arrays[requested_name] = np.asarray(feature_dict[resolved_name], dtype=np.float32)
    if missing:
        raise KeyError(
            "Missing residualisation nuisance columns for miner target: "
            + ", ".join(missing)
        )
    return resolved, arrays


def _should_drop_miner_nuisance_source(
    source_name: str, group_name: str, cfg: Dict[str, Any]
) -> bool:
    if not bool(cfg.get("drop_nuisance_features_from_miner", True)):
        return False
    if group_name == "regime" and source_name in MINER_NUISANCE_REGIME_SOURCE_NAMES:
        return True
    if group_name == "location" and bool(
        cfg.get("drop_location_nuisance_features_from_miner", False)
    ):
        return any(
            source_name.startswith(prefix)
            for prefix in MINER_OPTIONAL_LOCATION_NUISANCE_PREFIXES
        )
    return False


def _safe_finite_spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return np.nan
    return _safe_spearman(
        np.asarray(x[valid], dtype=np.float32),
        np.asarray(y[valid], dtype=np.float32),
    )


def _fit_linear_target_residualizer(
    target: np.ndarray,
    nuisance_arrays: Dict[str, np.ndarray],
    nuisance_resolution: Dict[str, str],
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    fit_mask = np.isfinite(target)
    for arr in nuisance_arrays.values():
        fit_mask &= np.isfinite(arr)
    fit_count = int(fit_mask.sum())
    if fit_count < len(nuisance_arrays) + 2:
        raise ValueError(
            f"Insufficient residualiser support: fit_count={fit_count}, predictors={len(nuisance_arrays)}"
        )
    design = np.column_stack(
        [
            np.ones(fit_count, dtype=np.float64),
            *[
                np.asarray(arr[fit_mask], dtype=np.float64)
                for arr in nuisance_arrays.values()
            ],
        ]
    )
    coeffs, _, _, _ = np.linalg.lstsq(
        design,
        np.asarray(target[fit_mask], dtype=np.float64),
        rcond=None,
    )
    weight_summary = {
        "weight_mean": np.nan,
        "weight_p5": np.nan,
        "weight_p50": np.nan,
        "weight_p95": np.nan,
    }
    if sample_weight is not None:
        weights_fit = np.asarray(sample_weight[fit_mask], dtype=np.float64)
        weights_fit = np.where(np.isfinite(weights_fit), weights_fit, 0.0)
        if np.sum(weights_fit > 0.0) < len(nuisance_arrays) + 2:
            raise ValueError(
                "Insufficient positive residualiser weights after filtering finite rows"
            )
        weighted_design = design * np.sqrt(weights_fit)[:, None]
        weighted_target = np.asarray(target[fit_mask], dtype=np.float64) * np.sqrt(
            weights_fit
        )
        coeffs, _, _, _ = np.linalg.lstsq(
            weighted_design,
            weighted_target,
            rcond=None,
        )
        weight_summary = {
            "weight_mean": float(np.mean(weights_fit)),
            "weight_p5": float(np.percentile(weights_fit, 5)),
            "weight_p50": float(np.percentile(weights_fit, 50)),
            "weight_p95": float(np.percentile(weights_fit, 95)),
        }
    else:
        coeffs, _, _, _ = np.linalg.lstsq(
            design,
            np.asarray(target[fit_mask], dtype=np.float64),
            rcond=None,
        )
    return {
        "intercept": float(coeffs[0]),
        "coefficients": [float(v) for v in coeffs[1:]],
        "fit_sample_count": fit_count,
        "fit_support_definition": "finite(raw_target)&finite(all_nuisance_columns) on fold-train partition",
        "nuisance_columns_requested": list(nuisance_arrays.keys()),
        "nuisance_columns_resolved": [
            nuisance_resolution[name] for name in nuisance_arrays.keys()
        ],
        **weight_summary,
    }


def _apply_linear_target_residualizer(
    target: np.ndarray,
    nuisance_arrays: Dict[str, np.ndarray],
    residualizer: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    valid_mask = np.isfinite(target)
    for arr in nuisance_arrays.values():
        valid_mask &= np.isfinite(arr)
    out = np.full(target.shape, np.nan, dtype=np.float32)
    if not valid_mask.any():
        return out, valid_mask
    pred = np.full(int(valid_mask.sum()), float(residualizer["intercept"]), dtype=np.float64)
    for coef, arr in zip(residualizer["coefficients"], nuisance_arrays.values()):
        pred += float(coef) * np.asarray(arr[valid_mask], dtype=np.float64)
    out[valid_mask] = (
        np.asarray(target[valid_mask], dtype=np.float64) - pred
    ).astype(np.float32, copy=False)
    return out, valid_mask


def _prepare_fold_effective_target_views(
    raw_target: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    target_name: str,
    residualise_target: bool,
    nuisance_feature_arrays: Optional[Dict[str, np.ndarray]],
    nuisance_feature_resolution: Optional[Dict[str, str]],
    apply_target_postprocessing: bool,
    X: np.ndarray,
    symbol_id: np.ndarray,
    cfg: Dict[str, Any],
    horizon: int,
    surprisal_bits: Optional[np.ndarray] = None,
    mfe_atr: Optional[np.ndarray] = None,
    mae_atr: Optional[np.ndarray] = None,
    side: str = "long",
) -> Tuple[Dict[int, Dict[str, np.ndarray]], np.ndarray, List[Dict[str, Any]]]:
    fold_views: Dict[int, Dict[str, np.ndarray]] = {}
    effective_target_oof = np.full(len(raw_target), np.nan, dtype=np.float32)
    residualizer_records: List[Dict[str, Any]] = []
    if apply_target_postprocessing:
        from extreme_price_movements.triad_targets import (
            apply_target_postprocessor,
            fit_target_postprocessor,
        )

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        y_tr_raw = np.asarray(raw_target[tr_idx], dtype=np.float32)
        y_va_raw = np.asarray(raw_target[va_idx], dtype=np.float32)
        X_tr = np.asarray(X[tr_idx], dtype=np.float32)
        symbol_id_tr = np.asarray(symbol_id[tr_idx])
        y_tr_effective = y_tr_raw.copy()
        y_va_effective = y_va_raw.copy()
        record: Dict[str, Any] = {
            "fold_id": fold_id,
            "raw_target_name": target_name,
            "effective_target_name": f"{target_name}_miner_effective",
            "target_representation": "residualised" if residualise_target else "raw",
            "residualisation_enabled": bool(residualise_target),
            "fit_sample_count": 0,
            "train_effective_valid_count": int(np.isfinite(y_tr_raw).sum()),
            "val_effective_valid_count": int(np.isfinite(y_va_raw).sum()),
            "train_excluded_missing_nuisance_pct": 0.0,
            "val_excluded_missing_nuisance_pct": 0.0,
            "nuisance_columns_requested": json.dumps(
                list((nuisance_feature_arrays or {}).keys())
            ),
            "nuisance_columns_resolved": json.dumps(
                [
                    (nuisance_feature_resolution or {}).get(name, name)
                    for name in (nuisance_feature_arrays or {}).keys()
                ]
            ),
        }
        if residualise_target:
            if not nuisance_feature_arrays:
                raise ValueError("Residualisation requested but no nuisance arrays were provided")
            residualizer_sample_weight = build_miner_sample_weights(
                y_tr_raw,
                X_tr,
                symbol_id_tr,
                cfg,
                horizon=horizon,
                surprisal_bits=(
                    None
                    if surprisal_bits is None
                    else np.asarray(surprisal_bits[tr_idx], dtype=np.float32)
                ),
                mfe_atr=(
                    None
                    if mfe_atr is None
                    else np.asarray(mfe_atr[tr_idx], dtype=np.float32)
                ),
                mae_atr=(
                    None
                    if mae_atr is None
                    else np.asarray(mae_atr[tr_idx], dtype=np.float32)
                ),
                side=side,
            )
            tr_nuisance = {
                name: np.asarray(arr[tr_idx], dtype=np.float32)
                for name, arr in nuisance_feature_arrays.items()
            }
            va_nuisance = {
                name: np.asarray(arr[va_idx], dtype=np.float32)
                for name, arr in nuisance_feature_arrays.items()
            }
            residualizer = _fit_linear_target_residualizer(
                y_tr_raw,
                tr_nuisance,
                nuisance_feature_resolution or {},
                sample_weight=residualizer_sample_weight,
            )
            y_tr_effective, tr_valid_mask = _apply_linear_target_residualizer(
                y_tr_raw,
                tr_nuisance,
                residualizer,
            )
            y_va_effective, va_valid_mask = _apply_linear_target_residualizer(
                y_va_raw,
                va_nuisance,
                residualizer,
            )
            record.update(
                {
                    "fit_sample_count": int(residualizer["fit_sample_count"]),
                    "intercept": float(residualizer["intercept"]),
                    "coefficients": json.dumps(residualizer["coefficients"]),
                    "fit_support_definition": residualizer["fit_support_definition"],
                    "train_effective_valid_count": int(tr_valid_mask.sum()),
                    "val_effective_valid_count": int(va_valid_mask.sum()),
                    "train_excluded_missing_nuisance_pct": float(
                        1.0 - (tr_valid_mask.sum() / max(len(tr_valid_mask), 1))
                    ),
                    "val_excluded_missing_nuisance_pct": float(
                        1.0 - (va_valid_mask.sum() / max(len(va_valid_mask), 1))
                    ),
                    "residualizer_weight_mean": float(residualizer["weight_mean"]),
                    "residualizer_weight_p5": float(residualizer["weight_p5"]),
                    "residualizer_weight_p50": float(residualizer["weight_p50"]),
                    "residualizer_weight_p95": float(residualizer["weight_p95"]),
                }
            )
            for requested_name, resolved_name in (
                nuisance_feature_resolution or {}
            ).items():
                record[f"resolved__{requested_name}"] = resolved_name
                record[f"corr_raw__{requested_name}"] = _safe_finite_spearman(
                    np.asarray(tr_nuisance[requested_name], dtype=np.float32),
                    y_tr_raw,
                )
                record[f"corr_effective__{requested_name}"] = _safe_finite_spearman(
                    np.asarray(tr_nuisance[requested_name], dtype=np.float32),
                    y_tr_effective,
                )

        if apply_target_postprocessing:
            pp = fit_target_postprocessor(target_name, y_tr_effective, mode="default")
            y_tr_processed = apply_target_postprocessor(y_tr_effective, pp)
            y_va_processed = apply_target_postprocessor(y_va_effective, pp)
        else:
            y_tr_processed = np.clip(y_tr_effective, -3.0, 3.0).astype(
                np.float32, copy=False
            )
            y_va_processed = np.asarray(y_va_effective, dtype=np.float32)

        effective_target_oof[va_idx] = y_va_processed
        record["train_processed_valid_count"] = int(np.isfinite(y_tr_processed).sum())
        record["val_processed_valid_count"] = int(np.isfinite(y_va_processed).sum())
        fold_views[fold_id] = {
            "y_tr_raw": y_tr_raw,
            "y_va_raw": y_va_raw,
            "y_tr_effective": np.asarray(y_tr_effective, dtype=np.float32),
            "y_va_effective": np.asarray(y_va_effective, dtype=np.float32),
            "y_tr_processed": np.asarray(y_tr_processed, dtype=np.float32),
            "y_va_processed": np.asarray(y_va_processed, dtype=np.float32),
        }
        residualizer_records.append(record)

    return fold_views, effective_target_oof, residualizer_records


@dataclass
class EvaluatedTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    confidence_score: float
    gross_trade_return: float
    symbol: str

def compute_ridge_pnl(
    trades: List[EvaluatedTrade],
    threshold_star: float,
    round_fee: float = 0.0015,
    min_weight: float = 0.05,
    max_weight: float = 0.15,
    convex_power: float = 2.0,
    starting_capital: float = 1.0,
    forbid_concurrent: bool = True,
    max_concurrent_per_symbol: int = 1,
    max_concurrent_total: int = 2
) -> Dict[str, Any]:
    eligible_trades = [
        t for t in trades
        if t.confidence_score >= threshold_star
    ]

    if len(eligible_trades) == 0:
        return {
            "ridge_pnl_gross_raw": 0.0,
            "ridge_pnl_raw": 0.0,
            "selected_trades": [],
            "weighted_gross_returns": [],
            "weighted_net_returns": [],
            "ending_gross_capital": starting_capital,
            "ending_capital": starting_capital,
        }

    eligible_trades.sort(key=lambda t: t.entry_time)

    if forbid_concurrent:
        selected_trades = []
        # Maintain active intervals separately per symbol
        active_intervals_per_symbol: Dict[str, List[pd.Timestamp]] = collections.defaultdict(list)
        # Global active intervals to enforce total concurrency
        active_intervals_global: List[pd.Timestamp] = []

        for t in eligible_trades:
            # Drop expired active intervals for that symbol
            active_intervals = active_intervals_per_symbol[t.symbol]
            active_intervals = [end_time for end_time in active_intervals if t.entry_time < end_time]
            active_intervals_per_symbol[t.symbol] = active_intervals

            # Drop expired active intervals globally
            active_intervals_global = [end_time for end_time in active_intervals_global if t.entry_time < end_time]

            # Enforce concurrency policy
            if len(active_intervals) < max_concurrent_per_symbol and len(active_intervals_global) < max_concurrent_total:
                selected_trades.append(t)
                active_intervals_per_symbol[t.symbol].append(t.exit_time)
                active_intervals_global.append(t.exit_time)
            else:
                continue
    else:
        selected_trades = eligible_trades

    if len(selected_trades) == 0:
        return {
            "ridge_pnl_gross_raw": 0.0,
            "ridge_pnl_raw": 0.0,
            "selected_trades": [],
            "weighted_gross_returns": [],
            "weighted_net_returns": [],
            "ending_gross_capital": starting_capital,
            "ending_capital": starting_capital,
        }

    weighted_gross_returns = []
    weighted_net_returns = []
    weighted_fee_returns = []
    gross_trade_returns = []
    sizing_weights = []

    for t in selected_trades:
        conf = t.confidence_score
        denom = max(1.0 - threshold_star, 1e-9)
        normalized_conf = np.clip((conf - threshold_star) / denom, 0.0, 1.0)

        position_weight = (
            min_weight
            + (max_weight - min_weight) * (normalized_conf ** convex_power)
        )
        sizing_weights.append(float(position_weight))
        gross_trade_returns.append(float(t.gross_trade_return))

        # Note: We assume gross_trade_return already incorporates the entry/exit logic,
        # but we must subtract the round-trip fee.
        weighted_gross_return = float(position_weight * t.gross_trade_return)
        net_trade_return = t.gross_trade_return - round_fee
        weighted_net_return = float(position_weight * net_trade_return)
        weighted_fee_return = float(position_weight * round_fee)
        weighted_gross_returns.append(weighted_gross_return)
        weighted_net_returns.append(weighted_net_return)
        weighted_fee_returns.append(weighted_fee_return)

    gross_capital = starting_capital
    for wr in weighted_gross_returns:
        gross_capital = gross_capital * (1.0 + wr)

    capital = starting_capital
    for wr in weighted_net_returns:
        capital = capital * (1.0 + wr)

    ridge_pnl_gross_raw = gross_capital - starting_capital
    ridge_pnl_raw = capital - starting_capital

    return {
        "ridge_pnl_gross_raw": ridge_pnl_gross_raw,
        "ridge_pnl_raw": ridge_pnl_raw,
        "selected_trades": selected_trades,
        "weighted_gross_returns": weighted_gross_returns,
        "weighted_net_returns": weighted_net_returns,
        "weighted_fee_returns": weighted_fee_returns,
        "avg_fee_per_trade": float(np.mean(weighted_fee_returns)) if weighted_fee_returns else 0.0,
        "avg_gross_move_per_trade": float(np.mean(np.abs(gross_trade_returns))) if gross_trade_returns else 0.0,
        "avg_position_weight": float(np.mean(sizing_weights)) if sizing_weights else 0.0,
        "ending_gross_capital": gross_capital,
        "ending_capital": capital,
    }


def compute_ridge_trade_sortino(
    realized_trades: List[EvaluatedTrade],
    threshold_star: float,
    round_fee: float = 0.0015,
    min_weight: float = 0.05,
    max_weight: float = 0.15,
    convex_power: float = 2.0,
    sortino_scale: float = 2.0,
    eps: float = 1e-9,
) -> Dict[str, Any]:
    """
    Compute a Ridge trade Sortino operating directly on the realized trade set.
    """
    if not realized_trades:
        return {
            "sizing_weights": [],
            "net_weighted_returns": [],
            "ridge_trade_sortino_raw": 0.0,
            "ridge_trade_sortino": 0.0,
        }

    denom = max(1.0 - threshold_star, eps)
    sizing_weights = []
    net_weighted_returns = []

    for t in realized_trades:
        normalized_score = np.clip((t.confidence_score - threshold_star) / denom, 0.0, 1.0)
        weight = min_weight + (max_weight - min_weight) * (normalized_score ** convex_power)
        sizing_weights.append(weight)
        net_weighted_returns.append(weight * (t.gross_trade_return - round_fee))

    realized = np.array(net_weighted_returns, dtype=float)

    mean_ret = float(np.mean(realized)) if realized.size else 0.0
    downside = np.minimum(realized, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2))) if realized.size else 0.0

    ridge_trade_sortino_raw = mean_ret / (downside_dev + eps)

    ridge_trade_sortino = float(
        np.tanh(max(ridge_trade_sortino_raw, 0.0) / sortino_scale)
    )

    return {
        "sizing_weights": sizing_weights,
        "net_weighted_returns": net_weighted_returns,
        "ridge_trade_sortino_raw": ridge_trade_sortino_raw,
        "ridge_trade_sortino": ridge_trade_sortino,
    }


def _compute_decile_spread_sharpe(
    predictions: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    top_frac: float = 0.2,
    min_obs: int = 20,
) -> float:
    """Compute top-vs-bottom spread Sharpe within mask."""
    valid = np.isfinite(predictions) & np.isfinite(target) & mask.astype(bool)
    if int(valid.sum()) < int(min_obs):
        return np.nan

    preds = predictions[valid]
    targs = target[valid]
    n = len(preds)
    k = max(1, int(np.floor(n * float(top_frac))))
    if n < (2 * k + 2):
        return np.nan

    order = np.argsort(preds)
    low_idx = order[:k]
    high_idx = order[-k:]
    spread = targs[high_idx] - targs[low_idx]
    if spread.size < 3:
        return np.nan
    spread_std = float(np.nanstd(spread, ddof=0))
    if spread_std <= 1e-12:
        return np.nan
    return float(np.nanmean(spread) / (spread_std + 1e-12))


def _compute_regression_beta_and_fit(
    predictions: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    min_obs: int = 20,
    tau: float = 0.35,
) -> Tuple[float, float]:
    """
    Fit target ~ prediction on masked observations and return:
    - beta (slope)
    - slope fit score in (0,1], where closer to beta=1 is better.
    """
    valid = np.isfinite(predictions) & np.isfinite(target) & mask.astype(bool)
    if int(valid.sum()) < int(min_obs):
        return np.nan, np.nan

    x = predictions[valid]
    y = target[valid]
    x_mean = float(np.nanmean(x))
    y_mean = float(np.nanmean(y))
    x_cent = x - x_mean
    y_cent = y - y_mean
    denom = float(np.dot(x_cent, x_cent))
    if denom <= 1e-12:
        return np.nan, np.nan
    beta = float(np.dot(x_cent, y_cent) / denom)

    # Point 8: Stabilize tau and fit calculation
    tau = max(float(tau), 1e-6)
    delta = abs(beta - 1.0)
    # Numerical safety for exp
    fit = float(np.exp(-min(delta / tau, 20.0)))
    return beta, fit


def compute_directional_sign_consistency(
    returns: np.ndarray,
    threshold: float = 0.01,
    min_effective_samples: int = 5,
    max_samples: int = 1000,
) -> float:
    """
    Compute sign_consistency using only rows where the return magnitude
    is large enough to be economically meaningful.
    """
    if returns.size == 0:
        return 0.5

    if returns.size > max_samples:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(returns.size, size=max_samples, replace=False)
        returns = returns[idx]

    valid_mask = np.isfinite(returns)
    if not np.any(valid_mask):
        return 0.5

    valid_returns = returns[valid_mask]

    # Vectorized check for eligible returns without allocating intermediate arrays of floats
    eligible_mask = np.abs(valid_returns) > threshold
    n_total = int(np.sum(eligible_mask))

    if n_total == 0:
        return 0.5

    eligible_returns = valid_returns[eligible_mask]
    n_pos = int(np.sum(eligible_returns > 0))
    n_neg = int(np.sum(eligible_returns < 0))

    sign_consistency = float(max(n_pos, n_neg) / n_total)

    if n_total < min_effective_samples:
        shrink_factor = float(n_total) / float(min_effective_samples)
        sign_consistency = 0.5 + (sign_consistency - 0.5) * shrink_factor

    return sign_consistency


@njit(cache=True)
def _compute_masks_from_instruction_matrix_numba(
    x_values: np.ndarray,
    context_values: np.ndarray,
    instr_source_type: np.ndarray,
    instr_source_idx: np.ndarray,
    instr_operator_code: np.ndarray,
    instr_threshold_value: np.ndarray,
    rule_offsets: np.ndarray,
    rule_lengths: np.ndarray,
) -> np.ndarray:
    n_rules = rule_offsets.shape[0]
    n_samples = x_values.shape[0]
    out = np.ones((n_rules, n_samples), dtype=np.bool_)

    for rule_idx in range(n_rules):
        offset = int(rule_offsets[rule_idx])
        length = int(rule_lengths[rule_idx])
        if length <= 0:
            continue
        for j in range(length):
            instr_idx = offset + j
            source_type = int(instr_source_type[instr_idx])
            source_idx = int(instr_source_idx[instr_idx])
            operator_code = int(instr_operator_code[instr_idx])
            threshold_value = float(instr_threshold_value[instr_idx])
            if source_type == 0:
                for sample_idx in range(n_samples):
                    if out[rule_idx, sample_idx]:
                        x_val = x_values[sample_idx, source_idx]
                        if operator_code == 0:
                            out[rule_idx, sample_idx] = x_val == threshold_value
                        elif operator_code == 1:
                            out[rule_idx, sample_idx] = x_val <= threshold_value
                        elif operator_code == 2:
                            out[rule_idx, sample_idx] = x_val > threshold_value
                        elif operator_code == 3:
                            out[rule_idx, sample_idx] = x_val < threshold_value
                        else:
                            out[rule_idx, sample_idx] = x_val >= threshold_value
            else:
                for sample_idx in range(n_samples):
                    if out[rule_idx, sample_idx]:
                        out[rule_idx, sample_idx] = (
                            context_values[source_idx, sample_idx] == threshold_value
                        )
    return out


@njit(cache=True)
def _compute_batch_cheap_stats_numba(
    mask_matrix: np.ndarray,
    returns: np.ndarray,
    day_codes: np.ndarray,
    n_day_buckets: int,
    sign_threshold: float = 0.01,
    sign_min_effective_samples: int = 5,
    sign_max_samples: int = 1000,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n_rules, n_samples = mask_matrix.shape
    support_counts = np.zeros(n_rules, dtype=np.int32)
    mean_returns = np.full(n_rules, np.nan, dtype=np.float32)
    std_returns = np.full(n_rules, np.nan, dtype=np.float32)
    sign_consistency = np.full(n_rules, 0.5, dtype=np.float32)
    tail_ratio = np.ones(n_rules, dtype=np.float32)
    mae = np.zeros(n_rules, dtype=np.float32)
    mfe = np.zeros(n_rules, dtype=np.float32)
    density_dispersion = np.zeros(n_rules, dtype=np.float32)

    for i in range(n_rules):
        count = 0
        sum_ret = 0.0
        sum_sq = 0.0
        neg_sum = 0.0
        neg_count = 0
        pos_sum = 0.0
        pos_count = 0
        eligible_count = 0
        pos_eligible = 0
        neg_eligible = 0
        sample_cap_count = 0

        for j in range(n_samples):
            if not mask_matrix[i, j]:
                continue
            r = returns[j]
            if not np.isfinite(r):
                continue
            count += 1
            sum_ret += r
            sum_sq += r * r
            if r < 0.0:
                neg_sum += r
                neg_count += 1
            elif r > 0.0:
                pos_sum += r
                pos_count += 1

            if sample_cap_count < sign_max_samples and abs(r) > sign_threshold:
                eligible_count += 1
                sample_cap_count += 1
                if r > 0.0:
                    pos_eligible += 1
                elif r < 0.0:
                    neg_eligible += 1

        support_counts[i] = count
        if count == 0:
            continue

        mean_r = sum_ret / count
        var_r = (sum_sq / count) - mean_r * mean_r
        if var_r < 0.0:
            var_r = 0.0
        mean_returns[i] = np.float32(mean_r)
        std_returns[i] = np.float32(np.sqrt(var_r))
        mae[i] = np.float32(neg_sum / neg_count) if neg_count > 0 else 0.0
        mfe[i] = np.float32(pos_sum / pos_count) if pos_count > 0 else 0.0

        if eligible_count > 0:
            sc = max(pos_eligible, neg_eligible) / eligible_count
            if eligible_count < sign_min_effective_samples:
                shrink = eligible_count / max(sign_min_effective_samples, 1)
                sc = 0.5 + (sc - 0.5) * shrink
            sign_consistency[i] = np.float32(sc)

        if count >= 20:
            selected = np.empty(count, dtype=np.float32)
            idx = 0
            for j in range(n_samples):
                if not mask_matrix[i, j]:
                    continue
                r = returns[j]
                if not np.isfinite(r):
                    continue
                selected[idx] = np.float32(r)
                idx += 1
            selected.sort()
            hi_idx = min(max(int(0.95 * (count - 1)), 0), count - 1)
            lo_idx = min(max(int(0.05 * (count - 1)), 0), count - 1)
            p95 = abs(float(selected[hi_idx]))
            p5 = abs(float(selected[lo_idx]))
            tail_ratio[i] = np.float32(p95 / (p5 + 1e-9))

        if n_day_buckets > 0:
            counts = np.zeros(n_day_buckets, dtype=np.int32)
            active_days = 0
            for j in range(n_samples):
                if not mask_matrix[i, j]:
                    continue
                day_code = day_codes[j]
                if day_code < 0 or day_code >= n_day_buckets:
                    continue
                counts[day_code] += 1
            mean_count = 0.0
            for d in range(n_day_buckets):
                mean_count += counts[d]
                if counts[d] > 0:
                    active_days += 1
            mean_count /= max(n_day_buckets, 1)
            if active_days > 0 and mean_count > 0.0:
                var_count = 0.0
                for d in range(n_day_buckets):
                    diff = counts[d] - mean_count
                    var_count += diff * diff
                var_count /= n_day_buckets
                density_dispersion[i] = np.float32(
                    np.sqrt(var_count) / (mean_count + 1e-9)
                )

    return (
        support_counts,
        mean_returns,
        std_returns,
        sign_consistency,
        tail_ratio,
        mae,
        mfe,
        density_dispersion,
    )


def _compute_metrics_batch_numba(
    mask_matrix: np.ndarray,
    returns: np.ndarray,
    day_codes: np.ndarray,
    n_day_buckets: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    return _compute_batch_cheap_stats_numba(
        mask_matrix,
        returns,
        day_codes,
        n_day_buckets,
    )


@njit(cache=True, fastmath=True)
def _compute_path_arrays_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    side_mult: float,
    horizon: int,
    final_ret: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized path array computation.
    Returns: mfe, mae, time_to_mfe, time_to_mae, path_length
    """
    n = len(close)
    mfe = np.full(n, np.nan, dtype=np.float32)
    mae = np.full(n, np.nan, dtype=np.float32)
    t_mfe = np.full(n, np.nan, dtype=np.float32)
    t_mae = np.full(n, np.nan, dtype=np.float32)
    path_len = np.full(n, np.nan, dtype=np.float32)

    max_i = n - int(horizon) - 1
    for i in range(max(max_i + 1, 0)):
        entry = close[i]
        exit_px = close[i + horizon]
        if not (np.isfinite(entry) and np.isfinite(exit_px) and abs(entry) > 1e-12):
            continue

        # Pre-allocate windows for better performance
        window_start = i + 1
        window_end = i + horizon + 1
        if window_end > n:
            continue

        # Single-pass computation
        max_fav_val = -np.inf
        max_adv_val = -np.inf
        max_fav_idx = -1
        max_adv_idx = -1

        sum_abs_ret = 0.0
        prev_px = entry

        if side_mult > 0:
            # Long position
            final_ret[i] = np.float32((exit_px - entry) / entry)
            for j in range(window_start, window_end):
                fav_val = (high[j] - entry) / entry
                adv_val = (entry - low[j]) / entry
                if np.isfinite(fav_val) and fav_val > max_fav_val:
                    max_fav_val = fav_val
                    max_fav_idx = j - i
                if np.isfinite(adv_val) and adv_val > max_adv_val:
                    max_adv_val = adv_val
                    max_adv_idx = j - i

                curr_px = close[j]
                if np.isfinite(curr_px) and np.isfinite(prev_px):
                    sum_abs_ret += abs((curr_px - prev_px) / prev_px)
                prev_px = curr_px
        else:
            # Short position
            final_ret[i] = np.float32((entry - exit_px) / entry)
            for j in range(window_start, window_end):
                fav_val = (entry - low[j]) / entry
                adv_val = (high[j] - entry) / entry
                if np.isfinite(fav_val) and fav_val > max_fav_val:
                    max_fav_val = fav_val
                    max_fav_idx = j - i
                if np.isfinite(adv_val) and adv_val > max_adv_val:
                    max_adv_val = adv_val
                    max_adv_idx = j - i

                curr_px = close[j]
                if np.isfinite(curr_px) and np.isfinite(prev_px):
                    sum_abs_ret += abs((prev_px - curr_px) / prev_px)
                prev_px = curr_px

        if max_fav_idx >= 0:
            mfe[i] = np.float32(max(max_fav_val, 0.0))
            t_mfe[i] = np.float32(max_fav_idx)
        if max_adv_idx >= 0:
            mae[i] = np.float32(max(max_adv_val, 0.0))
            t_mae[i] = np.float32(max_adv_idx)

        path_len[i] = np.float32(sum_abs_ret)

    return mfe, mae, t_mfe, t_mae, path_len


@njit(cache=True, fastmath=True)
def _compute_cheap_stats_batch_numba(
    masks: np.ndarray,
    target_ret: np.ndarray,
    day_codes: np.ndarray,
    n_day_buckets: int,
    total_symbol_days: float,
    support_min: float,
    support_max: float,
    target_support: float,
    preferred_support_min: float,
    preferred_support_max: float,
) -> np.ndarray:
    """
    Compute cheap stats for all rules in a vectorized batch.

    Args:
        masks: (n_rules, n_samples) boolean mask matrix
        target_ret: (n_samples,) target returns
        day_codes: (n_samples,) day bucket codes
        n_day_buckets: number of day buckets
        total_symbol_days: total symbol days for trade density computation

    Returns:
        (n_rules, n_stats) array of stats
        Stats order: support_pct, support_ok, support_score, avg_trades,
                    density_dispersion, tail_ratio, mae, mfe,
                    mean_ret_mask, std_ret_mask, ret_uplift, sign_consistency
    """
    n_rules = masks.shape[0]
    n_samples = masks.shape[1]
    results = np.zeros((n_rules, 12), dtype=np.float32)

    # Pre-compute global statistics
    mean_ret_global = np.float32(np.nanmean(target_ret))

    for i in range(n_rules):
        mask = masks[i]
        n_active = np.sum(mask)

        if n_active < 2:
            results[i, :] = np.nan
            continue

        # Support metrics
        support_pct = np.float32(n_active / n_samples)
        support_ok = np.float32(
            1.0 if support_min <= support_pct <= support_max else 0.0
        )

        if preferred_support_min <= support_pct <= preferred_support_max:
            support_score = 1.0
        elif support_pct < preferred_support_min:
            span = max(preferred_support_min - support_min, 1e-9)
            relative = np.clip((support_pct - support_min) / span, 0.0, 1.0)
            support_score = 0.2 + (1.0 - 0.2) * relative
        else:
            span = max(support_max - preferred_support_max, 1e-9)
            relative = np.clip((support_max - support_pct) / span, 0.0, 1.0)
            support_score = 0.2 + (1.0 - 0.2) * relative

        # Average trades per day
        avg_trades = np.float32(n_active / total_symbol_days)

        # Density dispersion
        if n_day_buckets > 0 and day_codes is not None:
            active_codes = day_codes[mask]
            active_codes = active_codes[active_codes >= 0]
            if active_codes.size > 0:
                counts = np.bincount(active_codes, minlength=n_day_buckets).astype(
                    np.float32
                )
                mean_count = np.mean(counts)
                density_dispersion = np.std(counts) / (mean_count + 1e-9)
            else:
                density_dispersion = 0.0
        else:
            density_dispersion = 0.0

        # Target return metrics
        target_ret_masked = target_ret[mask]
        mean_ret_mask = np.float32(np.nanmean(target_ret_masked))
        std_ret_mask = np.float32(np.nanstd(target_ret_masked))
        ret_uplift = mean_ret_mask - mean_ret_global

        # Tail ratio (95th percentile / 5th percentile)
        sorted_ret = np.sort(target_ret_masked)
        p95_idx = int(0.95 * len(sorted_ret))
        p5_idx = int(0.05 * len(sorted_ret))
        if p95_idx < len(sorted_ret) and p5_idx < len(sorted_ret):
            tail_ratio = np.abs(sorted_ret[p95_idx]) / (
                np.abs(sorted_ret[p5_idx]) + 1e-9
            )
        else:
            tail_ratio = 1.0

        # MFE/MAE
        cumsum_ret = np.cumsum(target_ret_masked)
        peak = np.max(cumsum_ret)
        trough = np.min(cumsum_ret)
        mfe = np.abs(peak) if peak > 0 else 0.0
        mae = np.abs(trough) if trough < 0 else 0.0

        # Sign consistency
        n_pos = np.sum(target_ret_masked > 0)
        n_neg = np.sum(target_ret_masked < 0)
        n_total = n_pos + n_neg
        if n_total > 0:
            sign_consistency = np.float32(max(n_pos, n_neg) / n_total)
        else:
            sign_consistency = 0.5

        # Store results
        results[i, 0] = support_pct
        results[i, 1] = support_ok
        results[i, 2] = support_score
        results[i, 3] = avg_trades
        results[i, 4] = density_dispersion
        results[i, 5] = tail_ratio
        results[i, 6] = mae
        results[i, 7] = mfe
        results[i, 8] = mean_ret_mask
        results[i, 9] = std_ret_mask
        results[i, 10] = ret_uplift
        results[i, 11] = sign_consistency

    return results


def _compute_path_arrays_from_ohlc(
    data: pd.DataFrame, side: str, horizon: int, fallback_final_ret: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Build per-event trade-path arrays required by trade-path quality metrics.
    Uses close/high/low forward windows when available; falls back gracefully.

    Optimized with Numba for 20-100x speedup.
    """
    n = len(data)
    final_ret = np.asarray(fallback_final_ret, dtype=np.float32).copy()
    mfe = np.full(n, np.nan, dtype=np.float32)
    mae = np.full(n, np.nan, dtype=np.float32)
    t_mfe = np.full(n, np.nan, dtype=np.float32)
    t_mae = np.full(n, np.nan, dtype=np.float32)
    path_len = np.full(n, np.nan, dtype=np.float32)

    if horizon <= 0:
        return {
            "mfe": mfe,
            "mae": mae,
            "final_ret": final_ret,
            "time_to_mfe": t_mfe,
            "time_to_mae": t_mae,
            "path_length": path_len,
        }

    required_cols = {"close", "high", "low"}
    if not required_cols.issubset(set(data.columns)):
        return {
            "mfe": mfe,
            "mae": mae,
            "final_ret": final_ret,
            "time_to_mfe": t_mfe,
            "time_to_mae": t_mae,
            "path_length": path_len,
        }

    close = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=np.float32)
    high = pd.to_numeric(data["high"], errors="coerce").to_numpy(dtype=np.float32)
    low = pd.to_numeric(data["low"], errors="coerce").to_numpy(dtype=np.float32)
    side_mult = np.float32(-1.0 if str(side).lower() == "short" else 1.0)

    # Use Numba-optimized implementation
    mfe, mae, t_mfe, t_mae, path_len = _compute_path_arrays_numba(
        close, high, low, side_mult, horizon, final_ret
    )

    return {
        "mfe": mfe,
        "mae": mae,
        "final_ret": final_ret,
        "time_to_mfe": t_mfe,
        "time_to_mae": t_mae,
        "path_length": path_len,
    }


@njit(fastmath=False, cache=True)
def _safe_tanh_scale_numba(x: float, scale: float) -> float:
    """Numba-optimized tanh scaling for stability."""
    safe_scale = max(float(scale), 1e-9)
    return np.tanh(x / safe_scale)


@njit(fastmath=False, cache=True)
def _compute_path_quality_terms_numba(
    mfe_mae_ratio: np.ndarray,
    realized_profit_consistency: np.ndarray,
    trajectory_smoothness: np.ndarray,
    fold_medians: np.ndarray,
    eps: float,
    ratio_cap: float,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Numba-optimized computation of path quality terms.
    Returns: (median_mfe_mae, p10_mfe_mae, median_realized_profit_consistency, median_trajectory_smoothness, median_fold_median, mad_fold, worst_fold, iqr_mfe_mae)
    """
    n = len(mfe_mae_ratio)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Compute statistics
    median_mfe_mae = np.nanmedian(mfe_mae_ratio)
    p10_mfe_mae = np.nanpercentile(mfe_mae_ratio, 10)
    median_realized_profit_consistency = np.nanmedian(realized_profit_consistency)
    median_trajectory_smoothness = np.nanmedian(trajectory_smoothness)

    # Fold statistics
    n_folds = len(fold_medians)
    if n_folds > 0:
        median_fold_median = np.nanmedian(fold_medians)
        mad_fold = np.nanmedian(np.abs(fold_medians - median_fold_median))
        worst_fold = np.nanmin(fold_medians)
    else:
        median_fold_median = np.nan
        mad_fold = np.nan
        worst_fold = np.nan

    # IQR calculation
    q25 = np.nanpercentile(mfe_mae_ratio, 25)
    q75 = np.nanpercentile(mfe_mae_ratio, 75)
    iqr_mfe_mae = q75 - q25

    return (
        median_mfe_mae,
        p10_mfe_mae,
        median_realized_profit_consistency,
        median_trajectory_smoothness,
        median_fold_median,
        mad_fold,
        worst_fold,
        iqr_mfe_mae,
    )


@njit(fastmath=False, cache=True)
def _compute_fold_medians_numba(
    ratio: np.ndarray,
    fold_codes: np.ndarray,
    n_folds: int,
) -> np.ndarray:
    fold_medians = np.empty(n_folds, dtype=np.float64)
    fold_medians[:] = np.nan

    for fold_code in range(n_folds):
        count = 0
        for i in range(ratio.size):
            if fold_codes[i] == fold_code and np.isfinite(ratio[i]):
                count += 1

        if count == 0:
            continue

        values = np.empty(count, dtype=np.float64)
        pos = 0
        for i in range(ratio.size):
            if fold_codes[i] == fold_code and np.isfinite(ratio[i]):
                values[pos] = ratio[i]
                pos += 1

        values.sort()
        mid = count // 2
        if count % 2 == 0:
            fold_medians[fold_code] = 0.5 * (values[mid - 1] + values[mid])
        else:
            fold_medians[fold_code] = values[mid]

    return fold_medians


def compute_trade_path_quality_metrics(
    mfe: np.ndarray,
    mae: np.ndarray,
    final_ret: np.ndarray,
    time_to_mfe: np.ndarray,
    time_to_mae: np.ndarray,
    path_length: np.ndarray,
    fold_id: np.ndarray,
    eps: float = 1e-6,
    ratio_cap: float = 12.0,
) -> Dict[str, Any]:
    """
    Robust regime-level trade-path quality metrics.
    """
    mfe_arr = np.asarray(mfe, dtype=np.float32)
    mae_arr = np.asarray(mae, dtype=np.float32)
    final_ret_arr = np.asarray(final_ret, dtype=np.float32)
    time_to_mfe_arr = np.asarray(time_to_mfe, dtype=np.float32)
    time_to_mae_arr = np.asarray(time_to_mae, dtype=np.float32)
    path_length_arr = np.asarray(path_length, dtype=np.float32)
    fold_arr = np.asarray(fold_id)

    valid = (
        np.isfinite(mfe_arr)
        & np.isfinite(mae_arr)
        & np.isfinite(final_ret_arr)
        & np.isfinite(time_to_mfe_arr)
        & np.isfinite(time_to_mae_arr)
        & np.isfinite(path_length_arr)
    )
    if fold_arr.dtype.kind in {"f", "i", "u"}:
        valid &= np.isfinite(fold_arr.astype(np.float64, copy=False))
    else:
        valid &= ~pd.isna(fold_arr)

    n_obs = int(np.sum(valid))
    if n_obs == 0:
        return {
            "quality_stability_score": np.nan,
            "trade_path_quality_score": np.nan,
            "n_obs": 0,
            "n_folds": 0,
        }

    mfe_valid = np.clip(mfe_arr[valid], 0.0, None)
    mae_valid = np.clip(mae_arr[valid], 0.0, None)
    final_ret_valid = final_ret_arr[valid]
    path_length_valid = path_length_arr[valid]
    ratio = np.clip(mfe_valid / (mae_valid + eps), 0.0, ratio_cap).astype(
        np.float64, copy=False
    )
    realized_profit_consistency = np.clip(
        final_ret_valid / (mae_valid + eps), -ratio_cap, ratio_cap
    ).astype(np.float64, copy=False)
    trajectory_smoothness = np.clip(
        np.abs(final_ret_valid) / (path_length_valid + eps), 0.0, 1.0
    ).astype(np.float64, copy=False)

    fold_valid = fold_arr[valid]
    fold_codes, uniques = pd.factorize(fold_valid, sort=False)
    if len(uniques) > 0:
        fold_medians_raw = _compute_fold_medians_numba(
            ratio,
            np.asarray(fold_codes, dtype=np.int32),
            int(len(uniques)),
        )
        fold_medians_arr = fold_medians_raw[np.isfinite(fold_medians_raw)]
    else:
        fold_medians_arr = np.empty(0, dtype=np.float64)

    # Use Numba-optimized computation
    (
        median_mfe_mae,
        p10_mfe_mae,
        median_realized_profit_consistency,
        median_trajectory_smoothness,
        median_fold_median,
        mad_fold,
        worst_fold,
        iqr_mfe_mae,
    ) = _compute_path_quality_terms_numba(
        ratio,
        realized_profit_consistency,
        trajectory_smoothness,
        fold_medians_arr,
        eps,
        ratio_cap,
    )

    n_folds = int(len(fold_medians_arr))

    rel_mad_fold = (
        mad_fold / (median_fold_median + eps)
        if np.isfinite(median_fold_median)
        else np.nan
    )
    rel_iqr_pooled = iqr_mfe_mae / (median_mfe_mae + eps)

    if np.isfinite(median_fold_median) and np.isfinite(worst_fold):
        # Base stability score
        quality_stability_score = float(
            median_fold_median / (1.0 + rel_mad_fold + rel_iqr_pooled)
        )

        # Worst-fold penalty: quadratic penalty starts when worst fold < 90% of median
        worst_fold_ratio = worst_fold / (median_fold_median + eps)
        penalty_threshold = 0.9

        if worst_fold_ratio >= penalty_threshold:
            worst_fold_penalty = 1.0  # No penalty
        else:
            # Quadratic penalty: (ratio / threshold)^2
            worst_fold_penalty = max((worst_fold_ratio / penalty_threshold) ** 2, 0.1)

        # Apply penalty to quality_stability_score before tanh scaling
        penalized_quality_score = quality_stability_score * worst_fold_penalty

        # Use Numba-optimized tanh scaling
        stability_term = max(_safe_tanh_scale_numba(penalized_quality_score, 3.0), 0.01)
    else:
        quality_stability_score = np.nan
        stability_term = np.nan

    # Use Numba-optimized tanh scaling for all terms
    smoothness_term = max(_safe_tanh_scale_numba(median_mfe_mae, 3.0), 0.01)
    survivability_term = (
        max(np.sqrt(max(_safe_tanh_scale_numba(p10_mfe_mae, 2.0), 0.0)), 0.01)
        if np.isfinite(p10_mfe_mae)
        else np.nan
    )
    realized_profit_consistency_term = (
        max(_safe_tanh_scale_numba(median_realized_profit_consistency, 3.0), 0.01)
        if np.isfinite(median_realized_profit_consistency)
        else np.nan
    )
    trajectory_smoothness_term = (
        max(float(np.clip(median_trajectory_smoothness, 0.0, 1.0)), 0.01)
        if np.isfinite(median_trajectory_smoothness)
        else np.nan
    )

    composite_terms = np.array(
        [
            smoothness_term,
            survivability_term,
            stability_term,
            realized_profit_consistency_term,
            trajectory_smoothness_term,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(composite_terms)):
        trade_path_quality_score = np.nan
    else:
        # Prevent any single weak component from collapsing the full product to zero.
        composite_terms = np.clip(composite_terms, 0.05, 1.0)
        trade_path_quality_score = float(np.prod(composite_terms))

    return {
        "quality_stability_score": quality_stability_score,
        "trade_path_quality_score": trade_path_quality_score,
        "path_smoothness_term": float(smoothness_term),
        "path_survivability_term": float(survivability_term),
        "path_stability_term": float(stability_term),
        "path_realized_profit_consistency_term": float(
            realized_profit_consistency_term
        ),
        "path_trajectory_smoothness_term": float(trajectory_smoothness_term),
        "n_obs": n_obs,
        "n_folds": n_folds,
    }


def _clip_returns(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    lo = float(np.nanpercentile(x, 2.0))
    hi = float(np.nanpercentile(x, 98.0))
    return np.clip(x, lo, hi)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Point 11: Improve NaN handling in cosine similarity
    mask_a = np.isfinite(a)
    mask_b = np.isfinite(b)
    if not np.any(mask_a) or not np.any(mask_b):
        return 0.0
    a_f = np.nan_to_num(a, 0.0)
    b_f = np.nan_to_num(b, 0.0)
    norm_a = np.linalg.norm(a_f)
    norm_b = np.linalg.norm(b_f)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a_f, b_f) / (norm_a * norm_b))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_valid = np.isfinite(a)
    b_valid = np.isfinite(b)
    valid = a_valid & b_valid
    if np.sum(valid) < 3:
        return np.nan
    # Check for constant arrays
    if np.all(a[valid] == a[valid][0]) or np.all(b[valid] == b[valid][0]):
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


# =============================================================================
# DATA STRUCTURES & METADATA
# =============================================================================


@dataclass(frozen=True)
class FeatureMetadata:
    feature_name: str
    feature_index: int
    group: str  # 'trigger', 'location', 'regime'
    source_name: str
    source_family: str
    source_type: str  # 'boolean', 'continuous'
    booleanization_method: Optional[str] = None
    threshold_type: Optional[str] = None
    threshold_value: Optional[float] = None
    threshold_upper_value: Optional[float] = None
    description: str = ""
    regime_family: Optional[str] = None

    @property
    def interaction_group(self) -> str:
        if self.group == "location":
            return "location"
        elif self.group == "regime":
            return (
                f"regime:{self.regime_family}"
                if self.regime_family
                else "regime:unknown"
            )
        return self.group


@dataclass(frozen=True)
class MiningStageSpec:
    stage_name: str
    active_groups: Tuple[str, ...]  # e.g. ("regime", "location")
    allow_groups_in_rule: Tuple[str, ...]  # same as above
    output_dir_name: str  # e.g. "stage_a_context"
    allowed_group_pairs: Tuple[Tuple[str, str], ...]
    slot_order: Tuple[str, ...] = ("trigger", "location", "regime")
    context_rule_keys: Optional[List[str]] = None
    use_context_features: bool = False
    context_feature_group_name: str = "context"
    require_uplift: bool = False


@dataclass(frozen=True)
class RuleCondition:
    feature_name: str
    feature_index: int
    group: str
    normalized_value: int  # 1 (feature==1) or 0 (feature==0)
    raw_operator: str  # '<=', '>', '==', etc.
    raw_threshold: float
    raw_decision_type: Optional[str] = None
    default_left: Optional[bool] = None
    missing_type: Optional[str] = None

    def __repr__(self):
        val_str = "==1" if self.normalized_value == 1 else "==0"
        return f"{self.group}:{self.feature_name}{val_str}"


@dataclass
class ExtractedRule:
    rule_id: str  # Instance-specific ID
    canonical_key: str  # Slot-based identity
    conditions: List[RuleCondition]
    model_id: str
    fold_id: int
    seed: int
    tree_index: int
    leaf_index: int
    leaf_value: float
    support_train: int
    support_val: int = 0
    source_target: str = "primary_target"  # Target name provenance
    source_horizon: int = 0  # Horizon in bars provenance
    path_gain_sum: float = 0.0
    total_samples: int = 1
    rule_model_importance_score: float = 0.0


@dataclass
class CompiledTree:
    split_feature: np.ndarray
    threshold: np.ndarray
    split_gain: np.ndarray
    left_child: np.ndarray
    right_child: np.ndarray
    leaf_value: np.ndarray
    leaf_count: np.ndarray
    leaf_index: np.ndarray
    is_leaf: np.ndarray


# =============================================================================
# FEATURE PREPARATION
# =============================================================================


class FeatureProcessor:
    def __init__(self):
        self.metadata: Dict[str, FeatureMetadata] = {}
        self.feature_names: List[str] = []
        self.rank_audit_rows = []
        self.bool_support_audit_rows = []

    @staticmethod
    def _regime_source_type(source_name: str) -> str:
        return str(RIDGE_FEATURE_META.get(source_name, {}).get("type", "continuous"))

    @staticmethod
    def _regime_source_family(source_name: str) -> str:
        return str(RIDGE_FEATURE_META.get(source_name, {}).get("family", "unknown"))

    @staticmethod
    def _is_reserved_target_side_feature(source_name: str) -> bool:
        name = str(source_name)
        return name.startswith("target_") or name.endswith("_surprisal")



    def prepare_features(
        self,
        feature_dict: Dict[str, np.ndarray],
        timestamps: np.ndarray,
        symbol_codes: np.ndarray,
        cfg: Dict[str, Any],
        active_groups: Optional[Sequence[str]] = None,
        extra_binary_features: Optional[Dict[str, np.ndarray]] = None,
        extra_feature_group: str = "context",
    ) -> Tuple[np.ndarray, List[FeatureMetadata], Dict[str, pd.DataFrame]]:
        """
        Groups and booleanizes features with quality hardening.
        """
        raw_cols = []
        raw_names = []
        self.metadata = {}
        self.feature_names = []

        # If active_groups is None, default to all primary groups
        if active_groups is None:
            active_groups = ("trigger", "location", "regime")

        raw_source_features_by_group = collections.defaultdict(set)
        dropped_nuisance_sources_by_group = collections.defaultdict(set)
        dropped_nuisance_generated_feature_count = {"count": 0}

        def _register_boolean_column(
            source_arr: np.ndarray,
            bool_name: str,
            bool_arr: np.ndarray,
            group_name: str,
            src: str,
            family: str,
            *,
            booleanization_method: str,
            threshold_type: str,
            threshold_value: Optional[float],
            threshold_upper_value: Optional[float],
            description: str,
            min_support: int,
        ) -> None:
            n_valid = int(np.sum(~np.isnan(source_arr)))
            support = int(np.nansum(bool_arr))
            support_pct = support / n_valid if n_valid > 0 else 0.0

            self.bool_support_audit_rows.append(
                {
                    "generated_boolean": bool_name,
                    "group": group_name,
                    "source_feature": src,
                    "support": support,
                    "support_pct": support_pct,
                }
            )

            if support < min_support or support_pct > 0.95:
                tprint(
                    f"WARNING: generated boolean {bool_name} has extreme support ({support}, {support_pct:.2%})"
                )

            self._add_metadata(
                bool_name,
                group_name,
                "boolean",
                source_name=src,
                source_family=family,
                booleanization_method=booleanization_method,
                threshold_type=threshold_type,
                threshold_value=threshold_value,
                threshold_upper_value=threshold_upper_value,
                description=description,
            )
            raw_cols.append(bool_arr)
            raw_names.append(bool_name)

        def _add_binary_feature(
            src: str, group_name: str, raw_arr: np.ndarray, family: str
        ):
            nan_rate_before = float(np.isnan(raw_arr).mean())
            self.rank_audit_rows.append(
                {
                    "source_feature": src,
                    "group": group_name,
                    "nan_rate_before": nan_rate_before,
                    "nan_rate_ts": nan_rate_before,
                }
            )
            raw_arr_f32 = np.asarray(raw_arr, dtype=np.float32)
            self._add_metadata(
                src,
                group_name,
                "boolean",
                source_name=src,
                source_family=family,
                description=f"Raw binary passthrough feature {src}",
            )
            raw_cols.append(raw_arr_f32)
            raw_names.append(src)

        def _add_continuous_features_as_booleans(sources, group_name):
            min_support = int(cfg.get("min_feature_support", 10))

            for src in sources:
                if self._is_reserved_target_side_feature(src):
                    tprint(
                        f"WARNING: skipping reserved target-side feature '{src}' in miner feature prep"
                    )
                    continue
                if _should_drop_miner_nuisance_source(src, group_name, cfg):
                    dropped_nuisance_sources_by_group[group_name].add(src)
                    if group_name in {"regime", "location"}:
                        dropped_nuisance_generated_feature_count["count"] += 1
                    continue
                if src in feature_dict:
                    raw_source_features_by_group[group_name].add(src)
                    raw_arr = feature_dict[src]
                    nan_rate_before = float(np.isnan(raw_arr).mean())

                    if group_name == "regime":
                        family = self._regime_source_family(src)
                        if self._regime_source_type(src) == "binary":
                            _add_binary_feature(src, group_name, raw_arr, family)
                            continue
                    elif group_name == "location":
                        family = LOC_CONTINUOUS_FAMILY_MAP.get(src, "context")
                    else:
                        family = src.split("_")[0] if "_" in src else group_name
                    
                    # Robust fallback for truncated or broken family names (e.g. "", "-", "_")
                    if not family or len(family) < 2:
                        family = group_name

                    # Heartbeat every 20 features to avoid 10-minute silence
                    if (len(raw_cols) + 1) % 20 == 0:
                        tprint(f"FeaturePrep Progress: {group_name} feature {len(raw_cols) + 1} processing ({src})")

                    # Redundant rank-normalization removed to eliminate O(N^2) bottleneck.
                    # nan_rate_ts falls back to raw NaN rate as no ranking transform is applied here.
                    nan_rate_ts = nan_rate_before

                    self.rank_audit_rows.append(
                        {
                            "source_feature": src,
                            "group": group_name,
                            "nan_rate_before": nan_rate_before,
                            "nan_rate_ts": nan_rate_ts,
                        }
                    )

                    raw_arr_f32 = np.asarray(raw_arr, dtype=np.float32)
                    self._add_metadata(
                        src,
                        group_name,
                        "continuous",
                        source_name=src,
                        source_family=family,
                        description=f"Continuous passthrough feature {src}",
                    )
                    raw_cols.append(raw_arr_f32)
                    raw_names.append(src)

        # 1. Trigger Features — DISABLED (all trigger features removed from pipeline)
        if "trigger" in active_groups:
            tprint("TRIGGER features disabled — skipping trigger group.")

        # 2. Location Features
        if "location" in active_groups:
            # Continuous location features are the sole location source family.
            _add_continuous_features_as_booleans(CONTINUOUS_LOCATION_COLS, "location")

        # 3. Regime Features (continuous -> hybrid booleanize)
        if "regime" in active_groups:
            time_keys_set = set(TIME_FEATURE_KEYS)
            regime_sources = sorted(
                list(
                    (set(RIDGE_FEATURE_COLS) | set(TEST_FEATURE_KEYS)) - time_keys_set
                )
            )
            _add_continuous_features_as_booleans(regime_sources, "regime")

        # 4. Extra Binary Features (e.g. Stage A Contexts)
        if extra_binary_features:
            for name, arr in extra_binary_features.items():
                if self._is_reserved_target_side_feature(name):
                    tprint(
                        f"WARNING: skipping reserved target-side extra feature '{name}' in miner feature prep"
                    )
                    continue
                raw_source_features_by_group[extra_feature_group].add(name)
                arr_f32 = arr.astype(np.float32)
                self._add_metadata(
                    name,
                    extra_feature_group,
                    "boolean",
                    source_name=name,
                    source_family=extra_feature_group,
                    description=f"Extra feature from {extra_feature_group}",
                )
                raw_cols.append(arr_f32)
                raw_names.append(name)

        if not raw_cols:
            return (
                np.empty((len(timestamps), 0)),
                [],
                pd.DataFrame(
                    columns=[
                        "feature_name",
                        "status",
                        "reason",
                        "support",
                        "group",
                        "regime_family",
                    ]
                ),
            )

        X_raw = np.column_stack(raw_cols)

        # Quality Hardening: Drop degenerate/duplicate columns
        X_clean, retained_names, audit_df = self._run_feature_quality_checks(
            X_raw, raw_names, cfg
        )

        if cfg.get("boolean_only", False):
            is_bool = np.array(
                [self.metadata[n].source_type == "boolean" for n in retained_names]
            )
            if np.any(is_bool):
                X_clean = X_clean[:, is_bool]
                retained_names = [n for i, n in enumerate(retained_names) if is_bool[i]]
            else:
                tprint("WARNING: boolean_only=True but no boolean features found!")

        if retained_names:
            grouped_indices: Dict[
                str, collections.deque[int]
            ] = collections.defaultdict(collections.deque)
            group_order = list(dict.fromkeys(active_groups))
            for idx, name in enumerate(retained_names):
                grouped_indices[self.metadata[name].group].append(idx)
            remaining_groups = [
                g for g in grouped_indices.keys() if g not in set(group_order)
            ]
            interleave_order: List[int] = []
            full_group_order = group_order + sorted(remaining_groups)
            while any(grouped_indices[g] for g in full_group_order):
                for g in full_group_order:
                    if grouped_indices[g]:
                        interleave_order.append(grouped_indices[g].popleft())
            if interleave_order and interleave_order != list(
                range(len(retained_names))
            ):
                X_clean = X_clean[:, interleave_order]
                retained_names = [retained_names[i] for i in interleave_order]

        if dropped_nuisance_sources_by_group:
            removed_sources = sum(
                len(sources) for sources in dropped_nuisance_sources_by_group.values()
            )
            removal_summary = ", ".join(
                f"{group}={len(sorted(sources))}"
                for group, sources in sorted(dropped_nuisance_sources_by_group.items())
            )
            tprint(
                f"Miner nuisance feature exclusion active: removed_sources={removed_sources}, removed_generated_features~={dropped_nuisance_generated_feature_count['count']}, {removal_summary}"
            )

        audit_df["group"] = [self.metadata[n].group for n in audit_df["feature_name"]]
        audit_df["regime_family"] = [
            self.metadata[n].regime_family for n in audit_df["feature_name"]
        ]

        raw_source_counts = {k: len(v) for k, v in raw_source_features_by_group.items()}

        # Summary by group
        group_summary = []
        all_groups = list(active_groups) if active_groups else []
        if extra_feature_group and extra_feature_group not in all_groups:
            all_groups.append(extra_feature_group)

        for g in all_groups:
            g_df = audit_df[audit_df["group"] == g]
            if g_df.empty:
                continue
            retained = g_df[g_df["status"] == "retained"]
            dropped = g_df[g_df["status"] == "dropped"]
            drop_reasons = dropped["reason"].value_counts().to_dict()

            support_stats = (
                retained["support"].describe()
                if not retained.empty
                else pd.Series(dtype=float)
            )

            group_summary.append(
                {
                    "group": g,
                    "raw_source_features": raw_source_counts.get(g, 0),
                    "generated_booleans": len(g_df),
                    "retained": len(retained),
                    "dropped": len(dropped),
                    "drop_reason_all_zeros": drop_reasons.get("all_zeros", 0),
                    "drop_reason_all_ones": drop_reasons.get("all_ones", 0),
                    "drop_reason_low_support": sum(
                        v
                        for k, v in drop_reasons.items()
                        if k.startswith("low_support")
                    ),
                    "drop_reason_duplicate": sum(
                        v
                        for k, v in drop_reasons.items()
                        if k.startswith("duplicate_of")
                    ),
                    "support_min": support_stats.get("min", np.nan),
                    "support_p25": support_stats.get("25%", np.nan),
                    "support_median": support_stats.get("50%", np.nan),
                    "support_p75": support_stats.get("75%", np.nan),
                    "support_max": support_stats.get("max", np.nan),
                }
            )

        feature_quality_summary_by_group = pd.DataFrame(group_summary)

        # Summary by regime family
        regime_df = audit_df[audit_df["group"] == "regime"]
        regime_summary = []
        if not regime_df.empty:
            for fam, f_df in regime_df.groupby("regime_family"):
                retained = f_df[f_df["status"] == "retained"]
                dropped = f_df[f_df["status"] == "dropped"]
                regime_summary.append(
                    {
                        "regime_family": fam,
                        "generated_booleans": len(f_df),
                        "retained": len(retained),
                        "dropped": len(dropped),
                    }
                )
        feature_quality_summary_by_regime_family = pd.DataFrame(regime_summary)

        if not feature_quality_summary_by_regime_family.empty:
            retained_counts = feature_quality_summary_by_regime_family.set_index(
                "regime_family"
            )["retained"]
            total_retained_regime = retained_counts.sum()
            if total_retained_regime > 0:
                max_fam = retained_counts.idxmax()
                max_val = retained_counts.max()
                if max_val / total_retained_regime > 0.5:
                    tprint(
                        f"WARNING: Regime family '{max_fam}' dominates retained regime features ({max_val}/{total_retained_regime})."
                    )

        # tprints
        total_raw = sum(raw_source_counts.values())
        total_gen = len(audit_df)
        total_retained = len(retained_names)
        tprint(
            f"FeaturePrep: total raw={total_raw}, generated={total_gen}, retained={total_retained}"
        )

        for g_sum in group_summary:
            tprint(
                f"  - {g_sum['group']}: retained {g_sum['retained']} / {g_sum['generated_booleans']} generated (from {g_sum['raw_source_features']} raw)"
            )

        dropped_df = audit_df[audit_df["status"] == "dropped"]
        if not dropped_df.empty:
            top_dropped = dropped_df["reason"].value_counts().head(10)
            tprint("Top 10 dropped features by reason:")
            for reason, count in top_dropped.items():
                tprint(f"  - {reason}: {count}")

        # Rank Audit tprints
        rank_audit_df = pd.DataFrame()
        if self.rank_audit_rows:
            rank_audit_df = pd.DataFrame(self.rank_audit_rows)
            rank_audit_df["worst_nan"] = rank_audit_df[
                ["nan_rate_before", "nan_rate_ts"]
            ].max(axis=1)
            top_nan = rank_audit_df.sort_values("worst_nan", ascending=False).head(10)
            tprint("Top 10 features with worst NaN rates:")
            for row in top_nan.itertuples(index=False, name=None):
                tprint(f"  - {row[0]}:{row[1]} -> before={row[2]:.2%}, ts={row[3]:.2%}")
        
        # Additional diagnostics for FeaturePrep
        tprint(f"DEBUG: audit_df size={len(audit_df)}, symbols={audit_df['symbol'].nunique() if 'symbol' in audit_df.columns else 'N/A'}")
        if 'reason' in audit_df.columns:
            tprint(f"DEBUG: Rejection reasons summary: {audit_df['reason'].value_counts().to_dict()}")

        bool_support_audit_df = pd.DataFrame()
        if self.bool_support_audit_rows:
            bool_support_audit_df = pd.DataFrame(self.bool_support_audit_rows)
            n_samples = len(timestamps)
            bool_support_audit_df["usable_support"] = np.minimum(
                bool_support_audit_df["support"],
                n_samples - bool_support_audit_df["support"],
            )
            top_imbal = bool_support_audit_df.sort_values("usable_support").head(10)
            tprint("Top 10 generated booleans with lowest usable support:")
            for row in top_imbal.itertuples(index=False, name=None):
                tprint(
                    f"  - {row[0]}:{row[1]}:{row[2]} -> support={row[3]}, usable={row[4]}"
                )

        if not rank_audit_df.empty and not bool_support_audit_df.empty:
            booleanization_support_audit = pd.merge(
                bool_support_audit_df,
                rank_audit_df.drop(columns=["worst_nan"]),
                on=["source_feature", "group"],
                how="left",
            )
        else:
            booleanization_support_audit = bool_support_audit_df

        audits = {
            "feature_quality_audit": audit_df,
            "feature_quality_summary_by_group": feature_quality_summary_by_group,
            "feature_quality_summary_by_regime_family": feature_quality_summary_by_regime_family,
            "booleanization_support_audit": booleanization_support_audit,
        }

        # Re-index metadata based on final retained columns
        retained_metadata = []
        old_metadata = {m.feature_name: m for m in self.metadata.values()}
        for i, name in enumerate(retained_names):
            m = old_metadata[name]
            # Create fresh copy with updated index
            new_m = FeatureMetadata(
                feature_name=m.feature_name,
                feature_index=i,
                group=m.group,
                source_name=m.source_name,
                source_family=m.source_family,
                source_type=m.source_type,
                booleanization_method=m.booleanization_method,
                threshold_type=m.threshold_type,
                threshold_value=m.threshold_value,
                threshold_upper_value=m.threshold_upper_value,
                description=m.description,
                regime_family=m.regime_family,
            )
            retained_metadata.append(new_m)

        return X_clean, retained_metadata, audits

    def _add_metadata(self, name, group, src_type, **kwargs):
        idx = len(self.feature_names)
        self.feature_names.append(name)

        source_name = kwargs.get("source_name", name)
        regime_family = None
        if group == "regime":
            if source_name in RIDGE_FEATURE_META:
                regime_family = RIDGE_FEATURE_META[source_name].get("family")
            else:
                regime_family = kwargs.get("source_family", "unknown")

        self.metadata[name] = FeatureMetadata(
            feature_name=name,
            feature_index=idx,
            group=group,
            source_name=source_name,
            source_family=kwargs.get("source_family", "unknown"),
            source_type=src_type,
            booleanization_method=kwargs.get("booleanization_method"),
            threshold_type=kwargs.get("threshold_type"),
            threshold_value=kwargs.get("threshold_value"),
            threshold_upper_value=kwargs.get("threshold_upper_value"),
            description=kwargs.get("description", ""),
            regime_family=regime_family,
        )



    def _run_feature_quality_checks(
        self, X: np.ndarray, names: List[str], cfg: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        """
        Drops degenerate or duplicate boolean columns.
        """
        n_samples = X.shape[0]
        min_support = int(cfg.get("min_feature_support", 10))

        audit_rows = []
        retained_indices = []
        retained_names = []
        # Deduplicate only within the same raw source feature.
        # Cross-source collisions are expected under aggressive booleanization
        # and must remain available so Stage A does not erase whole groups/families.
        hash_registry = {}

        for i, name in enumerate(names):
            col = X[:, i]
            metadata = self.metadata[name]
            finite_mask = np.isfinite(col)
            finite_count = int(np.sum(finite_mask))
            n_ones = int(np.sum(col == 1))
            n_zeros = int(np.sum(col == 0))

            dropped = False
            reason = "retained"

            if metadata.source_type == "continuous":
                if finite_count == 0:
                    dropped = True
                    reason = "all_nan"
                elif finite_count < min_support:
                    dropped = True
                    reason = f"low_finite_support_{finite_count}<{min_support}"
                else:
                    finite_vals = np.asarray(col[finite_mask], dtype=np.float32)
                    if np.nanstd(finite_vals) <= 1e-12:
                        dropped = True
                        reason = "near_constant_continuous"
                    else:
                        dedupe_scope = (metadata.group, metadata.source_name)
                        col_hash = hashlib.sha1(
                            np.asarray(np.round(finite_vals, 6), dtype=np.float32).tobytes()
                        ).hexdigest()
                        hash_key = (dedupe_scope, col_hash)
                        if hash_key in hash_registry:
                            dropped = True
                            reason = f"duplicate_of_{hash_registry[hash_key]}"
                        else:
                            hash_registry[hash_key] = name
            else:
                if n_ones == 0:
                    dropped = True
                    reason = "all_zeros"
                elif n_zeros == 0:
                    dropped = True
                    reason = "all_ones"
                elif n_ones < min_support:
                    dropped = True
                    reason = f"low_support_{int(n_ones)}<{min_support}"
                else:
                    dedupe_scope = (metadata.group, metadata.source_name)
                    col_hash = hashlib.sha1(col.tobytes()).hexdigest()
                    hash_key = (dedupe_scope, col_hash)
                    if hash_key in hash_registry:
                        dropped = True
                        reason = f"duplicate_of_{hash_registry[hash_key]}"
                    else:
                        hash_registry[hash_key] = name

            audit_rows.append(
                {
                    "feature_name": name,
                    "status": "dropped" if dropped else "retained",
                    "reason": reason,
                    "support": int(finite_count if metadata.source_type == "continuous" else n_ones),
                }
            )

            if not dropped:
                retained_indices.append(i)
                retained_names.append(name)

        if not retained_indices:
            return (
                np.empty((X.shape[0], 0)),
                [],
                pd.DataFrame(columns=["feature_name", "status", "reason", "support"]),
            )

        X_clean = X[:, retained_indices]
        return X_clean, retained_names, pd.DataFrame(audit_rows)


# =============================================================================
# MODEL TRAINING & CONSTRAINTS
# =============================================================================


@njit(parallel=True, cache=True, fastmath=True)
def _make_regime_weights_numba(
    returns: np.ndarray,
    symbols: np.ndarray,
    horizon: int,
    alpha: float,
    w_min: float,
    w_max: float,
    eps: float,
) -> np.ndarray:
    """Numba-optimized regime weights computation."""
    n = len(returns)
    weights = np.ones(n, dtype=np.float32)
    window = int(np.sqrt(horizon))

    # Get unique symbols and their indices
    unique_syms = np.unique(symbols)

    for sym in unique_syms:
        # Find indices for this symbol
        idx = np.where(symbols == sym)[0]
        if idx.size == 0:
            continue

        r = returns[idx]
        abs_ret = np.abs(r)
        local_mean_abs = np.zeros(idx.size, dtype=np.float32)

        # First pass: compute local mean absolute returns
        for j in range(idx.size):
            lo = max(0, j - window)
            hi = min(idx.size, j + window + 1)
            # Compute mean of absolute returns in window
            local_mean_abs[j] = np.mean(abs_ret[lo:hi])

        # Second pass: compute weights based on percentile and direction
        for j in range(idx.size):
            hist_lo = max(0, j - window)
            history = local_mean_abs[hist_lo : j + 1]
            # Compute percentile (fraction of history <= current value)
            percentile = np.mean(history <= local_mean_abs[j])

            lo = max(0, j - window)
            hi = min(idx.size, j + window + 1)
            local = r[lo:hi]

            # Direction agreement
            pos_frac = np.mean(local > 0)
            neg_frac = np.mean(local < 0)
            dir_agree = max(pos_frac, neg_frac)
            persistence = max(0.0, 2.0 * (dir_agree - 0.5))

            # Intensity and harmonic mean
            intensity = np.tanh(2.0 * percentile)
            harmonic = (2.0 * persistence * intensity) / (persistence + intensity + eps)
            weights[idx[j]] = np.float32(1.0 + alpha * harmonic)

    # Point 7: Clip carefully - respect 1.0 default if w_min/w_max are reached
    weights = np.clip(weights, np.float32(min(w_min, 1.0)), np.float32(max(w_max, 1.0)))
    return weights


def make_regime_weights(
    fwd_ret: np.ndarray,
    symbol_id: np.ndarray,
    horizon: int = 10,
    alpha: float = 1.0,
    w_min: float = 0.75,
    w_max: float = 2.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Regime-coherence harmonic sample weighting.

    Encourages rows that belong to persistent directional neighborhoods with
    elevated *relative* local intensity (vol-normalized via local percentile
    rank), without directly rewarding raw magnitude.

    Optimized with Numba for 10-50x speedup.
    """
    n = int(len(fwd_ret))
    if n == 0:
        return np.empty(0, dtype=np.float32)

    if len(symbol_id) != n:
        raise ValueError("symbol_id length must match fwd_ret length")

    returns = np.asarray(fwd_ret, dtype=np.float32)
    symbols = np.asarray(symbol_id)
    if symbols.dtype.kind not in ("i", "u"):
        symbols, _ = pd.factorize(symbols, sort=False)
        symbols = np.asarray(symbols, dtype=np.int32)
    else:
        symbols = np.asarray(symbols, dtype=np.int32)

    # Use Numba-optimized implementation
    return _make_regime_weights_numba(
        returns, symbols, horizon, alpha, w_min, w_max, eps
    )


def _support_preference_score_scalar(
    support_pct: float,
    *,
    target_pct: float,
    preferred_low_pct: float,
    preferred_high_pct: float,
) -> float:
    dist = abs(float(support_pct) - float(target_pct))
    base = max(0.0, 1.0 - dist / max(float(target_pct), 1e-9))
    preferred = 1.0 if preferred_low_pct <= support_pct <= preferred_high_pct else 0.0
    return float(0.7 * base + 0.3 * preferred)


def make_support_preference_weights(
    X: np.ndarray,
    *,
    target_pct: float = TARGET_SUPPORT,
    preferred_low_pct: float = PREFERRED_SUPPORT_MIN,
    preferred_high_pct: float = PREFERRED_SUPPORT_MAX,
    strength: float = 0.20,
    w_min: float = 0.85,
    w_max: float = 1.25,
) -> np.ndarray:
    """
    Softly upweight rows activated by boolean features whose support lies near the
    preferred band. This biases split discovery toward target-support regimes
    without imposing any hard gate.
    """
    n_rows = int(X.shape[0])
    if n_rows == 0:
        return np.empty(0, dtype=np.float32)

    X_bin = np.asarray(X > 0.5, dtype=np.float32)
    if X_bin.shape[1] == 0:
        return np.ones(n_rows, dtype=np.float32)

    support_pct = np.mean(X_bin, axis=0, dtype=np.float64)
    feature_pref = np.array(
        [
            _support_preference_score_scalar(
                float(p),
                target_pct=target_pct,
                preferred_low_pct=preferred_low_pct,
                preferred_high_pct=preferred_high_pct,
            )
            for p in support_pct
        ],
        dtype=np.float32,
    )
    global_pref = float(np.mean(feature_pref)) if feature_pref.size > 0 else 0.0
    active_count = np.sum(X_bin, axis=1, dtype=np.float32)
    active_pref_sum = X_bin @ feature_pref
    row_pref = np.where(
        active_count > 0.0,
        active_pref_sum / np.maximum(active_count, 1.0),
        global_pref,
    )
    centered = row_pref - global_pref
    weights = 1.0 + float(strength) * centered
    return np.clip(weights, w_min, w_max).astype(np.float32, copy=False)


def make_surprisal_sample_weights(
    surprisal_bits: np.ndarray,
    *,
    alpha: float = 0.20,
    reference_bits: float = 3.0,
    w_min: float = 1.0,
    w_max: float = 1.20,
) -> np.ndarray:
    """
    Build a bounded surprisal multiplier for sample weights.
    """
    if reference_bits <= 0:
        raise ValueError("reference_bits must be > 0")
    surprisal = np.asarray(surprisal_bits, dtype=np.float32)
    scaled = np.clip(surprisal / np.float32(reference_bits), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    weights = 1.0 + (float(alpha) * scaled)
    return np.clip(weights, w_min, w_max).astype(np.float32, copy=False)


def make_excursion_asymmetry_weights(
    mfe_atr: np.ndarray,
    mae_atr: np.ndarray,
    side: str,
    alpha: float = 0.15,
    w_min: float = 0.85,
    w_max: float = 1.20,
    neutral_value: float = 1.0,
) -> np.ndarray:
    """
    Mild miner-only weighting that prefers samples with directional excursion asymmetry.

    Parameters
    ----------
    mfe_atr : array
        Maximum favorable excursion in ATR units, non-negative.
    mae_atr : array
        Maximum adverse excursion in ATR units. Can be signed negative or absolute.
    side : {"long", "short"}
        Included for symmetry / future extensions. Current formulation is side-agnostic
        because asymmetry is defined as favorable vs adverse excursion.
    alpha : float
        Strength of weighting. Keep small.
    w_min, w_max : float
        Clip range for the asymmetry weight.
    neutral_value : float
        Default weight for rows where asymmetry cannot be computed.

    Returns
    -------
    np.ndarray
        Per-sample multiplicative weights.
    """
    mfe = np.asarray(mfe_atr, dtype=np.float64)
    mae = np.asarray(mae_atr, dtype=np.float64)

    # Treat MAE as magnitude
    mae_abs = np.abs(mae)

    valid = np.isfinite(mfe) & np.isfinite(mae_abs)
    w = np.full(mfe.shape, neutral_value, dtype=np.float64)

    # Positive score means favorable excursion dominates adverse excursion
    asym = np.zeros_like(mfe, dtype=np.float64)
    asym[valid] = mfe[valid] - mae_abs[valid]

    # Smooth bounded mapping; small alpha only
    w[valid] = 1.0 + alpha * np.tanh(asym[valid])

    return np.clip(w, w_min, w_max).astype(np.float32, copy=False)


def build_miner_sample_weights(
    target: np.ndarray,
    X: np.ndarray,
    symbol_id: np.ndarray,
    cfg: Dict[str, Any],
    *,
    horizon: int,
    surprisal_bits: Optional[np.ndarray] = None,
    mfe_atr: Optional[np.ndarray] = None,
    mae_atr: Optional[np.ndarray] = None,
    side: str = "long",
) -> np.ndarray:
    symbol_codes_tr, _ = pd.factorize(symbol_id, sort=False)
    sample_weight = make_regime_weights(
        target,
        symbol_codes_tr.astype(np.int32, copy=False),
        horizon=horizon,
    )
    sample_weight = sample_weight * make_support_preference_weights(
        X,
        target_pct=float(cfg.get("support_preference_target_pct", TARGET_SUPPORT)),
        preferred_low_pct=float(
            cfg.get("support_preference_preferred_low_pct", PREFERRED_SUPPORT_MIN)
        ),
        preferred_high_pct=float(
            cfg.get("support_preference_preferred_high_pct", PREFERRED_SUPPORT_MAX)
        ),
        strength=float(cfg.get("support_preference_strength", 0.20)),
        w_min=float(cfg.get("support_preference_weight_min", 0.85)),
        w_max=float(cfg.get("support_preference_weight_max", 1.25)),
    )
    if surprisal_bits is not None:
        sample_weight = sample_weight * make_surprisal_sample_weights(
            surprisal_bits,
            alpha=float(cfg.get("surprisal_weight_alpha", 0.20)),
            reference_bits=float(cfg.get("surprisal_weight_reference_bits", 3.0)),
            w_min=float(cfg.get("surprisal_weight_min", 1.0)),
            w_max=float(cfg.get("surprisal_weight_max", 1.20)),
        )

    use_excursion_asymmetry_weights = cfg.get("use_excursion_asymmetry_weights", True)
    if (
        use_excursion_asymmetry_weights
        and mfe_atr is not None
        and mae_atr is not None
    ):
        sample_weight = sample_weight * make_excursion_asymmetry_weights(
            mfe_atr=mfe_atr,
            mae_atr=mae_atr,
            side=side,
            alpha=0.15,
            w_min=0.85,
            w_max=1.20,
        )

    # Rebalance weights to [0.5, 2.0] using MinMax scaling instead of simple clip
    w_min_final = 0.5
    w_max_final = 2.0
    w_curr_min = np.min(sample_weight)
    w_curr_max = np.max(sample_weight)
    if w_curr_max > w_curr_min:
        sample_weight = w_min_final + (sample_weight - w_curr_min) * (w_max_final - w_min_final) / (w_curr_max - w_curr_min)
    else:
        sample_weight = np.full_like(sample_weight, (w_min_final + w_max_final) / 2.0)
    return sample_weight.astype(np.float32, copy=False)


def make_ridge_vol_weights(
    fwd_ret: np.ndarray,
    *,
    window: int = 20,
    w_min: float = 0.5,
    w_max: float = 2.0,
) -> np.ndarray:
    """
    Ridge sample weights by inverse rolling volatility.
    
    Heteroscedasticity correction: downweight high-volatility periods
    where Ridge's homoscedasticity assumption is violated.
    
    Parameters
    ----------
    fwd_ret : np.ndarray
        Forward returns array
    window : int
        Rolling window for volatility computation (default: 20)
    w_min : float
        Minimum weight (default: 0.5)
    w_max : float
        Maximum weight (default: 2.0)
        
    Returns
    -------
    np.ndarray
        Sample weights in [w_min, w_max]
    """
    import pandas as pd
    
    n = len(fwd_ret)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    
    # Compute rolling volatility using pandas
    vol = pd.Series(fwd_ret).rolling(window=window, min_periods=max(5, window // 4)).std().values
    
    # Fill NaN with median volatility
    vol_median = np.nanmedian(vol)
    vol = np.nan_to_num(vol, nan=vol_median)
    
    # Inverse volatility weights
    weights = 1.0 / (vol + 1e-6)
    
    # Normalize to median and clip
    weights = weights / (np.median(weights) + 1e-9)
    weights = np.clip(weights, w_min, w_max)
    
    return weights.astype(np.float32, copy=False)


def make_fee_aware_target_weights(
    target: np.ndarray,
    *,
    fee_buffer: float = 0.002,
    near_zero_weight: float = 0.25,
    large_target_weight: float = 1.25,
    large_target_multiple: float = 3.0,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Downweight targets that are too small to clear fees and modestly upweight
    targets that are materially larger than the fee buffer.
    """
    target = np.asarray(target, dtype=np.float32)
    if target.size == 0:
        return np.empty(0, dtype=np.float32)

    abs_target = np.abs(target)
    safe_fee = max(float(fee_buffer), eps)
    high_cutoff = max(float(large_target_multiple) * safe_fee, safe_fee + eps)
    weights = np.ones(target.shape[0], dtype=np.float32)

    near_mask = abs_target <= safe_fee
    high_mask = abs_target >= high_cutoff
    mid_mask = ~(near_mask | high_mask)

    weights[near_mask] = np.float32(near_zero_weight)
    weights[high_mask] = np.float32(large_target_weight)
    if np.any(mid_mask):
        rel = (abs_target[mid_mask] - safe_fee) / max(high_cutoff - safe_fee, eps)
        weights[mid_mask] = np.float32(
            near_zero_weight
        ) + rel.astype(np.float32) * np.float32(
            large_target_weight - near_zero_weight
        )

    return weights.astype(np.float32, copy=False)


class InteractionModel:
    """
    LightGBM is trained without strict interaction constraints.
    Structural validity of rule paths is enforced in:
        RuleExtractor._is_path_valid()
    using interaction_group metadata.
    """

    def __init__(
        self,
        metadata: List[FeatureMetadata],
        cfg: Dict[str, Any],
        side: str = "long",
        allowed_group_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        self.metadata = metadata
        self.cfg = cfg
        self.side = side
        self.allowed_group_pairs = allowed_group_pairs
        self.constraints = self._build_interaction_constraints()

    def _build_interaction_constraints(self) -> List[List[int]]:
        """
        Build interaction constraints for LightGBM.
        """
        # Interaction constraints removed per user request to allow more combinations.
        return []

    def _verify_constraints(
        self, constraints, trigger_idxs, location_idxs, regime_idxs
    ):
        """
        Hardening: Verify no same-group pairs.
        """
        for c in constraints:
            if len(c) == 2:
                idx1, idx2 = c
                m1 = self.metadata[idx1]
                m2 = self.metadata[idx2]
                if m1.group == m2.group:
                    raise ValueError(
                        f"Constraint violation: {m1.group}-{m2.group} pair ({idx1}, {idx2})"
                    )

        # Summary
        t_l = sum(
            1
            for c in constraints
            if len(c) == 2
            and {self.metadata[c[0]].group, self.metadata[c[1]].group}
            == {"trigger", "location"}
        )
        t_r = sum(
            1
            for c in constraints
            if len(c) == 2
            and {self.metadata[c[0]].group, self.metadata[c[1]].group}
            == {"trigger", "regime"}
        )
        l_r = sum(
            1
            for c in constraints
            if len(c) == 2
            and {self.metadata[c[0]].group, self.metadata[c[1]].group}
            == {"location", "regime"}
        )
        tprint(
            f"Constraints built: T-L={t_l}, T-R={t_r}, L-R={l_r}, Singletons={len(self.metadata)}"
        )

    def get_constraint_summary(self) -> Dict[str, Any]:
        import collections

        result = {
            "total_singletons": len(self.metadata),
            "total_constraints": (
                len(self.constraints) if self.constraints is not None else 0
            ),
            "mode": "training permissive / validation strict",
        }

        groups = set(m.group for m in self.metadata)
        for g in groups:
            result[f"num_{g}"] = sum(1 for m in self.metadata if m.group == g)

        regime_families = set(
            m.regime_family for m in self.metadata if m.group == "regime"
        )
        for rf in regime_families:
            result[f"num_regime_{rf}"] = sum(
                1
                for m in self.metadata
                if m.group == "regime" and m.regime_family == rf
            )

        if not self.constraints:
            return result

        summary = collections.defaultdict(int)
        for c in self.constraints:
            if len(c) == 1:
                summary["singleton"] += 1
                m = self.metadata[c[0]]
                summary[f"singleton_{m.group}"] += 1
            else:
                groups = set(self.metadata[i].group for i in c)
                if groups == {"regime"}:
                    summary["regime_cluster"] += 1
                elif groups == {"location"}:
                    summary["location_cluster"] += 1
                else:
                    summary["mixed_cluster"] += 1
        result.update(summary)
        return result

    def train_fold(
        self,
        X_tr,
        y_tr,
        symbol_id_tr,
        surprisal_bits_tr,
        X_va,
        y_va,
        fold_id: int,
        seed: int,
        target_type: str = "quantile",
        horizon: int = 10,
        mfe_atr_tr=None,
        mae_atr_tr=None,
        side: str = "long",
    ):
        """
        Train a LightGBM model on a single fold.

        Parameters
        ----------
        X_tr : np.ndarray
            Training features
        y_tr : np.ndarray
            Training targets
        symbol_id_tr : np.ndarray
            Training symbol ids aligned with y_tr
        X_va : np.ndarray
            Validation features
        y_va : np.ndarray
            Validation targets
        fold_id : int
            Fold identifier
        seed : int
            Random seed
        target_type : str
            Target type (deprecated, always uses quantile regression)
        mfe_atr_tr : np.ndarray, optional
            Maximum favorable excursion in ATR units
        mae_atr_tr : np.ndarray, optional
            Maximum adverse excursion in ATR units
        side : str
            Trading side ("long" or "short")

        Returns
        -------
        Tuple[LGBMRegressor, Dict[str, Any]]
            Trained model and fit metadata
        """
        from lightgbm import early_stopping, log_evaluation

        # ENFORCE FINITE TARGETS
        tr_mask = np.isfinite(y_tr)
        va_mask = np.isfinite(y_va)
        X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
        symbol_id_tr = symbol_id_tr[tr_mask]
        surprisal_bits_tr = (
            None
            if surprisal_bits_tr is None
            else np.asarray(surprisal_bits_tr)[tr_mask]
        )
        # Also filter mfe/mae if provided
        if mfe_atr_tr is not None:
            mfe_atr_tr = np.asarray(mfe_atr_tr)[tr_mask]
        if mae_atr_tr is not None:
            mae_atr_tr = np.asarray(mae_atr_tr)[tr_mask]
        X_va, y_va = X_va[va_mask], y_va[va_mask]

        if len(y_tr) < 100:
            tprint(
                f"WARNING: Fold {fold_id} has very few training samples ({len(y_tr)})"
            )
        if len(y_va) == 0:
            raise ValueError(f"Fold {fold_id} has no finite validation samples")

        # Sample weights for regime mining
        sample_weight = build_miner_sample_weights(
            y_tr,
            X_tr,
            symbol_id_tr,
            self.cfg,
            horizon=horizon,
            surprisal_bits=surprisal_bits_tr,
            mfe_atr=mfe_atr_tr,
            mae_atr=mae_atr_tr,
            side=side,
        )
        if mfe_atr_tr is not None and mae_atr_tr is not None:
            excursion_weights = make_excursion_asymmetry_weights(
                mfe_atr=mfe_atr_tr,
                mae_atr=mae_atr_tr,
                side=side,
                alpha=0.15,
                w_min=0.85,
                w_max=1.20,
            )
            tprint(
                f"Excursion weights [{side}] fold {fold_id} "
                f"mean={np.nanmean(excursion_weights):.4f} "
                f"p5={np.nanpercentile(excursion_weights, 5):.4f} "
                f"p50={np.nanpercentile(excursion_weights, 50):.4f} "
                f"p95={np.nanpercentile(excursion_weights, 95):.4f}"
            )

        X_tr = np.asarray(X_tr, dtype=np.float32, order="C")
        X_va = np.asarray(X_va, dtype=np.float32, order="C")
        y_lo = float(np.nanquantile(y_tr, 0.01))
        y_hi = float(np.nanquantile(y_tr, 0.99))
        y_tr_reg = np.clip(y_tr, y_lo, y_hi).astype(np.float32, copy=False)
        y_va_reg = np.clip(y_va, y_lo, y_hi).astype(np.float32, copy=False)

        max_depth = int(self.cfg.get("lgbm_max_depth", 5)) + 1
        num_leaves = int(self.cfg.get("lgbm_num_leaves", 64))

        lambda_l1 = float(self.cfg.get("lambda_l1", 0.0))
        lambda_l2 = float(self.cfg.get("lambda_l2", 0.0))
        feature_fraction = float(self.cfg.get("feature_fraction", 0.7))

        min_gain_to_split = float(self.cfg.get("min_gain_to_split", 0.00005))
        if "hpo_min_gain_to_split" in self.cfg:
            min_gain_to_split = float(self.cfg["hpo_min_gain_to_split"])

        min_leaf_frac = float(self.cfg.get("lgbm_min_leaf_frac", 0.001))
        miner_min_leaf_floor_frac = float(
            self.cfg.get("miner_min_leaf_floor_frac", 0.05)
        )
        effective_min_leaf_frac = max(min_leaf_frac, miner_min_leaf_floor_frac)
        min_data_in_leaf = max(10, int(effective_min_leaf_frac * X_tr.shape[0]))
        if "hpo_min_data_in_leaf" in self.cfg:
            min_data_in_leaf = max(10, int(self.cfg["hpo_min_data_in_leaf"]))

        # Use quantile loss for all targets (triad targets work with quantile regression)
        alpha_hpo = float(self.cfg.get("alpha_hpo", 0.65))
        alpha = alpha_hpo if self.side == "long" else (1.0 - alpha_hpo)

        # Override with dynamic HPO results if available
        if "hpo_best_alpha" in self.cfg:
            alpha = float(self.cfg["hpo_best_alpha"])

        params = {
            "objective": "quantile",
            "alpha": alpha,
            "metric": "quantile",
            "boosting_type": "gbdt",
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "min_gain_to_split": min_gain_to_split,
            "learning_rate": float(self.cfg.get("learning_rate", 0.01)),
            "n_estimators": 1000,
            "verbosity": -1,
            "random_state": seed,
            "extra_trees": True,
            "n_jobs": max(1, min(3, int(self.cfg.get("lgbm_n_jobs", 3)))),
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "feature_fraction": feature_fraction,
            "min_sum_hessian_in_leaf": float(
                self.cfg.get("min_sum_hessian_in_leaf", 1e-4)
            ),
        }
        target_mode = "quantile_regression"

        if self.constraints:
            params["interaction_constraints"] = self.constraints

        model = LGBMRegressor(**params)
        evals_result = {}
        model.fit(
            X_tr,
            y_tr_reg,
            sample_weight=sample_weight,
            eval_set=[(X_va, y_va_reg)],
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(period=0),
                # Record evaluation results
                lambda env: evals_result.setdefault(
                    env.iteration, env.evaluation_result_list
                ),
            ],
        )

        # Get best metric
        best_iter = model.best_iteration_
        best_val_metric = np.nan
        if best_iter in evals_result:
            for dataset_name, metric_name, val, is_higher_better in evals_result[
                best_iter
            ]:
                if metric_name == params["metric"]:
                    best_val_metric = val
                    break

        feature_importances_gain = model.booster_.feature_importance(
            importance_type="gain"
        )
        feature_importances_split = model.booster_.feature_importance(
            importance_type="split"
        )
        feature_importances_gain_full = np.asarray(
            feature_importances_gain, dtype=np.float32
        )
        feature_importances_split_full = np.asarray(
            feature_importances_split, dtype=np.float32
        )

        # Metadata persistence
        fit_meta = {
            "model_id": "lgbm_discovery",
            "fold_id": fold_id,
            "seed": seed,
            "best_iteration": best_iter,
            "best_val_metric": best_val_metric,
            "train_samples": X_tr.shape[0],
            "val_samples": X_va.shape[0],
            "params_hash": hashlib.sha1(str(params).encode()).hexdigest()[:8],
            "classification": False,
            "threshold_tr": np.nan,
            "threshold_va": np.nan,
            "target_mode": target_mode,
            "target_type": target_type,
            "feature_importances_gain": feature_importances_gain_full,
            "feature_importances_split": feature_importances_split_full,
            "selected_feature_count": int(X_tr.shape[1]),
            "selected_feature_fraction": 1.0,
            "params": params,
        }

        return model, fit_meta


# =============================================================================
# LEAF EXTRACTION & RULE SCORING
# =============================================================================


def bounded_path_term(path_gain_sum: float, path_gain_cap: float = 50.0) -> float:
    """
    Map raw cumulative split gain to [0, 1].
    Uses log compression + clipping so a few very large paths cannot dominate.
    """
    if path_gain_cap <= 0:
        raise ValueError("path_gain_cap must be > 0")
    x = max(0.0, float(path_gain_sum))
    return float(np.clip(np.log1p(x) / np.log1p(path_gain_cap), 0.0, 1.0))


def bounded_leaf_term(leaf_value: float, leaf_scale: float = 0.25) -> float:
    """
    Map absolute leaf value to [0, 1) using tanh.
    leaf_scale controls how quickly the term saturates.
    Smaller leaf_scale => faster saturation.
    """
    if leaf_scale <= 0:
        raise ValueError("leaf_scale must be > 0")
    x = abs(float(leaf_value)) / leaf_scale
    return float(np.tanh(x))


def support_band_score(
    support_pct: float,
    support_min: float = 0.10,
    support_max: float = 0.20,
    preferred_support_min: float = 0.10,
    preferred_support_max: float = 0.15,
    target_support: float = 0.125,
    hard_zero_outside_band: bool = True,
) -> float:
    """
    Return a support score in [0, 1].

    - 1.0 inside preferred band
    - linearly decays toward edges of allowed band
    - optionally 0.0 outside allowed band
    """
    s = float(support_pct)

    if support_min <= preferred_support_min <= preferred_support_max <= support_max:
        pass
    else:
        raise ValueError("Support bounds are inconsistent")

    if s < support_min or s > support_max:
        return 0.0 if hard_zero_outside_band else 0.05

    if preferred_support_min <= s <= preferred_support_max:
        return 1.0

    # Soft decay toward target_support inside allowed band
    return float(1.0 - min(abs(s - target_support) / max(target_support, 1e-9), 1.0))


def compute_rule_model_importance_score(
    path_gain_sum: float,
    leaf_value: float,
    support_train: int,
    total_samples: int,
    *,
    path_gain_cap: float = 50.0,
    leaf_scale: float = 0.25,
    support_min: float = 0.10,
    support_max: float = 0.20,
    preferred_support_min: float = 0.10,
    preferred_support_max: float = 0.15,
    target_support: float = 0.125,
    hard_zero_outside_band: bool = True,
    leaf_weight: float = 0.15,
) -> float:
    """
    Final bounded, path-based rule importance score.

    Design:
      - path_term is the anchor, bounded in [0, 1]
      - leaf_term is a bounded modifier, not a dominant driver
      - support_score enforces your desired support band

    Score shape:
      score = path_term * (1 + leaf_weight * leaf_term) * support_score

    Range:
      approximately [0, 1 + leaf_weight], then suppressed by support_score
    """
    if total_samples <= 0:
        return 0.0

    support_pct = float(support_train) / float(total_samples)

    path_term = bounded_path_term(path_gain_sum, path_gain_cap=path_gain_cap)
    leaf_term = bounded_leaf_term(leaf_value, leaf_scale=leaf_scale)
    support_score = support_band_score(
        support_pct=support_pct,
        support_min=support_min,
        support_max=support_max,
        preferred_support_min=preferred_support_min,
        preferred_support_max=preferred_support_max,
        target_support=target_support,
        hard_zero_outside_band=hard_zero_outside_band,
    )

    score = path_term * (1.0 + leaf_weight * leaf_term) * support_score
    return float(score)


class RuleExtractor:
    def __init__(
        self,
        metadata: List[FeatureMetadata],
        cfg: Dict[str, Any],
        slot_order: Sequence[str] = ("trigger", "location", "regime"),
        positive_only_groups: Optional[Sequence[str]] = None,
        required_positive_groups: Optional[Sequence[str]] = None,
        collapse_duplicate_groups: Optional[Sequence[str]] = None,
    ):
        self.metadata_lookup = {m.feature_index: m for m in metadata}
        self.cfg = cfg
        self.slot_order = slot_order
        self.positive_only_groups = set(positive_only_groups or [])
        self.required_positive_groups = set(required_positive_groups or [])
        self.collapse_duplicate_groups = set(collapse_duplicate_groups or [])
        self.rejection_audit = []
        self.total_leaf_paths = 0
        self.total_non_empty_paths = 0

    def extract_rules(
        self,
        model: LGBMRegressor,
        model_id: str,
        fold_id: int,
        seed: int,
        target_name: str = "primary_target",
        horizon: int = 0,
    ) -> List[ExtractedRule]:
        """
        Extract rules from a trained LightGBM model.

        Parameters
        ----------
        model : LGBMRegressor
            Trained LightGBM model
        model_id : str
            Identifier for the model
        fold_id : int
            Fold identifier
        seed : int
            Random seed used for training
        target_name : str
            Name of the target for provenance tracking
        horizon : int
            Horizon in bars for provenance tracking (default: 0)

        Returns
        -------
        List[ExtractedRule]
            List of extracted rules with target/horizon provenance
        """
        # ALWAYS use native booster dump for correct semantics according to fix spec
        dump = model.booster_.dump_model()
        rules = []
        self.rejection_audit = []  # For diagnostics
        self.total_leaf_paths = 0
        self.total_non_empty_paths = 0

        # Store target/horizon for provenance
        self._current_target_name = target_name
        self._current_horizon = horizon

        best_iteration = int(getattr(model, "best_iteration_", 0) or 0)
        tree_info = dump["tree_info"]
        if best_iteration > 0:
            tree_info = tree_info[:best_iteration]

        # Configs for pre-extraction pruning
        min_leaf_support_for_extraction = float(
            self.cfg.get("min_leaf_support_for_extraction", self.cfg.get("support_min_pct", SUPPORT_MIN))
        )
        max_leaf_support_for_extraction = float(
            self.cfg.get("max_leaf_support_for_extraction", self.cfg.get("support_max_pct", SUPPORT_MAX))
        )
        preextract_topk = int(self.cfg.get("preextract_topk_by_abs_leaf_value", 1000))

        # Pre-compile trees to find all candidate leaves
        compiled_trees = []
        all_leaves = []
        for tree_idx, tree in enumerate(tree_info):
            compiled_tree = self._compile_tree(tree["tree_structure"])
            compiled_trees.append(compiled_tree)
            total_samples = int(max(compiled_tree.leaf_count[0], 1))

            for node_idx in range(len(compiled_tree.is_leaf)):
                if compiled_tree.is_leaf[node_idx]:
                    leaf_value = float(compiled_tree.leaf_value[node_idx])
                    leaf_count = int(compiled_tree.leaf_count[node_idx])
                    leaf_support = leaf_count / total_samples
                    all_leaves.append((tree_idx, node_idx, leaf_value, leaf_support))

        raw_leaf_count = len(all_leaves)

        # Stage 1: Support-range filter
        support_filtered_leaves = [
            lf for lf in all_leaves
            if min_leaf_support_for_extraction <= lf[3] <= max_leaf_support_for_extraction
        ]
        support_ok_count = len(support_filtered_leaves)

        # Stage 2: Top-K by abs(leaf_value) with round-robin per tree
        # To preserve temporal/structural diversity from the boosting process, we select the top leaves
        # iteratively round-robin from each tree, rather than a global sort.
        leaves_by_tree: Dict[int, List[Tuple[int, int, float, float]]] = collections.defaultdict(list)
        for lf in support_filtered_leaves:
            leaves_by_tree[lf[0]].append(lf)

        for t_idx in leaves_by_tree:
            # Sort each tree's leaves by abs(leaf_value) descending, then node_idx
            leaves_by_tree[t_idx].sort(key=lambda x: (abs(x[2]), x[1]), reverse=True)

        top_leaves = []
        tree_indices = sorted(list(leaves_by_tree.keys()))
        pos = 0
        while len(top_leaves) < preextract_topk and tree_indices:
            trees_to_remove = []
            for t_idx in tree_indices:
                if len(top_leaves) >= preextract_topk:
                    break
                if pos < len(leaves_by_tree[t_idx]):
                    top_leaves.append(leaves_by_tree[t_idx][pos])
                else:
                    trees_to_remove.append(t_idx)

            for t_idx in trees_to_remove:
                tree_indices.remove(t_idx)
            pos += 1

        top_abs_leaf_count = len(top_leaves)

        # Create allowed set of (tree_idx, node_idx)
        allowed_leaves = {(lf[0], lf[1]) for lf in top_leaves}

        # Determine side for logging
        if "short" in model_id.lower() or "short" in getattr(self, "side", "").lower():
            side_str = "short"
        elif "long" in model_id.lower() or "long" in getattr(self, "side", "").lower():
            side_str = "long"
        else:
            side_str = "unknown"

        tprint(
            f"Pre-extract leaf prune {target_name} @ H{horizon} [{side_str}]: "
            f"raw={raw_leaf_count} support_ok={support_ok_count} top_abs_leaf={top_abs_leaf_count}"
        )

        for tree_idx, compiled_tree in enumerate(compiled_trees):
            total_samples = int(max(compiled_tree.leaf_count[0], 1))
            self._traverse_compiled_tree(
                compiled_tree,
                tree_idx,
                model_id,
                fold_id,
                seed,
                rules,
                total_samples=total_samples,
                allowed_leaves=allowed_leaves,
            )

        reject_counts = collections.Counter(r["reason"] for r in self.rejection_audit)

        tprint(
            f"Extracted {len(rules)} valid paths from {self.total_leaf_paths} total paths ({self.total_non_empty_paths} non-empty)."
        )
        if reject_counts:
            tprint("Top rejection reasons:")
            for reason, count in reject_counts.most_common(5):
                tprint(f"  - {reason}: {count}")

        return rules

    def _normalize_boolean_threshold(self, threshold: float) -> float:
        thr = float(threshold)
        if abs(thr - 0.5) <= 1e-4:
            return 0.5
        if abs(thr) <= 1e-8 or abs(thr) <= 1e-30:
            return 0.0
        return thr

    def _normalize_predicate(
        self, metadata: FeatureMetadata, threshold: float, direction: int
    ) -> Optional[Tuple[int, str, float]]:
        if metadata.source_type == "boolean":
            threshold = self._normalize_boolean_threshold(threshold)
            if direction == 1:
                return (0, "<=", threshold)
            return (1, ">", threshold)
        raw_threshold = float(threshold)
        if direction == 1:
            return (0, "<=", raw_threshold)
        return (1, ">", raw_threshold)

    def _compile_tree(self, root_node: Dict[str, Any]) -> CompiledTree:
        split_feature: List[int] = []
        threshold: List[float] = []
        split_gain: List[float] = []
        left_child: List[int] = []
        right_child: List[int] = []
        leaf_value: List[float] = []
        leaf_count: List[int] = []
        leaf_index: List[int] = []
        is_leaf: List[bool] = []

        queue: collections.deque[Tuple[Dict[str, Any], int]] = collections.deque()
        queue.append((root_node, -1))

        while queue:
            node, parent_idx = queue.popleft()
            node_idx = len(is_leaf)
            node_is_leaf = "leaf_value" in node
            is_leaf.append(node_is_leaf)
            if node_is_leaf:
                split_feature.append(-1)
                threshold.append(np.nan)
                split_gain.append(0.0)
                left_child.append(-1)
                right_child.append(-1)
                leaf_value.append(float(node.get("leaf_value", 0.0)))
                leaf_count.append(int(node.get("leaf_count", 0)))
                leaf_index.append(int(node.get("leaf_index", -1)))
            else:
                split_feat_idx = int(node["split_feature"])
                split_feature.append(split_feat_idx)
                meta = self.metadata_lookup.get(split_feat_idx)
                raw_threshold = float(node.get("threshold", 0.5))
                if meta is not None and meta.source_type == "boolean":
                    raw_threshold = float(self._normalize_boolean_threshold(raw_threshold))
                threshold.append(raw_threshold)
                split_gain.append(float(node.get("split_gain", 0.0)))
                left_child.append(-1)
                right_child.append(-1)
                leaf_value.append(np.nan)
                leaf_count.append(int(node.get("internal_count", 0)))
                leaf_index.append(-1)

                left_idx = len(is_leaf) + len(queue)
                queue.append((node["left_child"], node_idx))
                right_idx = len(is_leaf) + len(queue)
                queue.append((node["right_child"], node_idx))
                left_child[node_idx] = left_idx
                right_child[node_idx] = right_idx

        return CompiledTree(
            split_feature=np.asarray(split_feature, dtype=np.int32),
            threshold=np.asarray(threshold, dtype=np.float32),
            split_gain=np.asarray(split_gain, dtype=np.float32),
            left_child=np.asarray(left_child, dtype=np.int32),
            right_child=np.asarray(right_child, dtype=np.int32),
            leaf_value=np.asarray(leaf_value, dtype=np.float32),
            leaf_count=np.asarray(leaf_count, dtype=np.int32),
            leaf_index=np.asarray(leaf_index, dtype=np.int32),
            is_leaf=np.asarray(is_leaf, dtype=bool),
        )

    def _traverse_compiled_tree(
        self,
        tree: CompiledTree,
        tree_idx: int,
        model_id: str,
        fold_id: int,
        seed: int,
        rules: List[ExtractedRule],
        total_samples: int = 1,
        allowed_leaves: Optional[Set[Tuple[int, int]]] = None,
    ) -> None:
        stack: List[Tuple[int, List[RuleCondition], float]] = [(0, [], 0.0)]
        while stack:
            node_idx, conditions, current_gain = stack.pop()

            if tree.is_leaf[node_idx]:
                if allowed_leaves is not None and (tree_idx, node_idx) not in allowed_leaves:
                    continue

                self.total_leaf_paths += 1
                if not conditions:
                    self.rejection_audit.append(
                        {
                            "model_id": model_id,
                            "fold_id": fold_id,
                            "seed": seed,
                            "tree_idx": tree_idx,
                            "leaf_idx": int(tree.leaf_index[node_idx]),
                            "reason": "empty_path",
                        }
                    )
                    continue

                self.total_non_empty_paths += 1
                reduced_conditions, reduce_reason = self._reduce_conditions(conditions)
                if reduce_reason is not None:
                    self.rejection_audit.append(
                        {
                            "model_id": model_id,
                            "fold_id": fold_id,
                            "seed": seed,
                            "tree_idx": tree_idx,
                            "leaf_idx": int(tree.leaf_index[node_idx]),
                            "reason": reduce_reason,
                        }
                    )
                    continue

                is_valid, reason = self._is_path_valid(reduced_conditions)
                if not is_valid:
                    self.rejection_audit.append(
                        {
                            "model_id": model_id,
                            "fold_id": fold_id,
                            "seed": seed,
                            "tree_idx": tree_idx,
                            "leaf_idx": int(tree.leaf_index[node_idx]),
                            "reason": reason,
                        }
                    )
                    continue

                leaf_idx = int(tree.leaf_index[node_idx])
                prov_str = (
                    f"{model_id}_{fold_id}_{seed}_{tree_idx}_{leaf_idx}_"
                    f"{len(reduced_conditions)}"
                )
                rule_id = hashlib.sha1(prov_str.encode()).hexdigest()[:12]
                leaf_value = float(tree.leaf_value[node_idx])
                support_train = int(tree.leaf_count[node_idx])
                rule_model_importance_score = compute_rule_model_importance_score(
                    path_gain_sum=current_gain,
                    leaf_value=leaf_value,
                    support_train=support_train,
                    total_samples=total_samples,
                    path_gain_cap=50.0,
                    leaf_scale=0.25,
                    support_min=float(self.cfg.get("support_min_pct", SUPPORT_MIN)),
                    support_max=float(self.cfg.get("support_max_pct", SUPPORT_MAX)),
                    preferred_support_min=float(
                        self.cfg.get(
                            "objective_support_target_low_pct",
                            PREFERRED_SUPPORT_MIN,
                        )
                    ),
                    preferred_support_max=float(
                        self.cfg.get(
                            "objective_support_target_high_pct",
                            PREFERRED_SUPPORT_MAX,
                        )
                    ),
                    target_support=float(
                        self.cfg.get("target_support", TARGET_SUPPORT)
                    ),
                    hard_zero_outside_band=True,
                    leaf_weight=0.15,
                )

                rules.append(
                    ExtractedRule(
                        rule_id=rule_id,
                        canonical_key="",
                        conditions=list(reduced_conditions),
                        model_id=model_id,
                        fold_id=fold_id,
                        seed=seed,
                        tree_index=tree_idx,
                        leaf_index=leaf_idx,
                        leaf_value=leaf_value,
                        support_train=support_train,
                        source_target=getattr(
                            self, "_current_target_name", "primary_target"
                        ),
                        source_horizon=getattr(self, "_current_horizon", 0),
                        path_gain_sum=current_gain,
                        total_samples=total_samples,
                        rule_model_importance_score=rule_model_importance_score,
                    )
                )
                continue

            split_feat_idx = int(tree.split_feature[node_idx])
            m = self.metadata_lookup.get(split_feat_idx)
            if not m:
                continue

            current_split_gain = float(tree.split_gain[node_idx])
            raw_thr = float(tree.threshold[node_idx])

            for direction in (0, 1):
                norm = self._normalize_predicate(m, raw_thr, direction)
                if norm is None:
                    continue
                norm_val, raw_op, raw_thr = norm
                cond = RuleCondition(
                    feature_name=m.feature_name,
                    feature_index=split_feat_idx,
                    group=m.group,
                    normalized_value=norm_val,
                    raw_operator=raw_op,
                    raw_threshold=raw_thr,
                    raw_decision_type=None,
                    default_left=None,
                    missing_type=None,
                )
                child_idx = (
                    int(tree.left_child[node_idx])
                    if direction == 1
                    else int(tree.right_child[node_idx])
                )
                next_conditions = list(conditions)
                next_conditions.append(cond)
                stack.append(
                    (
                        child_idx,
                        next_conditions,
                        current_gain + current_split_gain,
                    )
                )

    def _reduce_conditions(
        self, conditions: List[RuleCondition]
    ) -> Tuple[Optional[List[RuleCondition]], Optional[str]]:
        """
        De-duplicate repeated predicates on the exact same feature while preserving
        multiple same-group positives. This allows Stage A to keep richer location
        and regime conjunctions instead of rejecting them as group violations.
        """
        by_group: Dict[str, List[RuleCondition]] = collections.defaultdict(list)
        group_order: List[str] = []
        for c in conditions:
            if c.group not in by_group:
                group_order.append(c.group)
            by_group[c.group].append(c)

        reduced: List[RuleCondition] = []
        for group in group_order:
            group_conditions = by_group[group]
            bool_feat_map: Dict[int, int] = {}
            continuous_bounds: Dict[int, Dict[str, Any]] = {}
            if group not in self.collapse_duplicate_groups:
                for c in group_conditions:
                    metadata = self.metadata_lookup.get(c.feature_index)
                    if metadata is not None and metadata.source_type == "continuous":
                        bound = continuous_bounds.setdefault(
                            c.feature_index,
                            {
                                "feature_name": c.feature_name,
                                "group": c.group,
                                "lower": None,
                                "upper": None,
                            },
                        )
                        if c.raw_operator in {">", ">="}:
                            prev = bound["lower"]
                            if prev is None or float(c.raw_threshold) > float(prev.raw_threshold):
                                bound["lower"] = c
                        elif c.raw_operator in {"<=", "<"}:
                            prev = bound["upper"]
                            if prev is None or float(c.raw_threshold) < float(prev.raw_threshold):
                                bound["upper"] = c
                        elif c.raw_operator == "==":
                            bound["lower"] = c
                            bound["upper"] = c
                        continue
                    prev = bool_feat_map.get(c.feature_index)
                    if prev is not None:
                        if prev != c.normalized_value:
                            return None, f"contradiction_{c.feature_name}"
                        continue
                    bool_feat_map[c.feature_index] = c.normalized_value
                    reduced.append(c)
                for feature_index, bound in continuous_bounds.items():
                    lower = bound["lower"]
                    upper = bound["upper"]
                    if lower is not None and upper is not None:
                        if float(lower.raw_threshold) > float(upper.raw_threshold):
                            return None, f"contradiction_{bound['feature_name']}"
                    if lower is not None:
                        reduced.append(lower)
                    if upper is not None and upper is not lower:
                        reduced.append(upper)
                continue

            for c in group_conditions:
                metadata = self.metadata_lookup.get(c.feature_index)
                if metadata is not None and metadata.source_type == "continuous":
                    bound = continuous_bounds.setdefault(
                        c.feature_index,
                        {
                            "feature_name": c.feature_name,
                            "group": c.group,
                            "lower": None,
                            "upper": None,
                        },
                    )
                    if c.raw_operator in {">", ">="}:
                        prev = bound["lower"]
                        if prev is None or float(c.raw_threshold) > float(prev.raw_threshold):
                            bound["lower"] = c
                    elif c.raw_operator in {"<=", "<"}:
                        prev = bound["upper"]
                        if prev is None or float(c.raw_threshold) < float(prev.raw_threshold):
                            bound["upper"] = c
                    elif c.raw_operator == "==":
                        bound["lower"] = c
                        bound["upper"] = c
                    continue
                if c.feature_index not in bool_feat_map:
                    bool_feat_map[c.feature_index] = c.normalized_value
                    reduced.append(c)
                elif bool_feat_map[c.feature_index] != c.normalized_value:
                    return None, f"contradiction_in_collapsed_group_{c.feature_name}"
            for feature_index, bound in continuous_bounds.items():
                lower = bound["lower"]
                upper = bound["upper"]
                if lower is not None and upper is not None:
                    if float(lower.raw_threshold) > float(upper.raw_threshold):
                        return None, f"contradiction_in_collapsed_group_{bound['feature_name']}"
                if lower is not None:
                    reduced.append(lower)
                if upper is not None and upper is not lower:
                    reduced.append(upper)

        return reduced, None

    def _is_path_valid(self, conditions: List[RuleCondition]) -> Tuple[bool, str]:
        """
        Hardened validation: Group limits, contradictions, and polarity.
        """
        if not conditions:
            return False, "empty_path"

        seen_groups = {}
        seen_features = {}

        for c in conditions:
            m = self.metadata_lookup.get(c.feature_index)
            if m is None:
                continue

            # Interaction group constraints removed per user request.
            # ig = m.interaction_group
            # prev_feat = seen_groups.get(ig)
            # if prev_feat is not None and prev_feat != c.feature_index:
            #     return False, f"interaction_group_violation_{ig}"
            # seen_groups[ig] = c.feature_index

            if m.source_type != "continuous":
                prev_val = seen_features.get(c.feature_index)
                if prev_val is not None and prev_val != c.normalized_value:
                    return False, f"contradiction_{c.feature_name}"
                seen_features[c.feature_index] = c.normalized_value

        # Polarity Check: reject only all-negative paths
        # A path is all-negative if NO condition has normalized_value == 1
        if not any(c.normalized_value == 1 for c in conditions):
            return False, "all_negative_path"

        for c in conditions:
            if c.group in self.positive_only_groups and c.normalized_value != 1:
                return False, f"negative_not_allowed_{c.group}"

        positive_groups = {c.group for c in conditions if c.normalized_value == 1}
        missing_required = sorted(self.required_positive_groups - positive_groups)
        if missing_required:
            return False, f"missing_required_group_{missing_required[0]}"

        return True, "valid"

    def _build_canonical_key(self, conditions: List[RuleCondition]) -> Optional[str]:
        """
        Deterministic slot-based key using slot_order.
        """
        slots = collections.defaultdict(list)
        for c in conditions:
            if c.group in self.slot_order:
                slots[c.group].append(c)

        out_slots = []
        for s in self.slot_order:
            group_conds = slots.get(s, [])
            if not group_conds:
                out_slots.append("(*)")
            else:
                # Sort by feature name for canonical ordering
                group_conds.sort(
                    key=lambda x: (x.feature_name, x.raw_operator, float(x.raw_threshold))
                )
                seen = set()
                joined = []
                for c in group_conds:
                    metadata = self.metadata_lookup.get(c.feature_index)
                    if metadata is not None and metadata.source_type == "continuous":
                        rep = (
                            f"{c.feature_name}{c.raw_operator}"
                            f"{format_condition_value(c.raw_threshold)}"
                        )
                    else:
                        rep = f"{c.feature_name}=={int(c.normalized_value)}"
                    if rep not in seen:
                        joined.append(rep)
                        seen.add(rep)
                out_slots.append(f"({'&'.join(joined)})")

        return "|".join(out_slots)


COMPOSITE_RULE_PATTERN = re.compile(r"^Composite\((.+)\)_OR_\((.+)\)$")


def split_composite_key(canonical_key: str) -> Optional[Tuple[str, str]]:
    match = COMPOSITE_RULE_PATTERN.match(canonical_key)
    if not match:
        return None
    return match.group(1), match.group(2)


CONDITION_OPERATOR_PATTERN = re.compile(r"^(?P<name>.+?)(?P<op><=|>=|==|<|>)(?P<value>.+)$")


def format_condition_value(value: float) -> str:
    return np.format_float_positional(
        float(value), precision=8, unique=True, fractional=False, trim="-"
    )


def parse_condition_string(cond_str: str) -> Optional[Tuple[str, str, str]]:
    match = CONDITION_OPERATOR_PATTERN.match(cond_str.strip())
    if match is None:
        return None
    return match.group("name"), match.group("op"), match.group("value")


def condition_operator_code(operator: str) -> int:
    mapping = {"==": 0, "<=": 1, ">": 2, "<": 3, ">=": 4}
    if operator not in mapping:
        raise ValueError(f"Unsupported condition operator: {operator}")
    return mapping[operator]


def parse_slot_map(
    canonical_key: str,
    slot_order: Sequence[str] = ("trigger", "location", "regime"),
) -> Dict[str, str]:
    parts = split_composite_key(canonical_key)
    if parts is not None:
        # For composite keys, return a dummy map since they don't map to a single triad slot
        return {group: "Composite" for group in slot_order}
    slots = canonical_key.split("|")
    if len(slots) != len(slot_order):
        raise ValueError(
            f"Key {canonical_key} has {len(slots)} slots but expected {len(slot_order)}"
        )
    return {group: slot.strip("()") for group, slot in zip(slot_order, slots)}


def build_stage_a_parent_key_from_slot_map(slot_map: Dict[str, str]) -> Optional[str]:
    loc = slot_map.get("location", "*")
    reg = slot_map.get("regime", "*")
    if loc == "*" and reg == "*":
        return None
    return f"(*)|({loc})|({reg})"


def iter_primitive_keys(canonical_key: str, depth: int = 0) -> List[str]:
    # Point 13: Add recursion depth limit
    if depth > 5:
        tprint(f"WARNING: Max recursion depth reached for key {canonical_key}")
        return [canonical_key]
    composite_parts = split_composite_key(canonical_key)
    if composite_parts is None:
        return [canonical_key]
    out: List[str] = []
    for part in composite_parts:
        out.extend(iter_primitive_keys(part, depth + 1))
    return out


def extract_feature_names_from_key(canonical_key: str) -> List[str]:
    names: List[str] = []
    for part in iter_primitive_keys(canonical_key):
        for slot in part.split("|"):
            slot_value = slot.strip("()")
            if slot_value == "*":
                continue
            for cond_str in slot_value.split("&"):
                parsed = parse_condition_string(cond_str)
                if parsed is None:
                    continue
                names.append(parsed[0])
    return sorted(set(names))


def infer_rule_side(
    canonical_key: str,
    mean_net_ret: Optional[float] = None,
    explicit_side: Optional[str] = None,
) -> str:
    if explicit_side:
        return explicit_side
    names = [name.lower() for name in extract_feature_names_from_key(canonical_key)]
    has_long = any(token in name for name in names for token in ("long", "bull", "up"))
    has_short = any(
        token in name for name in names for token in ("short", "bear", "down")
    )
    if has_long and has_short:
        return "mixed"
    if has_long:
        return "long"
    if has_short:
        return "short"
    if mean_net_ret is not None and np.isfinite(mean_net_ret):
        if mean_net_ret > 0:
            return "long"
        if mean_net_ret < 0:
            return "short"
    return "unknown"


def display_arity_for_key(canonical_key: str) -> int:
    composite_parts = split_composite_key(canonical_key)
    if composite_parts is not None:
        return max(display_arity_for_key(part) for part in composite_parts)

    total = 0
    for slot in canonical_key.split("|"):
        slot_value = slot.strip("()")
        if slot_value == "*":
            continue
        total += sum(
            1 for cond_str in slot_value.split("&") if parse_condition_string(cond_str)
        )
    return total


def structural_depth_for_key(canonical_key: str) -> int:
    composite_parts = split_composite_key(canonical_key)
    if composite_parts is not None:
        return sum(structural_depth_for_key(part) for part in composite_parts)
    return display_arity_for_key(canonical_key)


def build_walk_forward_folds(
    n_samples: int,
    n_folds: int,
    min_train_frac: float = 0.5,
    embargo: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_samples <= 1:
        return []
    min_train = max(1, int(np.floor(n_samples * min_train_frac)))
    min_train = min(min_train, n_samples - 1)
    remaining = n_samples - min_train
    if remaining <= 0:
        return []
    n_val_folds = min(max(1, n_folds), remaining)
    base_size = remaining // n_val_folds
    remainder = remaining % n_val_folds
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    va_start = min_train
    for fold_id in range(n_val_folds):
        fold_size = base_size + (1 if fold_id < remainder else 0)
        va_end = min(n_samples, va_start + fold_size)
        tr_end = max(0, va_start - embargo)
        # Point 10: Use int64 for large dataset indexing to prevent overflow
        tr_idx = np.arange(0, tr_end, dtype=np.int64)
        va_idx = np.arange(va_start, va_end, dtype=np.int64)
        if tr_idx.size == 0 or va_idx.size == 0:
            va_start = va_end
            continue
        if tr_idx.max() >= va_idx.min():
            raise ValueError(
                f"Invalid walk-forward fold {fold_id}: train leaks into validation"
            )
        folds.append((tr_idx, va_idx))
        va_start = va_end
    return folds


def _cap_fold_indices(indices: np.ndarray, max_rows: int) -> np.ndarray:
    """Deterministically downsample a fold while preserving temporal order."""
    if max_rows <= 0 or indices.size <= max_rows:
        return indices
    sample_pos = np.linspace(0, indices.size - 1, num=max_rows)
    sample_pos = np.asarray(sample_pos, dtype=np.int32)
    return indices[sample_pos]


class DictionaryMaskResolver:
    def __init__(
        self,
        mask_map: Dict[str, np.ndarray],
        parent_context_map: Optional[Dict[str, str]] = None,
        side_map: Optional[Dict[str, str]] = None,
    ):
        self.mask_map = {
            key: np.asarray(mask, dtype=bool) for key, mask in mask_map.items()
        }
        self.parent_context_map = parent_context_map or {}
        self.side_map = side_map or {}

    def register_mask(
        self,
        canonical_key: str,
        mask: np.ndarray,
        parent_context_key: Optional[str] = None,
        side: Optional[str] = None,
    ) -> None:
        self.mask_map[canonical_key] = np.asarray(mask, dtype=bool)
        if parent_context_key:
            self.parent_context_map[canonical_key] = parent_context_key
        if side:
            self.side_map[canonical_key] = side

    def get_mask(
        self, canonical_key: str, indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        composite_parts = split_composite_key(canonical_key)
        if composite_parts is not None:
            left = self.get_mask(composite_parts[0], indices)
            right = self.get_mask(composite_parts[1], indices)
            return left | right
        if canonical_key not in self.mask_map:
            raise KeyError(f"Cannot resolve mask for {canonical_key}")
        mask = self.mask_map[canonical_key]
        if indices is None:
            return mask.copy()
        return mask[indices]

    def get_masks_matrix(
        self, keys: List[str], indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.vstack([self.get_mask(key, indices=indices) for key in keys]).astype(
            bool, copy=False
        )

    def get_parent_context_key(self, canonical_key: str) -> Optional[str]:
        composite_parts = split_composite_key(canonical_key)
        if composite_parts is not None:
            left = self.get_parent_context_key(composite_parts[0])
            right = self.get_parent_context_key(composite_parts[1])
            return left if left == right else None
        return self.parent_context_map.get(canonical_key)

    def get_rule_side(self, canonical_key: str) -> Optional[str]:
        composite_parts = split_composite_key(canonical_key)
        if composite_parts is not None:
            left = self.get_rule_side(composite_parts[0])
            right = self.get_rule_side(composite_parts[1])
            return left if left == right else "mixed"
        return self.side_map.get(canonical_key)


class CanonicalRuleMaskResolver:
    """
    Resolves canonical rule strings into boolean masks across a dataset.
    """

    def __init__(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
        context_lookup: Optional[Dict[str, np.ndarray]] = None,
        context_key_map: Optional[Dict[str, str]] = None,
        slot_order: Sequence[str] = ("trigger", "location", "regime"),
    ):
        self.X = X
        self.metadata = metadata
        tprint("CanonicalRuleMaskResolver: processing context_lookup...")
        self.context_lookup = {
            key: np.asarray(val, dtype=bool)
            for key, val in (context_lookup or {}).items()
        }
        self.context_key_map = context_key_map or {}
        self.slot_order = tuple(slot_order)
        tprint("CanonicalRuleMaskResolver: processing name_to_idx...")
        self.name_to_idx = {m.feature_name: m.feature_index for m in metadata}
        tprint("CanonicalRuleMaskResolver: processing context_name_to_idx...")
        self.context_name_to_idx = {
            key: idx for idx, key in enumerate(self.context_lookup.keys())
        }
        self.parent_key_to_context_name = {
            parent_key: ctx_name
            for ctx_name, parent_key in self.context_key_map.items()
        }
        self._parsed_rule_cache: Dict[str, Dict[str, Any]] = {}
        # Point 15: Instance-level state instead of module globals
        self.malformed_key_count = 0
        self.unresolved_feature_count = 0
        self.unresolved_feature_names: Set[str] = set()
        tprint("CanonicalRuleMaskResolver: init complete.")

    def _slice_mask(
        self, mask: np.ndarray, indices: Optional[np.ndarray]
    ) -> np.ndarray:
        if indices is None:
            return mask.copy()
        return mask[indices]

    def _resolve_feature_mask(
        self, feature_name: str, target_val: int, indices: Optional[np.ndarray]
    ) -> np.ndarray:
        if feature_name in self.name_to_idx:
            values = (
                self.X[:, self.name_to_idx[feature_name]]
                if indices is None
                else self.X[indices, self.name_to_idx[feature_name]]
            )
            return values == target_val
        if feature_name in self.context_lookup:
            base_mask = self._slice_mask(self.context_lookup[feature_name], indices)
            return base_mask if target_val == 1 else ~base_mask
        raise KeyError(f"Unknown feature {feature_name} in canonical key")

    def _resolve_context_parent_mask(
        self, canonical_key: str, indices: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        try:
            slot_map = parse_slot_map(canonical_key, self.slot_order)
        except ValueError:
            slot_map = parse_slot_map(canonical_key, ("trigger", "location", "regime"))
        parent_key = build_stage_a_parent_key_from_slot_map(slot_map)
        if parent_key is None:
            return None
        ctx_name = self.parent_key_to_context_name.get(parent_key)
        if ctx_name is None:
            return None
        if not ctx_name.startswith("ctx__"):
            raise ValueError(
                f"Unexpected unresolved feature in canonical key: {ctx_name}"
            )
        return self._slice_mask(self.context_lookup[ctx_name], indices)

    def _parse_rule_spec(self, canonical_key: str) -> Dict[str, Any]:
        cached = self._parsed_rule_cache.get(canonical_key)
        if cached is not None:
            return cached

        composite_parts = split_composite_key(canonical_key)
        if composite_parts is not None:
            spec = {
                "is_composite": True,
                "left": composite_parts[0],
                "right": composite_parts[1],
            }
            self._parsed_rule_cache[canonical_key] = spec
            return spec

        try:
            slot_map = parse_slot_map(canonical_key, self.slot_order)
        except ValueError:
            slot_map = parse_slot_map(canonical_key, ("trigger", "location", "regime"))

        feature_indices: List[int] = []
        feature_operator_codes: List[int] = []
        feature_threshold_values: List[float] = []
        context_feature_names: List[str] = []
        context_target_values: List[int] = []
        unresolved: List[Tuple[str, str]] = []

        for group, slot_value in slot_map.items():
            if slot_value == "*":
                continue
            for cond_str in slot_value.split("&"):
                parsed = parse_condition_string(cond_str)
                if parsed is None:
                    self.malformed_key_count += 1
                    raise ValueError(f"Malformed slot {cond_str} in {canonical_key}")
                feature_name, operator, raw_value = parsed
                if feature_name in self.name_to_idx:
                    feature_idx = self.name_to_idx[feature_name]
                    feature_indices.append(feature_idx)
                    feature_operator_codes.append(condition_operator_code(operator))
                    feature_threshold_values.append(float(raw_value))
                elif feature_name in self.context_lookup:
                    if operator != "==":
                        raise ValueError(
                            f"Context feature {feature_name} only supports == predicates"
                        )
                    context_feature_names.append(feature_name)
                    context_target_values.append(int(float(raw_value)))
                else:
                    unresolved.append((group, feature_name))
                    self.unresolved_feature_count += 1
                    self.unresolved_feature_names.add(feature_name)

        parent_context_name: Optional[str] = None
        if unresolved:
            unresolved_groups = {g for g, _ in unresolved}
            unresolved_features = [f for _, f in unresolved]
            if not unresolved_groups.issubset({"location", "regime"}):
                raise KeyError(
                    f"Cannot resolve groups {unresolved_groups} for key {canonical_key}"
                )
            parent_key = build_stage_a_parent_key_from_slot_map(slot_map)
            if parent_key is not None:
                parent_context_name = self.parent_key_to_context_name.get(parent_key)
            allow_context_fallback = all(
                f.startswith("ctx__") for f in unresolved_features
            )
            if parent_context_name is None and not allow_context_fallback:
                raise KeyError(
                    f"Unresolved features {unresolved_features} in key {canonical_key}"
                )
            if allow_context_fallback and parent_context_name is None:
                raise KeyError(f"Cannot map {canonical_key} to a saved Stage A context")

        spec = {
            "is_composite": False,
            "feature_indices": np.asarray(feature_indices, dtype=np.int32),
            "feature_operator_codes": np.asarray(feature_operator_codes, dtype=np.int8),
            "feature_threshold_values": np.asarray(
                feature_threshold_values, dtype=np.float32
            ),
            "context_feature_names": tuple(context_feature_names),
            "context_target_values": np.asarray(context_target_values, dtype=np.int8),
            "parent_context_name": parent_context_name,
        }

        instr_source_type: List[int] = []
        instr_source_idx: List[int] = []
        instr_operator_code: List[int] = []
        instr_threshold_value: List[float] = []
        for idx, operator_code, threshold_value in zip(
            feature_indices, feature_operator_codes, feature_threshold_values
        ):
            instr_source_type.append(0)
            instr_source_idx.append(int(idx))
            instr_operator_code.append(int(operator_code))
            instr_threshold_value.append(float(threshold_value))
        for feature_name, target_val in zip(
            context_feature_names, context_target_values
        ):
            instr_source_type.append(1)
            instr_source_idx.append(int(self.context_name_to_idx[feature_name]))
            instr_operator_code.append(0)
            instr_threshold_value.append(float(target_val))
        if parent_context_name is not None:
            instr_source_type.append(1)
            instr_source_idx.append(int(self.context_name_to_idx[parent_context_name]))
            instr_operator_code.append(0)
            instr_threshold_value.append(1.0)
        spec["instr_source_type"] = np.asarray(instr_source_type, dtype=np.int8)
        spec["instr_source_idx"] = np.asarray(instr_source_idx, dtype=np.int32)
        spec["instr_operator_code"] = np.asarray(instr_operator_code, dtype=np.int8)
        spec["instr_threshold_value"] = np.asarray(
            instr_threshold_value, dtype=np.float32
        )

        self._parsed_rule_cache[canonical_key] = spec
        return spec

    def get_mask(
        self, canonical_key: str, indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        spec = self._parse_rule_spec(canonical_key)
        if spec.get("is_composite", False):
            return self.get_mask(spec["left"], indices) | self.get_mask(
                spec["right"], indices
            )

        n_samples = self.X.shape[0] if indices is None else len(indices)
        mask = np.ones(n_samples, dtype=bool)

        feature_indices = spec["feature_indices"]
        feature_operator_codes = spec["feature_operator_codes"]
        feature_threshold_values = spec["feature_threshold_values"]
        for idx, operator_code, threshold_value in zip(
            feature_indices, feature_operator_codes, feature_threshold_values
        ):
            values = self.X[:, idx] if indices is None else self.X[indices, idx]
            if int(operator_code) == 0:
                mask &= values == threshold_value
            elif int(operator_code) == 1:
                mask &= values <= threshold_value
            elif int(operator_code) == 2:
                mask &= values > threshold_value
            elif int(operator_code) == 3:
                mask &= values < threshold_value
            else:
                mask &= values >= threshold_value

        for feature_name, target_val in zip(
            spec["context_feature_names"], spec["context_target_values"]
        ):
            base_mask = self._slice_mask(self.context_lookup[feature_name], indices)
            mask &= base_mask if int(target_val) == 1 else ~base_mask

        parent_context_name = spec.get("parent_context_name")
        if parent_context_name is not None:
            context_mask = self._slice_mask(
                self.context_lookup[parent_context_name], indices
            )
            mask &= context_mask

        return mask

    def get_masks_matrix(
        self, keys: List[str], indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        n_rules = len(keys)
        n_samples = self.X.shape[0] if indices is None else len(indices)
        if n_rules == 0:
            return np.empty((0, n_samples), dtype=bool)

        x_values = self.X if indices is None else self.X[indices]
        x_values = np.asarray(x_values, dtype=np.float32, order="C")

        context_names = list(self.context_lookup.keys())
        if context_names:
            context_views = []
            for name in context_names:
                base = self.context_lookup[name]
                context_views.append(base if indices is None else base[indices])
            context_values = np.vstack(context_views).astype(np.int8, copy=False)
        else:
            context_values = np.empty((0, n_samples), dtype=np.int8)

        parsed_specs = [self._parse_rule_spec(key) for key in keys]
        non_composite_positions: List[int] = []
        composite_positions: List[int] = []
        instr_source_type_chunks: List[np.ndarray] = []
        instr_source_idx_chunks: List[np.ndarray] = []
        instr_operator_code_chunks: List[np.ndarray] = []
        instr_threshold_value_chunks: List[np.ndarray] = []
        rule_offsets: List[int] = []
        rule_lengths: List[int] = []
        offset = 0

        for pos, spec in enumerate(parsed_specs):
            if spec.get("is_composite", False):
                composite_positions.append(pos)
                continue
            non_composite_positions.append(pos)
            source_type = spec["instr_source_type"]
            source_idx = spec["instr_source_idx"]
            operator_code = spec["instr_operator_code"]
            threshold_value = spec["instr_threshold_value"]
            instr_source_type_chunks.append(source_type)
            instr_source_idx_chunks.append(source_idx)
            instr_operator_code_chunks.append(operator_code)
            instr_threshold_value_chunks.append(threshold_value)
            rule_offsets.append(offset)
            rule_lengths.append(int(len(source_type)))
            offset += int(len(source_type))

        mask_matrix = np.ones((n_rules, n_samples), dtype=bool)
        if non_composite_positions:
            if offset > 0:
                instr_source_type = np.concatenate(instr_source_type_chunks).astype(
                    np.int8, copy=False
                )
                instr_source_idx = np.concatenate(instr_source_idx_chunks).astype(
                    np.int32, copy=False
                )
                instr_operator_code = np.concatenate(instr_operator_code_chunks).astype(
                    np.int8, copy=False
                )
                instr_threshold_value = np.concatenate(
                    instr_threshold_value_chunks
                ).astype(np.float32, copy=False)
            else:
                instr_source_type = np.empty(0, dtype=np.int8)
                instr_source_idx = np.empty(0, dtype=np.int32)
                instr_operator_code = np.empty(0, dtype=np.int8)
                instr_threshold_value = np.empty(0, dtype=np.float32)

            non_composite_masks = _compute_masks_from_instruction_matrix_numba(
                x_values,
                context_values,
                instr_source_type,
                instr_source_idx,
                instr_operator_code,
                instr_threshold_value,
                np.asarray(rule_offsets, dtype=np.int32),
                np.asarray(rule_lengths, dtype=np.int32),
            )
            for row_idx, pos in enumerate(non_composite_positions):
                mask_matrix[pos] = non_composite_masks[row_idx]

        for pos in composite_positions:
            mask_matrix[pos] = self.get_mask(keys[pos], indices=indices)

        return mask_matrix

    def get_parent_context_key(self, canonical_key: str) -> Optional[str]:
        composite_parts = split_composite_key(canonical_key)
        if composite_parts is not None:
            left = self.get_parent_context_key(composite_parts[0])
            right = self.get_parent_context_key(composite_parts[1])
            return left if left == right else None

        try:
            slot_map = parse_slot_map(canonical_key, self.slot_order)
        except ValueError:
            slot_map = parse_slot_map(canonical_key, ("trigger", "location", "regime"))

        if "context" in slot_map and slot_map["context"] != "*":
            parsed_context = parse_condition_string(slot_map["context"])
            ctx_name = (
                parsed_context[0]
                if parsed_context is not None
                else slot_map["context"].split("==")[0]
            )
            return self.context_key_map.get(ctx_name)

        parent_key = build_stage_a_parent_key_from_slot_map(slot_map)
        if parent_key in self.parent_key_to_context_name:
            return parent_key
        return None

    def get_rule_side(self, canonical_key: str) -> Optional[str]:
        return infer_rule_side(canonical_key)


class RuleScorer:
    def __init__(
        self,
        metadata: List[FeatureMetadata],
        cfg: Dict[str, Any],
        mask_resolver: Optional[
            Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]
        ] = None,
    ):
        self.metadata = metadata
        self.cfg = cfg
        self.mask_resolver = mask_resolver
        self._path_quality_cache: Dict[str, Dict[str, Any]] = {}

    def _compute_required_hurdle(self, support_pct: float, display_arity: int) -> float:
        base_hurdle = float(self.cfg.get("prune_base_hurdle", 0.0002))
        target_support = float(self.cfg.get("prune_target_support_pct", TARGET_SUPPORT))
        complexity_bonus = float(
            self.cfg.get("prune_complexity_bonus_map", {}).get(str(display_arity), 0.0)
        )
        safe_support = max(float(support_pct), 0.0005)

        # Asymmetric U-shaped penalty favoring support around prune_target_support_pct (e.g., 10-15%)
        dist = safe_support - target_support
        # Punish lower support more heavily than higher support (asymmetry)
        penalty_multiplier = 1.0 + (
            10.0 * (dist**2) if dist < 0 else 5.0 * (dist**2)
        )

        return (base_hurdle * (1.0 - complexity_bonus)) * penalty_multiplier

    def _compute_support_objective_score(self, support_pct: float) -> float:
        """Return a bounded support-fit score for the HPO objective.

        The objective is intentionally flat across the preferred 7.5%-12.5% band,
        and anything outside the hard 5%-15% band is excluded entirely.
        """

        hard_min = float(
            self.cfg.get(
                "objective_support_min_pct",
                self.cfg.get("support_min_pct", SUPPORT_MIN),
            )
        )
        target_low = float(
            self.cfg.get("objective_support_target_low_pct", PREFERRED_SUPPORT_MIN)
        )
        target_high = float(
            self.cfg.get("objective_support_target_high_pct", PREFERRED_SUPPORT_MAX)
        )
        hard_max = float(
            self.cfg.get(
                "objective_support_max_pct",
                self.cfg.get("support_max_pct", SUPPORT_MAX),
            )
        )
        edge_floor = float(self.cfg.get("objective_support_edge_floor", 0.2))

        if (
            not np.isfinite(support_pct)
            or support_pct < hard_min
            or support_pct > hard_max
        ):
            return -np.inf

        if target_low <= support_pct <= target_high:
            return 1.0

        if support_pct < target_low:
            span = max(target_low - hard_min, 1e-9)
            relative = float(np.clip((support_pct - hard_min) / span, 0.0, 1.0))
        else:
            span = max(hard_max - target_high, 1e-9)
            relative = float(np.clip((hard_max - support_pct) / span, 0.0, 1.0))

        return float(edge_floor + (1.0 - edge_floor) * relative)

    def _compute_within_mask_ic(
        self,
        predictions: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        method: str = "spearman",
    ) -> Tuple[float, float]:
        """
        Compute IC within mask and delta from global IC.

        Args:
            predictions: Model predictions array
            target: Target values array
            mask: Boolean mask indicating regime activation
            method: Correlation method ("spearman" or "pearson")

        Returns:
            Tuple of (within_mask_ic, delta_ic) where delta_ic = within_mask_ic - global_ic
        """
        # Global IC
        valid = ~(np.isnan(predictions) | np.isnan(target))
        if valid.sum() < 10:
            return np.nan, np.nan

        if method == "spearman":
            global_ic = _safe_spearman(predictions[valid], target[valid])
        else:
            # Pearson correlation
            valid_preds = predictions[valid]
            valid_targets = target[valid]
            if np.all(valid_preds == valid_preds[0]) or np.all(
                valid_targets == valid_targets[0]
            ):
                global_ic = np.nan
            else:
                global_ic = float(np.corrcoef(valid_preds, valid_targets)[0, 1])

        # Within-mask IC
        mask_active = mask.astype(bool) & valid
        if mask_active.sum() < 10:
            return np.nan, np.nan

        if method == "spearman":
            within_ic = _safe_spearman(predictions[mask_active], target[mask_active])
        else:
            masked_preds = predictions[mask_active]
            masked_targets = target[mask_active]
            if np.all(masked_preds == masked_preds[0]) or np.all(
                masked_targets == masked_targets[0]
            ):
                within_ic = np.nan
            else:
                within_ic = float(np.corrcoef(masked_preds, masked_targets)[0, 1])

        delta_ic = within_ic - global_ic if not np.isnan(within_ic) else np.nan

        return within_ic, delta_ic

    def _compute_entropy_reduction(
        self,
        target: np.ndarray,
        mask: np.ndarray,
        n_bins: int = 20,
    ) -> float:
        """
        Compute reduction in target uncertainty conditional on mask.

        Uses variance proxy for entropy estimation.
        Higher is better - means the regime reduces target uncertainty.

        Args:
            target: Target values array
            mask: Boolean mask indicating regime activation
            n_bins: Number of bins for histogram estimation (ignored)

        Returns:
            Entropy reduction proxy (global entropy proxy - conditional entropy proxy).
            Positive means the mask concentrates the target distribution.
        """
        valid = ~np.isnan(target)
        target_valid = target[valid]

        if len(target_valid) < 100:
            return np.nan

        # Conditional entropy (within mask)
        mask_active = mask.astype(bool) & valid
        if mask_active.sum() < 50:
            return np.nan

        target_masked = target[mask_active]

        entropy_global = np.log(np.std(target_valid) + 1e-9)
        entropy_masked = np.log(np.std(target_masked) + 1e-9)

        # Reduction = global - conditional (positive is good)
        return float(entropy_global - entropy_masked)

    def score_key_oos(
        self,
        canonical_key: str,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        resolver: Optional[
            Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]
        ] = None,
        require_uplift: bool = False,
        parent_context_key: Optional[str] = None,
        discovery_count: int = 0,
        n_instances: Optional[int] = None,
        pipeline_stage: Optional[str] = None,
        explicit_side: Optional[str] = None,
        bounded_target: Optional[np.ndarray] = None,
        target_name: str = "primary_target",
        horizon: int = 0,
        predictions: Optional[np.ndarray] = None,
        path_mfe: Optional[np.ndarray] = None,
        path_mae: Optional[np.ndarray] = None,
        path_final_ret: Optional[np.ndarray] = None,
        path_time_to_mfe: Optional[np.ndarray] = None,
        path_time_to_mae: Optional[np.ndarray] = None,
        path_length: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Score rules with optional bounded target support.

        Args:
            canonical_key: Rule identifier
            fwd_ret: Forward returns array
            folds: List of (train_idx, val_idx) tuples for cross-validation
            resolver: Mask resolver for computing rule masks
            require_uplift: Whether to require uplift over parent context
            parent_context_key: Parent context key for uplift computation
            discovery_count: Number of times this rule was discovered
            n_instances: Total number of instances
            pipeline_stage: Pipeline stage identifier
            explicit_side: Explicit side (long/short) override
            bounded_target: Optional bounded target for within-mask IC computation
            target_name: Name of the target for logging
            horizon: Horizon identifier for logging
            predictions: Optional model predictions for within-mask IC computation

        Returns:
            Tuple of (summary_dict, fold_records_list)
        """
        resolver = resolver or self.mask_resolver
        if resolver is None:
            raise ValueError("RuleScorer requires a mask resolver")

        fold_records: List[Dict[str, Any]] = []
        epsilon = float(self.cfg.get("sign_dead_zone", 1e-6))

        if require_uplift and not parent_context_key:
            parent_context_key = resolver.get_parent_context_key(canonical_key)

        # Track within-mask IC metrics across folds
        within_mask_ic_values: List[float] = []
        delta_within_mask_ic_values: List[float] = []
        entropy_reduction_values: List[float] = []
        ic_sign_values: List[float] = []
        decile_spread_sharpe_values: List[float] = []
        slope_beta_values: List[float] = []
        slope_fit_values: List[float] = []

        # Track path quality observations across folds
        path_mfe_values: List[float] = []
        path_mae_values: List[float] = []
        path_final_ret_values: List[float] = []
        path_time_to_mfe_values: List[float] = []
        path_time_to_mae_values: List[float] = []
        path_length_values: List[float] = []
        path_fold_ids: List[int] = []

        for fold_id, (_, va_idx) in enumerate(folds):
            y_va = fwd_ret[va_idx]
            mask = resolver.get_mask(canonical_key, va_idx)
            support = int(mask.sum())
            baseline_support = 0
            baseline_ret = np.nan
            uplift = np.nan
            fold_ic = np.nan
            baseline_ic = np.nan
            delta_ic = np.nan
            within_mask_ic = np.nan
            delta_within_mask_ic = np.nan
            decile_spread_sharpe = np.nan
            regression_beta = np.nan
            regression_slope_fit = np.nan
            entropy_reduction = np.nan

            if parent_context_key:
                parent_mask = resolver.get_mask(parent_context_key, va_idx)
                baseline_support = int(parent_mask.sum())
                if baseline_support > 0:
                    baseline_ret = float(np.nanmean(y_va[parent_mask]))

                    # Compute baseline IC
                    valid_idx = np.isfinite(y_va) & np.isfinite(parent_mask)
                    if np.sum(valid_idx) >= 3:
                        baseline_ic = _safe_spearman(
                            parent_mask[valid_idx].astype(np.float32), y_va[valid_idx]
                        )

            if support > 0:
                masked_ret = y_va[mask]
                mean_ret = float(np.nanmean(masked_ret))
                std_ret = float(np.nanstd(masked_ret))
                if np.isfinite(baseline_ret):
                    uplift = mean_ret - baseline_ret
                sign = 1 if mean_ret > epsilon else (-1 if mean_ret < -epsilon else 0)

                # Compute mask IC
                valid_idx = np.isfinite(y_va) & np.isfinite(mask)
                if np.sum(valid_idx) >= 3:
                    fold_ic = _safe_spearman(
                        mask[valid_idx].astype(np.float32), y_va[valid_idx]
                    )

                if np.isfinite(fold_ic) and np.isfinite(baseline_ic):
                    delta_ic = fold_ic - baseline_ic

                # Compute within-mask IC if bounded_target and predictions available
                if bounded_target is not None and predictions is not None:
                    target_va = bounded_target[va_idx]
                    preds_va = predictions[va_idx]
                    within_mask_ic, delta_within_mask_ic = self._compute_within_mask_ic(
                        preds_va, target_va, mask
                    )
                    if np.isfinite(within_mask_ic):
                        within_mask_ic_values.append(within_mask_ic)
                        ic_sign_values.append(1.0 if within_mask_ic > 0 else -1.0)
                    if np.isfinite(delta_within_mask_ic):
                        delta_within_mask_ic_values.append(delta_within_mask_ic)

                    decile_spread_sharpe = _compute_decile_spread_sharpe(
                        preds_va, target_va, mask
                    )
                    if np.isfinite(decile_spread_sharpe):
                        decile_spread_sharpe_values.append(decile_spread_sharpe)

                    (
                        regression_beta,
                        regression_slope_fit,
                    ) = _compute_regression_beta_and_fit(preds_va, target_va, mask)
                    if np.isfinite(regression_beta):
                        slope_beta_values.append(regression_beta)
                    if np.isfinite(regression_slope_fit):
                        slope_fit_values.append(regression_slope_fit)

                if bounded_target is not None:
                    target_va = bounded_target[va_idx]
                    entropy_reduction = self._compute_entropy_reduction(target_va, mask)
                    if np.isfinite(entropy_reduction):
                        entropy_reduction_values.append(entropy_reduction)

                # Collect path stats if available
                if (
                    path_mfe is not None
                    and path_mae is not None
                    and path_final_ret is not None
                    and path_time_to_mfe is not None
                    and path_time_to_mae is not None
                    and path_length is not None
                ):
                    mask_idx = np.where(mask)[0]
                    if mask_idx.size > 0:
                        global_idx = va_idx[mask_idx]
                        path_mfe_values.extend(path_mfe[global_idx].tolist())
                        path_mae_values.extend(path_mae[global_idx].tolist())
                        path_final_ret_values.extend(
                            path_final_ret[global_idx].tolist()
                        )
                        path_time_to_mfe_values.extend(
                            path_time_to_mfe[global_idx].tolist()
                        )
                        path_time_to_mae_values.extend(
                            path_time_to_mae[global_idx].tolist()
                        )
                        path_length_values.extend(path_length[global_idx].tolist())
                        path_fold_ids.extend([fold_id] * int(mask_idx.size))
            else:
                mean_ret = np.nan
                std_ret = np.nan
                sign = 0

            fold_records.append(
                {
                    "canonical_key": canonical_key,
                    "fold_id": fold_id,
                    "support": support,
                    "support_pct": support / max(len(va_idx), 1),
                    "mean_ret": mean_ret,
                    "std_ret": std_ret,
                    "sign": sign,
                    "baseline_support": baseline_support,
                    "baseline_ret": baseline_ret,
                    "uplift": uplift,
                    "parent_context_key": parent_context_key,
                    "fold_ic": fold_ic,
                    "baseline_ic": baseline_ic,
                    "delta_ic": delta_ic,
                    "within_mask_ic": within_mask_ic,
                    "delta_within_mask_ic": delta_within_mask_ic,
                    "decile_spread_sharpe": decile_spread_sharpe,
                    "regression_beta": regression_beta,
                    "regression_slope_fit": regression_slope_fit,
                    "entropy_reduction": entropy_reduction,
                }
            )

        df_folds = pd.DataFrame(fold_records)
        present = df_folds[df_folds["support"] > 0]
        if present.empty:
            # Deconstruct for visibility
            slots = parse_slot_map(
                canonical_key,
                getattr(self, "slot_order", ("trigger", "location", "regime")),
            )

            summary = {
                "canonical_key": canonical_key,
                "trigger": slots.get("trigger", "*"),
                "location": slots.get("location", "*"),
                "regime": slots.get("regime", "*"),
                "mean_net_ret": np.nan,
                "directional_mean_ret": np.nan,
                "std_net_ret": np.nan,
                "mean_within_fold_std": np.nan,
                "mean_support_pct": 0.0,
                "std_support_pct": 0.0,
                "presence_freq": 0.0,
                "presence_freq_units": 0.0,
                "sign_consistency": 0.0,
                "min_support_actual": 0,
                "mean_uplift": np.nan,
                "mean_baseline_ret": np.nan,
                "mean_oos_ic": np.nan,
                "p25_oos_ic": np.nan,
                "p50_oos_ic": np.nan,
                "p75_oos_ic": np.nan,
                "mean_delta_ic": np.nan,
                "positive_ic_fraction": 0.0,
                "within_mask_ic": np.nan,
                "delta_within_mask_ic": np.nan,
                "mean_ic": np.nan,
                "ic_tstat": np.nan,
                "ic_sign_consistency": np.nan,
                "decile_spread_sharpe": np.nan,
                "mask_ic_uplift": np.nan,
                "regression_beta": np.nan,
                "regression_slope_fit": np.nan,
                "entropy_reduction": np.nan,
                "learnability_step_c_score": np.nan,
                "quality_stability_score": np.nan,
                "trade_path_quality_score": np.nan,
                "full_quality_score": np.nan,
                "source_target": target_name,
                "source_horizon": horizon,
                "composite_score": -np.inf,
                "required_hurdle": np.nan,
                "hurdle_excess": np.nan,
                "n_folds": 0,
                "discovery_count": discovery_count,
                "n_instances": 0 if n_instances is None else n_instances,
                "display_arity": display_arity_for_key(canonical_key),
                "structural_depth": structural_depth_for_key(canonical_key),
                "pipeline_stage": pipeline_stage or "unknown",
                "parent_context_key": parent_context_key,
                "side": infer_rule_side(canonical_key, explicit_side=explicit_side),
                "rule_type": (
                    "composite"
                    if split_composite_key(canonical_key) is not None
                    else f"{display_arity_for_key(canonical_key)}-way"
                ),
                "accepted": False,
                "rejection_reason": "no_validation_support",
            }
            return summary, fold_records

        mean_net_ret = float(present["mean_ret"].mean())
        std_net_ret = float(present["mean_ret"].std(ddof=0))
        mean_within_fold_std = (
            float(present["std_ret"].mean())
            if present["std_ret"].notna().any()
            else np.nan
        )
        mean_support_pct = float(present["support_pct"].mean())
        std_support_pct = float(present["support_pct"].std(ddof=0))
        presence_freq = float(len(present) / max(len(folds), 1))
        # Compute sign_consistency across pooled out-of-sample returns
        if path_final_ret_values:
            pooled_oos_returns = np.asarray(path_final_ret_values, dtype=float)
            sign_consistency = compute_directional_sign_consistency(pooled_oos_returns)
        else:
            # Fallback to computing from available fold means or just setting to 0.5
            sign_consistency = 0.5
        display_arity = display_arity_for_key(canonical_key)
        required_hurdle = self._compute_required_hurdle(mean_support_pct, display_arity)
        use_directional = bool(self.cfg.get("stage_a_directional", True)) and (
            (pipeline_stage or "") == "stage_a_context"
        )
        directional_mean_ret = (
            abs(mean_net_ret)
            if (use_directional and np.isfinite(mean_net_ret))
            else mean_net_ret
        )
        hurdle_excess = directional_mean_ret - required_hurdle
        mean_uplift = (
            float(present["uplift"].mean())
            if present["uplift"].notna().any()
            else np.nan
        )
        mean_baseline_ret = (
            float(present["baseline_ret"].mean())
            if present["baseline_ret"].notna().any()
            else np.nan
        )

        # OOS IC metrics
        mean_oos_ic = (
            float(present["fold_ic"].mean())
            if present["fold_ic"].notna().any()
            else np.nan
        )
        p25_oos_ic = (
            float(present["fold_ic"].quantile(0.25))
            if present["fold_ic"].notna().any()
            else np.nan
        )
        p50_oos_ic = (
            float(present["fold_ic"].quantile(0.50))
            if present["fold_ic"].notna().any()
            else np.nan
        )
        p75_oos_ic = (
            float(present["fold_ic"].quantile(0.75))
            if present["fold_ic"].notna().any()
            else np.nan
        )
        mean_delta_ic = (
            float(present["delta_ic"].mean())
            if present["delta_ic"].notna().any()
            else np.nan
        )

        positive_ic_fraction = (
            float((present["fold_ic"] > 0).mean())
            if present["fold_ic"].notna().any()
            else 0.0
        )

        # Within-mask IC metrics (computed from fold-level values)
        mean_within_mask_ic = (
            float(np.mean(within_mask_ic_values)) if within_mask_ic_values else np.nan
        )
        mean_delta_within_mask_ic = (
            float(np.mean(delta_within_mask_ic_values))
            if delta_within_mask_ic_values
            else np.nan
        )
        mean_ic = mean_within_mask_ic
        if len(within_mask_ic_values) >= 2:
            ic_std = float(np.nanstd(within_mask_ic_values, ddof=1))
            ic_tstat = (
                float(mean_ic / (ic_std / np.sqrt(len(within_mask_ic_values))))
                if ic_std > 1e-12
                else np.nan
            )
        else:
            ic_tstat = np.nan
        if ic_sign_values:
            major_sign = 1.0 if np.nanmean(ic_sign_values) >= 0 else -1.0
            ic_sign_consistency = float(
                np.mean(np.array(ic_sign_values, dtype=float) == major_sign)
            )
        else:
            ic_sign_consistency = np.nan
        decile_spread_sharpe = (
            float(np.nanmean(decile_spread_sharpe_values))
            if decile_spread_sharpe_values
            else np.nan
        )
        regression_beta = (
            float(np.nanmean(slope_beta_values)) if slope_beta_values else np.nan
        )
        regression_slope_fit = (
            float(np.nanmean(slope_fit_values)) if slope_fit_values else np.nan
        )
        mask_ic_uplift = mean_delta_within_mask_ic
        mean_entropy_reduction = (
            float(np.mean(entropy_reduction_values))
            if entropy_reduction_values
            else np.nan
        )

        path_obs = len(path_mfe_values)
        path_folds = len(set(path_fold_ids)) if path_fold_ids else 0
        if canonical_key in self._path_quality_cache:
            path_quality = self._path_quality_cache[canonical_key]
            path_quality_elapsed = 0.0
        else:
            path_quality_start = time.perf_counter()
            path_quality = compute_trade_path_quality_metrics(
                mfe=np.asarray(path_mfe_values, dtype=float),
                mae=np.asarray(path_mae_values, dtype=float),
                final_ret=np.asarray(path_final_ret_values, dtype=float),
                time_to_mfe=np.asarray(path_time_to_mfe_values, dtype=float),
                time_to_mae=np.asarray(path_time_to_mae_values, dtype=float),
                path_length=np.asarray(path_length_values, dtype=float),
                fold_id=np.asarray(path_fold_ids, dtype=int),
            )
            path_quality_elapsed = time.perf_counter() - path_quality_start
            self._path_quality_cache[canonical_key] = path_quality
        if path_quality_elapsed >= 1.0 or path_obs >= 5000:
            pass  # Trade path quality calculation (logging removed)
        trade_path_quality_score = float(
            path_quality.get("trade_path_quality_score", np.nan)
        )
        quality_stability_score = float(
            path_quality.get("quality_stability_score", np.nan)
        )
        path_smoothness_term = float(path_quality.get("path_smoothness_term", np.nan))
        path_survivability_term = float(
            path_quality.get("path_survivability_term", np.nan)
        )
        path_stability_term = float(path_quality.get("path_stability_term", np.nan))
        path_realized_profit_consistency_term = float(
            path_quality.get("path_realized_profit_consistency_term", np.nan)
        )
        path_trajectory_smoothness_term = float(
            path_quality.get("path_trajectory_smoothness_term", np.nan)
        )

        support_objective_score = self._compute_support_objective_score(
            mean_support_pct
        )

        # Step A: economic edge (support and presence intentionally removed)
        edge_ret_scale = float(self.cfg.get("score_edge_ret_scale", 0.002))
        s_edge_ret = _safe_tanh_scale(max(directional_mean_ret, 0.0), edge_ret_scale)
        s_edge_sign = float(np.clip(sign_consistency, 0.0, 1.0))
        s_edge_vol = float(1.0 / (1.0 + max(std_net_ret, 0.0)))
        s_edge = (
            s_edge_ret * s_edge_sign * s_edge_vol if np.isfinite(s_edge_ret) else np.nan
        )

        # Step C: enhanced learnability score
        ic_lo = float(self.cfg.get("step_c_ic_lo", -0.02))
        ic_hi = float(self.cfg.get("step_c_ic_hi", 0.05))
        ic_t_scale = float(self.cfg.get("step_c_ic_t_scale", 3.0))
        spread_scale = float(self.cfg.get("step_c_spread_sharpe_scale", 2.0))
        uplift_lo = float(self.cfg.get("step_c_uplift_lo", -0.01))
        uplift_hi = float(self.cfg.get("step_c_uplift_hi", 0.03))
        neutral_comp = float(self.cfg.get("score_component_neutral", 0.5))
        eps_score = 1e-6

        def _clip01(v: float) -> float:
            if not np.isfinite(v):
                return neutral_comp
            return float(np.clip(v, 0.0, 1.0))

        s_ic_mean = _clip01((mean_ic - ic_lo) / max(ic_hi - ic_lo, 1e-9))
        s_ic_t = _clip01(_safe_tanh_scale(max(ic_tstat, 0.0), ic_t_scale))
        s_ic_sign = _clip01(ic_sign_consistency)
        s_spread = _clip01(
            _safe_tanh_scale(max(decile_spread_sharpe, 0.0), spread_scale)
        )
        s_uplift = _clip01(
            (mask_ic_uplift - uplift_lo) / max(uplift_hi - uplift_lo, 1e-9)
        )
        s_slope = _clip01(regression_slope_fit)

        learnability_step_c_score = float(
            (s_ic_mean + eps_score) ** 0.20
            * (s_ic_t + eps_score) ** 0.15
            * (s_ic_sign + eps_score) ** 0.15
            * (s_spread + eps_score) ** 0.20
            * (s_uplift + eps_score) ** 0.20
            * (s_slope + eps_score) ** 0.10
        )

        s_path = (
            float(np.clip(trade_path_quality_score, 0.0, 1.0))
            if np.isfinite(trade_path_quality_score)
            else neutral_comp
        )
        s_edge_use = s_edge if np.isfinite(s_edge) else neutral_comp
        s_c_use = (
            learnability_step_c_score
            if np.isfinite(learnability_step_c_score)
            else neutral_comp
        )

        if not np.isfinite(support_objective_score):
            full_quality_score = -np.inf
        else:
            s_support_use = float(np.clip(support_objective_score, 0.0, 1.0))
            # Final quality score with requested weights:
            # edge 10%, path quality 35%, learnability 45%, support fit 10%.
            w_a, w_b, w_c, w_s = 0.10, 0.35, 0.45, 0.10
            full_quality_score = float(
                (s_edge_use + eps_score) ** w_a
                * (s_path + eps_score) ** w_b
                * (s_c_use + eps_score) ** w_c
                * (s_support_use + eps_score) ** w_s
            )
        composite_score_step1 = full_quality_score

        # Deconstruct for visibility
        slots = parse_slot_map(
            canonical_key,
            getattr(self, "slot_order", ("trigger", "location", "regime")),
        )

        summary = {
            "canonical_key": canonical_key,
            "trigger": slots.get("trigger", "*"),
            "location": slots.get("location", "*"),
            "regime": slots.get("regime", "*"),
            "mean_net_ret": mean_net_ret,
            "directional_mean_ret": directional_mean_ret,
            "std_net_ret": std_net_ret,
            "mean_within_fold_std": mean_within_fold_std,
            "mean_support_pct": mean_support_pct,
            "std_support_pct": std_support_pct,
            "presence_freq": presence_freq,
            "presence_freq_units": presence_freq,
            "sign_consistency": sign_consistency,
            "min_support_actual": int(present["support"].min()),
            "mean_uplift": mean_uplift,
            "mean_baseline_ret": mean_baseline_ret,
            "mean_oos_ic": mean_oos_ic,
            "p25_oos_ic": p25_oos_ic,
            "p50_oos_ic": p50_oos_ic,
            "p75_oos_ic": p75_oos_ic,
            "mean_delta_ic": mean_delta_ic,
            "positive_ic_fraction": positive_ic_fraction,
            "within_mask_ic": mean_within_mask_ic,
            "delta_within_mask_ic": mean_delta_within_mask_ic,
            "mean_ic": mean_ic,
            "ic_tstat": ic_tstat,
            "ic_sign_consistency": ic_sign_consistency,
            "decile_spread_sharpe": decile_spread_sharpe,
            "mask_ic_uplift": mask_ic_uplift,
            "regression_beta": regression_beta,
            "regression_slope_fit": regression_slope_fit,
            "entropy_reduction": mean_entropy_reduction,
            "learnability_step_c_score": learnability_step_c_score,
            "support_objective_score": support_objective_score,
            "trade_path_quality_score": trade_path_quality_score,
            "quality_stability_score": quality_stability_score,
            "path_smoothness_term": path_smoothness_term,
            "path_survivability_term": path_survivability_term,
            "path_stability_term": path_stability_term,
            "path_realized_profit_consistency_term": path_realized_profit_consistency_term,
            "path_trajectory_smoothness_term": path_trajectory_smoothness_term,
            "full_quality_score": full_quality_score,
            "source_target": target_name,
            "source_horizon": horizon,
            "composite_score_step1": composite_score_step1,
            "required_hurdle": required_hurdle,
            "hurdle_excess": hurdle_excess,
            "n_folds": int(len(present)),
            "discovery_count": int(discovery_count),
            "n_instances": int(len(present) if n_instances is None else n_instances),
            "display_arity": display_arity,
            "structural_depth": structural_depth_for_key(canonical_key),
            "pipeline_stage": pipeline_stage or "unknown",
            "parent_context_key": parent_context_key,
            "side": infer_rule_side(
                canonical_key, mean_net_ret=mean_net_ret, explicit_side=explicit_side
            ),
            "rule_type": (
                "composite"
                if split_composite_key(canonical_key) is not None
                else f"{display_arity}-way"
            ),
        }

        rejected: List[str] = []
        if summary["min_support_actual"] < int(
            self.cfg.get("min_support_count_validation", 10)
        ):
            rejected.append("low_support")
        if (
            not np.isfinite(summary["directional_mean_ret"])
            or summary["directional_mean_ret"] <= 0
        ):
            rejected.append("non_positive_directional_ret")

        _cost_pct = float(self.cfg.get("rule_economic_cost_pct", 0.003))
        _pooled_mask = np.zeros(len(fwd_ret), dtype=bool)
        for _, va_idx in folds:
            _pooled_mask[va_idx] |= resolver.get_mask(canonical_key, va_idx)
        if _pooled_mask.sum() >= 10:
            _masked_rets = fwd_ret[_pooled_mask]
            _k10 = max(1, int(0.10 * len(_masked_rets)))
            _top10_idx = np.argpartition(_masked_rets, -_k10)[-_k10:]
            _top10_mean = float(np.mean(_masked_rets[_top10_idx]))
            summary["top10_mean_ret"] = _top10_mean
            summary["top10_net_ret"] = _top10_mean - _cost_pct
            if _top10_mean - _cost_pct < 0:
                rejected.append("negative_top10_net_ret")
        else:
            summary["top10_mean_ret"] = np.nan
            summary["top10_net_ret"] = np.nan

        if require_uplift:
            if not np.isfinite(summary["mean_uplift"]):
                rejected.append("missing_uplift")
            elif summary["mean_uplift"] <= 0:
                rejected.append("non_positive_uplift")

        summary["accepted"] = len(rejected) == 0
        summary["rejection_reason"] = "|".join(rejected)

        # NEW: Classify rule type (ranking vs gate)
        rule_type_class = classify_rule_type(
            directional_mean_ret=summary.get("directional_mean_ret", np.nan),
            mean_uplift=summary.get("mean_uplift", np.nan),
            sign_consistency=summary.get("sign_consistency", 0.0),
            required_hurdle=summary.get("required_hurdle", 0.0),
        )
        summary["rule_type_class"] = rule_type_class

        # NEW: Production classification
        classification, diagnostics = classify_rule_production_quality(rule=summary)
        summary["production_classification"] = classification
        summary["classification_diagnostics"] = json.dumps(diagnostics)

        return summary, fold_records

    def score_registry_oos(
        self,
        keys: Sequence[str],
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        resolver: Optional[
            Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]
        ] = None,
        parent_context_map: Optional[Dict[str, str]] = None,
        require_uplift_keys: Optional[Sequence[str]] = None,
        discovery_count_map: Optional[Dict[str, int]] = None,
        n_instances_map: Optional[Dict[str, int]] = None,
        pipeline_stage_map: Optional[Dict[str, str]] = None,
        side_map: Optional[Dict[str, str]] = None,
        bounded_target: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        path_mfe: Optional[np.ndarray] = None,
        path_mae: Optional[np.ndarray] = None,
        path_final_ret: Optional[np.ndarray] = None,
        path_time_to_mfe: Optional[np.ndarray] = None,
        path_time_to_mae: Optional[np.ndarray] = None,
        path_length: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        resolver = resolver or self.mask_resolver
        if resolver is None:
            raise ValueError("RuleScorer requires a mask resolver")

        require_uplift_set = set(require_uplift_keys or [])
        summaries: List[Dict[str, Any]] = []
        audits: List[Dict[str, Any]] = []
        seen: set[str] = set()

        # Fast path scoring using NumbaRuleInferenceEngine if we have simple non-composite keys
        # and our resolver supports giving us the underlying X array.
        fast_path = False
        try:
            if isinstance(resolver, CanonicalRuleMaskResolver):
                simple_boolean_keys = True
                for key in keys:
                    if "Composite" in str(key):
                        simple_boolean_keys = False
                        break
                    for part in str(key).split("|"):
                        slot_value = part.strip("()")
                        if slot_value == "*":
                            continue
                        for cond_str in slot_value.split("&"):
                            parsed = parse_condition_string(cond_str)
                            if parsed is None or parsed[1] != "==":
                                simple_boolean_keys = False
                                break
                        if not simple_boolean_keys:
                            break
                    if not simple_boolean_keys:
                        break
                if simple_boolean_keys:
                    fast_registry = pd.DataFrame({"canonical_key": keys})
                    engine = NumbaRuleInferenceEngine(fast_registry, resolver.metadata)
                    mask_matrix = engine.apply(resolver.X)
                    fast_path = True
        except KeyError:
            fast_path = False

        for idx, key in enumerate(keys):
            if key in seen:
                continue
            seen.add(key)

            if fast_path and "Composite" not in key:
                # Override the mask in the resolver dynamically
                resolver.context_lookup[key] = mask_matrix[:, idx]

            summary, fold_records = self.score_key_oos(
                canonical_key=key,
                fwd_ret=fwd_ret,
                folds=folds,
                resolver=resolver,
                require_uplift=key in require_uplift_set,
                parent_context_key=(parent_context_map or {}).get(key),
                discovery_count=(discovery_count_map or {}).get(key, 0),
                n_instances=(n_instances_map or {}).get(key),
                pipeline_stage=(pipeline_stage_map or {}).get(key),
                explicit_side=(side_map or {}).get(key),
                bounded_target=bounded_target,
                predictions=predictions,
                path_mfe=path_mfe,
                path_mae=path_mae,
                path_final_ret=path_final_ret,
                path_time_to_mfe=path_time_to_mfe,
                path_time_to_mae=path_time_to_mae,
                path_length=path_length,
            )
            summaries.append(summary)
            audits.extend(fold_records)

        if not summaries:
            tprint("WARNING: No rules scored successfully. Returning empty registry.")
            return pd.DataFrame(), pd.DataFrame(audits)

        summary_df = pd.DataFrame(summaries).sort_values(
            ["accepted", "composite_score_step1"], ascending=[False, False]
        )

        # Scorer Reporting Diagnostics
        accepted_count = summary_df["accepted"].sum()
        rejected_count = len(summary_df) - accepted_count
        tprint(
            f"Scorer Input: {len(summary_df)} rules | Accepted: {accepted_count} | Rejected: {rejected_count}"
        )

        rejection_reasons = collections.Counter(
            reason.strip()
            for reasons in summary_df[~summary_df["accepted"]][
                "rejection_reason"
            ].dropna()
            for reason in reasons.split("|")
            if reason.strip()
        )
        if rejection_reasons:
            tprint("Top scorer rejection reasons:")
            for reason, count in rejection_reasons.most_common(5):
                tprint(f"  - {reason}: {count}")

        return summary_df, pd.DataFrame(audits)


@njit(cache=True, fastmath=True)
def tbm_outcomes_atr_nb(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    tp_atr: float,
    sl_atr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = close.shape[0]
    tp_first = np.zeros(n, dtype=np.int8)
    sl_first = np.zeros(n, dtype=np.int8)
    timeout = np.zeros(n, dtype=np.int8)

    if horizon <= 0:
        return tp_first, sl_first, timeout

    for i in range(n - horizon):
        entry = close[i]
        atr_i = max(atr[i], 1e-9)

        tp_price = entry + tp_atr * atr_i
        sl_price = entry - sl_atr * atr_i

        for j in range(i + 1, i + horizon + 1):
            hi = high[j]
            lo = low[j]

            hit_tp = hi >= tp_price
            hit_sl = lo <= sl_price

            if hit_tp and not hit_sl:
                tp_first[i] = 1
                break
            if hit_sl and not hit_tp:
                sl_first[i] = 1
                break
            if hit_tp and hit_sl:
                median = 0.5 * (hi + lo)
                d_tp = abs(median - tp_price)
                d_sl = abs(median - sl_price)
                if d_tp < d_sl:
                    tp_first[i] = 1
                elif d_sl < d_tp:
                    sl_first[i] = 1
                else:
                    timeout[i] = 1
                break
        else:
            timeout[i] = 1
    return tp_first, sl_first, timeout


class RulePruner:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def prune_for_assessment(
        self, scored_df: pd.DataFrame, all_rules: List[ExtractedRule], top_n: int = 50
    ) -> pd.DataFrame:
        """
        Prunes rules using a hybrid of Scorer metrics (OOS) and LGBM Native metrics (IS).
        """
        if scored_df.empty:
            return scored_df

        # 1. Aggregate Model-Native Metrics from ExtractedRule objects
        # We want to know how the model 'felt' about this canonical rule during training
        native_stats = []
        unique_keys = scored_df["canonical_key"].unique()

        for key in unique_keys:
            # Get all instances (across trees/folds/seeds) of this canonical rule
            instances = [r for r in all_rules if r.canonical_key == key]

            # Calculate Model conviction
            avg_leaf_val = np.mean([r.leaf_value for r in instances])
            total_is_support = np.sum([r.support_train for r in instances])
            occurrence_count = len(instances)  # How many trees used this rule?

            native_stats.append(
                {
                    "canonical_key": key,
                    "avg_model_conviction": abs(avg_leaf_val),
                    "total_is_support": total_is_support,
                    "discovery_count": occurrence_count,
                }
            )

        native_df = pd.DataFrame(native_stats)

        # 2. Merge Native metrics into the Scored Registry
        df = scored_df.merge(native_df, on="canonical_key", how="left")

        # 3. Hard Gates based on Model conviction
        # Reject rules that the model only used once or with very low importance (leaf value)
        min_conviction = float(self.cfg.get("min_avg_leaf_value", 0.001))
        min_discoveries = int(self.cfg.get("min_tree_discoveries", 2))

        dropped_conviction = (df["avg_model_conviction"] < min_conviction).sum()
        dropped_discoveries = (df["discovery_count"] < min_discoveries).sum()
        dropped_oos = (df["mean_net_ret"] <= 0).sum()

        tprint(f"RulePruner (Assessment Prep): Input {len(df)} rules")
        if dropped_conviction > 0:
            tprint(
                f"  - Rejected {dropped_conviction} rules (conviction < {min_conviction})"
            )
        if dropped_discoveries > 0:
            tprint(
                f"  - Rejected {dropped_discoveries} rules (discoveries < {min_discoveries})"
            )
        if dropped_oos > 0:
            tprint(f"  - Rejected {dropped_oos} rules (OOS mean_net_ret <= 0)")

        mask = (
            (df["avg_model_conviction"] >= min_conviction)
            & (df["discovery_count"] >= min_discoveries)
            & (df["mean_net_ret"] > 0)  # Basic OOS sanity check
        )

        pruned_df = df[mask].copy()
        tprint(
            f"RulePruner (Assessment Prep): Selected {len(pruned_df)} rules (pre-ranking)"
        )

        # 4. Final Ranking for Assessment
        # We rank by a hybrid of OOS performance and Model Discovery Count
        # Discovery Count is a great proxy for 'Structural Stability'
        pruned_df["prune_rank_score"] = pruned_df["composite_score"] * np.log1p(
            pruned_df["discovery_count"]
        )

        return pruned_df.sort_values("prune_rank_score", ascending=False).head(top_n)


class IndependentRulePruner:
    """
    Independent Rule Pruner (Hurdle Edition v2.0)
    Updated with:
    1. Hard Max Support Gate (to kill global 'nothingness' rules)
    2. Complexity Bonus (to reward 2-way and 3-way interactions)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base_hurdle = float(cfg.get("prune_base_hurdle", 0.0002))
        self.hurdle_aggressiveness = float(cfg.get("prune_hurdle_aggressiveness", 1.0))
        self.target_support = float(cfg.get("prune_target_support_pct", TARGET_SUPPORT))
        self.min_support_pct = float(cfg.get("support_min_pct", SUPPORT_MIN))
        self.min_sign_consistency = float(cfg.get("min_sign_consistency", 0.0))

        # New Gates
        self.max_support_pct = float(
            cfg.get("max_support_pct", SUPPORT_MAX)
        )  # Hard ceiling at 20%
        self.hurdle_gate_bottom_pctile = float(
            cfg.get("hurdle_gate_bottom_pctile", 0.20)
        )
        self.arity_bonus = cfg.get(
            "prune_complexity_bonus_map",
            {"1": 0.0, "2": 0.15, "3": 0.30, "4": 0.10, "5": 0.10, "6": 0.10},
        )

    def prune(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        # 1. Hard Gates: support bounds
        df["is_too_narrow"] = df["mean_support_pct"] < self.min_support_pct
        df["is_too_broad"] = df["mean_support_pct"] > self.max_support_pct

        # 2. Determine Rule Complexity from normalized metadata
        df["comp_bonus"] = df["display_arity"].apply(
            lambda val: float(self.arity_bonus.get(str(int(val)), 0.0))
        )

        # 3. Calculate the Complexity-Adjusted Hurdle
        # U-shaped penalty favoring target_support
        safe_support = df["mean_support_pct"].clip(lower=0.0005)
        dist = safe_support - self.target_support
        # Asymmetric penalty multiplier
        penalty_multiplier = self.hurdle_aggressiveness * (
            1.0 + np.where(dist < 0, 10.0 * (dist**2), 5.0 * (dist**2))
        )

        df["required_hurdle"] = (
            self.base_hurdle * (1.0 - df["comp_bonus"])
        ) * penalty_multiplier

        # 4. Hurdle remains as a downstream score/statistic only.
        df["hurdle_excess"] = df["mean_net_ret"] - df["required_hurdle"]

        # 5. Final Selection
        gate_summary = {
            "is_too_narrow_rejected": int(df["is_too_narrow"].sum()),
            "is_too_broad_rejected": int(df["is_too_broad"].sum()),
        }

        mask = (~df["is_too_narrow"]) & (~df["is_too_broad"])

        final_registry = df[mask]

        tprint(
            f"Pruning Gate-by-Gate Funnel: Total={len(df)} | "
            f"Narrow Rejected={gate_summary['is_too_narrow_rejected']} | "
            f"Broad Rejected={gate_summary['is_too_broad_rejected']} | "
            f"Final Accepted={len(final_registry)}"
        )

        # Save gate summary as attribute to extract later
        self.gate_summary = gate_summary

        return final_registry.sort_values("hurdle_excess", ascending=False)


def compute_tbm_outcomes_per_symbol(
    data: pd.DataFrame,
    horizon: int,
    tp_atr: float,
    sl_atr: float,
    side: str = "long",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute TBM outcomes independently within each symbol's time series.

    Assumes `data` has columns:
      - symbol
      - timestamp
      - close
      - high
      - low
      - atr

    Returns arrays aligned to `data.index`.
    """
    if data.empty:
        z = np.zeros(0, dtype=np.int8)
        return z, z, z

    # Preserve original row order for final alignment
    out_tp = np.zeros(len(data), dtype=np.int8)
    out_sl = np.zeros(len(data), dtype=np.int8)
    out_to = np.zeros(len(data), dtype=np.int8)

    # Sort once for temporal correctness inside each symbol
    work = data.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    work = work.sort_values(["symbol", "timestamp"], kind="mergesort")

    for sym, g in work.groupby("symbol", sort=False):
        idx = g["_orig_idx"].to_numpy()

        close = g["close"].to_numpy(dtype=np.float64, copy=False)
        high = g["high"].to_numpy(dtype=np.float64, copy=False)
        low = g["low"].to_numpy(dtype=np.float64, copy=False)
        atr = g["atr"].to_numpy(dtype=np.float64, copy=False)

        if side == "short":
            c, h, l = -close, -low, -high
        else:
            c, h, l = close, high, low

        tp_f, sl_f, to_f = tbm_outcomes_atr_nb(
            close=c,
            high=h,
            low=l,
            atr=atr,
            horizon=horizon,
            tp_atr=tp_atr,
            sl_atr=sl_atr,
        )

        out_tp[idx] = tp_f
        out_sl[idx] = sl_f
        out_to[idx] = to_f

    return out_tp, out_sl, out_to


def build_context_feature_dict_from_registry(
    registry: pd.DataFrame,
    data: pd.DataFrame,
    X_stage_a: np.ndarray,
    metadata_stage_a: List[FeatureMetadata],
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    if registry.empty:
        return {}, {}

    context_feature_dict: Dict[str, np.ndarray] = {}
    context_feature_to_stage_a_key: Dict[str, str] = {}
    resolver = CanonicalRuleMaskResolver(X_stage_a, metadata_stage_a)

    for row in registry.itertuples(index=False, name=None):
        key = row[0]  # canonical_key
        ctx_hash = hashlib.sha1(key.encode()).hexdigest()[:8]
        ctx_name = f"ctx__{ctx_hash}"
        mask = resolver.get_mask(key)
        context_feature_dict[ctx_name] = mask.astype(np.float32)
        context_feature_to_stage_a_key[ctx_name] = key

    return context_feature_dict, context_feature_to_stage_a_key


def build_rule_model_importance_scores(all_rules: List[ExtractedRule]) -> pd.DataFrame:
    """
    Aggregate model-native feature gain/split into canonical rule importance scores.
    """
    if not all_rules:
        return pd.DataFrame(
            columns=[
                "canonical_key",
                "rule_gain_score",
                "rule_split_score",
                "rule_model_importance_score",
            ]
        )

    instance_rows: List[Dict[str, Any]] = []
    for rule in all_rules:
        # We no longer calculate rule_gain_score and rule_split_score from feature importances
        # as it was legacy code. We only need the rule_model_importance_score.
        instance_rows.append(
            {
                "canonical_key": rule.canonical_key,
                "rule_gain_score": 0.0,
                "rule_split_score": 0.0,
                "rule_model_importance_score": rule.rule_model_importance_score,
            }
        )

    if not instance_rows:
        return pd.DataFrame(
            columns=[
                "canonical_key",
                "rule_gain_score",
                "rule_split_score",
                "rule_model_importance_score",
            ]
        )

    return (
        pd.DataFrame(instance_rows)
        .groupby("canonical_key", as_index=False)[
            ["rule_gain_score", "rule_split_score", "rule_model_importance_score"]
        ]
        .mean()
    )


def select_stage_a_contexts(
    stage_a_result: Dict[str, Any], cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    registry = stage_a_result.get("accepted_registry")
    if registry is None or registry.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["reason", "count"])

    registry = registry.copy()

    def _resolve_metric_col(*base_names: str) -> pd.Series:
        for base_name in base_names:
            for candidate in (base_name, f"{base_name}_x", f"{base_name}_y"):
                if candidate in registry.columns:
                    return registry[candidate]
        return pd.Series(np.nan, index=registry.index, dtype=np.float32)

    support_floor = float(
        cfg.get("min_context_support_pct", cfg.get("support_min_pct", SUPPORT_MIN))
    )
    registry["reject_support"] = _resolve_metric_col("mean_support_pct") < support_floor
    registry["reject_ret"] = _resolve_metric_col("directional_mean_ret") <= float(
        cfg.get("min_context_mean_ret", 0.0)
    )
    registry["reject_presence"] = False
    registry["reject_sign"] = False
    registry["reject_arity"] = False
    structural_series = _resolve_metric_col("is_structurally_sound").fillna(False)
    registry["reject_structural"] = ~structural_series.astype(bool)

    mask = ~(
        registry["reject_support"]
        | registry["reject_ret"]
        | registry["reject_presence"]
        | registry["reject_sign"]
        | registry["reject_arity"]
        | registry["reject_structural"]
    )

    selected = registry[mask]

    rejection_reasons = []
    for col in [
        "reject_support",
        "reject_ret",
        "reject_presence",
        "reject_sign",
        "reject_arity",
        "reject_structural",
    ]:
        rejection_reasons.append({"reason": col, "count": int(registry[col].sum())})
    rejection_summary = pd.DataFrame(rejection_reasons, columns=["reason", "count"])

    return selected, rejection_summary


def build_stage_a_rejection_map(
    stage_a_result: Dict[str, Any],
    winning_contexts: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Summarize the Stage A rejection funnel across scorer, pruner, assessor,
    and final context selection gates."""
    rows: List[Dict[str, Any]] = []

    def _append_stage_rows(
        stage_name: str,
        stage_order: int,
        input_count: int,
        gate_items: List[Tuple[str, str, int, str]],
        passed_count: int,
    ) -> None:
        for gate_name, metric_name, rejected_count, threshold in gate_items:
            rows.append(
                {
                    "stage_order": stage_order,
                    "stage_name": stage_name,
                    "gate_name": gate_name,
                    "metric_name": metric_name,
                    "threshold": threshold,
                    "input_count": int(input_count),
                    "rejected_count": int(rejected_count),
                    "passed_count": int(passed_count),
                }
            )

    scored_registry = stage_a_result.get("scored_registry", pd.DataFrame())
    scorer_accepted = stage_a_result.get("scorer_accepted", pd.DataFrame())
    if not scored_registry.empty:
        scorer_rejections = collections.Counter(
            reason.strip()
            for reasons in scored_registry.loc[
                ~scored_registry["accepted"].fillna(False), "rejection_reason"
            ].dropna()
            for reason in str(reasons).split("|")
            if reason.strip()
        )
        _append_stage_rows(
            stage_name="scorer",
            stage_order=1,
            input_count=len(scored_registry),
            gate_items=[
                (
                    "low_support",
                    "min_support_actual",
                    scorer_rejections.get("low_support", 0),
                    f">= {int(cfg.get('min_support_count_validation', 10))}",
                ),
                (
                    "non_positive_directional_ret",
                    "directional_mean_ret",
                    scorer_rejections.get("non_positive_directional_ret", 0),
                    "> 0",
                ),
                (
                    "below_hurdle",
                    "hurdle_excess",
                    scorer_rejections.get("below_hurdle", 0),
                    "> 0",
                ),
            ],
            passed_count=len(scorer_accepted),
        )

    candidate_registry = stage_a_result.get("candidate_registry", pd.DataFrame())
    scorer_accepted = stage_a_result.get("scorer_accepted", pd.DataFrame())
    if not scorer_accepted.empty:
        _append_stage_rows(
            stage_name="pruner",
            stage_order=2,
            input_count=len(scorer_accepted),
            gate_items=[
                (
                    "is_too_narrow",
                    "mean_support_pct",
                    int(
                        (
                            scorer_accepted.get(
                                "mean_support_pct",
                                pd.Series(index=scorer_accepted.index, dtype=float),
                            )
                            < float(cfg.get("support_min_pct", SUPPORT_MIN))
                        ).sum()
                    ),
                    f">= {float(cfg.get('support_min_pct', SUPPORT_MIN)):.4f}",
                ),
                (
                    "is_too_broad",
                    "mean_support_pct",
                    int(
                        (
                            scorer_accepted.get(
                                "mean_support_pct",
                                pd.Series(index=scorer_accepted.index, dtype=float),
                            )
                            > float(cfg.get("max_support_pct", SUPPORT_MAX))
                        ).sum()
                    ),
                    f"<= {float(cfg.get('max_support_pct', SUPPORT_MAX)):.4f}",
                ),
            ],
            passed_count=len(candidate_registry),
        )

    assessment_df = stage_a_result.get("assessment_df", pd.DataFrame())
    accepted_registry = stage_a_result.get("accepted_registry", pd.DataFrame())
    if not candidate_registry.empty:
        assessed_keys = set()
        if not assessment_df.empty and "canonical_key" in assessment_df.columns:
            assessed_keys = set(assessment_df["canonical_key"].astype(str))
        candidate_keys = set(candidate_registry["canonical_key"].astype(str))
        unassessed_count = len(candidate_keys - assessed_keys)
        assessment_rejections = collections.Counter()
        if not assessment_df.empty:
            assessment_rejections.update(
                reason
                for reason in assessment_df.loc[
                    ~assessment_df["is_structurally_sound"].fillna(False),
                    "rejection_reason",
                ].dropna()
                if str(reason).strip()
            )
        _append_stage_rows(
            stage_name="mask_assessor",
            stage_order=3,
            input_count=len(candidate_registry),
            gate_items=[
                (
                    "not_assessed_min_mask_support",
                    "mask_support_count",
                    unassessed_count,
                    ">= 20 rows",
                ),
                (
                    "low_trades_per_day",
                    "avg_trades_per_day",
                    assessment_rejections.get("low_trades_per_day", 0),
                    f">= {float(cfg.get('min_avg_trades_per_day_10_symbols', 0.1)):.4f}",
                ),
                (
                    "insufficient_baseline_oof_coverage",
                    "baseline_oof_coverage",
                    assessment_rejections.get("insufficient_baseline_oof_coverage", 0),
                    f">= {float(cfg.get('learnability_min_oof_coverage', 0.25)):.4f}",
                ),
                (
                    "insufficient_subset_oof_coverage",
                    "subset_oof_coverage",
                    assessment_rejections.get("insufficient_subset_oof_coverage", 0),
                    f">= {float(cfg.get('learnability_min_oof_coverage', 0.25)):.4f}",
                ),
                (
                    "missing_learnability",
                    "learn_eff_ratio",
                    assessment_rejections.get("missing_learnability", 0),
                    "must be finite",
                ),
                (
                    "low_sign_consistency",
                    "sign_consistency",
                    assessment_rejections.get("low_sign_consistency", 0),
                    f">= {float(cfg.get('min_sign_consistency', 0.0)):.4f}",
                ),
                (
                    "low_lift",
                    "learn_eff_ratio",
                    assessment_rejections.get("low_lift", 0),
                    ">= 1.1000",
                ),
                # NOTE: low_entropy_reduction gate removed per user request
            ],
            passed_count=len(accepted_registry),
        )

    if not accepted_registry.empty or not winning_contexts.empty:
        selected_registry, selection_summary = select_stage_a_contexts(
            stage_a_result, cfg
        )
        if winning_contexts is not None and not winning_contexts.empty:
            selected_registry = winning_contexts
        selection_counts = {
            str(row["reason"]): int(row["count"])
            for _, row in selection_summary.iterrows()
        }
        _append_stage_rows(
            stage_name="context_selector",
            stage_order=4,
            input_count=len(accepted_registry),
            gate_items=[
                (
                    "reject_support",
                    "mean_support_pct",
                    selection_counts.get("reject_support", 0),
                    f">= {float(cfg.get('min_context_support_pct', 0.01)):.4f}",
                ),
                (
                    "reject_ret",
                    "directional_mean_ret",
                    selection_counts.get("reject_ret", 0),
                    f"> {float(cfg.get('min_context_mean_ret', 0.0)):.4f}",
                ),
                (
                    "reject_presence",
                    "presence_freq",
                    selection_counts.get("reject_presence", 0),
                    f">= {float(cfg.get('min_context_presence_freq', cfg.get('min_presence_freq', 0.4))):.4f}",
                ),
                (
                    "reject_sign",
                    "sign_consistency",
                    selection_counts.get("reject_sign", 0),
                    f">= {float(cfg.get('min_context_sign_consistency', cfg.get('min_sign_consistency', 0.0))):.4f}",
                ),
                (
                    "reject_arity",
                    "display_arity",
                    selection_counts.get("reject_arity", 0),
                    f">= {int(cfg.get('min_context_display_arity', 2))}",
                ),
                (
                    "reject_structural",
                    "is_structurally_sound",
                    selection_counts.get("reject_structural", 0),
                    "must be True",
                ),
            ],
            passed_count=len(selected_registry),
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "stage_order",
                "stage_name",
                "gate_name",
                "metric_name",
                "threshold",
                "input_count",
                "rejected_count",
                "passed_count",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["stage_order", "gate_name"])
        .reset_index(drop=True)
    )


def create_pre_global_registry(side_results: Dict[str, Any]) -> pd.DataFrame:
    stage_a_selected = side_results.get("stage_a")
    if stage_a_selected is None or stage_a_selected.empty:
        stage_a_selected = pd.DataFrame()
    else:
        stage_a_selected = stage_a_selected.copy()
        stage_a_selected["origin_stage"] = "stage_a"
    return stage_a_selected


def log_stage_gate_diagnostics(
    stage_name: str, stage_result: Dict[str, Any], cfg: Dict[str, Any]
) -> None:
    scored = stage_result.get("scored_registry")
    scorer_accepted = stage_result.get("scorer_accepted")
    candidate = stage_result.get("candidate_registry")
    assessed = stage_result.get("assessment_df")
    accepted = stage_result.get("accepted_registry")
    tprint(
        f"{stage_name} gate counts: extracted={len(stage_result.get('all_extracted_rules', []))} "
        f"scored={0 if scored is None else len(scored)} "
        f"scorer_accepted={0 if scorer_accepted is None else len(scorer_accepted)} "
        f"candidate={0 if candidate is None else len(candidate)} "
        f"assessed={0 if assessed is None else len(assessed)} "
        f"accepted={0 if accepted is None else len(accepted)}"
    )
    if (
        scored is not None
        and not scored.empty
        and (scorer_accepted is None or scorer_accepted.empty)
    ):
        rejected = scored[~scored["accepted"]]
        if "rejection_reason" in rejected.columns:
            reason_counts = (
                rejected["rejection_reason"]
                .fillna("")
                .astype(str)
                .str.split("|", regex=False)
                .explode()
                .str.strip()
            )
            reason_counts = reason_counts[reason_counts != ""].value_counts().head(8)
            if not reason_counts.empty:
                tprint(
                    f"{stage_name} scorer rejection reasons: "
                    + ", ".join(
                        f"{reason}={count}" for reason, count in reason_counts.items()
                    )
                )
        if not rejected.empty:
            top_rejected = rejected.sort_values(
                ["hurdle_excess", "composite_score"], ascending=[False, False]
            ).head(10)
            cols = [
                c
                for c in [
                    "canonical_key",
                    "side",
                    "mean_net_ret",
                    "directional_mean_ret",
                    "mean_support_pct",
                    "presence_freq",
                    "sign_consistency",
                    "required_hurdle",
                    "hurdle_excess",
                    "rejection_reason",
                ]
                if c in top_rejected.columns
            ]
            tprint(
                f"{stage_name} top near-miss rules:\n{top_rejected[cols].to_string(index=False)}"
            )
        tprint(
            f"{stage_name} score summary: "
            f"mean_ret_med={rejected['mean_net_ret'].median():.6f} "
            f"support_med={rejected['mean_support_pct'].median():.4f} "
            f"presence_med={rejected['presence_freq'].median():.3f} "
            f"sign_consistency_med={rejected['sign_consistency'].median():.3f}"
        )


def run_mining_stage(
    data: pd.DataFrame,
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    X: np.ndarray,
    metadata: List[FeatureMetadata],
    cfg: Dict[str, Any],
    output_dir: Path,
    stage_name: str,
    allowed_group_pairs: Sequence[Tuple[str, str]],
    slot_order: Sequence[str] = ("trigger", "location", "regime"),
    folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    mask_resolver: Optional[
        Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]
    ] = None,
    require_uplift: bool = False,
    rule_key_rewriter: Optional[
        Callable[[str], Tuple[Optional[str], Optional[str]]]
    ] = None,
    pipeline_stage_name: Optional[str] = None,
    explicit_side: Optional[str] = None,
    target_name: str = "primary_target",
    horizon: int = 0,
    primary_target_override: Optional[np.ndarray] = None,
    sample_weight_surprisal_override: Optional[np.ndarray] = None,
    nuisance_feature_arrays: Optional[Dict[str, np.ndarray]] = None,
    nuisance_feature_resolution: Optional[Dict[str, str]] = None,
    run_step: str = "full",
    step1_input_dir: Optional[Path] = None,
    candidate_registry_override: Optional[pd.DataFrame] = None,
    bounded_target: Optional[np.ndarray] = None,
    target_nan_reasons: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run a single mining stage.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with event metadata
    fwd_ret : np.ndarray
        Forward returns (raw)
    fwd_ret_norm : np.ndarray
        Legacy normalized forward return array kept for compatibility fallback
    X : np.ndarray
        Feature matrix
    metadata : List[FeatureMetadata]
        Feature metadata
    cfg : Dict[str, Any]
        Configuration dictionary
    output_dir : Path
        Output directory for this stage
    stage_name : str
        Name of the stage
    allowed_group_pairs : Sequence[Tuple[str, str]]
        Allowed interaction group pairs
    slot_order : Sequence[str]
        Order of slots in rules
    folds : Optional[List[Tuple[np.ndarray, np.ndarray]]]
        Pre-computed folds
    mask_resolver : Optional[Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]]
        Mask resolver for rule evaluation
    require_uplift : bool
        Whether to require uplift over parent context
    rule_key_rewriter : Optional[Callable[[str], Tuple[Optional[str], Optional[str]]]]
        Function to rewrite rule keys
    pipeline_stage_name : Optional[str]
        Pipeline stage name for context
    explicit_side : Optional[str]
        Explicit side ('long' or 'short')
    target_name : str
        Name of the active target for provenance tracking
    horizon : int
        Horizon in bars for provenance tracking (default: 0)

    Returns
    -------
    Dict[str, Any]
        Stage results including extracted rules and registries
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tprint(f"--- RUNNING MINING STAGE: {stage_name} ---")

    # Log target provenance
    tprint(f"Target: {target_name} | Horizon: {horizon}")

    # Use primary_target_override (triad target) if provided; fall back to the
    # legacy normalized forward return only when no explicit primary target exists.
    primary_target = (
        primary_target_override if primary_target_override is not None else fwd_ret_norm
    )
    effective_target_name = f"{target_name}_miner_effective"
    residualise_target = bool(cfg.get("residualise_target_for_miner", True))

    # Relaxed completeness check per user request.
    # We only require the target to be finite and at least ONE feature to be present.
    _feature_any_finite = np.any(np.isfinite(X), axis=1)
    _nuisance_complete = np.ones(len(data), dtype=bool)
    if residualise_target and nuisance_feature_arrays:
        for arr in nuisance_feature_arrays.values():
            _nuisance_complete &= np.isfinite(arr)
    _complete = np.isfinite(primary_target) & _feature_any_finite & _nuisance_complete

    n_before = len(data)
    n_remain = int(_complete.sum())

    if n_remain < n_before:
        tprint(
            f"{pipeline_stage_name}: upstream filter invalidating {n_before - n_remain} rows "
            f"({100.0 * (n_before - n_remain) / max(n_before, 1):.1f}%) with missing features/targets. "
            f"{n_remain} rows will be used for training/metrics."
        )
        # We set target to NaN for incomplete rows so train_fold and Scorer skip them.
        # Use copies to avoid side effects on other stages.
        primary_target = primary_target.copy()
        primary_target[~_complete] = np.nan
        fwd_ret = fwd_ret.copy()
        fwd_ret[~_complete] = np.nan
        fwd_ret_norm = fwd_ret_norm.copy()
        fwd_ret_norm[~_complete] = np.nan

    if n_remain == 0:
        tprint(
            f"WARNING: {pipeline_stage_name} has 0 valid rows after filter. Skipping stage."
        )
        return {
            "accepted_registry": pd.DataFrame(),
            "candidate_registry": pd.DataFrame(),
            "all_extracted_rules": [],
            "all_rejection_audit": [],
            "all_split_usage": [],
            "fold_quality_reports": [],
            "model_fit_reports": [],
            "feature_importance_records": [],
        }

    path_arrays = _compute_path_arrays_from_ohlc(
        data=data,
        side=explicit_side or "long",
        horizon=int(horizon),
        fallback_final_ret=fwd_ret,
    )

    # Compute ATR-normalized mfe and mae for excursion asymmetry weighting
    if "atr" in data.columns:
        atr_array = data["atr"].to_numpy(dtype=np.float32)
        mfe_atr = np.where(
            np.isfinite(path_arrays["mfe"]) & (atr_array > 1e-12),
            path_arrays["mfe"] / atr_array,
            np.nan
        ).astype(np.float32)
        mae_atr = np.where(
            np.isfinite(path_arrays["mae"]) & (atr_array > 1e-12),
            path_arrays["mae"] / atr_array,
            np.nan
        ).astype(np.float32)
    else:
        # Fallback: use raw mfe/mae if ATR not available
        mfe_atr = path_arrays["mfe"]
        mae_atr = path_arrays["mae"]

    if folds is None:
        folds = build_walk_forward_folds(
            n_samples=len(data),
            n_folds=int(cfg.get("n_folds", 5)),
            min_train_frac=float(cfg.get("cv_min_train_frac", 0.5)),
            embargo=int(cfg.get("cv_embargo", 0)),
        )
    fold_orig_sizes: Dict[int, Tuple[int, int]] = {}
    training_folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        if tr_idx.size == 0 or va_idx.size == 0 or tr_idx.max() >= va_idx.min():
            raise ValueError(f"Invalid fold {fold_id} in {stage_name}")
        fold_orig_sizes[fold_id] = (len(tr_idx), len(va_idx))
        training_folds.append(
            (
                _cap_fold_indices(tr_idx, int(cfg.get("fold_train_row_cap", 75_000))),
                _cap_fold_indices(va_idx, int(cfg.get("fold_val_row_cap", 15_000))),
            )
        )
    folds = training_folds

    fold_target_views, effective_target_oof, residualizer_records = (
        _prepare_fold_effective_target_views(
            raw_target=np.asarray(primary_target, dtype=np.float32),
            folds=folds,
            target_name=target_name,
            residualise_target=residualise_target,
            nuisance_feature_arrays=nuisance_feature_arrays,
            nuisance_feature_resolution=nuisance_feature_resolution,
            apply_target_postprocessing=bounded_target is not None,
            X=X,
            symbol_id=data["symbol"].to_numpy(),
            cfg=cfg,
            horizon=horizon,
            surprisal_bits=sample_weight_surprisal_override,
            mfe_atr=mfe_atr,
            mae_atr=mae_atr,
            side=explicit_side or "long",
        )
    )
    if residualizer_records:
        atomic_to_csv(
            pd.DataFrame(residualizer_records),
            output_dir / "fold_target_residualizer_summary.csv",
        )
    tprint(
        f"Effective miner target: {effective_target_name} | residualised={residualise_target} | nuisance_columns={json.dumps(nuisance_feature_resolution or {}, sort_keys=True)}"
    )

    model_engine = InteractionModel(
        metadata, cfg, side=explicit_side, allowed_group_pairs=allowed_group_pairs
    )
    constraint_summary = model_engine.get_constraint_summary()
    with open(output_dir / "interaction_constraint_summary.json", "w") as f:
        json.dump(constraint_summary, f, indent=2)

    stage_input_feature_inventory = pd.DataFrame(
        [
            {
                "feature_name": m.feature_name,
                "group": m.group,
                "regime_family": m.regime_family,
                "interaction_group": m.interaction_group,
            }
            for m in metadata
        ]
    )
    stage_input_feature_inventory.to_csv(
        output_dir / "stage_input_feature_inventory.csv", index=False
    )

    if candidate_registry_override is not None:
        tprint(f"{stage_name}: resuming from stored step1 outcome at {step1_input_dir}")
        candidate_registry = candidate_registry_override.copy()
        candidate_registry["preset"] = cfg.get("preset", "exploration")
        candidate_registry = atomic_to_csv(
            candidate_registry,
            output_dir / "candidate_rule_registry.csv",
            expected_columns=list(candidate_registry.columns),
        )
        atomic_to_csv(
            candidate_registry,
            output_dir / "pruned_rule_registry.csv",
            expected_columns=list(candidate_registry.columns),
        )
        assessor = MaskAssessor(metadata, cfg, mask_resolver=mask_resolver)
        assessment_df = assessor.assess_rules(
            candidate_registry,
            X,
            data,
            fwd_ret,
            fwd_ret_norm,
            folds,
            step_mode="step2",
            step1_checkpoint_dir=step1_input_dir,
            checkpoint_output_dir=output_dir,
            bounded_target=effective_target_oof,
        )
        if not assessment_df.empty:
            atomic_to_csv(assessment_df, output_dir / "final_mask_assessment_audit.csv")
            
            # Merge assessment metrics back into the candidate registry so they are preserved
            # even for rules that ultimately fail the structural soundness check
            candidate_registry = candidate_registry.merge(
                assessment_df, on="canonical_key", how="left"
            )
            accepted_registry = candidate_registry.copy()
        else:
            accepted_registry = candidate_registry.iloc[0:0].copy()
        if "selected_for_final_registry" in accepted_registry.columns:
            accepted_registry = accepted_registry[
                accepted_registry["selected_for_final_registry"].fillna(False)
            ]
        elif "is_structurally_sound" in accepted_registry.columns:
            accepted_registry = accepted_registry[
                accepted_registry["is_structurally_sound"].fillna(False)
            ]
        accepted_registry["preset"] = cfg.get("preset", "exploration")
        accepted_registry = atomic_to_csv(
            accepted_registry,
            output_dir / "accepted_rule_registry.csv",
            expected_columns=list(candidate_registry.columns),
        )
        atomic_to_csv(
            accepted_registry,
            output_dir / "final_rule_registry.csv",
            expected_columns=list(accepted_registry.columns),
        )
        return {
            "X": X,
            "metadata": metadata,
            "folds": folds,
            "all_extracted_rules": [],
            "scored_registry": pd.DataFrame(),
            "parent_context_map": {},
            "scorer_accepted": pd.DataFrame(),
            "candidate_registry": candidate_registry,
            "assessment_df": assessment_df,
            "accepted_registry": accepted_registry,
            "output_dir": output_dir,
        }

    tprint(f"Constraints Mode: {constraint_summary.get('mode', 'unknown')}")
    tprint(
        f"Group Counts: "
        + ", ".join(
            [
                f"{k}={v}"
                for k, v in constraint_summary.items()
                if k.startswith("num_") and not k.startswith("num_regime_")
            ]
        )
    )
    tprint(
        f"Regime Family Counts: "
        + ", ".join(
            [
                f"{k.replace('num_regime_', '')}={v}"
                for k, v in constraint_summary.items()
                if k.startswith("num_regime_")
            ]
        )
    )

    positive_only_groups: Tuple[str, ...] = ()
    required_positive_groups: Tuple[str, ...] = ()
    collapse_duplicate_groups: Tuple[str, ...] = ()
    if pipeline_stage_name == "stage_a_context":
        collapse_duplicate_groups = ("location",)
        if not bool(cfg.get("stage_a_relax_positive_groups", True)):
            positive_only_groups = ("location", "regime")
            required_positive_groups = ("location", "regime")
    extractor = RuleExtractor(
        metadata,
        cfg,
        slot_order=slot_order,
        positive_only_groups=positive_only_groups,
        required_positive_groups=required_positive_groups,
        collapse_duplicate_groups=collapse_duplicate_groups,
    )
    all_extracted_rules = []
    all_rejection_audit = []
    all_split_usage = []
    seeds = cfg.get("seeds", [42])

    fold_quality_reports = []
    model_fit_reports = []
    feature_importance_records = []
    oof_pred_sum = np.zeros(len(data), dtype=np.float32)
    oof_pred_count = np.zeros(len(data), dtype=np.int32)

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        # All rows are guaranteed feature-complete (filtered upstream).
        if len(tr_idx) == 0 or len(va_idx) == 0:
            tprint(
                f"Skipping fold {fold_id}: empty split (train={len(tr_idx)}, val={len(va_idx)})."
            )
            continue

        orig_tr_rows, orig_va_rows = fold_orig_sizes.get(
            fold_id, (len(tr_idx), len(va_idx))
        )

        # Determine available features per group for logging
        group_to_features = collections.defaultdict(list)
        for m in metadata:
            group_to_features[m.group].append(m.feature_name)

        tprint(
            f"Fold {fold_id}: total_rows={len(tr_idx) + len(va_idx)} "
            f"train_rows={len(tr_idx)} val_rows={len(va_idx)} "
            f"(capped from total={orig_tr_rows + orig_va_rows}, "
            f"train={orig_tr_rows}, val={orig_va_rows})"
        )

        fold_view = fold_target_views[fold_id]
        _tr_tgt = fold_view["y_tr_processed"]
        tr_target_valid = _tr_tgt[~np.isnan(_tr_tgt)]
        if len(tr_target_valid) > 0:
            tr_mean = tr_target_valid.mean()
            tr_std = tr_target_valid.std()
            tr_p1 = np.percentile(tr_target_valid, 1)
            tr_p5 = np.percentile(tr_target_valid, 5)
            tr_p50 = np.percentile(tr_target_valid, 50)
            tr_p95 = np.percentile(tr_target_valid, 95)
            tr_p99 = np.percentile(tr_target_valid, 99)

            # Check severe clipping
            clipped = np.clip(tr_target_valid, -3.0, 3.0)
            clip_diff = (
                np.abs(tr_target_valid - clipped).sum() / np.abs(tr_target_valid).sum()
            )
            if clip_diff > 0.05:
                tprint(
                    f"WARNING: Severe target clipping in Fold {fold_id} ({clip_diff:.1%} diff)"
                )

        else:
            tr_mean, tr_std, tr_p1, tr_p5, tr_p50, tr_p95, tr_p99 = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

        fold_quality_reports.append(
            {
                "fold_id": fold_id,
                "tr_rows": len(tr_idx),
                "va_rows": len(va_idx),
                "raw_target_name": target_name,
                "effective_target_name": effective_target_name,
                "target_representation": (
                    "residualised" if residualise_target else "raw"
                ),
                "target_mean": tr_mean,
                "target_std": tr_std,
                "target_p1": tr_p1,
                "target_p5": tr_p5,
                "target_p50": tr_p50,
                "target_p95": tr_p95,
                "target_p99": tr_p99,
            }
        )

        tprint(
            f"Fold {fold_id}: Target {effective_target_name} -> mean={tr_mean:.4f}, std={tr_std:.4f}, p1={tr_p1:.4f}, p5={tr_p5:.4f}, p50={tr_p50:.4f}, p95={tr_p95:.4f}, p99={tr_p99:.4f}"
        )

        # All rows in tr_idx and va_idx are already complete (pre-filtered upstream).
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr_raw = fold_view["y_tr_raw"]
        y_va_raw = fold_view["y_va_raw"]
        symbol_id_tr = data["symbol"].to_numpy()[tr_idx]
        surprisal_bits_tr = (
            None
            if sample_weight_surprisal_override is None
            else sample_weight_surprisal_override[tr_idx]
        )

        tr_finite_mask = np.isfinite(fold_view["y_tr_processed"])
        va_finite_mask = np.isfinite(fold_view["y_va_processed"])
        if not tr_finite_mask.any() or not va_finite_mask.any():
            tprint(
                f"WARNING: Skipping fold {fold_id} because it has no finite "
                f"{'training' if not tr_finite_mask.any() else 'validation'} samples "
                f"for target {target_name} @ H{horizon}."
            )
            continue

        y_tr_clip = fold_view["y_tr_processed"]
        y_va_proc = fold_view["y_va_processed"]
        if bounded_target is not None:
            from extreme_price_movements.triad_targets import summarize_processed_target
            summarize_processed_target(
                f"{effective_target_name}_H{horizon}_fold{fold_id}_train", y_tr_clip
            )
            summarize_processed_target(
                f"{effective_target_name}_H{horizon}_fold{fold_id}_val", y_va_proc
            )
        if residualise_target and residualizer_records:
            residualizer_row = residualizer_records[fold_id]
            tprint(
                f"{stage_name} fold {fold_id}: target residualised fit_n={int(residualizer_row.get('fit_sample_count', 0))} train_valid={int(residualizer_row.get('train_effective_valid_count', 0))} val_valid={int(residualizer_row.get('val_effective_valid_count', 0))}"
            )

        for seed in seeds:
            # Use quantile regression for all targets (triad targets work with quantile regression)
            target_type = "quantile"
            fold_fit_start = time.perf_counter()
            tprint(
                f"{stage_name} fold {fold_id} seed {seed}: train start "
                f"rows_total={len(tr_idx) + len(va_idx)} train_rows={len(tr_idx)} val_rows={len(va_idx)} "
                f"features={X_tr.shape[1]} target_type={target_type} horizon={horizon}"
            )
            model, fit_meta = model_engine.train_fold(
                X_tr,
                y_tr_clip,
                symbol_id_tr,
                surprisal_bits_tr,
                X_va,
                y_va_proc,
                fold_id,
                seed,
                target_type=target_type,
                horizon=horizon,
                mfe_atr_tr=mfe_atr[tr_idx] if mfe_atr is not None else None,
                mae_atr_tr=mae_atr[tr_idx] if mae_atr is not None else None,
                side=explicit_side or "long",
            )
            fold_fit_elapsed = time.perf_counter() - fold_fit_start
            try:
                va_pred = model.predict(X_va)
                va_pred = np.asarray(va_pred, dtype=np.float32)
                if va_pred.ndim > 1:
                    va_pred = va_pred.ravel()
                oof_pred_sum[va_idx] += va_pred
                oof_pred_count[va_idx] += 1
            except Exception as pred_exc:
                tprint(
                    f"WARNING: could not compute OOF predictions for fold {fold_id} seed {seed}: {pred_exc}"
                )
            tprint(
                f"{stage_name} fold {fold_id} seed {seed}: "
                f"train_samples={fit_meta['train_samples']} val_samples={fit_meta['val_samples']} "
                f"best_iteration={fit_meta['best_iteration']} best_val_metric={fit_meta['best_val_metric']:.5f} "
                f"target_type={target_type} elapsed={fold_fit_elapsed:.2f}s "
                f"min_data_in_leaf={fit_meta['params']['min_data_in_leaf']} "
                f"alpha={fit_meta['params'].get('alpha', np.nan)} "
                f"min_gain_to_split={fit_meta['params'].get('min_gain_to_split', np.nan)}"
            )

            model_fit_reports.append(
                {
                    "fold_id": fold_id,
                    "seed": seed,
                    "best_iteration": fit_meta["best_iteration"],
                    "best_val_metric": fit_meta["best_val_metric"],
                    "max_depth": fit_meta["params"]["max_depth"],
                    "num_leaves": fit_meta["params"]["num_leaves"],
                    "min_data_in_leaf": fit_meta["params"]["min_data_in_leaf"],
                    "objective": fit_meta["params"]["objective"],
                    "alpha": fit_meta["params"].get("alpha", 0.5),
                    "metric": fit_meta["params"]["metric"],
                    "target_type": target_type,
                    "target_name": target_name,
                    "horizon": horizon,
                }
            )

            # Print hyperparams only on first fold/seed
            if fold_id == 0 and seed == seeds[0]:
                tprint(
                    f"Model Hyperparams: max_depth={fit_meta['params']['max_depth']}, num_leaves={fit_meta['params']['num_leaves']}, min_data_in_leaf={fit_meta['params']['min_data_in_leaf']}, objective={fit_meta['params']['objective']}, metric={fit_meta['params']['metric']}, n_estimators={fit_meta['params']['n_estimators']}, seeds={len(seeds)}, folds={len(folds)}"
                )

            # Extract Feature Importance
            gain_imp = fit_meta["feature_importances_gain"]
            split_imp = fit_meta["feature_importances_split"]

            fi_records = []
            for m in metadata:
                idx = m.feature_index
                gain = gain_imp[idx] if idx < len(gain_imp) else 0.0
                split = split_imp[idx] if idx < len(split_imp) else 0.0
                if gain > 0 or split > 0:
                    fi_records.append(
                        {
                            "fold_id": fold_id,
                            "seed": seed,
                            "feature_name": m.feature_name,
                            "group": m.group,
                            "regime_family": m.regime_family,
                            "source_family": m.source_family,
                            "gain": gain,
                            "split": split,
                        }
                    )
                    feature_importance_records.append(fi_records[-1])

            if fi_records:
                fi_df = pd.DataFrame(fi_records)
                total_gain = float(fi_df["gain"].sum())
                top_gain = fi_df.sort_values("gain", ascending=False).head(5)
                top10_gain = fi_df.sort_values("gain", ascending=False).head(10)
                top5_gain_share = (
                    float(top_gain["gain"].sum() / total_gain) if total_gain > 0.0 else np.nan
                )
                top10_gain_share = (
                    float(top10_gain["gain"].sum() / total_gain) if total_gain > 0.0 else np.nan
                )
                tprint("Top 5 features by gain:")
                for _, row in top_gain.iterrows():
                    gain_pct = (
                        (100.0 * float(row["gain"]) / total_gain) if total_gain > 0.0 else np.nan
                    )
                    tprint(
                        f"  - {row['feature_name']}: {row['gain']:.2f} ({gain_pct:.2f}% of total gain)"
                    )
                tprint(
                    "Feature gain concentration: "
                    f"top5={100.0 * top5_gain_share:.2f}% top10={100.0 * top10_gain_share:.2f}%"
                )

                top_fam = (
                    fi_df.groupby("regime_family")["split"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                )
                tprint("Top 5 regime families by split count:")
                for fam, count in top_fam.items():
                    if pd.notna(fam):
                        tprint(f"  - {fam}: {count}")

                top_loc_fam = (
                    fi_df[fi_df["group"] == "location"]
                    .groupby("source_family")["split"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                )
                tprint("Top 5 location families by split count:")
                for fam, count in top_loc_fam.items():
                    if pd.notna(fam):
                        tprint(f"  - {fam}: {count}")
            split_usage_df = collect_split_usage_from_model(
                model, metadata, fold_id, seed
            )
            if not split_usage_df.empty:
                all_split_usage.append(split_usage_df)
                group_summary = summarize_fold_feature_usage(split_usage_df)
                if not group_summary.empty:
                    summary_text = ", ".join(
                        f"{row.group}={int(row.used_feature_count)}f/{int(row.split_count)}s"
                        for row in group_summary.itertuples(index=False)
                    )
                    tprint(
                        f"{stage_name} fold {fold_id} seed {seed} feature usage by group: {summary_text}"
                    )
            extract_start = time.perf_counter()
            tprint(
                f"{stage_name} fold {fold_id} seed {seed}: extraction start "
                f"best_iteration={fit_meta['best_iteration']} "
                f"used_features={len(fi_records)} total_gain={float(np.sum(gain_imp)):.2f} "
                f"total_splits={int(np.sum(split_imp))}"
            )
            fold_rules = extractor.extract_rules(
                model,
                f"{stage_name}_model",
                fold_id,
                seed,
                target_name=target_name,
                horizon=horizon,
            )
            extract_elapsed = time.perf_counter() - extract_start
            rejected_paths = len(extractor.rejection_audit)
            tprint(
                f"{stage_name} fold {fold_id} seed {seed}: extraction done "
                f"valid_rules={len(fold_rules)} rejected_paths={rejected_paths} "
                f"total_leaf_paths={extractor.total_leaf_paths} "
                f"non_empty_paths={extractor.total_non_empty_paths} "
                f"elapsed={extract_elapsed:.2f}s"
            )
            all_extracted_rules.extend(fold_rules)
            if extractor.rejection_audit:
                all_rejection_audit.extend(extractor.rejection_audit)

    parent_context_map: Dict[str, str] = {}

    if all_extracted_rules:
        canonicalize_start = time.perf_counter()
        dropped_missing_key = 0
        for rule in all_extracted_rules:
            if not rule.canonical_key:
                canonical_key = extractor._build_canonical_key(rule.conditions)
                if canonical_key:
                    rule.canonical_key = canonical_key
                else:
                    dropped_missing_key += 1
        if dropped_missing_key > 0:
            all_extracted_rules = [r for r in all_extracted_rules if r.canonical_key]
        tprint(
            f"{stage_name}: canonical key materialization "
            f"rules={len(all_extracted_rules)} dropped={dropped_missing_key} "
            f"elapsed={time.perf_counter() - canonicalize_start:.2f}s"
        )

    if rule_key_rewriter is not None:
        rewritten_rules: List[ExtractedRule] = []
        for rule in all_extracted_rules:
            rewritten_key, parent_context_key = rule_key_rewriter(rule.canonical_key)
            if rewritten_key is None:
                continue
            rule.canonical_key = rewritten_key
            rewritten_rules.append(rule)
            if parent_context_key:
                parent_context_map[rewritten_key] = parent_context_key
        all_extracted_rules = rewritten_rules

    require_groups = cfg.get("require_groups", [])
    if require_groups:
        filtered_rules = []
        for rule in all_extracted_rules:
            # Check if all required groups are present in the rule conditions
            rule_groups = set(c.group for c in rule.conditions)
            # Some conditions might be context and we don't have conditions for them inside rule.conditions directly?
            # Actually, `rule.conditions` should have `group="context"` if we used context features.
            if all(rg in rule_groups for rg in require_groups):
                filtered_rules.append(rule)
        all_extracted_rules = filtered_rules

    if fold_quality_reports:
        fq_df = pd.DataFrame(fold_quality_reports)
        # Simplified columns after removing per-fold filtering
        data_cols = ["fold_id", "tr_rows", "va_rows"]
        tgt_cols = [
            "fold_id",
            "target_mean",
            "target_std",
            "target_p1",
            "target_p50",
            "target_p99",
        ]

        fq_df[data_cols].to_csv(
            output_dir / "fold_data_quality_report.csv", index=False
        )
        # Check if tgt_cols exist (they might be missing if all targets were NaN in a fold)
        actual_tgt_cols = [c for c in tgt_cols if c in fq_df.columns]
        fq_df[actual_tgt_cols].to_csv(
            output_dir / "fold_target_distribution_report.csv", index=False
        )

    if model_fit_reports:
        pd.DataFrame(model_fit_reports).to_csv(
            output_dir / "fold_model_fit_summary.csv", index=False
        )

    if feature_importance_records:
        fi_df = pd.DataFrame(feature_importance_records)
        fi_df.to_csv(output_dir / "fold_feature_importance_by_feature.csv", index=False)

        # Cache groupby operations
        fi_by_group_cache = fi_df.groupby(["fold_id", "seed", "group"])
        fi_by_family_cache = fi_df.groupby(["fold_id", "seed", "regime_family"])

        fi_by_group = fi_by_group_cache[["gain", "split"]].sum().reset_index()
        fi_by_group.to_csv(
            output_dir / "fold_feature_importance_by_group.csv", index=False
        )

        fi_by_family = fi_by_family_cache[["gain", "split"]].sum().reset_index()
        fi_by_family.to_csv(
            output_dir / "fold_feature_importance_by_regime_family.csv", index=False
        )

    if all_rejection_audit:
        audit_df = pd.DataFrame(all_rejection_audit)
        audit_df.to_csv(output_dir / "invalid_path_audit.csv", index=False)
        summarize_feature_usage(
            audit_df, output_dir / "invalid_path_reason_summary.csv", ["reason"]
        )

    if all_extracted_rules:
        shape_records = []
        family_combos = collections.defaultdict(int)
        for r in all_extracted_rules:
            # arity, structural depth, groups used, regime families used, location families used
            arity = display_arity_for_key(r.canonical_key)
            depth = structural_depth_for_key(r.canonical_key)
            groups_used = tuple(sorted(set(c.group for c in r.conditions)))

            regime_families = []
            location_families = []
            for c in r.conditions:
                if c.group == "regime":
                    fam = (
                        m.regime_family
                        if (m := metadata[c.feature_index])
                        else "unknown"
                    )
                    if fam:
                        regime_families.append(fam)
                elif c.group == "location":
                    fam = (
                        m.source_family
                        if (m := metadata[c.feature_index])
                        else "unknown"
                    )
                    if fam:
                        location_families.append(fam)

            regime_families_tuple = tuple(sorted(set(regime_families)))
            location_families_tuple = tuple(sorted(set(location_families)))
            # Combine regime and location families for the combo
            all_families_tuple = tuple(sorted(set(regime_families + location_families)))
            family_combos[all_families_tuple] += 1

            shape_records.append(
                {
                    "canonical_key": r.canonical_key,
                    "display_arity": arity,
                    "structural_depth": depth,
                    "groups_used": "|".join(groups_used),
                    "regime_families_used": "|".join(regime_families_tuple),
                    "location_families_used": "|".join(location_families_tuple),
                    "all_families_used": "|".join(all_families_tuple),
                }
            )

        pd.DataFrame(shape_records).to_csv(
            output_dir / "extracted_rule_shape_summary.csv", index=False
        )

        family_df = pd.DataFrame(
            [
                {"families_combo": "|".join(k) if k else "none", "count": v}
                for k, v in family_combos.items()
            ]
        )
        family_df.sort_values("count", ascending=False).to_csv(
            output_dir / "extracted_rule_family_combo_summary.csv", index=False
        )

        tprint("Top valid family combinations (regime + location):")
        for _, row in (
            family_df.sort_values("count", ascending=False).head(5).iterrows()
        ):
            tprint(f"  - {row['families_combo']}: {row['count']}")

    if all_split_usage:
        split_usage_all = pd.concat(all_split_usage, ignore_index=True)
        split_usage_all.to_csv(
            output_dir / "model_split_usage_detailed.csv", index=False
        )
        summarize_feature_usage(
            split_usage_all,
            output_dir / "model_split_usage_by_feature.csv",
            ["feature_name", "group"],
        )
        summarize_feature_usage(
            split_usage_all, output_dir / "model_split_usage_by_group.csv", ["group"]
        )
        summarize_fold_feature_usage(split_usage_all).to_csv(
            output_dir / "model_split_usage_by_fold_group.csv", index=False
        )
    else:
        split_usage_all = pd.DataFrame()

    fold_health_summary: Dict[str, Any] = {
        "total_fold_count": len(model_fit_reports),
        "healthy_fold_count": len(model_fit_reports),
        "healthy_fold_ratio": 1.0 if model_fit_reports else 0.0,
    }
    if model_fit_reports:
        fit_df = pd.DataFrame(model_fit_reports)
        if not split_usage_all.empty:
            split_totals = (
                split_usage_all.groupby(["fold_id", "seed"], as_index=False)[
                    "split_count"
                ]
                .sum()
                .rename(columns={"split_count": "total_splits"})
            )
        else:
            split_totals = pd.DataFrame(columns=["fold_id", "seed", "total_splits"])

        if feature_importance_records:
            gain_totals = (
                pd.DataFrame(feature_importance_records)
                .groupby(["fold_id", "seed"], as_index=False)["gain"]
                .sum()
                .rename(columns={"gain": "total_gain"})
            )
        else:
            gain_totals = pd.DataFrame(columns=["fold_id", "seed", "total_gain"])

        fold_health_df = fit_df.merge(split_totals, on=["fold_id", "seed"], how="left")
        fold_health_df = fold_health_df.merge(
            gain_totals, on=["fold_id", "seed"], how="left"
        )
        fold_health_df["total_splits"] = (
            pd.to_numeric(fold_health_df["total_splits"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        fold_health_df["total_gain"] = pd.to_numeric(
            fold_health_df["total_gain"], errors="coerce"
        ).fillna(0.0)
        min_healthy_best_iteration = int(cfg.get("min_healthy_fold_best_iteration", 2))
        min_healthy_total_splits = int(cfg.get("min_healthy_fold_total_splits", 20))
        min_healthy_total_gain = float(cfg.get("min_healthy_fold_total_gain", 10.0))
        fold_health_df["is_healthy_fold"] = (
            (
                pd.to_numeric(fold_health_df["best_iteration"], errors="coerce").fillna(
                    0
                )
                >= min_healthy_best_iteration
            )
            & (fold_health_df["total_splits"] >= min_healthy_total_splits)
            & (fold_health_df["total_gain"] >= min_healthy_total_gain)
        )
        fold_health_df["health_reason"] = np.where(
            fold_health_df["is_healthy_fold"],
            "healthy",
            "low_iterations_or_gain",
        )
        atomic_to_csv(fold_health_df, output_dir / "fold_health_summary.csv")

        healthy_fold_count = int(fold_health_df["is_healthy_fold"].sum())
        total_fold_count = int(len(fold_health_df))
        healthy_fold_ratio = float(healthy_fold_count) / max(total_fold_count, 1)
        fold_health_summary = {
            "total_fold_count": total_fold_count,
            "healthy_fold_count": healthy_fold_count,
            "healthy_fold_ratio": healthy_fold_ratio,
            "min_healthy_best_iteration": min_healthy_best_iteration,
            "min_healthy_total_splits": min_healthy_total_splits,
            "min_healthy_total_gain": min_healthy_total_gain,
        }
        tprint(
            "Fold health summary: "
            f"healthy={healthy_fold_count}/{total_fold_count} "
            f"ratio={healthy_fold_ratio:.3f} "
            f"(best_iteration>={min_healthy_best_iteration}, "
            f"total_splits>={min_healthy_total_splits}, "
            f"total_gain>={min_healthy_total_gain:.1f})"
        )

    rule_usage_df = collect_extracted_rule_feature_usage(all_extracted_rules, metadata)
    rule_usage_df.to_csv(
        output_dir / "extracted_rule_feature_usage_detailed.csv", index=False
    )
    summarize_feature_usage(
        rule_usage_df,
        output_dir / "extracted_rule_feature_usage_by_feature.csv",
        ["feature_name", "group"],
    )
    summarize_feature_usage(
        rule_usage_df,
        output_dir / "extracted_rule_feature_usage_by_group.csv",
        ["group"],
    )

    scorer = RuleScorer(metadata, cfg, mask_resolver=mask_resolver)
    unique_keys = sorted({rule.canonical_key for rule in all_extracted_rules})
    discovery_count_map = collections.Counter(
        rule.canonical_key for rule in all_extracted_rules
    )
    n_instances_map = discovery_count_map.copy()
    pipeline_stage_map = {
        key: (pipeline_stage_name or stage_name) for key in unique_keys
    }
    require_uplift_keys = unique_keys if require_uplift else []
    side_map = {key: explicit_side for key in unique_keys} if explicit_side else None
    oof_predictions = np.full(len(data), np.nan, dtype=np.float32)
    valid_oof = oof_pred_count > 0
    oof_predictions[valid_oof] = oof_pred_sum[valid_oof] / oof_pred_count[valid_oof]
    scorer_start = time.perf_counter()
    tprint(
        f"{stage_name}: scorer start "
        f"keys={len(unique_keys)} extracted_rules={len(all_extracted_rules)} "
        f"oof_valid_rows={int(valid_oof.sum())} folds={len(folds)}"
    )
    scored_registry, full_scorer_audit = scorer.score_registry_oos(
        keys=unique_keys,
        fwd_ret=fwd_ret,
        folds=folds,
        resolver=mask_resolver,
        parent_context_map=parent_context_map,
        require_uplift_keys=require_uplift_keys,
        discovery_count_map=discovery_count_map,
        n_instances_map=n_instances_map,
        pipeline_stage_map=pipeline_stage_map,
        side_map=side_map,
        bounded_target=effective_target_oof,
        predictions=oof_predictions,
        path_mfe=path_arrays["mfe"],
        path_mae=path_arrays["mae"],
        path_final_ret=path_arrays["final_ret"],
        path_time_to_mfe=path_arrays["time_to_mfe"],
        path_time_to_mae=path_arrays["time_to_mae"],
        path_length=path_arrays["path_length"],
    )
    scorer_elapsed = time.perf_counter() - scorer_start
    accepted_scored = (
        int(scored_registry["accepted"].sum()) if not scored_registry.empty else 0
    )
    tprint(
        f"{stage_name}: scorer done "
        f"keys={len(unique_keys)} scored={len(scored_registry)} accepted={accepted_scored} "
        f"audit_rows={len(full_scorer_audit)} elapsed={scorer_elapsed:.2f}s"
    )
    scored_registry["preset"] = cfg.get("preset", "exploration")
    atomic_to_csv(
        full_scorer_audit, output_dir / "fold_level_rule_aggregation_audit.csv"
    )
    scored_registry = atomic_to_csv(
        scored_registry,
        output_dir / "scored_rule_registry_full.csv",
        expected_columns=SCORER_REGISTRY_COLUMNS + ["preset"],
    )

    # Handle empty registry
    if scored_registry.empty:
        tprint(
            "WARNING: No rules scored. Skipping consolidation and returning empty results."
        )
        return {
            "scored_registry": scored_registry,
            "scorer_accepted": pd.DataFrame(),
            "accepted_registry": pd.DataFrame(),
            "final_registry": pd.DataFrame(),
            "candidate_registry": pd.DataFrame(),
            "assessment_df": pd.DataFrame(),
        }

    # Save scorer diagnostics
    rejection_reasons = collections.Counter(
        reason.strip()
        for reasons in scored_registry[~scored_registry["accepted"]][
            "rejection_reason"
        ].dropna()
        for reason in reasons.split("|")
        if reason.strip()
    )
    atomic_to_csv(
        pd.DataFrame(
            list(rejection_reasons.items()), columns=["rejection_reason", "count"]
        ),
        output_dir / "scorer_rejection_reason_summary.csv",
        expected_columns=["rejection_reason", "count"],
    )

    scorer_accepted = scored_registry[scored_registry["accepted"]]
    rule_importance_df = build_rule_model_importance_scores(all_extracted_rules)
    if not rule_importance_df.empty and not scorer_accepted.empty:
        scorer_accepted = scorer_accepted.merge(
            rule_importance_df, on="canonical_key", how="left"
        )
        importance_col = "rule_model_importance_score"
        finite_importance = pd.to_numeric(
            scorer_accepted[importance_col], errors="coerce"
        )
        finite_importance = finite_importance[np.isfinite(finite_importance)]
        if len(finite_importance) >= 2:
            cutoff = float(np.nanquantile(finite_importance, 0.30))
            before_count = len(scorer_accepted)
            scorer_accepted = scorer_accepted[
                pd.to_numeric(scorer_accepted[importance_col], errors="coerce").fillna(
                    -np.inf
                )
                > cutoff
            ]
            tprint(
                f"Model-importance pre-pruner cut: removed bottom 30% by "
                f"{importance_col} ({before_count} -> {len(scorer_accepted)}, cutoff={cutoff:.6f})"
            )

    pruner = IndependentRulePruner(cfg)
    candidate_registry = pruner.prune(scorer_accepted)
    candidate_registry["preset"] = cfg.get("preset", "exploration")
    candidate_registry = atomic_to_csv(
        candidate_registry,
        output_dir / "candidate_rule_registry.csv",
        expected_columns=list(scorer_accepted.columns) + ["preset"],
    )
    atomic_to_csv(
        candidate_registry,
        output_dir / "pruned_rule_registry.csv",
        expected_columns=list(candidate_registry.columns),
    )

    if hasattr(pruner, "gate_summary"):
        atomic_to_csv(
            pd.DataFrame([pruner.gate_summary]),
            output_dir / "pruner_gate_summary.csv",
        )

    if not candidate_registry.empty:
        arity_counts = candidate_registry["display_arity"].value_counts().reset_index()
        arity_counts.columns = ["display_arity", "count"]
        atomic_to_csv(arity_counts, output_dir / "pruner_arity_summary.csv")
        tprint("Accepted by Arity (Pruner):")
        for row in arity_counts.itertuples(index=False, name=None):
            tprint(f"  - {int(row[0])}: {int(row[1])}")

    assessor = MaskAssessor(metadata, cfg, mask_resolver=mask_resolver)
    assessment_df = assessor.assess_rules(
        candidate_registry,
        X,
        data,
        fwd_ret,
        fwd_ret_norm,
        folds,
        fold_health_summary=fold_health_summary,
        step_mode=run_step,
        step1_checkpoint_dir=step1_input_dir,
        checkpoint_output_dir=output_dir,
        bounded_target=effective_target_oof,
    )
    if not assessment_df.empty:
        atomic_to_csv(assessment_df, output_dir / "final_mask_assessment_audit.csv")

        # Clean merge: verify uniqueness of canonical_key in both sides
        if candidate_registry["canonical_key"].duplicated().any():
            raise ValueError("Duplicate canonical_key found in candidate_registry before final assessment merge")
        if assessment_df["canonical_key"].duplicated().any():
            raise ValueError("Duplicate canonical_key found in assessment_df before final assessment merge")

        # Select explicit canonical columns to avoid _x/_y leakage
        overlap_cols = list(set(candidate_registry.columns).intersection(assessment_df.columns))
        overlap_cols.remove("canonical_key")

        # Drop overlapping columns from candidate registry to prefer the final assessment versions
        candidate_registry_clean = candidate_registry.drop(columns=overlap_cols)

        accepted_registry = candidate_registry_clean.merge(
            assessment_df, on="canonical_key", how="left", validate="1:1"
        )

        if hasattr(assessor, "rejection_summary") and assessor.rejection_summary:
            atomic_to_csv(
                pd.DataFrame(
                    list(assessor.rejection_summary.items()),
                    columns=["reason", "count"],
                ),
                output_dir / "mask_assessment_rejection_summary.csv",
                expected_columns=["reason", "count"],
            )
    else:
        accepted_registry = (
            candidate_registry.iloc[0:0].copy()
            if run_step == "step1"
            else candidate_registry
        )

    if "is_structurally_sound" in accepted_registry.columns:
        accepted_registry = accepted_registry[
            accepted_registry["is_structurally_sound"].fillna(False)
        ]

    if "preset" not in accepted_registry.columns:
        accepted_registry["preset"] = cfg.get("preset", "exploration")

    # Ensure canonical schema by dropping duplicate columns before export
    accepted_registry = accepted_registry.loc[:, ~accepted_registry.columns.duplicated()]

    # Pre-export integrity pass
    _cols = accepted_registry.columns
    if any(c.endswith("_x") or c.endswith("_y") for c in _cols):
        bad_cols = [c for c in _cols if c.endswith("_x") or c.endswith("_y")]
        raise ValueError(f"Merge suffixes _x or _y detected in final registry: {bad_cols}")

    if "support_pct" in accepted_registry.columns:
        invalid_support = accepted_registry[~accepted_registry["support_pct"].between(0.0, 1.0, inclusive="both")]
        if not invalid_support.empty:
            raise ValueError(f"Found {len(invalid_support)} rows with support_pct out of [0, 1] bounds")

    for auc_col in ["mask_roc_auc", "mask_pr_auc", "global_roc_auc", "global_pr_auc"]:
        if auc_col in accepted_registry.columns:
            invalid_auc = accepted_registry[
                (~accepted_registry[auc_col].isna()) &
                (~accepted_registry[auc_col].between(0.0, 1.0, inclusive="both"))
            ]
            if not invalid_auc.empty:
                raise ValueError(f"Found {len(invalid_auc)} rows with {auc_col} out of [0, 1] bounds")

    accepted_registry = atomic_to_csv(
        accepted_registry,
        output_dir / "accepted_rule_registry.csv",
        expected_columns=list(candidate_registry.columns),
    )
    atomic_to_csv(
        accepted_registry,
        output_dir / "final_rule_registry.csv",
        expected_columns=list(accepted_registry.columns),
    )

    final_usage_df = collect_registry_feature_usage(accepted_registry, metadata)
    export_coverage_sanity_report(
        metadata, split_usage_all, rule_usage_df, final_usage_df, output_dir
    )

    return {
        "X": X,
        "metadata": metadata,
        "folds": folds,
        "all_extracted_rules": all_extracted_rules,
        "scored_registry": scored_registry,
        "parent_context_map": parent_context_map,
        "scorer_accepted": scorer_accepted,
        "candidate_registry": candidate_registry,
        "assessment_df": assessment_df,
        "accepted_registry": accepted_registry,
        "output_dir": output_dir,
    }


def _summarize_target_distribution(name: str, arr: np.ndarray, horizon: Optional[int] = None, side: Optional[str] = None):
    """
    Helper to log distribution stats for targets, useful for debugging downstream bounded range issues.
    """
    finite = np.isfinite(arr)
    frac_finite = finite.mean() if len(arr) > 0 else 0.0
    prefix = f"Target stats {name}"
    if horizon is not None:
        prefix += f" @ H{horizon}"
    if side is not None:
        prefix += f" [{side}]"

    if frac_finite == 0.0:
        tprint(f"{prefix}: frac_finite=0.0000 | No finite rows to summarize.")
        return

    valid_arr = arr[finite]
    p1 = np.percentile(valid_arr, 1)
    p5 = np.percentile(valid_arr, 5)
    p50 = np.median(valid_arr)
    p95 = np.percentile(valid_arr, 95)
    p99 = np.percentile(valid_arr, 99)
    std = np.std(valid_arr)

    tprint(
        f"{prefix}: frac_finite={frac_finite:.4f} median={p50:.4f} std={std:.4f} "
        f"p1={p1:.4f} p5={p5:.4f} p95={p95:.4f} p99={p99:.4f}"
    )


def _resolve_requested_assessor_source_names(
    cfg: Dict[str, Any], available_names: Set[str]
) -> List[str]:
    ordered_names: List[str] = []
    seen: Set[str] = set()

    for name in TEST_FEATURE_KEYS:
        if name in available_names and name not in seen:
            ordered_names.append(str(name))
            seen.add(str(name))

    requested_columns = tuple(
        cfg.get(
            "miner_target_residualization_columns",
            MINER_TARGET_RESIDUALIZATION_COLUMNS,
        )
    )
    for requested_name in requested_columns:
        candidates = MINER_TARGET_RESIDUALIZATION_ALIAS_MAP.get(
            str(requested_name), (str(requested_name),)
        )
        for candidate in candidates:
            candidate = str(candidate)
            if candidate in available_names and candidate not in seen:
                ordered_names.append(candidate)
                seen.add(candidate)
                break

    return ordered_names


def build_requested_assessor_feature_matrix(
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, List[FeatureMetadata], List[str]]:
    available_names = {str(name) for name in feature_dict.keys()}
    selected_names = _resolve_requested_assessor_source_names(cfg, available_names)

    if not selected_names:
        return np.empty((0, 0), dtype=np.float32), [], []

    cols: List[np.ndarray] = []
    metadata: List[FeatureMetadata] = []
    for idx, name in enumerate(selected_names):
        arr = np.asarray(feature_dict[name], dtype=np.float32)
        cols.append(arr)
        group = "location" if name in CONTINUOUS_LOCATION_COLS else "regime"
        source_family = (
            LOC_CONTINUOUS_FAMILY_MAP.get(name, "location")
            if group == "location"
            else (name.split("_")[0] if "_" in name else "regime")
        )
        metadata.append(
            FeatureMetadata(
                feature_name=name,
                feature_index=idx,
                group=group,
                source_name=name,
                source_family=source_family,
                source_type="continuous",
                description=f"Requested assessor raw feature {name}",
                regime_family=source_family if group == "regime" else None,
            )
        )

    X = np.column_stack(cols).astype(np.float32, copy=False)
    return X, metadata, selected_names


def build_rule_resolver_feature_matrix(
    feature_dict: Dict[str, np.ndarray],
    canonical_keys: Sequence[str],
) -> Tuple[np.ndarray, List[FeatureMetadata], List[str], List[str]]:
    available_names = {str(name) for name in feature_dict.keys()}
    requested_names: List[str] = []
    seen: Set[str] = set()
    unresolved_names: List[str] = []

    for canonical_key in canonical_keys:
        for name in extract_feature_names_from_key(str(canonical_key)):
            feature_name = str(name)
            if feature_name in available_names:
                if feature_name not in seen:
                    requested_names.append(feature_name)
                    seen.add(feature_name)
            elif feature_name not in unresolved_names:
                unresolved_names.append(feature_name)

    if not requested_names:
        return np.empty((0, 0), dtype=np.float32), [], [], unresolved_names

    cols: List[np.ndarray] = []
    metadata: List[FeatureMetadata] = []
    for idx, name in enumerate(requested_names):
        arr = np.asarray(feature_dict[name], dtype=np.float32)
        cols.append(arr)
        group = "location" if name in CONTINUOUS_LOCATION_COLS else "regime"
        source_family = (
            LOC_CONTINUOUS_FAMILY_MAP.get(name, "location")
            if group == "location"
            else (name.split("_")[0] if "_" in name else "regime")
        )
        metadata.append(
            FeatureMetadata(
                feature_name=name,
                feature_index=idx,
                group=group,
                source_name=name,
                source_family=source_family,
                source_type="continuous",
                description=f"Rule resolver raw feature {name}",
                regime_family=source_family if group == "regime" else None,
            )
        )

    X = np.column_stack(cols).astype(np.float32, copy=False)
    return X, metadata, requested_names, unresolved_names

def run_side_pipeline(
    side: str,
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    cfg: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    root_output_dir: Path,
    target_name: str = "primary_target",
    horizon: int = 0,
    bounded_target: Optional[np.ndarray] = None,
    bounded_target_surprisal: Optional[np.ndarray] = None,
    target_nan_reasons: Optional[np.ndarray] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the full side pipeline (long or short).

    Parameters
    ----------
    side : str
        'long' or 'short'
    data : pd.DataFrame
        DataFrame with event metadata
    feature_dict : Dict[str, np.ndarray]
        Dictionary of feature arrays
    fwd_ret : np.ndarray
        Forward returns (raw)
    fwd_ret_norm : np.ndarray
        Legacy normalized forward return array kept for compatibility fallback
    cfg : Dict[str, Any]
        Configuration dictionary
    folds : List[Tuple[np.ndarray, np.ndarray]]
        Pre-computed folds
    root_output_dir : Path
        Root output directory
    target_name : str
        Name of the active target for provenance tracking
    horizon : int
        Horizon in bars for provenance tracking (default: 0)

    Returns
    -------
    Dict[str, pd.DataFrame]
        Pipeline results including stage registries
    """
    tprint(f"--- RUNNING PIPELINE FOR SIDE: {side.upper()} ---")

    # Log target provenance
    tprint(f"Target: {target_name} | Horizon: {horizon}")

    side_fwd_ret = fwd_ret if side == "long" else -fwd_ret
    side_fwd_ret_norm = fwd_ret_norm if side == "long" else -fwd_ret_norm
    residualise_target = bool(cfg.get("residualise_target_for_miner", True))
    nuisance_feature_resolution: Dict[str, str] = {}
    nuisance_feature_arrays: Dict[str, np.ndarray] = {}
    nuisance_valid = np.ones(len(data), dtype=bool)
    if residualise_target:
        nuisance_feature_resolution, nuisance_feature_arrays = (
            _resolve_miner_nuisance_feature_arrays(feature_dict, cfg)
        )
        for arr in nuisance_feature_arrays.values():
            nuisance_valid &= np.isfinite(arr)
    effective_target_name = (
        f"{target_name}_miner_effective" if residualise_target else target_name
    )
    tprint(
        f"Miner target mode: raw={target_name} effective={effective_target_name} residualised={residualise_target}"
    )
    if residualise_target:
        tprint(
            "Residualiser nuisance columns: "
            + json.dumps(nuisance_feature_resolution, sort_keys=True)
        )
    tprint(
        f"Miner nuisance drop flags: drop_nuisance={bool(cfg.get('drop_nuisance_features_from_miner', True))} drop_location={bool(cfg.get('drop_location_nuisance_features_from_miner', False))} drop_continuous_parents={bool(cfg.get('drop_continuous_nuisance_parents_from_miner', True))}"
    )

    side_output_dir = root_output_dir / side
    side_output_dir.mkdir(parents=True, exist_ok=True)

    side_input_rows = int(len(data))
    side_input_symbols = (
        int(data["symbol"].nunique()) if "symbol" in data.columns else np.nan
    )
    atr_frac_for_targets = None
    if "atr" in data.columns and "close" in data.columns:
        atr_frac_for_targets = data["atr"].to_numpy(dtype=np.float32) / np.maximum(
            np.abs(data["close"].to_numpy(dtype=np.float32)), 1e-12
        )

    # --- STAGE A: CONTEXT MINING ---
    tprint(f"STAGE A: Context Mining (Regime x Location) [{side}]")
    fp_a = FeatureProcessor()
    X_a, metadata_a, audits_a = fp_a.prepare_features(
        feature_dict,
        data["timestamp"].to_numpy(),
        data["symbol"].to_numpy(),
        cfg,
        active_groups=("regime", "location"),
    )
    X_assessor, metadata_assessor, assessor_feature_names = (
        build_requested_assessor_feature_matrix(feature_dict, cfg)
    )
    tprint(
        "Assessor raw feature matrix: "
        f"requested_loaded={len(assessor_feature_names)} shape={X_assessor.shape}"
    )
    stage_a_output_dir = side_output_dir / "stage_a_context"
    stage_a_output_dir.mkdir(parents=True, exist_ok=True)

    for k, v in audits_a.items():
        if not v.empty:
            v.to_csv(stage_a_output_dir / f"{k}.csv", index=False)

    stage_a_spec = MiningStageSpec(
        stage_name="stage_a_context",
        active_groups=("regime", "location"),
        allow_groups_in_rule=("regime", "location"),
        output_dir_name="stage_a_context",
        allowed_group_pairs=(("regime", "location"),),
        slot_order=("trigger", "location", "regime"),
    )

    run_step = str(cfg.get("run_step", "full")).lower()
    if run_step == "step2":
        step1_base_dir = cfg.get("step1_dir")
        if not step1_base_dir:
            raise ValueError("step2 requires cfg['step1_dir']")
        step1_input_dir = resolve_stage_a_step1_dir(
            step1_base_dir, target_name=target_name, horizon=horizon, side=side
        )
        candidate_registry_override = cfg.get("global_step2_registry_override")
        if candidate_registry_override is None:
            candidate_registry_override = pd.read_csv(
                step1_input_dir / "step1_post_dedup_registry.csv"
            )
        tprint(
            f"Stage A [{target_name} @ H{horizon} {side}]: Running step 2 over "
            f"{len(candidate_registry_override)} registry rules."
        )
        tprint(
            f"Stage A [{target_name} @ H{horizon} {side}]: "
            "step2 resume mode active, skipping local mining/assessment and "
            "deferring evaluation to the pooled global assessor."
        )
        return {
            "stage_a": pd.DataFrame(),
            "metadata_a": metadata_a,
            "X_a": X_a,
            "metadata_assessor": metadata_assessor,
            "X_assessor": X_assessor,
        }

    if cfg.get("use_dynamic_hpo", False):
        tprint(f"--- DYNAMIC HPO: Tuning Stage A for {side.upper()} ---")
        try:
            # We use X_a and side_fwd_ret for HPO
            # The HPO script expects MAIN_MINER_PARAMS but we can build them from cfg
            hpo_main_params = {
                "objective": "quantile",
                "metric": "quantile",
                "learning_rate": float(cfg.get("learning_rate", 0.03)),
                "num_leaves": int(cfg.get("lgbm_num_leaves", 64)),
                "max_depth": int(cfg.get("lgbm_max_depth", 5)) + 1,
                "lambda_l1": float(cfg.get("lambda_l1", 0.0)),
                "lambda_l2": float(cfg.get("lambda_l2", 0.0)),
                "verbosity": -1,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "feature_fraction": 0.7,
                "n_jobs": max(1, min(3, int(cfg.get("lgbm_n_jobs", 3)))),
                "min_sum_hessian_in_leaf": float(
                    cfg.get("min_sum_hessian_in_leaf", 1e-4)
                ),
            }

            # Extract recent volatility if available, default to 1s
            if "atr" in data.columns:
                vol_array = data["atr"].to_numpy() / np.maximum(
                    data["close"].to_numpy(), 1e-9
                )
                vol_array = np.nan_to_num(vol_array, nan=1.0, posinf=1.0, neginf=1.0)
            else:
                vol_array = np.ones_like(side_fwd_ret)

            # Ensure vol_array is aligned with X_a
            # X_a is returned by prepare_features which currently returns aligned data,
            # but we explicitly slice it if needed. However, data length is assumed to match X_a length here
            # based on pipeline architecture.

            hpo_results = run_short_hpo_for_target_horizon(
                X=X_a,
                y=side_fwd_ret[: len(X_a)],
                vol=vol_array[: len(X_a)],
                main_params=hpo_main_params,
                seed=cfg.get("random_state", 42),
            )

            best_cfg = hpo_results.get("best_final_result")
            if best_cfg and best_cfg.valid:
                tprint(
                    f"Dynamic HPO Results: alpha={best_cfg.cfg.alpha:.3f}, "
                    f"min_gain={best_cfg.cfg.min_gain_to_split:.5f}, "
                    f"min_leaf_frac={best_cfg.cfg.min_leaf_frac:.5f}"
                )

                # Update cfg for this side pipeline
                cfg = cfg.copy()  # Avoid polluting other sides
                cfg["hpo_best_alpha"] = best_cfg.cfg.alpha
                cfg["hpo_min_gain_to_split"] = best_cfg.cfg.min_gain_to_split

                # Use train_n if available, otherwise fallback to full X_a size
                train_n = hpo_results.get("train_n", len(X_a))
                miner_min_leaf_floor_frac = float(
                    cfg.get("miner_min_leaf_floor_frac", 0.05)
                )
                cfg["hpo_min_data_in_leaf"] = max(
                    25,
                    int(
                        round(
                            max(best_cfg.cfg.min_leaf_frac, miner_min_leaf_floor_frac)
                            * train_n
                        )
                    ),
                )
            else:
                tprint(
                    f"Dynamic HPO failed or invalid: {best_cfg.reason if best_cfg else 'Unknown'}. Using defaults."
                )

        except Exception as e:
            tprint(f"Dynamic HPO encountered error: {e}. Using defaults.")
            traceback.print_exc()

    # Use bounded_target as the real training target if provided.
    # Return-based targets (returns_target, atr_norm_returns_target) MUST be sign-flipped for short sides.
    # Magnitude-based targets (target_eff, target_vame) should NOT be sign-flipped.
    if bounded_target is not None:
        side_target = np.asarray(bounded_target, dtype=np.float32).copy()
        if target_name in {"returns_target", "atr_norm_returns_target"}:
            if side == "short":
                side_target = -side_target
            fee_pct = float(cfg.get("training_label_round_trip_fee_pct", 0.002))
            if target_name == "returns_target":
                side_target = side_target - fee_pct
            else:
                fee_in_target_units = fee_pct / np.maximum(
                    np.asarray(atr_frac_for_targets, dtype=np.float32),
                    1e-3,
                )
                side_target = side_target - fee_in_target_units
            low_q = float(cfg.get("training_label_winsor_low_q", 0.01))
            high_q = float(cfg.get("training_label_winsor_high_q", 0.99))
            finite = np.isfinite(side_target)
            if finite.sum() >= 20 and 0.0 <= low_q < high_q <= 1.0:
                lo, hi = np.nanquantile(side_target[finite], [low_q, high_q])
                side_target = np.clip(side_target, lo, hi).astype(
                    np.float32, copy=False
                )
    else:
        side_target = (
            side_fwd_ret_norm  # legacy fallback only when no triad target is supplied
        )

    # ROOT CAUSE OF ZERO VALID ROWS:
    # Previously, the code implicitly assumed `[0, 1]` operational bounds by
    # dropping any rows outside that range, or via upstream filtering producing NaNs.
    # Downstream target bounding logic: use target-specific domain validation
    # Keep finite checks and target-specific sanity filters
    # Do NOT use one universal [-4, 4] filter for all targets
    feature_any_finite = np.any(np.isfinite(X_a), axis=1)

    # Check what causes zero valid rows precisely
    _finite_target_only = np.isfinite(side_target)

    # If using bounded target, apply target-specific domain validation
    # For signed targets (returns_target, atr_norm_returns_target), allow full range
    # For positive-only targets (target_eff, target_vame), ensure non-negative
    if bounded_target is not None:
        signed_targets = {"returns_target", "atr_norm_returns_target"}
        if target_name in signed_targets:
            # Signed targets: only require finite (no bounds)
            _domain_valid = _finite_target_only
        else:
            # Positive-only targets: require finite and non-negative
            _domain_valid = _finite_target_only & (side_target >= 0.0)
    else:
        # Legacy targets: only require finite
        _domain_valid = _finite_target_only

    target_finite = _domain_valid & nuisance_valid

    # We log the target distribution to help identify why targets are being dropped or bounded
    _summarize_target_distribution(target_name, side_target, horizon, side)
    complete_rows = feature_any_finite & target_finite
    stage_a_row_funnel = []
    prev_rows = max(side_input_rows, 1)
    for stage_name_funnel, rows_count, symbol_count in [
        ("side_input", side_input_rows, side_input_symbols),
        ("stage_a_feature_matrix", int(X_a.shape[0]), side_input_symbols),
        ("stage_a_target_finite", int(target_finite.sum()), side_input_symbols),
        (
            "stage_a_any_feature_finite",
            int(feature_any_finite.sum()),
            side_input_symbols,
        ),
        ("stage_a_complete_rows", int(complete_rows.sum()), side_input_symbols),
    ]:
        stage_a_row_funnel.append(
            {
                "stage": stage_name_funnel,
                "rows": rows_count,
                "symbols": symbol_count,
                "fraction_of_prev": float(rows_count / max(prev_rows, 1)),
            }
        )
        prev_rows = max(rows_count, 1)
    atomic_to_csv(
        _build_row_funnel_df(stage_a_row_funnel),
        stage_a_output_dir / "row_funnel.csv",
        index=False,
    )
    tprint(
        "Stage A row funnel: "
        + " -> ".join(f"{row['stage']}={row['rows']}" for row in stage_a_row_funnel)
    )

    if int(complete_rows.sum()) == 0:
        # Developer Note:
        # This skip condition checks if the MINER target (e.g. target_eff or target_vame)
        # has valid rows after applying its required bounds. It is completely normal and
        # legitimate for some miner targets to have 0 valid rows in a specific horizon/side combo
        # if the domain filters all data. This is a miner-only skip and DOES NOT affect
        # Ridge learnability, which safely and exclusively operates on `ridge_target_by_side`
        # (the vol-normalized forward returns).
        tprint(f"Skipping {target_name} @ H{horizon} [{side}] due to zero valid rows.")
        return None

    step1_input_dir: Optional[Path] = None
    candidate_registry_override: Optional[pd.DataFrame] = None

    stage_a_result = run_mining_stage(
        data,
        side_fwd_ret,
        side_fwd_ret_norm,
        X_a,
        metadata_a,
        cfg,
        stage_a_output_dir,
        stage_a_spec.stage_name,
        stage_a_spec.allowed_group_pairs,
        slot_order=stage_a_spec.slot_order,
        folds=folds,
        mask_resolver=CanonicalRuleMaskResolver(X_a, metadata_a),
        pipeline_stage_name="stage_a_context",
        explicit_side=side,
        target_name=target_name,
        horizon=horizon,
        primary_target_override=side_target,
        sample_weight_surprisal_override=bounded_target_surprisal,
        nuisance_feature_arrays=nuisance_feature_arrays,
        nuisance_feature_resolution=nuisance_feature_resolution,
        run_step=run_step,
        step1_input_dir=step1_input_dir,
        candidate_registry_override=candidate_registry_override,
        bounded_target=bounded_target,
        target_nan_reasons=target_nan_reasons,
    )
    log_stage_gate_diagnostics("Stage A", stage_a_result, cfg)

    if run_step == "step1":
        tprint(f"Stage A step1 only complete for {target_name} @ H{horizon} [{side}]")
        return {
            "stage_a": pd.DataFrame(),
            "stage_a_result": stage_a_result,
            "metadata_a": metadata_a,
            "X_a": X_a,
            "metadata_assessor": metadata_assessor,
            "X_assessor": X_assessor,
        }

    winning_contexts, stage_a_rejection_summary = select_stage_a_contexts(
        stage_a_result, cfg
    )
    atomic_to_csv(
        stage_a_rejection_summary,
        stage_a_output_dir / "stage_a_context_selection_summary.csv",
        expected_columns=["reason", "count"],
    )
    stage_a_rejection_map = build_stage_a_rejection_map(
        stage_a_result, winning_contexts, cfg
    )
    atomic_to_csv(
        stage_a_rejection_map,
        stage_a_output_dir / "stage_a_rejection_map.csv",
        expected_columns=[
            "stage_order",
            "stage_name",
            "gate_name",
            "metric_name",
            "threshold",
            "input_count",
            "rejected_count",
            "passed_count",
        ],
    )

    stage_a_accepted_count = len(stage_a_result.get("accepted_registry", []))
    tprint(
        f"Stage A accepted -> winning contexts funnel: {stage_a_accepted_count} -> {len(winning_contexts)}"
    )

    if len(winning_contexts) < 5 and stage_a_accepted_count > 10:
        tprint(
            f"WARNING: Very few contexts survived selection ({len(winning_contexts)} out of {stage_a_accepted_count})."
        )

    if not winning_contexts.empty:
        tprint("Top selected contexts by hurdle excess:")
        top_ctx = winning_contexts.sort_values("hurdle_excess", ascending=False).head(5)
        for row in top_ctx.itertuples(index=False, name=None):
            tprint(f"  - {row[0]}: hurdle_excess={row[1]:.5f}")

    if winning_contexts.empty:
        tprint("No contexts found in Stage A. Returning Stage A-only results.")
        return {
            "stage_a": winning_contexts,
            "stage_a_result": stage_a_result,
            "metadata_a": metadata_a,
            "X_a": X_a,
        }

    return {
        "stage_a": winning_contexts,
        "stage_a_result": stage_a_result,
        "metadata_a": metadata_a,
        "X_a": X_a,
        "metadata_assessor": metadata_assessor,
        "X_assessor": X_assessor,
    }


def run_lgbm_mask_generation_pipeline(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    tprint("=" * 80)
    tprint("LGBM MASK GENERATION PIPELINE: START")
    tprint("=" * 80)

    root_output_dir = build_run_output_dir(cfg)
    folds = build_walk_forward_folds(
        n_samples=len(data),
        n_folds=int(cfg.get("n_folds", 5)),
        min_train_frac=float(cfg.get("cv_min_train_frac", 0.5)),
        embargo=int(cfg.get("cv_embargo", 0)),
    )

    long_results = run_side_pipeline(
        "long", data, feature_dict, fwd_ret, fwd_ret_norm, cfg, folds, root_output_dir
    )
    short_results = run_side_pipeline(
        "short", data, feature_dict, fwd_ret, fwd_ret_norm, cfg, folds, root_output_dir
    )

    long_pre_global_raw = create_pre_global_registry(long_results)
    short_pre_global_raw = create_pre_global_registry(short_results)

    combined_pre_global_raw = pd.concat(
        [long_pre_global_raw, short_pre_global_raw], ignore_index=True
    )
    if combined_pre_global_raw.empty:
        tprint("No rules generated from either side. Aborting Global Consolidation.")
        return {
            "stage_a": pd.DataFrame(),
            "combined": pd.DataFrame(),
        }

    origin_counts = {"long_only": 0, "short_only": 0, "overlapping_keys": 0}
    l_keys = (
        set(long_pre_global_raw["canonical_key"])
        if not long_pre_global_raw.empty
        else set()
    )
    s_keys = (
        set(short_pre_global_raw["canonical_key"])
        if not short_pre_global_raw.empty
        else set()
    )

    origin_counts["overlapping_keys"] = len(l_keys.intersection(s_keys))
    origin_counts["long_only"] = len(l_keys - s_keys)
    origin_counts["short_only"] = len(s_keys - l_keys)

    pd.DataFrame([origin_counts]).to_csv(
        root_output_dir / "combined_registry_side_summary.csv", index=False
    )

    combined_pre_global = combined_pre_global_raw.sort_values(
        ["composite_score", "hurdle_excess"], ascending=False
    )
    combined_pre_global = combined_pre_global.drop_duplicates(
        subset=["canonical_key", "side"], keep="first"
    )
    combined_pre_global["preset"] = cfg.get("preset", "exploration")
    combined_pre_global.to_csv(
        root_output_dir / "combined_accepted_registry_pre_global.csv", index=False
    )

    combined_mask_map: Dict[str, np.ndarray] = {}
    combined_parent_context_map: Dict[str, str] = {}
    combined_side_map: Dict[str, str] = {}

    def extract_masks_from_results(results: Dict[str, Any], side: str):
        if results["X_a"] is None:
            return
        stage_a_accepted = results["stage_a"]

        stage_a_resolver = CanonicalRuleMaskResolver(
            results["X_a"], results["metadata_a"]
        )
        stage_a_accepted = results["stage_a"]
        if stage_a_accepted is not None and not stage_a_accepted.empty:
            for row in stage_a_accepted.itertuples(index=False, name=None):
                combined_mask_map[row[0]] = stage_a_resolver.get_mask(row[0])
                combined_side_map[row[0]] = row[1]

    extract_masks_from_results(long_results, "long")
    extract_masks_from_results(short_results, "short")

    combined_resolver = DictionaryMaskResolver(
        combined_mask_map,
        parent_context_map=combined_parent_context_map,
        side_map=combined_side_map,
    )

    # Merge metadata from long/short, removing duplicates
    all_metadata = []
    seen_meta = set()
    for res in [long_results, short_results]:
        if res["metadata_a"]:
            for m in res["metadata_a"]:
                if m.feature_name not in seen_meta:
                    all_metadata.append(m)
                    seen_meta.add(m.feature_name)

    tprint(
        f"Cross-Stage Funnel: Long ({len(long_pre_global_raw)}) + Short ({len(short_pre_global_raw)}) -> Combined Pre-Global ({len(combined_pre_global)})"
    )
    combined_global_registry = combined_pre_global.drop_duplicates(
        subset=["canonical_key", "side"], keep="first"
    )
    tprint(
        f"Global registry assembly resulted in {len(combined_global_registry)} rules."
    )
    combined_global_registry["preset"] = cfg.get("preset", "exploration")
    combined_global_registry.to_csv(
        root_output_dir / "combined_accepted_rule_registry.csv", index=False
    )

    # X_a represents regime + location features which are primary driver of correlation anyway
    X_for_ridge = None
    if long_results["X_a"] is not None:
        X_for_ridge = long_results["X_a"]
    elif short_results["X_a"] is not None:
        X_for_ridge = short_results["X_a"]

    portfolio_diversity_report = build_portfolio_diversity_report(
        combined_global_registry,
        combined_resolver,
        data,
        fwd_ret,
        X_for_ridge=X_for_ridge,
    )
    portfolio_diversity_report.to_csv(
        root_output_dir / "portfolio_diversity_report.csv", index=False
    )

    global malformed_key_count, unresolved_feature_count, unresolved_feature_names
    tprint(
        f"Canonical Key Diagnostics: malformed={malformed_key_count}, unresolved_features={unresolved_feature_count}"
    )
    if unresolved_feature_names:
        tprint(f"Unresolved features: {', '.join(list(unresolved_feature_names)[:10])}")

    audit_data = {
        "malformed_key_count": malformed_key_count,
        "unresolved_feature_count": unresolved_feature_count,
    }
    pd.DataFrame([audit_data]).to_csv(
        root_output_dir / "canonical_key_parse_audit.csv", index=False
    )

    # Final Registry Breakdowns
    if not combined_global_registry.empty:
        breakdown = (
            combined_global_registry.groupby(["side", "display_arity"])
            .size()
            .reset_index(name="count")
        )
        breakdown.to_csv(
            root_output_dir / "final_registry_breakdown_by_side_arity.csv", index=False
        )

        final_summary = {
            "total_accepted": len(combined_global_registry),
            "mean_support_pct": float(
                combined_global_registry["mean_support_pct"].mean()
            ),
            "median_support_pct": float(
                combined_global_registry["mean_support_pct"].median()
            ),
            "mean_hurdle_excess": float(
                combined_global_registry["hurdle_excess"].mean()
            ),
            "median_hurdle_excess": float(
                combined_global_registry["hurdle_excess"].median()
            ),
        }

        # Count by origin
        if "origin_stage" in combined_global_registry.columns:
            for origin, count in (
                combined_global_registry["origin_stage"].value_counts().items()
            ):
                final_summary[f"origin_{origin}"] = count

        # Count by rule type
        if "rule_type" in combined_global_registry.columns:
            for rtype, count in (
                combined_global_registry["rule_type"].value_counts().items()
            ):
                final_summary[f"type_{rtype}"] = count

        # Portfolio Diversity Highlights
        eff_rules = portfolio_diversity_report[
            portfolio_diversity_report["metric"] == "effective_independent_rules"
        ]["value"].values
        if len(eff_rules) > 0:
            final_summary["effective_independent_rules"] = float(eff_rules[0])

        top_rule_share = portfolio_diversity_report[
            portfolio_diversity_report["metric"] == "top_rule_share"
        ]["value"].values
        if len(top_rule_share) > 0:
            final_summary["top_rule_share"] = float(top_rule_share[0])

        with open(root_output_dir / "final_registry_summary.json", "w") as f:
            json.dump(final_summary, f, indent=2)

        tprint(f"Final Output Summary: {len(combined_global_registry)} rules.")
        tprint(f"  - Mean Support: {final_summary['mean_support_pct']:.2%}")
        tprint(f"  - Median Hurdle Excess: {final_summary['median_hurdle_excess']:.5f}")
        if "effective_independent_rules" in final_summary:
            tprint(
                f"  - Effective Independent Rules: {final_summary['effective_independent_rules']:.2f}"
            )

        tprint("Side Mix:")
        side_counts = combined_global_registry["side"].value_counts()
        for side, count in side_counts.items():
            tprint(f"  - {side}: {count}")

        tprint("Top 160 Final Diverse Rules (Thorough Report):")
        top_final = select_top_diverse_rules(
            combined_global_registry, combined_mask_map, top_n=160
        )
        for i, (_, row) in enumerate(top_final.iterrows(), start=1):
            tprint(
                f"  {i:2d}. [{row.get('side', 'unknown').upper()}] {row['canonical_key']}\n"
                f"      score={row.get('composite_score', 0):.3f} | "
                f"hurdle_excess={row.get('hurdle_excess', 0):.5f} | "
                f"support={row.get('mean_support_pct', 0):.2%} | "
                f"ret={row.get('directional_mean_ret', 0):.5f} | "
                f"uplift={row.get('mean_uplift', 0):.5f} | "
                f"presence={row.get('presence_freq', 0):.2%} | "
                f"sign_cons={row.get('sign_consistency', 0):.2%} | "
                f"arity={row.get('display_arity', 0)}"
            )
    else:
        tprint("Final Output Summary: 0 rules accepted.")

    tprint(
        f"Two-stage mining complete. Total accepted rules: {len(combined_global_registry)}"
    )
    return {
        "stage_a_long": long_results["stage_a"],
        "stage_a_short": short_results["stage_a"],
        "combined": combined_global_registry,
    }


def build_global_stage_a_ridge_shortlist(
    pooled_step1_frames: List[pd.DataFrame],
    X_ref: np.ndarray,
    metadata_ref: List[FeatureMetadata],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    global_cap = int(cfg.get("global_ridge_candidate_cap", 120))
    if not pooled_step1_frames:
        return pd.DataFrame()

    pooled = pd.concat(pooled_step1_frames, ignore_index=True, copy=False)
    if pooled.empty:
        return pd.DataFrame()

    support_min = float(cfg.get("support_min_pct", SUPPORT_MIN))
    support_max = float(cfg.get("max_support_pct", SUPPORT_MAX))
    support_col = (
        "mean_support_pct"
        if "mean_support_pct" in pooled.columns
        else ("support_pct" if "support_pct" in pooled.columns else None)
    )
    if support_col is not None:
        pooled[support_col] = pd.to_numeric(pooled[support_col], errors="coerce")
        pre_filter_n = len(pooled)
        pooled = pooled.loc[
            pooled[support_col].between(support_min, support_max, inclusive="both")
        ].copy()
        tprint(
            "Global stage2 shortlist support filter: "
            f"input={pre_filter_n} kept={len(pooled)} "
            f"support_col={support_col} range=[{support_min:.3f}, {support_max:.3f}]"
        )
        if pooled.empty:
            return pd.DataFrame()

    pooled = pooled.drop_duplicates(
        subset=["canonical_key", "source_target", "source_horizon", "side"],
        keep="first",
    ).copy()
    if "cheap_rank" not in pooled.columns:
        pooled["cheap_rank"] = 0.0
    pooled["cheap_rank"] = pd.to_numeric(pooled["cheap_rank"], errors="coerce").fillna(
        -np.inf
    )
    pooled = pooled.sort_values(
        ["cheap_rank", "source_target", "source_horizon", "side"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    selected = pooled.head(global_cap).copy()

    tprint(
        "Global stage2 shortlist: "
        f"input={len(pooled)} selected={len(selected)} cap={global_cap}"
    )

    return selected


def load_step1_slice_basket(
    step1_dir: Path,
    target_name: str,
    horizon: int,
    side: str,
    basket_size: int = 15,
    support_min_pct: float = SUPPORT_MIN,
    support_max_pct: float = SUPPORT_MAX,
) -> pd.DataFrame:
    post_dedup_path = step1_dir / "step1_post_dedup_registry.csv"
    candidate_path = step1_dir / "candidate_rule_registry.csv"

    frames: List[pd.DataFrame] = []
    selected_keys: Set[str] = set()

    def _filter_support_bounds(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        support_col = (
            "mean_support_pct"
            if "mean_support_pct" in df.columns
            else ("support_pct" if "support_pct" in df.columns else None)
        )
        if support_col is None:
            return df
        out = df.copy()
        out[support_col] = pd.to_numeric(out[support_col], errors="coerce")
        return out.loc[
            out[support_col].between(
                float(support_min_pct), float(support_max_pct), inclusive="both"
            )
        ].copy()

    if post_dedup_path.exists():
        post_df = pd.read_csv(post_dedup_path)
        if not post_df.empty:
            post_df = _filter_support_bounds(post_df)
        if not post_df.empty:
            post_df["canonical_key"] = post_df["canonical_key"].astype(str)
            post_df["_basket_source"] = "post_dedup"
            frames.append(post_df)
            selected_keys = set(post_df["canonical_key"].tolist())

    if len(selected_keys) < basket_size and candidate_path.exists():
        cand_df = pd.read_csv(candidate_path)
        if not cand_df.empty:
            cand_df = _filter_support_bounds(cand_df)
        if not cand_df.empty:
            cand_df["canonical_key"] = cand_df["canonical_key"].astype(str)
            cand_df = cand_df.loc[~cand_df["canonical_key"].isin(selected_keys)].copy()
            if "cheap_rank" in cand_df.columns:
                cand_df["cheap_rank"] = pd.to_numeric(
                    cand_df["cheap_rank"], errors="coerce"
                ).fillna(-np.inf)
                cand_df = cand_df.sort_values("cheap_rank", ascending=False)
            elif "composite_score" in cand_df.columns:
                cand_df["composite_score"] = pd.to_numeric(
                    cand_df["composite_score"], errors="coerce"
                ).fillna(-np.inf)
                cand_df = cand_df.sort_values("composite_score", ascending=False)
            cand_df["_basket_source"] = "candidate_backfill"
            needed = basket_size - len(selected_keys)
            if needed > 0:
                frames.append(cand_df.head(needed))

    if not frames:
        return pd.DataFrame()

    basket = pd.concat(frames, ignore_index=True, copy=False)
    basket = basket.drop_duplicates(subset=["canonical_key"], keep="first").copy()
    basket = basket.head(basket_size).copy()
    basket["source_target"] = target_name
    basket["source_horizon"] = horizon
    basket["side"] = side
    return basket


def run_lgbm_mask_generation_triad(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    triad_targets: Dict[str, Dict[int, np.ndarray]],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    target_nan_reasons: Optional[Union[np.ndarray, Dict[Tuple[str, int], np.ndarray]]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Triad-target version of the mask generation pipeline.

    Runs the Cartesian product:
        for horizon in horizons:
            for target_name in [eff, ela, vame]:
                for side in [long, short]:
                    run_side_pipeline(...)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with event metadata
    feature_dict : Dict[str, np.ndarray]
        Dictionary of feature arrays
    triad_targets : Dict[str, Dict[int, np.ndarray]]
        Nested dict of bounded targets: {target_name: {horizon: target_array}}
        e.g., {"target_eff": {5: array, 10: array, ...}, ...}
    fwd_ret : np.ndarray
        Array of forward returns for assessment
    fwd_ret_norm : np.ndarray
        Array of normalized forward returns for assessment
    cfg : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    Dict[str, Any]
        Merged discovery outputs with provenance. Structure:
        {
            "results_by_target_horizon": {
                ("target_eff", 5): {"long": ..., "short": ...},
                ...
            },
            "combined_registry": pd.DataFrame,
        }
    """
    tprint("=" * 80)
    tprint("TRIAD-TARGET TWO-STAGE LGBM MASK GENERATION: START")
    tprint("=" * 80)

    # Get horizons and target names from config or defaults
    horizons = cfg.get("triad_horizons", TRIAD_DEFAULT_HORIZONS)
    target_names = cfg.get("triad_target_names", TRIAD_DEFAULT_TARGET_NAMES)
    triad_run_step = str(cfg.get("run_step", "full")).lower()

    tprint(f"Horizons: {horizons}")
    tprint(f"Targets: {target_names}")
    tprint(
        f"Total combinations: {len(horizons) * len(target_names) * 2} (horizons × targets × sides)"
    )

    root_output_dir = build_run_output_dir(cfg)

    # Build folds once (shared across all combinations)
    folds = build_walk_forward_folds(
        n_samples=len(data),
        n_folds=int(cfg.get("n_folds", 5)),
        min_train_frac=float(cfg.get("cv_min_train_frac", 0.5)),
        embargo=int(cfg.get("cv_embargo", 0)),
    )

    # Store results by (target_name, horizon, side)
    results_by_target_horizon: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    # Track all registries for final combination
    all_registries: List[pd.DataFrame] = []
    pooled_step1_frames: List[pd.DataFrame] = []
    x_ref: Optional[np.ndarray] = None
    metadata_ref: List[FeatureMetadata] = []
    assessor_x_ref: Optional[np.ndarray] = None
    assessor_metadata_ref: List[FeatureMetadata] = []

    total_combinations = len(horizons) * len(target_names)
    current_combination = 0

    for horizon in horizons:
        for target_name in target_names:
            current_combination += 1
            tprint(f"\n{'='*60}")
            tprint(
                f"COMBINATION {current_combination}/{total_combinations}: {target_name} @ H{horizon}"
            )
            tprint(f"{'='*60}")

            # Get bounded target for this target_name/horizon
            bounded_target = None
            if target_name in triad_targets:
                if horizon in triad_targets[target_name]:
                    bounded_target = triad_targets[target_name][horizon]
                    tprint(
                        f"Using bounded target array: shape={bounded_target.shape}, "
                        f"range=[{np.nanmin(bounded_target):.3f}, {np.nanmax(bounded_target):.3f}]"
                    )
                else:
                    tprint(
                        f"WARNING: No bounded target for {target_name} @ H{horizon}, skipping"
                    )
                    continue
            else:
                tprint(
                    f"WARNING: Target {target_name} not found in triad_targets, skipping"
                )
                continue

            surprisal_key = f"{target_name}_surprisal"
            bounded_target_surprisal = None
            if (
                triad_targets.get(surprisal_key) is not None
                and horizon in triad_targets.get(surprisal_key, {})
            ):
                bounded_target_surprisal = triad_targets[surprisal_key][horizon]

            # Build horizon-specific output directory
            horizon_target_dir = root_output_dir / f"h{horizon}" / target_name
            horizon_target_dir.mkdir(parents=True, exist_ok=True)

            # Create horizon-specific config with target settings
            horizon_cfg = cfg.copy()
            horizon_cfg["target_name"] = target_name
            horizon_cfg["horizon"] = horizon
            horizon_cfg["run_step"] = (
                "step1" if triad_run_step == "full" else triad_run_step
            )

            # Apply per-target config from TRIAD_TARGET_CONFIGS
            if target_name in TRIAD_TARGET_CONFIGS:
                target_config = TRIAD_TARGET_CONFIGS[target_name]
                horizon_cfg["huber_alpha"] = target_config.get("huber_alpha", 0.9)
                horizon_cfg["learning_rate"] = target_config.get("learning_rate", 0.03)
                horizon_cfg["min_support_pct"] = target_config.get(
                    "min_support_pct", 0.05
                )
                horizon_cfg["ic_hurdle"] = target_config.get("ic_hurdle", 0.02)
                tprint(
                    f"Applied target config: huber_alpha={target_config.get('huber_alpha')}, "
                    f"lr={target_config.get('learning_rate')}"
                )

            # Apply horizon-specific config from HORIZON_CONFIGS
            if horizon in HORIZON_CONFIGS:
                horizon_config = HORIZON_CONFIGS[horizon]
                min_leaf_mult = horizon_config.get("min_data_in_leaf_multiplier", 1.0)
                base_min_leaf = int(horizon_cfg.get("min_data_in_leaf", 64))
                horizon_cfg["min_data_in_leaf"] = int(base_min_leaf * min_leaf_mult)
                tprint(
                    f"Applied horizon config: min_data_in_leaf_multiplier={min_leaf_mult}"
                )

            # For triad targets, we use the bounded_target itself as the "return" for scoring
            # so that RuleScorer and MaskAssessor see the actual edge we are mining.
            # Run both sides
            for side in ["long", "short"]:
                tprint(
                    f"\n--- Running {side.upper()} side for {target_name} @ H{horizon} ---"
                )

                side_results = run_side_pipeline(
                    target_nan_reasons=(
                        target_nan_reasons.get((target_name, horizon))
                        if isinstance(target_nan_reasons, dict)
                        else target_nan_reasons
                    ),
                    side=side,
                    data=data,
                    feature_dict=feature_dict,
                    fwd_ret=fwd_ret,
                    fwd_ret_norm=fwd_ret_norm,
                    cfg=horizon_cfg,
                    folds=folds,
                    root_output_dir=horizon_target_dir,
                    target_name=target_name,
                    horizon=horizon,
                    bounded_target=bounded_target,
                    bounded_target_surprisal=bounded_target_surprisal,
                )

                # Store results
                results_by_target_horizon[(target_name, horizon, side)] = side_results

                if side_results is not None:
                    if x_ref is None and side_results.get("X_a") is not None:
                        x_ref = side_results["X_a"]
                        metadata_ref = side_results.get("metadata_a", []) or []
                    if (
                        assessor_x_ref is None
                        and side_results.get("X_assessor") is not None
                    ):
                        assessor_x_ref = side_results["X_assessor"]
                        assessor_metadata_ref = (
                            side_results.get("metadata_assessor", []) or []
                        )

                    if not side_results.get("stage_a", pd.DataFrame()).empty:
                        stage_a_with_prov = side_results["stage_a"].copy()
                        stage_a_with_prov["source_target"] = target_name
                        stage_a_with_prov["source_horizon"] = horizon
                        stage_a_with_prov["side"] = side
                        all_registries.append(stage_a_with_prov)

                step1_base_dir_for_merge = (
                    Path(cfg.get("step1_dir"))
                    if str(cfg.get("run_step", "full")).lower() == "step2"
                    and cfg.get("step1_dir")
                    else root_output_dir
                )
                step1_dir = resolve_stage_a_step1_dir(
                    step1_base_dir_for_merge,
                    target_name=target_name,
                    horizon=horizon,
                    side=side,
                )
                basket_df = load_step1_slice_basket(
                    step1_dir=step1_dir,
                    target_name=target_name,
                    horizon=horizon,
                    side=side,
                    basket_size=int(cfg.get("global_ridge_per_slice_basket_size", 15)),
                    support_min_pct=float(cfg.get("support_min_pct", SUPPORT_MIN)),
                    support_max_pct=float(cfg.get("max_support_pct", SUPPORT_MAX)),
                )
                tprint(
                    "Stage A pooled basket load: "
                    f"target={target_name} horizon={horizon} side={side} "
                    f"rows={len(basket_df)} step1_dir={step1_dir}"
                )
                if not basket_df.empty:
                    pooled_step1_frames.append(basket_df)

    if pooled_step1_frames:
        tprint(
            "Collected Stage A post-dedup survivors: "
            f"frames={len(pooled_step1_frames)} total_rules={sum(len(df) for df in pooled_step1_frames)}"
        )
    else:
        tprint("Collected Stage A post-dedup survivors: frames=0 total_rules=0")

    # GLOBAL CONSOLIDATED STEP 2
    if triad_run_step in ["full", "step2"]:
        global_stage_a_pooled_registry = build_global_stage_a_ridge_shortlist(
            pooled_step1_frames=pooled_step1_frames,
            X_ref=x_ref,
            metadata_ref=metadata_ref,
            cfg=cfg,
        )
        tprint(
            f"Global consolidated Step 2 pooled registry: {len(global_stage_a_pooled_registry)} rules total."
        )

        if not global_stage_a_pooled_registry.empty:
            assessor_x_runtime = assessor_x_ref if assessor_x_ref is not None else x_ref
            assessor_metadata_runtime = (
                assessor_metadata_ref if assessor_metadata_ref else metadata_ref
            )
            resolver_x_runtime, resolver_metadata_runtime, resolver_feature_names, unresolved_rule_features = (
                build_rule_resolver_feature_matrix(
                    feature_dict,
                    global_stage_a_pooled_registry["canonical_key"]
                    .astype(str)
                    .tolist(),
                )
            )
            tprint(
                "Pooled Step 2 rule resolver matrix: "
                f"loaded={len(resolver_feature_names)} shape={resolver_x_runtime.shape} "
                f"unresolved={len(unresolved_rule_features)}"
            )
            if unresolved_rule_features:
                tprint(
                    "Pooled Step 2 unresolved rule features sample: "
                    + ", ".join(unresolved_rule_features[:10])
                )
            if resolver_x_runtime.shape[1] == 0:
                raise ValueError(
                    "Pooled Step 2 rule resolver matrix is empty; cannot assess rules."
                )
            # Prepare consolidated parameters for the global assessor phase
            global_horizon_target_dirs = {}
            for h in horizons:
                for t_name in target_names:
                    global_horizon_target_dirs[(t_name, int(h))] = (
                        root_output_dir / f"h{h}" / t_name
                    )

            # Map of (target_name, horizon) -> target_array for dynamic resolution
            # We also include surprisals if available
            assessment_targets = {}
            for t_name in target_names:
                for h in horizons:
                    if t_name in triad_targets and h in triad_targets[t_name]:
                        assessment_targets[(t_name, int(h))] = triad_targets[t_name][h]
                    
                    surprisal_key = f"{t_name}_surprisal"
                    if surprisal_key in triad_targets and h in triad_targets.get(surprisal_key, {}):
                        assessment_targets[(surprisal_key, int(h))] = triad_targets[surprisal_key][h]

            assessor = MaskAssessor(
                assessor_metadata_runtime,
                cfg,
                mask_resolver=CanonicalRuleMaskResolver(
                    resolver_x_runtime, resolver_metadata_runtime
                ),
            )
            
            # Pass all context necessary for sequential 'in bunches of 14' assessment
            final_assessment_df = assessor.assess_rules(
                global_stage_a_pooled_registry,
                assessor_x_runtime,
                data,
                fwd_ret, # fallback
                fwd_ret_norm, # fallback
                folds,
                step_mode="step2",
                checkpoint_output_dir=root_output_dir,
                triad_targets_map=assessment_targets,
                output_dirs_map=global_horizon_target_dirs,
                batch_size=1,
                target_nan_reasons=target_nan_reasons,
                skip_stage1_filtering=True,
            )

            if not final_assessment_df.empty:
                if "selected_for_final_registry" in final_assessment_df.columns:
                    final_rule_registry = final_assessment_df.loc[
                        final_assessment_df["selected_for_final_registry"].fillna(False)
                    ].copy()
                else:
                    final_rule_registry = final_assessment_df.loc[
                        final_assessment_df["is_structurally_sound"].fillna(False)
                    ].copy()
                # Sort by score_for_best_params as the main index
                if "score_for_best_params" in final_rule_registry.columns:
                    final_rule_registry = final_rule_registry.sort_values(
                        "score_for_best_params", ascending=False
                    ).reset_index(drop=True)
                final_rule_registry["preset"] = cfg.get("preset", "triad_exploration")
                atomic_to_csv(
                    final_rule_registry,
                    root_output_dir / "final_rule_registry.csv",
                )
                atomic_to_csv(
                    final_assessment_df,
                    root_output_dir / "global_final_mask_assessment_audit.csv",
                )
                if hasattr(assessor, "rejection_summary") and assessor.rejection_summary:
                    atomic_to_csv(
                        pd.DataFrame(
                            list(assessor.rejection_summary.items()),
                            columns=["reason", "count"],
                        ),
                        root_output_dir / "global_mask_assessment_rejection_summary.csv",
                        expected_columns=["reason", "count"],
                    )
                tprint(f"Global consolidated assessment complete. {len(final_assessment_df)} rules assessed.")
                return {
                    "results_by_target_horizon": {}, # Legacy compat
                    "combined_registry": final_assessment_df,
                    "global_results": final_assessment_df
                }

    if triad_run_step == "step1":
        top_level_step1_registry = (
            pd.concat(pooled_step1_frames, ignore_index=True, copy=False)
            if pooled_step1_frames
            else pd.DataFrame()
        )
        if not top_level_step1_registry.empty:
            top_level_step1_registry = top_level_step1_registry.drop_duplicates(
                subset=["canonical_key", "source_target", "source_horizon", "side"],
                keep="first",
            ).copy()
            top_level_cheap_gate_rows = collections.defaultdict(list)
            top_level_cheap_gate_result = {}
            top_level_bucket_cheap_ranks = collections.defaultdict(dict)
            for row in top_level_step1_registry.to_dict("records"):
                canonical_key = str(row.get("canonical_key", ""))
                side = str(row.get("side", "long"))
                try:
                    source_horizon = int(row.get("source_horizon", -1))
                except (TypeError, ValueError):
                    source_horizon = -1
                bucket_key = (side, source_horizon)
                cheap_rank = row.get("cheap_rank", row.get("composite_score", 0.0))
                try:
                    cheap_rank = float(cheap_rank)
                except (TypeError, ValueError):
                    cheap_rank = 0.0
                top_level_cheap_gate_rows[bucket_key].append((cheap_rank, canonical_key))
                top_level_bucket_cheap_ranks[bucket_key][canonical_key] = cheap_rank
                top_level_cheap_gate_result[(bucket_key, canonical_key)] = (False, "")

            checkpoint_path = save_stage_a_step1_checkpoint(
                output_dir=root_output_dir,
                candidate_registry=top_level_step1_registry,
                cheap_gate_rows=top_level_cheap_gate_rows,
                cheap_gate_result=top_level_cheap_gate_result,
                bucket_cheap_ranks=top_level_bucket_cheap_ranks,
                stage_a_matrices={},
            )
            tprint(f"TRIAD STEP1 top-level checkpoint saved to {checkpoint_path}")
        tprint("TRIAD STEP1 COMPLETE")
        return {
            "results_by_target_horizon": results_by_target_horizon,
            "combined_registry": pd.DataFrame(),
        }

    # Combine all registries
    tprint(f"\n{'='*60}")
    tprint("COMBINING ALL TRIAD RESULTS")
    tprint(f"{'='*60}")

    if all_registries:
        combined_registry = pd.concat(all_registries, ignore_index=True)
        combined_registry = combined_registry.drop_duplicates(
            subset=["canonical_key", "side", "source_target", "source_horizon"],
            keep="first",
        )
    else:
        combined_registry = pd.DataFrame()

    # Save combined registry
    combined_registry["preset"] = cfg.get("preset", "triad_exploration")
    combined_registry.to_csv(root_output_dir / "final_rule_registry.csv", index=False)

    # Summary statistics
    tprint(f"\nTRIAD TRAINING COMPLETE")
    tprint(f"Total combinations processed: {current_combination}/{total_combinations}")
    tprint(f"Total rules discovered: {len(combined_registry)}")

    if not combined_registry.empty:
        # Breakdown by target/horizon
        breakdown = (
            combined_registry.groupby(["source_target", "source_horizon", "side"])
            .size()
            .reset_index(name="count")
        )
        breakdown.to_csv(
            root_output_dir / "triad_breakdown_by_target_horizon.csv", index=False
        )
        tprint("\nBreakdown by target/horizon/side:")
        for row in breakdown.itertuples(index=False, name=None):
            tprint(f"  {row[0]} H{row[1]} {row[2]}: {row[3]}")

    # Save configuration used
    triad_config = {
        "horizons": horizons,
        "target_names": target_names,
        "total_combinations": total_combinations,
        "completed_combinations": current_combination,
        "total_rules": len(combined_registry),
    }
    with open(root_output_dir / "triad_config.json", "w") as f:
        json.dump(triad_config, f, indent=2)

    # Run merged discovery analysis
    tprint(f"\n{'='*60}")
    tprint("MERGED DISCOVERY ANALYSIS")
    tprint(f"{'='*60}")

    # Build discovery DataFrames for merge functions
    all_results: List[pd.DataFrame] = []
    for (t_name, h, side), result in results_by_target_horizon.items():
        if result is None:
            continue
        discovery_frames: List[pd.DataFrame] = []
        stage_a_result = result.get("stage_a_result", {})
        accepted_registry = stage_a_result.get("accepted_registry", pd.DataFrame())
        candidate_registry = stage_a_result.get("candidate_registry", pd.DataFrame())
        if isinstance(accepted_registry, pd.DataFrame) and not accepted_registry.empty:
            accepted_df = _drop_duplicate_columns(accepted_registry)
            accepted_df["source_target"] = t_name
            accepted_df["source_horizon"] = h
            accepted_df["side"] = side
            accepted_df["rule_status"] = "accepted"
            discovery_frames.append(accepted_df)
        if (
            isinstance(candidate_registry, pd.DataFrame)
            and not candidate_registry.empty
        ):
            candidate_df = _drop_duplicate_columns(candidate_registry)
            candidate_df["source_target"] = t_name
            candidate_df["source_horizon"] = h
            candidate_df["side"] = side
            candidate_df["rule_status"] = "candidate"
            discovery_frames.append(candidate_df)
        if discovery_frames:
            all_results.append(
                pd.concat(
                    [_drop_duplicate_columns(df) for df in discovery_frames],
                    axis=0,
                    ignore_index=True,
                    copy=False,
                )
            )

    # Merge discovery outputs
    merged_output = merge_discovery_outputs_across_targets(
        all_results=all_results,
        output_dir=str(root_output_dir),
    )

    # Create cross-target analysis
    cross_target_analysis = analyze_cross_target_rules(merged_output["merged_rules"])

    # Create target quality summary
    quality_summary = create_target_quality_summary(
        all_results=all_results,
        output_path=str(root_output_dir / "target_quality_summary.csv"),
    )

    # Save merged outputs
    if not merged_output["merged_rules"].empty:
        merged_output["merged_rules"].to_csv(
            root_output_dir / "merged_discovery_all.csv", index=False
        )
        tprint(
            f"Saved merged_discovery_all.csv: {len(merged_output['merged_rules'])} rules"
        )

    if not merged_output["dedup_rules"].empty:
        merged_output["dedup_rules"].to_csv(
            root_output_dir / "merged_discovery_dedup.csv", index=False
        )
        tprint(
            f"Saved merged_discovery_dedup.csv: {len(merged_output['dedup_rules'])} deduplicated rules"
        )

    # Save cross-target analysis
    with open(root_output_dir / "cross_target_analysis.json", "w") as f:
        json.dump(cross_target_analysis, f, indent=2, default=str)

    # Save triad run summary
    triad_run_summary = {
        "horizons": horizons,
        "target_names": target_names,
        "total_combinations": total_combinations,
        "completed_combinations": current_combination,
        "total_rules": len(combined_registry),
        "deduplicated_rules": len(merged_output["dedup_rules"]),
        "cross_target_rules_count": len(
            cross_target_analysis.get("cross_target_rules", [])
        ),
        "universal_rules_count": len(cross_target_analysis.get("universal_rules", [])),
        "merge_stats": merged_output.get("summary_stats", {}),
    }
    with open(root_output_dir / "triad_run_summary.json", "w") as f:
        json.dump(triad_run_summary, f, indent=2, default=str)

    tprint(f"\nTRIAD TRAINING COMPLETE")
    tprint(f"Total rules: {len(combined_registry)}")
    tprint(f"Deduplicated rules: {len(merged_output['dedup_rules'])}")
    tprint(
        f"Cross-target rules: {len(cross_target_analysis.get('cross_target_rules', []))}"
    )
    tprint(f"Universal rules: {len(cross_target_analysis.get('universal_rules', []))}")

    return {
        "results_by_target_horizon": {
            (t, h): {
                "long": results_by_target_horizon.get((t, h, "long"), {}),
                "short": results_by_target_horizon.get((t, h, "short"), {}),
            }
            for t in target_names
            for h in horizons
        },
        "combined_registry": combined_registry,
        "merged_output": merged_output,
        "cross_target_analysis": cross_target_analysis,
        "quality_summary": quality_summary,
    }


# =============================================================================
# MERGED DISCOVERY OUTPUT FUNCTIONS
# =============================================================================


def merge_discovery_outputs_across_targets(
    all_results: List[pd.DataFrame],
    output_dir: str,
    dedup_strategy: str = "canonical_key",
) -> Dict[str, Any]:
    """
    Merge discovered rules/contexts from all (target, horizon) runs.

    Parameters
    ----------
    all_results : List[pd.DataFrame]
        List of per-slice discovery DataFrames with provenance columns.
    output_dir : str
        Base output directory
    dedup_strategy : str
        How to deduplicate ("canonical_key" or "structural")

    Returns
    -------
    Dict[str, Any]
        Dict with:
        - merged_rules: pd.DataFrame with all rules and provenance
        - dedup_rules: pd.DataFrame deduplicated by canonical key
        - cross_target_rules: rules appearing in multiple targets
        - summary_stats: dict of merge statistics
    """
    tprint(f"Merging discovery outputs from {len(all_results)} result sets...")

    non_empty_frames = [
        _drop_duplicate_columns(df)
        for df in all_results
        if isinstance(df, pd.DataFrame) and not df.empty
    ]

    if not non_empty_frames:
        tprint("No rules to merge.")
        return {
            "merged_rules": pd.DataFrame(),
            "dedup_rules": pd.DataFrame(),
            "cross_target_rules": pd.DataFrame(),
            "summary_stats": {"total_rules": 0, "unique_canonical_keys": 0},
        }

    merged_df = pd.concat(non_empty_frames, axis=0, ignore_index=True, copy=False)
    tprint(f"Total rules collected: {len(merged_df)}")

    # Deduplicate by canonical key
    dedup_df = deduplicate_rules_by_canonical_key(merged_df, aggregation="mean")
    tprint(f"Unique canonical keys: {len(dedup_df)}")

    # Summary stats
    summary_stats = {
        "total_rules": len(merged_df),
        "unique_canonical_keys": len(dedup_df),
        "targets_represented": (
            merged_df["source_target"].nunique()
            if "source_target" in merged_df.columns
            else 0
        ),
        "horizons_represented": (
            merged_df["source_horizon"].nunique()
            if "source_horizon" in merged_df.columns
            else 0
        ),
        "sides_represented": (
            merged_df["side"].nunique() if "side" in merged_df.columns else 0
        ),
    }

    return {
        "merged_rules": merged_df,
        "dedup_rules": dedup_df,
        "cross_target_rules": (
            dedup_df[dedup_df["supporting_targets_count"] > 1]
            if "supporting_targets_count" in dedup_df.columns
            else pd.DataFrame()
        ),
        "summary_stats": summary_stats,
    }


def deduplicate_rules_by_canonical_key(
    rules: pd.DataFrame,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Deduplicate rules by canonical key, aggregating metrics.

    For each unique canonical_key:
    - Keep provenance from all sources
    - Aggregate numeric metrics (mean, min, max)
    - Track all source targets/horizons

    Parameters
    ----------
    rules : pd.DataFrame
        Rule DataFrame
    aggregation : str
        Aggregation method ("mean", "min", "max")

    Returns
    -------
    pd.DataFrame
        Deduplicated rules with aggregated metrics
    """
    if rules is None or rules.empty:
        return pd.DataFrame()
    df = rules.copy()

    if "canonical_key" not in df.columns:
        tprint("WARNING: No canonical_key column found, returning as-is")
        return df

    # Columns to aggregate (numeric metrics)
    ic_metric_cols = [
        "mean_oos_ic",
        "p25_oos_ic",
        "p50_oos_ic",
        "p75_oos_ic",
        "positive_ic_fraction",
        "within_mask_ic",
        "delta_within_mask_ic",
    ]
    rule_metric_cols = [
        "mean_net_ret",
        "std_net_ret",
        "mean_support_pct",
        "presence_freq",
        "sign_consistency",
        "entropy_reduction",
    ]
    all_metric_cols = ic_metric_cols + rule_metric_cols

    grouped = df.groupby("canonical_key", sort=False, dropna=False)
    agg_map: Dict[str, Any] = {}

    for col in [
        "source_target",
        "source_horizon",
        "side",
        "display_arity",
        "composite_score",
        "hurdle_excess",
        "conditions",
    ]:
        if col in df.columns:
            agg_map[col] = "first"

    metric_agg = "mean"
    if aggregation == "min":
        metric_agg = "min"
    elif aggregation == "max":
        metric_agg = "max"

    present_metric_cols = [col for col in all_metric_cols if col in df.columns]
    for col in present_metric_cols:
        agg_map[col] = metric_agg
    result_df = grouped.agg(agg_map).reset_index()

    if "presence_freq" in df.columns:
        result_df = result_df.merge(
            grouped["presence_freq"]
            .min()
            .rename("presence_freq_conservative")
            .reset_index(),
            on="canonical_key",
            how="left",
        )
    if "sign_consistency" in df.columns:
        result_df = result_df.merge(
            grouped["sign_consistency"]
            .min()
            .rename("sign_consistency_conservative")
            .reset_index(),
            on="canonical_key",
            how="left",
        )
    if "is_structurally_sound" in df.columns:
        result_df = result_df.merge(
            grouped["is_structurally_sound"]
            .all()
            .rename("is_structurally_sound")
            .reset_index(),
            on="canonical_key",
            how="left",
        )

    if "source_target" in df.columns:
        target_support = grouped["source_target"].agg(
            supporting_targets_count="nunique",
            targets_supporting_rule=lambda s: json.dumps(
                sorted(pd.Series(s).dropna().astype(str).unique().tolist())
            ),
        )
        result_df = result_df.merge(
            target_support.reset_index(), on="canonical_key", how="left"
        )
    else:
        result_df["supporting_targets_count"] = 1
        result_df["targets_supporting_rule"] = json.dumps([])

    if "source_horizon" in df.columns:
        horizon_support = grouped["source_horizon"].agg(
            supporting_horizons_count="nunique",
            horizons_supporting_rule=lambda s: json.dumps(
                sorted(pd.Series(s).dropna().unique().tolist())
            ),
        )
        result_df = result_df.merge(
            horizon_support.reset_index(), on="canonical_key", how="left"
        )
    else:
        result_df["supporting_horizons_count"] = 1
        result_df["horizons_supporting_rule"] = json.dumps([])

    ic_minmax_frames = []
    for col in [c for c in ic_metric_cols if c in df.columns]:
        ic_minmax = grouped[col].agg(["min", "max"]).reset_index()
        ic_minmax = ic_minmax.rename(columns={"min": f"{col}_min", "max": f"{col}_max"})
        ic_minmax_frames.append(ic_minmax)
    for frame in ic_minmax_frames:
        result_df = result_df.merge(frame, on="canonical_key", how="left")

    if "production_classification" in df.columns:
        prod_any = grouped["production_classification"].agg(
            lambda s: "production"
            if (pd.Series(s).astype(str) == "production").any()
            else "research"
        )
        result_df["merged_production_status"] = result_df["canonical_key"].map(prod_any)
    elif "production_status" in df.columns:
        prod_any = grouped["production_status"].agg(
            lambda s: "production"
            if (pd.Series(s).astype(str) == "production").any()
            else "exploration"
        )
        result_df["merged_production_status"] = result_df["canonical_key"].map(prod_any)
    else:
        result_df["merged_production_status"] = "research"

    if "rule_type_class" in df.columns:
        merged_rule_type_class = grouped["rule_type_class"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else "rejected"
        )
        result_df["merged_rule_type_class"] = result_df["canonical_key"].map(
            merged_rule_type_class
        )
    else:
        result_df["merged_rule_type_class"] = "rejected"

    if "rule_type" in df.columns:
        merged_rule_type = grouped["rule_type"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown"
        )
        result_df["rule_type"] = result_df["canonical_key"].map(merged_rule_type)

    # Sort by supporting targets count (descending), then by composite score
    sort_cols = []
    if "supporting_targets_count" in result_df.columns:
        sort_cols.append("supporting_targets_count")
    if "composite_score" in result_df.columns:
        sort_cols.append("composite_score")
    if sort_cols:
        result_df = result_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    return result_df.reset_index(drop=True)


def analyze_cross_target_rules(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze rules that appear across multiple targets.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged rules DataFrame with provenance

    Returns
    -------
    Dict[str, Any]
        - cross_target_rules: Rules appearing in 2+ targets
        - universal_rules: Rules appearing in all 3 targets
        - target_specific_rules: Rules unique to one target
        - cross_horizon_rules: Rules appearing at multiple horizons
        - single_horizon_rules: Rules at only one horizon
    """
    if merged_df.empty:
        return {
            "cross_target_rules": [],
            "universal_rules": [],
            "target_specific_rules": [],
            "cross_horizon_rules": [],
            "single_horizon_rules": [],
            "summary": {},
        }

    result = {
        "cross_target_rules": [],
        "universal_rules": [],
        "target_specific_rules": [],
        "cross_horizon_rules": [],
        "single_horizon_rules": [],
        "summary": {},
    }

    # Check if we have the necessary columns
    if "supporting_targets_count" not in merged_df.columns:
        return result

    # Cross-target rules (2+ targets)
    cross_target_mask = merged_df["supporting_targets_count"] >= 2
    result["cross_target_rules"] = merged_df[cross_target_mask]

    # Universal rules (all 3 targets)
    universal_mask = merged_df["supporting_targets_count"] >= 3
    result["universal_rules"] = merged_df[universal_mask]

    # Target-specific rules (1 target only)
    specific_mask = merged_df["supporting_targets_count"] == 1
    result["target_specific_rules"] = merged_df[specific_mask]

    # Cross-horizon rules
    if "supporting_horizons_count" in merged_df.columns:
        cross_horizon_mask = merged_df["supporting_horizons_count"] >= 2
        result["cross_horizon_rules"] = merged_df[cross_horizon_mask]

        single_horizon_mask = merged_df["supporting_horizons_count"] == 1
        result["single_horizon_rules"] = merged_df[single_horizon_mask]

    # Summary statistics
    result["summary"] = {
        "total_rules": len(merged_df),
        "cross_target_count": len(result["cross_target_rules"]),
        "universal_count": len(result["universal_rules"]),
        "target_specific_count": len(result["target_specific_rules"]),
        "cross_horizon_count": len(result["cross_horizon_rules"]),
        "single_horizon_count": len(result["single_horizon_rules"]),
    }

    tprint(
        f"Cross-target analysis: {len(result['cross_target_rules'])} cross-target, "
        f"{len(result['universal_rules'])} universal, "
        f"{len(result['target_specific_rules'])} target-specific"
    )

    return result


def classify_rule_type(
    directional_mean_ret: float,
    mean_uplift: float,
    sign_consistency: float,
    required_hurdle: float,
    ranking_excess: float = 0.0,
    gate_uplift_threshold: float = 0.0,
    min_sign_consistency: float = 0.0,
) -> str:
    """
    Classify rule as ranking vs gate regime using return-based metrics.

    Ranking regime:
    - directional_mean_ret exceeds required_hurdle (+ optional excess)
    - sign consistency is acceptable for directional sizing

    Gate/filter regime:
    - mean_uplift is positive (or above threshold)
    - sign consistency is acceptable for binary on/off gating

    Rejected:
    - Neither ranking nor gate characteristics

    Parameters
    ----------
    directional_mean_ret : float
        Direction-aligned average return for the rule.
    mean_uplift : float
        Mean uplift vs parent context.
    sign_consistency : float
        Fraction of folds whose sign agrees with the major sign.
    required_hurdle : float
        Required return hurdle based on support/arity.

    Returns
    -------
    str
        "ranking", "gate", or "rejected"
    """
    if not np.isfinite(directional_mean_ret):
        directional_mean_ret = -np.inf
    if not np.isfinite(mean_uplift):
        mean_uplift = -np.inf
    if not np.isfinite(sign_consistency):
        sign_consistency = 0.0
    if not np.isfinite(required_hurdle):
        required_hurdle = 0.0

    min_directional_mean_ret = float(required_hurdle) + float(ranking_excess)

    if (
        directional_mean_ret > min_directional_mean_ret
        and sign_consistency >= min_sign_consistency
    ):
        return "ranking"

    if mean_uplift > gate_uplift_threshold and sign_consistency >= min_sign_consistency:
        return "gate"

    return "rejected"


def classify_rule_production_quality(
    rule: Dict[str, Any],
    min_folds: int = 3,
    min_healthy_folds: int = 2,
    min_presence_freq: float = 0.75,
    min_directional_mean_ret: float = 0.0,
    min_support_threshold: int = 50,
    min_path_quality_score: float = 0.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Classify rule as production-quality, research, or rejected.

    Production-quality rules must meet ALL of:
    - n_folds >= min_folds
    - presence_freq >= min_presence_freq
    - directional_mean_ret > min_directional_mean_ret
    - min_support_actual >= min_support_actual
    - structurally sound
    - trade_path_quality_score >= min_path_quality_score

    Classification:
    - "production": Meets all criteria for deployment
    - "research": Meets some criteria, needs further study
    - "rejected": Fails critical criteria

    Parameters
    ----------
    rule : Dict[str, Any]
        Rule dictionary with metrics
    min_folds : int
        Minimum number of folds rule must appear in (default 3)
    min_presence_freq : float
        Minimum presence frequency across folds (default 0.75)
    min_directional_mean_ret : float
        Minimum directional return edge (default 0.0)
    min_support_threshold : int
        Minimum actual support count (default 50)
    min_path_quality_score : float
        Minimum trade path quality score (default 0.0)

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        (classification, diagnostics_dict)
        classification is "production", "research", or "rejected"
    """
    diagnostics: Dict[str, Any] = {
        "checks": {},
        "failures": [],
        "warnings": [],
    }

    # Extract metrics with safe defaults
    n_folds = rule.get("n_folds", 0)
    healthy_fold_count = int(rule.get("healthy_fold_count", n_folds) or 0)
    healthy_fold_ratio = float(rule.get("healthy_fold_ratio", 1.0) or 0.0)
    presence_freq = rule.get("presence_freq", 0.0)
    directional_mean_ret = rule.get("directional_mean_ret", np.nan)
    support_actual = rule.get("min_support_actual", 0)
    # hurdle_excess REMOVED - using directional returns instead
    is_structurally_sound = rule.get("is_structurally_sound", False)
    sign_consistency = rule.get("sign_consistency", 0.0)
    trade_path_quality_score = rule.get("trade_path_quality_score", np.nan)

    if not np.isfinite(directional_mean_ret):
        directional_mean_ret = -np.inf

    # Check 1: Fold count
    fold_check = n_folds >= min_folds
    diagnostics["checks"]["n_folds"] = {
        "value": n_folds,
        "threshold": min_folds,
        "passed": fold_check,
    }
    if not fold_check:
        diagnostics["failures"].append(f"n_folds={n_folds} < {min_folds}")

    healthy_fold_check = healthy_fold_count >= min_healthy_folds
    diagnostics["checks"]["healthy_fold_count"] = {
        "value": healthy_fold_count,
        "threshold": min_healthy_folds,
        "passed": healthy_fold_check,
    }
    diagnostics["checks"]["healthy_fold_ratio"] = {
        "value": healthy_fold_ratio,
        "threshold": float(min_healthy_folds) / max(float(min_folds), 1.0),
        "passed": healthy_fold_check,
    }
    if not healthy_fold_check:
        diagnostics["failures"].append(
            f"healthy_fold_count={healthy_fold_count} < {min_healthy_folds}"
        )
    elif healthy_fold_count < n_folds:
        diagnostics["warnings"].append(
            f"partial_fold_health={healthy_fold_count}/{n_folds}"
        )

    # Check 2: Presence frequency
    presence_check = presence_freq >= min_presence_freq
    diagnostics["checks"]["presence_freq"] = {
        "value": presence_freq,
        "threshold": min_presence_freq,
        "passed": presence_check,
    }
    if not presence_check:
        diagnostics["failures"].append(
            f"presence_freq={presence_freq:.3f} < {min_presence_freq}"
        )

    # Check 3: Directional return edge
    edge_check = directional_mean_ret > min_directional_mean_ret
    diagnostics["checks"]["directional_mean_ret"] = {
        "value": directional_mean_ret,
        "threshold": min_directional_mean_ret,
        "passed": edge_check,
    }
    if not edge_check:
        diagnostics["failures"].append(
            f"directional_mean_ret={directional_mean_ret:.6f} <= {min_directional_mean_ret:.6f}"
        )

    # Check 5: Support count
    support_check = support_actual >= min_support_threshold
    diagnostics["checks"]["min_support_actual"] = {
        "value": support_actual,
        "threshold": min_support_threshold,
        "passed": support_check,
    }
    if not support_check:
        diagnostics["failures"].append(
            f"min_support_actual={support_actual} < {min_support_threshold}"
        )

    # Check 6: Structural soundness
    structural_check = is_structurally_sound
    diagnostics["checks"]["is_structurally_sound"] = {
        "value": is_structurally_sound,
        "passed": structural_check,
    }
    if not structural_check:
        reason = rule.get("rejection_reason", "not_structurally_sound")
        diagnostics["failures"].append(f"rule_rejected: {reason}")

    # Check 7: Trade path quality score
    if not np.isfinite(trade_path_quality_score):
        trade_path_quality_score = -np.inf

    path_quality_check = trade_path_quality_score >= min_path_quality_score
    diagnostics["checks"]["trade_path_quality_score"] = {
        "value": trade_path_quality_score,
        "threshold": min_path_quality_score,
        "passed": path_quality_check,
    }
    if not path_quality_check:
        diagnostics["failures"].append(
            f"trade_path_quality_score={trade_path_quality_score:.6f} < {min_path_quality_score:.6f}"
        )

    # Check 8: Sign consistency (warning only) - disabled
    # if sign_consistency < 0.75:
    #     diagnostics["warnings"].append(f"low_sign_consistency={sign_consistency:.3f}")

    # Determine classification
    critical_failures = [
        f
        for f in diagnostics["failures"]
        if "n_folds" in f
        or "structurally_sound" in f
        or "trade_path_quality_score" in f
    ]

    if critical_failures:
        # Critical failures -> rejected
        classification = "rejected"
    elif len(diagnostics["failures"]) == 0:
        # All checks passed -> production
        classification = "production"
    elif len(diagnostics["failures"]) <= 2:
        # Minor failures -> research
        classification = "research"
    else:
        # Too many failures -> rejected
        classification = "rejected"

    diagnostics["classification"] = classification

    return classification, diagnostics


def compute_overall_target_quality_score(
    mean_directional_ret: float,
    mean_sign_consistency: float,
    mean_hurdle_excess: float,
    entropy_reduction: float,
    production_rule_count: int,
    total_rule_count: int,
) -> float:
    """
    Compute overall target quality score.

    Weighted combination:
    - 35%: mean_directional_ret (economic edge)
    - 25%: mean_sign_consistency (stability)
    - 15%: mean_hurdle_excess (economic surplus over hurdle)
    - 15%: entropy_reduction (regime separation)
    - 10%: production_rule_ratio (rule quality)

    Parameters
    ----------
    mean_directional_ret : float
        Mean directional return edge across accepted rules
    mean_sign_consistency : float
        Mean sign consistency across accepted rules
    mean_hurdle_excess : float
        Mean excess return over rule-specific hurdle
    entropy_reduction : float
        Entropy reduction from regime
    production_rule_count : int
        Number of production-quality rules
    total_rule_count : int
        Total number of rules

    Returns
    -------
    float
        Score in [0, 1] range
    """
    edge_score = np.clip((mean_directional_ret + 0.01) / 0.06, 0, 1)
    stability_score = np.clip(mean_sign_consistency, 0, 1)
    hurdle_score = np.clip((mean_hurdle_excess + 0.005) / 0.03, 0, 1)

    # Entropy reduction can be negative, normalize
    # Assume range [-0.5, 0.5]
    entropy_score = (entropy_reduction + 0.5) / 1.0
    entropy_score = np.clip(entropy_score, 0, 1)

    # Rule ratio: already in [0, 1]
    rule_ratio = production_rule_count / max(total_rule_count, 1)

    # Weighted combination
    score = (
        0.35 * edge_score
        + 0.25 * stability_score
        + 0.15 * hurdle_score
        + 0.15 * entropy_score
        + 0.10 * rule_ratio
    )

    return float(np.clip(score, 0, 1))


def create_triad_run_summary(
    all_results: List[Dict[str, Any]],
    merged_rules: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create summary of triad run including:
    - Per-target production rule counts
    - Per-horizon production rule counts
    - Cross-target rule counts
    - Overall quality assessment

    Parameters
    ----------
    all_results : List[Dict[str, Any]]
        List of result dicts from run_side_pipeline()
    merged_rules : Optional[pd.DataFrame]
        Merged/deduplicated rules DataFrame
    output_path : Optional[str]
        Path to save the summary JSON

    Returns
    -------
    Dict[str, Any]
        Summary dictionary with production rule counts and quality metrics
    """
    summary: Dict[str, Any] = {
        "per_target_summary": {},
        "per_horizon_summary": {},
        "cross_target_summary": {},
        "overall_quality": {},
    }

    # Per-target summary
    target_names = set()
    for result in all_results:
        target_name = result.get("target_name", "unknown")
        target_names.add(target_name)

        if target_name not in summary["per_target_summary"]:
            summary["per_target_summary"][target_name] = {
                "total_rules": 0,
                "production_rules": 0,
                "research_rules": 0,
                "rejected_rules": 0,
                "ranking_rules": 0,
                "gate_rules": 0,
            }

        accepted_rules = result.get("accepted_rules", [])
        summary["per_target_summary"][target_name]["total_rules"] += len(accepted_rules)

        for rule in accepted_rules:
            classification = rule.get("production_classification", "research")
            rule_type = rule.get("rule_type_class", "rejected")

            if classification == "production":
                summary["per_target_summary"][target_name]["production_rules"] += 1
            elif classification == "research":
                summary["per_target_summary"][target_name]["research_rules"] += 1
            else:
                summary["per_target_summary"][target_name]["rejected_rules"] += 1

            if rule_type == "ranking":
                summary["per_target_summary"][target_name]["ranking_rules"] += 1
            elif rule_type == "gate":
                summary["per_target_summary"][target_name]["gate_rules"] += 1

    # Per-horizon summary
    horizons = set()
    for result in all_results:
        horizon = result.get("horizon", 0)
        horizons.add(horizon)

        horizon_key = f"h{horizon}"
        if horizon_key not in summary["per_horizon_summary"]:
            summary["per_horizon_summary"][horizon_key] = {
                "total_rules": 0,
                "production_rules": 0,
                "research_rules": 0,
                "rejected_rules": 0,
                "ranking_rules": 0,
                "gate_rules": 0,
            }

        accepted_rules = result.get("accepted_rules", [])
        summary["per_horizon_summary"][horizon_key]["total_rules"] += len(
            accepted_rules
        )

        for rule in accepted_rules:
            classification = rule.get("production_classification", "research")
            rule_type = rule.get("rule_type_class", "rejected")

            if classification == "production":
                summary["per_horizon_summary"][horizon_key]["production_rules"] += 1
            elif classification == "research":
                summary["per_horizon_summary"][horizon_key]["research_rules"] += 1
            else:
                summary["per_horizon_summary"][horizon_key]["rejected_rules"] += 1

            if rule_type == "ranking":
                summary["per_horizon_summary"][horizon_key]["ranking_rules"] += 1
            elif rule_type == "gate":
                summary["per_horizon_summary"][horizon_key]["gate_rules"] += 1

    # Cross-target summary (from merged rules)
    if merged_rules is not None and not merged_rules.empty:
        cross_target_count = 0
        universal_count = 0
        production_cross_target_count = 0

        if "supporting_targets_count" in merged_rules.columns:
            cross_target_mask = merged_rules["supporting_targets_count"] >= 2
            cross_target_count = int(cross_target_mask.sum())

            universal_mask = merged_rules["supporting_targets_count"] >= 3
            universal_count = int(universal_mask.sum())

            # Production rules that are cross-target
            if "merged_production_status" in merged_rules.columns:
                production_cross_target = merged_rules[
                    cross_target_mask
                    & (merged_rules["merged_production_status"] == "production")
                ]
                production_cross_target_count = len(production_cross_target)

        summary["cross_target_summary"] = {
            "cross_target_rules": cross_target_count,
            "universal_rules": universal_count,
            "production_cross_target_rules": production_cross_target_count,
            "total_merged_rules": len(merged_rules),
        }

    # Overall quality assessment
    total_rules = sum(t["total_rules"] for t in summary["per_target_summary"].values())
    total_production = sum(
        t["production_rules"] for t in summary["per_target_summary"].values()
    )
    total_research = sum(
        t["research_rules"] for t in summary["per_target_summary"].values()
    )
    total_ranking = sum(
        t["ranking_rules"] for t in summary["per_target_summary"].values()
    )
    total_gate = sum(t["gate_rules"] for t in summary["per_target_summary"].values())

    summary["overall_quality"] = {
        "total_rules": total_rules,
        "production_rules": total_production,
        "research_rules": total_research,
        "production_ratio": total_production / max(total_rules, 1),
        "ranking_rules": total_ranking,
        "gate_rules": total_gate,
        "ranking_to_gate_ratio": total_ranking / max(total_gate, 1),
    }

    # Compute overall quality score
    if all_results:
        # Aggregate metrics across all results
        all_directional_ret = []
        all_sign_consistency = []
        all_hurdle_excess = []
        all_entropy_reduction = []

        for result in all_results:
            for rule in result.get("accepted_rules", []):
                if np.isfinite(rule.get("directional_mean_ret", np.nan)):
                    all_directional_ret.append(rule["directional_mean_ret"])
                if np.isfinite(rule.get("sign_consistency", np.nan)):
                    all_sign_consistency.append(rule["sign_consistency"])
                if np.isfinite(rule.get("hurdle_excess", np.nan)):
                    all_hurdle_excess.append(rule["hurdle_excess"])
                if np.isfinite(rule.get("entropy_reduction", np.nan)):
                    all_entropy_reduction.append(rule["entropy_reduction"])

        if all_directional_ret:
            overall_score = compute_overall_target_quality_score(
                mean_directional_ret=float(np.mean(all_directional_ret)),
                mean_sign_consistency=(
                    float(np.mean(all_sign_consistency))
                    if all_sign_consistency
                    else 0.0
                ),
                mean_hurdle_excess=(
                    float(np.mean(all_hurdle_excess)) if all_hurdle_excess else 0.0
                ),
                entropy_reduction=(
                    float(np.mean(all_entropy_reduction))
                    if all_entropy_reduction
                    else 0.0
                ),
                production_rule_count=total_production,
                total_rule_count=total_rules,
            )
            summary["overall_quality"]["quality_score"] = overall_score

    # Print summary
    tprint("=" * 60)
    tprint("TRIAD RUN SUMMARY")
    tprint("=" * 60)
    tprint(f"Total rules discovered: {total_rules}")
    tprint(
        f"Production-quality rules: {total_production} ({100*total_production/max(total_rules,1):.1f}%)"
    )
    tprint(f"Research-grade rules: {total_research}")
    tprint(f"Ranking regimes: {total_ranking}")
    tprint(f"Gate regimes: {total_gate}")
    tprint("")
    tprint("Per-target breakdown:")
    for target_name, target_summary in summary["per_target_summary"].items():
        tprint(
            f"  {target_name}: {target_summary['production_rules']} production / {target_summary['total_rules']} total"
        )
    tprint("")
    tprint("Per-horizon breakdown:")
    for horizon_key, horizon_summary in summary["per_horizon_summary"].items():
        tprint(
            f"  {horizon_key}: {horizon_summary['production_rules']} production / {horizon_summary['total_rules']} total"
        )
    if "quality_score" in summary["overall_quality"]:
        tprint(
            f"\nOverall quality score: {summary['overall_quality']['quality_score']:.3f}"
        )
    tprint("=" * 60)

    # Save to file if path provided
    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        tprint(f"Saved triad run summary to {output_path}")

    return summary


class MaskAssessor:
    def __init__(
        self,
        metadata: List[FeatureMetadata],
        cfg: Dict[str, Any],
        mask_resolver: Optional[CanonicalRuleMaskResolver] = None,
    ):
        self.metadata = metadata
        self.cfg = cfg
        self.mask_resolver = mask_resolver
        self._ridge_feature_indices_cache: Optional[np.ndarray] = None

    def _get_ridge_feature_indices(self) -> np.ndarray:
        cached = self._ridge_feature_indices_cache
        if cached is not None:
            return cached
        # Exclude time-based features from ridge/assessor feature selection
        time_keys_set = set(TIME_FEATURE_KEYS)
        test_keys_set = set(TEST_FEATURE_KEYS) - time_keys_set
        requested_columns = tuple(
            self.cfg.get(
                "miner_target_residualization_columns",
                MINER_TARGET_RESIDUALIZATION_COLUMNS,
            )
        )
        residualization_source_names: Set[str] = set()
        for requested_name in requested_columns:
            candidates = MINER_TARGET_RESIDUALIZATION_ALIAS_MAP.get(
                str(requested_name), (str(requested_name),)
            )
            for candidate in candidates:
                if any(m.source_name == candidate for m in self.metadata):
                    residualization_source_names.add(str(candidate))
                    break
        assessor_feature_names = test_keys_set | residualization_source_names
        ridge_feats = [
            i
            for i, m in enumerate(self.metadata)
            if (
                getattr(m, "source_name", None) in assessor_feature_names
                or getattr(m, "feature_name", None) in assessor_feature_names
            )
        ]
        tprint(
            "MaskAssessor feature selector: "
            f"requested_matches={len(ridge_feats)} requested_names={len(assessor_feature_names)} "
            f"total_stage_a_features={len(self.metadata)}"
        )
        cached = np.asarray(ridge_feats, dtype=np.int32)
        self._ridge_feature_indices_cache = cached
        return cached

    def _build_ridge_design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Build the assessor design matrix without re-slicing when X already matches
        the assessor metadata one-for-one.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D assessor matrix, got shape={X_arr.shape}")

        ridge_feats = self._get_ridge_feature_indices()
        if ridge_feats.size == 0:
            return np.empty((len(X_arr), 0), dtype=np.float32)

        if X_arr.shape[1] == len(self.metadata):
            X_ridge = np.asarray(X_arr, dtype=np.float32, order="C")
        else:
            X_ridge = np.asarray(X_arr[:, ridge_feats], dtype=np.float32, order="C")

        if X_ridge.shape[1] != ridge_feats.size:
            raise ValueError(
                "Assessor design-matrix width mismatch: "
                f"expected={ridge_feats.size} actual={X_ridge.shape[1]}"
            )

        return X_ridge

    @staticmethod
    def _compute_total_symbol_days(data: pd.DataFrame) -> Optional[float]:
        """Precomputes the total_symbol_days for a given dataset."""
        if "timestamp" not in data.columns or "symbol" not in data.columns:
            return None

        start_ts = time.perf_counter()
        timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
        valid_rows = timestamps.notna().to_numpy()
        if not np.any(valid_rows):
            return None

        symbols = data.loc[valid_rows, "symbol"].astype(str).to_numpy()
        day_values = (
            timestamps.loc[valid_rows].dt.floor("D").to_numpy(dtype="datetime64[ns]")
        )
        symbol_codes, _ = pd.factorize(symbols, sort=False)
        day_codes, _ = pd.factorize(day_values, sort=False)
        pair_codes = np.stack(
            [
                symbol_codes.astype(np.int64, copy=False),
                day_codes.astype(np.int64, copy=False),
            ],
            axis=1,
        )
        _, pair_counts = np.unique(pair_codes, axis=0, return_counts=True)
        typical_rows_per_symbol_day = float(np.median(pair_counts))
        if (
            not np.isfinite(typical_rows_per_symbol_day)
            or typical_rows_per_symbol_day <= 0
        ):
            return None

        total_symbol_days = float(valid_rows.sum()) / typical_rows_per_symbol_day
        if total_symbol_days <= 0:
            return None

        elapsed = time.perf_counter() - start_ts
        tprint(
            "Total symbol-day precompute: "
            f"rows={int(valid_rows.sum())} unique_symbol_days={int(len(pair_counts))} "
            f"typical_rows_per_symbol_day={typical_rows_per_symbol_day:.2f} "
            f"total_symbol_days={total_symbol_days:.2f} elapsed={elapsed:.2f}s"
        )

        return total_symbol_days

    @staticmethod
    def _compute_avg_trades_per_day(
        mask: np.ndarray, total_symbol_days: Optional[float]
    ) -> float:
        selected_count = int(np.sum(mask))
        if selected_count == 0:
            return 0.0

        if total_symbol_days is None or total_symbol_days <= 0:
            return float(selected_count)

        trades_per_day_per_symbol = selected_count / total_symbol_days
        return float(trades_per_day_per_symbol * 10.0)

    @staticmethod
    def _compute_top_quartile_precision(
        oof_preds: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        tp_f: np.ndarray,
        fwd_ret_threshold: float = 0.5,
        top_pct: float = 0.75,
        min_samples: int = 20,
    ) -> float:
        """
        Compute precision of top-quartile Ridge predictions.
        
        Precision = % of predictions in top 25% (by rank) that either:
        - Hit take-profit first (tp_f == 1), OR
        - Have forward return > fwd_ret_threshold
        
        Parameters
        ----------
        oof_preds : np.ndarray
            Ridge out-of-fold predictions
        y : np.ndarray
            Forward returns (for threshold check)
        mask : np.ndarray
            Boolean mask defining subset
        tp_f : np.ndarray
            Take-profit first hit flags (1=TP hit first)
        fwd_ret_threshold : float
            Forward return threshold for positive outcome (default 0.5)
        top_pct : float
            Percentile threshold for top predictions (default 0.75 = top 25%)
        min_samples : int
            Minimum samples required for valid computation
            
        Returns
        -------
        float
            Top-quartile precision in [0, 1], or NaN if insufficient samples
        """
        # Get valid samples within mask
        valid_mask = (
            mask.astype(bool) 
            & np.isfinite(oof_preds) 
            & np.isfinite(y)
            & np.isfinite(tp_f)
        )
        
        if np.sum(valid_mask) < min_samples:
            return np.nan
        
        # Get predictions within mask
        preds_masked = oof_preds[valid_mask]
        y_masked = y[valid_mask]
        tp_f_masked = tp_f[valid_mask]
        
        # Compute percentile threshold (top 25% by default)
        threshold = np.percentile(preds_masked, top_pct * 100)
        
        # Select top predictions
        top_mask = preds_masked >= threshold
        n_top = np.sum(top_mask)
        
        if n_top < 5:  # Need at least 5 samples in top quartile
            return np.nan
        
        # Positive outcome: TP hit first OR fwd_ret > threshold
        positive_outcome = (tp_f_masked == 1) | (y_masked > fwd_ret_threshold)
        
        # Precision = positive outcomes in top quartile / total in top quartile
        precision = np.sum(positive_outcome[top_mask]) / n_top
        
        return float(precision)

    @staticmethod
    def _compute_oof_learnability_score(
        oof_preds: np.ndarray,
        y: np.ndarray,
        coverage_denominator: np.ndarray,
        min_predicted_points: int = 100,
    ) -> Tuple[float, float]:
        # Ensure we only compute metrics on the exact masked subset.
        predicted_mask = np.isfinite(oof_preds) & np.isfinite(y) & coverage_denominator.astype(bool)
        coverage_base_mask = np.isfinite(y) & coverage_denominator.astype(bool)
        coverage_base = int(np.sum(coverage_base_mask))
        predicted_count = int(np.sum(predicted_mask))
        coverage = float(predicted_count / coverage_base) if coverage_base > 0 else 0.0

        if predicted_count < min_predicted_points:
            return np.nan, coverage

        y_predicted = y[predicted_mask]
        preds = oof_preds[predicted_mask]
        unique_y = np.unique(y_predicted)
        is_binary_target = unique_y.size <= 2 and np.all(np.isin(unique_y, [0.0, 1.0]))

        if is_binary_target:
            if unique_y.size < 2:
                return np.nan, coverage
            try:
                auc = roc_auc_score(y_predicted, preds)
                return max(auc, 1.0 - auc), coverage
            except ValueError:
                return np.nan, coverage

        if np.nanstd(y_predicted) < 1e-12 or np.nanstd(preds) < 1e-12:
            return np.nan, coverage

        score = np.corrcoef(y_predicted, preds)[0, 1]
        if not np.isfinite(score):
            return np.nan, coverage
        return float(score), coverage

    @staticmethod
    def _compute_oof_classification_metrics(
        oof_preds: np.ndarray,
        y: np.ndarray,
        coverage_denominator: np.ndarray,
        min_predicted_points: int = 100,
    ) -> Dict[str, float]:
        # Ensure we only compute metrics on the exact masked subset.
        predicted_mask = np.isfinite(oof_preds) & np.isfinite(y) & coverage_denominator.astype(bool)
        coverage_base_mask = np.isfinite(y) & coverage_denominator.astype(bool)
        coverage_base = int(np.sum(coverage_base_mask))
        predicted_count = int(np.sum(predicted_mask))
        coverage = float(predicted_count / coverage_base) if coverage_base > 0 else 0.0

        if predicted_count < min_predicted_points:
            return {
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "coverage": coverage,
            }

        y_predicted = np.asarray(y[predicted_mask], dtype=np.float32)
        preds = np.asarray(oof_preds[predicted_mask], dtype=np.float32)
        y_binary = (y_predicted > 0.0).astype(np.int8)

        if np.unique(y_binary).size < 2 or np.nanstd(preds) < 1e-12:
            return {
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "coverage": coverage,
            }

        try:
            roc_auc = float(roc_auc_score(y_binary, preds))
        except ValueError:
            roc_auc = np.nan
        try:
            pr_auc = float(average_precision_score(y_binary, preds))
        except ValueError:
            pr_auc = np.nan

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "coverage": coverage,
        }

    def _ridge_learnability_thresholds(
        self, y: np.ndarray
    ) -> Tuple[bool, int, int, int]:
        """
        Return (is_binary_target, min_train, min_val, min_predicted_points).

        Triad targets are continuous and should not use binary-positive thresholds.
        """
        finite_y = np.asarray(y, dtype=np.float32)
        finite_y = finite_y[np.isfinite(finite_y)]
        if finite_y.size == 0:
            return False, 200, 100, 100

        unique_y = np.unique(finite_y)
        is_binary_target = unique_y.size <= 2 and np.all(np.isin(unique_y, [0.0, 1.0]))
        if is_binary_target:
            min_train = int(self.cfg.get("learnability_min_train_positives", 1000))
            min_val = int(self.cfg.get("learnability_min_val_positives", 1000))
            min_pred = int(
                self.cfg.get("learnability_min_predicted_points_binary", 1000)
            )
        else:
            min_train = int(
                self.cfg.get("learnability_min_train_samples_continuous", 500)
            )
            min_val = int(self.cfg.get("learnability_min_val_samples_continuous", 100))
            min_pred = int(
                self.cfg.get("learnability_min_predicted_points_continuous", 100)
            )
        return is_binary_target, min_train, min_val, min_pred

    @staticmethod
    def _pct_rank(series: pd.Series) -> pd.Series:
        if series.empty:
            return pd.Series(dtype=np.float32, index=series.index)
        return series.rank(method="average", pct=True).astype(np.float32)

    def _get_step2_model_profile_params(self, model_profile: str) -> Dict[str, Any]:
        profile = str(model_profile).lower()
        if profile == "strong":
            boosting_type = str(self.cfg.get("step2_strong_boosting_type", "goss"))
            params = {
                "max_depth": int(self.cfg.get("step2_strong_max_depth", 4)),
                "n_estimators": int(self.cfg.get("step2_strong_n_estimators", 100)),
                "min_child_samples": int(self.cfg.get("step2_strong_min_child_samples", 20)),
                "min_data_in_leaf": int(self.cfg.get("step2_strong_min_data_in_leaf", 20)),
                "lambda_l1": float(self.cfg.get("step2_strong_lambda_l1", 10.0)),
                "lambda_l2": float(self.cfg.get("step2_strong_lambda_l2", 2.0)),
                "min_gain_to_split": float(self.cfg.get("step2_strong_min_gain_to_split", 0.001)),
                "subsample": float(self.cfg.get("step2_strong_subsample", 0.7)),
                "subsample_freq": int(self.cfg.get("step2_strong_subsample_freq", 1)),
                "feature_fraction": float(self.cfg.get("step2_strong_feature_fraction", 0.7)),
                "boosting_type": boosting_type,
            }
            if boosting_type.lower() == "goss":
                params["subsample"] = 1.0
                params["subsample_freq"] = 0
            return params
        return {
            "max_depth": int(self.cfg.get("step2_weak_max_depth", 3)),
            "n_estimators": int(self.cfg.get("step2_weak_n_estimators", 5)),
            "min_child_samples": int(self.cfg.get("step2_weak_min_child_samples", 20)),
            "min_data_in_leaf": int(self.cfg.get("step2_weak_min_data_in_leaf", 20)),
            "lambda_l1": float(self.cfg.get("step2_weak_lambda_l1", 0.0)),
            "lambda_l2": float(self.cfg.get("step2_weak_lambda_l2", 0.0)),
            "min_gain_to_split": float(self.cfg.get("step2_weak_min_gain_to_split", 0.001)),
            "subsample": float(self.cfg.get("step2_weak_subsample", 1.0)),
            "subsample_freq": int(self.cfg.get("step2_weak_subsample_freq", 0)),
            "feature_fraction": float(self.cfg.get("step2_weak_feature_fraction", 1.0)),
            "boosting_type": str(self.cfg.get("step2_weak_boosting_type", "gbdt")),
        }

    def _compute_score_for_best_params(
        self,
        assessment_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute score_for_best_params with 9 weighted components.
        
        Formula:
        - 0.3 * overall_mask_uplift
        - 0.075 * Directional accuracy @ ATR * (sqrt(horizon) / 2)
        - 0.075 * Directional accuracy @ ATR * (sqrt(horizon))
        - 0.1 * Directional accuracy (sign) at horizon time
        - 0.1 * IC stability (across months & assets)
        - 0.1 * Rank IC
        - 0.1 * IC @ Top10% predictions
        - 0.1 * Conditional mean return
        - 0.1 * Return-per-risk inside regime (mean / downside deviation)
        
        Then: score_for_best_params = base_score × sqrt(support%)
        
        All terms: winsorize at p5/95th, MinMax 0.1-1 between all candidates.
        """
        if assessment_df.empty:
            return assessment_df
        
        result = assessment_df.copy()
        
        # Helper to winsorize and minmax scale to [0.1, 1.0]
        def _winsorize_minmax(series: pd.Series, low_p=5, high_p=95) -> pd.Series:
            valid = series.replace([np.inf, -np.inf], np.nan).dropna()
            if valid.empty:
                return pd.Series(0.55, index=series.index)  # middle of 0.1-1.0
            
            p_low = np.percentile(valid, low_p)
            p_high = np.percentile(valid, high_p)
            clipped = np.clip(series, p_low, p_high)
            
            c_min = float(clipped.min())
            c_max = float(clipped.max())
            span = max(c_max - c_min, 1e-9)
            
            # Scale to [0.1, 1.0]
            scaled = 0.1 + 0.9 * (clipped - c_min) / span
            return scaled.fillna(0.55)
        
        # 1. overall_mask_uplift (weight: 0.3)
        result["_s1_overall_mask_uplift"] = _winsorize_minmax(result.get("overall_mask_uplift", pd.Series(np.nan, index=result.index)))
        
        # 2 & 3. Directional accuracy @ ATR components (weights: 0.075 each)
        # Using decile_monotonic_spearman as proxy for directional accuracy
        horizon = result.get("source_horizon", pd.Series(100, index=result.index))
        sqrt_h = np.sqrt(pd.to_numeric(horizon, errors="coerce").fillna(100))
        
        dir_acc = result.get("decile_monotonic_spearman", pd.Series(np.nan, index=result.index))
        result["_s2_dir_acc_atr_half"] = _winsorize_minmax(dir_acc * (sqrt_h / 2))
        result["_s3_dir_acc_atr_full"] = _winsorize_minmax(dir_acc * sqrt_h)
        
        # 4. Directional accuracy (sign) at horizon time (weight: 0.1)
        # Using sign_consistency as proxy
        result["_s4_dir_acc_sign"] = _winsorize_minmax(result.get("sign_consistency", pd.Series(np.nan, index=result.index)))
        
        # 5. IC stability (weight: 0.1)
        # Using fold_sign_consistency as proxy for IC stability across folds
        result["_s5_ic_stability"] = _winsorize_minmax(result.get("fold_sign_consistency", pd.Series(np.nan, index=result.index)))
        
        # 6. Rank IC (weight: 0.1)
        # Using mask_oof_corr (the IC within mask) as proxy
        result["_s6_rank_ic"] = _winsorize_minmax(result.get("mask_oof_corr", pd.Series(np.nan, index=result.index)))
        
        # 7. IC @ Top10% predictions (weight: 0.1)
        # Using top_decile_mean_target correlation
        result["_s7_ic_top10"] = _winsorize_minmax(result.get("top_decile_mean_target", pd.Series(np.nan, index=result.index)))
        
        # 8. Conditional mean return (weight: 0.1)
        # Using mean_ret_mask
        result["_s8_conditional_mean"] = _winsorize_minmax(result.get("mean_ret_mask", pd.Series(np.nan, index=result.index)))
        
        # 9. Return-per-risk inside regime (weight: 0.1)
        # Computing mean_ret_mask / downside_std
        mean_ret = pd.to_numeric(result.get("mean_ret_mask", pd.Series(np.nan, index=result.index)), errors="coerce")
        # Use weekly_sortino as proxy for return-per-risk
        ret_per_risk = pd.to_numeric(result.get("weekly_sortino", pd.Series(np.nan, index=result.index)), errors="coerce")
        result["_s9_ret_per_risk"] = _winsorize_minmax(ret_per_risk)
        
        # Compute base score with weights
        base_score = (
            0.3 * result["_s1_overall_mask_uplift"]
            + 0.075 * result["_s2_dir_acc_atr_half"]
            + 0.075 * result["_s3_dir_acc_atr_full"]
            + 0.1 * result["_s4_dir_acc_sign"]
            + 0.1 * result["_s5_ic_stability"]
            + 0.1 * result["_s6_rank_ic"]
            + 0.1 * result["_s7_ic_top10"]
            + 0.1 * result["_s8_conditional_mean"]
            + 0.1 * result["_s9_ret_per_risk"]
        )
        
        # Apply linear support scaling: 5% → 1.0, 20% → 1.2 (reward higher support)
        support_pct = pd.to_numeric(result.get("support_pct", pd.Series(0.05, index=result.index)), errors="coerce").fillna(0.05)
        # Normalize to [0, 1] range where 5% = 0 and 20% = 1
        support_normalized = ((support_pct - 0.05) / 0.15).clip(0, 1)
        support_factor = 1.0 + 0.2 * support_normalized
        result["score_for_best_params"] = base_score * support_factor
        
        # Clean up temporary columns
        temp_cols = [c for c in result.columns if c.startswith("_s")]
        result = result.drop(columns=temp_cols)
        
        return result

    def _apply_final_topk_selection(
        self,
        assessment_df: pd.DataFrame,
        mask_lookup: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        if assessment_df.empty:
            return assessment_df

        result = assessment_df.copy()
        result["selected_for_final_registry"] = False
        result["final_candidate_rank_score"] = np.nan
        result["final_selection_order"] = np.nan
        result["trade_composite_score"] = np.nan
        if "stage1_composite_score" not in result.columns:
            result["stage1_composite_score"] = pd.to_numeric(
                result.get("composite_score", np.nan), errors="coerce"
            )

        eligible = result.copy()
        if eligible.empty:
            return result

        weekly_vol = pd.to_numeric(eligible.get("weekly_volatility", np.nan), errors="coerce")
        weekly_sortino = pd.to_numeric(eligible.get("weekly_sortino", np.nan), errors="coerce")
        fold_pnl_std = pd.to_numeric(eligible.get("fold_pnl_std", np.nan), errors="coerce")
        if fold_pnl_std.isna().all():
            fold_pnl_std = weekly_vol.copy()

        pnl_rank = self._pct_rank(pd.to_numeric(eligible["ridge_pnl_raw"], errors="coerce").fillna(-np.inf))
        sortino_7d_rank = self._pct_rank(
            pd.to_numeric(eligible.get("ridge_trade_sortino_7d", np.nan), errors="coerce").fillna(0.0)
        )
        sortino_30d_rank = self._pct_rank(
            pd.to_numeric(eligible.get("ridge_trade_sortino_30d", np.nan), errors="coerce").fillna(0.0)
        )
        sortino_90d_rank = self._pct_rank(
            pd.to_numeric(eligible.get("ridge_trade_sortino_90d", np.nan), errors="coerce").fillna(0.0)
        )
        sortino_composite_rank = self._pct_rank(
            pd.to_numeric(eligible["ridge_trade_sortino_composite"], errors="coerce").fillna(0.0)
        )
        multi_horizon_sortino_rank = (
            0.20 * sortino_7d_rank
            + 0.30 * sortino_30d_rank
            + 0.30 * sortino_90d_rank
            + 0.20 * sortino_composite_rank
        )
        weekly_vol_rank = 1.0 - self._pct_rank(
            weekly_vol.fillna(weekly_vol.max() if weekly_vol.notna().any() else 0.0)
        )
        weekly_sortino_rank = self._pct_rank(weekly_sortino.fillna(0.0))
        pos_fold_rank = self._pct_rank(pd.to_numeric(eligible.get("positive_fold_fraction", np.nan), errors="coerce").fillna(0.0))
        fold_std_rank = 1.0 - self._pct_rank(fold_pnl_std.fillna(fold_pnl_std.max() if fold_pnl_std.notna().any() else 0.0))
        fold_stability = 0.7 * pos_fold_rank + 0.3 * fold_std_rank
        active_symbol_day_rank = self._pct_rank(
            pd.to_numeric(
                eligible.get("avg_pnl_per_active_symbol_day", np.nan),
                errors="coerce",
            ).fillna(-np.inf)
        )
        trade_density_rank = self._pct_rank(
            pd.to_numeric(
                eligible.get("trades_per_symbol_day_above_threshold_star", np.nan),
                errors="coerce",
            ).fillna(0.0)
        )
        weekly_stability_rank = 0.5 * weekly_sortino_rank + 0.5 * weekly_vol_rank

        eligible["trade_composite_score"] = (
            0.35 * pnl_rank
            + 0.25 * multi_horizon_sortino_rank
            + 0.15 * active_symbol_day_rank
            + 0.10 * trade_density_rank
            + 0.10 * weekly_stability_rank
            + 0.05 * fold_stability
        )

        # Compute score_for_best_params if not already present
        if "score_for_best_params" not in result.columns:
            result = self._compute_score_for_best_params(result)

        if "score_for_best_params" in result.columns:
            eligible["score_for_best_params"] = pd.to_numeric(
                result.loc[eligible.index, "score_for_best_params"], errors="coerce"
            )
        else:
            eligible["score_for_best_params"] = np.nan

        # Use score_for_best_params as the base score for Pareto selection, with robust fallbacks
        eligible["final_candidate_rank_score"] = pd.to_numeric(
            eligible["score_for_best_params"], errors="coerce"
        )
        missing_rank = ~np.isfinite(eligible["final_candidate_rank_score"].to_numpy())
        if np.any(missing_rank):
            fallback_trade = pd.to_numeric(
                eligible.get("trade_composite_score", np.nan), errors="coerce"
            ).to_numpy()
            fallback_stage = pd.to_numeric(
                eligible.get("composite_score", np.nan), errors="coerce"
            ).to_numpy()
            final_rank = eligible["final_candidate_rank_score"].to_numpy(dtype=float, copy=True)
            final_rank[missing_rank] = fallback_trade[missing_rank]
            still_missing = ~np.isfinite(final_rank)
            final_rank[still_missing] = fallback_stage[still_missing]
            eligible["final_candidate_rank_score"] = final_rank

        eligible["fold_stability"] = fold_stability.astype(np.float32)
        
        overlap_penalty = float(self.cfg.get("final_selection_overlap_penalty", 0.35))
        support_overlap_weight = float(self.cfg.get("final_selection_support_overlap_weight", 0.5))
        ic_overlap_weight = float(self.cfg.get("final_selection_ic_overlap_weight", 0.5))
        top_k = int(self.cfg.get("final_selected_rule_cap", 20))
        selected: List[Any] = []
        remaining = eligible.sort_values("final_candidate_rank_score", ascending=False).index.tolist()

        def _ic_series(idx: Any) -> np.ndarray:
            """Get IC series for a rule, returning empty array if not available."""
            ic_data = eligible.loc[idx, "ic_series"] if "ic_series" in eligible.columns else None
            if ic_data is None:
                return np.array([])
            if isinstance(ic_data, np.ndarray):
                return ic_data
            if isinstance(ic_data, (list, tuple)):
                return np.asarray(ic_data, dtype=np.float32)
            return np.array([])

        ic_series_cache = {idx: _ic_series(idx) for idx in eligible.index}

        def _pair_overlap(idx_a: Any, idx_b: Any) -> float:
            key_a = str(eligible.loc[idx_a, "canonical_key"])
            key_b = str(eligible.loc[idx_b, "canonical_key"])
            mask_a = np.asarray(mask_lookup.get(key_a), dtype=bool)
            mask_b = np.asarray(mask_lookup.get(key_b), dtype=bool)
            support_overlap = 0.0
            if mask_a.size > 0 and mask_b.size > 0 and mask_a.shape == mask_b.shape:
                supp_a = float(np.sum(mask_a))
                supp_b = float(np.sum(mask_b))
                if supp_a > 0.0 and supp_b > 0.0:
                    inter = float(np.sum(mask_a & mask_b))
                    support_overlap = float((2.0 * inter) / max(supp_a + supp_b, 1.0))

            # IC series correlation overlap (instead of PnL correlation)
            ic_overlap = 0.0
            ic_a = ic_series_cache.get(idx_a, np.array([]))
            ic_b = ic_series_cache.get(idx_b, np.array([]))
            if len(ic_a) > 0 and len(ic_b) > 0:
                # Align to common length (both should be same number of folds)
                min_len = min(len(ic_a), len(ic_b))
                ic_a_aligned = ic_a[:min_len]
                ic_b_aligned = ic_b[:min_len]
                # Remove NaN pairs
                valid_mask = np.isfinite(ic_a_aligned) & np.isfinite(ic_b_aligned)
                if np.sum(valid_mask) >= 2:
                    ic_a_valid = ic_a_aligned[valid_mask]
                    ic_b_valid = ic_b_aligned[valid_mask]
                    if np.std(ic_a_valid) > 1e-12 and np.std(ic_b_valid) > 1e-12:
                        ic_corr = float(np.clip(np.corrcoef(ic_a_valid, ic_b_valid)[0, 1], -1.0, 1.0))
                        ic_overlap = max(ic_corr, 0.0)  # Only positive correlation is "overlap"

            total_weight = max(support_overlap_weight + ic_overlap_weight, 1e-12)
            raw_overlap = float(
                (
                    support_overlap_weight * support_overlap
                    + ic_overlap_weight * ic_overlap
                )
                / total_weight
            )

            # Apply reduction if side or horizon differ (cumulative)
            # Different side: 30% less penalty (multiply by 0.7)
            # Different horizon: 20% less penalty (multiply by 0.8)
            side_a = str(eligible.loc[idx_a, "side"]) if "side" in eligible.columns else ""
            side_b = str(eligible.loc[idx_b, "side"]) if "side" in eligible.columns else ""
            horizon_a = float(eligible.loc[idx_a, "source_horizon"]) if "source_horizon" in eligible.columns else 0.0
            horizon_b = float(eligible.loc[idx_b, "source_horizon"]) if "source_horizon" in eligible.columns else 0.0

            reduction_factor = 1.0
            if side_a != side_b:
                reduction_factor *= 0.7  # 30% less penalty
            if abs(horizon_a - horizon_b) > 1e-6:
                reduction_factor *= 0.8  # 20% less penalty

            return raw_overlap * reduction_factor

        quadrant_order = [("long", 3), ("long", 10), ("short", 3), ("short", 10)]
        min_per_quadrant = 3
        for side_name, horizon_value in quadrant_order:
            if len(selected) >= top_k:
                break
            group_mask = (
                eligible["side"].astype(str).eq(side_name)
                & pd.to_numeric(
                    eligible.get("source_horizon", np.nan), errors="coerce"
                ).eq(float(horizon_value))
            )
            group_candidates = eligible.loc[group_mask].sort_values(
                "final_candidate_rank_score", ascending=False
            ).index.tolist()
            selected_count = 0
            for idx in group_candidates:
                if len(selected) >= top_k:
                    break
                if idx in remaining and idx not in selected:
                    selected.append(idx)
                    remaining.remove(idx)
                    selected_count += 1
                    if selected_count >= min_per_quadrant:
                        break

        # Group remaining by (side, horizon) for overlap-constrained selection
        from collections import defaultdict
        remaining_by_group = defaultdict(list)
        for idx in remaining:
            side_val = str(eligible.loc[idx, "side"]) if "side" in eligible.columns else "unknown"
            horizon_val = float(eligible.loc[idx, "source_horizon"]) if "source_horizon" in eligible.columns else 0.0
            remaining_by_group[(side_val, horizon_val)].append(idx)
        
        # Pre-group selected by (side, horizon)
        selected_by_group = defaultdict(list)
        for s_idx in selected:
            side_val = str(eligible.loc[s_idx, "side"]) if "side" in eligible.columns else "unknown"
            horizon_val = float(eligible.loc[s_idx, "source_horizon"]) if "source_horizon" in eligible.columns else 0.0
            selected_by_group[(side_val, horizon_val)].append(s_idx)
        
        while remaining and len(selected) < top_k:
            best_idx = None
            best_score = -np.inf
            for idx in remaining:
                base_score = float(eligible.loc[idx, "final_candidate_rank_score"])
                # Only compute overlap against strategies with same (side, horizon)
                side_val = str(eligible.loc[idx, "side"]) if "side" in eligible.columns else "unknown"
                horizon_val = float(eligible.loc[idx, "source_horizon"]) if "source_horizon" in eligible.columns else 0.0
                same_group_selected = selected_by_group.get((side_val, horizon_val), [])
                max_overlap = max((_pair_overlap(idx, s) for s in same_group_selected), default=0.0)
                score = base_score - overlap_penalty * max_overlap
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
            # Update selected_by_group
            side_val = str(eligible.loc[best_idx, "side"]) if "side" in eligible.columns else "unknown"
            horizon_val = float(eligible.loc[best_idx, "source_horizon"]) if "source_horizon" in eligible.columns else 0.0
            selected_by_group[(side_val, horizon_val)].append(best_idx)

        if selected:
            result.loc[selected, "selected_for_final_registry"] = True
            result.loc[selected, "final_candidate_rank_score"] = eligible.loc[selected, "final_candidate_rank_score"].to_numpy()
            result.loc[selected, "final_selection_order"] = np.arange(1, len(selected) + 1, dtype=np.int32)
        result.loc[eligible.index, "trade_composite_score"] = eligible["trade_composite_score"].to_numpy()
        result.loc[eligible.index, "composite_score"] = eligible["trade_composite_score"].to_numpy()

        return result

    def assess_rules(
        self,
        registry: pd.DataFrame,
        X: np.ndarray,
        data: pd.DataFrame,
        fwd_ret: np.ndarray,
        fwd_ret_norm: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        fold_health_summary: Optional[Dict[str, Any]] = None,
        step_mode: str = "full",
        step1_checkpoint_dir: Optional[Path] = None,
        checkpoint_output_dir: Optional[Path] = None,
        bounded_target: Optional[np.ndarray] = None,
        triad_targets_map: Optional[Dict[Tuple[str, int], np.ndarray]] = None,
        output_dirs_map: Optional[Dict[Tuple[str, int], Path]] = None,
        batch_size: int = 14,
        target_nan_reasons: Optional[
            Union[np.ndarray, Dict[Tuple[str, int], np.ndarray]]
        ] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if registry.empty:
            return registry

        tprint(
            f"Assessing {len(registry)} rules for Structural Alpha & Learnability..."
        )
        tprint(f"Stage A: Starting assessment - preparing data and configuration...")
        fold_health_summary = dict(fold_health_summary or {})
        total_fold_count = int(
            fold_health_summary.get("total_fold_count", len(folds)) or 0
        )
        healthy_fold_count = int(
            fold_health_summary.get("healthy_fold_count", total_fold_count) or 0
        )
        healthy_fold_ratio = (
            float(healthy_fold_count) / max(float(total_fold_count), 1.0)
            if total_fold_count > 0
            else 1.0
        )
        if total_fold_count > 0:
            tprint(
                "Stage A: Fold health summary "
                f"healthy={healthy_fold_count}/{total_fold_count} "
                f"ratio={healthy_fold_ratio:.3f}"
            )
        step_mode = str(step_mode or "full").lower()
        if step_mode not in {"full", "step1", "step2"}:
            raise ValueError(f"Unsupported step mode: {step_mode}")
        skip_stage1_filtering = bool(kwargs.get("skip_stage1_filtering", False))
        assessment_results = []

        # Prepare TBM data
        close = (
            data["close"].to_numpy() if "close" in data.columns else np.zeros(len(data))
        )
        high = (
            data["high"].to_numpy() if "high" in data.columns else np.zeros(len(data))
        )
        low = data["low"].to_numpy() if "low" in data.columns else np.zeros(len(data))
        atr = (
            data["atr"].to_numpy()
            if "atr" in data.columns
            else np.full(len(data), 0.001)
        )
        atr_frac_for_targets = atr / np.maximum(np.abs(close), 1e-12)

        adaptive_enabled = bool(self.cfg.get("adaptive_tp_sl_enabled", False))

        min_oof_coverage = float(self.cfg.get("learnability_min_oof_coverage", 0.05))
        min_avg_trades = float(self.cfg.get("min_avg_trades_per_day_10_symbols", 0.1))
        min_sign_consistency = float(self.cfg.get("min_sign_consistency", 0.0))
        min_mean_target_value = float(self.cfg.get("min_mean_target_value", 0.003))

        if bounded_target is not None:
            ridge_target_by_side = {"long": bounded_target, "short": bounded_target}
        else:
            ridge_target_by_side = {"long": fwd_ret_norm, "short": -fwd_ret_norm}

        tprint(f"Stage A Target Alignment:")
        tprint(f"  Miner target bounded: {bounded_target is not None}")
        tprint(
            "  Baseline/Ridge learnability target: "
            + (
                "bounded_target effective miner target"
                if bounded_target is not None
                else "ridge_target_by_side (vol-normalized signed forward return)"
            )
        )

        mean_ret_global_by_side = {
            "long": float(np.nanmean(ridge_target_by_side["long"])),
            "short": float(np.nanmean(ridge_target_by_side["short"])),
        }
        feature_to_regime_family = {
            m.feature_name: m.regime_family
            for m in self.metadata
            if getattr(m, "regime_family", None)
        }
        baseline_cache: Dict[str, Dict[str, float]] = {}

        mask_cache: Dict[str, np.ndarray] = {}
        cheap_stats_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
        contextual_cheap_stats_cache: Dict[
            Tuple[str, str, str, int], Dict[str, float]
        ] = {}
        directional_edge_floor = float(self.cfg.get("directional_edge_floor", 0.0))
        min_candidates_per_bucket = int(self.cfg.get("min_candidates_per_bucket", 50))
        support_min = float(self.cfg.get("support_min_pct", SUPPORT_MIN))
        support_max = float(self.cfg.get("support_max_pct", SUPPORT_MAX))
        target_support = float(self.cfg.get("target_support_pct", TARGET_SUPPORT))
        preferred_support_min = float(
            self.cfg.get("objective_support_target_low_pct", PREFERRED_SUPPORT_MIN)
        )
        preferred_support_max = float(
            self.cfg.get("objective_support_target_high_pct", PREFERRED_SUPPORT_MAX)
        )
        persisted_support_ok_by_key: Dict[str, bool] = {}
        persisted_support_pct_by_key: Dict[str, float] = {}
        support_source_col = None
        for candidate_col in ("mean_support_pct", "support_pct"):
            if candidate_col in registry.columns:
                support_source_col = candidate_col
                break
        if support_source_col is not None:
            support_series = pd.to_numeric(
                registry[support_source_col], errors="coerce"
            )
            canonical_series = registry["canonical_key"].astype(str)
            for canonical_key, support_val in zip(canonical_series, support_series):
                if np.isfinite(support_val):
                    support_float = float(support_val)
                    persisted_support_pct_by_key[canonical_key] = support_float
                    persisted_support_ok_by_key[canonical_key] = bool(
                        support_min <= support_float <= support_max
                    )

        # Precompute day buckets once to avoid per-rule timestamp parsing/groupby.
        day_codes: Optional[np.ndarray] = None
        n_day_buckets = 0
        if "timestamp" in data.columns:
            timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
            if timestamps.notna().any():
                day_labels, _ = pd.factorize(timestamps.dt.date, sort=False)
                day_codes = day_labels.astype(np.int32, copy=False)
                n_day_buckets = int(day_labels.max() + 1)

        total_symbol_days = self._compute_total_symbol_days(data)

        reg_canonical_keys = registry["canonical_key"].astype(str).to_numpy()
        if "side" in registry.columns:
            reg_side = registry["side"].astype(str).to_numpy()
        else:
            reg_side = np.full(len(registry), "long", dtype=object)
        if "source_horizon" in registry.columns:
            reg_source_horizon = registry["source_horizon"].to_numpy()
        else:
            reg_source_horizon = np.full(len(registry), -1)

        unique_rule_keys = list(dict.fromkeys(reg_canonical_keys.tolist()))
        key_to_mask_idx = {key: idx for idx, key in enumerate(unique_rule_keys)}
        batch_mask_start = time.perf_counter()
        if self.mask_resolver is not None:
            unique_mask_matrix = self.mask_resolver.get_masks_matrix(unique_rule_keys)
        else:
            unique_mask_matrix = np.vstack(
                [self._get_mask_for_rule(key, X) for key in unique_rule_keys]
            ).astype(bool, copy=False)
        batch_mask_elapsed = time.perf_counter() - batch_mask_start
        tprint(
            "Stage A: Batch mask resolution "
            f"rules={len(unique_rule_keys)} samples={unique_mask_matrix.shape[1]} "
            f"elapsed={batch_mask_elapsed:.2f}s"
        )
        for idx, key in enumerate(unique_rule_keys):
            mask_cache[key] = unique_mask_matrix[idx]

        batch_day_codes = (
            day_codes
            if day_codes is not None
            else np.full(len(data), -1, dtype=np.int32)
        )

        def _batch_fill_cheap_stats_cache(side: str, side_returns: np.ndarray) -> None:
            side_returns = np.asarray(side_returns, dtype=np.float32)
            batch_stats_start = time.perf_counter()
            (
                support_counts_arr,
                mean_ret_arr,
                std_ret_arr,
                sign_consistency_arr,
                tail_ratio_arr,
                mae_arr,
                mfe_arr,
                density_dispersion_arr,
            ) = _compute_metrics_batch_numba(
                unique_mask_matrix,
                side_returns,
                batch_day_codes,
                int(n_day_buckets),
            )
            batch_stats_elapsed = time.perf_counter() - batch_stats_start
            tprint(
                "Stage A: Batch cheap stats "
                f"side={side} rules={len(unique_rule_keys)} "
                f"elapsed={batch_stats_elapsed:.2f}s"
            )
            mean_ret_global = mean_ret_global_by_side[side]
            for idx, canonical_key in enumerate(unique_rule_keys):
                support_count = int(support_counts_arr[idx])
                support_pct = float(support_count / max(len(data), 1))
                support_ok = support_min <= support_pct <= support_max
                if preferred_support_min <= support_pct <= preferred_support_max:
                    support_score = 1.0
                elif support_pct < preferred_support_min:
                    span = max(preferred_support_min - support_min, 1e-9)
                    relative = np.clip((support_pct - support_min) / span, 0.0, 1.0)
                    support_score = float(0.2 + (1.0 - 0.2) * relative)
                else:
                    span = max(support_max - preferred_support_max, 1e-9)
                    relative = np.clip((support_max - support_pct) / span, 0.0, 1.0)
                    support_score = float(0.2 + (1.0 - 0.2) * relative)

                if total_symbol_days is None or total_symbol_days <= 0:
                    avg_trades = float(support_count)
                else:
                    avg_trades = float((support_count / total_symbol_days) * 10.0)

                mean_ret_mask = float(mean_ret_arr[idx])
                cheap_stats_cache[(canonical_key, side)] = {
                    "support_count": support_count,
                    "support_pct": support_pct,
                    "support_ok": float(support_ok),
                    "support_score": support_score,
                    "avg_trades": avg_trades,
                    "density_dispersion": float(density_dispersion_arr[idx]),
                    "tail_ratio": float(tail_ratio_arr[idx]),
                    "mae": float(mae_arr[idx]),
                    "mfe": float(mfe_arr[idx]),
                    "mean_ret_global": mean_ret_global,
                    "mean_ret_mask": mean_ret_mask,
                    "std_ret_mask": float(std_ret_arr[idx]),
                    "ret_uplift": mean_ret_mask - mean_ret_global,
                    "sign_consistency": float(sign_consistency_arr[idx]),
                }

        _batch_fill_cheap_stats_cache("long", ridge_target_by_side["long"])
        _batch_fill_cheap_stats_cache("short", ridge_target_by_side["short"])

        def _compute_single_cheap_stats(
            side: str,
            mask: np.ndarray,
            side_returns: np.ndarray,
        ) -> Dict[str, float]:
            side_returns = np.asarray(side_returns, dtype=np.float32)
            support_count = int(np.sum(mask))
            support_pct = float(support_count / max(len(data), 1))
            support_ok = float(support_min <= support_pct <= support_max)
            if preferred_support_min <= support_pct <= preferred_support_max:
                support_score = 1.0
            elif support_pct < preferred_support_min:
                span = max(preferred_support_min - support_min, 1e-9)
                relative = np.clip((support_pct - support_min) / span, 0.0, 1.0)
                support_score = float(0.2 + (1.0 - 0.2) * relative)
            else:
                span = max(support_max - preferred_support_max, 1e-9)
                relative = np.clip((support_max - support_pct) / span, 0.0, 1.0)
                support_score = float(0.2 + (1.0 - 0.2) * relative)

            masked_returns = np.asarray(side_returns[mask], dtype=np.float32)
            finite_returns = masked_returns[np.isfinite(masked_returns)]
            mean_ret_global = float(np.nanmean(side_returns))
            mean_ret_mask = float(np.nanmean(finite_returns)) if finite_returns.size else np.nan
            std_ret_mask = (
                float(np.nanstd(_clip_returns(finite_returns)))
                if finite_returns.size
                else np.nan
            )
            sign_consistency = (
                compute_directional_sign_consistency(finite_returns)
                if finite_returns.size
                else 0.5
            )
            if finite_returns.size >= 20:
                abs_returns = np.abs(finite_returns)
                p95 = float(np.nanpercentile(abs_returns, 95))
                p5 = float(np.nanpercentile(abs_returns, 5))
                tail_ratio = p95 / (p5 + 1e-9)
            else:
                tail_ratio = 1.0

            return {
                "support_count": support_count,
                "support_pct": support_pct,
                "support_ok": support_ok,
                "support_score": support_score,
                "avg_trades": self._compute_avg_trades_per_day(mask, total_symbol_days),
                "density_dispersion": 0.0,
                "tail_ratio": float(tail_ratio),
                "mae": 0.0,
                "mfe": 0.0,
                "mean_ret_global": mean_ret_global,
                "mean_ret_mask": mean_ret_mask,
                "std_ret_mask": std_ret_mask,
                "ret_uplift": mean_ret_mask - mean_ret_global,
                "sign_consistency": float(sign_consistency),
            }

        def _get_or_compute_cheap_stats(
            canonical_key: str,
            side: str,
            mask: np.ndarray,
            *,
            target_name: Optional[str] = None,
            horizon_key: Optional[int] = None,
            side_returns: Optional[np.ndarray] = None,
        ) -> Dict[str, float]:
            if (
                side_returns is not None
                and target_name is not None
                and horizon_key is not None
                and triad_targets_map is not None
            ):
                contextual_key = (
                    canonical_key,
                    side,
                    str(target_name),
                    int(horizon_key),
                )
                cached_stats = contextual_cheap_stats_cache.get(contextual_key)
                if cached_stats is not None:
                    return cached_stats
                stats = _compute_single_cheap_stats(side, mask, side_returns)
                contextual_cheap_stats_cache[contextual_key] = stats
                return stats

            cache_key = (canonical_key, side)
            cached_stats = cheap_stats_cache.get(cache_key)
            if cached_stats is not None:
                return cached_stats
            return _compute_single_cheap_stats(side, mask, ridge_target_by_side[side])

        def _persisted_support_ok(canonical_key: str, cheap: Dict[str, float]) -> bool:
            if canonical_key in persisted_support_ok_by_key:
                return persisted_support_ok_by_key[canonical_key]
            return bool(cheap["support_ok"])

        def _infer_regime_family_combo(canonical_key: str) -> str:
            families = sorted(
                {
                    feature_to_regime_family[name]
                    for name in extract_feature_names_from_key(canonical_key)
                    if name in feature_to_regime_family
                }
            )
            return "|".join(families) if families else "none"

        # Bucket-level floor calibration and protection list to avoid over-pruning.
        bucket_path_floor: Dict[Tuple[str, int, str], float] = {}
        # Precompute bucket-level top-decile caps for density dispersion and tail risk.
        bucket_density_values: Dict[
            Tuple[str, int], List[Tuple[str, float]]
        ] = collections.defaultdict(list)
        bucket_tail_values: Dict[
            Tuple[str, int], List[Tuple[str, float]]
        ] = collections.defaultdict(list)
        bucket_path_quality_values: Dict[
            Tuple[str, int], List[Tuple[str, float]]
        ] = collections.defaultdict(list)
        bucket_stability_values: Dict[
            Tuple[str, int], List[Tuple[str, float]]
        ] = collections.defaultdict(list)
        bucket_sign_consistency_values: Dict[
            Tuple[str, int], List[Tuple[str, float]]
        ] = collections.defaultdict(list)
        bucket_mean_ret_values: Dict[
            Tuple[str, int], List[Tuple[str, float]]
        ] = collections.defaultdict(list)
        bucket_hurdle_values: Dict[
            Tuple[str, int], List[Tuple[str, float]]
        ] = collections.defaultdict(list)

        rejected_by_support: set = set()

        effective_step_mode = step_mode
        if step_mode == "step2" and step1_checkpoint_dir is None:
            tprint(
                "Stage A: step2 requested without step1_checkpoint_dir; "
                "falling back to in-memory cheap-gate recomputation."
            )
            effective_step_mode = "full"

        if skip_stage1_filtering:
            tprint(
                "Stage A: step2 pooled assessment active; "
                "skipping overlap pruning and ridge shortlist reduction, "
                "but still enforcing support_out_of_range upstream."
            )
            cheap_gate_rows = collections.defaultdict(list)
            cheap_gate_result = {}
            bucket_cheap_ranks = collections.defaultdict(dict)
            stage_a_matrices = {}
            seen_keys_for_direct_assessment = set()
            for pre_row in registry.to_dict("records"):
                canonical_key = str(pre_row["canonical_key"])
                side = str(pre_row.get("side", "long"))
                if side not in ridge_target_by_side:
                    side = "long"
                horizon_raw = pre_row.get("source_horizon", -1)
                try:
                    horizon_key = int(horizon_raw) if pd.notna(horizon_raw) else -1
                except (TypeError, ValueError):
                    horizon_key = -1
                bucket_key = (side, horizon_key)
                tracker = (bucket_key, canonical_key)
                if tracker in seen_keys_for_direct_assessment:
                    continue
                seen_keys_for_direct_assessment.add(tracker)
                mask = mask_cache.get(canonical_key)
                if mask is None:
                    if self.mask_resolver:
                        mask = self.mask_resolver.get_mask(canonical_key)
                    else:
                        mask = self._get_mask_for_rule(canonical_key, X)
                    mask_cache[canonical_key] = mask

                if np.sum(mask) < 20:
                    cheap_gate_result[(bucket_key, canonical_key)] = (
                        True,
                        "support_out_of_range",
                    )
                    continue

                target_name = str(pre_row.get("source_target", "unknown"))
                side_returns = ridge_target_by_side[side]
                if triad_targets_map is not None:
                    side_returns = triad_targets_map.get(
                        (target_name, horizon_key), side_returns
                    )
                cheap = _get_or_compute_cheap_stats(
                    canonical_key,
                    side,
                    mask,
                    target_name=target_name,
                    horizon_key=horizon_key,
                    side_returns=side_returns,
                )
                if not _persisted_support_ok(canonical_key, cheap):
                    cheap_gate_result[(bucket_key, canonical_key)] = (
                        True,
                        "support_out_of_range",
                    )
                    continue

                cheap_gate_rows[bucket_key].append((0.0, canonical_key))
                bucket_cheap_ranks[bucket_key][canonical_key] = 0.0
                cheap_gate_result[(bucket_key, canonical_key)] = (False, "")
        elif effective_step_mode == "step2":
            step1_payload = load_stage_a_step1_checkpoint(step1_checkpoint_dir)
            cheap_gate_rows = step1_payload["cheap_gate_rows"]
            cheap_gate_result = step1_payload["cheap_gate_result"]
            bucket_cheap_ranks = step1_payload["bucket_cheap_ranks"]
            stage_a_matrices = step1_payload["stage_a_matrices"]
            selected_keys_runtime = {
                str(k)
                for k in registry.get("canonical_key", pd.Series(dtype=str)).astype(str)
            }
            if selected_keys_runtime:
                cheap_gate_rows = {
                    bucket_key: [
                        (cheap_rank, canonical_key)
                        for cheap_rank, canonical_key in entries
                        if str(canonical_key) in selected_keys_runtime
                    ]
                    for bucket_key, entries in cheap_gate_rows.items()
                }
                cheap_gate_rows = {
                    bucket_key: entries
                    for bucket_key, entries in cheap_gate_rows.items()
                    if entries
                }
                cheap_gate_result = {
                    (bucket_key, canonical_key): outcome
                    for (
                        bucket_key,
                        canonical_key,
                    ), outcome in cheap_gate_result.items()
                    if str(canonical_key) in selected_keys_runtime
                }
                bucket_cheap_ranks = {
                    bucket_key: {
                        canonical_key: rank
                        for canonical_key, rank in ranks.items()
                        if str(canonical_key) in selected_keys_runtime
                    }
                    for bucket_key, ranks in bucket_cheap_ranks.items()
                }
            tprint(
                f"Stage A: Loaded step1 checkpoint from {step1_checkpoint_dir} "
                f"with {sum(len(v) for v in cheap_gate_rows.values())} post-dedup rules"
            )
        else:
            seen_keys_per_bucket = collections.defaultdict(set)
            reg_trade_path_quality = (
                pd.to_numeric(
                    registry["trade_path_quality_score"], errors="coerce"
                ).to_numpy()
                if "trade_path_quality_score" in registry.columns
                else np.full(len(registry), np.nan, dtype=np.float32)
            )
            reg_quality_stability = (
                pd.to_numeric(
                    registry["quality_stability_score"], errors="coerce"
                ).to_numpy()
                if "quality_stability_score" in registry.columns
                else np.full(len(registry), np.nan, dtype=np.float32)
            )
            reg_hurdle_excess = (
                pd.to_numeric(registry["hurdle_excess"], errors="coerce").to_numpy()
                if "hurdle_excess" in registry.columns
                else np.full(len(registry), np.nan, dtype=np.float32)
            )

            # Build registry_key_to_row mapping once (used later for overlap dedup)
            registry_key_to_row: Dict[str, int] = {}

            tprint(
                f"Stage A: Pass 1 - Computing cheap stats for all {len(registry)} rules..."
            )
            for row_idx in range(len(registry)):
                canonical_key = reg_canonical_keys[row_idx]
                side = str(reg_side[row_idx])
                if side not in ridge_target_by_side:
                    side = "long"

                horizon_raw = reg_source_horizon[row_idx]
                try:
                    horizon_key = int(horizon_raw) if pd.notna(horizon_raw) else -1
                except (TypeError, ValueError):
                    horizon_key = -1
                target_name = str(
                    registry.iloc[row_idx].get("source_target", "unknown")
                )

                bucket_key = (side, horizon_key)

                if canonical_key in seen_keys_per_bucket[bucket_key]:
                    continue
                seen_keys_per_bucket[bucket_key].add(canonical_key)
                registry_key_to_row[canonical_key] = row_idx

                mask = mask_cache[canonical_key]
                if np.sum(mask) < 20:
                    continue

                side_returns = ridge_target_by_side[side]
                if triad_targets_map is not None:
                    side_returns = triad_targets_map.get(
                        (target_name, horizon_key), side_returns
                    )
                cheap = _get_or_compute_cheap_stats(
                    canonical_key,
                    side,
                    mask,
                    target_name=target_name,
                    horizon_key=horizon_key,
                    side_returns=side_returns,
                )
                if not _persisted_support_ok(canonical_key, cheap):
                    rejected_by_support.add(canonical_key)

                bucket_density_values[bucket_key].append(
                    (canonical_key, float(cheap["density_dispersion"]))
                )
                bucket_tail_values[bucket_key].append(
                    (canonical_key, float(cheap["tail_ratio"]))
                )
                path_quality = float(reg_trade_path_quality[row_idx])
                stability_score = float(reg_quality_stability[row_idx])
                if np.isfinite(path_quality):
                    bucket_path_quality_values[bucket_key].append(
                        (canonical_key, path_quality)
                    )
                if np.isfinite(stability_score):
                    bucket_stability_values[bucket_key].append(
                        (canonical_key, stability_score)
                    )
                sign_consistency = float(cheap["sign_consistency"])
                mean_ret_mask = float(cheap["mean_ret_mask"])
                hurdle_excess = float(reg_hurdle_excess[row_idx])
                bucket_sign_consistency_values[bucket_key].append(
                    (canonical_key, sign_consistency)
                )
                bucket_mean_ret_values[bucket_key].append(
                    (canonical_key, mean_ret_mask)
                )
                if np.isfinite(hurdle_excess):
                    bucket_hurdle_values[bucket_key].append(
                        (canonical_key, hurdle_excess)
                    )

        if not skip_stage1_filtering:
            def _normalize_to_1_2(
                arr: np.ndarray, higher_is_better: bool = True
            ) -> np.ndarray:
                valid = np.isfinite(arr)
                if not np.any(valid):
                    return np.full_like(arr, 1.5)
                valid_vals = arr[valid]
                if len(valid_vals) == 0:
                    return np.full_like(arr, 1.5)
                p2 = np.percentile(valid_vals, 2)
                p98 = np.percentile(valid_vals, 98)
                clipped = np.clip(arr, p2, p98)
                min_val = np.nanmin(clipped)
                max_val = np.nanmax(clipped)
                if max_val - min_val < 1e-9:
                    return np.full_like(arr, 1.5)

                if higher_is_better:
                    return 1.0 + (clipped - min_val) / (max_val - min_val)
                else:
                    return 1.0 + (max_val - clipped) / (max_val - min_val)

            bucket_protected_keys: Dict[Tuple[str, int, str], set[str]] = {}
            bucket_cheap_ranks = collections.defaultdict(dict)

            all_bucket_keys = set(bucket_mean_ret_values.keys())
            for b_key in all_bucket_keys:
                keys = [k for k, v in bucket_mean_ret_values[b_key]]
                if not keys:
                    continue

                m_ret_dict = dict(bucket_mean_ret_values[b_key])
                m_path_dict = dict(bucket_path_quality_values[b_key])
                m_sign_dict = dict(bucket_sign_consistency_values[b_key])
                m_tail_dict = dict(bucket_tail_values[b_key])
                m_dens_dict = dict(bucket_density_values[b_key])

                ret_arr = np.array([m_ret_dict.get(k, np.nan) for k in keys])
                path_arr = np.array([m_path_dict.get(k, np.nan) for k in keys])
                sign_arr = np.array([m_sign_dict.get(k, np.nan) for k in keys])
                tail_arr = np.array([m_tail_dict.get(k, np.nan) for k in keys])
                dens_arr = np.array([m_dens_dict.get(k, np.nan) for k in keys])

                n_ret = _normalize_to_1_2(ret_arr, higher_is_better=True)
                n_path = _normalize_to_1_2(path_arr, higher_is_better=True)
                n_sign = _normalize_to_1_2(sign_arr, higher_is_better=True)
                n_tail = _normalize_to_1_2(tail_arr, higher_is_better=False)
                n_dens = _normalize_to_1_2(dens_arr, higher_is_better=False)
                ranks = n_sign * (n_path**1.5) * n_ret * np.sqrt(n_tail + n_dens)

                ranked_items = []
                for i, k in enumerate(keys):
                    rank_val = ranks[i]
                    if np.isnan(rank_val):
                        rank_val = -np.inf
                    bucket_cheap_ranks[b_key][k] = float(rank_val)
                    ranked_items.append((rank_val, k))

                ranked_items.sort(key=lambda x: x[0], reverse=True)
                top_k = [k for r, k in ranked_items[: max(min_candidates_per_bucket, 0)]]
                bucket_protected_keys[b_key] = set(top_k)

            tprint(
                f"Stage A: Pass 1 complete - computed cheap stats for {len(registry)} rules"
            )

            pctile_bottom_cut = float(self.cfg.get("cheap_gate_bottom_pctile", 0.20))

            all_rejected = set(rejected_by_support)
            if all_rejected:
                tprint(
                    f"Stage A cheap gate (0): support_out_of_range rejected {len(all_rejected)} rules"
                )

            bucket_sign_consistency_floor: Dict[Tuple[str, int], float] = {}
            for bucket_key, tuples in bucket_sign_consistency_values.items():
                vals = np.asarray(
                    [v for k, v in tuples if k not in all_rejected], dtype=float
                )
                finite_vals = vals[np.isfinite(vals)]
                bucket_sign_consistency_floor[bucket_key] = (
                    float(np.nanquantile(finite_vals, pctile_bottom_cut))
                    if finite_vals.size > 0
                    else -np.inf
                )

            bucket_hurdle_floor: Dict[Tuple[str, int], float] = {}
            bucket_hurdle_values_surviving = collections.defaultdict(list)
            for bucket_key, tuples in bucket_hurdle_values.items():
                for canonical_key, hurdle_excess in tuples:
                    if canonical_key not in all_rejected:
                        bucket_hurdle_values_surviving[bucket_key].append(
                            (canonical_key, hurdle_excess)
                        )
            for bucket_key, tuples in bucket_hurdle_values_surviving.items():
                vals = np.asarray([v for _, v in tuples], dtype=float)
                finite_vals = vals[np.isfinite(vals)]
                bucket_hurdle_floor[bucket_key] = (
                    float(np.nanquantile(finite_vals, pctile_bottom_cut))
                    if finite_vals.size > 0
                    else -np.inf
                )

            rejected_by_hurdle: set = set()
            for bucket_key, tuples in bucket_hurdle_values_surviving.items():
                floor = bucket_hurdle_floor.get(bucket_key, -np.inf)
                for canonical_key, hurdle_excess in tuples:
                    if hurdle_excess < floor:
                        rejected_by_hurdle.add(canonical_key)

            n_hurdle_rejected = len(rejected_by_hurdle)
            if n_hurdle_rejected > 0:
                tprint(
                    f"Stage A cheap gate (1.1): beats_hurdle "
                    f"  rejected {n_hurdle_rejected} rules (bottom {pctile_bottom_cut:.0%} per bucket)"
                )
            all_rejected |= rejected_by_hurdle

            bucket_stability_floor: Dict[Tuple[str, int], float] = {}
            for bucket_key, tuples in bucket_stability_values.items():
                vals = np.asarray(
                    [v for k, v in tuples if k not in all_rejected], dtype=float
                )
                finite_vals = vals[np.isfinite(vals)]
                bucket_stability_floor[bucket_key] = (
                    float(np.nanquantile(finite_vals, 0.20))
                    if finite_vals.size > 0
                    else -np.inf
                )

            rejected_by_stability: set = set()
            for bucket_key, tuples in bucket_stability_values.items():
                floor = bucket_stability_floor.get(bucket_key, -np.inf)
                protected = bucket_protected_keys.get(bucket_key, set())
                for canonical_key, stability in tuples:
                    if (
                        canonical_key not in all_rejected
                        and stability < floor
                        and canonical_key not in protected
                    ):
                        rejected_by_stability.add(canonical_key)

            n_stability_rejected = len(rejected_by_stability)
            if n_stability_rejected > 0:
                tprint(
                    f"Stage A cheap gate (1.5): quality_stability "
                    f"  rejected {n_stability_rejected} rules (bottom 20% per bucket)"
                )

            all_rejected |= rejected_by_stability

            bucket_density_values_surviving = collections.defaultdict(list)
            for bucket_key, tuples in bucket_density_values.items():
                for canonical_key, val in tuples:
                    if canonical_key not in all_rejected:
                        bucket_density_values_surviving[bucket_key].append(
                            (canonical_key, val)
                        )
            bucket_density_cap: Dict[Tuple[str, int], float] = {}
            for bucket_key, tuples in bucket_density_values_surviving.items():
                vals = np.asarray([v for _, v in tuples], dtype=float)
                finite_vals = vals[np.isfinite(vals)]
                bucket_density_cap[bucket_key] = (
                    float(np.nanquantile(finite_vals, 0.90))
                    if finite_vals.size > 0
                    else np.inf
                )

            bucket_tail_values_surviving = collections.defaultdict(list)
            for bucket_key, tuples in bucket_tail_values.items():
                for canonical_key, val in tuples:
                    if canonical_key not in all_rejected:
                        bucket_tail_values_surviving[bucket_key].append(
                            (canonical_key, val)
                        )
            bucket_tail_cap: Dict[Tuple[str, int], float] = {}
            for bucket_key, tuples in bucket_tail_values_surviving.items():
                vals = np.asarray([v for _, v in tuples], dtype=float)
                finite_vals = vals[np.isfinite(vals)]
                bucket_tail_cap[bucket_key] = (
                    float(np.nanquantile(finite_vals, 0.90))
                    if finite_vals.size > 0
                    else np.inf
                )

            bucket_path_quality_surviving = collections.defaultdict(list)
            for bucket_key, tuples in bucket_path_quality_values.items():
                for canonical_key, val in tuples:
                    if canonical_key not in all_rejected:
                        bucket_path_quality_surviving[bucket_key].append(
                            (canonical_key, val)
                        )
            bucket_path_floor: Dict[Tuple[str, int, str], float] = {}
            for bucket_key, tuples in bucket_path_quality_surviving.items():
                vals = np.asarray([v for _, v in tuples], dtype=float)
                finite_vals = vals[np.isfinite(vals)]
                bucket_path_floor[bucket_key] = (
                    float(np.nanquantile(finite_vals, pctile_bottom_cut))
                    if finite_vals.size > 0
                    else -np.inf
                )

            rejected_by_path: set = set()
            for bucket_key, tuples in bucket_path_quality_values.items():
                floor = bucket_path_floor.get(bucket_key, -np.inf)
                protected = bucket_protected_keys.get(bucket_key, set())
                for canonical_key, path in tuples:
                    if (
                        canonical_key not in all_rejected
                        and path < floor
                        and canonical_key not in protected
                    ):
                        rejected_by_path.add(canonical_key)

            n_path_rejected = len(rejected_by_path)
            if n_path_rejected > 0:
                tprint(
                    f"Stage A cheap gate (2.5): path_quality "
                    f"  rejected {n_path_rejected} rules (bottom {pctile_bottom_cut:.0%} per bucket)"
                )
            all_rejected |= rejected_by_path

            tprint(
                f"Stage A: Cheap gates complete - {len(registry) - len(all_rejected)} rules survived cheap gates"
            )

            cheap_gate_rows = collections.defaultdict(list)
            cheap_gate_result = {}
            seen_keys_for_cheap_gate = set()

            for pre_row in registry.to_dict("records"):
                canonical_key = str(pre_row["canonical_key"])
                side = str(pre_row["side"])
                if side not in ridge_target_by_side:
                    side = "long"

                horizon_raw = pre_row.get("source_horizon", -1)
                try:
                    horizon_key = int(horizon_raw) if pd.notna(horizon_raw) else -1
                except (TypeError, ValueError):
                    horizon_key = -1
                target_name = str(pre_row.get("source_target", "unknown"))
                bucket_key = (side, horizon_key)
                key_tracker = (bucket_key, canonical_key)
                if key_tracker in seen_keys_for_cheap_gate:
                    continue
                seen_keys_for_cheap_gate.add(key_tracker)

                mask = mask_cache.get(canonical_key)
                if mask is None:
                    if self.mask_resolver:
                        mask = self.mask_resolver.get_mask(canonical_key)
                    else:
                        mask = self._get_mask_for_rule(canonical_key, X)
                    mask_cache[canonical_key] = mask
                if np.sum(mask) < 20:
                    continue

                side_returns = ridge_target_by_side[side]
                if triad_targets_map is not None:
                    side_returns = triad_targets_map.get(
                        (target_name, horizon_key), side_returns
                    )
                cheap = _get_or_compute_cheap_stats(
                    canonical_key,
                    side,
                    mask,
                    target_name=target_name,
                    horizon_key=horizon_key,
                    side_returns=side_returns,
                )

                rejected = False
                rejection_reason = ""
                if not _persisted_support_ok(canonical_key, cheap):
                    rejected, rejection_reason = True, "support_out_of_range"
                elif float(
                    pre_row.get("quality_stability_score", np.nan)
                ) < bucket_stability_floor.get(bucket_key, -np.inf):
                    if canonical_key not in bucket_protected_keys.get(bucket_key, set()):
                        rejected, rejection_reason = True, "low_stability_floor"
                elif float(
                    pre_row.get("trade_path_quality_score", np.nan)
                ) < bucket_path_floor.get(bucket_key, -np.inf):
                    if canonical_key not in bucket_protected_keys.get(bucket_key, set()):
                        rejected, rejection_reason = True, "low_path_quality_floor"
                elif float(cheap["density_dispersion"]) > bucket_density_cap.get(
                    bucket_key, np.inf
                ):
                    rejected, rejection_reason = True, "high_density_dispersion_top_decile"
                elif float(cheap["tail_ratio"]) > bucket_tail_cap.get(
                    bucket_key, np.inf
                ):
                    rejected, rejection_reason = True, "high_tail_risk_top_decile"

                if (
                    rejected
                    and canonical_key in bucket_protected_keys.get(bucket_key, set())
                    and rejection_reason in {"low_path_quality_floor", "low_stability_floor"}
                ):
                    rejected = False

                if rejected:
                    cheap_gate_result[(bucket_key, canonical_key)] = (
                        True,
                        rejection_reason,
                    )
                    continue

                cheap_rank = bucket_cheap_ranks.get(bucket_key, {}).get(
                    canonical_key, -np.inf
                )
                cheap_gate_result[(bucket_key, canonical_key)] = (False, "")
                cheap_gate_rows[bucket_key].append((cheap_rank, canonical_key))

            OVERLAP_THRESHOLD = 0.975
            SUPPORT_RATIO_MIN = 0.70
            DEDUP_SUBSAMPLE_SIZE = 10000
            eps = 1e-8

            surviving_keys_by_bucket = collections.defaultdict(list)
            for bucket_key, entries in cheap_gate_rows.items():
                for _, canonical_key in entries:
                    surviving_keys_by_bucket[bucket_key].append(canonical_key)

            stage_a_matrices = {}

            n_total_surviving = sum(len(v) for v in surviving_keys_by_bucket.values())
            if n_total_surviving > 0:
                rng = np.random.default_rng(seed=42)
                n_rows = X.shape[0]
                if n_rows > DEDUP_SUBSAMPLE_SIZE:
                    sub_idx = rng.choice(n_rows, size=DEDUP_SUBSAMPLE_SIZE, replace=False)
                    sub_idx = np.sort(sub_idx)
                else:
                    sub_idx = np.arange(n_rows)
                n_subsample = float(len(sub_idx))

                registry_key_to_row = {}
                for i, row_dict in enumerate(registry.to_dict("records")):
                    ck = str(row_dict.get("canonical_key", ""))
                    if ck:
                        registry_key_to_row[ck] = i

                n_dedup_rejected = 0
                overlap_dedup_start_ts = time.perf_counter()
                tprint(
                    "Stage A: Overlap deduplication start - "
                    f"buckets={len(surviving_keys_by_bucket)} "
                    f"input_rules={sum(len(v) for v in surviving_keys_by_bucket.values())}"
                )
                for bucket_key, surviving_keys in surviving_keys_by_bucket.items():
                    bucket_overlap_start_ts = time.perf_counter()
                    n_rules = len(surviving_keys)
                    contexts = [None] * n_rules
                    gains = np.zeros(n_rules, dtype=np.float32)
                    sign_consistencies = np.zeros(n_rules, dtype=np.float32)
                    mean_returns = np.zeros(n_rules, dtype=np.float32)
                    std_returns = np.zeros(n_rules, dtype=np.float32)
                    supports = np.zeros(n_rules, dtype=np.float32)

                    for idx, canonical_key in enumerate(surviving_keys):
                        mask = mask_cache.get(canonical_key)
                        if mask is None:
                            if self.mask_resolver:
                                mask = self.mask_resolver.get_mask(canonical_key)
                            else:
                                mask = self._get_mask_for_rule(canonical_key, X)
                            mask_cache[canonical_key] = mask

                        mask_sub = mask[sub_idx]
                        contexts[idx] = mask_sub
                        supports[idx] = float(np.mean(mask_sub))

                        cheap = _get_or_compute_cheap_stats(
                            canonical_key, bucket_key[0], mask
                        )

                        row_idx = registry_key_to_row.get(canonical_key)
                        if row_idx is not None:
                            pre_row = registry.iloc[row_idx]
                            gain_val = float(
                                pre_row.get("rule_model_importance_score", 0.0)
                            )
                        else:
                            gain_val = 0.0
                        gains[idx] = gain_val
                        sign_consistencies[idx] = float(cheap["sign_consistency"])
                        mean_returns[idx] = float(cheap.get("mean_ret_mask", 0.0))
                        std_returns[idx] = float(cheap.get("std_ret_mask", 0.0))

                    tprint(
                        f"Stage A: Overlap deduplication - processing {len(surviving_keys)} surviving rules in bucket {bucket_key}"
                    )

                    context_matrix = np.column_stack(
                        [c.astype(np.int32, copy=False) for c in contexts]
                    )
                    intersections = context_matrix.T @ context_matrix
                    initial_n_rules = n_rules
                    sub_supports = np.diag(intersections).astype(float)

                    stage_a_matrices[bucket_key] = {
                        "key_to_idx": {k: idx for idx, k in enumerate(surviving_keys)},
                        "intersections": intersections,
                        "supports": sub_supports,
                        "n_subsample": n_subsample,
                    }

                    bucket_top_target = int(
                        self.cfg.get("overlap_dedup_bucket_top_target", 15)
                    )
                    score_order = np.argsort(-gains)
                    if n_rules > bucket_top_target:
                        accepted_indices = list(score_order[:bucket_top_target])
                    else:
                        accepted_indices = list(score_order)

                    surviving_indices = accepted_indices
                    final_surviving_set = {surviving_keys[i] for i in surviving_indices}

                    for i, k in enumerate(surviving_keys):
                        if k not in final_surviving_set:
                            cheap_gate_result[(bucket_key, k)] = (
                                True,
                                "top_n_cutoff",
                            )
                            n_dedup_rejected += 1

                    final_surviving_keys = [surviving_keys[i] for i in surviving_indices]

                    tprint(
                        f"Stage A: Bucket {bucket_key} dedup complete - "
                        f"{len(final_surviving_keys)} rules kept (from {initial_n_rules}), "
                        f"elapsed={time.perf_counter() - bucket_overlap_start_ts:.2f}s"
                    )

                    if final_surviving_keys:
                        stage_a_matrices[bucket_key] = {
                            "key_to_idx": {
                                k: idx for idx, k in enumerate(final_surviving_keys)
                            },
                            "intersections": intersections[
                                np.ix_(surviving_indices, surviving_indices)
                            ],
                            "supports": sub_supports[surviving_indices],
                            "n_subsample": n_subsample,
                        }
                    else:
                        stage_a_matrices.pop(bucket_key, None)

                if n_dedup_rejected > 0:
                    tprint(
                        f"Stage A cheap gate (3): overlap deduplication "
                        f"rejected {n_dedup_rejected} rules (overlap>{OVERLAP_THRESHOLD:.0%}, "
                        f"support_ratio>{SUPPORT_RATIO_MIN:.0%})"
                    )

                cheap_gate_rows_deduped = collections.defaultdict(list)
                for bucket_key, entries in cheap_gate_rows.items():
                    for cheap_rank, canonical_key in entries:
                        rejected, _ = cheap_gate_result.get(
                            (bucket_key, canonical_key), (False, "")
                        )
                        if not rejected:
                            cheap_gate_rows_deduped[bucket_key].append(
                                (cheap_rank, canonical_key)
                            )
                cheap_gate_rows = cheap_gate_rows_deduped

                tprint(
                    f"Stage A: Overlap deduplication complete - {sum(len(v) for v in cheap_gate_rows.values())} rules total survived"
                )
                tprint(
                    "Stage A: Overlap deduplication end - "
                    f"total_survivors={sum(len(v) for v in cheap_gate_rows.values())} "
                    f"total_rejected={n_dedup_rejected} "
                    f"elapsed={time.perf_counter() - overlap_dedup_start_ts:.2f}s"
                )

                if checkpoint_output_dir is not None:
                    checkpoint_path = save_stage_a_step1_checkpoint(
                        output_dir=checkpoint_output_dir,
                        candidate_registry=registry,
                        cheap_gate_rows=cheap_gate_rows,
                        cheap_gate_result=cheap_gate_result,
                        bucket_cheap_ranks=bucket_cheap_ranks,
                        stage_a_matrices=stage_a_matrices,
                    )
                    tprint(f"Stage A: Step1 checkpoint saved to {checkpoint_path}")

                if effective_step_mode == "step1":
                    tprint("Stage A: Step1 complete. Skipping post-dedup step2 assessment.")
                    return pd.DataFrame()

        # Helper for universe-relative uplift baselines
        def _compute_baseline_population_metrics(target_returns: np.ndarray, ts_array: np.ndarray):
            res = {
                "p5": np.nan,
                "mean_p75_ret": np.nan,
                "weekly_sortino": np.nan,
                "monthly_sortino": np.nan,
            }
            valid = np.isfinite(target_returns)
            if not np.any(valid):
                return res

            rets = target_returns[valid]
            ts = ts_array[valid]

            if len(rets) < 5:
                return res

            res["p5"] = float(np.percentile(rets, 5))

            p75_thresh = float(np.percentile(rets, 75))
            top_rets = rets[rets > p75_thresh]
            if len(top_rets) > 0:
                res["mean_p75_ret"] = float(np.mean(top_rets))

            # Periodic Sortino helper
            def _periodic_sortino(freq):
                if len(rets) < 2:
                    return np.nan
                df = pd.DataFrame({"ts": ts, "ret": rets})
                if df["ts"].dt.tz is not None:
                    df["ts"] = df["ts"].dt.tz_convert("UTC").dt.tz_localize(None)

                if freq == "W":
                    df["period"] = df["ts"].dt.floor("D") - pd.to_timedelta(df["ts"].dt.dayofweek, unit="D")
                else: # "M"
                    df["period"] = df["ts"].dt.floor("D") - pd.to_timedelta(df["ts"].dt.day - 1, unit="D")

                period_pnl = df.groupby("period")["ret"].sum()
                if len(period_pnl) < 2:
                    return np.nan

                mean_ret = float(period_pnl.mean())
                downside = np.minimum(period_pnl.to_numpy(dtype=np.float32), 0.0)
                downside_dev = float(np.sqrt(np.mean(downside**2)))

                if downside_dev > 1e-9:
                    return float(mean_ret / downside_dev)
                return np.nan

            res["weekly_sortino"] = _periodic_sortino("W")
            res["monthly_sortino"] = _periodic_sortino("M")
            return res

        valid_ts_mask = pd.notna(data["timestamp"])
        global_ts = pd.to_datetime(data.loc[valid_ts_mask, "timestamp"], errors="coerce", utc=True).to_numpy()

        # 0. Infrastructure: Component Extraction
        # Pre-calculating baselines for all relevant (target, horizon, side) combinations
        # that exist in the registry to avoid redundant expensive AUC/Entropy calculations.
        unique_contexts = set()
        selected_records = registry.to_dict("records")
        for row in selected_records:
            t_name = str(row.get("source_target", "unknown"))
            h_key = int(row.get("source_horizon", -1))
            s_side = str(row.get("side", "long"))
            unique_contexts.add((t_name, h_key, s_side))

        # TBM outcomes must respect each rule's own horizon (+2 bars flexibility) in pooled mode.
        cfg_assessment_horizon = int(self.cfg.get("horizon", 100))
        assessment_horizons = sorted(
            {
                max(int(h_key), 1)
                for _, h_key, _ in unique_contexts
                if pd.notna(h_key) and int(h_key) > 0
            }
        )
        if not assessment_horizons:
            assessment_horizons = [max(cfg_assessment_horizon, 1)]
        unique_tbm_contexts = sorted(
            {
                (max(int(h_key), 1) + 2, str(s_side))
                for _, h_key, s_side in unique_contexts
                if pd.notna(h_key) and int(h_key) > 0
            }
        )
        if not unique_tbm_contexts:
            unique_tbm_contexts = [
                (max(cfg_assessment_horizon, 1) + 2, "long"),
                (max(cfg_assessment_horizon, 1) + 2, "short"),
            ]

        horizons_cfg = self.cfg.get("triad_horizons", assessment_horizons)
        h_ref = max(horizons_cfg) if horizons_cfg else max(assessment_horizons)
        base_tp_atr = float(self.cfg.get("tbm_tp_atr", 1.25))
        base_sl_atr = float(self.cfg.get("tbm_sl_atr", 0.50))

        tprint(
            "TBM Configuration: per-rule assessment horizons="
            f"{assessment_horizons} -> realized horizons={[h for h, _ in unique_tbm_contexts]} "
            f"(rule horizon + 2), adaptive_tp_sl={adaptive_enabled}, h_ref={h_ref}"
        )

        tbm_outcome_cache: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if unique_tbm_contexts:
            tprint(
                f"Stage A: Pre-calculating TBM outcomes for {len(unique_tbm_contexts)} unique (H+2, Side) pairs..."
            )
            for tbm_horizon, tbm_side in unique_tbm_contexts:
                scale_factor = np.sqrt(float(tbm_horizon) / max(float(h_ref), 1.0))
                tp_atr = base_tp_atr * scale_factor
                sl_atr = base_sl_atr * scale_factor
                tbm_outcome_cache[(tbm_horizon, tbm_side)] = compute_tbm_outcomes_per_symbol(
                    data=data,
                    horizon=tbm_horizon,
                    tp_atr=tp_atr,
                    sl_atr=sl_atr,
                    side=tbm_side,
                )
            
        tprint(f"Stage A: Pre-calculating baselines for {len(unique_contexts)} unique target contexts...")

        baseline_cache = {}
        for t_name, h_key, s_side in unique_contexts:
            # Resolve the correct target array for this context
            current_baseline_target = None
            if triad_targets_map is not None:
                current_baseline_target = triad_targets_map.get((t_name, h_key))
            
            if current_baseline_target is None:
                # Fallback to local side-return if specific map missing
                current_baseline_target = bounded_target if bounded_target is not None else fwd_ret
            current_baseline_gross = self._resolve_side_gross_returns(
                side=s_side,
                source_horizon=h_key,
                triad_targets_map=triad_targets_map,
                fallback_fwd_ret=fwd_ret,
            )

            # Baseline metrics must be evaluated on the specific target used for the rule
            baseline_metrics = self._compute_baseline_auc(
                X,
                current_baseline_target,
                folds,
                eval_returns=current_baseline_gross,
                positive_return_threshold=float(
                    self.cfg.get("ridge_cost_pct", 0.003)
                ),
            )
            global_rets = current_baseline_gross[valid_ts_mask]
            uplift_baseline = _compute_baseline_population_metrics(global_rets, global_ts)

            baseline_cache[(t_name, h_key, s_side)] = {
                "global_auc": float(baseline_metrics["global_auc"]),
                "global_roc_auc": float(baseline_metrics["global_roc_auc"]),
                "global_pr_auc": float(baseline_metrics["global_pr_auc"]),
                "global_top_quartile_precision": float(baseline_metrics["global_top_quartile_precision"]),
                "global_cov": float(baseline_metrics["global_cov"]),
                "global_entropy": float(self._compute_entropy(current_baseline_target)),
                **uplift_baseline,
            }

        path_excursion_cache: Dict[Tuple[int, str], Dict[str, np.ndarray]] = {}
        unique_path_contexts = {
            (int(h_key), str(s_side))
            for _, h_key, s_side in unique_contexts
            if int(h_key) > 0
        }
        if unique_path_contexts and {"close", "high", "low"}.issubset(data.columns):
            atr_array = None
            if "atr" in data.columns:
                atr_array = pd.to_numeric(data["atr"], errors="coerce").to_numpy(dtype=np.float32)
            fallback_path_ret = np.asarray(fwd_ret, dtype=np.float32)
            for h_key, s_side in unique_path_contexts:
                try:
                    path_arrays = _compute_path_arrays_from_ohlc(
                        data=data,
                        side=s_side,
                        horizon=max(int(h_key), 1),
                        fallback_final_ret=fallback_path_ret,
                    )
                except Exception as exc:
                    tprint(
                        f"WARNING: Step2 path excursion precompute failed for H{h_key} [{s_side}]: {exc}"
                    )
                    continue
                if atr_array is not None:
                    path_mfe_ctx = np.where(
                        np.isfinite(path_arrays["mfe"]) & (atr_array > 1e-12),
                        path_arrays["mfe"] / atr_array,
                        np.nan,
                    ).astype(np.float32)
                    path_mae_ctx = np.where(
                        np.isfinite(path_arrays["mae"]) & (atr_array > 1e-12),
                        path_arrays["mae"] / atr_array,
                        np.nan,
                    ).astype(np.float32)
                else:
                    path_mfe_ctx = np.asarray(path_arrays["mfe"], dtype=np.float32)
                    path_mae_ctx = np.asarray(path_arrays["mae"], dtype=np.float32)
                path_excursion_cache[(h_key, s_side)] = {
                    "mfe": path_mfe_ctx,
                    "mae": path_mae_ctx,
                }
                
        max_ridge_candidates_total = int(
            self.cfg.get("max_ridge_candidates_total", 80)
        )
        family_rarity_bonus_strength = float(
            self.cfg.get("family_rarity_bonus_strength", 0.05)
        )
        family_rarity_bonus_cap = float(self.cfg.get("family_rarity_bonus_cap", 0.05))
        overlap_free_zone = float(self.cfg.get("ridge_overlap_free_zone", 0.30))
        cheap_rank_exponent = float(self.cfg.get("ridge_cheap_rank_exponent", 1.3))
        overlap_penalty_exponent = float(
            self.cfg.get("ridge_overlap_penalty_exponent", 1.8)
        )
        support_ratio_min = float(self.cfg.get("ridge_support_ratio_min", 0.70))
        penalty_strength = float(self.cfg.get("ridge_support_penalty_strength", 1.0))
        boost_strength = float(self.cfg.get("ridge_support_boost_strength", 1.0))
        center = TARGET_SUPPORT
        half_width = 0.025

        bucket_ridge_rows: Dict[
            Tuple[str, int], List[Tuple[float, str]]
        ] = collections.defaultdict(list)
        family_combo_count_by_bucket: Dict[
            Tuple[str, int], Dict[str, int]
        ] = collections.defaultdict(dict)
        family_rarity_bonus_by_key: Dict[
            Tuple[str, int], Dict[str, float]
        ] = collections.defaultdict(dict)
        tprint(
            f"Stage A: Starting global ridge regression selection..."
        )

        global_pool = []

        for bucket_key, entries in cheap_gate_rows.items():
            side = bucket_key[0]
            bucket_context_coverages: List[float] = []
            for _, canonical_key in entries:
                matching_rows = registry.loc[
                    registry["canonical_key"].astype(str) == str(canonical_key)
                ]
                if matching_rows.empty:
                    continue
                source_target = str(matching_rows.iloc[0].get("source_target", "unknown"))
                source_horizon_raw = matching_rows.iloc[0].get("source_horizon", -1)
                try:
                    source_horizon = (
                        int(source_horizon_raw)
                        if pd.notna(source_horizon_raw)
                        else int(bucket_key[1])
                    )
                except (TypeError, ValueError):
                    source_horizon = int(bucket_key[1])
                ctx_key = (source_target, source_horizon, side)
                baseline_data = baseline_cache.get(ctx_key)
                if baseline_data is not None:
                    bucket_context_coverages.append(
                        float(baseline_data.get("global_cov", np.nan))
                    )
            finite_coverages = [c for c in bucket_context_coverages if np.isfinite(c)]
            baseline_oof_coverage = (
                float(min(finite_coverages)) if finite_coverages else 1.0
            )
            if baseline_oof_coverage < min_oof_coverage:
                for _, canonical_key in entries:
                    cheap_gate_result[(bucket_key, canonical_key)] = (
                        True,
                        "insufficient_baseline_oof_coverage",
                    )
                continue
            bucket_ridge_rows[bucket_key].extend(entries)
            family_counts = collections.Counter(
                _infer_regime_family_combo(canonical_key)
                for _, canonical_key in entries
            )
            family_combo_count_by_bucket[bucket_key] = dict(family_counts)
            total_bucket = max(sum(family_counts.values()), 1)
            n_family_combos = max(len(family_counts), 1)
            target_share_family = 1.0 / float(n_family_combos)

            for family_combo, count in family_counts.items():
                actual_share = float(count) / total_bucket
                deficit = target_share_family - actual_share
                if deficit > 0:
                    bonus = min(
                        deficit * family_rarity_bonus_strength, family_rarity_bonus_cap
                    )
                else:
                    bonus = 0.0
                for _, canonical_key in entries:
                    if _infer_regime_family_combo(canonical_key) == family_combo:
                        family_rarity_bonus_by_key[bucket_key][canonical_key] = bonus

            global_pool.extend([
                (bucket_key, rank, canonical_key)
                for rank, canonical_key in entries
            ])

        tprint(
            f"Stage A: Ridge global pool size = {len(global_pool)} rules across "
            f"{len(set(bk for bk, _, _ in global_pool))} buckets "
            f"(target={max_ridge_candidates_total})"
)
        for bk in sorted(set(bk for bk, _, _ in global_pool)):
            n = sum(1 for b, _, _ in global_pool if b == bk)
            tprint(f"  bucket {bk}: {n} rules in pool")

        self.bucket_ridge_keys = collections.defaultdict(set)

        if True: # Always skip cascade dedup as per user request to only dedup at bucket level
            tprint("Stage A: Pool fits within target or cascade explicitly skipped, skipping cross-bucket cascade dedup")
            for bucket_key, rank, canonical_key in global_pool:
                self.bucket_ridge_keys[bucket_key].add(canonical_key)
        else:
            # ----------------------------------------------------------------
            # CASCADE DEDUP WITH GLOBAL MASKS + CROSS-BUCKET PENALTY
            # ----------------------------------------------------------------
            # 1. Build flat lists with raw boolean masks for ALL candidates.
            # 2. Score rules and sort descending (best first).
            # 3. Walk down overlap thresholds [0.95 → 0.6]; at each threshold,
            #    run a greedy "keep if max effective overlap < threshold" pass.
            #    Stop when ≤ max_ridge_candidates_total rules remain.
            # 4. Refill from rejected pool (score-ordered) if we end up < target.
            # Cross-bucket overlaps are discounted:
            #   - different side:            multiplier = cross_side_overlap_mult
            #   - same side, diff horizon:   multiplier = cross_horizon_overlap_mult
            #   - same side + same horizon:  multiplier = 1.0
            cross_side_overlap_mult = float(
                self.cfg.get("ridge_cross_side_overlap_mult", 0.50)
            )
            cross_horizon_overlap_mult = float(
                self.cfg.get("ridge_cross_horizon_overlap_mult", 0.70)
            )
            dedup_thresholds = list(
                self.cfg.get(
                    "ridge_dedup_thresholds",
                    [0.95, 0.925, 0.90, 0.875, 0.85, 0.825, 0.80, 0.75, 0.70, 0.65, 0.60],
                )
            )

            global_valid_keys = []
            global_valid_ranks = []
            global_bucket_keys = []
            global_masks = []
            global_supports = []
            global_scores = []

            for bucket_key, rank, canonical_key in global_pool:
                if canonical_key in mask_cache:
                    rule_mask = mask_cache[canonical_key]
                elif self.mask_resolver:
                    rule_mask = self.mask_resolver.get_mask(canonical_key)
                    mask_cache[canonical_key] = rule_mask
                else:
                    rule_mask = self._get_mask_for_rule(canonical_key, X)
                    mask_cache[canonical_key] = rule_mask

                supp = int(np.sum(rule_mask))
                if supp < 1:
                    continue

                s = supp / max(len(rule_mask), 1)
                if s < (center - half_width):
                    w = 1.0 - penalty_strength * (center - half_width - s) / (center - half_width)
                elif s < center:
                    w = 1.0 + boost_strength * (s - (center - half_width)) / half_width
                elif s < (center + half_width):
                    w = 1.0 + boost_strength * ((center + half_width) - s) / half_width
                else:
                    w = 1.0 - penalty_strength * (s - (center + half_width)) / (center + half_width)
                w = float(np.clip(w, 0.1, 2.0))

                family_bonus = family_rarity_bonus_by_key.get(bucket_key, {}).get(canonical_key, 0.0)
                score = rank * (1.0 + w) * (1.0 + family_bonus)

                global_valid_keys.append(canonical_key)
                global_valid_ranks.append(rank)
                global_bucket_keys.append(bucket_key)
                global_masks.append(rule_mask.astype(bool))
                global_supports.append(supp)
                global_scores.append(score)

            if not global_valid_keys:
                pass  # nothing to select
            else:
                supports_arr = np.array(global_supports, dtype=np.float32)
                scores_arr = np.array(global_scores, dtype=np.float64)

                # Sort ALL candidates by score descending (highest quality first)
                order = np.argsort(-scores_arr)

                def effective_f1(i: int, j: int) -> float:
                    """F1 overlap between rules i and j, with cross-bucket discount."""
                    supp_i = supports_arr[i]
                    supp_j = supports_arr[j]
                    supp_ratio = min(supp_i, supp_j) / max(supp_i, supp_j + 1e-9)
                    if supp_ratio < support_ratio_min:
                        return 0.0
                    inter = float(np.sum(global_masks[i] & global_masks[j]))
                    raw_f1 = 2.0 * inter / (supp_i + supp_j + 1e-9)
                    # Apply discount for cross-bucket pairs
                    bi = global_bucket_keys[i]
                    bj = global_bucket_keys[j]
                    if bi[0] != bj[0]:  # different side
                        return raw_f1 * cross_side_overlap_mult
                    elif bi[1] != bj[1]:  # same side, different horizon
                        return raw_f1 * cross_horizon_overlap_mult
                    return raw_f1  # same bucket: full overlap

                def run_dedup_pass(threshold: float):
                    """Greedy dedup: iterate in score order, keep rule if its
                    effective overlap with every kept rule is below threshold."""
                    kept = []
                    rejected_pool = []
                    for idx in order:
                        max_ov = max(
                            (effective_f1(idx, k) for k in kept), default=0.0
                        )
                        if max_ov < threshold:
                            kept.append(idx)
                        else:
                            rejected_pool.append(idx)
                    return kept, rejected_pool

                selected_indices = list(order)   # fallback: keep all (sorted)
                rejected_pool_final: List[int] = []

                for threshold in dedup_thresholds:
                    kept, rejected = run_dedup_pass(threshold)
                    tprint(
                        f"Stage A: Ridge dedup @ threshold={threshold:.3f} "
                        f"-> kept={len(kept)} rejected={len(rejected)}"
                    )
                    if len(kept) <= max_ridge_candidates_total:
                        selected_indices = kept
                        rejected_pool_final = rejected
                        break

                # Refill from rejected pool (diversity-aware, squared overlap penalty)
                if len(selected_indices) < max_ridge_candidates_total and rejected_pool_final:
                    deficit = max_ridge_candidates_total - len(selected_indices)
                    alpha = float(self.cfg.get("ridge_refill_diversity_alpha", 0.3))
                    
                    diversity_scores = []
                    for idx in rejected_pool_final:
                        base_score = scores_arr[idx]
                        max_ov = max((effective_f1(idx, k) for k in selected_indices), default=0.0)
                        overlap_penalty = max(0.0, max_ov - 0.3)
                        selection_score = base_score - alpha * (overlap_penalty ** 2)
                        diversity_scores.append((idx, selection_score))
                    
                    diversity_scores.sort(key=lambda x: x[1], reverse=True)
                    refill = [ds[0] for ds in diversity_scores[:deficit]]
                    selected_indices = selected_indices + refill
                    
                    tprint(
                        f"Stage A: Ridge diversity refill +{len(refill)} "
                        f"-> total={len(selected_indices)} (alpha={alpha:.2f})"
                    )

                for i in selected_indices:
                    self.bucket_ridge_keys[global_bucket_keys[i]].add(global_valid_keys[i])



        total_ridge_selected = sum(
            len(keys) for keys in self.bucket_ridge_keys.values()
        )
        tprint(
            f"Stage A: Ridge regression selection complete - {total_ridge_selected} rules selected for final assessment"
        )

        # 0. Infrastructure: Component Extraction
        # Pre-calculating baselines and TBM outcomes for all relevant combinations
        # that exist in the registry to avoid redundant expensive calculations.
        unique_contexts = set()
        unique_tbm_contexts = set()
        selected_records = registry.to_dict("records")
        for row in selected_records:
            t_name = str(row.get("source_target", "unknown"))
            h_key = int(row.get("source_horizon", -1))
            s_side = str(row.get("side", "long"))
            unique_contexts.add((t_name, h_key, s_side))
            unique_tbm_contexts.add((max(int(h_key) + 2, 1), s_side))
            
        tprint(f"Stage A: Pre-calculating baselines for {len(unique_contexts)} unique target contexts...")

        valid_ts_mask = pd.notna(data["timestamp"])
        global_ts = pd.to_datetime(data.loc[valid_ts_mask, "timestamp"], errors="coerce", utc=True).to_numpy()

        baseline_cache = {}
        for t_name, h_key, s_side in unique_contexts:
            # Resolve the correct target array for this context
            current_baseline_target = None
            if triad_targets_map is not None:
                current_baseline_target = triad_targets_map.get((t_name, h_key))
            
            if current_baseline_target is None:
                current_baseline_target = bounded_target if bounded_target is not None else fwd_ret
            current_baseline_target = self._transform_side_target(
                target_name=t_name,
                target_values=current_baseline_target,
                side=s_side,
                atr_frac=atr_frac_for_targets,
            )
            current_baseline_gross = self._resolve_side_gross_returns(
                side=s_side,
                source_horizon=h_key,
                triad_targets_map=triad_targets_map,
                fallback_fwd_ret=fwd_ret,
            )

            baseline_metrics = self._compute_baseline_auc(
                X,
                current_baseline_target,
                folds,
                eval_returns=current_baseline_gross,
                positive_return_threshold=float(
                    self.cfg.get("ridge_cost_pct", 0.003)
                ),
            )
            global_rets = current_baseline_gross[valid_ts_mask]
            uplift_baseline = _compute_baseline_population_metrics(global_rets, global_ts)

            baseline_cache[(t_name, h_key, s_side)] = {
                "global_auc": float(baseline_metrics["global_auc"]),
                "global_roc_auc": float(baseline_metrics["global_roc_auc"]),
                "global_pr_auc": float(baseline_metrics["global_pr_auc"]),
                "global_top_quartile_precision": float(baseline_metrics["global_top_quartile_precision"]),
                "global_cov": float(baseline_metrics["global_cov"]),
                "global_entropy": float(self._compute_entropy(current_baseline_target)),
                **uplift_baseline,
            }

        total_ridge_selected = len(selected_records)
        tprint(
            f"Stage A: Starting final assessment for {total_ridge_selected} rules (batches of {batch_size})..."
        )

        final_assessment_start_ts = time.perf_counter()
        assessed_progress = 0
        for row in selected_records:
            ridge_details: Dict[str, Any] = {}
            assessed_progress += 1
            if assessed_progress == 1 or assessed_progress % batch_size == 0 or assessed_progress == total_ridge_selected:
                tprint(
                    f"Stage A: Final assessment progress {assessed_progress}/{total_ridge_selected} "
                    f"rules (bunch of {batch_size})"
                )
            canonical_key = str(row["canonical_key"])
            if canonical_key in mask_cache:
                mask = mask_cache[canonical_key]
            elif self.mask_resolver:
                mask = self.mask_resolver.get_mask(canonical_key)
                mask_cache[canonical_key] = mask
            else:
                mask = self._get_mask_for_rule(canonical_key, X)
                mask_cache[canonical_key] = mask
            
            if np.sum(mask) < 20: # Sanity check for rule support
                continue

            # 0. Infrastructure: Component Extraction
            slots = parse_slot_map(
                canonical_key,
                self.cfg.get("slot_order", ("trigger", "location", "regime")),
            )

            side = str(row.get("side", "long"))
            horizon_raw = row.get("source_horizon", -1)
            try:
                horizon_key = int(horizon_raw) if pd.notna(horizon_raw) else -1
            except (TypeError, ValueError):
                horizon_key = -1
            target_name = str(row.get("source_target", "unknown"))
            tbm_horizon = max(horizon_key + 2, 1)

            # Dynamic Target Resolution:
            # If a global triad_targets_map is provided, we pick the rules source target.
            # Otherwise, we fallback to the provided bounded_target or fwd_ret.
            current_target_ret = None
            if triad_targets_map is not None:
                current_target_ret = triad_targets_map.get((target_name, horizon_key))
            
            if current_target_ret is None:
                current_target_ret = bounded_target if bounded_target is not None else fwd_ret
            current_target_ret = self._transform_side_target(
                target_name=target_name,
                target_values=current_target_ret,
                side=side,
                atr_frac=atr_frac_for_targets,
            )
            current_gross_return_ret = self._resolve_side_gross_returns(
                side=side,
                source_horizon=horizon_key,
                triad_targets_map=triad_targets_map,
                fallback_fwd_ret=fwd_ret,
            )
            current_gross_return_ret = self._resolve_side_gross_returns(
                side=side,
                source_horizon=horizon_key,
                triad_targets_map=triad_targets_map,
                fallback_fwd_ret=fwd_ret,
            )
            current_target_nan_reasons = (
                target_nan_reasons.get((target_name, horizon_key))
                if isinstance(target_nan_reasons, dict)
                else target_nan_reasons
            )

            # The bucket grouping is determined strictly by side and horizon
            group_bucket_key = (side, horizon_key)

            target_ret = current_target_ret
            
            # Context-aware baseline lookup
            ctx_key = (target_name, horizon_key, side)
            baseline_data = baseline_cache.get(ctx_key, {})
            
            global_auc = float(baseline_data.get("global_auc", 0.5))
            global_roc_auc = float(baseline_data.get("global_roc_auc", 0.5))
            global_pr_auc = float(baseline_data.get("global_pr_auc", 0.5))
            global_top_quartile_precision = float(baseline_data.get("global_top_quartile_precision", 0.5))
            global_entropy = float(baseline_data.get("global_entropy", 1.0))
            baseline_oof_coverage = float(baseline_data.get("global_cov", 1.0))

            # 1. Triple Barrier
            # Check if adaptive TP/SL is enabled
            adaptive_enabled = self.cfg.get("adaptive_tp_sl_enabled", False)
            if adaptive_enabled:
                # Compute adaptive TP/SL for this specific rule
                adaptive_tp_atr, adaptive_sl_atr = self._compute_adaptive_tp_sl(
                    mask=mask,
                    fwd_ret=current_gross_return_ret,
                    atr=data["atr"].to_numpy(),
                    oof_preds=ridge_details.get("oof_preds", np.full(len(data), np.nan)),
                    close=data["close"].to_numpy(),
                    horizon=tbm_horizon,
                    side=side,
                )
                
                # Compute TBM outcomes with adaptive TP/SL
                tbm_outcomes = compute_tbm_outcomes_per_symbol(
                    data=data,
                    horizon=tbm_horizon,
                    tp_atr=adaptive_tp_atr,
                    sl_atr=adaptive_sl_atr,
                    side=side,
                )
                rule_tp_f, rule_sl_f, rule_to_f = tbm_outcomes
            else:
                # Use pre-calculated TBM outcomes for this (horizon+2, side)
                tbm_outcomes = tbm_outcome_cache.get((tbm_horizon, side))
                if tbm_outcomes is None:
                    # Final fallback if cache missing
                    tbm_outcomes = compute_tbm_outcomes_per_symbol(
                        data=data,
                        horizon=tbm_horizon,
                        tp_atr=float(self.cfg.get("tp_atr", 2.0)),
                        sl_atr=float(self.cfg.get("sl_atr", 2.0)),
                        side=side,
                    )
                    tbm_outcome_cache[(tbm_horizon, side)] = tbm_outcomes
                
                rule_tp_f, rule_sl_f, rule_to_f = tbm_outcomes

            tbm_metrics = self._compute_tbm_metrics(
                mask, rule_tp_f, rule_sl_f, rule_to_f, current_gross_return_ret
            )

            # 2-6. Cached cheap stats (no Ridge work)
            cheap = _get_or_compute_cheap_stats(
                canonical_key,
                side,
                mask,
                target_name=target_name,
                horizon_key=horizon_key,
                side_returns=current_gross_return_ret,
            )

            sign_consistency = float(cheap["sign_consistency"])
            support_pct = float(cheap["support_pct"])
            support_ok = bool(cheap["support_ok"])
            support_score = float(cheap["support_score"])
            avg_trades = float(cheap["avg_trades"])
            density_dispersion = float(cheap["density_dispersion"])
            tail_ratio = float(cheap["tail_ratio"])
            mae = float(cheap["mae"])
            mfe = float(cheap["mfe"])
            mean_ret_global = float(cheap["mean_ret_global"])
            mean_ret_mask = float(cheap["mean_ret_mask"])
            ret_uplift = float(cheap["ret_uplift"])

            rejected, rejection_reason = cheap_gate_result.get(
                (group_bucket_key, canonical_key), (False, "")
            )

            # 7. Learnability (Efficiency Frontier) - expensive section
            subset_oof_coverage = 0.0
            mask_auc = np.nan
            mask_roc_auc = np.nan
            mask_pr_auc = np.nan
            mask_top_quartile_precision = np.nan
            auc_lift = np.nan
            top_quartile_precision_lift = np.nan
            entropy_red = np.nan
            ridge_round_fee = float(self.cfg.get("ridge_cost_pct", 0.003))
            ridge_trade_metrics: Dict[str, Any] = {
                "threshold_star": np.nan,
                "threshold_star_lowest_positive": np.nan,
                "threshold_star_optimal_pnl": np.nan,
                "threshold_star_best_pnl_threshold": np.nan,
                "ridge_pnl_gross_raw": 0.0,
                "ridge_pnl_gross_raw_at_optimal_threshold": np.nan,
                "ridge_pnl_raw": 0.0,
                "ridge_pnl_raw_at_optimal_threshold": np.nan,
                "avg_pnl_per_day": np.nan,
                "avg_pnl_per_active_symbol_day": np.nan,
                "ridge_trade_sortino_7d": 0.0,
                "ridge_trade_sortino_30d": 0.0,
                "ridge_trade_sortino_90d": 0.0,
                "ridge_trade_sortino_composite": 0.0,
                "trades_per_symbol_day_above_threshold_star": 0.0,
                "valid_symbol_days_observed": 0,
                "total_trades": 0,
                "threshold_search_mode": "grid",
                "n_quantiles_evaluated": 0,
                "n_thresholds_evaluated": 0,
                "n_unique_thresholds_evaluated": 0,
                "score_min": np.nan,
                "score_max": np.nan,
                "score_std": np.nan,
                "n_unique_scores": 0,
                "rejected": True,
                "reject_reason": {"reason": "did_not_run_ridge"},
                "realized_trades": [],
                "gross_weighted_returns": [],
                "net_weighted_returns": []
            }
            if not rejected:
                run_ridge = False
                if (
                    hasattr(self, "bucket_ridge_keys")
                    and group_bucket_key in self.bucket_ridge_keys
                ):
                    if canonical_key in self.bucket_ridge_keys[group_bucket_key]:
                        run_ridge = True

                if run_ridge:
                    ridge_start = time.time()
                    tprint(
                        f"Stage A: Ridge learnability start {assessed_progress}/{total_ridge_selected} "
                        f"key={canonical_key[:120]}"
                    )
                    source_horizon = int(row.get("source_horizon", bucket_key[1]))
                    rule_side = str(row.get("side", side))
                    path_excursions = path_excursion_cache.get((source_horizon, rule_side), {})
                    ridge_details = self._compute_subset_ridge_details(
                        X,
                        target_ret,
                        mask,
                        folds,
                        tp_f=rule_tp_f,
                        target_nan_reasons=current_target_nan_reasons,
                        path_mfe=path_excursions.get("mfe"),
                        path_mae=path_excursions.get("mae"),
                        side=side,
                    )
                    mask_oof_corr = float(ridge_details["mask_oof_corr"])
                    mask_oof_r2 = float(ridge_details["mask_oof_r2"])
                    mask_top_quartile_precision = float(ridge_details.get("top_quartile_precision", np.nan))
                    subset_oof_coverage = float(ridge_details["coverage"])
                    ridge_trade_metrics = self._compute_ranked_ridge_trade_metrics(
                        data=data,
                        directional_returns=current_gross_return_ret,
                        mask=mask,
                        folds=folds,
                        horizon=source_horizon,
                        oof_preds=np.asarray(
                            ridge_details["oof_preds"], dtype=np.float32
                        ),
                        round_fee=ridge_round_fee,
                    )
                    tprint(
                        f"Stage A: Ridge learnability done {assessed_progress}/{total_ridge_selected} "
                        f"key={canonical_key[:120]} "
                        f"mask_oof_corr={mask_oof_corr if np.isfinite(mask_oof_corr) else np.nan:.6f} "
                        f"coverage={subset_oof_coverage:.4f} "
                        f"elapsed={time.time() - ridge_start:.2f}s"
                    )
                    mask_top_quartile_precision = self._compute_top_quartile_precision(
                        oof_preds=np.asarray(ridge_details["oof_preds"], dtype=np.float32),
                        y=np.asarray(current_gross_return_ret, dtype=np.float32),
                        mask=mask,
                        tp_f=rule_tp_f,
                        fwd_ret_threshold=float(
                            self.cfg.get("ridge_cost_pct", 0.003)
                        ),
                        top_pct=0.75,
                        min_samples=20,
                    )
                    if np.isfinite(mask_top_quartile_precision) and np.isfinite(global_top_quartile_precision):
                        top_quartile_precision_lift = mask_top_quartile_precision - global_top_quartile_precision
                else:
                    subset_oof_coverage = float(np.mean(mask))
                    rejected, rejection_reason = True, "not_in_top_ridge_candidates"

                mask_entropy = self._compute_entropy(target_ret[mask])
                entropy_red = 1.0 - (mask_entropy / (global_entropy + 1e-9))
                if subset_oof_coverage < min_oof_coverage:
                    rejected, rejection_reason = (
                        True,
                        "insufficient_subset_oof_coverage",
                    )

            # 8. Event-based Expected Value
            if adaptive_enabled:
                tp_payoff = float(adaptive_tp_atr)
                sl_payoff = float(adaptive_sl_atr)
            else:
                tp_payoff = float(self.cfg.get("tp_atr", 2.0))
                sl_payoff = float(self.cfg.get("sl_atr", 2.0))
            timeout_payoff = float(np.nanmean(current_gross_return_ret[mask]))

            ev_per_event = (
                tbm_metrics["tp_rate"] * tp_payoff
                - tbm_metrics["sl_rate"] * sl_payoff
                + tbm_metrics["timeout_rate"] * timeout_payoff
            )

            # Apply ev_per_event hard gate
            if ev_per_event <= 0.0 and not rejected:
                rejected, rejection_reason = True, "ev_per_event_less_than_or_equal_to_zero"
                
                # Diagnostic logging for EV-based rejection
                fee_frac = (ridge_round_fee / abs(mean_ret_mask)) if abs(mean_ret_mask) > 1e-9 else np.nan
                tprint(
                    f"DIAGNOSTIC: Rule rejected (Low EV) key={canonical_key[:60]}... "
                    f"GrossEV={ev_per_event:.6f} AvgMove={mean_ret_mask:.6f} HitRate={tbm_metrics['tp_rate']:.2%} "
                    f"MAE={mae:.6f} MFE={mfe:.6f} FeeRatio={fee_frac:.2f}"
                )

            # If the threshold finding logic failed
            if ridge_trade_metrics.get("rejected", False) and not rejected:
                rejected = True
                rejection_reason = ridge_trade_metrics.get("reject_reason", {}).get("reason", "threshold_star_search_failed")
                
                if rejection_reason == "no positive post-fee profit threshold":
                    # Diagnostic logging for post-fee rejection
                    fee_frac = (ridge_round_fee / abs(mean_ret_mask)) if abs(mean_ret_mask) > 1e-9 else np.nan

                    # Extract best TP/SL and profit information from reject_reason
                    best_pnl_candidate = ridge_trade_metrics.get("reject_reason", {}).get("best_pnl_candidate", np.nan)
                    threshold_star_optimal = ridge_trade_metrics.get("reject_reason", {}).get("threshold_star_optimal_pnl", np.nan)

                    # Get TP/SL information (adaptive if enabled, otherwise use default)
                    if adaptive_enabled:
                        tp_atr_str = f"{adaptive_tp_atr:.2f}ATR"
                        sl_atr_str = f"{adaptive_sl_atr:.2f}ATR"
                    else:
                        tp_atr_str = "default"
                        sl_atr_str = "default"

                    # Compute average post-fee profit per trade from the best threshold candidate.
                    avg_trades_est = cheap.get("avg_trades", np.nan)
                    if np.isfinite(best_pnl_candidate) and np.isfinite(avg_trades_est) and avg_trades_est > 0:
                        support_pct = cheap.get("support_pct", 0.0)
                        n_samples = len(data)
                        total_trades_est = n_samples * support_pct * avg_trades_est
                        if total_trades_est > 0:
                            avg_net_profit_per_trade = best_pnl_candidate / total_trades_est
                        else:
                            avg_net_profit_per_trade = np.nan
                    else:
                        avg_net_profit_per_trade = np.nan

                    # AvgMove should be absolute value (directional return)
                    avg_move = abs(mean_ret_mask)

                    tprint(
                        f"DIAGNOSTIC: Rule rejected (Post-Fee) key={canonical_key[:60]}... "
                        f"TP={tp_atr_str} SL={sl_atr_str} "
                        f"BestThresh={threshold_star_optimal:.3f} BestNetPnL={best_pnl_candidate:.6f} "
                        f"AvgNetProfit={avg_net_profit_per_trade:.6f} "
                        f"GrossEV={ev_per_event:.6f} AvgMove={avg_move:.6f} HitRate={tbm_metrics['tp_rate']:.2%} "
                        f"MAE={mae:.6f} MFE={mfe:.6f} FeeRatio={fee_frac:.2f}"
                    )

            # Fetch cheap_rank for Final Regime Score
            cheap_rank = bucket_cheap_ranks.get(group_bucket_key, {}).get(
                canonical_key, -np.inf
            )
            if not np.isfinite(cheap_rank):
                cheap_rank = 0.0
            family_rarity_bonus = float(
                family_rarity_bonus_by_key.get(group_bucket_key, {}).get(
                    canonical_key, 0.0
                )
            )

            # Use composite_score_step1 from row instead of computing regime_score
            composite_score_step1 = float(row.get("composite_score_step1", np.nan))
            if not np.isfinite(composite_score_step1):
                composite_score_step1 = 0.0

            # Compute new regime_score with overall_mask_uplift
            overall_mask_uplift = np.nan
            regime_score = (
                0.5 * overall_mask_uplift
                + 0.25 * ret_uplift
                + 0.25 * ev_per_event
                + family_rarity_bonus
            )
            if not np.isfinite(regime_score):
                regime_score = composite_score_step1

            # Production classification
            rule_for_classification = {
                "n_folds": row.get("n_folds", 0),
                "healthy_fold_count": healthy_fold_count,
                "healthy_fold_ratio": healthy_fold_ratio,
                "presence_freq": row.get("presence_freq", 0.0),
                "directional_mean_ret": row.get("directional_mean_ret", np.nan),
                "min_support_actual": row.get("min_support_actual", 0),
                "hurdle_excess": row.get("hurdle_excess", np.nan),
                "trade_path_quality_score": row.get("trade_path_quality_score", np.nan),
                "is_structurally_sound": not rejected,
                "sign_consistency": sign_consistency,
            }
            (
                production_classification,
                classification_diagnostics,
            ) = classify_rule_production_quality(rule=rule_for_classification)

            # Rule type classification
            rule_type_class = classify_rule_type(
                directional_mean_ret=row.get("directional_mean_ret", np.nan),
                mean_uplift=row.get("mean_uplift", np.nan),
                sign_consistency=sign_consistency,
                required_hurdle=row.get("required_hurdle", 0.0),
            )

            assessment_results.append(
                {
                    "canonical_key": row["canonical_key"],
                    "source_target": target_name,
                    "source_horizon": horizon_key,
                    "side": side,
                    "trigger": slots.get("trigger", "*"),
                    "location": slots.get("location", "*"),
                    "regime": slots.get("regime", "*"),
                    "regime_score": regime_score,
                    "is_structurally_sound": not rejected,
                    "rejection_reason": rejection_reason,
                    "support_count": cheap.get("support_count", int(np.sum(mask))),
                    "support_pct": support_pct,
                    "directional_mean_ret": float(
                        row.get("directional_mean_ret", np.nan)
                    ),
                    "presence_freq": float(row.get("presence_freq", np.nan)),
                    "min_support_actual": float(row.get("min_support_actual", np.nan)),
                    "n_folds": int(row.get("n_folds", 0) or 0),
                    "healthy_fold_count": healthy_fold_count,
                    "healthy_fold_ratio": healthy_fold_ratio,
                    "mean_uplift": float(row.get("mean_uplift", np.nan)),
                    "required_hurdle": float(row.get("required_hurdle", np.nan)),
                    "hurdle_excess": float(row.get("hurdle_excess", np.nan)),
                    "trade_path_quality_score": float(
                        row.get("trade_path_quality_score", np.nan)
                    ),
                    "quality_stability_score": float(
                        row.get("quality_stability_score", np.nan)
                    ),
                    "full_quality_score": float(row.get("full_quality_score", np.nan)),
                    "composite_score": float(row.get("composite_score", np.nan)),
                    "rule_gain_score": float(row.get("rule_gain_score", np.nan)),
                    "rule_split_score": float(row.get("rule_split_score", np.nan)),
                    "rule_model_importance_score": float(
                        row.get("rule_model_importance_score", np.nan)
                    ),
                    "family_rarity_bonus": family_rarity_bonus,
                    "support_ok": support_ok,
                    "support_score": support_score,
                    "avg_trades_per_day": avg_trades,
                    "density_dispersion": density_dispersion,
                    "tail_ratio": tail_ratio,
                    "mae": mae,
                    "mfe": mfe,
                    "mean_ret_global": mean_ret_global,
                    "mean_ret_mask": mean_ret_mask,
                    "ret_uplift": ret_uplift,
                    "auc_lift": auc_lift,
                    "top_quartile_precision": mask_top_quartile_precision,
                    "top_quartile_precision_lift": top_quartile_precision_lift,
                    "learn_eff_ratio": np.nan,  # Deprecated - same as auc_lift
                    "subset_oof_coverage": subset_oof_coverage,
                    "baseline_oof_coverage": baseline_oof_coverage,
                    "mask_oof_corr": mask_auc,
                    "mask_roc_auc": mask_roc_auc,
                    "mask_pr_auc": mask_pr_auc,
                    "global_oof_corr": global_auc,
                    "global_roc_auc": global_roc_auc,
                    "global_pr_auc": global_pr_auc,
                    "global_entropy": global_entropy,
                    "entropy_reduction": entropy_red,
                    "tp_rate": tbm_metrics["tp_rate"],
                    "sl_rate": tbm_metrics["sl_rate"],
                    "timeout_rate": tbm_metrics["timeout_rate"],
                    "ev_per_trade": tbm_metrics["ev_per_trade"],
                    "ev_per_event": ev_per_event,
                    "win_rate_conditional": tbm_metrics["win_rate_conditional"],
                    "win_rate_unconditional": tbm_metrics["win_rate_unconditional"],
                    "threshold_star": ridge_trade_metrics.get("threshold_star", np.nan),
                    "threshold_star_lowest_positive": ridge_trade_metrics.get("threshold_star_lowest_positive", np.nan),
                    "threshold_star_optimal_pnl": ridge_trade_metrics.get("threshold_star_optimal_pnl", np.nan),
                    "threshold_star_best_pnl_threshold": ridge_trade_metrics.get("threshold_star_best_pnl_threshold", np.nan),
                    "ridge_pnl_gross_raw": ridge_trade_metrics.get("ridge_pnl_gross_raw", 0.0),
                    "ridge_pnl_gross_raw_at_optimal_threshold": ridge_trade_metrics.get("ridge_pnl_gross_raw_at_optimal_threshold", np.nan),
                    "ridge_pnl_raw": ridge_trade_metrics.get("ridge_pnl_raw", 0.0),
                    "ridge_pnl_raw_at_optimal_threshold": ridge_trade_metrics.get("ridge_pnl_raw_at_optimal_threshold", np.nan),
                    "avg_trades_per_day": ridge_trade_metrics.get("avg_trades_per_day", np.nan),
                    "avg_pnl_per_day": ridge_trade_metrics.get("avg_pnl_per_day", np.nan),
                    "avg_pnl_per_active_symbol_day": ridge_trade_metrics.get("avg_pnl_per_active_symbol_day", np.nan),
                    "ridge_trade_sortino_7d": ridge_trade_metrics.get("ridge_trade_sortino_7d", 0.0),
                    "ridge_trade_sortino_30d": ridge_trade_metrics.get("ridge_trade_sortino_30d", 0.0),
                    "ridge_trade_sortino_90d": ridge_trade_metrics.get("ridge_trade_sortino_90d", 0.0),
                    "ridge_trade_sortino_composite": ridge_trade_metrics.get("ridge_trade_sortino_composite", 0.0),
                    "trades_per_symbol_day_above_threshold_star": ridge_trade_metrics.get("trades_per_symbol_day_above_threshold_star", 0.0),
                    "valid_symbol_days_observed": ridge_trade_metrics.get("valid_symbol_days_observed", 0),
                    "total_trades": ridge_trade_metrics.get("total_trades", 0),
                    "threshold_search_mode": ridge_trade_metrics.get("threshold_search_mode", "grid"),
                    "threshold_selection_policy": ridge_trade_metrics.get("threshold_selection_policy", np.nan),
                    "n_quantiles_evaluated": ridge_trade_metrics.get("n_quantiles_evaluated", 0),
                    "n_thresholds_evaluated": ridge_trade_metrics.get("n_thresholds_evaluated", 0),
                    "n_unique_thresholds_evaluated": ridge_trade_metrics.get("n_unique_thresholds_evaluated", 0),
                    "score_min": ridge_trade_metrics.get("score_min", np.nan),
                    "score_max": ridge_trade_metrics.get("score_max", np.nan),
                    "score_std": ridge_trade_metrics.get("score_std", np.nan),
                    "n_unique_scores": ridge_trade_metrics.get("n_unique_scores", 0),
                    "realized_trades": ridge_trade_metrics.get("realized_trades", []),
                    "gross_weighted_returns": ridge_trade_metrics.get("gross_weighted_returns", []),
                    "net_weighted_returns": ridge_trade_metrics.get("net_weighted_returns", []),
                    "weighted_fee_returns": ridge_trade_metrics.get("weighted_fee_returns", []),
                    "avg_fee_per_trade": ridge_trade_metrics.get("avg_fee_per_trade", np.nan),
                    "avg_gross_move_per_trade": ridge_trade_metrics.get("avg_gross_move_per_trade", np.nan),
                    "avg_position_weight": ridge_trade_metrics.get("avg_position_weight", np.nan),
                    "learnability_step_c_score": float(
                        row.get("learnability_step_c_score", np.nan)
                    ),
                    "production_classification": production_classification,
                    "classification_diagnostics": json.dumps(
                        classification_diagnostics
                    ),
                    "rule_type_class": rule_type_class,
                    "mask_oof_corr": ridge_details.get("mask_oof_corr", np.nan),
                    "mask_oof_r2": ridge_details.get("mask_oof_r2", np.nan),
                    "fold_sign_consistency": ridge_details.get("fold_sign_consistency", np.nan),
                    "positive_fold_fraction": ridge_details.get("positive_fold_fraction", np.nan),
                    "negative_fold_fraction": ridge_details.get("negative_fold_fraction", np.nan),
                    "fold_pnl_std": ridge_details.get("fold_pnl_std", np.nan),
                    "ic_series": ridge_details.get("ic_series", np.array([])),  # Per-fold IC for Pareto overlap
                    "decile_monotonic_spearman": ridge_details.get("decile_monotonic_spearman", np.nan),
                    "top_decile_mean_target": ridge_details.get("top_decile_mean_target", np.nan),
                    "bottom_decile_mean_target": ridge_details.get("bottom_decile_mean_target", np.nan),
                    "decile_spread_mean": ridge_details.get("decile_spread_mean", np.nan),
                    "target_nan_total_train": ridge_details.get("target_nan_total_train", 0),
                    "target_nan_total_val": ridge_details.get("target_nan_total_val", 0),
                    "stage1_model_profile": "weak",
                    "stage2_model_profile": "weak",
                    "stage2_rescored": False,
                    "stage1_top70_survivor": False,
                    "selected_for_final_registry": False,
                }
            )            # 5b. Compute Universe-Relative Uplift Metrics for the Mask
            mask_ts = global_ts[mask[valid_ts_mask]]
            mask_rets = target_ret[valid_ts_mask][mask[valid_ts_mask]]

            mask_metrics = _compute_baseline_population_metrics(mask_rets, mask_ts)

            eps = 1e-9

            # Tail p5 ratio and uplift
            p5_mask = mask_metrics["p5"]
            p5_global = float(baseline_data.get("p5", np.nan))
            if np.isfinite(p5_mask) and np.isfinite(p5_global):
                tail_p5_ratio = float(p5_mask / p5_global) if abs(p5_global) > eps else np.nan
                tail_uplift = 10.0 * (p5_mask - p5_global) / max(abs(p5_mask) + abs(p5_global), eps)
            else:
                tail_p5_ratio = np.nan
                tail_uplift = np.nan

            # Mean p75 ratio and uplift
            p75_mask = mask_metrics["mean_p75_ret"]
            p75_global = float(baseline_data.get("mean_p75_ret", np.nan))
            if np.isfinite(p75_mask) and np.isfinite(p75_global):
                mean_p75_ratio = float(p75_mask / p75_global) if abs(p75_global) > eps else np.nan
                mean_75_uplift = 10.0 * (p75_mask - p75_global) / max(abs(p75_mask) + abs(p75_global), eps)
            else:
                mean_p75_ratio = np.nan
                mean_75_uplift = np.nan

            # Weekly sortino ratio and uplift
            w_sort_mask = mask_metrics["weekly_sortino"]
            w_sort_global = float(baseline_data.get("weekly_sortino", np.nan))
            if np.isfinite(w_sort_mask) and np.isfinite(w_sort_global):
                weekly_sortino_ratio = float(w_sort_mask / w_sort_global) if abs(w_sort_global) > eps else np.nan
                sortino_weekly_uplift = 10.0 * (w_sort_mask - w_sort_global) / max(abs(w_sort_mask) + abs(w_sort_global), eps)
            else:
                weekly_sortino_ratio = np.nan
                sortino_weekly_uplift = np.nan

            # Monthly sortino ratio and uplift
            m_sort_mask = mask_metrics["monthly_sortino"]
            m_sort_global = float(baseline_data.get("monthly_sortino", np.nan))
            if np.isfinite(m_sort_mask) and np.isfinite(m_sort_global):
                monthly_sortino_ratio = float(m_sort_mask / m_sort_global) if abs(m_sort_global) > eps else np.nan
                sortino_monthly_uplift = 10.0 * (m_sort_mask - m_sort_global) / max(abs(m_sort_mask) + abs(m_sort_global), eps)
            else:
                monthly_sortino_ratio = np.nan
                sortino_monthly_uplift = np.nan

            # Rank IC uplift (same scaling formula as other uplift components)
            rank_ic_mask = float(row.get("mask_ic_uplift", np.nan))
            rank_ic_global = float(row.get("delta_within_mask_ic", 0.0))  # Baseline reference
            if np.isfinite(rank_ic_mask):
                # Scale similarly to other uplift components: 10.0 * (mask - reference) / normalization
                rank_ic_uplift = 10.0 * rank_ic_mask
            else:
                rank_ic_uplift = np.nan

            # Composite mask uplift with 5 components now
            uplift_components = [
                tail_uplift,
                mean_75_uplift,
                sortino_weekly_uplift,
                sortino_monthly_uplift,
                rank_ic_uplift,
            ]
            valid_uplifts = [u for u in uplift_components if np.isfinite(u)]
            if len(valid_uplifts) == 5:
                overall_mask_uplift = sum(valid_uplifts) / 5.0
            else:
                overall_mask_uplift = np.nan

            assessment_results[-1].update({
                "tail_p5_ratio": tail_p5_ratio,
                "mean_p75_ratio": mean_p75_ratio,
                "monthly_sortino_ratio": monthly_sortino_ratio,
                "weekly_sortino_ratio": weekly_sortino_ratio,
                "tail_uplift": tail_uplift,
                "mean_75_uplift": mean_75_uplift,
                "sortino_monthly_uplift": sortino_monthly_uplift,
                "sortino_weekly_uplift": sortino_weekly_uplift,
                "overall_mask_uplift": overall_mask_uplift,
                # Store raw metrics for debugging/reporting
                "p5_mask": p5_mask,
                "p5_global": p5_global,
                "p75_mask": p75_mask,
                "p75_global": p75_global,
                "w_sort_mask": w_sort_mask,
                "w_sort_global": w_sort_global,
                "m_sort_mask": m_sort_mask,
                "m_sort_global": m_sort_global,
            })

            if "train_target_nan_reasons" in ridge_details and ridge_details["train_target_nan_reasons"] is not None:
                for k, v in ridge_details["train_target_nan_reasons"].items():
                    assessment_results[-1][f"train_target_nan_{k}"] = v
            if "val_target_nan_reasons" in ridge_details and ridge_details["val_target_nan_reasons"] is not None:
                for k, v in ridge_details["val_target_nan_reasons"].items():
                    assessment_results[-1][f"val_target_nan_{k}"] = v

            # Print the per-candidate target-drop summary message
            if "train_target_nan_reasons" in ridge_details:
                train_reasons = ridge_details.get("train_target_nan_reasons", {})
                val_reasons = ridge_details.get("val_target_nan_reasons", {})

                # Helper to format the dict
                def _format_reasons(r_dict):
                    return (
                        f"horizon_exceeded={r_dict.get('horizon_exceeded', 0)}, "
                        f"barrier_unresolved={r_dict.get('barrier_unresolved', 0)}, "
                        f"ambiguous_bar={r_dict.get('ambiguous_bar', 0)}, "
                        f"outside_support_mask={r_dict.get('outside_support_mask', 0)}, "
                        f"neutral_filtered={r_dict.get('neutral_filtered', 0)}, "
                        f"current_close_missing={r_dict.get('current_close_missing', 0)}, "
                        f"atr_missing={r_dict.get('atr_missing', 0)}, "
                        f"transformed_target_nonfinite={r_dict.get('transformed_target_nonfinite', 0)}, "
                        f"symbol_alignment_missing={r_dict.get('symbol_alignment_missing', 0)}, "
                        f"future_close_missing={r_dict.get('future_close_missing', 0)}, "
                        f"other_target_nan={r_dict.get('other_target_nan', 0)}"
                    )

                # Merged reasons for print log
                merged_reasons = {
                    k: train_reasons.get(k, 0) + val_reasons.get(k, 0)
                    for k in set(train_reasons) | set(val_reasons)
                }

                tprint(
                    f"Stage A: Ridge target-drop summary key={canonical_key} "
                    f"train_target_nan={ridge_details.get('target_nan_total_train', 0)} "
                    f"val_target_nan={ridge_details.get('target_nan_total_val', 0)} "
                    f"reasons[{_format_reasons(merged_reasons)}]"
                )
        assessment_df = pd.DataFrame(assessment_results)
        if assessment_df.empty:
            tprint("Stage A: No candidates to assess. Yielding empty dataframe.")
            return assessment_df

        def _compute_trade_weekly_volatility(
            realized_trades: Any,
            net_returns: Any,
        ) -> float:
            if (
                not isinstance(realized_trades, list)
                or not isinstance(net_returns, list)
                or len(realized_trades) == 0
                or len(realized_trades) != len(net_returns)
            ):
                return np.nan
            entry_times = pd.Series([t.entry_time for t in realized_trades])
            if entry_times.empty:
                return np.nan
            if entry_times.dt.tz is not None:
                entry_times = entry_times.dt.tz_convert("UTC").dt.tz_localize(None)
            monday_floors = entry_times.dt.floor("D") - pd.to_timedelta(
                entry_times.dt.dayofweek, unit="D"
            )
            week_pnl = pd.DataFrame(
                {"week": monday_floors, "pnl": net_returns}
            ).groupby("week")["pnl"].sum()
            if len(week_pnl) < 2:
                return np.nan
            weekly_vol = week_pnl.std(ddof=0)
            return float(weekly_vol) if not pd.isna(weekly_vol) else np.nan

        assessment_df["stage1_ridge_pnl_raw"] = pd.to_numeric(
            assessment_df.get("ridge_pnl_raw", 0.0), errors="coerce"
        ).fillna(0.0)
        assessment_df["stage1_weak_sortino"] = pd.to_numeric(
            assessment_df.get("ridge_trade_sortino_composite", 0.0), errors="coerce"
        ).fillna(0.0)
        assessment_df["stage1_weak_weekly_std"] = assessment_df.apply(
            lambda row: _compute_trade_weekly_volatility(
                row.get("realized_trades", []),
                row.get("net_weighted_returns", []),
            ),
            axis=1,
        )
        weak_pnl_rank = self._pct_rank(assessment_df["stage1_ridge_pnl_raw"])
        weak_sortino_rank = self._pct_rank(assessment_df["stage1_weak_sortino"])
        weak_weekly_std_rank = self._pct_rank(
            assessment_df["stage1_weak_weekly_std"].fillna(
                assessment_df["stage1_weak_weekly_std"].max()
                if assessment_df["stage1_weak_weekly_std"].notna().any()
                else 0.0
            )
        )
        assessment_df["weak_filter_score"] = (
            0.70 * weak_pnl_rank
            + 0.20 * weak_sortino_rank
            + 0.10 * (1.0 - weak_weekly_std_rank)
        )
        assessment_df = assessment_df.sort_values(
            ["weak_filter_score", "stage1_ridge_pnl_raw", "canonical_key"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        assessment_df["stage1_net_pnl_rank"] = (
            assessment_df["stage1_ridge_pnl_raw"].rank(method="first", ascending=False).astype(np.int32)
        )
        assessment_df["stage1_weak_filter_rank"] = np.arange(1, len(assessment_df) + 1, dtype=np.int32)
        survivor_frac = float(self.cfg.get("stage1_top_survivor_fraction", 0.70))
        survivor_count = max(1, int(np.ceil(len(assessment_df) * survivor_frac)))
        survivor_keys = set(
            assessment_df.head(survivor_count)["canonical_key"].astype(str).tolist()
        )
        assessment_df["stage1_top70_survivor"] = assessment_df["canonical_key"].astype(str).isin(survivor_keys)
        assessment_df["stage1_top60_weak_filter_survivor"] = assessment_df["stage1_top70_survivor"]

        selected_records_by_key = {
            str(record.get("canonical_key")): record for record in selected_records
        }
        learnability_reasons = {
            "not_in_top_ridge_candidates",
            "insufficient_subset_oof_coverage",
            "missing_learnability",
            "missing_top_quartile_precision",
            "top_quartile_precision_not_positive",
            "ev_per_event_less_than_or_equal_to_zero",
            "no positive post-fee profit threshold",
            "threshold_star_search_failed",
            "insufficient trades per symbol day",
        }
        strong_rescore_count = 0
        adaptive_enabled_global = bool(self.cfg.get("adaptive_tp_sl_enabled", False))
        for df_idx in assessment_df.index[assessment_df["stage1_top70_survivor"]]:
            canonical_key = str(assessment_df.at[df_idx, "canonical_key"])
            row_dict = selected_records_by_key.get(canonical_key)
            if row_dict is None:
                continue
            current_reason = str(assessment_df.at[df_idx, "rejection_reason"] or "")
            if current_reason == "support_out_of_range":
                continue

            if canonical_key in mask_cache:
                mask = mask_cache[canonical_key]
            elif self.mask_resolver:
                mask = self.mask_resolver.get_mask(canonical_key)
                mask_cache[canonical_key] = mask
            else:
                mask = self._get_mask_for_rule(canonical_key, X)
                mask_cache[canonical_key] = mask

            if int(np.sum(mask)) < 20:
                continue

            side = str(row_dict.get("side", "long"))
            horizon_raw = row_dict.get("source_horizon", -1)
            try:
                horizon_key = int(horizon_raw) if pd.notna(horizon_raw) else -1
            except (TypeError, ValueError):
                horizon_key = -1
            target_name = str(row_dict.get("source_target", "unknown"))
            tbm_horizon = max(horizon_key + 2, 1)

            current_target_ret = None
            if triad_targets_map is not None:
                current_target_ret = triad_targets_map.get((target_name, horizon_key))
            if current_target_ret is None:
                current_target_ret = bounded_target if bounded_target is not None else fwd_ret
            current_target_ret = self._transform_side_target(
                target_name=target_name,
                target_values=current_target_ret,
                side=side,
                atr_frac=atr_frac_for_targets,
            )

            ctx_key = (target_name, horizon_key, side)
            baseline_data = baseline_cache.get(ctx_key, {})
            global_auc = float(baseline_data.get("global_auc", 0.5))
            global_top_quartile_precision = float(baseline_data.get("global_top_quartile_precision", 0.5))

            path_excursions = path_excursion_cache.get((horizon_key, side), {})
            strong_details = self._compute_subset_ridge_details(
                X,
                current_target_ret,
                mask,
                folds,
                tp_f=None,
                target_nan_reasons=current_target_nan_reasons,
                path_mfe=path_excursions.get("mfe"),
                path_mae=path_excursions.get("mae"),
                model_profile="strong",
                side=side,
            )

            ridge_round_fee = float(self.cfg.get("ridge_cost_pct", 0.003))
            if adaptive_enabled_global:
                adaptive_tp_atr, adaptive_sl_atr = self._compute_adaptive_tp_sl(
                    mask=mask,
                    fwd_ret=current_gross_return_ret,
                    atr=data["atr"].to_numpy(),
                    oof_preds=np.asarray(strong_details["oof_preds"], dtype=np.float32),
                    close=data["close"].to_numpy(),
                    horizon=tbm_horizon,
                    side=side,
                )
                tbm_outcomes = compute_tbm_outcomes_per_symbol(
                    data=data,
                    horizon=tbm_horizon,
                    tp_atr=adaptive_tp_atr,
                    sl_atr=adaptive_sl_atr,
                    side=side,
                )
                tp_payoff = float(adaptive_tp_atr)
                sl_payoff = float(adaptive_sl_atr)
            else:
                tbm_outcomes = tbm_outcome_cache.get((tbm_horizon, side))
                if tbm_outcomes is None:
                    tbm_outcomes = compute_tbm_outcomes_per_symbol(
                        data=data,
                        horizon=tbm_horizon,
                        tp_atr=float(self.cfg.get("tp_atr", 2.0)),
                        sl_atr=float(self.cfg.get("sl_atr", 2.0)),
                        side=side,
                    )
                    tbm_outcome_cache[(tbm_horizon, side)] = tbm_outcomes
                tp_payoff = float(self.cfg.get("tp_atr", 2.0))
                sl_payoff = float(self.cfg.get("sl_atr", 2.0))

            rule_tp_f, rule_sl_f, rule_to_f = tbm_outcomes
            strong_tbm_metrics = self._compute_tbm_metrics(
                mask, rule_tp_f, rule_sl_f, rule_to_f, current_gross_return_ret
            )
            strong_top_q = self._compute_top_quartile_precision(
                oof_preds=np.asarray(strong_details["oof_preds"], dtype=np.float32),
                y=np.asarray(current_gross_return_ret, dtype=np.float32),
                mask=mask,
                tp_f=rule_tp_f,
                fwd_ret_threshold=float(
                    self.cfg.get("ridge_cost_pct", 0.003)
                ),
                top_pct=0.75,
                min_samples=20,
            )
            strong_trade_metrics = self._compute_ranked_ridge_trade_metrics(
                data=data,
                directional_returns=current_gross_return_ret,
                mask=mask,
                folds=folds,
                horizon=horizon_key,
                oof_preds=np.asarray(strong_details["oof_preds"], dtype=np.float32),
                round_fee=ridge_round_fee,
            )

            strong_auc = float(strong_details.get("subset_auc", np.nan))
            strong_auc_lift = (
                strong_auc - global_auc
                if np.isfinite(strong_auc) and np.isfinite(global_auc)
                else np.nan
            )
            strong_top_q_lift = (
                float(strong_top_q) - global_top_quartile_precision
                if np.isfinite(strong_top_q) and np.isfinite(global_top_quartile_precision)
                else np.nan
            )
            timeout_payoff = float(np.nanmean(current_gross_return_ret[mask]))
            strong_ev_per_event = (
                strong_tbm_metrics["tp_rate"] * tp_payoff
                - strong_tbm_metrics["sl_rate"] * sl_payoff
                + strong_tbm_metrics["timeout_rate"] * timeout_payoff
            )

            preserve_rejection = current_reason not in ("", *learnability_reasons)
            strong_rejected = bool(preserve_rejection)
            strong_rejection_reason = current_reason if preserve_rejection else ""
            if not strong_rejected and float(strong_details.get("coverage", 0.0)) < min_oof_coverage:
                strong_rejected = True
                strong_rejection_reason = "insufficient_subset_oof_coverage"
            elif not strong_rejected and strong_ev_per_event <= 0.0:
                strong_rejected = True
                strong_rejection_reason = "ev_per_event_less_than_or_equal_to_zero"
            elif not strong_rejected and bool(strong_trade_metrics.get("rejected", False)):
                strong_rejected = True
                strong_rejection_reason = str(
                    strong_trade_metrics.get("reject_reason", {}).get("reason", "threshold_star_search_failed")
                )

            assessment_df.at[df_idx, "is_structurally_sound"] = not strong_rejected
            assessment_df.at[df_idx, "rejection_reason"] = strong_rejection_reason
            assessment_df.at[df_idx, "stage2_model_profile"] = "strong"
            assessment_df.at[df_idx, "stage2_rescored"] = True
            assessment_df.at[df_idx, "auc_lift"] = strong_auc_lift
            assessment_df.at[df_idx, "top_quartile_precision"] = float(strong_top_q) if np.isfinite(strong_top_q) else np.nan
            assessment_df.at[df_idx, "top_quartile_precision_lift"] = strong_top_q_lift
            assessment_df.at[df_idx, "subset_oof_coverage"] = float(strong_details.get("coverage", np.nan))
            assessment_df.at[df_idx, "mask_oof_corr"] = strong_details.get("mask_oof_corr", np.nan)
            assessment_df.at[df_idx, "mask_oof_r2"] = strong_details.get("mask_oof_r2", np.nan)
            assessment_df.at[df_idx, "mask_roc_auc"] = strong_details.get("subset_roc_auc", np.nan)
            assessment_df.at[df_idx, "mask_pr_auc"] = strong_details.get("subset_pr_auc", np.nan)
            assessment_df.at[df_idx, "fold_sign_consistency"] = strong_details.get("fold_sign_consistency", np.nan)
            assessment_df.at[df_idx, "positive_fold_fraction"] = strong_details.get("positive_fold_fraction", np.nan)
            assessment_df.at[df_idx, "negative_fold_fraction"] = strong_details.get("negative_fold_fraction", np.nan)
            assessment_df.at[df_idx, "fold_pnl_std"] = strong_details.get("fold_pnl_std", np.nan)
            assessment_df.at[df_idx, "decile_monotonic_spearman"] = strong_details.get("decile_monotonic_spearman", np.nan)
            assessment_df.at[df_idx, "top_decile_mean_target"] = strong_details.get("top_decile_mean_target", np.nan)
            assessment_df.at[df_idx, "bottom_decile_mean_target"] = strong_details.get("bottom_decile_mean_target", np.nan)
            assessment_df.at[df_idx, "decile_spread_mean"] = strong_details.get("decile_spread_mean", np.nan)
            assessment_df.at[df_idx, "target_nan_total_train"] = strong_details.get("target_nan_total_train", 0)
            assessment_df.at[df_idx, "target_nan_total_val"] = strong_details.get("target_nan_total_val", 0)
            assessment_df.at[df_idx, "tp_rate"] = strong_tbm_metrics.get("tp_rate", np.nan)
            assessment_df.at[df_idx, "sl_rate"] = strong_tbm_metrics.get("sl_rate", np.nan)
            assessment_df.at[df_idx, "timeout_rate"] = strong_tbm_metrics.get("timeout_rate", np.nan)
            assessment_df.at[df_idx, "ev_per_trade"] = strong_tbm_metrics.get("ev_per_trade", np.nan)
            assessment_df.at[df_idx, "ev_per_event"] = strong_ev_per_event
            assessment_df.at[df_idx, "win_rate_conditional"] = strong_tbm_metrics.get("win_rate_conditional", np.nan)
            assessment_df.at[df_idx, "win_rate_unconditional"] = strong_tbm_metrics.get("win_rate_unconditional", np.nan)
            for metric_col in [
                "threshold_star",
                "threshold_star_lowest_positive",
                "threshold_star_optimal_pnl",
                "threshold_star_best_pnl_threshold",
                "ridge_pnl_gross_raw",
                "ridge_pnl_gross_raw_at_optimal_threshold",
                "ridge_pnl_raw",
                "ridge_pnl_raw_at_optimal_threshold",
                "avg_trades_per_day",
                "avg_pnl_per_day",
                "avg_pnl_per_active_symbol_day",
                "ridge_trade_sortino_7d",
                "ridge_trade_sortino_30d",
                "ridge_trade_sortino_90d",
                "ridge_trade_sortino_composite",
                "trades_per_symbol_day_above_threshold_star",
                "valid_symbol_days_observed",
                "total_trades",
                "threshold_search_mode",
                "threshold_selection_policy",
                "n_quantiles_evaluated",
                "n_thresholds_evaluated",
                "n_unique_thresholds_evaluated",
                "score_min",
                "score_max",
                "score_std",
                "n_unique_scores",
                "realized_trades",
                "gross_weighted_returns",
                "net_weighted_returns",
                "weighted_fee_returns",
                "avg_fee_per_trade",
                "avg_gross_move_per_trade",
                "avg_position_weight",
            ]:
                assessment_df.at[df_idx, metric_col] = strong_trade_metrics.get(metric_col, assessment_df.at[df_idx, metric_col])
            strong_rescore_count += 1

        if strong_rescore_count > 0:
            tprint(
                f"Stage A: Strong Step2 rescoring complete for {strong_rescore_count}/{survivor_count} stage1 survivors"
            )
            
        # 6. Final Ranking Normalization
        # We apply MinMax scaling to each final-ranking term across the entire candidate cohort.

        def _normalize_column(col_name: str, new_col_name: str):
            if col_name not in assessment_df.columns:
                assessment_df[new_col_name] = 0.0
                return

            valid_vals = assessment_df[col_name].replace([np.inf, -np.inf], np.nan).dropna()
            if valid_vals.empty:
                assessment_df[new_col_name] = 0.0
                return

            c_min = float(valid_vals.min())
            c_max = float(valid_vals.max())
            span = max(c_max - c_min, 1e-9)

            # Ensure stable fallback when max == min
            if span <= 1e-9:
                if c_max > 0:
                    assessment_df[new_col_name] = 1.0
                else:
                    assessment_df[new_col_name] = 0.0
                return

            assessment_df[new_col_name] = np.clip(
                (assessment_df[col_name] - c_min) / span, 0.0, 1.0
            ).fillna(0.0)

        # Base terms for final ranking
        # Heuristic default for intraday crypto. Raw ridge-pnl ~ 3% is meaningful.
        pnl_ref = 0.03
        assessment_df["ridge_pnl_norm"] = 1.0 - np.exp(
            -np.maximum(assessment_df["ridge_pnl_raw"], 0.0) / pnl_ref
        )

        # Normalize final-ranking inputs jointly across the assessed candidate table before scoring.
        _normalize_column("ridge_pnl_norm", "ridge_pnl_norm_norm")
        _normalize_column("ridge_trade_sortino_composite", "ridge_trade_sortino_composite_norm")
        _normalize_column("overall_mask_uplift", "overall_mask_uplift_norm")
        _normalize_column("ev_per_event", "ev_per_event_norm")

        # 10. worst_penalty (computed from the fully normalized inputs)
        def _compute_worst_penalty(row):
            # All components are in [0,1]
            worst_malus = min(
                row.get("ridge_pnl_norm_norm", 0.0),
                row.get("ridge_trade_sortino_composite_norm", 0.0),
                row.get("overall_mask_uplift_norm", 0.0),
                row.get("ev_per_event_norm", 0.0)
            )
            return 1.0 - worst_malus

        assessment_df["worst_penalty"] = assessment_df.apply(_compute_worst_penalty, axis=1)

        # 1. Final base score (V3 update: replace auc_lift_norm with overall_mask_uplift_norm)
        assessment_df["base_regime_score"] = (
            0.30 * assessment_df["ridge_pnl_norm_norm"]
            + 0.30 * assessment_df["ridge_trade_sortino_composite_norm"]
            + 0.30 * assessment_df["overall_mask_uplift_norm"]
            + 0.10 * assessment_df["ev_per_event_norm"]
        )

        # Helper to compute weekly metrics from realized trades
        #
        # ridge_trade_sortino_*: evaluates trade-level realized execution quality
        # weekly_sortino: evaluates weekly aggregation stability / smoothness quality
        def _compute_weekly_metrics(row):
            realized_trades = row.get("realized_trades", [])
            net_returns = row.get("net_weighted_returns", [])

            # Default empty metrics
            metrics = pd.Series({
                "weekly_volatility": np.nan,
                "weekly_sortino_raw": np.nan,
                "weekly_sortino": np.nan
            })

            if not isinstance(net_returns, list) or len(net_returns) == 0 or len(realized_trades) != len(net_returns):
                return metrics

            # Build a weekly PnL series from entry times floored to the week safely
            entry_times = pd.Series([t.entry_time for t in realized_trades])
            if entry_times.dt.tz is not None:
                entry_times = entry_times.dt.tz_convert("UTC").dt.tz_localize(None)

            # Robust timezone-safe weekly bucketing: floor to day, then subtract dayofweek to get Monday
            # entry_times is already tz-naive here
            monday_floors = entry_times.dt.floor("D") - pd.to_timedelta(entry_times.dt.dayofweek, unit="D")
            df_trades = pd.DataFrame({
                "week": monday_floors,
                "pnl": net_returns
            })

            week_pnl = df_trades.groupby("week")["pnl"].sum()

            if len(week_pnl) < 2:
                return metrics

            weekly_vol = week_pnl.std(ddof=0)
            metrics["weekly_volatility"] = float(weekly_vol) if not pd.isna(weekly_vol) else np.nan

            # Downside computation for Sortino
            mean_ret = float(week_pnl.mean())
            downside = np.minimum(week_pnl.to_numpy(dtype=np.float32), 0.0)
            downside_dev = float(np.sqrt(np.mean(downside**2)))

            if downside_dev > 1e-9:
                sortino_raw = mean_ret / downside_dev
                metrics["weekly_sortino_raw"] = float(sortino_raw)

                # Bounded [0, 1] Sortino via tanh scaling
                sortino_scale = 2.0
                metrics["weekly_sortino"] = float(np.tanh(max(sortino_raw, 0.0) / sortino_scale))

            return metrics

        # Compute the weekly metrics for audit/analysis
        weekly_metrics_df = assessment_df.apply(_compute_weekly_metrics, axis=1)
        for col in ["weekly_volatility", "weekly_sortino_raw", "weekly_sortino"]:
            assessment_df[col] = weekly_metrics_df[col]

        assessment_df["regime_score"] = assessment_df["base_regime_score"]
        assessment_df = self._apply_final_topk_selection(assessment_df, mask_cache)

        # Drop temporary lists from assessment df
        if "realized_trades" in assessment_df.columns:
            assessment_df = assessment_df.drop(columns=["realized_trades"])
        if "net_weighted_returns" in assessment_df.columns:
            assessment_df = assessment_df.drop(columns=["net_weighted_returns"])

        assessed_count = len(assessment_df)
        sound_count = assessment_df["is_structurally_sound"].sum()
        rejected_count = assessed_count - sound_count

        tprint(
            f"Stage A: Final assessment complete - {assessed_count} assessed | {sound_count} structurally sound | {rejected_count} rejected"
        )
        tprint(
            "Stage A: Final assessment phase end - "
            f"assessed={assessed_count} accepted={sound_count} rejected={rejected_count} "
            f"elapsed={time.perf_counter() - final_assessment_start_ts:.2f}s"
        )

        rejection_counts = assessment_df[~assessment_df["is_structurally_sound"]][
            "rejection_reason"
        ].value_counts()
        if not rejection_counts.empty:
            tprint("Top Assessor Rejection Reasons:")
            for reason, count in rejection_counts.items():
                tprint(f"  - {reason}: {count}")

        # Save rejection summary as attribute
        self.rejection_summary = rejection_counts.to_dict()

        top_sound = (
            assessment_df[assessment_df["is_structurally_sound"]]
            .sort_values("regime_score", ascending=False)
            .head(5)
        )
        if not top_sound.empty:
            tprint("Top 5 Structurally Sound Rules by Regime Score:")
            for row in top_sound[["canonical_key", "regime_score"]].to_dict("records"):
                regime_score = row.get("regime_score", np.nan)
                if isinstance(regime_score, (int, float, np.floating)) and np.isfinite(
                    regime_score
                ):
                    tprint(
                        f"  - {row['canonical_key'][:120]} regime_score={float(regime_score):.6f}"
                    )
                else:
                    tprint(
                        f"  - {row.get('canonical_key', '<unknown>')}: {regime_score}"
                    )

        final_selected = assessment_df[
            assessment_df["selected_for_final_registry"].fillna(False)
        ].sort_values("final_selection_order", ascending=True)
        if not final_selected.empty:
            tprint("Final diversified top-k rules:")
            for row in final_selected[
                [
                    "final_selection_order",
                    "canonical_key",
                    "ridge_pnl_raw",
                    "ridge_trade_sortino_composite",
                    "final_candidate_rank_score",
                ]
            ].to_dict("records"):
                tprint(
                    "  - "
                    f"order={int(row['final_selection_order'])} "
                    f"key={str(row['canonical_key'])[:120]} "
                    f"net_pnl={float(row['ridge_pnl_raw']):.6f} "
                    f"sortino={float(row['ridge_trade_sortino_composite']):.6f} "
                    f"score={float(row['final_candidate_rank_score']):.6f}"
                )

        return assessment_df

    def _simulate_tp_sl_ev_with_path(
        self,
        fwd_ret: np.ndarray,
        atr_frac: np.ndarray,
        tp_atr: float,
        sl_atr: float,
        horizon: int,
        side: str,
    ) -> float:
        """
        Simulate expected value given TP/SL parameters using actual path data.

        Uses final return as approximation when path data is not available.
        """
        # Normalize returns by ATR
        ret_atr = fwd_ret / (atr_frac + 1e-12)

        # For long: positive return is profit, negative is loss
        # For short: negative return is profit, positive is loss
        if side == "long":
            # TP hit if ret_atr >= tp_atr
            # SL hit if ret_atr <= -sl_atr
            tp_hit = ret_atr >= tp_atr
            sl_hit = ret_atr <= -sl_atr
            # Use final return as approximation of path
            profit = np.where(tp_hit, tp_atr, np.where(sl_hit, -sl_atr, ret_atr))
        else:
            # Short: TP hit if ret_atr <= -tp_atr
            # SL hit if ret_atr >= sl_atr
            tp_hit = ret_atr <= -tp_atr
            sl_hit = ret_atr >= sl_atr
            # Use final return as approximation of path
            profit = np.where(tp_hit, tp_atr, np.where(sl_hit, -sl_atr, -ret_atr))

        return float(np.mean(profit))

    def _compute_adaptive_tp_sl(
        self,
        mask: np.ndarray,
        fwd_ret: np.ndarray,
        atr: np.ndarray,
        oof_preds: np.ndarray,
        close: np.ndarray,
        horizon: int,
        side: str,
        confidence_levels: list = None,
        tp_grid_atr: list = None,
        sl_ratio_grid: list = None,
        raw_tp_min: float = 0.01,  # 1%
        raw_tp_max: float = 0.03,  # 3%
        min_valid: int = 100,
        min_conf_samples: int = 20,
    ) -> tuple:
        """
        Compute adaptive TP/SL for a rule based on historical performance.

        TP is ATR-normalized but constrained to ~1-3% raw returns.
        Uses fractional ATR (atr/price) for proper dimensional consistency.
        """
        if confidence_levels is None:
            confidence_levels = self.cfg.get("adaptive_tp_sl_conf_levels", [0.8])
        if tp_grid_atr is None:
            tp_grid_atr = self.cfg.get("adaptive_tp_sl_grid", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        if sl_ratio_grid is None:
            sl_ratio_grid = self.cfg.get("adaptive_tp_sl_sl_ratio_grid", [0.3, 0.5, 0.7, 0.9])

        if not np.any(mask):
            return 1.25, 0.50  # Fallback to defaults

        # Use fractional ATR (atr/price) for proper dimensional consistency
        atr_frac = atr / (np.abs(close) + 1e-12)

        valid_mask = mask & np.isfinite(fwd_ret) & np.isfinite(atr_frac) & np.isfinite(oof_preds) & np.isfinite(close)
        if np.sum(valid_mask) < min_valid:
            return 1.25, 0.50  # Insufficient data

        optimal_tps = []
        optimal_sls = []

        for conf_level in confidence_levels:
            # Side-specific confidence threshold
            scores = oof_preds[valid_mask]
            if side == "long":
                # For long: top predictions (highest scores)
                threshold = np.percentile(scores, conf_level * 100)
                conf_mask = valid_mask & (oof_preds >= threshold)
            else:
                # For short: bottom predictions (lowest scores, most negative)
                threshold = np.percentile(scores, (1.0 - conf_level) * 100)
                conf_mask = valid_mask & (oof_preds <= threshold)

            if np.sum(conf_mask) < min_conf_samples:
                continue  # Skip if insufficient samples at this confidence level

            # Compute median fractional ATR for this confidence subset
            median_atr_frac = np.median(atr_frac[conf_mask])

            # Filter TP candidates to ensure raw returns are in [1%, 3%]
            tp_candidates = []
            for tp_atr in tp_grid_atr:
                tp_raw = tp_atr * median_atr_frac
                if raw_tp_min <= tp_raw <= raw_tp_max:
                    tp_candidates.append(tp_atr)

            # If no candidates satisfy raw return constraint, find nearest valid values
            if not tp_candidates:
                def dist_to_band(x, lo, hi):
                    if x < lo:
                        return lo - x
                    if x > hi:
                        return x - hi
                    return 0.0

                # Compute distance for each candidate
                candidates_with_dist = []
                for tp_atr in tp_grid_atr:
                    tp_raw = tp_atr * median_atr_frac
                    dist = dist_to_band(tp_raw, raw_tp_min, raw_tp_max)
                    candidates_with_dist.append((dist, tp_atr))

                # Sort by distance and take closest
                candidates_with_dist.sort(key=lambda x: x[0])
                # Take all candidates with same distance as the closest
                min_dist = candidates_with_dist[0][0]
                tp_candidates = [tp for dist, tp in candidates_with_dist if dist == min_dist]

            best_ev = -np.inf
            best_tp = 1.25
            best_sl = 0.50

            # Grid search over valid TP candidates and SL ratios
            for tp_atr in tp_candidates:
                for sl_ratio in sl_ratio_grid:
                    sl_atr = tp_atr * sl_ratio

                    # Simulate trades with these TP/SL using actual path data
                    ev = self._simulate_tp_sl_ev_with_path(
                        fwd_ret[conf_mask],
                        atr_frac[conf_mask],
                        tp_atr,
                        sl_atr,
                        horizon,
                        side,
                    )

                    if ev > best_ev:
                        best_ev = ev
                        best_tp = tp_atr
                        best_sl = sl_atr

            optimal_tps.append(best_tp)
            optimal_sls.append(best_sl)

        # Average optimal TP/SL across confidence levels
        if optimal_tps:
            tp_avg = np.mean(optimal_tps)
            sl_avg = np.mean(optimal_sls)
        else:
            tp_avg, sl_avg = 1.25, 0.50

        return tp_avg, sl_avg

    def _compute_tbm_metrics(self, mask, tp_f, sl_f, to_f, fwd_ret) -> Dict[str, float]:
        """Compute triple barrier metrics."""
        m = mask.astype(bool)
        if not np.any(m):
            return {
                "tp_rate": 0.0,
                "sl_rate": 0.0,
                "timeout_rate": 0.0,
                "ev_per_trade": 0.0,
                "win_rate_conditional": 0.0,
                "win_rate_unconditional": 0.0,
            }

        tp = np.sum(tp_f[m])
        sl = np.sum(sl_f[m])
        to = np.sum(to_f[m])
        total = np.sum(m)

        ev = np.nanmean(fwd_ret[m])

        # Conditional on a barrier hit
        win_rate_conditional = tp / (tp + sl + 1e-9)

        # Unconditional share of selected events
        win_rate_unconditional = tp / (total + 1e-9)

        return {
            "tp_rate": float(tp / total),
            "sl_rate": float(sl / total),
            "timeout_rate": float(to / total),
            "ev_per_trade": float(ev),
            "win_rate_conditional": float(win_rate_conditional),
            "win_rate_unconditional": float(win_rate_unconditional),
        }

    def _compute_cvar(self, returns, alpha=0.05) -> float:
        """Compute Conditional Value at Risk."""
        if len(returns) == 0:
            return 0.0
        n = len(returns)
        cutoff_idx = max(int(n * alpha), 1)
        sorted_rets = np.sort(returns)
        return float(np.mean(sorted_rets[:cutoff_idx]))

    def _compute_tail_ratio(self, returns) -> float:
        """Compute tail ratio (95th percentile / 5th percentile)."""
        if len(returns) < 20:
            # Root cause: insufficient samples for reliable tail ratio
            return 1.0
        p95 = abs(np.percentile(returns, 95))
        p5 = abs(np.percentile(returns, 5))
        return float(p95 / (p5 + 1e-9))

    def _compute_mae_mfe(self, returns) -> Tuple[float, float]:
        """
        Compute Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE).

        MAE: Average of negative returns (average loss)
        MFE: Average of positive returns (average gain)
        """
        if len(returns) == 0:
            return 0.0, 0.0

        returns = returns[np.isfinite(returns)]
        if len(returns) == 0:
            return 0.0, 0.0

        negative_returns = returns[returns < 0]
        positive_returns = returns[returns > 0]

        mae = float(np.mean(negative_returns)) if len(negative_returns) > 0 else 0.0
        mfe = float(np.mean(positive_returns)) if len(positive_returns) > 0 else 0.0

        return mae, mfe

    def _compute_subset_auc(self, X, fwd_ret, mask, folds) -> Tuple[float, float]:
        """Compute AUC for a subset of data defined by mask."""
        details = self._compute_subset_ridge_details(X, fwd_ret, mask, folds)
        return float(details["subset_auc"]), float(details["coverage"])

    def _transform_side_target(
        self,
        target_name: str,
        target_values: Optional[np.ndarray],
        side: str,
        atr_frac: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if target_values is None:
            return None
        arr = np.asarray(target_values, dtype=np.float32).copy()
        target_name = str(target_name)
        side = str(side).lower()

        if target_name not in {"returns_target", "atr_norm_returns_target"}:
            return arr

        signed = arr if side != "short" else -arr

        fee_pct = float(self.cfg.get("training_label_round_trip_fee_pct", 0.002))
        if target_name == "returns_target":
            transformed = signed - fee_pct
        else:
            if atr_frac is None:
                transformed = signed - fee_pct
            else:
                atr_frac_arr = np.asarray(atr_frac, dtype=np.float32)
                fee_in_target_units = fee_pct / np.maximum(atr_frac_arr, 1e-3)
                transformed = signed - fee_in_target_units

        low_q = float(self.cfg.get("training_label_winsor_low_q", 0.01))
        high_q = float(self.cfg.get("training_label_winsor_high_q", 0.99))
        finite = np.isfinite(transformed)
        if finite.sum() >= 20 and 0.0 <= low_q < high_q <= 1.0:
            lo, hi = np.nanquantile(transformed[finite], [low_q, high_q])
            transformed = np.clip(transformed, lo, hi)

        return np.asarray(transformed, dtype=np.float32)

    def _resolve_side_gross_returns(
        self,
        *,
        side: str,
        source_horizon: int,
        triad_targets_map: Optional[Dict[Tuple[str, int], np.ndarray]],
        fallback_fwd_ret: np.ndarray,
    ) -> np.ndarray:
        gross_target = None
        if triad_targets_map is not None:
            gross_target = triad_targets_map.get(("returns_target", int(source_horizon)))
        if gross_target is None:
            gross_target = fallback_fwd_ret
        gross_arr = np.asarray(gross_target, dtype=np.float32).copy()
        if str(side).lower() == "short":
            gross_arr = -gross_arr
        return gross_arr

    def _compute_subset_ridge_details(
        self,
        X,
        fwd_ret,
        mask,
        folds,
        tp_f: np.ndarray = None,
        target_nan_reasons: Optional[np.ndarray] = None,
        path_mfe: Optional[np.ndarray] = None,
        path_mae: Optional[np.ndarray] = None,
        model_profile: str = "weak",
        side: str = "long",
    ) -> Dict[str, Any]:
        """Compute Ridge OOF details for a subset of data defined by mask."""
        if not np.any(mask):
            return {
                "subset_auc": np.nan,
                "subset_roc_auc": np.nan,
                "subset_pr_auc": np.nan,
                "top_quartile_precision": np.nan,
                "coverage": 0.0,
                "oof_preds": np.full(len(X), np.nan, dtype=np.float32),
                "folds_used": 0,
                "folds_skipped": 0,
                "mask_oof_corr": np.nan,
                "mask_oof_r2": np.nan,
                "fold_sign_consistency": np.nan,
                "positive_fold_fraction": np.nan,
                "negative_fold_fraction": np.nan,
                "decile_monotonic_spearman": np.nan,
                "top_decile_mean_target": np.nan,
                "bottom_decile_mean_target": np.nan,
                "decile_spread_mean": np.nan,
            }

        ridge_feats = self._get_ridge_feature_indices()
        if ridge_feats.size == 0:
            return {
                "subset_auc": np.nan,
                "subset_roc_auc": np.nan,
                "subset_pr_auc": np.nan,
                "top_quartile_precision": np.nan,
                "coverage": 0.0,
                "oof_preds": np.full(len(X), np.nan, dtype=np.float32),
                "folds_used": 0,
                "folds_skipped": 0,
                "mask_oof_corr": np.nan,
                "mask_oof_r2": np.nan,
                "fold_sign_consistency": np.nan,
                "positive_fold_fraction": np.nan,
                "negative_fold_fraction": np.nan,
                "decile_monotonic_spearman": np.nan,
                "top_decile_mean_target": np.nan,
                "bottom_decile_mean_target": np.nan,
                "decile_spread_mean": np.nan,
            }
        subset_auc_start = time.perf_counter()
        X_ridge = self._build_ridge_design_matrix(X)

        if X_ridge.shape[1] >= 2:
            col0_data = X_ridge[:, 0]
            all_same = all(
                np.allclose(X_ridge[:, i], col0_data, equal_nan=True)
                for i in range(X_ridge.shape[1])
            )
            if all_same:
                tprint(
                    "WARNING: Assessor design matrix columns are identical after "
                    f"construction (shape={X_ridge.shape})."
                )
        
        y = np.asarray(fwd_ret, dtype=np.float32)
        y[~np.isfinite(fwd_ret)] = np.nan

        # --- Defensive NaN Protection: Discard sparse features to save rows ---
        final_ridge_feats_subset = np.arange(X_ridge.shape[1])
        if np.any(mask):
            X_mask = X_ridge[mask]
            nan_rates = np.isnan(X_mask).mean(axis=0)
            offenders = np.where(nan_rates > 0.10)[0]
            if 0 < len(offenders) < (0.10 * X_ridge.shape[1]):
                ridge_meta = [self.metadata[i] for i in ridge_feats] if hasattr(self, "metadata") else []
                discard_names = []
                for fi in offenders:
                    name = ridge_meta[fi].source_name if fi < len(ridge_meta) else f"feat_{fi}"
                    discard_names.append(name)
                
                tprint(f"  DEFENSIVE: Discarding {len(offenders)} Ridge features with >10% NaNs to save rows: {', '.join(discard_names)}")
                final_ridge_feats_subset = np.delete(final_ridge_feats_subset, offenders)
                X_ridge = X_ridge[:, final_ridge_feats_subset]

        (
            is_binary_target,
            min_train_req,
            min_val_req,
            min_pred_points,
        ) = self._ridge_learnability_thresholds(y)
        model_profile = str(model_profile).lower()
        model_params = self._get_step2_model_profile_params(model_profile)

        # Compute OOF predictions using Ridge
        from sklearn.linear_model import Ridge

        oof_preds = np.full(len(X), np.nan, dtype=np.float32)
        rng = np.random.RandomState(42)

        fold_filter_time = 0.0
        fit_predict_time = 0.0
        folds_used = 0
        folds_skipped = 0
        for fold_id, (tr_idx, va_idx) in enumerate(folds):
            fold_stage_start = time.perf_counter()
            # Apply mask to fold indices
            tr_masked = tr_idx[mask[tr_idx]]
            va_masked = va_idx[mask[va_idx]]

            X_tr, X_va = X_ridge[tr_masked], X_ridge[va_masked]
            y_tr, y_va = y[tr_masked], y[va_masked]
            mfe_tr = None
            mae_tr = None
            if path_mfe is not None and path_mae is not None:
                mfe_tr = np.asarray(path_mfe[tr_masked], dtype=np.float32)
                mae_tr = np.asarray(path_mae[tr_masked], dtype=np.float32)

            # Filter valid samples (y must be finite, and ALL ridge features must be finite)
            valid_tr = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)
            valid_va = np.isfinite(y_va) & np.all(np.isfinite(X_va), axis=1)

            # Log if many samples are dropped due to target or feature finiteness
            n_tr_before = len(y_tr)
            n_tr_after = np.sum(valid_tr)
            if n_tr_before > 0 and n_tr_after < n_tr_before:
                target_nan_mask = ~np.isfinite(y_tr)
                n_tr_target_nan = np.sum(target_nan_mask)
                n_tr_feat_nan = np.sum((~target_nan_mask) & (~np.all(np.isfinite(X_tr), axis=1)))

                target_reasons = {
                    "horizon_exceeded": 0,
                    "barrier_unresolved": 0,
                    "ambiguous_bar": 0,
                    "outside_support_mask": 0,
                    "neutral_filtered": 0,
                    "current_close_missing": 0,
                    "atr_missing": 0,
                    "transformed_target_nonfinite": 0,
                    "symbol_alignment_missing": 0,
                    "future_close_missing": 0,
                    "other_target_nan": 0,
                }
                if target_nan_reasons is not None and n_tr_target_nan > 0:
                    fold_reasons = target_nan_reasons[tr_masked][target_nan_mask]
                    unique_reasons, counts = np.unique(fold_reasons, return_counts=True)
                    for reason, count in zip(unique_reasons, counts):
                        if reason in target_reasons:
                            target_reasons[reason] += count
                        else:
                            target_reasons["other_target_nan"] += count
                else:
                    target_reasons["other_target_nan"] += n_tr_target_nan
                
                # Mask Enrichment Diagnostic
                global_nan_rate = np.isnan(y[tr_idx]).mean()
                mask_nan_rate = np.isnan(y_tr).mean()
                
                tprint(
                    f"WARNING: Fold {fold_id} Ridge training: Dropped {n_tr_before - n_tr_after}/{n_tr_before} "
                    f"({100*(1-n_tr_after/n_tr_before):.1f}%) samples. "
                    f"NaN Enrichment: global={global_nan_rate:.1%} -> mask={mask_nan_rate:.1%} "
                    f"[Target NaN: {n_tr_target_nan}, Feature NaN: {n_tr_feat_nan}] "
                    f"TargetNaNReasons[horizon_exceeded={target_reasons['horizon_exceeded']}, barrier_unresolved={target_reasons['barrier_unresolved']}, ambiguous_bar={target_reasons['ambiguous_bar']}, outside_support_mask={target_reasons['outside_support_mask']}, neutral_filtered={target_reasons['neutral_filtered']}, current_close_missing={target_reasons['current_close_missing']}, atr_missing={target_reasons['atr_missing']}, transformed_target_nonfinite={target_reasons['transformed_target_nonfinite']}, symbol_alignment_missing={target_reasons['symbol_alignment_missing']}, future_close_missing={target_reasons['future_close_missing']}, other_target_nan={target_reasons['other_target_nan']}]"
                )
                
                if n_tr_feat_nan > 0:
                    nan_rates = np.isnan(X_tr).mean(axis=0)
                    bad_idx = np.where(nan_rates > 0.001)[0]
                    if len(bad_idx) > 0:
                        ridge_meta = [self.metadata[i] for i in ridge_feats] if hasattr(self, "metadata") else []
                        sorted_bad = sorted(bad_idx, key=lambda i: nan_rates[i], reverse=True)
                        for fi in sorted_bad[:5]:
                            fname = ridge_meta[fi].source_name if fi < len(ridge_meta) else f"feat_{fi}"
                            tprint(f"    - Offending Ridge Feature: {fname} ({nan_rates[fi]:.1%} NaN)")

            n_va_before = len(y_va)
            n_va_after = np.sum(valid_va)
            if n_va_before > 0 and n_va_after < n_va_before:
                target_nan_mask = ~np.isfinite(y_va)
                n_va_target_nan = np.sum(target_nan_mask)
                n_va_feat_nan = np.sum((~target_nan_mask) & (~np.all(np.isfinite(X_va), axis=1)))

                target_reasons = {
                    "horizon_exceeded": 0,
                    "barrier_unresolved": 0,
                    "ambiguous_bar": 0,
                    "outside_support_mask": 0,
                    "neutral_filtered": 0,
                    "current_close_missing": 0,
                    "atr_missing": 0,
                    "transformed_target_nonfinite": 0,
                    "symbol_alignment_missing": 0,
                    "future_close_missing": 0,
                    "other_target_nan": 0,
                }
                if target_nan_reasons is not None and n_va_target_nan > 0:
                    fold_reasons = target_nan_reasons[va_masked][target_nan_mask]
                    unique_reasons, counts = np.unique(fold_reasons, return_counts=True)
                    for reason, count in zip(unique_reasons, counts):
                        if reason in target_reasons:
                            target_reasons[reason] += count
                        else:
                            target_reasons["other_target_nan"] += count
                else:
                    target_reasons["other_target_nan"] += n_va_target_nan

                tprint(
                    f"WARNING: Fold {fold_id} Ridge validation: Dropped {n_va_before - n_va_after}/{n_va_before} "
                    f"({100*(1-n_va_after/n_va_before):.1f}%) samples. "
                    f"[Target NaN: {n_va_target_nan}, Feature NaN: {n_va_feat_nan}] "
                    f"TargetNaNReasons[horizon_exceeded={target_reasons['horizon_exceeded']}, barrier_unresolved={target_reasons['barrier_unresolved']}, ambiguous_bar={target_reasons['ambiguous_bar']}, outside_support_mask={target_reasons['outside_support_mask']}, neutral_filtered={target_reasons['neutral_filtered']}, current_close_missing={target_reasons['current_close_missing']}, atr_missing={target_reasons['atr_missing']}, transformed_target_nonfinite={target_reasons['transformed_target_nonfinite']}, symbol_alignment_missing={target_reasons['symbol_alignment_missing']}, future_close_missing={target_reasons['future_close_missing']}, other_target_nan={target_reasons['other_target_nan']}]"
                )

            X_tr_clean = X_tr[valid_tr]
            y_tr_clean = y_tr[valid_tr]
            X_va_clean = X_va[valid_va]
            y_va_clean = y_va[valid_va]
            mfe_tr_clean = None
            mae_tr_clean = None
            if mfe_tr is not None and mae_tr is not None:
                mfe_tr_clean = mfe_tr[valid_tr]
                mae_tr_clean = mae_tr[valid_tr]
            fold_filter_time += time.perf_counter() - fold_stage_start

            # Defensive check for any remaining NaNs (should not happen with valid_tr/valid_va)
            if not np.all(np.isfinite(X_tr_clean)) or not np.all(
                np.isfinite(y_tr_clean)
            ):
                folds_skipped += 1
                continue

            if is_binary_target:
                pos_tr = int(np.sum(y_tr_clean == 1))
                pos_va = int(np.sum(y_va_clean == 1))
                neg_tr = int(np.sum(y_tr_clean == 0))
                neg_va = int(np.sum(y_va_clean == 0))
                if pos_tr < min_train_req or pos_va < min_val_req:
                    folds_skipped += 1
                    continue
            else:
                if len(X_tr_clean) < min_train_req or len(X_va_clean) < min_val_req:
                    folds_skipped += 1
                    continue
                if np.nanstd(y_tr_clean) < 1e-12 or np.nanstd(y_va_clean) < 1e-12:
                    folds_skipped += 1
                    continue

            # Use all available training data (no subsampling)
            X_tr_subsample = X_tr_clean
            y_tr_subsample = y_tr_clean

            # Fit shallow LGBM (acting as proxy for former Ridge step)
            from sklearn.preprocessing import RobustScaler
            from lightgbm import LGBMRegressor
            from sklearn.pipeline import Pipeline

            # Helper: MinMax scale to custom range
            def _minmax_scale(arr, w_min, w_max):
                arr_min, arr_max = np.min(arr), np.max(arr)
                if arr_max > arr_min:
                    return w_min + (arr - arr_min) * (w_max - w_min) / (arr_max - arr_min)
                else:
                    return np.full_like(arr, (w_min + w_max) / 2.0)

            # Compute inverse-volatility sample weights for heteroscedasticity correction
            # Per-component MinMax to [0.5, 2.0]
            vol_weights_raw = make_ridge_vol_weights(
                y_tr_subsample,
                window=20,
                w_min=0.5,
                w_max=2.0,
            )
            vol_weights = _minmax_scale(vol_weights_raw, 0.5, 2.0)
            
            # Fee-aware weights get higher priority (wider range: 0.5 to 4.0)
            fee_weights_raw = make_fee_aware_target_weights(
                y_tr_subsample,
                fee_buffer=float(
                    self.cfg.get("training_label_round_trip_fee_pct", 0.002)
                ),
                near_zero_weight=float(
                    self.cfg.get("step2_fee_buffer_near_zero_weight", 0.5)
                ),
                large_target_weight=float(
                    self.cfg.get("step2_fee_buffer_large_target_weight", 2.0)
                ),
                large_target_multiple=float(
                    self.cfg.get("step2_fee_buffer_large_target_multiple", 3.0)
                ),
            )
            fee_weights = _minmax_scale(fee_weights_raw, 0.5, 4.0)
            
            # Excursion weights get reduced priority (narrower range: 0.5 to 2.0)
            if mfe_tr_clean is not None and mae_tr_clean is not None:
                exc_weights_raw = make_excursion_asymmetry_weights(
                    mfe_atr=mfe_tr_clean,
                    mae_atr=mae_tr_clean,
                    side=side,
                    alpha=float(self.cfg.get("step2_excursion_weight_alpha", 0.35)),
                    w_min=float(self.cfg.get("step2_excursion_weight_min", 0.8)),
                    w_max=float(self.cfg.get("step2_excursion_weight_max", 1.3)),
                )
                exc_weights = _minmax_scale(exc_weights_raw, 0.5, 2.0)
            else:
                exc_weights = np.ones_like(vol_weights)
            
            # Additive combination centered at 1.0
            lgbm_sample_weight = 1.0 + (vol_weights - 1.0) + (fee_weights - 1.0) + (exc_weights - 1.0)
            
            # Final clip to [0.25, 4.0] to prevent extreme outliers
            lgbm_sample_weight = np.clip(lgbm_sample_weight, 0.25, 4.0)
            lgbm_sample_weight = lgbm_sample_weight.astype(np.float32, copy=False)
            
            fit_start = time.perf_counter()
            # LGBM is a tree model - no scaling needed, removed RobustScaler
            model = LGBMRegressor(
                max_depth=int(model_params["max_depth"]), 
                n_estimators=int(model_params["n_estimators"]), 
                min_child_samples=int(model_params["min_child_samples"]),
                min_data_in_leaf=int(model_params["min_data_in_leaf"]),
                lambda_l1=float(model_params["lambda_l1"]),
                lambda_l2=float(model_params["lambda_l2"]),
                min_gain_to_split=float(model_params["min_gain_to_split"]),
                subsample=float(model_params["subsample"]),
                subsample_freq=int(model_params["subsample_freq"]),
                feature_fraction=float(model_params["feature_fraction"]),
                boosting_type=str(model_params["boosting_type"]),
                random_state=42,
                n_jobs=max(1, min(4, int(self.cfg.get("lgbm_n_jobs", 3)))),
                verbosity=-1,
            )
            
            # Feature-target correlation diagnostics - only on first fold
            if fold_id == 0:
                try:
                    # Diagnostic: Verify feature matrix integrity
                    tprint(f"  Feature matrix shape: {X_tr_subsample.shape}")
                    feat_vars = np.nanvar(X_tr_subsample, axis=0)
                    tprint(f"  Feature variance: min={feat_vars.min():.6f}, max={feat_vars.max():.6f}, n_zero_var={np.sum(feat_vars < 1e-12)}/{len(feat_vars)}")
                    tprint(f"  Feature range: col0=[{X_tr_subsample[:,0].min():.4f}, {X_tr_subsample[:,0].max():.4f}], col1=[{X_tr_subsample[:,1].min():.4f}, {X_tr_subsample[:,1].max():.4f}]")
                    
                    feat_target_corr = np.array([
                        np.corrcoef(X_tr_subsample[:, i], y_tr_subsample)[0, 1]
                        for i in range(X_tr_subsample.shape[1])
                    ])
                    max_abs_corr = np.nanmax(np.abs(feat_target_corr))
                    n_significant = np.sum(np.abs(feat_target_corr) > 0.05)
                    tprint(f"  Feature-target correlation: max_abs={max_abs_corr:.4f}, "
                           f"n_significant_5pct={n_significant}/{len(feat_target_corr)}")
                    if max_abs_corr < 0.05:
                        tprint(f"  WARNING: No features have significant correlation with target!")
                except Exception as e:
                    tprint(f"  Could not compute feature-target correlations: {e}")
            
            model.fit(X_tr_subsample, y_tr_subsample, sample_weight=lgbm_sample_weight)
            
            # Feature importance diagnostics - only on first fold to avoid spam
            if fold_id == 0:
                try:
                    importances = model.feature_importances_
                    ridge_meta = [self.metadata[i] for i in ridge_feats] if hasattr(self, "metadata") and self.metadata else []
                    feature_names = [m.source_name if hasattr(m, 'source_name') else f"feat_{i}" 
                                     for i, m in enumerate(ridge_meta)] if ridge_meta else [f"feat_{i}" for i in range(len(importances))]
                    
                    # Check if model is learning
                    max_importance = np.max(importances) if len(importances) > 0 else 0
                    if max_importance < 1e-6:
                        tprint(f"  WARNING: Fold {fold_id} LGBM feature importances all near-zero - model not learning!")
                    else:
                        # Top 10 features by importance
                        sorted_idx = np.argsort(importances)[::-1]
                        top_n = min(10, len(sorted_idx))
                        tprint(f"  Fold {fold_id} Top {top_n} LGBM features by importance (max={max_importance:.4f}):")
                        for i in range(top_n):
                            idx = sorted_idx[i]
                            feat_name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                            tprint(f"    {i+1}. {feat_name}: {importances[idx]:.4f}")
                        
                        # Summary stats
                        n_zero = np.sum(importances == 0)
                        tprint(f"  Feature importance summary: mean={np.mean(importances):.4f}, "
                               f"n_zero={n_zero}/{len(importances)}")
                except Exception as e:
                    tprint(f"  Could not extract feature importances: {e}")
            
            preds = model.predict(X_va_clean)
            fit_predict_time += time.perf_counter() - fit_start
            folds_used += 1

            # Store predictions
            oof_preds[va_masked[valid_va]] = preds

        subset_auc, coverage = self._compute_oof_learnability_score(
            oof_preds, y, mask, min_predicted_points=min_pred_points
        )
        class_metrics = self._compute_oof_classification_metrics(
            oof_preds, y, mask, min_predicted_points=min_pred_points
        )
        
        # Compute top-quartile precision if TP flags are provided
        top_quartile_precision = np.nan
        if tp_f is not None and tp_f.size > 0:
            top_quartile_precision = self._compute_top_quartile_precision(
                oof_preds=oof_preds,
                y=y,
                mask=mask,
                tp_f=tp_f,
                fwd_ret_threshold=0.5,
                top_pct=0.75,
                min_samples=20,
            )
        

        # --- Compute Expanded Learnability Metrics ---
        mask_oof_corr = np.nan
        mask_oof_r2 = np.nan
        fold_sign_consistency = np.nan
        positive_fold_fraction = np.nan
        negative_fold_fraction = np.nan
        fold_pnl_std = np.nan
        decile_monotonic_spearman = np.nan
        top_decile_mean_target = np.nan
        bottom_decile_mean_target = np.nan
        decile_spread_mean = np.nan

        target_nan_total_train = 0
        target_nan_total_val = 0
        train_target_nan_reasons = {
            "horizon_exceeded": 0,
            "barrier_unresolved": 0,
            "ambiguous_bar": 0,
            "outside_support_mask": 0,
            "neutral_filtered": 0,
            "current_close_missing": 0,
            "atr_missing": 0,
            "transformed_target_nonfinite": 0,
            "symbol_alignment_missing": 0,
            "future_close_missing": 0,
            "other_target_nan": 0,
        }
        val_target_nan_reasons = {
            "horizon_exceeded": 0,
            "barrier_unresolved": 0,
            "ambiguous_bar": 0,
            "outside_support_mask": 0,
            "neutral_filtered": 0,
            "current_close_missing": 0,
            "atr_missing": 0,
            "transformed_target_nonfinite": 0,
            "symbol_alignment_missing": 0,
            "future_close_missing": 0,
            "other_target_nan": 0,
        }

        for tr_idx, va_idx in folds:
            tr_masked = tr_idx[mask[tr_idx]]
            va_masked = va_idx[mask[va_idx]]

            y_tr = y[tr_masked]
            target_nan_mask_tr = ~np.isfinite(y_tr)
            target_nan_total_train += np.sum(target_nan_mask_tr)

            if target_nan_reasons is not None and np.sum(target_nan_mask_tr) > 0:
                fold_reasons = target_nan_reasons[tr_masked][target_nan_mask_tr]
                u_reasons, counts = np.unique(fold_reasons, return_counts=True)
                for r, c in zip(u_reasons, counts):
                    if r in train_target_nan_reasons:
                        train_target_nan_reasons[r] += c
                    else:
                        train_target_nan_reasons["other_target_nan"] += c
            else:
                train_target_nan_reasons["other_target_nan"] += np.sum(target_nan_mask_tr)

            y_va = y[va_masked]
            target_nan_mask_va = ~np.isfinite(y_va)
            target_nan_total_val += np.sum(target_nan_mask_va)

            if target_nan_reasons is not None and np.sum(target_nan_mask_va) > 0:
                fold_reasons = target_nan_reasons[va_masked][target_nan_mask_va]
                u_reasons, counts = np.unique(fold_reasons, return_counts=True)
                for r, c in zip(u_reasons, counts):
                    if r in val_target_nan_reasons:
                        val_target_nan_reasons[r] += c
                    else:
                        val_target_nan_reasons["other_target_nan"] += c
            else:
                val_target_nan_reasons["other_target_nan"] += np.sum(target_nan_mask_va)


        valid_mask = mask.astype(bool) & np.isfinite(y) & np.isfinite(oof_preds)
        effective_rows = int(np.sum(valid_mask))
        per_fold_ic: List[float] = []
        ic_series = np.asarray([], dtype=np.float32)

        if effective_rows > 0:
            preds_valid = oof_preds[valid_mask]
            targets_valid = y[valid_mask]

            # B1. Masked OOF Correlation
            if np.std(preds_valid) > 0 and np.std(targets_valid) > 0 and len(preds_valid) > 1:
                mask_oof_corr = np.corrcoef(preds_valid, targets_valid)[0, 1]

            # B2. Masked R2
            mean_y = np.mean(targets_valid)
            ss_tot = np.sum((targets_valid - mean_y) ** 2)
            ss_res = np.sum((targets_valid - preds_valid) ** 2)
            if ss_tot > 0:
                mask_oof_r2 = 1.0 - (ss_res / ss_tot)
            else:
                tprint("WARNING: Cannot compute mask_oof_r2 due to zero variance in targets.")

            # B3. Fold sign consistency and per-fold IC series
            fold_means = []
            for fold_id, (tr_idx, va_idx) in enumerate(folds):
                va_masked_idx = va_idx[mask[va_idx]]
                valid_fold_mask = np.isfinite(y[va_masked_idx]) & np.isfinite(oof_preds[va_masked_idx])
                fold_targets = y[va_masked_idx][valid_fold_mask]
                fold_preds = oof_preds[va_masked_idx][valid_fold_mask]
                if len(fold_targets) > 0:
                    fold_means.append(np.mean(fold_targets))
                    # Compute per-fold IC if we have enough samples
                    if len(fold_targets) >= 10 and np.std(fold_targets) > 0 and np.std(fold_preds) > 0:
                        fold_ic = float(np.corrcoef(fold_preds, fold_targets)[0, 1])
                        per_fold_ic.append(fold_ic)
                    else:
                        per_fold_ic.append(np.nan)
            ic_series = np.asarray(per_fold_ic, dtype=np.float32)
            if fold_means:
                n_pos = sum(1 for m in fold_means if m > 0)
                n_neg = sum(1 for m in fold_means if m < 0)
                n_nonzero = n_pos + n_neg
                fold_pnl_std = float(
                    np.nanstd(np.asarray(fold_means, dtype=np.float32), ddof=0)
                )

                positive_fold_fraction = n_pos / len(fold_means)
                negative_fold_fraction = n_neg / len(fold_means)

                if n_nonzero > 0:
                    fold_sign_consistency = max(n_pos, n_neg) / n_nonzero
                else:
                    tprint("WARNING: Cannot compute fold_sign_consistency due to zero fold means.")

            # B4. Decile monotonicity
            if effective_rows >= 10:
                import scipy.stats
                order = np.argsort(preds_valid)
                binned = np.array_split(order, 10)
                decile_means = []
                for b in binned:
                    if len(b) > 0:
                        decile_means.append(np.mean(targets_valid[b]))

                if len(decile_means) >= 5:
                    top_decile_mean_target = decile_means[-1]
                    bottom_decile_mean_target = decile_means[0]
                    decile_spread_mean = top_decile_mean_target - bottom_decile_mean_target

                    spearman_corr, _ = scipy.stats.spearmanr(np.arange(1, len(decile_means) + 1), decile_means)
                    if np.isfinite(spearman_corr):
                        decile_monotonic_spearman = float(spearman_corr)
                else:
                    tprint("WARNING: Cannot compute decile_monotonic_spearman due to insufficient populated deciles.")
            else:
                tprint("WARNING: Cannot compute decile_monotonic_spearman due to insufficient effective rows.")

        total_elapsed = time.perf_counter() - subset_auc_start
        if total_elapsed >= 0.20:
            tprint(
                "Stage A: Ridge subset AUC internals "
                f"folds_used={folds_used} folds_skipped={folds_skipped} "
                f"filter_elapsed={fold_filter_time:.2f}s fit_predict_elapsed={fit_predict_time:.2f}s "
                f"total_elapsed={total_elapsed:.2f}s"
            )
        return {
            "subset_auc": float(subset_auc) if np.isfinite(subset_auc) else np.nan,
            "subset_roc_auc": (
                float(class_metrics["roc_auc"])
                if np.isfinite(class_metrics["roc_auc"])
                else np.nan
            ),
            "subset_pr_auc": (
                float(class_metrics["pr_auc"])
                if np.isfinite(class_metrics["pr_auc"])
                else np.nan
            ),
            "top_quartile_precision": (
                float(top_quartile_precision)
                if np.isfinite(top_quartile_precision)
                else np.nan
            ),
            "coverage": float(coverage),
            "oof_preds": oof_preds,
            "folds_used": int(folds_used),
            "folds_skipped": int(folds_skipped),
            "mask_oof_corr": float(mask_oof_corr) if np.isfinite(mask_oof_corr) else np.nan,
            "mask_oof_r2": float(mask_oof_r2) if np.isfinite(mask_oof_r2) else np.nan,
            "fold_sign_consistency": float(fold_sign_consistency) if np.isfinite(fold_sign_consistency) else np.nan,
            "positive_fold_fraction": float(positive_fold_fraction) if np.isfinite(positive_fold_fraction) else np.nan,
            "negative_fold_fraction": float(negative_fold_fraction) if np.isfinite(negative_fold_fraction) else np.nan,
            "fold_pnl_std": float(fold_pnl_std) if np.isfinite(fold_pnl_std) else np.nan,
            "ic_series": ic_series,  # Per-fold IC values for Pareto overlap
            "decile_monotonic_spearman": float(decile_monotonic_spearman) if np.isfinite(decile_monotonic_spearman) else np.nan,
            "top_decile_mean_target": float(top_decile_mean_target) if np.isfinite(top_decile_mean_target) else np.nan,
            "bottom_decile_mean_target": float(bottom_decile_mean_target) if np.isfinite(bottom_decile_mean_target) else np.nan,
            "decile_spread_mean": float(decile_spread_mean) if np.isfinite(decile_spread_mean) else np.nan,
            "target_nan_total_train": target_nan_total_train,
            "target_nan_total_val": target_nan_total_val,
            "train_target_nan_reasons": train_target_nan_reasons,
            "val_target_nan_reasons": val_target_nan_reasons,
        }

    def _compute_ranked_ridge_trade_metrics(
        self,
        data: pd.DataFrame,
        directional_returns: np.ndarray,
        mask: np.ndarray,
        oof_preds: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]] = None,
        *,
        round_fee: float = 0.0015,
        min_weight: float = 0.05,
        max_weight: float = 0.15,
        convex_power: float = 2.0,
        eps: float = 1e-9,
        forbid_concurrent: bool = True,
        horizon: int = 1,
    ) -> Dict[str, Any]:
        def _empty_metrics() -> Dict[str, Any]:
            return {
                "threshold_star": np.nan,
                "threshold_star_lowest_positive": np.nan,
                "threshold_star_optimal_pnl": np.nan,
                "threshold_star_best_pnl_threshold": np.nan,
                "threshold_star_best_gross_pnl_threshold": np.nan,
                "ridge_pnl_gross_raw_at_optimal_threshold": np.nan,
                "ridge_pnl_gross_raw": 0.0,
                "ridge_pnl_raw_at_optimal_threshold": np.nan,
                "ridge_pnl_raw": 0.0,
                "avg_trades_per_day": np.nan,
                "avg_pnl_per_day": np.nan,
                "avg_pnl_per_active_symbol_day": np.nan,
                "ridge_trade_sortino_7d": 0.0,
                "ridge_trade_sortino_30d": 0.0,
                "ridge_trade_sortino_90d": 0.0,
                "ridge_trade_sortino_composite": 0.0,
                "trades_per_symbol_day_above_threshold_star": 0.0,
                "valid_symbol_days_observed": 0,
                "total_trades": 0,
                "threshold_search_mode": "grid",
                "n_quantiles_evaluated": 0,
                "n_thresholds_evaluated": 0,
                "n_unique_thresholds_evaluated": 0,
                "score_min": np.nan,
                "score_max": np.nan,
                "score_std": np.nan,
                "n_unique_scores": 0,
                "mask_signal_mean": np.nan,
                "mask_signal_std": np.nan,
                "mask_sharpe": np.nan,
                "rejected": True,
                "reject_reason": {"reason": "no valid threshold_star or no trades"},
                "realized_trades": [],
                "gross_weighted_returns": [],
                "net_weighted_returns": []
            }

        def _reject_metrics(reason_payload: Dict[str, Any], **metric_overrides: Any) -> Dict[str, Any]:
            res = _empty_metrics()
            res.update(metric_overrides)
            res["rejected"] = True
            res["reject_reason"] = reason_payload
            return res

        valid = (
            mask.astype(bool)
            & np.isfinite(directional_returns)
            & np.isfinite(oof_preds)
        )
        if int(np.sum(valid)) == 0:
            return _empty_metrics()

        scores = np.asarray(oof_preds[valid], dtype=np.float32)
        score_min = np.nan
        score_max = np.nan
        score_std = np.nan
        n_unique_scores = 0
        # Normalize scores to pseudo-confidence [0, 1] using rank-based scoring for robust threshold selection
        if len(scores) > 0:
            score_min = float(np.min(scores))
            score_max = float(np.max(scores))
            score_std = float(np.std(scores))
            n_unique_scores = int(np.unique(scores).size)
            if score_max > score_min:
                # Use true rank-based scoring instead of min-max normalization
                # This is more robust with low-variance predictions
                ranks = scipy.stats.rankdata(scores, method='average')
                # Normalize ranks to [0, 1] range (0 = lowest rank, 1 = highest rank)
                confidence_scores = (ranks - 1.0) / (len(scores) - 1.0) if len(scores) > 1 else np.zeros_like(scores)
            else:
                confidence_scores = np.zeros_like(scores)
        else:
            confidence_scores = np.array([])

        gross_returns = np.asarray(directional_returns[valid], dtype=np.float32)
        symbols = data.loc[valid, "symbol"].astype(str).to_numpy()
        timestamps = pd.to_datetime(
            data.loc[valid, "timestamp"], errors="coerce", utc=True
        )

        valid_ts = timestamps.notna().to_numpy()
        if not np.any(valid_ts):
            return _empty_metrics()

        confidence_scores = confidence_scores[valid_ts]
        gross_returns = gross_returns[valid_ts]
        symbols = symbols[valid_ts]
        timestamps = timestamps[valid_ts].to_numpy()

        # Calculate total valid symbol days observed from the candidate's valid subset
        # Floor the valid timestamps to the day level before counting unique days to get true symbol-days
        valid_days = pd.Series(timestamps).dt.floor("D")
        valid_df = pd.DataFrame({"symbol": symbols, "day": valid_days.to_numpy()}).drop_duplicates()
        valid_symbol_days_observed = len(valid_df)
        valid_calendar_days_observed = valid_df["day"].nunique()

        if valid_symbol_days_observed == 0 or valid_calendar_days_observed == 0:
             return _empty_metrics()

        # Check if actual exit time 't1' is available in data.
        # The event frame currently initializes a placeholder t1 = t0 + 1 second;
        # that placeholder must not be used for concurrency-aware monetization.
        if "t1" in data.columns:
            # We must map back the valid_ts rows to the original data index to fetch t1 correctly.
            # valid_ts applies to the valid mask subset.
            valid_indices = np.where(valid)[0]
            valid_indices = valid_indices[valid_ts]

            # Fetch corresponding exit times from data
            t1_vals = pd.to_datetime(data.iloc[valid_indices]["t1"], errors="coerce", utc=True).to_numpy()
            if len(t1_vals) > 0:
                inferred_deltas = pd.to_timedelta(t1_vals - timestamps, errors="coerce")
                finite_deltas = inferred_deltas[pd.notna(inferred_deltas)]
                median_delta = (
                    finite_deltas.median()
                    if len(finite_deltas) > 0
                    else pd.Timedelta(seconds=0)
                )
                if median_delta <= pd.Timedelta(minutes=1):
                    t1_vals = None
        else:
            t1_vals = None

        # Build trade objects
        all_trades = []
        for i in range(len(timestamps)):
             entry_t = timestamps[i]
             exit_t = None

             # Prefer actual exit time if available, otherwise fallback to horizon proxy
             # Validate that t1 > entry_time and that it is not NaT
             if t1_vals is not None:
                 t1_candidate = t1_vals[i]
                 if pd.notna(t1_candidate) and t1_candidate > entry_t:
                     exit_t = t1_candidate

             if exit_t is None:
                 # Fallback: estimate exit time using horizon proxy safely
                 exit_t = entry_t + pd.Timedelta(hours=horizon)

             # Ensure consistent timezone handling
             if getattr(exit_t, 'tz', None) is None and getattr(entry_t, 'tz', None) is not None:
                 exit_t = exit_t.tz_localize('UTC')

             all_trades.append(EvaluatedTrade(
                 entry_time=entry_t,
                 exit_time=exit_t,
                 confidence_score=float(confidence_scores[i]),
                gross_trade_return=float(gross_returns[i]),
                 symbol=symbols[i]
             ))

        # 3. Per-rule threshold search on score quantiles, optimized against net PnL.
        quantile_low = float(self.cfg.get("ridge_threshold_search_low_quantile", 0.50))
        quantile_high = float(self.cfg.get("ridge_threshold_search_high_quantile", 0.95))
        quantile_low = float(np.clip(quantile_low, 0.0, 1.0))
        quantile_high = float(np.clip(max(quantile_high, quantile_low + 1e-3), 0.0, 1.0))
        max_trials = max(1, int(self.cfg.get("ridge_threshold_search_trials", 7)))
        max_concurrent_per_symbol = 1
        max_concurrent_total = int(self.cfg.get("ridge_max_concurrent_total", 2))
        threshold_selection_policy = str(
            self.cfg.get("ridge_threshold_selection_policy", "best_net_pnl")
        ).strip().lower()

        threshold_star_lowest_positive = None
        threshold_star_optimal_pnl = None
        threshold_star_best_pnl_threshold = None
        threshold_star_best_gross_pnl_threshold = None  # Track best gross PnL threshold
        best_pnl_raw = -np.inf
        best_pnl_gross_raw = np.nan

        best_candidate: Optional[Dict[str, Any]] = None
        current_lowest_positive = np.inf
        threshold_eval_cache: Dict[float, Dict[str, Any]] = {}

        def _evaluate_threshold_value(
            threshold_value: float,
            quantile_level: float = np.nan,
        ) -> Dict[str, Any]:
            threshold_value = float(np.clip(threshold_value, 0.0, 1.0))
            threshold_key = float(np.round(threshold_value, 8))
            cached = threshold_eval_cache.get(threshold_key)
            if cached is not None:
                return cached

            pnl_res = compute_ridge_pnl(
                trades=all_trades,
                threshold_star=threshold_value,
                round_fee=round_fee,
                min_weight=min_weight,
                max_weight=max_weight,
                convex_power=convex_power,
                starting_capital=1.0,
                forbid_concurrent=forbid_concurrent,
                max_concurrent_per_symbol=max_concurrent_per_symbol,
                max_concurrent_total=max_concurrent_total,
            )
            result = {
                "quantile": quantile_level,
                "threshold": threshold_value,
                "pnl_res": pnl_res,
                "pnl_gross_raw": float(pnl_res["ridge_pnl_gross_raw"]),
                "pnl_raw": float(pnl_res["ridge_pnl_raw"]),
            }
            threshold_eval_cache[threshold_key] = result
            return result

        def _evaluate_threshold_for_quantile(quantile_level: float) -> Dict[str, Any]:
            quantile_level = float(np.clip(quantile_level, quantile_low, quantile_high))
            threshold_value = float(np.nanquantile(confidence_scores, quantile_level))
            return _evaluate_threshold_value(threshold_value, quantile_level)

        def _register_threshold_candidate(candidate: Dict[str, Any]) -> None:
            nonlocal best_candidate
            nonlocal best_pnl_raw
            nonlocal best_pnl_gross_raw
            nonlocal threshold_star_best_pnl_threshold
            nonlocal threshold_star_best_gross_pnl_threshold
            nonlocal threshold_star_optimal_pnl
            nonlocal current_lowest_positive
            nonlocal threshold_star_lowest_positive
            pnl_raw = float(candidate["pnl_raw"])
            pnl_gross = float(candidate.get("pnl_gross_raw", np.nan))
            threshold_value = float(candidate["threshold"])
            if pnl_raw > best_pnl_raw:
                best_pnl_raw = pnl_raw
                best_pnl_gross_raw = pnl_gross
                best_candidate = candidate
                threshold_star_best_pnl_threshold = threshold_value
                threshold_star_optimal_pnl = threshold_value
            # Track best gross PnL threshold separately
            if np.isfinite(pnl_gross) and pnl_gross > 0.0:
                if threshold_star_best_gross_pnl_threshold is None:
                    threshold_star_best_gross_pnl_threshold = threshold_value
            if pnl_raw > 0.0 and threshold_value < current_lowest_positive:
                current_lowest_positive = threshold_value
                threshold_star_lowest_positive = threshold_value

        quantile_grid = np.linspace(
            quantile_low,
            quantile_high,
            num=max_trials + 2,
            dtype=np.float64,
        )
        _register_threshold_candidate(_evaluate_threshold_value(0.0, 0.0))
        for quantile_level in quantile_grid:
            candidate = _evaluate_threshold_for_quantile(float(quantile_level))
            _register_threshold_candidate(candidate)

        n_quantiles_evaluated = int(len(quantile_grid) + 1)
        n_thresholds_evaluated = int(len(threshold_eval_cache))
        n_unique_thresholds_evaluated = n_thresholds_evaluated

        # Compute mask signal quality (Sharpe-like ratio within mask)
        masked_returns = directional_returns[mask.astype(bool) & np.isfinite(directional_returns)]
        mask_signal_mean = float(np.nanmean(masked_returns)) if len(masked_returns) > 0 else 0.0
        mask_signal_std = float(np.nanstd(masked_returns)) if len(masked_returns) > 0 else 0.0
        mask_sharpe = mask_signal_mean / (mask_signal_std + 1e-9) if mask_signal_std > 0 else 0.0
        
        # Configurable thresholds for research-grade acceptance
        min_gross_pnl_threshold = float(self.cfg.get("ridge_min_gross_pnl_threshold", 0.0))  # Default: any positive gross
        min_mask_sharpe_threshold = float(self.cfg.get("ridge_min_mask_sharpe_threshold", 0.3))  # Default: 0.3 Sharpe
        
        forced_reject_reason = None
        # NEW: Accept if gross PnL is positive AND mask has sufficient signal quality
        has_gross_profit = best_pnl_gross_raw > min_gross_pnl_threshold if np.isfinite(best_pnl_gross_raw) else False
        has_signal_quality = mask_sharpe >= min_mask_sharpe_threshold
        
        # Reject only if no gross profit AND no positive post-fee threshold
        if threshold_star_lowest_positive is None and not (has_gross_profit and has_signal_quality):
            if best_candidate is None:
                return _reject_metrics(
                    {
                        "reason": "no positive post-fee profit threshold",
                        "best_pnl_candidate": float(best_pnl_raw) if best_pnl_raw != -np.inf else 0.0,
                        "best_pnl_gross": float(best_pnl_gross_raw) if np.isfinite(best_pnl_gross_raw) else 0.0,
                        "mask_signal_mean": mask_signal_mean,
                        "mask_signal_std": mask_signal_std,
                        "mask_sharpe": mask_sharpe,
                        "threshold_star_optimal_pnl": threshold_star_optimal_pnl,
                        "threshold_star_best_pnl_threshold": threshold_star_best_pnl_threshold,
                        "threshold_star_best_gross_pnl_threshold": threshold_star_best_gross_pnl_threshold,
                        "n_quantiles_evaluated": n_quantiles_evaluated,
                        "n_thresholds_evaluated": n_thresholds_evaluated,
                        "n_unique_thresholds_evaluated": n_unique_thresholds_evaluated,
                        "score_min": score_min,
                        "score_max": score_max,
                        "score_std": score_std,
                        "n_unique_scores": n_unique_scores,
                    },
                    threshold_star_optimal_pnl=(
                        float(threshold_star_optimal_pnl)
                        if threshold_star_optimal_pnl is not None and np.isfinite(threshold_star_optimal_pnl)
                        else np.nan
                    ),
                    threshold_star_best_pnl_threshold=(
                        float(threshold_star_best_pnl_threshold)
                        if threshold_star_best_pnl_threshold is not None and np.isfinite(threshold_star_best_pnl_threshold)
                        else np.nan
                    ),
                    threshold_star_best_gross_pnl_threshold=(
                        float(threshold_star_best_gross_pnl_threshold)
                        if threshold_star_best_gross_pnl_threshold is not None and np.isfinite(threshold_star_best_gross_pnl_threshold)
                        else np.nan
                    ),
                    ridge_pnl_raw_at_optimal_threshold=(
                        float(best_pnl_raw) if best_pnl_raw != -np.inf else np.nan
                    ),
                    ridge_pnl_gross_raw_at_optimal_threshold=best_pnl_gross_raw,
                    valid_symbol_days_observed=valid_symbol_days_observed,
                    mask_signal_mean=mask_signal_mean,
                    mask_signal_std=mask_signal_std,
                    mask_sharpe=mask_sharpe,
                    n_quantiles_evaluated=n_quantiles_evaluated,
                    n_thresholds_evaluated=n_thresholds_evaluated,
                    n_unique_thresholds_evaluated=n_unique_thresholds_evaluated,
                    score_min=score_min,
                    score_max=score_max,
                    score_std=score_std,
                    n_unique_scores=n_unique_scores,
                )
            # Use gross PnL threshold if net PnL failed but gross passed
            if threshold_star_lowest_positive is None and has_gross_profit and has_signal_quality:
                threshold_star = float(threshold_star_best_gross_pnl_threshold if threshold_star_best_gross_pnl_threshold is not None else best_candidate["threshold"])
            else:
                threshold_star = float(best_candidate["threshold"])
            forced_reject_reason = {
                "reason": "no positive post-fee profit threshold (but passed gross+signal criteria)" if (has_gross_profit and has_signal_quality) else "no positive post-fee profit threshold",
                "best_pnl_candidate": float(best_pnl_raw) if best_pnl_raw != -np.inf else 0.0,
                "best_pnl_gross": float(best_pnl_gross_raw) if np.isfinite(best_pnl_gross_raw) else 0.0,
                "mask_signal_mean": mask_signal_mean,
                "mask_signal_std": mask_signal_std,
                "mask_sharpe": mask_sharpe,
                "threshold_star_optimal_pnl": threshold_star_optimal_pnl,
                "threshold_star_best_pnl_threshold": threshold_star_best_pnl_threshold,
                "threshold_star_best_gross_pnl_threshold": threshold_star_best_gross_pnl_threshold,
                "threshold_star_fallback": threshold_star,
                "passed_gross_criteria": has_gross_profit and has_signal_quality,
                "n_quantiles_evaluated": n_quantiles_evaluated,
                "n_thresholds_evaluated": n_thresholds_evaluated,
                "n_unique_thresholds_evaluated": n_unique_thresholds_evaluated,
                "score_min": score_min,
                "score_max": score_max,
                "score_std": score_std,
                "n_unique_scores": n_unique_scores,
            }
        else:
            if (
                threshold_selection_policy == "lowest_positive"
                and threshold_star_lowest_positive is not None
            ):
                threshold_star = float(threshold_star_lowest_positive)
            else:
                threshold_star = float(
                    threshold_star_best_pnl_threshold
                    if threshold_star_best_pnl_threshold is not None
                    and np.isfinite(threshold_star_best_pnl_threshold)
                    else threshold_star_lowest_positive
                )

        # 4. Final Realization and Trade-rate hard gate
        final_pnl_res = compute_ridge_pnl(
            trades=all_trades,
            threshold_star=threshold_star,
            round_fee=round_fee,
            min_weight=min_weight,
            max_weight=max_weight,
            convex_power=convex_power,
            starting_capital=1.0,
            forbid_concurrent=forbid_concurrent,
            max_concurrent_per_symbol=max_concurrent_per_symbol,
            max_concurrent_total=max_concurrent_total,
        )

        final_trades = final_pnl_res["selected_trades"]
        total_trades = len(final_trades)

        # Validation: check concurrency
        if total_trades > 0 and forbid_concurrent:
            events = []
            for t in final_trades:
                events.append((t.entry_time, 1, t.symbol))
                events.append((t.exit_time, -1, t.symbol))
            events.sort(key=lambda x: (x[0], x[1])) # Exit first on exact same time

            symbol_open = collections.defaultdict(int)
            total_open = 0
            for dt, delta, sym in events:
                symbol_open[sym] += delta
                total_open += delta
                if symbol_open[sym] > 1:
                    return _reject_metrics(
                        {"reason": "validation_failed: >1 trade per symbol"},
                        threshold_star=threshold_star,
                        threshold_star_lowest_positive=(
                            float(threshold_star_lowest_positive)
                            if threshold_star_lowest_positive is not None and np.isfinite(threshold_star_lowest_positive)
                            else np.nan
                        ),
                        threshold_star_optimal_pnl=(
                            float(threshold_star_optimal_pnl)
                            if threshold_star_optimal_pnl is not None and np.isfinite(threshold_star_optimal_pnl)
                            else np.nan
                        ),
                        threshold_star_best_pnl_threshold=(
                            float(threshold_star_best_pnl_threshold)
                            if threshold_star_best_pnl_threshold is not None and np.isfinite(threshold_star_best_pnl_threshold)
                            else np.nan
                        ),
                        ridge_pnl_raw_at_optimal_threshold=(
                            float(best_pnl_raw) if best_pnl_raw != -np.inf else np.nan
                        ),
                        ridge_pnl_gross_raw_at_optimal_threshold=best_pnl_gross_raw,
                        ridge_pnl_gross_raw=float(final_pnl_res.get("ridge_pnl_gross_raw", 0.0)),
                        ridge_pnl_raw=float(final_pnl_res.get("ridge_pnl_raw", 0.0)),
                        valid_symbol_days_observed=valid_symbol_days_observed,
                        total_trades=total_trades,
                        threshold_search_mode="grid",
                        threshold_selection_policy=threshold_selection_policy,
                        n_quantiles_evaluated=n_quantiles_evaluated,
                        n_thresholds_evaluated=n_thresholds_evaluated,
                        n_unique_thresholds_evaluated=n_unique_thresholds_evaluated,
                        score_min=score_min,
                        score_max=score_max,
                        score_std=score_std,
                        n_unique_scores=n_unique_scores,
                        realized_trades=final_trades,
                        gross_weighted_returns=final_pnl_res.get("weighted_gross_returns", []),
                        net_weighted_returns=final_pnl_res.get("weighted_net_returns", []),
                    )
                if total_open > max_concurrent_total:
                    return _reject_metrics(
                        {"reason": f"validation_failed: >{max_concurrent_total} total concurrent trades"},
                        threshold_star=threshold_star,
                        threshold_star_lowest_positive=(
                            float(threshold_star_lowest_positive)
                            if threshold_star_lowest_positive is not None and np.isfinite(threshold_star_lowest_positive)
                            else np.nan
                        ),
                        threshold_star_optimal_pnl=(
                            float(threshold_star_optimal_pnl)
                            if threshold_star_optimal_pnl is not None and np.isfinite(threshold_star_optimal_pnl)
                            else np.nan
                        ),
                        threshold_star_best_pnl_threshold=(
                            float(threshold_star_best_pnl_threshold)
                            if threshold_star_best_pnl_threshold is not None and np.isfinite(threshold_star_best_pnl_threshold)
                            else np.nan
                        ),
                        ridge_pnl_raw_at_optimal_threshold=(
                            float(best_pnl_raw) if best_pnl_raw != -np.inf else np.nan
                        ),
                        ridge_pnl_gross_raw_at_optimal_threshold=best_pnl_gross_raw,
                        ridge_pnl_gross_raw=float(final_pnl_res.get("ridge_pnl_gross_raw", 0.0)),
                        ridge_pnl_raw=float(final_pnl_res.get("ridge_pnl_raw", 0.0)),
                        valid_symbol_days_observed=valid_symbol_days_observed,
                        total_trades=total_trades,
                        threshold_search_mode="grid",
                        threshold_selection_policy=threshold_selection_policy,
                        n_quantiles_evaluated=n_quantiles_evaluated,
                        n_thresholds_evaluated=n_thresholds_evaluated,
                        n_unique_thresholds_evaluated=n_unique_thresholds_evaluated,
                        score_min=score_min,
                        score_max=score_max,
                        score_std=score_std,
                        n_unique_scores=n_unique_scores,
                        realized_trades=final_trades,
                        gross_weighted_returns=final_pnl_res.get("weighted_gross_returns", []),
                        net_weighted_returns=final_pnl_res.get("weighted_net_returns", []),
                    )

        # Alignment Note:
        # Numerator (`total_trades`) counts realized execution intervals defined by their entry_time.
        # Denominator (`valid_symbol_days_observed`) counts unique (symbol, entry_day) combinations in the valid universe.
        # Thus, this metric safely measures: "realized trades initiated per valid symbol-day".
        trades_per_symbol_day_above_threshold_star = total_trades / valid_symbol_days_observed

        if trades_per_symbol_day_above_threshold_star < 0.05:
            return _reject_metrics(
                {
                    "reason": "insufficient trades per symbol day",
                    "trades_per_symbol_day_above_threshold_star": trades_per_symbol_day_above_threshold_star,
                },
                threshold_star=threshold_star,
                threshold_star_lowest_positive=(
                    float(threshold_star_lowest_positive)
                    if threshold_star_lowest_positive is not None and np.isfinite(threshold_star_lowest_positive)
                    else np.nan
                ),
                threshold_star_optimal_pnl=(
                    float(threshold_star_optimal_pnl)
                    if threshold_star_optimal_pnl is not None and np.isfinite(threshold_star_optimal_pnl)
                    else np.nan
                ),
                ridge_pnl_raw_at_optimal_threshold=(
                    float(best_pnl_raw) if best_pnl_raw != -np.inf else np.nan
                ),
                ridge_pnl_gross_raw_at_optimal_threshold=best_pnl_gross_raw,
                threshold_star_best_pnl_threshold=(
                    float(threshold_star_best_pnl_threshold)
                    if threshold_star_best_pnl_threshold is not None and np.isfinite(threshold_star_best_pnl_threshold)
                    else np.nan
                ),
                ridge_pnl_gross_raw=float(final_pnl_res.get("ridge_pnl_gross_raw", 0.0)),
                ridge_pnl_raw=float(final_pnl_res.get("ridge_pnl_raw", 0.0)),
                trades_per_symbol_day_above_threshold_star=trades_per_symbol_day_above_threshold_star,
                valid_symbol_days_observed=valid_symbol_days_observed,
                total_trades=total_trades,
                threshold_search_mode="grid",
                threshold_selection_policy=threshold_selection_policy,
                n_quantiles_evaluated=n_quantiles_evaluated,
                n_thresholds_evaluated=n_thresholds_evaluated,
                n_unique_thresholds_evaluated=n_unique_thresholds_evaluated,
                score_min=score_min,
                score_max=score_max,
                score_std=score_std,
                n_unique_scores=n_unique_scores,
                realized_trades=final_trades,
                gross_weighted_returns=final_pnl_res.get("weighted_gross_returns", []),
                net_weighted_returns=final_pnl_res.get("weighted_net_returns", []),
            )

        # 5. compute metrics consistently using realized trades
        # We compute trade-level Sortino composites over multiple lookback windows.

        # The latest entry time across all candidate's valid rows
        latest_valid_entry = timestamps.max() if len(timestamps) > 0 else pd.Timestamp.now(tz="UTC")

        def _get_sortino_for_window(days_lookback: int) -> float:
            cutoff = latest_valid_entry - pd.Timedelta(days=days_lookback)
            window_trades = [t for t in final_trades if t.entry_time >= cutoff]
            if len(window_trades) < 2:
                # Stable default if insufficient trades in the window
                return 0.0

            res = compute_ridge_trade_sortino(
                realized_trades=window_trades,
                threshold_star=threshold_star,
                round_fee=round_fee,
                min_weight=min_weight,
                max_weight=max_weight,
                convex_power=convex_power
            )
            return res["ridge_trade_sortino"]

        sortino_7d = _get_sortino_for_window(7)
        sortino_30d = _get_sortino_for_window(30)
        sortino_90d = _get_sortino_for_window(90)

        ridge_trade_sortino_composite = (
            (sortino_7d + sortino_30d + sortino_90d) / 3.0
        )

        # 6. Compute avg PnL metrics
        ridge_pnl_gross_raw = final_pnl_res["ridge_pnl_gross_raw"]
        ridge_pnl_raw = final_pnl_res["ridge_pnl_raw"]

        # Denominators are purely derived from the candidate's valid universe
        avg_trades_per_day = total_trades / max(valid_calendar_days_observed, 1)
        avg_pnl_per_day = ridge_pnl_raw / max(valid_calendar_days_observed, 1)

        if total_trades > 0:
            active_days_series = pd.Series([t.entry_time for t in final_trades]).dt.floor("D")
            active_symbols_series = pd.Series([t.symbol for t in final_trades])
            active_symbol_days = pd.DataFrame({"symbol": active_symbols_series, "day": active_days_series}).drop_duplicates().shape[0]
            avg_pnl_per_active_symbol_day = ridge_pnl_raw / max(active_symbol_days, 1)
        else:
            avg_pnl_per_active_symbol_day = np.nan

        return {
            "threshold_star": threshold_star,
            "threshold_star_lowest_positive": threshold_star_lowest_positive,
            "threshold_star_optimal_pnl": threshold_star_optimal_pnl,
            "threshold_star_best_pnl_threshold": threshold_star_best_pnl_threshold,
            "threshold_star_best_gross_pnl_threshold": threshold_star_best_gross_pnl_threshold,
            "ridge_pnl_gross_raw_at_optimal_threshold": best_pnl_gross_raw,
            "ridge_pnl_gross_raw": ridge_pnl_gross_raw,
            "ridge_pnl_raw_at_optimal_threshold": best_pnl_raw,
            "ridge_pnl_raw": ridge_pnl_raw,
            "avg_trades_per_day": avg_trades_per_day,
            "avg_pnl_per_day": avg_pnl_per_day,
            "avg_pnl_per_active_symbol_day": avg_pnl_per_active_symbol_day,
            "ridge_trade_sortino_7d": sortino_7d,
            "ridge_trade_sortino_30d": sortino_30d,
            "ridge_trade_sortino_90d": sortino_90d,
            "ridge_trade_sortino_composite": ridge_trade_sortino_composite,
            "trades_per_symbol_day_above_threshold_star": trades_per_symbol_day_above_threshold_star,
            "valid_symbol_days_observed": valid_symbol_days_observed,
            "total_trades": total_trades,
            "threshold_search_mode": "grid",
            "threshold_selection_policy": threshold_selection_policy,
            "n_quantiles_evaluated": n_quantiles_evaluated,
            "n_thresholds_evaluated": n_thresholds_evaluated,
            "n_unique_thresholds_evaluated": n_unique_thresholds_evaluated,
            "score_min": score_min,
            "score_max": score_max,
            "score_std": score_std,
            "n_unique_scores": n_unique_scores,
            "mask_signal_mean": mask_signal_mean,
            "mask_signal_std": mask_signal_std,
            "mask_sharpe": mask_sharpe,
            "rejected": forced_reject_reason is not None,
            "reject_reason": forced_reject_reason,
            "realized_trades": final_trades,
            "gross_weighted_returns": final_pnl_res.get("weighted_gross_returns", []),
            "net_weighted_returns": final_pnl_res.get("weighted_net_returns", []),
            "weighted_fee_returns": final_pnl_res.get("weighted_fee_returns", []),
            "avg_fee_per_trade": float(final_pnl_res.get("avg_fee_per_trade", 0.0)),
            "avg_gross_move_per_trade": float(final_pnl_res.get("avg_gross_move_per_trade", 0.0)),
            "avg_position_weight": float(final_pnl_res.get("avg_position_weight", 0.0)),
        }

    def _compute_entropy(self, y) -> float:
        """Compute entropy proxy of the target distribution."""
        if len(y) == 0:
            return 0.0
        if np.all(np.isin(y, [0, 1])):
            p1 = np.mean(y)
            if p1 <= 0 or p1 >= 1:
                return 0.0
            return float(-(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1)))
        else:
            return float(np.log(np.std(y) + 1e-9))

    def _compute_baseline_auc(
        self,
        X: np.ndarray,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        eval_returns: Optional[np.ndarray] = None,
        positive_return_threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute baseline AUC using ridge features across all folds.
        Uses only 50% of the data for Ridge model training.
        """
        ridge_feats = self._get_ridge_feature_indices()
        if ridge_feats.size == 0:
            return {
                "global_auc": np.nan,
                "global_roc_auc": np.nan,
                "global_pr_auc": np.nan,
                "global_top_quartile_precision": np.nan,
                "global_cov": 0.0,
            }

        X_ridge = self._build_ridge_design_matrix(X)
        y = np.asarray(fwd_ret, dtype=np.float32)
        y[~np.isfinite(fwd_ret)] = np.nan
        (
            is_binary_target,
            min_train_req,
            min_val_req,
            min_pred_points,
        ) = self._ridge_learnability_thresholds(y)

        # Compute OOF predictions using Ridge (use 50% of data)
        from sklearn.linear_model import Ridge

        oof_preds = np.full(len(X), np.nan, dtype=np.float32)
        rng = np.random.RandomState(42)

        for fold_id, (tr_idx, va_idx) in enumerate(folds):
            X_tr, X_va = X_ridge[tr_idx], X_ridge[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # Filter valid samples
            valid_tr = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)
            valid_va = np.isfinite(y_va) & np.all(np.isfinite(X_va), axis=1)

            X_tr_clean = X_tr[valid_tr]
            y_tr_clean = y_tr[valid_tr]
            X_va_clean = X_va[valid_va]
            y_va_clean = y_va[valid_va]

            if is_binary_target:
                pos_tr = int(np.sum(y_tr_clean == 1))
                pos_va = int(np.sum(y_va_clean == 1))
                neg_tr = int(np.sum(y_tr_clean == 0))
                neg_va = int(np.sum(y_va_clean == 0))
                if pos_tr < min_train_req or pos_va < min_val_req:
                    tprint(
                        f"Skipping fold {fold_id} in baseline Ridge evaluation due to insufficient positive samples. "
                        f"Train: {pos_tr} pos / {neg_tr} neg. Validation: {pos_va} pos / {neg_va} neg. "
                        f"(Required: train>={min_train_req} pos, val>={min_val_req} pos)"
                    )
                    continue
            else:
                if len(X_tr_clean) < min_train_req or len(X_va_clean) < min_val_req:
                    tprint(
                        f"Skipping fold {fold_id} in baseline Ridge evaluation due to insufficient continuous samples. "
                        f"Train={len(X_tr_clean)} Validation={len(X_va_clean)} "
                        f"(Required: train>={min_train_req}, val>={min_val_req})"
                    )
                    continue
                if np.nanstd(y_tr_clean) < 1e-12 or np.nanstd(y_va_clean) < 1e-12:
                    tprint(
                        f"Skipping fold {fold_id} in baseline Ridge evaluation due to near-zero continuous target variance."
                    )
                    continue

            # Use all available training data (no subsampling)
            X_tr_subsample = X_tr_clean
            y_tr_subsample = y_tr_clean

            # Fit shallow LGBM (acting as proxy for former Ridge step)
            from sklearn.preprocessing import RobustScaler
            from lightgbm import LGBMRegressor
            from sklearn.pipeline import Pipeline

            # Compute inverse-volatility sample weights for heteroscedasticity correction
            lgbm_sample_weight = make_ridge_vol_weights(
                y_tr_subsample,
                window=20,
                w_min=0.5,
                w_max=2.0,
            )
            lgbm_sample_weight = lgbm_sample_weight * make_fee_aware_target_weights(
                y_tr_subsample,
                fee_buffer=float(
                    self.cfg.get("training_label_round_trip_fee_pct", 0.002)
                ),
                near_zero_weight=float(
                    self.cfg.get("step2_fee_buffer_near_zero_weight", 0.5)
                ),
                large_target_weight=float(
                    self.cfg.get("step2_fee_buffer_large_target_weight", 2.0)
                ),
                large_target_multiple=float(
                    self.cfg.get("step2_fee_buffer_large_target_multiple", 3.0)
                ),
            )
            # Rebalance weights to [0.5, 2.0] using MinMax scaling instead of simple clip
            w_min_final = float(self.cfg.get("step2_sample_weight_min_final", 0.5))
            w_max_final = float(self.cfg.get("step2_sample_weight_max_final", 2.0))
            w_curr_min = np.min(lgbm_sample_weight)
            w_curr_max = np.max(lgbm_sample_weight)
            if w_curr_max > w_curr_min:
                lgbm_sample_weight = w_min_final + (lgbm_sample_weight - w_curr_min) * (w_max_final - w_min_final) / (w_curr_max - w_curr_min)
            else:
                lgbm_sample_weight = np.full_like(lgbm_sample_weight, (w_min_final + w_max_final) / 2.0)
            lgbm_sample_weight = lgbm_sample_weight.astype(np.float32, copy=False)
            
            # LGBM is a tree model - no scaling needed, removed RobustScaler
            model = LGBMRegressor(
                max_depth=3, 
                n_estimators=5, 
                min_child_samples=20,
                min_data_in_leaf=20,
                random_state=42,
                n_jobs=max(1, min(4, int(self.cfg.get("lgbm_n_jobs", 3)))),
                verbosity=-1,
            )
            model.fit(X_tr_subsample, y_tr_subsample, sample_weight=lgbm_sample_weight)
            preds = model.predict(X_va_clean)

            # Store predictions
            oof_preds[va_idx[valid_va]] = preds

        global_auc, global_cov = self._compute_oof_learnability_score(
            oof_preds, y, np.isfinite(y), min_predicted_points=min_pred_points
        )
        class_metrics = self._compute_oof_classification_metrics(
            oof_preds, y, np.isfinite(y), min_predicted_points=min_pred_points
        )
        
        eval_y = (
            np.asarray(eval_returns, dtype=np.float32)
            if eval_returns is not None
            else y
        )
        if positive_return_threshold is None:
            positive_return_threshold = float(
                self.cfg.get("ridge_cost_pct", 0.003)
            )

        # Compute global top-quartile precision (no mask = all data)
        global_top_quartile_precision = self._compute_top_quartile_precision(
            oof_preds=oof_preds,
            y=eval_y,
            mask=np.isfinite(eval_y),  # All valid samples
            tp_f=np.zeros(len(y), dtype=np.int8),  # No TP data for global, use fwd_ret threshold only
            fwd_ret_threshold=float(positive_return_threshold),
            top_pct=0.75,
            min_samples=20,
        )
        return {
            "global_auc": float(global_auc) if np.isfinite(global_auc) else np.nan,
            "global_roc_auc": (
                float(class_metrics["roc_auc"])
                if np.isfinite(class_metrics["roc_auc"])
                else np.nan
            ),
            "global_pr_auc": (
                float(class_metrics["pr_auc"])
                if np.isfinite(class_metrics["pr_auc"])
                else np.nan
            ),
            "global_top_quartile_precision": (
                float(global_top_quartile_precision)
                if np.isfinite(global_top_quartile_precision)
                else np.nan
            ),
            "global_cov": float(global_cov),
        }

    def _get_mask_for_rule(self, key: str, X: np.ndarray) -> np.ndarray:
        """
        Parses '(F1==1)|(LOC1==0)|(*)' into a boolean mask.
        """
        parts = key.split("|")
        mask = np.ones(X.shape[0], dtype=bool)
        for p in parts:
            p = p.strip("()")
            if p == "*":
                continue
            for cond_str in p.split("&"):
                parsed = parse_condition_string(cond_str)
                if parsed is None:
                    continue
                fname, operator, val_part = parsed
                val = float(val_part)
                # Find matching metadata for feature index
                f_idx = next(
                    m.feature_index for m in self.metadata if m.feature_name == fname
                )
                if operator == "==":
                    mask &= X[:, f_idx] == val
                elif operator == "<=":
                    mask &= X[:, f_idx] <= val
                elif operator == ">":
                    mask &= X[:, f_idx] > val
                elif operator == "<":
                    mask &= X[:, f_idx] < val
                elif operator == ">=":
                    mask &= X[:, f_idx] >= val
        return mask


# =============================================================================
# NUMBA-OPTIMIZED INFERENCE ENGINE
# =============================================================================


@njit(parallel=True, cache=True, fastmath=True)
def _generate_masks_numba_kernel(
    X: np.ndarray,
    cond_feat_idxs: np.ndarray,
    cond_vals: np.ndarray,
    rule_ptr: np.ndarray,
) -> np.ndarray:
    """
    Highly optimized kernel to apply N-rules to M-samples.
    rule_ptr: array of indices marking the start/end of each rule's conditions.
    """
    n_samples = X.shape[0]
    n_rules = len(rule_ptr) - 1
    out = np.ones((n_samples, n_rules), dtype=np.bool_)

    # Parallelize across samples for high-throughput
    for i in prange(n_samples):
        for r in range(n_rules):
            start = rule_ptr[r]
            end = rule_ptr[r + 1]

            # Intersection of conditions (AND logic within a path)
            for c in range(start, end):
                f_idx = cond_feat_idxs[c]
                target_val = cond_vals[c]

                # Check if boolean feature matches target normalized value
                if X[i, f_idx] != target_val:
                    out[i, r] = False
                    break
    return out


class NumbaRuleInferenceEngine:
    def __init__(self, registry: pd.DataFrame, metadata: List[FeatureMetadata]):
        self.registry = registry
        self.metadata = metadata
        self.name_to_idx = {m.feature_name: m.feature_index for m in metadata}

        # Flattened structures for Numba
        self.feat_idxs = []
        self.target_vals = []
        self.rule_ptrs = [0]

        self._compile_registry()

    def _compile_registry(self):
        """Pre-processes strings into flat integer arrays for Numba."""
        for row in self.registry.itertuples(index=False, name=None):
            key = row[0]  # canonical_key
            # Note: For Composite rules, we currently treat them as their
            # atomic components in the registry or manage them as separate vectors.
            # This engine handles standard (T|L|R) path logic.
            conditions = self._parse_key(key)

            for f_idx, val in conditions:
                self.feat_idxs.append(f_idx)
                self.target_vals.append(val)

            self.rule_ptrs.append(len(self.feat_idxs))

        self.feat_idxs_np = np.array(self.feat_idxs, dtype=np.int32)
        self.target_vals_np = np.array(self.target_vals, dtype=np.int32)
        self.rule_ptrs_np = np.array(self.rule_ptrs, dtype=np.int32)

    def _parse_key(self, key: str) -> List[Tuple[int, int]]:
        parts = key.split("|")
        parsed = []
        for p in parts:
            p = p.strip("()")
            if p == "*":
                continue
            for cond_str in p.split("&"):
                if "==" not in cond_str:
                    continue
                name, val = cond_str.split("==")
                if name in self.name_to_idx:
                    parsed.append((self.name_to_idx[name], int(val)))
                else:
                    raise KeyError(f"Feature {name} not found in metadata.")
        return parsed

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Entry point for inference."""
        # X must be float32 or int for Numba kernel
        return _generate_masks_numba_kernel(
            X.astype(np.float32),
            self.feat_idxs_np,
            self.target_vals_np,
            self.rule_ptrs_np,
        )


# =============================================================================
# FEATURE USAGE AUDIT HELPERS
# =============================================================================


def export_feature_group_summary(
    metadata: List[FeatureMetadata], output_dir: Path
) -> pd.DataFrame:
    """Export feature metadata and group summary."""
    rows = []
    for m in metadata:
        rows.append(
            {
                "feature_name": m.feature_name,
                "feature_index": m.feature_index,
                "group": m.group,
                "source_name": m.source_name,
                "source_family": m.source_family,
                "source_type": m.source_type,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "retained_feature_metadata.csv", index=False)

    summary = df.groupby("group").size().reset_index(name="retained_feature_count")
    summary.to_csv(output_dir / "retained_feature_group_summary.csv", index=False)
    return df


def collect_split_usage_from_model(
    model, metadata: List[FeatureMetadata], fold_id: int, seed: int
) -> pd.DataFrame:
    """
    Count split-feature usage directly from LightGBM tree dump.
    """
    idx_to_meta = {m.feature_index: m for m in metadata}
    dump = model.booster_.dump_model()

    counts = collections.Counter()

    def walk(node):
        if "split_feature" in node:
            counts[node["split_feature"]] += 1
            walk(node["left_child"])
            walk(node["right_child"])

    for tree in dump["tree_info"]:
        walk(tree["tree_structure"])

    rows = []
    for feat_idx, split_count in counts.items():
        m = idx_to_meta.get(feat_idx)
        if m is None:
            continue
        rows.append(
            {
                "fold_id": fold_id,
                "seed": seed,
                "feature_index": feat_idx,
                "feature_name": m.feature_name,
                "group": m.group,
                "source_name": m.source_name,
                "source_family": m.source_family,
                "split_count": split_count,
            }
        )

    return pd.DataFrame(rows)


def summarize_fold_feature_usage(split_usage_df: pd.DataFrame) -> pd.DataFrame:
    if split_usage_df.empty:
        return pd.DataFrame(
            columns=["fold_id", "seed", "group", "used_feature_count", "split_count"]
        )
    grouped = (
        split_usage_df.groupby(["fold_id", "seed", "group"], as_index=False)
        .agg(
            used_feature_count=("feature_name", "nunique"),
            split_count=("split_count", "sum"),
        )
        .sort_values(["fold_id", "seed", "group"])
    )
    return grouped


def collect_extracted_rule_feature_usage(
    rules: List[ExtractedRule], metadata: List[FeatureMetadata]
) -> pd.DataFrame:
    """Collect feature usage from extracted rules."""
    idx_to_meta = {m.feature_index: m for m in metadata}
    rows = []

    for r in rules:
        used = set()
        for c in r.conditions:
            if c.feature_index in used:
                continue
            used.add(c.feature_index)
            m = idx_to_meta[c.feature_index]
            rows.append(
                {
                    "canonical_key": r.canonical_key,
                    "rule_id": r.rule_id,
                    "fold_id": r.fold_id,
                    "seed": r.seed,
                    "feature_index": c.feature_index,
                    "feature_name": m.feature_name,
                    "group": m.group,
                    "source_name": m.source_name,
                    "source_family": m.source_family,
                    "normalized_value": c.normalized_value,
                }
            )

    return pd.DataFrame(rows)


def collect_registry_feature_usage(
    registry: pd.DataFrame, metadata: List[FeatureMetadata]
) -> pd.DataFrame:
    """Collect feature usage from final registry."""
    name_to_meta = {m.feature_name: m for m in metadata}
    rows = []

    for row in registry.itertuples(index=False, name=None):
        canonical_key = row[0]  # canonical_key
        for feature_name in extract_feature_names_from_key(canonical_key):
            m = name_to_meta.get(feature_name)
            if m is None:
                continue
            rows.append(
                {
                    "canonical_key": canonical_key,
                    "feature_index": m.feature_index,
                    "feature_name": m.feature_name,
                    "group": m.group,
                    "source_name": m.source_name,
                    "source_family": m.source_family,
                }
            )

    return pd.DataFrame(rows)


def select_top_diverse_rules(
    registry: pd.DataFrame,
    mask_map: Dict[str, np.ndarray],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Select top `top_n` candidates using greedy, iterative selection:
    - Recompute overlap against already accepted rules
    - Recompute selection_score for all remaining candidates
    - Pick the best acceptable candidate
    """
    if registry.empty:
        return registry

    score_col = "base_regime_score" if "base_regime_score" in registry.columns else "regime_score"
    working_reg = registry.copy().reset_index(drop=True)

    selected_idx = []
    remaining_idx = set(working_reg.index)

    def dice_overlap(mask_a, mask_b):
        intersection = float(np.sum(mask_a & mask_b))
        sum_a = float(np.sum(mask_a))
        sum_b = float(np.sum(mask_b))
        if sum_a + sum_b == 0:
            return 0.0
        return 2.0 * intersection / (sum_a + sum_b)

    while len(selected_idx) < top_n and remaining_idx:
        best_score = -np.inf
        best_idx = None
        best_overlap_penalty = 0.0

        # Precompute the union mask of already accepted rules to save time
        accepted_union_mask = None
        if selected_idx:
            accepted_masks = [mask_map.get(working_reg.loc[s_idx, "canonical_key"]) for s_idx in selected_idx if mask_map.get(working_reg.loc[s_idx, "canonical_key"]) is not None]
            if accepted_masks:
                accepted_union_mask = np.logical_or.reduce(accepted_masks)

        to_remove = []
        updates = []

        # Iterate over a sorted list to ensure deterministic traversal
        current_remaining = sorted(list(remaining_idx))
        for idx in current_remaining:
            row = working_reg.loc[idx]
            key = str(row["canonical_key"])
            rule_mask = mask_map.get(key)
            if rule_mask is None:
                to_remove.append(idx)
                continue

            base_regime_score = float(row.get(score_col, 0.0))
            worst_penalty = float(row.get("worst_penalty", 0.0))

            pairwise_raw_overlap = 0.0
            eligible_pairwise_overlaps = []
            for s_idx in selected_idx:
                sel = working_reg.loc[s_idx]
                s_key = sel["canonical_key"]
                s_mask = mask_map.get(s_key)
                if s_mask is not None:
                    raw_overlap = dice_overlap(rule_mask, s_mask)
                    different_sides = str(row.get("side")) != str(sel.get("side"))
                    different_horizons = str(row.get("source_horizon")) != str(sel.get("source_horizon"))
                    side_factor = 0.3 if different_sides else 0.0
                    horizon_factor = 0.2 if different_horizons else 0.0
                    difference_leniency = max(0.0, side_factor + horizon_factor)
                    effective_overlap = raw_overlap * (1.0 - difference_leniency)
                    eligible_pairwise_overlaps.append(effective_overlap)

            if eligible_pairwise_overlaps:
                pairwise_effective_overlap = max(eligible_pairwise_overlaps)

            pairwise_overlap_penalty = pairwise_effective_overlap if pairwise_effective_overlap >= 0.30 else 0.0

            union_raw_overlap = 0.0
            union_effective_overlap = 0.0
            if accepted_union_mask is not None:
                # Compute raw overlap with union mask
                union_raw_overlap = dice_overlap(rule_mask, accepted_union_mask)

                # Compute conservative leniency for union: if ANY selected rule differs
                # in side or horizon from candidate, apply the full leniency
                candidate_side = str(row.get("side"))
                candidate_horizon = str(row.get("source_horizon"))
                different_sides = any(
                    str(working_reg.loc[s_idx, "side"]) != candidate_side
                    for s_idx in selected_idx
                )
                different_horizons = any(
                    str(working_reg.loc[s_idx, "source_horizon"]) != candidate_horizon
                    for s_idx in selected_idx
                )
                side_factor = 0.3 if different_sides else 0.0
                horizon_factor = 0.2 if different_horizons else 0.0
                difference_leniency = max(0.0, side_factor + horizon_factor)
                union_effective_overlap = union_raw_overlap * (1.0 - difference_leniency)

            union_overlap_penalty = union_effective_overlap if union_effective_overlap >= 0.40 else 0.0

            # Hard reject conditions - use effective_overlap consistently
            if pairwise_effective_overlap >= 0.70 or union_effective_overlap >= 0.75:
                to_remove.append(idx)
                continue

            overlap_penalty = max(
                pairwise_overlap_penalty,
                0.70 * union_overlap_penalty,
            )

            selection_score = (
                base_regime_score
                - 0.2 * (overlap_penalty ** 2)
                - 0.1 * (worst_penalty ** 2)
            )

            # Store metrics for batch update to avoid in-loop mutation of dataframe
            updates.append((idx, overlap_penalty, selection_score))

            if selection_score > best_score:
                best_score = selection_score
                best_idx = idx
                best_overlap_penalty = overlap_penalty

        # Apply updates outside the iteration loop safely
        for idx, overlap_penalty, selection_score in updates:
            working_reg.at[idx, "overlap_penalty"] = overlap_penalty
            working_reg.at[idx, "selection_score"] = selection_score

        # Apply removals
        remaining_idx.difference_update(to_remove)

        if best_idx is not None and np.isfinite(best_score):
            selected_idx.append(best_idx)
            if best_idx in remaining_idx:
                remaining_idx.remove(best_idx)
        else:
            break

    return working_reg.loc[selected_idx]


def build_portfolio_diversity_report(
    registry: pd.DataFrame,
    resolver: Union[CanonicalRuleMaskResolver, DictionaryMaskResolver],
    data: pd.DataFrame,
    fwd_ret: np.ndarray,
    X_for_ridge: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    del fwd_ret, X_for_ridge
    rows: List[Dict[str, Any]] = []
    if registry.empty:
        return pd.DataFrame(rows)

    mask_map = {
        key: resolver.get_mask(key) for key in registry["canonical_key"].tolist()
    }
    activation_counts = {key: int(mask.sum()) for key, mask in mask_map.items()}
    total_activations = sum(activation_counts.values())
    if total_activations > 0:
        shares = (
            np.array(list(activation_counts.values()), dtype=np.float64)
            / total_activations
        )
        effective_rules = 1.0 / np.sum(shares**2)
        top_rule_share = float(np.max(shares))
    else:
        effective_rules = 0.0
        top_rule_share = 0.0

    rows.append(
        {"category": "summary", "metric": "top_rule_share", "value": top_rule_share}
    )
    rows.append(
        {
            "category": "summary",
            "metric": "effective_independent_rules",
            "value": effective_rules,
        }
    )

    keys = registry["canonical_key"].tolist()
    top_keys = (
        registry.sort_values(["composite_score", "hurdle_excess"], ascending=False)
        .head(20)["canonical_key"]
        .tolist()
    )
    for key_a, key_b in itertools.combinations(top_keys, 2):
        mask_a = mask_map[key_a]
        mask_b = mask_map[key_b]
        intersection = float(np.sum(mask_a & mask_b))
        union = float(np.sum(mask_a | mask_b))
        support_a = float(np.sum(mask_a))
        support_b = float(np.sum(mask_b))
        jaccard = intersection / union if union > 0 else 0.0
        overlap_coeff = (
            intersection / min(support_a, support_b)
            if min(support_a, support_b) > 0
            else 0.0
        )
        rows.append(
            {
                "category": "pairwise",
                "metric": "jaccard_overlap",
                "item_a": key_a,
                "item_b": key_b,
                "value": jaccard,
            }
        )
        rows.append(
            {
                "category": "pairwise",
                "metric": "overlap_coeff",
                "item_a": key_a,
                "item_b": key_b,
                "value": overlap_coeff,
            }
        )

    if "symbol" in data.columns:
        symbol_series = data["symbol"].astype(str)
        for key, mask in mask_map.items():
            counts = symbol_series[mask].value_counts(normalize=True)
            for symbol, share in counts.items():
                rows.append(
                    {
                        "category": "coverage_symbol",
                        "metric": key,
                        "item_a": symbol,
                        "value": float(share),
                    }
                )

    for side, count in registry["side"].fillna("unknown").value_counts().items():
        rows.append(
            {"category": "coverage_side", "metric": side, "value": float(count)}
        )

    if "timestamp" in data.columns:
        hours = pd.to_datetime(data["timestamp"]).dt.hour.fillna(-1)
        for key, mask in mask_map.items():
            counts = hours[mask].value_counts(normalize=True)
            for hour, share in counts.items():
                rows.append(
                    {
                        "category": "coverage_hour",
                        "metric": key,
                        "item_a": int(hour),
                        "value": float(share),
                    }
                )

    regime_family_counts = collections.Counter()
    for key in keys:
        for feature_name in extract_feature_names_from_key(key):
            if feature_name.startswith("reg_"):
                regime_family = (
                    feature_name.split("_")[1] if "_" in feature_name else feature_name
                )
                regime_family_counts[regime_family] += 1
    for family, count in regime_family_counts.items():
        rows.append(
            {
                "category": "coverage_regime_family",
                "metric": family,
                "value": float(count),
            }
        )

    return pd.DataFrame(rows)


def apply_label_step_sliceplanner_filter(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    events = pd.DataFrame(
        {
            "event_id": np.arange(len(data), dtype=np.int64),
            "symbol": data["symbol"].astype(object).to_numpy(copy=False),
            "t0": pd.to_datetime(data["timestamp"], utc=True, errors="coerce"),
            "t1": pd.to_datetime(data["timestamp"], utc=True, errors="coerce"),
        }
    )

    planner_cfg = build_mining_sliceplanner_config(cfg)
    bundle = SlicePlanner(planner_cfg).build(events)

    train_indices: set[int] = set()
    for plan in bundle["consumer_plans"].get("regime_search", []):
        if plan.tag in {"fit_inner", "fit_outer", "predict_inner"}:
            train_indices.update(np.asarray(plan.fit_idx, dtype=np.int64).tolist())

    if not train_indices:
        return (
            data,
            feature_dict,
            fwd_ret,
            {
                "sliceplanner_applied": False,
                "reason": "no_training_indices",
                "rows_before": int(len(data)),
                "rows_after": int(len(data)),
                "symbols_before": int(data["symbol"].nunique()),
                "symbols_after": int(data["symbol"].nunique()),
            },
        )

    keep_idx = np.array(sorted(train_indices), dtype=np.int64)
    filtered_data = data.iloc[keep_idx].reset_index(drop=True)
    filtered_features = {
        name: np.asarray(values)[keep_idx] for name, values in feature_dict.items()
    }
    filtered_fwd_ret = np.asarray(fwd_ret)[keep_idx]

    metadata = {
        "sliceplanner_applied": True,
        "preset": planner_cfg.preset.preset_name,
        "rows_before": int(len(data)),
        "rows_after": int(len(filtered_data)),
        "symbols_before": int(data["symbol"].nunique()),
        "symbols_after": int(filtered_data["symbol"].nunique()),
        "row_fraction_kept": float(len(filtered_data) / max(len(data), 1)),
    }
    return filtered_data, filtered_features, filtered_fwd_ret, metadata


def build_mining_sliceplanner_config(
    cfg: Optional[Dict[str, Any]] = None,
) -> SlicePlannerConfig:
    planner_cfg = SlicePlannerConfig.fast_defaults(schema=EventSchema())
    cfg = cfg or {}
    outer_n_folds = cfg.get("sliceplanner_outer_n_folds")
    if outer_n_folds is not None:
        planner_cfg = replace(
            planner_cfg,
            preset=replace(
                planner_cfg.preset,
                outer=replace(planner_cfg.preset.outer, n_folds=int(outer_n_folds)),
            ),
        )
    return planner_cfg


def estimate_pretrim_start_ts(
    end_ts: pd.Timestamp,
    cfg: Dict[str, Any],
) -> pd.Timestamp:
    planner_cfg = build_mining_sliceplanner_config(cfg)
    outer = planner_cfg.preset.outer
    outer_folds = int(outer.n_folds or cfg.get("sliceplanner_outer_n_folds", 8) or 8)
    warmup_days = int(cfg.get("sliceplanner_warmup_days", 90))
    total_span = (
        outer.train_span
        + (outer.valid_span or pd.Timedelta(0))
        + outer.test_span
        + outer.step_span * max(outer_folds - 1, 0)
        + pd.Timedelta(days=warmup_days)
    )
    return end_ts - total_span


def build_label_step_sliceplanner_keep_idx(
    timestamps: pd.Index,
    symbols: pd.Index,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    ts_arr = np.repeat(timestamps.to_numpy(), len(symbols))
    events = pd.DataFrame(
        {
            "event_id": np.arange(len(ts_arr), dtype=np.int64),
            "symbol": np.tile(symbols.to_numpy(dtype=object), len(timestamps)),
            "t0": pd.to_datetime(ts_arr, utc=True, errors="coerce"),
            "t1": pd.to_datetime(ts_arr, utc=True, errors="coerce"),
        }
    )

    planner_cfg = build_mining_sliceplanner_config(cfg)
    bundle = SlicePlanner(planner_cfg).build(events)
    clean_events = bundle["events"]

    train_event_ids: set[int] = set()
    for plan in bundle["consumer_plans"].get("regime_search", []):
        if plan.tag in {"fit_inner", "fit_outer", "predict_inner"}:
            fit_idx = np.asarray(plan.fit_idx, dtype=np.int64)
            if fit_idx.size == 0:
                continue
            train_event_ids.update(
                clean_events.iloc[fit_idx]["event_id"].to_numpy(dtype=np.int64).tolist()
            )

    total_rows = int(len(events))
    symbols_before = int(len(symbols))
    if not train_event_ids:
        keep_idx = np.arange(total_rows, dtype=np.int64)
        metadata = {
            "sliceplanner_applied": False,
            "reason": "no_training_indices",
            "preset": planner_cfg.preset.preset_name,
            "rows_before": total_rows,
            "rows_after": total_rows,
            "symbols_before": symbols_before,
            "symbols_after": symbols_before,
            "row_fraction_kept": 1.0,
        }
        return keep_idx, metadata

    keep_idx = np.array(sorted(train_event_ids), dtype=np.int64)
    kept_symbol_count = int(np.unique(keep_idx % max(len(symbols), 1)).size)
    metadata = {
        "sliceplanner_applied": True,
        "preset": planner_cfg.preset.preset_name,
        "rows_before": total_rows,
        "rows_after": int(len(keep_idx)),
        "symbols_before": symbols_before,
        "symbols_after": kept_symbol_count,
        "row_fraction_kept": float(len(keep_idx) / max(total_rows, 1)),
    }
    return keep_idx, metadata


def _extract_selected_wide_values(
    df: pd.DataFrame,
    common_idx: pd.Index,
    common_syms: pd.Index,
    time_idx: np.ndarray,
    sym_idx: np.ndarray,
    dtype: Optional[np.dtype] = np.float32,
) -> np.ndarray:
    aligned = df.reindex(index=common_idx, columns=common_syms)
    values = aligned.to_numpy()
    extracted = values[time_idx, sym_idx]
    if dtype is None:
        return extracted
    return extracted.astype(dtype, copy=False)


def apply_robust_data_filtering(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    target_nan_reasons: Optional[np.ndarray] = None,
    overlap_threshold: float = 0.8,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Apply robust filtering strategy:
    1. Filter out rows with no feature availability.
    2. Identify the feature with maximum availability as the 'reference'.
    3. Drop features that have less than 80% availability overlap with the reference.
    4. Drop only the specific (symbol, timestamp) row if it is missing any
       of the retained features.
    """
    n_rows_initial = len(data)
    if n_rows_initial == 0 or not feature_dict:
        return (
            data,
            feature_dict,
            fwd_ret,
            fwd_ret_norm,
            target_nan_reasons,
            {"dropped_features": [], "dropped_rows": 0},
        )

    # 1. Filter out rows with NO features available
    has_any_feature = np.zeros(n_rows_initial, dtype=bool)
    for v in feature_dict.values():
        has_any_feature |= np.isfinite(v)

    data = data.loc[has_any_feature].reset_index(drop=True)
    feature_dict = {k: v[has_any_feature] for k, v in feature_dict.items()}
    fwd_ret = fwd_ret[has_any_feature]
    fwd_ret_norm = fwd_ret_norm[has_any_feature]
    if target_nan_reasons is not None:
        target_nan_reasons = target_nan_reasons[has_any_feature]
    n_rows_any = len(data)

    # 2. Identify reference feature (max availability)
    availability = {k: np.isfinite(v).mean() for k, v in feature_dict.items()}
    ref_feat = max(availability, key=availability.get)
    ref_mask = np.isfinite(feature_dict[ref_feat])

    # 3. Prune features based on overlap with reference
    retained_features = {}
    dropped_features = []

    features_with_binary_values = 0
    total_features_initial = len(feature_dict)

    for k, v in feature_dict.items():
        overlap = np.isfinite(v[ref_mask]).mean()
        if overlap >= overlap_threshold:
            retained_features[k] = v

            # Heuristic to check if feature has only binary values (0/1/NaN)
            unique_vals = np.unique(v[np.isfinite(v)])
            if set(unique_vals).issubset({0.0, 1.0}):
                features_with_binary_values += 1
        else:
            dropped_features.append((k, overlap))

    # 4. Final row pruning: fully available for the specific symbol/timestamp row
    all_finite = np.ones(len(data), dtype=bool)
    for v in retained_features.values():
        all_finite &= np.isfinite(v)

    # Filtering is necessary, but we only drop the specific (symbol, timestamp) row
    final_keep_mask = all_finite

    data_final = data.loc[final_keep_mask].reset_index(drop=True)
    features_final = {
        k: retained_features[k][final_keep_mask] for k in retained_features
    }

    # Check if there are any NaNs left after filtering
    nan_features_detected = []
    for k, v in features_final.items():
        if np.isnan(v).any():
            nan_features_detected.append(k)
    fwd_ret_final = fwd_ret[final_keep_mask]
    fwd_ret_norm_final = fwd_ret_norm[final_keep_mask]
    target_nan_reasons_final = target_nan_reasons[final_keep_mask] if target_nan_reasons is not None else None

    meta = {
        "rows_initial": n_rows_initial,
        "rows_after_any_feat": n_rows_any,
        "rows_final": len(data_final),
        "dropped_rows": n_rows_initial - len(data_final),
        "reference_feature": ref_feat,
        "dropped_features": dropped_features,
        "retained_count": len(retained_features),
        "total_features_initial": total_features_initial,
        "features_with_binary_values": features_with_binary_values,
        "nan_features_detected": nan_features_detected,
    }

    return data_final, features_final, fwd_ret_final, fwd_ret_norm_final, target_nan_reasons_final, meta


def filter_complete_feature_rows(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Retain only symbol-timestamp rows where every extracted feature value is finite.

    Each entry in ``feature_dict`` is expected to already be reduced to the selected
    event rows via ``feat_values[time_idx, compact_sym_idx]``. Missing values for
    other symbols elsewhere in the universe must not affect a retained row for the
    current symbol/timestamp.
    """
    n_rows = len(data)
    if n_rows == 0 or not feature_dict:
        return (
            data,
            feature_dict,
            fwd_ret,
            fwd_ret_norm,
            {
                "rows_before": int(n_rows),
                "rows_after": int(n_rows),
                "dropped_rows": 0,
                "drop_fraction": 0.0,
                "worst_features": [],
                "worst_symbols": [],
            },
        )

    keep_mask = np.ones(n_rows, dtype=bool)
    missing_counts: List[Tuple[str, int]] = []
    for name, values in feature_dict.items():
        arr = np.asarray(values)
        finite_mask = np.isfinite(arr)
        if arr.ndim > 1:
            finite_mask = np.all(finite_mask, axis=1)
        keep_mask &= finite_mask
        missing_counts.append((str(name), int((~finite_mask).sum())))

    filtered_data = data.loc[keep_mask].reset_index(drop=True)
    filtered_features = {
        name: np.asarray(values)[keep_mask] for name, values in feature_dict.items()
    }
    filtered_fwd_ret = np.asarray(fwd_ret)[keep_mask]
    filtered_fwd_ret_norm = np.asarray(fwd_ret_norm)[keep_mask]
    missing_counts.sort(key=lambda item: item[1], reverse=True)
    dropped_rows = int((~keep_mask).sum())
    symbol_drop_counts: List[Tuple[str, int]] = []
    if "symbol" in data.columns:
        dropped_by_symbol = (
            data.loc[~keep_mask, "symbol"]
            .astype(str)
            .value_counts()
            .sort_values(ascending=False)
        )
        symbol_drop_counts = [
            (str(symbol), int(count))
            for symbol, count in dropped_by_symbol.head(10).items()
        ]
    meta = {
        "rows_before": int(n_rows),
        "rows_after": int(len(filtered_data)),
        "dropped_rows": dropped_rows,
        "drop_fraction": float(dropped_rows / max(n_rows, 1)),
        "worst_features": missing_counts[:10],
        "worst_symbols": symbol_drop_counts,
    }
    return (
        filtered_data,
        filtered_features,
        filtered_fwd_ret,
        filtered_fwd_ret_norm,
        meta,
    )


@njit(parallel=True, cache=True, fastmath=True)
def _compute_atr_wide_numba(
    high_wide: np.ndarray,
    low_wide: np.ndarray,
    close_wide: np.ndarray,
    atr_period: int,
) -> np.ndarray:
    """Numba-optimized ATR computation for multiple symbols."""
    n_ts, n_syms = high_wide.shape
    atr_wide = np.zeros((n_ts, n_syms), dtype=np.float32)

    for sym_idx in prange(n_syms):
        high_sym = high_wide[:, sym_idx]
        low_sym = low_wide[:, sym_idx]
        close_sym = close_wide[:, sym_idx]

        tr = np.zeros(n_ts, dtype=np.float32)
        if n_ts > 1:
            # Vectorized True Range computation
            for i in range(1, n_ts):
                tr[i] = max(
                    high_sym[i] - low_sym[i],
                    max(
                        abs(high_sym[i] - close_sym[i - 1]),
                        abs(low_sym[i] - close_sym[i - 1]),
                    ),
                )

        if n_ts > atr_period:
            atr_sym = np.zeros(n_ts, dtype=np.float32)
            # Initialize with mean of first atr_period values
            atr_sym[:atr_period] = np.mean(tr[:atr_period])
            # EWMA computation
            for i in range(atr_period, n_ts):
                atr_sym[i] = (atr_sym[i - 1] * (atr_period - 1) + tr[i]) / atr_period
        else:
            fallback = np.mean(tr[1:]) if n_ts > 1 else 0.001
            atr_sym = np.full(n_ts, fallback, dtype=np.float32)

        atr_wide[:, sym_idx] = atr_sym

    return atr_wide


def compute_atr_wide(
    high_wide: np.ndarray,
    low_wide: np.ndarray,
    close_wide: np.ndarray,
    atr_period: int = 14,
) -> np.ndarray:
    """
    Compute Average True Range (ATR) for multiple symbols in parallel.

    Optimized with Numba for 10-30x speedup.
    """
    # Use Numba-optimized implementation
    return _compute_atr_wide_numba(high_wide, low_wide, close_wide, atr_period)


def summarize_feature_usage(
    df: pd.DataFrame, output_path: Path, groupby_cols: List[str]
) -> None:
    """Summarize feature usage by specified columns."""
    if df.empty:
        pd.DataFrame(columns=groupby_cols + ["usage_count"]).to_csv(
            output_path, index=False
        )
        return

    summary = df.groupby(groupby_cols).size().reset_index(name="usage_count")
    summary = summary.sort_values("usage_count", ascending=False)
    summary.to_csv(output_path, index=False)


def export_coverage_sanity_report(
    metadata: List[FeatureMetadata],
    split_usage_all: pd.DataFrame,
    rule_usage_df: pd.DataFrame,
    final_usage_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Export a comprehensive coverage sanity report."""
    all_features = pd.DataFrame(
        [
            {
                "feature_name": m.feature_name,
                "group": m.group,
                "source_name": m.source_name,
                "source_family": m.source_family,
            }
            for m in metadata
        ]
    )

    split_counts = (
        split_usage_all.groupby("feature_name")["split_count"]
        .sum()
        .reset_index()
        .rename(columns={"split_count": "model_split_count"})
        if not split_usage_all.empty
        else pd.DataFrame(columns=["feature_name", "model_split_count"])
    )

    extracted_counts = (
        rule_usage_df.groupby("feature_name")
        .size()
        .reset_index(name="extracted_rule_count")
        if not rule_usage_df.empty
        else pd.DataFrame(columns=["feature_name", "extracted_rule_count"])
    )

    final_counts = (
        final_usage_df.groupby("feature_name")
        .size()
        .reset_index(name="final_registry_count")
        if not final_usage_df.empty
        else pd.DataFrame(columns=["feature_name", "final_registry_count"])
    )

    report = (
        all_features.merge(split_counts, on="feature_name", how="left")
        .merge(extracted_counts, on="feature_name", how="left")
        .merge(final_counts, on="feature_name", how="left")
        .fillna(0)
    )

    for c in ["model_split_count", "extracted_rule_count", "final_registry_count"]:
        report[c] = report[c].astype(int)

    report["used_in_model"] = report["model_split_count"] > 0
    report["used_in_extracted_rules"] = report["extracted_rule_count"] > 0
    report["used_in_final_registry"] = report["final_registry_count"] > 0

    report.to_csv(output_dir / "feature_coverage_sanity_report.csv", index=False)

    summary = (
        report.groupby("group")[
            ["used_in_model", "used_in_extracted_rules", "used_in_final_registry"]
        ]
        .sum()
        .reset_index()
    )
    summary.to_csv(
        output_dir / "final_registry_feature_usage_by_group.csv", index=False
    )

    return report


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================


def apply_cfg_preset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    preset = str(cfg.get("preset", "exploration")).lower()
    out = dict(cfg)
    defaults = {
        "exploration": {
            "min_feature_support": 2,
            "min_support_count_validation": 5,
            "min_tree_discoveries": 1,
            "min_presence_freq": 0.33,
            "min_sign_consistency": 0.0,
            "support_min_pct": SUPPORT_MIN,
            "support_max_pct": SUPPORT_MAX,
            "objective_support_min_pct": SUPPORT_MIN,
            "objective_support_target_low_pct": PREFERRED_SUPPORT_MIN,
            "objective_support_target_high_pct": PREFERRED_SUPPORT_MAX,
            "objective_support_max_pct": SUPPORT_MAX,
            "objective_support_edge_floor": 0.2,
            "prune_base_hurdle": 0.00005,
            "prune_target_support_pct": TARGET_SUPPORT,
            "prune_support_exp": 0.5,
            "min_context_support_pct": 0.005,
            "min_context_mean_ret": 0.0,
            "cv_min_train_frac": 0.5,
            "cv_embargo": 0,
            "max_support_pct": SUPPORT_MAX,
            "stage_a_directional": True,
            "stage_a_relax_positive_groups": True,
            "residualise_target_for_miner": True,
            "miner_target_residualization_columns": list(
                MINER_TARGET_RESIDUALIZATION_COLUMNS
            ),
            "drop_nuisance_features_from_miner": True,
            "drop_continuous_nuisance_parents_from_miner": True,
            "drop_location_nuisance_features_from_miner": False,
            "global_ridge_candidate_cap": 120,
        },
        "production": {
            "min_feature_support": 5,
            "min_support_count_validation": 10,
            "min_tree_discoveries": 2,
            "min_presence_freq": 0.4,
            "support_min_pct": SUPPORT_MIN,
            "support_max_pct": SUPPORT_MAX,
            "objective_support_min_pct": SUPPORT_MIN,
            "objective_support_target_low_pct": PREFERRED_SUPPORT_MIN,
            "objective_support_target_high_pct": PREFERRED_SUPPORT_MAX,
            "objective_support_max_pct": SUPPORT_MAX,
            "objective_support_edge_floor": 0.2,
            "prune_base_hurdle": 0.00010,
            "prune_target_support_pct": TARGET_SUPPORT,
            "prune_support_exp": 0.5,
            "min_context_support_pct": 0.01,
            "min_context_mean_ret": 0.0,
            "cv_min_train_frac": 0.5,
            "cv_embargo": 0,
            "max_support_pct": SUPPORT_MAX,
            "stage_a_directional": True,
            "stage_a_relax_positive_groups": True,
            "residualise_target_for_miner": True,
            "miner_target_residualization_columns": list(
                MINER_TARGET_RESIDUALIZATION_COLUMNS
            ),
            "drop_nuisance_features_from_miner": True,
            "drop_continuous_nuisance_parents_from_miner": True,
            "drop_location_nuisance_features_from_miner": False,
            "global_ridge_candidate_cap": 120,
        },
    }
    if preset not in defaults:
        raise ValueError(f"Unknown preset {preset}")
    for key, value in defaults[preset].items():
        out.setdefault(key, value)
    out["preset"] = preset
    out.setdefault(
        "prune_complexity_bonus_map",
        {"1": 0.0, "2": 0.15, "3": 0.30, "4": 0.10, "5": 0.10, "6": 0.10},
    )
    out.setdefault("n_folds", 5)
    out.setdefault("pairwise_top_n", 20)
    return out


def apply_test_mode(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a lightweight deterministic test-mode profile.

    Test mode is intended for quicker end-to-end smoke runs:
    - 3 folds
    - 300 symbols
    - 4 years of lookback
    """
    out = dict(cfg)
    out["n_folds"] = 3
    out["sliceplanner_outer_n_folds"] = 3
    out["mask_opt_max_symbols"] = 400
    out["mask_opt_lookback_years"] = 4.0
    out["support_max_pct"] = 0.22
    out["objective_support_max_pct"] = 0.22
    out["max_support_pct"] = 0.22
    out["test_mode"] = True
    return out


def apply_smoke_test_mode(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a very lightweight smoke-test profile.
    - 3 folds
    - 100 symbols
    - 2 years of lookback
    """
    out = dict(cfg)
    out["n_folds"] = 3
    out["sliceplanner_outer_n_folds"] = 3
    out["mask_opt_max_symbols"] = 100
    out["mask_opt_lookback_years"] = 2.0
    out["support_max_pct"] = 0.22
    out["objective_support_max_pct"] = 0.22
    out["max_support_pct"] = 0.22
    out["test_mode"] = True
    return out


def run_lgbm_mask_generation(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    cfg: Dict[str, Any],
):
    cfg = apply_cfg_preset(cfg)
    output_dir = Path(cfg.get("output_dir", "./lgbm_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    nuisance_feature_resolution: Dict[str, str] = {}
    nuisance_feature_arrays: Dict[str, np.ndarray] = {}
    if bool(cfg.get("residualise_target_for_miner", True)):
        nuisance_feature_resolution, nuisance_feature_arrays = (
            _resolve_miner_nuisance_feature_arrays(feature_dict, cfg)
        )
    fp = FeatureProcessor()
    X, metadata, audits = fp.prepare_features(
        feature_dict, data["timestamp"].to_numpy(), data["symbol"].to_numpy(), cfg
    )
    for k, v in audits.items():
        if not v.empty:
            v.to_csv(output_dir / f"{k}.csv", index=False)
    export_feature_group_summary(metadata, output_dir)
    result = run_mining_stage(
        data=data,
        fwd_ret=fwd_ret,
        fwd_ret_norm=fwd_ret_norm,
        X=X,
        metadata=metadata,
        cfg=cfg,
        output_dir=output_dir,
        stage_name="single_stage",
        allowed_group_pairs=(
            ("trigger", "location"),
            ("trigger", "regime"),
            ("location", "regime"),
        ),
        slot_order=("trigger", "location", "regime"),
        folds=build_walk_forward_folds(
            n_samples=len(data),
            n_folds=int(cfg.get("n_folds", 5)),
            min_train_frac=float(cfg.get("cv_min_train_frac", 0.5)),
            embargo=int(cfg.get("cv_embargo", 0)),
        ),
        mask_resolver=CanonicalRuleMaskResolver(X, metadata),
        pipeline_stage_name="single_stage",
        nuisance_feature_arrays=nuisance_feature_arrays,
        nuisance_feature_resolution=nuisance_feature_resolution,
    )
    return result["accepted_registry"]


def _flatten_wide_frame(
    df: pd.DataFrame, common_idx: pd.Index, common_syms: pd.Index
) -> np.ndarray:
    return df.reindex(index=common_idx, columns=common_syms).to_numpy().flatten()


def list_preload_training_symbols(
    store: PartitionedOHLCVStore,
    cfg: Dict[str, Any],
    max_symbols: int = 0,
) -> List[str]:
    """Return the same training universe used by the label step, before heavy data loading."""
    train_symbols = get_training_universe(None, cfg, store, ts_sig=None)
    if max_symbols > 0:
        return list(train_symbols[:max_symbols])
    return list(train_symbols)


# =============================================================================
# TARGET QUALITY METRICS
# =============================================================================


def compute_target_quality_metrics(
    target: np.ndarray,
    target_name: str,
    horizon: int,
    predictions: Optional[np.ndarray] = None,
    fold_ics: Optional[List[float]] = None,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive target quality metrics.

    Args:
        target: Target values array
        target_name: Name of the target for logging
        horizon: Horizon identifier for logging
        predictions: Optional model predictions for IC computation
        fold_ics: Optional list of fold-level ICs
        mask: Optional mask for within-mask metrics

    Returns:
        Dict with target quality metrics including:
        - target_name, horizon
        - target_mean, target_std, target_p05, p25, p50, p75, p95
        - target_nonzero_fraction
        - target_top_decile_support
        - target_autocorr_1
        - mean_oos_ic, p25_oos_ic, p50_oos_ic, p75_oos_ic, positive_ic_fraction
        - entropy_reduction (if predictions and mask available)
        - within_mask_ic, delta_within_mask_ic (if predictions and mask available)
    """
    valid = np.isfinite(target)
    target_valid = target[valid]

    result: Dict[str, Any] = {
        "target_name": target_name,
        "horizon": horizon,
    }

    if len(target_valid) < 10:
        result.update(
            {
                "target_mean": np.nan,
                "target_std": np.nan,
                "target_p05": np.nan,
                "target_p25": np.nan,
                "target_p50": np.nan,
                "target_p75": np.nan,
                "target_p95": np.nan,
                "target_nonzero_fraction": np.nan,
                "target_top_decile_support": np.nan,
                "target_autocorr_1": np.nan,
                "mean_oos_ic": np.nan,
                "p25_oos_ic": np.nan,
                "p50_oos_ic": np.nan,
                "p75_oos_ic": np.nan,
                "positive_ic_fraction": np.nan,
                "entropy_reduction": np.nan,
                "within_mask_ic": np.nan,
                "delta_within_mask_ic": np.nan,
            }
        )
        return result

    # Basic statistics
    result["target_mean"] = float(np.mean(target_valid))
    result["target_std"] = float(np.std(target_valid))
    result["target_p05"] = float(np.percentile(target_valid, 5))
    result["target_p25"] = float(np.percentile(target_valid, 25))
    result["target_p50"] = float(np.percentile(target_valid, 50))
    result["target_p75"] = float(np.percentile(target_valid, 75))
    result["target_p95"] = float(np.percentile(target_valid, 95))

    # Non-zero fraction
    nonzero_count = np.sum(target_valid != 0)
    result["target_nonzero_fraction"] = float(nonzero_count / len(target_valid))

    # Top decile support (fraction of values in top 10%)
    top_decile_threshold = np.percentile(np.abs(target_valid), 90)
    top_decile_count = np.sum(np.abs(target_valid) >= top_decile_threshold)
    result["target_top_decile_support"] = float(top_decile_count / len(target_valid))

    # Autocorrelation at lag 1
    if len(target_valid) > 10:
        target_centered = target_valid - np.mean(target_valid)
        autocorr_1 = np.corrcoef(target_centered[:-1], target_centered[1:])[0, 1]
        result["target_autocorr_1"] = (
            float(autocorr_1) if np.isfinite(autocorr_1) else np.nan
        )
    else:
        result["target_autocorr_1"] = np.nan

    # OOS IC metrics from fold ICs
    if fold_ics is not None and len(fold_ics) > 0:
        valid_ics = [ic for ic in fold_ics if np.isfinite(ic)]
        if valid_ics:
            result["mean_oos_ic"] = float(np.mean(valid_ics))
            result["p25_oos_ic"] = float(np.percentile(valid_ics, 25))
            result["p50_oos_ic"] = float(np.percentile(valid_ics, 50))
            result["p75_oos_ic"] = float(np.percentile(valid_ics, 75))
            result["positive_ic_fraction"] = float(
                sum(1 for ic in valid_ics if ic > 0) / len(valid_ics)
            )
        else:
            result["mean_oos_ic"] = np.nan
            result["p25_oos_ic"] = np.nan
            result["p50_oos_ic"] = np.nan
            result["p75_oos_ic"] = np.nan
            result["positive_ic_fraction"] = np.nan
    else:
        result["mean_oos_ic"] = np.nan
        result["p25_oos_ic"] = np.nan
        result["p50_oos_ic"] = np.nan
        result["p75_oos_ic"] = np.nan
        result["positive_ic_fraction"] = np.nan

    # Entropy reduction and within-mask IC (if predictions and mask available)
    if predictions is not None and mask is not None:
        # Compute entropy reduction proxy
        if len(target_valid) >= 100:
            mask_active = mask.astype(bool) & valid
            if mask_active.sum() >= 50:
                target_masked = target[mask_active]
                entropy_global = np.log(np.std(target_valid) + 1e-9)
                entropy_masked = np.log(np.std(target_masked) + 1e-9)
                result["entropy_reduction"] = float(entropy_global - entropy_masked)
            else:
                result["entropy_reduction"] = np.nan
        else:
            result["entropy_reduction"] = np.nan

        # Compute within-mask IC
        pred_valid = ~(np.isnan(predictions) | np.isnan(target))
        if pred_valid.sum() >= 10:
            global_ic = _safe_spearman(predictions[pred_valid], target[pred_valid])

            mask_active = mask.astype(bool) & pred_valid
            if mask_active.sum() >= 10:
                within_ic = _safe_spearman(
                    predictions[mask_active], target[mask_active]
                )
                result["within_mask_ic"] = within_ic
                result["delta_within_mask_ic"] = (
                    within_ic - global_ic if np.isfinite(within_ic) else np.nan
                )
            else:
                result["within_mask_ic"] = np.nan
                result["delta_within_mask_ic"] = np.nan
        else:
            result["within_mask_ic"] = np.nan
            result["delta_within_mask_ic"] = np.nan
    else:
        result["entropy_reduction"] = np.nan
        result["within_mask_ic"] = np.nan
        result["delta_within_mask_ic"] = np.nan

    return result


def compute_cross_target_correlation(
    targets: Dict[str, np.ndarray],
    horizon: int,
    correlation_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Compute correlation matrix between targets at same horizon.

    Args:
        targets: Dict mapping target_name -> target_array
        horizon: Horizon identifier for logging
        correlation_threshold: Threshold for flagging high correlations

    Returns:
        Dict with:
        - correlation_matrix: Dict[(target_a, target_b)] -> correlation
        - high_correlation_pairs: List of pairs with corr > threshold
        - quality_flags: List of any issues
    """
    result: Dict[str, Any] = {
        "horizon": horizon,
        "correlation_matrix": {},
        "high_correlation_pairs": [],
        "quality_flags": [],
    }

    target_names = list(targets.keys())
    n_targets = len(target_names)

    if n_targets < 2:
        result["quality_flags"].append("insufficient_targets_for_correlation")
        return result

    target_series = {
        name: pd.Series(np.asarray(values, dtype=np.float32).reshape(-1))
        for name, values in targets.items()
    }
    target_df = pd.DataFrame(target_series)
    if target_df.shape[1] < 2:
        result["quality_flags"].append("insufficient_targets_for_correlation")
        return result

    ranked_df = target_df.rank(method="average", na_option="keep")
    corr_df = ranked_df.corr(method="pearson", min_periods=10)

    for i in range(n_targets):
        for j in range(i + 1, n_targets):
            name_a = target_names[i]
            name_b = target_names[j]
            corr = float(corr_df.at[name_a, name_b])
            if not np.isfinite(corr):
                continue
            result["correlation_matrix"][(name_a, name_b)] = corr
            if abs(corr) > correlation_threshold:
                result["high_correlation_pairs"].append(
                    {"pair": (name_a, name_b), "correlation": corr}
                )

    # Add quality flags for high correlations
    if result["high_correlation_pairs"]:
        result["quality_flags"].append(
            f"high_correlation_count_{len(result['high_correlation_pairs'])}"
        )

    return result


def create_target_quality_summary(
    all_results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create summary table of target quality across all targets and horizons.

    Args:
        all_results: Results from all (target, horizon) runs, where each dict
                     contains rule summaries with target quality metrics
        output_path: Optional path to save the summary CSV

    Returns:
        DataFrame with columns:
        - target_name
        - horizon
        - target_mean, target_std, target_p05, p25, p50, p75, p95
        - mean_oos_ic, p25_oos_ic, p50_oos_ic, p75_oos_ic
        - positive_ic_fraction
        - entropy_reduction
        - mean_delta_ic
        - rule_count
        - structurally_sound_rule_count
        - production_rule_count
        - median_support
        - cross_target_corrs (as dict or JSON string)
        - overall_target_quality_score
    """
    if not all_results:
        return pd.DataFrame()

    # Concatenate all results into a single DataFrame for vectorized aggregation
    all_df = (
        pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    )
    if all_df.empty:
        return pd.DataFrame()

    # Pre-calculating boolean flags for counts
    if "accepted" in all_df.columns:
        all_df["is_accepted"] = all_df["accepted"].fillna(False).astype(bool)
    else:
        all_df["is_accepted"] = False

    if "composite_score" in all_df.columns:
        all_df["is_production"] = all_df["is_accepted"] & (all_df["composite_score"] > 0)
    else:
        all_df["is_production"] = False

    # Aggregation map for vectorized summary
    agg_map = {
        "target_mean": "mean",
        "target_std": "mean",
        "mean_oos_ic": "mean",
        "p25_oos_ic": "mean",
        "p50_oos_ic": "mean",
        "p75_oos_ic": "mean",
        "positive_ic_fraction": "mean",
        "entropy_reduction": "mean",
        "mean_delta_ic": "mean",
        "min_support_actual": "median",
        "is_accepted": "sum",
        "is_production": "sum",
    }

    # Only include keys that exist in the columns
    actual_agg_map = {k: v for k, v in agg_map.items() if k in all_df.columns}

    summary_df = (
        all_df.groupby(["source_target", "source_horizon"], as_index=False)
        .agg(actual_agg_map)
        .rename(
            columns={
                "source_target": "target_name",
                "source_horizon": "horizon",
                "is_accepted": "structurally_sound_rule_count",
                "is_production": "production_rule_count",
                "min_support_actual": "median_support",
            }
        )
    )

    # Compute rule_count separately
    counts_df = (
        all_df.groupby(["source_target", "source_horizon"])
        .size()
        .reset_index(name="rule_count")
    )
    summary_df = pd.merge(
        summary_df,
        counts_df.rename(
            columns={"source_target": "target_name", "source_horizon": "horizon"}
        ),
        on=["target_name", "horizon"],
    )

    # Compute overall target quality score (Vectorized)
    summary_df["overall_target_quality_score"] = 0.0
    quality_components = pd.Series(0, index=summary_df.index)

    if "mean_oos_ic" in summary_df.columns:
        mask = summary_df["mean_oos_ic"].notna()
        summary_df.loc[mask, "overall_target_quality_score"] += (
            summary_df.loc[mask, "mean_oos_ic"] * 10
        )
        quality_components[mask] += 1

    if "positive_ic_fraction" in summary_df.columns:
        mask = summary_df["positive_ic_fraction"].notna()
        summary_df.loc[mask, "overall_target_quality_score"] += (
            summary_df.loc[mask, "positive_ic_fraction"] * 0.5
        )
        quality_components[mask] += 1

    if "entropy_reduction" in summary_df.columns:
        mask = summary_df["entropy_reduction"].notna()
        summary_df.loc[mask, "overall_target_quality_score"] += (
            summary_df.loc[mask, "entropy_reduction"].clip(lower=0) * 2
        )
        quality_components[mask] += 1

    if "mean_delta_ic" in summary_df.columns:
        mask = summary_df["mean_delta_ic"].notna()
        summary_df.loc[mask, "overall_target_quality_score"] += (
            summary_df.loc[mask, "mean_delta_ic"].clip(lower=0) * 5
        )
        quality_components[mask] += 1

    summary_df["overall_target_quality_score"] /= quality_components.clip(lower=1)
    # Sort by target name and horizon
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["target_name", "horizon"])

    # Save to file if path provided
    if output_path and not summary_df.empty:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        tprint(f"Target quality summary saved to {output_path}")

    return summary_df


if __name__ == "__main__":
    import argparse
    import glob

    from extreme_price_movements.config import CFG
    from extreme_price_movements.data_store import (
        PartitionedOHLCVStore,
        load_features_selected,
        to_panel,
    )
    from extreme_price_movements.pipeline_steps import _feature_snapshot_health_issues

    parser = argparse.ArgumentParser(description="Full LGBM Mask Generation Run")
    parser.add_argument(
        "--data-root",
        default="/Users/remyroche/Documents/Ares/data",
        help="Data root path",
    )
    parser.add_argument("--feature-path", help="Optional feature path override")
    parser.add_argument(
        "--lookback-years",
        type=float,
        default=0.0,
        help="Years of data to load before SlicePlanner filtering; 0 means no manual limit",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="Max symbols to load before SlicePlanner filtering; 0 means no manual limit",
    )
    parser.add_argument(
        "--output-dir", default="./production_lgbm_outputs", help="Output directory"
    )
    parser.add_argument(
        "--preset",
        choices=["exploration", "production"],
        default="production",
        help="Threshold preset",
    )
    parser.add_argument(
        "--boolean-only",
        action="store_true",
        help="Only use boolean features (triggers)",
    )
    parser.add_argument(
        "--use-dynamic-hpo",
        action="store_true",
        help="Run dynamic HPO for Stage A",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run the smaller configuration (300 symbols, 3y)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the tiny configuration (100 symbols, 2y)",
    )
    parser.add_argument(
        "--run-step",
        choices=["full", "step1", "step2"],
        default="full",
        help="Run the full pipeline, stop after Stage A dedup (step1), or resume from a stored step1 outcome (step2)",
    )
    parser.add_argument(
        "--step1-dir",
        help="Previous run root or specific stage_a_context directory to use as step1 input for --run-step step2",
    )
    parser.add_argument(
        "--triad-horizons",
        type=str,
        default="5,10",
        help="Comma-separated list of horizons for triad targets (default: 5,10)",
    )
    args = parser.parse_args()

    cfg = dict(CFG)
    cfg["data_root"] = args.data_root
    cfg["output_dir"] = args.output_dir
    cfg["preset"] = args.preset
    cfg["run_step"] = args.run_step
    if args.step1_dir:
        cfg["step1_dir"] = args.step1_dir
    cfg.setdefault("sliceplanner_outer_n_folds", 8)
    cfg.setdefault("sliceplanner_warmup_days", 90)
    
    # Adaptive TP/SL configuration
    cfg.setdefault("adaptive_tp_sl_enabled", True)
    cfg.setdefault("adaptive_tp_sl_conf_levels", [0.8])
    cfg.setdefault("adaptive_tp_sl_grid", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    cfg.setdefault("adaptive_tp_sl_sl_ratio_grid", [0.3, 0.5, 0.7, 0.9])
    
    if args.test_mode or args.smoke_test:
        if args.smoke_test:
            cfg = apply_smoke_test_mode(cfg)
        else:
            cfg = apply_test_mode(cfg)

        if not args.max_symbols:
            args.max_symbols = int(cfg.get("mask_opt_max_symbols", 30))
        if not args.lookback_years:
            args.lookback_years = float(cfg.get("mask_opt_lookback_years", 2.0))

    # Triad target configuration (always use triad targets)
    cfg["use_triad_targets"] = True
    if args.triad_horizons:
        cfg["triad_horizons"] = [int(h.strip()) for h in args.triad_horizons.split(",")]
    else:
        cfg["triad_horizons"] = TRIAD_DEFAULT_HORIZONS
    cfg["triad_target_names"] = TRIAD_DEFAULT_TARGET_NAMES

    cfg = apply_cfg_preset(cfg)

    root_output_dir = build_run_output_dir(cfg)
    tprint(
        f"LGBM Full Run: root={args.data_root} | lookback={args.lookback_years}y | symbols={args.max_symbols} | run_step={args.run_step}"
    )
    tprint(f"LGBM Output Dir: {root_output_dir}")
    if args.step1_dir:
        tprint(f"Step1 input dir: {args.step1_dir}")

    # 1. Data Store & Symbols
    store = PartitionedOHLCVStore(
        root_dir=cfg["data_root"], timeframe=cfg.get("timeframe", "1h")
    )
    symbols = list_preload_training_symbols(store, cfg, max_symbols=args.max_symbols)
    tprint(f"Selected {len(symbols)} pre-load training-universe symbols")
    feature_dir = os.path.join(cfg["data_root"], "features")
    feature_files = sorted(glob.glob(os.path.join(feature_dir, "202[0-9]*")))
    feature_path = args.feature_path or (feature_files[-1] if feature_files else None)

    if not feature_path:
        feature_files = sorted(glob.glob("202[0-9]*"))
        feature_path = feature_files[-1] if feature_files else None

    if not feature_path:
        tprint("ERROR: No feature path found.")
        exit(1)

    ts_str = os.path.basename(feature_path)
    try:
        feature_snapshot_ts = pd.Timestamp(ts_str.replace("_", " "))
    except Exception:
        feature_snapshot_ts = pd.Timestamp.now(tz="UTC")
    if feature_snapshot_ts.tzinfo is None:
        feature_snapshot_ts = feature_snapshot_ts.tz_localize("UTC")

    start_ts = estimate_pretrim_start_ts(feature_snapshot_ts, cfg)
    if args.lookback_years and args.lookback_years > 0:
        # Floor start_ts to the hour to ensure alignment with hourly features
        manual_start = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(
            days=int(365.25 * args.lookback_years)
        )
        start_ts = max(start_ts, manual_start)
    tprint(
        f"Pre-trim start_ts={start_ts} derived from planner horizon (floored to hour)"
    )

    # 2. Load OHLCV
    dfs_by_symbol: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            df = store.load(s, start_ts=start_ts)
            if not df.empty:
                dfs_by_symbol[s] = df
        except Exception:
            continue

    if not dfs_by_symbol:
        tprint("ERROR: No data loaded.")
        exit(1)

    panel = to_panel(dfs_by_symbol)
    common_idx = panel["close"].index
    common_syms = panel["close"].columns

    # 3. Prepare planner-bounded indices before loading features
    fwd_hours = int(cfg.get("mask_opt_forward_hours", 5))
    fwd_ret_wide, fwd_ret_reasons_wide = generate_fwd_ret_with_reasons(panel, fwd_hours)

    common_idx = panel["close"].index
    common_syms = panel["close"].columns
    n_ts, n_syms = len(common_idx), len(common_syms)

    # TZ Normalization for alignment
    common_idx_naive = (
        common_idx.tz_localize(None) if common_idx.tz is not None else common_idx
    )

    tprint(
        f"Panel: {n_ts} timestamps x {n_syms} symbols. TZ: {common_idx.tz} -> Naive. Top syms: {list(common_syms[:3])}"
    )

    planner_filter_start = time.perf_counter()
    keep_idx, planner_filter_meta = build_label_step_sliceplanner_keep_idx(
        common_idx, common_syms, cfg=cfg
    )
    time_idx = (keep_idx // n_syms).astype(np.int32, copy=False)
    sym_idx = (keep_idx % n_syms).astype(np.int32, copy=False)
    kept_sym_positions = np.unique(sym_idx)
    kept_syms = common_syms.take(kept_sym_positions)
    compact_sym_idx = np.searchsorted(kept_sym_positions, sym_idx).astype(
        np.int32, copy=False
    )
    tprint(
        f"SlicePlanner keep-index build complete: kept_rows={len(keep_idx)} "
        f"in {time.perf_counter() - planner_filter_start:.1f}s"
    )
    global_row_funnel = []
    prev_rows = max(int(n_ts * n_syms), 1)
    for stage_name_funnel, rows_count, symbol_count in [
        ("raw_panel", int(n_ts * n_syms), int(n_syms)),
        ("sliceplanner_keep_idx", int(len(keep_idx)), int(len(kept_syms))),
    ]:
        global_row_funnel.append(
            {
                "stage": stage_name_funnel,
                "rows": rows_count,
                "symbols": symbol_count,
                "fraction_of_prev": float(rows_count / max(prev_rows, 1)),
            }
        )
        prev_rows = max(rows_count, 1)

    # 4. Load features only for planner-surviving symbols
    ts = feature_snapshot_ts

    tprint(
        f"Loading features from {feature_path} for {len(kept_syms)} planner-surviving symbols..."
    )
    requested_feature_keys = sorted(
        set(
            list(CFG.get("FEATURE_SELECTION_KEYS", []))
            + list(TEST_FEATURE_KEYS)
            + RIDGE_FEATURE_COLS
            + list(CONTINUOUS_LOCATION_COLS)
        )
    )
    tprint(
        f"Requested feature keys: {len(requested_feature_keys)} (TRIGGER features disabled)"
    )

    if args.boolean_only:
        cfg["boolean_only"] = True
        tprint(
            "MODE: boolean_only. Continuous features will be converted to booleans via thresholding."
        )
    if args.use_dynamic_hpo:
        cfg["use_dynamic_hpo"] = True
    feat_dict_raw = load_features_selected(
        ts=ts,
        root_dir=os.path.dirname(os.path.dirname(feature_path)),
        feature_keys=requested_feature_keys,
        symbols=list(map(str, kept_syms)),
        start_ts=start_ts,
    )
    if feat_dict_raw is None:
        feat_dict_raw = {}
    health_issues = _feature_snapshot_health_issues(feat_dict_raw)
    if health_issues:
        raise RuntimeError(
            "Loaded feature snapshot is unhealthy for miner consumption: "
            + ", ".join(health_issues)
            + ". Regenerate features through the feature pipeline first "
            "(e.g. `python3 -u extreme_price_movements/run_pipeline.py features "
            "--force-feature-recompute`)."
        )

    # Check for missing features - RIDGE_FEATURE_COLS and FEATURE_SELECTION_KEYS are required,
    # but CONTINUOUS_LOCATION_COLS are optional (they'll be skipped if missing)
    required_keys = set(
        list(CFG.get("FEATURE_SELECTION_KEYS", [])) + RIDGE_FEATURE_COLS
    )
    missing_required_keys = sorted(required_keys - set(feat_dict_raw))
    if missing_required_keys:
        tprint(
            f"WARNING: Feature snapshot incomplete. "
            f"Missing {len(missing_required_keys)} required keys: {missing_required_keys[:20]}"
        )
        tprint(
            "Continuing with available features only. Missing required keys will be "
            "excluded from this miner run."
        )
        test_feature_keys = set(CFG.get("TEST_FEATURE_KEYS", []))
        missing_test_keys = sorted(
            test_feature_keys.intersection(missing_required_keys)
        )
        if missing_test_keys:
            tprint(
                f"WARNING: {len(missing_test_keys)} TEST_FEATURE_KEYS are also missing: "
                f"{missing_test_keys[:20]}"
            )

    # Check for optional location features
    optional_location_keys = set(CONTINUOUS_LOCATION_COLS)
    missing_location_keys = sorted(optional_location_keys - set(feat_dict_raw))
    if missing_location_keys:
        tprint(
            f"WARNING: {len(missing_location_keys)} optional location features missing. "
            f"These will be skipped: {missing_location_keys[:10]}{'...' if len(missing_location_keys) > 10 else ''}"
        )

    # 5. Load features
    ts_arr = common_idx.to_numpy()[time_idx]
    ts_pd = pd.to_datetime(ts_arr, utc=True)
    symbol_arr = common_syms.to_numpy(dtype=object)[sym_idx]
    close_selected = _extract_selected_wide_values(
        panel["close"], common_idx, common_syms, time_idx, sym_idx
    )
    high_selected = _extract_selected_wide_values(
        panel["high"], common_idx, common_syms, time_idx, sym_idx
    )
    low_selected = _extract_selected_wide_values(
        panel["low"], common_idx, common_syms, time_idx, sym_idx
    )
    if "volume" not in panel:
        raise RuntimeError("volume column is required for triad target generation")
    volume_selected = _extract_selected_wide_values(
        panel["volume"], common_idx, common_syms, time_idx, sym_idx
    )
    if "open" in panel:
        open_selected = _extract_selected_wide_values(
            panel["open"], common_idx, common_syms, time_idx, sym_idx
        )
    else:
        open_selected = close_selected
    stack_start = time.perf_counter()
    data_final = pd.DataFrame(
        {
            "event_id": np.arange(len(keep_idx), dtype=np.int64),
            "timestamp": ts_arr,
            "symbol": symbol_arr,
            "close": close_selected,
            "high": high_selected,
            "low": low_selected,
            "volume": volume_selected,
            "t0": ts_pd.to_numpy(),
            "t1": (ts_pd + pd.Timedelta(seconds=1)).to_numpy(),
            "open": open_selected,
        }
    )
    tprint(
        f"Filtered event frame built: rows={len(data_final)} cols={len(data_final.columns)} "
        f"in {time.perf_counter() - stack_start:.1f}s"
    )
    global_row_funnel.append(
        {
            "stage": "event_frame_built",
            "rows": int(len(data_final)),
            "symbols": int(data_final["symbol"].nunique()),
            "fraction_of_prev": float(len(data_final) / max(prev_rows, 1)),
        }
    )
    prev_rows = max(int(len(data_final)), 1)

    atr_start = time.perf_counter()
    high_wide = (
        panel["high"]
        .reindex(index=common_idx, columns=common_syms)
        .to_numpy(dtype=np.float32)
    )
    low_wide = (
        panel["low"]
        .reindex(index=common_idx, columns=common_syms)
        .to_numpy(dtype=np.float32)
    )
    close_wide = (
        panel["close"]
        .reindex(index=common_idx, columns=common_syms)
        .to_numpy(dtype=np.float32)
    )
    atr_wide = compute_atr_wide(high_wide, low_wide, close_wide, atr_period=14)
    # Compute ATR as percentage of close price
    atr_pct_matrix = np.where(close_wide > 1e-9, atr_wide / close_wide, 0.0).astype(
        np.float32
    )
    data_final["atr"] = atr_wide[time_idx, sym_idx]
    tprint(
        f"ATR computed in wide form and extracted in {time.perf_counter() - atr_start:.1f}s"
    )

    fwd_ret_start = time.perf_counter()
    fwd_ret_matrix = fwd_ret_wide.reindex(
        index=common_idx, columns=common_syms
    ).to_numpy(dtype=np.float32)
    fwd_ret_reasons_matrix = fwd_ret_reasons_wide.reindex(
        index=common_idx, columns=common_syms
    ).to_numpy(dtype=object)

    target_signal = fwd_ret_matrix / np.maximum(np.sqrt(atr_pct_matrix), 1e-9)

    fwd_ret_norm_matrix = fwd_ret_matrix / np.maximum(atr_pct_matrix, 1e-9)
    fwd_ret_final = fwd_ret_matrix[time_idx, sym_idx]
    fwd_ret_norm_final = fwd_ret_norm_matrix[time_idx, sym_idx]
    fwd_ret_reasons_final = fwd_ret_reasons_matrix[time_idx, sym_idx]
    tprint(
        f"Forward returns extracted for kept rows in {time.perf_counter() - fwd_ret_start:.1f}s"
    )

    feature_align_start = time.perf_counter()
    feat_final: Dict[str, np.ndarray] = {}
    feature_items = list(feat_dict_raw.items())
    feature_log_every = max(1, len(feature_items) // 10)
    extraction_trace_features = {
        "atr_change_rate",
        "atr_percentile",
        "choppiness_index_20",
        "dist_ema20_atr",
        "loc_range_pos_24",
    }
    alignment_issue_samples: List[str] = []
    alignment_issue_count = 0
    for feat_idx, (k, df_feat) in enumerate(feature_items, start=1):
        if isinstance(df_feat, pd.DataFrame):
            feat_df = df_feat
            if (
                isinstance(feat_df.index, pd.DatetimeIndex)
                and feat_df.index.tz is not None
            ):
                feat_df = feat_df.tz_localize(None)

            if len(feat_final) == 0:
                overlap = common_idx_naive.intersection(feat_df.index)
                tprint(
                    f"Alignment Check: overlap={len(overlap)}/{len(common_idx_naive)}"
                )
                if len(overlap) == 0:
                    tprint(f"Panel Index Sample: {common_idx_naive[:2].tolist()}")
                    tprint(f"Feat Index Sample: {feat_df.index[:2].tolist()}")
                feat_cols = pd.Index(feat_df.columns.map(str))
                kept_syms_str = pd.Index(kept_syms.map(str))
                present_syms = kept_syms_str.intersection(feat_cols)
                missing_syms_live = kept_syms_str.difference(feat_cols)
                extra_syms_live = feat_cols.difference(kept_syms_str)
                tprint(
                    f"Live feature column audit [{k}]: "
                    f"feat_cols={len(feat_cols)} kept_syms={len(kept_syms_str)} "
                    f"present={len(present_syms)} missing={len(missing_syms_live)} extra={len(extra_syms_live)}"
                )
                tprint(
                    f"Live feature column audit [{k}] kept_syms_sample={kept_syms_str[:10].tolist()}"
                )
                tprint(
                    f"Live feature column audit [{k}] feat_cols_sample={feat_cols[:10].tolist()}"
                )
                if len(missing_syms_live) > 0:
                    tprint(
                        f"Live feature column audit [{k}] missing_sample={missing_syms_live[:10].tolist()}"
                    )
                if len(extra_syms_live) > 0:
                    tprint(
                        f"Live feature column audit [{k}] extra_sample={extra_syms_live[:10].tolist()}"
                    )

            missing_ts = common_idx_naive.difference(feat_df.index)
            missing_syms = kept_syms.difference(feat_df.columns)
            if len(missing_ts) > 0 or len(missing_syms) > 0:
                alignment_issue_count += 1
                if len(alignment_issue_samples) < 10:
                    ts_sample = [str(ts) for ts in missing_ts[:3].tolist()]
                    sym_sample = [str(sym) for sym in missing_syms[:3].tolist()]
                    alignment_issue_samples.append(
                        f"{k}: missing_ts={len(missing_ts)} sample_ts={ts_sample} "
                        f"missing_syms={len(missing_syms)} sample_syms={sym_sample}"
                    )

            feat_df_aligned = feat_df.reindex(index=common_idx_naive, columns=kept_syms)
            feat_values = feat_df_aligned.to_numpy(dtype=np.float32)
            feat_final[k] = feat_values[time_idx, compact_sym_idx]
            if k in extraction_trace_features:
                aligned_finite = int(np.isfinite(feat_values).sum())
                extracted_finite = int(np.isfinite(feat_final[k]).sum())
                sample_symbol_counts = []
                sample_symbols = [str(sym) for sym in kept_syms[:5]]
                for sample_sym in sample_symbols:
                    sample_mask = (
                        data_final["symbol"].to_numpy(dtype=object, copy=False)
                        == sample_sym
                    )
                    sample_symbol_counts.append(
                        f"{sample_sym}={int(np.isfinite(feat_final[k][sample_mask]).sum())}"
                    )
                tprint(
                    f"Extraction trace [{k}]: aligned_finite={aligned_finite} "
                    f"extracted_finite={extracted_finite} "
                    f"samples={{" + ", ".join(sample_symbol_counts) + "}}"
                )
            if feat_idx % feature_log_every == 0 or feat_idx == len(feature_items):
                tprint(
                    f"Feature extraction progress: {feat_idx}/{len(feature_items)} "
                    f"({100.0 * feat_idx / len(feature_items):.1f}%) in "
                    f"{time.perf_counter() - feature_align_start:.1f}s"
                )
    tprint(
        f"Feature extraction complete: {len(feat_final)} feature arrays "
        f"in {time.perf_counter() - feature_align_start:.1f}s"
    )
    if alignment_issue_count > 0:
        tprint(
            f"Feature alignment issues detected for {alignment_issue_count} features during "
            "timestamp/symbol reindexing."
        )
        for sample in alignment_issue_samples:
            tprint(f"Alignment issue sample: {sample}")

    availability_start = time.perf_counter()
    symbol_values = data_final["symbol"].to_numpy(dtype=object, copy=False)
    unique_symbols, symbol_codes = np.unique(symbol_values, return_inverse=True)
    total_rows_by_symbol = np.bincount(
        symbol_codes, minlength=len(unique_symbols)
    ).astype(np.int64, copy=False)
    feature_availability_rows: List[Dict[str, Any]] = []
    feature_symbol_availability_rows: List[Dict[str, Any]] = []
    total_rows_pre_filter = max(len(data_final), 1)

    for feature_name, values in feat_final.items():
        finite_mask = np.isfinite(values)
        overall_finite_count = int(finite_mask.sum())
        feature_availability_rows.append(
            {
                "feature": feature_name,
                "overall_finite_row_count": overall_finite_count,
                "overall_finite_row_pct": float(
                    overall_finite_count / total_rows_pre_filter
                ),
            }
        )
        if overall_finite_count == 0:
            continue
        counts_by_symbol = np.bincount(
            symbol_codes,
            weights=finite_mask.astype(np.int32),
            minlength=len(unique_symbols),
        ).astype(np.int64, copy=False)
        for idx, (sym, finite_count) in enumerate(
            zip(unique_symbols, counts_by_symbol)
        ):
            feature_symbol_availability_rows.append(
                {
                    "feature": feature_name,
                    "symbol": str(sym),
                    "finite_row_count": int(finite_count),
                    "finite_row_pct_within_symbol": float(
                        finite_count / max(int(total_rows_by_symbol[idx]), 1)
                    ),
                }
            )

    feature_availability_df = pd.DataFrame(feature_availability_rows).sort_values(
        ["overall_finite_row_count", "feature"], ascending=[False, True]
    )
    atomic_to_csv(
        feature_availability_df,
        root_output_dir / "pre_robust_feature_availability.csv",
        index=False,
    )
    feature_symbol_availability_df = pd.DataFrame(
        feature_symbol_availability_rows
    ).sort_values(
        ["feature", "finite_row_count", "symbol"], ascending=[True, False, True]
    )
    atomic_to_csv(
        feature_symbol_availability_df,
        root_output_dir / "pre_robust_feature_symbol_availability.csv",
        index=False,
    )
    if not feature_availability_df.empty:
        top_availability = feature_availability_df.head(5)
        tprint(
            "Pre-robust feature availability top 5: "
            + ", ".join(
                f"{row.feature}={int(row.overall_finite_row_count)}"
                for row in top_availability.itertuples(index=False)
            )
        )
    tprint(
        f"Pre-robust availability reports saved in {time.perf_counter() - availability_start:.1f}s"
    )

    (
        data_final,
        feat_final,
        fwd_ret_final,
        fwd_ret_norm_final,
        fwd_ret_reasons_final,
        robust_meta,
    ) = apply_robust_data_filtering(
        data_final,
        feat_final,
        fwd_ret_final,
        fwd_ret_norm_final,
        target_nan_reasons=fwd_ret_reasons_final,
        overlap_threshold=0.8,
    )

    tprint(
        f"Robust Data Filter complete: rows_initial={robust_meta['rows_initial']} "
        f"rows_after_any_feat={robust_meta['rows_after_any_feat']} "
        f"rows_final={robust_meta['rows_final']} "
        f"dropped_rows={robust_meta['dropped_rows']} "
        f"retained_features={robust_meta['retained_count']} "
        f"reference={robust_meta['reference_feature']} "
        f"total_features_initial={robust_meta['total_features_initial']} "
        f"features_with_binary_values={robust_meta['features_with_binary_values']}"
    )
    global_row_funnel.append(
        {
            "stage": "robust_data_filter",
            "rows": int(len(data_final)),
            "symbols": int(data_final["symbol"].nunique()),
            "fraction_of_prev": float(len(data_final) / max(prev_rows, 1)),
        }
    )
    atomic_to_csv(
        _build_row_funnel_df(global_row_funnel),
        root_output_dir / "global_row_funnel.csv",
        index=False,
    )
    tprint(
        "Global row funnel: "
        + " -> ".join(f"{row['stage']}={row['rows']}" for row in global_row_funnel)
    )

    if robust_meta["dropped_features"]:
        tprint(
            "Dropped sparse features (low overlap with reference): "
            + ", ".join(
                f"{name}({overlap:.2%})"
                for name, overlap in robust_meta["dropped_features"][:5]
            )
        )

    if robust_meta.get("nan_features_detected"):
        tprint(
            f"WARNING: NaN features detected during filtering: {robust_meta['nan_features_detected']}"
        )

    tprint(
        "SlicePlanner label-step filter: "
        f"rows {planner_filter_meta['rows_before']} -> {planner_filter_meta['rows_after']} | "
        f"symbols {planner_filter_meta['symbols_before']} -> {planner_filter_meta['symbols_after']} | "
        f"applied={planner_filter_meta['sliceplanner_applied']} | "
        f"elapsed={time.perf_counter() - planner_filter_start:.1f}s"
    )

    # Check non-zero features
    non_zero_feats = 0
    for k, v in feat_final.items():
        if v.size > 0 and np.nanmax(np.abs(v)) > 0:
            non_zero_feats += 1
    tprint(
        f"Final Input: {len(data_final)} rows. {non_zero_feats}/{len(feat_final)} features have non-zero values."
    )

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg["output_dir"]) / "sliceplanner_filter_summary.json", "w") as f:
        json.dump(planner_filter_meta, f, indent=2, default=str)
    with open(Path(cfg["output_dir"]) / "run_config_snapshot.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # Triad target mode is now the default and only mode
    tprint("=" * 60)
    tprint("TRIAD TARGET MODE")
    tprint("=" * 60)

    # Import triad target computation
    from extreme_price_movements.triad_targets import compute_triad_targets_for_horizons

    # Compute triad targets for all configured horizons
    horizons = cfg.get("triad_horizons", TRIAD_DEFAULT_HORIZONS)
    target_names = cfg.get("triad_target_names", TRIAD_DEFAULT_TARGET_NAMES)

    tprint(f"Computing triad targets for horizons: {horizons}")
    tprint(f"Target names: {target_names}")

    # Compute triad targets using data_final (long format)
    # compute_triad_targets_for_horizons expects a DataFrame with close, high, low, volume, atr columns
    triad_results_by_horizon = compute_triad_targets_for_horizons(
        df=data_final,
        horizons=horizons,
        atr_col="atr",
    )

    # Convert to the format expected by run_lgbm_mask_generation_triad:
    # {target_name: {horizon: target_array}}
    triad_targets: Dict[str, Dict[int, np.ndarray]] = {
        "target_eff": {},
        "target_vame": {},
        "target_eff_surprisal": {},
        "target_vame_surprisal": {},
        "returns_target": {},
        "atr_norm_returns_target": {},
    }

    # Re-derive alignment indices after robust filtering to ensure target arrays match data_final
    time_idx_final = np.searchsorted(common_idx, data_final["timestamp"])
    sym_idx_final = np.searchsorted(common_syms, data_final["symbol"])
    
    # Use columns from data_final if available, fallback to panel if not.
    # Note: data_final is already flat and aligned to 122040 rows.
    close_final = data_final["close"].to_numpy(dtype=np.float32) if "close" in data_final.columns else panel["close"].to_numpy(dtype=np.float32)[time_idx_final, sym_idx_final]
    atr_final = data_final["atr"].to_numpy(dtype=np.float32) if "atr" in data_final.columns else panel["close"].to_numpy(dtype=np.float32)[time_idx_final, sym_idx_final] * 0.01 # Fallback dummy if missing

    # Pre-compute close and atr for the new return-based miner targets.
    # Note: since this is panel data (flattened via time_idx, sym_idx),
    # using shift on close_wide is correct as it avoids cross-symbol leak.
    close_wide_local = panel["close"].reindex(index=common_idx, columns=common_syms)
    triad_target_nan_reasons: Dict[Tuple[str, int], np.ndarray] = {}
    for horizon, df_targets in triad_results_by_horizon.items():
        # Add new targets
        # Using percentage forward return aligned at time t:
        # (close[t+h] - close[t]) / close[t] = close[t+h] / close[t] - 1
        fwd_pct_ret_wide = close_wide_local.shift(-horizon) / close_wide_local - 1.0
        fwd_pct_ret_final = fwd_pct_ret_wide.to_numpy(dtype=np.float32)[time_idx_final, sym_idx_final]

        ret_reason_wide = pd.DataFrame(
            "",
            index=fwd_pct_ret_wide.index,
            columns=fwd_pct_ret_wide.columns,
            dtype=object,
        )
        current_close_wide = close_wide_local
        future_close_wide = close_wide_local.shift(-horizon)
        ret_reason_wide = ret_reason_wide.mask(
            current_close_wide.isna(),
            TargetNaNReason.CURRENT_CLOSE_MISSING,
        )
        ret_reason_wide = ret_reason_wide.mask(
            future_close_wide.isna() & (ret_reason_wide == ""),
            TargetNaNReason.FUTURE_CLOSE_MISSING,
        )
        ret_reason_wide = ret_reason_wide.mask(
            ~np.isfinite(fwd_pct_ret_wide) & (ret_reason_wide == ""),
            TargetNaNReason.TRANSFORMED_TARGET_NONFINITE,
        )
        if horizon > 0 and len(ret_reason_wide) >= horizon:
            ret_reason_wide.iloc[-horizon:, :] = TargetNaNReason.HORIZON_EXCEEDED
        ret_reason_final = ret_reason_wide.to_numpy(dtype=object)[
            time_idx_final, sym_idx_final
        ]
        aligned_close_final = current_close_wide.to_numpy(dtype=np.float32)[
            time_idx_final, sym_idx_final
        ]
        symbol_alignment_missing_final = np.isfinite(close_final) & ~np.isfinite(
            aligned_close_final
        )
        ret_reason_final[
            symbol_alignment_missing_final & (ret_reason_final == "")
        ] = TargetNaNReason.SYMBOL_ALIGNMENT_MISSING

        current_close_missing_final = ~np.isfinite(close_final)
        ret_reason_final[current_close_missing_final & (ret_reason_final == "")] = (
            TargetNaNReason.CURRENT_CLOSE_MISSING
        )

        # Compute ATR-normalized forward percentage returns
        # Floor the ATR fraction at 0.001 (10bps) to prevent explosion from tiny ATR values.
        # Clip the final target at [-10, 10] to ensure model stability across extreme outliers.
        atr_frac = np.maximum(atr_final / np.maximum(close_final, 1e-12), 0.001)
        fwd_pct_ret_atr_norm_final = np.clip(fwd_pct_ret_final / atr_frac, -10, 10)
        atr_ret_reason_final = ret_reason_final.copy()
        atr_ret_reason_final[~np.isfinite(atr_final) & (atr_ret_reason_final == "")] = (
            TargetNaNReason.ATR_MISSING
        )
        atr_ret_reason_final[
            ~np.isfinite(fwd_pct_ret_atr_norm_final)
            & (atr_ret_reason_final == "")
        ] = TargetNaNReason.TRANSFORMED_TARGET_NONFINITE

        triad_targets["returns_target"][horizon] = fwd_pct_ret_final
        triad_targets["atr_norm_returns_target"][horizon] = fwd_pct_ret_atr_norm_final
        triad_target_nan_reasons[("returns_target", horizon)] = ret_reason_final
        triad_target_nan_reasons[("atr_norm_returns_target", horizon)] = (
            atr_ret_reason_final
        )

        # Target variance diagnostics
        valid_mask = np.isfinite(fwd_pct_ret_final)
        n_valid = valid_mask.sum()
        if n_valid > 0:
            std_raw = np.nanstd(fwd_pct_ret_final)
            std_norm = np.nanstd(fwd_pct_ret_atr_norm_final)
            min_raw, max_raw = np.nanmin(fwd_pct_ret_final), np.nanmax(fwd_pct_ret_final)
            min_norm, max_norm = np.nanmin(fwd_pct_ret_atr_norm_final), np.nanmax(fwd_pct_ret_atr_norm_final)
            tprint(f"  returns_target_{horizon}: {n_valid}/{len(fwd_pct_ret_final)} valid, "
                   f"std={std_raw:.6f}, range=[{min_raw:.4f}, {max_raw:.4f}]")
            tprint(f"  atr_norm_returns_target_{horizon}: {np.isfinite(fwd_pct_ret_atr_norm_final).sum()}/{len(fwd_pct_ret_atr_norm_final)} valid, "
                   f"std={std_norm:.4f}, range=[{min_norm:.2f}, {max_norm:.2f}]")
        else:
            tprint(f"  WARNING: returns_target_{horizon}: 0 valid values!")

        for target_base in [
            "target_eff",
            "target_vame",
            "target_eff_surprisal",
            "target_vame_surprisal",
        ]:
            if target_base not in target_names:
                continue
            col_name = f"{target_base}_{horizon}"
            if col_name in df_targets.columns:
                triad_targets[target_base][horizon] = df_targets[col_name].to_numpy(
                    dtype=np.float32
                )
                valid_count = int(
                    np.isfinite(triad_targets[target_base][horizon]).sum()
                )
                tprint(
                    f"  {col_name}: {valid_count}/{len(triad_targets[target_base][horizon])} valid values"
                )

    # Run triad-target mask generation
    run_lgbm_mask_generation_triad(
        data_final,
        feat_final,
        triad_targets,
        fwd_ret_final,
        fwd_ret_norm_final,
        triad_target_nan_reasons,
        cfg,
    )
