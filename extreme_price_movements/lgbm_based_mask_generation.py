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
TRIAD_DEFAULT_HORIZONS: List[int] = [3, 10]

# Default triad target names
TRIAD_DEFAULT_TARGET_NAMES: List[str] = ["target_eff", "target_ela", "target_vame"]

# Per-target configuration for triad targets
TRIAD_TARGET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "target_eff": {
        "huber_alpha": 1.0,
        "learning_rate": 0.03,
        "min_support_pct": 0.05,
        "ic_hurdle": 0.02,
        "description": "Efficiency: direct vs actual path ratio",
    },
    "target_ela": {
        "huber_alpha": 2.0,
        "learning_rate": 0.02,
        "min_support_pct": 0.04,
        "ic_hurdle": 0.015,
        "description": "Elasticity: reversion tendency at extremes",
    },
    "target_vame": {
        "huber_alpha": 0.5,
        "learning_rate": 0.04,
        "min_support_pct": 0.06,
        "ic_hurdle": 0.025,
        "description": "Volume-adjusted momentum efficiency",
    },
}

# Horizon-specific configuration multipliers
HORIZON_CONFIGS: Dict[int, Dict[str, Any]] = {
    3: {"min_data_in_leaf_multiplier": 0.8, "description": "Very short-term"},
    10: {"min_data_in_leaf_multiplier": 1.0, "description": "Short-term"},
}

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
        Target name for triad mode (e.g., 'target_eff', 'target_ela', 'target_vame')
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
    for (side, source_horizon), entries in cheap_gate_rows.items():
        for cheap_rank, canonical_key in entries:
            post_dedup_rows.append(
                {
                    "side": side,
                    "source_horizon": source_horizon,
                    "cheap_rank": cheap_rank,
                    "canonical_key": canonical_key,
                }
            )
    atomic_to_csv(
        pd.DataFrame(post_dedup_rows),
        output_dir / "step1_post_dedup_registry.csv",
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
    instr_target_value: np.ndarray,
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
            target_val = int(instr_target_value[instr_idx])
            if source_type == 0:
                for sample_idx in range(n_samples):
                    if out[rule_idx, sample_idx]:
                        out[rule_idx, sample_idx] = (
                            x_values[sample_idx, source_idx] == target_val
                        )
            else:
                for sample_idx in range(n_samples):
                    if out[rule_idx, sample_idx]:
                        out[rule_idx, sample_idx] = (
                            context_values[source_idx, sample_idx] == target_val
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

    def _compute_rank_norm(
        self, arr: np.ndarray, symbol_codes: np.ndarray
    ) -> np.ndarray:
        """
        Compute a rank-normalized source in [0, 1] per symbol.

        We keep this fold-local and symbol-local so downstream thresholds can
        be saved and replayed during OOF/OOS/inference without recomputing
        any dynamic cut points.
        """
        values = np.asarray(arr, dtype=np.float32)
        codes, _ = pd.factorize(symbol_codes, sort=False)
        codes = np.asarray(codes, dtype=np.int32)
        out = np.full(values.shape[0], np.nan, dtype=np.float32)

        for code in np.unique(codes):
            idx = np.flatnonzero(codes == code)
            if idx.size == 0:
                continue
            g = values[idx]
            valid = np.isfinite(g)
            if int(np.sum(valid)) <= 1:
                continue

            g_valid = g[valid]
            order = np.argsort(g_valid, kind="mergesort")
            ranks = np.empty(g_valid.shape[0], dtype=np.float32)
            ranks[order] = np.arange(1, g_valid.shape[0] + 1, dtype=np.float32)
            norm = (ranks - 1.0) / max(float(g_valid.shape[0] - 1), 1.0)

            out_idx = idx[valid]
            out[out_idx] = norm.astype(np.float32, copy=False)

        return out

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

            nan_mask = np.isnan(raw_arr)
            bool_arr = (raw_arr > 0.5).astype(np.float32, copy=False)
            bool_arr = np.asarray(bool_arr, dtype=np.float32)
            bool_arr[nan_mask] = np.nan
            bool_name = f"{group_name[:3]}_{src}_raw"
            _register_boolean_column(
                source_arr=raw_arr,
                bool_name=bool_name,
                bool_arr=bool_arr,
                group_name=group_name,
                src=src,
                family=family,
                booleanization_method="raw_binary",
                threshold_type="binary",
                threshold_value=0.5,
                threshold_upper_value=None,
                description="Raw binary feature > 0.5",
                min_support=int(cfg.get("min_feature_support", 10)),
            )

        def _add_continuous_features_as_booleans(sources, group_name):
            min_support = int(cfg.get("min_feature_support", 10))

            for src in sources:
                if self._is_reserved_target_side_feature(src):
                    tprint(
                        f"WARNING: skipping reserved target-side feature '{src}' in miner feature prep"
                    )
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

                    ts_ranks = self._compute_rank_norm(raw_arr, symbol_codes)

                    nan_rate_ts = float(np.isnan(ts_ranks).mean())

                    self.rank_audit_rows.append(
                        {
                            "source_feature": src,
                            "group": group_name,
                            "nan_rate_before": nan_rate_before,
                            "nan_rate_ts": nan_rate_ts,
                        }
                    )

                    # NaN-preserving booleanization.
                    # ts_ranks is NaN wherever the raw feature was NaN for that symbol.
                    # We must propagate NaN through to the boolean column so that
                    # filter_complete_feature_rows correctly drops rows missing feature data.
                    nan_mask = np.isnan(ts_ranks)

                    def _bool_float(condition: np.ndarray) -> np.ndarray:
                        out = condition.astype(np.float32)
                        out[nan_mask] = np.nan
                        return out

                    for q in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                        # Top quantiles (>= q)
                        bool_name_top = f"{group_name[:3]}_{src}_ts_top{int(q*100)}"
                        bool_arr_top = _bool_float(ts_ranks >= q)
                        _register_boolean_column(
                            source_arr=ts_ranks,
                            bool_name=bool_name_top,
                            bool_arr=bool_arr_top,
                            group_name=group_name,
                            src=src,
                            family=family,
                            booleanization_method="rank_norm",
                            threshold_type="top_quantile",
                            threshold_value=q,
                            threshold_upper_value=None,
                            description=f"TS Rank >= {q}",
                            min_support=min_support,
                        )

                        # Bottom quantiles (<= q)
                        bool_name_bot = f"{group_name[:3]}_{src}_ts_bot{int(q*100)}"
                        bool_arr_bot = _bool_float(ts_ranks <= q)
                        _register_boolean_column(
                            source_arr=ts_ranks,
                            bool_name=bool_name_bot,
                            bool_arr=bool_arr_bot,
                            group_name=group_name,
                            src=src,
                            family=family,
                            booleanization_method="rank_norm",
                            threshold_type="bot_quantile",
                            threshold_value=q,
                            threshold_upper_value=None,
                            description=f"TS Rank <= {q}",
                            min_support=min_support,
                        )

                    # Expanded median bands
                    for q_band in [0.20, 0.25, 0.30, 0.40]:
                        q_band_upper = 1.0 - q_band
                        band_name = f"{group_name[:3]}_{src}_ts_band{int(q_band*100)}_{int(q_band_upper*100)}"
                        band_arr = _bool_float(
                            (ts_ranks >= q_band) & (ts_ranks <= q_band_upper)
                        )

                        _register_boolean_column(
                            source_arr=ts_ranks,
                            bool_name=band_name,
                            bool_arr=band_arr,
                            group_name=group_name,
                            src=src,
                            family=family,
                            booleanization_method="rank_norm",
                            threshold_type="band_quantile",
                            threshold_value=q_band,
                            threshold_upper_value=q_band_upper,
                            description=f"TS Rank inside {int(q_band*100)}-{int(q_band_upper*100)} band",
                            min_support=min_support,
                        )

        # 1. Trigger Features — DISABLED (all trigger features removed from pipeline)
        if "trigger" in active_groups:
            tprint("TRIGGER features disabled — skipping trigger group.")

        # 2. Location Features
        if "location" in active_groups:
            # Continuous location features are the sole location source family.
            _add_continuous_features_as_booleans(CONTINUOUS_LOCATION_COLS, "location")

        # 3. Regime Features (continuous -> hybrid booleanize)
        if "regime" in active_groups:
            regime_sources = sorted(
                list(set(RIDGE_FEATURE_COLS) | set(TEST_FEATURE_KEYS))
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
        if self.rank_audit_rows:
            rank_audit_df = pd.DataFrame(self.rank_audit_rows)
            rank_audit_df["worst_nan"] = rank_audit_df[
                ["nan_rate_before", "nan_rate_ts"]
            ].max(axis=1)
            top_nan = rank_audit_df.sort_values("worst_nan", ascending=False).head(10)
            tprint("Top 10 features with worst NaN rates:")
            for row in top_nan.itertuples(index=False, name=None):
                tprint(f"  - {row[0]}:{row[1]} -> before={row[2]:.2%}, ts={row[3]:.2%}")
        else:
            rank_audit_df = pd.DataFrame()

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
        else:
            bool_support_audit_df = pd.DataFrame()

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

    def _compute_ts_ranks(
        self, arr: np.ndarray, symbol_codes: np.ndarray
    ) -> np.ndarray:
        """
        Time-series ranking using pandas for vectorization speed.
        """
        s = pd.Series(arr, index=symbol_codes)

        # Fix root cause of 0.5 artifacts: if cross-section has no variance, return NaN
        def _rank_safe(g):
            if g.nunique() <= 1:
                return np.full(len(g), np.nan)
            return g.rank(pct=True)

        # Using transform for speed while preserving index
        ranks = s.groupby(level=0, sort=False).transform(_rank_safe)
        return ranks.values

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
            n_ones = np.sum(col == 1)
            n_zeros = np.sum(col == 0)

            dropped = False
            reason = "retained"

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
                # Duplicate check via hash, scoped to the originating source feature.
                metadata = self.metadata[name]
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
                    "support": int(n_ones),
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
        X_va, y_va = X_va[va_mask], y_va[va_mask]

        if len(y_tr) < 100:
            tprint(
                f"WARNING: Fold {fold_id} has very few training samples ({len(y_tr)})"
            )
        if len(y_va) == 0:
            raise ValueError(f"Fold {fold_id} has no finite validation samples")

        # Sample weights for regime mining
        symbol_codes_tr, _ = pd.factorize(symbol_id_tr, sort=False)
        sample_weight = make_regime_weights(
            y_tr, symbol_codes_tr.astype(np.int32, copy=False), horizon=horizon
        )
        sample_weight = sample_weight * make_support_preference_weights(
            X_tr,
            target_pct=float(
                self.cfg.get("support_preference_target_pct", TARGET_SUPPORT)
            ),
            preferred_low_pct=float(
                self.cfg.get(
                    "support_preference_preferred_low_pct", PREFERRED_SUPPORT_MIN
                )
            ),
            preferred_high_pct=float(
                self.cfg.get(
                    "support_preference_preferred_high_pct", PREFERRED_SUPPORT_MAX
                )
            ),
            strength=float(self.cfg.get("support_preference_strength", 0.20)),
            w_min=float(self.cfg.get("support_preference_weight_min", 0.85)),
            w_max=float(self.cfg.get("support_preference_weight_max", 1.25)),
        )
        if surprisal_bits_tr is not None:
            sample_weight = sample_weight * make_surprisal_sample_weights(
                surprisal_bits_tr,
                alpha=float(self.cfg.get("surprisal_weight_alpha", 0.20)),
                reference_bits=float(
                    self.cfg.get("surprisal_weight_reference_bits", 3.0)
                ),
                w_min=float(self.cfg.get("surprisal_weight_min", 1.0)),
                w_max=float(self.cfg.get("surprisal_weight_max", 1.20)),
            )
        sample_weight = np.clip(
            sample_weight,
            float(self.cfg.get("sample_weight_final_min", 0.75)),
            float(self.cfg.get("sample_weight_final_max", 1.5)),
        ).astype(np.float32, copy=False)

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

        min_gain_to_split = float(self.cfg.get("min_gain_to_split", 0.0))
        if "hpo_min_gain_to_split" in self.cfg:
            min_gain_to_split = float(self.cfg["hpo_min_gain_to_split"])

        min_leaf_frac = float(self.cfg.get("lgbm_min_leaf_frac", 0.001))
        min_data_in_leaf = max(10, int(min_leaf_frac * X_tr.shape[0]))
        if "hpo_min_data_in_leaf" in self.cfg:
            min_data_in_leaf = max(10, int(self.cfg["hpo_min_data_in_leaf"]))

        # Use quantile loss for all targets (triad targets work with quantile regression)
        alpha_hpo = float(self.cfg.get("alpha_hpo", 0.90))
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
                if metric_name == "l2":
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

        for tree_idx, tree in enumerate(tree_info):
            compiled_tree = self._compile_tree(tree["tree_structure"])
            total_samples = int(max(compiled_tree.leaf_count[0], 1))
            self._traverse_compiled_tree(
                compiled_tree,
                tree_idx,
                model_id,
                fold_id,
                seed,
                rules,
                total_samples=total_samples,
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
        self, threshold: float, direction: int
    ) -> Optional[Tuple[int, str, float]]:
        """
        Simplified and hardened normalization for [0, 1] boolean features.
        LightGBM JSON format:
        Left child (direction 1) is 'value <= threshold'
        Right child (direction 0) is 'value > threshold'
        """
        threshold = self._normalize_boolean_threshold(threshold)

        # Direction 1: Left (<= 0.5) -> Feature is 0
        if direction == 1:
            return (0, "<=", threshold)

        # Direction 0: Right (> 0.5) -> Feature is 1
        else:
            return (1, ">", threshold)

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
                split_feature.append(int(node["split_feature"]))
                threshold.append(
                    float(self._normalize_boolean_threshold(node.get("threshold", 0.5)))
                )
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
    ) -> None:
        stack: List[Tuple[int, List[RuleCondition], float]] = [(0, [], 0.0)]
        while stack:
            node_idx, conditions, current_gain = stack.pop()

            if tree.is_leaf[node_idx]:
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
                norm = self._normalize_predicate(raw_thr, direction)
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
            if group not in self.collapse_duplicate_groups:
                feat_map: Dict[int, int] = {}
                for c in group_conditions:
                    prev = feat_map.get(c.feature_index)
                    if prev is not None:
                        if prev != c.normalized_value:
                            return None, f"contradiction_{c.feature_name}"
                        continue
                    feat_map[c.feature_index] = c.normalized_value
                    reduced.append(c)
                continue

            # Point 14: Implement actual collapsing for duplicate groups
            # For now, we take the FIRST condition for each unique feature in the group
            # (or we could take the most restrictive, but first is safer for now)
            feat_map: Dict[int, int] = {}
            for c in group_conditions:
                if c.feature_index not in feat_map:
                    feat_map[c.feature_index] = c.normalized_value
                    reduced.append(c)
                elif feat_map[c.feature_index] != c.normalized_value:
                    # If duplicate features in same group have DIFFERENT values, it's a contradiction
                    return None, f"contradiction_in_collapsed_group_{c.feature_name}"

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
                group_conds.sort(key=lambda x: x.feature_name)
                # Deduplicate same feature identical conditions
                seen = set()
                joined = []
                for c in group_conds:
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
                if "==" not in cond_str:
                    continue
                names.append(cond_str.split("==")[0])
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
        total += sum(1 for cond_str in slot_value.split("&") if "==" in cond_str)
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
        self.context_lookup = {
            key: np.asarray(val, dtype=bool)
            for key, val in (context_lookup or {}).items()
        }
        self.context_key_map = context_key_map or {}
        self.slot_order = tuple(slot_order)
        self.name_to_idx = {m.feature_name: m.feature_index for m in metadata}
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
        target_values: List[int] = []
        context_feature_names: List[str] = []
        context_target_values: List[int] = []
        unresolved: List[Tuple[str, str]] = []

        for group, slot_value in slot_map.items():
            if slot_value == "*":
                continue
            for cond_str in slot_value.split("&"):
                if "==" not in cond_str:
                    self.malformed_key_count += 1
                    raise ValueError(f"Malformed slot {cond_str} in {canonical_key}")
                feature_name, target_val_raw = cond_str.split("==")
                target_val = int(target_val_raw)
                if feature_name in self.name_to_idx:
                    feature_indices.append(self.name_to_idx[feature_name])
                    target_values.append(target_val)
                elif feature_name in self.context_lookup:
                    context_feature_names.append(feature_name)
                    context_target_values.append(target_val)
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
            "target_values": np.asarray(target_values, dtype=np.int8),
            "context_feature_names": tuple(context_feature_names),
            "context_target_values": np.asarray(context_target_values, dtype=np.int8),
            "parent_context_name": parent_context_name,
        }

        instr_source_type: List[int] = []
        instr_source_idx: List[int] = []
        instr_target_value: List[int] = []
        for idx, target_val in zip(feature_indices, target_values):
            instr_source_type.append(0)
            instr_source_idx.append(int(idx))
            instr_target_value.append(int(target_val))
        for feature_name, target_val in zip(
            context_feature_names, context_target_values
        ):
            instr_source_type.append(1)
            instr_source_idx.append(int(self.context_name_to_idx[feature_name]))
            instr_target_value.append(int(target_val))
        if parent_context_name is not None:
            instr_source_type.append(1)
            instr_source_idx.append(int(self.context_name_to_idx[parent_context_name]))
            instr_target_value.append(1)
        spec["instr_source_type"] = np.asarray(instr_source_type, dtype=np.int8)
        spec["instr_source_idx"] = np.asarray(instr_source_idx, dtype=np.int32)
        spec["instr_target_value"] = np.asarray(instr_target_value, dtype=np.int8)

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
        target_values = spec["target_values"]
        for idx, target_val in zip(feature_indices, target_values):
            values = self.X[:, idx] if indices is None else self.X[indices, idx]
            mask &= values == target_val

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
        instr_target_value_chunks: List[np.ndarray] = []
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
            target_value = spec["instr_target_value"]
            instr_source_type_chunks.append(source_type)
            instr_source_idx_chunks.append(source_idx)
            instr_target_value_chunks.append(target_value)
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
                instr_target_value = np.concatenate(instr_target_value_chunks).astype(
                    np.int8, copy=False
                )
            else:
                instr_source_type = np.empty(0, dtype=np.int8)
                instr_source_idx = np.empty(0, dtype=np.int32)
                instr_target_value = np.empty(0, dtype=np.int8)

            non_composite_masks = _compute_masks_from_instruction_matrix_numba(
                x_values,
                context_values,
                instr_source_type,
                instr_source_idx,
                instr_target_value,
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
            ctx_name = slot_map["context"].split("==")[0]
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

        hard_min = float(self.cfg.get("objective_support_min_pct", SUPPORT_MIN))
        target_low = float(
            self.cfg.get("objective_support_target_low_pct", PREFERRED_SUPPORT_MIN)
        )
        target_high = float(
            self.cfg.get("objective_support_target_high_pct", PREFERRED_SUPPORT_MAX)
        )
        hard_max = float(self.cfg.get("objective_support_max_pct", SUPPORT_MAX))
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
            tprint(
                "Stage A: Trade path quality "
                f"path_obs={path_obs} folds={path_folds} "
                f"n_obs={path_quality.get('n_obs', 0)} n_folds={path_quality.get('n_folds', 0)} "
                f"quality_stability_score={path_quality.get('quality_stability_score', np.nan):.4f} "
                f"trade_path_quality_score={path_quality.get('trade_path_quality_score', np.nan):.4f} "
                f"elapsed={path_quality_elapsed:.2f}s"
            )
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
        composite_score = full_quality_score

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
            "composite_score": composite_score,
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
        if summary["hurdle_excess"] <= 0:
            rejected.append("below_hurdle")
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
            ["accepted", "composite_score"], ascending=[False, False]
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
    registry["reject_support"] = registry["mean_support_pct"] < float(
        cfg.get("min_context_support_pct", 0.01)
    )
    registry["reject_ret"] = registry["directional_mean_ret"] <= float(
        cfg.get("min_context_mean_ret", 0.0)
    )
    registry["reject_presence"] = registry["presence_freq"] < float(
        cfg.get("min_context_presence_freq", cfg.get("min_presence_freq", 0.4))
    )
    registry["reject_sign"] = registry["sign_consistency"] < float(
        cfg.get("min_context_sign_consistency", cfg.get("min_sign_consistency", 0.0))
    )
    registry["reject_arity"] = registry["display_arity"] < int(
        cfg.get("min_context_display_arity", 2)
    )
    registry["reject_structural"] = ~registry.get(
        "is_structurally_sound", pd.Series(True, index=registry.index)
    ).fillna(False)

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
    run_step: str = "full",
    step1_input_dir: Optional[Path] = None,
    candidate_registry_override: Optional[pd.DataFrame] = None,
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

    # Relaxed completeness check per user request.
    # We only require the target to be finite and at least ONE feature to be present.
    _feature_any_finite = np.any(np.isfinite(X), axis=1)
    _complete = np.isfinite(primary_target) & _feature_any_finite

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

    if folds is None:
        folds = build_walk_forward_folds(
            n_samples=len(data),
            n_folds=int(cfg.get("n_folds", 5)),
            min_train_frac=float(cfg.get("cv_min_train_frac", 0.5)),
            embargo=int(cfg.get("cv_embargo", 0)),
        )
    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        if tr_idx.size == 0 or va_idx.size == 0 or tr_idx.max() >= va_idx.min():
            raise ValueError(f"Invalid fold {fold_id} in {stage_name}")

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
            folds,
            step_mode="step2",
            step1_checkpoint_dir=step1_input_dir,
            checkpoint_output_dir=output_dir,
        )
        if not assessment_df.empty:
            atomic_to_csv(assessment_df, output_dir / "final_mask_assessment_audit.csv")
            accepted_registry = candidate_registry.merge(
                assessment_df, on="canonical_key", how="left"
            )
        else:
            accepted_registry = candidate_registry.iloc[0:0].copy()
        if "is_structurally_sound" in accepted_registry.columns:
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

        orig_tr_rows = len(tr_idx)
        orig_va_rows = len(va_idx)
        tr_idx = _cap_fold_indices(tr_idx, int(cfg.get("fold_train_row_cap", 50_000)))
        va_idx = _cap_fold_indices(va_idx, int(cfg.get("fold_val_row_cap", 10_000)))

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

        # Target distribution summary (using primary_target, which is the real training target)
        _tr_tgt = primary_target[tr_idx]
        tr_target_valid = _tr_tgt[~np.isnan(_tr_tgt)]
        if len(tr_target_valid) > 0:
            tr_mean = tr_target_valid.mean()
            tr_std = tr_target_valid.std()
            tr_p1 = np.percentile(tr_target_valid, 1)
            tr_p50 = np.percentile(tr_target_valid, 50)
            tr_p99 = np.percentile(tr_target_valid, 99)

            # Check severe clipping
            clipped = np.clip(tr_target_valid, tr_p1, tr_p99)
            clip_diff = (
                np.abs(tr_target_valid - clipped).sum() / np.abs(tr_target_valid).sum()
            )
            if clip_diff > 0.05:
                tprint(
                    f"WARNING: Severe target clipping in Fold {fold_id} ({clip_diff:.1%} diff)"
                )

        else:
            tr_mean, tr_std, tr_p1, tr_p50, tr_p99 = (
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
                "target_mean": tr_mean,
                "target_std": tr_std,
                "target_p1": tr_p1,
                "target_p50": tr_p50,
                "target_p99": tr_p99,
            }
        )

        tprint(
            f"Fold {fold_id}: Target {target_name} -> mean={tr_mean:.4f}, std={tr_std:.4f}, p1={tr_p1:.4f}, p50={tr_p50:.4f}, p99={tr_p99:.4f}"
        )

        # All rows in tr_idx and va_idx are already complete (pre-filtered upstream).
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr_raw = primary_target[tr_idx]
        y_va_raw = primary_target[va_idx]
        symbol_id_tr = data["symbol"].to_numpy()[tr_idx]
        surprisal_bits_tr = (
            None
            if sample_weight_surprisal_override is None
            else sample_weight_surprisal_override[tr_idx]
        )

        tr_finite_mask = np.isfinite(y_tr_raw)
        va_finite_mask = np.isfinite(y_va_raw)
        if not tr_finite_mask.any() or not va_finite_mask.any():
            tprint(
                f"WARNING: Skipping fold {fold_id} because it has no finite "
                f"{'training' if not tr_finite_mask.any() else 'validation'} samples "
                f"for target {target_name} @ H{horizon}."
            )
            continue

        y_tr_clip = np.clip(
            y_tr_raw,
            np.nanquantile(y_tr_raw, 0.01),
            np.nanquantile(y_tr_raw, 0.99),
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
                y_va_raw,
                fold_id,
                seed,
                target_type=target_type,
                horizon=horizon,
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
                            "gain": gain,
                            "split": split,
                        }
                    )
                    feature_importance_records.append(fi_records[-1])

            if fi_records:
                fi_df = pd.DataFrame(fi_records)
                top_gain = fi_df.sort_values("gain", ascending=False).head(5)
                tprint("Top 5 features by gain:")
                for row in top_gain.itertuples(index=False, name=None):
                    tprint(f"  - {row[0]}: {row[1]:.2f}")

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
            # arity, structural depth, groups used, regime families used
            arity = display_arity_for_key(r.canonical_key)
            depth = structural_depth_for_key(r.canonical_key)
            groups_used = tuple(sorted(set(c.group for c in r.conditions)))

            regime_families = []
            for c in r.conditions:
                if c.group == "regime":
                    fam = (
                        m.regime_family
                        if (m := metadata[c.feature_index])
                        else "unknown"
                    )
                    if fam:
                        regime_families.append(fam)

            regime_families_tuple = tuple(sorted(set(regime_families)))
            family_combos[regime_families_tuple] += 1

            shape_records.append(
                {
                    "canonical_key": r.canonical_key,
                    "display_arity": arity,
                    "structural_depth": depth,
                    "groups_used": "|".join(groups_used),
                    "regime_families_used": "|".join(regime_families_tuple),
                }
            )

        pd.DataFrame(shape_records).to_csv(
            output_dir / "extracted_rule_shape_summary.csv", index=False
        )

        family_df = pd.DataFrame(
            [
                {"regime_families_combo": "|".join(k), "count": v}
                for k, v in family_combos.items()
            ]
        )
        family_df.sort_values("count", ascending=False).to_csv(
            output_dir / "extracted_rule_family_combo_summary.csv", index=False
        )

        tprint("Top valid family combinations:")
        for _, row in (
            family_df.sort_values("count", ascending=False).head(5).iterrows()
        ):
            tprint(f"  - {row['regime_families_combo']}: {row['count']}")

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
        bounded_target=primary_target,
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
        folds,
        fold_health_summary=fold_health_summary,
        step_mode=run_step,
        step1_checkpoint_dir=step1_input_dir,
        checkpoint_output_dir=output_dir,
    )
    if not assessment_df.empty:
        atomic_to_csv(assessment_df, output_dir / "final_mask_assessment_audit.csv")
        accepted_registry = candidate_registry.merge(
            assessment_df, on="canonical_key", how="left"
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

    side_output_dir = root_output_dir / side
    side_output_dir.mkdir(parents=True, exist_ok=True)

    side_input_rows = int(len(data))
    side_input_symbols = (
        int(data["symbol"].nunique()) if "symbol" in data.columns else np.nan
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
                cfg["hpo_min_data_in_leaf"] = max(
                    25, int(round(best_cfg.cfg.min_leaf_frac * train_n))
                )
            else:
                tprint(
                    f"Dynamic HPO failed or invalid: {best_cfg.reason if best_cfg else 'Unknown'}. Using defaults."
                )

        except Exception as e:
            tprint(f"Dynamic HPO encountered error: {e}. Using defaults.")
            traceback.print_exc()

    # Use bounded_target as the real training target if provided.
    # For short side, flip sign (higher target_eff = more favourable for that side).
    if bounded_target is not None:
        side_target = -bounded_target if side == "short" else bounded_target
    else:
        side_target = (
            side_fwd_ret_norm  # legacy fallback only when no triad target is supplied
        )

    feature_any_finite = np.any(np.isfinite(X_a), axis=1)
    target_finite = np.isfinite(side_target)
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

    run_step = str(cfg.get("run_step", "full")).lower()
    step1_input_dir: Optional[Path] = None
    candidate_registry_override: Optional[pd.DataFrame] = None
    if run_step == "step2":
        step1_base_dir = cfg.get("step1_dir")
        if not step1_base_dir:
            raise ValueError("step2 requires cfg['step1_dir']")
        step1_input_dir = resolve_stage_a_step1_dir(
            step1_base_dir, target_name=target_name, horizon=horizon, side=side
        )
        candidate_registry_override = pd.read_csv(
            step1_input_dir / "candidate_rule_registry.csv"
        )
        has_global_slice_filter = "global_step2_selected_keys_by_slice" in cfg
        selected_keys_by_slice = cfg.get("global_step2_selected_keys_by_slice", {})
        slice_sel_key = _make_slice_selection_key(
            target_name or "unknown", int(horizon or -1), side
        )
        selected_keys = selected_keys_by_slice.get(slice_sel_key)
        if has_global_slice_filter:
            selected_keys = selected_keys or set()
            selected_keys = {str(k) for k in selected_keys}
            if "canonical_key" in candidate_registry_override.columns:
                candidate_registry_override = candidate_registry_override[
                    candidate_registry_override["canonical_key"]
                    .astype(str)
                    .isin(selected_keys)
                ].copy()
        tprint(
            f"Stage A: Loaded stored step1 candidate registry from {step1_input_dir} "
            f"({len(candidate_registry_override)} rules)"
        )

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
        run_step=run_step,
        step1_input_dir=step1_input_dir,
        candidate_registry_override=candidate_registry_override,
    )
    log_stage_gate_diagnostics("Stage A", stage_a_result, cfg)

    if run_step == "step1":
        tprint(f"Stage A step1 only complete for {target_name} @ H{horizon} [{side}]")
        return {
            "stage_a": pd.DataFrame(),
            "stage_a_result": stage_a_result,
            "metadata_a": metadata_a,
            "X_a": X_a,
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

        tprint("Top 15 Final Diverse Rules (Thorough Report):")
        top_final = select_top_diverse_rules(
            combined_global_registry, combined_mask_map, top_n=15
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
) -> Dict[str, set[str]]:
    global_cap = int(cfg.get("global_ridge_candidate_cap", 80))
    if not pooled_step1_frames or X_ref is None or len(metadata_ref) == 0:
        return {}

    pooled = pd.concat(pooled_step1_frames, ignore_index=True, copy=False)
    if pooled.empty:
        return {}

    pooled = pooled.drop_duplicates(
        subset=["canonical_key", "source_target", "source_horizon", "side"],
        keep="first",
    ).copy()
    if "cheap_rank" not in pooled.columns:
        pooled["cheap_rank"] = 0.0

    resolver = CanonicalRuleMaskResolver(X_ref, metadata_ref)
    unique_keys = list(dict.fromkeys(pooled["canonical_key"].astype(str).tolist()))
    key_to_idx = {k: i for i, k in enumerate(unique_keys)}

    rng = np.random.default_rng(42)
    n_rows = X_ref.shape[0]
    subsample_size = int(
        min(max(int(cfg.get("global_overlap_subsample_size", 10000)), 1000), n_rows)
    )
    sub_idx = (
        np.sort(rng.choice(n_rows, size=subsample_size, replace=False))
        if n_rows > subsample_size
        else np.arange(n_rows)
    )
    mask_matrix = resolver.get_masks_matrix(unique_keys)[:, sub_idx].astype(
        np.int8, copy=False
    )
    intersections = mask_matrix @ mask_matrix.T
    supports = np.diag(intersections).astype(np.float32)

    pooled["canonical_key"] = pooled["canonical_key"].astype(str)
    pooled["mask_idx"] = pooled["canonical_key"].map(key_to_idx).astype(np.int32)
    pooled["cheap_rank"] = pd.to_numeric(pooled["cheap_rank"], errors="coerce").fillna(
        -np.inf
    )
    pooled["support_sub"] = pooled["mask_idx"].map(lambda i: float(supports[int(i)]))
    pooled = pooled[pooled["support_sub"] > 0].copy()
    pooled = pooled.sort_values(
        ["cheap_rank", "source_target", "source_horizon", "side"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    selected_rows: List[int] = []
    support_ratio_min = float(cfg.get("global_overlap_support_ratio_min", 0.70))
    same_bucket_thr = float(cfg.get("global_overlap_same_bucket_threshold", 0.975))
    diff_dim_bonus = float(cfg.get("global_overlap_diff_dim_bonus", 0.01))
    both_diff_bonus = float(cfg.get("global_overlap_both_diff_bonus", 0.01))

    for row_idx, row in pooled.iterrows():
        if len(selected_rows) >= global_cap:
            break
        cand_mask_idx = int(row["mask_idx"])
        cand_support = float(row["support_sub"])
        keep = True
        for sel_idx in selected_rows:
            sel = pooled.iloc[sel_idx]
            sel_mask_idx = int(sel["mask_idx"])
            sel_support = float(sel["support_sub"])
            inter = float(intersections[cand_mask_idx, sel_mask_idx])
            if inter < 1.0:
                continue
            min_support = max(min(cand_support, sel_support), 1.0)
            overlap = inter / min_support
            support_ratio = min(cand_support, sel_support) / max(
                max(cand_support, sel_support), 1e-9
            )
            if support_ratio < support_ratio_min:
                continue
            threshold = same_bucket_thr
            side_diff = str(row["side"]) != str(sel["side"])
            horizon_diff = int(row["source_horizon"]) != int(sel["source_horizon"])
            if side_diff or horizon_diff:
                threshold += diff_dim_bonus
            if side_diff and horizon_diff:
                threshold += both_diff_bonus
            threshold = min(threshold, 0.995)
            if overlap > threshold:
                keep = False
                break
        if keep:
            selected_rows.append(int(row_idx))

    selected = (
        pooled.iloc[selected_rows].copy() if selected_rows else pooled.iloc[0:0].copy()
    )
    tprint(
        "Global stage2 shortlist: "
        f"input={len(pooled)} selected={len(selected)} cap={global_cap}"
    )

    selected_by_slice: Dict[str, set[str]] = collections.defaultdict(set)
    for row in selected.itertuples(index=False):
        slice_key = _make_slice_selection_key(
            str(row.source_target), int(row.source_horizon), str(row.side)
        )
        selected_by_slice[slice_key].add(str(row.canonical_key))
    return dict(selected_by_slice)


def run_lgbm_mask_generation_triad(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    triad_targets: Dict[str, Dict[int, np.ndarray]],
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
                surprisal_key in triad_targets
                and horizon in triad_targets[surprisal_key]
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
            dummy_fwd_ret = bounded_target.astype(np.float32, copy=True)
            dummy_fwd_ret_norm = bounded_target.astype(np.float32, copy=True)

            # Run both sides
            for side in ["long", "short"]:
                tprint(
                    f"\n--- Running {side.upper()} side for {target_name} @ H{horizon} ---"
                )

                side_results = run_side_pipeline(
                    side=side,
                    data=data,
                    feature_dict=feature_dict,
                    fwd_ret=dummy_fwd_ret,
                    fwd_ret_norm=dummy_fwd_ret_norm,
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

                if triad_run_step == "full":
                    if x_ref is None and side_results.get("X_a") is not None:
                        x_ref = side_results["X_a"]
                        metadata_ref = side_results.get("metadata_a", []) or []
                    step1_dir = resolve_stage_a_step1_dir(
                        root_output_dir,
                        target_name=target_name,
                        horizon=horizon,
                        side=side,
                    )
                    post_dedup_path = step1_dir / "step1_post_dedup_registry.csv"
                    if post_dedup_path.exists():
                        post_df = pd.read_csv(post_dedup_path)
                        if not post_df.empty:
                            post_df["source_target"] = target_name
                            post_df["source_horizon"] = horizon
                            post_df["side"] = side
                            pooled_step1_frames.append(post_df)
                else:
                    # Add provenance to registry
                    if not side_results["stage_a"].empty:
                        stage_a_with_prov = side_results["stage_a"].copy()
                        stage_a_with_prov["source_target"] = target_name
                        stage_a_with_prov["source_horizon"] = horizon
                        all_registries.append(stage_a_with_prov)

    if triad_run_step == "full":
        global_selected_keys_by_slice = build_global_stage_a_ridge_shortlist(
            pooled_step1_frames=pooled_step1_frames,
            X_ref=x_ref,
            metadata_ref=metadata_ref,
            cfg=cfg,
        )
        tprint(
            "Global step2 shortlist by slice: "
            + ", ".join(
                f"{k}={len(v)}"
                for k, v in sorted(global_selected_keys_by_slice.items())
            )
        )

        results_by_target_horizon = {}
        all_registries = []
        for horizon in horizons:
            for target_name in target_names:
                bounded_target = (
                    triad_targets.get(target_name, {}).get(horizon)
                    if target_name in triad_targets
                    else None
                )
                if bounded_target is None:
                    continue
                surprisal_key = f"{target_name}_surprisal"
                bounded_target_surprisal = None
                if (
                    surprisal_key in triad_targets
                    and horizon in triad_targets[surprisal_key]
                ):
                    bounded_target_surprisal = triad_targets[surprisal_key][horizon]
                horizon_target_dir = root_output_dir / f"h{horizon}" / target_name
                horizon_cfg = cfg.copy()
                horizon_cfg["target_name"] = target_name
                horizon_cfg["horizon"] = horizon
                horizon_cfg["run_step"] = "step2"
                horizon_cfg["step1_dir"] = str(root_output_dir)
                horizon_cfg[
                    "global_step2_selected_keys_by_slice"
                ] = global_selected_keys_by_slice
                if target_name in TRIAD_TARGET_CONFIGS:
                    target_config = TRIAD_TARGET_CONFIGS[target_name]
                    horizon_cfg["huber_alpha"] = target_config.get("huber_alpha", 0.9)
                    horizon_cfg["learning_rate"] = target_config.get(
                        "learning_rate", 0.03
                    )
                    horizon_cfg["min_support_pct"] = target_config.get(
                        "min_support_pct", 0.05
                    )
                    horizon_cfg["ic_hurdle"] = target_config.get("ic_hurdle", 0.02)
                if horizon in HORIZON_CONFIGS:
                    horizon_config = HORIZON_CONFIGS[horizon]
                    min_leaf_mult = horizon_config.get(
                        "min_data_in_leaf_multiplier", 1.0
                    )
                    base_min_leaf = int(horizon_cfg.get("min_data_in_leaf", 64))
                    horizon_cfg["min_data_in_leaf"] = int(base_min_leaf * min_leaf_mult)
                dummy_fwd_ret = bounded_target.astype(np.float32, copy=True)
                dummy_fwd_ret_norm = bounded_target.astype(np.float32, copy=True)
                for side in ["long", "short"]:
                    tprint(
                        f"\n--- Running GLOBAL STEP2 {side.upper()} side for {target_name} @ H{horizon} ---"
                    )
                    side_results = run_side_pipeline(
                        side=side,
                        data=data,
                        feature_dict=feature_dict,
                        fwd_ret=dummy_fwd_ret,
                        fwd_ret_norm=dummy_fwd_ret_norm,
                        cfg=horizon_cfg,
                        folds=folds,
                        root_output_dir=horizon_target_dir,
                        target_name=target_name,
                        horizon=horizon,
                        bounded_target=bounded_target,
                        bounded_target_surprisal=bounded_target_surprisal,
                    )
                    results_by_target_horizon[
                        (target_name, horizon, side)
                    ] = side_results
                    if not side_results["stage_a"].empty:
                        stage_a_with_prov = side_results["stage_a"].copy()
                        stage_a_with_prov["source_target"] = target_name
                        stage_a_with_prov["source_horizon"] = horizon
                        all_registries.append(stage_a_with_prov)

    if triad_run_step == "step1":
        tprint("TRIAD STEP1 COMPLETE")
        tprint(
            "Stored stage_a_context step1 checkpoints for all processed slices. "
            "No post-dedup selection was run."
        )
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
        result_df = result_df.sort_values(sort_cols, ascending=[False, False])

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

    if (
        directional_mean_ret > (required_hurdle + ranking_excess)
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
    hurdle_excess = rule.get("hurdle_excess", np.nan)
    is_structurally_sound = rule.get("is_structurally_sound", False)
    sign_consistency = rule.get("sign_consistency", 0.0)
    trade_path_quality_score = rule.get("trade_path_quality_score", np.nan)

    if not np.isfinite(directional_mean_ret):
        directional_mean_ret = -np.inf
    if not np.isfinite(hurdle_excess):
        hurdle_excess = -np.inf

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

    # Check 4: Hurdle excess
    # Keep this as a diagnostic only; it should not drive rejection.
    hurdle_check = hurdle_excess > 0.0
    diagnostics["checks"]["hurdle_excess"] = {
        "value": hurdle_excess,
        "threshold": 0.0,
        "passed": hurdle_check,
    }

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
        diagnostics["failures"].append(f"structural_soundness_fail: {reason}")

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
        test_keys_set = set(TEST_FEATURE_KEYS)
        ridge_feats = [
            i for i, m in enumerate(self.metadata) if m.source_name in test_keys_set
        ]
        cached = np.asarray(ridge_feats, dtype=np.int32)
        self._ridge_feature_indices_cache = cached
        return cached

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
    def _compute_oof_learnability_score(
        oof_preds: np.ndarray,
        y: np.ndarray,
        coverage_denominator: np.ndarray,
        min_predicted_points: int = 100,
    ) -> Tuple[float, float]:
        predicted_mask = np.isfinite(oof_preds) & np.isfinite(y)
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
        predicted_mask = np.isfinite(oof_preds) & np.isfinite(y)
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

    def assess_rules(
        self,
        registry: pd.DataFrame,
        X: np.ndarray,
        data: pd.DataFrame,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        fold_health_summary: Optional[Dict[str, Any]] = None,
        step_mode: str = "full",
        step1_checkpoint_dir: Optional[Path] = None,
        checkpoint_output_dir: Optional[Path] = None,
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

        # TBM horizon should match assessment horizon + 2 bars
        assessment_horizon = int(self.cfg.get("horizon", 100))
        horizon = assessment_horizon + 2

        # Use the maximum horizon in the current run as reference for TP/SL scaling
        horizons = self.cfg.get("triad_horizons", [100])
        h_ref = max(horizons) if horizons else 100

        # Scale TP and SL by sqrt(H / H_ref)
        scale_factor = np.sqrt(horizon / h_ref)

        # Base TP/SL multipliers (at reference horizon)
        base_tp_atr = float(self.cfg.get("tbm_tp_atr", 1.25))
        base_sl_atr = float(self.cfg.get("tbm_sl_atr", 0.50))

        # Scaled TP/SL for current horizon
        tp_atr = base_tp_atr * scale_factor
        sl_atr = base_sl_atr * scale_factor

        tprint(
            f"TBM Configuration: H={horizon} bars (assessment H={assessment_horizon}), "
            f"TP={tp_atr:.3f} ATR, SL={sl_atr:.3f} ATR, scale_factor={scale_factor:.3f} (h_ref={h_ref})"
        )

        tp_f_long, sl_f_long, to_f_long = compute_tbm_outcomes_per_symbol(
            data=data,
            horizon=horizon,
            tp_atr=tp_atr,
            sl_atr=sl_atr,
            side="long",
        )

        tp_f_short, sl_f_short, to_f_short = compute_tbm_outcomes_per_symbol(
            data=data,
            horizon=horizon,
            tp_atr=tp_atr,
            sl_atr=sl_atr,
            side="short",
        )

        min_oof_coverage = float(self.cfg.get("learnability_min_oof_coverage", 0.05))
        min_avg_trades = float(self.cfg.get("min_avg_trades_per_day_10_symbols", 0.1))
        min_sign_consistency = float(self.cfg.get("min_sign_consistency", 0.0))
        min_mean_target_value = float(self.cfg.get("min_mean_target_value", 0.003))

        target_ret_by_side = {"long": fwd_ret, "short": -fwd_ret}
        mean_ret_global_by_side = {
            "long": float(np.nanmean(fwd_ret)),
            "short": float(np.nanmean(-fwd_ret)),
        }
        feature_to_regime_family = {
            m.feature_name: m.regime_family
            for m in self.metadata
            if getattr(m, "regime_family", None)
        }
        tbm_side_map = {
            "long": (tp_f_long, sl_f_long, to_f_long),
            "short": (tp_f_short, sl_f_short, to_f_short),
        }
        baseline_cache: Dict[str, Dict[str, float]] = {}

        mask_cache: Dict[str, np.ndarray] = {}
        cheap_stats_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
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

        _batch_fill_cheap_stats_cache("long", target_ret_by_side["long"])
        _batch_fill_cheap_stats_cache("short", target_ret_by_side["short"])

        def _get_or_compute_cheap_stats(
            canonical_key: str, side: str, mask: np.ndarray
        ) -> Dict[str, float]:
            cache_key = (canonical_key, side)
            cached_stats = cheap_stats_cache.get(cache_key)
            if cached_stats is not None:
                return cached_stats
            support_pct = float(np.mean(mask))
            return {
                "support_pct": support_pct,
                "support_ok": float(support_min <= support_pct <= support_max),
                "support_score": 1.0,
                "avg_trades": self._compute_avg_trades_per_day(mask, total_symbol_days),
                "density_dispersion": 0.0,
                "tail_ratio": 1.0,
                "mae": 0.0,
                "mfe": 0.0,
                "mean_ret_global": mean_ret_global_by_side[side],
                "mean_ret_mask": float(np.nanmean(target_ret_by_side[side][mask])),
                "std_ret_mask": float(
                    np.nanstd(_clip_returns(target_ret_by_side[side][mask]))
                ),
                "ret_uplift": float(np.nanmean(target_ret_by_side[side][mask]))
                - mean_ret_global_by_side[side],
                "sign_consistency": 0.5,
            }

        def _infer_regime_family_combo(canonical_key: str) -> str:
            families = sorted(
                {
                    feature_to_regime_family[name]
                    for name in extract_feature_names_from_key(canonical_key)
                    if name in feature_to_regime_family
                }
            )
            return "|".join(families) if families else "none"

        if step_mode == "step2":
            if step1_checkpoint_dir is None:
                raise ValueError("step2 requires step1_checkpoint_dir")
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
                if side not in target_ret_by_side:
                    side = "long"

                horizon_raw = reg_source_horizon[row_idx]
                try:
                    horizon_key = int(horizon_raw) if pd.notna(horizon_raw) else -1
                except (TypeError, ValueError):
                    horizon_key = -1

                bucket_key = (side, horizon_key)

                if canonical_key in seen_keys_per_bucket[bucket_key]:
                    continue
                seen_keys_per_bucket[bucket_key].add(canonical_key)
                registry_key_to_row[canonical_key] = row_idx

                mask = mask_cache[canonical_key]
                if np.sum(mask) < 20:
                    continue

                cheap = _get_or_compute_cheap_stats(canonical_key, side, mask)
                if not bool(cheap["support_ok"]):
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

            # Map so that the "best" raw value always becomes 2.0, and the "worst" raw value becomes 1.0.
            if higher_is_better:
                return 1.0 + (clipped - min_val) / (max_val - min_val)
            else:
                return 1.0 + (max_val - clipped) / (max_val - min_val)

        bucket_protected_keys: Dict[Tuple[str, int, str], set[str]] = {}
        bucket_cheap_ranks: Dict[
            Tuple[str, int, str], Dict[str, float]
        ] = collections.defaultdict(dict)

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

            # For lower-is-better metrics, normalizing them this way maps their best (lowest) value to 2.0
            n_tail = _normalize_to_1_2(tail_arr, higher_is_better=False)
            n_dens = _normalize_to_1_2(dens_arr, higher_is_better=False)

            # Because lower-is-better metrics are mapped such that 2.0 is BEST, we must put them
            # in the numerator (multiplying them) rather than the denominator to maintain a monotonic score.
            # Using directional_mean_ret directly instead of sqrt(directional_mean_ret).
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
        bucket_hurdle_values_surviving: Dict[
            Tuple[str, int], List[Tuple[str, float]]
        ] = collections.defaultdict(list)
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

        bucket_density_values_surviving: Dict[
            Tuple[str, int], List[Tuple[str, float]]
        ] = collections.defaultdict(list)
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

        bucket_tail_values_surviving: Dict[
            Tuple[str, int], List[Tuple[str, float]]
        ] = collections.defaultdict(list)
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

        bucket_path_quality_surviving: Dict[
            Tuple[str, int, str], List[Tuple[str, float]]
        ] = collections.defaultdict(list)
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

        cheap_gate_rows: Dict[
            Tuple[str, int], List[Tuple[float, str]]
        ] = collections.defaultdict(list)
        cheap_gate_result: Dict[Tuple[Tuple[str, int], str], Tuple[bool, str]] = {}

        seen_keys_for_cheap_gate = set()

        for pre_row in registry.to_dict("records"):
            canonical_key = str(pre_row["canonical_key"])
            side = str(pre_row["side"])
            if side not in target_ret_by_side:
                side = "long"

            horizon_raw = pre_row.get("source_horizon", -1)
            try:
                horizon_key = int(horizon_raw) if pd.notna(horizon_raw) else -1
            except (TypeError, ValueError):
                horizon_key = -1

            bucket_key = (side, horizon_key)

            # Ensure we only process each canonical key once per bucket_key since target metrics
            # are inherently agnostic to triad targets
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

            cheap = _get_or_compute_cheap_stats(canonical_key, side, mask)
            sign_consistency = float(cheap["sign_consistency"])

            rejected = False
            rejection_reason = ""
            if not bool(cheap["support_ok"]):
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
            elif float(cheap["tail_ratio"]) > bucket_tail_cap.get(bucket_key, np.inf):
                rejected, rejection_reason = True, "high_tail_risk_top_decile"

            if (
                rejected
                and canonical_key in bucket_protected_keys.get(bucket_key, set())
                and rejection_reason
                in {
                    "low_path_quality_floor",
                    "low_stability_floor",
                }
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

        OVERLAP_THRESHOLD = 0.95
        SUPPORT_RATIO_MIN = 0.70
        DEDUP_SUBSAMPLE_SIZE = 10000
        DEDUP_STOP_TARGET = int(self.cfg.get("overlap_dedup_stop_target", 80))
        eps = 1e-8

        surviving_keys_by_bucket: Dict[
            Tuple[str, int, str], List[str]
        ] = collections.defaultdict(list)
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

            registry_key_to_row: Dict[str, int] = {}
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
                # Pre-allocate arrays for better performance
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

                # Build a dense subsample matrix directly. The current bucket sizes are
                # small enough that this is safer than hand-assembling sparse COO indices.
                context_matrix = np.column_stack(
                    [c.astype(np.int32, copy=False) for c in contexts]
                )
                intersections = context_matrix.T @ context_matrix
                n_rules = len(surviving_keys)
                sub_supports = np.diag(intersections).astype(float)

                # Store matrices for soft F1/Dice penalty downstream
                stage_a_matrices[bucket_key] = {
                    "key_to_idx": {k: idx for idx, k in enumerate(surviving_keys)},
                    "intersections": intersections,
                    "supports": sub_supports,
                    "n_subsample": n_subsample,
                }

                threshold_ladder = [0.975, 0.95, 0.925, 0.90, 0.875, 0.85]
                final_keep = [True] * n_rules
                initial_n_rules = n_rules
                prev_keep = np.array(final_keep, dtype=bool)

                # Convert lists to numpy arrays for vectorized operations
                mean_returns_arr = np.array(mean_returns, dtype=np.float32)
                std_returns_arr = np.array(std_returns, dtype=np.float32)
                supports_arr = np.array(supports, dtype=np.float32)
                sub_supports_arr = np.array(sub_supports, dtype=np.float32)

                tprint(
                    f"Stage A: Starting overlap deduplication threshold ladder for bucket {bucket_key}..."
                )

                for threshold in threshold_ladder:
                    tprint(
                        f"Stage A: Overlap dedup bucket {bucket_key} at threshold {threshold:.2f} "
                        f"(active={int(np.sum(final_keep))}/{n_rules})"
                    )
                    prev_keep = np.asarray(final_keep, dtype=bool).copy()
                    keep = np.ones(n_rules, dtype=bool)

                    # Vectorized overlap checking
                    # Create mask for pairs with same sign returns
                    sign_mask = (
                        mean_returns_arr[:, None] * mean_returns_arr[None, :] >= 0
                    )

                    # Create mask for valid overlaps (intersection >= 1)
                    valid_inter_mask = intersections >= 1

                    # Compute overlap matrix
                    min_supports = np.minimum(
                        sub_supports_arr[:, None], sub_supports_arr[None, :]
                    )
                    overlap_matrix = intersections / np.maximum(min_supports, 1.0)

                    # Compute support ratio matrix
                    supp_ratio_matrix = np.minimum(
                        supports_arr[:, None], supports_arr[None, :]
                    ) / np.maximum(
                        np.maximum(supports_arr[:, None], supports_arr[None, :]), 1e-9
                    )

                    # Create overlap threshold mask
                    overlap_mask = overlap_matrix > threshold
                    supp_ratio_mask = supp_ratio_matrix > SUPPORT_RATIO_MIN

                    # Combined mask for pairs to check
                    check_mask = (
                        sign_mask & valid_inter_mask & overlap_mask & supp_ratio_mask
                    )

                    # Only check upper triangular (i < j)
                    tri_mask = np.triu(np.ones((n_rules, n_rules), dtype=bool), k=1)
                    check_mask = check_mask & tri_mask

                    # Get indices of pairs to check
                    i_indices, j_indices = np.where(check_mask)

                    if len(i_indices) == 0:
                        continue

                    # Vectorized rule quality computation
                    rq_i = np.abs(mean_returns_arr[i_indices]) / (
                        std_returns_arr[i_indices] + eps
                    )
                    rq_j = np.abs(mean_returns_arr[j_indices]) / (
                        std_returns_arr[j_indices] + eps
                    )
                    rq_i = np.where(
                        std_returns_arr[i_indices] > eps,
                        rq_i,
                        np.abs(mean_returns_arr[i_indices]),
                    )
                    rq_j = np.where(
                        std_returns_arr[j_indices] > eps,
                        rq_j,
                        np.abs(mean_returns_arr[j_indices]),
                    )

                    # Determine which to drop
                    drop_i = (rq_i < rq_j) & keep[i_indices]
                    drop_j = (rq_j < rq_i) & keep[j_indices]
                    drop_both = (rq_i == rq_j) & keep[i_indices] & keep[j_indices]

                    # Apply drops
                    keep[i_indices[drop_i]] = False
                    keep[j_indices[drop_j]] = False

                    # For ties, drop the one with lower support
                    if np.any(drop_both):
                        tie_i = i_indices[drop_both]
                        tie_j = j_indices[drop_both]
                        drop_lower_support = supports_arr[tie_i] < supports_arr[tie_j]
                        keep[tie_i[drop_lower_support]] = False
                        keep[tie_j[~drop_lower_support]] = False

                    candidate_keep = np.asarray(final_keep, dtype=bool) & keep
                    candidate_count = int(np.sum(candidate_keep))
                    prev_count = int(np.sum(prev_keep))

                    if candidate_count <= DEDUP_STOP_TARGET:
                        if prev_count > DEDUP_STOP_TARGET:
                            ranked_prev = sorted(
                                np.where(prev_keep)[0].tolist(),
                                key=lambda i: bucket_cheap_ranks.get(
                                    bucket_key, {}
                                ).get(surviving_keys[i], -np.inf),
                                reverse=True,
                            )
                            selected_prev = set(ranked_prev[:DEDUP_STOP_TARGET])
                            final_keep = np.array(
                                [i in selected_prev for i in range(n_rules)], dtype=bool
                            )
                            tprint(
                                f"Stage A: Overlap dedup bucket {bucket_key} would undershoot target "
                                f"at threshold {threshold:.2f}; trimming previous survivor set "
                                f"from {prev_count} to {DEDUP_STOP_TARGET}"
                            )
                        else:
                            final_keep = candidate_keep
                            tprint(
                                f"Stage A: Overlap dedup bucket {bucket_key} reached <={DEDUP_STOP_TARGET} survivors at "
                                f"threshold {threshold:.2f}"
                            )
                        break

                    final_keep = candidate_keep

                surviving_indices = [i for i, k in enumerate(final_keep) if k]
                if len(surviving_indices) > DEDUP_STOP_TARGET:

                    def _score(i):
                        rq = (
                            abs(mean_returns[i]) / (std_returns[i] + eps)
                            if std_returns[i] > eps
                            else abs(mean_returns[i])
                        )
                        gain_b = min(max(np.sqrt(max(gains[i], 0.0)), 0.0), 1.0)
                        sign_b = min(max(sign_consistencies[i], 0.0), 1.0)
                        rq_b = min(max(rq, 0.0), 1.0)
                        return gain_b * sign_b * rq_b

                    surviving_indices.sort(key=_score, reverse=True)
                    surviving_indices = surviving_indices[:DEDUP_STOP_TARGET]

                final_surviving_set = {surviving_keys[i] for i in surviving_indices}

                for i, k in enumerate(surviving_keys):
                    if k not in final_surviving_set:
                        cheap_gate_result[(bucket_key, k)] = (
                            True,
                            "deduplicated_overlap",
                        )
                        n_dedup_rejected += 1

                final_surviving_keys = [surviving_keys[i] for i in surviving_indices]

                tprint(
                    f"Stage A: Overlap deduplication complete for bucket {bucket_key} - {len(final_surviving_keys)} rules survived (from {initial_n_rules})"
                )
                tprint(
                    f"Stage A: Overlap dedup bucket summary {bucket_key} - "
                    f"input_rules={initial_n_rules} output_rules={len(final_surviving_keys)} "
                    f"rejected={initial_n_rules - len(final_surviving_keys)} "
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

            cheap_gate_rows_deduped: Dict[
                Tuple[str, int], List[Tuple[float, str]]
            ] = collections.defaultdict(list)
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

            if step_mode == "step1":
                tprint("Stage A: Step1 complete. Skipping post-dedup step2 assessment.")
                return pd.DataFrame()

        # Cache baseline learnability once per side after cheap structural filtering.
        for side, target_ret in target_ret_by_side.items():
            baseline_metrics = self._compute_baseline_auc(X, target_ret, folds)
            baseline_cache[side] = {
                "global_auc": float(baseline_metrics["global_auc"])
                if np.isfinite(baseline_metrics["global_auc"])
                else np.nan,
                "global_roc_auc": float(baseline_metrics["global_roc_auc"])
                if np.isfinite(baseline_metrics["global_roc_auc"])
                else np.nan,
                "global_pr_auc": float(baseline_metrics["global_pr_auc"])
                if np.isfinite(baseline_metrics["global_pr_auc"])
                else np.nan,
                "global_cov": float(baseline_metrics["global_cov"]),
                "global_entropy": float(self._compute_entropy(target_ret)),
            }
            if np.nanstd(target_ret) < 1e-9:
                tprint(
                    f"WARNING: Root cause for degenerate metrics: {side} target has zero variance!"
                )

        max_ridge_candidates_per_bucket = int(
            self.cfg.get("max_ridge_candidates_per_bucket", 80)
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
            f"Stage A: Starting ridge regression selection for {len(cheap_gate_rows)} buckets..."
        )

        self.bucket_ridge_keys = {}
        for bucket_key, entries in cheap_gate_rows.items():
            side = bucket_key[0]
            baseline_oof_coverage = float(baseline_cache[side]["global_cov"])
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
            for _, canonical_key in entries:
                combo = _infer_regime_family_combo(canonical_key)
                combo_share = family_counts.get(combo, 0) / float(total_bucket)
                underrep = max(target_share_family - combo_share, 0.0) / max(
                    target_share_family, 1e-9
                )
                bonus = min(
                    family_rarity_bonus_cap,
                    family_rarity_bonus_strength * underrep,
                )
                family_rarity_bonus_by_key[bucket_key][canonical_key] = float(bonus)
            if family_counts:
                top_combo, top_count = max(family_counts.items(), key=lambda kv: kv[1])
                tprint(
                    f"Stage A: Family rarity bonus bucket {bucket_key} "
                    f"combos={len(family_counts)} top_combo={top_combo} top_count={top_count}"
                )

        for bucket_key, entries in bucket_ridge_rows.items():
            if not entries:
                continue

            surviving_keys = [k for _, k in entries]
            surviving_ranks = [r for r, _ in entries]
            family_bonus_arr = np.array(
                [
                    family_rarity_bonus_by_key.get(bucket_key, {}).get(k, 0.0)
                    for k in surviving_keys
                ],
                dtype=float,
            )

            matrices = stage_a_matrices.get(bucket_key)
            if not matrices or len(surviving_keys) <= max_ridge_candidates_per_bucket:
                entries.sort(
                    key=lambda item: (
                        item[0]
                        + family_rarity_bonus_by_key.get(bucket_key, {}).get(
                            item[1], 0.0
                        )
                    ),
                    reverse=True,
                )
                self.bucket_ridge_keys[bucket_key] = {
                    key for _, key in entries[: max(max_ridge_candidates_per_bucket, 0)]
                }
                continue

            key_to_idx = matrices["key_to_idx"]
            intersections = matrices["intersections"]
            supports = matrices["supports"]
            n_subsample_bucket = matrices.get("n_subsample", 10000.0)

            # Filter matrices for valid entries
            idx_list = []
            valid_keys = []
            valid_ranks = []
            for i, k in enumerate(surviving_keys):
                if k in key_to_idx:
                    idx_list.append(key_to_idx[k])
                    valid_keys.append(k)
                    valid_ranks.append(surviving_ranks[i])

            if len(valid_keys) <= max_ridge_candidates_per_bucket:
                self.bucket_ridge_keys[bucket_key] = set(valid_keys)
                continue

            sub_intersections = intersections[np.ix_(idx_list, idx_list)]
            sub_supports_arr = supports[idx_list]

            # Compute Support Weight Multiplier
            s_arr = sub_supports_arr / n_subsample_bucket
            w_mult_arr = np.ones(len(valid_keys), dtype=float)

            for i, s in enumerate(s_arr):
                if s < (center - half_width):
                    w = 1.0 - penalty_strength * (center - half_width - s) / (
                        center - half_width
                    )
                elif s < center:
                    w = 1.0 + boost_strength * (s - (center - half_width)) / half_width
                elif s < (center + half_width):
                    w = 1.0 + boost_strength * ((center + half_width) - s) / half_width
                else:
                    w = 1.0 - penalty_strength * (s - (center + half_width)) / (
                        center + half_width
                    )

                # Point 12: Broaden clip bounds to allow meaningful boosting
                w_mult_arr[i] = np.clip(w, 0.1, 2.0)

            supp_i = sub_supports_arr[:, None]
            supp_j = sub_supports_arr[None, :]

            # Compute F1/Dice matrix
            f1_overlap_matrix = 2.0 * sub_intersections / (supp_i + supp_j + 1e-9)

            # Compute Support Ratio matrix to ignore similarities where support sizes differ greatly
            supp_ratio_matrix = np.minimum(supp_i, supp_j) / np.maximum(
                supp_i, supp_j + 1e-9
            )

            # Effective F1 (Same-side is naturally guaranteed because bucket keys split by side)
            effective_f1_overlap_matrix = f1_overlap_matrix * (
                supp_ratio_matrix >= support_ratio_min
            )

            # Normalize cheap ranks to [0.05, 1.0] positive bounded range
            raw_cr = np.array(valid_ranks)
            min_cr = np.min(raw_cr)
            max_cr = np.max(raw_cr)
            if max_cr - min_cr < 1e-9:
                norm_cr = np.full(len(valid_keys), 1.0)
            else:
                norm_cr = 0.05 + 0.95 * (raw_cr - min_cr) / (max_cr - min_cr)

            # Diversified Top-K Greedy Selection
            # Pre-allocate arrays for better performance
            max_candidates = min(max_ridge_candidates_per_bucket, len(valid_keys))
            selected_indices = np.zeros(max_candidates, dtype=np.int32)
            remaining_mask = np.ones(len(valid_keys), dtype=bool)

            # Select highest support-weighted cheap_rank first
            initial_scores = norm_cr * (1.0 + w_mult_arr) * (1.0 + family_bonus_arr)
            best_idx = int(np.argmax(initial_scores))
            selected_indices[0] = best_idx
            remaining_mask[best_idx] = False
            n_selected = 1

            while n_selected < max_candidates and remaining_mask.any():
                remaining_indices = np.where(remaining_mask)[0]
                if len(remaining_indices) == 0:
                    break

                # Vectorized overlap computation for all remaining candidates
                selected_so_far = selected_indices[:n_selected]
                max_f1_overlap = np.max(
                    effective_f1_overlap_matrix[remaining_indices][:, selected_so_far],
                    axis=1,
                )
                overlap_excess = np.maximum(0.0, max_f1_overlap - overlap_free_zone) / (
                    1.0 - overlap_free_zone + 1e-9
                )
                adjusted_scores = (
                    norm_cr[remaining_indices] ** cheap_rank_exponent
                ) * ((1.0 - overlap_excess) ** overlap_penalty_exponent)
                final_ranking = (
                    adjusted_scores
                    * (1.0 + w_mult_arr[remaining_indices])
                    * (1.0 + family_bonus_arr[remaining_indices])
                )

                best_next_idx = remaining_indices[int(np.argmax(final_ranking))]
                selected_indices[n_selected] = best_next_idx
                remaining_mask[best_next_idx] = False
                n_selected += 1

            self.bucket_ridge_keys[bucket_key] = {
                valid_keys[i] for i in selected_indices
            }

        total_ridge_selected = sum(
            len(keys) for keys in self.bucket_ridge_keys.values()
        )
        tprint(
            f"Stage A: Ridge regression selection complete - {total_ridge_selected} rules selected for final assessment"
        )

        final_assessment_start_ts = time.perf_counter()
        tprint(
            f"Stage A: Starting final assessment for {total_ridge_selected} rules..."
        )

        selected_key_set = {
            key for keys in self.bucket_ridge_keys.values() for key in keys
        }
        selected_registry = registry[
            registry["canonical_key"].astype(str).isin(selected_key_set)
        ].copy()
        selected_records = selected_registry.to_dict("records")

        tprint(
            f"Stage A: Final assessment narrowed registry from {len(registry)} to {len(selected_records)} ridge-selected rules"
        )
        tprint(
            "Stage A: Final assessment phase start - "
            f"selected_rules={len(selected_records)} total_registry={len(registry)}"
        )

        assessed_progress = 0
        for row in selected_records:
            assessed_progress += 1
            if assessed_progress == 1 or assessed_progress % 25 == 0:
                tprint(
                    f"Stage A: Final assessment progress {assessed_progress}/{total_ridge_selected} "
                    f"rules"
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
            if np.sum(mask) < 20:
                continue

            # 0. Infrastructure: Component Extraction
            slots = parse_slot_map(
                canonical_key,
                self.cfg.get("slot_order", ("trigger", "location", "regime")),
            )

            side = str(row.get("side", "long"))
            if side not in target_ret_by_side:
                side = "long"
            horizon_raw = row.get("source_horizon", -1)
            try:
                horizon_key = int(horizon_raw) if pd.notna(horizon_raw) else -1
            except (TypeError, ValueError):
                horizon_key = -1
            target_key = str(row.get("source_target", "unknown"))

            # The bucket grouping is determined strictly by side and horizon
            group_bucket_key = (side, horizon_key)

            target_ret = target_ret_by_side[side]
            global_auc = float(baseline_cache[side]["global_auc"])
            global_roc_auc = float(baseline_cache[side]["global_roc_auc"])
            global_pr_auc = float(baseline_cache[side]["global_pr_auc"])
            global_entropy = float(baseline_cache[side]["global_entropy"])
            baseline_oof_coverage = float(baseline_cache[side]["global_cov"])

            # 1. Triple Barrier
            rule_tp_f, rule_sl_f, rule_to_f = tbm_side_map[side]

            tbm_metrics = self._compute_tbm_metrics(
                mask, rule_tp_f, rule_sl_f, rule_to_f, target_ret
            )

            # 2-6. Cached cheap stats (no Ridge work)
            cheap = _get_or_compute_cheap_stats(canonical_key, side, mask)

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
            lift = np.nan
            entropy_red = np.nan
            ridge_trade_metrics: Dict[str, Any] = {
                "ridge_profitable_top_pct": np.nan,
                "ridge_profitable_score_threshold": np.nan,
                "ridge_profitable_trade_count": 0,
                "ridge_profitable_trades_per_day": 0.0,
                "ridge_profitable_win_rate": np.nan,
                "ridge_profitable_avg_net_ret": np.nan,
                "ridge_profitable_avg_pnl_per_day": np.nan,
                "ridge_profitable_total_net_pnl": np.nan,
                "ridge_best_top_pct": np.nan,
                "ridge_best_total_net_pnl": np.nan,
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
                    ridge_details = self._compute_subset_ridge_details(
                        X, target_ret, mask, folds
                    )
                    mask_auc = float(ridge_details["subset_auc"])
                    mask_roc_auc = float(ridge_details["subset_roc_auc"])
                    mask_pr_auc = float(ridge_details["subset_pr_auc"])
                    subset_oof_coverage = float(ridge_details["coverage"])
                    ridge_trade_metrics = self._compute_ranked_ridge_trade_metrics(
                        data=data,
                        directional_returns=target_ret,
                        mask=mask,
                        oof_preds=np.asarray(
                            ridge_details["oof_preds"], dtype=np.float32
                        ),
                    )
                    tprint(
                        f"Stage A: Ridge learnability done {assessed_progress}/{total_ridge_selected} "
                        f"key={canonical_key[:120]} "
                        f"mask_oof_corr={mask_auc if np.isfinite(mask_auc) else np.nan:.6f} "
                        f"coverage={subset_oof_coverage:.4f} "
                        f"elapsed={time.time() - ridge_start:.2f}s"
                    )
                    if np.isfinite(mask_auc) and np.isfinite(global_auc):
                        lift = mask_auc - global_auc
                else:
                    subset_oof_coverage = float(np.mean(mask))
                    lift = 0.0  # Neutral lift
                    rejected, rejection_reason = True, "not_in_top_ridge_candidates"

                mask_entropy = self._compute_entropy(target_ret[mask])
                entropy_red = 1.0 - (mask_entropy / (global_entropy + 1e-9))
                if subset_oof_coverage < min_oof_coverage:
                    rejected, rejection_reason = (
                        True,
                        "insufficient_subset_oof_coverage",
                    )
                elif not np.isfinite(lift):
                    rejected, rejection_reason = True, "missing_learnability"
                elif run_ridge and lift < 0.01:  # Lower threshold for lift (was 1.10)
                    rejected, rejection_reason = True, "low_lift"

            # 8. Event-based Expected Value
            tp_payoff = tp_atr  # TP payoff in ATR units
            sl_payoff = sl_atr  # SL payoff in ATR units
            timeout_payoff = mean_ret_mask  # Average return for timeouts

            ev_per_event = (
                tbm_metrics["tp_rate"] * tp_payoff
                - tbm_metrics["sl_rate"] * sl_payoff
                + tbm_metrics["timeout_rate"] * timeout_payoff
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

            # 9. Final Regime Score
            regime_score = (
                0.30 * cheap_rank
                + 0.20 * lift
                + 0.20 * ret_uplift
                + 0.20 * ev_per_event
                + 0.10 * (mask_auc if np.isfinite(mask_auc) else 0.0)
                + family_rarity_bonus
            )

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
                    "trigger": slots.get("trigger", "*"),
                    "location": slots.get("location", "*"),
                    "regime": slots.get("regime", "*"),
                    "regime_score": regime_score,
                    "is_structurally_sound": not rejected,
                    "rejection_reason": rejection_reason,
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
                    "lift": lift,
                    "learn_eff_ratio": np.nan,  # Deprecated - same as lift
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
                    "ridge_profitable_top_pct": ridge_trade_metrics[
                        "ridge_profitable_top_pct"
                    ],
                    "ridge_profitable_score_threshold": ridge_trade_metrics[
                        "ridge_profitable_score_threshold"
                    ],
                    "ridge_profitable_trade_count": ridge_trade_metrics[
                        "ridge_profitable_trade_count"
                    ],
                    "ridge_profitable_trades_per_day": ridge_trade_metrics[
                        "ridge_profitable_trades_per_day"
                    ],
                    "ridge_profitable_win_rate": ridge_trade_metrics[
                        "ridge_profitable_win_rate"
                    ],
                    "ridge_profitable_avg_net_ret": ridge_trade_metrics[
                        "ridge_profitable_avg_net_ret"
                    ],
                    "ridge_profitable_avg_pnl_per_day": ridge_trade_metrics[
                        "ridge_profitable_avg_pnl_per_day"
                    ],
                    "ridge_profitable_avg_pnl_per_active_symbol_day": ridge_trade_metrics[
                        "ridge_profitable_avg_pnl_per_active_symbol_day"
                    ],
                    "ridge_profitable_daily_pnl_std": ridge_trade_metrics[
                        "ridge_profitable_daily_pnl_std"
                    ],
                    "ridge_profitable_daily_sortino": ridge_trade_metrics[
                        "ridge_profitable_daily_sortino"
                    ],
                    "ridge_profitable_avg_position_weight": ridge_trade_metrics[
                        "ridge_profitable_avg_position_weight"
                    ],
                    "ridge_profitable_total_net_pnl": ridge_trade_metrics[
                        "ridge_profitable_total_net_pnl"
                    ],
                    "ridge_best_top_pct": ridge_trade_metrics["ridge_best_top_pct"],
                    "ridge_best_total_net_pnl": ridge_trade_metrics[
                        "ridge_best_total_net_pnl"
                    ],
                    "ridge_best_score_threshold": ridge_trade_metrics[
                        "ridge_best_score_threshold"
                    ],
                    "ridge_best_trade_count": ridge_trade_metrics[
                        "ridge_best_trade_count"
                    ],
                    "ridge_best_trades_per_day": ridge_trade_metrics[
                        "ridge_best_trades_per_day"
                    ],
                    "ridge_best_win_rate": ridge_trade_metrics["ridge_best_win_rate"],
                    "ridge_best_avg_net_ret": ridge_trade_metrics[
                        "ridge_best_avg_net_ret"
                    ],
                    "ridge_best_avg_pnl_per_day": ridge_trade_metrics[
                        "ridge_best_avg_pnl_per_day"
                    ],
                    "ridge_best_avg_pnl_per_active_symbol_day": ridge_trade_metrics[
                        "ridge_best_avg_pnl_per_active_symbol_day"
                    ],
                    "ridge_best_daily_pnl_std": ridge_trade_metrics[
                        "ridge_best_daily_pnl_std"
                    ],
                    "ridge_best_daily_sortino": ridge_trade_metrics[
                        "ridge_best_daily_sortino"
                    ],
                    "ridge_best_avg_position_weight": ridge_trade_metrics[
                        "ridge_best_avg_position_weight"
                    ],
                    "ridge_midpoint_top_pct": ridge_trade_metrics[
                        "ridge_midpoint_top_pct"
                    ],
                    "ridge_midpoint_score_threshold": ridge_trade_metrics[
                        "ridge_midpoint_score_threshold"
                    ],
                    "ridge_midpoint_trade_count": ridge_trade_metrics[
                        "ridge_midpoint_trade_count"
                    ],
                    "ridge_midpoint_trades_per_day": ridge_trade_metrics[
                        "ridge_midpoint_trades_per_day"
                    ],
                    "ridge_midpoint_win_rate": ridge_trade_metrics[
                        "ridge_midpoint_win_rate"
                    ],
                    "ridge_midpoint_avg_net_ret": ridge_trade_metrics[
                        "ridge_midpoint_avg_net_ret"
                    ],
                    "ridge_midpoint_avg_pnl_per_day": ridge_trade_metrics[
                        "ridge_midpoint_avg_pnl_per_day"
                    ],
                    "ridge_midpoint_avg_pnl_per_active_symbol_day": ridge_trade_metrics[
                        "ridge_midpoint_avg_pnl_per_active_symbol_day"
                    ],
                    "ridge_midpoint_daily_pnl_std": ridge_trade_metrics[
                        "ridge_midpoint_daily_pnl_std"
                    ],
                    "ridge_midpoint_daily_sortino": ridge_trade_metrics[
                        "ridge_midpoint_daily_sortino"
                    ],
                    "ridge_midpoint_avg_position_weight": ridge_trade_metrics[
                        "ridge_midpoint_avg_position_weight"
                    ],
                    "ridge_midpoint_total_net_pnl": ridge_trade_metrics[
                        "ridge_midpoint_total_net_pnl"
                    ],
                    "learnability_step_c_score": float(
                        row.get("learnability_step_c_score", np.nan)
                    ),
                    "production_classification": production_classification,
                    "classification_diagnostics": json.dumps(
                        classification_diagnostics
                    ),
                    "rule_type_class": rule_type_class,
                }
            )

        assessment_df = pd.DataFrame(assessment_results)
        if assessment_df.empty:
            tprint(f"Stage A: Final assessment complete - no rules passed all gates")
            tprint(
                "Stage A: Final assessment phase end - "
                f"assessed=0 accepted=0 rejected=0 elapsed={time.perf_counter() - final_assessment_start_ts:.2f}s"
            )
            return assessment_df

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
                        f"  - {row.get('canonical_key', '<unknown>')}: {float(regime_score):.3f}"
                    )
                else:
                    tprint(
                        f"  - {row.get('canonical_key', '<unknown>')}: {regime_score}"
                    )

        return assessment_df

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

    def _compute_subset_ridge_details(self, X, fwd_ret, mask, folds) -> Dict[str, Any]:
        """Compute Ridge OOF details for a subset of data defined by mask."""
        if not np.any(mask):
            return {
                "subset_auc": np.nan,
                "subset_roc_auc": np.nan,
                "subset_pr_auc": np.nan,
                "coverage": 0.0,
                "oof_preds": np.full(len(X), np.nan, dtype=np.float32),
                "folds_used": 0,
                "folds_skipped": 0,
            }

        ridge_feats = self._get_ridge_feature_indices()
        if ridge_feats.size == 0:
            return {
                "subset_auc": np.nan,
                "subset_roc_auc": np.nan,
                "subset_pr_auc": np.nan,
                "coverage": 0.0,
                "oof_preds": np.full(len(X), np.nan, dtype=np.float32),
                "folds_used": 0,
                "folds_skipped": 0,
            }
        subset_auc_start = time.perf_counter()
        X_ridge = np.asarray(X[:, ridge_feats], dtype=np.float32, order="C")
        y = np.asarray(fwd_ret, dtype=np.float32)
        y[~np.isfinite(fwd_ret)] = np.nan
        (
            is_binary_target,
            min_train_req,
            min_val_req,
            min_pred_points,
        ) = self._ridge_learnability_thresholds(y)

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

            # Filter valid samples (y must be finite, and ALL ridge features must be finite)
            valid_tr = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)
            valid_va = np.isfinite(y_va) & np.all(np.isfinite(X_va), axis=1)

            X_tr_clean = X_tr[valid_tr]
            y_tr_clean = y_tr[valid_tr]
            X_va_clean = X_va[valid_va]
            y_va_clean = y_va[valid_va]
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

            # Subsample to 50% of training data, capped at 50,000 samples
            n_samples = len(X_tr_clean)
            n_subsample = max(20, min(50000, int(n_samples * 0.5)))
            subsample_idx = rng.choice(n_samples, size=n_subsample, replace=False)

            y_tr_subsample_pre = y_tr_clean[subsample_idx]
            pos_mask_pre = y_tr_subsample_pre == 1
            neg_mask_pre = y_tr_subsample_pre == 0

            # Cap positive samples at 5000 and match negative samples to the same index
            if np.sum(pos_mask_pre) > 5000:
                pos_indices = np.where(pos_mask_pre)[0]
                pos_keep = pos_indices[:5000]

                # Maintain original positive/negative ratio within the subset but limit by the positive cutoff
                neg_indices = np.where(neg_mask_pre)[0]
                # Stop including negative samples once we've reached the positive cutoff's relative point in the array
                max_neg_index = (
                    pos_keep[-1] if len(pos_keep) > 0 else len(subsample_idx)
                )
                neg_keep = neg_indices[neg_indices < max_neg_index]

                final_keep_idx = np.sort(np.concatenate([pos_keep, neg_keep]))
                subsample_idx = subsample_idx[final_keep_idx]

            X_tr_subsample = X_tr_clean[subsample_idx]
            y_tr_subsample = y_tr_clean[subsample_idx]

            # Fit Ridge on subsampled data
            fit_start = time.perf_counter()
            model = Ridge(alpha=1.0, solver="auto")
            model.fit(X_tr_subsample, y_tr_subsample)
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
            "coverage": float(coverage),
            "oof_preds": oof_preds,
            "folds_used": int(folds_used),
            "folds_skipped": int(folds_skipped),
        }

    @staticmethod
    def _compute_ranked_ridge_trade_metrics(
        data: pd.DataFrame,
        directional_returns: np.ndarray,
        mask: np.ndarray,
        oof_preds: np.ndarray,
        *,
        round_fee: float = 0.0015,
        min_weight: float = 0.05,
        max_weight: float = 0.15,
        convex_power: float = 2.0,
        eps: float = 1e-9,
        ranked_top_pcts: Sequence[float] = (
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.875,
            0.90,
            0.925,
            0.95,
            0.975,
            1.00,
        ),
    ) -> Dict[str, Any]:
        def _empty_metrics() -> Dict[str, Any]:
            return {
                "top_pct": np.nan,
                "score_threshold": np.nan,
                "trade_count": 0,
                "trades_per_day": 0.0,
                "win_rate": np.nan,
                "avg_net_ret": np.nan,
                "avg_pnl_per_day": np.nan,
                "avg_pnl_per_active_symbol_day": np.nan,
                "daily_pnl_std": np.nan,
                "daily_sortino": np.nan,
                "avg_position_weight": np.nan,
                "total_net_pnl": np.nan,
            }

        valid = (
            mask.astype(bool)
            & np.isfinite(directional_returns)
            & np.isfinite(oof_preds)
        )
        if int(np.sum(valid)) == 0:
            return {
                "ridge_best_top_pct": np.nan,
                "ridge_best_total_net_pnl": np.nan,
                "ridge_profitable": _empty_metrics(),
                "ridge_best": _empty_metrics(),
                "ridge_midpoint": _empty_metrics(),
            }

        scores = np.asarray(oof_preds[valid], dtype=np.float32)
        net_returns = np.asarray(directional_returns[valid], dtype=np.float32) - float(
            round_fee
        )
        symbols = data.loc[valid, "symbol"].astype(str).to_numpy()
        timestamps = pd.to_datetime(
            data.loc[valid, "timestamp"], errors="coerce", utc=True
        )
        valid_ts = timestamps.notna().to_numpy()
        if not np.any(valid_ts):
            return {
                "ridge_best_top_pct": np.nan,
                "ridge_best_total_net_pnl": np.nan,
                "ridge_profitable": _empty_metrics(),
                "ridge_best": _empty_metrics(),
                "ridge_midpoint": _empty_metrics(),
            }

        scores = scores[valid_ts]
        net_returns = net_returns[valid_ts]
        symbols = symbols[valid_ts]
        timestamps = timestamps[valid_ts]
        days = timestamps.dt.floor("D")
        observed_days = max(int(days.nunique()), 1)

        order = np.argsort(scores)[::-1]
        scores_sorted = scores[order]
        net_sorted = net_returns[order]
        symbols_sorted = symbols[order]
        days_sorted = days.to_numpy()[order]

        profitable_metrics: Optional[Dict[str, Any]] = None
        threshold_metrics: Dict[float, Dict[str, Any]] = {}
        best_total_net_pnl = -np.inf
        best_top_pct = np.nan

        for top_pct in ranked_top_pcts:
            k = max(1, int(np.ceil(len(scores_sorted) * float(top_pct))))
            sel_scores = scores_sorted[:k]
            sel_net = net_sorted[:k]
            sel_symbols = symbols_sorted[:k]
            sel_days = days_sorted[:k]
            score_threshold = float(sel_scores[k - 1])
            score_max = float(sel_scores[0]) if k > 0 else score_threshold
            score_denom = max(score_max - score_threshold, eps)
            normalized_scores = np.clip(
                (sel_scores - score_threshold) / score_denom, 0.0, 1.0
            )
            weights = min_weight + (max_weight - min_weight) * (
                normalized_scores**convex_power
            )
            weighted_net = weights.astype(np.float32) * sel_net.astype(np.float32)
            total_net_pnl = float(np.sum(weighted_net))
            avg_net_ret = float(np.mean(weighted_net))
            win_rate = float(np.mean(weighted_net > 0.0))
            trades_per_day = float(k / observed_days)
            day_pnl = pd.Series(weighted_net).groupby(sel_days).sum()
            avg_pnl_per_day = float(day_pnl.mean()) if len(day_pnl) > 0 else np.nan
            daily_pnl_std = float(day_pnl.std(ddof=0)) if len(day_pnl) > 0 else np.nan
            if len(day_pnl) > 0:
                downside = np.minimum(day_pnl.to_numpy(dtype=np.float32), 0.0)
                downside_dev = float(np.sqrt(np.mean(downside**2)))
                daily_sortino = float(avg_pnl_per_day / (downside_dev + eps))
            else:
                daily_sortino = np.nan
            active_symbol_days = max(
                int(
                    pd.DataFrame({"symbol": sel_symbols, "day": sel_days})
                    .drop_duplicates()
                    .shape[0]
                ),
                1,
            )
            avg_pnl_per_active_symbol_day = float(total_net_pnl / active_symbol_days)
            avg_position_weight = (
                float(np.mean(weights)) if len(weights) > 0 else np.nan
            )

            current_metrics = {
                "top_pct": float(top_pct),
                "score_threshold": score_threshold,
                "trade_count": int(k),
                "trades_per_day": trades_per_day,
                "win_rate": win_rate,
                "avg_net_ret": avg_net_ret,
                "avg_pnl_per_day": avg_pnl_per_day,
                "avg_pnl_per_active_symbol_day": avg_pnl_per_active_symbol_day,
                "daily_pnl_std": daily_pnl_std,
                "daily_sortino": daily_sortino,
                "avg_position_weight": avg_position_weight,
                "total_net_pnl": total_net_pnl,
            }
            threshold_metrics[float(top_pct)] = current_metrics

            if total_net_pnl > best_total_net_pnl:
                best_total_net_pnl = total_net_pnl
                best_top_pct = float(top_pct)

            if profitable_metrics is None and avg_net_ret > 0.0:
                profitable_metrics = current_metrics

        if profitable_metrics is None:
            profitable_metrics = _empty_metrics()

        best_metrics = (
            threshold_metrics.get(float(best_top_pct), _empty_metrics())
            if np.isfinite(best_top_pct)
            else _empty_metrics()
        )

        midpoint_metrics = _empty_metrics()
        profitable_top_pct = profitable_metrics.get("top_pct", np.nan)
        if np.isfinite(profitable_top_pct):
            midpoint_top_pct = 0.5 * (float(profitable_top_pct) + 1.0)
            eligible_midpoints = [
                p for p in ranked_top_pcts if float(p) >= midpoint_top_pct
            ]
            midpoint_choice = (
                float(eligible_midpoints[0]) if eligible_midpoints else 1.0
            )
            midpoint_metrics = threshold_metrics.get(midpoint_choice, _empty_metrics())

        out = {
            "ridge_best_top_pct": float(best_top_pct)
            if np.isfinite(best_top_pct)
            else np.nan,
            "ridge_best_total_net_pnl": (
                float(best_total_net_pnl) if np.isfinite(best_total_net_pnl) else np.nan
            ),
        }
        prefix_map = {
            "ridge_profitable": profitable_metrics,
            "ridge_best": best_metrics,
            "ridge_midpoint": midpoint_metrics,
        }
        for prefix, metrics in prefix_map.items():
            for key, value in metrics.items():
                out[f"{prefix}_{key}"] = value
        return out

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
                "global_cov": 0.0,
            }

        X_ridge = np.asarray(X[:, ridge_feats], dtype=np.float32, order="C")
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

            # Subsample to 50% of training data, capped at 50,000 samples
            n_samples = len(X_tr_clean)
            n_subsample = max(20, min(50000, int(n_samples * 0.5)))
            subsample_idx = rng.choice(n_samples, size=n_subsample, replace=False)

            y_tr_subsample_pre = y_tr_clean[subsample_idx]
            pos_mask_pre = y_tr_subsample_pre == 1
            neg_mask_pre = y_tr_subsample_pre == 0

            # Cap positive samples at 5000 and match negative samples to the same index
            if np.sum(pos_mask_pre) > 5000:
                pos_indices = np.where(pos_mask_pre)[0]
                pos_keep = pos_indices[:5000]

                # Maintain original positive/negative ratio within the subset but limit by the positive cutoff
                neg_indices = np.where(neg_mask_pre)[0]
                # Stop including negative samples once we've reached the positive cutoff's relative point in the array
                max_neg_index = (
                    pos_keep[-1] if len(pos_keep) > 0 else len(subsample_idx)
                )
                neg_keep = neg_indices[neg_indices < max_neg_index]

                final_keep_idx = np.sort(np.concatenate([pos_keep, neg_keep]))
                subsample_idx = subsample_idx[final_keep_idx]

            X_tr_subsample = X_tr_clean[subsample_idx]
            y_tr_subsample = y_tr_clean[subsample_idx]

            # Fit Ridge on subsampled data
            model = Ridge(alpha=1.0, solver="auto")
            model.fit(X_tr_subsample, y_tr_subsample)
            preds = model.predict(X_va_clean)

            # Store predictions
            oof_preds[va_idx[valid_va]] = preds

        global_auc, global_cov = self._compute_oof_learnability_score(
            oof_preds, y, np.isfinite(y), min_predicted_points=min_pred_points
        )
        class_metrics = self._compute_oof_classification_metrics(
            oof_preds, y, np.isfinite(y), min_predicted_points=min_pred_points
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
                if "==" not in cond_str:
                    continue
                fname, val_part = cond_str.split("==")
                val = int(val_part)
                # Find matching metadata for feature index
                f_idx = next(
                    m.feature_index for m in self.metadata if m.feature_name == fname
                )
                mask &= X[:, f_idx] == val
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
    max_overlap: float = 0.4,
    max_side_in_top: int = 9,
) -> pd.DataFrame:
    """
    Select top `top_n` diverse rules:
    - Sort by composite_score
    - Ensure top `top_n` has at most `max_side_in_top` of the same side (long/short)
      IF there are enough valid rules of the other side to fill the quota.
    - Ensure jaccard similarity between any two selected rules is <= max_overlap
    """
    if registry.empty:
        return registry

    # Point 5: Reset index to ensure safe integer-based slicing later
    sorted_reg = registry.sort_values("composite_score", ascending=False).reset_index(
        drop=True
    )

    selected_idx = []
    selected_sides = {"long": 0, "short": 0}

    for idx, row in sorted_reg.iterrows():
        if len(selected_idx) >= top_n:
            break

        key = str(row["canonical_key"])
        side = str(row.get("side", "unknown"))
        mask = mask_map.get(key)
        if mask is None:
            continue

        # Check side constraint for the selected rules
        if len(selected_idx) < top_n and side in selected_sides:
            if selected_sides[side] >= max_side_in_top:
                other_side = "short" if side == "long" else "long"
                slots_to_fill = top_n - len(selected_idx)
                valid_other_side = 0

                # Check remaining items
                curr_pos = sorted_reg.index.get_loc(idx)
                remaining_indices = sorted_reg.index[curr_pos + 1 :]

                for rem_idx in remaining_indices:
                    rem_row = sorted_reg.loc[rem_idx]
                    if rem_row.get("side", "unknown") == other_side:
                        rem_mask = mask_map.get(rem_row["canonical_key"])
                        if rem_mask is not None:
                            too_similar = False
                            for s_idx in selected_idx:
                                s_key_inner = sorted_reg.loc[s_idx, "canonical_key"]
                                s_mask = mask_map.get(s_key_inner)
                                if s_mask is not None:
                                    intersection = float(np.sum(rem_mask & s_mask))
                                    union = float(np.sum(rem_mask | s_mask))
                                    jaccard = intersection / union if union > 0 else 0.0
                                    if jaccard > max_overlap:
                                        too_similar = True
                                        break

                            if not too_similar:
                                valid_other_side += 1
                                if valid_other_side >= slots_to_fill:
                                    break

                # If we have enough valid rules of the other side to fill the top_n spots,
                # skip this one. Otherwise, allow the side count to exceed max_side_in_top.
                if valid_other_side >= slots_to_fill:
                    continue

        # Check overlap constraint
        too_similar = False
        for s_idx in selected_idx:
            s_key = sorted_reg.loc[s_idx, "canonical_key"]
            s_mask = mask_map.get(s_key)
            if s_mask is None:
                continue

            intersection = float(np.sum(mask & s_mask))
            union = float(np.sum(mask | s_mask))
            jaccard = intersection / union if union > 0 else 0.0

            if jaccard > max_overlap:
                too_similar = True
                break

        if not too_similar:
            selected_idx.append(idx)
            if side in selected_sides:
                selected_sides[side] += 1

    if len(selected_idx) < min(top_n, len(registry)) and max_overlap < 0.8:
        return select_top_diverse_rules(
            registry, mask_map, top_n, max_overlap + 0.1, max_side_in_top
        )

    return sorted_reg.loc[selected_idx]


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
    overlap_threshold: float = 0.8,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
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
    n_rows_any = len(data)

    # 2. Identify reference feature (max availability)
    availability = {k: np.isfinite(v).mean() for k, v in feature_dict.items()}
    ref_feat = max(availability, key=availability.get)
    ref_mask = np.isfinite(feature_dict[ref_feat])

    # 3. Prune features based on overlap with reference
    retained_features = {}
    dropped_features = []

    continuous_features_converted = 0
    total_features_initial = len(feature_dict)

    for k, v in feature_dict.items():
        overlap = np.isfinite(v[ref_mask]).mean()
        if overlap >= overlap_threshold:
            retained_features[k] = v

            # Heuristic to check if continuous feature transformed to binary (e.g. only 0/1/NaN)
            unique_vals = np.unique(v[np.isfinite(v)])
            if set(unique_vals).issubset({0.0, 1.0}):
                # Assuming this implies it's been transformed into binaries successfully if it has binary values
                continuous_features_converted += 1
        else:
            dropped_features.append((k, overlap))

    # 4. Final row pruning: fully available for the specific symbol/timestamp row
    all_finite = np.ones(len(data), dtype=bool)
    for v in retained_features.values():
        all_finite &= np.isfinite(v)

    # Filtering is necessary, but we only drop the specific (symbol, timestamp) row
    final_keep_mask = all_finite

    data_final = data.loc[final_keep_mask].reset_index(drop=True)
    features_final = {k: v[final_keep_mask] for k in retained_features}

    # Check if there are any NaNs left after filtering
    nan_features_detected = []
    for k, v in features_final.items():
        if np.isnan(v).any():
            nan_features_detected.append(k)
    fwd_ret_final = fwd_ret[final_keep_mask]
    fwd_ret_norm_final = fwd_ret_norm[final_keep_mask]

    meta = {
        "rows_initial": n_rows_initial,
        "rows_after_any_feat": n_rows_any,
        "rows_final": len(data_final),
        "dropped_rows": n_rows_initial - len(data_final),
        "reference_feature": ref_feat,
        "dropped_features": dropped_features,
        "retained_count": len(retained_features),
        "total_features_initial": total_features_initial,
        "continuous_features_converted": continuous_features_converted,
        "nan_features_detected": nan_features_detected,
    }

    return data_final, features_final, fwd_ret_final, fwd_ret_norm_final, meta


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
    - 200 symbols
    - 3 years of lookback
    """
    out = dict(cfg)
    out["n_folds"] = 3
    out["sliceplanner_outer_n_folds"] = 3
    out["mask_opt_max_symbols"] = 200
    out["mask_opt_lookback_years"] = 3.0
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
    all_df["is_accepted"] = all_df.get("accepted", False).fillna(False).astype(bool)
    all_df["is_production"] = all_df["is_accepted"] & (
        all_df.get("composite_score", -np.inf) > 0
    )

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
        help="Run the smaller smoke-test configuration",
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
        default="3,10",
        help="Comma-separated list of horizons for triad targets (default: 3,10)",
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
    if args.test_mode:
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
    fwd_ret_wide = (
        panel["close"].pct_change(fwd_hours, fill_method=None).shift(-fwd_hours)
    )

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
    target_signal = fwd_ret_matrix / np.maximum(np.sqrt(atr_pct_matrix), 1e-9)

    fwd_ret_norm_matrix = fwd_ret_matrix / np.maximum(atr_pct_matrix, 1e-9)
    fwd_ret_final = fwd_ret_matrix[time_idx, sym_idx]
    fwd_ret_norm_final = fwd_ret_norm_matrix[time_idx, sym_idx]
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
        robust_meta,
    ) = apply_robust_data_filtering(
        data_final,
        feat_final,
        fwd_ret_final,
        fwd_ret_norm_final,
        overlap_threshold=0.8,
    )

    tprint(
        "Robust Data Filter complete: "
        f"rows_initial={robust_meta['rows_initial']} "
        f"rows_after_any_feat={robust_meta['rows_after_any_feat']} "
        f"rows_final={robust_meta['rows_final']} "
        f"dropped_rows={robust_meta['dropped_rows']} "
        f"retained_features={robust_meta['retained_count']} "
        f"reference={robust_meta['reference_feature']} "
        f"total_features_initial={robust_meta['total_features_initial']} "
        f"continuous_features_converted={robust_meta['continuous_features_converted']}"
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
        "target_ela": {},
        "target_vame": {},
        "target_eff_surprisal": {},
        "target_ela_surprisal": {},
        "target_vame_surprisal": {},
    }

    for horizon, df_targets in triad_results_by_horizon.items():
        for target_base in [
            "target_eff",
            "target_ela",
            "target_vame",
            "target_eff_surprisal",
            "target_ela_surprisal",
            "target_vame_surprisal",
        ]:
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
    run_lgbm_mask_generation_triad(data_final, feat_final, triad_targets, cfg)
