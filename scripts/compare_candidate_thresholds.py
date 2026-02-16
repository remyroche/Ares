#!/usr/bin/env python
"""
Candidate Selection Threshold Comparison (Optimized)

Compares Fixed, ATR-normalized, and Volume-Weighted candidate selection methods.
Uses ExtraTrees with the same parameters as the target race in training.py.

Usage:
    python scripts/compare_candidate_thresholds.py \\
        --features data/features/20260214_190000 \\
        --panel data/klines \\
        --output reports/candidate_threshold_comparison.csv
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import logging
from copy import deepcopy
from typing import Dict, Optional, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-use from extreme_price_movements
from extreme_price_movements.purged_cv import PurgedKFold
from extreme_price_movements.config import MODEL_FEATURES, CFG
from extreme_price_movements.data_store import load_features as load_features_pipeline
from extreme_price_movements.training import (
    build_hourly_training_set_and_weights,
    build_grid_aggregated_tb_cache,
)
from extreme_price_movements.utils import tprint
from extreme_price_movements import fast_funcs as ff

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


ET_REGRESSOR_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_leaf": 30,
    "max_features": "sqrt",
    "n_jobs": 3,
    "random_state": 42,
}

RIDGE_SCREEN_ALPHA = 0.5
RIDGE_SCREEN_TOP_FRAC = 0.25
OOF_MAX_SAMPLES = 1_800_000
OOF_RIDGE_MAX_TRAIN_SAMPLES = 700_000
FEATURE_CHUNK_SIZE = 8
_PSUTIL_WARNED = False


def get_memory_mb() -> float:
    """Return process resident memory in MB when available."""
    global _PSUTIL_WARNED
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))
    except Exception:
        if not _PSUTIL_WARNED:
            logger.warning("psutil unavailable; memory usage will not be reported in tlog output")
            _PSUTIL_WARNED = True
        return float("nan")


def tlog(message: str):
    """Unified timestamped progress + memory logging."""
    mem = get_memory_mb()
    if np.isfinite(mem):
        tprint(f"{message} | mem={mem:.1f}MB")
    else:
        tprint(message)


class DataContainer:
    """
    Encapsulates aligned panel and feature data for optimized access.
    Holds both DataFrames (for legacy compatibility) and numpy arrays (for speed).
    """
    def __init__(self, panel: dict, feats: dict, float_dtype: np.dtype = np.float32):
        self.panel_dfs = panel
        self.feat_dfs = feats
        self.float_dtype = float_dtype

        self.common_index = None
        self.common_columns = None

        # Aligned numpy arrays (name -> 2D array)
        self.panel_arr = {}
        self.feat_arr = {}

        # Aligned DataFrames (name -> DF)
        self.aligned_panel_dfs = {}
        self.aligned_feat_dfs = {}

    def align(self):
        """Align all data to the intersection of indices and columns."""
        tlog("Aligning data indices and columns...")

        # 1. Start with panel close
        if "close" not in self.panel_dfs:
            raise ValueError("Panel must have 'close' column")

        common_idx = self.panel_dfs["close"].index
        common_cols = self.panel_dfs["close"].columns

        # 2. Intersect with all features
        for name, df in self.feat_dfs.items():
            common_idx = common_idx.intersection(df.index)
            common_cols = common_cols.intersection(df.columns)

        self.common_index = common_idx
        self.common_columns = common_cols

        tlog(f"Aligned grid: {len(common_idx)} rows x {len(common_cols)} cols")

        # 3. Reindex and store
        # Panel
        for name, df in self.panel_dfs.items():
            if isinstance(df, pd.DataFrame):
                aligned_df = df.reindex(index=common_idx, columns=common_cols).astype(self.float_dtype, copy=False)
                self.aligned_panel_dfs[name] = aligned_df
                self.panel_arr[name] = aligned_df.to_numpy(dtype=self.float_dtype, copy=False)

        # Features
        for name, df in self.feat_dfs.items():
            aligned_df = df.reindex(index=common_idx, columns=common_cols).astype(self.float_dtype, copy=False)
            self.aligned_feat_dfs[name] = aligned_df
            self.feat_arr[name] = aligned_df.to_numpy(dtype=self.float_dtype, copy=False)

        # Clear original references to free memory
        self.panel_dfs = None
        self.feat_dfs = None
        gc.collect()
        tlog("Data alignment complete.")

    def get_feature_matrix(self, feature_names: List[str], row_idx: np.ndarray, col_idx: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Efficiently construct feature matrix X (samples x features) using aligned arrays.
        """
        valid_feats = [f for f in feature_names if f in self.feat_arr]
        if not valid_feats:
            return np.empty((len(row_idx), 0), dtype=self.float_dtype), []

        n_samples = len(row_idx)
        n_feats = len(valid_feats)
        X = np.empty((n_samples, n_feats), dtype=self.float_dtype)

        for i, name in enumerate(valid_feats):
            arr = self.feat_arr[name]
            # Use advanced indexing
            X[:, i] = arr[row_idx, col_idx]

        return X, valid_feats

    def get_column(self, name: str, source: str = "feat") -> Optional[np.ndarray]:
        """Get the full 2D array for a feature or panel column."""
        if source == "feat":
            return self.feat_arr.get(name)
        elif source == "panel":
            return self.panel_arr.get(name)
        return None


def cast_dataframe_dtype(df: pd.DataFrame, float_dtype: np.dtype) -> pd.DataFrame:
    """Cast numeric columns to requested dtypes for memory control."""
    for col in df.columns:
        col_type = df[col].dtype
        if np.issubdtype(col_type, np.floating):
            if col_type != float_dtype:
                df[col] = df[col].astype(float_dtype, copy=False)
        elif np.issubdtype(col_type, np.integer) and np.dtype(col_type).itemsize > 4:
            df[col] = df[col].astype(np.int32, copy=False)
    return df


def cast_features_dtype(feats: dict, float_dtype: np.dtype) -> dict:
    """Cast all feature DataFrames to the requested float dtype."""
    for name, df in feats.items():
        if isinstance(df, pd.DataFrame):
            feats[name] = cast_dataframe_dtype(df, float_dtype=float_dtype)
    return feats


def compute_global_sign_consistency(
    data: DataContainer,
) -> Optional[np.ndarray]:
    """Compute sign consistency using aligned data."""
    sc_arr = None
    sc_source = "none"

    if "sign_consistency" in data.feat_arr:
        sc_arr = data.feat_arr["sign_consistency"]
        sc_source = "features.sign_consistency"
    elif "sign_consistency_12h" in data.feat_arr:
        sc_arr = data.feat_arr["sign_consistency_12h"]
        sc_source = "features.sign_consistency_12h"
    else:
        # Fallback calculation on aligned arrays
        base_ret = data.feat_arr.get("ret6h")
        if base_ret is None:
            base_ret = data.feat_arr.get("ret24h")

        if base_ret is not None:
            # We need to do rolling mean on the array.
            # Since data is aligned (Time x Symbol), we can wrap in DF or use bottleneck/numba if available.
            # For simplicity, wrap in DF since this is a one-time precompute.
            base_ret_df = pd.DataFrame(base_ret, index=data.common_index, columns=data.common_columns)
            sign_mean = np.sign(base_ret_df).rolling(12, min_periods=6).mean().abs()
            sc_arr = sign_mean.to_numpy(dtype=data.float_dtype, copy=False)
            sc_source = "features.sign_roll_mean_abs_12"
        elif "close" in data.panel_arr:
            try:
                # Fallback to panel
                close_arr = data.panel_arr["close"]
                close_df = pd.DataFrame(close_arr, index=data.common_index, columns=data.common_columns)
                sc_arr_numba = ff.numba_sign_consistency(close_df, 12)
                sc_arr = sc_arr_numba.astype(data.float_dtype)
                sc_source = "panel.numba_sign_consistency"
            except Exception as exc:
                tlog(f"Sign-consistency fallback failed: {exc}")
                sc_arr = None
                sc_source = "none"

    if sc_arr is not None:
        sample = sc_arr.reshape(-1)
        sample = sample[np.isfinite(sample)]
        if sample.size > 0:
            q99 = float(np.quantile(sample[::100], 0.99)) # subsample for speed
            if q99 > 1.5:
                sc_arr = (sc_arr / np.float32(100.0)).astype(data.float_dtype)
        tlog(f"Sign-consistency source={sc_source}")

    return sc_arr


def precompute_filter_masks(
    data: DataContainer,
    range_thresholds: list[float],
    vol_thresholds: list[float],
    sc_thresholds: list[float],
) -> dict:
    """Precompute boolean filter masks using aligned numpy arrays."""
    n_rows, n_cols = len(data.common_index), len(data.common_columns)
    true_arr = np.ones((n_rows, n_cols), dtype=bool)

    # Helper to find first available feature array
    def _get_first(names):
        for name in names:
            if name in data.feat_arr:
                return data.feat_arr[name]
        return None

    range_arr = _get_first(["range_12h_pct", "range_16h_pct", "range_pct"])
    vol_arr = _get_first(["volatility_zscore", "vol_z"])
    sc_arr = compute_global_sign_consistency(data)

    range_masks: dict[float, np.ndarray] = {}
    vol_masks: dict[float, np.ndarray] = {}
    sc_masks: dict[float, np.ndarray] = {}

    if range_arr is not None:
        for thr in sorted({float(v) for v in range_thresholds}):
            range_masks[thr] = range_arr > thr

    if vol_arr is not None:
        for thr in sorted({float(v) for v in vol_thresholds}):
            vol_masks[thr] = vol_arr > thr

    if sc_arr is not None:
        for thr in sorted({float(v) for v in sc_thresholds}):
            thr_eff = thr / 100.0 if thr > 1.0 else thr
            sc_masks[thr] = sc_arr >= thr_eff

    return {
        "true_arr": true_arr,
        "range_masks": range_masks,
        "vol_masks": vol_masks,
        "sc_masks": sc_masks,
    }


def to_panel_dict(panel_df: pd.DataFrame) -> dict:
    """Convert OHLCV dataframe into training-compatible wide panel dict."""
    if isinstance(panel_df, dict):
        return panel_df

    ts_candidates = ["timestamp", "ts", "datetime", "date", "open_time"]
    sym_candidates = ["symbol", "asset", "ticker"]

    ts_col = next((c for c in ts_candidates if c in panel_df.columns), None)
    sym_col = next((c for c in sym_candidates if c in panel_df.columns), None)
    if ts_col is None or sym_col is None:
        raise ValueError("Panel data must include timestamp and symbol columns")

    df = panel_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=[ts_col, sym_col])

    panel = {}
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            wide = df.pivot_table(index=ts_col, columns=sym_col, values=col, aggfunc="last")
            wide = wide.sort_index().sort_index(axis=1)
            panel[col] = wide.astype(np.float32)

    if not {"high", "low", "close"}.issubset(panel.keys()):
        raise ValueError("Panel data must include at least high/low/close columns")

    return panel


def _series_to_terciles(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    if x.dropna().nunique() < 3:
        return pd.Series(1, index=x.index, dtype=np.int8)
    try:
        bins = pd.qcut(x.rank(method="first"), q=3, labels=False, duplicates="drop")
        return bins.fillna(1).astype(np.int8)
    except Exception:
        return pd.Series(1, index=x.index, dtype=np.int8)


def build_proxy_mkt_gates(feats_dfs: dict) -> pd.DataFrame:
    """Build minimal market gates DataFrame required by training set builder."""
    # Use aligned DataFrames from DataContainer
    vol_df = feats_dfs.get("volatility_zscore")
    if vol_df is None:
        vol_df = feats_dfs.get("vol_z")
    trend_df = feats_dfs.get("trend_pct")
    if trend_df is None:
        trend_df = feats_dfs.get("ret6h")
    if trend_df is None:
        trend_df = feats_dfs.get("ret24h")

    if vol_df is None or trend_df is None:
        # Fallback
        idx = next(iter(feats_dfs.values())).index
        return pd.DataFrame({"G_VOL": 1, "G_TREND": 1}, index=idx, dtype=np.int8)

    vol_s = vol_df.median(axis=1)
    trend_s = trend_df.median(axis=1)
    return pd.DataFrame(
        {
            "G_VOL": _series_to_terciles(vol_s),
            "G_TREND": _series_to_terciles(trend_s),
        },
        index=vol_df.index,
    )


def fingerprint_candidate_mask(mask_arr: np.ndarray) -> str:
    """Compact deterministic fingerprint for caching."""
    packed = np.packbits(mask_arr.astype(np.uint8, copy=False))
    h = hashlib.blake2b(digest_size=16)
    h.update(str(mask_arr.shape).encode("ascii"))
    h.update(packed.tobytes())
    return h.hexdigest()


def _bucket_move_bucket(side: str, kind: str) -> str:
    """Map trade bucket to move bucket used by pipeline filtering."""
    if side == "long":
        cand_filter = "worst" if kind == "mr" else "best"
    else:
        cand_filter = "best" if kind == "mr" else "worst"
    return "up" if cand_filter == "best" else "down"


def _build_bucket_candidate_mask_arr(
    candidate_mask_arr: np.ndarray,
    data: DataContainer,
    move_bucket: str,
) -> np.ndarray:
    """Create per-bucket candidate mask (up/down) using trend sign filter."""
    trend_arr = data.feat_arr.get("trend_pct")
    if trend_arr is None:
        return candidate_mask_arr

    if move_bucket == "up":
        trend_mask = trend_arr > 0
    else:
        trend_mask = trend_arr <= 0

    return candidate_mask_arr & trend_mask


def infer_stage_label(config_id: str) -> str:
    """Infer sweep stage from config id."""
    if "_S3" in config_id:
        return "Stage 3"
    if "_FULL_" in config_id:
        return "Stage 2"
    return "Stage 1"


def evaluate_training_slices(
    candidate_mask_arr: np.ndarray,
    data: DataContainer,
    cfg_variant: dict,
    horizons: list,
    cache: Optional[dict] = None,
    cache_key: Optional[tuple] = None,
    sample_frac: float = 1.0,
) -> list:
    """Evaluate MR/TF x long/short slices using aligned DataFrames."""

    # Construct DataFrame mask from array for compatibility with training.py
    candidate_mask = pd.DataFrame(candidate_mask_arr, index=data.common_index, columns=data.common_columns)

    if sample_frac < 1.0:
        n_rows = len(candidate_mask)
        n_sample = int(n_rows * sample_frac)
        np.random.seed(42)  # Reproducible subsampling
        sample_idx = np.sort(np.random.choice(n_rows, n_sample, replace=False))
        candidate_mask = candidate_mask.iloc[sample_idx].copy()
        tlog(f"Training slices: subsampled {n_sample}/{n_rows} rows ({sample_frac*100:.0f}%)")
    else:
        sample_idx = None

    if cache is not None and cache_key is not None and cache_key in cache:
        tlog("Training-slice cache hit")
        return [dict(r) for r in cache[cache_key]]

    tlog("Training-slice evaluation start")
    rows = []

    # We must use aligned feats for training set construction
    mkt_gates = build_proxy_mkt_gates(data.aligned_feat_dfs)
    ts_end = candidate_mask.index.max()

    # Re-use build_grid_aggregated_tb_cache with aligned panel/feats
    tb_cache_by_h_side, geom_cache_by_h_side = build_grid_aggregated_tb_cache(
        panel=data.aligned_panel_dfs,
        feats=data.aligned_feat_dfs,
        cfg=cfg_variant,
        horizons=horizons,
        trade_sides=["long", "short"],
    )
    bucket_mask_cache: dict[tuple[str, str], pd.DataFrame] = {}

    for side in ["long", "short"]:
        for kind in ["mr", "tf"]:
            tlog(f"Training slice loop: side={side}, kind={kind}")
            cand_filter = "unknown"
            if side == "long":
                cand_filter = "worst" if kind == "mr" else "best"
            else:
                cand_filter = "best" if kind == "mr" else "worst"
            trend_filter = "up" if cand_filter == "best" else "down"
            feat_key = "tf_feature_keys" if kind == "tf" else "mr_feature_keys"
            move_bucket = _bucket_move_bucket(side, kind)

            if (side, kind) not in bucket_mask_cache:
                trend_arr = data.feat_arr.get("trend_pct")
                if trend_arr is not None:
                     trend_aligned = trend_arr
                     if sample_idx is not None:
                         trend_aligned = trend_aligned[sample_idx]

                     if move_bucket == "up":
                         t_mask = trend_aligned > 0
                     else:
                         t_mask = trend_aligned <= 0

                     # candidate_mask is already a DataFrame
                     b_mask = candidate_mask & pd.DataFrame(t_mask, index=candidate_mask.index, columns=candidate_mask.columns)
                else:
                     b_mask = candidate_mask

                bucket_mask_cache[(side, kind)] = b_mask.fillna(False)

            bucket_mask = bucket_mask_cache[(side, kind)]
            bucket_n = int(bucket_mask.values.sum())
            tlog(
                f"Training slice bucket mask: side={side}, kind={kind}, "
                f"move_bucket={move_bucket}, selected={bucket_n}"
            )
            skip_remaining_horizons = False

            for h_i, h in enumerate(horizons):
                if skip_remaining_horizons:
                    rows.append({
                        "slice": f"{side}_{kind}", "side": side, "kind": kind, "horizon": h,
                        "n_samples": 0, "label_pos_rate": 0, "mean_ret_bps": 0,
                        "sharpe": 0, "sortino": 0, "weighted_ret_bps": 0,
                    })
                    continue

                tlog(f"Building training set: side={side}, kind={kind}, H={h}")
                # Inject geom features into feats dict (inplace modification of aligned DF dict)
                if (h, side) in geom_cache_by_h_side:
                    data.aligned_feat_dfs["__geom_n_tp__"] = geom_cache_by_h_side[(h, side)]["n_tp"]
                    data.aligned_feat_dfs["__geom_n_sl__"] = geom_cache_by_h_side[(h, side)]["n_sl"]
                    data.aligned_feat_dfs["__geom_n_to__"] = geom_cache_by_h_side[(h, side)]["n_to"]

                # We need to pass the aligned dicts
                X, y_bin, y_ret, cols, w, meta_idx = build_hourly_training_set_and_weights(
                    panel=data.aligned_panel_dfs,
                    feats=data.aligned_feat_dfs,
                    mkt_gates=mkt_gates,
                    cfg=cfg_variant,
                    syms=list(data.common_columns),
                    ts_end=ts_end,
                    p_exh_hist=None,
                    H=h,
                    model_kind=kind,
                    trend_filter=trend_filter,
                    feature_key=feat_key,
                    extra_feature_keys=[],
                    label_method="triple_barrier",
                    side=side,
                    _cached_cand_mask=bucket_mask,
                    _cached_tb=tb_cache_by_h_side.get((h, side)),
                )

                structurally_empty = X is None and (y_ret is None or len(y_ret) == 0)
                if X is None or y_ret is None or len(y_ret) < 50:
                    tlog(f"Slice skipped (insufficient samples): side={side}, kind={kind}, H={h}")
                    rows.append({
                        "slice": f"{side}_{kind}", "side": side, "kind": kind, "horizon": h,
                        "n_samples": 0, "label_pos_rate": 0, "mean_ret_bps": 0,
                        "sharpe": 0, "sortino": 0, "weighted_ret_bps": 0,
                    })
                    if structurally_empty:
                        skip_remaining_horizons = True
                    continue

                y_ret = np.asarray(y_ret, dtype=np.float32)
                y_bin = np.asarray(y_bin, dtype=np.float32)
                w = np.asarray(w, dtype=np.float32)
                mean_ret = float(np.nanmean(y_ret))
                vol_ret = float(np.nanstd(y_ret))
                downside = y_ret[y_ret < 0]
                sharpe = float(mean_ret / vol_ret * np.sqrt(8760)) if vol_ret > 1e-12 else 0.0
                sortino = (
                    float(mean_ret / np.nanstd(downside) * np.sqrt(8760))
                    if len(downside) > 1 and np.nanstd(downside) > 1e-12
                    else 0.0
                )
                weighted_ret = float(np.average(y_ret, weights=np.clip(w, 1e-8, None)))

                rows.append(
                    {
                        "slice": f"{side}_{kind}",
                        "side": side,
                        "kind": kind,
                        "horizon": h,
                        "n_samples": int(len(y_ret)),
                        "label_pos_rate": float(np.nanmean(y_bin >= 0.5)),
                        "mean_ret_bps": mean_ret * 1e4,
                        "sharpe": sharpe,
                        "sortino": sortino,
                        "weighted_ret_bps": weighted_ret * 1e4,
                    }
                )
                tlog(f"Slice done: side={side}, kind={kind}, H={h}, n={len(y_ret)}")

    if cache is not None and cache_key is not None:
        cache[cache_key] = [dict(r) for r in rows]
    tlog("Training-slice evaluation done")
    return rows


def aggregate_slice_rows(slice_rows: list) -> dict:
    """Aggregate horizon-level slice rows into per-slice summary metrics."""
    if not slice_rows:
        return {
            "slice_overall_sharpe": 0.0,
            "slice_overall_sortino": 0.0,
            "slice_total_samples": 0,
            "slice_metrics_json": "{}",
        }

    sdf = pd.DataFrame(slice_rows)
    if sdf.empty:
        return {
            "slice_overall_sharpe": 0.0,
            "slice_overall_sortino": 0.0,
            "slice_total_samples": 0,
            "slice_metrics_json": "{}",
        }

    grouped = sdf.groupby("slice", as_index=False).apply(
        lambda g: pd.Series(
            {
                "n_samples": float(g["n_samples"].sum()),
                "mean_ret_bps": float(np.average(g["mean_ret_bps"], weights=np.clip(g["n_samples"], 1, None))),
                "weighted_ret_bps": float(np.average(g["weighted_ret_bps"], weights=np.clip(g["n_samples"], 1, None))),
                "sharpe": float(np.average(g["sharpe"], weights=np.clip(g["n_samples"], 1, None))),
                "sortino": float(np.average(g["sortino"], weights=np.clip(g["n_samples"], 1, None))),
            }
        )
    ).reset_index(drop=True)

    total_samples = float(grouped["n_samples"].sum())
    overall_sharpe = (
        float(np.average(grouped["sharpe"], weights=np.clip(grouped["n_samples"], 1, None)))
        if total_samples > 0 else 0.0
    )
    overall_sortino = (
        float(np.average(grouped["sortino"], weights=np.clip(grouped["n_samples"], 1, None)))
        if total_samples > 0 else 0.0
    )

    metrics_json = json.dumps(
        {
            row["slice"]: {
                "n_samples": row["n_samples"],
                "mean_ret_bps": row["mean_ret_bps"],
                "weighted_ret_bps": row["weighted_ret_bps"],
                "sharpe": row["sharpe"],
                "sortino": row["sortino"],
            }
            for _, row in grouped.iterrows()
        }
    )

    return {
        "slice_overall_sharpe": overall_sharpe,
        "slice_overall_sortino": overall_sortino,
        "slice_total_samples": int(total_samples),
        "slice_metrics_json": metrics_json,
    }


def preprocess_atr(
    atr_pct: pd.DataFrame,
    floor: float = 0.005,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> Dict[str, pd.DataFrame]:
    """Apply ATR floor + winsorization and return derived denominator variants."""
    atr_filtered = atr_pct.where(atr_pct >= floor)
    coverage = float(np.isfinite(atr_filtered.to_numpy(dtype=np.float32, copy=False)).mean())
    if coverage < 0.05:
        logger.warning(
            "ATR floor removed most samples (coverage=%.2f%%). Falling back to raw atr_pct before winsorization.",
            coverage * 100.0,
        )
        atr_filtered = atr_pct.copy()
    lo = atr_filtered.quantile(lower_q, axis=1)
    hi = atr_filtered.quantile(upper_q, axis=1)
    atr_wins = atr_filtered.clip(lower=lo, upper=hi, axis=0)
    atr_robust = atr_wins.ewm(span=72, min_periods=24, adjust=False).mean()
    atr_robust = atr_robust.where(atr_filtered.notna())
    atr_robust = atr_robust.where(np.isfinite(atr_robust), atr_wins)
    return {
        "atr_filtered": atr_filtered,
        "atr_wins": atr_wins,
        "atr_robust": atr_robust,
    }


# =============================================================================
# Candidate Selection Functions (Optimized)
# =============================================================================

def precompute_selection_metrics(data: DataContainer) -> dict:
    """Precompute selection metrics once and reuse across configs."""
    ret_base = data.feat_arr.get("ret6h")
    if ret_base is None:
        ret_base = data.feat_arr.get("ret24h")
    if ret_base is None:
        raise ValueError("ret6h/ret24h not found in features")

    metrics = {"fixed": ret_base}
    atr_effective = None

    if "atr_pct" in data.aligned_feat_dfs:
        atr_df = data.aligned_feat_dfs["atr_pct"]
        atr_pack = preprocess_atr(atr_df)

        atr_robust_arr = atr_pack["atr_robust"].to_numpy(dtype=data.float_dtype, copy=False)
        atr_effective = atr_robust_arr

        metrics["atr"] = ret_base / (atr_robust_arr + 1e-12)
        metrics["atr_robust"] = metrics["atr"]

    rvol_z = data.feat_arr.get("rvol_z")
    volu_z = data.feat_arr.get("volu_z")
    if rvol_z is not None and volu_z is not None:
        vol_combined = ((rvol_z + volu_z) / 2)
        metrics["vol_weight"] = np.abs(ret_base) * np.clip(vol_combined, 0, None) * np.sign(ret_base)

    return {
        "metrics": metrics,
        "atr_effective": atr_effective,
    }

def apply_quality_filters_array(
    base_mask_arr: np.ndarray,
    filter_masks: Optional[dict] = None,
    min_range_pct: Optional[float] = None,
    min_vol_zscore: Optional[float] = None,
    min_sign_consistency: Optional[float] = None,
) -> np.ndarray:
    """Apply quality filters in-place on a bool ndarray to avoid DataFrame copies."""
    if filter_masks is None:
        return base_mask_arr

    filtered = base_mask_arr.copy()

    if min_range_pct is not None:
        range_mask = filter_masks["range_masks"].get(float(min_range_pct))
        if range_mask is not None:
            filtered &= range_mask
    if min_vol_zscore is not None:
        vol_mask = filter_masks["vol_masks"].get(float(min_vol_zscore))
        if vol_mask is not None:
            filtered &= vol_mask
    if min_sign_consistency is not None:
        sc_mask = filter_masks["sc_masks"].get(float(min_sign_consistency))
        if sc_mask is not None:
            filtered &= sc_mask
    return filtered


def compute_cross_sectional_base_mask(metric_arr: np.ndarray, pct: float) -> np.ndarray:
    """Compute cross-sectional base signs as int8 ndarray (1=Top, -1=Bottom)."""
    valid = np.isfinite(metric_arr)
    n_valid = valid.sum(axis=1)
    k = np.maximum(1, (n_valid * pct).astype(int))

    n_rows, n_cols = metric_arr.shape
    sign_arr = np.zeros_like(metric_arr, dtype=np.int8)

    # Iterate rows for top-k/bottom-k selection (Numpy loop is efficient enough for typical N)
    for i in range(n_rows):
        if n_valid[i] < 2:
            continue

        row_vals = metric_arr[i]
        valid_idx = np.flatnonzero(valid[i])
        n_v = len(valid_idx)
        if n_v == 0: continue

        k_i = k[i]
        vals = row_vals[valid_idx]

        if k_i >= n_v:
             # If selecting all, assign signs based on median? Or just +1?
             # Fallback: everything valid is selected. Split by median.
             med = np.median(vals)
             sign_arr[i, valid_idx[vals > med]] = 1
             sign_arr[i, valid_idx[vals <= med]] = -1
             continue

        # Bottom k (Short/MeanRev Long) -> -1
        part_idx = np.argpartition(vals, k_i)
        bot_local_idx = part_idx[:k_i]
        sign_arr[i, valid_idx[bot_local_idx]] = -1

        # Top k (Long/MeanRev Short) -> 1
        part_idx_top = np.argpartition(vals, n_v - k_i)
        top_local_idx = part_idx_top[n_v - k_i:]
        sign_arr[i, valid_idx[top_local_idx]] = 1

    return sign_arr


def expand_signed_mask(base_sign_arr: np.ndarray, offsets: list[int]) -> np.ndarray:
    """Expand signed candidate timestamps, propagating signs (latest wins or non-zero)."""
    if not offsets:
        return base_sign_arr
    expanded = base_sign_arr.copy()

    # We want to propagate signs. If overlap, what to do?
    # Simple strategy: prioritize existing non-zero, or let new shifts overwrite?
    # Usually expansion means "if signal at t, then t+1 is also active".
    # So we propagate forward (positive offsets).
    # If base is at t, and offset is +1, then t+1 gets signal from t.
    # We iterate offsets.

    for off in offsets:
        off = int(off)
        shifted = np.zeros_like(base_sign_arr)
        if off > 0:
            shifted[off:] = base_sign_arr[:-off]
        elif off < 0:
            shifted[:off] = base_sign_arr[-off:]
        else:
            shifted = base_sign_arr

        # Merge: if expanded is 0, take shifted. If both non-zero, keep expanded (priority to original/closer).
        mask_empty = expanded == 0
        expanded[mask_empty] = shifted[mask_empty]

    return expanded


def select_candidates_cross_sectional(
    metric_arr: np.ndarray,
    pct: float,
    filter_masks: Optional[dict] = None,
    min_range_pct: float = None,
    min_vol_zscore: float = None,
    min_sign_consistency: float = None,
    base_mask_cache: Optional[dict] = None,
    base_cache_key: Optional[tuple] = None,
) -> np.ndarray:
    """
    Unified cross-sectional selection returning signed int8 array (+1/-1/0).
    """
    if base_mask_cache is not None and base_cache_key is not None and base_cache_key in base_mask_cache:
        base_sign_arr = base_mask_cache[base_cache_key]
    else:
        base_sign_arr = compute_cross_sectional_base_mask(metric_arr, pct)
        if base_mask_cache is not None and base_cache_key is not None:
            base_mask_cache[base_cache_key] = base_sign_arr

    has_filters = any(
        [
            min_range_pct is not None,
            min_vol_zscore is not None,
            min_sign_consistency is not None,
        ]
    )
    if has_filters:
        # Quality filters are boolean. Apply them to zero-out signs.
        # We need a small adapter since apply_quality_filters_array expected bool.
        # Let's just reimplement the masking here for clarity with signs.

        sign_arr = base_sign_arr.copy()

        # Construct combined boolean mask
        keep_mask = np.ones(sign_arr.shape, dtype=bool)
        if min_range_pct is not None:
            range_mask = filter_masks["range_masks"].get(float(min_range_pct))
            if range_mask is not None: keep_mask &= range_mask
        if min_vol_zscore is not None:
            vol_mask = filter_masks["vol_masks"].get(float(min_vol_zscore))
            if vol_mask is not None: keep_mask &= vol_mask
        if min_sign_consistency is not None:
            sc_mask = filter_masks["sc_masks"].get(float(min_sign_consistency))
            if sc_mask is not None: keep_mask &= sc_mask

        sign_arr[~keep_mask] = 0
    else:
        sign_arr = base_sign_arr.copy()

    return sign_arr


# =============================================================================
# Metric Computation Functions
# =============================================================================

def compute_ks_statistic(pos_values: np.ndarray, neg_values: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic between two distributions."""
    if len(pos_values) < 10 or len(neg_values) < 10:
        return 0.0
    try:
        return stats.ks_2samp(pos_values, neg_values).statistic
    except Exception:
        return 0.0


def safe_pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Safe Pearson correlation that avoids divide-by-zero warnings."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    x_std = np.nanstd(x)
    y_std = np.nanstd(y)
    if not np.isfinite(x_std) or not np.isfinite(y_std) or x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def compute_snr(pos_values: np.ndarray, neg_values: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio between positive and negative distributions."""
    if len(pos_values) < 2 or len(neg_values) < 2:
        return 0.0
    try:
        mean_diff = abs(np.nanmean(pos_values) - np.nanmean(neg_values))
        var_sum = np.nanvar(pos_values) + np.nanvar(neg_values)
        if var_sum <= 0:
            return 0.0
        return mean_diff / np.sqrt(var_sum)
    except Exception:
        return 0.0


def compute_feature_target_correlation(
    data: DataContainer,
    available_features: list,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    target_values: np.ndarray,
    top_n: int = 20
) -> float:
    """
    Compute mean |IC| of top features with target using vectorized ops.
    """
    if not available_features:
        return 0.0

    y = target_values
    valid_y = np.isfinite(y)

    if valid_y.sum() < 5:
        return 0.0

    ics = []

    for feat_name in available_features:
        feat_arr = data.get_column(feat_name, source="feat")
        if feat_arr is None:
            continue

        x = feat_arr[row_idx, col_idx]

        mask = valid_y & np.isfinite(x)
        if mask.sum() < 5:
            continue

        corr = safe_pearson_corr(x[mask], y[mask])
        if np.isfinite(corr):
            ics.append(abs(corr))

    if not ics:
        return 0.0

    ics_sorted = sorted(ics, reverse=True)[:top_n]
    return np.mean(ics_sorted)


def _ridge_select_topk(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray],
    top_frac: float,
    alpha: float,
) -> np.ndarray:
    """Select top-fraction feature subset using RobustScaler + Ridge |coef| ranking."""
    n_features = X_train.shape[1]
    k = int(max(1, min(n_features, np.ceil(float(top_frac) * n_features))))
    if n_features <= k:
        return np.arange(n_features, dtype=np.int32)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train, sample_weight=sample_weight)

    coef = np.asarray(ridge.coef_)
    if coef.ndim > 1:
        coef = np.nanmean(np.abs(coef), axis=0)
    else:
        coef = np.abs(coef)
    coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)

    idx = np.argpartition(coef, -k)[-k:]
    idx = idx[np.argsort(coef[idx])[::-1]]
    return idx.astype(np.int32, copy=False)


def _uniform_subsample_idx(n: int, k: int, seed: int) -> np.ndarray:
    if k >= n:
        return np.arange(n, dtype=np.int32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    idx.sort()
    return idx.astype(np.int32, copy=False)


def run_oof_cv(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray = None,
    n_splits: int = 3,
    purge: int = 12,  # 12 hours in index units
    random_state: int = 42,
    float_dtype: np.dtype = np.float32,
    ridge_alpha: float = RIDGE_SCREEN_ALPHA,
    ridge_top_frac: float = RIDGE_SCREEN_TOP_FRAC,
    use_extratrees: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Run purged k-fold CV with two-stage modeling.
    """
    n_samples = len(y)
    oof = np.full(n_samples, np.nan, dtype=float_dtype)
    
    y_valid = np.isfinite(y)
    y_filled = np.where(y_valid, y, np.nanmedian(y[y_valid])).astype(float_dtype, copy=False)
    
    time_idx = np.arange(n_samples, dtype=np.int32)
    pkf = PurgedKFold(n_splits=n_splits, purge=purge, embargo=2)
    
    splits = list(pkf.split(time_idx))
    selected_ks = []

    for fold_i, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y_filled[train_idx]
        sw_train = sample_weights[train_idx] if sample_weights is not None else None

        # Ridge screening subsample
        if len(train_idx) > OOF_RIDGE_MAX_TRAIN_SAMPLES:
            ridge_sub_idx = _uniform_subsample_idx(
                len(train_idx),
                OOF_RIDGE_MAX_TRAIN_SAMPLES,
                seed=(random_state + fold_i * 1009),
            )
            X_ridge = X_train[ridge_sub_idx]
            y_ridge = y_train[ridge_sub_idx]
            sw_ridge = sw_train[ridge_sub_idx] if sw_train is not None else None
        else:
            X_ridge = X_train
            y_ridge = y_train
            sw_ridge = sw_train

        selected_idx = _ridge_select_topk(
            X_train=X_ridge,
            y_train=y_ridge,
            sample_weight=sw_ridge,
            top_frac=ridge_top_frac,
            alpha=ridge_alpha,
        )
        selected_ks.append(int(len(selected_idx)))

        X_train_sel = X_train[:, selected_idx]
        X_val_sel = X_val[:, selected_idx]

        if use_extratrees:
            model = ExtraTreesRegressor(**{**ET_REGRESSOR_PARAMS, "random_state": random_state})
            model.fit(X_train_sel, y_train, sample_weight=sw_train)
            oof[val_idx] = model.predict(X_val_sel)
        else:
            from sklearn.linear_model import Ridge
            ridge_model = Ridge(alpha=ridge_alpha)
            ridge_model.fit(X_train_sel, y_train, sample_weight=sw_train)
            oof[val_idx] = ridge_model.predict(X_val_sel)

        del X_train, X_val, X_train_sel, X_val_sel, y_train, sw_train, X_ridge, y_ridge, sw_ridge
        gc.collect()

    diagnostics = {
        "ridge_selected_k_mean": float(np.mean(selected_ks)) if selected_ks else 0.0,
    }
    return oof, diagnostics


def compute_learnability_metrics(
    sign_arr: np.ndarray,
    data: DataContainer,
    available_features: list,
    float_dtype: np.dtype,
    use_extratrees: bool = False,
) -> dict:
    """
    Compute learnability metrics using Directional Sharpe (Momentum assumption).
    """
    tlog("Metrics: start")

    # Non-zero signs are selected
    row_idx, col_idx = np.nonzero(sign_arr)
    n_candidates = len(row_idx)

    if n_candidates == 0:
        return {
            "n_candidates_mean": 0, "ic": 0, "sharpe": 0,
            "slice_metrics_json": "{}", "error": "No candidates",
            "mean_feat_ic": 0, "mean_return_bps": 0
        }

    ret_arr = data.get_column("ret6h")
    if ret_arr is None: ret_arr = data.get_column("ret24h")

    target_arr = ret_arr

    candidate_returns_raw = ret_arr[row_idx, col_idx]
    candidate_target = target_arr[row_idx, col_idx]
    candidate_signs = sign_arr[row_idx, col_idx]

    valid_mask = np.isfinite(candidate_returns_raw) & np.isfinite(candidate_target)
    candidate_returns_raw = candidate_returns_raw[valid_mask]
    candidate_target = candidate_target[valid_mask]
    candidate_signs = candidate_signs[valid_mask]

    # Calculate Directional Returns (Momentum: Long Tops, Short Bottoms)
    candidate_returns = candidate_returns_raw * candidate_signs
    
    row_idx = row_idx[valid_mask]
    col_idx = col_idx[valid_mask]
    n_candidates = len(candidate_returns)
    
    if n_candidates < 50:
        return {"n_candidates_mean": 0, "ic": 0}

    n_candidates_mean = float(n_candidates / len(data.common_index))
    mean_return_bps = float(np.mean(candidate_returns) * 1e4)

    # Sharpe on Directional Returns
    sharpe = float(np.mean(candidate_returns) / np.std(candidate_returns) * np.sqrt(8760)) if np.std(candidate_returns) > 0 else 0.0
    
    # KS / SNR using target > 0 as positive class proxy
    pos_rets = candidate_returns[candidate_target > 0]
    neg_rets = candidate_returns[candidate_target <= 0]
    
    ks_stat = compute_ks_statistic(pos_rets, neg_rets)
    snr = compute_snr(pos_rets, neg_rets)
    hit_rate = float(np.mean(candidate_returns > 0))

    downside = candidate_returns[candidate_returns < 0]
    sortino = float(np.mean(candidate_returns) / np.std(downside) * np.sqrt(8760)) if len(downside) > 1 and np.std(downside) > 0 else 0.0

    tlog("Metrics: computing feature correlations")
    mean_feat_ic = compute_feature_target_correlation(
        data, available_features, row_idx, col_idx, candidate_target, top_n=20
    )
    
    ic, oof_mae, oof_directional_acc = 0.0, 0.0, 0.0
    ridge_diag = {}
    
    if len(available_features) >= 10:
        tlog("Metrics: constructing feature matrix for OOF")
        X, used_feats = data.get_feature_matrix(available_features, row_idx, col_idx)
        y = candidate_target

        if len(y) > OOF_MAX_SAMPLES:
            sub_idx = _uniform_subsample_idx(len(y), OOF_MAX_SAMPLES, seed=42)
            X = X[sub_idx]
            y = y[sub_idx]

        if X.shape[1] > 0 and len(y) > 100:
            tlog(f"Metrics: running OOF CV on {X.shape[0]}x{X.shape[1]}")
            oof, ridge_diag = run_oof_cv(
                X, y, float_dtype=float_dtype,
                ridge_alpha=RIDGE_SCREEN_ALPHA, ridge_top_frac=RIDGE_SCREEN_TOP_FRAC,
                use_extratrees=use_extratrees
            )

            oof_valid = np.isfinite(oof)
            if oof_valid.sum() >= 50:
                ic = safe_pearson_corr(oof[oof_valid], y[oof_valid])
                oof_mae = float(np.mean(np.abs(oof[oof_valid] - y[oof_valid])))
                oof_directional_acc = float(np.mean(np.sign(oof[oof_valid]) == np.sign(y[oof_valid])))

    return {
        "n_candidates_mean": n_candidates_mean,
        "mean_return_bps": mean_return_bps,
        "sharpe": sharpe,
        "sortino": sortino,
        "hit_rate": hit_rate,
        "ks_stat": ks_stat,
        "snr": snr,
        "mean_feat_ic": mean_feat_ic,
        "ic": ic,
        "oof_mae": oof_mae,
        "oof_directional_acc": oof_directional_acc,
        **ridge_diag
    }


def run_comparison(
    feature_path: str,
    panel_path: str,
    output_path: str,
    dtype: str = "float32",
    max_features: Optional[int] = None,
    stage3: bool = False,
    winners: list = None,
    use_extratrees: bool = False,
):
    """
    Main comparison runner (Optimized).
    """
    logger.info("=" * 60)
    logger.info("Candidate Selection Threshold Comparison (Optimized)")
    logger.info("=" * 60)
    
    float_dtype = np.float32 if dtype == "float32" else np.float64
    
    # 1. Load Data
    tlog("Loading features...")
    # Support both directory and pipeline formats
    import glob
    import re
    symbol_files = glob.glob(os.path.join(feature_path, "symbol=*.parquet"))
    
    if symbol_files:
        match = re.search(r'(\d{8}_\d{6})', feature_path)
        if match:
            ts_str = match.group(1)
            ts = pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S")
            features_dir = os.path.dirname(feature_path)
            root_dir = os.path.dirname(features_dir) if features_dir.endswith('features') else os.path.dirname(feature_path)
            feats = load_features_pipeline(ts, root_dir)
            if feats is None:
                raise ValueError(f"Failed to load features from {feature_path}")
            feats = cast_features_dtype(feats, float_dtype=float_dtype)
        else:
            raise ValueError(f"Could not parse timestamp from path: {feature_path}")
    else:
        # Generic loader
        if os.path.isdir(feature_path):
            feats = {}
            for fname in os.listdir(feature_path):
                if fname.endswith(".parquet"):
                    fpath = os.path.join(feature_path, fname)
                    feat_name = fname.replace(".parquet", "")
                    try:
                        df = pd.read_parquet(fpath)
                        if isinstance(df.index, pd.MultiIndex):
                            df = df.unstack()
                            if df.columns.nlevels > 1: df.columns = df.columns.droplevel(0)
                        feats[feat_name] = df
                    except Exception as e:
                        logger.warning(f"Failed to load {fname}: {e}")
            feats = cast_features_dtype(feats, float_dtype=float_dtype)
        else:
            raise NotImplementedError("Single file loading not fully implemented in optimization script")

    tlog("Loading panel data...")
    if os.path.isfile(panel_path):
        panel_raw = pd.read_parquet(panel_path)
    else:
        dfs = []
        for root, dirs, files in os.walk(panel_path):
            for f in files:
                if f.endswith(".parquet"):
                    df = pd.read_parquet(os.path.join(root, f))
                    if "symbol" not in df.columns:
                        sym = None
                        parts = root.split(os.sep)
                        for p in parts:
                            if p.startswith("symbol="):
                                sym = p.replace("symbol=", "")
                                break
                        if sym: df["symbol"] = sym
                    dfs.append(df)
        panel_raw = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    panel = to_panel_dict(panel_raw)
    
    # 2. Data Container & Alignment
    tlog("Initializing DataContainer and aligning...")
    data = DataContainer(panel, feats, float_dtype=float_dtype)
    data.align()

    # 3. Precomputations
    tlog("Precomputing metrics and masks...")
    metric_pack = precompute_selection_metrics(data)
    metric_by_mode_arr = {
        "fixed": metric_pack["metrics"]["fixed"].to_numpy(dtype=float_dtype, copy=False),
        "atr": metric_pack["metrics"]["atr"].to_numpy(dtype=float_dtype, copy=False) if "atr" in metric_pack["metrics"] else None,
        "vol_weight": metric_pack["metrics"]["vol_weight"].to_numpy(dtype=float_dtype, copy=False) if "vol_weight" in metric_pack["metrics"] else None,
    }

    # Configuration setup (Mirrors original script logic)
    configs = []
    
    # Default values
    default_range_pct = 0.07
    default_vol_zscore = 1.6
    default_sign_consistency = 0.70
    pct_grid = [0.06]
    modes = [("F", "fixed"), ("A", "atr"), ("VW", "vol_weight")]
    
    # Stage 1: Filter Sweeps
    for mode_prefix, mode_name in modes:
        for pct in pct_grid:
            # Range sweep
            for range_pct in [0.06, 0.07, 0.08]:
                configs.append({
                    "config_id": f"{mode_prefix}_P{int(pct*100):02d}_R{int(range_pct*100):02d}",
                    "mode": mode_name, "pct": pct,
                    "min_range_pct": range_pct, "min_vol_zscore": default_vol_zscore, "min_sign_consistency": default_sign_consistency
                })
            # Vol sweep
            for vol_z in [1.4, 1.6, 1.8]:
                configs.append({
                    "config_id": f"{mode_prefix}_P{int(pct*100):02d}_V{int(vol_z*10):02d}",
                    "mode": mode_name, "pct": pct,
                    "min_range_pct": default_range_pct, "min_vol_zscore": vol_z, "min_sign_consistency": default_sign_consistency
                })
            # SC sweep
            for sc in [0.60, 0.70, 0.80]:
                configs.append({
                    "config_id": f"{mode_prefix}_P{int(pct*100):02d}_S{int(sc*100):02d}",
                    "mode": mode_name, "pct": pct,
                    "min_range_pct": default_range_pct, "min_vol_zscore": default_vol_zscore, "min_sign_consistency": sc
                })

    # Stage 2: FULL + Expansion
    expansion_variants = [
        ("none", []), ("full", [-12, -8, -4, 4, 8, 12, 16]),
        ("neg48", [-4, -8]), ("pos48", [4, 8]), ("sym48", [-4, -8, 4, 8]),
    ]
    for mode_prefix, mode_name in modes:
        for pct in pct_grid:
            configs.append({
                "config_id": f"{mode_prefix}_P{int(pct*100):02d}_FULL",
                "mode": mode_name, "pct": pct,
                "min_range_pct": default_range_pct, "min_vol_zscore": default_vol_zscore, "min_sign_consistency": default_sign_consistency
            })

    expanded_configs = []
    for cfg in configs:
        if "_FULL" in cfg["config_id"]:
            for exp_name, exp_offsets in expansion_variants:
                cfg_e = dict(cfg)
                cfg_e["expansion_name"] = exp_name
                cfg_e["expansion_offsets"] = list(exp_offsets)
                cfg_e["config_id"] = f"{cfg['config_id']}_E{exp_name.upper()}"
                expanded_configs.append(cfg_e)

    configs = [c for c in configs if "_FULL" not in c["config_id"]] + expanded_configs
    
    # Stage 3
    if stage3 and winners:
        stage3_pcts = [0.05, 0.06, 0.07]
        stage3_configs = []
        for cfg in configs:
            base_id = cfg["config_id"].split("_E")[0] if "_E" in cfg["config_id"] else cfg["config_id"]
            if base_id in winners:
                for new_pct in stage3_pcts:
                    if new_pct != cfg.get("pct", 0.06):
                        new_cfg = dict(cfg)
                        old_pct = int(cfg.get("pct", 0.06) * 100)
                        new_cfg["config_id"] = cfg["config_id"].replace(f"P{old_pct:02d}", f"P{int(new_pct*100):02d}") + "_S3"
                        new_cfg["pct"] = new_pct
                        stage3_configs.append(new_cfg)
        if stage3_configs:
            configs = stage3_configs
            use_extratrees = True

    # Precompute masks for all thresholds
    range_thresholds = [c.get("min_range_pct") for c in configs if c.get("min_range_pct")]
    vol_thresholds = [c.get("min_vol_zscore") for c in configs if c.get("min_vol_zscore")]
    sc_thresholds = [c.get("min_sign_consistency") for c in configs if c.get("min_sign_consistency")]
    
    filter_mask_pack = precompute_filter_masks(
        data,
        range_thresholds=range_thresholds,
        vol_thresholds=vol_thresholds,
        sc_thresholds=sc_thresholds
    )

    available_model_features = [f for f in MODEL_FEATURES if f in data.feat_arr]
    if max_features:
        available_model_features = available_model_features[:max_features]

    results = []
    base_mask_cache = {}
    
    # 4. Loop
    for cfg in configs:
        config_id = cfg["config_id"]
        mode = cfg["mode"]
        pct = cfg["pct"]
        
        tlog(f"Running {config_id}...")
        
        # Select Candidates (Vectorized)
        metric_arr = metric_by_mode_arr.get(mode)
        if metric_arr is None: continue

        # Returns int8 signed array
        sign_arr = select_candidates_cross_sectional(
            metric_arr, pct,
            filter_masks=filter_mask_pack,
            min_range_pct=cfg.get("min_range_pct"),
            min_vol_zscore=cfg.get("min_vol_zscore"),
            min_sign_consistency=cfg.get("min_sign_consistency"),
            base_mask_cache=base_mask_cache,
            base_cache_key=(mode, pct)
        )

        expansion_offsets = cfg.get("expansion_offsets", [])
        if expansion_offsets:
            sign_arr = expand_signed_mask(sign_arr, expansion_offsets)

        # Boolean mask for training slices
        cand_mask_arr = sign_arr != 0
            
        # Metrics (uses signed array for directional sharpe)
        metrics = compute_learnability_metrics(
            sign_arr, data, available_model_features, float_dtype, use_extratrees
        )

        # Training Slices (needs boolean mask)
        cfg_variant = deepcopy(CFG)
        cfg_variant["train_extreme_pct_hourly"] = pct

        slice_rows = evaluate_training_slices(
            cand_mask_arr, data, cfg_variant, horizons=[2, 4, 8], sample_frac=0.33
        )
        metrics.update(aggregate_slice_rows(slice_rows))

        # Add config fields to result
        res = dict(cfg)
        res.update(metrics)
        results.append(res)

        # Print summary line
        print(f"  {config_id:<25} IC={metrics.get('ic',0):.4f} Sharpe={metrics.get('sharpe',0):.2f}")

    # Output
    results_df = pd.DataFrame(results)
    output_dir = os.path.dirname(output_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    tlog(f"Saved results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare candidate selection thresholds (Optimized)"
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--panel", required=False)
    parser.add_argument("--output", default="reports/candidate_threshold_comparison.csv")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-features", type=int, default=60)
    parser.add_argument("--stage3", action="store_true")
    parser.add_argument("--winners", nargs="+", default=[])
    
    args = parser.parse_args()
    
    run_comparison(
        args.features, args.panel, args.output,
        dtype=args.dtype, max_features=args.max_features,
        stage3=args.stage3, winners=args.winners
    )
