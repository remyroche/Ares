#!/usr/bin/env python
"""
Candidate Selection Threshold Comparison

Compares Fixed, ATR-normalized, and Volume-Weighted candidate selection methods.
Uses ExtraTrees with the same parameters as the target race in training.py.

Usage:
    python scripts/compare_candidate_thresholds.py \
        --features data/features/20260214_190000 \
        --panel data/klines \
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
from typing import Dict, Optional, Any
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
    panel: Optional[dict],
    feats: dict,
    float_dtype: np.dtype,
) -> Optional[pd.DataFrame]:
    """Compute sign consistency once globally (prefer training-aligned feature source)."""
    sc_df = None
    sc_source = "none"

    # Prefer feature-based sign consistency to match training universe behavior.
    if "sign_consistency" in feats:
        sc_df = feats["sign_consistency"]
        sc_source = "features.sign_consistency"
    elif "sign_consistency_12h" in feats:
        sc_df = feats["sign_consistency_12h"]
        sc_source = "features.sign_consistency_12h"
    else:
        # Training-aligned fallback used in apply_training_filters:
        # abs(rolling_mean(sign(base_ret), 12, min_periods=6))
        # where base_ret is ret6h, or ret24h if ret6h unavailable.
        base_ret = feats.get("ret6h")
        if base_ret is None:
            base_ret = feats.get("ret24h")
        if base_ret is not None:
            sign_mean = np.sign(base_ret).rolling(12, min_periods=6).mean().abs()
            sc_df = sign_mean.astype(float_dtype, copy=False)
            sc_source = "features.sign_roll_mean_abs_12"
        elif panel is not None and "close" in panel:
            # Final fallback for robustness only (may not match production exactly).
            try:
                sc_arr = ff.numba_sign_consistency(panel["close"], 12)
                sc_df = pd.DataFrame(sc_arr, index=panel["close"].index, columns=panel["close"].columns)
                sc_source = "panel.numba_sign_consistency"
            except Exception as exc:
                tlog(f"Sign-consistency fallback failed: {exc}")
                sc_df = None
                sc_source = "none"

    if sc_df is None:
        tlog("Sign-consistency source: unavailable")
        return None

    sc_df = sc_df.astype(float_dtype, copy=False)
    sample_arr = sc_df.to_numpy(dtype=float_dtype, copy=False)
    # Lightweight sample for scale detection/stat logging.
    row_step = max(1, sample_arr.shape[0] // 512)
    col_step = max(1, sample_arr.shape[1] // 64)
    sample = sample_arr[::row_step, ::col_step].reshape(-1)
    sample = sample[np.isfinite(sample)]

    sc_scale = "unknown"
    if sample.size > 0:
        q50 = float(np.quantile(sample, 0.50))
        q90 = float(np.quantile(sample, 0.90))
        q99 = float(np.quantile(sample, 0.99))
        if q99 > 1.5:
            # Convert 0..100 style scale to 0..1 to match config thresholds (0.70, 0.80, 0.90).
            sc_df = (sc_df / np.float32(100.0)).astype(float_dtype, copy=False)
            sc_scale = "percent_to_ratio"
        else:
            sc_scale = "ratio"
        tlog(
            f"Sign-consistency source={sc_source}, scale={sc_scale}, "
            f"sample_q50={q50:.4f}, sample_q90={q90:.4f}, sample_q99={q99:.4f}"
        )
    else:
        tlog(f"Sign-consistency source={sc_source}, sample has no finite values")

    return sc_df


def _first_available_feature(feats: dict, names: list[str]) -> Optional[pd.DataFrame]:
    for name in names:
        if name in feats:
            return feats[name]
    return None


def precompute_filter_masks(
    feats: dict,
    panel: Optional[dict],
    target_index: pd.Index,
    target_columns: pd.Index,
    range_thresholds: list[float],
    vol_thresholds: list[float],
    sc_thresholds: list[float],
    float_dtype: np.dtype,
) -> dict:
    """Precompute boolean filter masks once per threshold value."""
    true_arr = np.ones((len(target_index), len(target_columns)), dtype=bool)

    range_feat = _first_available_feature(feats, ["range_12h_pct", "range_16h_pct", "range_pct"])
    vol_feat = _first_available_feature(feats, ["volatility_zscore", "vol_z"])
    sc_df = compute_global_sign_consistency(panel=panel, feats=feats, float_dtype=float_dtype)

    range_masks: dict[float, np.ndarray] = {}
    vol_masks: dict[float, np.ndarray] = {}
    sc_masks: dict[float, np.ndarray] = {}

    if range_feat is not None:
        range_aligned = range_feat.reindex(index=target_index, columns=target_columns)
        range_arr = range_aligned.to_numpy(dtype=float_dtype, copy=False)
        for thr in sorted({float(v) for v in range_thresholds}):
            range_masks[thr] = range_arr > thr
    if vol_feat is not None:
        vol_aligned = vol_feat.reindex(index=target_index, columns=target_columns)
        vol_arr = vol_aligned.to_numpy(dtype=float_dtype, copy=False)
        for thr in sorted({float(v) for v in vol_thresholds}):
            vol_masks[thr] = vol_arr > thr
    if sc_df is not None:
        sc_aligned = sc_df.reindex(index=target_index, columns=target_columns)
        sc_arr = sc_aligned.to_numpy(dtype=float_dtype, copy=False)
        sc_finite_ratio = float(np.isfinite(sc_arr).mean())
        tlog(
            "Sign-consistency aligned coverage: "
            f"finite={sc_finite_ratio:.2%}, shape={sc_arr.shape}"
        )
        for thr in sorted({float(v) for v in sc_thresholds}):
            thr_eff = thr / 100.0 if thr > 1.0 else thr
            mask = sc_arr >= thr_eff
            sc_masks[thr] = mask
            tlog(
                f"Sign-consistency threshold mask: thr={thr:.4f} "
                f"(effective={thr_eff:.4f}) pass={int(mask.sum())}"
            )
    else:
        tlog("Sign-consistency mask build skipped: feature not available")

    return {
        "index": target_index,
        "columns": target_columns,
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


def build_proxy_mkt_gates(feats: dict) -> pd.DataFrame:
    """Build minimal market gates DataFrame required by training set builder."""
    vol_df = feats.get("volatility_zscore")
    if vol_df is None:
        vol_df = feats.get("vol_z")
    trend_df = feats.get("trend_pct")
    if trend_df is None:
        trend_df = feats.get("ret6h")
    if trend_df is None:
        trend_df = feats.get("ret24h")

    if vol_df is None or trend_df is None:
        idx = next(iter(feats.values())).index
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


def apply_training_filters(
    candidate_mask: pd.DataFrame,
    feats: dict,
    min_range_pct: Optional[float] = None,
    min_vol_zscore: Optional[float] = None,
    min_sign_consistency: Optional[float] = None,
) -> pd.DataFrame:
    """Apply training-like candidate prefilters to a candidate mask."""
    filt = pd.DataFrame(True, index=candidate_mask.index, columns=candidate_mask.columns)

    if min_range_pct is not None:
        range_feat = feats.get("range_12h_pct")
        if range_feat is None:
            range_feat = feats.get("range_16h_pct")
        if range_feat is None:
            range_feat = feats.get("range_pct")
        if range_feat is not None:
            filt &= range_feat >= float(min_range_pct)

    if min_vol_zscore is not None:
        vol_feat = feats.get("volatility_zscore")
        if vol_feat is None:
            vol_feat = feats.get("vol_z")
        if vol_feat is not None:
            filt &= vol_feat >= float(min_vol_zscore)

    if min_sign_consistency is not None:
        sc_feat = feats.get("sign_consistency")
        if sc_feat is None:
            base_ret = feats.get("ret6h")
            if base_ret is None:
                base_ret = feats.get("ret24h")
            if base_ret is not None:
                sign_mean = np.sign(base_ret).rolling(12, min_periods=6).mean().abs()
                sc_feat = sign_mean.astype(np.float32)
        if sc_feat is not None:
            filt &= sc_feat >= float(min_sign_consistency)

    return (candidate_mask & filt).fillna(False)


def fingerprint_candidate_mask(candidate_mask: pd.DataFrame) -> str:
    """Compact deterministic fingerprint for caching expensive downstream computations."""
    arr = candidate_mask.to_numpy(dtype=bool, copy=False).reshape(-1)
    packed = np.packbits(arr.astype(np.uint8, copy=False))
    h = hashlib.blake2b(digest_size=16)
    h.update(str(candidate_mask.shape).encode("ascii"))
    h.update(packed.tobytes())
    return h.hexdigest()


def _normalize_symbol(sym: Any) -> str:
    s = str(sym).upper()
    return "".join(ch for ch in s if ch.isalnum())


def align_candidate_mask_to_panel_symbols(candidate_mask: pd.DataFrame, panel: dict) -> pd.DataFrame:
    """Align candidate mask symbols to panel close symbols (exact first, normalized fallback)."""
    if "close" not in panel:
        return candidate_mask
    panel_cols = list(panel["close"].columns)
    panel_set = set(panel_cols)
    cand_cols = list(candidate_mask.columns)

    if all(c in panel_set for c in cand_cols):
        return candidate_mask

    norm_to_panel: dict[str, str] = {}
    dup_norm = set()
    for p in panel_cols:
        k = _normalize_symbol(p)
        if k in norm_to_panel and norm_to_panel[k] != p:
            dup_norm.add(k)
        else:
            norm_to_panel[k] = p
    for k in dup_norm:
        norm_to_panel.pop(k, None)

    rename_map: dict[Any, Any] = {}
    for c in cand_cols:
        if c in panel_set:
            continue
        mapped = norm_to_panel.get(_normalize_symbol(c))
        if mapped is not None:
            rename_map[c] = mapped

    aligned = candidate_mask.rename(columns=rename_map)
    if aligned.columns.has_duplicates:
        aligned = aligned.T.groupby(level=0).any().T
    keep_cols = [c for c in panel_cols if c in set(aligned.columns)]
    # Safety: if mapping fails completely, keep original mask rather than dropping all columns.
    if len(keep_cols) == 0:
        return candidate_mask
    aligned = aligned.reindex(columns=keep_cols, fill_value=False)
    return aligned


def training_slice_precheck(candidate_mask: pd.DataFrame, panel: dict) -> tuple[bool, str]:
    """Fast precheck to decide if training slices can produce rows."""
    if "close" not in panel:
        return False, "panel.close missing"
    close_df = panel["close"]
    if candidate_mask.empty:
        return False, "candidate mask empty"
    overlap = close_df.columns.intersection(candidate_mask.columns)
    if len(overlap) == 0:
        return False, "no symbol overlap between candidate mask and panel close"
    if int(candidate_mask.to_numpy(dtype=bool, copy=False).sum()) == 0:
        return False, "candidate mask has zero selected events"
    return True, ""


def _bucket_move_bucket(side: str, kind: str) -> str:
    """Map trade bucket to move bucket used by pipeline filtering."""
    if side == "long":
        cand_filter = "worst" if kind == "mr" else "best"
    else:
        cand_filter = "best" if kind == "mr" else "worst"
    return "up" if cand_filter == "best" else "down"


def _build_bucket_candidate_mask(
    candidate_mask: pd.DataFrame,
    feats: dict,
    move_bucket: str,
) -> pd.DataFrame:
    """Create per-bucket candidate mask (up/down) using trend sign filter."""
    trend_df = feats.get("trend_pct")
    if trend_df is None:
        # If trend is unavailable, keep original mask.
        return candidate_mask
    if not isinstance(trend_df, pd.DataFrame):
        if isinstance(trend_df, np.ndarray) and trend_df.shape == candidate_mask.shape:
            trend_df = pd.DataFrame(trend_df, index=candidate_mask.index, columns=candidate_mask.columns)
        else:
            tlog("Bucket mask: trend_pct is non-DataFrame/incompatible; skipping move-bucket filter")
            return candidate_mask
    trend_aligned = trend_df.reindex(index=candidate_mask.index, columns=candidate_mask.columns)
    if move_bucket == "up":
        trend_mask = trend_aligned > 0
    else:
        trend_mask = trend_aligned <= 0
    return (candidate_mask & trend_mask).fillna(False)


def infer_stage_label(config_id: str) -> str:
    """Infer sweep stage from config id."""
    if "_S3" in config_id:
        return "Stage 3"
    if "_FULL_" in config_id:
        return "Stage 2"
    return "Stage 1"


def evaluate_training_slices(
    candidate_mask: pd.DataFrame,
    feats: dict,
    panel: dict,
    cfg_variant: dict,
    horizons: list,
    cache: Optional[dict] = None,
    cache_key: Optional[tuple] = None,
    sample_frac: float = 1.0,
) -> list:
    """Evaluate MR/TF x long/short slices reusing training labeling & sampling logic."""
    # Defensive: training barrier/grid code expects atr_pct as DataFrame aligned to panel.
    close_df = panel.get("close")
    if close_df is not None and "atr_pct" in feats and not isinstance(feats["atr_pct"], pd.DataFrame):
        atr_obj = feats["atr_pct"]
        if isinstance(atr_obj, np.ndarray) and atr_obj.shape == close_df.shape:
            feats["atr_pct"] = pd.DataFrame(atr_obj, index=close_df.index, columns=close_df.columns)
            tlog("Training slices: coerced feats['atr_pct'] ndarray -> DataFrame")
        else:
            feats["atr_pct"] = pd.DataFrame(0.02, index=close_df.index, columns=close_df.columns, dtype=np.float32)
            tlog(
                "Training slices: atr_pct was non-DataFrame with incompatible shape; "
                "replaced with fallback 0.02 DataFrame"
            )

    # Apply sample subsampling for faster execution
    if sample_frac < 1.0:
        n_rows = len(candidate_mask)
        n_sample = int(n_rows * sample_frac)
        np.random.seed(42)  # Reproducible subsampling
        sample_idx = np.sort(np.random.choice(n_rows, n_sample, replace=False))
        candidate_mask = candidate_mask.iloc[sample_idx].copy()
        tlog(f"Training slices: subsampled {n_sample}/{n_rows} rows ({sample_frac*100:.0f}%)")
    if cache is not None and cache_key is not None and cache_key in cache:
        tlog("Training-slice cache hit")
        return [dict(r) for r in cache[cache_key]]

    tlog("Training-slice evaluation start")
    rows = []
    mkt_gates = build_proxy_mkt_gates(feats)
    ts_end = candidate_mask.index.max()
    tb_cache_by_h_side, geom_cache_by_h_side = build_grid_aggregated_tb_cache(
        panel=panel,
        feats=feats,
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
                bucket_mask_cache[(side, kind)] = _build_bucket_candidate_mask(
                    candidate_mask=candidate_mask,
                    feats=feats,
                    move_bucket=move_bucket,
                )
            bucket_mask = bucket_mask_cache[(side, kind)]
            syms = list(bucket_mask.columns)
            bucket_n = int(bucket_mask.to_numpy(dtype=bool, copy=False).sum())
            tlog(
                f"Training slice bucket mask: side={side}, kind={kind}, "
                f"move_bucket={move_bucket}, selected={bucket_n}"
            )
            skip_remaining_horizons = False

            for h_i, h in enumerate(horizons):
                if skip_remaining_horizons:
                    rows.append(
                        {
                            "slice": f"{side}_{kind}",
                            "side": side,
                            "kind": kind,
                            "horizon": h,
                            "n_samples": 0,
                            "label_pos_rate": 0,
                            "mean_ret_bps": 0,
                            "sharpe": 0,
                            "sortino": 0,
                            "weighted_ret_bps": 0,
                        }
                    )
                    continue

                tlog(f"Building training set: side={side}, kind={kind}, H={h}")
                if (h, side) in geom_cache_by_h_side:
                    feats["__geom_n_tp__"] = geom_cache_by_h_side[(h, side)]["n_tp"]
                    feats["__geom_n_sl__"] = geom_cache_by_h_side[(h, side)]["n_sl"]
                    feats["__geom_n_to__"] = geom_cache_by_h_side[(h, side)]["n_to"]
                X, y_bin, y_ret, cols, w, meta_idx = build_hourly_training_set_and_weights(
                    panel=panel,
                    feats=feats,
                    mkt_gates=mkt_gates,
                    cfg=cfg_variant,
                    syms=syms,
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
                    rows.append(
                        {
                            "slice": f"{side}_{kind}",
                            "side": side,
                            "kind": kind,
                            "horizon": h,
                            "n_samples": 0,
                            "label_pos_rate": 0,
                            "mean_ret_bps": 0,
                            "sharpe": 0,
                            "sortino": 0,
                            "weighted_ret_bps": 0,
                        }
                    )
                    if structurally_empty:
                        remaining = len(horizons) - (h_i + 1)
                        if remaining > 0:
                            tlog(
                                f"Short-circuit remaining horizons for side={side}, kind={kind} "
                                f"after empty training set at H={h} (remaining={remaining})"
                            )
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
# Candidate Selection Functions
# =============================================================================

def precompute_selection_metrics(feats: dict, float_dtype: np.dtype) -> dict:
    """Precompute selection metrics once and reuse across configs.

    ATR-normalized paths use robust ATR denominator (winsorized + EWM smoothed)
    consistently to reduce denominator noise/spikes.
    """
    ret_base = feats.get("ret6h")
    if ret_base is None:
        ret_base = feats.get("ret24h")
    if ret_base is None:
        raise ValueError("ret6h/ret24h not found in features")
    ret_base = ret_base.astype(float_dtype, copy=False)

    metrics = {"fixed": ret_base}

    atr_effective = None
    atr_pct = feats.get("atr_pct")
    if atr_pct is not None:
        atr_pack = preprocess_atr(atr_pct.astype(float_dtype, copy=False))
        # Use robust denominator everywhere for ATR mode.
        atr_effective = atr_pack["atr_robust"].astype(float_dtype, copy=False)
        metrics["atr"] = (ret_base / atr_pack["atr_robust"]).astype(float_dtype, copy=False)
        metrics["atr_robust"] = (ret_base / atr_pack["atr_robust"]).astype(float_dtype, copy=False)

    rvol_z = feats.get("rvol_z")
    volu_z = feats.get("volu_z")
    if rvol_z is not None and volu_z is not None:
        vol_combined = ((rvol_z.astype(float_dtype, copy=False) + volu_z.astype(float_dtype, copy=False)) / 2).astype(float_dtype, copy=False)
        metrics["vol_weight"] = (ret_base.abs() * vol_combined.clip(lower=0) * np.sign(ret_base)).astype(float_dtype, copy=False)

    return {
        "metrics": metrics,
        "atr_effective": atr_effective,
    }


def select_candidates_non_cross_sectional_atr(
    metric: pd.DataFrame,
    pct: float,
    ewm_span: int = 240,
    min_periods: int = 72,
) -> pd.DataFrame:
    """
    Non-cross-sectional ATR-normalized selection.

    Uses per-symbol EWM z-score over time and selects tail events by |z| >= z_thr,
    where z_thr is mapped from pct using a normal quantile.
    """
    z_thr = float(stats.norm.ppf(1.0 - pct))
    ewm_mean = metric.ewm(span=ewm_span, min_periods=min_periods, adjust=False).mean()
    ewm_std = metric.ewm(span=ewm_span, min_periods=min_periods, adjust=False).std()
    ewm_std = ewm_std.replace(0, np.nan)

    z = (metric - ewm_mean) / ewm_std
    abs_z = z.abs()
    flat = abs_z.stack(future_stack=True).dropna()
    if len(flat) == 0:
        return pd.DataFrame(False, index=metric.index, columns=metric.columns)

    # target average selection density ~= cross-sectional two-sided density (2*pct)
    thr = float(flat.quantile(max(0.0, 1.0 - 2.0 * pct)))
    thr = max(thr, z_thr)
    mask = abs_z.ge(thr) & metric.notna()
    return mask.fillna(False)


def select_candidates_cross_sectional_side_aware(
    metric: pd.DataFrame,
    long_pct: float,
    short_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-sectional side-aware selection with independent long/short thresholds."""
    n_symbols = metric.shape[1]
    k_long = max(1, int(n_symbols * long_pct))
    k_short = max(1, int(n_symbols * short_pct))

    ranks = metric.rank(axis=1, method="first")
    valid_counts = metric.notna().sum(axis=1)

    vc = valid_counts.values[:, np.newaxis]
    r = ranks.values
    valid = metric.notna().values

    long_arr = (r > (vc - k_long)) & valid
    short_arr = (r <= k_short) & valid

    min_needed = max(k_long, k_short)
    invalid_rows = valid_counts.values < min_needed
    long_arr[invalid_rows, :] = False
    short_arr[invalid_rows, :] = False

    mask_arr = long_arr | short_arr
    sign_arr = np.zeros_like(r, dtype=np.int8)
    sign_arr[long_arr] = 1
    sign_arr[short_arr] = -1

    return (
        pd.DataFrame(mask_arr, index=metric.index, columns=metric.columns),
        pd.DataFrame(sign_arr, index=metric.index, columns=metric.columns),
    )


def select_candidates_atr_quintiles(
    metric: pd.DataFrame,
    atr_effective: pd.DataFrame,
    pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select tails within ATR quintiles using vectorized bucket assignment + NumPy selection."""
    m_arr = metric.to_numpy(dtype=np.float32, copy=False)
    a_arr = atr_effective.to_numpy(dtype=np.float32, copy=False)
    n_rows, n_cols = m_arr.shape

    # Vectorized ATR quintile assignment per row (0..4), NaN where invalid.
    # Equivalent intent to qcut per row, with much lower overhead.
    atr_rank_pct = atr_effective.rank(axis=1, method="first", pct=True).to_numpy(dtype=np.float32, copy=False)
    bucket_arr = np.floor(atr_rank_pct * 5.0).astype(np.int8, copy=False)
    bucket_arr = np.clip(bucket_arr, 0, 4)

    valid = np.isfinite(m_arr) & np.isfinite(a_arr)
    mask_arr = np.zeros((n_rows, n_cols), dtype=bool)
    sign_arr = np.zeros((n_rows, n_cols), dtype=np.int8)

    for i in range(n_rows):
        valid_i = valid[i]
        if int(valid_i.sum()) < 20:
            continue
        m_i = m_arr[i]
        b_i = bucket_arr[i]
        for b in range(5):
            idx = np.flatnonzero(valid_i & (b_i == b))
            n_bucket = int(len(idx))
            if n_bucket < 4:
                continue
            k = max(1, int(n_bucket * pct))
            vals = m_i[idx]

            # bottom k
            bot_local = np.argpartition(vals, k - 1)[:k]
            bot_idx = idx[bot_local]
            # top k
            top_local = np.argpartition(vals, n_bucket - k)[n_bucket - k:]
            top_idx = idx[top_local]

            mask_arr[i, bot_idx] = True
            sign_arr[i, bot_idx] = -1
            mask_arr[i, top_idx] = True
            sign_arr[i, top_idx] = 1

    return (
        pd.DataFrame(mask_arr, index=metric.index, columns=metric.columns, dtype=bool),
        pd.DataFrame(sign_arr, index=metric.index, columns=metric.columns, dtype=np.int8),
    )


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

    filtered = np.array(base_mask_arr, copy=True, dtype=bool)

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


def compute_cross_sectional_base_mask(metric: pd.DataFrame, pct: float) -> np.ndarray:
    """Compute cross-sectional base mask as ndarray (top/bottom pct)."""
    n_symbols = metric.shape[1]
    k = max(1, int(n_symbols * pct))

    ranks = metric.rank(axis=1, method="first")
    valid_counts = metric.notna().sum(axis=1)

    vc = valid_counts.to_numpy()[:, np.newaxis]
    r = ranks.to_numpy(copy=False)
    valid = metric.notna().to_numpy(copy=False)

    mask_arr = ((r > (vc - k)) | (r <= k)) & valid
    mask_arr[valid_counts.to_numpy() < k, :] = False
    return mask_arr


def expand_candidate_mask(base_mask: pd.DataFrame, offsets: list[int]) -> pd.DataFrame:
    """Expand candidate timestamps by OR-ing shifted copies of the base mask."""
    if not offsets:
        return base_mask
    expanded = base_mask.copy()
    for off in offsets:
        shifted = base_mask.shift(int(off)).fillna(False)
        expanded |= shifted
    return expanded


def select_candidates_cross_sectional(
    metric: pd.DataFrame,
    pct: float,
    filter_masks: Optional[dict] = None,
    min_range_pct: float = None,
    min_vol_zscore: float = None,
    min_sign_consistency: float = None,
    base_mask_cache: Optional[dict] = None,
    base_cache_key: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Unified cross-sectional selection for all three modes.
    
    Parameters
    ----------
    metric : pd.DataFrame
        Precomputed metric DataFrame used for ranking
    pct : float
        Percentage of symbols to select from top and bottom (e.g., 0.08 = 8%)
    filter_masks : dict, optional
        Precomputed boolean masks keyed by threshold values
    min_range_pct : float, optional
        Minimum 12h range filter (None = no filter)
    min_vol_zscore : float, optional
        Minimum volatility z-score filter (None = no filter)
    min_sign_consistency : float, optional
        Minimum sign consistency filter (None = no filter)
    
    Returns
    -------
    pd.DataFrame
        Boolean mask of selected candidates (True = selected)
    """
    if base_mask_cache is not None and base_cache_key is not None and base_cache_key in base_mask_cache:
        base_mask_arr = base_mask_cache[base_cache_key]
    else:
        base_mask_arr = compute_cross_sectional_base_mask(metric, pct)
        if base_mask_cache is not None and base_cache_key is not None:
            base_mask_cache[base_cache_key] = base_mask_arr

    has_filters = any(
        [
            min_range_pct is not None,
            min_vol_zscore is not None,
            min_sign_consistency is not None,
        ]
    )
    if has_filters:
        mask_arr = apply_quality_filters_array(
            base_mask_arr=base_mask_arr,
            filter_masks=filter_masks,
            min_range_pct=min_range_pct,
            min_vol_zscore=min_vol_zscore,
            min_sign_consistency=min_sign_consistency,
        )
    else:
        mask_arr = np.array(base_mask_arr, copy=True, dtype=bool)

    return pd.DataFrame(mask_arr, index=metric.index, columns=metric.columns, dtype=bool)


def build_long_form_tables(
    feats: dict,
    target_col: str,
    float_dtype: np.dtype,
    atr_reference: Optional[pd.DataFrame] = None,
) -> dict:
    """Build reusable long-form base table once; feature tables are materialized lazily."""
    target = feats.get(target_col)
    if target is None:
        target = feats.get("ret24h")
        if target is None:
            raise ValueError(f"Target column {target_col} or ret24h not found in features")

    ret_base = feats.get("ret6h")
    if ret_base is None:
        ret_base = feats.get("ret24h")
    if ret_base is None:
        raise ValueError("ret6h/ret24h not found in features")

    # Keep ATR diagnostics aligned with the same denominator used by ATR mode ranking.
    atr_pct = atr_reference
    if atr_pct is None:
        atr_pct = feats.get("atr_pct")
    if atr_pct is None:
        atr_pct = pd.DataFrame(np.nan, index=ret_base.index, columns=ret_base.columns)

    valid_counts = target.notna().sum(axis=1)
    threshold = target.quantile(0.65, axis=1)
    labels = target.gt(threshold, axis=0).astype(np.float32)
    labels = labels.where(valid_counts >= 10, np.nan)

    base_long = pd.DataFrame(
        {
            "ret_base": ret_base.stack(future_stack=True).astype(float_dtype, copy=False),
            "target": target.stack(future_stack=True).astype(float_dtype, copy=False),
            "atr_pct": atr_pct.stack(future_stack=True).astype(float_dtype, copy=False),
            "label": labels.stack(future_stack=True).astype(np.float32, copy=False),
        }
    )

    return {
        "base_long": base_long,
        "feats": feats,
        "float_dtype": float_dtype,
        "feature_series_cache": {},
    }


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
    precomputed: dict,
    target_candidates: pd.Series,
    available_features: list,
    candidate_row_idx: np.ndarray,
    candidate_col_idx: np.ndarray,
    candidate_index_ref: pd.Index,
    candidate_columns_ref: pd.Index,
    top_n: int = 20
) -> float:
    """
    Compute mean |IC| of top features with target.
    
    Uses direct 2D gathers from feature matrices to avoid expensive full stacking.
    """
    if not available_features:
        logger.warning("No MODEL_FEATURES found in feature data")
        return 0.0

    target_idx = target_candidates.index
    if len(target_idx) == 0:
        return 0.0
    y_all = target_candidates.to_numpy(dtype=np.float32, copy=False)

    ics = []
    for feat_name in available_features[:top_n * 2]:  # Check more than needed
        feat_df = precomputed.get("feats", {}).get(feat_name)
        if feat_df is None:
            continue
        if (
            feat_df.shape[0] != len(candidate_index_ref)
            or feat_df.shape[1] != len(candidate_columns_ref)
            or not feat_df.index.equals(candidate_index_ref)
            or not feat_df.columns.equals(candidate_columns_ref)
        ):
            feat_df = feat_df.reindex(index=candidate_index_ref, columns=candidate_columns_ref)
        feat_arr = feat_df.to_numpy(dtype=np.float32, copy=False)
        x_all = feat_arr[candidate_row_idx, candidate_col_idx]
        del feat_arr, feat_df
        valid = np.isfinite(x_all) & np.isfinite(y_all)
        if valid.sum() == 0:
            continue
        joined = pd.DataFrame(
            {
                "x": x_all[valid],
                "y": y_all[valid],
            },
            index=target_idx[valid],
        )

        if joined.empty:
            continue

        timestamp_ics = joined.groupby(level=0).apply(
            lambda g: safe_pearson_corr(
                g["x"].to_numpy(dtype=np.float32, copy=False),
                g["y"].to_numpy(dtype=np.float32, copy=False),
            ) if len(g) >= 5 else np.nan
        )
        timestamp_ics = timestamp_ics[np.isfinite(timestamp_ics)]

        if not timestamp_ics.empty:
            ics.append(np.abs(timestamp_ics.values).mean())

    if not ics:
        return 0.0

    # Return mean of top N by |IC|
    ics_sorted = sorted(ics, reverse=True)[:top_n]
    return np.mean(ics_sorted)


def get_stacked_feature_series(
    precomputed: dict,
    feat_name: str,
    use_cache: bool = True,
) -> Optional[pd.Series]:
    """Lazy stack + cache feature series as MultiIndex(ts, symbol)->value."""
    cache = precomputed.get("feature_series_cache")
    if use_cache:
        if cache is None:
            cache = {}
            precomputed["feature_series_cache"] = cache
        if feat_name in cache:
            return cache[feat_name]
    feats = precomputed.get("feats", {})
    if feat_name not in feats:
        return None
    float_dtype = precomputed.get("float_dtype", np.float32)
    s = feats[feat_name].stack(future_stack=True).astype(float_dtype, copy=False)
    if use_cache:
        cache[feat_name] = s
    return s


def materialize_feature_matrix(
    precomputed: dict,
    available_features: list,
    candidate_row_idx: np.ndarray,
    candidate_col_idx: np.ndarray,
    candidate_index_ref: pd.Index,
    candidate_columns_ref: pd.Index,
    float_dtype: np.dtype,
) -> tuple[np.ndarray, list[str]]:
    """Materialize candidate feature matrix in chunks to cap peak memory."""
    feats = precomputed.get("feats", {})
    used = [f for f in available_features if f in feats]
    if not used:
        return np.empty((len(candidate_row_idx), 0), dtype=float_dtype), []

    X = np.empty((len(candidate_row_idx), len(used)), dtype=float_dtype)
    for start in range(0, len(used), FEATURE_CHUNK_SIZE):
        end = min(start + FEATURE_CHUNK_SIZE, len(used))
        chunk_feats = used[start:end]
        for offset, feat_name in enumerate(chunk_feats):
            feat_df = feats.get(feat_name)
            if feat_df is None:
                X[:, start + offset] = np.nan
                continue
            if (
                feat_df.shape[0] != len(candidate_index_ref)
                or feat_df.shape[1] != len(candidate_columns_ref)
                or not feat_df.index.equals(candidate_index_ref)
                or not feat_df.columns.equals(candidate_columns_ref)
            ):
                feat_df = feat_df.reindex(index=candidate_index_ref, columns=candidate_columns_ref)
            feat_arr = feat_df.to_numpy(dtype=float_dtype, copy=False)
            vals = feat_arr[candidate_row_idx, candidate_col_idx]
            X[:, start + offset] = vals
            del vals, feat_arr, feat_df
        gc.collect()

    return X, used


def _safe_jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 1.0
    return float(len(a & b) / u)


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
    Run purged k-fold CV with two-stage modeling:
    RobustScaler + Ridge screen (top-k by |coef|) -> ExtraTrees.
    
    If use_extratrees=False, only runs Ridge regression for fast IC estimation.
    """
    n_samples = len(y)
    oof = np.full(n_samples, np.nan, dtype=float_dtype)
    
    # Handle NaN in target
    y_valid = np.isfinite(y)
    y_filled = np.where(y_valid, y, np.nanmedian(y[y_valid])).astype(float_dtype, copy=False)
    
    # Create time index for purging
    time_idx = np.arange(n_samples, dtype=np.int32)
    
    pkf = PurgedKFold(n_splits=n_splits, purge=purge, embargo=2)
    
    splits = list(pkf.split(time_idx))
    selected_sets: list[set[int]] = []
    selected_ks: list[int] = []

    for fold_i, (train_idx, val_idx) in enumerate(splits, start=1):
        tlog(f"OOF fold {fold_i}/{len(splits)}: train={len(train_idx)}, val={len(val_idx)}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y_filled[train_idx]

        sw_train = sample_weights[train_idx] if sample_weights is not None else None
        if len(train_idx) > OOF_RIDGE_MAX_TRAIN_SAMPLES:
            ridge_sub_idx = _uniform_subsample_idx(
                len(train_idx),
                OOF_RIDGE_MAX_TRAIN_SAMPLES,
                seed=(random_state + fold_i * 1009),
            )
            X_ridge = X_train[ridge_sub_idx]
            y_ridge = y_train[ridge_sub_idx]
            sw_ridge = sw_train[ridge_sub_idx] if sw_train is not None else None
            tlog(
                f"OOF fold {fold_i}: Ridge subsample "
                f"{len(ridge_sub_idx)}/{len(train_idx)}"
            )
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
        selected_sets.append(set(selected_idx.tolist()))
        selected_ks.append(int(len(selected_idx)))
        tlog(f"OOF fold {fold_i}: Ridge-screen selected {len(selected_idx)} candidate generation variants")

        X_train_sel = X_train[:, selected_idx]
        X_val_sel = X_val[:, selected_idx]

        if use_extratrees:
            # Train ExtraTrees with target race parameters on screened features
            model = ExtraTreesRegressor(**{**ET_REGRESSOR_PARAMS, "random_state": random_state})
            model.fit(X_train_sel, y_train, sample_weight=sw_train)
            oof[val_idx] = model.predict(X_val_sel)
        else:
            # Use Ridge only for fast IC estimation (Stage 1 & 2)
            from sklearn.linear_model import Ridge
            ridge_model = Ridge(alpha=ridge_alpha)
            ridge_model.fit(X_train_sel, y_train, sample_weight=sw_train)
            oof[val_idx] = ridge_model.predict(X_val_sel)

        del X_train, X_val, X_train_sel, X_val_sel, y_train, sw_train, X_ridge, y_ridge, sw_ridge
        if use_extratrees:
            del model
        else:
            del ridge_model
        gc.collect()

    jaccards = []
    for i in range(len(selected_sets)):
        for j in range(i + 1, len(selected_sets)):
            jaccards.append(_safe_jaccard(selected_sets[i], selected_sets[j]))

    replacement_rates = []
    for i in range(1, len(selected_sets)):
        prev_set = selected_sets[i - 1]
        curr_set = selected_sets[i]
        denom = max(1, len(curr_set))
        replacement_rates.append(float(len(curr_set - prev_set) / denom))

    diagnostics = {
        "ridge_alpha": float(ridge_alpha),
        "ridge_top_frac": float(ridge_top_frac),
        "ridge_selected_k_mean": float(np.mean(selected_ks)) if selected_ks else 0.0,
        "ridge_jaccard_median": float(np.median(jaccards)) if jaccards else 1.0,
        "ridge_replacement_rate_median": float(np.median(replacement_rates)) if replacement_rates else 0.0,
    }
    tlog(
        "OOF ridge stability: "
        f"jaccard_med={diagnostics['ridge_jaccard_median']:.3f}, "
        f"replace_med={diagnostics['ridge_replacement_rate_median']:.3f}"
    )
    return oof, diagnostics


def compute_learnability_metrics(
    candidate_mask: pd.DataFrame,
    precomputed: dict,
    available_features: list,
    float_dtype: np.dtype,
    side_sign: Optional[pd.DataFrame] = None,
    use_extratrees: bool = False,
) -> dict:
    """
    Compute all learnability metrics for a candidate selection method.
    
    Parameters
    ----------
    candidate_mask : pd.DataFrame
        Boolean mask of selected candidates
    precomputed : dict
        Dictionary containing precomputed long-form base/feature tables
    available_features : list
        Cached list of available model feature names
    float_dtype : np.dtype
        Dtype used for feature matrix materialization
    
    Returns
    -------
    dict
        Dictionary of computed metrics
    """
    tlog("Metrics: start")
    base_long = precomputed["base_long"]

    mask_arr = candidate_mask.to_numpy(dtype=bool, copy=False)
    candidate_row_idx, candidate_col_idx = np.nonzero(mask_arr)
    flat_idx = np.flatnonzero(mask_arr.ravel(order="C"))
    candidate_index = base_long.index[flat_idx]

    if side_sign is None:
        sign_arr = np.ones_like(mask_arr, dtype=np.int8)
    else:
        sign_arr = side_sign.to_numpy(dtype=np.int8, copy=False)
    side_sign_long = pd.Series(sign_arr.ravel(order="C"), index=base_long.index)

    if len(candidate_index) == 0:
        logger.warning("No candidates selected for this configuration")
        tlog("Metrics: no candidates")
        return {
            "n_candidates_mean": 0,
            "ic": 0,
            "ic_std": 0,
            "ks_stat": 0,
            "snr": 0,
            "class_balance": 0,
            "mean_feat_ic": 0,
            "sharpe": 0,
            "candidate_rate": 0,
            "mean_return_bps": 0,
            "volatility_bps": 0,
            "sortino": 0,
            "hit_rate": 0,
            "tail_ratio": 0,
            "ic_spearman": 0,
            "oof_mae": 0,
            "oof_directional_acc": 0,
            "ridge_alpha": RIDGE_SCREEN_ALPHA,
            "ridge_top_frac": RIDGE_SCREEN_TOP_FRAC,
            "ridge_selected_k_mean": 0,
            "ridge_jaccard_median": 0,
            "ridge_replacement_rate_median": 0,
            "mean_abs_ret6h": 0,
            "median_abs_ret6h": 0,
            "ret6h_q01": 0,
            "ret6h_q05": 0,
            "atr_mean": 0,
            "atr_q10": 0,
            "atr_q50": 0,
            "atr_q90": 0,
            "atr_decile_worst": -1,
            "atr_decile_worst_share": 0,
            "atr_decile_pnl_json": "{}",
        }

    candidates = base_long.iloc[flat_idx]
    side_sign_candidates = side_sign_long.loc[candidate_index]

    candidate_returns_raw = candidates["ret_base"].to_numpy(dtype=float_dtype, copy=False)
    candidate_target = candidates["target"].to_numpy(dtype=float_dtype, copy=False)
    candidate_signs = side_sign_candidates.to_numpy(dtype=float_dtype, copy=False)

    valid_mask = np.isfinite(candidate_returns_raw) & np.isfinite(candidate_target)
    candidate_returns_raw = candidate_returns_raw[valid_mask]
    candidate_target = candidate_target[valid_mask]
    candidate_signs = candidate_signs[valid_mask]
    candidate_returns = candidate_returns_raw * candidate_signs
    
    if len(candidate_returns) < 50:
        logger.warning(f"Too few candidates: {len(candidate_returns)}")
        tlog(f"Metrics: too few candidates ({len(candidate_returns)})")
        return {
            "n_candidates_mean": 0,
            "ic": 0,
            "ic_std": 0,
            "ks_stat": 0,
            "snr": 0,
            "class_balance": 0,
            "mean_feat_ic": 0,
            "sharpe": 0,
            "candidate_rate": 0,
            "mean_return_bps": 0,
            "volatility_bps": 0,
            "sortino": 0,
            "hit_rate": 0,
            "tail_ratio": 0,
            "ic_spearman": 0,
            "oof_mae": 0,
            "oof_directional_acc": 0,
            "ridge_alpha": RIDGE_SCREEN_ALPHA,
            "ridge_top_frac": RIDGE_SCREEN_TOP_FRAC,
            "ridge_selected_k_mean": 0,
            "ridge_jaccard_median": 0,
            "ridge_replacement_rate_median": 0,
            "mean_abs_ret6h": 0,
            "median_abs_ret6h": 0,
            "ret6h_q01": 0,
            "ret6h_q05": 0,
            "atr_mean": 0,
            "atr_q10": 0,
            "atr_q50": 0,
            "atr_q90": 0,
            "atr_decile_worst": -1,
            "atr_decile_worst_share": 0,
            "atr_decile_pnl_json": "{}",
        }
    
    tlog(f"Metrics: computing descriptive stats for {len(candidate_returns)} samples")
    # 1. Candidate count (mean per timestamp)
    n_candidates_mean = float(mask_arr.sum(axis=1).mean())
    candidate_rate = float(mask_arr.mean())

    abs_ret = np.abs(candidate_returns_raw)
    mean_abs_ret6h = float(np.mean(abs_ret)) if len(abs_ret) > 0 else 0.0
    median_abs_ret6h = float(np.median(abs_ret)) if len(abs_ret) > 0 else 0.0
    ret6h_q01 = float(np.nanpercentile(candidate_returns_raw, 1)) if len(candidate_returns_raw) > 0 else 0.0
    ret6h_q05 = float(np.nanpercentile(candidate_returns_raw, 5)) if len(candidate_returns_raw) > 0 else 0.0

    candidate_labels = candidates["label"].to_numpy(dtype=np.float32, copy=False)
    candidate_labels = candidate_labels[valid_mask]
    label_finite = np.isfinite(candidate_labels)
    pos_values = candidate_returns_raw[label_finite & (candidate_labels == 1.0)]
    neg_values = candidate_returns_raw[label_finite & (candidate_labels == 0.0)]
    
    # 3. KS Statistic
    ks_stat = compute_ks_statistic(pos_values, neg_values)
    
    # 4. Signal-to-Noise Ratio
    snr = compute_snr(pos_values, neg_values)
    
    # 5. Class Balance (positive label rate among candidates)
    class_balance = float(np.mean(candidate_labels[label_finite])) if np.any(label_finite) else 0.0

    tlog("Metrics: computing feature-target correlation")
    # 6. Feature-Target Correlation
    mean_feat_ic = compute_feature_target_correlation(
        precomputed=precomputed,
        target_candidates=candidates["target"],
        available_features=available_features,
        candidate_row_idx=candidate_row_idx,
        candidate_col_idx=candidate_col_idx,
        candidate_index_ref=candidate_mask.index,
        candidate_columns_ref=candidate_mask.columns,
        top_n=20,
    )
    
    # 7. Sharpe Ratio (annualized)
    if len(candidate_returns) > 1 and np.std(candidate_returns) > 0:
        # Assuming hourly data: 24 * 365 = 8760 hours per year
        sharpe = np.mean(candidate_returns) / np.std(candidate_returns) * np.sqrt(8760)
    else:
        sharpe = 0

    mean_return_bps = float(np.mean(candidate_returns) * 1e4) if len(candidate_returns) > 0 else 0.0
    volatility_bps = float(np.std(candidate_returns) * 1e4) if len(candidate_returns) > 0 else 0.0
    downside = candidate_returns[candidate_returns < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = float(np.mean(candidate_returns) / np.std(downside) * np.sqrt(8760))
    else:
        sortino = 0.0
    hit_rate = float(np.mean(candidate_returns > 0)) if len(candidate_returns) > 0 else 0.0
    p95 = np.nanpercentile(candidate_returns, 95) if len(candidate_returns) > 0 else 0.0
    p05 = np.nanpercentile(candidate_returns, 5) if len(candidate_returns) > 0 else 0.0
    tail_ratio = float(p95 / abs(p05)) if abs(p05) > 1e-12 else 0.0

    atr_selected = candidates["atr_pct"].to_numpy(dtype=float_dtype, copy=False)
    atr_selected = atr_selected[valid_mask]
    atr_selected = atr_selected[np.isfinite(atr_selected)]
    atr_mean = float(np.mean(atr_selected)) if len(atr_selected) > 0 else 0.0
    atr_q10 = float(np.nanpercentile(atr_selected, 10)) if len(atr_selected) > 0 else 0.0
    atr_q50 = float(np.nanpercentile(atr_selected, 50)) if len(atr_selected) > 0 else 0.0
    atr_q90 = float(np.nanpercentile(atr_selected, 90)) if len(atr_selected) > 0 else 0.0

    atr_decile_worst = -1
    atr_decile_worst_share = 0.0
    atr_decile_pnl_json = "{}"
    atr_raw = candidates["atr_pct"].to_numpy(dtype=float_dtype, copy=False)
    atr_raw = atr_raw[valid_mask]
    valid_atr_for_deciles = np.isfinite(atr_raw) & np.isfinite(candidate_returns)
    if valid_atr_for_deciles.sum() >= 100:
        atr_vals = atr_raw[valid_atr_for_deciles]
        pnl_vals = candidate_returns[valid_atr_for_deciles]
        try:
            decile = pd.qcut(atr_vals, q=10, labels=False, duplicates="drop")
            dec_df = pd.DataFrame({"decile": decile, "pnl": pnl_vals})
            dec_pnl = dec_df.groupby("decile")["pnl"].sum().sort_index()
            atr_decile_pnl_json = json.dumps({int(k): float(v) for k, v in dec_pnl.items()})
            worst = dec_pnl.idxmin()
            atr_decile_worst = int(worst)
            total_neg = abs(dec_pnl[dec_pnl < 0].sum())
            if total_neg > 1e-12 and dec_pnl[worst] < 0:
                atr_decile_worst_share = float(abs(dec_pnl[worst]) / total_neg)
        except Exception:
            pass
    
    tlog("Metrics: entering OOF/IC block")
    # 8. Information Coefficient (requires OOF predictions)
    if len(available_features) >= 10:
        X_all, used_features = materialize_feature_matrix(
            precomputed=precomputed,
            available_features=available_features,
            candidate_row_idx=candidate_row_idx,
            candidate_col_idx=candidate_col_idx,
            candidate_index_ref=candidate_mask.index,
            candidate_columns_ref=candidate_mask.columns,
            float_dtype=float_dtype,
        )
        y_arr = candidates["target"].to_numpy(dtype=float_dtype, copy=False)
        if X_all.shape[1] >= 10 and len(used_features) >= 10:
            valid_rows = np.isfinite(y_arr) & np.isfinite(X_all).all(axis=1)
        else:
            valid_rows = np.zeros(len(y_arr), dtype=bool)

        if valid_rows.sum() >= 100:
            X = X_all[valid_rows]
            y = y_arr[valid_rows]
            ts_vals = candidate_index.get_level_values(0)[valid_rows]

            if len(y) >= 100:
                if len(y) > OOF_MAX_SAMPLES:
                    n_before_oof = len(y)
                    sub_idx = _uniform_subsample_idx(len(y), OOF_MAX_SAMPLES, seed=42)
                    X = X[sub_idx]
                    y = y[sub_idx]
                    ts_vals = ts_vals[sub_idx]
                    tlog(f"Metrics: OOF downsample applied {len(y)}/{n_before_oof} rows")
                tlog(f"Metrics: running OOF CV on {X.shape[0]}x{X.shape[1]}")
                oof, oof_diag = run_oof_cv(
                    X,
                    y,
                    float_dtype=float_dtype,
                    ridge_alpha=RIDGE_SCREEN_ALPHA,
                    ridge_top_frac=RIDGE_SCREEN_TOP_FRAC,
                    use_extratrees=use_extratrees,
                )

                oof_valid = np.isfinite(oof)
                if oof_valid.sum() >= 50:
                    ic = safe_pearson_corr(oof[oof_valid], y[oof_valid])
                    ic_spearman = stats.spearmanr(oof[oof_valid], y[oof_valid], nan_policy="omit").statistic
                    oof_mae = float(np.mean(np.abs(oof[oof_valid] - y[oof_valid])))
                    oof_directional_acc = float(np.mean(np.sign(oof[oof_valid]) == np.sign(y[oof_valid])))

                    oof_df = pd.DataFrame({"oof": oof, "y": y, "ts": ts_vals})
                    oof_df = oof_df[oof_df["oof"].notna()]
                    timestamp_ics = []
                    for ts, group in oof_df.groupby("ts"):
                        if len(group) >= 5:
                            ts_ic = safe_pearson_corr(
                                group["oof"].to_numpy(dtype=float_dtype, copy=False),
                                group["y"].to_numpy(dtype=float_dtype, copy=False),
                            )
                            if np.isfinite(ts_ic):
                                timestamp_ics.append(ts_ic)
                    ic_std = np.std(timestamp_ics) if timestamp_ics else 0
                    del oof_df
                    gc.collect()
                else:
                    ic, ic_std, ic_spearman, oof_mae, oof_directional_acc = 0, 0, 0, 0, 0
                ridge_alpha_used = float(oof_diag.get("ridge_alpha", RIDGE_SCREEN_ALPHA))
                ridge_top_frac_used = float(oof_diag.get("ridge_top_frac", RIDGE_SCREEN_TOP_FRAC))
                ridge_selected_k_mean = float(oof_diag.get("ridge_selected_k_mean", 0.0))
                ridge_jaccard_median = float(oof_diag.get("ridge_jaccard_median", 0.0))
                ridge_replacement_rate_median = float(oof_diag.get("ridge_replacement_rate_median", 0.0))
            else:
                ic, ic_std, ic_spearman, oof_mae, oof_directional_acc = 0, 0, 0, 0, 0
                ridge_alpha_used = RIDGE_SCREEN_ALPHA
                ridge_top_frac_used = RIDGE_SCREEN_TOP_FRAC
                ridge_selected_k_mean = 0.0
                ridge_jaccard_median = 0.0
                ridge_replacement_rate_median = 0.0
        else:
            ic, ic_std, ic_spearman, oof_mae, oof_directional_acc = 0, 0, 0, 0, 0
            ridge_alpha_used = RIDGE_SCREEN_ALPHA
            ridge_top_frac_used = RIDGE_SCREEN_TOP_FRAC
            ridge_selected_k_mean = 0.0
            ridge_jaccard_median = 0.0
            ridge_replacement_rate_median = 0.0
    else:
        ic, ic_std, ic_spearman, oof_mae, oof_directional_acc = 0, 0, 0, 0, 0
        ridge_alpha_used = RIDGE_SCREEN_ALPHA
        ridge_top_frac_used = RIDGE_SCREEN_TOP_FRAC
        ridge_selected_k_mean = 0.0
        ridge_jaccard_median = 0.0
        ridge_replacement_rate_median = 0.0
    
    tlog("Metrics: done")
    return {
        "n_candidates_mean": n_candidates_mean,
        "ic": ic if np.isfinite(ic) else 0,
        "ic_std": ic_std if np.isfinite(ic_std) else 0,
        "ks_stat": ks_stat,
        "snr": snr,
        "class_balance": class_balance,
        "mean_feat_ic": mean_feat_ic,
        "sharpe": sharpe if np.isfinite(sharpe) else 0,
        "candidate_rate": candidate_rate,
        "mean_return_bps": mean_return_bps,
        "volatility_bps": volatility_bps,
        "sortino": sortino if np.isfinite(sortino) else 0,
        "hit_rate": hit_rate,
        "tail_ratio": tail_ratio if np.isfinite(tail_ratio) else 0,
        "ic_spearman": ic_spearman if np.isfinite(ic_spearman) else 0,
        "oof_mae": oof_mae,
        "oof_directional_acc": oof_directional_acc,
        "ridge_alpha": ridge_alpha_used,
        "ridge_top_frac": ridge_top_frac_used,
        "ridge_selected_k_mean": ridge_selected_k_mean,
        "ridge_jaccard_median": ridge_jaccard_median,
        "ridge_replacement_rate_median": ridge_replacement_rate_median,
        "mean_abs_ret6h": mean_abs_ret6h,
        "median_abs_ret6h": median_abs_ret6h,
        "ret6h_q01": ret6h_q01,
        "ret6h_q05": ret6h_q05,
        "atr_mean": atr_mean,
        "atr_q10": atr_q10,
        "atr_q50": atr_q50,
        "atr_q90": atr_q90,
        "atr_decile_worst": atr_decile_worst,
        "atr_decile_worst_share": atr_decile_worst_share,
        "atr_decile_pnl_json": atr_decile_pnl_json,
    }


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_features_from_parquet(feature_path: str) -> dict:
    """
    Load feature data from parquet files.
    
    Expects either:
    - A directory with individual parquet files per feature
    - A single parquet file with multi-index (timestamp, symbol)
    """
    feats = {}
    
    if os.path.isfile(feature_path):
        # Single parquet file
        logger.info(f"Loading features from single file: {feature_path}")
        df = pd.read_parquet(feature_path)
        
        # Check if multi-index
        if isinstance(df.index, pd.MultiIndex):
            # Unstack to get feature x (timestamp, symbol) format
            if "feature" in df.columns:
                for feat_name in df["feature"].unique():
                    feat_df = df[df["feature"] == feat_name].drop(columns=["feature"])
                    feat_df = feat_df.unstack()
                    feat_df.columns = feat_df.columns.droplevel(0)
                    feats[feat_name] = feat_df
            else:
                # Assume columns are features
                for col in df.columns:
                    feat_df = df[[col]].unstack()
                    feat_df.columns = feat_df.columns.droplevel(0)
                    feats[col] = feat_df
        else:
            # Assume columns are symbols, need to check format
            logger.warning("Unexpected parquet format, attempting to load as-is")
            for col in df.columns:
                feats[col] = df[[col]]
    
    elif os.path.isdir(feature_path):
        # Directory with parquet files
        logger.info(f"Loading features from directory: {feature_path}")
        
        for fname in os.listdir(feature_path):
            if fname.endswith(".parquet"):
                fpath = os.path.join(feature_path, fname)
                feat_name = fname.replace(".parquet", "")
                try:
                    df = pd.read_parquet(fpath)
                    
                    # Check format
                    if isinstance(df.index, pd.MultiIndex):
                        df = df.unstack()
                        if df.columns.nlevels > 1:
                            df.columns = df.columns.droplevel(0)
                    
                    feats[feat_name] = df
                    logger.debug(f"Loaded {feat_name}: {df.shape}")
                except Exception as e:
                    logger.warning(f"Failed to load {fname}: {e}")
        
        logger.info(f"Loaded {len(feats)} features")
    
    else:
        raise FileNotFoundError(f"Feature path not found: {feature_path}")
    
    return feats


def load_panel_data(panel_path: str) -> pd.DataFrame:
    """
    Load panel data (OHLCV) for computing returns if needed.
    """
    if os.path.isfile(panel_path):
        return pd.read_parquet(panel_path)
    elif os.path.isdir(panel_path):
        # Load all parquet files in directory
        dfs = []
        for root, dirs, files in os.walk(panel_path):
            for f in files:
                if f.endswith(".parquet"):
                    fpath = os.path.join(root, f)
                    df = pd.read_parquet(fpath)

                    # Derive symbol from partition path when missing (e.g., .../symbol=BTC_USDT/...)
                    if "symbol" not in df.columns:
                        sym = None
                        parts = root.split(os.sep)
                        for p in parts:
                            if p.startswith("symbol="):
                                sym = p.replace("symbol=", "")
                                break
                        if sym is not None:
                            df["symbol"] = sym

                    dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
    
    raise FileNotFoundError(f"Panel path not found: {panel_path}")


# =============================================================================
# Main Comparison Runner
# =============================================================================

def run_comparison(
    feature_path: str,
    panel_path: str,
    output_path: str,
    dtype: str = "float32",
    max_features: Optional[int] = None,
    stage3: bool = False,
    winners: list = None,
    use_extratrees: bool = False,  # Default: Ridge only for Stage 1 & 2
):
    """
    Main comparison runner.
    
    Parameters
    ----------
    feature_path : str
        Path to feature data (directory or single parquet file)
        Can be either:
        - A timestamp directory like 'data/features/20260214_190000' (per-symbol format)
        - A directory with per-feature parquet files
    panel_path : str
        Path to panel data (klines/OHLCV)
    output_path : str
        Path for output CSV
    """
    logger.info("=" * 60)
    logger.info("Candidate Selection Threshold Comparison")
    logger.info("=" * 60)
    tlog("Starting comparison run")
    
    # Load data - try pipeline format first, then fallback to generic loader
    logger.info(f"Loading features from: {feature_path}")
    
    # Check if this is a pipeline-style timestamp directory (has symbol=*.parquet files)
    import glob
    symbol_files = glob.glob(os.path.join(feature_path, "symbol=*.parquet"))
    
    float_dtype = np.float32 if dtype == "float32" else np.float64
    tlog(f"Configured dtype={dtype}")
    tlog(
        "OOF model stack defaults: "
        f"RobustScaler->Ridge(alpha={RIDGE_SCREEN_ALPHA}) top_frac={RIDGE_SCREEN_TOP_FRAC:.0%}->ExtraTrees"
    )

    if symbol_files:
        # Pipeline format: per-symbol files, need to parse timestamp from path
        logger.info("Detected pipeline per-symbol format")
        import re
        # Extract timestamp from path like 'data/features/20260214_190000'
        match = re.search(r'(\d{8}_\d{6})', feature_path)
        if match:
            ts_str = match.group(1)
            ts = pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S")
            # Determine root_dir (parent of 'features' directory)
            features_dir = os.path.dirname(feature_path)
            root_dir = os.path.dirname(features_dir) if features_dir.endswith('features') else os.path.dirname(feature_path)
            feats = load_features_pipeline(ts, root_dir)
            if feats is None:
                raise ValueError(f"Failed to load features from {feature_path}")
            logger.info(f"Loaded {len(feats)} features via pipeline loader")
            logger.info(f"Casting features to {dtype}...")
            tlog("Casting feature DataFrames")
            feats = cast_features_dtype(feats, float_dtype=float_dtype)
            gc.collect()
        else:
            raise ValueError(f"Could not parse timestamp from path: {feature_path}")
    else:
        # Generic format: per-feature files
        tlog("Loading generic feature parquet layout")
        feats = load_features_from_parquet(feature_path)
        feats = cast_features_dtype(feats, float_dtype=float_dtype)
        gc.collect()
    
    # Check required features
    if "ret6h" not in feats and "ret24h" not in feats:
        logger.error("Required feature 'ret6h' (or fallback 'ret24h') not found in data")
        return
    
    # Log available features
    available_model_features = [f for f in MODEL_FEATURES if f in feats]
    if max_features is not None and max_features > 0:
        available_model_features = available_model_features[:max_features]
    logger.info(f"Available MODEL_FEATURES: {len(available_model_features)}/{len(MODEL_FEATURES)}")
    tlog(f"Using {len(available_model_features)} model features")

    if panel_path is None:
        raise ValueError("--panel is required for training-aligned MR/TF long/short slice evaluation")
    tlog("Loading panel data")
    panel_raw = load_panel_data(panel_path)
    panel = to_panel_dict(panel_raw)
    tlog(f"Loaded panel with close shape={panel['close'].shape}")

    tlog("Precomputing selection metrics")
    metric_pack = precompute_selection_metrics(feats, float_dtype=float_dtype)
    metric_by_mode = metric_pack["metrics"]
    tlog(f"Precomputed metric modes: {list(metric_by_mode.keys())}")

    tlog("Building long-form precomputed tables")
    precomputed = build_long_form_tables(
        feats=feats,
        target_col="ret6h",
        float_dtype=float_dtype,
        atr_reference=metric_pack.get("atr_effective"),
    )
    tlog("Built long-form base table (features materialized lazily)")

    # Keep feats for filter application, but we'll pass it to selection functions
    # Define test configurations with filter variants
    # Filter parameter ranges:
    # - min_range_pct: [0.06, 0.07, 0.08]
    # - min_vol_zscore: [1.4, 1.6, 1.8]
    # - min_sign_consistency: [0.60, 0.70, 0.80]
    
    configs = []
    
    # Default values for filters
    default_pct = 0.06
    default_range_pct = 0.07
    default_vol_zscore = 1.6
    default_sign_consistency = 0.70
    default_tp_lo = 0.02
    default_tp_hi = 0.06
    default_sl_mult = 0.50
    pct_grid = [0.06]  # Single pct for initial runs
    
    # Expansion variants
    expansion_variants = [
        ("none", []),
        ("full", [-12, -8, -4, 4, 8, 12, 16]),
        ("neg48", [-4, -8]),
        ("pos48", [4, 8]),
        ("sym48", [-4, -8, 4, 8]),
    ]
    
    # Modes to test
    modes = [("F", "fixed"), ("A", "atr"), ("VW", "vol_weight")]
    
    # Use 33% of samples for faster execution
    SAMPLE_FRAC = 0.33
    
    # =============================================================================
    # STAGE 1: Filter sweep (27 configs)
    # 3 modes × 3 filter sweeps × 3 values each
    # No expansions in this stage - just filter sweeps to find best values
    # =============================================================================
    tlog("Stage 1 setup: building filter-sweep configs")
    # One-at-a-time filter sweeps for each mode (pipeline-aligned defaults on other filters)
    for mode_prefix, mode_name in modes:
        for pct in pct_grid:
            for range_pct in [0.06, 0.07, 0.08]:
                configs.append(
                    {
                        "config_id": f"{mode_prefix}_P{int(pct * 100):02d}_R{int(range_pct * 100):02d}",
                        "mode": mode_name,
                        "pct": pct,
                        "min_range_pct": range_pct,
                        "min_vol_zscore": default_vol_zscore,
                        "min_sign_consistency": default_sign_consistency,
                    }
                )
            for vol_z in [1.4, 1.6, 1.8]:
                configs.append(
                    {
                        "config_id": f"{mode_prefix}_P{int(pct * 100):02d}_V{int(vol_z * 10):02d}",
                        "mode": mode_name,
                        "pct": pct,
                        "min_range_pct": default_range_pct,
                        "min_vol_zscore": vol_z,
                        "min_sign_consistency": default_sign_consistency,
                    }
                )
            for sc in [0.60, 0.70, 0.80]:
                configs.append(
                    {
                        "config_id": f"{mode_prefix}_P{int(pct * 100):02d}_S{int(sc * 100):02d}",
                        "mode": mode_name,
                        "pct": pct,
                        "min_range_pct": default_range_pct,
                        "min_vol_zscore": default_vol_zscore,
                        "min_sign_consistency": sc,
                    }
                )

    tlog(f"Stage 1 setup done: {len(configs)} configs")

    # =============================================================================
    # STAGE 2: Expansion variants (15 configs)
    # 3 modes × 5 expansion variants
    # Uses best filter values from Stage 1
    # =============================================================================
    tlog("Stage 2 setup: adding FULL configs + expansion variants")
    # Full filter combination (training pipeline defaults)
    for mode_prefix, mode_name in modes:
        for pct in pct_grid:
            configs.append(
                {
                    "config_id": f"{mode_prefix}_P{int(pct * 100):02d}_FULL",
                    "mode": mode_name,
                    "pct": pct,
                    "min_range_pct": default_range_pct,
                    "min_vol_zscore": default_vol_zscore,
                    "min_sign_consistency": default_sign_consistency,
                    "barrier_sl_base_mult": default_sl_mult,
                    "barrier_k_tp": 1.0,
                }
            )

    # Add expansion variants to FULL configs only (15 configs total)
    expanded_configs = []
    for cfg in configs:
        if "_FULL" in cfg["config_id"]:
            for exp_name, exp_offsets in expansion_variants:
                cfg_e = dict(cfg)
                cfg_e["expansion_name"] = exp_name
                cfg_e["expansion_offsets"] = list(exp_offsets)
                cfg_e["config_id"] = f"{cfg['config_id']}_E{exp_name.upper()}"
                expanded_configs.append(cfg_e)
    
    # Keep non-FULL configs (filter sweeps) and add expanded FULL configs
    non_full_configs = [cfg for cfg in configs if "_FULL" not in cfg["config_id"]]
    configs = non_full_configs + expanded_configs
    tlog(f"Stage 2 setup done: {len(configs)} configs")

    # =============================================================================
    # STAGE 3: PCT variations for winners (12 configs)
    # Added when --stage3 flag is used with --winners
    # 4 winners × 3 pct values = 12 configs
    # Stage 3 always uses ExtraTrees for final selection
    # =============================================================================
    if stage3 and winners:
        tlog(f"Stage 3 setup: building winner pct-variation configs for winners={winners}")
        # Force ExtraTrees for Stage 3
        use_extratrees = True
        
        stage3_pcts = [0.05, 0.06, 0.07]
        stage3_configs = []
        for cfg in configs:
            # Check if this config matches any of the winners
            # Winners should match the base config_id (without expansion suffix)
            base_id = cfg["config_id"].split("_E")[0] if "_E" in cfg["config_id"] else cfg["config_id"]
            if base_id in winners:
                for new_pct in stage3_pcts:
                    if new_pct != cfg.get("pct", 0.06):
                        new_cfg = dict(cfg)
                        # Replace pct in config_id
                        old_pct = int(cfg.get("pct", 0.06) * 100)
                        new_cfg["config_id"] = cfg["config_id"].replace(f"P{old_pct:02d}", f"P{int(new_pct*100):02d}")
                        new_cfg["config_id"] = new_cfg["config_id"] + "_S3"  # Mark as stage 3
                        new_cfg["pct"] = new_pct
                        stage3_configs.append(new_cfg)
        if stage3_configs:
            tlog(f"Stage 3 setup done: {len(stage3_configs)} configs")
            configs = stage3_configs  # Replace with stage 3 configs only
        else:
            tlog(f"Stage 3 setup done: no configs matched winners={winners}")
    else:
        tlog("Stage 3 setup skipped")
    
    range_thresholds = [cfg["min_range_pct"] for cfg in configs if cfg.get("min_range_pct") is not None]
    vol_thresholds = [cfg["min_vol_zscore"] for cfg in configs if cfg.get("min_vol_zscore") is not None]
    sc_thresholds = [cfg["min_sign_consistency"] for cfg in configs if cfg.get("min_sign_consistency") is not None]
    metric_ref = metric_by_mode.get("fixed")
    if metric_ref is None:
        metric_ref = next(iter(metric_by_mode.values()))
    tlog("Precomputing filter masks")
    filter_mask_pack = precompute_filter_masks(
        feats=feats,
        panel=panel,
        target_index=metric_ref.index,
        target_columns=metric_ref.columns,
        range_thresholds=range_thresholds,
        vol_thresholds=vol_thresholds,
        sc_thresholds=sc_thresholds,
        float_dtype=float_dtype,
    )

    base_mask_cache: dict[tuple[str, float], np.ndarray] = {}
    training_slice_cache: dict[tuple[Any, ...], list] = {}
    disable_training_slices = False

    tlog(f"Prepared {len(configs)} configs for execution")
    results = []
    slice_results = []
    
    for cfg in configs:
        config_id = cfg["config_id"]
        mode = cfg["mode"]
        pct = cfg["pct"]
        candidate_mask = None
        side_sign = None
        
        logger.info("-" * 40)
        stage_label = infer_stage_label(config_id)
        logger.info(f"Running config [{stage_label}]: {config_id} (mode={mode}, pct={pct})")
        tlog(f"Config start: {config_id}")
        
        try:
            if mode not in {"fixed", "atr", "vol_weight"}:
                raise ValueError(f"Unsupported mode in default sweep: '{mode}'")
            if mode not in metric_by_mode:
                raise ValueError(f"Required features missing for mode '{mode}'")

            # Extract filter parameters from config
            min_range_pct = cfg.get("min_range_pct")
            min_vol_zscore = cfg.get("min_vol_zscore")
            min_sign_consistency = cfg.get("min_sign_consistency")
            expansion_name = cfg.get("expansion_name", "none")
            expansion_offsets = cfg.get("expansion_offsets", [])

            tlog(f"Selecting candidates: mode={mode}, pct={pct}")
            candidate_mask_base = select_candidates_cross_sectional(
                metric_by_mode[mode],
                pct,
                filter_masks=filter_mask_pack,
                min_range_pct=min_range_pct,
                min_vol_zscore=min_vol_zscore,
                min_sign_consistency=min_sign_consistency,
                base_mask_cache=base_mask_cache,
                base_cache_key=(mode, float(pct)),
            )
            # Troubleshooting: show where candidates are filtered out.
            raw_base_arr = base_mask_cache.get((mode, float(pct)))
            if raw_base_arr is not None:
                n_raw = int(raw_base_arr.sum())
                dbg_arr = np.array(raw_base_arr, copy=True, dtype=bool)
                n_after_range = n_raw
                n_after_vol = n_raw
                n_after_sc = n_raw
                if min_range_pct is not None:
                    range_mask = filter_mask_pack["range_masks"].get(float(min_range_pct))
                    if range_mask is not None:
                        dbg_arr &= range_mask
                        n_after_range = int(dbg_arr.sum())
                if min_vol_zscore is not None:
                    vol_mask = filter_mask_pack["vol_masks"].get(float(min_vol_zscore))
                    if vol_mask is not None:
                        dbg_arr &= vol_mask
                        n_after_vol = int(dbg_arr.sum())
                if min_sign_consistency is not None:
                    sc_mask = filter_mask_pack["sc_masks"].get(float(min_sign_consistency))
                    if sc_mask is not None:
                        dbg_arr &= sc_mask
                        n_after_sc = int(dbg_arr.sum())
                tlog(
                    "Candidate filter breakdown: "
                    f"raw={n_raw}, after_range={n_after_range}, "
                    f"after_vol={n_after_vol}, after_sign={n_after_sc}"
                )
            base_selected_n = int(candidate_mask_base.to_numpy(dtype=bool, copy=False).sum())
            tlog(f"Candidate base mask: selected={base_selected_n}")
            candidate_mask = expand_candidate_mask(candidate_mask_base, expansion_offsets)
            expanded_selected_n = int(candidate_mask.to_numpy(dtype=bool, copy=False).sum())
            if expansion_offsets:
                tlog(
                    f"Applied candidate expansion ({expansion_name}): offsets={expansion_offsets}, "
                    f"selected={expanded_selected_n}"
                )
            else:
                tlog(f"Candidate expansion skipped: selected={expanded_selected_n}")

            candidate_mask = align_candidate_mask_to_panel_symbols(candidate_mask, panel)
            aligned_selected_n = int(candidate_mask.to_numpy(dtype=bool, copy=False).sum())
            panel_overlap_cols = int(
                len(candidate_mask.columns.intersection(panel["close"].columns))
            ) if "close" in panel else 0
            tlog(
                f"Candidate alignment: selected={aligned_selected_n}, "
                f"cols={candidate_mask.shape[1]}, panel_overlap_cols={panel_overlap_cols}"
            )
            if base_selected_n > 0 and aligned_selected_n == 0:
                tlog(
                    "Troubleshoot: candidates collapsed to zero after expansion/alignment. "
                    "Check filter thresholds and symbol naming consistency."
                )
            del candidate_mask_base
            
            # Compute metrics
            tlog(f"Computing learnability metrics: {config_id}")
            metrics = compute_learnability_metrics(
                candidate_mask=candidate_mask,
                precomputed=precomputed,
                available_features=available_model_features,
                float_dtype=float_dtype,
                side_sign=side_sign,
                use_extratrees=use_extratrees,
            )

            cfg_variant = deepcopy(CFG)
            cfg_variant["train_extreme_pct_hourly"] = pct
            
            # Unified barrier factory params (v3 - single source of truth)
            # These replace the old train_tp_lo, train_tp_hi, train_sl_mult params
            cfg_variant["barrier_k_tp"] = float(cfg.get("barrier_k_tp", 1.0))
            cfg_variant["barrier_sl_base_mult"] = float(cfg.get("barrier_sl_base_mult", 0.5))
            cfg_variant["barrier_disp_floor"] = float(cfg.get("barrier_disp_floor", 0.1))
            cfg_variant["barrier_z_max"] = float(cfg.get("barrier_z_max", 3.0))
            cfg_variant["barrier_k_reg"] = float(cfg.get("barrier_k_reg", 0.3))
            cfg_variant["barrier_m_lo"] = float(cfg.get("barrier_m_lo", 0.7))
            cfg_variant["barrier_m_hi"] = float(cfg.get("barrier_m_hi", 1.5))
            cfg_variant["barrier_sl_lo"] = float(cfg.get("barrier_sl_lo", 0.4))
            cfg_variant["barrier_sl_hi"] = float(cfg.get("barrier_sl_hi", 0.7))
            cfg_variant["barrier_z_gate"] = float(cfg.get("barrier_z_gate", 1.0))
            cfg_variant["barrier_tp_lo"] = float(cfg.get("barrier_tp_lo", 0.02))
            cfg_variant["barrier_tp_hi"] = float(cfg.get("barrier_tp_hi", 0.06))
            cfg_variant["label_horizon_base"] = float(cfg.get("label_horizon_base", 4))
            
            if cfg.get("min_range_pct") is not None:
                cfg_variant["train_min_range_pct"] = float(cfg["min_range_pct"])
            if cfg.get("min_vol_zscore") is not None:
                cfg_variant["train_min_vol_zscore"] = float(cfg["min_vol_zscore"])
            if cfg.get("min_sign_consistency") is not None:
                cfg_variant["min_feat_sign_consistency"] = float(cfg["min_sign_consistency"])

            training_cache_key = (
                fingerprint_candidate_mask(candidate_mask),
                float(cfg_variant.get("train_extreme_pct_hourly", 0.0)),
                cfg_variant.get("train_min_range_pct"),
                cfg_variant.get("train_min_vol_zscore"),
                cfg_variant.get("min_feat_sign_consistency"),
                cfg_variant.get("barrier_k_tp"),
                cfg_variant.get("barrier_sl_base_mult"),
                cfg_variant.get("barrier_disp_floor"),
                tuple(expansion_offsets),
                tuple(cfg_variant.get("label_horizons_hours", [2, 4, 8])),
            )

            if disable_training_slices:
                tlog("Training-slice stage skipped: disabled after prior structural precheck failure")
                training_slice_rows = []
            else:
                ok_slices, why_not = training_slice_precheck(candidate_mask, panel)
                if not ok_slices:
                    tlog(f"Training-slice stage skipped: {why_not}")
                    if why_not in {
                        "panel.close missing",
                        "no symbol overlap between candidate mask and panel close",
                    }:
                        disable_training_slices = True
                    training_slice_rows = []
                else:
                    tlog(f"Evaluating training slices: {config_id}")
                    training_slice_rows = evaluate_training_slices(
                        candidate_mask=candidate_mask,
                        feats=feats,
                        panel=panel,
                        cfg_variant=cfg_variant,
                        horizons=cfg_variant.get("label_horizons_hours", [2, 4, 8]),
                        cache=training_slice_cache,
                        cache_key=training_cache_key,
                        sample_frac=SAMPLE_FRAC,
                    )
            for r in training_slice_rows:
                slice_results.append(
                    {
                        "config_id": config_id,
                        "mode": mode,
                        "pct": pct,
                        "min_range_pct": cfg.get("min_range_pct", None),
                        "min_vol_zscore": cfg.get("min_vol_zscore", None),
                        "min_sign_consistency": cfg.get("min_sign_consistency", None),
                        "barrier_k_tp": cfg.get("barrier_k_tp", 1.0),
                        "barrier_sl_base_mult": cfg.get("barrier_sl_base_mult", 0.5),
                        "barrier_disp_floor": cfg.get("barrier_disp_floor", 0.1),
                        "expansion_name": expansion_name,
                        "expansion_offsets": ",".join(str(o) for o in expansion_offsets) if expansion_offsets else "",
                        **r,
                    }
                )
            metrics.update(aggregate_slice_rows(training_slice_rows))
            
            result = {
                "config_id": config_id,
                "mode": mode,
                "pct": pct,
                "min_range_pct": cfg.get("min_range_pct", None),
                "min_vol_zscore": cfg.get("min_vol_zscore", None),
                "min_sign_consistency": cfg.get("min_sign_consistency", None),
                "barrier_k_tp": cfg.get("barrier_k_tp", 1.0),
                "barrier_sl_base_mult": cfg.get("barrier_sl_base_mult", 0.5),
                "barrier_disp_floor": cfg.get("barrier_disp_floor", 0.1),
                "expansion_name": expansion_name,
                "expansion_offsets": ",".join(str(o) for o in expansion_offsets) if expansion_offsets else "",
                **metrics
            }
            
            results.append(result)
            
            logger.info(f"  Candidates/timestamp: {metrics['n_candidates_mean']:.1f}")
            logger.info(f"  IC: {metrics['ic']:.4f} ± {metrics['ic_std']:.4f}")
            logger.info(f"  KS: {metrics['ks_stat']:.4f}, SNR: {metrics['snr']:.4f}")
            logger.info(f"  Class balance: {metrics['class_balance']:.2%}")
            logger.info(f"  Sharpe: {metrics['sharpe']:.2f}")
            logger.info(
                f"  HitRate: {metrics['hit_rate']:.2%} | Sortino: {metrics['sortino']:.2f} | "
                f"Return(bps): {metrics['mean_return_bps']:.2f} ± {metrics['volatility_bps']:.2f}"
            )
            logger.info(
                f"  SliceSharpe: {metrics['slice_overall_sharpe']:.2f} | "
                f"SliceSortino: {metrics['slice_overall_sortino']:.2f} | "
                f"SliceN: {metrics['slice_total_samples']}"
            )
            tlog(f"Config done: {config_id}")
            
            # Free memory
            del candidate_mask
            del metrics
            gc.collect()
            
        except Exception as e:
            logger.exception(f"Failed to run config {config_id}: {e}")
            results.append({
                "config_id": config_id,
                "mode": mode,
                "pct": pct,
                "min_range_pct": cfg.get("min_range_pct", None),
                "min_vol_zscore": cfg.get("min_vol_zscore", None),
                "min_sign_consistency": cfg.get("min_sign_consistency", None),
                "barrier_k_tp": cfg.get("barrier_k_tp", 1.0),
                "barrier_sl_base_mult": cfg.get("barrier_sl_base_mult", 0.5),
                "barrier_disp_floor": cfg.get("barrier_disp_floor", 0.1),
                "expansion_name": cfg.get("expansion_name", "none"),
                "expansion_offsets": ",".join(str(o) for o in cfg.get("expansion_offsets", [])) if cfg.get("expansion_offsets") else "",
                "n_candidates_mean": 0,
                "ic": 0,
                "ic_std": 0,
                "ks_stat": 0,
                "snr": 0,
                "class_balance": 0,
                "mean_feat_ic": 0,
                "sharpe": 0,
                "candidate_rate": 0,
                "mean_return_bps": 0,
                "volatility_bps": 0,
                "sortino": 0,
                "hit_rate": 0,
                "tail_ratio": 0,
                "ic_spearman": 0,
                "oof_mae": 0,
                "oof_directional_acc": 0,
                "ridge_alpha": RIDGE_SCREEN_ALPHA,
                "ridge_top_frac": RIDGE_SCREEN_TOP_FRAC,
                "ridge_selected_k_mean": 0,
                "ridge_jaccard_median": 0,
                "ridge_replacement_rate_median": 0,
                "mean_abs_ret6h": 0,
                "median_abs_ret6h": 0,
                "ret6h_q01": 0,
                "ret6h_q05": 0,
                "atr_mean": 0,
                "atr_q10": 0,
                "atr_q50": 0,
                "atr_q90": 0,
                "atr_decile_worst": -1,
                "atr_decile_worst_share": 0,
                "atr_decile_pnl_json": "{}",
                "slice_overall_sharpe": 0,
                "slice_overall_sortino": 0,
                "slice_total_samples": 0,
                "slice_metrics_json": "{}",
                "error": str(e)
            })
        finally:
            candidate_mask = None
            side_sign = None
            feature_cache = precomputed.get("feature_series_cache")
            if isinstance(feature_cache, dict) and feature_cache:
                feature_cache.clear()
                tlog(f"Per-config cleanup: cleared stacked feature cache for {config_id}")
            gc.collect()
            tlog(f"Config cleanup done: {config_id}")
    
    tlog("Building results dataframe")
    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Learnability-first policy ranking:
    # Primary: maximize IC
    # Constraint: IC_std <= median IC_std across valid configs
    # Secondary: maximize KS then SNR
    # Tie-break: Sharpe, Sortino, mean_return_bps
    valid_icstd = results_df["ic_std"].to_numpy(dtype=float)
    valid_icstd = valid_icstd[np.isfinite(valid_icstd)]
    icstd_threshold = float(np.median(valid_icstd)) if len(valid_icstd) > 0 else 0.0
    results_df["policy_icstd_threshold"] = icstd_threshold
    results_df["policy_pass_stability"] = results_df["ic_std"] <= icstd_threshold
    sort_view = results_df.sort_values(
        by=[
            "policy_pass_stability",
            "ic",
            "ks_stat",
            "snr",
            "sharpe",
            "sortino",
            "mean_return_bps",
        ],
        ascending=[False, False, False, False, False, False, False],
    ).reset_index(drop=True)
    sort_view["policy_rank"] = np.arange(1, len(sort_view) + 1, dtype=np.int32)
    results_df = results_df.merge(sort_view[["config_id", "policy_rank"]], on="config_id", how="left")
    if len(sort_view) > 0:
        best_row = sort_view.iloc[0]
        tlog(
            "Policy best config: "
            f"{best_row['config_id']} | pass={bool(best_row['policy_pass_stability'])} "
            f"IC={best_row['ic']:.4f} IC_std={best_row['ic_std']:.4f} "
            f"KS={best_row['ks_stat']:.4f} SNR={best_row['snr']:.4f}"
        )
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    tlog("Saving output files")
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    if slice_results:
        slice_output = output_path.replace(".csv", "_slices.csv")
        pd.DataFrame(slice_results).to_csv(slice_output, index=False)
        logger.info(f"Slice-level results saved to: {slice_output}")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")
    
    # Print summary table
    print("\n" + "=" * 140)
    print("CANDIDATE SELECTION THRESHOLD COMPARISON RESULTS")
    print("=" * 140)
    print(f"{'Config':<10} {'Mode':<12} {'Pct':>5} {'Range':>6} {'VolZ':>5} {'SignC':>5} "
          f"{'N_Cand':>7} {'IC':>7} {'IC_std':>7} {'KS':>6} {'SNR':>6} {'Bal':>6} {'Sharpe':>7} {'Hit':>6} {'Sort':>7}")
    print("-" * 140)
    
    for _, row in results_df.iterrows():
        range_str = f"{row['min_range_pct']:.2f}" if pd.notna(row.get('min_range_pct')) else "-"
        volz_str = f"{row['min_vol_zscore']:.1f}" if pd.notna(row.get('min_vol_zscore')) else "-"
        signc_str = f"{row['min_sign_consistency']:.0%}" if pd.notna(row.get('min_sign_consistency')) else "-"
        print(f"{row['config_id']:<10} {row['mode']:<12} {row['pct']:>5.2f} "
              f"{range_str:>6} {volz_str:>5} {signc_str:>5} "
              f"{row['n_candidates_mean']:>7.1f} {row['ic']:>7.4f} {row['ic_std']:>7.4f} "
              f"{row['ks_stat']:>6.3f} {row['snr']:>6.2f} {row['class_balance']:>5.1%} "
              f"{row['sharpe']:>7.2f} {row['hit_rate']:>5.1%} {row['sortino']:>7.2f}")

    if len(sort_view) > 0:
        best = sort_view.iloc[0]
        print(
            f"Policy best: {best['config_id']} | Rank=1 | "
            f"Pass={bool(best['policy_pass_stability'])} | "
            f"IC={best['ic']:.4f} IC_std={best['ic_std']:.4f} "
            f"KS={best['ks_stat']:.4f} SNR={best['snr']:.4f} "
            f"Sharpe={best['sharpe']:.2f}"
        )

    print("=" * 140)
    
    return results_df


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare candidate selection methods for extreme price movements"
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Path to feature data (directory or parquet file)"
    )
    parser.add_argument(
        "--panel",
        required=False,
        default=None,
        help="Path to panel data (klines/OHLCV) - optional if features contain returns"
    )
    parser.add_argument(
        "--output",
        default="reports/candidate_threshold_comparison.csv",
        help="Output CSV path (default: reports/candidate_threshold_comparison.csv)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Floating-point dtype for feature matrices (default: float32)"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=60,
        help="Cap number of MODEL_FEATURES used for OOF modeling (default: 60)"
    )
    parser.add_argument(
        "--stage3",
        action="store_true",
        help="Enable stage 3: run winners with pct=[0.05, 0.06, 0.07]. Use with --winners (or auto-runs after stage 1+2)"
    )
    parser.add_argument(
        "--winners",
        nargs="+",
        default=[],
        help="List of winning config_ids from stage 2 to test in stage 3 (auto-selected if not provided)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # =============================================================================
    # Auto-select winners if not provided and prepare for potential Stage 3
    # =============================================================================
    auto_stage3 = args.stage3 or len(args.winners) > 0
    
    run_comparison(
        args.features,
        args.panel,
        args.output,
        dtype=args.dtype,
        max_features=args.max_features,
        stage3=args.stage3,
        winners=args.winners,
    )
    
    # =============================================================================
    # Auto-run Stage 3 if requested
    # =============================================================================
    if auto_stage3:
        # Load results to find top winners if not provided
        if not args.winners:
            try:
                import pandas as pd
                prev_results = pd.read_csv(args.output)
                # Find top 4 configs by slice_overall_sortino from FULL_E* configs
                full_configs = prev_results[prev_results['config_id'].str.contains('FULL_E')]
                if len(full_configs) > 0:
                    top_winners = full_configs.nlargest(4, 'slice_overall_sortino')['config_id'].tolist()
                    tlog(f"Auto-selected top 4 winners: {top_winners}")
                else:
                    tlog("No FULL_E configs found, cannot auto-select winners")
                    top_winners = []
            except Exception as e:
                tlog(f"Could not auto-select winners: {e}")
                top_winners = []
        else:
            top_winners = args.winners
        
        if top_winners:
            # Run Stage 3 with top winners
            stage3_output = args.output.replace('.csv', '_stage3.csv')
            tlog(f"Auto-running Stage 3 with winners: {top_winners}")
            run_comparison(
                args.features,
                args.panel,
                stage3_output,
                dtype=args.dtype,
                max_features=args.max_features,
                stage3=True,
                winners=top_winners,
                use_extratrees=True,  # Use ExtraTrees for Stage 3
            )
