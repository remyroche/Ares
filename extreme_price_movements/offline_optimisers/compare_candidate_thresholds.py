#!/usr/bin/env python
"""
Candidate Selection Threshold Comparison

Compares Fixed, ATR-normalized, and Volume-Weighted candidate selection methods.
Uses ExtraTrees with the same parameters as the target race in training.py.

Usage:
    python -m extreme_price_movements.offline_optimisers.compare_candidate_thresholds \
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
import glob
import logging
from copy import deepcopy
from collections import OrderedDict
from typing import Dict, Optional, Any, Iterable
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import RobustScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Re-use from extreme_price_movements
from extreme_price_movements.purged_cv import PurgedKFold
from extreme_price_movements.config import (
    TEST_FEATURE_KEYS,
    CFG,
    PERP_FEATURE_KEYS,
    enable_perp_feature_keys,
)
from extreme_price_movements.data_store import (
    load_features as load_features_pipeline,
    PartitionedOHLCVStore,
    to_panel,
)
from extreme_price_movements.training import (
    build_hourly_training_set_and_weights,
    build_grid_aggregated_tb_cache,
)
from extreme_price_movements.sample_weights import compute_avg_uniqueness
from extreme_price_movements.utils import tprint
from extreme_price_movements import fast_funcs as ff
from extreme_price_movements.training_defaults import (
    get_candidate_filter_defaults,
    get_barrier_factory_defaults,
    get_target_race_model_defaults,
)
from extreme_price_movements.offline_optimisers.params_store import (
    REPORTS_DIR,
    save_best_params_csv,
    apply_offline_optimizer_best_params,
    CANDIDATE_BEST_PARAMS_CSV,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)



OOF_MAX_SAMPLES = 600_000
OOF_RIDGE_MAX_TRAIN_SAMPLES = 700_000
FEATURE_CHUNK_SIZE = 8
_PSUTIL_WARNED = False
MAX_GEOMETRY_CACHE_KEYS = 1
STAGE1_OOF_MAX_SAMPLES = 250_000
STAGE1_OOF_SPLITS = 2
STAGE23_OOF_SPLITS = 3
STAGE1_SYMBOL_SUBSAMPLE_STEP = 1
STAGE23_SYMBOL_SUBSAMPLE_STEP = 1
GEOMETRY_CACHE_MAX_MB = 1536
EPS = 1e-12
BASE_ROUND_TRIP_FEE_PCT = 0.3
BASE_ROUND_TRIP_FEE_DEC = BASE_ROUND_TRIP_FEE_PCT / 100.0
_LEGACY_ROUND_TRIP_FEE_PCT = 0.5
_LEGACY_ROUND_TRIP_FEE_DEC = _LEGACY_ROUND_TRIP_FEE_PCT / 100.0


def _is_close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def _normalize_fee_default(value: Optional[float], *, legacy: float, new_default: float) -> float:
    """Map unset/legacy fee defaults to the new baseline while preserving explicit overrides."""
    if value is None:
        return float(new_default)
    v = float(value)
    if _is_close(v, legacy):
        return float(new_default)
    return v


def _append_suffix(path: str, suffix: str) -> str:
    norm = str(path).rstrip("/\\")
    if norm.endswith(suffix):
        return norm
    return f"{norm}{suffix}"


def _resolve_runtime_cfg(*, perps: bool = False, data_root: Optional[str] = None) -> dict:
    cfg = apply_offline_optimizer_best_params(deepcopy(CFG))
    if data_root:
        cfg["data_root"] = str(data_root)
    if perps:
        cfg["use_perps"] = True
        cfg["data_root"] = _append_suffix(cfg.get("data_root", "../data"), "_perp")
        cfg = enable_perp_feature_keys(cfg)
        existing_test = list(cfg.get("test_feature_keys", TEST_FEATURE_KEYS))
        cfg["test_feature_keys"] = list(dict.fromkeys(existing_test + list(PERP_FEATURE_KEYS)))
    cfg["label_round_trip_fee_pct"] = _normalize_fee_default(
        cfg.get("label_round_trip_fee_pct"),
        legacy=_LEGACY_ROUND_TRIP_FEE_PCT,
        new_default=BASE_ROUND_TRIP_FEE_PCT,
    )
    cfg["sample_weight_fee_rt"] = _normalize_fee_default(
        cfg.get("sample_weight_fee_rt"),
        legacy=_LEGACY_ROUND_TRIP_FEE_DEC,
        new_default=BASE_ROUND_TRIP_FEE_DEC,
    )
    cfg["optimiser_fee_pct"] = _normalize_fee_default(
        cfg.get("optimiser_fee_pct"),
        legacy=_LEGACY_ROUND_TRIP_FEE_DEC,
        new_default=BASE_ROUND_TRIP_FEE_DEC,
    )
    return cfg


def _find_latest_feature_dir(data_root: str) -> Optional[str]:
    feat_dir = os.path.join(data_root, "features")
    if not os.path.isdir(feat_dir):
        return None
    dirs = sorted(glob.glob(os.path.join(feat_dir, "20*")))
    if not dirs:
        return None
    return dirs[-1]

_TARGET_RACE_DEFAULTS = get_target_race_model_defaults(CFG)
ET_REGRESSOR_PARAMS = dict(_TARGET_RACE_DEFAULTS["et_params"])
RIDGE_SCREEN_ALPHA = float(_TARGET_RACE_DEFAULTS["ridge_screen_alpha"])
RIDGE_SCREEN_TOP_FRAC = float(_TARGET_RACE_DEFAULTS["ridge_screen_top_frac"])


def cleanup_run_caches() -> None:
    """Best-effort cleanup of in-process caches before each optimizer run."""
    gc.collect()
    try:
        import pyarrow as pa  # type: ignore

        pa.default_memory_pool().release_unused()
    except Exception:
        pass


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


def apply_deprado_concurrency_weight(
    weights: np.ndarray,
    meta_idx: Any,
    horizon_hours: int,
    enable: bool = True,
) -> np.ndarray:
    """Apply de Prado concurrency uniqueness scaling to slice weights (compare-only path)."""
    w = np.asarray(weights, dtype=np.float32)
    if not enable or w.size == 0:
        return w
    if not isinstance(meta_idx, pd.MultiIndex) or meta_idx.nlevels < 1:
        return w

    ts = pd.to_datetime(meta_idx.get_level_values(0), utc=True, errors="coerce")
    valid = pd.Series(ts).notna().values
    if valid.sum() < 4:
        return w

    t_start = pd.DatetimeIndex(ts[valid])
    t_end = t_start + pd.Timedelta(hours=int(max(1, horizon_hours)))
    label_times = pd.DataFrame({"t_start": t_start, "t_end": t_end}, index=np.where(valid)[0])
    grid = pd.DatetimeIndex(np.unique(np.concatenate([t_start.values, t_end.values]))).sort_values()

    uniq = compute_avg_uniqueness(label_times=label_times, time_grid=grid)
    u = np.ones(len(w), dtype=np.float32)
    if len(uniq) > 0:
        u_vals = np.asarray(uniq.values, dtype=np.float32)
        u_idx = np.asarray(uniq.index.values, dtype=int)
        u[u_idx] = u_vals

    u = np.nan_to_num(u, nan=1.0, posinf=1.0, neginf=1.0)
    mean_u = float(np.mean(u)) if len(u) else 1.0
    if mean_u > 1e-8:
        u = u / mean_u
    w2 = w * u
    w2 = np.clip(w2, 1e-8, None)
    return w2.astype(np.float32)


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


def normalize_feature_time_indices(feats: dict) -> dict:
    """Normalize feature DatetimeIndex objects to UTC-naive to avoid tz mismatch drops."""
    for name, df in feats.items():
        if not isinstance(df, pd.DataFrame):
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            continue
        try:
            idx_norm = pd.to_datetime(df.index, utc=True).tz_localize(None)
            if not df.index.equals(idx_norm):
                df2 = df.copy(deep=False)
                df2.index = idx_norm
                feats[name] = df2
        except Exception:
            continue
    return feats


def select_symbol_subset(all_symbols: list[str], step: int = 3, limit: int = 0) -> list[str]:
    """Select every `step`-th symbol in alphabetical order, optionally capped by `limit`."""
    syms_sorted = sorted({str(s) for s in all_symbols})
    selected = syms_sorted[::max(1, int(step))]
    lim = int(limit)
    return selected if lim <= 0 else selected[:lim]


def subset_feature_universe(feats: dict, symbols: list[str]) -> dict:
    """Subset all feature dataframes to a reduced symbol universe."""
    sym_set = set(symbols)
    dropped = 0
    for name, df in feats.items():
        if not isinstance(df, pd.DataFrame):
            continue
        keep_cols = [c for c in df.columns if c in sym_set]
        if keep_cols:
            feats[name] = df.reindex(columns=keep_cols)
        else:
            dropped += 1
    if dropped > 0:
        tlog(f"Symbol subset removed all columns for {dropped} feature tables")
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


def _resolve_feature_aligned_array(
    feats: dict,
    candidate_names: list[str],
    target_index: pd.Index,
    target_columns: pd.Index,
    float_dtype: np.dtype,
) -> Optional[np.ndarray]:
    """Return aligned feature array for the first available feature name."""
    feat = _first_available_feature(feats, candidate_names)
    if feat is None:
        return None
    aligned = feat.reindex(index=target_index, columns=target_columns)
    return aligned.to_numpy(dtype=float_dtype, copy=False)


def apply_tail_filter_on_prefilter(
    prefilter_arr: np.ndarray,
    tail_mode: Optional[str],
    top_frac: Optional[float],
    vol_tail_arr: Optional[np.ndarray],
    entropy_tail_arr: Optional[np.ndarray],
) -> np.ndarray:
    """Apply a top-tail filter computed strictly inside an existing prefilter mask.

    The row-wise quantile thresholds are estimated only on points already selected
    by `prefilter_arr` (never on the full dataset).
    """
    mode = str(tail_mode or "none").lower()
    frac = float(top_frac or 0.0)
    if mode in {"none", "off", ""} or frac <= 0.0 or frac >= 1.0:
        return prefilter_arr

    out = np.array(prefilter_arr, copy=True, dtype=bool)
    n_rows = out.shape[0]
    q = 1.0 - frac

    for i in range(n_rows):
        row_base = out[i]
        n_base = int(row_base.sum())
        if n_base < 8:
            continue

        keep_vol = None
        keep_entropy = None

        if vol_tail_arr is not None and mode in {"vol24_top", "vol24_top20", "vol_or_entropy_top20"}:
            row_vol = vol_tail_arr[i]
            valid = row_base & np.isfinite(row_vol)
            if int(valid.sum()) >= 8:
                thr = np.nanquantile(row_vol[valid], q)
                keep_vol = valid & (row_vol >= thr)

        if entropy_tail_arr is not None and mode in {"entropy_top", "entropy_top20", "vol_or_entropy_top20"}:
            row_ent = entropy_tail_arr[i]
            valid = row_base & np.isfinite(row_ent)
            if int(valid.sum()) >= 8:
                thr = np.nanquantile(row_ent[valid], q)
                keep_entropy = valid & (row_ent >= thr)

        if mode in {"vol24_top", "vol24_top20"} and keep_vol is not None:
            out[i] = keep_vol
        elif mode in {"entropy_top", "entropy_top20"} and keep_entropy is not None:
            out[i] = keep_entropy
        elif mode == "vol_or_entropy_top20":
            if keep_vol is not None and keep_entropy is not None:
                out[i] = keep_vol | keep_entropy
            elif keep_vol is not None:
                out[i] = keep_vol
            elif keep_entropy is not None:
                out[i] = keep_entropy

    return out


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
    aligned_mask = align_candidate_mask_to_panel_symbols(candidate_mask, panel)
    overlap = close_df.columns.intersection(aligned_mask.columns)
    if len(overlap) == 0:
        return False, "no symbol overlap between candidate mask and panel close"
    if int(aligned_mask.to_numpy(dtype=bool, copy=False).sum()) == 0:
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

    trend_aligned_source = trend_df
    overlap_cols = trend_aligned_source.columns.intersection(candidate_mask.columns)
    overlap_n = len(overlap_cols)
    if overlap_n == 0:
        # Try normalized symbol matching (e.g. BTC/USDT vs BTC_USDT).
        norm_to_trend: dict[str, str] = {}
        dup_norm: set[str] = set()
        for c in trend_aligned_source.columns:
            k = _normalize_symbol(c)
            if k in norm_to_trend and norm_to_trend[k] != c:
                dup_norm.add(k)
            else:
                norm_to_trend[k] = c
        for k in dup_norm:
            norm_to_trend.pop(k, None)

        rename_map: dict[Any, Any] = {}
        for c in candidate_mask.columns:
            src = norm_to_trend.get(_normalize_symbol(c))
            if src is not None:
                rename_map[src] = c
        if rename_map:
            trend_aligned_source = trend_aligned_source.rename(columns=rename_map)
            if trend_aligned_source.columns.has_duplicates:
                trend_aligned_source = trend_aligned_source.T.groupby(level=0).last().T
            overlap_cols = trend_aligned_source.columns.intersection(candidate_mask.columns)
            overlap_n = len(overlap_cols)
            tlog(
                "Bucket mask: recovered overlap via normalized symbol mapping; "
                f"overlap={overlap_n}/{candidate_mask.shape[1]}"
            )
        if overlap_n == 0:
            tlog(
                "Bucket mask: trend_pct has zero column overlap with candidate mask; "
                f"trend_cols={trend_df.shape[1]}, candidate_cols={candidate_mask.shape[1]} | skipping move-bucket filter"
            )
            return candidate_mask
    if overlap_n < max(5, int(0.1 * candidate_mask.shape[1])):
        tlog(
            "Bucket mask: low trend/candidate column overlap; "
            f"overlap={overlap_n}/{candidate_mask.shape[1]}"
        )

    trend_aligned = trend_aligned_source.reindex(index=candidate_mask.index, columns=candidate_mask.columns)
    trend_arr = trend_aligned.to_numpy(dtype=np.float32, copy=False)
    finite_ratio = float(np.isfinite(trend_arr).mean())
    if finite_ratio < 0.05:
        tlog(
            f"Bucket mask: trend_pct alignment has low finite coverage ({finite_ratio:.2%}) "
            f"for shape={trend_arr.shape}; skipping move-bucket filter"
        )
        return candidate_mask
    if finite_ratio < 0.50:
        tlog(
            f"Bucket mask: degraded trend_pct finite coverage ({finite_ratio:.2%}) "
            f"for shape={trend_arr.shape}; applying filter with caution"
        )

    if move_bucket == "up":
        trend_mask = trend_aligned > 0
    else:
        trend_mask = trend_aligned <= 0

    bucket_mask = (candidate_mask & trend_mask).fillna(False)
    base_n = int(candidate_mask.to_numpy(dtype=bool, copy=False).sum())
    bucket_n = int(bucket_mask.to_numpy(dtype=bool, copy=False).sum())
    if base_n > 0 and bucket_n == 0:
        finite = np.isfinite(trend_arr)
        up_share = float(np.nanmean(trend_arr > 0)) if finite.any() else 0.0
        down_share = float(np.nanmean(trend_arr <= 0)) if finite.any() else 0.0
        if finite.any():
            vals = trend_arr[finite]
            q10 = float(np.nanpercentile(vals, 10))
            q50 = float(np.nanpercentile(vals, 50))
            q90 = float(np.nanpercentile(vals, 90))
            bucket_stats = f"trend_q10={q10:.4g},trend_q50={q50:.4g},trend_q90={q90:.4g}"
        else:
            bucket_stats = "trend_quantiles=nan"
        tlog(
            "Bucket mask collapsed to zero after trend filter: "
            f"move_bucket={move_bucket}, pre_selected={base_n}, post_selected={bucket_n}, "
            f"finite_fraction={finite_ratio:.2%}, trend_up_share={up_share:.2%}, trend_down_share={down_share:.2%}, "
            f"{bucket_stats}, sub_mask_shape={bucket_mask.shape}"
        )
    return bucket_mask


def cache_geometry_entry(
    cache: "OrderedDict[tuple[Any, ...], tuple[dict, dict]]",
    key: tuple[Any, ...],
    value: tuple[dict, dict],
    max_keys: int = MAX_GEOMETRY_CACHE_KEYS,
) -> None:
    """Store geometry cache entry with LRU eviction to cap memory growth."""
    if key in cache:
        cache.move_to_end(key)
        cache[key] = value
        return
    cache[key] = value
    while len(cache) > max_keys:
        evicted_key, _ = cache.popitem(last=False)
        tlog(f"Training-slice geometry cache evicted oldest key hash={hash(evicted_key)}")


def _estimate_geometry_cache_size_bytes(tb_cache_by_h_side: dict, geom_cache_by_h_side: dict) -> int:
    total = 0
    for tb_triplet in tb_cache_by_h_side.values():
        if isinstance(tb_triplet, tuple):
            for df in tb_triplet:
                if isinstance(df, pd.DataFrame):
                    total += int(df.memory_usage(index=True, deep=True).sum())
    for geom_pack in geom_cache_by_h_side.values():
        if isinstance(geom_pack, dict):
            for df in geom_pack.values():
                if isinstance(df, pd.DataFrame):
                    total += int(df.memory_usage(index=True, deep=True).sum())
    return total


def _geometry_cache_dir(
    output_path: str,
    feature_path: str,
    panel_close: pd.DataFrame,
    shared_geometry_key: tuple[Any, ...],
) -> str:
    reports_dir = os.path.abspath(os.path.dirname(output_path) or ".")
    root = os.path.join(reports_dir, ".geometry_cache")
    cols_hash = hashlib.sha1(
        ",".join(map(str, panel_close.columns)).encode("utf-8")
    ).hexdigest()[:16]
    sig = {
        "feature_path": os.path.abspath(feature_path),
        "panel_shape": [int(panel_close.shape[0]), int(panel_close.shape[1])],
        "panel_start": str(panel_close.index.min()),
        "panel_end": str(panel_close.index.max()),
        "panel_cols_hash": cols_hash,
        "shared_geometry_key": [str(v) for v in shared_geometry_key],
    }
    cache_id = hashlib.sha1(json.dumps(sig, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return os.path.join(root, cache_id)


def load_persisted_geometry_cache(cache_dir: str) -> Optional[tuple[dict, dict]]:
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        entries = manifest.get("entries", [])
        if not entries:
            return None
        tb_cache_by_h_side: dict[tuple[int, str], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        geom_cache_by_h_side: dict[tuple[int, str], dict[str, pd.DataFrame]] = {}
        for item in entries:
            h = int(item["h"])
            side = str(item["side"])
            lbl = pd.read_parquet(os.path.join(cache_dir, item["tb_labels"]))
            ret = pd.read_parquet(os.path.join(cache_dir, item["tb_returns"]))
            qual = pd.read_parquet(os.path.join(cache_dir, item["tb_quality"]))
            n_tp = pd.read_parquet(os.path.join(cache_dir, item["geom_n_tp"]))
            n_sl = pd.read_parquet(os.path.join(cache_dir, item["geom_n_sl"]))
            n_to = pd.read_parquet(os.path.join(cache_dir, item["geom_n_to"]))
            tb_cache_by_h_side[(h, side)] = (lbl, ret, qual)
            geom_cache_by_h_side[(h, side)] = {"n_tp": n_tp, "n_sl": n_sl, "n_to": n_to}
        return tb_cache_by_h_side, geom_cache_by_h_side
    except Exception as exc:
        tlog(f"Persisted geometry cache load failed: {exc}")
        return None


def save_persisted_geometry_cache(
    cache_dir: str,
    tb_cache_by_h_side: dict,
    geom_cache_by_h_side: dict,
    max_mb: int = GEOMETRY_CACHE_MAX_MB,
) -> bool:
    try:
        est_bytes = _estimate_geometry_cache_size_bytes(tb_cache_by_h_side, geom_cache_by_h_side)
        max_bytes = int(max_mb) * 1024 * 1024
        if est_bytes > max_bytes:
            tlog(
                "Skipping persisted geometry cache write: "
                f"estimated_size={est_bytes / (1024**2):.1f}MB > limit={max_mb}MB"
            )
            return False

        os.makedirs(cache_dir, exist_ok=True)
        entries = []
        for (h, side), tb_triplet in sorted(tb_cache_by_h_side.items(), key=lambda x: (int(x[0][0]), str(x[0][1]))):
            if (h, side) not in geom_cache_by_h_side:
                continue
            lbl, ret, qual = tb_triplet
            geom_pack = geom_cache_by_h_side[(h, side)]
            base = f"h{int(h)}_{side}"
            files = {
                "tb_labels": f"{base}_tb_labels.parquet",
                "tb_returns": f"{base}_tb_returns.parquet",
                "tb_quality": f"{base}_tb_quality.parquet",
                "geom_n_tp": f"{base}_geom_n_tp.parquet",
                "geom_n_sl": f"{base}_geom_n_sl.parquet",
                "geom_n_to": f"{base}_geom_n_to.parquet",
            }
            lbl.to_parquet(os.path.join(cache_dir, files["tb_labels"]), compression="zstd")
            ret.to_parquet(os.path.join(cache_dir, files["tb_returns"]), compression="zstd")
            qual.to_parquet(os.path.join(cache_dir, files["tb_quality"]), compression="zstd")
            geom_pack["n_tp"].to_parquet(os.path.join(cache_dir, files["geom_n_tp"]), compression="zstd")
            geom_pack["n_sl"].to_parquet(os.path.join(cache_dir, files["geom_n_sl"]), compression="zstd")
            geom_pack["n_to"].to_parquet(os.path.join(cache_dir, files["geom_n_to"]), compression="zstd")
            entries.append({"h": int(h), "side": side, **files})

        manifest = {
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "estimated_size_mb": float(est_bytes / (1024**2)),
            "entries": entries,
        }
        with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        tlog(
            "Persisted geometry cache saved: "
            f"entries={len(entries)}, est_size={est_bytes / (1024**2):.1f}MB"
        )
        return True
    except Exception as exc:
        tlog(f"Persisted geometry cache save failed: {exc}")
        return False


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
    geometry_cache: Optional[dict] = None,
    geometry_cache_key: Optional[tuple] = None,
    precomputed_geometry: Optional[tuple[dict, dict]] = None,
    sample_weight_sink: Optional[list[dict]] = None,
    config_id: Optional[str] = None,
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
    tp_ratio = float(cfg_variant.get("barrier_k_tp", 1.0))
    sl_ratio = float(cfg_variant.get("barrier_sl_base_mult", 0.5))
    tp_sl_ratio = f"{tp_ratio:.2f}:{sl_ratio:.2f}"
    rows = []
    mkt_gates = build_proxy_mkt_gates(feats)
    ts_end = candidate_mask.index.max()
    bucket_mask_cache: dict[tuple[str, str], pd.DataFrame] = {}

    # Fast-fail before expensive grid/label precomputation.
    # If every side/kind bucket is empty, training slices are structurally impossible.
    bucket_nonzero_counts: dict[tuple[str, str], int] = {}
    for side in ["long", "short"]:
        for kind in ["mr", "tf"]:
            move_bucket = _bucket_move_bucket(side, kind)
            bucket_mask = _build_bucket_candidate_mask(
                candidate_mask=candidate_mask,
                feats=feats,
                move_bucket=move_bucket,
            )
            bucket_mask_cache[(side, kind)] = bucket_mask
            bucket_nonzero_counts[(side, kind)] = int(bucket_mask.to_numpy(dtype=bool, copy=False).sum())

    total_bucket_selected = int(sum(bucket_nonzero_counts.values()))
    if total_bucket_selected == 0:
        tlog(
            "Training-slice fast skip: all side/kind bucket masks are empty; "
            "skipping geometry precompute"
        )
        logger.warning(
            "All training-slice buckets empty after bucketing; slice metrics are not informative for this config"
        )
        for side in ["long", "short"]:
            for kind in ["mr", "tf"]:
                for h in horizons:
                    rows.append(
                        {
                            "slice": f"{side}_{kind}",
                            "side": side,
                            "kind": kind,
                            "horizon": h,
                            "n_samples": 0,
                            "n_days": 0,
                            "opportunities_per_day": 0.0,
                            "tp_sl_ratio": tp_sl_ratio,
                            "label_pos_rate": 0,
                            "mean_ret_bps": 0,
                            "sharpe": 0,
                            "sortino": 0,
                            "weighted_ret_bps": 0,
                        }
                    )
        if cache is not None and cache_key is not None:
            cache[cache_key] = [dict(r) for r in rows]
        tlog("Training-slice evaluation done")
        return rows

    if precomputed_geometry is not None:
        tb_cache_by_h_side, geom_cache_by_h_side = precomputed_geometry
        tlog("Training-slice geometry reuse: using precomputed shared geometry")
    elif (
        geometry_cache is not None
        and geometry_cache_key is not None
        and geometry_cache_key in geometry_cache
    ):
        tb_cache_by_h_side, geom_cache_by_h_side = geometry_cache[geometry_cache_key]
        if isinstance(geometry_cache, OrderedDict):
            geometry_cache.move_to_end(geometry_cache_key)
        tlog(
            "Training-slice geometry cache hit: "
            f"key_hash={hash(geometry_cache_key)}, horizons={tuple(horizons)}, "
            f"k_tp={cfg_variant.get('barrier_k_tp')}, sl_base={cfg_variant.get('barrier_sl_base_mult')}, "
            f"tp_lo={cfg_variant.get('barrier_tp_lo')}, tp_hi={cfg_variant.get('barrier_tp_hi')}"
        )
    else:
        tlog(
            "Training-slice geometry cache miss: building grid "
            f"key_hash={hash(geometry_cache_key) if geometry_cache_key is not None else 'none'}, "
            f"horizons={tuple(horizons)}, k_tp={cfg_variant.get('barrier_k_tp')}, "
            f"sl_base={cfg_variant.get('barrier_sl_base_mult')}, "
            f"tp_lo={cfg_variant.get('barrier_tp_lo')}, tp_hi={cfg_variant.get('barrier_tp_hi')}"
        )
        tb_cache_by_h_side, geom_cache_by_h_side = build_grid_aggregated_tb_cache(
            panel=panel,
            feats=feats,
            cfg=cfg_variant,
            horizons=horizons,
            trade_sides=["long", "short"],
        )
        if geometry_cache is not None and geometry_cache_key is not None:
            if isinstance(geometry_cache, OrderedDict):
                cache_geometry_entry(geometry_cache, geometry_cache_key, (tb_cache_by_h_side, geom_cache_by_h_side))
            else:
                geometry_cache[geometry_cache_key] = (tb_cache_by_h_side, geom_cache_by_h_side)

    for side in ["long", "short"]:
        for kind in ["mr", "tf"]:
            tlog(f"Training slice loop: side={side}, kind={kind}")
            cand_filter = "unknown"
            if side == "long":
                cand_filter = "worst" if kind == "mr" else "best"
            else:
                cand_filter = "best" if kind == "mr" else "worst"
            trend_filter = "up" if cand_filter == "best" else "down"
            if (side == "long" and kind == "tf") or (side == "short" and kind == "mr"):
                trend_filter = "down"
            feat_key = "tf_feature_keys" if kind == "tf" else "mr_feature_keys"
            move_bucket = _bucket_move_bucket(side, kind)
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
                            "n_days": 0,
                            "opportunities_per_day": 0.0,
                            "tp_sl_ratio": tp_sl_ratio,
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
                w = apply_deprado_concurrency_weight(
                    weights=w,
                    meta_idx=meta_idx,
                    horizon_hours=int(h),
                    enable=bool(cfg_variant.get("compare_use_deprado_concurrency_weight", True)),
                )
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
                if isinstance(meta_idx, pd.MultiIndex):
                    ts_vals = pd.to_datetime(meta_idx.get_level_values(0), utc=True, errors="coerce")
                else:
                    ts_vals = pd.to_datetime(pd.Index([None] * len(y_ret)), utc=True, errors="coerce")
                ts_vals = pd.Index(ts_vals).dropna()
                n_days = int(ts_vals.normalize().nunique()) if len(ts_vals) > 0 else 0
                opportunities_per_day = float(len(y_ret) / max(1, n_days))

                rows.append(
                    {
                        "slice": f"{side}_{kind}",
                        "side": side,
                        "kind": kind,
                        "horizon": h,
                        "n_samples": int(len(y_ret)),
                        "n_days": n_days,
                        "opportunities_per_day": opportunities_per_day,
                        "tp_sl_ratio": tp_sl_ratio,
                        "label_pos_rate": float(np.nanmean(y_bin >= 0.5)),
                        "mean_ret_bps": mean_ret * 1e4,
                        "sharpe": sharpe,
                        "sortino": sortino,
                        "weighted_ret_bps": weighted_ret * 1e4,
                    }
                )
                if sample_weight_sink is not None:
                    if isinstance(meta_idx, pd.MultiIndex):
                        ts_vals = meta_idx.get_level_values(0)
                        sym_vals = meta_idx.get_level_values(1) if meta_idx.nlevels > 1 else pd.Index([None] * len(meta_idx))
                    else:
                        ts_vals = pd.Index([None] * len(w))
                        sym_vals = pd.Index([None] * len(w))
                    w_clipped = np.asarray(np.clip(w, 1e-8, None), dtype=np.float32)
                    for i in range(len(w_clipped)):
                        sample_weight_sink.append(
                            {
                                "config_id": config_id or "unknown",
                                "side": side,
                                "kind": kind,
                                "horizon": int(h),
                                "timestamp": ts_vals[i] if i < len(ts_vals) else None,
                                "symbol": sym_vals[i] if i < len(sym_vals) else None,
                                "sample_weight": float(w_clipped[i]),
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
            "slice_overall_opportunities_per_day": 0.0,
            "slice_total_samples": 0,
            "slice_metrics_json": "{}",
        }

    sdf = pd.DataFrame(slice_rows)
    if sdf.empty:
        return {
            "slice_overall_sharpe": 0.0,
            "slice_overall_sortino": 0.0,
            "slice_overall_opportunities_per_day": 0.0,
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
                "opportunities_per_day": float(np.average(g["opportunities_per_day"], weights=np.clip(g["n_samples"], 1, None))),
                "tp_sl_ratio": str(g["tp_sl_ratio"].dropna().iloc[0]) if g["tp_sl_ratio"].notna().any() else "",
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
    overall_opportunities_per_day = (
        float(np.average(grouped["opportunities_per_day"], weights=np.clip(grouped["n_samples"], 1, None)))
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
                "opportunities_per_day": row["opportunities_per_day"],
                "tp_sl_ratio": row["tp_sl_ratio"],
            }
            for _, row in grouped.iterrows()
        }
    )

    return {
        "slice_overall_sharpe": overall_sharpe,
        "slice_overall_sortino": overall_sortino,
        "slice_overall_opportunities_per_day": overall_opportunities_per_day,
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


def _cusum_zscore_std(ret_df: pd.DataFrame, span: int = 96, z_cap: float = 8.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standard z-score using EWM sigma with clipping."""
    sigma = ret_df.ewm(span=int(span), min_periods=max(8, int(span // 4)), adjust=False).std()
    sigma = sigma.abs().replace(0, np.nan)
    z = (ret_df / (sigma + EPS)).clip(lower=-float(z_cap), upper=float(z_cap))
    return z.astype(np.float32), sigma.astype(np.float32)


def _cusum_zscore_mad(ret_df: pd.DataFrame, win: int = 256, z_cap: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robust z-score using rolling MAD."""
    minp = max(10, int(win // 5))
    med = ret_df.rolling(int(win), min_periods=minp).median()
    mad = (ret_df - med).abs().rolling(int(win), min_periods=minp).median()
    scale = 1.4826 * mad
    z = ((ret_df - med) / (scale + EPS)).clip(lower=-float(z_cap), upper=float(z_cap))
    return z.astype(np.float32), scale.astype(np.float32)


def _cusum_zscore_iqr(ret_df: pd.DataFrame, win: int = 256, z_cap: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robust z-score using rolling IQR."""
    minp = max(10, int(win // 5))
    q1 = ret_df.rolling(int(win), min_periods=minp).quantile(0.25)
    q3 = ret_df.rolling(int(win), min_periods=minp).quantile(0.75)
    iqr = q3 - q1
    scale = iqr / 1.349
    med = ret_df.rolling(int(win), min_periods=minp).median()
    z = ((ret_df - med) / (scale + EPS)).clip(lower=-float(z_cap), upper=float(z_cap))
    return z.astype(np.float32), scale.astype(np.float32)


def _cusum_remove_drift(z_df: pd.DataFrame, mu_win: int = 64, method: str = "ewm") -> pd.DataFrame:
    """Remove micro-drift from z before CUSUM accumulation."""
    if str(method).lower() == "ewm":
        mu = z_df.ewm(span=int(mu_win), adjust=False, min_periods=max(10, int(mu_win // 4))).mean()
    else:
        mu = z_df.rolling(int(mu_win), min_periods=max(10, int(mu_win // 4))).mean()
    return (z_df - mu).astype(np.float32)


def _build_cusum_z_for_frame(
    ret_df: pd.DataFrame,
    z_mode: str = "std_clip",
    z_cap: float = 8.0,
    robust_win: int = 256,
    drift_remove: bool = True,
    mu_win: int = 64,
    drift_method: str = "ewm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build robust z-score frame and companion scale frame for CUSUM."""
    mode = str(z_mode).lower()
    if mode == "std_clip":
        z_df, sigma_df = _cusum_zscore_std(ret_df, span=96, z_cap=z_cap)
    elif mode == "mad":
        z_df, sigma_df = _cusum_zscore_mad(ret_df, win=robust_win, z_cap=z_cap)
    elif mode == "iqr":
        z_df, sigma_df = _cusum_zscore_iqr(ret_df, win=robust_win, z_cap=z_cap)
    else:
        raise ValueError("cusum_z_mode must be one of: std_clip, mad, iqr")

    if bool(drift_remove):
        z_df = _cusum_remove_drift(z_df, mu_win=mu_win, method=drift_method)

    return z_df.astype(np.float32), sigma_df.astype(np.float32)


def _compute_cusum_strength_from_z(
    z_df: pd.DataFrame,
    h_in: float,
    shock_z: Optional[float] = 5.0,
    cooldown: int = 3,
) -> pd.DataFrame:
    """Trigger-only directional CUSUM impulse detector (no episode closure dependency)."""
    z_arr = z_df.to_numpy(dtype=np.float32, copy=False)
    out = np.zeros_like(z_arr, dtype=np.float32)
    n_rows, n_cols = z_arr.shape

    h_in = float(max(h_in, 1e-6))
    cooldown = int(max(0, cooldown))
    shock_z = None if shock_z is None else float(max(shock_z, 0.0))

    for c in range(n_cols):
        col = z_arr[:, c]
        s_pos = 0.0
        s_neg = 0.0
        cd = 0

        for i in range(n_rows):
            zi = float(col[i])
            if not np.isfinite(zi):
                s_pos = 0.0
                s_neg = 0.0
                if cd > 0:
                    cd -= 1
                continue

            if cd > 0:
                cd -= 1
                # Keep trigger-only semantics: do not accumulate during cooldown.
                s_pos = 0.0
                s_neg = 0.0
                continue

            # Instantaneous shock trigger for flash/gap-like impulses.
            if shock_z is not None and abs(zi) >= shock_z:
                out[i, c] = np.float32(np.sign(zi) * abs(zi))
                s_pos = 0.0
                s_neg = 0.0
                cd = cooldown
                continue

            # Directional CUSUM impulse accumulation.
            s_pos = max(0.0, s_pos + zi)
            s_neg = min(0.0, s_neg + zi)

            if s_pos >= h_in:
                out[i, c] = np.float32(s_pos)
                s_pos = 0.0
                s_neg = 0.0
                cd = cooldown
            elif s_neg <= -h_in:
                out[i, c] = np.float32(s_neg)
                s_pos = 0.0
                s_neg = 0.0
                cd = cooldown

    return pd.DataFrame(out, index=z_df.index, columns=z_df.columns, dtype=np.float32)


def _conditional_expand_with_z_and_sign(
    base_mask: pd.DataFrame,
    base_sign: pd.DataFrame,
    offsets: Iterable[int],
    z_df: Optional[pd.DataFrame],
    ret_df: Optional[pd.DataFrame],
    sigma_df: Optional[pd.DataFrame] = None,
    z_min: float = 1.0,
    sign_pct: float = 0.6,
    consistency_bars: int = 5,
    vol_ratio: float = 1.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand candidates but keep only bars supported by z-gate + sign consistency."""
    expanded = base_mask.copy()
    sign_out = base_sign.copy().astype(np.int8)
    if not offsets:
        return expanded, sign_out

    for off in offsets:
        shifted_mask = base_mask.shift(int(off)).fillna(False)
        shifted_sign = base_sign.shift(int(off)).fillna(0).astype(np.int8)

        cond = shifted_mask
        if z_df is not None:
            z_shift = z_df.shift(int(off))
            cond = cond & z_shift.abs().ge(float(z_min)).fillna(False)
        if ret_df is not None:
            r_shift = ret_df.shift(int(off))
            sign_match = (np.sign(r_shift) == shifted_sign).where(shifted_sign != 0, False)
            sm = sign_match.rolling(int(max(1, consistency_bars)), min_periods=1).mean()
            cond = cond & sm.ge(float(sign_pct)).fillna(False)
        if sigma_df is not None:
            sig_shift = sigma_df.shift(int(off))
            sig_base = sigma_df.where(base_mask)
            sig_base_med = sig_base.rolling(int(max(2, consistency_bars)), min_periods=1).median()
            vr = sig_shift / (sig_base_med + EPS)
            cond = cond & vr.ge(float(vol_ratio)).fillna(False)

        expanded |= cond
        fill_mask = (sign_out == 0) & cond & (shifted_sign != 0)
        sign_out = sign_out.where(~fill_mask, shifted_sign)

    sign_out = sign_out.where(expanded, 0).astype(np.int8)
    return expanded.fillna(False), sign_out


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

    cusum_pack: Dict[str, Any] = {"z": None, "sigma": None, "strength_by_h": {}, "atr_adjusted": False}
    ret_for_cusum = feats.get("ret1h")
    if ret_for_cusum is None:
        ret_for_cusum = ret_base
    ret_for_cusum = ret_for_cusum.astype(float_dtype, copy=False)
    if isinstance(atr_effective, pd.DataFrame):
        atr_for_cusum = atr_effective.reindex(index=ret_for_cusum.index, columns=ret_for_cusum.columns)
        ret_for_cusum = (ret_for_cusum / (atr_for_cusum.abs() + EPS)).astype(float_dtype, copy=False)
        cusum_pack["atr_adjusted"] = True
    z_mode = str(CFG.get("cusum_z_mode", "std_clip"))
    z_cap = float(CFG.get("cusum_z_cap", 8.0))
    robust_win = int(CFG.get("cusum_robust_win", 256))
    drift_remove = bool(CFG.get("cusum_drift_remove", True))
    mu_win = int(CFG.get("cusum_mu_win", 64))
    drift_method = str(CFG.get("cusum_drift_method", "ewm"))
    z_df, sigma_df = _build_cusum_z_for_frame(
        ret_for_cusum,
        z_mode=z_mode,
        z_cap=z_cap,
        robust_win=robust_win,
        drift_remove=drift_remove,
        mu_win=mu_win,
        drift_method=drift_method,
    )
    cusum_pack["z"] = z_df.astype(float_dtype, copy=False)
    cusum_pack["sigma"] = sigma_df.astype(float_dtype, copy=False)
    shock_z = CFG.get("cusum_shock_z", 5.0)
    if shock_z is not None:
        shock_z = float(shock_z)
    cooldown = int(CFG.get("cusum_impulse_cooldown", 3))
    use_sparse_strength = bool(CFG.get("cusum_strength_sparse", True))
    for h in (5.0, 6.0, 7.0):
        key = f"{h:.1f}"
        strength_df = _compute_cusum_strength_from_z(
            z_df,
            h_in=h,
            shock_z=shock_z,
            cooldown=cooldown,
        ).astype(float_dtype, copy=False)
        if not strength_df.index.equals(ret_for_cusum.index) or not strength_df.columns.equals(ret_for_cusum.columns):
            raise ValueError(f"CUSUM strength alignment mismatch for h={key}")
        if use_sparse_strength:
            strength_df = strength_df.astype(pd.SparseDtype(float_dtype, 0.0))
        cusum_pack["strength_by_h"][key] = strength_df
    # default CUSUM metric in common map
    metrics["cusum"] = cusum_pack["strength_by_h"].get("6.0")

    rvol_z = feats.get("rvol_z")
    volu_z = feats.get("volu_z")
    if rvol_z is not None and volu_z is not None:
        vol_combined = ((rvol_z.astype(float_dtype, copy=False) + volu_z.astype(float_dtype, copy=False)) / 2).astype(float_dtype, copy=False)
        vol_tilt = (1.0 + 0.12 * vol_combined.clip(lower=0.0, upper=3.0)).astype(float_dtype, copy=False)
        atr_base = metrics.get("atr", ret_base).astype(float_dtype, copy=False)
        metrics["atr_vol_weight"] = (atr_base * vol_tilt).astype(float_dtype, copy=False)
    else:
        metrics["atr_vol_weight"] = metrics.get("atr", ret_base).astype(float_dtype, copy=False)

    return {
        "metrics": metrics,
        "atr_effective": atr_effective,
        "cusum_pack": cusum_pack,
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


def compute_cross_sectional_base_mask_and_sign(
    metric: pd.DataFrame,
    pct: float,
    eligible_mask: Optional[np.ndarray] = None,
    low_count_policy: str = "keep_all",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cross-sectional base mask + sign with deterministic low-count handling."""
    arr = metric.to_numpy(dtype=np.float32, copy=False)
    n_rows, n_cols = arr.shape
    k = max(1, int(n_cols * pct))

    valid = np.isfinite(arr)
    if eligible_mask is not None:
        valid &= np.asarray(eligible_mask, dtype=bool)
    valid_counts = valid.sum(axis=1)

    # Deterministic low-trigger handling for sparse modes (e.g., CUSUM):
    # - keep_all: keep all valid points when count<k
    # - drop: drop timestamp entirely when count<k
    low_count_policy = str(low_count_policy).lower()
    row_keep_all = valid_counts < k

    arr_for_top = np.where(valid, arr, -np.inf)
    arr_for_bot = np.where(valid, arr, np.inf)

    # O(N) partial selection avoids full per-row sort for better scaling.
    top_idx = np.argpartition(arr_for_top, kth=max(n_cols - k, 0), axis=1)[:, -k:]
    bot_idx = np.argpartition(arr_for_bot, kth=max(k - 1, 0), axis=1)[:, :k]

    row_ids = np.repeat(np.arange(n_rows, dtype=np.int32), k)
    top_flat = top_idx.reshape(-1)
    bot_flat = bot_idx.reshape(-1)

    top_valid = valid[row_ids, top_flat]
    bot_valid = valid[row_ids, bot_flat]

    mask_arr = np.zeros((n_rows, n_cols), dtype=bool)
    sign_arr = np.zeros((n_rows, n_cols), dtype=np.int8)

    rr_top = row_ids[top_valid]
    cc_top = top_flat[top_valid]
    rr_bot = row_ids[bot_valid]
    cc_bot = bot_flat[bot_valid]

    mask_arr[rr_top, cc_top] = True
    sign_arr[rr_top, cc_top] = 1
    mask_arr[rr_bot, cc_bot] = True
    sign_arr[rr_bot, cc_bot] = -1

    if np.any(row_keep_all):
        if low_count_policy == "keep_all":
            keep_rows = np.where(row_keep_all)[0]
            valid_keep = valid[keep_rows]
            arr_keep = arr[keep_rows]
            pos = valid_keep & (arr_keep > 0)
            neg = valid_keep & (arr_keep < 0)
            mask_arr[keep_rows] = valid_keep
            sign_keep = np.zeros(valid_keep.shape, dtype=np.int8)
            sign_keep[pos] = 1
            sign_keep[neg] = -1
            sign_arr[keep_rows] = sign_keep
        elif low_count_policy == "drop":
            drop_rows = np.where(row_keep_all)[0]
            mask_arr[drop_rows] = False
            sign_arr[drop_rows] = 0

    return mask_arr, sign_arr


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
    return_sign: bool = False,
    prefilter_arr: Optional[np.ndarray] = None,
    low_count_policy: str = "keep_all",
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
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
        cached = base_mask_cache[base_cache_key]
        if isinstance(cached, tuple) and len(cached) == 2:
            base_mask_arr, base_sign_arr = cached
        else:
            base_mask_arr = cached
            base_sign_arr = np.zeros_like(base_mask_arr, dtype=np.int8)
    else:
        base_mask_arr, base_sign_arr = compute_cross_sectional_base_mask_and_sign(
            metric,
            pct,
            eligible_mask=prefilter_arr,
            low_count_policy=low_count_policy,
        )
        if base_mask_cache is not None and base_cache_key is not None:
            base_mask_cache[base_cache_key] = (base_mask_arr, base_sign_arr)

    # Filters are now applied pre-ranking through prefilter_arr for deterministic ranks.
    mask_arr = np.asarray(base_mask_arr, dtype=bool)

    mask_df = pd.DataFrame(mask_arr, index=metric.index, columns=metric.columns, dtype=bool)
    if not return_sign:
        return mask_df

    sign_filtered = np.where(mask_arr, base_sign_arr, 0).astype(np.int8, copy=False)
    sign_df = pd.DataFrame(sign_filtered, index=metric.index, columns=metric.columns, dtype=np.int8)
    return mask_df, sign_df


def gather_candidate_values(arr: np.ndarray, row_idx: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
    """Gather candidate values from 2D feature array using integer indices."""
    return arr[row_idx, col_idx]


class DataContainer:
    """Holds pre-aligned feature matrices and fast gather utilities."""

    def __init__(
        self,
        feats: dict,
        index: pd.Index,
        columns: pd.Index,
        feature_names: list[str],
        float_dtype: np.dtype,
    ):
        self.index = index
        self.columns = columns
        self.float_dtype = float_dtype
        self.feat_arr: dict[str, np.ndarray] = {}

        for f in feature_names:
            df = feats.get(f)
            if df is None or not isinstance(df, pd.DataFrame):
                continue
            if (
                df.shape[0] != len(index)
                or df.shape[1] != len(columns)
                or not df.index.equals(index)
                or not df.columns.equals(columns)
            ):
                df = df.reindex(index=index, columns=columns)
            self.feat_arr[f] = np.ascontiguousarray(df.to_numpy(dtype=float_dtype, copy=False))

    def get_feature_matrix(
        self,
        feature_names: list[str],
        row_idx: np.ndarray,
        col_idx: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        used = [f for f in feature_names if f in self.feat_arr]
        if not used:
            return np.empty((len(row_idx), 0), dtype=self.float_dtype), []
        X = np.empty((len(row_idx), len(used)), dtype=self.float_dtype)
        for j, f in enumerate(used):
            X[:, j] = gather_candidate_values(self.feat_arr[f], row_idx, col_idx)
        return X, used

    def compute_feature_target_ic(
        self,
        feature_names: list[str],
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        y: np.ndarray,
        top_n: int = 20,
    ) -> float:
        if len(y) == 0:
            return 0.0
        y = np.asarray(y, dtype=np.float32)
        ics: list[float] = []
        n_ts = len(self.index)

        for feat_name in feature_names[: top_n * 2]:
            arr = self.feat_arr.get(feat_name)
            if arr is None:
                continue
            x = gather_candidate_values(arr, row_idx, col_idx)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 20:
                continue

            codes = row_idx[valid].astype(np.int32, copy=False)
            # Avoid casting to float64 for intermediate vectors to save memory,
            # trusting bincount to handle accumulation precision or float32 limits.
            xv = x[valid]
            yv = y[valid]

            cnt = np.bincount(codes, minlength=n_ts).astype(np.float64)
            sx = np.bincount(codes, weights=xv, minlength=n_ts)
            sy = np.bincount(codes, weights=yv, minlength=n_ts)
            sxx = np.bincount(codes, weights=xv * xv, minlength=n_ts)
            syy = np.bincount(codes, weights=yv * yv, minlength=n_ts)
            sxy = np.bincount(codes, weights=xv * yv, minlength=n_ts)

            ok = cnt >= 5.0
            if not np.any(ok):
                continue
            cnt_ok = cnt[ok]
            cov = sxy[ok] - (sx[ok] * sy[ok] / cnt_ok)
            varx = sxx[ok] - (sx[ok] * sx[ok] / cnt_ok)
            vary = syy[ok] - (sy[ok] * sy[ok] / cnt_ok)
            denom = np.sqrt(np.maximum(varx, 0.0) * np.maximum(vary, 0.0))
            corr = np.zeros_like(cov)
            nz = denom > 1e-12
            corr[nz] = cov[nz] / denom[nz]
            corr = corr[np.isfinite(corr)]
            if corr.size > 0:
                ics.append(float(np.mean(np.abs(corr))))

        if not ics:
            return 0.0
        ics_sorted = sorted(ics, reverse=True)[:top_n]
        return float(np.mean(ics_sorted))


def build_long_form_tables(
    feats: dict,
    target_col: str,
    float_dtype: np.dtype,
    atr_reference: Optional[pd.DataFrame] = None,
    tb_base: Optional[dict] = None,
) -> dict:
    """Build reusable long-form base table once; feature tables are materialized lazily."""
    def _norm_sym(s: Any) -> str:
        s = str(s).upper()
        return "".join(ch for ch in s if ch.isalnum())

    def _align_panel_cols_to_feature_cols(df: pd.DataFrame, target_cols: pd.Index) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame(index=ret_base.index, columns=target_cols, dtype=float)
        src_cols = list(df.columns)
        tgt_cols = list(target_cols)
        direct_hit = sum(1 for c in tgt_cols if c in df.columns)
        if direct_hit == len(tgt_cols):
            return df.reindex(index=ret_base.index, columns=target_cols)
        norm_to_src: dict[str, str] = {}
        dup_norm: set[str] = set()
        for c in src_cols:
            k = _norm_sym(c)
            if k in norm_to_src and norm_to_src[k] != c:
                dup_norm.add(k)
            else:
                norm_to_src[k] = c
        for k in dup_norm:
            norm_to_src.pop(k, None)
        rename_map: dict[Any, Any] = {}
        for c in src_cols:
            k = _norm_sym(c)
            tgt = next((t for t in tgt_cols if _norm_sym(t) == k), None)
            if tgt is not None:
                rename_map[c] = tgt
        aligned = df.rename(columns=rename_map)
        if aligned.columns.has_duplicates:
            aligned = aligned.T.groupby(level=0).last().T
        return aligned.reindex(index=ret_base.index, columns=target_cols)

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

    if (
        isinstance(tb_base, dict)
        and isinstance(tb_base.get("ret_long"), pd.DataFrame)
        and isinstance(tb_base.get("ret_short"), pd.DataFrame)
        and isinstance(tb_base.get("label_long"), pd.DataFrame)
        and isinstance(tb_base.get("label_short"), pd.DataFrame)
    ):
        # Legacy fallback targets/labels (feature-space aligned) used when
        # aggregated geometry coverage is partial due symbol naming/schema gaps.
        valid_counts = target.notna().sum(axis=1)
        threshold = target.quantile(0.65, axis=1)
        legacy_label_long_df = target.gt(threshold, axis=0).astype(np.float32)
        legacy_label_long_df = legacy_label_long_df.where(valid_counts >= 10, np.nan)
        legacy_label_short_df = (1.0 - legacy_label_long_df).astype(np.float32)

        ret_long = _align_panel_cols_to_feature_cols(tb_base["ret_long"], ret_base.columns)
        ret_short = _align_panel_cols_to_feature_cols(tb_base["ret_short"], ret_base.columns)
        label_long = _align_panel_cols_to_feature_cols(tb_base["label_long"], ret_base.columns)
        label_short = _align_panel_cols_to_feature_cols(tb_base["label_short"], ret_base.columns)

        ret_base_s = ret_base.stack(future_stack=True).astype(float_dtype, copy=False)
        ret_long_s = ret_long.stack(future_stack=True).astype(float_dtype, copy=False)
        ret_short_s = ret_short.stack(future_stack=True).astype(float_dtype, copy=False)
        label_long_s = (label_long == 1).stack(future_stack=True).astype(np.float32, copy=False)
        label_short_s = (label_short == 1).stack(future_stack=True).astype(np.float32, copy=False)

        legacy_target_long_s = target.stack(future_stack=True).astype(float_dtype, copy=False)
        legacy_target_short_s = (-target).stack(future_stack=True).astype(float_dtype, copy=False)
        legacy_label_long_s = legacy_label_long_df.stack(future_stack=True).astype(np.float32, copy=False)
        legacy_label_short_s = legacy_label_short_df.stack(future_stack=True).astype(np.float32, copy=False)

        cov_ret_long = float(np.isfinite(ret_long_s.to_numpy(dtype=np.float32, copy=False)).mean())
        cov_ret_short = float(np.isfinite(ret_short_s.to_numpy(dtype=np.float32, copy=False)).mean())
        cov_lbl_long = float(np.isfinite(label_long_s.to_numpy(dtype=np.float32, copy=False)).mean())
        cov_lbl_short = float(np.isfinite(label_short_s.to_numpy(dtype=np.float32, copy=False)).mean())
        tlog(
            "Aggregated base coverage before fallback: "
            f"ret_long={cov_ret_long:.2%}, ret_short={cov_ret_short:.2%}, "
            f"label_long={cov_lbl_long:.2%}, label_short={cov_lbl_short:.2%}"
        )

        # Fill missing aggregated values with side-consistent legacy proxies.
        ret_long_s = ret_long_s.where(np.isfinite(ret_long_s), legacy_target_long_s).astype(float_dtype, copy=False)
        ret_short_s = ret_short_s.where(np.isfinite(ret_short_s), legacy_target_short_s).astype(float_dtype, copy=False)
        label_long_s = label_long_s.where(np.isfinite(label_long_s), legacy_label_long_s).astype(np.float32, copy=False)
        label_short_s = label_short_s.where(np.isfinite(label_short_s), legacy_label_short_s).astype(np.float32, copy=False)

        cov_ret_long_after = float(np.isfinite(ret_long_s.to_numpy(dtype=np.float32, copy=False)).mean())
        cov_ret_short_after = float(np.isfinite(ret_short_s.to_numpy(dtype=np.float32, copy=False)).mean())
        cov_lbl_long_after = float(np.isfinite(label_long_s.to_numpy(dtype=np.float32, copy=False)).mean())
        cov_lbl_short_after = float(np.isfinite(label_short_s.to_numpy(dtype=np.float32, copy=False)).mean())
        tlog(
            "Aggregated base coverage after fallback: "
            f"ret_long={cov_ret_long_after:.2%}, ret_short={cov_ret_short_after:.2%}, "
            f"label_long={cov_lbl_long_after:.2%}, label_short={cov_lbl_short_after:.2%}"
        )

        base_long = pd.DataFrame(
            {
                "ret_base": ret_base_s,
                "ret_long": ret_long_s,
                "ret_short": ret_short_s,
                "target_long": ret_long_s,
                "target_short": ret_short_s,
                "label_long": label_long_s,
                "label_short": label_short_s,
                "atr_pct": atr_pct.stack(future_stack=True).astype(float_dtype, copy=False),
            }
        )
        # Keep legacy aliases for compatibility (defaults to long side).
        base_long["target"] = base_long["target_long"]
        base_long["label"] = base_long["label_long"]
    else:
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


def estimate_pooled_ic_std(
    oof_values: np.ndarray,
    y_values: np.ndarray,
    ts_values: np.ndarray,
    min_group_size: int = 20,
) -> tuple[float, str, int, int, int]:
    """Estimate IC uncertainty from pooled chunks when per-timestamp IC is sparse."""
    # Ensure inputs are handled efficiently, respecting float32 preference
    valid = np.isfinite(oof_values) & np.isfinite(y_values)
    if valid.sum() < max(100, min_group_size * 3):
        return 0.0, "insufficient", 0, len(oof_values), 0

    oof = oof_values[valid]
    y = y_values[valid]
    ts = ts_values[valid]

    # Vectorized timestamp encoding
    ts_codes, unique_ts = pd.factorize(ts, sort=True)
    n_ts = len(unique_ts)

    # Use bincount for vectorized per-timestamp correlation
    counts = np.bincount(ts_codes, minlength=n_ts)
    group_mask = counts >= 2

    # Check if we have enough groups for reliable statistics
    if group_mask.sum() >= 10:
        # Fully vectorized correlation calculation
        sum_x = np.bincount(ts_codes, weights=oof, minlength=n_ts)
        sum_y = np.bincount(ts_codes, weights=y, minlength=n_ts)
        # float32 squared can overflow/lose precision, but respecting 32-bit request
        sum_xx = np.bincount(ts_codes, weights=oof * oof, minlength=n_ts)
        sum_yy = np.bincount(ts_codes, weights=y * y, minlength=n_ts)
        sum_xy = np.bincount(ts_codes, weights=oof * y, minlength=n_ts)

        n = counts[group_mask]
        sx = sum_x[group_mask]
        sy = sum_y[group_mask]
        sxx = sum_xx[group_mask]
        syy = sum_yy[group_mask]
        sxy = sum_xy[group_mask]

        numer = n * sxy - sx * sy
        denom_x = n * sxx - sx * sx
        denom_y = n * syy - sy * sy
        denom_sq = denom_x * denom_y

        # Floating point safety
        valid_denom = denom_sq > 1e-12
        if valid_denom.sum() >= 10:
            numer = numer[valid_denom]
            denom_sq = denom_sq[valid_denom]
            denom = np.sqrt(denom_sq)
            corrs = numer / denom
            return float(np.std(corrs)), "per_timestamp", len(corrs), len(oof), 0

    # Fallback to pooled chunks if per-timestamp is sparse
    # Create sorted dataframe only for fallback path
    df = pd.DataFrame({"oof": oof, "y": y, "ts": ts}).sort_values("ts")
    n_groups = max(5, min(30, len(df) // max(min_group_size, 1)))
    if n_groups < 2:
        return 0.0, "insufficient", 0, len(df), 0

    group_ids = np.floor(np.linspace(0, n_groups, len(df), endpoint=False)).astype(np.int32)
    pooled_ics = []
    # Small loop over chunks (max 30 iterations)
    for g in range(n_groups):
        chunk = df[group_ids == g]
        if len(chunk) >= min_group_size:
            chunk_ic = safe_pearson_corr(
                chunk["oof"].to_numpy(dtype=np.float32, copy=False),
                chunk["y"].to_numpy(dtype=np.float32, copy=False),
            )
            if np.isfinite(chunk_ic):
                pooled_ics.append(chunk_ic)

    return (
        float(np.std(pooled_ics)) if len(pooled_ics) >= 2 else 0.0,
        "pooled_chunked",
        0,
        len(df),
        len(pooled_ics),
    )


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


def _scores_to_pseudo_proba(scores: np.ndarray) -> np.ndarray:
    """Map arbitrary model scores into (0,1) via empirical rank transform."""
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    if n == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    p = np.empty(n, dtype=np.float64)
    p[order] = (np.arange(n, dtype=np.float64) + 0.5) / n
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def compute_feature_target_correlation(
    data_container: DataContainer,
    available_features: list,
    candidate_row_idx: np.ndarray,
    candidate_col_idx: np.ndarray,
    target_values: np.ndarray,
    top_n: int = 20,
) -> float:
    """Compute mean |IC| of top features with target via vectorized bincount aggregation."""
    if not available_features:
        return 0.0
    return data_container.compute_feature_target_ic(
        feature_names=available_features,
        row_idx=candidate_row_idx,
        col_idx=candidate_col_idx,
        y=target_values,
        top_n=top_n,
    )


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
    data_container: DataContainer,
    available_features: list,
    candidate_row_idx: np.ndarray,
    candidate_col_idx: np.ndarray,
    float_dtype: np.dtype,
) -> tuple[np.ndarray, list[str]]:
    """Materialize candidate feature matrix directly from cached arrays."""
    X, used = data_container.get_feature_matrix(
        feature_names=available_features,
        row_idx=candidate_row_idx,
        col_idx=candidate_col_idx,
    )
    if X.dtype != float_dtype:
        X = X.astype(float_dtype, copy=False)
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
    ts_values: Optional[np.ndarray] = None,
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
    
    pkf = PurgedKFold(n_splits=n_splits, purge=purge, embargo=2)
    if ts_values is not None and len(ts_values) == n_samples:
        ts_codes, unique_ts = pd.factorize(pd.Series(ts_values), sort=True)
        ts_codes = ts_codes.astype(np.int32, copy=False)
        time_idx = np.arange(len(unique_ts), dtype=np.int32)
        ts_splits = list(pkf.split(time_idx))
        splits = []
        for train_ts_idx, val_ts_idx in ts_splits:
            train_mask = np.isin(ts_codes, train_ts_idx)
            val_mask = np.isin(ts_codes, val_ts_idx)
            train_idx = np.flatnonzero(train_mask)
            val_idx = np.flatnonzero(val_mask)
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
    else:
        # Fallback to row-order splitting when timestamp groups are unavailable.
        time_idx = np.arange(n_samples, dtype=np.int32)
        splits = list(pkf.split(time_idx))
    if not splits:
        return oof, {
            "ridge_alpha": float(ridge_alpha),
            "ridge_top_frac": float(ridge_top_frac),
            "ridge_selected_k_mean": 0.0,
            "ridge_jaccard_median": 0.0,
            "ridge_replacement_rate_median": 0.0,
        }
    selected_sets: list[set[int]] = []
    selected_ks: list[int] = []
    model_name = "ExtraTrees" if use_extratrees else "Ridge"

    for fold_i, (train_idx, val_idx) in enumerate(splits, start=1):
        tlog(
            f"OOF fold {fold_i}/{len(splits)} [{model_name}]: "
            f"train={len(train_idx)}, val={len(val_idx)}"
        )
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
        tlog(f"OOF fold {fold_i}: Ridge-screen selected {len(selected_idx)} features")

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


def run_oof_cv_classifier(
    X: np.ndarray,
    y_cls: np.ndarray,
    ts_values: Optional[np.ndarray] = None,
    sample_weights: np.ndarray = None,
    n_splits: int = 3,
    purge: int = 12,
    random_state: int = 42,
    ridge_alpha: float = RIDGE_SCREEN_ALPHA,
    ridge_top_frac: float = RIDGE_SCREEN_TOP_FRAC,
) -> np.ndarray:
    """Run purged OOF CV for binary classification with ExtraTreesClassifier."""
    n_samples = len(y_cls)
    oof_proba = np.full(n_samples, np.nan, dtype=np.float32)

    pkf = PurgedKFold(n_splits=n_splits, purge=purge, embargo=2)
    if ts_values is not None and len(ts_values) == n_samples:
        ts_codes, unique_ts = pd.factorize(pd.Series(ts_values), sort=True)
        ts_codes = ts_codes.astype(np.int32, copy=False)
        time_idx = np.arange(len(unique_ts), dtype=np.int32)
        ts_splits = list(pkf.split(time_idx))
        splits = []
        for train_ts_idx, val_ts_idx in ts_splits:
            train_mask = np.isin(ts_codes, train_ts_idx)
            val_mask = np.isin(ts_codes, val_ts_idx)
            train_idx = np.flatnonzero(train_mask)
            val_idx = np.flatnonzero(val_mask)
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
    else:
        time_idx = np.arange(n_samples, dtype=np.int32)
        splits = list(pkf.split(time_idx))
    if not splits:
        return oof_proba

    for fold_i, (train_idx, val_idx) in enumerate(splits, start=1):
        tlog(f"OOF-CLF fold {fold_i}/{len(splits)} [ExtraTreesClassifier]: train={len(train_idx)}, val={len(val_idx)}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y_cls[train_idx]
        sw_train = sample_weights[train_idx] if sample_weights is not None else None

        selected_idx = _ridge_select_topk(
            X_train=X_train,
            y_train=y_train.astype(np.float32, copy=False),
            sample_weight=sw_train,
            top_frac=ridge_top_frac,
            alpha=ridge_alpha,
        )
        X_train_sel = X_train[:, selected_idx]
        X_val_sel = X_val[:, selected_idx]

        clf = ExtraTreesClassifier(**{**ET_REGRESSOR_PARAMS, "random_state": random_state})
        clf.fit(X_train_sel, y_train, sample_weight=sw_train)
        proba = clf.predict_proba(X_val_sel)
        oof_proba[val_idx] = proba[:, 1].astype(np.float32, copy=False)

        del X_train, X_val, X_train_sel, X_val_sel, y_train, sw_train, clf
        gc.collect()

    return oof_proba


def compute_learnability_metrics(
    candidate_mask: pd.DataFrame,
    precomputed: dict,
    available_features: list,
    float_dtype: np.dtype,
    side_sign: Optional[pd.DataFrame] = None,
    use_extratrees: bool = False,
    oof_max_samples: int = OOF_MAX_SAMPLES,
    oof_n_splits: int = STAGE23_OOF_SPLITS,
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

    if side_sign is None:
        sign_arr = np.ones_like(mask_arr, dtype=np.int8)
    else:
        sign_arr = side_sign.to_numpy(dtype=np.int8, copy=False)
    side_sign_flat = sign_arr.ravel(order="C")

    if len(flat_idx) == 0:
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
            "auc": 0,
            "brier": 0,
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
            "atr_decile_worst_share": np.nan,
            "atr_decile_pnl_json": "{}",
        }

    use_geom_side = {"target_long", "target_short", "label_long", "label_short", "ret_long", "ret_short"}.issubset(base_long.columns)
    candidate_returns_raw = base_long["ret_base"].to_numpy(dtype=float_dtype, copy=False)[flat_idx]
    candidate_signs = side_sign_flat[flat_idx].astype(float_dtype, copy=False)
    if use_geom_side:
        target_long = base_long["target_long"].to_numpy(dtype=float_dtype, copy=False)[flat_idx]
        target_short = base_long["target_short"].to_numpy(dtype=float_dtype, copy=False)[flat_idx]
        ret_long = base_long["ret_long"].to_numpy(dtype=float_dtype, copy=False)[flat_idx]
        ret_short = base_long["ret_short"].to_numpy(dtype=float_dtype, copy=False)[flat_idx]
        label_long = base_long["label_long"].to_numpy(dtype=np.float32, copy=False)[flat_idx]
        label_short = base_long["label_short"].to_numpy(dtype=np.float32, copy=False)[flat_idx]
        use_long = candidate_signs >= 0
        candidate_target_full = np.where(use_long, target_long, target_short).astype(float_dtype, copy=False)
        candidate_returns_full = np.where(use_long, ret_long, ret_short).astype(float_dtype, copy=False)
        candidate_labels_full = np.where(use_long, label_long, label_short).astype(np.float32, copy=False)
    else:
        candidate_target_full = base_long["target"].to_numpy(dtype=float_dtype, copy=False)[flat_idx]
        candidate_returns_full = candidate_returns_raw
        candidate_labels_full = base_long["label"].to_numpy(dtype=np.float32, copy=False)[flat_idx]

    # Label robustness: if barrier labels are degenerate or missing, fallback to
    # direction-of-return proxy to keep class metrics meaningful.
    lbl_valid = np.isfinite(candidate_labels_full)
    pos_rate_raw = float(np.mean(candidate_labels_full[lbl_valid] > 0.5)) if np.any(lbl_valid) else float("nan")
    if (not np.any(lbl_valid)) or (np.isfinite(pos_rate_raw) and (pos_rate_raw < 0.01 or pos_rate_raw > 0.99)):
        proxy_labels = (candidate_returns_full > 0).astype(np.float32, copy=False)
        if np.any(lbl_valid):
            candidate_labels_full = np.where(lbl_valid, candidate_labels_full, proxy_labels).astype(np.float32, copy=False)
        else:
            candidate_labels_full = proxy_labels.astype(np.float32, copy=False)
        # If original labels are almost single-class, fully switch to proxy.
        if np.isfinite(pos_rate_raw) and (pos_rate_raw < 0.01 or pos_rate_raw > 0.99):
            candidate_labels_full = proxy_labels.astype(np.float32, copy=False)
        pos_rate_new = float(np.mean(candidate_labels_full > 0.5)) if len(candidate_labels_full) > 0 else 0.0
        tlog(
            "Metrics labels fallback applied: "
            f"orig_pos_rate={pos_rate_raw if np.isfinite(pos_rate_raw) else float('nan'):.4f}, "
            f"new_pos_rate={pos_rate_new:.4f}"
        )

    valid_mask = np.isfinite(candidate_returns_full) & np.isfinite(candidate_target_full)
    candidate_returns_raw = candidate_returns_full[valid_mask]
    candidate_target = candidate_target_full[valid_mask]
    candidate_signs = candidate_signs[valid_mask]
    candidate_labels = candidate_labels_full[valid_mask]
    candidate_signs = np.where(candidate_signs == 0, 1, candidate_signs)
    # Keep return metrics unbiased with respect to the oracle side assignment used
    # for candidate construction. Directional trading quality is evaluated via OOF.
    candidate_returns = candidate_returns_raw
    candidate_ts = np.asarray(candidate_mask.index.to_numpy()[candidate_row_idx])[valid_mask]
    
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
            "auc": 0,
            "brier": 0,
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
            "atr_decile_worst_share": np.nan,
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
    data_container: DataContainer = precomputed["data_container"]
    mean_feat_ic = compute_feature_target_correlation(
        data_container=data_container,
        available_features=available_features,
        candidate_row_idx=candidate_row_idx,
        candidate_col_idx=candidate_col_idx,
        target_values=(candidate_target_full.astype(np.float32, copy=False)),
        top_n=20,
    )
    
    # 7. Sharpe/Sortino on timestamp-aggregated (hourly) returns.
    hourly_returns = pd.Series(candidate_returns, index=pd.Index(candidate_ts)).groupby(level=0).mean()
    hr = hourly_returns.to_numpy(dtype=np.float64, copy=False)
    if len(hr) > 1 and np.std(hr) > 1e-12:
        sharpe = float(np.mean(hr) / np.std(hr) * np.sqrt(24.0 * 365.0))
    else:
        sharpe = 0.0

    mean_return_bps = float(np.mean(candidate_returns) * 1e4) if len(candidate_returns) > 0 else 0.0
    volatility_bps = float(np.std(candidate_returns) * 1e4) if len(candidate_returns) > 0 else 0.0
    downside = hr[hr < 0]
    if len(downside) > 1 and np.std(downside) > 1e-12:
        sortino = float(np.mean(hr) / np.std(downside) * np.sqrt(24.0 * 365.0))
    else:
        sortino = 0.0
    hit_rate = float(np.mean(candidate_returns > 0)) if len(candidate_returns) > 0 else 0.0
    p95 = np.nanpercentile(candidate_returns, 95) if len(candidate_returns) > 0 else 0.0
    p05 = np.nanpercentile(candidate_returns, 5) if len(candidate_returns) > 0 else 0.0
    tail_ratio = float(p95 / abs(p05)) if abs(p05) > 1e-12 else 0.0

    atr_selected = base_long["atr_pct"].to_numpy(dtype=float_dtype, copy=False)[flat_idx]
    atr_selected = atr_selected[valid_mask]
    atr_selected = atr_selected[np.isfinite(atr_selected)]
    atr_mean = float(np.mean(atr_selected)) if len(atr_selected) > 0 else 0.0
    atr_q10 = float(np.nanpercentile(atr_selected, 10)) if len(atr_selected) > 0 else 0.0
    atr_q50 = float(np.nanpercentile(atr_selected, 50)) if len(atr_selected) > 0 else 0.0
    atr_q90 = float(np.nanpercentile(atr_selected, 90)) if len(atr_selected) > 0 else 0.0

    atr_decile_worst = -1
    atr_decile_worst_share = 0.0
    atr_decile_pnl_json = "{}"
    atr_raw = base_long["atr_pct"].to_numpy(dtype=float_dtype, copy=False)[flat_idx]
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
    
    auc = 0.0
    brier = 0.0
    ic_oof_n = 0
    auc_oof_n = 0
    brier_oof_n = 0
    ic_source = "none"
    auc_source = "none"
    brier_source = "none"

    tlog("Metrics: entering OOF/IC block")
    # 8. Information Coefficient (requires OOF predictions)
    if len(available_features) >= 10:
        X_all, used_features = materialize_feature_matrix(
            data_container=data_container,
            available_features=available_features,
            candidate_row_idx=candidate_row_idx,
            candidate_col_idx=candidate_col_idx,
            float_dtype=float_dtype,
        )
        y_arr = candidate_target_full.astype(float_dtype, copy=False)
        if X_all.shape[1] >= 10 and len(used_features) >= 10:
            valid_rows = np.isfinite(y_arr) & np.isfinite(X_all).all(axis=1)
        else:
            valid_rows = np.zeros(len(y_arr), dtype=bool)

        if valid_rows.sum() >= 100:
            X = X_all[valid_rows]
            y = y_arr[valid_rows]
            ts_vals = np.asarray(candidate_mask.index.to_numpy()[candidate_row_idx])[valid_rows]

            if len(y) >= 100:
                if len(y) > int(oof_max_samples):
                    n_before_oof = len(y)
                    sub_idx = _uniform_subsample_idx(len(y), int(oof_max_samples), seed=42)
                    X = X[sub_idx]
                    y = y[sub_idx]
                    ts_vals = ts_vals[sub_idx]
                    tlog(f"Metrics: OOF downsample applied {len(y)}/{n_before_oof} rows")
                tlog(f"Metrics: running OOF CV on {X.shape[0]}x{X.shape[1]}")
                oof, oof_diag = run_oof_cv(
                    X,
                    y,
                    ts_values=np.asarray(ts_vals),
                    n_splits=int(oof_n_splits),
                    float_dtype=float_dtype,
                    ridge_alpha=RIDGE_SCREEN_ALPHA,
                    ridge_top_frac=RIDGE_SCREEN_TOP_FRAC,
                    use_extratrees=use_extratrees,
                )

                oof_valid = np.isfinite(oof)
                if oof_valid.sum() >= 50:
                    ic_oof_n = int(oof_valid.sum())
                    ic_source = "oof_regression"
                    ic = safe_pearson_corr(oof[oof_valid], y[oof_valid])
                    ic_spearman = stats.spearmanr(oof[oof_valid], y[oof_valid], nan_policy="omit").statistic
                    oof_mae = float(np.mean(np.abs(oof[oof_valid] - y[oof_valid])))
                    oof_directional_acc = float(np.mean(np.sign(oof[oof_valid]) == np.sign(y[oof_valid])))

                    ic_std, ic_std_mode, ic_ts_n, ic_n, ic_chunks_n = estimate_pooled_ic_std(
                        oof_values=oof[oof_valid],
                        y_values=y[oof_valid],
                        ts_values=np.asarray(ts_vals[oof_valid]),
                    )
                    tlog(
                        "IC uncertainty estimate: "
                        f"mode={ic_std_mode}, n_candidates={ic_n}, n_timestamps={ic_ts_n}, n_chunks={ic_chunks_n}, "
                        f"ic_std={ic_std:.6f}"
                    )
                else:
                    ic, ic_std, ic_spearman, oof_mae, oof_directional_acc = 0, 0, 0, 0, 0

                # Always compute classification diagnostics from OOF regression scores.
                y_cls_raw = candidate_labels_full[valid_rows]
                y_cls = (y_cls_raw > 0.5).astype(np.int8, copy=False)
                if len(y_cls) >= 100 and y_cls.sum() > 0 and y_cls.sum() < len(y_cls):
                    y_cls_valid = y_cls[oof_valid].astype(np.int32, copy=False)
                    score_valid = oof[oof_valid].astype(np.float64, copy=False)
                    if len(y_cls_valid) >= 50 and y_cls_valid.sum() > 0 and y_cls_valid.sum() < len(y_cls_valid):
                        try:
                            auc = float(roc_auc_score(y_cls_valid, score_valid))
                            auc_oof_n = int(len(y_cls_valid))
                            auc_source = "oof_regression_scores"
                        except Exception:
                            auc = 0.0
                        p_rank = _scores_to_pseudo_proba(score_valid)
                        if len(p_rank) == len(y_cls_valid):
                            brier = float(brier_score_loss(y_cls_valid, p_rank))
                            brier_oof_n = int(len(y_cls_valid))
                            brier_source = "oof_regression_rank_proba"

                # If classifier OOF is enabled, override with calibrated probability metrics.
                if use_extratrees:
                    y_cls_raw = candidate_labels_full[valid_rows]
                    y_cls = (y_cls_raw > 0.5).astype(np.int8, copy=False)
                    if len(y_cls) >= 100 and y_cls.sum() > 0 and y_cls.sum() < len(y_cls):
                        oof_proba = run_oof_cv_classifier(
                            X,
                            y_cls,
                            ts_values=np.asarray(ts_vals),
                            n_splits=int(oof_n_splits),
                            ridge_alpha=RIDGE_SCREEN_ALPHA,
                            ridge_top_frac=RIDGE_SCREEN_TOP_FRAC,
                        )
                        proba_valid = np.isfinite(oof_proba)
                        if proba_valid.sum() >= 50:
                            y_cls_valid = y_cls[proba_valid].astype(np.int32, copy=False)
                            p_valid = np.clip(oof_proba[proba_valid], 1e-6, 1 - 1e-6)
                            if y_cls_valid.sum() > 0 and y_cls_valid.sum() < len(y_cls_valid):
                                auc = float(roc_auc_score(y_cls_valid, p_valid))
                                brier = float(brier_score_loss(y_cls_valid, p_valid))
                                auc_oof_n = int(len(y_cls_valid))
                                brier_oof_n = int(len(y_cls_valid))
                                auc_source = "oof_classifier_proba"
                                brier_source = "oof_classifier_proba"
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
        "ic_source": ic_source,
        "ic_oof_n": int(ic_oof_n),
        "auc_source": auc_source,
        "auc_oof_n": int(auc_oof_n),
        "brier_source": brier_source,
        "brier_oof_n": int(brier_oof_n),
        "auc": auc if np.isfinite(auc) else 0,
        "brier": brier if np.isfinite(brier) else 0,
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


def load_selected_features_from_symbol_parquets(
    feature_dir: str,
    wanted_features: set[str],
    float_dtype: np.dtype,
) -> dict:
    """Load only selected columns from pipeline symbol parquet files."""
    files = sorted(glob.glob(os.path.join(feature_dir, "symbol=*.parquet")))
    if not files:
        return {}

    # Discover available columns once to avoid per-file schema scans.
    sample_cols = set(pd.read_parquet(files[0]).columns)
    select_cols = [c for c in sorted(wanted_features) if c in sample_cols]
    if not select_cols:
        tlog("Selective symbol-parquet loader: no requested columns found")
        return {}

    feat_buffers: dict[str, dict[str, pd.Series]] = {k: {} for k in select_cols}
    total_files = len(files)
    progress_every = 25 if total_files >= 100 else 10
    start = pd.Timestamp.utcnow().timestamp()

    for i, fpath in enumerate(files, start=1):
        try:
            fname = os.path.basename(fpath)
            sym = fname.replace("symbol=", "").replace(".parquet", "")
            read_cols = list(select_cols)
            # Keep original symbol if present.
            if "__symbol__" in sample_cols:
                read_cols.append("__symbol__")
            df = pd.read_parquet(fpath, columns=read_cols)
            if "__symbol__" in df.columns and not df.empty:
                real_sym = str(df["__symbol__"].iloc[0])
                df = df.drop(columns=["__symbol__"], errors="ignore")
            else:
                real_sym = sym.replace("_", "/", 1)
            for k in select_cols:
                if k in df.columns:
                    feat_buffers[k][real_sym] = pd.to_numeric(df[k], errors="coerce").astype(float_dtype, copy=False)
            del df
            if i % progress_every == 0 or i == total_files:
                elapsed = pd.Timestamp.utcnow().timestamp() - start
                tlog(
                    f"Selective feature load progress: {i}/{total_files} files "
                    f"({(i / total_files) * 100:.1f}%) in {elapsed:.1f}s"
                )
        except Exception as exc:
            logger.warning(f"Selective feature load failed for {fpath}: {exc}")

    feats_out: dict[str, pd.DataFrame] = {}
    for k, sym_map in feat_buffers.items():
        if sym_map:
            feats_out[k] = pd.DataFrame(sym_map).sort_index()
    feat_buffers.clear()
    gc.collect()
    tlog(f"Selective loader built {len(feats_out)} feature matrices")
    return feats_out


def load_panel_data(
    panel_path: str,
    symbols: Optional[list[str]] = None,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load panel data lazily with projection/predicate pushdown when possible."""
    def _symbol_values_for_filter(requested_symbols: Optional[list[str]]) -> Optional[list[str]]:
        if not requested_symbols:
            return None
        vals: set[str] = set()
        for s in requested_symbols:
            s_str = str(s)
            vals.add(s_str)
            vals.add(s_str.replace("/", "_"))
        return sorted(vals)

    def _list_parquet_files(root_path: str) -> list[str]:
        parquet_files: list[str] = []
        for root, _, files in os.walk(root_path):
            for fname in files:
                if fname.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, fname))
        return parquet_files

    def _resolve_projection(requested: Optional[list[str]], available: list[str]) -> list[str]:
        if requested is None:
            return list(available)
        available_set = set(available)
        projected = [c for c in requested if c in available_set]
        # Support common timestamp aliasing in panel stores.
        if "timestamp" in requested and "timestamp" not in available_set and "ts" in available_set:
            projected.append("ts")
        # De-duplicate while preserving order (avoids duplicate-key DataFrames like ['ts', ..., 'ts']).
        deduped: list[str] = []
        seen: set[str] = set()
        for col in projected:
            if col not in seen:
                deduped.append(col)
                seen.add(col)
        return deduped

    symbol_filter_values = _symbol_values_for_filter(symbols)

    if os.path.isfile(panel_path):
        try:
            df = pd.read_parquet(panel_path, columns=columns)
        except Exception:
            df = pd.read_parquet(panel_path)
            if columns is not None:
                keep = [c for c in columns if c in df.columns]
                if keep:
                    df = df[keep]
        if symbol_filter_values is not None and "symbol" in df.columns:
            df = df[df["symbol"].astype(str).isin(symbol_filter_values)]
        return df
    elif os.path.isdir(panel_path):
        parquet_files = _list_parquet_files(panel_path)
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under panel path: {panel_path}")
        try:
            import pyarrow.dataset as ds
            import pyarrow.compute as pc

            dataset = ds.dataset(parquet_files, format="parquet", partitioning="hive")
            wanted_cols = _resolve_projection(columns, list(dataset.schema.names))
            filt = None
            if symbol_filter_values:
                if "symbol" in dataset.schema.names:
                    filt = pc.field("symbol").isin(symbol_filter_values)
            table = dataset.to_table(columns=wanted_cols, filter=filt)
            return table.to_pandas()
        except Exception as exc:
            logger.warning(f"PyArrow lazy panel load failed ({exc}); falling back to batch parquet reads")
            dfs = []
            for fpath in parquet_files:
                root = os.path.dirname(fpath)
                try:
                    df = pd.read_parquet(fpath, columns=columns)
                except Exception:
                    df = pd.read_parquet(fpath)
                    keep = _resolve_projection(columns, list(df.columns))
                    if keep:
                        df = df[keep]
                if "symbol" not in df.columns:
                    sym = None
                    for p in root.split(os.sep):
                        if p.startswith("symbol="):
                            sym = p.replace("symbol=", "")
                            break
                    if sym is not None:
                        df["symbol"] = sym
                if symbol_filter_values is not None and "symbol" in df.columns:
                    df = df[df["symbol"].astype(str).isin(symbol_filter_values)]
                if len(df) > 0:
                    dfs.append(df)
            if dfs:
                return pd.concat(dfs, ignore_index=True)

    raise FileNotFoundError(f"Panel path not found: {panel_path}")


def load_panel_from_store(
    cfg: dict,
    symbols: Optional[list[str]] = None,
) -> dict:
    """Load panel directly from partitioned OHLCV store (pipeline-compatible)."""
    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg.get("timeframe", "1h"))
    chosen_syms = [str(s) for s in (symbols or [])]
    if not chosen_syms:
        ohlcv_dir = os.path.join(cfg["data_root"], "ohlcv")
        if os.path.isdir(ohlcv_dir):
            chosen_syms = sorted(
                {
                    d.replace("symbol=", "").replace("_", "/", 1)
                    for d in os.listdir(ohlcv_dir)
                    if d.startswith("symbol=") and os.path.isdir(os.path.join(ohlcv_dir, d))
                }
            )
    dfs: dict[str, pd.DataFrame] = {}
    for sym in chosen_syms:
        try:
            df = store.load(sym)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        dfs[sym] = df
    if not dfs:
        raise ValueError(f"No OHLCV data found in store under {cfg['data_root']}")
    return to_panel(dfs)


# =============================================================================
# Main Comparison Runner
# =============================================================================

def run_comparison(
    feature_path: Optional[str],
    panel_path: Optional[str],
    output_path: str,
    dtype: str = "float32",
    max_features: Optional[int] = None,
    stage3: bool = False,
    winners: list = None,
    use_extratrees: bool = False,  # Default: Ridge only for Stage 1 & 2
    symbol_step: int = 3,
    symbol_limit: int = 0,
    save_sample_weights: bool = False,
    tail_filter_mode: str = "auto_compare",
    tail_filter_top_frac: float = 0.20,
    runtime_cfg: Optional[dict] = None,
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
    cleanup_run_caches()
    tlog("Starting comparison run")

    runtime_cfg = deepcopy(runtime_cfg) if runtime_cfg is not None else apply_offline_optimizer_best_params(deepcopy(CFG))
    
    # Load data - try pipeline format first, then fallback to generic loader
    if not feature_path:
        raise ValueError("feature_path is required (pass --features or resolve via --data-root)")
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
    tlog(f"OOF estimator for this run: {'ExtraTrees' if use_extratrees else 'Ridge'}")

    if symbol_files:
        # Pipeline format: per-symbol files, need to parse timestamp from path
        logger.info("Detected pipeline per-symbol format")
        requested_model_features = runtime_cfg.get("test_feature_keys", TEST_FEATURE_KEYS)
        if max_features is not None and max_features > 0:
            requested_model_features = requested_model_features[:max_features]
        requested_model_features = [str(f) for f in requested_model_features]
        structural_keys = {
            "ret6h",
            "ret24h",
            "ret1h",
            "atr_pct",
            "trend_pct",
            "range_12h_pct",
            "range_16h_pct",
            "range_pct",
            "volatility_zscore",
            "vol_z",
            "rvol_z",
            "volu_z",
            "sign_consistency",
            "sign_consistency_12h",
        }
        wanted_features = set(requested_model_features) | structural_keys
        tlog(
            "Selective pipeline load: "
            f"requested_model={len(requested_model_features)}, structural={len(structural_keys)}"
        )
        feats = load_selected_features_from_symbol_parquets(
            feature_dir=feature_path,
            wanted_features=wanted_features,
            float_dtype=float_dtype,
        )
        if not feats:
            raise ValueError(f"Failed selective feature load from {feature_path}")
        logger.info(f"Loaded {len(feats)} selected features via pipeline loader")
        gc.collect()
    else:
        # Generic format: per-feature files
        tlog("Loading generic feature parquet layout")
        feats = load_features_from_parquet(feature_path)
        feats = cast_features_dtype(feats, float_dtype=float_dtype)
        gc.collect()

    # Keep training/event-scoring logic healthy even when range_16h_pct is absent.
    if "range_16h_pct" not in feats:
        if "range_12h_pct" in feats:
            feats["range_16h_pct"] = feats["range_12h_pct"]
            tlog("Derived missing range_16h_pct from range_12h_pct")
        elif "range_pct" in feats:
            feats["range_16h_pct"] = feats["range_pct"]
            tlog("Derived missing range_16h_pct from range_pct")

    # Normalize feature timestamps to UTC-naive once to prevent downstream
    # tz-aware vs tz-naive alignment drops in training-slice label joins.
    feats = normalize_feature_time_indices(feats)
    
    # Check required features
    if "ret6h" not in feats and "ret24h" not in feats:
        logger.error("Required feature 'ret6h' (or fallback 'ret24h') not found in data")
        return

    # Symbol universe selection defaults to every 3rd symbol in alphabetical order.
    base_ret_df = feats.get("ret6h") if isinstance(feats.get("ret6h"), pd.DataFrame) else feats.get("ret24h")
    if not isinstance(base_ret_df, pd.DataFrame):
        raise ValueError("ret6h/ret24h must be DataFrame for symbol-universe selection")
    all_symbols = sorted({str(c) for c in base_ret_df.columns})
    selected_symbols = select_symbol_subset(
        list(base_ret_df.columns),
        step=max(1, int(symbol_step)),
        limit=(int(symbol_limit) if int(symbol_limit) > 0 else len(all_symbols)),
    )
    if not selected_symbols:
        raise ValueError("Symbol subset selection produced empty universe")
    if len(selected_symbols) == len(all_symbols):
        tlog(f"Symbol universe selection: using full set ({len(selected_symbols)} symbols)")
    else:
        tlog(
            "Symbol universe reduction applied: "
            f"source={len(base_ret_df.columns)} -> selected={len(selected_symbols)} "
            f"(step={max(1, int(symbol_step))}, limit={int(symbol_limit) if int(symbol_limit) > 0 else 'all'})"
        )
    feats = subset_feature_universe(feats, selected_symbols)

    # Log available features
    test_feature_universe = runtime_cfg.get("test_feature_keys", TEST_FEATURE_KEYS)
    available_model_features = [f for f in test_feature_universe if f in feats]
    if not available_model_features:
        raise ValueError(
            "No configured test_feature_keys are available in loaded features; "
            "cannot run learnability comparison on the requested test feature set"
        )
    if max_features is not None and max_features > 0:
        available_model_features = available_model_features[:max_features]
    logger.info(f"Available test_feature_keys: {len(available_model_features)}/{len(test_feature_universe)}")
    tlog(f"Using {len(available_model_features)} test features")

    # OOM guard: keep only model features plus minimal structural inputs.
    structural_keys = {
        "ret6h",
        "ret24h",
        "atr_pct",
        "trend_pct",
        "range_12h_pct",
        "range_16h_pct",
        "range_pct",
        "volatility_zscore",
        "vol_z",
        "rvol_z",
        "volu_z",
        "sign_consistency",
        "sign_consistency_12h",
    }
    keep_feature_keys = structural_keys | set(available_model_features)
    before_feature_count = len(feats)
    feats = {k: v for k, v in feats.items() if k in keep_feature_keys}
    tlog(
        "Feature pruning applied: "
        f"{before_feature_count} -> {len(feats)} keys "
        f"(model={len(available_model_features)}, structural={len(structural_keys)})"
    )
    # Ensure training-slice builders use the same compact test feature universe.
    runtime_cfg["mr_feature_keys"] = list(available_model_features)
    runtime_cfg["tf_feature_keys"] = list(available_model_features)

    if panel_path is None:
        tlog("No --panel provided; loading panel from partitioned store data_root")
        panel = load_panel_from_store(runtime_cfg, symbols=selected_symbols)
    else:
        tlog("Loading panel data")
        panel_raw = load_panel_data(
            panel_path,
            symbols=selected_symbols,
            columns=["timestamp", "datetime", "open_time", "date", "symbol", "open", "high", "low", "close", "volume"],
        )
        panel = to_panel_dict(panel_raw)
    tlog(f"Loaded panel with close shape={panel['close'].shape}")

    tlog("Precomputing selection metrics")
    metric_pack = precompute_selection_metrics(feats, float_dtype=float_dtype)
    metric_by_mode = metric_pack["metrics"]
    tlog(f"Precomputed metric modes: {list(metric_by_mode.keys())}")

    tlog("Precomputing aggregated TP/SL geometry base for learnability metrics")
    tb_base = None
    try:
        horizons_cfg = runtime_cfg.get("label_horizons_hours", [2, 4, 8])
        base_h = int(runtime_cfg.get("label_horizon_base", 4))
        if base_h not in [int(h) for h in horizons_cfg]:
            base_h = int(horizons_cfg[0]) if horizons_cfg else 4
        tb_cache_base, _geom_cache_base = build_grid_aggregated_tb_cache(
            panel=panel,
            feats=feats,
            cfg=runtime_cfg,
            horizons=[base_h],
            trade_sides=["long", "short"],
        )
        long_pair = tb_cache_base.get((base_h, "long"))
        short_pair = tb_cache_base.get((base_h, "short"))
        if long_pair is not None and short_pair is not None:
            tb_base = {
                "horizon": base_h,
                "ret_long": long_pair[1],
                "ret_short": short_pair[1],
                "label_long": long_pair[0],
                "label_short": short_pair[0],
            }
            tlog(f"Aggregated geometry base ready (horizon={base_h}h)")
        else:
            tlog("Aggregated geometry base unavailable for selected horizon; fallback to legacy target/labels")
    except Exception as exc:
        logger.warning(f"Aggregated geometry base precompute failed ({exc}); falling back to legacy target/labels")

    tlog("Building long-form precomputed tables")
    precomputed = build_long_form_tables(
        feats=feats,
        target_col="ret6h",
        float_dtype=float_dtype,
        atr_reference=metric_pack.get("atr_effective"),
        tb_base=tb_base,
    )
    base_ref_df = feats.get("ret6h") if isinstance(feats.get("ret6h"), pd.DataFrame) else feats.get("ret24h")
    data_container = DataContainer(
        feats=feats,
        index=base_ref_df.index,
        columns=base_ref_df.columns,
        feature_names=available_model_features,
        float_dtype=float_dtype,
    )
    precomputed["data_container"] = data_container
    tlog("Built long-form base table + pre-aligned DataContainer arrays")

    # Keep feats for filter application, but we'll pass it to selection functions
    # Define test configurations with filter variants
    # Filter parameter ranges:
    # - min_range_pct: [0.06, 0.07, 0.08]
    # - min_vol_zscore: [1.4, 1.6, 1.8]
    # - min_sign_consistency: [0.60, 0.70, 0.80]
    
    configs = []
    
    # Default values for filters
    candidate_defaults = get_candidate_filter_defaults(runtime_cfg)
    barrier_defaults = get_barrier_factory_defaults(runtime_cfg)

    default_pct = float(candidate_defaults["train_extreme_pct_hourly"])
    default_range_pct = float(candidate_defaults["train_min_range_pct"])
    default_vol_zscore = float(candidate_defaults["train_min_vol_zscore"])
    default_sign_consistency = float(candidate_defaults["min_feat_sign_consistency"])
    default_tp_lo = float(barrier_defaults["barrier_tp_lo"])
    default_tp_hi = float(barrier_defaults["barrier_tp_hi"])
    default_k_tp = float(barrier_defaults["barrier_k_tp"])
    default_sl_mult = float(barrier_defaults["barrier_sl_base_mult"])
    default_cusum_h = float(runtime_cfg.get("cusum_h", 6.0))
    default_cusum_z_gate = float(runtime_cfg.get("cusum_z_gate", 0.5))
    pct_grid = [0.06, 0.20]  # Broader pct grid for Stage 1/2 comparisons
    
    # Expansion variants
    expansion_variants = [
        ("none", []),
        ("full", [-12, -8, -4, 4, 8, 12, 16]),
        ("neg48", [-4, -8]),
        ("pos48", [4, 8]),
        ("sym48", [-4, -8, 4, 8]),
    ]
    
    # Modes to test
    modes = [("F", "fixed"), ("A", "atr"), ("AVW", "atr_vol_weight"), ("C", "cusum")]
    
    # Use 33% of samples for faster execution
    SAMPLE_FRAC = 1.00
    
    # =============================================================================
    # STAGE 1: Filter sweep (54 configs)
    # 3 modes × 2 pct values × (4 range + 3 vol + 2 sign-consistency)
    # No expansions in this stage - just filter sweeps to find best values
    # =============================================================================
    tlog("Stage 1 setup: building filter-sweep configs")
    # One-at-a-time filter sweeps for each mode (pipeline-aligned defaults on other filters)
    for mode_prefix, mode_name in modes:
        for pct in pct_grid:
            for range_pct in [0.06, 0.07, 0.08, 0.09]:
                configs.append(
                    {
                        "config_id": f"{mode_prefix}_P{int(pct * 100):02d}_R{int(range_pct * 100):02d}",
                        "mode": mode_name,
                        "pct": pct,
                        "min_range_pct": range_pct,
                        "min_vol_zscore": default_vol_zscore,
                        "min_sign_consistency": default_sign_consistency,
                        "cusum_h": default_cusum_h,
                        "cusum_z_gate": default_cusum_z_gate,
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
                        "cusum_h": default_cusum_h,
                        "cusum_z_gate": default_cusum_z_gate,
                    }
                )
            for sc in [0.60, 0.70]:
                configs.append(
                    {
                        "config_id": f"{mode_prefix}_P{int(pct * 100):02d}_S{int(sc * 100):02d}",
                        "mode": mode_name,
                        "pct": pct,
                        "min_range_pct": default_range_pct,
                        "min_vol_zscore": default_vol_zscore,
                        "min_sign_consistency": sc,
                        "cusum_h": default_cusum_h,
                        "cusum_z_gate": default_cusum_z_gate,
                    }
                )

    tlog(f"Stage 1 setup done: {len(configs)} configs")

    # =============================================================================
    # STAGE 2: Expansion variants (30 configs)
    # 3 modes × 2 pct values × 5 expansion variants
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
                    "barrier_k_tp": default_k_tp,
                    "cusum_h": default_cusum_h,
                    "cusum_z_gate": default_cusum_z_gate,
                }
            )

    # Add expansion variants to FULL configs only (30 configs total)
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
    # STAGE 3: PCT variations for winners (up to 20 configs)
    # Added when --stage3 flag is used with --winners
    # 4 winners × 5 pct values = 20 configs (before skipping original pct)
    # Stage 3 always uses ExtraTrees for final selection
    # =============================================================================
    if stage3 and winners:
        tlog(f"Stage 3 setup: building winner pct-variation configs for winners={winners}")
        # Force ExtraTrees for Stage 3
        use_extratrees = True
        
        stage3_pcts = [0.05, 0.06, 0.07, 0.10, 0.20]
        stage3_configs = []
        for cfg in configs:
            # Check if this config matches any of the winners
            # Winners should match the base config_id (without expansion suffix)
            base_id = cfg["config_id"].split("_E")[0] if "_E" in cfg["config_id"] else cfg["config_id"]
            if cfg["config_id"] in winners or base_id in winners:
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
            # Deduplicate same config_id produced by duplicate winners.
            dedup_stage3 = {str(c["config_id"]): c for c in stage3_configs}
            stage3_configs = list(dedup_stage3.values())
            tlog(f"Stage 3 setup done: {len(stage3_configs)} configs")
            tlog(f"Stage 3 pct grid: {sorted({float(c['pct']) for c in stage3_configs})}")
            configs = stage3_configs  # Replace with stage 3 configs only
        else:
            tlog(f"Stage 3 setup done: no configs matched winners={winners}")
    else:
        tlog("Stage 3 setup skipped")

    # Tail-regime variants: by default, compare with/without tail filter.
    requested_tail_mode = str(tail_filter_mode or "auto_compare").lower()
    tail_top_frac = float(tail_filter_top_frac)
    if requested_tail_mode in {"auto", "auto_compare", "default"}:
        tail_variants = [
            ("TBASE", "none"),
            ("TVE20", "vol_or_entropy_top20"),
        ]
    elif requested_tail_mode in {"none", "off", ""}:
        tail_variants = [("", "none")]
    else:
        tail_variants = [("", requested_tail_mode)]

    if len(tail_variants) > 1:
        expanded_tail_configs = []
        for cfg in configs:
            for suffix, mode_name in tail_variants:
                cfg_t = dict(cfg)
                cfg_t["tail_filter_mode"] = mode_name
                cfg_t["tail_filter_top_frac"] = tail_top_frac
                cfg_t["config_id"] = f"{cfg['config_id']}_{suffix}" if suffix else cfg["config_id"]
                expanded_tail_configs.append(cfg_t)
        configs = expanded_tail_configs
        tlog(
            "Tail filter auto-compare enabled: "
            f"variants={[m for _, m in tail_variants]}, top_frac={tail_top_frac:.0%}, "
            f"expanded_configs={len(configs)}"
        )
    else:
        mode_name = tail_variants[0][1]
        for cfg in configs:
            cfg["tail_filter_mode"] = mode_name
            cfg["tail_filter_top_frac"] = tail_top_frac
        tlog(f"Tail filter fixed mode: {mode_name}, top_frac={tail_top_frac:.0%}")

    # Final safety dedupe for config ids before execution.
    cfg_map = {str(c["config_id"]): c for c in configs}
    if len(cfg_map) < len(configs):
        tlog(f"Deduplicated configs by config_id: {len(configs)} -> {len(cfg_map)}")
    configs = list(cfg_map.values())
    
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

    vol_tail_arr = _resolve_feature_aligned_array(
        feats,
        ["rv_24h", "rv_48h", "volatility_zscore", "vol_z", "rvol_z"],
        target_index=metric_ref.index,
        target_columns=metric_ref.columns,
        float_dtype=float_dtype,
    )
    entropy_tail_arr = _resolve_feature_aligned_array(
        feats,
        ["spectral_entropy_ret_24", "shannon_entropy_ret_16", "perm_entropy_ret_24", "volume_entropy_24"],
        target_index=metric_ref.index,
        target_columns=metric_ref.columns,
        float_dtype=float_dtype,
    )
    tlog(
        "Tail filter features: "
        f"requested_mode={tail_filter_mode}, top_frac={float(tail_filter_top_frac):.0%}, "
        f"vol_feature={'yes' if vol_tail_arr is not None else 'no'}, "
        f"entropy_feature={'yes' if entropy_tail_arr is not None else 'no'}"
    )

    base_mask_cache: dict[tuple[str, float], np.ndarray] = {}
    training_slice_cache: dict[tuple[Any, ...], list] = {}
    training_slice_geometry_cache: OrderedDict[tuple[Any, ...], tuple[dict, dict]] = OrderedDict()
    disable_training_slices = False
    shared_geometry_key: Optional[tuple[Any, ...]] = None
    shared_geometry_value: Optional[tuple[dict, dict]] = None

    # Prewarm geometry once for the common barrier setup used by most configs.
    try:
        geom_cfg = deepcopy(runtime_cfg)
        for k, v in barrier_defaults.items():
            geom_cfg[k] = geom_cfg.get(k, v)
        horizons_hours = list(geom_cfg.get("label_horizons_hours", [2, 4, 8]))
        if any(float(h) <= 2.0 for h in horizons_hours):
            geom_cfg["barrier_tp_lo"] = min(float(geom_cfg.get("barrier_tp_lo", 0.02)), 0.004)
        elif any(float(h) <= 4.0 for h in horizons_hours):
            geom_cfg["barrier_tp_lo"] = min(float(geom_cfg.get("barrier_tp_lo", 0.02)), 0.008)
        geom_cfg["barrier_tp_hi"] = max(float(geom_cfg.get("barrier_tp_hi", 0.06)), float(geom_cfg.get("barrier_tp_lo", 0.02)))

        shared_geometry_key = (
            tuple(geom_cfg.get("label_horizons_hours", [2, 4, 8])),
            geom_cfg.get("barrier_k_tp"),
            geom_cfg.get("barrier_sl_base_mult"),
            geom_cfg.get("barrier_disp_floor"),
            geom_cfg.get("barrier_z_max"),
            geom_cfg.get("barrier_k_reg"),
            geom_cfg.get("barrier_m_lo"),
            geom_cfg.get("barrier_m_hi"),
            geom_cfg.get("barrier_sl_lo"),
            geom_cfg.get("barrier_sl_hi"),
            geom_cfg.get("barrier_z_gate"),
            geom_cfg.get("barrier_tp_lo"),
            geom_cfg.get("barrier_tp_hi"),
        )
        tlog("Prewarming shared training-slice geometry cache")
        cache_dir = _geometry_cache_dir(
            output_path=output_path,
            feature_path=feature_path,
            panel_close=panel["close"],
            shared_geometry_key=shared_geometry_key,
        )
        persisted = load_persisted_geometry_cache(cache_dir)
        if persisted is not None:
            shared_tb, shared_geom = persisted
            tlog(f"Loaded persisted geometry cache from {cache_dir}")
        else:
            shared_tb, shared_geom = build_grid_aggregated_tb_cache(
                panel=panel,
                feats=feats,
                cfg=geom_cfg,
                horizons=geom_cfg.get("label_horizons_hours", [2, 4, 8]),
                trade_sides=["long", "short"],
            )
            save_persisted_geometry_cache(
                cache_dir=cache_dir,
                tb_cache_by_h_side=shared_tb,
                geom_cache_by_h_side=shared_geom,
                max_mb=GEOMETRY_CACHE_MAX_MB,
            )
        shared_geometry_value = (shared_tb, shared_geom)
        cache_geometry_entry(training_slice_geometry_cache, shared_geometry_key, shared_geometry_value)
        tlog("Shared training-slice geometry cache ready")
    except Exception as exc:
        tlog(f"Shared geometry prewarm skipped: {exc}")

    tlog(f"Prepared {len(configs)} configs for execution")
    results = []
    slice_results = []
    sample_weight_rows: list[dict] = []
    
    for cfg in configs:
        config_id = cfg["config_id"]
        mode = cfg["mode"]
        pct = cfg["pct"]
        candidate_mask_metrics = None
        candidate_mask_panel = None
        side_sign_metrics = None
        side_sign_panel = None
        
        logger.info("-" * 40)
        stage_label = infer_stage_label(config_id)
        is_stage1 = stage_label == "Stage 1"
        logger.info(f"Running config [{stage_label}]: {config_id} (mode={mode}, pct={pct})")
        tlog(f"Config start: {config_id} | tail_mode={cfg.get('tail_filter_mode', tail_filter_mode)} top_frac={float(cfg.get('tail_filter_top_frac', tail_filter_top_frac)):.0%}")
        
        try:
            if mode not in {"fixed", "atr", "atr_vol_weight", "cusum"}:
                raise ValueError(f"Unsupported mode in default sweep: '{mode}'")
            if mode != "cusum" and mode not in metric_by_mode:
                raise ValueError(f"Required features missing for mode '{mode}'")

            # Extract filter parameters from config
            min_range_pct = cfg.get("min_range_pct")
            min_vol_zscore = cfg.get("min_vol_zscore")
            min_sign_consistency = cfg.get("min_sign_consistency")
            expansion_name = cfg.get("expansion_name", "none")
            expansion_offsets = cfg.get("expansion_offsets", [])

            tlog(f"Selecting candidates: mode={mode}, pct={pct}")
            metric_for_mode = metric_by_mode.get(mode)
            prefilter_arr = None
            low_count_policy = str(cfg.get("cusum_low_count_policy", "keep_all"))
            if mode == "cusum":
                cusum_pack = metric_pack.get("cusum_pack", {}) if isinstance(metric_pack, dict) else {}
                strength_by_h = cusum_pack.get("strength_by_h", {}) if isinstance(cusum_pack, dict) else {}
                cusum_h = float(cfg.get("cusum_h", default_cusum_h))
                metric_for_mode = strength_by_h.get(f"{cusum_h:.1f}")
                if metric_for_mode is None:
                    raise ValueError(f"CUSUM metric unavailable for h={cusum_h:.1f}")

            # Pre-ranking quality filters: rank only among valid points.
            if filter_mask_pack is not None:
                prefilter_arr = np.array(filter_mask_pack.get("true_arr"), copy=True, dtype=bool)
                if min_range_pct is not None:
                    range_mask = filter_mask_pack["range_masks"].get(float(min_range_pct))
                    if range_mask is not None:
                        prefilter_arr &= range_mask
                if min_vol_zscore is not None:
                    vol_mask = filter_mask_pack["vol_masks"].get(float(min_vol_zscore))
                    if vol_mask is not None:
                        prefilter_arr &= vol_mask
                if min_sign_consistency is not None:
                    sc_mask = filter_mask_pack["sc_masks"].get(float(min_sign_consistency))
                    if sc_mask is not None:
                        prefilter_arr &= sc_mask
                prefilter_arr = apply_tail_filter_on_prefilter(
                    prefilter_arr=prefilter_arr,
                    tail_mode=cfg.get("tail_filter_mode", tail_filter_mode),
                    top_frac=float(cfg.get("tail_filter_top_frac", tail_filter_top_frac)),
                    vol_tail_arr=vol_tail_arr,
                    entropy_tail_arr=entropy_tail_arr,
                )
            if mode == "cusum" and metric_for_mode is not None:
                trig = metric_for_mode.to_numpy(dtype=np.float32, copy=False)
                trig_mask = np.isfinite(trig) & (np.abs(trig) > 0)
                prefilter_arr = trig_mask if prefilter_arr is None else (prefilter_arr & trig_mask)

            candidate_mask_base, side_sign_base = select_candidates_cross_sectional(
                metric_for_mode,
                pct,
                filter_masks=filter_mask_pack,
                min_range_pct=min_range_pct,
                min_vol_zscore=min_vol_zscore,
                min_sign_consistency=min_sign_consistency,
                base_mask_cache=base_mask_cache,
                base_cache_key=(mode, float(pct), low_count_policy),
                return_sign=True,
                prefilter_arr=prefilter_arr,
                low_count_policy=low_count_policy,
            )
            # Troubleshooting: show where candidates are filtered out.
            raw_cached = base_mask_cache.get((mode, float(pct), low_count_policy))
            raw_base_arr = raw_cached[0] if isinstance(raw_cached, tuple) else raw_cached
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
            if mode == "cusum":
                cusum_pack = metric_pack.get("cusum_pack", {}) if isinstance(metric_pack, dict) else {}
                z_df = cusum_pack.get("z") if isinstance(cusum_pack, dict) else None
                z_gate = float(cfg.get("cusum_z_gate", default_cusum_z_gate))
                if isinstance(z_df, pd.DataFrame):
                    z_gate_mask = z_df.abs().ge(z_gate).reindex(index=candidate_mask_base.index, columns=candidate_mask_base.columns).fillna(False)
                    candidate_mask_base = (candidate_mask_base & z_gate_mask).fillna(False)
                    side_sign_base = side_sign_base.where(candidate_mask_base, 0).astype(np.int8)
                    tlog(f"CUSUM z-gate applied: z>={z_gate:.2f}")

            base_selected_n = int(candidate_mask_base.to_numpy(dtype=bool, copy=False).sum())
            tlog(f"Candidate base mask: selected={base_selected_n}")

            if mode == "cusum" and expansion_offsets:
                cusum_pack = metric_pack.get("cusum_pack", {}) if isinstance(metric_pack, dict) else {}
                z_df = cusum_pack.get("z") if isinstance(cusum_pack, dict) else None
                ret_df = feats.get("ret1h") if isinstance(feats.get("ret1h"), pd.DataFrame) else feats.get("ret6h")
                candidate_mask_metrics, side_sign_metrics = _conditional_expand_with_z_and_sign(
                    base_mask=candidate_mask_base,
                    base_sign=side_sign_base,
                    offsets=expansion_offsets,
                    z_df=z_df if isinstance(z_df, pd.DataFrame) else None,
                    ret_df=ret_df if isinstance(ret_df, pd.DataFrame) else None,
                    sigma_df=(cusum_pack.get("sigma") if isinstance(cusum_pack.get("sigma"), pd.DataFrame) else None),
                    z_min=float(cfg.get("cusum_expand_z_min", 1.0)),
                    sign_pct=float(cfg.get("cusum_expand_sign_pct", 0.6)),
                    consistency_bars=int(cfg.get("cusum_expand_consistency_bars", 5)),
                    vol_ratio=float(cfg.get("cusum_expand_vol_ratio", 1.2)),
                )
            else:
                candidate_mask_metrics = expand_candidate_mask(candidate_mask_base, expansion_offsets)
                side_sign_metrics = side_sign_base.copy()
                if expansion_offsets:
                    for off in expansion_offsets:
                        shifted = side_sign_base.shift(int(off)).fillna(0).astype(np.int8)
                        # Preserve first non-zero sign when expanded overlap occurs.
                        fill_mask = (side_sign_metrics == 0) & (shifted != 0)
                        side_sign_metrics = side_sign_metrics.where(~fill_mask, shifted)
                side_sign_metrics = side_sign_metrics.where(candidate_mask_metrics, 0).astype(np.int8)
            expanded_selected_n = int(candidate_mask_metrics.to_numpy(dtype=bool, copy=False).sum())
            if expansion_offsets:
                tlog(
                    f"Applied candidate expansion ({expansion_name}): offsets={expansion_offsets}, "
                    f"selected={expanded_selected_n}"
                )
            else:
                tlog(f"Candidate expansion skipped: selected={expanded_selected_n}")

            stage_symbol_step = STAGE1_SYMBOL_SUBSAMPLE_STEP if is_stage1 else STAGE23_SYMBOL_SUBSAMPLE_STEP
            if stage_symbol_step > 1:
                keep_cols = list(candidate_mask_metrics.columns[::stage_symbol_step])
                if keep_cols and len(keep_cols) < candidate_mask_metrics.shape[1]:
                    candidate_mask_metrics = candidate_mask_metrics.reindex(columns=keep_cols, fill_value=False)
                    side_sign_metrics = side_sign_metrics.reindex(columns=keep_cols, fill_value=0).astype(np.int8)
                    stage1_selected_n = int(candidate_mask_metrics.to_numpy(dtype=bool, copy=False).sum())
                    tlog(
                        f"{stage_label} symbol subsample applied: "
                        f"step={stage_symbol_step}, cols={len(keep_cols)}, selected={stage1_selected_n}"
                    )

            # Keep learnability metrics in feature-space coordinates; build a separate
            # panel-aligned mask only for training-slice stage.
            candidate_mask_panel = align_candidate_mask_to_panel_symbols(candidate_mask_metrics, panel)
            side_sign_panel = side_sign_metrics.reindex(
                index=candidate_mask_panel.index,
                columns=candidate_mask_panel.columns,
                fill_value=0,
            ).astype(np.int8)
            side_sign_panel = side_sign_panel.where(candidate_mask_panel, 0).astype(np.int8)
            aligned_selected_n = int(candidate_mask_panel.to_numpy(dtype=bool, copy=False).sum())
            panel_overlap_cols = int(
                len(candidate_mask_panel.columns.intersection(panel["close"].columns))
            ) if "close" in panel else 0
            tlog(
                f"Candidate alignment: selected={aligned_selected_n}, "
                f"cols={candidate_mask_panel.shape[1]}, panel_overlap_cols={panel_overlap_cols}"
            )
            if base_selected_n > 0 and aligned_selected_n == 0:
                tlog(
                    "Troubleshoot: candidates collapsed to zero after expansion/alignment. "
                    "Check filter thresholds and symbol naming consistency."
                )
            del candidate_mask_base
            
            # Compute metrics
            tlog(f"Computing learnability metrics: {config_id}")
            oof_n_splits = STAGE1_OOF_SPLITS if is_stage1 else STAGE23_OOF_SPLITS
            oof_max_samples = STAGE1_OOF_MAX_SAMPLES if is_stage1 else OOF_MAX_SAMPLES
            metrics = compute_learnability_metrics(
                candidate_mask=candidate_mask_metrics,
                precomputed=precomputed,
                available_features=available_model_features,
                float_dtype=float_dtype,
                side_sign=side_sign_metrics,
                use_extratrees=use_extratrees,
                oof_max_samples=oof_max_samples,
                oof_n_splits=oof_n_splits,
            )

            cfg_variant = deepcopy(runtime_cfg)
            cfg_variant["train_extreme_pct_hourly"] = pct
            
            # Unified barrier factory params (v3 - single source of truth)
            # These replace the old train_tp_lo, train_tp_hi, train_sl_mult params
            cfg_variant["barrier_k_tp"] = float(cfg.get("barrier_k_tp", barrier_defaults["barrier_k_tp"]))
            cfg_variant["barrier_sl_base_mult"] = float(cfg.get("barrier_sl_base_mult", barrier_defaults["barrier_sl_base_mult"]))
            cfg_variant["barrier_disp_floor"] = float(cfg.get("barrier_disp_floor", barrier_defaults["barrier_disp_floor"]))
            cfg_variant["barrier_z_max"] = float(cfg.get("barrier_z_max", barrier_defaults["barrier_z_max"]))
            cfg_variant["barrier_k_reg"] = float(cfg.get("barrier_k_reg", barrier_defaults["barrier_k_reg"]))
            cfg_variant["barrier_m_lo"] = float(cfg.get("barrier_m_lo", barrier_defaults["barrier_m_lo"]))
            cfg_variant["barrier_m_hi"] = float(cfg.get("barrier_m_hi", barrier_defaults["barrier_m_hi"]))
            cfg_variant["barrier_sl_lo"] = float(cfg.get("barrier_sl_lo", barrier_defaults["barrier_sl_lo"]))
            cfg_variant["barrier_sl_hi"] = float(cfg.get("barrier_sl_hi", barrier_defaults["barrier_sl_hi"]))
            cfg_variant["barrier_z_gate"] = float(cfg.get("barrier_z_gate", barrier_defaults["barrier_z_gate"]))
            cfg_variant["barrier_tp_lo"] = float(cfg.get("barrier_tp_lo", barrier_defaults["barrier_tp_lo"]))
            cfg_variant["barrier_tp_hi"] = float(cfg.get("barrier_tp_hi", barrier_defaults["barrier_tp_hi"]))
            cfg_variant["label_horizon_base"] = float(cfg.get("label_horizon_base", barrier_defaults["label_horizon_base"]))

            horizons_hours = list(cfg_variant.get("label_horizons_hours", [2, 4, 8]))
            if any(float(h) <= 2.0 for h in horizons_hours):
                cfg_variant["barrier_tp_lo"] = min(float(cfg_variant["barrier_tp_lo"]), 0.004)
            elif any(float(h) <= 4.0 for h in horizons_hours):
                cfg_variant["barrier_tp_lo"] = min(float(cfg_variant["barrier_tp_lo"]), 0.008)
            cfg_variant["barrier_tp_hi"] = max(float(cfg_variant["barrier_tp_hi"]), float(cfg_variant["barrier_tp_lo"]))
            
            if cfg.get("min_range_pct") is not None:
                cfg_variant["train_min_range_pct"] = float(cfg["min_range_pct"])
            if cfg.get("min_vol_zscore") is not None:
                cfg_variant["train_min_vol_zscore"] = float(cfg["min_vol_zscore"])
            if cfg.get("min_sign_consistency") is not None:
                cfg_variant["min_feat_sign_consistency"] = float(cfg["min_sign_consistency"])

            training_cache_key = (
                fingerprint_candidate_mask(candidate_mask_panel),
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
            geometry_cache_key = (
                tuple(cfg_variant.get("label_horizons_hours", [2, 4, 8])),
                cfg_variant.get("barrier_k_tp"),
                cfg_variant.get("barrier_sl_base_mult"),
                cfg_variant.get("barrier_disp_floor"),
                cfg_variant.get("barrier_z_max"),
                cfg_variant.get("barrier_k_reg"),
                cfg_variant.get("barrier_m_lo"),
                cfg_variant.get("barrier_m_hi"),
                cfg_variant.get("barrier_sl_lo"),
                cfg_variant.get("barrier_sl_hi"),
                cfg_variant.get("barrier_z_gate"),
                cfg_variant.get("barrier_tp_lo"),
                cfg_variant.get("barrier_tp_hi"),
            )

            if disable_training_slices:
                tlog("Training-slice stage skipped: disabled after prior structural precheck failure")
                training_slice_rows = []
            else:
                ok_slices, why_not = training_slice_precheck(candidate_mask_panel, panel)
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
                    shared_geom = (
                        shared_geometry_value
                        if shared_geometry_key is not None and geometry_cache_key == shared_geometry_key
                        else None
                    )
                    training_slice_rows = evaluate_training_slices(
                        candidate_mask=candidate_mask_panel,
                        feats=feats,
                        panel=panel,
                        cfg_variant=cfg_variant,
                        horizons=cfg_variant.get("label_horizons_hours", [2, 4, 8]),
                        cache=training_slice_cache,
                        cache_key=training_cache_key,
                        sample_frac=SAMPLE_FRAC,
                        geometry_cache=training_slice_geometry_cache,
                        geometry_cache_key=geometry_cache_key,
                        precomputed_geometry=shared_geom,
                        sample_weight_sink=sample_weight_rows if save_sample_weights else None,
                        config_id=config_id,
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
            
            logger.info(
                f"  Candidates/timestamp: {metrics['n_candidates_mean']:.3f} "
                f"(rate={metrics.get('candidate_rate', 0.0):.4%})"
            )
            logger.info(f"  IC: {metrics['ic']:.4f} ± {metrics['ic_std']:.4f}")
            logger.info(f"  KS: {metrics['ks_stat']:.4f}, SNR: {metrics['snr']:.4f}")
            logger.info(f"  Class balance: {metrics['class_balance']:.2%}")
            logger.info(f"  Sharpe: {metrics['sharpe']:.2f}")
            logger.info(
                f"  HitRate: {metrics['hit_rate']:.2%} | Sortino: {metrics['sortino']:.2f} | "
                f"Return(bps): {metrics['mean_return_bps']:.2f} ± {metrics['volatility_bps']:.2f}"
            )
            logger.info(f"  Clf AUC: {metrics.get('auc', 0.0):.4f} | Brier: {metrics.get('brier', 0.0):.4f}")
            logger.info(
                f"  OOF provenance: IC[{metrics.get('ic_source','none')} n={int(metrics.get('ic_oof_n',0))}] "
                f"AUC[{metrics.get('auc_source','none')} n={int(metrics.get('auc_oof_n',0))}] "
                f"Brier[{metrics.get('brier_source','none')} n={int(metrics.get('brier_oof_n',0))}]"
            )
            logger.info(
                f"  SliceSharpe: {metrics['slice_overall_sharpe']:.2f} | "
                f"SliceSortino: {metrics['slice_overall_sortino']:.2f} | "
                f"SliceOpp/Day: {metrics.get('slice_overall_opportunities_per_day', 0.0):.2f} | "
                f"SliceN: {metrics['slice_total_samples']}"
            )
            logger.info(
                f"  TP:SL used: {float(cfg.get('barrier_k_tp', 1.0)):.2f}:{float(cfg.get('barrier_sl_base_mult', 0.5)):.2f}"
            )
            tlog(f"Config done: {config_id}")
            
            # Free memory
            del candidate_mask_metrics
            del candidate_mask_panel
            del side_sign_metrics
            del side_sign_panel
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
                "n_candidates_mean": np.nan,
                "ic": np.nan,
                "ic_std": np.nan,
                "ks_stat": np.nan,
                "snr": np.nan,
                "class_balance": np.nan,
                "mean_feat_ic": np.nan,
                "sharpe": np.nan,
                "candidate_rate": np.nan,
                "mean_return_bps": np.nan,
                "volatility_bps": np.nan,
                "sortino": np.nan,
                "hit_rate": np.nan,
                "tail_ratio": np.nan,
                "ic_spearman": np.nan,
                "oof_mae": np.nan,
                "oof_directional_acc": np.nan,
                "ic_source": "none",
                "ic_oof_n": 0,
                "auc_source": "none",
                "auc_oof_n": 0,
                "brier_source": "none",
                "brier_oof_n": 0,
                "auc": np.nan,
                "brier": np.nan,
                "ridge_alpha": RIDGE_SCREEN_ALPHA,
                "ridge_top_frac": RIDGE_SCREEN_TOP_FRAC,
                "ridge_selected_k_mean": np.nan,
                "ridge_jaccard_median": np.nan,
                "ridge_replacement_rate_median": np.nan,
                "mean_abs_ret6h": np.nan,
                "median_abs_ret6h": np.nan,
                "ret6h_q01": np.nan,
                "ret6h_q05": np.nan,
                "atr_mean": np.nan,
                "atr_q10": np.nan,
                "atr_q50": np.nan,
                "atr_q90": np.nan,
                "atr_decile_worst": -1,
                "atr_decile_worst_share": np.nan,
                "atr_decile_pnl_json": "{}",
                "slice_overall_sharpe": np.nan,
                "slice_overall_sortino": np.nan,
                "slice_overall_opportunities_per_day": np.nan,
                "slice_total_samples": 0,
                "slice_metrics_json": "{}",
                "error": str(e)
            })
        finally:
            candidate_mask_metrics = None
            candidate_mask_panel = None
            side_sign_metrics = None
            side_sign_panel = None
            feature_cache = precomputed.get("feature_series_cache")
            if isinstance(feature_cache, dict) and feature_cache:
                feature_cache.clear()
                tlog(f"Per-config cleanup: cleared stacked feature cache for {config_id}")
            gc.collect()
            tlog(f"Config cleanup done: {config_id}")
    
    tlog("Building results dataframe")
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    if "config_id" in results_df.columns:
        n_before = len(results_df)
        results_df = results_df.drop_duplicates(subset=["config_id"], keep="first").reset_index(drop=True)
        n_after = len(results_df)
        if n_after < n_before:
            tlog(f"Deduplicated results by config_id: {n_before} -> {n_after}")

    # Learnability-first policy ranking with classifier-aware global score.
    error_series = results_df.get("error", pd.Series("", index=results_df.index)).fillna("").astype(str)
    results_df["has_error"] = error_series.str.len() > 0
    valid_policy_mask = ~results_df["has_error"].to_numpy(dtype=bool)

    results_df["auc"] = pd.to_numeric(results_df.get("auc", np.nan), errors="coerce")
    results_df["brier"] = pd.to_numeric(results_df.get("brier", np.nan), errors="coerce")

    valid_icstd = pd.to_numeric(results_df.loc[valid_policy_mask, "ic_std"], errors="coerce").to_numpy(dtype=float)
    valid_icstd = valid_icstd[np.isfinite(valid_icstd)]
    icstd_threshold = float(np.median(valid_icstd)) if len(valid_icstd) > 0 else float("nan")
    results_df["policy_icstd_threshold"] = icstd_threshold
    results_df["policy_pass_stability"] = False
    if np.isfinite(icstd_threshold):
        results_df.loc[valid_policy_mask, "policy_pass_stability"] = (
            pd.to_numeric(results_df.loc[valid_policy_mask, "ic_std"], errors="coerce") <= icstd_threshold
        )

    score = pd.Series(np.nan, index=results_df.index, dtype=float)
    if bool(np.any(valid_policy_mask)):
        valid_idx = results_df.index[valid_policy_mask]
        rank_ic = results_df.loc[valid_idx, "ic"].rank(pct=True, method="average")
        rank_ks = results_df.loc[valid_idx, "ks_stat"].rank(pct=True, method="average")
        rank_snr = results_df.loc[valid_idx, "snr"].rank(pct=True, method="average")
        rank_sharpe = results_df.loc[valid_idx, "sharpe"].rank(pct=True, method="average")
        rank_sortino = results_df.loc[valid_idx, "sortino"].rank(pct=True, method="average")
        rank_auc = results_df.loc[valid_idx, "auc"].rank(pct=True, method="average")
        rank_brier = 1.0 - results_df.loc[valid_idx, "brier"].rank(pct=True, method="average")

        score.loc[valid_idx] = (
            0.22 * rank_ic
            + 0.16 * rank_ks
            + 0.14 * rank_snr
            + 0.12 * rank_sharpe
            + 0.12 * rank_sortino
            + 0.14 * rank_auc
            + 0.10 * rank_brier
        )
        # Small stability bonus/penalty keeps unstable IC_std from dominating rank.
        score.loc[valid_idx] = score.loc[valid_idx] * np.where(
            results_df.loc[valid_idx, "policy_pass_stability"].to_numpy(dtype=bool),
            1.03,
            0.97,
        )

    results_df["global_score"] = score

    sort_view = results_df.sort_values(
        by=["has_error", "global_score", "policy_pass_stability", "ic", "ks_stat", "snr"],
        ascending=[True, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    sort_view["global_rank"] = np.arange(1, len(sort_view) + 1, dtype=np.int32)
    sort_view["policy_rank"] = sort_view["global_rank"]
    results_df = results_df.merge(
        sort_view[["config_id", "policy_rank", "global_rank"]],
        on="config_id",
        how="left",
    )
    if len(sort_view) > 0:
        best_row = sort_view.iloc[0]
        tlog(
            "Policy best config: "
            f"{best_row['config_id']} | pass={bool(best_row['policy_pass_stability'])} "
            f"global_score={best_row['global_score']:.4f} "
            f"IC={best_row['ic']:.4f} IC_std={best_row['ic_std']:.4f} "
            f"AUC={best_row.get('auc', 0.0):.4f} Brier={best_row.get('brier', 0.0):.4f}"
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
        slices_df = pd.DataFrame(slice_results)
        if {"config_id", "slice", "horizon"}.issubset(slices_df.columns):
            n_before = len(slices_df)
            slices_df = slices_df.drop_duplicates(subset=["config_id", "slice", "horizon"], keep="first")
            n_after = len(slices_df)
            if n_after < n_before:
                tlog(f"Deduplicated slice rows: {n_before} -> {n_after}")
        slices_df.to_csv(slice_output, index=False)
        logger.info(f"Slice-level results saved to: {slice_output}")
    if save_sample_weights:
        sw_output = output_path.replace(".csv", "_sample_weights.csv")
        if sample_weight_rows:
            sw_df = pd.DataFrame(sample_weight_rows)
            sw_df.to_csv(sw_output, index=False)
            logger.info(f"Sample weights saved to: {sw_output}")
        else:
            logger.warning(
                "Sample-weight export requested but no slice weights were generated; "
                "downstream should use existing/default sample-weight values as fallback"
            )
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")

    try:
        if len(sort_view) > 0:
            best = sort_view.iloc[0]
            best_params = {
                "train_extreme_pct_hourly": float(best.get("pct", default_pct)),
                "train_min_range_pct": float(best.get("min_range_pct", default_range_pct)) if pd.notna(best.get("min_range_pct")) else default_range_pct,
                "train_min_vol_zscore": float(best.get("min_vol_zscore", default_vol_zscore)) if pd.notna(best.get("min_vol_zscore")) else default_vol_zscore,
                "min_feat_sign_consistency": float(best.get("min_sign_consistency", default_sign_consistency)) if pd.notna(best.get("min_sign_consistency")) else default_sign_consistency,
            }
            save_best_params_csv(CANDIDATE_BEST_PARAMS_CSV, best_params, metadata={"source": "compare_candidate_thresholds"})
            logger.info(f"Saved best params CSV: {CANDIDATE_BEST_PARAMS_CSV}")
    except Exception as exc:
        logger.warning(f"Failed to persist candidate best params CSV: {exc}")
    
    # Print summary table
    print("\n" + "=" * 172)
    print("CANDIDATE SELECTION THRESHOLD COMPARISON RESULTS")
    print("=" * 172)
    print(f"{'Config':<10} {'Mode':<12} {'Pct':>5} {'Range':>6} {'VolZ':>5} {'SignC':>5} "
          f"{'N_Cand':>7} {'IC':>7} {'IC_std':>7} {'AUC':>6} {'Brier':>6} {'KS':>6} {'SNR':>6} "
          f"{'Bal':>6} {'Sharpe':>7} {'Hit':>6} {'Sort':>7} {'GScore':>7} {'Rank':>5}")
    print("-" * 172)
    
    for _, row in results_df.iterrows():
        range_str = f"{row['min_range_pct']:.2f}" if pd.notna(row.get('min_range_pct')) else "-"
        volz_str = f"{row['min_vol_zscore']:.1f}" if pd.notna(row.get('min_vol_zscore')) else "-"
        signc_str = f"{row['min_sign_consistency']:.0%}" if pd.notna(row.get('min_sign_consistency')) else "-"
        global_rank = int(row["global_rank"]) if pd.notna(row.get("global_rank")) else 0
        print(f"{row['config_id']:<10} {row['mode']:<12} {row['pct']:>5.2f} "
              f"{range_str:>6} {volz_str:>5} {signc_str:>5} "
              f"{row['n_candidates_mean']:>7.1f} {row['ic']:>7.4f} {row['ic_std']:>7.4f} "
              f"{row.get('auc', 0.0):>6.3f} {row.get('brier', 0.0):>6.3f} "
              f"{row['ks_stat']:>6.3f} {row['snr']:>6.2f} {row['class_balance']:>5.1%} "
              f"{row['sharpe']:>7.2f} {row['hit_rate']:>5.1%} {row['sortino']:>7.2f} "
              f"{row.get('global_score', 0.0):>7.4f} {global_rank:>5d}")

    if len(sort_view) > 0:
        best = sort_view.iloc[0]
        print(
            f"Policy best: {best['config_id']} | Rank=1 | "
            f"Pass={bool(best['policy_pass_stability'])} | "
            f"GlobalScore={best.get('global_score', 0.0):.4f} | "
            f"IC={best['ic']:.4f} IC_std={best['ic_std']:.4f} "
            f"AUC={best.get('auc', 0.0):.4f} Brier={best.get('brier', 0.0):.4f} "
            f"Sharpe={best['sharpe']:.2f} Sortino={best['sortino']:.2f}"
        )
    print("\nGLOBAL RANKING (best -> worst)")
    rank_cols = ["global_rank", "config_id", "global_score", "ic", "auc", "brier", "sharpe", "sortino"]
    print(sort_view[rank_cols].to_string(index=False))

    print("=" * 172)
    
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
        required=False,
        default=None,
        help="Path to feature data (directory or parquet file). If omitted, auto-detect latest under --data-root/features."
    )
    parser.add_argument(
        "--panel",
        required=False,
        default=None,
        help="Path to panel data (klines/OHLCV). If omitted, loads from partitioned store in --data-root."
    )
    parser.add_argument(
        "--data-root",
        required=False,
        default=None,
        help="Override cfg data_root for auto-detection"
    )
    parser.add_argument(
        "--perps",
        action="store_true",
        help="Use perp mode data/features (_perp root + perp feature keys)"
    )
    parser.add_argument(
        "--output",
        default=str(REPORTS_DIR / "candidate_threshold_comparison.csv"),
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
        help="Cap number of test_feature_keys used for OOF modeling (default: 60)"
    )
    parser.add_argument(
        "--stage3",
        action="store_true",
        help="Force stage 3 auto-run after stage 2 (default behavior unless --skip-stage3 is set)"
    )
    parser.add_argument(
        "--skip-stage3",
        action="store_true",
        help="Run only stage 2 and skip stage 3"
    )
    parser.add_argument(
        "--winners",
        nargs="+",
        default=[],
        help="List of winning config_ids from stage 2 to test in stage 3 (auto-selected if not provided)"
    )
    parser.add_argument(
        "--symbol-step",
        type=int,
        default=3,
        help="Select every N-th symbol in sorted universe (default: 3)"
    )
    parser.add_argument(
        "--symbol-limit",
        type=int,
        default=0,
        help="Max number of symbols to keep after step sampling (default: 0 = no cap)"
    )
    parser.add_argument(
        "--save-sample-weights",
        action="store_true",
        help="Export training-slice sample weights to <output>_sample_weights.csv for downstream base/meta models"
    )
    parser.add_argument(
        "--tail-filter-mode",
        choices=["auto_compare", "none", "vol24_top20", "entropy_top20", "vol_or_entropy_top20"],
        default="auto_compare",
        help=(
            "Tail-regime policy. auto_compare (default) runs with/without tail filter; "
            "Quantiles are computed only within each row's pre-existing mask."
        ),
    )
    parser.add_argument(
        "--tail-filter-top-frac",
        type=float,
        default=0.20,
        help="Top fraction for tail-filter quantiles (default: 0.20). Used by auto_compare tail variant too.",
    )

    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    runtime_cfg = _resolve_runtime_cfg(perps=bool(args.perps), data_root=args.data_root)
    feature_path = args.features or _find_latest_feature_dir(runtime_cfg["data_root"])
    if not feature_path:
        raise ValueError(
            f"Could not auto-detect features under {runtime_cfg['data_root']}/features; provide --features explicitly."
        )

    # =============================================================================
    # Stage 2 always runs first; Stage 3 auto-runs unless explicitly skipped.
    # =============================================================================
    auto_stage3 = (not args.skip_stage3) or args.stage3 or len(args.winners) > 0

    tlog("Starting Stage 2 run")
    run_comparison(
        feature_path,
        args.panel,
        args.output,
        dtype=args.dtype,
        max_features=args.max_features,
        stage3=False,
        winners=[],
        symbol_step=args.symbol_step,
        symbol_limit=args.symbol_limit,
        save_sample_weights=args.save_sample_weights,
        tail_filter_mode=args.tail_filter_mode,
        tail_filter_top_frac=args.tail_filter_top_frac,
        runtime_cfg=runtime_cfg,
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
                # Find top 4 FULL_E configs by global score from Stage 2 output.
                full_configs = prev_results[prev_results['config_id'].str.contains('FULL_E')]
                if len(full_configs) > 0:
                    score_col = "global_score" if "global_score" in full_configs.columns else "policy_rank"
                    if score_col == "global_score":
                        top_winners = full_configs.nlargest(4, score_col)["config_id"].tolist()
                    else:
                        top_winners = full_configs.nsmallest(4, score_col)["config_id"].tolist()
                    top_winners = list(dict.fromkeys([str(w) for w in top_winners]))
                    tlog(f"Auto-selected top 4 winners: {top_winners}")
                else:
                    tlog("No FULL_E configs found, cannot auto-select winners")
                    top_winners = []
            except Exception as e:
                tlog(f"Could not auto-select winners: {e}")
                top_winners = []
        else:
            top_winners = list(dict.fromkeys([str(w) for w in args.winners]))
        
        if top_winners:
            # Run Stage 3 with top winners
            stage3_output = args.output.replace('.csv', '_stage3.csv')
            tlog(f"Starting Stage 3 run with winners={top_winners} and pct grid [0.05, 0.06, 0.07, 0.10, 0.20]")
            run_comparison(
                feature_path,
                args.panel,
                stage3_output,
                dtype=args.dtype,
                max_features=args.max_features,
                stage3=True,
                winners=top_winners,
                use_extratrees=True,  # Use ExtraTrees for Stage 3
                symbol_step=args.symbol_step,
                symbol_limit=args.symbol_limit,
                save_sample_weights=args.save_sample_weights,
                tail_filter_mode=args.tail_filter_mode,
                tail_filter_top_frac=args.tail_filter_top_frac,
                runtime_cfg=runtime_cfg,
            )
        else:
            tlog("Stage 3 skipped: no winners selected")
    else:
        tlog("Stage 3 disabled via --skip-stage3")
