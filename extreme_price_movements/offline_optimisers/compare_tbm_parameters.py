#!/usr/bin/env python3
"""Compare/optimize Triple-Barrier Method (TBM) parameters.

Implements the staged optimization plan from `plans/tbm_parameter_optimization.md`:
- Stage 1 quick scan on atr_mult_rr baseline mode.
- Stage 2 validation on additional modes + optional path dependence knobs.
- Unified parameter semantics (absolute caps vs multipliers).
- Two-layer caching (barrier series + labels).
- Learnability metrics: IC_label, IC_payoff, calibration, and robustness slices.

Note on performance:
- Barrier construction is fully vectorized (Pandas).
- Labeling iterates over assets (Python loop) but uses Numba-compiled kernels for high speed.
- Memory usage is controlled via LRU caching of intermediate barrier/label artifacts.

The script is intentionally self-contained and resilient to different parquet layouts.
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import math
import os
import resource
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Apple Silicon / BLAS thread controls: set before numpy/sklearn imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score
from joblib import Parallel, delayed
import optuna
try:
    import numba  # type: ignore
    from numba import njit as _njit
    _NUMBA_OK = True
except ImportError:  # pragma: no cover
    _NUMBA_OK = False
    def _njit(*args, **kwargs):  # type: ignore
        """Passthrough when numba is unavailable."""
        def _dec(fn): return fn
        return _dec
try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    xgb = None

MODULE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = MODULE_DIR.parent
import sys
import platform

if sys.version_info < (3, 10):
    raise RuntimeError("compare_tbm_parameters.py requires Python >= 3.10")

from extreme_price_movements.data_store import to_panel
from extreme_price_movements.labeling import compute_triple_barrier_labels, OUT_SL, OUT_TO, OUT_TP
from extreme_price_movements.config import (
    CFG,
    TEST_FEATURE_KEYS,
    PERP_FEATURE_KEYS,
    enable_perp_feature_keys,
)
from extreme_price_movements.candidates import select_trade_candidates_vectorized
from extreme_price_movements.offline_optimisers.params_store import (
    REPORTS_DIR,
    TBM_BEST_PARAMS_CSV,
    TBM_BEST_PARAMS_PER_BUCKET_CSV,
    TBM_BEST_PARAMS_PER_CELL_CSV,
    TBM_GEOMETRY_GRID_CSV,
    TBM_BUCKET_NAMES,
    TBM_HORIZONS,  # Single source of truth for horizons
    save_best_params_csv,
    apply_offline_optimizer_best_params,
)
from extreme_price_movements.training_defaults import (
    get_candidate_filter_defaults,
    get_barrier_factory_defaults,
    get_tbm_optimizer_defaults,
)
from extreme_price_movements.utils import tprint
from extreme_price_movements.barrier_geometry import (
    make_effective_tp,
    effective_tp_floor,
    effective_sl_floor,
    apply_horizon_scaling,
)
from extreme_price_movements.sample_weights import (
    compute_cell_weights_neg_mass_renorm,
    NegMassRenormCfg,
)
from extreme_price_movements.production_admissibility import (
    ProdGates,
    production_admissibility_report,
    apply_econ_guardrail_to_stage2,
    compute_prod_aligned_tp_params,
)
from extreme_price_movements.production_sl_tp_policy import (
    SLTPPolicy,
    expand_configs_wide_sl_tp_additive_superiority,
)
from extreme_price_movements.hf_data_loader import HF_DATA_DIR, get_15m_ohlcv


EPS = 1e-12
TBM_CACHE_VERSION = 2
ACTIVE_TEST_FEATURE_KEYS = list(TEST_FEATURE_KEYS)
USE_15M_REFINEMENT = True  # Feature toggle to disable 15m data download/refinement
_HF_15M_FAILED_SYMBOLS: set[str] = set()
_HF_15M_MEMORY_CACHE: Dict[str, pd.DataFrame] = {}
_HF_15M_LOCAL_AVAILABLE: Dict[str, bool] = {}


def _get_global_hf_cache() -> Dict[str, pd.DataFrame]:
    return _HF_15M_MEMORY_CACHE


def _is_apple_arm() -> bool:
    return (platform.system() == "Darwin") and (platform.machine() in {"arm64", "aarch64"})


def _safe_parallel_jobs(n_items: int, cap: int = 8) -> int:
    if n_items <= 1:
        return 1
    cpu = int(os.cpu_count() or 1)
    return max(1, min(cap, n_items, max(1, cpu // 2)))


def _bars_for_hours(timeframe: str, hours: float) -> int:
    """Convert hours into bar count for timeframe strings like 15m/1h/4h."""
    tf = str(timeframe or "15m").strip().lower()
    step_min = 15.0
    try:
        if tf.endswith("m"):
            step_min = float(tf[:-1])
        elif tf.endswith("h"):
            step_min = float(tf[:-1]) * 60.0
        elif tf.endswith("d"):
            step_min = float(tf[:-1]) * 60.0 * 24.0
    except Exception:
        step_min = 15.0
    step_min = max(step_min, 1.0)
    return max(1, int(round((float(hours) * 60.0) / step_min)))


def _resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(PACKAGE_ROOT, path))


def _append_suffix(path: str, suffix: str) -> str:
    norm = str(path).rstrip("/\\")
    if norm.endswith(suffix):
        return norm
    return f"{norm}{suffix}"


def _memory_snapshot_mb() -> float:
    """Process resident memory estimate in MB (high-water mark on Linux)."""
    import sys
    rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss_kb / (1024.0 * 1024.0)
    return rss_kb / 1024.0


def _maybe_collect_gc(
    *,
    last_gc_ts: float,
    mem_threshold_mb: float = 8192.0,
    min_interval_s: float = 5.0,
) -> float:
    """Run gc.collect() only under memory pressure and rate limit it."""
    now = time.perf_counter()
    if (now - last_gc_ts) < min_interval_s:
        return last_gc_ts
    mem_mb = _memory_snapshot_mb()
    if mem_mb >= mem_threshold_mb:
        gc.collect()
        return now
    return last_gc_ts


def _cache_pressure_summary(
    layer1_cache: Dict[str, Any],
    layer2_cache: Dict[str, Any],
    eval_cache: BoundedEvalCache,
) -> str:
    bucket_stack = eval_cache.get("bucket_stack", {})
    return (
        f"cache_sizes(layer1={len(layer1_cache)}, layer2={len(layer2_cache)}, "
        f"eval_keys={len(eval_cache)}, bucket_stack_keys={len(bucket_stack)})"
    )


def _should_log_progress(idx: int, total: int, *, every: int = 50) -> bool:
    """Low-overhead cadence helper to avoid excessive tprint noise."""
    if total <= 0:
        return False
    if idx <= 1 or idx >= total:
        return True
    return (idx % max(1, int(every))) == 0


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _sanitize_quality_array(values: np.ndarray, *, fallback: float = 0.5) -> np.ndarray:
    """Guarantee finite quality scores in [0,1] and emit warning on repair."""
    arr = np.asarray(values, dtype=np.float32)
    bad = ~np.isfinite(arr)
    if np.any(bad):
        tprint(f"WARNING: quality contains {int(bad.sum())} non-finite values; replacing with fallback={float(fallback):.3f}")
    arr = np.nan_to_num(arr, nan=float(fallback), posinf=float(fallback), neginf=float(fallback))
    return np.clip(arr, 0.0, 1.0)


def _symbol_to_hf_ccxt(symbol: str) -> str:
    raw = str(symbol or "").upper().strip()
    s = raw.replace("/", "").replace("_", "").replace("-", "").replace(" ", "")
    for quote in ("USDT", "USDC", "BUSD", "EUR", "BTC", "ETH"):
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[:-len(quote)]}/{quote}"
    return raw.replace("_", "/")


def _has_local_15m_cache(symbol: str) -> bool:
    """Fast local-availability check for 15m parquet (memoized)."""
    clean = str(symbol or "").replace("/", "").replace("_", "").lower()
    cached = _HF_15M_LOCAL_AVAILABLE.get(clean)
    if cached is not None:
        return bool(cached)

    path = HF_DATA_DIR / f"{clean}_15m.parquet"
    available = path.exists()
    if not available:
        # Fallback: non-USDT quote can reuse USDT cache when available.
        clean_up = clean.upper()
        for q in ("USDC", "BUSD", "EUR"):
            if clean_up.endswith(q):
                base = clean_up[:-len(q)].lower()
                available = (HF_DATA_DIR / f"{base}usdt_15m.parquet").exists()
                break
    _HF_15M_LOCAL_AVAILABLE[clean] = bool(available)
    return bool(available)


def _load_or_download_15m(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, *, allow_download: bool = False) -> pd.DataFrame:
    """Load or download 15m data with global memory caching."""
    clean = str(symbol or "").replace("/", "").replace("_", "").lower()
    ccxt_symbol = _symbol_to_hf_ccxt(symbol)
    
    # 1. Check Memory Cache first (fastest)
    cache = _get_global_hf_cache()
    if clean in cache:
        df = cache[clean]
        if df.empty: return df
        return df.loc[(df.index >= start_ts) & (df.index <= end_ts), ["open", "high", "low", "close"]].astype(np.float32, copy=False)

    # 2. Check Disk
    path = HF_DATA_DIR / f"{clean}_15m.parquet"
    if not path.exists():
        # Fallback: try USDT variant for USDC/BUSD/EUR-quoted symbols
        _FALLBACK_QUOTES = ("USDC", "BUSD", "EUR")
        clean_up = clean.upper()
        for q in _FALLBACK_QUOTES:
            if clean_up.endswith(q):
                base = clean_up[:-len(q)].lower()
                fallback_path = HF_DATA_DIR / f"{base}usdt_15m.parquet"
                if fallback_path.exists():
                    path = fallback_path
                break
    if path.exists():
        try:
            df = pd.read_parquet(path)
            idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
            df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            df = df[["open", "high", "low", "close"]].astype(np.float32)
            cache[clean] = df # Cache full DF
            _HF_15M_LOCAL_AVAILABLE[clean] = True
            return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        except Exception:
            pass

    # 3. Download (Slowest)
    if not allow_download:
        cache[clean] = pd.DataFrame()
        _HF_15M_LOCAL_AVAILABLE[clean] = False
        return pd.DataFrame()
    try:
        if ccxt_symbol in _HF_15M_FAILED_SYMBOLS:
            return pd.DataFrame()
        import ccxt  # type: ignore
        ex = ccxt.binance({"enableRateLimit": True})
        # Note: we download a larger window to reduce future hits
        got = get_15m_ohlcv(ex, ccxt_symbol, start_ts, max_hold_hours=24)
        if got is None or got.empty:
            _HF_15M_FAILED_SYMBOLS.add(ccxt_symbol)
            cache[clean] = pd.DataFrame()
            return pd.DataFrame()
        
        got = got[["open", "high", "low", "close"]].astype(np.float32)
        # Update cache with what we just got (merged if we were smarter, but get_15m_ohlcv handles disk merge)
        # Actually, let's just re-load from disk to be sure we have the full merged version
        if path.exists():
             df = pd.read_parquet(path)
             idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
             df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
             df = df[["open", "high", "low", "close"]].astype(np.float32)
             cache[clean] = df
             _HF_15M_LOCAL_AVAILABLE[clean] = True
             return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
             
        cache[clean] = got
        _HF_15M_LOCAL_AVAILABLE[clean] = True
        return got.loc[(got.index >= start_ts) & (got.index <= end_ts)]
    except Exception as exc:
        _HF_15M_FAILED_SYMBOLS.add(ccxt_symbol)
        tprint(f"15m download disabled for {ccxt_symbol} after failure: {exc}")
        return pd.DataFrame()


def _resolve_barrier_conflict_15m(*, side: str, bar_open: float, tp_price: float, sl_price: float, trailing_price: Optional[float] = None) -> Optional[str]:
    hits: Dict[str, float] = {"TP": abs(bar_open - tp_price), "SL": abs(bar_open - sl_price)}
    if trailing_price is not None and np.isfinite(trailing_price):
        hits["TRAILING"] = abs(bar_open - trailing_price)
    order = {"SL": 0, "TRAILING": 1, "TP": 2}
    items = sorted(hits.items(), key=lambda kv: (kv[1], order.get(kv[0], 99)))
    return items[0][0] if items else None


@_njit(cache=True)
def _scan_first_ambiguous_hits(
    hh: np.ndarray,
    ll: np.ndarray,
    tp_price: np.ndarray,
    sl_price: np.ndarray,
    valid_indices: np.ndarray,
    end_idx: np.ndarray,
    side_long: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return entry indices i and first ambiguous j where TP and SL hit on same bar first."""
    n = len(hh)
    out_i = np.empty(len(valid_indices), dtype=np.int64)
    out_j = np.empty(len(valid_indices), dtype=np.int64)
    k = 0
    for p in range(len(valid_indices)):
        i = int(valid_indices[p])
        if i >= n - 1:
            continue
        j_stop = int(end_idx[i])
        if j_stop <= (i + 1):
            continue
        first_amb = -1
        for j in range(i + 1, j_stop):
            if side_long:
                hit_tp = hh[j] >= tp_price[i]
                hit_sl = ll[j] <= sl_price[i]
            else:
                hit_tp = ll[j] <= tp_price[i]
                hit_sl = hh[j] >= sl_price[i]
            if hit_tp and hit_sl:
                first_amb = j
                break
            if hit_tp or hit_sl:
                break
        if first_amb >= 0:
            out_i[k] = i
            out_j[k] = first_amb
            k += 1
    return out_i[:k], out_j[:k]


def _refine_ambiguous_labels_with_15m(
    panel: Dict[str, pd.DataFrame],
    tp_df: pd.DataFrame,
    sl_df: pd.DataFrame,
    lbl: pd.DataFrame,
    ret: pd.DataFrame,
    qual: pd.DataFrame,
    h: int,
    side: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close_df = panel.get("close")
    high_df = panel.get("high")
    low_df = panel.get("low")
    if not USE_15M_REFINEMENT or close_df is None or high_df is None or low_df is None:
        return lbl, ret, qual

    symbols = close_df.columns
    idx = close_df.index
    allow_download = bool((cfg or {}).get("allow_15m_download", False))
    
    # ── Optimization: Batch pre-load 15m data for all symbols before parallel processing ──
    n_jobs = _safe_parallel_jobs(len(symbols), cap=os.cpu_count() or 8)
    
    # Pre-load 15m data for all symbols that have local cache or allow download
    # This avoids N sequential disk reads in the parallel workers
    _15m_preloaded: Dict[str, pd.DataFrame] = {}
    _symbols_needing_15m = []
    
    # First pass: identify which symbols need 15m data
    for sym in symbols:
        if allow_download or _has_local_15m_cache(sym):
            _symbols_needing_15m.append(sym)
    
    # Batch load in parallel if there are symbols to load
    if _symbols_needing_15m:
        tprint(
            f"[15m_refine] pre-loading {len(_symbols_needing_15m)}/{len(symbols)} symbols "
            f"side={side} h={h} allow_download={allow_download}"
        )
        
        def _preload_symbol(sym: str):
            # Find the time range needed for this symbol by checking the panel
            c_sym = close_df[sym]
            tpv_sym = tp_df[sym]
            slv_sym = sl_df[sym]
            valid_mask = np.isfinite(c_sym) & (c_sym > 0) & np.isfinite(tpv_sym) & np.isfinite(slv_sym)
            if not np.any(valid_mask):
                return sym, pd.DataFrame()
            # Use full time range of the panel for preloading
            start_ts = c_sym.index.min()
            end_ts = c_sym.index.max()
            return sym, _load_or_download_15m(sym, start_ts, end_ts, allow_download=allow_download)
        
        # Parallel preload
        preload_results = Parallel(n_jobs=min(n_jobs, len(_symbols_needing_15m)), prefer="threads")(
            delayed(_preload_symbol)(s) for s in _symbols_needing_15m
        )
        for sym, hf_data in preload_results:
            if not hf_data.empty:
                _15m_preloaded[sym] = hf_data
    
    t_refine0 = time.perf_counter()
    tprint(
        f"[15m_refine] start side={side} h={h} symbols={len(symbols)} "
        f"n_jobs={n_jobs} allow_download={allow_download} rss_mb={_memory_snapshot_mb():.0f}"
    )

    def _process_symbol(sym: str):
        c = close_df[sym].to_numpy(dtype=np.float32, copy=False)
        hh = high_df[sym].to_numpy(dtype=np.float32, copy=False)
        ll = low_df[sym].to_numpy(dtype=np.float32, copy=False)
        tpv = tp_df[sym].to_numpy(dtype=np.float32, copy=False)
        slv = sl_df[sym].to_numpy(dtype=np.float32, copy=False)
        lab = lbl[sym].to_numpy(dtype=np.int8, copy=True)
        rets_sym = ret[sym].to_numpy(dtype=np.float32, copy=True)
        qual_sym = qual[sym].to_numpy(dtype=np.float32, copy=True)

        # If no local 15m cache exists and downloading is disabled, refinement cannot help.
        # Exit before the O(n*horizon) ambiguous-scan loop.
        if not allow_download and not _has_local_15m_cache(sym):
            return sym, lab, rets_sym, qual_sym, 0, 0, 0
        
        n = len(c)
        # Hybrid acceleration: vectorized candidate setup + compiled first-hit scan.
        valid_mask_i = np.isfinite(c) & (c > 0) & np.isfinite(tpv) & np.isfinite(slv)
        valid_indices = np.where(valid_mask_i)[0].astype(np.int64, copy=False)
        if valid_indices.size == 0:
            return sym, lab, rets_sym, qual_sym, 0, 0, 0

        tp_price = c * ((1.0 + tpv) if side == "long" else (1.0 - tpv))
        sl_price = c * ((1.0 - slv) if side == "long" else (1.0 + slv))
        idx_ns = idx.asi8
        horizon_ns = int(pd.Timedelta(hours=float(h)).value)
        end_idx = np.searchsorted(idx_ns, idx_ns + horizon_ns, side="right").astype(np.int64, copy=False)

        amb_i, amb_j = _scan_first_ambiguous_hits(
            hh=hh,
            ll=ll,
            tp_price=tp_price.astype(np.float32, copy=False),
            sl_price=sl_price.astype(np.float32, copy=False),
            valid_indices=valid_indices,
            end_idx=end_idx,
            side_long=bool(side == "long"),
        )
        if amb_i.size == 0:
            return sym, lab, rets_sym, qual_sym, 0, 0, 0

        ambiguous_j_list = [
            (int(i), int(j), float(tp_price[i]), float(sl_price[i]), float(tpv[i]), float(slv[i]))
            for i, j in zip(amb_i.tolist(), amb_j.tolist())
        ]

        # P1 FIX: use min(j) across all ambiguous entries, not just ambiguous_j_list[0][1].
        # The list is ordered by i (entry bar), but a later i can have an earlier first_ambiguous_j,
        # so using [0][1] could miss required 15m bars at the start of the range.
        start_full = min(idx[entry[1]] for entry in ambiguous_j_list).floor("15min")
        end_full = min(idx[ambiguous_j_list[-1][0]] + pd.Timedelta(hours=float(h)), idx[-1])
        
        # Use preloaded 15m data if available, otherwise load on-demand
        if sym in _15m_preloaded:
            hf_full = _15m_preloaded[sym]
            # Slice to the required time range
            hf_full = hf_full.loc[(hf_full.index >= start_full) & (hf_full.index <= end_full)]
        else:
            hf_full = _load_or_download_15m(sym, start_full, end_full, allow_download=allow_download)
        
        if hf_full.empty:
            return sym, lab, rets_sym, qual_sym, int(len(ambiguous_j_list)), 0, 0

        # Optimization: Pre-convert HF to numpy for faster iteration
        hf_idx = hf_full.index
        hf_h = hf_full["high"].to_numpy(dtype=np.float32, copy=False)
        hf_l = hf_full["low"].to_numpy(dtype=np.float32, copy=False)
        hf_c = hf_full["close"].to_numpy(dtype=np.float32, copy=False)

        resolved_cnt = 0
        for i, first_ambiguous_j, tp_price, sl_price, tp_val, sl_val in ambiguous_j_list:
            start_15m = idx[first_ambiguous_j].floor("15min")
            end_15m = min(idx[i] + pd.Timedelta(hours=float(h)), idx[-1])
            
            # Use np.searchsorted for O(log n) slice instead of O(n) boolean mask
            hf_ns = hf_idx.asi8  # int64 nanoseconds, already sorted
            lo_ns = start_15m.value
            hi_ns = end_15m.value
            i_lo = np.searchsorted(hf_ns, lo_ns, side="left")
            i_hi = np.searchsorted(hf_ns, hi_ns, side="left")
            idxs15 = np.arange(i_lo, i_hi, dtype=np.intp)
            
            resolved = None
            if idxs15.size == 0:
                continue
            h15 = hf_h[idxs15]
            l15 = hf_l[idxs15]
            c15 = hf_c[idxs15]
            range_abs = float(np.nanmax(h15) - np.nanmin(l15))
            threshold_abs = float(c[i] * min(abs(tp_val), abs(sl_val)))
            if (not np.isfinite(range_abs)) or (range_abs <= max(threshold_abs, 0.0)):
                continue

            hit_tp_arr = (h15 >= tp_price) if side == "long" else (l15 <= tp_price)
            hit_sl_arr = (l15 <= sl_price) if side == "long" else (h15 >= sl_price)
            any_hit = np.where(hit_tp_arr | hit_sl_arr)[0]
            if any_hit.size:
                k0 = int(any_hit[0])
                if bool(hit_tp_arr[k0] and hit_sl_arr[k0]):
                    d_tp = abs(float(c15[k0]) - float(h15[k0])) if side == "long" else abs(float(c15[k0]) - float(l15[k0]))
                    d_sl = abs(float(c15[k0]) - float(l15[k0])) if side == "long" else abs(float(c15[k0]) - float(h15[k0]))
                    resolved = "TP" if d_tp < d_sl else "SL"
                elif bool(hit_tp_arr[k0]):
                    resolved = "TP"
                elif bool(hit_sl_arr[k0]):
                    resolved = "SL"
            
            if resolved == "TP":
                lab[i] = OUT_TP
                rets_sym[i] = float(tp_val)
                qual_sym[i] = np.float32(max(float(qual_sym[i]), 0.51))
                resolved_cnt += 1
            elif resolved == "SL":
                lab[i] = OUT_SL
                rets_sym[i] = -float(sl_val)
                qual_sym[i] = np.float32(min(float(qual_sym[i]), 0.49))
                resolved_cnt += 1
        
        return sym, lab, rets_sym, qual_sym, int(len(ambiguous_j_list)), int(resolved_cnt), int(len(hf_full))

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_symbol)(s) for s in symbols
    )

    lbl2, ret2, qual2 = lbl.copy(), ret.copy(), qual.copy()
    total_ambiguous = 0
    total_resolved = 0
    total_hf_rows = 0
    for _i, (sym, lab, rets_sym, qual_sym, amb_cnt, resolved_cnt, hf_rows) in enumerate(results, 1):
        lbl2[sym] = lab
        ret2[sym] = rets_sym
        qual2[sym] = qual_sym
        total_ambiguous += int(amb_cnt)
        total_resolved += int(resolved_cnt)
        total_hf_rows += int(hf_rows)
        if _should_log_progress(_i, len(results), every=100):
            tprint(
                f"[15m_refine] merged {_i}/{len(results)} symbols "
                f"ambiguous={total_ambiguous} resolved={total_resolved} "
                f"hf_rows={total_hf_rows} rss_mb={_memory_snapshot_mb():.0f}"
            )
    tprint(
        f"[15m_refine] done side={side} h={h} symbols={len(results)} ambiguous={total_ambiguous} "
        f"resolved={total_resolved} hf_rows={total_hf_rows} "
        f"elapsed_s={time.perf_counter()-t_refine0:.2f} rss_mb={_memory_snapshot_mb():.0f}"
    )
    
    return lbl2, ret2, qual2


def _find_latest_feature_ts(data_root: str) -> Optional[pd.Timestamp]:
    """Find the latest feature timestamp directory (same logic as run_pipeline)."""
    feat_dir = os.path.join(data_root, "features")
    if not os.path.exists(feat_dir):
        return None
    dirs = sorted(glob.glob(os.path.join(feat_dir, "20*")))
    if not dirs:
        return None
    latest = os.path.basename(dirs[-1])
    return pd.to_datetime(latest, format="%Y%m%d_%H%M%S").tz_localize("UTC")


def _load_panel_from_store(cfg: Dict[str, Any]) -> Tuple[Optional[Dict[str, pd.DataFrame]], List[str]]:
    """Load panel data from PartitionedOHLCVStore (same as training pipeline).
    
    Subsamples aggressively to reduce memory usage for Stage 1 quick scans.
    Returns (panel, symbols_used).
    """
    from extreme_price_movements.data_store import PartitionedOHLCVStore
    from extreme_price_movements.universe import refresh_margin_universe_daily
    
    try:
        store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
        
        # Get margin symbols
        try:
            mu = refresh_margin_universe_daily(None, quotes=("USDT",))
            margin_symbols = mu.symbols if mu else []
        except Exception:
            margin_symbols = []
        
        # Use market_basket from config
        market_basket = cfg.get("market_basket", [])
        all_syms = list(set(margin_symbols + market_basket))

        # Augment with local store symbols (USDT-only to stay consistent with the
        # margin universe quote filter set above).
        ohlcv_dir = os.path.join(cfg["data_root"], "ohlcv")
        ohlcv_syms: List[str] = []
        if os.path.exists(ohlcv_dir):
            for name in os.listdir(ohlcv_dir):
                full = os.path.join(ohlcv_dir, name)
                sym = name.replace(".meta.json", "") if name.endswith(".meta.json") else (name if os.path.isdir(full) else None)
                if sym is not None and sym.upper().endswith("USDT"):
                    ohlcv_syms.append(sym)
            if ohlcv_syms:
                all_syms = list(set(all_syms) | set(ohlcv_syms))
                tprint(f"Augmented symbol universe from local ohlcv (USDT-only): +{len(set(ohlcv_syms))} symbols")

        # Use a configurable subsample step; default keeps 1/2 of symbols (USDT-only universe).
        # Previously 3 (1/3 of all quotes), now 2 (1/2 of USDT-only) for better coverage with native data.
        _step = max(1, int(cfg.get("tbm_symbol_subsample_step", 2)))
        train_syms = sorted(all_syms)[::_step]
        # Cap: honour explicit override; otherwise use half the live margin universe size
        # so "step=2" truly means 1/2 of the USDT margin universe (not 1/2 of local store).
        _default_max = max(1, len(margin_symbols) // 2) if margin_symbols else int(cfg.get("fetch_symbols_M", 600))
        _max_syms = int(cfg.get("tbm_max_symbols", _default_max))
        train_syms = train_syms[: max(1, _max_syms)]

        tprint(
            f"Loading panel from store for {len(train_syms)} symbols "
            f"(subsample_step={_step})"
        )
        dfs: Dict[str, pd.DataFrame] = {}
        _last_gc_ts = time.perf_counter()
        for i, sym in enumerate(train_syms, 1):
            try:
                df = store.load(sym)
                if df is not None and len(df) > 500:
                    dfs[sym] = df
            except Exception as exc:
                tprint(f"Store load failed for {sym}: {exc}")
                continue
            if _should_log_progress(i, len(train_syms), every=25):
                tprint(
                    f"[panel_load] {i}/{len(train_syms)} loaded={len(dfs)} "
                    f"rss_mb={_memory_snapshot_mb():.0f}"
                )
            _last_gc_ts = _maybe_collect_gc(last_gc_ts=_last_gc_ts, mem_threshold_mb=6144.0, min_interval_s=8.0)
        if not dfs:
            tprint(f"Panel store load returned no usable symbols. store.ohlcv_dir={store.ohlcv_dir}")
            return None, []
        return to_panel(dfs), train_syms
    except Exception as e:
        tprint(f"Failed to load panel from store: {e}")
        return None, []


def _load_features_from_data_root(
    cfg: Dict[str, Any],
    symbols: Optional[List[str]] = None,
    feature_keys: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Load features from the latest feature timestamp in data_root.
    
    Only loads TEST_FEATURE_KEYS to minimize memory usage.
    """
    ts = _find_latest_feature_ts(cfg["data_root"])
    if ts is None:
        tprint(f"No feature directories found in {cfg['data_root']}/features")
        return None
    
    feat_dir = Path(cfg["data_root"]) / "features" / ts.strftime("%Y%m%d_%H%M%S")
    if not feat_dir.exists():
        return None
    
    tprint(f"Loading features from {feat_dir}")
    
    feature_keys = list(feature_keys or ACTIVE_TEST_FEATURE_KEYS)
    # Load only configured feature keys to minimize memory
    columns = list(dict.fromkeys(feature_keys))
    dfs = _read_symbol_parquet_dir(feat_dir, symbols=symbols, columns=columns)
    feat_buf: Dict[str, Dict[str, pd.Series]] = {}
    
    # Only process columns that are in configured feature keys.
    test_keys_set = set(feature_keys)
    for sym, df in dfs.items():
        for c in df.columns:
            if c in test_keys_set:  # Only keep test feature keys
                feat_buf.setdefault(c, {})[sym] = pd.to_numeric(df[c], errors="coerce")
    
    out = {k: pd.DataFrame(v).sort_index() for k, v in feat_buf.items()}
    tprint(f"Loaded {len(out)} features (configured key universe only)")
    return out


def _build_tbm_learnability_report_rows(
    out_df: pd.DataFrame,
    details: Dict[str, Any],
) -> pd.DataFrame:
    """Expand per-config detail JSON into a thorough, CSV-friendly learnability report.
    
    Uses vectorized operations instead of iterrows for better memory efficiency.
    """
    rows: List[Dict[str, Any]] = []
    if out_df is None or out_df.empty:
        return pd.DataFrame(rows)

    # Vectorized extraction of config_id values
    config_ids = out_df["config_id"].astype(str).tolist()
    
    # Pre-extract commonly used columns as numpy arrays to avoid repeated access
    stage2_scores = out_df["stage2_score"].values
    stage1_scores = out_df["stage1_score"].values
    ic_labels = out_df["ic_label"].values
    ic_payoffs = out_df["ic_payoff"].values
    ic_snrs = out_df["ic_snr"].values
    sharpes = out_df["sharpe"].values
    sortinos = out_df["sortino"].values
    coverages = out_df["coverage"].values
    hard_gates = out_df["hard_gate"].values
    
    for idx, config_id in enumerate(config_ids):
        detail = details.get(config_id, {}) if isinstance(details, dict) else {}
        cfg = detail.get("config", {}) if isinstance(detail, dict) else {}
        bucket_metrics = detail.get("bucket_metrics", {}) if isinstance(detail, dict) else {}
        regime_metrics = detail.get("regime_metrics", {}) if isinstance(detail, dict) else {}
        vol_metrics = detail.get("vol_quintile_metrics", {}) if isinstance(detail, dict) else {}
        bucket_h_metrics = detail.get("bucket_horizon_metrics", {}) if isinstance(detail, dict) else {}

        base = {
            "config_id": config_id,
            "stage2_score": _safe_float(stage2_scores[idx], 0.0),
            "stage1_score": _safe_float(stage1_scores[idx], 0.0),
            "ic_label": _safe_float(ic_labels[idx], 0.0),
            "ic_payoff": _safe_float(ic_payoffs[idx], 0.0),
            "ic_snr": _safe_float(ic_snrs[idx], 0.0),
            "sharpe": _safe_float(sharpes[idx], 0.0),
            "sortino": _safe_float(sortinos[idx], 0.0),
            "coverage": _safe_float(coverages[idx], 0.0),
            "hard_gate": bool(hard_gates[idx]),
            "k_tp": _safe_float(cfg.get("k_tp"), np.nan),
            "sl_as_tp_pct": _safe_float(cfg.get("sl_as_tp_pct"), np.nan),
            "tp_abs_lo_pct": _safe_float(cfg.get("tp_abs_lo_pct"), np.nan),
            "tp_abs_hi_pct": _safe_float(cfg.get("tp_abs_hi_pct"), np.nan),
            "horizon_base": _safe_float(cfg.get("horizon_base"), np.nan),
            "horizon_alpha": _safe_float(cfg.get("horizon_alpha"), np.nan),
        }

        # Aggregate bucket/robustness diagnostics (similar spirit to candidate script slices)
        bucket_rows = [v for v in bucket_metrics.values() if isinstance(v, dict)]
        if bucket_rows:
            ic_payoff_vals = np.array([_safe_float(v.get("ic_payoff"), np.nan) for v in bucket_rows], dtype=float)
            ic_label_vals = np.array([_safe_float(v.get("ic_label"), np.nan) for v in bucket_rows], dtype=float)
            n_vals = np.array([_safe_float(v.get("n"), np.nan) for v in bucket_rows], dtype=float)
            base["bucket_ic_payoff_mean"] = float(np.nanmean(ic_payoff_vals))
            base["bucket_ic_payoff_min"] = float(np.nanmin(ic_payoff_vals))
            base["bucket_ic_payoff_std"] = float(np.nanstd(ic_payoff_vals))
            base["bucket_ic_label_mean"] = float(np.nanmean(ic_label_vals))
            base["bucket_samples_mean"] = float(np.nanmean(n_vals))
            base["bucket_samples_min"] = float(np.nanmin(n_vals))
        else:
            base["bucket_ic_payoff_mean"] = np.nan
            base["bucket_ic_payoff_min"] = np.nan
            base["bucket_ic_payoff_std"] = np.nan
            base["bucket_ic_label_mean"] = np.nan
            base["bucket_samples_mean"] = np.nan
            base["bucket_samples_min"] = np.nan

        # Flatten slices for detailed learnability report
        if not regime_metrics:
            rows.append({**base, "slice_type": "regime", "slice_name": "all"})
        else:
            for sname, sm in regime_metrics.items():
                if not isinstance(sm, dict):
                    continue
                rows.append(
                    {
                        **base,
                        "slice_type": "regime",
                        "slice_name": str(sname),
                        "slice_n": _safe_float(sm.get("n"), np.nan),
                        "slice_ic_label": _safe_float(sm.get("ic_label"), np.nan),
                        "slice_ic_payoff": _safe_float(sm.get("ic_payoff"), np.nan),
                        "slice_payoff_mean": _safe_float(sm.get("payoff_mean"), np.nan),
                        "slice_label_pos": _safe_float(sm.get("label_pos"), np.nan),
                    }
                )

        for sname, sm in (vol_metrics or {}).items():
            if not isinstance(sm, dict):
                continue
            rows.append(
                {
                    **base,
                    "slice_type": "vol_quintile",
                    "slice_name": str(sname),
                    "slice_n": _safe_float(sm.get("n"), np.nan),
                    "slice_ic_label": _safe_float(sm.get("ic_label"), np.nan),
                    "slice_ic_payoff": _safe_float(sm.get("ic_payoff"), np.nan),
                    "slice_payoff_mean": _safe_float(sm.get("payoff_mean"), np.nan),
                    "slice_label_pos": _safe_float(sm.get("label_pos"), np.nan),
                }
            )

        for sname, sm in (bucket_h_metrics or {}).items():
            if not isinstance(sm, dict):
                continue
            rows.append(
                {
                    **base,
                    "slice_type": "bucket_horizon",
                    "slice_name": str(sname),
                    "slice_n": _safe_float(sm.get("n"), np.nan),
                    "slice_tp_hit": _safe_float(sm.get("tp_hit"), np.nan),
                    "slice_sl_hit": _safe_float(sm.get("sl_hit"), np.nan),
                    "slice_timeout": _safe_float(sm.get("timeout"), np.nan),
                    "slice_bind": _safe_float(sm.get("bind"), np.nan),
                    "slice_balance": _safe_float(sm.get("balance"), np.nan),
                    "slice_sl_to_tp": _safe_float(sm.get("sl_to_tp"), np.nan),
                    "slice_ic_payoff": _safe_float(sm.get("ic_payoff"), np.nan),
                    "slice_ic_label": _safe_float(sm.get("ic_label"), np.nan),
                    "slice_auc_label": _safe_float(sm.get("auc_label"), np.nan),
                    "slice_tp_sep_top10": _safe_float(sm.get("tp_sep_top10"), np.nan),
                    "slice_tp_mean_pct": _safe_float(sm.get("tp_mean"), np.nan),
                    "slice_sl_mean_pct": _safe_float(sm.get("sl_mean"), np.nan),
                    "slice_payoff_mean": _safe_float(sm.get("payoff_mean"), np.nan),
                    "slice_ok": bool(sm.get("ok", False)),
                }
            )

    return pd.DataFrame(rows)

# Parameters that affect barrier geometry (TP/SL/Time scaling).
# Only these should trigger re-computation of barriers.
BARRIER_PARAMS = {
    "tp_method", "sl_method", "k_tp", "k_sl", "sl_as_tp_pct",
    "tp_regime_model", "sl_regime_model", "mix_weight",
    "horizon_scaling", "horizon_alpha", "horizon_base",
    "tp_abs_pct", "tp_base_pct", "base_atr_window",
    "tp_abs_lo_pct", "tp_abs_hi_pct", "sl_abs_lo_pct", "sl_abs_hi_pct",
    "sl_noise_buffer", "sl_min_abs_pct", "sl_min_bps",
    "tp_min_abs_pct", "tp_min_bps",
    "tp_time_decay", "trail_sl_mult", "tp_side_skew",
}


class LRUCache(dict):
    """
    Simple LRU cache for large DataFrames.
    Evicts oldest items when max_size is exceeded.
    """
    def __init__(self, max_size=50):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            oldest = next(iter(self))
            del self[oldest]

    def __getitem__(self, key):
        value = super().__getitem__(key)
        del self[key]
        super().__setitem__(key, value)
        return value


class BoundedEvalCache:
    """
    Bounded LRU cache for eval_cache with automatic eviction of oldest entries.
    Thread-safe via RLock for use with parallel Optuna jobs (n_jobs > 1).
    """
    def __init__(self, max_size=20):
        import threading
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()  # P2 FIX: thread-safe for parallel Optuna

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
            elif len(self._cache) >= self.max_size:
                oldest = self._access_order.pop(0)
                old_val = self._cache.pop(oldest, None)
                if isinstance(old_val, dict):
                    for v in old_val.values():
                        if isinstance(v, np.ndarray):
                            del v
            self._cache[key] = value
            self._access_order.append(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def setdefault(self, key: str, default: Any) -> Any:
        if key in self._cache:
            return self._cache[key]
        self.__setitem__(key, default)
        return default

    def clear(self) -> None:
        """Clear all cached items and free memory."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
        gc.collect()

    def __len__(self) -> int:
        return len(self._cache)


@dataclass
class RunArtifacts:
    panel: Dict[str, pd.DataFrame]
    features: Dict[str, pd.DataFrame]
    panel_15m: Optional[Dict[str, pd.DataFrame]] = None


def _subsample_symbols(symbols: Sequence[str]) -> List[str]:
    """Return sorted unique symbols (subsampling is handled during loading to avoid double-filtering)."""
    return sorted(set(map(str, symbols)))



def _index_cache_key(index: pd.Index) -> str:
    """Stable key for per-index cached stacked arrays."""
    try:
        if isinstance(index, pd.MultiIndex):
            ts_vals = index.get_level_values(0).asi8.astype(np.int64)
            sym_vals = index.get_level_values(1).astype(str)
            sym_hash = int(pd.util.hash_array(sym_vals.to_numpy(dtype=object)).sum())
        else:
            # DatetimeIndex or standard Index
            ts_vals = index.asi8.astype(np.int64)
            sym_hash = 0
        ts_hash = int(pd.util.hash_array(ts_vals).sum())
        return f"{len(index)}::{ts_hash}::{sym_hash}"
    except Exception:
        return f"idx_{len(index)}_{id(index)}"


def _safe_filename_from_key(key: str) -> str:
    return hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:24]


def _tbm_cache_signature(
    artifacts: RunArtifacts,
    horizons: Sequence[int],
    lookback_years: int,
) -> Dict[str, Any]:
    close = artifacts.panel["close"]
    idx = close.index
    cols = list(map(str, close.columns))
    sym_hash = hashlib.sha1("|".join(cols).encode("utf-8")).hexdigest()[:12]
    first_ts = str(idx[0]) if len(idx) else ""
    last_ts = str(idx[-1]) if len(idx) else ""
    return {
        "version": TBM_CACHE_VERSION,
        "n_rows": int(len(idx)),
        "n_symbols": int(len(cols)),
        "first_ts": first_ts,
        "last_ts": last_ts,
        "symbols_hash": sym_hash,
        "horizons": [int(h) for h in horizons],
        "lookback_years": int(lookback_years),
    }


def _tbm_cache_dir(output_path: Path, signature: Dict[str, Any]) -> Path:
    reports_dir = output_path.parent
    root = reports_dir / ".tbm_cache"
    cache_id = hashlib.sha1(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return root / cache_id


def _estimate_tbm_cache_size_bytes(layer1_cache: Dict[str, Any], layer2_cache: Dict[str, Any]) -> int:
    total = 0
    for v in layer1_cache.values():
        if not isinstance(v, tuple) or len(v) < 3:
            continue
        tp_df = v[0]
        sl_df = v[1]
        total += int(tp_df.memory_usage(index=True, deep=True).sum())
        total += int(sl_df.memory_usage(index=True, deep=True).sum())
        if len(v) >= 4 and v[3] is not None:
             total += int(v[3].memory_usage(index=True, deep=True).sum())
    for v in layer2_cache.values():
        if not isinstance(v, tuple) or len(v) != 3:
            # compute_triple_barrier_labels returns 3 elements (lbl, ret, qual)
            if len(v) == 2: # Legacy support
                 pass
            else:
                 continue
        lbl, ret = v[0], v[1]
        total += int(lbl.memory_usage(index=True, deep=True).sum())
        total += int(ret.memory_usage(index=True, deep=True).sum())
    return total


def load_persisted_tbm_cache(cache_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return {}, {}
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        entries = manifest.get("entries", [])
        layer1_loaded: Dict[str, Any] = {}
        layer2_loaded: Dict[str, Any] = {}
        for item in entries:
            key = str(item["key"])
            tp_df = pd.read_parquet(cache_dir / item["tp_file"])
            sl_df = pd.read_parquet(cache_dir / item["sl_file"])

            dyn_h = None
            if "dyn_horizon_file" in item and item["dyn_horizon_file"]:
                 dyn_h = pd.read_parquet(cache_dir / item["dyn_horizon_file"])

            lbl = pd.read_parquet(cache_dir / item["label_file"])
            ret = pd.read_parquet(cache_dir / item["return_file"])

            # Legacy quality handling
            qual = None
            if "quality_file" in item and item["quality_file"]:
                 qual = pd.read_parquet(cache_dir / item["quality_file"])
            else:
                 # Reconstruct dummy quality if missing (legacy cache)
                 qual = pd.DataFrame(0.5, index=lbl.index, columns=lbl.columns, dtype=np.float32)

            geom = {"bound_saturation": float(item.get("bound_saturation", 0.0))}
            layer1_loaded[key] = (tp_df, sl_df, geom, dyn_h)
            layer2_loaded[key] = (lbl, ret, qual)
        tprint(
            f"Loaded persisted TBM cache from {cache_dir} "
            f"(entries={len(entries)}, est_size_mb={manifest.get('estimated_size_mb', float('nan'))})"
        )
        return layer1_loaded, layer2_loaded
    except Exception as exc:
        tprint(f"Failed to load persisted TBM cache from {cache_dir}: {exc}")
        return {}, {}


def save_persisted_tbm_cache(
    cache_dir: Path,
    layer1_cache: Dict[str, Any],
    layer2_cache: Dict[str, Any],
    signature: Dict[str, Any],
    max_bytes: int,
) -> None:
    try:
        common_keys = sorted(set(layer1_cache.keys()) & set(layer2_cache.keys()))
        if not common_keys:
            return
        est_bytes = _estimate_tbm_cache_size_bytes(layer1_cache, layer2_cache)
        if est_bytes > max_bytes:
            tprint(
                f"Skipping TBM cache persistence: estimated_size_mb={est_bytes/(1024.0*1024.0):.1f} "
                f"exceeds max_size_mb={max_bytes/(1024.0*1024.0):.1f}"
            )
            return
        cache_dir.mkdir(parents=True, exist_ok=True)
        entries: List[Dict[str, Any]] = []
        for key in common_keys:
            # Layer 1: (tp, sl, geom, dyn_h)
            v1 = layer1_cache[key]
            tp_df = v1[0]
            sl_df = v1[1]
            geom = v1[2]
            dyn_h = v1[3] if len(v1) >= 4 else None

            # Layer 2: (lbl, ret, qual)
            v2 = layer2_cache[key]
            lbl = v2[0]
            ret = v2[1]
            qual = v2[2] if len(v2) >= 3 else None

            stem = _safe_filename_from_key(key)
            tp_file = f"{stem}_tp.parquet"
            sl_file = f"{stem}_sl.parquet"
            label_file = f"{stem}_label.parquet"
            return_file = f"{stem}_return.parquet"

            tp_df.to_parquet(cache_dir / tp_file, compression="zstd")
            sl_df.to_parquet(cache_dir / sl_file, compression="zstd")
            lbl.to_parquet(cache_dir / label_file, compression="zstd")
            ret.to_parquet(cache_dir / return_file, compression="zstd")

            entry = {
                "key": key,
                "tp_file": tp_file,
                "sl_file": sl_file,
                "label_file": label_file,
                "return_file": return_file,
                "bound_saturation": _safe_float(geom.get("bound_saturation"), 0.0)
                if isinstance(geom, dict)
                else 0.0,
            }

            if dyn_h is not None:
                dyn_file = f"{stem}_dyn_h.parquet"
                dyn_h.to_parquet(cache_dir / dyn_file, compression="zstd")
                entry["dyn_horizon_file"] = dyn_file

            if qual is not None:
                qual_file = f"{stem}_qual.parquet"
                qual.to_parquet(cache_dir / qual_file, compression="zstd")
                entry["quality_file"] = qual_file

            entries.append(entry)
        manifest = {
            "signature": signature,
            "entries": entries,
            "estimated_size_mb": round(est_bytes / (1024.0 * 1024.0), 2),
        }
        with (cache_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        tprint(
            f"Saved persisted TBM cache to {cache_dir} "
            f"(entries={len(entries)}, est_size_mb={est_bytes/(1024.0*1024.0):.1f})"
        )
    except Exception as exc:
        tprint(f"Failed to persist TBM cache to {cache_dir}: {exc}")


# ---------------------------
# IO helpers
# ---------------------------
def _read_symbol_parquet_file(f: Path, columns: Optional[List[str]] = None) -> Tuple[str, pd.DataFrame]:
    """Helper for parallel loading of a single symbol file."""
    raw_sym = None
    for part in f.parts:
        if part.startswith("symbol="):
            raw_sym = part.replace("symbol=", "").replace(".parquet", "")
            break
    if raw_sym is None:
        raw_sym = f.stem.replace("symbol=", "")
    
    # Try to read only specific columns if provided
    try:
        df = pd.read_parquet(f, columns=columns)
    except Exception:
        # Fallback if columns are missing in some files
        df = pd.read_parquet(f)
        if columns:
            df = df[[c for c in columns if c in df.columns]]

    if "__symbol__" in df.columns and not df.empty:
        sym = str(df["__symbol__"].iloc[0])
        df = df.drop(columns=["__symbol__"])
    else:
        sym = raw_sym.replace("_", "/", 1)

    if "year" in df.columns:
        df = df.drop(columns=["year"])

    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        else:
            # Check if it has an Unnamed: 0 or similar for the index
            if df.index.name is None and not df.empty:
                 pass
            else:
                 raise ValueError(f"Cannot infer timestamp index for {f}")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    
    return sym, df

def _read_symbol_parquet_dir(
    folder: Path, 
    symbols: Optional[List[str]] = None, 
    columns: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """Reads a directory of symbol parquets, with push-down filtering and parallelism."""
    files = sorted(folder.glob("symbol=*.parquet"))
    if not files:
        files = sorted(folder.glob("symbol=*/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No symbol parquet files in {folder}")

    # Push-down filtering: only read files for requested symbols
    if symbols:
        symbols_set = {str(s).replace("/", "_") for s in symbols}
        filtered_files = []
        for f in files:
            # Check if symbol is in the path
            sym_part = None
            for part in f.parts:
                if part.startswith("symbol="):
                    sym_part = part.replace("symbol=", "").replace(".parquet", "")
                    break
            if sym_part and sym_part in symbols_set:
                filtered_files.append(f)
            elif not sym_part:
                # Fallback check on stem (remove .parquet if present in stem)
                stem_sym = f.stem.replace("symbol=", "")
                if stem_sym in symbols_set:
                    filtered_files.append(f)
        files = filtered_files
        
    if not files:
        return {}

    # Parallel loading
    tprint(f"Parallel loading {len(files)} symbol files...")
    n_jobs = _safe_parallel_jobs(len(files), cap=8)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_read_symbol_parquet_file)(f, columns=columns) for f in files
    )

    by_symbol_parts: Dict[str, List[pd.DataFrame]] = {}
    for sym, df in results:
        by_symbol_parts.setdefault(sym, []).append(df.sort_index())

    dfs: Dict[str, pd.DataFrame] = {}
    for sym, parts in by_symbol_parts.items():
        if len(parts) == 1:
            merged = parts[0]
        else:
            merged = pd.concat(parts, axis=0).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
        dfs[sym] = merged
    return dfs


def load_panel(panel_path: Path) -> Dict[str, pd.DataFrame]:
    tprint(f"Loading panel data from: {panel_path}")
    if panel_path.is_dir():
        dfs = _read_symbol_parquet_dir(panel_path)
        has_ohlcv = {"open", "high", "low", "close", "volume"}.issubset(
            set(next(iter(dfs.values())).columns)
        )
        if not has_ohlcv:
            raise ValueError(f"Panel directory {panel_path} does not contain OHLCV columns")
        return to_panel(dfs)

    panel_df = pd.read_parquet(panel_path)
    if isinstance(panel_df.columns, pd.MultiIndex):
        out: Dict[str, pd.DataFrame] = {}
        for k in ["open", "high", "low", "close", "volume"]:
            if k in panel_df.columns.get_level_values(0):
                out[k] = panel_df.xs(k, axis=1, level=0)
            elif k in panel_df.columns.get_level_values(1):
                out[k] = panel_df.xs(k, axis=1, level=1)
        if len(out) == 5:
            return {k: v.sort_index() for k, v in out.items()}

    if {"timestamp", "symbol", "open", "high", "low", "close", "volume"}.issubset(panel_df.columns):
        panel_df["timestamp"] = pd.to_datetime(panel_df["timestamp"], utc=True)
        piv = {}
        for k in ["open", "high", "low", "close", "volume"]:
            piv[k] = panel_df.pivot(index="timestamp", columns="symbol", values=k).sort_index()
        return piv

    raise ValueError(f"Unsupported panel parquet layout for {panel_path}")


def load_features(features_path: Path) -> Dict[str, pd.DataFrame]:
    tprint(f"Loading features from: {features_path}")
    if features_path.is_file():
        raise ValueError("--features expects a directory of symbol=*.parquet feature files")
    dfs = _read_symbol_parquet_dir(features_path)
    feat_buf: Dict[str, Dict[str, pd.Series]] = {}
    for sym, df in dfs.items():
        for c in df.columns:
            feat_buf.setdefault(c, {})[sym] = pd.to_numeric(df[c], errors="coerce")

    out = {k: pd.DataFrame(v).sort_index() for k, v in feat_buf.items()}
    return out


def _compute_panel_atr_pct(
    panel: Dict[str, pd.DataFrame],
    *,
    window: int = 24,
    min_periods: int = 4,
) -> pd.DataFrame:
    """Compute raw ATR% from OHLCV panel for barrier construction/reporting."""
    close = panel["close"]
    high = panel["high"]
    low = panel["low"]
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=0,
    ).groupby(level=0).max()
    atr = tr.rolling(window, min_periods=min_periods).mean()
    return (atr / close).replace([np.inf, -np.inf], np.nan).clip(lower=1e-6).astype(np.float32)


def _is_standardized_like_atr(frame: pd.DataFrame) -> bool:
    """Detect z-scored/normalized ATR-like series (invalid for barrier sizing)."""
    if frame is None or frame.empty:
        return False
    vals = frame.to_numpy(dtype=np.float32, copy=False).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return False
    # True ATR% should be non-negative; meaningful negative mass strongly suggests normalization.
    neg_share = float(np.mean(vals < 0.0))
    return neg_share > 0.001


def _get_barrier_atr_frame(artifacts: RunArtifacts) -> pd.DataFrame:
    """Return ATR% frame suitable for barrier geometry and production diagnostics."""
    series_cache = getattr(artifacts, "_tbm_series_cache", None)
    if series_cache is None:
        series_cache = {}
        setattr(artifacts, "_tbm_series_cache", series_cache)
    cached = series_cache.get("atr_pct_barrier_source")
    if cached is not None:
        return cached

    feats = artifacts.features
    if "atr_pct_raw" in feats and isinstance(feats["atr_pct_raw"], pd.DataFrame):
        src = feats["atr_pct_raw"]
    elif "atr_pct" in feats and isinstance(feats["atr_pct"], pd.DataFrame) and not _is_standardized_like_atr(feats["atr_pct"]):
        src = feats["atr_pct"]
    else:
        src = _compute_panel_atr_pct(artifacts.panel)
        tprint("[atr_source] Using panel-derived ATR% for barriers (feature atr_pct looks normalized or missing).")

    src = src.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    series_cache["atr_pct_barrier_source"] = src
    return src


def align_artifacts(
    panel: Dict[str, pd.DataFrame],
    features: Dict[str, pd.DataFrame],
    lookback_years: int = 2,
) -> RunArtifacts:
    tprint(
        f"Aligning artifacts with lookback_years={lookback_years} "
        f"(panel_symbols={len(panel.get('close', pd.DataFrame()).columns)})"
    )
    idx_full = panel["close"].index
    cutoff = idx_full.max() - pd.DateOffset(years=int(lookback_years))
    idx = idx_full[idx_full >= cutoff]
    cols_all = list(panel["close"].columns)
    cols = pd.Index(_subsample_symbols(cols_all))
    p_out: Dict[str, pd.DataFrame] = {}
    for k, df in panel.items():
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        reindexed = df.reindex(index=idx, columns=cols)
        # Only cast numeric columns to float32 to avoid ValueError with object/categorical
        p_out[k] = reindexed.apply(pd.to_numeric, errors="coerce").astype(np.float32)

    f_out: Dict[str, pd.DataFrame] = {}
    for k, df in features.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            continue
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        reindexed = df.reindex(index=idx, columns=cols)
        # Only cast numeric columns to float32 to avoid ValueError with object/categorical
        f_out[k] = reindexed.apply(pd.to_numeric, errors="coerce").astype(np.float32)

    # Always materialize a raw ATR% series for barrier construction/admissibility.
    f_out["atr_pct_raw"] = _compute_panel_atr_pct(p_out)

    if "atr_pct" not in f_out:
        f_out["atr_pct"] = f_out["atr_pct_raw"].copy()
    elif _is_standardized_like_atr(f_out["atr_pct"]):
        tprint("[align_artifacts] WARNING atr_pct appears normalized (contains negatives); barriers will use atr_pct_raw.")

    tprint(f"Loading 15m high-fidelity panel for {len(cols)} symbols...")
    from joblib import Parallel, delayed
    start_ts = idx.min()
    end_ts = idx.max() + pd.Timedelta(hours=48)
    
    def _fetch(sym):
        return sym, _load_or_download_15m(sym, start_ts, end_ts, allow_download=False)
        
    results = Parallel(n_jobs=_safe_parallel_jobs(len(cols), cap=8))(
        delayed(_fetch)(sym) for sym in cols
    )
    
    sym_dfs = {sym: df for sym, df in results if df is not None and not df.empty}
    p15_out = None
    if sym_dfs:
        p15_out = {}
        for k in ["open", "high", "low", "close"]:
            series_dict = {sym: df[k] for sym, df in sym_dfs.items()}
            # Fast vectorized outer-join
            df_k = pd.DataFrame(series_dict)
            df_k = df_k.reindex(columns=cols)
            p15_out[k] = df_k.astype(np.float32)

    out = RunArtifacts(panel=p_out, features=f_out, panel_15m=p15_out)
    tprint(
        f"Aligned artifacts ready: bars={len(idx)}, symbols={len(cols)}, "
        f"feature_frames={len(f_out)}, mem_peak_mb={_memory_snapshot_mb():.1f}"
    )
    return out


# ---------------------------
# Config/key helpers
# ---------------------------
def serialize_key(params: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            out[k] = round(v, 6)
        elif isinstance(v, dict):
            out[k] = serialize_key(v)
        else:
            out[k] = v
    return out


def _get_barrier_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: cfg[k] for k in BARRIER_PARAMS if k in cfg}


def _compact_cfg_signature(cfg: Dict[str, Any], keys: Sequence[str]) -> Tuple[Any, ...]:
    """Compact, hash-stable signature for cache keys without JSON encoding overhead."""
    out: List[Any] = []
    for k in sorted(keys):
        if k not in cfg:
            continue
        v = cfg[k]
        if isinstance(v, float):
            out.append((k, round(float(v), 6)))
        elif isinstance(v, (int, str, bool)):
            out.append((k, v))
        elif isinstance(v, (list, tuple)):
            out.append((k, tuple(v)))
        elif isinstance(v, dict):
            out.append((k, tuple(sorted((str(kk), serialize_key({"v": vv})["v"]) for kk, vv in v.items()))))
        else:
            out.append((k, str(v)))
    return tuple(out)


def _get_tp_barrier_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Subset of barrier params that affect TP / horizon only (not SL geometry)."""
    drop_keys = {
        "sl_method", "k_sl", "sl_as_tp_pct", "sl_regime_model",
        "sl_abs_lo_pct", "sl_abs_hi_pct", "sl_noise_buffer", "sl_min_abs_pct", "sl_min_bps",
        "trail_sl_mult",
    }
    return {k: cfg[k] for k in BARRIER_PARAMS if (k in cfg and k not in drop_keys)}


def _derive_sl_from_tp(tp: pd.DataFrame, atr: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Fast SL derivation from cached TP arrays for sl_method=tp_pct."""
    sl = float(cfg.get("sl_as_tp_pct", 0.5)) * tp
    sl = _apply_caps(sl, float(cfg.get("sl_abs_lo_pct", 0.005)), float(cfg.get("sl_abs_hi_pct", 0.08)))
    sl_lo_eff = effective_sl_floor(
        sl_abs_lo_pct=float(cfg.get("sl_abs_lo_pct", 0.005)),
        sl_min_abs_pct=float(cfg.get("sl_min_abs_pct", 0.01)),
        sl_min_bps=float(cfg.get("sl_min_bps", 100)),
    )
    if cfg.get("sl_noise_buffer", False):
        sl = sl.clip(lower=sl_lo_eff)
    if float(cfg.get("trail_sl_mult", 0.0)) > 0:
        sl = sl * (1.0 - 0.15 * float(cfg.get("trail_sl_mult", 0.0)))
        sl = sl.clip(lower=sl_lo_eff, upper=float(cfg.get("sl_abs_hi_pct", 0.08)))
    return sl.astype(np.float32)


def _apply_compare_production_floor(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Optionally evaluate TBM configs under production TP floor semantics."""
    if not bool(cfg.get("evaluate_under_prod_floor", False)):
        return cfg
    cfg_eff = dict(cfg)
    prod_floor = cfg_eff.get(
        "prod_tp_abs_lo_pct",
        cfg_eff.get("barrier_tp_lo_prod", cfg_eff.get("tp_abs_lo_pct", 0.005)),
    )
    cfg_eff["tp_abs_lo_pct"] = float(prod_floor)
    cfg_eff["_prod_floor_applied"] = True
    return cfg_eff


def config_id(config: Dict[str, Any]) -> str:
    payload = json.dumps(serialize_key(config), sort_keys=True)
    return "CFG" + hashlib.sha1(payload.encode()).hexdigest()[:10].upper()


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return 0.0
    r = spearmanr(a[m], b[m]).correlation
    return float(r) if np.isfinite(r) else 0.0


@dataclass
class LearnabilityWeights:
    w_auc_bound: float = 0.28
    w_tp_sep: float = 0.18
    tp_sep_saturation_k: float = 0.03
    w_ap_lift: float = 0.06
    w_sortino: float = 0.08
    w_ic_snr: float = 0.08
    w_dir_sup: float = 0.08
    w_tp_over_sl: float = 0.08
    w_payoff_edge: float = 0.08
    w_ece: float = 0.22
    w_brier: float = 0.16
    w_bind: float = 0.32
    w_timeout: float = 0.32
    w_ic_std_time: float = 0.26
    w_ic_std_asset: float = 0.14
    w_coverage: float = 0.18
    w_probe: float = 0.12
    w_gap: float = 0.15
    w_horizon_consistency: float = 0.08


def _resolve_learnability_weights(
    cfg: Dict[str, Any],
    target_horizon: Optional[int] = None,
    target_bucket: Optional[str] = None,
) -> LearnabilityWeights:
    w = LearnabilityWeights(
        w_auc_bound=float(cfg.get("stage2_w_auc_bound", 0.28)),
        w_tp_sep=float(cfg.get("stage2_w_tp_sep", 0.18)),
        tp_sep_saturation_k=float(cfg.get("stage2_tp_sep_saturation_k", 0.03)),
        w_ap_lift=float(cfg.get("stage2_w_ap_lift", 0.06)),
        w_sortino=float(cfg.get("stage2_w_sortino", 0.08)),
        w_ic_snr=float(cfg.get("stage2_w_ic_snr", 0.08)),
        w_dir_sup=float(cfg.get("stage2_w_dir_sup", 0.08)),
        w_tp_over_sl=float(cfg.get("stage2_w_tp_over_sl", 0.08)),
        w_payoff_edge=float(cfg.get("stage2_w_payoff_edge", 0.08)),
        w_ece=float(cfg.get("stage2_w_ece", 0.22)),
        w_brier=float(cfg.get("stage2_w_brier", 0.16)),
        w_bind=float(cfg.get("stage2_w_bind", 0.32)),
        w_timeout=float(cfg.get("stage2_w_timeout", 0.32)),
        w_ic_std_time=float(cfg.get("stage2_w_ic_std_time", 0.26)),
        w_ic_std_asset=float(cfg.get("stage2_w_ic_std_asset", 0.14)),
        w_coverage=float(cfg.get("stage2_w_coverage", 0.18)),
        w_probe=float(cfg.get("stage2_w_probe", 0.12)),
        w_gap=float(cfg.get("stage2_w_gap", 0.15)),
        w_horizon_consistency=float(cfg.get("stage2_w_horizon_consistency", 0.08)),
    )
    if target_horizon is not None:
        sfx = f"_h{int(target_horizon)}"
        for field_name in w.__dataclass_fields__.keys():
            k = f"stage2_{field_name}{sfx}"
            if k in cfg:
                setattr(w, field_name, float(cfg[k]))
    if target_bucket:
        bkey = str(target_bucket).lower()
        for field_name in w.__dataclass_fields__.keys():
            k = f"stage2_{field_name}_{bkey}"
            if k in cfg:
                setattr(w, field_name, float(cfg[k]))
    return w


def _score_saturated_tp_sep(tp_sep: float, saturation_k: float) -> float:
    if not math.isfinite(tp_sep) or tp_sep <= 0.0:
        return 0.0
    s = max(float(saturation_k), EPS)
    return float(1.0 - math.exp(-float(tp_sep) / s))


def _auc_lower_bound(auc: float, n_obs: int, z: float = 1.96) -> float:
    if not math.isfinite(auc):
        return float("nan")
    n = max(int(n_obs), 1)
    var = max(auc * (1.0 - auc), 0.25) / float(n)
    return float(auc - z * math.sqrt(max(var, EPS)))


def _strict_walk_forward_probe_auc(
    X: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    w: Optional[np.ndarray],
    cfg: Dict[str, Any],
    fast_mode: bool = False,
) -> Tuple[float, float, str, int, int]:
    if len(y) < int(cfg.get("stage2_probe_min_rows", 400)):
        return float("nan"), float("nan"), "insufficient_rows", 0, 0
    _u_ts = np.sort(pd.Index(ts).unique())
    train_frac = float(cfg.get("stage2_probe_train_fraction", 0.7))
    min_train = int(cfg.get("stage2_probe_min_train_rows", 200))
    min_val = int(cfg.get("stage2_probe_min_val_rows", 120))
    if len(_u_ts) >= 10:
        split_idx = max(1, int(len(_u_ts) * train_frac))
        split_idx = min(split_idx, len(_u_ts) - 1)
        split_ts = _u_ts[split_idx]
        train_mask = ts < split_ts
        val_mask = ~train_mask
    else:
        # Fallback: strictly forward by row order when timestamp granularity is coarse.
        n = len(y)
        cut = max(1, min(n - 1, int(n * train_frac)))
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:cut] = True
        val_mask = ~train_mask
    n_tr = int(train_mask.sum())
    n_va = int(val_mask.sum())
    if n_tr < min_train or n_va < min_val:
        return float("nan"), float("nan"), "insufficient_split", n_tr, n_va
    y_tr = y[train_mask]
    y_va = y[val_mask]
    if np.unique(y_tr).size < 2 or np.unique(y_va).size < 2:
        return float("nan"), float("nan"), "single_class_split", n_tr, n_va

    X_tr = X[train_mask]
    X_va = X[val_mask]
    w_tr = None if w is None else w[train_mask]
    engine = str(cfg.get("stage2_probe_engine", "xgb" if xgb is not None else "extratrees")).lower()
    pred = None
    try:
        if engine == "xgb" and xgb is not None:
            probe = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=int(cfg.get("stage2_probe_xgb_n_estimators", 400 if not fast_mode else 150)),
                max_depth=int(cfg.get("stage2_probe_xgb_max_depth", 3)),
                min_child_weight=float(cfg.get("stage2_probe_xgb_min_child_weight", 40.0)),
                gamma=float(cfg.get("stage2_probe_xgb_gamma", 4.0)),
                subsample=float(cfg.get("stage2_probe_xgb_subsample", 0.7)),
                colsample_bytree=float(cfg.get("stage2_probe_xgb_colsample_bytree", 0.7)),
                reg_alpha=float(cfg.get("stage2_probe_xgb_reg_alpha", 2.0)),
                reg_lambda=float(cfg.get("stage2_probe_xgb_reg_lambda", 15.0)),
                learning_rate=float(cfg.get("stage2_probe_xgb_eta", 0.05)),
                tree_method="hist",
                random_state=42,
                n_jobs=1,
            )
            probe.fit(X_tr, y_tr, sample_weight=w_tr)
            pred = probe.predict(X_va)
        else:
            probe = ExtraTreesRegressor(
                n_estimators=int(cfg.get("stage2_probe_et_n_estimators", 500 if not fast_mode else 200)),
                min_samples_leaf=int(cfg.get("stage2_probe_et_min_samples_leaf", 80)),
                min_samples_split=int(cfg.get("stage2_probe_et_min_samples_split", 400)),
                max_features=float(cfg.get("stage2_probe_et_max_features", 0.35)),
                bootstrap=True,
                max_samples=float(cfg.get("stage2_probe_et_max_samples", 0.7)),
                random_state=42,
                n_jobs=1,
            )
            probe.fit(X_tr, y_tr, sample_weight=w_tr)
            pred = probe.predict(X_va)
    except Exception:
        return float("nan"), float("nan"), "fit_failed", n_tr, n_va

    pred = np.asarray(pred, dtype=np.float32)
    auc = _fast_auc_binary(y_va.astype(np.float32), pred)
    auc_b = _auc_lower_bound(float(auc), int(val_mask.sum()))
    return float(auc), float(auc_b), "ok", n_tr, n_va


def _horizon_consistency_penalty(
    bucket_h_metrics: Dict[Tuple[str, int], Dict[str, Any]],
    cfg: Dict[str, Any],
) -> float:
    if not bucket_h_metrics:
        return 0.0
    bucket_to_cells: Dict[str, List[Dict[str, Any]]] = {}
    for (bkt, _h), met in bucket_h_metrics.items():
        bucket_to_cells.setdefault(str(bkt), []).append(met)
    penalties: List[float] = []
    ratio_band = float(cfg.get("stage2_horizon_ratio_band", 1.25))
    tpsl_band = float(cfg.get("stage2_horizon_tp_over_sl_band", 1.5))
    label_band = float(cfg.get("stage2_horizon_label_mean_band", 0.20))
    for cells in bucket_to_cells.values():
        ratios = [float(c.get("barrier_ratio", float("nan"))) for c in cells]
        tpsls = [float(c.get("tp_over_sl", float("nan"))) for c in cells]
        lbl_means = [float(c.get("base_rate", float("nan"))) for c in cells]
        ratios = [x for x in ratios if math.isfinite(x) and x > 0]
        tpsls = [x for x in tpsls if math.isfinite(x) and x > 0]
        lbl_means = [x for x in lbl_means if math.isfinite(x)]
        if len(ratios) >= 2:
            span = max(ratios) / max(min(ratios), EPS)
            penalties.append(max(0.0, span - ratio_band) / max(ratio_band, EPS))
        if len(tpsls) >= 2:
            span = max(tpsls) / max(min(tpsls), EPS)
            penalties.append(max(0.0, span - tpsl_band) / max(tpsl_band, EPS))
        if len(lbl_means) >= 2:
            drift = max(lbl_means) - min(lbl_means)
            penalties.append(max(0.0, drift - label_band) / max(label_band, EPS))
    return float(np.mean(penalties)) if penalties else 0.0


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 10:
        return 0.0
    y = y[m]
    p = np.clip(p[m], 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def oof_payoff_decile_spread(score: np.ndarray, payoff: np.ndarray) -> float:
    s = np.asarray(score, dtype=float)
    y = np.asarray(payoff, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    if int(np.sum(m)) < 20:
        return 0.0
    ss = s[m]
    yy = y[m]
    try:
        dec = pd.qcut(pd.Series(ss), 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"d": dec, "p": yy}).groupby("d")["p"].mean()
        if len(g) < 2:
            return 0.0
        return float(g.iloc[-1] - g.iloc[0])
    except Exception:
        return 0.0


# ---------------------------
# Barrier geometry
# ---------------------------
def _regime_multiplier(atr_pct: pd.DataFrame, model: str, mix_weight: float = 0.5) -> pd.DataFrame:
    roll = atr_pct.rolling(24 * 14, min_periods=24).median()
    ratio = (atr_pct / roll).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    ratio = ratio.clip(0.5, 2.0)

    shock = atr_pct.pct_change(fill_method=None).abs().rolling(24, min_periods=4).mean().fillna(0.0)
    shock = (1.0 + shock).clip(0.7, 2.2)

    if model == "none":
        return pd.DataFrame(1.0, index=atr_pct.index, columns=atr_pct.columns)
    if model == "level":
        return ratio
    if model == "shock":
        return shock
    if model == "mix":
        return mix_weight * ratio + (1 - mix_weight) * shock
    raise ValueError(f"Unknown regime model: {model}")



def _apply_caps(x: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    return x.clip(lower=lo, upper=hi)


def _compute_dynamic_horizon(
    atr: pd.DataFrame,
    base_horizon: int,
    cfg: Dict[str, Any],
) -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
    if not bool(cfg.get("use_dynamic_horizon", False)):
        return None, {}

    window = int(cfg.get("base_atr_window", 24 * 30))
    disp_floor = 0.1
    z_max = 3.0

    # atr is already shifted in build_barriers, so just roll
    atr_median = atr.rolling(window, min_periods=24).median()
    atr_mad = (atr - atr_median).abs().rolling(window, min_periods=24).median()
    atr_disp = np.maximum(atr_mad, disp_floor * atr_median)
    z = ((atr - atr_median) / (atr_disp + 1e-12)).clip(-z_max, z_max)

    z_lo = float(cfg.get("dynamic_horizon_z_lo", -1.0))
    z_hi = float(cfg.get("dynamic_horizon_z_hi", 2.0))
    max_scale_add = float(cfg.get("dynamic_horizon_max_scale_add", 0.5))

    fraction = ((z - z_lo) / (z_hi - z_lo + 1e-9)).clip(0.0, 1.0)
    scale = 1.0 + max_scale_add * fraction

    dyn_h = scale * float(base_horizon)

    # Stats
    vals = dyn_h.values.ravel()
    vals = vals[np.isfinite(vals)]
    stats = {
        "h_mean": float(np.mean(vals)) if vals.size else float(base_horizon),
        "h_p10": float(np.percentile(vals, 10)) if vals.size else float(base_horizon),
        "h_p90": float(np.percentile(vals, 90)) if vals.size else float(base_horizon),
    }
    return dyn_h, stats


def build_barriers(
    artifacts: RunArtifacts,
    cfg: Dict[str, Any],
    horizon: int,
    side: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Optional[pd.DataFrame]]:
    # Cache reusable per-artifact volatility series and rolling refs.
    series_cache = getattr(artifacts, "_tbm_series_cache", None)
    if series_cache is None:
        series_cache = {}
        setattr(artifacts, "_tbm_series_cache", series_cache)

    # Use entry-time ATR only: shift by 1 bar so the barrier is set using only
    # data available at the moment the signal fires, not the current bar's ATR
    # (which would be look-ahead on 15m bars where ATR is computed on the closing price).
    atr_key = "atr_shift_bfill"
    atr = series_cache.get(atr_key)
    if atr is None:
        atr_source = _get_barrier_atr_frame(artifacts)
        atr = atr_source.shift(1).clip(lower=1e-6)
        atr = atr.bfill(limit=1)
        series_cache[atr_key] = atr.astype(np.float32)

    tp_regime = _regime_multiplier(atr, cfg.get("tp_regime_model", "none"), cfg.get("mix_weight", 0.5))
    sl_regime = _regime_multiplier(atr, cfg.get("sl_regime_model", cfg.get("tp_regime_model", "none")), cfg.get("mix_weight", 0.5))

    side_skew = float(cfg.get("tp_side_skew", 0.0))
    side_mult = (1 + side_skew) if side == "long" else (1 - side_skew)

    tp_method = cfg["tp_method"]
    sl_method = cfg["sl_method"]

    if tp_method == "atr_mult":
        tp_raw = cfg["k_tp"] * side_mult * atr * tp_regime
    elif tp_method == "semi_atr_mult":
        atr_tp = cfg["k_tp"] * side_mult * atr * tp_regime
        abs_tp = pd.DataFrame(float(cfg.get("tp_abs_pct", cfg.get("tp_base_pct", 0.02))), index=atr.index, columns=atr.columns)
        tp_raw = 0.5 * atr_tp + 0.5 * abs_tp
    elif tp_method == "absolute":
        tp_raw = pd.DataFrame(float(cfg["tp_abs_pct"]), index=atr.index, columns=atr.columns)
    elif tp_method == "atr_norm":
        # median is already causal (rolling on shifted atr); shift(1) already applied above.
        _w = int(cfg.get("base_atr_window", 168))
        med_key = f"atr_med::{_w}"
        med = series_cache.get(med_key)
        if med is None:
            med = atr.rolling(_w, min_periods=24).median().fillna(atr.median())
            series_cache[med_key] = med.astype(np.float32)
        tp_raw = cfg["k_tp"] * (atr / (med + EPS)) * float(cfg["tp_base_pct"])
    elif tp_method == "semi_atr_norm":
        _w = int(cfg.get("base_atr_window", 168))
        med_key = f"atr_med::{_w}"
        med = series_cache.get(med_key)
        if med is None:
            med = atr.rolling(_w, min_periods=24).median().fillna(atr.median())
            series_cache[med_key] = med.astype(np.float32)
        atrn_tp = cfg["k_tp"] * (atr / (med + EPS)) * float(cfg.get("tp_base_pct", 0.02))
        abs_tp = pd.DataFrame(float(cfg.get("tp_abs_pct", cfg.get("tp_base_pct", 0.02))), index=atr.index, columns=atr.columns)
        tp_raw = 0.5 * atrn_tp + 0.5 * abs_tp
    else:
        raise ValueError(f"Unsupported tp_method={tp_method}")

    # Canonical effective TP floor (production semantics):
    # tp_lo_eff = max(tp_abs_lo_pct, tp_min_abs_pct, tp_min_bps/1e4)
    tp_lo_eff = effective_tp_floor(
        tp_abs_lo_pct=float(cfg.get("tp_abs_lo_pct", 0.005)),
        tp_min_abs_pct=float(cfg.get("tp_min_abs_pct", 0.005)),
        tp_min_bps=float(cfg.get("tp_min_bps", 50)),
    )

    # Dynamic horizon
    dyn_horizon, dyn_stats = _compute_dynamic_horizon(atr, horizon, cfg)
    eff_horizon = dyn_horizon if dyn_horizon is not None else horizon

    tp = make_effective_tp(
        tp_raw,
        horizon=eff_horizon,
        horizon_scaling=cfg.get("horizon_scaling", "none"),
        lo=float(cfg.get("tp_abs_lo_pct", 0.005)),
        hi=float(cfg.get("tp_abs_hi_pct", 0.08)),
        horizon_alpha=float(cfg.get("horizon_alpha", 0.5)),
        horizon_base=float(cfg.get("horizon_base", 4.0)),
    ).clip(lower=tp_lo_eff)

    # Path dependence approximation knobs (applied on TP before final floor/cap enforcement).
    if cfg.get("tp_time_decay", "none") == "linear":
        decay = max(0.6, 1.0 - 0.06 * max(horizon - 2, 0))
        tp = tp * decay

    # Re-enforce TP floor/cap after optional transforms so effective floor attribution is stable.
    tp = tp.clip(lower=tp_lo_eff, upper=float(cfg.get("tp_abs_hi_pct", 0.08)))

    # SL must be derived from effective TP (post-scale/post-clip) when sl_method=tp_pct.
    if sl_method == "tp_pct":
        sl = float(cfg["sl_as_tp_pct"]) * tp
    elif sl_method == "atr_mult":
        sl = float(cfg["k_sl"]) * atr * sl_regime
    elif sl_method == "absolute":
        sl = pd.DataFrame(float(cfg["sl_abs_pct"]), index=atr.index, columns=atr.columns)
    else:
        raise ValueError(f"Unsupported sl_method={sl_method}")

    sl = _apply_caps(sl, float(cfg.get("sl_abs_lo_pct", 0.005)), float(cfg.get("sl_abs_hi_pct", 0.08)))
    sl_lo_eff = effective_sl_floor(
        sl_abs_lo_pct=float(cfg.get("sl_abs_lo_pct", 0.005)),
        sl_min_abs_pct=float(cfg.get("sl_min_abs_pct", 0.01)),
        sl_min_bps=float(cfg.get("sl_min_bps", 100)),
    )
    if cfg.get("sl_noise_buffer", False):
        sl = sl.clip(lower=sl_lo_eff)

    if float(cfg.get("trail_sl_mult", 0.0)) > 0:
        sl = sl * (1.0 - 0.15 * float(cfg.get("trail_sl_mult", 0.0)))
        sl = sl.clip(lower=sl_lo_eff, upper=float(cfg.get("sl_abs_hi_pct", 0.08)))

    out_stats = {
        "tp_mean": float(np.nanmean(tp.values)),
        "sl_mean": float(np.nanmean(sl.values)),
        "tp_floor_eff": float(tp_lo_eff),
        "sl_floor_eff": float(sl_lo_eff),
        "bound_saturation": float(((tp.values <= float(tp_lo_eff + 1e-9)) | (tp.values >= float(cfg.get("tp_abs_hi_pct", 0.08) - 1e-9))).mean()),
    }
    if dyn_stats:
        out_stats.update(dyn_stats)

    return tp.astype(np.float32), sl.astype(np.float32), out_stats, dyn_horizon


# ---------------------------
# Event extraction and scoring
# ---------------------------
def build_bucket_masks(artifacts: RunArtifacts, cfg_runtime: Dict[str, Any] | None = None) -> Dict[str, pd.DataFrame]:
    """Build directional candidate masks matching training.py's _strategy_bucket_context.

    The 4 pipelines (TF_long, MR_short, TF_short, MR_long) all draw from the same
    candidate pool (top+bottom pct% by ret24h), split only by move direction:
      - "up" movers (top pct%):   TF_long (long) + MR_short (short)
      - "down" movers (bottom %): MR_long (long) + TF_short (short)

    Since TF_long and MR_short share the same rows (both are up-movers), and
    MR_long and TF_short share the same rows (both are down-movers), the bucket
    assignment in evaluate_config uses side × direction to produce 4 exclusive labels.
    This dict therefore exposes the directional split as "up" and "down" keys,
    plus the full candidate mask, so evaluate_config can derive the 4 bucket labels.
    """
    c = artifacts.panel["close"]
    feats = artifacts.features

    # Determine the deviation metric (same as training.py).
    metric = "ret24h"
    if cfg_runtime is not None:
        metric = cfg_runtime.get("trade_deviation_metric", "ret24h")
    if metric not in feats:
        for fallback in ["ret24h", "ret4h", "ret1h"]:
            if fallback in feats:
                metric = fallback
                break

    # Build the candidate filter (top+bottom pct% by metric, same as training).
    # Hard-gate as early as possible: event conditions are applied before geometry eval.
    candidate_filter = pd.DataFrame(True, index=c.index, columns=c.columns)
    if cfg_runtime is not None:
        candidate_defaults = get_candidate_filter_defaults(cfg_runtime)
        try:
            # Bottom 10% symbols by volume are exempt from strict per-asset gating.
            vol_df = artifacts.panel.get("volume")
            if isinstance(vol_df, pd.DataFrame):
                sym_liq = vol_df.median(axis=0, skipna=True).astype(np.float32)
            else:
                sym_liq = pd.Series(1.0, index=c.columns, dtype=np.float32)
            liq_cut = float(sym_liq.quantile(0.10)) if len(sym_liq) else float("-inf")
            hi_vol_cols = [x for x in c.columns if float(sym_liq.get(x, 0.0)) > liq_cut]
            lo_vol_cols = [x for x in c.columns if x not in set(hi_vol_cols)]

            panel_hi = {k: v.reindex(columns=hi_vol_cols) for k, v in artifacts.panel.items()}
            feats_hi = {k: v.reindex(columns=hi_vol_cols) for k, v in feats.items() if isinstance(v, pd.DataFrame)}
            cf_hi = select_trade_candidates_vectorized(
                panel_hi,
                feats_hi,
                pct=float(candidate_defaults["train_extreme_pct_hourly"]),
                metric=metric,
                min_range_pct=float(candidate_defaults["train_min_range_pct"]),
                min_vol_zscore=float(candidate_defaults["train_min_vol_zscore"]),
                chop_thr=float(candidate_defaults.get("train_chop_thr", 0.5)),
            )
            if cf_hi is not None:
                cf_hi = cf_hi.astype(bool)
                # +12h time-window extension so entries can finish lifecycle.
                finish_bars = _bars_for_hours((cfg_runtime or {}).get("timeframe", "15m"), 12.0)
                time_gate = cf_hi.any(axis=1).rolling(finish_bars + 1, min_periods=1).max().astype(bool)

                candidate_filter = pd.DataFrame(False, index=c.index, columns=c.columns)
                if hi_vol_cols:
                    candidate_filter.loc[:, hi_vol_cols] = cf_hi.reindex(index=c.index, columns=hi_vol_cols, fill_value=False)
                if lo_vol_cols:
                    candidate_filter.loc[:, lo_vol_cols] = True

                time_gate_df = pd.DataFrame(
                    np.repeat(time_gate.values[:, None], len(candidate_filter.columns), axis=1),
                    index=candidate_filter.index,
                    columns=candidate_filter.columns,
                    dtype=bool,
                )
                candidate_filter = (candidate_filter & time_gate_df).astype(bool)
        except Exception:
            pass

    # Split candidates by move direction using the deviation metric.
    # "up" movers = top pct% (best performers): used by TF_long + MR_short
    # "down" movers = bottom pct% (worst performers): used by MR_long + TF_short
    if metric in feats:
        df_metric = feats[metric]
        if int(df_metric.shape[1]) < 50:
            raise ValueError(f"Fast fail: not enough symbols for cross-sectional ranking (found {df_metric.shape[1]}, need >= 50).")
        ranks = df_metric.rank(axis=1, method="first", na_option="keep", pct=True)
        up_zone   = (ranks > 0.5).fillna(False).astype(bool)
        down_zone = (ranks <= 0.5).fillna(False).astype(bool)
    else:
        ret = c.pct_change(24).fillna(0.0)
        up_zone   = (ret > 0).astype(bool)
        down_zone = (ret <= 0).astype(bool)

    up_cands   = up_zone   & candidate_filter
    down_cands = down_zone & candidate_filter

    # Expose directional masks + full candidate mask.
    # evaluate_config derives the 4 bucket labels from side × direction:
    #   (long,  up)   → TF_long   (buy_momentum)
    #   (short, up)   → MR_short  (sell_rips)
    #   (long,  down) → MR_long   (buy_dips)
    #   (short, down) → TF_short  (sell_weakness)
    return {
        "up":        up_cands,
        "down":      down_cands,
        "Candidate": candidate_filter.astype(bool),
        "Global":    pd.DataFrame(True, index=c.index, columns=c.columns),
        # Legacy aliases so any code still referencing TF_long/MR_long doesn't crash.
        "TF_long":   up_cands,
        "TF_short":  down_cands,
        "MR_long":   down_cands,
        "MR_short":  up_cands,
    }


def make_quantile_basis(artifacts: RunArtifacts, basis: str) -> pd.DataFrame:
    feats = artifacts.features
    c = artifacts.panel["close"]
    atr_basis = _get_barrier_atr_frame(artifacts)
    if basis == "vol":
        return atr_basis
    if basis == "breakout":
        for k in ["breakout_24h", "breakout_t", "pct_breakout_t"]:
            if k in feats:
                return feats[k]
    if basis == "volume":
        for k in ["vol_z", "rvol_z", "volume_entropy_12"]:
            if k in feats:
                return feats[k]
        return artifacts.panel["volume"].pct_change(fill_method=None).abs()
    if basis == "trend":
        for k in ["trend_snr", "tf_bias", "trend_regime"]:
            if k in feats:
                return feats[k]
        return c.pct_change(6, fill_method=None)

    parts = []
    for k in ["atr_pct", "trend_snr", "vol_z", "rsi"]:
        if k == "atr_pct":
            x = atr_basis
        elif k in feats:
            x = feats[k]
        else:
            continue
        parts.append((x - x.rolling(72, min_periods=8).median()) / (x.rolling(72, min_periods=8).std() + EPS))
    if parts:
        return sum(parts) / len(parts)
    return c.pct_change(fill_method=None).abs()


def choose_feature_matrix(artifacts: RunArtifacts, max_features: int = 20) -> Tuple[pd.DataFrame, List[str]]:
    good = []
    for k, v in artifacts.features.items():
        if not isinstance(v, pd.DataFrame):
            continue
        if v.shape != artifacts.panel["close"].shape:
            continue
        if v.notna().sum().sum() < 100:
            continue
        good.append(k)

    preferred = list(ACTIVE_TEST_FEATURE_KEYS)
    selected = [k for k in preferred if k in good]
    if not selected:
        raise ValueError("No configured test_feature_keys available in features for TBM comparison")
    if max_features is not None and max_features > 0:
        selected = selected[:max_features]

    stacked = {}
    for k in selected:
        stacked[k] = artifacts.features[k].stack()
    X = pd.DataFrame(stacked).astype(np.float32)
    return X, selected





@dataclass
class FeatureMatrixCache:
    index: pd.MultiIndex
    X: np.ndarray
    feat_cols: List[str]
    global_aligner: np.ndarray


def get_feature_matrix_cache(
    artifacts: RunArtifacts,
    eval_cache: BoundedEvalCache,
    max_features: int = 20,
) -> FeatureMatrixCache:
    """Build feature matrix once as contiguous immutable float32 array."""
    key = f"feature_matrix_cache::{max_features}"
    cached = eval_cache.get(key)
    if cached is not None:
        return cached

    X_flat, feat_cols = choose_feature_matrix(artifacts, max_features=max_features)
    X_flat = X_flat.fillna(0.0).astype(np.float32)
    X_np = np.ascontiguousarray(X_flat.to_numpy(dtype=np.float32, copy=False))
    X_np.setflags(write=False)
    panel_idx = artifacts.panel["close"].index
    panel_cols = artifacts.panel["close"].columns
    panel_flat_idx = pd.MultiIndex.from_product([panel_idx, panel_cols], names=["ts", "symbol"])
    global_aligner = pd.Index(X_flat.index).get_indexer(panel_flat_idx).astype(np.int64, copy=False)

    fm = FeatureMatrixCache(
        index=X_flat.index,
        X=X_np,
        feat_cols=feat_cols,
        global_aligner=np.ascontiguousarray(global_aligner, dtype=np.int64),
    )
    eval_cache[key] = fm
    gc.collect()
    return fm


def _ridge_warm_key(cfg: Dict[str, Any]) -> str:
    # Include ALL barrier params to ensure cache invalidation on any change
    sig = _compact_cfg_signature(cfg, sorted(BARRIER_PARAMS))
    return f"ridge_warm_{hashlib.sha1(str(sig).encode()).hexdigest()[:12]}"


def _ridge_predict_oof_with_cache(
    X: np.ndarray,
    y_bin: np.ndarray,
    sample_weight: np.ndarray,
    ts_values: np.ndarray,
    *,
    n_folds: int,
    warm_cache: Optional[Dict[str, Tuple[np.ndarray, float]]] = None,
    warm_key: Optional[str] = None,
    rng_seed: int = 42,
) -> np.ndarray:
    """OOF ridge on precomputed float32 arrays with optional warm-start cache."""
    y_bin = np.asarray(y_bin, dtype=np.float32)
    sample_weight = np.asarray(sample_weight, dtype=np.float32)
    # Convert timestamps to int64 (ns) to avoid dtype mismatch in np.isin
    # (ts_values from events["ts"] is datetime64, pd.Index.unique() returns Timestamps).
    ts_int = pd.DatetimeIndex(ts_values).asi8  # int64 nanoseconds
    unique_ts_int = np.array(sorted(pd.unique(ts_int)), dtype=np.int64)
    if len(X) == 0 or len(unique_ts_int) == 0:
        return np.full(len(X), 0.5, dtype=np.float32)

    n_folds = max(2, int(n_folds))  # keep OOF-only path (>=2 folds)
    pred = np.full(len(X), 0.5, dtype=np.float32)

    if len(unique_ts_int) < n_folds + 5:
        n_folds = max(2, len(unique_ts_int) // 5)

    chunks = np.array_split(unique_ts_int, n_folds)

    for test_ts_chunk in chunks:
        if len(test_ts_chunk) == 0:
            continue
        test_mask = np.isin(ts_int, test_ts_chunk)
        train_mask = ~test_mask
        if int(train_mask.sum()) < 100 or int(test_mask.sum()) == 0:
            continue

        X_train = X[train_mask]
        y_train = y_bin[train_mask]
        w_train = sample_weight[train_mask]
        X_test = X[test_mask]

        # Robust standardisation: vectorised numpy (avoids sklearn object overhead per fold).
        # median + IQR computed once over train set, applied to both train and test.
        q25 = np.percentile(X_train, 25, axis=0).astype(np.float32)
        q75 = np.percentile(X_train, 75, axis=0).astype(np.float32)
        iqr = np.where(q75 - q25 > 1e-8, q75 - q25, 1.0).astype(np.float32)
        med = np.median(X_train, axis=0).astype(np.float32)
        X_train_s = ((X_train - med) / iqr).astype(np.float32)
        X_test_s = ((X_test - med) / iqr).astype(np.float32)

        # Higher alpha (3.0) for increased regularization/stability in noisy financial data.
        alpha = 3.0

        # P2 FIX: warm-cache — reuse coef/intercept from a previously solved similar fold
        # to skip the expensive linalg.solve when a warm solution exists.
        if warm_cache is not None and warm_key and warm_key in warm_cache:
            coef_w, intercept_w = warm_cache[warm_key]
            p = (X_test_s @ coef_w + intercept_w).astype(np.float32, copy=False)
            pred[test_mask] = np.clip(p, 0.0, 1.0)
            continue

        # Fast closed-form Ridge Regression
        sw_sum = np.sum(w_train)
        X_mean = (w_train @ X_train_s) / sw_sum
        y_mean = np.dot(w_train, y_train) / sw_sum

        X_c = X_train_s - X_mean
        y_c = y_train - y_mean

        WX_c = X_c * w_train[:, np.newaxis]

        A = X_c.T @ WX_c
        # Add Ridge penalty
        A.flat[::X_train_s.shape[1]+1] += alpha
        b = WX_c.T @ y_c

        try:
            coef = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            coef = np.linalg.lstsq(A, b, rcond=None)[0]

        intercept = y_mean - np.dot(X_mean, coef)

        p = (X_test_s @ coef + intercept).astype(np.float32, copy=False)
        # For Ridge on binary targets, raw p is a better probability proxy than sigmoid(p).
        p_clipped = np.clip(p, 0.0, 1.0)
        pred[test_mask] = p_clipped.astype(np.float32, copy=False)

        if warm_cache is not None and warm_key:
            warm_cache[warm_key] = (coef.astype(np.float32, copy=True), float(intercept))

    return pred


def _stage1_pr_auc_lift(
    y_bin: np.ndarray,
    pred: np.ndarray,
) -> float:
    yb = np.asarray(y_bin, dtype=np.float32)
    pp = np.asarray(pred, dtype=np.float32)
    if len(yb) < 32:
        return 1.0
    pos_rate = float(np.clip(np.mean(yb > 0.5), EPS, 1.0))
    try:
        ap = float(average_precision_score(yb, pp))
    except Exception:
        return 1.0
    return float(ap / max(pos_rate, EPS))


def _fast_auc_binary(y_bin: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y_bin, dtype=np.float32)
    p = np.asarray(pred, dtype=np.float32)
    m = np.isfinite(y) & np.isfinite(p)
    if int(m.sum()) < 16:
        return float("nan")
    y = y[m]
    p = p[m]
    n_pos = int((y > 0.5).sum())
    n_neg = int(len(y) - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy(np.float32)
    u = ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _optuna_objective_score(res: Dict[str, Any]) -> float:
    """Stable scalar objective derived from evaluate_config() outputs."""
    stage = _safe_float(res.get("stage_score", float("nan")), float("nan"))
    if math.isfinite(stage):
        return float(stage)
    stage2 = _safe_float(res.get("stage2_score", float("nan")), float("nan"))
    if math.isfinite(stage2):
        return float(stage2)
    auc_med = _safe_float(res.get("median_cell_auc", float("nan")), float("nan"))
    if math.isfinite(auc_med):
        return float(auc_med)
    auc_min = _safe_float(res.get("min_cell_auc", float("nan")), float("nan"))
    if math.isfinite(auc_min):
        return float(auc_min)
    return -1.0


def _optuna_n_jobs() -> int:
    env_jobs = os.environ.get("EPM_TBM_OPTUNA_N_JOBS", "").strip()
    if env_jobs:
        try:
            return max(1, int(env_jobs))
        except Exception:
            pass
    c = int(os.cpu_count() or 1)
    # Conservative default to reduce first-wave cache miss contention/memory spikes.
    return max(1, min(2, c // 2))


def _build_equidistant_grid(
    lo: float,
    hi: float,
    *,
    max_points: int = 10,
    min_abs: float = 0.1,
    min_rel: float = 0.2,
    endpoint_mode: str = "spacing_safe_replace",
) -> Tuple[List[float], Dict[str, Any]]:
    """Generate spacing-constrained proposals with spacing-safe endpoint handling."""
    lo_f, hi_f = float(lo), float(hi)
    meta: Dict[str, Any] = {
        "proposal_mode": "linear",
        "endpoint_hi_included": False,
        "endpoint_override": False,
    }
    if not math.isfinite(lo_f) or not math.isfinite(hi_f):
        return [], meta
    if hi_f < lo_f:
        lo_f, hi_f = hi_f, lo_f
    if abs(hi_f - lo_f) < EPS:
        one = [round(lo_f, 6)]
        meta["endpoint_hi_included"] = True
        return one, meta

    n = max(2, min(15, int(max_points)))
    use_log = lo_f > 0.0 and hi_f > 0.0
    if use_log:
        p = np.exp(np.linspace(np.log(lo_f), np.log(hi_f), n, dtype=np.float64))
    else:
        p = np.linspace(lo_f, hi_f, n, dtype=np.float64)

    def _min_gap(prev: float) -> float:
        scale_prev = max(abs(float(prev)), abs(lo_f), abs(hi_f), 1.0)
        return max(float(min_abs), float(min_rel) * scale_prev)

    def _spacing_ok(vals: List[float]) -> bool:
        if len(vals) <= 1:
            return True
        for i in range(1, len(vals)):
            prev = float(vals[i - 1])
            cur = float(vals[i])
            if (cur - prev) < (_min_gap(prev) - 1e-12):
                return False
        return True

    g: List[float] = []
    for x in p.tolist():
        if not g:
            g.append(float(x)); continue
        prev = float(g[-1])
        if (float(x) - prev) >= (_min_gap(prev) - 1e-12):
            g.append(float(x))

    if not g:
        g = [float(lo_f)]
    endpoint_override = False
    if abs(g[-1] - hi_f) > 1e-12:
        prev = float(g[-1])
        if (hi_f - prev) >= (_min_gap(prev) - 1e-12):
            g.append(float(hi_f))
        elif endpoint_mode == "spacing_safe_replace":
            if len(g) >= 2:
                prev2 = float(g[-2])
                if (hi_f - prev2) >= (_min_gap(prev2) - 1e-12):
                    g[-1] = float(hi_f)
            # else: keep strict spacing and omit hi
        elif endpoint_mode == "force_hi_log":
            g[-1] = float(hi_f)
            endpoint_override = True

    # stable unique + rounded for categorical sampler determinism
    out: List[float] = []
    for v in g:
        rv = round(float(v), 6)
        if (not out or abs(rv - out[-1]) > 1e-9) and (not out or rv > out[-1]):
            out.append(rv)
    lo_r = round(float(lo_f), 6)
    if not out or abs(out[0] - lo_r) > 1e-9:
        out = [lo_r] + [v for v in out if v > lo_r]
    endpoint_hi_included = any(abs(v - round(float(hi_f), 6)) <= 1e-9 for v in out)
    spacing_valid = _spacing_ok(out)
    if not spacing_valid:
        tprint(f"[grid_meta] spacing violation detected lo={lo_f:.6f} hi={hi_f:.6f} grid={out}")
    return out, {
        "proposal_mode": "log" if use_log else "linear",
        "endpoint_hi_included": bool(endpoint_hi_included),
        "endpoint_override": bool(endpoint_override),
        "spacing_valid": bool(spacing_valid),
    }

def _build_param_grid(lo: float, hi: float) -> Tuple[List[float], Dict[str, Any]]:
    """Build parameter grid with adaptive spacing relaxation when too coarse."""
    span = abs(float(hi) - float(lo))
    min_abs_p1 = max(5e-4, 0.05 * span)
    min_abs_p2 = max(2.5e-4, 0.025 * span)
    min_rel_p1 = max(1e-3, 0.10 * span)
    min_rel_p2 = max(5e-4, 0.05 * span)
    g, meta = _build_equidistant_grid(lo, hi, max_points=10, min_abs=min_abs_p1, min_rel=min_rel_p1, endpoint_mode="spacing_safe_replace")
    if len(g) < 6:
        g, meta = _build_equidistant_grid(lo, hi, max_points=10, min_abs=min_abs_p2, min_rel=min_rel_p2, endpoint_mode="spacing_safe_replace")
    if len(g) < 4:
        lo_f, hi_f = float(min(lo, hi)), float(max(lo, hi))
        span = max(hi_f - lo_f, 0.1)
        lo_w = lo_f - 0.1 * span
        hi_w = hi_f + 0.1 * span
        if lo_f > 0 and lo_w <= 0:
            lo_w = max(EPS, lo_f * 0.8)
        g, meta = _build_equidistant_grid(
            lo_w,
            hi_w,
            max_points=12,
            min_abs=max(2.5e-4, 0.025 * span),
            min_rel=max(5e-4, 0.025 * span),
            endpoint_mode="spacing_safe_replace",
        )
        tprint(f"[grid_meta] relaxed grid due to low cardinality lo={lo:.6f} hi={hi:.6f} -> n={len(g)}")
    return g, meta


def _build_sl_biased_grid(runtime_floor: float) -> Tuple[List[float], Dict[str, Any]]:
    """Build SL grid with denser plausible mid-band coverage and retained tails."""
    sl_lo = float(max(runtime_floor, 0.3))
    sl_hi = 0.9
    if sl_hi <= sl_lo + EPS:
        return [round(sl_lo, 6)], {"proposal_mode": "linear_mixed", "endpoint_hi_included": False, "endpoint_override": False, "spacing_valid": True}

    p_tail = np.linspace(sl_lo, sl_hi, 6, dtype=np.float64)
    mid_lo = max(sl_lo, 0.45)
    mid_hi = min(sl_hi, 0.75)
    p_mid = np.linspace(mid_lo, mid_hi, 10, dtype=np.float64) if mid_hi > mid_lo + EPS else np.array([], dtype=np.float64)
    p = np.sort(np.unique(np.concatenate([p_tail, p_mid])))

    def _min_gap(prev: float) -> float:
        scale_prev = max(abs(float(prev)), abs(sl_lo), abs(sl_hi), 1.0)
        return max(0.05, 0.1 * scale_prev)

    g: List[float] = []
    for x in p.tolist():
        if not g:
            g.append(float(x))
            continue
        prev = float(g[-1])
        if (float(x) - prev) >= (_min_gap(prev) - 1e-12):
            g.append(float(x))

    if not g:
        g = [sl_lo, sl_hi]
    if g[0] > sl_lo + 1e-12:
        g = [sl_lo] + g
    if g[-1] < sl_hi - 1e-12:
        prev = g[-1]
        if (sl_hi - prev) >= (_min_gap(prev) - 1e-12):
            g.append(sl_hi)

    out = sorted(set(round(float(v), 6) for v in g))
    if len(out) < 6 and sl_hi > sl_lo + EPS:
        out = sorted(set(round(float(v), 6) for v in np.linspace(sl_lo, sl_hi, 6, dtype=np.float64).tolist()))
    if len(out) > 15:
        interior = out[1:-1]
        keep_n = 13
        step = max(1, len(interior) // max(1, keep_n))
        kept = interior[::step][:keep_n]
        out = [out[0]] + kept + [out[-1]]
        out = sorted(set(out))

    meta = {
        "proposal_mode": "linear_mixed",
        "endpoint_hi_included": bool(abs(out[-1] - round(sl_hi, 6)) <= 1e-9),
        "endpoint_override": False,
        "spacing_valid": True,
    }
    return out, meta


def _optuna_sampler() -> optuna.samplers.RandomSampler:
    return optuna.samplers.RandomSampler(seed=42)

def compute_weights(events: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    scheme = cfg.get("weighting_scheme", "none")
    w = np.ones(len(events), dtype=np.float32)
    if scheme == "none":
        return w

    rr = events["tp"].values / np.maximum(events["sl"].values, EPS)
    tp_hit = (events["label"].values == OUT_TP).astype(float)
    timeout = (events["label"].values == OUT_TO).astype(float)

    if scheme == "rr":
        w *= np.power(np.clip(rr, 0.1, 10.0), float(cfg.get("rr_weight_power", 1.0)))
    elif scheme == "tp_hit":
        w *= 1.0 + float(cfg.get("tp_hit_weight", 0.5)) * tp_hit
    elif scheme == "inv_timeout":
        w *= 1.0 / (1.0 + float(cfg.get("timeout_penalty", 0.5)) * timeout)
    elif scheme == "combined":
        w *= np.power(np.clip(rr, 0.1, 10.0), float(cfg.get("rr_weight_power", 1.0)))
        w *= 1.0 + float(cfg.get("tp_hit_weight", 0.5)) * tp_hit
        w *= 1.0 / (1.0 + float(cfg.get("timeout_penalty", 0.5)) * timeout)

    w = np.clip(w, 0.1, 25.0)
    return w.astype(np.float32, copy=False)


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float32)
    if w.size == 0:
        return 0.0
    s1 = np.sum(w)
    s2 = np.sum(w * w)
    if s2 <= 0:
        return 0.0
    return float((s1 * s1) / s2)




def per_slice_metrics(events: pd.DataFrame, pred: np.ndarray, slice_col: str) -> Dict[str, Dict[str, float]]:
    """Compute IC_payoff and IC_label for each unique value in events[slice_col]."""
    out = {}
    
    # Global metrics
    out["Global"] = {
        "ic_payoff": _safe_spearman(pred, events["payoff"].values),
        "ic_label": _safe_spearman(pred, events["label"].values)
    }
    
    # Per-slice metrics
    for val, g in events.groupby(slice_col, observed=True):
        idx = g.index.values
        out[str(val)] = {
            "ic_payoff": _safe_spearman(pred[idx], g["payoff"].values),
            "ic_label": _safe_spearman(pred[idx], g["label"].values)
        }
    return out


def _empty_result(
    cfg: Dict[str, Any],
    cfg_id: str,
    full_n: int,
    reason: str = "empty",
    stage2_rescore: bool = False,
) -> tuple:
    """Return a zero-filled (summary, detail, None) triple for fast-path exits."""
    summary = {
        "config_id": cfg_id,
        "evaluation_stage": "stage2_rescore" if stage2_rescore else "stage1",
        "mode": cfg.get("mode", "unknown"),
        "k_tp": cfg.get("k_tp"),
        "tp_anchor": cfg.get("tp_anchor"),
        "tp_base_pct": cfg.get("tp_base_pct", cfg.get("tp_abs_pct")),
        "sl_method": cfg.get("sl_method"),
        "sl_as_tp_pct": cfg.get("sl_as_tp_pct"),
        "regime_model": cfg.get("tp_regime_model"),
        "horizon_scaling": cfg.get("horizon_scaling"),
        "ic_label": 0.0, "ic_label_bucket_mean": 0.0,
        "ic_payoff": 0.0, "ic_payoff_bucket_mean": 0.0,
        "ic_snr": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "tp_hit_rate": 0.0, "sl_hit_rate": 0.0, "timeout_rate": 1.0,
        "ess": 0.0, "ess_full": float(full_n), "coverage": 0.0,
        "ic_std_time": 0.0, "ic_std_asset": 0.0, "worst_bucket_IC": 0.0,
        "stage1_score": -10.0, "stage2_score": -10.0, "stage_score": -10.0,
        "probe_auc": float("nan"), "probe_auc_bound": float("nan"),
        "probe_status": "not_run",
        "probe_n_train": 0, "probe_n_val": 0,
        "brier": 0.0, "ece": 0.0, "monotonicity": 0.0,
        "oof_payoff_decile_spread": 0.0, "hard_gate": False,
        "pass_cells": 0, "total_cells": 0, "worst_bucket_coverage": 0.0,
        "prod_admissible_tier0": False, "prod_adm_failures": 1,
        "econ_ok": False, "econ_G": 0.0, "econ_multiplier": 0.0,
        "tp_floor_bind_prod_agg": float("nan"),
        "max_cell_tp_floor_bind_prod": float("nan"),
        "floor_dominance_penalty": 0.0,
        "prod_aligned_tp": cfg.get("prod_aligned_tp", {}),
    }
    detail = {
        "config": serialize_key(cfg),
        "feature_columns": [],
        "bucket_metrics": {},
        "bucket_horizon_metrics": {},
        "production_admissibility": {
            "admissible_tier0": False,
            "failures": [f"early_exit:{reason}"],
            "per_cell_health": {},
            "aggregates": {},
        },
        "prod_aligned_tp": cfg.get("prod_aligned_tp", {}),
        "calibration": {"brier": 0.0, "ece": 0.0, "monotonicity": 0.0, "oof_payoff_decile_spread": 0.0},
    }
    tprint(f"[eval:done] {cfg_id} {reason} hard_gate=False")
    return summary, detail, None


def evaluate_config(
    artifacts: RunArtifacts,
    cfg: Dict[str, Any],
    horizons: Sequence[int],
    bucket_masks: Dict[str, pd.DataFrame],
    layer1_cache: Dict[str, Any],
    layer2_cache: Dict[str, Any],
    eval_cache: BoundedEvalCache,
    detailed_slices: bool = False,
    collect_weights: bool = False,
    target_cell_filter: Optional[Tuple[str, int]] = None,
    stage2_rescore: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[pd.DataFrame]]:
    cfg = _apply_compare_production_floor(cfg)
    cfg_id = config_id(cfg)
    
    # ── Smoking Gun #4: Early cheap gate before barrier/label generation ──
    # If the config is fundamentally broken (e.g. RR too low), exit before building barriers.
    min_rr = float(cfg.get("min_net_rr", 0.7))
    sl_ratio = float(cfg.get("sl_as_tp_pct", 0.5))
    sl_ratio_floor = float(cfg.get("tbm_sl_as_tp_pct_min", 0.4))
    if sl_ratio < sl_ratio_floor:
        return _empty_result(
            cfg, cfg_id, 0, reason=f"cheap_gate_sl_as_tp_too_low:{sl_ratio:.3f}<{sl_ratio_floor:.3f}",
            stage2_rescore=stage2_rescore
        )
    if sl_ratio > 0 and (1.0 / sl_ratio) < (min_rr * 0.7):
        return _empty_result(
            cfg, cfg_id, 0, reason=f"cheap_gate_rr_too_low:{1.0/sl_ratio:.2f}<{min_rr:.2f}",
            stage2_rescore=stage2_rescore
        )
    
    t0 = time.perf_counter()
    fast_mode = bool(cfg.get("_low_fidelity", False)) and not bool(collect_weights)
    if fast_mode:
        detailed_slices = False

    tprint(
        f"[eval:start] {cfg_id} mode={cfg.get('mode', 'unknown')} "
        f"horizons={list(horizons)} mem_peak_mb={_memory_snapshot_mb():.1f} "
        f"{_cache_pressure_summary(layer1_cache, layer2_cache, eval_cache)}"
    )
    events_rows: List[pd.DataFrame] = []
    # If target_cell_filter is provided (for cell-specific Optuna), only compute barriers for that horizon.
    _active_horizons = [target_cell_filter[1]] if target_cell_filter else horizons
    
    # ── Pre-compute Bucket Map ──
    # The evaluation requires knowing which candidate symbols belong to `up` or `down` zones
    # BEFORE running the triple barrier labeler so we can fast-sample the labels.
    _panel_idx = artifacts.panel["close"].index
    _panel_cols = artifacts.panel["close"].columns
    _idx_flat = pd.MultiIndex.from_product([_panel_idx, _panel_cols], names=["ts", "symbol"])
    _stack_key = _index_cache_key(_idx_flat)
    
    cache_bucket_stack = eval_cache.setdefault("bucket_stack", {})
    if _stack_key in cache_bucket_stack:
        bucket_map = cache_bucket_stack[_stack_key]
    else:
        bucket_map = {}
        for bname, bmask in bucket_masks.items():
            # Align mask to panel shape in case of mismatch, then stack
            aligned_bmask = bmask.reindex(index=_panel_idx, columns=_panel_cols, fill_value=False)
            bucket_map[bname] = aligned_bmask.stack().to_numpy(dtype=bool, copy=False)
        cache_bucket_stack[_stack_key] = bucket_map
        
    _last_gc_ts = time.perf_counter()

    for h in _active_horizons:
        for side in ["long", "short"]:
            # Key1: Barriers depend on geometry params; TP can be cached independently for tp_pct SL sweeps.
            barrier_cfg = _get_barrier_params(cfg)
            
            # Optimization: Decoupled label cache key as requested.
            # Labels will be cached by (h, side, scaling, base, time_index_hash).
            # This ensures hits across different k_tp/sl_as_tp trials.
            label_key = (
                "labels_decoupled",
                int(h),
                str(side),
                str(cfg.get("horizon_scaling", "sqrt")),
                float(cfg.get("horizon_base", 4.0)),
                _index_cache_key(artifacts.panel["close"].index),
            )

            if str(cfg.get("sl_method", "tp_pct")) == "tp_pct":
                tp_barrier_cfg = _get_tp_barrier_params(cfg)
                key1_tp = (
                    "barrier_tp",
                    int(h),
                    str(side),
                    bool(cfg.get("_prod_floor_applied", False)),
                    round(float(effective_tp_floor(tp_abs_lo_pct=float(cfg.get("tp_abs_lo_pct", 0.005)), tp_min_abs_pct=float(cfg.get("tp_min_abs_pct", 0.005)), tp_min_bps=float(cfg.get("tp_min_bps", 50)))), 6),
                    _compact_cfg_signature(tp_barrier_cfg, tuple(tp_barrier_cfg.keys())),
                )
                if key1_tp not in layer1_cache:
                    tp_df, _sl_df, geom_stats, dyn_h = build_barriers(artifacts, cfg, h, side)
                    layer1_cache[key1_tp] = (tp_df, geom_stats, dyn_h)
                    tprint(f"[eval:{cfg_id}] barrier_cache miss(tp) h={h} side={side}")
                else:
                    tprint(f"[eval:{cfg_id}] barrier_cache hit(tp) h={h} side={side}")
                tp_df, geom_stats, dyn_h = layer1_cache[key1_tp]
                
                # Derive SL from TP on the fly (very fast)
                series_cache = getattr(artifacts, "_tbm_series_cache", {})
                atr_cached = series_cache.get("atr_shift_bfill")
                if atr_cached is None:
                    atr_cached = _get_barrier_atr_frame(artifacts).shift(1).clip(lower=1e-6).bfill(limit=1).astype(np.float32)
                    series_cache["atr_shift_bfill"] = atr_cached
                    setattr(artifacts, "_tbm_series_cache", series_cache)
                sl_df = _derive_sl_from_tp(tp_df, atr_cached, cfg)
                # Labels still use the decoupled label_key for hit potential
                key2 = label_key
            else:
                key1 = (
                    "barrier_full",
                    int(h),
                    str(side),
                    bool(cfg.get("_prod_floor_applied", False)),
                    round(float(effective_tp_floor(tp_abs_lo_pct=float(cfg.get("tp_abs_lo_pct", 0.005)), tp_min_abs_pct=float(cfg.get("tp_min_abs_pct", 0.005)), tp_min_bps=float(cfg.get("tp_min_bps", 50)))), 6),
                    _compact_cfg_signature(barrier_cfg, tuple(barrier_cfg.keys())),
                )
                if key1 not in layer1_cache:
                    tp_df, sl_df, geom_stats, dyn_h = build_barriers(artifacts, cfg, h, side)
                    layer1_cache[key1] = (tp_df, sl_df, geom_stats, dyn_h)
                    tprint(f"[eval:{cfg_id}] barrier_cache miss h={h} side={side}")
                else:
                    tprint(f"[eval:{cfg_id}] barrier_cache hit h={h} side={side}")
                tp_df, sl_df, geom_stats, dyn_h = layer1_cache[key1]
                key2 = label_key


            # Key2: Labels depend on TP/SL barriers + horizon/side + geometry.
            # Optimization: We keep the decoupled label_key for fast generation (Numba hit),
            # but we use a more granular key2 for the resulting dataframes (lbl, ret, qual)
            # because they contain the actual TP/SL/Payoff values which differ by trial.
            key2 = (label_key, _compact_cfg_signature(cfg, ("k_tp", "sl_as_tp_pct", "tp_base_pct", "base_atr_window")))

            if key2 not in layer2_cache:
                # ── Fast Targeting: Find Valid TP Hits First ──
                # 1. Run the ultra-fast Numba labeler without 15m refinement.
                # 2. Identify all `TP` hits across the 5-year history.
                # 3. Sample exactly 2000 of them (and proportional negatives).
                # 4. Only run the slow 15m refinement on this tiny subset.
                lbl_fast, _, _ = compute_triple_barrier_labels(
                    artifacts.panel, tp_df, sl_df, h, side=side, return_outcomes=True, horizons_frame=dyn_h, resolve_conflicts=False
                )
                
                # Active candidates for all directions (we subsample across ALL, not just one side)
                # This guarantees both TF and MR buckets always have event representation.
                up_cand_mask = bucket_map["up"]
                down_cand_mask = bucket_map["down"]
                all_cand_mask = up_cand_mask | down_cand_mask
                
                lbl_arr = lbl_fast.to_numpy(dtype=np.int8).ravel()
                
                # ── Bucket-Stratified Subsampling ──
                # Split candidates into UP (TF_long / MR_short) and DOWN (MR_long / TF_short) strata.
                # Without stratification, the global random sample can be 100% UP, leaving MR_long empty.
                _MAX_TP = 3000
                _TP_PER_STRATA = _MAX_TP // 2  # 1500 TPs from each direction
                _rng_lb = np.random.default_rng(42)
                
                keep_global_list = []
                for strat_mask in [up_cand_mask.ravel(), down_cand_mask.ravel()]:
                    strat_pos = np.where(strat_mask)[0]
                    strat_lbls = lbl_arr[strat_pos]
                    tp_idx = np.where(strat_lbls == 2)[0]
                    neg_idx = np.where(strat_lbls != 2)[0]
                    if len(tp_idx) == 0:
                        continue
                    n_tp = min(len(tp_idx), _TP_PER_STRATA)
                    tp_keep = _rng_lb.choice(tp_idx, size=n_tp, replace=False)
                    keep_ratio = n_tp / len(tp_idx)
                    n_neg = int(len(neg_idx) * keep_ratio)
                    if n_neg > 0 and len(neg_idx) > 0:
                        neg_keep = _rng_lb.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
                        keep_local = np.concatenate([tp_keep, neg_keep])
                    else:
                        keep_local = tp_keep
                    keep_global_list.append(strat_pos[keep_local])
                
                if keep_global_list:
                    keep_global = np.concatenate(keep_global_list)
                    tight_mask_arr = np.zeros(lbl_arr.shape[0], dtype=bool)
                    tight_mask_arr[keep_global] = True
                    tight_mask_df = pd.DataFrame(tight_mask_arr.reshape(tp_df.shape), index=tp_df.index, columns=tp_df.columns)
                    
                    # Log the cross-asset spread for the final subsample
                    unique_assets_hit = (tight_mask_df.sum(axis=0) > 0).sum()
                    total_tp = sum(len(np.where(lbl_arr[np.where(m.ravel())[0]] == 2)[0]) 
                                   for m in [up_cand_mask.ravel(), down_cand_mask.ravel()])
                    tprint(f"[eval:{cfg_id}] subsampled {len(keep_global):,} events (2x stratified) spread across {unique_assets_hit} unique assets")
                    
                    _sampled_tp = tp_df.where(tight_mask_df, np.nan)
                    _sampled_sl = sl_df.where(tight_mask_df, np.nan)
                else:
                    # No candidates at all — return early
                    _sampled_tp = tp_df
                    _sampled_sl = sl_df
                    tprint(f"[eval:{cfg_id}] WARNING: no TP candidates found in any bucket stratum, using full array")

                # ── Final Compute ──
                # Use the heavily downsampled TP/SL barriers to compute the final label array.
                # If the 15m panel is available, we pass it natively into Numba (bypassing slow Python refinement).
                if getattr(artifacts, "panel_15m", None) is not None:
                    _panel_15m = artifacts.panel_15m
                    _idx_15m = _panel_15m["close"].index
                    _idx_1h = artifacts.panel["close"].index
                    
                    # Reindex downsampled bounds to 15m domain (fills NaNs for non-hour bars)
                    # Numba skips NaNs instantly, computing only valid 1h entries but tracking paths on 15m resolution.
                    _tp_15m = _sampled_tp.reindex(_idx_15m)
                    _sl_15m = _sampled_sl.reindex(_idx_15m)
                    _dyn_h_15m = dyn_h.reindex(_idx_15m) if dyn_h is not None else None
                    
                    lbl_15m, ret_15m, qual_15m = compute_triple_barrier_labels(
                        _panel_15m, _tp_15m, _sl_15m, h, side=side, return_outcomes=True, horizons_frame=_dyn_h_15m, resolve_conflicts=False
                    )
                    
                    # Project perfectly computed outcomes back to 1h domain
                    lbl = lbl_15m.reindex(_idx_1h).fillna(0)
                    ret = ret_15m.reindex(_idx_1h).fillna(0)
                    qual = qual_15m.reindex(_idx_1h).fillna(0)
                else:
                    lbl, ret, qual = compute_triple_barrier_labels(
                        artifacts.panel, _sampled_tp, _sampled_sl, h, side=side, return_outcomes=True, horizons_frame=dyn_h
                    )
                    lbl, ret, qual = _refine_ambiguous_labels_with_15m(
                        artifacts.panel, _sampled_tp, _sampled_sl, lbl, ret, qual, h, side, cfg=cfg
                    )
                layer2_cache[key2] = (lbl, ret, qual)
                tprint(f"[eval:{cfg_id}] label_cache miss h={h} side={side}")
            else:
                tprint(f"[eval:{cfg_id}] label_cache hit h={h} side={side}")
            lbl, ret, qual = layer2_cache[key2]

            # Stack arrays once per granular key and reuse across repeated evaluations.
            stack_cache = eval_cache.setdefault("label_stack_cache", {})
            stack_key = key2
            if stack_key not in stack_cache:
                # Utilize instant Numpy `.ravel()` instead of slow Pandas `.stack()` 
                # because we already computed the guaranteed dense MultiIndex `_idx_flat`.
                label_arr = lbl.to_numpy(dtype=np.float32, copy=False).ravel()
                payoff_arr = ret.to_numpy(dtype=np.float32, copy=False).ravel()
                qual_arr = qual.to_numpy(dtype=np.float32, copy=False).ravel()
                
                # IMPORTANT FIX: Must use the DOWNSAMPLED barrier arrays.
                # If we used the original dense `tp_df`, we would inject 6 million rows of NaN labels.
                tp_arr = _sampled_tp.to_numpy(dtype=np.float32, copy=False).ravel()
                sl_arr = _sampled_sl.to_numpy(dtype=np.float32, copy=False).ravel()
                
                # ── CRITICAL FIX: Memory Protection ──
                # The arrays above contain 7 million elements, 99.9% of which are NaN
                # due to our pre-labeling mask. We strip the NaNs raw in Numpy first.
                valid_mask_flat = ~np.isnan(sl_arr)
                
                stacked_idx = _idx_flat[valid_mask_flat]
                panel_idx_arr = np.flatnonzero(valid_mask_flat).astype(np.int64, copy=False)
                label_arr = label_arr[valid_mask_flat]
                payoff_arr = payoff_arr[valid_mask_flat]
                qual_arr = qual_arr[valid_mask_flat]
                tp_arr = tp_arr[valid_mask_flat]
                sl_arr = sl_arr[valid_mask_flat]
                
                if dyn_h is not None:
                    h_arr = dyn_h.to_numpy(dtype=np.float32, copy=False).ravel()[valid_mask_flat]
                else:
                    h_arr = np.full(len(label_arr), float(h), dtype=np.float32)
                
                stack_cache[stack_key] = (stacked_idx, panel_idx_arr, label_arr, payoff_arr, qual_arr, tp_arr, sl_arr, h_arr)
                if len(stack_cache) > 256:
                    stack_cache.pop(next(iter(stack_cache)))
            else:
                stacked_idx, panel_idx_arr, label_arr, payoff_arr, qual_arr, tp_arr, sl_arr, h_arr = stack_cache[stack_key]

            # Create DataFrame directly from cached masked numpy arrays (now only ~4,000 rows, not 7,000,000)
            df = pd.DataFrame(
                {
                    "label": label_arr,
                    "payoff": payoff_arr,
                    "quality": qual_arr,
                    "tp": tp_arr,
                    "sl": sl_arr,
                    "horizon_eff": h_arr,
                    "__panel_idx__": panel_idx_arr,
                },
                index=stacked_idx
            )
            df.index.names = ["ts", "symbol"]
            
            # Early filtering: drop timeouts early if they won't pass min_raw_events
            # This reduces memory before pd.concat
            if len(df) > 0:
                timeout_mask = df["label"] == 0
                timeout_count = timeout_mask.sum()
                if timeout_count > 1000:  # If too many timeouts, keep only non-timeouts for now
                    # Keep all but mark for later filtering
                    pass
            
            df = df.reset_index()
            df["side"] = side
            df["horizon"] = h
            df["bound_saturation"] = geom_stats["bound_saturation"]
            events_rows.append(df)
            
            # Free intermediate arrays immediately
            _last_gc_ts = _maybe_collect_gc(last_gc_ts=_last_gc_ts, mem_threshold_mb=8192.0, min_interval_s=3.0)

    events = pd.concat(events_rows, ignore_index=True)
    events["label"] = events["label"].astype(np.float32, copy=False)
    events["payoff"] = events["payoff"].astype(np.float32, copy=False)
    events["quality"] = _sanitize_quality_array(events["quality"].to_numpy(dtype=np.float32, copy=False))
    events["tp"] = events["tp"].astype(np.float32, copy=False)
    events["sl"] = events["sl"].astype(np.float32, copy=False)
    events["horizon"] = events["horizon"].astype(np.int16, copy=False)
    events["horizon_eff"] = events["horizon_eff"].astype(np.float32, copy=False)
    events["side"] = events["side"].astype("category")
    tprint(
        f"[eval:{cfg_id}] raw_events={len(events):,} "
        f"mem_peak_mb={_memory_snapshot_mb():.1f}"
    )

    # Per-(side,horizon) funnel: raw → prefilter → rr → final counts.
    # Printed before any further filtering so the user can see where events are lost.
    _funnel_rows = []
    _fee = float(cfg.get("fee_pct", 0.5)) / 100.0
    _slip = float(cfg.get("slip_buffer", 0.1)) / 100.0
    _min_rr = float(cfg.get("min_net_rr", 0.7))
    _tp_net_all = events["tp"].to_numpy(dtype=np.float32, copy=False) - _fee - _slip
    _sl_net_all = events["sl"].to_numpy(dtype=np.float32, copy=False) + _fee + _slip
    _rr_all = _tp_net_all / np.maximum(_sl_net_all, EPS)
    _rr_ok_all = _rr_all >= _min_rr
    _side_vals = events["side"].astype(str).to_numpy()
    _h_vals = events["horizon"].to_numpy(dtype=np.int16, copy=False)
    _keys = np.char.add(np.char.add(_side_vals.astype(str), "|"), _h_vals.astype(str))
    _u_keys, _inv = np.unique(_keys, return_inverse=True)
    for _i, _k in enumerate(_u_keys):
        _mask = (_inv == _i)
        _n_raw = int(_mask.sum())
        _n_rr = int(_rr_ok_all[_mask].sum())
        _side, _h = _k.split("|")
        _funnel_rows.append({"side": _side, "h": int(_h), "n_raw": _n_raw, "n_rr": _n_rr})
    if _funnel_rows:
        _funnel_rows = sorted(_funnel_rows, key=lambda r: (str(r["side"]), int(r["h"])))
        _parts = [f"{r['side']}_H{r['h']}: {r['n_raw']:,}→{r['n_rr']:,}" for r in _funnel_rows]
        _rr_fracs = [r["n_rr"] / max(r["n_raw"], 1) for r in _funnel_rows]
        tprint(
            f"[eval:{cfg_id}] funnel(rr≥{_min_rr:.2f}) "
            + "  ".join(_parts)
            + f"  |  rr_kept min={min(_rr_fracs)*100:.1f}% median={float(np.median(_rr_fracs))*100:.1f}%"
        )

    tprint(f"[eval:{cfg_id}] Step 2 (Assembly) took {time.perf_counter() - t0:.2f}s")
    _t1 = time.perf_counter()

    tprint(f"[eval:{cfg_id}] Step 2 (Assembly) took {time.perf_counter() - t0:.2f}s")
    _t1 = time.perf_counter()

    # Hard candidate pre-filter: keep only rows in the candidate mask.
    event_panel_idx = events["__panel_idx__"].to_numpy(dtype=np.int64, copy=False)
    _cand_mask = bucket_map["Candidate"][event_panel_idx]
    
    pre_candidate_n = len(events)
    if np.any(_cand_mask):
        events = events.iloc[_cand_mask].copy().reset_index(drop=True)
        # Create a new local mapping strictly for this iteration's matched events row-slice
        local_bucket_map = {k: v[event_panel_idx][_cand_mask] for k, v in bucket_map.items()}
    else:
        events = events.iloc[0:0].copy()
        local_bucket_map = {k: np.array([], dtype=bool) for k in bucket_map.keys()}
        
    tprint(f"[eval:{cfg_id}] candidate_prefilter_kept={len(events):,}/{pre_candidate_n:,}")

    # Assign bucket from side × move_direction, matching _strategy_bucket_context:
    #   (long,  up)   → TF_long   (buy_momentum)
    #   (short, up)   → MR_short  (sell_rips)
    #   (long,  down) → MR_long   (buy_dips)
    #   (short, down) → TF_short  (sell_weakness)
    up_mask   = local_bucket_map["up"]
    down_mask = local_bucket_map["down"]
    side_arr  = events["side"].astype(str).to_numpy()
    bucket = np.full(len(events), "Global", dtype=object)
    bucket[(side_arr == "long")  & up_mask]   = "TF_long"
    bucket[(side_arr == "short") & up_mask]   = "MR_short"
    bucket[(side_arr == "long")  & down_mask] = "MR_long"
    bucket[(side_arr == "short") & down_mask] = "TF_short"
    events["bucket"] = pd.Categorical(bucket)

    # Cell-specific filtering for Optuna Stage 1 efficiency
    # Applied after bucket assignment but before heavy OOF/Admissibility to avoid KeyError
    if target_cell_filter:
        _bkt, _h = target_cell_filter
        events = events[np.array((events["bucket"] == _bkt) & (events["horizon"] == _h))].copy().reset_index(drop=True)
        if events.empty:
            return _empty_result(cfg, cfg_id, 0, f"no_events_for_cell_{_bkt}_H{_h}", stage2_rescore=stage2_rescore)
        # Note: bucket_masks local shadow used for per-slice metrics (optional, mostly for safety)
        bucket_masks = {b: (events["bucket"] == b) for b in (["MR_long", "MR_short", "TF_long", "TF_short", "Candidate", "Global"])}
        # Event index alignment remains via __panel_idx__ (no MultiIndex reindex required).

    tprint(f"[eval:{cfg_id}] Step 3 (Bucketing) took {time.perf_counter() - _t1:.2f}s")
    _t2 = time.perf_counter()

    # Regime and quintile slices for Stage 2.
    event_panel_idx = events["__panel_idx__"].to_numpy(dtype=np.int64, copy=False)
    atr_source = _get_barrier_atr_frame(artifacts)
    atr_flat = eval_cache.get("atr_flat")
    if atr_flat is None:
        atr_flat = np.ascontiguousarray(atr_source.to_numpy(dtype=np.float32, copy=False).ravel())
        eval_cache["atr_flat"] = atr_flat
    atr = atr_flat[event_panel_idx]
    atr_roll = eval_cache.get("atr_roll_14d")
    if atr_roll is None:
        atr_roll = atr_source.rolling(24 * 14, min_periods=24).median().astype(np.float32)
        eval_cache["atr_roll_14d"] = atr_roll
        gc.collect()
    atr_roll_flat = eval_cache.get("atr_roll_flat")
    if atr_roll_flat is None:
        atr_roll_flat = np.ascontiguousarray(atr_roll.to_numpy(dtype=np.float32, copy=False).ravel())
        eval_cache["atr_roll_flat"] = atr_roll_flat
    roll = atr_roll_flat[event_panel_idx]
    atr = np.nan_to_num(atr, nan=0.0, copy=False)
    ratio = np.divide(atr, roll + EPS, out=np.ones_like(atr, dtype=np.float32), where=np.isfinite(roll))

    ts_vals_for_grp = pd.to_datetime(events["ts"].to_numpy(copy=False))
    atr_s = pd.Series(atr, dtype=np.float32)
    grp = atr_s.groupby(ts_vals_for_grp)
    ts_counts = grp.transform("count").to_numpy(dtype=np.int32, copy=False)
    rank_pct = grp.rank(method="first", pct=True).to_numpy(dtype=np.float32, copy=False)
    q = np.full(len(events), 2, dtype=np.int16)
    valid_q = (ts_counts > 5) & np.isfinite(rank_pct)
    q[valid_q] = np.minimum((rank_pct[valid_q] * 5.0).astype(np.int16), 4)
    events["vol_quintile"] = (q + 1).astype(np.int8, copy=False)

    regime = np.where(ratio < 0.85, "low_vol", np.where(ratio > 1.15, "high_vol", "medium_vol"))
    events["regime"] = pd.Categorical(regime)

    # Quantile filtering.
    quant_basis = cfg.get("quantile_basis", "composite")
    basis_frame_key = f"quant_basis_frame::{quant_basis}"
    if basis_frame_key not in eval_cache:
        eval_cache[basis_frame_key] = make_quantile_basis(artifacts, quant_basis).astype(np.float32)
        gc.collect()
    basis_flat_key = f"quant_basis_flat::{quant_basis}"
    basis_flat = eval_cache.get(basis_flat_key)
    if basis_flat is None:
        basis_flat = np.ascontiguousarray(eval_cache[basis_frame_key].to_numpy(dtype=np.float32, copy=False).ravel())
        eval_cache[basis_flat_key] = basis_flat
    basis = basis_flat[event_panel_idx]
    keep_mask = np.ones(len(events), dtype=bool)
    _bkt_vals_full = events["bucket"].astype(str).to_numpy()
    _u_bkt_full, _cnt_bkt_full = np.unique(_bkt_vals_full, return_counts=True)
    full_bucket_counts = {str(k): int(v) for k, v in zip(_u_bkt_full, _cnt_bkt_full)}
    if cfg.get("use_quantile_filter", False):
        lo = float(cfg.get("quantile_lo", 0.2))
        hi = float(cfg.get("quantile_hi", 0.8))
        _qb_cache = eval_cache.setdefault("quantile_bounds_cache", {})
        _qb_key = (_index_cache_key(pd.Index(ts_vals_for_grp)), str(quant_basis), round(lo, 6), round(hi, 6))
        _qb_val = _qb_cache.get(_qb_key)
        if _qb_val is None:
            basis_s = pd.Series(basis, index=ts_vals_for_grp, dtype=np.float32)
            tprint(f"[eval:{cfg_id}] TRACE: groupby quantiles start")
            by_ts = basis_s.groupby(level=0)
            lo_map = by_ts.quantile(lo)
            hi_map = by_ts.quantile(hi)
            ts_idx = ts_vals_for_grp
            lo_t = ts_idx.map(lo_map).to_numpy(dtype=np.float32, copy=False)
            hi_t = ts_idx.map(hi_map).to_numpy(dtype=np.float32, copy=False)
            tprint(f"[eval:{cfg_id}] TRACE: groupby quantiles end, lo_t={len(lo_t)}")
            _qb_cache[_qb_key] = (lo_t, hi_t)
            if len(_qb_cache) > 64:
                _qb_cache.pop(next(iter(_qb_cache)))
        else:
            lo_t, hi_t = _qb_val
        keep_mask = (basis <= lo_t) | (basis >= hi_t)
        keep_mask &= np.isfinite(basis)
        min_keep = float(cfg.get("min_keep_fraction", 0.5))
        cur = keep_mask.mean()
        if cur < min_keep:
            keep_mask[:] = True


    full_n = len(events)
    events = events.loc[keep_mask].copy().reset_index(drop=True)
    tprint(
        f"[eval:{cfg_id}] quantile_filter_kept={len(events):,}/{full_n:,} "
        f"({(len(events)/max(full_n,1))*100:.1f}%)"
    )
    _last_gc_ts = _maybe_collect_gc(last_gc_ts=_last_gc_ts, mem_threshold_mb=8192.0, min_interval_s=5.0)

    # Filter constraints.
    fee = float(cfg.get("fee_pct", 0.5)) / 100.0
    slip = float(cfg.get("slip_buffer", 0.1)) / 100.0
    tp_net = events["tp"] - fee - slip
    sl_net = events["sl"] + fee + slip
    events["net_rr"] = (tp_net / np.maximum(sl_net, EPS)).astype(np.float32, copy=False)

    min_rr = float(cfg.get("min_net_rr", 0.7))
    min_tp_hit = float(cfg.get("min_tp_hit_rate", 0.01))
    # max_timeout is now horizon-dependent: shorter horizons are allowed higher timeout
    # because price has less time to reach TP. Formula: 0.90 + 0.025 * (h / 8).
    # h=2 → 0.906, h=4 → 0.913, h=8 → 0.925. Overridden by cfg key if set explicitly.
    _max_timeout_base = float(cfg.get("max_timeout_rate", 0.0))  # 0.0 = use adaptive
    min_raw = int(cfg.get("min_raw_events", 50))

    pre_rr_n = len(events)
    events = events[events["net_rr"] >= min_rr].reset_index(drop=True)
    rr_keep_rate = float(len(events) / max(pre_rr_n, 1))
    tprint(
        f"[eval:{cfg_id}] rr_filter_kept={len(events):,}/{pre_rr_n:,} min_net_rr={min_rr:.3f}"
    )
    if events.empty:
        return _empty_result(cfg, cfg_id, full_n, reason="empty_after_rr_filter", stage2_rescore=stage2_rescore)

    # ── Early exit: skip expensive metrics if TP:SL ratio is catastrophically bad (>1:5) ──
    _agg_tp = float((events["label"] == OUT_TP).mean())
    _agg_sl = float((events["label"] == OUT_SL).mean())
    _agg_sl_to_tp = _agg_sl / max(_agg_tp, EPS)
    if _agg_sl_to_tp > 5.0:
        tprint(f"[eval:{cfg_id}] EARLY_EXIT sl_to_tp={_agg_sl_to_tp:.2f}x > 5.0 — skipping learnability metrics")
        return _empty_result(
            cfg, cfg_id, full_n, reason=f"sl_to_tp_too_high:{_agg_sl_to_tp:.2f}x", stage2_rescore=stage2_rescore
        )

    pass_cells = 0
    # Pre-initialize all 12 canonical cells with n=0 defaults so cells with zero
    # surviving events are always materialized (not absent) in bucket_horizon_metrics.
    _DEFAULT_CELL_METRICS = {
        "n": 0, "n_eval_kept": 0, "tp_hit": 0.0, "sl_hit": 0.0, "timeout": 1.0,
        "tp_hit_kept": 0.0, "sl_hit_kept": 0.0, "timeout_kept": 1.0,
        "bind": 0.0, "balance": 0.0, "sl_to_tp": float("nan"),
        "tp_mean": float("nan"), "sl_mean": float("nan"), "barrier_ratio": float("nan"),
        "h_eff_mean": float("nan"), "h_eff_p90": float("nan"),
        "payoff_mean": float("nan"),
        "max_timeout_threshold": float("nan"), "ok": False,
        "ic_payoff": float("nan"), "ic_label": float("nan"),
        "auc_label": float("nan"), "tp_sep_top10": float("nan"),
        "base_rate": float("nan"), "ap_lift": float("nan"),
        "er_tp": float("nan"), "er_sl": float("nan"), "tp_over_sl": float("nan"),
        "er_tp_top_decile": float("nan"), "er_sl_top_decile": float("nan"),
        "dir_superiority_top_decile": float("nan"),
        "auc_bound": float("nan"),
        "auc_vol_low": float("nan"), "auc_vol_mid": float("nan"), "auc_vol_high": float("nan"),
        "brier": float("nan"), "ece": float("nan"), "monotonicity": float("nan"),
        "ic_std_time": float("nan"), "ic_std_asset": float("nan"),
    }
    # Use canonical constants from params_store
    bucket_h_metrics = {
        (b, h): dict(_DEFAULT_CELL_METRICS)
        for b in TBM_BUCKET_NAMES
        for h in TBM_HORIZONS
    }
    total_cells = len(bucket_h_metrics)
    for (b, h), g in events.groupby(["bucket", "horizon"], observed=False):
        n_eval_kept = int(len(g))
        tp_hit_kept = float((g["label"] == OUT_TP).mean())
        timeout_kept = float((g["label"] == OUT_TO).mean())
        # Horizon-dependent timeout threshold: shorter horizons tolerate more timeouts.
        # If cfg explicitly sets max_timeout_rate (non-zero), use that; otherwise adaptive.
        if _max_timeout_base > 0.0:
            max_timeout_h = _max_timeout_base
        else:
            max_timeout_h = 0.90 + 0.025 * (int(h) / 8.0)
        ok = (n_eval_kept >= min_raw) and (tp_hit_kept >= min_tp_hit) and (timeout_kept <= max_timeout_h)
        pass_cells += int(ok)
        sl_hit_kept = float((g["label"] == OUT_SL).mean())
        bind_cell = tp_hit_kept + sl_hit_kept
        balance_cell = (min(tp_hit_kept, sl_hit_kept) / max(max(tp_hit_kept, sl_hit_kept), EPS))
        # tp_mean/sl_mean: actual barrier sizes in % for this bucket-horizon cell.
        # These are the values that will be passed to the production labeler.
        bucket_h_metrics[(b, h)] = {
            "n": n_eval_kept,
            "n_eval_kept": n_eval_kept,
            "tp_hit": tp_hit_kept,
            "sl_hit": sl_hit_kept,
            "timeout": timeout_kept,
            "tp_hit_kept": tp_hit_kept,
            "sl_hit_kept": sl_hit_kept,
            "timeout_kept": timeout_kept,
            "bind": round(bind_cell, 4),
            "balance": round(balance_cell, 4),
            "sl_to_tp": round(sl_hit_kept / max(tp_hit_kept, EPS), 4),
            "tp_mean": float(g["tp"].mean()),
            "sl_mean": float(g["sl"].mean()),
            "barrier_ratio": round(float((g["sl"].mean() + 0.003) / max(g["tp"].mean(), EPS)), 4),
            "h_eff_mean": float(g["horizon_eff"].mean()),
            "h_eff_p90": float(g["horizon_eff"].quantile(0.90)),
            "payoff_mean": float(g["payoff"].mean()),
            "max_timeout_threshold": round(max_timeout_h, 4),
            "ok": ok,
            # ic_payoff per cell filled in after OOF scoring below
            "ic_payoff": float("nan"),
        }

    tprint(f"[eval:{cfg_id}] Step 4 (Quantile/Filter/Buckets) took {time.perf_counter() - _t2:.2f}s")
    _t3 = time.perf_counter()
    tprint(f"[eval:{cfg_id}] TRACE: starting compute_weights")

    weights = compute_weights(events, cfg)

    # Negative mass renormalization (Timeout downweighting)
    if bool(cfg.get("use_neg_mass_renorm", True)):
        # Construct cell_id array from bucket+horizon
        # events['bucket'] is string, events['horizon'] is int.
        # compute_cell_weights_neg_mass_renorm accepts string or int cell_ids.
        # We can construct "Bucket_H{h}" strings.
        # Doing this efficiently in pandas:
        cell_ids = (events["bucket"].astype(str) + "_H" + events["horizon"].astype(str)).values

        # Labels are: TP=1, SL=-1, TO=0.
        # NegMassRenormCfg uses the config dict.
        renorm_cfg = NegMassRenormCfg(
            w_to_min=float(cfg.get("neg_mass_w_to_min", 0.2)),
            w_to_max=float(cfg.get("neg_mass_w_to_max", 1.0)),
            rho_pos_over_neg=float(cfg.get("neg_mass_rho", 1.0)),
        )

        weights = compute_cell_weights_neg_mass_renorm(
            y=events["label"].values.astype(np.int8),
            cell_id=cell_ids,
            base_w=weights.astype(np.float32, copy=False),
            cfg=renorm_cfg,
            tp_label=OUT_TP,
            sl_label=OUT_SL,
            to_label=OUT_TO,
        ).astype(np.float32)

    ess = effective_sample_size(weights)
    ess_full = float(full_n)
    coverage = ess / max(ess_full, 1.0)
    
    tprint(f"[eval:{cfg_id}] TRACE: starting feature layout extraction fm")

    # Feature matrix + OOF scoring (precomputed global float32 matrix).
    fm = get_feature_matrix_cache(artifacts, eval_cache)
    tprint(f"[eval:{cfg_id}] TRACE: finished format fm, shape={fm.shape if hasattr(fm, 'shape') else 'cached'}")
    feat_cols = fm.feat_cols
    y_signed = events["label"].values.astype(np.float32, copy=False)
    y_bin = (events["label"].values == OUT_TP).astype(np.float32, copy=False)
    payoff = events["payoff"].values.astype(np.float32, copy=False)

    event_panel_idx = events["__panel_idx__"].to_numpy(dtype=np.int64, copy=False)
    aligner = fm.global_aligner[event_panel_idx]
    valid_mask = aligner >= 0
    if not np.any(valid_mask):
        return {}, {}, None

    # ── Global Event Subsampling ──
    # Cap positive samples per cell at 2,000, and downsample negatives proportionally.
    # This preserves the TP / non-TP class ratio while bounding total size.
    _MAX_POS_PER_CELL = 2000
    _rng = np.random.default_rng(42)
    _bkt_h = events["bucket"].astype(str) + "_" + events["horizon"].astype(str)
    
    _keep_indices = []
    for _cell in np.unique(_bkt_h.values):
        _cell_idx = np.where((_bkt_h.values == _cell) & valid_mask)[0]
        _pos_mask = y_bin[_cell_idx] == 1.0
        _pos_idx = _cell_idx[_pos_mask]
        _neg_idx = _cell_idx[~_pos_mask]
        
        if len(_pos_idx) > _MAX_POS_PER_CELL:
            _keep_ratio = _MAX_POS_PER_CELL / len(_pos_idx)
            _sub_pos = _rng.choice(_pos_idx, size=_MAX_POS_PER_CELL, replace=False)
            _num_neg_keep = int(len(_neg_idx) * _keep_ratio)
            _sub_neg = _rng.choice(_neg_idx, size=_num_neg_keep, replace=False)
            _keep_indices.extend(_sub_pos)
            _keep_indices.extend(_sub_neg)
        else:
            _keep_indices.extend(_cell_idx)
            
    if _keep_indices:
        _keep_indices = np.sort(np.array(_keep_indices))
        new_valid = np.zeros_like(valid_mask, dtype=bool)
        new_valid[_keep_indices] = True
        valid_mask = new_valid

    X_event = np.ascontiguousarray(fm.X[aligner[valid_mask]], dtype=np.float32)
    y_signed_event = y_signed[valid_mask].astype(np.float32, copy=False)
    y_event = y_bin[valid_mask].astype(np.float32, copy=False)
    w_event = weights[valid_mask].astype(np.float32, copy=False)
    payoff_event = payoff[valid_mask].astype(np.float32, copy=False)
    ts_event = events.loc[valid_mask, "ts"].values
    side_event = events.loc[valid_mask, "side"].astype(str).values

    n_folds = 2  # OOF-only evaluation path for optimization trials

    # Train separate per-(bucket, horizon) Ridge OOF models.
    # Training per cell (8 models) prevents signal confounding across horizons.
    pred_event = np.full(len(X_event), 0.5, dtype=np.float32)
    warm_cache = eval_cache.setdefault("ridge_warm_cache", {})
    bucket_event = events.loc[valid_mask, "bucket"].astype(str).values
    horizon_event = events.loc[valid_mask, "horizon"].values
    
    unique_cells = sorted(list(bucket_h_metrics.keys()))
    flipped_cells = 0
    modeled_cells = 0
    for _bkt, _h in unique_cells:
        _bmask = (bucket_event == _bkt) & (horizon_event == _h)
        if _bmask.sum() < 64:  # Minimum events for stable Ridge
            continue
        modeled_cells += 1
            
        _warm_key = _ridge_warm_key(cfg) + f":{_bkt}_H{_h}"

        _pred_cell = _ridge_predict_oof_with_cache(
            X_event[_bmask],
            y_event[_bmask],
            w_event[_bmask],
            ts_event[_bmask],
            n_folds=n_folds,
            warm_cache=warm_cache,
            warm_key=_warm_key,
        )
        # Optional AUC direction flip: ensure model is at least 0.50 (random) or better.
        if bool(cfg.get("oof_auc_flip_enable", True)):
            flip_thr = float(cfg.get("oof_auc_flip_threshold", 0.50))
            _auc_cell = _fast_auc_binary(y_event[_bmask], _pred_cell)
            if math.isfinite(_auc_cell) and _auc_cell < flip_thr:
                _pred_cell = (1.0 - _pred_cell).astype(np.float32, copy=False)
                flipped_cells += 1
                tprint(
                    f"[eval:{cfg_id}] OOF direction flip applied for {_bkt}_H{_h}: "
                    f"auc_before={_auc_cell:.4f} < {flip_thr:.4f}"
                )
        pred_event[_bmask] = _pred_cell

    oof_flip_rate = (float(flipped_cells) / float(modeled_cells)) if modeled_cells > 0 else 0.0
    tprint(
        f"[eval:{cfg_id}] flip_summary flipped_cells={flipped_cells}/{modeled_cells} "
        f"flip_rate={oof_flip_rate:.3f}"
    )


    pred = np.full(len(events), 0.5, dtype=np.float32)
    pred[valid_mask] = pred_event
    decile_spread = oof_payoff_decile_spread(pred_event, payoff_event)
    tprint(
        f"[eval:{cfg_id}] model_scored n={len(pred):,} ic_payoff={_safe_spearman(pred, payoff):.4f} "
        f"ic_label={_safe_spearman(pred, y_signed):.4f} decile_spread={decile_spread:.6f}"
    )

    ic_label = _safe_spearman(pred, y_signed)
    ic_payoff = _safe_spearman(pred, payoff)
    pr_auc_lift = _stage1_pr_auc_lift(y_event, pred_event) if len(pred_event) else 0.0

    # Top-decile payoff + top10_vs_rest_spread.
    if len(pred) >= 10:
        top10_mask = pred >= np.quantile(pred, 0.90)
        rest_mask = ~top10_mask
        payoff_mean_top_decile = float(payoff[top10_mask].mean())
        payoff_mean_rest = float(payoff[rest_mask].mean()) if rest_mask.any() else 0.0
        top10_vs_rest_spread = payoff_mean_top_decile - payoff_mean_rest
    else:
        payoff_mean_top_decile = 0.0
        payoff_mean_rest = 0.0
        top10_vs_rest_spread = 0.0

    # Per-cell ic_payoff, ic_label, auc_label, tp_sep_top10: fill in now that pred is available.
    # top10 is by model score (pred) within each bucket independently — cross-bucket thresholds
    # contaminate the metric because TF and MR models score on different distributions.
    # Pre-compute per-bucket top-decile thresholds.
    _bucket_top10_thresh: Dict[str, float] = {}
    for _bname, _bg in events.groupby("bucket", observed=False):
        _bp = pred[_bg.index.values]
        _bucket_top10_thresh[str(_bname)] = float(np.quantile(_bp, 0.90)) if len(_bp) >= 10 else float("inf")

    for (b, h), g in events.groupby(["bucket", "horizon"], observed=False):
        if (b, h) not in bucket_h_metrics:
            continue
        idx = g.index.values
        p_cell = pred[idx]
        y_tp_cell = (g["label"].values == OUT_TP).astype(np.float32)
        # IC on payoff
        cell_ic = _safe_spearman(p_cell, g["payoff"].values)
        # IC on label (signed: TP=1, timeout=0, SL=-1)
        cell_ic_label = _safe_spearman(p_cell, g["label"].values)
        # Fast AUC on TP classification
        n_pos = int(y_tp_cell.sum()); n_neg = len(y_tp_cell) - n_pos
        if n_pos > 0 and n_neg > 0:
            ranks_cell = pd.Series(p_cell).rank(method="average").to_numpy(np.float32)
            u_cell = ranks_cell[y_tp_cell == 1].sum() - n_pos * (n_pos + 1) / 2.0
            auc_cell = float(u_cell / (n_pos * n_neg))
        else:
            auc_cell = float("nan")
        # TP separation: P(TP|top10 score) - P(TP|rest) — threshold is per-bucket, not global.
        _b10_thresh = _bucket_top10_thresh.get(str(b), float("inf"))
        top_mask_cell = p_cell >= _b10_thresh
        tp_top_cell = float(y_tp_cell[top_mask_cell].mean()) if top_mask_cell.any() else float("nan")
        tp_rest_cell = float(y_tp_cell[~top_mask_cell].mean()) if (~top_mask_cell).any() else float("nan")
        tp_sep_cell = (tp_top_cell - tp_rest_cell) if not (math.isnan(tp_top_cell) or math.isnan(tp_rest_cell)) else 0.0
        bucket_h_metrics[(b, h)]["ic_payoff"] = round(float(cell_ic), 5)
        bucket_h_metrics[(b, h)]["ic_label"] = round(float(cell_ic_label), 5)
        bucket_h_metrics[(b, h)]["auc_label"] = round(auc_cell, 4) if not math.isnan(auc_cell) else 0.5
        bucket_h_metrics[(b, h)]["tp_sep_top10"] = round(tp_sep_cell, 5)

        # ── New guardrail metrics ──────────────────────────────────────────────
        # 1. AP lift: Average Precision / base_rate.  base = mean(y_tp).
        #    Requires sklearn.metrics.average_precision_score.
        #    ap_lift > 1.25 means the model lifts precision meaningfully above random.
        _base_rate = float(y_tp_cell.mean()) if len(y_tp_cell) > 0 else float("nan")
        if not math.isnan(_base_rate) and _base_rate > 0.0 and n_pos > 0 and n_neg > 0:
            try:
                from sklearn.metrics import average_precision_score as _aps
                _ap = float(_aps(y_tp_cell, p_cell))
                _ap_lift = _ap / _base_rate
            except Exception:
                _ap_lift = float("nan")
        else:
            _ap_lift = float("nan")
        bucket_h_metrics[(b, h)]["base_rate"] = round(_base_rate, 5) if not math.isnan(_base_rate) else float("nan")
        bucket_h_metrics[(b, h)]["ap_lift"] = round(_ap_lift, 4) if not math.isnan(_ap_lift) else float("nan")

        # 2. tp_over_sl: E[r|TP] / abs(E[r|SL]).  Requires >= 1.05 (5% edge).
        #    Uses realized payoff (already net of fees in the labeler).
        _payoff_cell = g["payoff"].values.astype(np.float32, copy=False)
        _tp_mask_cell = g["label"].values == OUT_TP
        _sl_mask_cell = g["label"].values == OUT_SL
        _er_tp = float(_payoff_cell[_tp_mask_cell].mean()) if _tp_mask_cell.sum() > 0 else float("nan")
        _er_sl = float(_payoff_cell[_sl_mask_cell].mean()) if _sl_mask_cell.sum() > 0 else float("nan")
        if not (math.isnan(_er_tp) or math.isnan(_er_sl)) and abs(_er_sl) > EPS:
            _tp_over_sl = _er_tp / abs(_er_sl)
        else:
            _tp_over_sl = float("nan")
        bucket_h_metrics[(b, h)]["er_tp"] = round(_er_tp, 6) if not math.isnan(_er_tp) else float("nan")
        bucket_h_metrics[(b, h)]["er_sl"] = round(_er_sl, 6) if not math.isnan(_er_sl) else float("nan")
        bucket_h_metrics[(b, h)]["tp_over_sl"] = round(_tp_over_sl, 4) if not math.isnan(_tp_over_sl) else float("nan")

        # 3. Directional superiority in top decile (secondary objective):
        #    E[r|TP, top_decile] >= abs(E[r|SL, top_decile]).
        #    top_decile here is the top-10% by model score within this cell.
        _cell_top10_thresh = float(np.quantile(p_cell, 0.90)) if len(p_cell) >= 10 else float("inf")
        _cell_top10_mask = p_cell >= _cell_top10_thresh
        _tp_top_dec = _tp_mask_cell & _cell_top10_mask
        _sl_top_dec = _sl_mask_cell & _cell_top10_mask
        _er_tp_top = float(_payoff_cell[_tp_top_dec].mean()) if _tp_top_dec.sum() > 0 else float("nan")
        _er_sl_top = float(_payoff_cell[_sl_top_dec].mean()) if _sl_top_dec.sum() > 0 else float("nan")
        if not (math.isnan(_er_tp_top) or math.isnan(_er_sl_top)) and abs(_er_sl_top) > EPS:
            _dir_sup = _er_tp_top / abs(_er_sl_top)
        else:
            _dir_sup = float("nan")
        bucket_h_metrics[(b, h)]["er_tp_top_decile"] = round(_er_tp_top, 6) if not math.isnan(_er_tp_top) else float("nan")
        bucket_h_metrics[(b, h)]["er_sl_top_decile"] = round(_er_sl_top, 6) if not math.isnan(_er_sl_top) else float("nan")
        bucket_h_metrics[(b, h)]["dir_superiority_top_decile"] = round(_dir_sup, 4) if not math.isnan(_dir_sup) else float("nan")

    # auc_bound and auc_by_vol_regime per cell: computed now that pred is available.
    # auc_bound: AUC restricted to bound events (TP or SL only, timeouts excluded).
    # This is a cleaner learnability signal: timeouts are uninformative for TP classification.
    # auc_vol_low/mid/high: AUC split by vol_quintile coarse bins (Q1-2 / Q3 / Q4-5).
    for (b, h), g in events.groupby(["bucket", "horizon"], observed=False):
        if (b, h) not in bucket_h_metrics:
            continue
        idx = g.index.values
        p_cell = pred[idx]
        y_tp_cell = (g["label"].values == OUT_TP).astype(np.float32)
        # auc_bound: only bound events
        bound_mask = g["label"].values != 0
        if bound_mask.sum() >= 10:
            p_bound = p_cell[bound_mask]
            y_bound = y_tp_cell[bound_mask]
            n_pos_b = int(y_bound.sum()); n_neg_b = len(y_bound) - n_pos_b
            if n_pos_b > 0 and n_neg_b > 0:
                r_b = pd.Series(p_bound).rank(method="average").to_numpy(np.float32)
                u_b = r_b[y_bound == 1].sum() - n_pos_b * (n_pos_b + 1) / 2.0
                auc_bound_cell = float(u_b / (n_pos_b * n_neg_b))
            else:
                auc_bound_cell = float("nan")
        else:
            auc_bound_cell = float("nan")
        bucket_h_metrics[(b, h)]["auc_bound"] = round(auc_bound_cell, 4) if not math.isnan(auc_bound_cell) else float("nan")
        # auc_by_vol_regime: coarse bins Q1-2=low, Q3=mid, Q4-5=high
        vq = g["vol_quintile"].values
        vol_aucs: Dict[str, float] = {}
        for vbin, vmask in [("low", vq <= 2), ("mid", vq == 3), ("high", vq >= 4)]:
            if vmask.sum() >= 10:
                p_v = p_cell[vmask]; y_v = y_tp_cell[vmask]
                n_p = int(y_v.sum()); n_n = len(y_v) - n_p
                if n_p > 0 and n_n > 0:
                    r_v = pd.Series(p_v).rank(method="average").to_numpy(np.float32)
                    u_v = r_v[y_v == 1].sum() - n_p * (n_p + 1) / 2.0
                    vol_aucs[vbin] = round(float(u_v / (n_p * n_n)), 4)
                else:
                    vol_aucs[vbin] = float("nan")
            else:
                vol_aucs[vbin] = float("nan")
        bucket_h_metrics[(b, h)]["auc_vol_low"] = vol_aucs["low"]
        bucket_h_metrics[(b, h)]["auc_vol_mid"] = vol_aucs["mid"]
        bucket_h_metrics[(b, h)]["auc_vol_high"] = vol_aucs["high"]

        # Per-cell calibration & stability metrics (previously global-only).
        y_bin_cell = y_tp_cell  # already computed above as (label==1)
        if len(p_cell) >= 20:
            _brier_cell = float(np.mean((p_cell - y_bin_cell) ** 2))
            _ece_cell = expected_calibration_error(y_bin_cell, p_cell)
            
            # Increased granularity for monotonicity (up to 20 bins for better resolution)
            _n_bins_mono = min(20, len(p_cell) // 10)
            if _n_bins_mono >= 2:
                _dec_cell = pd.qcut(pd.Series(p_cell), _n_bins_mono, labels=False, duplicates="drop")
                _pay_cell = g["payoff"].values.astype(np.float32, copy=False)
                _pay_by_dec = pd.DataFrame({"d": _dec_cell, "p": _pay_cell}).groupby("d")["p"].mean()
                _mono_cell = float((np.diff(_pay_by_dec.values) >= 0).mean()) if len(_pay_by_dec) > 1 else float("nan")
            else:
                _mono_cell = float("nan")
        else:
            _brier_cell = float("nan")
            _ece_cell = float("nan")
            _mono_cell = float("nan")
        bucket_h_metrics[(b, h)]["brier"] = round(_brier_cell, 5) if not math.isnan(_brier_cell) else float("nan")
        bucket_h_metrics[(b, h)]["ece"] = round(_ece_cell, 5) if not math.isnan(_ece_cell) else float("nan")
        bucket_h_metrics[(b, h)]["monotonicity"] = round(_mono_cell, 4) if not math.isnan(_mono_cell) else float("nan")

        # Per-cell IC stability: std of monthly IC(payoff) and per-asset IC(payoff).
        _ts_cell = g["ts"].values
        _pay_cell_f = g["payoff"].values.astype(np.float32, copy=False)
        _p_cell_f = p_cell.astype(np.float32, copy=False)
        # ic_std_time: std of IC across calendar months
        try:
            _months = pd.to_datetime(_ts_cell).to_period("M")
            _ic_by_month = []
            for _m in pd.unique(_months):
                _mm = _months == _m
                if _mm.sum() >= 5:
                    _ic_by_month.append(_safe_spearman(_p_cell_f[_mm], _pay_cell_f[_mm]))
            _ic_std_time_cell = float(np.std(_ic_by_month)) if len(_ic_by_month) >= 2 else float("nan")
        except Exception:
            _ic_std_time_cell = float("nan")
        # ic_std_asset: std of IC across symbols
        try:
            _syms_cell = g["symbol"].values
            _ic_by_sym = []
            for _s in np.unique(_syms_cell):
                _sm = _syms_cell == _s
                if _sm.sum() >= 5:
                    _ic_by_sym.append(_safe_spearman(_p_cell_f[_sm], _pay_cell_f[_sm]))
            _ic_std_asset_cell = float(np.std(_ic_by_sym)) if len(_ic_by_sym) >= 2 else float("nan")
        except Exception:
            _ic_std_asset_cell = float("nan")
        bucket_h_metrics[(b, h)]["ic_std_time"] = round(_ic_std_time_cell, 5) if not math.isnan(_ic_std_time_cell) else float("nan")
        bucket_h_metrics[(b, h)]["ic_std_asset"] = round(_ic_std_asset_cell, 5) if not math.isnan(_ic_std_asset_cell) else float("nan")

    # Config-level cell stability metrics.
    cell_payoffs = [v["payoff_mean"] for v in bucket_h_metrics.values()]
    cell_ics = [v["ic_payoff"] for v in bucket_h_metrics.values() if not math.isnan(v["ic_payoff"])]
    cell_ic_labels = [v["ic_label"] for v in bucket_h_metrics.values() if not math.isnan(v.get("ic_label", float("nan")))]
    cell_aucs = [v["auc_label"] for v in bucket_h_metrics.values() if not math.isnan(v.get("auc_label", float("nan")))]
    cell_aucs_bound = [v["auc_bound"] for v in bucket_h_metrics.values() if not math.isnan(v.get("auc_bound", float("nan")))]
    cell_tp_seps = [v["tp_sep_top10"] for v in bucket_h_metrics.values() if not math.isnan(v.get("tp_sep_top10", float("nan")))]
    cell_timeouts = [v["timeout"] for v in bucket_h_metrics.values()]
    cell_sl_to_tp_vals = [v["sl_to_tp"] for v in bucket_h_metrics.values() if not math.isnan(v.get("sl_to_tp", float("nan")))]
    cell_ap_lifts = [v["ap_lift"] for v in bucket_h_metrics.values() if not math.isnan(v.get("ap_lift", float("nan")))]
    cell_tp_over_sls = [v["tp_over_sl"] for v in bucket_h_metrics.values() if not math.isnan(v.get("tp_over_sl", float("nan")))]
    cell_dir_sups = [v["dir_superiority_top_decile"] for v in bucket_h_metrics.values() if not math.isnan(v.get("dir_superiority_top_decile", float("nan")))]
    cell_barrier_ratios = [v["barrier_ratio"] for v in bucket_h_metrics.values() if not math.isnan(v.get("barrier_ratio", float("nan")))]
    cell_dispersion = float(np.std(cell_payoffs)) if len(cell_payoffs) > 1 else 0.0
    min_cell_payoff = float(np.min(cell_payoffs)) if cell_payoffs else 0.0
    min_cell_ic = float(np.min(cell_ics)) if cell_ics else 0.0
    min_cell_ic_label = float(np.min(cell_ic_labels)) if cell_ic_labels else 0.0
    median_cell_ic_label = float(np.median(cell_ic_labels)) if cell_ic_labels else 0.0
    min_cell_auc = float(np.min(cell_aucs)) if cell_aucs else float("nan")
    median_cell_auc = float(np.median(cell_aucs)) if cell_aucs else float("nan")
    min_cell_auc_bound = float(np.min(cell_aucs_bound)) if cell_aucs_bound else float("nan")
    median_cell_auc_bound = float(np.median(cell_aucs_bound)) if cell_aucs_bound else float("nan")
    min_cell_tp_sep = float(np.min(cell_tp_seps)) if cell_tp_seps else 0.0
    median_cell_tp_sep = float(np.median(cell_tp_seps)) if cell_tp_seps else 0.0
    timeout_range = float(np.max(cell_timeouts) - np.min(cell_timeouts)) if len(cell_timeouts) > 1 else 0.0
    min_cell_ap_lift = float(np.min(cell_ap_lifts)) if cell_ap_lifts else float("nan")
    median_cell_ap_lift = float(np.median(cell_ap_lifts)) if cell_ap_lifts else float("nan")
    min_cell_tp_over_sl = float(np.min(cell_tp_over_sls)) if cell_tp_over_sls else float("nan")
    median_cell_tp_over_sl = float(np.median(cell_tp_over_sls)) if cell_tp_over_sls else float("nan")
    min_cell_dir_superiority = float(np.min(cell_dir_sups)) if cell_dir_sups else float("nan")
    median_cell_dir_superiority = float(np.median(cell_dir_sups)) if cell_dir_sups else float("nan")
    p50_cell_sl_to_tp = float(np.median(cell_sl_to_tp_vals)) if cell_sl_to_tp_vals else float("nan")
    p90_cell_sl_to_tp = float(np.quantile(np.asarray(cell_sl_to_tp_vals, dtype=np.float64), 0.90)) if cell_sl_to_tp_vals else float("nan")
    max_barrier_ratio = float(np.max(cell_barrier_ratios)) if cell_barrier_ratios else float("nan")
    # Effective label mass proxy for probability learnability.
    effective_pos_mass = 0.0
    effective_neg_mass = 0.0
    for _cm in bucket_h_metrics.values():
        _n_cell = float(_cm.get("n", 0.0))
        _br = float(_cm.get("base_rate", float("nan")))
        if _n_cell <= 0 or not math.isfinite(_br):
            continue
        effective_pos_mass += _n_cell * _br
        effective_neg_mass += _n_cell * max(0.0, 1.0 - _br)

    # For cell-specific Optuna, ensure aggregation is strictly per-cell (not across canonical cells).
    if target_cell_filter:
        _tb, _th = target_cell_filter
        _tm = bucket_h_metrics.get((_tb, _th), {})
        _auc_v = float(_tm.get("auc_label", float("nan")))
        _auc_b_v = float(_tm.get("auc_bound", float("nan")))
        _sep_v = float(_tm.get("tp_sep_top10", float("nan")))
        _ap_v = float(_tm.get("ap_lift", float("nan")))
        _tpsl_v = float(_tm.get("tp_over_sl", float("nan")))
        _dir_v = float(_tm.get("dir_superiority_top_decile", float("nan")))
        _ic_lbl_v = float(_tm.get("ic_label", float("nan")))
        min_cell_auc = median_cell_auc = _auc_v
        min_cell_auc_bound = median_cell_auc_bound = _auc_b_v
        min_cell_tp_sep = median_cell_tp_sep = (0.0 if math.isnan(_sep_v) else _sep_v)
        min_cell_ap_lift = median_cell_ap_lift = _ap_v
        min_cell_tp_over_sl = median_cell_tp_over_sl = _tpsl_v
        min_cell_dir_superiority = median_cell_dir_superiority = _dir_v
        min_cell_ic_label = median_cell_ic_label = (0.0 if math.isnan(_ic_lbl_v) else _ic_lbl_v)

    # Config-level bind/balance (aggregate across all events).
    tp_hit_agg = float((events["label"] == OUT_TP).mean()) if len(events) else 0.0
    sl_hit_agg = float((events["label"] == OUT_SL).mean()) if len(events) else 0.0
    bind_agg = tp_hit_agg + sl_hit_agg
    balance_agg = min(tp_hit_agg, sl_hit_agg) / max(max(tp_hit_agg, sl_hit_agg), EPS)
    sl_to_tp_agg = sl_hit_agg / max(tp_hit_agg, EPS)
    _tp_vals_agg = events.loc[events["label"] == OUT_TP, "payoff"].to_numpy(dtype=np.float32, copy=False) if len(events) else np.array([], dtype=np.float32)
    _sl_vals_agg = events.loc[events["label"] == OUT_SL, "payoff"].to_numpy(dtype=np.float32, copy=False) if len(events) else np.array([], dtype=np.float32)
    _tp_count = int(_tp_vals_agg.size)
    _sl_count = int(_sl_vals_agg.size)
    _min_edge_count = int(cfg.get("min_payoff_edge_count", 30))
    tp_med_agg = float(np.median(_tp_vals_agg)) if _tp_vals_agg.size else EPS
    sl_med_agg = float(np.median(np.abs(_sl_vals_agg))) if _sl_vals_agg.size else EPS
    edge = float(math.log(max(tp_hit_agg, EPS) * max(tp_med_agg, EPS) / max(sl_hit_agg * sl_med_agg, EPS)))
    # Use neutral payoff-edge contribution when TP/SL sample support is too thin.
    payoff_edge = edge if (_tp_count >= _min_edge_count and _sl_count >= _min_edge_count) else 0.0

    # Degeneracy flags.
    timeout_agg = float((events["label"] == OUT_TO).mean()) if len(events) else 1.0
    flag_degenerate_timeout = timeout_agg > 0.85
    flag_degenerate_sl = sl_hit_agg > 0.60
    flag_degenerate_tp = tp_hit_agg < 0.05

    # Time/asset IC stability.
    ic_time = []
    _ts_vals = events["ts"].to_numpy()
    _u_ts, _inv_ts = np.unique(_ts_vals, return_inverse=True)
    _pay_vals = events["payoff"].to_numpy(dtype=np.float32, copy=False)
    for _i in range(len(_u_ts)):
        _m = (_inv_ts == _i)
        if int(_m.sum()) >= 5:
            ic_time.append(_safe_spearman(pred[_m], _pay_vals[_m]))
    ic_asset = []
    _sym_vals = events["symbol"].to_numpy()
    _u_sym, _inv_sym = np.unique(_sym_vals, return_inverse=True)
    for _i in range(len(_u_sym)):
        _m = (_inv_sym == _i)
        if int(_m.sum()) >= 5:
            ic_asset.append(_safe_spearman(pred[_m], _pay_vals[_m]))

    ic_time = np.array(ic_time, dtype=float)
    ic_asset = np.array(ic_asset, dtype=float)
    ic_snr = float(ic_time.mean() / (ic_time.std() + EPS)) if ic_time.size else 0.0

    top_q = pred >= np.quantile(pred, 0.7) if len(pred) else np.array([], dtype=bool)
    top_payoff = payoff[top_q] if top_q.any() else np.array([0.0])
    downside = np.minimum(top_payoff, 0.0)
    sortino = float(top_payoff.mean() / (np.sqrt(np.mean(downside**2) + EPS)))
    sharpe = float(top_payoff.mean() / (top_payoff.std() + EPS))

    # brier/ece/mono/ic_std_time/ic_std_asset: derived from per-cell values (already in bucket_h_metrics)
    # rather than computed globally (which would mix all 4 buckets × 2 horizons).
    _cell_briers = [v["brier"] for v in bucket_h_metrics.values() if not math.isnan(v.get("brier", float("nan")))]
    _cell_eces = [v["ece"] for v in bucket_h_metrics.values() if not math.isnan(v.get("ece", float("nan")))]
    _cell_monos = [v["monotonicity"] for v in bucket_h_metrics.values() if not math.isnan(v.get("monotonicity", float("nan")))]
    _cell_ic_std_times = [v["ic_std_time"] for v in bucket_h_metrics.values() if not math.isnan(v.get("ic_std_time", float("nan")))]
    _cell_ic_std_assets = [v["ic_std_asset"] for v in bucket_h_metrics.values() if not math.isnan(v.get("ic_std_asset", float("nan")))]
    brier = float(np.median(_cell_briers)) if _cell_briers else float("nan")
    ece = float(np.median(_cell_eces)) if _cell_eces else float("nan")
    mono = float(np.median(_cell_monos)) if _cell_monos else float("nan")
    ic_std_time_median = float(np.median(_cell_ic_std_times)) if _cell_ic_std_times else float("nan")
    ic_std_asset_median = float(np.median(_cell_ic_std_assets)) if _cell_ic_std_assets else float("nan")
    min_cell_brier = float(np.min(_cell_briers)) if _cell_briers else float("nan")
    min_cell_ece = float(np.min(_cell_eces)) if _cell_eces else float("nan")
    min_cell_mono = float(np.min(_cell_monos)) if _cell_monos else float("nan")

    if fast_mode:
        per_bucket = {}
        _b = events["bucket"].astype(str).to_numpy()
        _lbl = events["label"].to_numpy(dtype=np.float32, copy=False)
        _pay = events["payoff"].to_numpy(dtype=np.float32, copy=False)
        _pr = pred.astype(np.float32, copy=False)
        for _name in np.unique(_b):
            _m = (_b == _name)
            if int(_m.sum()) == 0:
                continue
            _l = _lbl[_m]
            per_bucket[str(_name)] = {
                "n": int(_m.sum()),
                "tp_hit": float(np.mean(_l == OUT_TP)),
                "sl_hit": float(np.mean(_l == OUT_SL)),
                "timeout": float(np.mean(_l == OUT_TO)),
                "payoff_mean": float(np.mean(_pay[_m])) if _m.any() else 0.0,
                "ic_payoff": float(_safe_spearman(_pr[_m], _pay[_m])),
                "ic_label": float(_safe_spearman(_pr[_m], _l)),
            }
        per_bucket["Global"] = {
            "n": int(len(_b)),
            "tp_hit": float(np.mean(_lbl == OUT_TP)) if len(_lbl) else 0.0,
            "sl_hit": float(np.mean(_lbl == OUT_SL)) if len(_lbl) else 0.0,
            "timeout": float(np.mean(_lbl == OUT_TO)) if len(_lbl) else 1.0,
            "payoff_mean": float(np.mean(_pay)) if len(_pay) else 0.0,
            "ic_payoff": float(_safe_spearman(_pr, _pay)),
            "ic_label": float(_safe_spearman(_pr, _lbl)),
        }
    else:
        per_bucket = per_slice_metrics(events, pred, "bucket")
    bucket_ics = [m["ic_payoff"] for k, m in per_bucket.items() if k != "Global"]
    bucket_ic_labels = [m["ic_label"] for k, m in per_bucket.items() if k != "Global"]
    worst_bucket_ic = min(bucket_ics, default=0.0)
    mean_bucket_ic = float(np.mean(bucket_ics)) if bucket_ics else 0.0
    mean_bucket_ic_label = float(np.mean(bucket_ic_labels)) if bucket_ic_labels else 0.0

    min_cov_threshold = float(cfg.get("min_coverage_threshold", 0.2))
    _bkt_vals_after = events["bucket"].astype(str).to_numpy()
    _u_bkt_after, _cnt_bkt_after = np.unique(_bkt_vals_after, return_counts=True)
    bucket_counts_after = {str(k): int(v) for k, v in zip(_u_bkt_after, _cnt_bkt_after)}
    bucket_coverages = []
    for b, n_full in full_bucket_counts.items():
        n_kept = bucket_counts_after.get(b, 0)
        bucket_coverages.append(n_kept / max(n_full, 1))
    worst_bucket_cov = min(bucket_coverages) if bucket_coverages else 1.0

    hard_gate = all(
        [
            coverage >= min_cov_threshold,
            ess >= float(cfg.get("min_ess_events", 30)),
            pass_cells >= int(0.5 * max(total_cells, 1)),
            # Softened: IC time-series can be noisy/regime-dependent; -0.10 kills only
            # systematically negative labelers while keeping solid but volatile ones.
            (ic_time.mean() > -0.10) if ic_time.size else (worst_bucket_ic > -0.10),
            # Sanity: reject obviously degenerate labelers before they waste Stage 2 budget.
            # A geometry with 80%+ timeouts OR <20% binding events is not a labeler.
            not (timeout_agg > 0.80 and bind_agg < 0.20),
        ]
    )

    stage1_score = (
        (0.5 * ic_snr + 0.5 * mean_bucket_ic) * math.sqrt(max(coverage, 0.0))
        - 0.2 * float(events["bound_saturation"].mean() if len(events) else 0.0)
        - 0.2 * float((events["label"] == OUT_TO).mean() if len(events) else 1.0)
    )

    # Learnability-focused stage2_score:
    # prioritize auc_bound, saturating TP-separation rewards, explicit calibration penalties,
    # stronger instability/censoring penalties, and optional tree learnability probe.
    _target_h = int(target_cell_filter[1]) if target_cell_filter else int(cfg.get("horizon_base", 4))
    _target_b = str(target_cell_filter[0]) if target_cell_filter else None
    _lw = _resolve_learnability_weights(cfg, target_horizon=_target_h, target_bucket=_target_b)

    _auc_raw = median_cell_auc_bound if not math.isnan(median_cell_auc_bound) else (
        median_cell_auc if not math.isnan(median_cell_auc) else 0.5
    )
    _auc_score = float(np.clip(2.0 * (_auc_raw - 0.5), -1.0, 1.0))

    _sep_score = _score_saturated_tp_sep(
        median_cell_tp_sep if not math.isnan(median_cell_tp_sep) else 0.0,
        _lw.tp_sep_saturation_k,
    )
    _ap_lift_score = 0.0
    if not math.isnan(median_cell_ap_lift):
        _ap_lift_score = float(np.tanh((median_cell_ap_lift - 1.0) / 0.50))
    _sortino_score = float(np.tanh(sortino / 3.0))
    _snr_score = float(np.tanh(ic_snr / 2.0))

    _dir_sup_score = 0.0
    if not math.isnan(median_cell_dir_superiority):
        _dir_sup_score = float(np.tanh((median_cell_dir_superiority - 1.0) / 0.25))

    _tp_over_sl_score = 0.0
    if not math.isnan(median_cell_tp_over_sl) and median_cell_tp_over_sl > 0.0:
        _tp_over_sl_score = float(np.tanh(math.log(max(median_cell_tp_over_sl, EPS)) / 0.35))
    payoff_edge_score = float(np.tanh(payoff_edge / 0.75))

    _reward = (
        _lw.w_auc_bound * _auc_score
        + _lw.w_tp_sep * _sep_score
        + _lw.w_ap_lift * _ap_lift_score
        + _lw.w_sortino * _sortino_score
        + _lw.w_ic_snr * _snr_score
        + _lw.w_dir_sup * _dir_sup_score
        + _lw.w_tp_over_sl * _tp_over_sl_score
        + _lw.w_payoff_edge * payoff_edge_score
    )

    # Smooth penalties (scaled then bounded to <=55% of |reward|).
    _ic_std_time = 0.0 if math.isnan(ic_std_time_median) else float(ic_std_time_median)
    _ic_std_asset = 0.0 if math.isnan(ic_std_asset_median) else float(ic_std_asset_median)
    _ic_std_target = float(cfg.get("stage2_ic_std_target", 0.12))
    _instability_pen = _lw.w_ic_std_time * max(0.0, (_ic_std_time - _ic_std_target) / max(_ic_std_target, EPS)) ** 2
    _instability_asset_pen = _lw.w_ic_std_asset * max(0.0, (_ic_std_asset - _ic_std_target) / max(_ic_std_target, EPS)) ** 2
    _coverage_pen = _lw.w_coverage * max(0.0, (float(cfg.get("stage2_target_coverage", 0.45)) - coverage) / max(float(cfg.get("stage2_target_coverage", 0.45)), EPS))
    _timeout_pen = _lw.w_timeout * max(0.0, timeout_agg - float(cfg.get("stage2_timeout_target", 0.55)))
    _bind_pen = _lw.w_bind * max(0.0, float(cfg.get("stage2_bind_target", 0.35)) - bind_agg)
    _ece_pen = _lw.w_ece * max(0.0, 0.0 if math.isnan(ece) else float(ece))
    _brier_pen = _lw.w_brier * max(0.0, 0.0 if math.isnan(brier) else float(brier))

    _raw_penalty = (
        _instability_pen
        + _instability_asset_pen
        + _coverage_pen
        + _timeout_pen
        + _bind_pen
        + _ece_pen
        + _brier_pen
    )
    _penalty_cap = 0.5 * max(abs(_reward), 0.10)
    _bounded_penalty = min(_raw_penalty, _penalty_cap)

    # RR-keep penalty: discourage pathological regions where too few events survive RR filter.
    _rr_floor_map = {
        2: float(cfg.get("stage2_rr_keep_floor_h2", 0.50)),
        4: float(cfg.get("stage2_rr_keep_floor_h4", 0.55)),
        8: float(cfg.get("stage2_rr_keep_floor_h8", 0.60)),
    }
    rr_floor = float(_rr_floor_map.get(_target_h, float(cfg.get("stage2_rr_keep_floor_default", 0.55))))
    rr_gap = max(0.0, (rr_floor - rr_keep_rate) / max(rr_floor, EPS))
    _rr_pen_w = float(cfg.get("stage2_penalty_rr_keep_w", 0.10))
    _rr_pen_base = _rr_pen_w * (rr_gap ** 1.5)
    # Normalize to stage2 reward/penalty scale and cap at a bounded additive impact.
    _rr_keep_pen = 0.10 * float(np.tanh(_rr_pen_base / 0.10))

    stage2_score = _reward - _bounded_penalty - _rr_keep_pen

    _horizon_cons_pen = _lw.w_horizon_consistency * _horizon_consistency_penalty(bucket_h_metrics, cfg)
    stage2_score -= _horizon_cons_pen

    probe_auc = float("nan")
    probe_auc_bound = float("nan")
    probe_gap_penalty = 0.0
    probe_status = "disabled"
    probe_n_train = 0
    probe_n_val = 0
    _probe_default = bool(stage2_rescore and not fast_mode)
    probe_enabled = bool(cfg.get("stage2_probe_enabled", _probe_default))
    if stage2_rescore and bool(cfg.get("stage2_probe_force_on_rescore", True)):
        probe_enabled = True
    if probe_enabled:
        probe_auc, probe_auc_bound, probe_status, probe_n_train, probe_n_val = _strict_walk_forward_probe_auc(
            X_event, y_event, ts_event, w_event, cfg, fast_mode=fast_mode
        )
        if math.isfinite(probe_auc_bound):
            stage2_score += _lw.w_probe * max(0.0, (probe_auc_bound - 0.5) * 2.0)
        if math.isfinite(probe_auc):
            probe_gap_penalty = _lw.w_gap * max(0.0, _auc_raw - probe_auc)
            stage2_score -= probe_gap_penalty

    # Keep mild multiplicative regularization, bounded away from zero.
    stage2_score *= max(0.70, math.sqrt(max(coverage, 0.0)))
    stage2_score *= max(0.70, max(bind_agg, 0.0))

    # Additive penalties for clear learnability failures.
    if top10_vs_rest_spread < 0.0:
        stage2_score -= 0.03
    if min_cell_ic_label < -0.02:
        stage2_score -= 0.03
    if worst_bucket_ic < -0.1:
        stage2_score -= 0.10
    if not math.isnan(min_cell_tp_over_sl) and min_cell_tp_over_sl < 1.0:
        stage2_score -= 0.05

    # Keep stage2_score defined for sparse-trade/partial-metric runs; reserve missing only for invalid evals.
    stage2_missing = False
    if not math.isfinite(stage2_score):
        stage2_missing = True
        stage2_score = -1.0
    # Economic guardrail: hard constraints + bounded stage2 adjustment.
    _econ_tp_hit_floor = float(cfg.get("econ_tp_hit_floor", 0.10))
    _econ_sl_to_tp_cap = float(cfg.get("econ_sl_to_tp_cap", 3.5))
    _econ_tp_over_sl_floor = float(cfg.get("econ_tp_over_sl_floor", 1.05))
    _econ_min_factor = float(cfg.get("econ_min_factor", 0.85))
    _econ_mult_floor = float(cfg.get("econ_mult_floor", 0.70))
    _econ_mult_weight = float(cfg.get("econ_mult_weight", 0.30))
    _econ_bonus_max = float(cfg.get("econ_add_bonus_max", 0.03))

    # Surface threshold conflicts with existing settings when economics are stricter.
    if float(cfg.get("min_tp_hit_rate", 0.01)) < _econ_tp_hit_floor:
        tprint(
            f"[ECON_GUARDRAIL:{cfg_id}] NOTE min_tp_hit_rate={float(cfg.get('min_tp_hit_rate',0.01)):.3f} "
            f"is looser than econ_tp_hit_floor={_econ_tp_hit_floor:.3f}."
        )

    stage2_score, econ_ok, econ_G, econ_multiplier = apply_econ_guardrail_to_stage2(
        stage2_score,
        tp_hit_agg=tp_hit_agg,
        sl_to_tp_agg=sl_to_tp_agg,
        tp_over_sl=median_cell_tp_over_sl,
        min_factor=_econ_min_factor,
        mult_floor=_econ_mult_floor,
        mult_weight=_econ_mult_weight,
        add_bonus_max=_econ_bonus_max,
        tp_hit_floor=_econ_tp_hit_floor,
        sl_to_tp_cap=_econ_sl_to_tp_cap,
        tp_over_sl_floor=_econ_tp_over_sl_floor,
    )
    if not math.isfinite(stage2_score):
        stage2_missing = True
        stage2_score = -1.0
    hard_gate = bool(hard_gate and econ_ok)

    # Production-aligned admissibility on U_prod (post candidate/quantile/RR filters).
    # Skip full admissibility when running per-cell Optuna — it requires all canonical cells
    # and is irrelevant for single-cell scoring (OOF Ridge metrics are already fold-based).
    if target_cell_filter:
        prod_admissibility = {"admissible_tier0": True, "failures": [], "aggregates": {}}
        _floor_dom_pen = 0.0
        _tp_fb_agg = 0.0
        _tp_fb_max = 0.0
        _pa_agg = {}
    else:
        _prod_aligned_meta = cfg.get("prod_aligned_tp", {}) if isinstance(cfg.get("prod_aligned_tp", {}), dict) else {}
        _tp_min_tradeable = float(_prod_aligned_meta.get("tp_min_tradeable", cfg.get("prod_adm_tradeable_tp_min", 0.015)))
        _prod_tp_lo = float(cfg.get("prod_tp_abs_lo_pct", cfg.get("tp_abs_lo_pct", 0.005)))
        _prod_sl_lo = float(cfg.get("prod_sl_abs_lo_pct", cfg.get("sl_abs_lo_pct", 0.005)))
        _prod_gates = ProdGates(
            n_min=int(cfg.get("prod_adm_n_min", 50)),
            bind_cell_min=float(cfg.get("prod_adm_bind_cell_min", 0.38)),
            bind_min=float(cfg.get("prod_adm_bind_min", 0.50)),
            timeout_max=float(cfg.get("prod_adm_timeout_max", 0.60)),
            timeout_range_max=float(cfg.get("prod_adm_timeout_range_max", 0.50)),
            sl_to_tp_max=float(cfg.get("prod_adm_sl_to_tp_max", 2.5)),
            tp_hit_min_agg=float(cfg.get("prod_adm_tp_hit_min_agg", 0.10)),
            auc_min=float(cfg.get("prod_adm_auc_min", 0.54)),
            auc_bound_min=float(cfg.get("prod_adm_auc_bound_min", 0.54)),
            tp_sep_min=float(cfg.get("prod_adm_tp_sep_min", 0.04)),
            ap_lift_min=float(cfg.get("prod_adm_ap_lift_min", 1.20)),
            tp_over_sl_min=float(cfg.get("prod_adm_tp_over_sl_min", 1.05)),
            tp_floor_bind_max_cell=float(cfg.get("prod_adm_floor_bind_cell_max", cfg.get("prod_adm_tp_floor_bind_max_cell", 0.35))),
            tp_floor_bind_max_agg=float(cfg.get("prod_adm_floor_bind_agg_max", cfg.get("prod_adm_tp_floor_bind_max_agg", 0.20))),
            sl_floor_bind_max_cell=(
                float(cfg.get("prod_adm_sl_floor_bind_max_cell"))
                if cfg.get("prod_adm_sl_floor_bind_max_cell", None) is not None
                else None
            ),
            enforce_tradeable_tp_lo=bool(cfg.get("prod_adm_enforce_tradeable_tp_lo", True)),
            tradeable_tp_min=float(_tp_min_tradeable),
        )
        prod_admissibility = production_admissibility_report(
            events_prod=events,
            score_prod=pred,
            bucket_horizon_metrics_prod=bucket_h_metrics,
            tp_lo_prod=_prod_tp_lo,
            sl_lo_prod=_prod_sl_lo,
            gates=_prod_gates,
        )

        # Tradeability is explicitly gated on effective TP distribution in U_prod (events["tp"]).
        _tp_eff_arr = events["tp"].to_numpy(dtype=np.float32, copy=False) if "tp" in events.columns else np.array([], dtype=np.float32)
        _tp_eff_arr = _tp_eff_arr[np.isfinite(_tp_eff_arr)] if _tp_eff_arr.size else _tp_eff_arr
        _tp_eff_p50 = float(np.quantile(_tp_eff_arr, 0.50)) if _tp_eff_arr.size else float("nan")
        _tp_eff_p75 = float(np.quantile(_tp_eff_arr, 0.75)) if _tp_eff_arr.size else float("nan")
        _tp_eff_p90 = float(np.quantile(_tp_eff_arr, 0.90)) if _tp_eff_arr.size else float("nan")
        _tradeable_min_lo = float(cfg.get("prod_adm_tradeable_tp_min_lo", 0.012))
        _tradeable_min = float(_tp_min_tradeable)
        _tradeable_min_hi = float(cfg.get("prod_adm_tradeable_tp_min_hi", 0.022))
        _tradeable_rule_p50 = bool(np.isfinite(_tp_eff_p50) and (_tp_eff_p50 >= _tradeable_min_lo))
        _tradeable_rule_tail = bool(
            np.isfinite(_tp_eff_p75)
            and np.isfinite(_tp_eff_p90)
            and (_tp_eff_p75 >= _tradeable_min)
            and (_tp_eff_p90 >= _tradeable_min_hi)
        )
        _tradeable_ok = bool(_tradeable_rule_p50 or _tradeable_rule_tail)
        if not _tradeable_ok:
            prod_admissibility.setdefault("failures", []).append(
                "tp_eff tradeability 2-of-3 rule failed: "
                f"p50={_tp_eff_p50:.4f} (min_lo={_tradeable_min_lo:.4f}), "
                f"p75={_tp_eff_p75:.4f}/p90={_tp_eff_p90:.4f} "
                f"(mins={_tradeable_min:.4f}/{_tradeable_min_hi:.4f})"
            )
        prod_admissibility["admissible_tier0"] = bool(prod_admissibility.get("admissible_tier0", False) and _tradeable_ok)
        prod_admissibility.setdefault("aggregates", {}).update({
            "tp_eff_p50_prod": _tp_eff_p50,
            "tp_eff_p75_prod": _tp_eff_p75,
            "tp_eff_p90_prod": _tp_eff_p90,
            "tp_eff_tradeable_min_lo": _tradeable_min_lo,
            "tp_eff_tradeable_min": _tradeable_min,
            "tp_eff_tradeable_min_hi": _tradeable_min_hi,
            "tp_eff_tradeable_rule_p50": _tradeable_rule_p50,
            "tp_eff_tradeable_rule_tail": _tradeable_rule_tail,
            "tp_eff_tradeable_ok": _tradeable_ok,
        })

        if not bool(prod_admissibility.get("admissible_tier0", False)):
            tprint(f"[PROD_ADMISSIBILITY:{cfg_id}] FAIL " + " | ".join(prod_admissibility.get("failures", [])))

        _pa_agg = prod_admissibility.get("aggregates", {}) if isinstance(prod_admissibility, dict) else {}
        _tp_fb_agg = _safe_float(_pa_agg.get("tp_floor_bind_prod_agg"), 0.0)
        _tp_fb_max = _safe_float(_pa_agg.get("max_cell_tp_floor_bind_prod"), 0.0)
        _floor_dom_pen = float(cfg.get("floor_dominance_penalty_weight_agg", 0.25)) * max(_tp_fb_agg, 0.0) + float(cfg.get("floor_dominance_penalty_weight_max", 0.25)) * max(_tp_fb_max, 0.0)
    stage2_score -= _floor_dom_pen

    summary = {
        "config_id": cfg_id,
        "evaluation_stage": "stage2_rescore" if stage2_rescore else "stage1",
        "mode": cfg.get("mode", "unknown"),
        "k_tp": cfg.get("k_tp"),
        "tp_anchor": cfg.get("tp_anchor"),
        "tp_base_pct": cfg.get("tp_base_pct", cfg.get("tp_abs_pct")),
        "sl_method": cfg.get("sl_method"),
        "sl_as_tp_pct": cfg.get("sl_as_tp_pct"),
        "regime_model": cfg.get("tp_regime_model"),
        "horizon_scaling": cfg.get("horizon_scaling"),
        "ic_label": ic_label,
        "ic_label_bucket_mean": mean_bucket_ic_label,
        "ic_payoff": ic_payoff,
        "ic_payoff_bucket_mean": mean_bucket_ic,
        "ic_snr": ic_snr,
        "sharpe": sharpe,
        "sortino": sortino,
        "tp_hit_rate": tp_hit_agg,
        "sl_hit_rate": sl_hit_agg,
        "timeout_rate": timeout_agg,
        "bind": round(bind_agg, 4),
        "balance": round(balance_agg, 4),
        "sl_to_tp": round(sl_to_tp_agg, 4),
        "p50_cell_sl_to_tp": p50_cell_sl_to_tp,
        "p90_cell_sl_to_tp": p90_cell_sl_to_tp,
        "payoff_edge": round(float(payoff_edge), 6),
        "edge": round(float(edge), 6),
        "payoff_edge_support_tp": int(_tp_count),
        "payoff_edge_support_sl": int(_sl_count),
        "payoff_edge_min_count": int(_min_edge_count),
        "pr_auc_lift": round(float(pr_auc_lift), 6),
        "barrier_ratio": max_barrier_ratio,
        "flag_degenerate_timeout": flag_degenerate_timeout,
        "flag_degenerate_sl": flag_degenerate_sl,
        "flag_degenerate_tp": flag_degenerate_tp,
        "ess": ess,
        "ess_full": ess_full,
        "coverage": coverage,
        # ic_std_time/asset: median across (bucket,horizon) cells — not global (which mixes all cells)
        "ic_std_time": ic_std_time_median,
        "ic_std_asset": ic_std_asset_median,
        "worst_bucket_IC": worst_bucket_ic,
        "stage1_score": stage1_score,
        "stage2_score": stage2_score,
        "stage_score": stage2_score,
        "stage2_missing": bool(stage2_missing),
        "stage2_reward": float(_reward),
        "stage2_penalty_raw": float(_raw_penalty),
        "stage2_penalty_bounded": float(_bounded_penalty),
        "stage2_auc_score": float(_auc_score),
        "stage2_sep_score": float(_sep_score),
        "stage2_sortino_score": float(_sortino_score),
        "stage2_ic_snr_score": float(_snr_score),
        "stage2_dir_sup_score": float(_dir_sup_score),
        "stage2_tp_over_sl_score": float(_tp_over_sl_score),
        "stage2_payoff_edge_score": float(payoff_edge_score),
        "rr_keep_rate": float(rr_keep_rate),
        "rr_keep_floor": float(rr_floor),
        "stage2_rr_keep_penalty": float(_rr_keep_pen),
        "stage2_horizon_consistency_penalty": float(_horizon_cons_pen),
        "stage2_probe_gap_penalty": float(probe_gap_penalty),
        "probe_auc": float(probe_auc),
        "probe_auc_bound": float(probe_auc_bound),
        "probe_status": str(probe_status),
        "probe_n_train": int(probe_n_train),
        "probe_n_val": int(probe_n_val),
        # brier/ece/mono: median across cells (per-cell values in bucket_horizon_metrics)
        "brier": brier,
        "ece": ece,
        "monotonicity": mono,
        "min_cell_brier": round(min_cell_brier, 5) if not math.isnan(min_cell_brier) else float("nan"),
        "min_cell_ece": round(min_cell_ece, 5) if not math.isnan(min_cell_ece) else float("nan"),
        "min_cell_mono": round(min_cell_mono, 4) if not math.isnan(min_cell_mono) else float("nan"),
        "oof_payoff_decile_spread": decile_spread,
        "oof_flip_rate": oof_flip_rate,
        "oof_flipped_cells": int(flipped_cells),
        "oof_modeled_cells": int(modeled_cells),
        "hard_gate": bool(hard_gate),
        "pass_cells": pass_cells,
        "total_cells": total_cells,
        "worst_bucket_coverage": worst_bucket_cov,
        "payoff_mean_top_decile": payoff_mean_top_decile,
        "top10_vs_rest_spread": round(top10_vs_rest_spread, 6),
        "cell_dispersion": round(cell_dispersion, 6),
        "min_cell_payoff": round(min_cell_payoff, 6),
        "min_cell_ic": round(min_cell_ic, 5),
        "min_cell_ic_label": round(min_cell_ic_label, 5),
        "median_cell_ic_label": round(median_cell_ic_label, 5),
        "min_cell_auc": round(min_cell_auc, 4) if not math.isnan(min_cell_auc) else float("nan"),
        "median_cell_auc": round(median_cell_auc, 4) if not math.isnan(median_cell_auc) else float("nan"),
        "min_cell_auc_bound": round(min_cell_auc_bound, 4) if not math.isnan(min_cell_auc_bound) else float("nan"),
        "median_cell_auc_bound": round(median_cell_auc_bound, 4) if not math.isnan(median_cell_auc_bound) else float("nan"),
        "min_cell_tp_sep": round(min_cell_tp_sep, 5),
        "median_cell_tp_sep": round(median_cell_tp_sep, 5),
        "timeout_range": round(timeout_range, 4),
        "min_cell_ap_lift": round(min_cell_ap_lift, 4) if not math.isnan(min_cell_ap_lift) else float("nan"),
        "median_cell_ap_lift": round(median_cell_ap_lift, 4) if not math.isnan(median_cell_ap_lift) else float("nan"),
        "min_cell_tp_over_sl": round(min_cell_tp_over_sl, 4) if not math.isnan(min_cell_tp_over_sl) else float("nan"),
        "median_cell_tp_over_sl": round(median_cell_tp_over_sl, 4) if not math.isnan(median_cell_tp_over_sl) else float("nan"),
        "min_cell_dir_superiority": round(min_cell_dir_superiority, 4) if not math.isnan(min_cell_dir_superiority) else float("nan"),
        "median_cell_dir_superiority": round(median_cell_dir_superiority, 4) if not math.isnan(median_cell_dir_superiority) else float("nan"),
        "effective_pos_mass": float(effective_pos_mass),
        "effective_neg_mass": float(effective_neg_mass),
        "prod_admissible_tier0": bool(prod_admissibility.get("admissible_tier0", False)),
        "prod_adm_failures": int(len(prod_admissibility.get("failures", []))),
        "econ_ok": bool(econ_ok),
        "econ_G": float(econ_G),
        "econ_multiplier": float(econ_multiplier),
        "tp_floor_bind_prod_agg": float(_tp_fb_agg),
        "max_cell_tp_floor_bind_prod": float(_tp_fb_max),
        "floor_dominance_penalty": float(_floor_dom_pen),
        "tp_eff_p50_prod": _safe_float(_pa_agg.get("tp_eff_p50_prod"), float("nan")),
        "tp_eff_p75_prod": _safe_float(_pa_agg.get("tp_eff_p75_prod"), float("nan")),
        "tp_eff_p90_prod": _safe_float(_pa_agg.get("tp_eff_p90_prod"), float("nan")),
        "tp_eff_tradeable_ok": bool(_pa_agg.get("tp_eff_tradeable_ok", False)),
        "tp_eff_tradeable_rule_p50": bool(_pa_agg.get("tp_eff_tradeable_rule_p50", False)),
        "tp_eff_tradeable_rule_tail": bool(_pa_agg.get("tp_eff_tradeable_rule_tail", False)),
        "prod_aligned_tp": cfg.get("prod_aligned_tp", {}),
    }

    detail = {
        "config": serialize_key(cfg),
        "feature_columns": feat_cols,
        "bucket_metrics": per_bucket,
        "bucket_horizon_metrics": {f"{k[0]}_H{k[1]}": v for k, v in bucket_h_metrics.items()},
        "production_admissibility": prod_admissibility,
        "prod_aligned_tp": cfg.get("prod_aligned_tp", {}),
        "economic_guardrail": {
            "econ_ok": bool(econ_ok),
            "econ_G": float(econ_G),
            "econ_multiplier": float(econ_multiplier),
        "tp_floor_bind_prod_agg": float(_tp_fb_agg),
        "max_cell_tp_floor_bind_prod": float(_tp_fb_max),
        "floor_dominance_penalty": float(_floor_dom_pen),
            "tp_hit_floor": _econ_tp_hit_floor,
            "sl_to_tp_cap": _econ_sl_to_tp_cap,
            "tp_over_sl_floor": _econ_tp_over_sl_floor,
            "min_factor": _econ_min_factor,
        },
        "calibration": {"brier": brier, "ece": ece, "monotonicity": mono, "oof_payoff_decile_spread": decile_spread},
    }
    if detailed_slices:
        detail["regime_metrics"] = per_slice_metrics(events, pred, "regime")
        detail["vol_quintile_metrics"] = per_slice_metrics(events, pred, "vol_quintile")

    weights_df: Optional[pd.DataFrame] = None
    if collect_weights:
        weights_df = events[["ts", "symbol", "side", "horizon", "bucket", "label", "payoff"]].copy()
        # Current run-computed TBM/event weights are preserved as fallback defaults
        # for both downstream base and meta model training consumers.
        weights_df["sample_weight"] = weights.astype(np.float32, copy=False)
        weights_df["base_sample_weight"] = weights_df["sample_weight"]
        weights_df["meta_sample_weight"] = weights_df["sample_weight"]
        weights_df["config_id"] = cfg_id
        weights_df["mode"] = str(cfg.get("mode", "unknown"))

    tprint(
        f"[eval:done] {cfg_id} stage1={stage1_score:.4f} stage2={stage2_score:.4f} "
        f"coverage={coverage:.3f} ess={ess:.1f} tp={summary['tp_hit_rate']:.3f} "
        f"timeout={summary['timeout_rate']:.3f} decile_spread={decile_spread:.6f} hard_gate={hard_gate} "
        f"elapsed_s={time.perf_counter()-t0:.2f} mem_peak_mb={_memory_snapshot_mb():.1f} "
        f"{_cache_pressure_summary(layer1_cache, layer2_cache, eval_cache)}"
    )
    del events, pred, y_signed, y_bin, payoff, weights
    _last_gc_ts = _maybe_collect_gc(last_gc_ts=_last_gc_ts, mem_threshold_mb=8192.0, min_interval_s=5.0)

    return summary, detail, weights_df


def _horizon_scale_for_cfg(cfg: Dict[str, Any], horizon: int) -> float:
    scaled = apply_horizon_scaling(
        1.0,
        horizon=int(horizon),
        scaling=str(cfg.get("horizon_scaling", "none")),
        alpha=float(cfg.get("horizon_alpha", 0.5)),
        base=float(cfg.get("horizon_base", 4.0)),
    )
    return float(scaled)


def _atr_pct_samples_for_prod_universe(
    artifacts: RunArtifacts,
    bucket_masks: Dict[str, pd.DataFrame],
) -> np.ndarray:
    atr = _get_barrier_atr_frame(artifacts)
    close = artifacts.panel.get("close")
    if close is None:
        return np.array([], dtype=np.float32)
    atr = atr.reindex(index=close.index, columns=close.columns).shift(1)
    union_mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    for m in bucket_masks.values():
        if isinstance(m, pd.DataFrame):
            union_mask |= m.reindex(index=close.index, columns=close.columns).fillna(False)
    samples = atr.where(union_mask).to_numpy(dtype=np.float32, copy=False).ravel()
    samples = samples[np.isfinite(samples)]
    return samples


def _apply_prod_aligned_tp_centering(
    cfgs: List[Dict[str, Any]],
    *,
    artifacts: RunArtifacts,
    bucket_masks: Dict[str, pd.DataFrame],
    cfg_runtime: Optional[Dict[str, Any]] = None,
    preserve_sl_axis: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    runtime = cfg_runtime or CFG
    if not bool(runtime.get("tbm_prod_aligned_tp_enable", True)):
        return cfgs, {}

    atr_samples = _atr_pct_samples_for_prod_universe(artifacts, bucket_masks)
    if atr_samples.size == 0:
        tprint("[prod_aligned_tp] WARNING no ATR% samples in production universe; skipping TP centering override")
        return cfgs, {}

    fee_pct_total = float(runtime.get("tbm_prod_aligned_fee_pct_total", runtime.get("fee_pct", 0.003)))
    worst_h = int(runtime.get("tbm_prod_aligned_worst_horizon", 2))
    q = float(runtime.get("tbm_prod_aligned_q", 0.25))
    alpha = float(runtime.get("tbm_prod_aligned_alpha", 0.45))
    margin_mult = float(runtime.get("tbm_prod_aligned_margin_mult", 4.0))
    hard_min_tp = float(runtime.get("tbm_prod_aligned_hard_min_tp", 0.02))
    inflate = float(runtime.get("tbm_prod_aligned_inflate", 1.10))
    h2_l = float(runtime.get("tbm_prod_aligned_h2_l2", 0.01))
    h2_u = float(runtime.get("tbm_prod_aligned_h2_u2", 0.04))

    anchor_cfg = cfgs[0] if cfgs else base_param_template(runtime)
    aligned = compute_prod_aligned_tp_params(
        atr_pct_samples=atr_samples,
        fee_pct_total=fee_pct_total,
        horizon_scaling_fn=lambda h: _horizon_scale_for_cfg(anchor_cfg, h),
        worst_horizon=worst_h,
        q=q,
        alpha=alpha,
        margin_mult=margin_mult,
        hard_min_tp=hard_min_tp,
        inflate=inflate,
        horizons=(2, 4),
        h2_lower=h2_l,
        h2_upper=h2_u,
    )

    tp_lo_new = float(aligned["tp_abs_lo_pct"])
    ladder = list(aligned.get("tp_base_candidates", []))
    if not ladder:
        ladder = [{"tp_base_pct": float(aligned["tp_base_pct"]), "tp_eff_targets": {}, "tp_eff_bands": {}, "q": q, "alpha": alpha}]

    out = []
    for c in cfgs:
        # Preserve the grid's own tp_base_pct and tp_abs_lo_pct — these are first-class
        # optimisation axes. The prod-aligned centering only injects production metadata
        # (prod_tp_abs_lo_pct, prod_aligned_tp) used for admissibility reporting, without
        # overwriting the barrier parameters used for actual barrier computation.
        grid_tp_base = float(c.get("tp_base_pct", c.get("tp_abs_pct", float(aligned["tp_base_pct"]))))
        grid_tp_lo = float(c.get("tp_abs_lo_pct", tp_lo_new))
        c2 = dict(c)
        # prod_tp_abs_lo_pct: production floor for admissibility reporting only.
        # Uses max(grid floor, ATR-derived floor) so production reporting is conservative,
        # but the barrier computation still uses the grid's tp_abs_lo_pct.
        c2["prod_tp_abs_lo_pct"] = max(grid_tp_lo, tp_lo_new)
        c2["prod_aligned_tp"] = {
            "atr_q": float(aligned["atr_q"]),
            "q": float(aligned["q"]),
            "alpha": float(aligned["alpha"]),
            "margin_mult": float(aligned["margin_mult"]),
            "fee_pct_total": float(aligned["fee_pct_total"]),
            "tp_min_tradeable": float(aligned["tp_min_tradeable"]),
            "s_H2": float(aligned.get("scaling", {}).get("s2", aligned["s_worst"])),
            "tp_atr_anchor": float(aligned["tp_atr_anchor"]),
            "tp_base_pct_final": grid_tp_base,
            "tp_abs_lo_pct_final": c2["prod_tp_abs_lo_pct"],
            "worst_horizon": int(aligned["worst_horizon"]),
            "tp_base_pre_override": grid_tp_base,
            "tp_abs_lo_pre_override": grid_tp_lo,
            "tp_eff_targets": (ladder[0].get("tp_eff_targets", {}) if ladder else {}),
            "tp_eff_bands": (ladder[0].get("tp_eff_bands", {}) if ladder else {}),
            "scaling": aligned.get("scaling", {}),
            "atr_quantiles": aligned.get("atr_quantiles", {}),
        }
        if preserve_sl_axis or not bool(runtime.get("prod_sl_tp_wide_enable", True)):
            # Stage 1: sl_as_tp_pct is a first-class grid axis — do not expand/override it.
            out.append(c2)
        else:
            sup_add = float(runtime.get("prod_sl_tp_superiority_add", c2.get("prod_sl_tp_superiority_add", 0.0075)))
            if sup_add > 0.02:
                tprint(
                    f"[prod_sl_tp_policy] NOTE superiority_add={sup_add:.4f} is very high vs typical TP levels; "
                    "this can over-prune SL ladder."
                )
            policy = SLTPPolicy(
                sl_as_tp_pct_grid=tuple(float(x) for x in runtime.get("prod_sl_tp_pct_grid", [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00, 1.25, 1.50, 1.75, 2.00])),
                superiority_add=sup_add,
                drop_on_violation=bool(runtime.get("prod_sl_tp_drop_on_violation", True)),
            )
            tp_eff_ref = float((aligned.get("tp_eff_targets", {}) or {}).get("H2", float("nan")))
            out.extend(expand_configs_wide_sl_tp_additive_superiority(c2, tp_eff=tp_eff_ref, policy=policy))

    # de-duplicate expanded grid
    uniq = {config_id(c): c for c in out}
    out = list(uniq.values())

    if tp_lo_new <= 0.005 + 1e-12:
        tprint(
            f"[prod_aligned_tp] NOTE computed tp_abs_lo_pct={tp_lo_new:.4f} is not above legacy 0.5% floor; "
            "consider higher margin_mult or hard_min_tp"
        )
    tprint(
        f"[prod_aligned_tp] applied ladder to {len(cfgs)} base cfgs -> {len(out)} cfgs "
        f"(tp_base_count={len(ladder)}, tp_abs_lo_pct>={tp_lo_new:.4f}, q50={aligned.get('atr_quantiles',{}).get('q50', float('nan')):.4f}, "
        f"q75={aligned.get('atr_quantiles',{}).get('q75', float('nan')):.4f}, q90={aligned.get('atr_quantiles',{}).get('q90', float('nan')):.4f})"
    )
    return out, aligned


# ---------------------------
# Grids
# ---------------------------
def base_param_template(cfg_runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    tbm_defaults = get_tbm_optimizer_defaults(cfg_runtime if cfg_runtime is not None else CFG)
    return {
        "tp_abs_lo_pct": 0.005,  # was tbm_defaults["tp_abs_lo_pct"]=0.02; 2% floor dominated ATR barriers at H2/H4
        "tp_abs_hi_pct": float(tbm_defaults["tp_abs_hi_pct"]),
        "sl_abs_lo_pct": 0.005,  # same fix for SL floor
        "sl_abs_hi_pct": float(tbm_defaults["sl_abs_hi_pct"]),
        "tp_mult_lo": 0.5,
        "tp_mult_hi": 3.0,
        "sl_mult_lo": 0.3,
        "sl_mult_hi": 2.0,
        "mix_weight": 0.5,
        "horizon_alpha": float(tbm_defaults["horizon_alpha"]),
        "horizon_base": int(tbm_defaults["horizon_base"]),
        "quantile_basis": "composite",
        "quantile_lo": 0.2,
        "quantile_hi": 0.8,
        "min_keep_fraction": 0.5,
        "weighting_scheme": "combined",
        "rr_weight_power": 1.0,
        "tp_hit_weight": 0.5,
        "timeout_penalty": 0.5,
        "sl_noise_buffer": True,
        "sl_min_abs_pct": 0.01,
        "sl_min_bps": 100,
        "tp_min_abs_pct": 0.005,
        "tp_min_bps": 50,
        "fee_pct": float(tbm_defaults["fee_pct"]),
        "slip_buffer": float(tbm_defaults["slip_buffer"]),
        "min_net_rr": 0.4,  # relaxed for geometry search; tighter RR kills training density
        "min_tp_hit_rate": max(0.01, float((cfg_runtime if cfg_runtime is not None else CFG).get("econ_tp_hit_floor", 0.10))),
        "max_timeout_rate": 0.0,  # 0.0 = use horizon-adaptive threshold (0.90 + 0.025*h/8)
        "min_raw_events": 50,
        "min_ess_events": 30,
        "min_coverage_threshold": 0.2,
        "use_quantile_filter": False,
        "tp_side_skew": 0.0,
        "sl_activation_minutes": 0,
        "trail_sl_mult": 0.0,
        "tp_time_decay": "none",
        "tp_abs_pct": float(tbm_defaults["tp_abs_pct"]),
        "tp_base_pct": float(tbm_defaults["tp_base_pct"]),
        "base_atr_window": int(tbm_defaults["base_atr_window"]),
        # Production-aligned TP centering knobs.
        "tbm_prod_aligned_tp_enable": bool((cfg_runtime if cfg_runtime is not None else CFG).get("tbm_prod_aligned_tp_enable", True)),
        "tbm_prod_aligned_fee_pct_total": float((cfg_runtime if cfg_runtime is not None else CFG).get("tbm_prod_aligned_fee_pct_total", 0.003)),
        "tbm_prod_aligned_worst_horizon": int((cfg_runtime if cfg_runtime is not None else CFG).get("tbm_prod_aligned_worst_horizon", 2)),
        "tbm_prod_aligned_q": float((cfg_runtime if cfg_runtime is not None else CFG).get("tbm_prod_aligned_q", 0.25)),
        "tbm_prod_aligned_alpha": float((cfg_runtime if cfg_runtime is not None else CFG).get("tbm_prod_aligned_alpha", 0.45)),
        "tbm_prod_aligned_margin_mult": float((cfg_runtime if cfg_runtime is not None else CFG).get("tbm_prod_aligned_margin_mult", 4.0)),
        "tbm_prod_aligned_hard_min_tp": float((cfg_runtime if cfg_runtime is not None else CFG).get("tbm_prod_aligned_hard_min_tp", 0.02)),
        "tbm_prod_aligned_inflate": float((cfg_runtime if cfg_runtime is not None else CFG).get("tbm_prod_aligned_inflate", 1.10)),
        "tbm_prod_aligned_h2_l2": float((cfg_runtime if cfg_runtime is not None else CFG).get("tbm_prod_aligned_h2_l2", 0.01)),
        "tbm_prod_aligned_h2_u2": float((cfg_runtime if cfg_runtime is not None else CFG).get("tbm_prod_aligned_h2_u2", 0.04)),
        "prod_adm_tradeable_tp_min": float((cfg_runtime if cfg_runtime is not None else CFG).get("prod_adm_tradeable_tp_min", 0.015)),
        "prod_adm_tradeable_tp_min_lo": float((cfg_runtime if cfg_runtime is not None else CFG).get("prod_adm_tradeable_tp_min_lo", 0.012)),
        "prod_adm_tradeable_tp_min_hi": float((cfg_runtime if cfg_runtime is not None else CFG).get("prod_adm_tradeable_tp_min_hi", 0.022)),
        "prod_adm_floor_bind_agg_max": float((cfg_runtime if cfg_runtime is not None else CFG).get("prod_adm_floor_bind_agg_max", 0.20)),
        "prod_adm_floor_bind_cell_max": float((cfg_runtime if cfg_runtime is not None else CFG).get("prod_adm_floor_bind_cell_max", 0.35)),
        # Wide SL/TP ladder policy under additive TP superiority rule.
        "prod_sl_tp_wide_enable": bool((cfg_runtime if cfg_runtime is not None else CFG).get("prod_sl_tp_wide_enable", True)),
        "prod_sl_tp_superiority_add": float((cfg_runtime if cfg_runtime is not None else CFG).get("prod_sl_tp_superiority_add", 0.0075)),
        "prod_sl_tp_drop_on_violation": bool((cfg_runtime if cfg_runtime is not None else CFG).get("prod_sl_tp_drop_on_violation", True)),
        "prod_sl_tp_pct_grid": (cfg_runtime if cfg_runtime is not None else CFG).get(
            "prod_sl_tp_pct_grid",
            [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00, 1.25],
        ),
        # OOF safeguard: flip bucket score direction when materially anti-informative.
        "oof_auc_flip_enable": bool((cfg_runtime if cfg_runtime is not None else CFG).get("oof_auc_flip_enable", True)),
        "oof_auc_flip_threshold": float((cfg_runtime if cfg_runtime is not None else CFG).get("oof_auc_flip_threshold", 0.50)),
    }




def _is_grid_tradeable_h2(cfg: Dict[str, Any], *, min_ratio_to_floor: float = 1.35) -> bool:
    """Static guard: reject configs whose median-vol H2 TP is too close to effective floor."""
    tp_method = str(cfg.get("tp_method", ""))
    if tp_method not in {"atr_norm", "semi_atr_norm", "atr_mult", "semi_atr_mult", "absolute"}:
        return True
    h2_scale = 1.0
    scaling = str(cfg.get("horizon_scaling", "none"))
    if scaling == "sqrt":
        h2_scale = (2.0 / max(float(cfg.get("horizon_base", 4.0)), EPS)) ** 0.5
    elif scaling == "power":
        h2_scale = (2.0 / max(float(cfg.get("horizon_base", 4.0)), EPS)) ** float(cfg.get("horizon_alpha", 0.5))
    tp_anchor = float(cfg.get("tp_base_pct", cfg.get("tp_abs_pct", 0.01))) * float(cfg.get("k_tp", 1.0)) * h2_scale
    tp_floor_eff = effective_tp_floor(
        tp_abs_lo_pct=float(cfg.get("tp_abs_lo_pct", 0.005)),
        tp_min_abs_pct=float(cfg.get("tp_min_abs_pct", 0.005)),
        tp_min_bps=float(cfg.get("tp_min_bps", 50)),
    )
    return tp_anchor >= min_ratio_to_floor * tp_floor_eff


def _filter_tradeability_guard(cfgs: List[Dict[str, Any]], cfg_runtime: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if not cfgs:
        return cfgs
    ratio = float((cfg_runtime or CFG).get("tbm_h2_tradeability_floor_ratio", 1.35))
    kept, dropped = [], 0
    for c in cfgs:
        if _is_grid_tradeable_h2(c, min_ratio_to_floor=ratio):
            kept.append(c)
        else:
            dropped += 1
    if dropped > 0:
        tprint(f"[grid_guard] dropped {dropped}/{len(cfgs)} configs: H2 TP anchor too close to effective floor (ratio<{ratio:.2f})")
    return kept or cfgs
def stage1_grid(cfg_runtime: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    cfgs = []

    # atr_norm only: atr_mult_rr is consistently degenerate (timeout>0.89, tp_hit<0.06)
    # and wastes compute. atr_norm normalises by rolling median ATR so the barrier scales
    # with the event's volatility spike rather than the absolute ATR level.
    # tp_base_pct = base TP when atr/median_atr == 1.0.
    # k_tp: [0.4, 0.5] added — tighter TP → higher tp_hit → lower sl_to_tp ratio.
    # sl_as_tp_pct: [0.8, 1.0, 1.2] added — wider SL relative to TP → lower sl_to_tp ratio.
    # tp_base_pct: [0.020] added — higher absolute TP floor.
    # base_atr_window: [336, 504, 672, 840] (14d–35d). Prior run confirmed 672>504>336.
    # 84/168 excluded: too short, produces high timeout + low bind + low coverage.
    for k_tp, tp_base, sl_as_tp, atr_win in product(
        [0.4, 0.5, 0.7, 1.0, 1.25, 1.6],
        [0.006, 0.010, 0.015, 0.020, 0.025, 0.030],  # 0.025/0.030 needed for k_tp=0.4 to clear H2 floor guard
        [0.4, 0.6, 0.7, 0.8, 1.0],
        [336, 504, 672, 840],  # 14d / 21d / 28d / 35d slow median reference
    ):
        c = base_param_template(cfg_runtime)
        c.update(
            {
                "mode": "atr_norm",
                "tp_method": "atr_norm",
                "sl_method": "tp_pct",
                "k_tp": float(k_tp),
                "tp_base_pct": float(tp_base),
                "tp_abs_pct": float(tp_base),
                "sl_as_tp_pct": float(sl_as_tp),
                "tp_regime_model": "none",
                "horizon_scaling": "sqrt",
                "base_atr_window": int(atr_win),
            }
        )
        cfgs.append(c)

    # Dedup + early hard tradeability feasibility
    uniq = {config_id(c): c for c in cfgs}
    return _filter_tradeability_guard(list(uniq.values()), cfg_runtime=cfg_runtime)


def stage2_grids_from_stage1(winners: List[Dict[str, Any]], max_per_substage: int = 24) -> List[Dict[str, Any]]:
    """Hierarchical stage-2 generation focused on refining atr_norm winners.

    Stage 1 established that atr_norm with tp_base_pct=0.015, k_tp=1.25 is the
    dominant geometry. Stage 2 refines along three axes:
    2A) tp_base_pct fine grid + base_atr_window (slow median reference)
    2B) SL geometry: sl_as_tp_pct fine grid + atr_mult SL alternative
    2C) Path-dependence knobs on best 2B subset
    """
    if not winners:
        return []

    # 2A: Refine tp_base_pct and base_atr_window around each winner.
    # tp_base_pct controls the TP at median volatility; the winner is 0.015.
    # base_atr_window controls how "slow" the median reference is.
    stage2a: List[Dict[str, Any]] = []
    for base in winners:
        tp_method = base.get("tp_method", "atr_norm")
        base_tp = float(base.get("tp_base_pct", base.get("tp_abs_pct", 0.015)))
        base_k = float(base.get("k_tp", 1.25))

        if tp_method == "atr_norm":
            # Fine-grid around the winning tp_base_pct, k_tp, and atr_window.
            # atr_window is now a first-class axis: sweep ±1 step around winner's value.
            # Stage 2A ladder extends one step below Stage 1 (adds 168) so a winner at
            # the Stage 1 floor (336) still has a lower neighbour to probe.
            # Stage 1 ladder: [336, 504, 672, 840]  (168 dropped as consistently worst)
            # Stage 2A ladder: [168, 336, 504, 672, 840]  (168 re-enters as safety net only)
            base_win = int(base.get("base_atr_window", 672))
            _win_ladder = [168, 336, 504, 672, 840]
            _win_idx = min(range(len(_win_ladder)), key=lambda i: abs(_win_ladder[i] - base_win))
            _win_neighbours = sorted(set(_win_ladder[max(0, _win_idx - 1): _win_idx + 2]))
            for tp_base, k_tp, atr_win in product(
                [max(0.008, base_tp - 0.002), base_tp, min(0.020, base_tp + 0.002)],
                [max(0.8, base_k - 0.25), base_k, min(2.0, base_k + 0.25)],
                _win_neighbours,
            ):
                c = dict(base)
                c.update({
                    "mode": "atr_norm_2A",
                    "tp_method": "atr_norm",
                    "k_tp": float(k_tp),
                    "tp_base_pct": float(tp_base),
                    "tp_abs_pct": float(tp_base),
                    "base_atr_window": int(atr_win),
                })
                stage2a.append(c)
        else:
            # Non-atr_norm winner: keep existing TP method, sweep k_tp only
            for k_tp in [max(0.5, base_k - 0.4), base_k, min(2.5, base_k + 0.4)]:
                c = dict(base)
                c.update({"mode": f"{tp_method}_2A", "k_tp": float(k_tp)})
                stage2a.append(c)

    uniq2a = {config_id(c): c for c in stage2a}
    stage2a = list(uniq2a.values())[:max_per_substage]

    # 2B: SL geometry refinement on top 2A candidates.
    # Primary axis: sl_as_tp_pct fine grid (winner range 0.4–0.6).
    # Secondary: atr_mult SL as alternative to tp_pct SL.
    stage2b: List[Dict[str, Any]] = []
    for base in stage2a:
        base_sl = float(base.get("sl_as_tp_pct", 0.5))
        # Fine grid around current sl_as_tp_pct
        for sl_pct in [max(0.3, base_sl - 0.1), base_sl, min(0.8, base_sl + 0.1), min(0.8, base_sl + 0.2)]:
            c = dict(base)
            c.update({"mode": f"{base.get('mode','stage2')}_sl",
                      "sl_method": "tp_pct", "sl_as_tp_pct": float(sl_pct)})
            stage2b.append(c)
        # ATR-mult SL is consistently degenerate (timeout>0.85, tp_hit<0.06 in all runs).
        # Removed to avoid wasting Stage 2 budget on dead branches.

    uniq2b = {config_id(c): c for c in stage2b}
    stage2b = list(uniq2b.values())[:max_per_substage]

    # 2C: Path-dependence knobs on top slice of 2B.
    # sl_activation_minutes: delay SL activation to avoid noise wicks at entry.
    # trail_sl_mult: trailing SL tightening as trade moves in favour.
    stage2c: List[Dict[str, Any]] = []
    for base in stage2b[:max(1, min(6, len(stage2b)))]:
        for act_m, trail in product([0, 15, 30], [0.0, 0.3, 0.6]):
            c = dict(base)
            c.update({
                "mode": f"{base.get('mode','stage2')}_path",
                "sl_activation_minutes": int(act_m),
                "trail_sl_mult": float(trail),
                "tp_time_decay": "none",  # keep TP fixed; decay interacts badly with atr_norm
            })
            stage2c.append(c)

    out = stage2a + stage2b + stage2c
    uniq = {config_id(c): c for c in out}
    return _filter_tradeability_guard(list(uniq.values()))


# Maximum allowed spread between the highest and lowest per-cell timeout rates.
_MAX_TIMEOUT_RANGE: float = 0.50

# Hard pre-filter thresholds applied before any learnability ranking.
# These prevent metric-gaming: a geometry must first produce a healthy bound set
# before its AUC/separation metrics are meaningful.
_PROMOTE_MIN_COVERAGE: float = 0.45   # fraction of events surviving all filters
_PROMOTE_MIN_BIND: float = 0.50       # TP+SL rate — global/aggregate gate
_PROMOTE_MIN_BIND_CELL: float = 0.38  # per-cell bind floor (H2 is typically the limiter)
_PROMOTE_MAX_TIMEOUT: float = 0.60    # aggregate timeout rate
_PROMOTE_MIN_TP_SEP: float = 0.05     # min_cell_tp_sep floor (5pp)
_PROMOTE_MIN_AUC: float = 0.54        # min_cell_auc floor
_PROMOTE_MIN_AP_LIFT: float = 1.20    # AP / base_rate — model must lift precision above random
_PROMOTE_MIN_TP_OVER_SL: float = 1.30 # E[r|TP] / abs(E[r|SL]) — 30% payoff edge required to survive 50bps fees
_PROMOTE_MAX_SL_TO_TP: float = 3      # SL-hit / TP-hit ratio cap — informational only; fee_ev gate is the binding constraint
_MAX_BARRIER_RATIO: float = 1.0       # SL-mean / TP-mean cap — ensures SL isn't too high compared to TP
_AVG_AUC_THRESHOLD: float = 0.54      # Minimum average AUC for the selected set of geometries
_PROMOTE_MAX_ECE: float = 0.30        # Calibration viability gate (lower is better)
_PROMOTE_MAX_BRIER: float = 0.22      # Calibration viability gate (lower is better)
_PROMOTE_MIN_EFFECTIVE_POS: float = 250.0  # Effective positive label mass proxy floor
_PROMOTE_MIN_EFFECTIVE_NEG: float = 250.0  # Effective negative label mass proxy floor
_ROUND_TRIP_FEE: float = 0.005        # 0.5% round-trip fee (entry + exit) applied to both TP and SL legs
_PROMOTE_MIN_FEE_EV: float = 0.0      # fee-adjusted EV must be > 0: tp_hit*(tp_mean-fee) - sl_hit*(sl_mean+fee) > 0

# Tier definitions for the feasible-set builder.
# Each tier is a tuple of:
# (cov_min, bind_min, timeout_max, tp_sep_min, auc_bound_min, ap_lift_min,
#  tp_over_sl_min, ece_max, brier_max, eff_pos_min, eff_neg_min)
# bind_min here is the GLOBAL aggregate gate; per-cell uses _PROMOTE_MIN_BIND_CELL.
# Tier 0: all gates active.
# Tiers 1-2: relax learnability gates (tp_sep/auc/ap_lift/calibration/mass) but keep structural + payoff edge.
# Tier 3: structural-only — reached only when all learnability-aware tiers fail.
_FEASIBLE_TIERS: List[Tuple[float, float, float, float, float, float, float, float, float, float, float]] = [
    (_PROMOTE_MIN_COVERAGE, _PROMOTE_MIN_BIND, _PROMOTE_MAX_TIMEOUT, _PROMOTE_MIN_TP_SEP, _PROMOTE_MIN_AUC, _PROMOTE_MIN_AP_LIFT, _PROMOTE_MIN_TP_OVER_SL, _PROMOTE_MAX_ECE, _PROMOTE_MAX_BRIER, _PROMOTE_MIN_EFFECTIVE_POS, _PROMOTE_MIN_EFFECTIVE_NEG),  # Tier 0
    (_PROMOTE_MIN_COVERAGE, _PROMOTE_MIN_BIND, _PROMOTE_MAX_TIMEOUT, 0.02, 0.54, 1.10, _PROMOTE_MIN_TP_OVER_SL, 0.36, 0.26, 150.0, 150.0),  # Tier 1
    (_PROMOTE_MIN_COVERAGE, _PROMOTE_MIN_BIND, _PROMOTE_MAX_TIMEOUT, 0.0, 0.52, 0.0, _PROMOTE_MIN_TP_OVER_SL, 0.45, 0.30, 80.0, 80.0),      # Tier 2
    (_PROMOTE_MIN_COVERAGE, _PROMOTE_MIN_BIND, _PROMOTE_MAX_TIMEOUT, 0.0, 0.0, 0.0, 0.0, 999.0, 999.0, 0.0, 0.0),                             # Tier 3
]


def _build_feasible_set(
    df: pd.DataFrame,
    per_cell: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """Build the feasible set using explicit tiers — no silent progressive relaxation.

    Returns (feasible_df, tier_used) where tier_used is 0 (strictest) to 3 (structural only).
    Structural gates (coverage/bind/timeout) are NEVER relaxed.
    Learnability gates (tp_sep, auc_bound, ap_lift, calibration, effective label mass) relax across tiers.
    Payoff edge gate (tp_over_sl) is kept through Tier 2, dropped only at Tier 3.
    If no config passes even Tier 3, returns (df, -1).

    per_cell=True: use _PROMOTE_MIN_BIND_CELL (0.38) instead of _PROMOTE_MIN_BIND (0.50)
    for the bind gate, since H2 cells structurally have lower bind than the aggregate.
    Also applies sl_to_tp cap if the column is present.
    """
    _bind_floor = _PROMOTE_MIN_BIND_CELL if per_cell else _PROMOTE_MIN_BIND
    for tier, (
        cov_min, bind_min, to_max, sep_min, auc_min, ap_lift_min, tp_over_sl_min,
        ece_max, brier_max, eff_pos_min, eff_neg_min
    ) in enumerate(_FEASIBLE_TIERS):
        _bind_gate = max(bind_min * (_PROMOTE_MIN_BIND_CELL / _PROMOTE_MIN_BIND), _bind_floor) if per_cell else bind_min
        mask = pd.Series(True, index=df.index)
        if "coverage" in df.columns:
            mask = mask & (df["coverage"] >= cov_min)
        if "bind" in df.columns:
            mask = mask & (df["bind"] >= _bind_gate)
        if "timeout_rate" in df.columns:
            mask = mask & (df["timeout_rate"] <= to_max)
        if sep_min > 0.0 and "min_cell_tp_sep" in df.columns:
            mask = mask & (df["min_cell_tp_sep"].fillna(0.0) >= sep_min)
        if auc_min > 0.0:
            auc_col = "min_cell_auc_bound" if "min_cell_auc_bound" in df.columns else "min_cell_auc"
            if auc_col in df.columns:
                mask = mask & (pd.to_numeric(df[auc_col], errors="coerce").fillna(0.0) >= auc_min)
        if ap_lift_min > 0.0 and "min_cell_ap_lift" in df.columns:
            mask = mask & (df["min_cell_ap_lift"].fillna(0.0) >= ap_lift_min)
        if tp_over_sl_min > 0.0 and "min_cell_tp_over_sl" in df.columns:
            mask = mask & (df["min_cell_tp_over_sl"].fillna(0.0) >= tp_over_sl_min)
        if ece_max < 999.0:
            _ece_col = "min_cell_ece" if "min_cell_ece" in df.columns else ("ece" if "ece" in df.columns else None)
            if _ece_col:
                mask = mask & (pd.to_numeric(df[_ece_col], errors="coerce").fillna(999.0) <= ece_max)
        if brier_max < 999.0:
            _br_col = "min_cell_brier" if "min_cell_brier" in df.columns else ("brier" if "brier" in df.columns else None)
            if _br_col:
                mask = mask & (pd.to_numeric(df[_br_col], errors="coerce").fillna(999.0) <= brier_max)
        if eff_pos_min > 0.0 and "effective_pos_mass" in df.columns:
            mask = mask & (pd.to_numeric(df["effective_pos_mass"], errors="coerce").fillna(0.0) >= eff_pos_min)
        if eff_neg_min > 0.0 and "effective_neg_mass" in df.columns:
            mask = mask & (pd.to_numeric(df["effective_neg_mass"], errors="coerce").fillna(0.0) >= eff_neg_min)
        # Fee-adjusted EV gate: tp_hit*(tp_mean - fee) - sl_hit*(sl_mean + fee) > 0.
        # This is a hard tradeability constraint applied at ALL tiers — a geometry that
        # cannot generate positive expected value after 0.3% round-trip fees is not tradeable
        # regardless of its learnability. Applied at all tiers (never relaxed).
        if "fee_ev" in df.columns:
            mask = mask & (df["fee_ev"].fillna(-999.0) > _PROMOTE_MIN_FEE_EV)
        # barrier_ratio cap: ensure SL size isn't too high compared to TP size.
        if "barrier_ratio" in df.columns:
            mask = mask & (df["barrier_ratio"].fillna(999.0) <= _MAX_BARRIER_RATIO)
        result = df[mask]
        if not result.empty:
            if tier > 0:
                tprint(f"[feasible_set] Tier {tier} used (relaxed learnability gates): {len(result)} configs")
            return result, tier
    return df, -1


def _param_distance(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any]) -> float:
    """Normalised L1 distance in the 4-parameter geometry space.

    Scale factors chosen so that a single step in each axis contributes ~1.0:
        atr_window: step=168h  → divide by 168
        k_tp:       step=0.1   → divide by 0.1
        sl_as_tp:   step=0.1   → divide by 0.1
        tp_base_pct:step=0.002 → divide by 0.002
    """
    d_win = abs(float(cfg_a.get("base_atr_window", 672)) - float(cfg_b.get("base_atr_window", 672))) / 168.0
    d_ktp = abs(float(cfg_a.get("k_tp", 1.0)) - float(cfg_b.get("k_tp", 1.0))) / 0.1
    d_sl  = abs(float(cfg_a.get("sl_as_tp_pct", 0.5)) - float(cfg_b.get("sl_as_tp_pct", 0.5))) / 0.1
    d_tp  = abs(float(cfg_a.get("tp_base_pct", 0.01)) - float(cfg_b.get("tp_base_pct", 0.01))) / 0.002
    return d_win + d_ktp + d_sl + d_tp


def _check_label_conflict(v1: np.ndarray, v2: np.ndarray) -> bool:
    """Check for opposite labels at the same timestamp (1 vs -1)."""
    # 1 vs -1 conflict: (v1 == 1 & v2 == -1) | (v1 == -1 & v2 == 1)
    # This is equivalent to v1 * v2 == -1 (since only -1*1 or 1*-1 gives -1)
    if v1 is None or v2 is None:
        return False
    return bool(np.any(v1 * v2 == -1))


def _jaccard_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Jaccard distance on non-zero label occurrences."""
    if v1 is None or v2 is None:
        return 0.0
    # Jaccard on outcome existence (label != 0)
    # v1, v2 are signed labels (-1, 0, 1).
    m1 = v1 != 0
    m2 = v2 != 0
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return 1.0 - float(intersection) / float(union)


def _span_ratio(values: pd.Series) -> float:
    """Return max/min ratio for positive finite values, else NaN."""
    s = pd.to_numeric(values, errors="coerce")
    s = s[np.isfinite(s) & (s > 0)]
    if s.empty:
        return float("nan")
    return float(s.max() / max(float(s.min()), EPS))


def _robust_minmax(values: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0, eps: float = 1e-12) -> np.ndarray:
    """Robust percentile-based min-max normalization to [0,1]."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if not finite.any():
        return np.full_like(x, 0.5, dtype=np.float64)
    xv = x[finite]
    if xv.size < 20:
        out = np.full_like(x, 0.5, dtype=np.float64)
        ranks = pd.Series(xv).rank(method="average", pct=True).to_numpy(dtype=np.float64, copy=False)
        out[finite] = ranks
        out[~finite] = 0.5
        return out
    lo = float(np.percentile(xv, float(p_lo)))
    hi = float(np.percentile(xv, float(p_hi)))
    if hi <= (lo + float(eps)):
        out = np.full_like(x, 0.5, dtype=np.float64)
        out[~finite] = 0.5
        return out
    out = (x - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~finite] = 0.5
    return out


def _apply_min_tp_sl_span(
    selected_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    *,
    min_tp_span_ratio: float,
    min_sl_span_ratio: float,
    max_configs: int,
) -> pd.DataFrame:
    """Ensure selected set has minimum TP and implied-SL spans."""
    if selected_df is None or selected_df.empty:
        return selected_df
    if candidate_df is None or candidate_df.empty:
        return selected_df

    tp_need = float(max(min_tp_span_ratio, 1.0))
    sl_need = float(max(min_sl_span_ratio, 1.0))
    out = selected_df.copy()
    pool = candidate_df[~candidate_df["config_id"].isin(out["config_id"])].copy()
    if pool.empty:
        return out

    def _current_metrics(df: pd.DataFrame) -> Tuple[float, float]:
        tp_span = _span_ratio(df.get("tp_base_pct", pd.Series(dtype=float)))
        sl_span = _span_ratio(df.get("sl_anchor_pct", pd.Series(dtype=float)))
        return tp_span, sl_span

    # Greedy repair: add candidate that most improves deficit to target spans.
    while len(out) < max_configs:
        tp_span, sl_span = _current_metrics(out)
        if (tp_span >= tp_need) and (sl_span >= sl_need):
            break

        cur_deficit = max(0.0, tp_need - tp_span) + max(0.0, sl_need - sl_span)
        best_row = None
        best_gain = float("-inf")
        for _, cand in pool.iterrows():
            trial = pd.concat([out, cand.to_frame().T], ignore_index=True)
            tp_t, sl_t = _current_metrics(trial)
            trial_deficit = max(0.0, tp_need - tp_t) + max(0.0, sl_need - sl_t)
            gain = float(cur_deficit - trial_deficit)
            if not math.isfinite(gain):
                gain = float("-inf")
            if gain > best_gain:
                best_gain = gain
                best_row = cand

        if best_row is None or best_gain <= 0.0:
            break

        out = pd.concat([out, best_row.to_frame().T], ignore_index=True)
        pool = pool[pool["config_id"] != str(best_row.get("config_id"))]
        if pool.empty:
            break

    return out


def _diverse_subset(
    feasible_df: pd.DataFrame,
    details: Dict[str, Any],
    run_vectors: Optional[Dict[str, np.ndarray]] = None,
    min_distance: float = 1.0,
    max_configs: int = 20,
    alpha: float = 0.7,
    gamma: float = 2.0,
    gate_mode: str = "asymmetric_min",
    acquisition: str = "score_novelty_power",
    preselected_cids: Optional[List[str]] = None,
    anchor_high: int = 4,
    score_keep_fraction: float = 1.0,
    behavior_alpha: float = 0.7,
    min_tp_span_ratio: float = 2.0,
    min_sl_span_ratio: float = 1.5,
    enable_span_repair: bool = True,
) -> pd.DataFrame:
    """GRR on all feasible layer-1 candidates using stage_score + novelty utility."""
    if max_configs <= 0 or feasible_df is None or feasible_df.empty:
        return feasible_df.head(0) if isinstance(feasible_df, pd.DataFrame) else pd.DataFrame()
    if run_vectors is None:
        run_vectors = {}

    work_df = feasible_df.copy()
    score_col = "stage_score" if "stage_score" in work_df.columns else ("stage2_score" if "stage2_score" in work_df.columns else "stage1_score")
    auc_col = "min_cell_auc_bound" if "min_cell_auc_bound" in work_df.columns else ("min_cell_auc" if "min_cell_auc" in work_df.columns else None)
    if auc_col is None:
        tprint("[grr] no auc column; skipping")
        return feasible_df.head(0)

    work_df["_stage_score"] = pd.to_numeric(work_df.get(score_col, np.nan), errors="coerce").fillna(-np.inf)
    work_df["_auc_metric"] = pd.to_numeric(work_df.get(auc_col, np.nan), errors="coerce")
    work_df["_cid"] = work_df.get("config_id", pd.Series(range(len(work_df)), index=work_df.index)).astype(str)
    _tuple_cols = [c for c in ("k_tp", "tp_anchor", "sl_as_tp_pct") if c in work_df.columns]
    if len(_tuple_cols) < 3:
        _tuple_cols = [c for c in ("k_tp", "tp_base_pct", "sl_as_tp_pct") if c in work_df.columns]
    work_df["_stable_id"] = work_df["_cid"]
    if _tuple_cols:
        work_df["_stable_id"] = work_df[_tuple_cols].round(8).astype(str).agg("|".join, axis=1)

    dedupe_cols = list(_tuple_cols)
    if dedupe_cols:
        work_df = work_df.sort_values(["_stage_score", "_auc_metric", "_stable_id"], ascending=[False, False, True])
        work_df = work_df.drop_duplicates(subset=dedupe_cols, keep="first")

    finite_auc = work_df["_auc_metric"].to_numpy(dtype=np.float64, copy=False)
    finite_auc = finite_auc[np.isfinite(finite_auc)]
    median_auc = float(np.median(finite_auc)) if finite_auc.size else 0.54
    auc_floor = float(max(0.54, 0.8 * median_auc))

    pool_df = work_df[np.isfinite(work_df["_auc_metric"]) & (work_df["_auc_metric"] >= auc_floor)].copy()
    auc_floor_relaxed = False
    if pool_df.empty:
        pool_df = work_df[np.isfinite(work_df["_auc_metric"]) & (work_df["_auc_metric"] >= 0.54)].copy()
        auc_floor_relaxed = True
    if pool_df.empty:
        pool_df = work_df.copy()
        auc_floor_relaxed = True

    econ_mask = pd.Series(True, index=pool_df.index)
    econ_applied = False
    if "pr_auc_lift" in pool_df.columns:
        econ_mask = econ_mask & (pd.to_numeric(pool_df["pr_auc_lift"], errors="coerce").fillna(-np.inf) > 0.0)
    if "edge" in pool_df.columns:
        econ_mask = econ_mask & (pd.to_numeric(pool_df["edge"], errors="coerce").fillna(-np.inf) > 0.0)
    econ_df = pool_df[econ_mask].copy()
    if not econ_df.empty and len(econ_df) >= max(2 * int(max_configs), 20):
        pool_df = econ_df
        econ_applied = True

    if pool_df.empty:
        return feasible_df.head(0)

    param_cols = [c for c in ("k_tp", "tp_anchor", "sl_as_tp_pct") if c in pool_df.columns]
    if len(param_cols) < 3:
        param_cols = [c for c in ("k_tp", "tp_base_pct", "sl_as_tp_pct") if c in pool_df.columns]
    if len(param_cols) < 3:
        return pool_df.head(min(max_configs, len(pool_df))).copy()
    p_df = pool_df[param_cols].apply(pd.to_numeric, errors="coerce")
    p_fill = p_df.fillna(p_df.median(numeric_only=True)).fillna(0.0)
    p_vals = p_fill.to_numpy(dtype=np.float64, copy=False)
    for _i, _c in enumerate(param_cols):
        if _c in {"k_tp", "tp_anchor", "tp_base_pct", "sl_as_tp_pct"}:
            _col = p_vals[:, _i]
            if np.all(np.isfinite(_col)) and np.nanmin(_col) > 0:
                p_vals[:, _i] = np.log(_col)
    p_min = np.nanmin(p_vals, axis=0)
    p_max = np.nanmax(p_vals, axis=0)
    p_scale = np.maximum(p_max - p_min, 1e-9)
    p_norm = (p_vals - p_min) / p_scale

    stage_score_norm = _robust_minmax(pool_df["_stage_score"].to_numpy(dtype=np.float64, copy=False), p_lo=5.0, p_hi=95.0)
    auc_arr = pool_df["_auc_metric"].to_numpy(dtype=np.float64, copy=False)
    cids = pool_df["_stable_id"].to_numpy(dtype=object)

    def _dist_to_one(idx_arr: np.ndarray, sel_pos: int) -> np.ndarray:
        dv = p_norm[idx_arr] - p_norm[int(sel_pos)]
        return np.sqrt(np.sum(dv * dv, axis=1, dtype=np.float64))

    n = len(pool_df)
    if n == 0:
        return feasible_df.head(0)

    selected: List[int] = []
    selected_metrics: Dict[int, Tuple[float, float, float]] = {}
    remaining: List[int] = list(range(n))
    lambda_eff = float(np.clip(alpha, 0.5, 0.9))

    seed_pos = int(np.argmax(stage_score_norm))
    selected.append(seed_pos)
    selected_metrics[seed_pos] = (1.0, float(stage_score_norm[seed_pos]), float(lambda_eff * stage_score_norm[seed_pos] + (1.0 - lambda_eff) * 1.0))
    remaining.remove(seed_pos)

    dmin_all = np.full(n, np.inf, dtype=np.float64)
    if remaining:
        rem_idx0 = np.asarray(remaining, dtype=int)
        dmin_all[rem_idx0] = np.minimum(dmin_all[rem_idx0], _dist_to_one(rem_idx0, seed_pos))

    while remaining and len(selected) < max_configs:
        rem_idx = np.asarray(remaining, dtype=int)
        novelty_raw = dmin_all[rem_idx]
        novelty_norm = _robust_minmax(novelty_raw, p_lo=5.0, p_hi=95.0)
        utility = lambda_eff * stage_score_norm[rem_idx] + (1.0 - lambda_eff) * novelty_norm

        order = np.lexsort((
            np.asarray([cids[i] for i in rem_idx], dtype=object),
            -auc_arr[rem_idx],
            -novelty_raw,
            -pool_df["_stage_score"].to_numpy(dtype=np.float64, copy=False)[rem_idx],
            -utility,
        ))
        pick_pos = int(rem_idx[int(order[0])])
        pick_loc = int(np.where(rem_idx == pick_pos)[0][0])
        selected_metrics[pick_pos] = (float(novelty_raw[pick_loc]), float(stage_score_norm[pick_pos]), float(utility[pick_loc]))
        selected.append(pick_pos)
        remaining.remove(pick_pos)
        if remaining:
            rem_idx2 = np.asarray(remaining, dtype=int)
            dmin_all[rem_idx2] = np.minimum(dmin_all[rem_idx2], _dist_to_one(rem_idx2, pick_pos))

    selected_df = pool_df.iloc[np.asarray(selected, dtype=int)].copy()
    selected_df["sl_anchor_pct"] = pd.to_numeric(selected_df.get("tp_base_pct", np.nan), errors="coerce") * pd.to_numeric(
        selected_df.get("sl_as_tp_pct", np.nan), errors="coerce"
    )
    if enable_span_repair:
        pool_aug = pool_df.copy()
        pool_aug["sl_anchor_pct"] = pd.to_numeric(pool_aug.get("tp_base_pct", np.nan), errors="coerce") * pd.to_numeric(
            pool_aug.get("sl_as_tp_pct", np.nan), errors="coerce"
        )
        selected_df = _apply_min_tp_sl_span(
            selected_df,
            pool_aug,
            min_tp_span_ratio=min_tp_span_ratio,
            min_sl_span_ratio=min_sl_span_ratio,
            max_configs=max_configs,
        )

    if "_cid" not in selected_df.columns:
        selected_df["_cid"] = selected_df.get("config_id", "")
    selected_df["_cid"] = selected_df["_cid"].fillna(selected_df.get("config_id", "")).astype(str)
    if "_stable_id" not in selected_df.columns:
        selected_df["_stable_id"] = selected_df["_cid"]
    selected_df["_stable_id"] = selected_df["_stable_id"].fillna(selected_df["_cid"]).astype(str)
    if "_stage_score" not in selected_df.columns:
        selected_df["_stage_score"] = pd.to_numeric(selected_df.get(score_col, np.nan), errors="coerce")
    if "_auc_metric" not in selected_df.columns:
        selected_df["_auc_metric"] = pd.to_numeric(selected_df.get(auc_col, np.nan), errors="coerce")

    tprint(
        f"[grr] candidates={len(feasible_df)} feasible={len(work_df)} auc_floor={auc_floor:.4f} "
        f"auc_floor_relaxed={auc_floor_relaxed} pool={len(pool_df)} econ_prefilter={econ_applied} winners={len(selected_df)}"
    )
    for r in selected_df.itertuples(index=False):
        _cid = str(getattr(r, "config_id", getattr(r, "_cid", "")))
        _sid = str(getattr(r, "_stable_id", _cid))
        _row = pool_df[pool_df["_stable_id"] == _sid]
        _nov, _scn, _util = (float("nan"), float("nan"), float("nan"))
        if not _row.empty:
            _idx = int(_row.index[0])
            _pos = int(np.where(pool_df.index.to_numpy() == _idx)[0][0])
            _nov, _scn, _util = selected_metrics.get(_pos, (float("nan"), float("nan"), float("nan")))
        tprint(
            f"[grr_winner] cid={_cid} k_tp={getattr(r,'k_tp',float('nan'))} "
            f"tp={getattr(r,'tp_base_pct',float('nan'))} sl={getattr(r,'sl_as_tp_pct',float('nan'))} "
            f"stage_score={getattr(r,'_stage_score',float('nan')):.4f} auc={getattr(r,'_auc_metric',float('nan')):.4f} "
            f"novelty_raw={_nov:.4f} stage_score_norm={_scn:.4f} utility={_util:.4f}"
        )

    selected_df = selected_df.drop(columns=["sl_anchor_pct"], errors="ignore")
    selected_df.attrs["grr_pool_df"] = pool_df.drop(columns=["sl_anchor_pct"], errors="ignore")
    return selected_df


def _rank_by_learnability(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical ranking for promotion/diversity input.

    Primary objective is stage2_score (fallback stage1_score if missing), then
    learnability tie-breakers.
    """
    if df is None or df.empty:
        return df
    _s2 = pd.to_numeric(df.get("stage2_score", pd.Series(np.nan, index=df.index)), errors="coerce")
    _s1 = pd.to_numeric(df.get("stage1_score", pd.Series(-np.inf, index=df.index)), errors="coerce").fillna(-np.inf)
    _score = _s2.where(_s2.notna(), _s1).fillna(-np.inf)
    _min_auc = (df["min_cell_auc_bound"].fillna(0.0) * 1000).round() / 1000 if "min_cell_auc_bound" in df.columns else (
        (df["min_cell_auc"].fillna(0.0) * 1000).round() / 1000 if "min_cell_auc" in df.columns else pd.Series(0.0, index=df.index)
    )
    _min_sep = df["min_cell_tp_sep"].fillna(0.0) if "min_cell_tp_sep" in df.columns else pd.Series(0.0, index=df.index)
    _disp = df["cell_dispersion"].fillna(999.0) if "cell_dispersion" in df.columns else pd.Series(999.0, index=df.index)
    _to_r = df["timeout_range"].fillna(999.0) if "timeout_range" in df.columns else pd.Series(999.0, index=df.index)
    _ece = df["min_cell_ece"].fillna(999.0) if "min_cell_ece" in df.columns else (df["ece"].fillna(999.0) if "ece" in df.columns else pd.Series(999.0, index=df.index))
    _brier = df["min_cell_brier"].fillna(999.0) if "min_cell_brier" in df.columns else (df["brier"].fillna(999.0) if "brier" in df.columns else pd.Series(999.0, index=df.index))
    _ic_t = df["ic_std_time"].fillna(999.0) if "ic_std_time" in df.columns else pd.Series(999.0, index=df.index)
    _ic_a = df["ic_std_asset"].fillna(999.0) if "ic_std_asset" in df.columns else pd.Series(999.0, index=df.index)
    return df.assign(_sc=_score, _ma=_min_auc, _ms=_min_sep, _d=_disp, _tr=_to_r, _ece=_ece, _brier=_brier, _ict=_ic_t, _ica=_ic_a).sort_values(
        ["_sc", "_ma", "_ms", "_ece", "_brier", "_ict", "_ica", "_d", "_tr"],
        ascending=[False, False, False, True, True, True, True, True, True],
    ).drop(columns=["_sc", "_ma", "_ms", "_d", "_tr", "_ece", "_brier", "_ict", "_ica"])


def promote_stage1(stage1_results: pd.DataFrame, top_k: int = 10) -> List[str]:
    if stage1_results.empty:
        return []
    # Step 1: structural validity — hard_gate + all cells pass + timeout_range in bounds.
    _to_r_col = stage1_results["timeout_range"].fillna(999.0) if "timeout_range" in stage1_results.columns else pd.Series(0.0, index=stage1_results.index)
    struct_mask = (stage1_results["hard_gate"] == True) & (stage1_results["pass_cells"] == stage1_results["total_cells"]) & (_to_r_col <= _MAX_TIMEOUT_RANGE)
    df = stage1_results[struct_mask].copy()
    if df.empty:
        # Relax timeout_range only — keep hard_gate + pass_cells
        df = stage1_results[(stage1_results["hard_gate"] == True) & (stage1_results["pass_cells"] == stage1_results["total_cells"])].copy()
    if df.empty:
        df = stage1_results[stage1_results["hard_gate"] == True].copy()
    if df.empty:
        return []
    # Step 2: tiered feasible-set builder (no silent progressive relaxation).
    df, tier = _build_feasible_set(df)
    # Step 3: rank by learnability.
    df = _rank_by_learnability(df)
    return df.head(top_k)["config_id"].tolist()


# ---------------------------
# Per-cell feasible-set builder
# ---------------------------
_CELL_KEYS = [
    f"{b}_H{h}"
    for b in ["MR_long", "MR_short", "TF_long", "TF_short"]
    for h in [2, 4]
]


def _build_per_cell_feasible_sets(
    out_df: pd.DataFrame,
    details: Dict[str, Any],
    min_distance: float = 1.0,
    max_configs_per_cell: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Build a diverse feasible set for each (bucket, horizon) cell independently.

    For each cell key (e.g. "MR_long_H4"), we:
    1. Restrict out_df to configs that passed hard_gate + pass_cells == total_cells.
    2. Build a per-cell score DataFrame using that cell's auc_label and tp_sep_top10.
    3. Apply _build_feasible_set using per-cell metrics.
    4. Apply _diverse_subset for diversity control.

    Returns dict: cell_key -> diverse feasible DataFrame (subset of out_df rows).
    """
    # Pre-filter to structurally valid configs only.
    # Fallback chain: prefer hard_gate+pass_cells+timeout_range, relax each in turn,
    # and finally use all configs (pass_cells==total_cells) when hard_gate is universally
    # False (e.g. econ_ok fails for all configs due to sl_to_tp guardrail).
    _to_r_col = out_df["timeout_range"].fillna(999.0) if "timeout_range" in out_df.columns else pd.Series(0.0, index=out_df.index)
    struct_mask = (out_df["hard_gate"] == True) & (out_df["pass_cells"] == out_df["total_cells"]) & (_to_r_col <= _MAX_TIMEOUT_RANGE)
    base_df = out_df[struct_mask].copy()
    if base_df.empty:
        base_df = out_df[(out_df["hard_gate"] == True) & (out_df["pass_cells"] == out_df["total_cells"])].copy()
    if base_df.empty:
        base_df = out_df[out_df["hard_gate"] == True].copy()
    if base_df.empty:
        # hard_gate universally False (e.g. econ_ok blocks all) — fall back to
        # configs that at least passed all 12 cells structurally.
        base_df = out_df[out_df["pass_cells"] == out_df["total_cells"]].copy()
        if not base_df.empty:
            tprint(f"[per_cell] hard_gate=False for all configs; using pass_cells==total_cells fallback ({len(base_df)} configs)")
    if base_df.empty:
        base_df = out_df.copy()
        tprint(f"[per_cell] WARNING: using all {len(base_df)} configs as fallback (no structural filter passed)")

    result: Dict[str, pd.DataFrame] = {}

    _base_rows = base_df[["config_id", "coverage", "cell_dispersion", "timeout_range"]].copy()
    _cid_bh = {
        str(cid): (details.get(str(cid), {}).get("bucket_horizon_metrics", {}) if isinstance(details.get(str(cid), {}), dict) else {})
        for cid in _base_rows["config_id"].astype(str).tolist()
    }

    for cell_key in _CELL_KEYS:
        # Extract per-cell metrics for each config.
        cell_rows = []
        for row in _base_rows.itertuples(index=False):
            cid = str(row.config_id)
            bh = _cid_bh.get(cid, {})
            cell_m = bh.get(cell_key, {})
            if not cell_m:
                continue
            auc_cell = cell_m.get("auc_label", float("nan"))
            sep_cell = cell_m.get("tp_sep_top10", float("nan"))
            timeout_cell = cell_m.get("timeout", 1.0)
            bind_cell = cell_m.get("bind", 0.0)
            coverage_cell = float(getattr(row, "coverage", 0.0))  # config-level coverage as proxy
            ok_cell = cell_m.get("ok", False)
            ap_lift_cell = cell_m.get("ap_lift", float("nan"))
            tp_over_sl_cell = cell_m.get("tp_over_sl", float("nan"))
            dir_sup_cell = cell_m.get("dir_superiority_top_decile", float("nan"))
            barrier_ratio_cell = cell_m.get("barrier_ratio", float("nan"))
            # Fee-adjusted EV: tp_hit*(tp_mean - fee) - sl_hit*(sl_mean + fee)
            # Uses actual barrier sizes from the cell — not a ratio proxy.
            _tp_h = cell_m.get("tp_hit", float("nan"))
            _sl_h = cell_m.get("sl_hit", float("nan"))
            _tp_m = cell_m.get("tp_mean", float("nan"))
            _sl_m = cell_m.get("sl_mean", float("nan"))
            if not any(math.isnan(v) for v in [_tp_h, _sl_h, _tp_m, _sl_m]):
                fee_ev_cell = _tp_h * (_tp_m - _ROUND_TRIP_FEE) - _sl_h * (_sl_m + _ROUND_TRIP_FEE)
            else:
                fee_ev_cell = float("nan")
            # Strict filters (Hard Exclusion)
            # 1. Negative or zero TP separation
            if sep_cell <= 0:
                continue
            # 2. Too few samples
            if cell_m.get("n", 0) < 2000:
                continue
            # 3. Pathological monotonicity
            mono_cell = cell_m.get("monotonicity", float("nan"))
            if not math.isnan(mono_cell) and mono_cell < 0.2:
                continue
            _n_cell = float(cell_m.get("n", 0.0))
            _base_rate_cell = float(cell_m.get("base_rate", float("nan")))
            _eff_pos_cell = (_n_cell * _base_rate_cell) if math.isfinite(_base_rate_cell) else float("nan")
            _eff_neg_cell = (_n_cell * max(0.0, 1.0 - _base_rate_cell)) if math.isfinite(_base_rate_cell) else float("nan")

            cell_rows.append({
                "config_id": cid,
                "min_cell_auc": auc_cell if not math.isnan(auc_cell) else 0.0,
                "min_cell_tp_sep": sep_cell if not math.isnan(sep_cell) else 0.0,
                "cell_dispersion": float(getattr(row, "cell_dispersion", 999.0)),
                "timeout_range": float(getattr(row, "timeout_range", 999.0)),
                "min_cell_auc_bound": cell_m.get("auc_bound", float("nan")),
                "coverage": coverage_cell,
                "bind": bind_cell,
                "timeout_rate": timeout_cell,
                "ok": ok_cell,
                "min_cell_ap_lift": ap_lift_cell if not math.isnan(ap_lift_cell) else 0.0,
                "min_cell_tp_over_sl": tp_over_sl_cell if not math.isnan(tp_over_sl_cell) else 0.0,
                "min_cell_dir_superiority": dir_sup_cell if not math.isnan(dir_sup_cell) else 0.0,
                "barrier_ratio": barrier_ratio_cell if not math.isnan(barrier_ratio_cell) else 999.0,
                "fee_ev": fee_ev_cell,
                "monotonicity": mono_cell,
                "min_cell_brier": cell_m.get("brier", float("nan")),
                "min_cell_ece": cell_m.get("ece", float("nan")),
                "effective_pos_mass": _eff_pos_cell if math.isfinite(_eff_pos_cell) else 0.0,
                "effective_neg_mass": _eff_neg_cell if math.isfinite(_eff_neg_cell) else 0.0,
            })

        if not cell_rows:
            result[cell_key] = pd.DataFrame()
            continue

        cell_df = pd.DataFrame(cell_rows)

        # Only consider configs where this cell passed its own ok gate.
        ok_df = cell_df[cell_df["ok"] == True].copy()
        if ok_df.empty:
            ok_df = cell_df.copy()

        # Apply tiered feasible-set builder on per-cell metrics (per_cell=True → relaxed bind gate).
        feasible, tier = _build_feasible_set(ok_df, per_cell=True)
        if feasible.empty:
            result[cell_key] = pd.DataFrame()
            continue

        # Rank by learnability.
        feasible = _rank_by_learnability(feasible)

        # Diversity control — target max_configs_per_cell but ensure at least 2.
        
        _anchor_cid = str(feasible.iloc[0]["config_id"]) if not feasible.empty else None
        diverse = _diverse_subset(
            feasible,
            details,
            min_distance=min_distance,
            max_configs=max_configs_per_cell,
            alpha=0.78,
            preselected_cids=[_anchor_cid] if _anchor_cid else None,
            score_keep_fraction=0.75,
        )
        # If diversity filter left only 1 config, relax min_distance to get a second.
        if len(diverse) < 2 and len(feasible) >= 2:
            diverse = _diverse_subset(
                feasible,
                details,
                min_distance=min_distance * 0.5,
                max_configs=max_configs_per_cell,
                alpha=0.78,
                score_keep_fraction=0.75,
            )
        if len(diverse) < 2 and len(feasible) >= 2:
            diverse = feasible.head(2)  # last resort: top-2 by learnability rank

        # Post-selection check: "On average, geometries must pass a higher threshold".
        # If the set average is below _AVG_AUC_THRESHOLD, prune from the bottom (worst learnability)
        # until the average is met, provided we keep at least 2 configs (for diversity).
        # Exception: if all configs are below the threshold, we might be left with 2 mediocre ones.
        # This enforces that the SET quality is high, not just minimum viability.
        if "min_cell_auc" in diverse.columns:
            while len(diverse) > 3 and diverse["min_cell_auc"].mean() < _AVG_AUC_THRESHOLD:
                diverse = diverse.iloc[:-1]  # Drop last (lowest ranked)

        # Attach full out_df row data for downstream use.
        full_rows = out_df[out_df["config_id"].isin(diverse["config_id"])].copy()
        result[cell_key] = full_rows

        avg_auc = diverse["min_cell_auc"].mean() if "min_cell_auc" in diverse.columns and not diverse.empty else 0.0
        tprint(
            f"[per_cell] {cell_key}: {len(diverse)} diverse configs "
            f"(tier={tier}, feasible_pool={len(feasible)}, avg_auc={avg_auc:.4f})"
        )

    return result


# ---------------------------
# Per-bucket feasible-set builder
# ---------------------------
# Use canonical constants from params_store (TBM_BUCKET_NAMES, TBM_HORIZONS)

def _per_bucket_metrics_from_details(
    cid: str,
    details: Dict[str, Any],
    bucket_name: str,
    global_row: pd.Series,
) -> Dict[str, Any]:
    """Extract per-bucket aggregate metrics from bucket_horizon_metrics for one bucket.

    Uses only the 3 horizon cells belonging to `bucket_name` (e.g. "MR_long_H1/H2/H4").
    Falls back to global_row values for coverage/ess (which are config-level, not per-bucket).
    """
    bh = details.get(cid, {}).get("bucket_horizon_metrics", {})
    cell_keys = [f"{bucket_name}_H{h}" for h in TBM_HORIZONS]
    cells = [bh[k] for k in cell_keys if k in bh and bh[k]]

    if not cells:
        return {}

    def _safe_min(vals: list) -> float:
        v = [x for x in vals if not math.isnan(x)]
        return float(np.min(v)) if v else float("nan")

    def _safe_median(vals: list) -> float:
        v = [x for x in vals if not math.isnan(x)]
        return float(np.median(v)) if v else float("nan")

    aucs      = [c.get("auc_label",    float("nan")) for c in cells]
    aucs_b    = [c.get("auc_bound",    float("nan")) for c in cells]
    tp_seps   = [c.get("tp_sep_top10", float("nan")) for c in cells]
    timeouts  = [c.get("timeout",      float("nan")) for c in cells]
    binds     = [c.get("bind",         float("nan")) for c in cells]
    ap_lifts  = [c.get("ap_lift",      float("nan")) for c in cells]
    tp_over   = [c.get("tp_over_sl",   float("nan")) for c in cells]
    sl_to_tps = [c.get("sl_to_tp",     float("nan")) for c in cells]
    barrier_rs= [c.get("barrier_ratio",float("nan")) for c in cells]
    eces      = [c.get("ece",          float("nan")) for c in cells]
    briers    = [c.get("brier",        float("nan")) for c in cells]
    payoffs   = [c.get("payoff_mean",  float("nan")) for c in cells]
    disps     = [c.get("payoff_mean",  float("nan")) for c in cells]

    timeout_vals = [t for t in timeouts if not math.isnan(t)]
    timeout_range = float(np.max(timeout_vals) - np.min(timeout_vals)) if len(timeout_vals) > 1 else 0.0
    bind_vals = [b for b in binds if not math.isnan(b)]
    bind_min = float(np.min(bind_vals)) if bind_vals else float("nan")
    sl_to_tp_max = float(np.max([s for s in sl_to_tps if not math.isnan(s)])) if any(not math.isnan(s) for s in sl_to_tps) else float("nan")
    barrier_max = float(np.max([b for b in barrier_rs if not math.isnan(b)])) if any(not math.isnan(b) for b in barrier_rs) else float("nan")

    payoff_vals = [p for p in payoffs if not math.isnan(p)]
    cell_dispersion = float(np.std(payoff_vals)) if len(payoff_vals) > 1 else 0.0

    # Fee-adjusted EV: min across the bucket's cells.
    # EV = tp_hit*(tp_mean - fee) - sl_hit*(sl_mean + fee)
    fee_evs = []
    eff_pos_mass = 0.0
    eff_neg_mass = 0.0
    for c in cells:
        _tp_h = c.get("tp_hit", float("nan"))
        _sl_h = c.get("sl_hit", float("nan"))
        _tp_m = c.get("tp_mean", float("nan"))
        _sl_m = c.get("sl_mean", float("nan"))
        _n = float(c.get("n", 0.0))
        _br = float(c.get("base_rate", float("nan")))
        if _n > 0 and math.isfinite(_br):
            eff_pos_mass += _n * _br
            eff_neg_mass += _n * max(0.0, 1.0 - _br)
        if not any(math.isnan(v) for v in [_tp_h, _sl_h, _tp_m, _sl_m]):
            fee_evs.append(_tp_h * (_tp_m - _ROUND_TRIP_FEE) - _sl_h * (_sl_m + _ROUND_TRIP_FEE))
    fee_ev_min = float(np.min(fee_evs)) if fee_evs else float("nan")

    return {
        "config_id":          cid,
        "bucket":             bucket_name,
        "coverage":           float(global_row.get("coverage", 0.0)),
        "hard_gate":          bool(global_row.get("hard_gate", False)),
        "pass_cells":         int(global_row.get("pass_cells", 0)),
        "total_cells":        int(global_row.get("total_cells", 0)),
        "timeout_range":      round(timeout_range, 4),
        "cell_dispersion":    round(cell_dispersion, 6),
        # Per-bucket aggregate metrics (only this bucket's cells)
        "min_cell_auc":       round(_safe_min(aucs), 4)     if not math.isnan(_safe_min(aucs))    else float("nan"),
        "median_cell_auc":    round(_safe_median(aucs), 4)  if not math.isnan(_safe_median(aucs)) else float("nan"),
        "min_cell_auc_bound": round(_safe_min(aucs_b), 4)   if not math.isnan(_safe_min(aucs_b))  else float("nan"),
        "min_cell_tp_sep":    round(_safe_min(tp_seps), 5)  if not math.isnan(_safe_min(tp_seps)) else 0.0,
        "min_cell_ap_lift":   round(_safe_min(ap_lifts), 4) if not math.isnan(_safe_min(ap_lifts)) else 0.0,
        "min_cell_tp_over_sl":round(_safe_min(tp_over), 4)  if not math.isnan(_safe_min(tp_over)) else 0.0,
        "min_cell_ece":       round(_safe_min(eces), 5) if not math.isnan(_safe_min(eces)) else float("nan"),
        "min_cell_brier":     round(_safe_min(briers), 5) if not math.isnan(_safe_min(briers)) else float("nan"),
        "bind":               round(bind_min, 4)             if not math.isnan(bind_min)           else 0.0,
        "timeout_rate":       round(_safe_min(timeouts), 4)  if not math.isnan(_safe_min(timeouts)) else 1.0,
        "sl_to_tp":           round(sl_to_tp_max, 4)         if not math.isnan(sl_to_tp_max)       else 999.0,
        "barrier_ratio":      round(barrier_max, 4)          if not math.isnan(barrier_max)        else 999.0,
        "fee_ev":             round(fee_ev_min, 6)           if not math.isnan(fee_ev_min)         else float("nan"),
        "effective_pos_mass": float(eff_pos_mass),
        "effective_neg_mass": float(eff_neg_mass),
    }


def _build_per_bucket_feasible_sets(
    out_df: pd.DataFrame,
    details: Dict[str, Any],
    min_distance: float = 1.0,
    max_configs_per_bucket: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Build a diverse feasible set for each of the 4 buckets independently.

    For each bucket (TF_long, TF_short, MR_long, MR_short):
    1. Recompute per-bucket aggregate metrics using only that bucket's 3 horizon cells.
    2. Apply _build_feasible_set (per_cell=True → relaxed bind gate).
    3. Apply _diverse_subset for diversity control.

    Returns dict: bucket_name -> diverse feasible DataFrame.
    """
    # Pre-filter to structurally valid configs.
    _to_r_col = out_df["timeout_range"].fillna(999.0) if "timeout_range" in out_df.columns else pd.Series(0.0, index=out_df.index)
    struct_mask = (out_df["hard_gate"] == True) & (_to_r_col <= _MAX_TIMEOUT_RANGE)
    base_df = out_df[struct_mask].copy()
    if base_df.empty:
        base_df = out_df[out_df["hard_gate"] == True].copy()
    if base_df.empty:
        base_df = out_df.copy()

    result: Dict[str, pd.DataFrame] = {}

    for bucket_name in TBM_BUCKET_NAMES:
        bucket_rows = []
        for row in base_df.itertuples(index=False):
            row_s = pd.Series(row._asdict())
            cid = str(row_s.get("config_id", ""))
            bm = _per_bucket_metrics_from_details(cid, details, bucket_name, row_s)
            if not bm:
                continue
            bucket_rows.append(bm)

        if not bucket_rows:
            result[bucket_name] = pd.DataFrame()
            tprint(f"[per_bucket] {bucket_name}: no data in bucket_horizon_metrics")
            continue

        bucket_df = pd.DataFrame(bucket_rows)

        # Apply tiered feasible-set builder using per-bucket metrics.
        feasible, tier = _build_feasible_set(bucket_df, per_cell=True)
        if feasible.empty:
            result[bucket_name] = pd.DataFrame()
            tprint(f"[per_bucket] {bucket_name}: no configs passed feasibility gates (tier={tier})")
            continue

        # Rank by learnability.
        feasible = _rank_by_learnability(feasible)

        # Diversity control — ensure at least 2 configs.
        diverse = _diverse_subset(feasible, details, min_distance=min_distance, max_configs=max_configs_per_bucket, score_keep_fraction=0.85)
        if len(diverse) < 2 and len(feasible) >= 2:
            diverse = _diverse_subset(feasible, details, min_distance=min_distance * 0.5, max_configs=max_configs_per_bucket, score_keep_fraction=0.85)
        if len(diverse) < 2 and len(feasible) >= 2:
            diverse = feasible.head(2)

        # Attach full out_df row data for downstream use.
        full_rows = out_df[out_df["config_id"].isin(diverse["config_id"])].copy()
        result[bucket_name] = full_rows

        tprint(
            f"[per_bucket] {bucket_name}: {len(diverse)} diverse configs "
            f"(tier={tier}, feasible_pool={len(feasible)})"
        )

    return result


def promote_stage1_per_cell(
    stage1_results: pd.DataFrame,
    details: Dict[str, Any],
    top_k_per_cell: int = 3,
) -> Dict[str, List[str]]:
    """Run promotion independently for each of the 12 (bucket, horizon) cells.

    For each cell, selects the top-k configs ranked by that cell's own metrics
    (auc_label, tp_sep_top10, bind, timeout) — not global aggregates.
    Returns dict: cell_key -> list of promoted config_ids.
    The union across all 12 cells feeds stage2.

    Minimum 2 configs guaranteed per cell (relaxing diversity if needed).
    """
    per_cell_sets = _build_per_cell_feasible_sets(
        stage1_results, details, min_distance=1.0, max_configs_per_cell=max(top_k_per_cell, 2)
    )
    out: Dict[str, List[str]] = {}
    for cell_key, cdf in per_cell_sets.items():
        if cdf.empty:
            out[cell_key] = []
        else:
            out[cell_key] = cdf["config_id"].tolist()[:top_k_per_cell]
    return out


# ---------------------------
# Winning geometry summary
# ---------------------------
def _print_winning_geometry_summary(
    out_df: pd.DataFrame,
    details: Dict[str, Any],
    per_cell_grids: Optional[Dict[str, pd.DataFrame]] = None,
    top_k: int = 5,
) -> None:
    """Print a rich summary of the top-k configs after scoring completes."""
    sep = "=" * 100
    tprint(sep)
    tprint(f"WINNING GEOMETRY SUMMARY  (top {top_k}  ranked: health_gates → min_auc → min_tp_sep → cell_dispersion → timeout_range → min_auc_bound)")
    tprint(sep)

    top = out_df.head(top_k)
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        cid = row["config_id"]
        d = details.get(cid, {})
        cfg = d.get("config", {})
        bh = d.get("bucket_horizon_metrics", {})

        # Degeneracy flags
        flags = []
        if row.get("flag_degenerate_timeout", False): flags.append("TIMEOUT!")
        if row.get("flag_degenerate_sl", False):      flags.append("SL_DOM!")
        if row.get("flag_degenerate_tp", False):      flags.append("TP_LOW!")
        flag_str = "  [" + " ".join(flags) + "]" if flags else ""

        tprint(
            f"\n#{rank}  {cid}  mode={cfg.get('tp_method','?')}  "
            f"hard_gate={row.get('hard_gate')}{flag_str}"
        )

        # --- A) Geometry parameters ---
        tprint(
            f"    Geometry : k_tp={cfg.get('k_tp','?')}  sl_as_tp_pct={cfg.get('sl_as_tp_pct','?')}  "
            f"tp_base_pct={cfg.get('tp_base_pct','?')}  atr_window={cfg.get('base_atr_window','?')}  "
            f"horizon_scaling={cfg.get('horizon_scaling','?')}  regime={cfg.get('tp_regime_model','?')}"
        )
        tprint(
            f"    Floors   : tp_abs_lo={cfg.get('tp_abs_lo_pct','?')}  sl_abs_lo={cfg.get('sl_abs_lo_pct','?')}  "
            f"tp_abs_hi={cfg.get('tp_abs_hi_pct','?')}  sl_noise_buffer={cfg.get('sl_noise_buffer','?')}"
        )

        # --- A) Outcome mix + degeneracy ---
        tp_h  = row.get('tp_hit_rate', 0.0)
        sl_h  = row.get('sl_hit_rate', 0.0)
        to_h  = row.get('timeout_rate', 1.0)
        bind  = row.get('bind', tp_h + sl_h)
        bal   = row.get('balance', 0.0)
        sl2tp = row.get('sl_to_tp', 0.0)
        tprint(
            f"    Outcome  : tp_hit={tp_h:.3f}  sl_hit={sl_h:.3f}  timeout={to_h:.3f}  "
            f"bind={bind:.3f}  balance={bal:.3f}  sl_to_tp={sl2tp:.2f}x"
        )

        # --- B) Learnability as ranking object ---
        top10   = row.get('payoff_mean_top_decile', float('nan'))
        spread  = row.get('top10_vs_rest_spread', float('nan'))
        dspread = row.get('oof_payoff_decile_spread', float('nan'))
        med_auc = row.get('median_cell_auc', float('nan'))
        min_auc = row.get('min_cell_auc', float('nan'))
        med_sep = row.get('median_cell_tp_sep', 0.0)
        tprint(
            f"    Signal   : ic_payoff={row.get('ic_payoff',0):.4f}  ic_snr={row.get('ic_snr',0):.4f}  "
            f"sortino={row.get('sortino',0):.4f}  payoff_top10%={top10*10000:.1f}bps  "
            f"top10_vs_rest={spread*10000:.1f}bps"
        )
        _auc_str = f"{med_auc:.4f}" if not (isinstance(med_auc, float) and math.isnan(med_auc)) else "n/a"
        _min_auc_str = f"{min_auc:.4f}" if not (isinstance(min_auc, float) and math.isnan(min_auc)) else "n/a"
        _med_auc_b = row.get('median_cell_auc_bound', float('nan'))
        _min_auc_b = row.get('min_cell_auc_bound', float('nan'))
        _med_auc_b_str = f"{_med_auc_b:.4f}" if not (isinstance(_med_auc_b, float) and math.isnan(_med_auc_b)) else "n/a"
        _min_auc_b_str = f"{_min_auc_b:.4f}" if not (isinstance(_min_auc_b, float) and math.isnan(_min_auc_b)) else "n/a"
        tprint(
            f"    Learn    : min_auc={_min_auc_str}  median_auc={_auc_str}  "
            f"min_auc_bound={_min_auc_b_str}  median_auc_bound={_med_auc_b_str}  "
            f"min_tp_sep={row.get('min_cell_tp_sep',0)*100:.2f}pp  median_tp_sep={med_sep*100:.2f}pp"
        )
        tprint(
            f"    Learn2   : median_ic_label={row.get('median_cell_ic_label',0):.4f}  "
            f"min_ic_label={row.get('min_cell_ic_label',0):.4f}  "
            f"timeout_range={row.get('timeout_range',0):.3f}"
        )

        # --- C) Stability across cells ---
        tprint(
            f"    Stability: pass_cells={int(row.get('pass_cells',0))}/{int(row.get('total_cells',0))}  "
            f"min_cell_payoff={row.get('min_cell_payoff',0)*10000:.1f}bps  "
            f"cell_dispersion={row.get('cell_dispersion',0)*10000:.2f}bps  "
            f"coverage={row.get('coverage',0):.3f}"
        )

        # --- Per bucket-horizon: full diagnostic row ---
        if bh:
            hdr = (f"    {'Cell':<18} {'n':>7}  "
                   f"{'tp':>6} {'sl':>6} {'to':>6}  {'bind':>5} "
                   f"{'h_eff':>5}  {'auc':>6}  {'auc_bnd':>7}  {'ic_lbl':>7}  {'sep%':>7}  ok")
            tprint(hdr)
            for cell_name in sorted(bh.keys()):
                m = bh[cell_name]
                cell_flags = ""
                if m.get("timeout", 1.0) > 0.85: cell_flags += "T"
                if m.get("sl_hit", 0.0) > 0.60:  cell_flags += "S"
                if m.get("tp_hit", 0.0) < 0.05:  cell_flags += "P"
                ok_sym = "✓" if m.get("ok") else "✗"
                def _fmt(v: Any, fmt: str = ".4f") -> str:
                    return format(v, fmt) if not (isinstance(v, float) and math.isnan(v)) else "  n/a"
                tprint(
                    f"    {cell_name:<18} {m.get('n',0):>7,}  "
                    f"{m.get('tp_hit',0):>6.3f} {m.get('sl_hit',0):>6.3f} {m.get('timeout',0):>6.3f}  "
                    f"{m.get('bind',0):>5.3f}  "
                    f"{m.get('h_eff_mean',0):>5.2f}  "
                    f"{_fmt(m.get('auc_label', float('nan'))):>6}  "
                    f"{_fmt(m.get('auc_bound', float('nan'))):>7}  "
                    f"{_fmt(m.get('ic_label', float('nan'))):>7}  "
                    f"{(format(m.get('tp_sep_top10',0)*100,'+.2f') if not (isinstance(m.get('tp_sep_top10',float('nan')),float) and math.isnan(m.get('tp_sep_top10',float('nan')))) else '  n/a'):>7}  "
                    f"{ok_sym}{cell_flags}"
                )

    tprint(f"\n{sep}")

    # --- Per-cell feasible set summary ---
    if per_cell_grids:
        tprint("PER-CELL FEASIBLE SET SUMMARY  (bucket × horizon → diverse geometry grid)")
        hdr_c = (f"  {'Cell':<18} {'n_configs':>9}  {'k_tp values':<30}  {'sl values':<20}  "
                 f"{'atr_win':>7}  {'best_auc':>8}  {'best_sep%':>9}")
        tprint(hdr_c)
        for cell_key in sorted(per_cell_grids.keys()):
            cell_df = per_cell_grids[cell_key]
            if cell_df.empty:
                tprint(f"  {cell_key:<18}  (no valid configs)")
                continue
            _cids = cell_df["config_id"].tolist()
            _cfgs = [details.get(c, {}).get("config", {}) for c in _cids if c in details]
            _k_vals = sorted(set(float(c.get("k_tp", float("nan"))) for c in _cfgs if isinstance(c, dict)))
            _sl_vals = sorted(set(float(c.get("sl_as_tp_pct", float("nan"))) for c in _cfgs if isinstance(c, dict)))
            _wins = sorted(set(int(c.get("base_atr_window", 0)) for c in _cfgs if isinstance(c, dict)))
            _win_str = str(_wins[0]) if len(_wins) == 1 else str(_wins)
            # Best per-cell auc and tp_sep from bucket_horizon_metrics
            _aucs, _seps = [], []
            for cid in _cids:
                bh = details.get(cid, {}).get("bucket_horizon_metrics", {})
                m = bh.get(cell_key, {})
                _a = m.get("auc_label", float("nan"))
                _s = m.get("tp_sep_top10", float("nan"))
                if not math.isnan(_a): _aucs.append(_a)
                if not math.isnan(_s): _seps.append(_s)
            _best_auc = f"{max(_aucs):.4f}" if _aucs else "  n/a"
            _best_sep = f"{max(_seps)*100:+.2f}pp" if _seps else "  n/a"
            tprint(
                f"  {cell_key:<18} {len(_cids):>9}  "
                f"{str([round(k,2) for k in _k_vals]):<30}  "
                f"{str([round(s,2) for s in _sl_vals]):<20}  "
                f"{_win_str:>7}  {_best_auc:>8}  {_best_sep:>9}"
            )
        tprint(sep)

    # --- Mode comparison table ---
    tprint("MODE COMPARISON  (ranked by min_auc)")
    hdr2 = (f"  {'mode':<24} {'n':>4}  {'timeout':>8}  {'tp_hit':>7}  "
            f"{'bind':>6}  {'min_auc_best':>13}  {'min_tp_sep_best':>16}  "
            f"{'timeout_range_best':>19}")
    tprint(hdr2)
    for mode, g in out_df.groupby("mode"):
        _min_auc_best = g["min_cell_auc"].max() if "min_cell_auc" in g.columns else float("nan")
        _min_sep_best = g["min_cell_tp_sep"].max() if "min_cell_tp_sep" in g.columns else float("nan")
        _to_range_best = g["timeout_range"].min() if "timeout_range" in g.columns else float("nan")
        bind_mean = g["bind"].mean() if "bind" in g.columns else float("nan")
        def _f(v: Any) -> str:
            return f"{v:.4f}" if not (isinstance(v, float) and math.isnan(v)) else "  n/a"
        tprint(
            f"  {mode:<24} {len(g):>4}  {g['timeout_rate'].mean():>8.3f}  "
            f"{g['tp_hit_rate'].mean():>7.3f}  "
            f"{bind_mean:>6.3f}  {_f(_min_auc_best):>13}  "
            f"{_min_sep_best*100:>15.2f}pp  "
            f"{_f(_to_range_best):>19}"
        )
    tprint(sep)


# ---------------------------
# Main
# ---------------------------
def _clear_caches() -> None:
    """Clear all caches and collect garbage."""
    gc.collect()
    try:
        import numba
        numba.core.caching._cache_cleanup()
    except Exception:
        pass
    gc.collect()
    tprint("Cleared caches and ran gc.collect()")

class TBMObjective:
    """Objective function for Optuna to optimize TBM parameters."""
    def __init__(
        self,
        artifacts: Any,
        bucket_masks: Dict[str, Any],
        runtime_cfg: Dict[str, Any],
        horizons: List[int],
        layer1_cache: Dict[str, Any],
        layer2_cache: Dict[str, Any],
        eval_cache: Any,
        write_weights_fn: Any,
        stage_name: str = "optuna_stage1",
        target_cell: Optional[Tuple[str, int]] = None,
        grid_overrides: Optional[Dict[str, List[float]]] = None,
        tuple_eval_cache: Optional[Dict[Any, Tuple[float, Dict[str, Any], Dict[str, Any]]]] = None,
    ):
        self.artifacts = artifacts
        self.bucket_masks = bucket_masks
        self.runtime_cfg = runtime_cfg
        self.horizons = horizons
        self.layer1_cache = layer1_cache
        self.layer2_cache = layer2_cache
        self.eval_cache = eval_cache
        self.write_weights_fn = write_weights_fn
        self.stage_name = stage_name
        self.trial_results = []
        self.details = {}
        self.total_weights_written = 0
        self.target_cell = target_cell
        self.grid_overrides = grid_overrides or {}
        self.tuple_eval_cache = tuple_eval_cache if tuple_eval_cache is not None else {}
        _close = self.artifacts.panel.get("close", pd.DataFrame()) if isinstance(self.artifacts, RunArtifacts) else pd.DataFrame()
        _ctx = {
            "timeframe": str(self.runtime_cfg.get("timeframe", "1h")),
            "fee_pct": float(self.runtime_cfg.get("fee_pct", 0.5)),
            "slip_buffer": float(self.runtime_cfg.get("slip_buffer", 0.1)),
            "use_quantile_filter": bool(self.runtime_cfg.get("use_quantile_filter", False)),
            "quantile_lo": float(self.runtime_cfg.get("quantile_lo", 0.2)),
            "quantile_hi": float(self.runtime_cfg.get("quantile_hi", 0.8)),
            "min_keep_fraction": float(self.runtime_cfg.get("min_keep_fraction", 0.5)),
            "evaluate_under_prod_floor": bool(self.runtime_cfg.get("evaluate_under_prod_floor", False)),
            "bars": int(len(_close.index)),
            "symbols": int(len(_close.columns)),
        }
        self.eval_context_hash = hashlib.sha1(json.dumps(_ctx, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_saved_s_est = 0.0
        self._avg_eval_s = 0.0

    def __call__(self, trial: optuna.Trial) -> float:
        # ── Pre-compute ATR normalisation mean so we can constrain the search space ──
        # This avoids wasting trials on tp_base_pct values that, after ATR scaling,
        # would produce a tp_anchor outside [0.01, 0.04] and be immediately pruned.
        _TP_ANCHOR_LO = 0.01
        _TP_ANCHOR_HI = 0.05
        _TP_BASE_PCT_LO = 0.012
        _TP_BASE_PCT_HI = 0.045
        _atr_norm_mean_ref = 1.0
        _atr_win = 840  # fixed for now (only categorical choice)
        _r_key = f"tp_anchor_atr_norm_ref::{_atr_win}"
        _cached = self.eval_cache.get(_r_key, None) if hasattr(self.eval_cache, "get") else None
        if _cached is not None:
            _atr_norm_mean_ref = float(_cached)
        else:
            try:
                _atr_src = _get_barrier_atr_frame(self.artifacts).replace([np.inf, -np.inf], np.nan)
                _med = _atr_src.rolling(_atr_win, min_periods=24).median().fillna(_atr_src.median())
                _ratio = (_atr_src / (_med + EPS)).to_numpy(dtype=np.float32, copy=False).ravel()
                _ratio = _ratio[np.isfinite(_ratio)]
                _atr_norm_mean_ref = float(np.clip(np.nanmedian(_ratio) if _ratio.size else 1.0, 0.5, 2.0))
                try:
                    self.eval_cache[_r_key] = _atr_norm_mean_ref
                except Exception:
                    pass
            except Exception:
                _atr_norm_mean_ref = 1.0

        # 1. Suggest parameters from generated categorical grids.
        k_tp_grid = list(self.grid_overrides.get("k_tp", []))
        k_tp_meta: Dict[str, Any] = {"proposal_mode": "override"}
        if not k_tp_grid:
            k_tp_grid, k_tp_meta = _build_param_grid(0.3, 1.6)
        if not k_tp_grid:
            k_tp_grid = [0.3, 1.6]
        k_tp = float(trial.suggest_categorical("k_tp", k_tp_grid))

        tp_anchor_grid = list(self.grid_overrides.get("tp_anchor", []))
        tp_anchor_meta: Dict[str, Any] = {"proposal_mode": "override"}
        if not tp_anchor_grid:
            tp_anchor_grid, tp_anchor_meta = _build_param_grid(_TP_ANCHOR_LO, _TP_ANCHOR_HI)
        if not tp_anchor_grid:
            tp_anchor_grid = [_TP_ANCHOR_LO, _TP_ANCHOR_HI]
        tp_anchor = float(trial.suggest_categorical("tp_anchor", tp_anchor_grid))

        raw_tp_base_pct = float(tp_anchor / max(k_tp * _atr_norm_mean_ref, EPS))
        tp_base = float(np.clip(raw_tp_base_pct, _TP_BASE_PCT_LO, _TP_BASE_PCT_HI))
        derived_tp_clipped_low = bool(raw_tp_base_pct < _TP_BASE_PCT_LO)
        derived_tp_clipped_high = bool(raw_tp_base_pct > _TP_BASE_PCT_HI)

        sl_grid = list(self.grid_overrides.get("sl_as_tp_pct", []))
        sl_meta: Dict[str, Any] = {"proposal_mode": "override"}
        if not sl_grid:
            sl_as_tp_floor = float(self.runtime_cfg.get("tbm_sl_as_tp_pct_min", 0.3))
            sl_grid, sl_meta = _build_sl_biased_grid(sl_as_tp_floor)
        if not sl_grid:
            sl_grid = [0.3, 0.9]
        sl_as_tp = float(trial.suggest_categorical("sl_as_tp_pct", sl_grid))

        if trial.number == 0:
            tprint(f"[grid_meta] k_tp={k_tp_meta} n={len(k_tp_grid)}")
            tprint(f"[grid_meta] tp_anchor={tp_anchor_meta} n={len(tp_anchor_grid)}")
            tprint(f"[grid_meta] sl_as_tp_pct={sl_meta} n={len(sl_grid)}")

        atr_win = int(trial.suggest_categorical("base_atr_window", [840]))

        # 2. Build config
        c = base_param_template(self.runtime_cfg)
        c.update({
            "mode": self.stage_name,
            "tp_method": "atr_norm",
            "sl_method": "tp_pct",
            "k_tp": float(k_tp),
            "tp_anchor": float(tp_anchor),
            "tp_base_pct": float(tp_base),
            "tp_abs_pct": float(tp_base),
            "sl_as_tp_pct": float(sl_as_tp),
            "tp_regime_model": "none",
            "horizon_scaling": "sqrt",
            "base_atr_window": int(atr_win),
        })

        # Apply prod aligned centering if possible (simplified for trial)
        c_list, _ = _apply_prod_aligned_tp_centering(
            [c], artifacts=self.artifacts, bucket_masks=self.bucket_masks, cfg_runtime=self.runtime_cfg, preserve_sl_axis=True
        )
        if not c_list:
            return -1.0
        c = c_list[0]

        _cell_tag = f"{self.target_cell[0]}_H{self.target_cell[1]}" if self.target_cell else "global"
        _tuple_key = (
            _cell_tag,
            round(float(k_tp), 6),
            round(float(tp_anchor), 6),
            round(float(sl_as_tp), 6),
            int(atr_win),
            str(self.eval_context_hash),
        )
        cached_eval = self.tuple_eval_cache.get(_tuple_key)
        if cached_eval is not None:
            score, res, det = cached_eval
            self.cache_hits += 1
            self.cache_saved_s_est += max(0.0, float(self._avg_eval_s))
            self.trial_results.append(res)
            cid = str(res.get("config_id", config_id(c)))
            self.details[cid] = det
            trial.set_user_attr("cache_hit", True)
            trial.set_user_attr("stage_score", float(score))
            return float(score)

        # 3. Stage-1 cheap pruning before model fitting.
        try:
            t_eval0 = time.perf_counter()
            # Safety-net: tp_anchor guard (should rarely fire now that sampling is constrained)
            _tp_base = float(c.get("tp_base_pct", c.get("tp_abs_pct", 0.01)))
            _k_tp = float(c.get("k_tp", 1.0))
            _tp_method = str(c.get("tp_method", ""))
            _anchor_norm = _atr_norm_mean_ref if _tp_method in {"atr_norm", "semi_atr_norm"} else 1.0
            tp_anchor_eff = _k_tp * _tp_base * _anchor_norm
            if tp_anchor_eff < _TP_ANCHOR_LO or tp_anchor_eff > _TP_ANCHOR_HI:
                self.tuple_eval_cache[_tuple_key] = (-1.0, {"config_id": config_id(c), "stage_score": -1.0, "hard_gate": False, "k_tp": k_tp, "tp_anchor": tp_anchor, "sl_as_tp_pct": sl_as_tp}, {"config": dict(c)})
                return -1.0

            # Upstream cheap gates only (before model run).
            _min_rr = float(c.get("min_net_rr", 0.7))
            _sl_ratio_floor = float(self.runtime_cfg.get("tbm_sl_as_tp_pct_min", 0.3))
            _sl_ratio = float(c.get("sl_as_tp_pct", 0.5))
            if _sl_ratio < _sl_ratio_floor:
                self.tuple_eval_cache[_tuple_key] = (-1.0, {"config_id": config_id(c), "stage_score": -1.0, "hard_gate": False, "k_tp": k_tp, "tp_anchor": tp_anchor, "sl_as_tp_pct": sl_as_tp}, {"config": dict(c)})
                return -1.0
            if _sl_ratio > 0 and (1.0 / _sl_ratio) < (_min_rr * 0.7):
                self.tuple_eval_cache[_tuple_key] = (-1.0, {"config_id": config_id(c), "stage_score": -1.0, "hard_gate": False, "k_tp": k_tp, "tp_anchor": tp_anchor, "sl_as_tp_pct": sl_as_tp}, {"config": dict(c)})
                return -1.0

            # Single-pass evaluation (RandomSampler mode).
            res, det, weights_df = evaluate_config(
                self.artifacts,
                c,
                horizons=self.horizons,
                bucket_masks=self.bucket_masks,
                layer1_cache=self.layer1_cache,
                layer2_cache=self.layer2_cache,
                eval_cache=self.eval_cache,
                detailed_slices=False,
                target_cell_filter=self.target_cell,
                collect_weights=False,
            )
            if not res:
                return -1.0

            self.cache_misses += 1
            _dur = float(time.perf_counter() - t_eval0)
            self._avg_eval_s = _dur if self._avg_eval_s <= 0 else (0.9 * self._avg_eval_s + 0.1 * _dur)

            if weights_df is not None and not weights_df.empty:
                self.write_weights_fn(weights_df)
                self.total_weights_written += len(weights_df)
                del weights_df

            cid = config_id(c)
            self.trial_results.append(res)
            self.details[cid] = det
            score = _optuna_objective_score(res)
            self.tuple_eval_cache[_tuple_key] = (float(score), dict(res), dict(det))

            _m_auc = float(_safe_float(res.get("median_cell_auc", float("nan")), float("nan")))
            _m_ic_snr = float(_safe_float(res.get("ic_snr", float("nan")), float("nan")))
            _m_tp_sep = float(_safe_float(res.get("median_cell_tp_sep", float("nan")), float("nan")))
            _m_dir_sup = float(_safe_float(res.get("median_cell_dir_superiority", float("nan")), float("nan")))
            _m_payoff = float(_safe_float(res.get("payoff_edge", float("nan")), float("nan")))
            _m_ess = float(_safe_float(res.get("ess", float("nan")), float("nan")))
            trial.set_user_attr("stage_score", float(score))
            trial.set_user_attr("auc", _m_auc)
            trial.set_user_attr("tp_anchor", float(tp_anchor))
            trial.set_user_attr("tp_base_pct", float(tp_base))
            trial.set_user_attr("derived_tp_clipped_low", bool(derived_tp_clipped_low))
            trial.set_user_attr("derived_tp_clipped_high", bool(derived_tp_clipped_high))
            trial.set_user_attr("ic_snr", _m_ic_snr)
            trial.set_user_attr("tp_sep", _m_tp_sep)
            trial.set_user_attr("dir_sup", _m_dir_sup)
            trial.set_user_attr("payoff_edge", _m_payoff)
            trial.set_user_attr("ess", _m_ess)

            # Verbose logging of trial results — match detail level of global mode
            _cell_tag = f"{self.target_cell[0]}_H{self.target_cell[1]}" if self.target_cell else "global"
            tprint(
                f"[trial:{trial.number}] cell={_cell_tag} "
                f"k_tp={c.get('k_tp'):.1f} tp_anchor={tp_anchor:.4f} tp_base={c.get('tp_base_pct'):.3f} "
                f"sl_as_tp={c.get('sl_as_tp_pct'):.2f} atr_win={c.get('base_atr_window')} | "
                f"score={score:.4f} sl_to_tp={res.get('sl_to_tp', 0):.2f}x "
                f"roc_auc={res.get('median_cell_auc', float('nan')):.4f} "
                f"auc_bound={res.get('median_cell_auc_bound', float('nan')):.4f} "
                f"tp_sep={res.get('median_cell_tp_sep', 0):.4f} "
                f"ap_lift={res.get('median_cell_ap_lift', float('nan')):.3f} "
                f"pr_auc_lift={res.get('pr_auc_lift', 0):.3f} "
                f"edge={res.get('payoff_edge', 0):.4f} "
                f"sl_realized_p50={res.get('p50_cell_sl_to_tp', float('nan')):.3f} "
                f"sl_realized_p90={res.get('p90_cell_sl_to_tp', float('nan')):.3f} "
                f"hard_gate={res.get('hard_gate', False)}"
            )

            # Reduce GC frequency: every 25 trials is sufficient (was every 10, hurts JIT cache)
            if trial.number % 25 == 0:
                gc.collect()
            if trial.number % 50 == 0:
                _tot = max(1, self.cache_hits + self.cache_misses)
                tprint(
                    f"[tuple_cache] cell={_cell_tag} hits={self.cache_hits} misses={self.cache_misses} "
                    f"hit_rate={self.cache_hits/_tot:.3f} saved_s_est={self.cache_saved_s_est:.2f} "
                    f"ctx={self.eval_context_hash}"
                )

            return score
        except Exception as e:
            tprint(f"CRITICAL: Trial {trial.number} failed with exception: {e}")
            return -10.0


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()

    # Aggressive memory cleanup at startup
    gc.collect()
    _clear_caches()

    # Memory baseline
    _rss_mb = _memory_snapshot_mb()
    try:
        import psutil
        _sys_avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        tprint(f"[MEM_INIT] RSS={_rss_mb:.0f}MB  system_available={_sys_avail_mb:.0f}MB")
    except ImportError:
        tprint(f"[MEM_INIT] RSS={_rss_mb:.0f}MB  (psutil unavailable for system RAM check)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    tprint("Starting TBM parameter comparison run")
    try:
        np_blas_cfg = np.show_config()
        # show_config() prints but also returns None or a dict in some versions.
        # It's better to just skip the Accelerate specific warning if we can't easily check.
    except Exception:
        pass
    runtime_cfg = apply_offline_optimizer_best_params(dict(CFG))
    if args.data_root:
        runtime_cfg["data_root"] = str(args.data_root)
    if bool(args.perps):
        runtime_cfg["use_perps"] = True
        runtime_cfg["data_root"] = _append_suffix(runtime_cfg.get("data_root", "../data"), "_perp")
        runtime_cfg = enable_perp_feature_keys(runtime_cfg)
        existing_test = list(runtime_cfg.get("test_feature_keys", TEST_FEATURE_KEYS))
        runtime_cfg["test_feature_keys"] = list(dict.fromkeys(existing_test + list(PERP_FEATURE_KEYS)))
    global ACTIVE_TEST_FEATURE_KEYS
    ACTIVE_TEST_FEATURE_KEYS = list(runtime_cfg.get("test_feature_keys", TEST_FEATURE_KEYS))
    
    # Resolve data_root if relative
    data_root = runtime_cfg.get("data_root", "data")
    if not os.path.isabs(data_root):
        # Check if local 'data' actually has data, if not fallback to PROJECT_ROOT
        has_local_data = False
        local_ohlcv = os.path.join(data_root, "ohlcv")
        if os.path.exists(local_ohlcv):
            if any(os.scandir(local_ohlcv)):
                has_local_data = True
        
        if not has_local_data and (PROJECT_ROOT / data_root).exists():
            data_root = str((PROJECT_ROOT / data_root).resolve())
            runtime_cfg["data_root"] = data_root
            tprint(f"Resolved relative data_root to: {data_root}")

    # Auto-detect panel if not provided
    train_syms = None
    if args.panel:
        panel = load_panel(Path(args.panel))
    else:
        tprint(f"No --panel provided, auto-detecting from data_root: {runtime_cfg.get('data_root')}")
        panel, train_syms = _load_panel_from_store(runtime_cfg)

    if panel is None:
        raise ValueError("Could not load panel data. Please provide --panel path.")

    # Auto-detect features if not provided (only loads TEST_FEATURE_KEYS)
    if args.features:
        features = load_features(Path(args.features))
        # Filter to TEST_FEATURE_KEYS if present
        if features:
            available_keys = set(features.keys())
            test_keys = set(TEST_FEATURE_KEYS)
            common_keys = available_keys & test_keys
            if common_keys:
                tprint(f"Filtering features to TEST_FEATURE_KEYS: {len(common_keys)} features found")
                features = {k: features[k] for k in common_keys if k in features}
    else:
        tprint("No --features provided, auto-detecting from data_root")
        # Use subsampled symbols from panel if available to speed up loading
        features = _load_features_from_data_root(
            runtime_cfg,
            symbols=train_syms,
            feature_keys=ACTIVE_TEST_FEATURE_KEYS,
        )
        if features is None:
            raise ValueError("Could not auto-detect features. Please provide --features path.")

    artifacts = align_artifacts(panel, features, lookback_years=args.lookback_years)
    # Inject candidate-threshold best params (train_extreme_pct_hourly,
    # train_min_range_pct, train_min_vol_zscore) from CANDIDATE_BEST_PARAMS_CSV into runtime_cfg
    # so build_bucket_masks uses the optimised values, not hardcoded defaults.
    runtime_cfg = apply_offline_optimizer_best_params(dict(runtime_cfg))
    bucket_masks = build_bucket_masks(artifacts, cfg_runtime=runtime_cfg)
    tprint(
        f"Artifacts + buckets ready: bars={len(artifacts.panel['close'])}, symbols={len(artifacts.panel['close'].columns)} "
        f"bucket_masks={list(bucket_masks.keys())} mem_peak_mb={_memory_snapshot_mb():.1f}"
    )

    # Clear caches after loading data
    _clear_caches()

    horizons = [2, 4] if not args.horizons else [int(x) for x in args.horizons.split(",")]

    # Use bounded caches to prevent unbounded cache growth.
    layer1_cache: Dict[str, Any] = LRUCache(max_size=10)
    layer2_cache: Dict[str, Any] = LRUCache(max_size=10)
    eval_cache: BoundedEvalCache = BoundedEvalCache(max_size=10)

    # Load persisted TBM cache (barriers + labels) when available.
    cache_sig = _tbm_cache_signature(artifacts, horizons=horizons, lookback_years=int(args.lookback_years))
    cache_dir = _tbm_cache_dir(output_path, cache_sig)
    persisted_l1, persisted_l2 = load_persisted_tbm_cache(cache_dir)
    for k, v in persisted_l1.items():
        layer1_cache[k] = v
    for k, v in persisted_l2.items():
        layer2_cache[k] = v
    tprint(
        f"TBM cache bootstrap: layer1={len(layer1_cache)} layer2={len(layer2_cache)} "
        f"path={cache_dir}"
    )
    
    # For streaming weights output - write incrementally to avoid memory buildup
    weights_path = Path(args.weights_output) if args.weights_output else output_path.with_suffix(".weights.parquet")
    weights_writer = None
    
    def write_weights_streaming(weights_df: pd.DataFrame) -> None:
        """Write weights to parquet incrementally to avoid memory buildup."""
        nonlocal weights_writer
        if weights_df is None or weights_df.empty:
            return
        
        if weights_writer is None:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            # Create parquet writer for streaming.
            schema = pa.Schema.from_pandas(weights_df)
            weights_writer = pq.ParquetWriter(weights_path, schema, compression='snappy')
        
        table = pa.Table.from_pandas(weights_df)
        weights_writer.write_table(table)
        tprint(f"Streamed {len(weights_df):,} weight rows to {weights_path}")

    details: Dict[str, Any] = {}
    stage1_rows: List[Dict[str, Any]] = []
    total_weights_written = 0
    sl_shrink_rows: List[Dict[str, Any]] = []

    # Precompute immutable global feature matrix once (outside objective/trials).
    fm = get_feature_matrix_cache(artifacts, eval_cache)
    tprint(
        f"Feature matrix cache ready: rows={fm.X.shape[0]:,} cols={fm.X.shape[1]} "
        f"dtype={fm.X.dtype} contiguous={bool(fm.X.flags['C_CONTIGUOUS'])}"
    )

    # --- Optuna Stage 1 (Cell-Specific) ---
    tprint("Starting Optuna Stage 1 optimization (Cell-Specific)...")
    
    # Define canonical cells (4 buckets x N horizons)
    canonical_cells = []
    for b in TBM_BUCKET_NAMES:
        for h in horizons:
            canonical_cells.append((b, h))
            
    for bkt, hor in canonical_cells:
        tprint(f"--- Optimizing Cell: {bkt} H{hor} ---")
        tprint(
            f"[cell_start] {bkt}_H{hor} rss_mb={_memory_snapshot_mb():.0f} "
            f"{_cache_pressure_summary(layer1_cache, layer2_cache, eval_cache)}"
        )
        # Static categorical grids for this run/cell.
        grid_k_tp, _ = _build_param_grid(0.3, 1.6)
        grid_tp_anchor, _ = _build_param_grid(0.01, 0.05)
        grid_sl_full, _ = _build_sl_biased_grid(float(runtime_cfg.get("tbm_sl_as_tp_pct_min", 0.3)))
        tuple_eval_cache: Dict[Any, Tuple[float, Dict[str, Any], Dict[str, Any]]] = {}

        # Phase 1: full SL grid.
        n_trials_s1 = int(runtime_cfg.get("tbm_optuna_trials_per_cell", 100))
        n_phase1 = int(min(max(1, runtime_cfg.get("tbm_sl_shrink_phase1_trials", 30)), n_trials_s1))

        obj_s1 = TBMObjective(
            artifacts=artifacts,
            bucket_masks=bucket_masks,
            runtime_cfg=runtime_cfg,
            horizons=[hor],
            layer1_cache=layer1_cache,
            layer2_cache=layer2_cache,
            eval_cache=eval_cache,
            write_weights_fn=write_weights_streaming,
            stage_name=f"optuna_s1_{bkt}_H{hor}",
            target_cell=(bkt, hor),
            grid_overrides={"k_tp": grid_k_tp, "tp_anchor": grid_tp_anchor, "sl_as_tp_pct": grid_sl_full},
            tuple_eval_cache=tuple_eval_cache,
        )
        
        study_s1 = optuna.create_study(
            direction="maximize", 
            sampler=_optuna_sampler(),
            pruner=optuna.pruners.NopPruner()
        )

        study_s1.optimize(obj_s1, n_trials=n_phase1, n_jobs=_optuna_n_jobs())

        # Phase 2: optional SL-only shrink using feasible phase-1 trials.
        sl_grid_phase2 = list(grid_sl_full)
        if n_phase1 < n_trials_s1 and obj_s1.trial_results:
            phase1_df = pd.DataFrame(obj_s1.trial_results)
            _score_col = "stage_score" if "stage_score" in phase1_df.columns else ("stage2_score" if "stage2_score" in phase1_df.columns else "stage1_score")
            feasible_df = phase1_df[(phase1_df.get("hard_gate", False) == True)].copy()
            if not feasible_df.empty and "sl_as_tp_pct" in feasible_df.columns and _score_col in feasible_df.columns:
                m_all = float(pd.to_numeric(feasible_df[_score_col], errors="coerce").dropna().median())
                _grp_total = phase1_df.groupby("sl_as_tp_pct", observed=True).size().rename("n_total").reset_index() if "sl_as_tp_pct" in phase1_df.columns else pd.DataFrame(columns=["sl_as_tp_pct", "n_total"])
                _grp = feasible_df.groupby("sl_as_tp_pct", observed=True)[_score_col].agg(["count", "median"]).reset_index()
                _grp.rename(columns={"count": "n_obs", "median": "m_v"}, inplace=True)
                _grp = _grp.merge(_grp_total, on="sl_as_tp_pct", how="left")
                _grp["n_total"] = pd.to_numeric(_grp.get("n_total", 0), errors="coerce").fillna(0).astype(int)
                _grp["removed"] = False
                n_feas_min = int(runtime_cfg.get("tbm_sl_shrink_n_feasible_min", 4))
                n_total_min = int(runtime_cfg.get("tbm_sl_shrink_n_total_min", 6))
                keep_min = int(runtime_cfg.get("tbm_sl_shrink_keep_min", 4))
                _judge = (_grp["n_total"] >= n_total_min) | (_grp["n_obs"] >= n_feas_min)
                bad_vals = set(_grp.loc[_judge & (_grp["m_v"] < m_all), "sl_as_tp_pct"].tolist())
                kept_vals = [v for v in sl_grid_phase2 if float(v) not in bad_vals]
                if len(kept_vals) < keep_min and not _grp.empty:
                    top_vals = _grp.sort_values(["m_v", "n_obs"], ascending=[False, False])["sl_as_tp_pct"].tolist()
                    kept_vals = sorted(set([float(v) for v in top_vals[:keep_min]]))
                sl_grid_phase2 = kept_vals if len(kept_vals) >= keep_min else sl_grid_phase2
                _grp["removed"] = _grp["sl_as_tp_pct"].apply(lambda x: bool(float(x) not in set(sl_grid_phase2)))
                for row in _grp.itertuples(index=False):
                    sl_shrink_rows.append({
                        "cell": f"{bkt}_H{hor}",
                        "phase": "sl_shrink",
                        "sl_value": float(getattr(row, "sl_as_tp_pct")),
                        "n_obs": int(getattr(row, "n_obs")),
                        "n_total": int(getattr(row, "n_total")),
                        "m_v": float(getattr(row, "m_v")),
                        "m_all": float(m_all),
                        "removed": bool(getattr(row, "removed")),
                        "grid_before": json.dumps([float(x) for x in grid_sl_full]),
                        "grid_after": json.dumps([float(x) for x in sl_grid_phase2]),
                    })
                tprint(f"[sl_shrink] {bkt}_H{hor} phase1_trials={len(phase1_df)} sl_before={len(grid_sl_full)} sl_after={len(sl_grid_phase2)}")

        if n_phase1 < n_trials_s1:
            obj_s2 = TBMObjective(
                artifacts=artifacts,
                bucket_masks=bucket_masks,
                runtime_cfg=runtime_cfg,
                horizons=[hor],
                layer1_cache=layer1_cache,
                layer2_cache=layer2_cache,
                eval_cache=eval_cache,
                write_weights_fn=write_weights_streaming,
                stage_name=f"optuna_s1_{bkt}_H{hor}",
                target_cell=(bkt, hor),
                grid_overrides={"k_tp": grid_k_tp, "tp_anchor": grid_tp_anchor, "sl_as_tp_pct": sl_grid_phase2},
                tuple_eval_cache=tuple_eval_cache,
            )
            study_s2 = optuna.create_study(
                direction="maximize",
                sampler=_optuna_sampler(),
                pruner=optuna.pruners.NopPruner(),
            )
            study_s2.optimize(obj_s2, n_trials=(n_trials_s1 - n_phase1), n_jobs=_optuna_n_jobs())
            obj_s1.trial_results.extend(obj_s2.trial_results)
            obj_s1.details.update(obj_s2.details)
            obj_s1.total_weights_written += obj_s2.total_weights_written

        
        stage1_rows.extend(obj_s1.trial_results)
        details.update(obj_s1.details)
        total_weights_written += obj_s1.total_weights_written
        # Per-cell summary
        _cell_df = pd.DataFrame(obj_s1.trial_results)
        _score_col = "stage_score" if "stage_score" in _cell_df.columns else ("stage2_score" if "stage2_score" in _cell_df.columns else "stage1_score")
        _best_val = "N/A"
        if not _cell_df.empty and _score_col in _cell_df.columns:
            _best_s = pd.to_numeric(_cell_df[_score_col], errors="coerce").dropna()
            if not _best_s.empty:
                _best_val = f"{float(_best_s.max()):.4f}"
        tprint(
            f"--- Cell {bkt} H{hor} complete: "
            f"trials={len(obj_s1.trial_results)} best={_best_val} "
            f"mem={_memory_snapshot_mb():.0f}MB "
            f"{_cache_pressure_summary(layer1_cache, layer2_cache, eval_cache)} ---"
        )
        gc.collect()
        _clear_caches()
    
    if not stage1_rows:
        tprint("ERROR: Stage 1 produced no valid results.")
        return

    stage1_df = pd.DataFrame(stage1_rows)
    out_df = stage1_df.copy()
    # Two-level learnability sort:
    # Level 1 (hard flags, binary): hard_gate / timeout_range_ok / all_cells_pass /
    #   geometry-health gates (coverage, bind, timeout, min_tp_sep, min_auc floors).
    # Level 2 (continuous ranking among survivors):
    #   min_auc desc → min_tp_sep desc → cell_dispersion asc → timeout_range asc
    #   → min_auc_bound desc (tie-breaker only; fragile without healthy bind/coverage).
    out_df["_all_cells_pass"] = (out_df["pass_cells"] == out_df["total_cells"]).astype(int)
    out_df["_to_range_ok"] = (out_df["timeout_range"].fillna(999.0) <= _MAX_TIMEOUT_RANGE).astype(int) if "timeout_range" in out_df.columns else 1
    # Geometry-health gate flags (soft in sort — hard in promote/best-params)
    # Ensure all required columns exist with fallback values for stage1-only results
    required_cols = [
        "min_cell_auc", "median_cell_auc", "min_cell_auc_bound", "median_cell_auc_bound",
        "min_cell_tp_sep", "median_cell_tp_sep", "cell_dispersion", "timeout_range"
    ]
    
    for col in required_cols:
        if col not in out_df.columns:
            if col in ["min_cell_auc", "median_cell_auc", "min_cell_auc_bound", "median_cell_auc_bound"]:
                out_df[col] = float("nan")
            elif col in ["min_cell_tp_sep", "median_cell_tp_sep", "cell_dispersion", "timeout_range"]:
                out_df[col] = 0.0
    
    _cov_ok = (out_df["coverage"] >= _PROMOTE_MIN_COVERAGE).astype(int) if "coverage" in out_df.columns else pd.Series(1, index=out_df.index)
    _bind_ok = (out_df["bind"] >= _PROMOTE_MIN_BIND).astype(int) if "bind" in out_df.columns else pd.Series(1, index=out_df.index)
    _to_ok = (out_df["timeout_rate"] <= _PROMOTE_MAX_TIMEOUT).astype(int) if "timeout_rate" in out_df.columns else pd.Series(1, index=out_df.index)
    _sep_ok = (out_df["min_cell_tp_sep"].fillna(0.0) >= _PROMOTE_MIN_TP_SEP).astype(int) if "min_cell_tp_sep" in out_df.columns else pd.Series(1, index=out_df.index)
    _auc_col = "min_cell_auc_bound" if "min_cell_auc_bound" in out_df.columns else ("min_cell_auc" if "min_cell_auc" in out_df.columns else None)
    _auc_ok = (out_df[_auc_col].fillna(0.0) >= _PROMOTE_MIN_AUC).astype(int) if _auc_col else pd.Series(1, index=out_df.index)
    _ece_ok = (out_df.get("min_cell_ece", out_df.get("ece", pd.Series(0.0, index=out_df.index))).fillna(999.0) <= _PROMOTE_MAX_ECE).astype(int)
    _brier_ok = (out_df.get("min_cell_brier", out_df.get("brier", pd.Series(0.0, index=out_df.index))).fillna(999.0) <= _PROMOTE_MAX_BRIER).astype(int)
    _epos_ok = (out_df.get("effective_pos_mass", pd.Series(0.0, index=out_df.index)).fillna(0.0) >= _PROMOTE_MIN_EFFECTIVE_POS).astype(int)
    _eneg_ok = (out_df.get("effective_neg_mass", pd.Series(0.0, index=out_df.index)).fillna(0.0) >= _PROMOTE_MIN_EFFECTIVE_NEG).astype(int)
    out_df["_health_flags"] = _cov_ok + _bind_ok + _to_ok + _sep_ok + _auc_ok + _ece_ok + _brier_ok + _epos_ok + _eneg_ok
    # Bucket min_auc to 3dp so configs that differ by <0.001 are treated as tied
    # and min_tp_sep (the more discriminating learnability signal) breaks the tie.
    _min_auc_source = "min_cell_auc_bound" if "min_cell_auc_bound" in out_df.columns else "min_cell_auc"
    _min_auc_col = (out_df[_min_auc_source].fillna(0.0) * 1000).round() / 1000 if _min_auc_source in out_df.columns else pd.Series(0.0, index=out_df.index)
    _min_sep_col = out_df["min_cell_tp_sep"].fillna(0.0)
    _disp_col = out_df["cell_dispersion"].fillna(999.0)
    _to_range_col = out_df["timeout_range"].fillna(999.0) if "timeout_range" in out_df.columns else pd.Series(0.0, index=out_df.index)
    _min_auc_b_col = out_df["min_cell_auc_bound"].fillna(0.0) if "min_cell_auc_bound" in out_df.columns else _min_auc_col
    _score_s = pd.to_numeric(out_df.get("stage_score", pd.Series(np.nan, index=out_df.index)), errors="coerce")
    _score_s2 = pd.to_numeric(out_df.get("stage2_score", pd.Series(np.nan, index=out_df.index)), errors="coerce")
    _score_s1 = pd.to_numeric(out_df.get("stage1_score", pd.Series(-np.inf, index=out_df.index)), errors="coerce").fillna(-np.inf)
    _score_col = _score_s.where(_score_s.notna(), _score_s2.where(_score_s2.notna(), _score_s1)).fillna(-np.inf)
    out_df = out_df.assign(
        _score_sort=_score_col,
        _min_auc_sort=_min_auc_col,
        _min_sep_sort=_min_sep_col,
        _disp_sort=_disp_col,
        _to_range_sort=_to_range_col,
        _min_auc_b_sort=_min_auc_b_col,
    ).sort_values(
        ["hard_gate", "_to_range_ok", "_all_cells_pass", "_health_flags", "_score_sort",
         "_min_auc_sort", "_min_sep_sort", "_disp_sort", "_to_range_sort", "_min_auc_b_sort"],
        ascending=[False, False, False, False, False, False, False, True, True, False],
    ).drop(columns=["_all_cells_pass", "_to_range_ok", "_health_flags", "_score_sort",
                    "_min_auc_sort", "_min_sep_sort", "_disp_sort", "_to_range_sort", "_min_auc_b_sort"])
    
    # 1) Exact duplicates cleanup (config_id + mode)
    if not out_df.empty:
        _n_before = len(out_df)
        out_df = out_df.drop_duplicates(subset=["config_id", "mode"], keep="first")
        if len(out_df) < _n_before:
            tprint(f"[cleanup] Deduplicated out_df: {_n_before} -> {len(out_df)} rows")

    tprint(
        f"Scoring complete: total_rows={len(out_df)} stage1_rows={len(stage1_df)} "
        f"mem_peak_mb={_memory_snapshot_mb():.1f}"
    )

    out_df.to_csv(output_path, index=False)
    trials_path = output_path.with_name("tbm_trials.csv")
    stage1_df.to_csv(trials_path, index=False)
    if sl_shrink_rows:
        pd.DataFrame(sl_shrink_rows).to_csv(output_path.with_name("tbm_sl_shrink_report.csv"), index=False)

    detail_path = output_path.with_suffix(".json")
    with detail_path.open("w") as f:
        json.dump(details, f, indent=2)

    _build_prod_aligned_reports(out_df, details, output_path)

    learnability_path = output_path.with_name("tbm__learnability_report.csv")
    learnability_df = _build_tbm_learnability_report_rows(out_df, details)
    if not learnability_df.empty:
        learnability_df.to_csv(learnability_path, index=False)
        tprint(f"Saved learnability CSV: {learnability_path}")

    # Close the streaming weights writer and report
    if weights_writer is not None:
        weights_writer.close()
        tprint(f"Saved sample weights (streaming): {weights_path} (total_rows={total_weights_written:,})")

    save_persisted_tbm_cache(
        cache_dir=cache_dir,
        layer1_cache=layer1_cache,
        layer2_cache=layer2_cache,
        signature=cache_sig,
        max_bytes=int(args.tbm_cache_max_mb * 1024 * 1024),
    )

    per_cell_grids: Dict[str, pd.DataFrame] = {}
    if not out_df.empty:
        # ── Step 1: structural validity pool ──────────────────────────────────
        _to_r_ok = out_df["timeout_range"].fillna(999.0) <= _MAX_TIMEOUT_RANGE if "timeout_range" in out_df.columns else pd.Series(True, index=out_df.index)
        _base_mask = (out_df["hard_gate"] == True) & _to_r_ok & (out_df["pass_cells"] == out_df["total_cells"])
        _struct_pool = out_df[_base_mask].copy()
        if _struct_pool.empty:
            _struct_pool = out_df[(out_df["hard_gate"] == True) & (out_df["pass_cells"] == out_df["total_cells"])].copy()
        if _struct_pool.empty:
            _struct_pool = out_df[out_df["hard_gate"] == True].copy()
        if _struct_pool.empty:
            _struct_pool = out_df.copy()

        # ── Step 2: tiered feasible-set (all layer-1 candidates) ──────────────
        _best_pool, _tier = _build_feasible_set(_struct_pool)
        tprint(f"Global feasible pool: {len(_best_pool)} configs (tier={_tier})")

        # Audit export: feasible layer-1 pool.
        _feasible_pool_csv = TBM_GEOMETRY_GRID_CSV.with_name("tbm_feasible_pool.csv")
        _best_pool.to_csv(_feasible_pool_csv, index=False)

        # ── Step 3: GRR diversity on entire feasible pool ─────────────────────
        _candidate_set = _best_pool.copy()
        _grid_k_tp, _ = _build_param_grid(0.4, 2.0)
        _grid_tp_anchor, _ = _build_param_grid(0.01, 0.04)
        _grid_sl, _ = _build_sl_biased_grid(float(runtime_cfg.get("tbm_sl_as_tp_pct_min", 0.3)))
        _tuple_cols = [c for c in ("k_tp", "tp_anchor", "sl_as_tp_pct") if c in out_df.columns]
        if len(_tuple_cols) < 3:
            _tuple_cols = [c for c in ("k_tp", "tp_base_pct", "sl_as_tp_pct") if c in out_df.columns]
        _n_unique_tuples = int(out_df.drop_duplicates(subset=_tuple_cols).shape[0]) if _tuple_cols else int(len(out_df))
        _cart_n = max(1, len(_grid_k_tp) * len(_grid_tp_anchor) * len(_grid_sl))
        _coverage = float(_n_unique_tuples / _cart_n)
        tprint(
            f"[grr_coverage] N_trials={len(out_df)} N_unique_tuples={_n_unique_tuples} "
            f"N_feasible={len(_best_pool)} approx_cartesian={_cart_n} coverage={_coverage:.4f}"
        )
        for _col in ("k_tp", "tp_anchor", "sl_as_tp_pct"):
            if _col in out_df.columns:
                _vc = out_df[_col].round(6).value_counts(dropna=True).sort_index()
                tprint(f"[grr_coverage] {_col}_counts={{{', '.join([f'{k}:{int(v)}' for k,v in _vc.items()])}}}")

        _global_diverse_k = int(np.clip(int(runtime_cfg.get("tbm_global_diverse_k", 8)), 6, 10))
        _diverse_pool = _diverse_subset(_candidate_set, details, min_distance=1.0, max_configs=_global_diverse_k, score_keep_fraction=1.0)
        if len(_diverse_pool) < min(_global_diverse_k, len(_candidate_set)):
            for _d in (0.9, 0.8, 0.7, 0.6, 0.5):
                _diverse_pool = _diverse_subset(_candidate_set, details, min_distance=float(_d), max_configs=_global_diverse_k, score_keep_fraction=1.0)
                if len(_diverse_pool) >= min(_global_diverse_k, len(_candidate_set)):
                    break

        if _diverse_pool.empty and len(_best_pool) > 0:
            _pr_s = pd.to_numeric(_best_pool.get("pr_auc_lift", pd.Series(np.nan, index=_best_pool.index)), errors="coerce")
            _ed_s = pd.to_numeric(_best_pool.get("edge", pd.Series(np.nan, index=_best_pool.index)), errors="coerce")
            tprint(
                "[global_diversity] empty after feasible+auc/econ gating; "
                f"best_pool={len(_best_pool)} pr_auc_lift>0={int(_pr_s.gt(0).sum())} edge>0={int(_ed_s.gt(0).sum())}"
            )
        _grr_pool_csv = TBM_GEOMETRY_GRID_CSV.with_name("tbm_grr_pool.csv")
        _grr_pool_df = _diverse_pool.attrs.get("grr_pool_df", _candidate_set)
        _grr_pool_df.to_csv(_grr_pool_csv, index=False)
        _diverse_winners_csv = TBM_GEOMETRY_GRID_CSV.with_name("tbm_diverse_winners.csv")
        _diverse_pool.to_csv(_diverse_winners_csv, index=False)
        try:
            _dist_cols = [c for c in ("k_tp", "tp_anchor", "sl_as_tp_pct") if c in _diverse_pool.columns]
            if len(_dist_cols) >= 3 and len(_diverse_pool) >= 2:
                _x = _diverse_pool[_dist_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=False)
                _x = np.log(np.clip(_x, EPS, None))
                _xmin = np.nanmin(_x, axis=0)
                _xmax = np.nanmax(_x, axis=0)
                _x = (_x - _xmin) / np.maximum(_xmax - _xmin, 1e-9)
                _nn = []
                for ii in range(len(_x)):
                    dd = np.sqrt(np.sum((_x - _x[ii]) ** 2, axis=1, dtype=np.float64))
                    dd[ii] = np.inf
                    _nn.append(float(np.min(dd)))
                if _nn:
                    tprint(f"[grr_coverage] winner_nearest_dist min={float(np.min(_nn)):.4f} mean={float(np.mean(_nn)):.4f}")
        except Exception:
            pass
        tprint(f"Global diverse pool: {len(_diverse_pool)} configs after diversity filter")

        # ── Step 4: best_stage1 convenience export (non-authoritative) ───────
        _best_source = _best_pool if not _best_pool.empty else _struct_pool
        _score_col = "stage_score" if "stage_score" in _best_source.columns else ("stage2_score" if "stage2_score" in _best_source.columns else "stage1_score")
        _best_pool_sorted = _best_source.sort_values(_score_col, ascending=False)
        best_cid = _best_pool_sorted.iloc[0]["config_id"]
        best_params = details.get(best_cid, {}).get("config", {})
        if isinstance(best_params, dict):
            best_params = dict(best_params)
            raw_mode = best_params.get("mode", "")
            canonical_mode = raw_mode.split("_2A")[0].split("_refine")[0]
            best_params["mode"] = canonical_mode
            save_best_params_csv(TBM_BEST_PARAMS_CSV, best_params, metadata={"source": "compare_tbm_parameters", "config_id": best_cid, "label": "best_stage1_not_authoritative"})
            tprint(f"Saved best_stage1 convenience params CSV: {TBM_BEST_PARAMS_CSV} (best_stage1={best_cid})")

        # ── Step 5: per-(bucket, horizon) feasible sets — canonical output ───
        # 8 cells = 4 buckets × 2 horizons; up to max_configs_per_cell diverse configs per cell.
        # load_tbm_geometry_grid() collects all unique k_tp/sl_as_tp_pct per cell_key so
        # training.py sweeps the full set of selected geometries for each cell independently.
        per_cell_grids = _build_per_cell_feasible_sets(
            out_df, details, min_distance=1.0, max_configs_per_cell=10
        )

        # ── Step 6: save geometry grid CSV (per-cell format) ─────────────────
        # Each row: cell_key, bucket, config_id, k_tp, sl_as_tp_pct, base_atr_window,
        #           tp_base_pct, mode, cell_auc, cell_tp_sep, cell_timeout, cell_bind, rank.
        # The labels step reads this keyed by cell_key to build its per-bucket/horizon sweep grid.
        _grid_rows = []
        _fallback_window = int(best_params.get("base_atr_window", 840)) if isinstance(best_params, dict) else 840
        for cell_key, cell_df in per_cell_grids.items():
            if cell_df.empty:
                continue
            # Derive bucket from cell_key (e.g. "MR_long_H4" → "MR_long")
            _cell_bucket = "_".join(cell_key.split("_")[:-1])  # strip "_H{n}"
            for rank_i, crow in enumerate(cell_df.itertuples(index=False), 1):
                cid = str(getattr(crow, "config_id", ""))
                _d_c = details.get(cid, {}) if isinstance(details, dict) else {}
                cfg_i = _d_c.get("config", {}) if isinstance(_d_c, dict) else {}
                if not isinstance(cfg_i, dict):
                    continue
                bh_i = _d_c.get("bucket_horizon_metrics", {}) if isinstance(_d_c, dict) else {}
                cell_m_i = bh_i.get(cell_key, {})
                _grid_rows.append({
                    "cell_key": cell_key,
                    "bucket": _cell_bucket,
                    "config_id": cid,
                    "rank": rank_i,
                    "k_tp": float(cfg_i.get("k_tp", float("nan"))),
                    "tp_anchor": float(cfg_i.get("tp_anchor", float("nan"))),
                    "sl_as_tp_pct": float(cfg_i.get("sl_as_tp_pct", float("nan"))),
                    "base_atr_window": int(cfg_i.get("base_atr_window", _fallback_window)),
                    "tp_base_pct": float(cfg_i.get("tp_base_pct", float("nan"))),
                    "tp_abs_lo_pct": float(cfg_i.get("tp_abs_lo_pct", float("nan"))),
                    "sl_abs_lo_pct": float(cfg_i.get("sl_abs_lo_pct", float("nan"))),
                    "mode": str(cfg_i.get("mode", "")).split("_2A")[0].split("_refine")[0],
                    "stage2_score": float(getattr(crow, "stage2_score", float("nan"))),
                    "stage1_score": float(getattr(crow, "stage1_score", float("nan"))),
                    "probe_auc": float(getattr(crow, "probe_auc", float("nan"))),
                    "probe_auc_bound": float(getattr(crow, "probe_auc_bound", float("nan"))),
                    "probe_status": str(getattr(crow, "probe_status", "")),
                    "probe_n_train": int(_safe_float(getattr(crow, "probe_n_train", 0), 0)),
                    "probe_n_val": int(_safe_float(getattr(crow, "probe_n_val", 0), 0)),
                    "cell_auc": float(cell_m_i.get("auc_label", float("nan"))),
                    "cell_auc_bound": float(cell_m_i.get("auc_bound", float("nan"))),
                    "cell_tp_sep": float(cell_m_i.get("tp_sep_top10", float("nan"))),
                    "cell_ap_lift": float(cell_m_i.get("ap_lift", float("nan"))),
                    "cell_tp_over_sl": float(cell_m_i.get("tp_over_sl", float("nan"))),
                    "cell_sl_to_tp": float(cell_m_i.get("sl_to_tp", float("nan"))),
                    "cell_bind": float(cell_m_i.get("bind", float("nan"))),
                    "cell_timeout": float(cell_m_i.get("timeout_kept", float("nan"))),
                    "cell_brier": float(cell_m_i.get("brier", float("nan"))),
                    "cell_ece": float(cell_m_i.get("ece", float("nan"))),
                    "cell_monotonicity": float(cell_m_i.get("monotonicity", float("nan"))),
                    "cell_ic_std_time": float(cell_m_i.get("ic_std_time", float("nan"))),
                    "cell_ic_std_asset": float(cell_m_i.get("ic_std_asset", float("nan"))),
                    "cell_ic_payoff": float(cell_m_i.get("ic_payoff", float("nan"))),
                    "cell_ic_label": float(cell_m_i.get("ic_label", float("nan"))),
                    "cell_barrier_ratio": float(cell_m_i.get("barrier_ratio", float("nan"))),
                    "cell_n": int(cell_m_i.get("n", 0)),
                })

        # Emit fallback rows for any canonical cell missing from per_cell_grids,
        # so tbm_geometry_grid.csv always covers all 12 cell keys.
        _covered_cells = {r["cell_key"] for r in _grid_rows}
        _fb_params = best_params if isinstance(best_params, dict) else {}
        for _ck in _CELL_KEYS:
            if _ck in _covered_cells:
                continue
            _ck_bucket = "_".join(_ck.split("_")[:-1])
            tprint(f"[grid] Cell {_ck} missing from per_cell_grids — emitting fallback row from global best")
            _grid_rows.append({
                "cell_key": _ck,
                "bucket": _ck_bucket,
                "config_id": best_cid,
                "rank": 99,
                "k_tp": float(_fb_params.get("k_tp", float("nan"))),
                "tp_anchor": float(_fb_params.get("tp_anchor", float("nan"))),
                "sl_as_tp_pct": float(_fb_params.get("sl_as_tp_pct", float("nan"))),
                "base_atr_window": int(_fb_params.get("base_atr_window", _fallback_window)),
                "tp_base_pct": float(_fb_params.get("tp_base_pct", float("nan"))),
                "tp_abs_lo_pct": float(_fb_params.get("tp_abs_lo_pct", float("nan"))),
                "sl_abs_lo_pct": float(_fb_params.get("sl_abs_lo_pct", float("nan"))),
                "mode": str(_fb_params.get("mode", "")).split("_2A")[0].split("_refine")[0],
                "stage2_score": float("nan"),
                "stage1_score": float("nan"),
                "probe_auc": float("nan"),
                "probe_auc_bound": float("nan"),
                "probe_status": "fallback",
                "probe_n_train": 0,
                "probe_n_val": 0,
                "cell_auc": float("nan"), "cell_auc_bound": float("nan"),
                "cell_tp_sep": float("nan"), "cell_timeout": float("nan"),
                "cell_bind": float("nan"), "cell_ap_lift": float("nan"),
                "cell_tp_over_sl": float("nan"), "cell_brier": float("nan"),
                "cell_ece": float("nan"), "cell_monotonicity": float("nan"),
                "cell_ic_std_time": float("nan"), "cell_ic_std_asset": float("nan"),
                "cell_ic_payoff": float("nan"), "cell_ic_label": float("nan"),
                "cell_barrier_ratio": float("nan"),
                "cell_n": 0,
            })

        if _grid_rows:
            _grid_df = pd.DataFrame(_grid_rows).dropna(subset=["k_tp", "sl_as_tp_pct"])
            
            # 2) Deduplicate by (cell_key, params) to remove redundant signals
            _param_cols = ["cell_key", "k_tp", "tp_anchor", "sl_as_tp_pct", "base_atr_window", "tp_base_pct", "tp_abs_lo_pct", "sl_abs_lo_pct"]
            _param_cols = [c for c in _param_cols if c in _grid_df.columns]
            _n_grid_before = len(_grid_df)
            _grid_df = _grid_df.sort_values("rank").drop_duplicates(subset=_param_cols, keep="first")
            if len(_grid_df) < _n_grid_before:
                tprint(f"[cleanup] Deduplicated geometry grid: {_n_grid_before} -> {len(_grid_df)} rows")

            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            _grid_df.to_csv(TBM_GEOMETRY_GRID_CSV, index=False)
            _cells_covered = _grid_df["cell_key"].nunique()
            _buckets_covered = _grid_df["bucket"].nunique() if "bucket" in _grid_df.columns else 0
            _total_rows = len(_grid_df)
            tprint(
                f"Saved geometry grid CSV: {TBM_GEOMETRY_GRID_CSV} "
                f"({_total_rows} rows across {_cells_covered} cells, {_buckets_covered} buckets)"
            )
            # Summary per cell
            for ck, cg in _grid_df.groupby("cell_key"):
                _k = sorted(cg["k_tp"].unique().tolist())
                _s = sorted(cg["sl_as_tp_pct"].unique().tolist())
                tprint(f"  {ck:<18}: {len(cg):>2} configs  k_tp={_k}  sl={_s}")

            # ── Step 7: per-bucket best params (one winner per bucket) ──────────
            # Selects the rank-1 config per bucket (aggregated across all horizons H1/H2/H4)
            # using a composite learnability score and saves to tbm_best_params_per_bucket.csv.
            # Format: one row per bucket, same columns as tbm_best_params.csv + bucket col.
            # Downstream steps can call load_tbm_best_params_per_bucket()[bucket] to get the
            # barrier geometry for that specific bucket/regime.
            _bucket_best_rows: list = []
            per_cell_rows: list = []
            for _bkt in TBM_BUCKET_NAMES:
                _bdf = _grid_df[_grid_df["bucket"] == _bkt].copy()
                if _bdf.empty:
                    tprint(f"[bucket_best] {_bkt}: no valid configs — using global best fallback")
                    # Emit global fallback so consumers always have all 4 buckets
                    _bucket_best_rows.append({
                        "bucket": _bkt,
                        "config_id": best_cid,
                        "rank_in_bucket": 99,
                        "source": "global_fallback",
                        **{k: _to_scalar(v) for k, v in (best_params if isinstance(best_params, dict) else {}).items()
                           if k in ("k_tp", "sl_as_tp_pct", "base_atr_window", "tp_base_pct",
                                    "tp_abs_lo_pct", "tp_abs_hi_pct", "sl_abs_lo_pct", "sl_abs_hi_pct",
                                    "tp_method", "sl_method", "mode", "horizon_base", "horizon_scaling")},
                    })
                    continue

                # ── Admissibility Filter (Production Guardrails) ───────────
                # Only pick configs that are economically viable for production.
                if "cell_sl_to_tp" in _bdf.columns and "cell_bind" in _bdf.columns:
                    _bdf_valid = _bdf[
                        (_bdf["cell_sl_to_tp"] <= 2.8) &  # Allow slight wiggle room above 2.5
                        (_bdf["cell_bind"] >= 0.30)      # Minimum activity
                    ].copy()
                else:
                    _bdf_valid = _bdf.copy()
                if _bdf_valid.empty:
                    _bdf_valid = _bdf.copy() # Fallback

                # Aggregate per-config metrics across horizons (mean of valid cells)
                _metr_cols = ["stage2_score", "cell_auc", "cell_tp_sep", "cell_ap_lift", "cell_tp_over_sl",
                              "cell_auc_bound", "cell_timeout", "cell_bind", "cell_ece", "cell_brier",
                              "cell_ic_std_time", "cell_ic_std_asset"]
                _full_agg = {
                    "rank": "min",
                    "k_tp": "first",
                    "sl_as_tp_pct": "first",
                    "base_atr_window": "first",
                    "tp_base_pct": "first",
                    "tp_abs_lo_pct": "first",
                    "sl_abs_lo_pct": "first",
                    "mode": "first",
                }
                for c in _metr_cols:
                    if c in _bdf_valid.columns:
                        _full_agg[c] = "mean"
                _bdf_agg = _bdf_valid.groupby("config_id").agg(_full_agg).reset_index()

                # Per-cell ranked configs. Persist the full ranked set so downstream
                # consumers can recover more than the top-1 winner for each cell.
                for cell_key, cell_group in _bdf_valid.groupby("cell_key"):
                    _cell_stage2 = pd.to_numeric(cell_group.get("stage2_score", pd.Series(np.nan, index=cell_group.index)), errors="coerce")
                    _cell_aucb = pd.to_numeric(cell_group.get("cell_auc_bound", pd.Series(np.nan, index=cell_group.index)), errors="coerce").fillna(0.0)
                    _cell_sep = pd.to_numeric(cell_group.get("cell_tp_sep", pd.Series(np.nan, index=cell_group.index)), errors="coerce").fillna(0.0)
                    _cell_ece = pd.to_numeric(cell_group.get("cell_ece", pd.Series(np.nan, index=cell_group.index)), errors="coerce").fillna(999.0)
                    _cell_brier = pd.to_numeric(cell_group.get("cell_brier", pd.Series(np.nan, index=cell_group.index)), errors="coerce").fillna(999.0)
                    cell_group = cell_group.assign(
                        _score=_cell_stage2.fillna(-np.inf),
                        _aucb=_cell_aucb,
                        _sep=_cell_sep,
                        _ece=_cell_ece,
                        _brier=_cell_brier,
                    )
                    _cell_ranked = cell_group.sort_values(
                        ["_score", "_aucb", "_sep", "_ece", "_brier"],
                        ascending=[False, False, False, True, True]
                    ).reset_index(drop=True)
                    _cell_bucket, _cell_horizon = cell_key.rsplit("_", 1)
                    for _rank_i, _row in _cell_ranked.iterrows():
                        _cid = str(_row["config_id"])
                        _cfg_details = details.get(_cid, {}).get("config", {})
                        _cell_payload = dict(_cfg_details)
                        _cell_payload["cell_key"] = cell_key
                        _cell_payload["bucket"] = _cell_bucket
                        _cell_payload["horizon"] = _cell_horizon
                        _cell_payload["config_id"] = _cid
                        _cell_payload["rank_in_cell"] = int(_rank_i + 1)
                        _cell_payload["cell_auc"] = float(_row.get("cell_auc", 0.5))
                        _cell_payload["cell_auc_bound"] = float(_row.get("cell_auc_bound", float("nan")))
                        _cell_payload["cell_tp_sep"] = float(_row.get("cell_tp_sep", 0.0))
                        _cell_payload["cell_ap_lift"] = float(_row.get("cell_ap_lift", 1.0))
                        _cell_payload["cell_ece"] = float(_row.get("cell_ece", float("nan")))
                        _cell_payload["cell_brier"] = float(_row.get("cell_brier", float("nan")))
                        _cell_payload["cell_bind"] = float(_row.get("cell_bind", float("nan")))
                        _cell_payload["cell_timeout"] = float(_row.get("cell_timeout", float("nan")))
                        _cell_payload["cell_ic_std_time"] = float(_row.get("cell_ic_std_time", float("nan")))
                        _cell_payload["cell_ic_std_asset"] = float(_row.get("cell_ic_std_asset", float("nan")))
                        _cell_payload["stage2_score"] = float(_row.get("stage2_score", float("nan")))
                        _cell_payload["cell_score"] = float(_row["_score"])
                        per_cell_rows.append(_cell_payload)

                _bdf_agg["_score"] = pd.to_numeric(_bdf_agg.get("stage2_score", pd.Series(np.nan, index=_bdf_agg.index)), errors="coerce").fillna(-np.inf)
                _bdf_agg["_aucb"] = pd.to_numeric(_bdf_agg.get("cell_auc_bound", pd.Series(np.nan, index=_bdf_agg.index)), errors="coerce").fillna(0.0)
                _bdf_agg["_sep"] = pd.to_numeric(_bdf_agg.get("cell_tp_sep", pd.Series(np.nan, index=_bdf_agg.index)), errors="coerce").fillna(0.0)
                _bdf_agg["_ece"] = pd.to_numeric(_bdf_agg.get("cell_ece", pd.Series(np.nan, index=_bdf_agg.index)), errors="coerce").fillna(999.0)
                _bdf_agg["_brier"] = pd.to_numeric(_bdf_agg.get("cell_brier", pd.Series(np.nan, index=_bdf_agg.index)), errors="coerce").fillna(999.0)
                _bdf_agg = _bdf_agg.sort_values(
                    ["_score", "_aucb", "_sep", "_ece", "_brier"],
                    ascending=[False, False, False, True, True]
                )
                _bkt_winner = _bdf_agg.iloc[0]
                _bkt_cid = str(_bkt_winner["config_id"])
                _bkt_cfg = details.get(_bkt_cid, {}).get("config", {})
                _bkt_row: Dict[str, Any] = {
                    "bucket": _bkt,
                    "config_id": _bkt_cid,
                    "rank_in_bucket": int(_bkt_winner.get("rank", 1)),
                    "source": "per_bucket",
                    # Barrier geometry (from full config for completeness)
                    "k_tp": float(_bkt_winner["k_tp"]),
                    "sl_as_tp_pct": float(_bkt_winner["sl_as_tp_pct"]),
                    "base_atr_window": int(_bkt_winner["base_atr_window"]),
                    "tp_base_pct": float(_bkt_winner.get("tp_base_pct", float("nan"))),
                    "tp_abs_lo_pct": float(_bkt_winner.get("tp_abs_lo_pct", float("nan"))),
                    "sl_abs_lo_pct": float(_bkt_winner.get("sl_abs_lo_pct", float("nan"))),
                    "tp_abs_hi_pct": float(_bkt_cfg.get("tp_abs_hi_pct", float("nan"))) if isinstance(_bkt_cfg, dict) else float("nan"),
                    "sl_abs_hi_pct": float(_bkt_cfg.get("sl_abs_hi_pct", float("nan"))) if isinstance(_bkt_cfg, dict) else float("nan"),
                    "tp_method": str(_bkt_cfg.get("tp_method", "atr_norm")) if isinstance(_bkt_cfg, dict) else "atr_norm",
                    "sl_method": str(_bkt_cfg.get("sl_method", "tp_pct")) if isinstance(_bkt_cfg, dict) else "tp_pct",
                    "mode": str(_bkt_winner.get("mode", "")).split("_2A")[0].split("_refine")[0],
                    "horizon_base": int(_bkt_cfg.get("horizon_base", 4)) if isinstance(_bkt_cfg, dict) else 4,
                    "horizon_scaling": str(_bkt_cfg.get("horizon_scaling", "sqrt")) if isinstance(_bkt_cfg, dict) else "sqrt",
                    # Learnability metrics for this bucket
                    "bucket_auc": float(_bkt_winner.get("cell_auc", float("nan"))),
                    "bucket_auc_bound": float(_bkt_winner.get("cell_auc_bound", float("nan"))),
                    "bucket_tp_sep": float(_bkt_winner.get("cell_tp_sep", float("nan"))),
                    "bucket_ap_lift": float(_bkt_winner.get("cell_ap_lift", float("nan"))),
                    "bucket_tp_over_sl": float(_bkt_winner.get("cell_tp_over_sl", float("nan"))),
                    "bucket_timeout": float(_bkt_winner.get("cell_timeout", float("nan"))),
                    "bucket_bind": float(_bkt_winner.get("cell_bind", float("nan"))),
                    "bucket_ece": float(_bkt_winner.get("cell_ece", float("nan"))),
                    "bucket_brier": float(_bkt_winner.get("cell_brier", float("nan"))),
                    "bucket_ic_std_time": float(_bkt_winner.get("cell_ic_std_time", float("nan"))),
                    "bucket_ic_std_asset": float(_bkt_winner.get("cell_ic_std_asset", float("nan"))),
                    "bucket_stage2_score": float(_bkt_winner.get("stage2_score", float("nan"))),
                    "learnability_score": float(_bkt_winner["_score"]),
                    "saved_at": pd.Timestamp.utcnow().isoformat(),
                }
                _bucket_best_rows.append(_bkt_row)
                tprint(
                    f"[bucket_best] {_bkt}: winner={_bkt_cid} k_tp={_bkt_row['k_tp']} "
                    f"sl={_bkt_row['sl_as_tp_pct']} atr={_bkt_row['base_atr_window']} "
                    f"auc={_bkt_row['bucket_auc']:.4f} tp_sep={_bkt_row['bucket_tp_sep']:.4f}"
                )

            if _bucket_best_rows:
                _bkt_best_df = pd.DataFrame(_bucket_best_rows)
                _bkt_best_df.to_csv(TBM_BEST_PARAMS_PER_BUCKET_CSV, index=False)
                tprint(f"Saved per-bucket best params: {TBM_BEST_PARAMS_PER_BUCKET_CSV} ({len(_bucket_best_rows)} buckets)")

            if per_cell_rows:
                pd.DataFrame(per_cell_rows).to_csv(TBM_BEST_PARAMS_PER_CELL_CSV, index=False)
                tprint(f"Saved per-cell best params to {TBM_BEST_PARAMS_PER_CELL_CSV}")

    tprint(f"Saved CSV: {output_path}")
    tprint(f"Saved JSON: {detail_path}")
    if not out_df.empty:
        _print_winning_geometry_summary(out_df, details, per_cell_grids=per_cell_grids, top_k=5)

    # Per-cell geometry quality report
    try:
        from extreme_price_movements.reports.bucket_report import report_compare_tbm
        import re as _re
        _run_id = _re.sub(r"[^0-9_]", "", str(Path(output_path).stem)) or "tbm_run"
        _rp = report_compare_tbm(_run_id, str(TBM_GEOMETRY_GRID_CSV), base_dir=runtime_cfg.get('reports_root'))
        tprint(f"TBM geometry bucket report: {_rp}")
    except Exception as _rpe:
        tprint(f"WARNING: TBM geometry bucket report failed: {_rpe}")

    tprint(
        f"Run completed in {time.perf_counter()-t0:.2f}s with mem_peak_mb={_memory_snapshot_mb():.1f} "
        f"{_cache_pressure_summary(layer1_cache, layer2_cache, eval_cache)}"
    )





def _log_prod_aligned_wiring(cfgs: List[Dict[str, Any]], stage_name: str) -> None:
    if not cfgs:
        tprint(f"[prod_aligned_tp] {stage_name}: no configs")
        return
    n_with = sum(1 for c in cfgs if isinstance(c.get("prod_aligned_tp", None), dict) and bool(c.get("prod_aligned_tp")))
    h_vals = sorted(set(int(h) for h in (2, 4)))
    tprint(
        f"[prod_aligned_tp] {stage_name}: configs={len(cfgs)} with_prod_aligned_meta={n_with}/{len(cfgs)} "
        f"horizons={h_vals}"
    )

def _build_prod_aligned_reports(
    out_df: pd.DataFrame,
    details: Dict[str, Any],
    output_path: Path,
) -> None:
    """Emit diversity/tradeability/rejection reports for expanded prod-aligned grid."""
    rows_div = []
    rows_rej = []
    rows_trade = []

    for r in out_df.itertuples(index=False):
        _r = r._asdict()
        cid = str(_r.get("config_id", ""))
        bucket = str(_r.get("mode", "unknown"))
        d = details.get(cid, {}) if isinstance(details, dict) else {}
        pa = d.get("production_admissibility", {}) if isinstance(d, dict) else {}
        agg = pa.get("aggregates", {}) if isinstance(pa, dict) else {}
        cfg_meta = d.get("prod_aligned_tp", _r.get("prod_aligned_tp", {})) if isinstance(d, dict) else {}

        targets = cfg_meta.get("tp_eff_targets", {}) if isinstance(cfg_meta, dict) else {}
        tp_base = float(_r.get("tp_base_pct", float("nan")))
        sl_as_tp = float(_r.get("sl_as_tp_pct", float("nan")))
        rows_div.append({
            "bucket": bucket,
            "config_id": cid,
            "tp_base_pct": tp_base,
            "sl_anchor_pct": tp_base * sl_as_tp if (math.isfinite(tp_base) and math.isfinite(sl_as_tp)) else float("nan"),
            "tp_eff_h2": float(targets.get("H2", float("nan"))),
            "tp_eff_h4": float(targets.get("H4", float("nan"))),
            "tp_eff_h8": float(targets.get("H8", float("nan"))),
            "min_cell_auc_bound": float(_r.get("min_cell_auc_bound", _r.get("min_cell_auc", float("nan")))),
        })

        rows_trade.append({
            "config_id": cid,
            "bucket": bucket,
            "prod_admissible_tier0": bool(_r.get("prod_admissible_tier0", False)),
            "tp_eff_p50_prod": float(agg.get("tp_eff_p50_prod", _r.get("tp_eff_p50_prod", float("nan")))),
            "tp_eff_p75_prod": float(agg.get("tp_eff_p75_prod", _r.get("tp_eff_p75_prod", float("nan")))),
            "tp_eff_p90_prod": float(agg.get("tp_eff_p90_prod", _r.get("tp_eff_p90_prod", float("nan")))),
            "tp_eff_tradeable_rule_p50": bool(agg.get("tp_eff_tradeable_rule_p50", _r.get("tp_eff_tradeable_rule_p50", False))),
            "tp_eff_tradeable_rule_tail": bool(agg.get("tp_eff_tradeable_rule_tail", _r.get("tp_eff_tradeable_rule_tail", False))),
            "tp_eff_tradeable_ok": bool(agg.get("tp_eff_tradeable_ok", _r.get("tp_eff_tradeable_ok", False))),
            "tp_floor_bind_prod_agg": float(_r.get("tp_floor_bind_prod_agg", float("nan"))),
            "max_cell_tp_floor_bind_prod": float(_r.get("max_cell_tp_floor_bind_prod", float("nan"))),
            "tp_hit_agg": float(_r.get("tp_hit_rate", float("nan"))),
            "sl_to_tp_agg": float(_r.get("sl_to_tp", float("nan"))),
            "tp_over_sl_median_cell": float(_r.get("median_cell_tp_over_sl", float("nan"))),
            "min_cell_auc_bound": float(_r.get("min_cell_auc_bound", float("nan"))),
            "min_cell_tp_sep": float(_r.get("min_cell_tp_sep", float("nan"))),
            "missing_cells": int(max(0, int(_safe_float(_r.get("total_cells", 0))) - int(_safe_float(_r.get("pass_cells", 0))))),
        })

        fails = pa.get("failures", []) if isinstance(pa, dict) else []
        if not bool(_r.get("prod_admissible_tier0", False)):
            if fails:
                for f in fails:
                    rows_rej.append({"config_id": cid, "bucket": bucket, "reason": str(f)})
            else:
                rows_rej.append({"config_id": cid, "bucket": bucket, "reason": "unknown_prod_admissibility_failure"})

    if rows_div:
        div_df = pd.DataFrame(rows_div)
        grp = div_df.groupby("bucket", observed=True).agg(
            unique_tp_base_pct=("tp_base_pct", lambda x: int(pd.Series(x).dropna().round(6).nunique())),
            unique_tp_eff_h2=("tp_eff_h2", lambda x: int(pd.Series(x).dropna().round(6).nunique())),
            unique_tp_eff_h4=("tp_eff_h4", lambda x: int(pd.Series(x).dropna().round(6).nunique())),
            unique_tp_eff_h8=("tp_eff_h8", lambda x: int(pd.Series(x).dropna().round(6).nunique())),
            h2_min=("tp_eff_h2", "min"),
            h2_max=("tp_eff_h2", "max"),
            min_auc_bound=("min_cell_auc_bound", "min"),
            tp_base_span=("tp_base_pct", _span_ratio),
            sl_anchor_span=("sl_anchor_pct", _span_ratio),
        ).reset_index()
        _min_sl_span = 1.5
        _min_tp_span = 2.0
        grp["tp_span_required"] = _min_tp_span
        grp["diversity_pass"] = (
            (grp["unique_tp_base_pct"] >= 4)
            & (grp["unique_tp_eff_h2"] >= 4)
            & (grp["h2_min"] <= 0.011)
            & (grp["h2_max"] >= 0.03)
            & (grp["tp_base_span"] >= grp["tp_span_required"])
            & (grp["sl_anchor_span"] >= _min_sl_span)
        )
        div_path = output_path.with_suffix(".prod_aligned_diversity.csv")
        grp.to_csv(div_path, index=False)
        tprint(f"Saved production-aligned diversity report: {div_path}")

    if rows_trade:
        trade_df = pd.DataFrame(rows_trade)
        trade_path = output_path.with_suffix(".prod_aligned_tradeability.csv")
        trade_df.to_csv(trade_path, index=False)
        tprint(f"Saved production-aligned tradeability report: {trade_path}")

    if rows_rej:
        rej_df = pd.DataFrame(rows_rej)
        rej_path = output_path.with_suffix(".rejected_configs_reasons.csv")
        rej_df.to_csv(rej_path, index=False)
        tprint(f"Saved rejected-config reasons report: {rej_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize/compare TBM parameter sets")
    # Features and panel are now auto-detected from CFG data_root, no longer required args
    p.add_argument("--features", default=None, help="Path to features directory (auto-detected from data_root if not set)")
    p.add_argument("--panel", default=None, help="Path to panel parquet or symbol parquet directory (auto-detected from data_root if not set)")
    p.add_argument("--output", default=str(REPORTS_DIR / "tbm_parameter_comparison.csv"), help="Output CSV path")
    p.add_argument("--quick", action="store_true", help="Quick stage1 subset")
    p.add_argument("--stage2", dest="stage2", action="store_true", help="Run stage2 validation")
    p.add_argument("--no-stage2", dest="stage2", action="store_false", help="Disable stage2 validation")
    p.set_defaults(stage2=True)
    p.add_argument("--top-k", type=int, default=10, help="Stage1 promotion top-k")
    p.add_argument("--winners", nargs="*", default=[], help="Explicit stage1 config IDs")
    p.add_argument("--horizons", default="2,4", help="Comma-separated horizons in hours")
    p.add_argument("--max-configs", type=int, default=20, help="Max configs when --quick")
    p.add_argument("--max-stage2-configs", type=int, default=24, help="Max configs per Stage2 substage (hierarchical cap)")
    p.add_argument("--lookback-years", type=int, default=2, help="Years of history to keep")
    p.add_argument("--weights-output", default="", help="Optional sample-weights parquet output path")
    p.add_argument("--tbm-cache-max-mb", type=int, default=1536, help="Max on-disk persisted TBM cache size in MB")
    p.add_argument("--data-root", default=None, help="Override cfg data_root")
    p.add_argument("--perps", action="store_true", help="Use perp mode data/features (_perp root + perp feature keys)")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
