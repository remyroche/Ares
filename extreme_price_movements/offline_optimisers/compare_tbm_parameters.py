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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extreme_price_movements.data_store import to_panel
from extreme_price_movements.labeling import compute_triple_barrier_labels
from extreme_price_movements.config import CFG, TEST_FEATURE_KEYS
from extreme_price_movements.candidates import select_trade_candidates_vectorized
from extreme_price_movements.offline_optimisers.params_store import (
    REPORTS_DIR,
    TBM_BEST_PARAMS_CSV,
    save_best_params_csv,
    apply_offline_optimizer_best_params,
)
from extreme_price_movements.training_defaults import (
    get_candidate_filter_defaults,
    get_barrier_factory_defaults,
    get_tbm_optimizer_defaults,
)
from extreme_price_movements.utils import tprint


EPS = 1e-12


def _memory_snapshot_mb() -> float:
    """Process resident memory estimate in MB (high-water mark on Linux)."""
    rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss_kb / 1024.0


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


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


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


def _load_panel_from_store(cfg: Dict[str, Any]) -> Optional[Dict[str, pd.DataFrame]]:
    """Load panel data from PartitionedOHLCVStore (same as training pipeline).
    
    Subsamples aggressively to reduce memory usage for Stage 1 quick scans.
    """
    from extreme_price_movements.data_store import PartitionedOHLCVStore
    from extreme_price_movements.universe import refresh_margin_universe_daily
    
    try:
        store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
        
        # Get margin symbols
        mu = refresh_margin_universe_daily(None, quotes=("USDT", "USDC", "BUSD", "EUR"))
        margin_symbols = mu.symbols if mu else []
        
        # Use market_basket from config and limit total symbols
        market_basket = cfg.get("market_basket", [])
        all_syms = list(set(margin_symbols + market_basket))
        
        # Aggressive subsample: take every 4th asset for Stage 1
        train_syms = all_syms[::4]
        # Limit to max 30 symbols for Stage 1 quick runs
        train_syms = train_syms[:30]
        
        tprint(f"Loading panel from store for {len(train_syms)} symbols (Stage 1 subsampled)")
        dfs = store.load_symbols(train_syms)
        if not dfs:
            return None
        return to_panel(dfs)
    except Exception as e:
        tprint(f"Failed to load panel from store: {e}")
        return None


def _load_features_from_data_root(cfg: Dict[str, Any]) -> Optional[Dict[str, pd.DataFrame]]:
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
    
    # Load only TEST_FEATURE_KEYS to minimize memory
    dfs = _read_symbol_parquet_dir(feat_dir)
    feat_buf: Dict[str, Dict[str, pd.Series]] = {}
    
    # Only process columns that are in TEST_FEATURE_KEYS
    test_keys_set = set(TEST_FEATURE_KEYS)
    for sym, df in dfs.items():
        for c in df.columns:
            if c in test_keys_set:  # Only keep test feature keys
                feat_buf.setdefault(c, {})[sym] = pd.to_numeric(df[c], errors="coerce")
    
    out = {k: pd.DataFrame(v).sort_index() for k, v in feat_buf.items()}
    tprint(f"Loaded {len(out)} features (TEST_FEATURE_KEYS only)")
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
                    "slice_timeout": _safe_float(sm.get("timeout"), np.nan),
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
    "quantile_window", "tp_quantile",
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
    Bounded cache for eval_cache with automatic eviction of oldest entries.
    Handles nested dicts for bucket_stack and other eval caches.
    """
    def __init__(self, max_size=20):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __getitem__(self, key: str) -> Any:
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Evict oldest
            oldest = self._access_order.pop(0)
            old_val = self._cache.pop(oldest, None)
            # If it's a dict with large arrays, try to free memory
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
        self._cache.clear()
        self._access_order.clear()
        gc.collect()

    def __len__(self) -> int:
        return len(self._cache)


@dataclass
class RunArtifacts:
    panel: Dict[str, pd.DataFrame]
    features: Dict[str, pd.DataFrame]


def _subsample_symbols(symbols: Sequence[str]) -> List[str]:
    """Deterministic symbol subsample: alphabetical, keep every 4th token.
    
    Reduced from every 2nd to every 4th for Stage 1 memory efficiency.
    """
    syms_sorted = sorted(set(map(str, symbols)))
    return syms_sorted[::4] if syms_sorted else []


def _index_cache_key(index: pd.MultiIndex) -> str:
    """Stable key for per-index cached stacked arrays."""
    ts_vals = index.get_level_values(0).asi8.astype(np.int64, copy=False)
    sym_vals = index.get_level_values(1).astype(str)
    ts_hash = int(pd.util.hash_array(ts_vals).sum())
    sym_hash = int(pd.util.hash_array(sym_vals.to_numpy(dtype=object, copy=False)).sum())
    return f"{len(index)}::{ts_hash}::{sym_hash}"


# ---------------------------
# IO helpers
# ---------------------------
def _read_symbol_parquet_dir(folder: Path) -> Dict[str, pd.DataFrame]:
    files = sorted(folder.glob("symbol=*.parquet"))
    if not files:
        raise FileNotFoundError(f"No symbol parquet files in {folder}")

    dfs: Dict[str, pd.DataFrame] = {}
    for f in files:
        raw_sym = f.stem.replace("symbol=", "")
        df = pd.read_parquet(f)
        if "__symbol__" in df.columns and not df.empty:
            sym = str(df["__symbol__"].iloc[0])
            df = df.drop(columns=["__symbol__"])
        else:
            sym = raw_sym.replace("_", "/", 1)

        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            else:
                raise ValueError(f"Cannot infer timestamp index for {f}")

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        dfs[sym] = df.sort_index()
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
        p_out[k] = df.reindex(index=idx, columns=cols).astype(np.float32)

    f_out: Dict[str, pd.DataFrame] = {}
    for k, df in features.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            continue
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        f_out[k] = df.reindex(index=idx, columns=cols).astype(np.float32)

    if "atr_pct" not in f_out:
        close = p_out["close"]
        high = p_out["high"]
        low = p_out["low"]
        tr = pd.concat(
            [
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=0,
        ).groupby(level=0).max()
        atr = tr.rolling(24, min_periods=4).mean()
        f_out["atr_pct"] = (atr / close).clip(lower=1e-6).fillna(0.01).astype(np.float32)

    out = RunArtifacts(panel=p_out, features=f_out)
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


def config_id(config: Dict[str, Any]) -> str:
    payload = json.dumps(serialize_key(config), sort_keys=True)
    return "CFG" + hashlib.sha1(payload.encode()).hexdigest()[:10].upper()


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return 0.0
    r = spearmanr(a[m], b[m]).correlation
    return float(r) if np.isfinite(r) else 0.0


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


# ---------------------------
# Barrier geometry
# ---------------------------
def _regime_multiplier(atr_pct: pd.DataFrame, model: str, mix_weight: float = 0.5) -> pd.DataFrame:
    roll = atr_pct.rolling(24 * 14, min_periods=24).median()
    ratio = (atr_pct / roll).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    ratio = ratio.clip(0.5, 2.0)

    shock = atr_pct.pct_change().abs().rolling(24, min_periods=4).mean().fillna(0.0)
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


def _horizon_scale(h: int, scaling: str, alpha: float = 0.5, base: float = 4.0) -> float:
    if scaling == "none":
        return 1.0
    if scaling == "sqrt":
        return math.sqrt(max(h, EPS) / max(base, EPS))
    if scaling == "power":
        return (max(h, EPS) / max(base, EPS)) ** alpha
    raise ValueError(f"Unknown horizon scaling: {scaling}")


def _apply_caps(x: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    return x.clip(lower=lo, upper=hi)


def build_barriers(
    artifacts: RunArtifacts,
    cfg: Dict[str, Any],
    horizon: int,
    side: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    atr = artifacts.features["atr_pct"].clip(lower=1e-6)

    h_scale = _horizon_scale(
        h=horizon,
        scaling=cfg.get("horizon_scaling", "none"),
        alpha=float(cfg.get("horizon_alpha", 0.5)),
        base=float(cfg.get("horizon_base", 4.0)),
    )

    tp_regime = _regime_multiplier(atr, cfg.get("tp_regime_model", "none"), cfg.get("mix_weight", 0.5))
    sl_regime = _regime_multiplier(atr, cfg.get("sl_regime_model", cfg.get("tp_regime_model", "none")), cfg.get("mix_weight", 0.5))

    side_skew = float(cfg.get("tp_side_skew", 0.0))
    side_mult = (1 + side_skew) if side == "long" else (1 - side_skew)

    tp_method = cfg["tp_method"]
    sl_method = cfg["sl_method"]

    if tp_method == "atr_mult":
        tp = cfg["k_tp"] * side_mult * atr * tp_regime * h_scale
    elif tp_method == "semi_atr_mult":
        atr_tp = cfg["k_tp"] * side_mult * atr * tp_regime * h_scale
        abs_tp = pd.DataFrame(float(cfg.get("tp_abs_pct", cfg.get("tp_base_pct", 0.02))), index=atr.index, columns=atr.columns)
        tp = 0.5 * atr_tp + 0.5 * abs_tp
    elif tp_method == "absolute":
        tp = pd.DataFrame(float(cfg["tp_abs_pct"]), index=atr.index, columns=atr.columns)
    elif tp_method == "atr_norm":
        med = atr.rolling(int(cfg.get("base_atr_window", 168)), min_periods=24).median().fillna(atr.median())
        tp = cfg["k_tp"] * (atr / (med + EPS)) * float(cfg["tp_base_pct"])
    elif tp_method == "semi_atr_norm":
        med = atr.rolling(int(cfg.get("base_atr_window", 168)), min_periods=24).median().fillna(atr.median())
        atrn_tp = cfg["k_tp"] * (atr / (med + EPS)) * float(cfg.get("tp_base_pct", 0.02))
        abs_tp = pd.DataFrame(float(cfg.get("tp_abs_pct", cfg.get("tp_base_pct", 0.02))), index=atr.index, columns=atr.columns)
        tp = 0.5 * atrn_tp + 0.5 * abs_tp
    elif tp_method == "rolling_quantile":
        # Calculate absolute returns for quantile estimation
        # Use close-to-close returns
        abs_ret = artifacts.panel["close"].pct_change().abs()

        # Calculate rolling quantile
        window = int(cfg.get("quantile_window", 720))
        q = float(cfg.get("tp_quantile", 0.95))

        # Pre-clip extreme outliers (winsorization) to prevent skewing
        # Clip at 99.9th percentile of the full series or fixed threshold
        # For simplicity/speed in rolling window, just clip raw series at a high threshold (e.g. 20%)
        # or rely on quantile robustness. Here we rely on quantile robustness but user can add winsorization later.

        base_val = abs_ret.rolling(window, min_periods=max(24, window // 10)).quantile(q)

        # Fill initial NaNs with expanding quantile or median * multiplier fallback
        if base_val.isna().any().any():
             # Fallback to ATR-based estimate for warmup period
             fallback = 2.0 * atr
             base_val = base_val.fillna(fallback)

        # Apply scaling:
        # base_val is 1-hour quantile (since panel is hourly).
        # We need to scale to the target horizon 'h'.
        # Re-use h_scale logic for consistent horizon scaling.
        # Also apply regime multiplier (optional, but consistent with other methods).
        tp = base_val * h_scale * tp_regime
    else:
        raise ValueError(f"Unsupported tp_method={tp_method}")

    if sl_method == "tp_pct":
        sl = float(cfg["sl_as_tp_pct"]) * tp
    elif sl_method == "atr_mult":
        sl = float(cfg["k_sl"]) * atr * sl_regime
    elif sl_method == "absolute":
        sl = pd.DataFrame(float(cfg["sl_abs_pct"]), index=atr.index, columns=atr.columns)
    else:
        raise ValueError(f"Unsupported sl_method={sl_method}")

    tp = _apply_caps(tp, float(cfg.get("tp_abs_lo_pct", 0.005)), float(cfg.get("tp_abs_hi_pct", 0.08)))
    sl = _apply_caps(sl, float(cfg.get("sl_abs_lo_pct", 0.005)), float(cfg.get("sl_abs_hi_pct", 0.08)))

    if cfg.get("sl_noise_buffer", False):
        sl_min = max(float(cfg.get("sl_min_abs_pct", 0.01)), float(cfg.get("sl_min_bps", 100)) / 10000.0)
        sl = sl.clip(lower=sl_min)

    tp_min = max(float(cfg.get("tp_min_abs_pct", 0.005)), float(cfg.get("tp_min_bps", 50)) / 10000.0)
    tp = tp.clip(lower=tp_min)

    # Path dependence approximation knobs (applied as geometry transforms).
    if cfg.get("tp_time_decay", "none") == "linear":
        decay = max(0.6, 1.0 - 0.06 * max(horizon - 2, 0))
        tp = tp * decay
    if float(cfg.get("trail_sl_mult", 0.0)) > 0:
        sl = sl * (1.0 - 0.15 * float(cfg.get("trail_sl_mult", 0.0)))

    out_stats = {
        "tp_mean": float(np.nanmean(tp.values)),
        "sl_mean": float(np.nanmean(sl.values)),
        "bound_saturation": float(((tp.values <= float(cfg.get("tp_abs_lo_pct", 0.005))) | (tp.values >= float(cfg.get("tp_abs_hi_pct", 0.08)))).mean()),
    }
    return tp.astype(np.float32), sl.astype(np.float32), out_stats


# ---------------------------
# Event extraction and scoring
# ---------------------------
def build_bucket_masks(artifacts: RunArtifacts, cfg_runtime: Dict[str, Any] | None = None) -> Dict[str, pd.DataFrame]:
    c = artifacts.panel["close"]
    feats = artifacts.features

    tf_source = None
    for k in ["trend_snr", "tf_bias", "trend_regime", "ret4h", "ret1h"]:
        if k in feats:
            tf_source = feats[k]
            break
    if tf_source is None:
        tf_source = c.pct_change(4)

    mr_source = None
    for k in ["rsi", "dist_ema_fast", "zscore_close", "ret1h_z"]:
        if k in feats:
            mr_source = feats[k]
            break
    if mr_source is None:
        ema = c.ewm(span=24, min_periods=6).mean()
        mr_source = (ema - c) / (ema.abs() + EPS)

    tf_long = (tf_source > 0).astype(bool)
    tf_short = (tf_source <= 0).astype(bool)
    mr_long = (mr_source > 0).astype(bool)
    mr_short = (mr_source <= 0).astype(bool)

    candidate_filter = pd.DataFrame(True, index=c.index, columns=c.columns)
    if cfg_runtime is not None:
        candidate_defaults = get_candidate_filter_defaults(cfg_runtime)
        try:
            candidate_filter = select_trade_candidates_vectorized(
                artifacts.panel,
                artifacts.features,
                pct=float(candidate_defaults["train_extreme_pct_hourly"]),
                metric=cfg_runtime.get("trade_deviation_metric", "ret24h"),
                min_range_pct=float(candidate_defaults["train_min_range_pct"]),
                min_vol_zscore=float(candidate_defaults["train_min_vol_zscore"]),
            )
            if candidate_filter is None:
                candidate_filter = pd.DataFrame(True, index=c.index, columns=c.columns)
        except Exception:
            candidate_filter = pd.DataFrame(True, index=c.index, columns=c.columns)

    tf_long = tf_long & candidate_filter
    tf_short = tf_short & candidate_filter
    mr_long = mr_long & candidate_filter
    mr_short = mr_short & candidate_filter

    return {
        "TF_long": tf_long,
        "TF_short": tf_short,
        "MR_long": mr_long,
        "MR_short": mr_short,
        "Global": pd.DataFrame(True, index=c.index, columns=c.columns),
    }


def make_quantile_basis(artifacts: RunArtifacts, basis: str) -> pd.DataFrame:
    feats = artifacts.features
    c = artifacts.panel["close"]
    if basis == "vol" and "atr_pct" in feats:
        return feats["atr_pct"]
    if basis == "breakout":
        for k in ["breakout_24h", "breakout_t", "pct_breakout_t"]:
            if k in feats:
                return feats[k]
    if basis == "volume":
        for k in ["vol_z", "rvol_z", "volume_entropy_12"]:
            if k in feats:
                return feats[k]
        return artifacts.panel["volume"].pct_change().abs()
    if basis == "trend":
        for k in ["trend_snr", "tf_bias", "trend_regime"]:
            if k in feats:
                return feats[k]
        return c.pct_change(6)

    parts = []
    for k in ["atr_pct", "trend_snr", "vol_z", "rsi"]:
        if k in feats:
            x = feats[k]
            parts.append((x - x.rolling(72, min_periods=8).median()) / (x.rolling(72, min_periods=8).std() + EPS))
    if parts:
        return sum(parts) / len(parts)
    return c.pct_change().abs()


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

    preferred = CFG.get("test_feature_keys", TEST_FEATURE_KEYS)
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


def get_stacked_feature_matrix(
    artifacts: RunArtifacts,
    eval_cache: BoundedEvalCache,
    max_features: int = 20,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build (once) and cache stacked feature matrix in float32."""
    key = f"feature_matrix::{max_features}"
    if key not in eval_cache:
        X_flat, feat_cols = choose_feature_matrix(artifacts, max_features=max_features)
        eval_cache[key] = (X_flat, feat_cols)
        gc.collect()
    return eval_cache[key]


def get_stacked_array(
    eval_cache: BoundedEvalCache,
    cache_name: str,
    frame: pd.DataFrame,
    stacked_index: pd.MultiIndex,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    """Cache DataFrame.stack().reindex(...) as dense numpy array."""
    cache = eval_cache.setdefault(cache_name, {})
    key = f"{_index_cache_key(stacked_index)}::{id(frame)}"
    if key not in cache:
        arr = frame.stack().reindex(stacked_index).to_numpy(dtype=dtype, copy=False)
        cache[key] = np.asarray(arr, dtype=dtype)
    return cache[key]


def compute_weights(events: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    scheme = cfg.get("weighting_scheme", "none")
    w = np.ones(len(events), dtype=np.float32)
    if scheme == "none":
        return w

    rr = events["tp"].values / np.maximum(events["sl"].values, EPS)
    tp_hit = (events["label"].values == 1).astype(float)
    timeout = (events["label"].values == 0).astype(float)

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


def oof_predictions_by_time(
    X: pd.DataFrame,
    y_bin: np.ndarray,
    sample_weight: np.ndarray,
    n_folds: int = 2,  # 2 for stage1, 3 for stage2, 4 for stage3
) -> np.ndarray:
    ts = X.index.get_level_values(0)
    unique_ts = np.array(sorted(pd.Index(ts).unique()))
    if len(unique_ts) < n_folds + 5:
        n_folds = max(2, len(unique_ts) // 5)
    if n_folds < 2:
        return np.full(len(X), 0.5)

    chunks = np.array_split(unique_ts, n_folds)
    pred = np.full(len(X), 0.5, dtype=np.float32)
    Xv = X.fillna(0.0).to_numpy(dtype=np.float32, copy=False)

    for i, test_ts in enumerate(chunks):
        if len(test_ts) == 0:
            continue
        test_mask = ts.isin(test_ts)
        train_mask = ~test_mask
        if train_mask.sum() < 100 or test_mask.sum() == 0:
            continue

        model = Ridge(alpha=1.0, random_state=42)
        model.fit(Xv[train_mask], y_bin[train_mask], sample_weight=sample_weight[train_mask])
        p = model.predict(Xv[test_mask])
        pred[test_mask] = 1 / (1 + np.exp(-p))

    return np.clip(pred, 0.0, 1.0).astype(np.float32, copy=False)


def train_and_predict_per_bucket(
    events: pd.DataFrame,
    X_flat: pd.DataFrame,
    sample_weight: np.ndarray,
    n_folds: int = 2,  # 2 for stage1, 3 for stage2, 4 for stage3
) -> np.ndarray:
    pred = np.full(len(events), 0.5, dtype=np.float32)
    mi = pd.MultiIndex.from_arrays([events["ts"], events["symbol"]])
    for bname, g in events.groupby("bucket"):
        idx = g.index.to_numpy(dtype=np.int64)
        if len(idx) < 100:
            continue
        Xb = X_flat.reindex(mi[idx]).fillna(0.0)
        yb = (events.loc[idx, "label"].values == 1).astype(np.float32)
        wb = sample_weight[idx].astype(np.float32, copy=False)
        pred[idx] = oof_predictions_by_time(Xb, yb, wb, n_folds=n_folds)
        del Xb, yb, wb
        gc.collect()
    return pred


def per_slice_metrics(events: pd.DataFrame, score: np.ndarray, slice_col: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, g in events.groupby(slice_col):
        idx = g.index.values
        y = g["label"].values
        payoff = g["payoff"].values
        s = score[idx]
        out[str(key)] = {
            "n": int(len(g)),
            "ic_label": _safe_spearman(s, y),
            "ic_payoff": _safe_spearman(s, payoff),
            "tp_hit_rate": float((y == 1).mean()),
            "timeout_rate": float((y == 0).mean()),
        }
    return out


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
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[pd.DataFrame]]:
    cfg_id = config_id(cfg)
    t0 = time.perf_counter()
    tprint(
        f"[eval:start] {cfg_id} mode={cfg.get('mode', 'unknown')} "
        f"horizons={list(horizons)} mem_peak_mb={_memory_snapshot_mb():.1f} "
        f"{_cache_pressure_summary(layer1_cache, layer2_cache, eval_cache)}"
    )
    events_rows: List[pd.DataFrame] = []

    for h in horizons:
        for side in ["long", "short"]:
            # Key1: Barriers depends only on geometric params.
            barrier_cfg = _get_barrier_params(cfg)
            key1 = json.dumps(serialize_key({"h": h, "side": side, "cfg": barrier_cfg}), sort_keys=True)

            if key1 not in layer1_cache:
                tp_df, sl_df, geom_stats = build_barriers(artifacts, cfg, h, side)
                layer1_cache[key1] = (tp_df, sl_df, geom_stats)
                tprint(f"[eval:{cfg_id}] barrier_cache miss h={h} side={side}")
            else:
                tprint(f"[eval:{cfg_id}] barrier_cache hit h={h} side={side}")
            tp_df, sl_df, geom_stats = layer1_cache[key1]

            # Key2: Labels depends fully on barriers (key1) + horizon/side (in key1).
            # Note: compute_triple_barrier_labels uses JIT logic that may interpret TP
            # as trailing activation. It does NOT use sl_activation_minutes.
            # So key2 is effectively just key1.
            key2 = key1

            if key2 not in layer2_cache:
                lbl, ret = compute_triple_barrier_labels(artifacts.panel, tp_df, sl_df, h, side=side)
                layer2_cache[key2] = (lbl, ret)
                tprint(f"[eval:{cfg_id}] label_cache miss h={h} side={side}")
            else:
                tprint(f"[eval:{cfg_id}] label_cache hit h={h} side={side}")
            lbl, ret = layer2_cache[key2]

            # Stack once and create DataFrame efficiently
            # Use numpy arrays directly to avoid multiple .stack() calls
            label_arr = lbl.stack().to_numpy(dtype=np.float32)
            payoff_arr = ret.stack().to_numpy(dtype=np.float32)
            tp_arr = tp_df.stack().to_numpy(dtype=np.float32)
            sl_arr = sl_df.stack().to_numpy(dtype=np.float32)
            
            # Get the stacked index once
            stacked_idx = lbl.stack().index
            
            # Create DataFrame directly from numpy arrays
            df = pd.DataFrame(
                {
                    "label": label_arr,
                    "payoff": payoff_arr,
                    "tp": tp_arr,
                    "sl": sl_arr,
                },
                index=stacked_idx
            )
            df.index.names = ["ts", "symbol"]
            
            # Early filtering: drop NaNs before concatenation
            df = df.dropna(subset=["label", "payoff", "tp", "sl"])
            
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
            del label_arr, payoff_arr, tp_arr, sl_arr, stacked_idx
            gc.collect()

    events = pd.concat(events_rows, ignore_index=True)
    tprint(
        f"[eval:{cfg_id}] raw_events={len(events):,} "
        f"mem_peak_mb={_memory_snapshot_mb():.1f}"
    )

    # Bucket tagging.
    stacked_index = pd.MultiIndex.from_arrays([events["ts"], events["symbol"]])
    stack_key = _index_cache_key(stacked_index)
    cache_bucket_stack = eval_cache.setdefault("bucket_stack", {})
    if stack_key in cache_bucket_stack:
        bucket_map = cache_bucket_stack[stack_key]
    else:
        bucket_map: Dict[str, np.ndarray] = {}
        for bname, bmask in bucket_masks.items():
            bucket_map[bname] = bmask.stack().reindex(stacked_index).fillna(False).to_numpy(dtype=bool)
        cache_bucket_stack[stack_key] = bucket_map

    bucket = np.full(len(events), "Global", dtype=object)
    for bname in ["MR_long", "MR_short", "TF_long", "TF_short"]:
        m = bucket_map[bname]
        bucket[m] = bname
    events["bucket"] = bucket

    # Regime and quintile slices for Stage 2.
    atr = get_stacked_array(
        eval_cache,
        "atr_stack",
        artifacts.features["atr_pct"],
        stacked_index,
        dtype=np.float32,
    )
    atr_roll = eval_cache.get("atr_roll_14d")
    if atr_roll is None:
        atr_roll = artifacts.features["atr_pct"].rolling(24 * 14, min_periods=24).median().astype(np.float32)
        eval_cache["atr_roll_14d"] = atr_roll
        gc.collect()
    roll = get_stacked_array(
        eval_cache,
        "atr_roll_stack",
        atr_roll,
        stacked_index,
        dtype=np.float32,
    )
    atr = np.nan_to_num(atr, nan=0.0, copy=False)
    ratio = np.divide(atr, roll + EPS, out=np.ones_like(atr, dtype=np.float32), where=np.isfinite(roll))

    atr_s = pd.Series(atr, index=stacked_index, dtype=np.float32)
    ts_counts = atr_s.groupby(level=0).transform("count").to_numpy(dtype=np.int32, copy=False)
    rank_pct = atr_s.groupby(level=0).rank(method="first", pct=True).to_numpy(dtype=np.float32, copy=False)
    q = np.full(len(events), 2, dtype=np.int16)
    valid_q = (ts_counts > 5) & np.isfinite(rank_pct)
    q[valid_q] = np.minimum((rank_pct[valid_q] * 5.0).astype(np.int16), 4)
    events["vol_quintile"] = q + 1

    regime = np.where(ratio < 0.85, "low_vol", np.where(ratio > 1.15, "high_vol", "medium_vol"))
    events["regime"] = regime

    # Quantile filtering.
    quant_basis = cfg.get("quantile_basis", "composite")
    basis_frame_key = f"quant_basis_frame::{quant_basis}"
    if basis_frame_key not in eval_cache:
        eval_cache[basis_frame_key] = make_quantile_basis(artifacts, quant_basis).astype(np.float32)
        gc.collect()
    basis = get_stacked_array(
        eval_cache,
        "quant_basis_stack",
        eval_cache[basis_frame_key],
        stacked_index,
        dtype=np.float32,
    )
    keep_mask = np.ones(len(events), dtype=bool)
    full_bucket_counts = events.groupby("bucket").size().to_dict()
    if cfg.get("use_quantile_filter", False):
        lo = float(cfg.get("quantile_lo", 0.2))
        hi = float(cfg.get("quantile_hi", 0.8))
        basis_s = pd.Series(basis, index=stacked_index, dtype=np.float32)
        by_ts = basis_s.groupby(level=0)
        lo_map = by_ts.quantile(lo)
        hi_map = by_ts.quantile(hi)
        ts_idx = stacked_index.get_level_values(0)
        lo_t = ts_idx.map(lo_map).to_numpy(dtype=np.float32, copy=False)
        hi_t = ts_idx.map(hi_map).to_numpy(dtype=np.float32, copy=False)
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

    # Filter constraints.
    fee = float(cfg.get("fee_pct", 0.5)) / 100.0
    slip = float(cfg.get("slip_buffer", 0.1)) / 100.0
    tp_net = events["tp"] - fee - slip
    sl_net = events["sl"] + fee + slip
    events["net_rr"] = tp_net / np.maximum(sl_net, EPS)

    min_rr = float(cfg.get("min_net_rr", 0.7))
    min_tp_hit = float(cfg.get("min_tp_hit_rate", 0.01))
    max_timeout = float(cfg.get("max_timeout_rate", 0.95))
    min_raw = int(cfg.get("min_raw_events", 50))

    pre_rr_n = len(events)
    events = events[events["net_rr"] >= min_rr].reset_index(drop=True)
    tprint(
        f"[eval:{cfg_id}] rr_filter_kept={len(events):,}/{pre_rr_n:,} min_net_rr={min_rr:.3f}"
    )

    pass_cells = 0
    total_cells = 0
    bucket_h_metrics = {}
    for (b, h), g in events.groupby(["bucket", "horizon"]):
        total_cells += 1
        tp_hit = float((g["label"] == 1).mean())
        timeout = float((g["label"] == 0).mean())
        ok = (len(g) >= min_raw) and (tp_hit >= min_tp_hit) and (timeout <= max_timeout)
        pass_cells += int(ok)
        bucket_h_metrics[(b, h)] = {"n": int(len(g)), "tp_hit": tp_hit, "timeout": timeout, "ok": ok}

    weights = compute_weights(events, cfg)
    ess = effective_sample_size(weights)
    ess_full = float(full_n)
    coverage = ess / max(ess_full, 1.0)

    # Feature matrix + OOF scoring.
    X_flat, feat_cols = get_stacked_feature_matrix(artifacts, eval_cache)
    y_signed = events["label"].astype(np.float32).values
    y_bin = (events["label"].values == 1).astype(np.float32)
    payoff = events["payoff"].astype(np.float32).values

    pred = train_and_predict_per_bucket(events, X_flat, weights, n_folds=5)
    tprint(
        f"[eval:{cfg_id}] model_scored n={len(pred):,} ic_payoff={_safe_spearman(pred, payoff):.4f} "
        f"ic_label={_safe_spearman(pred, y_signed):.4f}"
    )

    ic_label = _safe_spearman(pred, y_signed)
    ic_payoff = _safe_spearman(pred, payoff)

    # Time/asset IC stability.
    ic_time = []
    for _, g in events.groupby("ts"):
        idx = g.index.values
        ic_time.append(_safe_spearman(pred[idx], g["payoff"].values))
    ic_asset = []
    for _, g in events.groupby("symbol"):
        idx = g.index.values
        ic_asset.append(_safe_spearman(pred[idx], g["payoff"].values))

    ic_time = np.array(ic_time, dtype=float)
    ic_asset = np.array(ic_asset, dtype=float)
    ic_snr = float(ic_time.mean() / (ic_time.std() + EPS)) if ic_time.size else 0.0

    top_q = pred >= np.quantile(pred, 0.7) if len(pred) else np.array([], dtype=bool)
    top_payoff = payoff[top_q] if top_q.any() else np.array([0.0])
    downside = np.minimum(top_payoff, 0.0)
    sortino = float(top_payoff.mean() / (np.sqrt(np.mean(downside**2) + EPS)))
    sharpe = float(top_payoff.mean() / (top_payoff.std() + EPS))

    brier = float(np.mean((pred - y_bin) ** 2)) if len(pred) else 0.0
    ece = expected_calibration_error(y_bin, pred)
    dec = pd.qcut(pd.Series(pred), 10, labels=False, duplicates="drop") if len(pred) >= 20 else pd.Series(np.zeros(len(pred), dtype=int))
    payoff_by_dec = pd.DataFrame({"d": dec, "p": payoff}).groupby("d")["p"].mean()
    mono = float((np.diff(payoff_by_dec.values) >= 0).mean()) if len(payoff_by_dec) > 1 else 0.0

    per_bucket = per_slice_metrics(events, pred, "bucket")
    bucket_ics = [m["ic_payoff"] for k, m in per_bucket.items() if k != "Global"]
    bucket_ic_labels = [m["ic_label"] for k, m in per_bucket.items() if k != "Global"]
    worst_bucket_ic = min(bucket_ics, default=0.0)
    mean_bucket_ic = float(np.mean(bucket_ics)) if bucket_ics else 0.0
    mean_bucket_ic_label = float(np.mean(bucket_ic_labels)) if bucket_ic_labels else 0.0

    min_cov_threshold = float(cfg.get("min_coverage_threshold", 0.2))
    bucket_counts_after = events.groupby("bucket").size().to_dict()
    bucket_coverages = []
    for b, n_full in full_bucket_counts.items():
        n_kept = bucket_counts_after.get(b, 0)
        bucket_coverages.append(n_kept / max(n_full, 1))
    worst_bucket_cov = min(bucket_coverages) if bucket_coverages else 1.0

    hard_gate = all(
        [
            coverage >= min_cov_threshold,
            ess >= float(cfg.get("min_ess_events", 30)),
            pass_cells >= int(0.7 * max(total_cells, 1)),
            (ic_time.min() > 0.0) if ic_time.size else (worst_bucket_ic > -0.05),
        ]
    )

    stage1_score = (
        (0.5 * ic_snr + 0.5 * mean_bucket_ic) * math.sqrt(max(coverage, 0.0))
        - 0.2 * float(events["bound_saturation"].mean() if len(events) else 0.0)
        - 0.2 * float((events["label"] == 0).mean() if len(events) else 1.0)
    )

    stage2_score = (
        0.35 * mean_bucket_ic
        + 0.25 * ic_snr
        + 0.20 * sortino
        - 0.15 * float(ic_time.std() if ic_time.size else 0.0)
        - 0.10 * float(ic_asset.std() if ic_asset.size else 0.0)
    )
    if worst_bucket_ic < -0.1:
        stage2_score -= 0.5

    summary = {
        "config_id": cfg_id,
        "mode": cfg.get("mode", "unknown"),
        "k_tp": cfg.get("k_tp"),
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
        "tp_hit_rate": float((events["label"] == 1).mean()) if len(events) else 0.0,
        "sl_hit_rate": float((events["label"] == -1).mean()) if len(events) else 0.0,
        "timeout_rate": float((events["label"] == 0).mean()) if len(events) else 1.0,
        "ess": ess,
        "ess_full": ess_full,
        "coverage": coverage,
        "ic_std_time": float(ic_time.std() if ic_time.size else 0.0),
        "ic_std_asset": float(ic_asset.std() if ic_asset.size else 0.0),
        "worst_bucket_IC": worst_bucket_ic,
        "stage1_score": stage1_score,
        "stage2_score": stage2_score,
        "brier": brier,
        "ece": ece,
        "monotonicity": mono,
        "hard_gate": bool(hard_gate),
        "pass_cells": pass_cells,
        "total_cells": total_cells,
        "worst_bucket_coverage": worst_bucket_cov,
    }

    detail = {
        "config": serialize_key(cfg),
        "feature_columns": feat_cols,
        "bucket_metrics": per_bucket,
        "bucket_horizon_metrics": {f"{k[0]}_H{k[1]}": v for k, v in bucket_h_metrics.items()},
        "calibration": {"brier": brier, "ece": ece, "monotonicity": mono},
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
        f"timeout={summary['timeout_rate']:.3f} hard_gate={hard_gate} "
        f"elapsed_s={time.perf_counter()-t0:.2f} mem_peak_mb={_memory_snapshot_mb():.1f} "
        f"{_cache_pressure_summary(layer1_cache, layer2_cache, eval_cache)}"
    )
    del events, pred, y_signed, y_bin, payoff, weights
    gc.collect()

    return summary, detail, weights_df


# ---------------------------
# Grids
# ---------------------------
def base_param_template(cfg_runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    tbm_defaults = get_tbm_optimizer_defaults(cfg_runtime if cfg_runtime is not None else CFG)
    return {
        "tp_abs_lo_pct": float(tbm_defaults["tp_abs_lo_pct"]),
        "tp_abs_hi_pct": float(tbm_defaults["tp_abs_hi_pct"]),
        "sl_abs_lo_pct": float(tbm_defaults["sl_abs_lo_pct"]),
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
        "min_net_rr": 0.7,
        "min_tp_hit_rate": 0.01,
        "max_timeout_rate": 0.95,
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
        "quantile_window": 720,  # 30 days
        "tp_quantile": 0.95,
    }


def stage1_grid(cfg_runtime: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    cfgs = []
    # Standard ATR Multiplier Grid
    for k_tp, sl_as_tp, regime_model, h_scaling in product(
        [0.8, 1.0, 1.25, 1.6, 2.0],
        [0.4, 0.5, 0.6, 0.7],
        ["none", "mix"],
        ["none", "sqrt"],
    ):
        c = base_param_template(cfg_runtime)
        c.update(
            {
                "mode": "atr_mult_rr",
                "tp_method": "atr_mult",
                "sl_method": "tp_pct",
                "k_tp": float(k_tp),
                "sl_as_tp_pct": float(sl_as_tp),
                "tp_regime_model": regime_model,
                "sl_regime_model": regime_model,
                "horizon_scaling": h_scaling,
            }
        )
        cfgs.append(c)

    # Rolling Quantile Grid (Alternative to k_tp)
    # Replaces k_tp iteration with tp_quantile iteration
    for tp_quantile, sl_as_tp, regime_model, h_scaling in product(
        [0.90, 0.95, 0.98, 0.99],
        [0.4, 0.5, 0.6, 0.7],
        ["none", "mix"],
        ["none", "sqrt"],
    ):
        c = base_param_template(cfg_runtime)
        c.update(
            {
                "mode": "rolling_quantile_rr",
                "tp_method": "rolling_quantile",
                "sl_method": "tp_pct",
                "tp_quantile": float(tp_quantile),
                "quantile_window": 720, # Fixed 30-day window for stage 1
                "sl_as_tp_pct": float(sl_as_tp),
                "tp_regime_model": regime_model,
                "sl_regime_model": regime_model,
                "horizon_scaling": h_scaling,
            }
        )
        cfgs.append(c)

    return cfgs


def stage2_grids_from_stage1(winners: List[Dict[str, Any]], max_per_substage: int = 24) -> List[Dict[str, Any]]:
    """Hierarchical stage-2 generation to avoid curse of dimensionality.

    Substages are intentionally shallow and capped:
    2A) TP family refinement (incl. semi_atr_* modes)
    2B) SL/asymmetry refinement on top 2A set
    2C) Path dependence add-on only for top subset
    """
    if not winners:
        return []

    # 2A: refine TP family around winners (single-axis exploration)
    stage2a: List[Dict[str, Any]] = []
    for base in winners:
        base_tp_abs = float(base.get("tp_abs_pct", base.get("tp_base_pct", 0.02)))
        for mode, k_tp in product(["atr_mult", "semi_atr_mult", "atr_norm", "semi_atr_norm"], [0.9, 1.0, 1.25]):
            c = dict(base)
            c.update(
                {
                    "mode": f"{mode}_refine",
                    "tp_method": mode,
                    "k_tp": float(k_tp),
                    "tp_abs_pct": base_tp_abs,
                    "tp_base_pct": base_tp_abs,
                    "base_atr_window": int(base.get("base_atr_window", 168)),
                }
            )
            stage2a.append(c)

    # Dedup + cap to keep dimensionality controlled.
    uniq2a = {config_id(c): c for c in stage2a}
    stage2a = list(uniq2a.values())[:max_per_substage]

    # 2B: SL geometry / asymmetry (single-axis from 2A candidates)
    stage2b: List[Dict[str, Any]] = []
    for base in stage2a:
        for sl_method in ["tp_pct", "atr_mult", "absolute"]:
            c = dict(base)
            c["sl_method"] = sl_method
            c["mode"] = f"{base.get('mode', 'stage2')}_sl"
            if sl_method == "tp_pct":
                for v in [0.45, 0.6]:
                    cc = dict(c)
                    cc["sl_as_tp_pct"] = float(v)
                    stage2b.append(cc)
            elif sl_method == "atr_mult":
                for ksl in [0.5, 0.8]:
                    for skew in [0.0, 0.1]:
                        cc = dict(c)
                        cc["k_sl"] = float(ksl)
                        cc["tp_side_skew"] = float(skew)
                        stage2b.append(cc)
            else:
                for sl_abs in [0.01, 0.015]:
                    cc = dict(c)
                    cc["sl_abs_pct"] = float(sl_abs)
                    stage2b.append(cc)

    uniq2b = {config_id(c): c for c in stage2b}
    stage2b = list(uniq2b.values())[:max_per_substage]

    # 2C: path dependence only on top slice from stage2b candidates.
    stage2c: List[Dict[str, Any]] = []
    for base in stage2b[: max(1, min(8, len(stage2b)) )]:
        for act_m, trail, decay in product([0, 30], [0.0, 0.5], ["none", "linear"]):
            c = dict(base)
            c.update(
                {
                    "mode": f"{base.get('mode', 'stage2')}_path",
                    "sl_activation_minutes": int(act_m),
                    "trail_sl_mult": float(trail),
                    "tp_time_decay": decay,
                }
            )
            stage2c.append(c)

    out = stage2a + stage2b + stage2c
    uniq = {config_id(c): c for c in out}
    return list(uniq.values())


def promote_stage1(stage1_results: pd.DataFrame, top_k: int = 10) -> List[str]:
    if stage1_results.empty:
        return []
    df = stage1_results.copy()
    df = df[df["hard_gate"] == True]
    if df.empty:
        return []
    bucket = max(1, top_k // 2)
    top_ic = df.sort_values("ic_payoff", ascending=False).head(bucket)
    top_score = df.sort_values("stage1_score", ascending=False).head(top_k - bucket)
    pick = pd.concat([top_ic, top_score], axis=0).drop_duplicates(subset=["config_id"])  # pareto-ish proxy
    pick = pick.sort_values(["stage1_score", "ic_payoff"], ascending=False).head(top_k)
    return pick["config_id"].tolist()


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


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    
    # Clear caches at the start of each run
    _clear_caches()
    
    tprint("Starting TBM parameter comparison run")
    runtime_cfg = apply_offline_optimizer_best_params(dict(CFG))
    
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
        features = _load_features_from_data_root(runtime_cfg)
        if features is None:
            raise ValueError("Could not auto-detect features. Please provide --features path.")
    
    # Auto-detect panel if not provided
    if args.panel:
        panel = load_panel(Path(args.panel))
    else:
        tprint("No --panel provided, auto-detecting from data_root")
        panel = _load_panel_from_store(runtime_cfg)

    if panel is None:
        raise ValueError("Could not load panel data. Please provide --panel path.")

    artifacts = align_artifacts(panel, features, lookback_years=args.lookback_years)
    bucket_masks = build_bucket_masks(artifacts, cfg_runtime=runtime_cfg)
    tprint(
        f"Artifacts + buckets ready: bars={len(artifacts.panel['close'])}, symbols={len(artifacts.panel['close'].columns)} "
        f"bucket_masks={list(bucket_masks.keys())} mem_peak_mb={_memory_snapshot_mb():.1f}"
    )

    # Clear caches after loading data
    _clear_caches()

    # Use BoundedEvalCache to prevent unbounded eval_cache growth
    layer1_cache: Dict[str, Any] = LRUCache(max_size=10)
    layer2_cache: Dict[str, Any] = LRUCache(max_size=10)
    eval_cache: BoundedEvalCache = BoundedEvalCache(max_size=10)
    
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
            # Create parquet writer for streaming
            import pyarrow as pa
            import pyarrow.parquet as pq
            schema = pa.Schema.from_pandas(weights_df)
            weights_writer = pq.ParquetWriter(weights_path, schema, compression='snappy')
        
        table = pa.Table.from_pandas(weights_df)
        weights_writer.write_table(table)
        tprint(f"Streamed {len(weights_df):,} weight rows to {weights_path}")

    stage1_cfgs = stage1_grid(runtime_cfg)
    barrier_defaults = get_barrier_factory_defaults(runtime_cfg)
    for _cfg in stage1_cfgs:
        _cfg.setdefault("k_tp", float(barrier_defaults["barrier_k_tp"]))
        _cfg.setdefault("sl_as_tp_pct", float(barrier_defaults["barrier_sl_base_mult"]))
        _cfg.setdefault("tp_abs_lo_pct", float(barrier_defaults["barrier_tp_lo"]))
        _cfg.setdefault("tp_abs_hi_pct", float(barrier_defaults["barrier_tp_hi"]))
        _cfg.setdefault("horizon_base", float(barrier_defaults["label_horizon_base"]))
    if args.quick:
        stage1_cfgs = stage1_cfgs[: max(1, args.max_configs)]
    tprint(
        f"Stage1 config count={len(stage1_cfgs)} quick={args.quick} horizons={args.horizons} "
        f"{_cache_pressure_summary(layer1_cache, layer2_cache, eval_cache)}"
    )

    stage1_rows = []
    details: Dict[str, Any] = {}
    total_weights_written = 0
    horizons = [2, 4, 8] if not args.horizons else [int(x) for x in args.horizons.split(",")]

    for i, cfg in enumerate(stage1_cfgs, 1):
        s, d, weights_df = evaluate_config(
            artifacts,
            cfg,
            horizons=horizons,
            bucket_masks=bucket_masks,
            layer1_cache=layer1_cache,
            layer2_cache=layer2_cache,
            eval_cache=eval_cache,
            detailed_slices=False,
            collect_weights=True,
        )
        stage1_rows.append(s)
        details[s["config_id"]] = d
        
        # Stream weights to parquet instead of accumulating in memory
        if weights_df is not None and not weights_df.empty:
            write_weights_streaming(weights_df)
            total_weights_written += len(weights_df)
            del weights_df  # Free memory immediately
        
        # Regular gc.collect() every iteration to prevent memory buildup
        gc.collect()
        
        if i % 5 == 0:
            top = max(stage1_rows, key=lambda x: x.get("stage1_score", -1e9)) if stage1_rows else {}
            tprint(
                f"[stage1] progress={i}/{len(stage1_cfgs)} best_cfg={top.get('config_id', 'n/a')} "
                f"best_stage1={top.get('stage1_score', float('nan')):.4f} "
                f"best_ic_payoff={top.get('ic_payoff', float('nan')):.4f} "
                f"mem_peak_mb={_memory_snapshot_mb():.1f} "
                f"{_cache_pressure_summary(layer1_cache, layer2_cache, eval_cache)}"
            )
            # More aggressive cache cleaning every 5 iterations
            _clear_caches()

    stage1_df = pd.DataFrame(stage1_rows)

    winners = []
    if args.winners:
        winners = [x.strip() for x in args.winners if x.strip()]
    elif args.stage2:
        winners = promote_stage1(stage1_df, top_k=args.top_k)

    stage2_df = pd.DataFrame()
    if args.stage2 and winners:
        tprint(f"Stage2 enabled with {len(winners)} winners from Stage1")
        id_to_cfg = {config_id(c): c for c in stage1_cfgs}
        base_cfgs = [id_to_cfg[w] for w in winners if w in id_to_cfg]
        stage2_cfgs = stage2_grids_from_stage1(base_cfgs, max_per_substage=args.max_stage2_configs)
        tprint(f"Stage2 config count={len(stage2_cfgs)} (max_per_substage={args.max_stage2_configs})")

        rows = []
        for _cfg in stage2_cfgs:
            _cfg.setdefault("k_tp", float(barrier_defaults["barrier_k_tp"]))
            _cfg.setdefault("sl_as_tp_pct", float(barrier_defaults["barrier_sl_base_mult"]))
            _cfg.setdefault("tp_abs_lo_pct", float(barrier_defaults["barrier_tp_lo"]))
            _cfg.setdefault("tp_abs_hi_pct", float(barrier_defaults["barrier_tp_hi"]))
            _cfg.setdefault("horizon_base", float(barrier_defaults["label_horizon_base"]))
        for i, cfg in enumerate(stage2_cfgs, 1):
            s, d, weights_df = evaluate_config(
                artifacts,
                cfg,
                horizons=horizons,
                bucket_masks=bucket_masks,
                layer1_cache=layer1_cache,
                layer2_cache=layer2_cache,
                eval_cache=eval_cache,
                detailed_slices=True,
                collect_weights=True,
            )
            rows.append(s)
            details[s["config_id"]] = d
            
            # Stream weights to parquet instead of accumulating in memory
            if weights_df is not None and not weights_df.empty:
                write_weights_streaming(weights_df)
                total_weights_written += len(weights_df)
                del weights_df  # Free memory immediately
            
            if i % 5 == 0:
                top2 = max(rows, key=lambda x: x.get("stage2_score", -1e9)) if rows else {}
                tprint(
                    f"[stage2] progress={i}/{len(stage2_cfgs)} best_cfg={top2.get('config_id', 'n/a')} "
                    f"best_stage2={top2.get('stage2_score', float('nan')):.4f} "
                    f"best_sortino={top2.get('sortino', float('nan')):.4f} "
                    f"mem_peak_mb={_memory_snapshot_mb():.1f} "
                    f"{_cache_pressure_summary(layer1_cache, layer2_cache, eval_cache)}"
                )
                gc.collect()

        stage2_df = pd.DataFrame(rows)

    out_df = stage1_df if stage2_df.empty else pd.concat([stage1_df, stage2_df], ignore_index=True)
    out_df = out_df.sort_values(["stage2_score", "stage1_score", "ic_payoff"], ascending=False)
    tprint(
        f"Scoring complete: total_rows={len(out_df)} stage1_rows={len(stage1_df)} stage2_rows={len(stage2_df)} "
        f"mem_peak_mb={_memory_snapshot_mb():.1f}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    detail_path = output_path.with_suffix(".json")
    with detail_path.open("w") as f:
        json.dump(details, f, indent=2)

    learnability_path = output_path.with_name("tbm__learnability_report.csv")
    learnability_df = _build_tbm_learnability_report_rows(out_df, details)
    if not learnability_df.empty:
        learnability_df.to_csv(learnability_path, index=False)
        tprint(f"Saved learnability CSV: {learnability_path}")

    # Close the streaming weights writer and report
    if weights_writer is not None:
        weights_writer.close()
        tprint(f"Saved sample weights (streaming): {weights_path} (total_rows={total_weights_written:,})")

    if not out_df.empty:
        best = out_df.iloc[0].to_dict()
        best_params = details.get(best.get("config_id"), {}).get("config", {})
        if isinstance(best_params, dict):
            save_best_params_csv(TBM_BEST_PARAMS_CSV, best_params, metadata={"source": "compare_tbm_parameters", "config_id": best.get("config_id")})
            tprint(f"Saved best params CSV: {TBM_BEST_PARAMS_CSV}")

    tprint(f"Saved CSV: {output_path}")
    tprint(f"Saved JSON: {detail_path}")
    if not out_df.empty:
        top = out_df.iloc[0]
        tprint(
            f"Best config summary: config_id={top.get('config_id', 'n/a')} "
            f"stage2={float(top.get('stage2_score', 0.0)):.4f} "
            f"stage1={float(top.get('stage1_score', 0.0)):.4f} "
            f"ic_payoff={float(top.get('ic_payoff', 0.0)):.4f} "
            f"sortino={float(top.get('sortino', 0.0)):.4f}"
        )
    tprint(
        f"Run completed in {time.perf_counter()-t0:.2f}s with mem_peak_mb={_memory_snapshot_mb():.1f} "
        f"{_cache_pressure_summary(layer1_cache, layer2_cache, eval_cache)}"
    )



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize/compare TBM parameter sets")
    # Features and panel are now auto-detected from CFG data_root, no longer required args
    p.add_argument("--features", default=None, help="Path to features directory (auto-detected from data_root if not set)")
    p.add_argument("--panel", default=None, help="Path to panel parquet or symbol parquet directory (auto-detected from data_root if not set)")
    p.add_argument("--output", default=str(REPORTS_DIR / "tbm_parameter_comparison.csv"), help="Output CSV path")
    p.add_argument("--quick", action="store_true", help="Quick stage1 subset")
    p.add_argument("--stage2", action="store_true", help="Run stage2 validation")
    p.add_argument("--top-k", type=int, default=10, help="Stage1 promotion top-k")
    p.add_argument("--winners", nargs="*", default=[], help="Explicit stage1 config IDs")
    p.add_argument("--horizons", default="2,4,8", help="Comma-separated horizons in hours")
    p.add_argument("--max-configs", type=int, default=20, help="Max configs when --quick")
    p.add_argument("--max-stage2-configs", type=int, default=24, help="Max configs per Stage2 substage (hierarchical cap)")
    p.add_argument("--lookback-years", type=int, default=2, help="Years of history to keep")
    p.add_argument("--weights-output", default="", help="Optional sample-weights parquet output path")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
