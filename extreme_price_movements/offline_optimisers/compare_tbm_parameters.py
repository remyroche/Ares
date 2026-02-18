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
    TBM_GEOMETRY_GRID_CSV,
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
TBM_CACHE_VERSION = 1


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
        dfs: Dict[str, pd.DataFrame] = {}
        for sym in train_syms:
            try:
                df = store.load(sym)
            except Exception as exc:
                tprint(f"Store load failed for {sym}: {exc}")
                continue
            if df is None or df.empty:
                continue
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(set(df.columns)):
                continue
            dfs[sym] = df
        if not dfs:
            tprint("Panel store load returned no usable symbols")
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
        if not isinstance(v, tuple) or len(v) != 3:
            continue
        tp_df, sl_df, _geom = v
        total += int(tp_df.memory_usage(index=True, deep=True).sum())
        total += int(sl_df.memory_usage(index=True, deep=True).sum())
    for v in layer2_cache.values():
        if not isinstance(v, tuple) or len(v) != 2:
            continue
        lbl, ret = v
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
            lbl = pd.read_parquet(cache_dir / item["label_file"])
            ret = pd.read_parquet(cache_dir / item["return_file"])
            geom = {"bound_saturation": float(item.get("bound_saturation", 0.0))}
            layer1_loaded[key] = (tp_df, sl_df, geom)
            layer2_loaded[key] = (lbl, ret)
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
            tp_df, sl_df, geom = layer1_cache[key]
            lbl, ret = layer2_cache[key]
            stem = _safe_filename_from_key(key)
            tp_file = f"{stem}_tp.parquet"
            sl_file = f"{stem}_sl.parquet"
            label_file = f"{stem}_label.parquet"
            return_file = f"{stem}_return.parquet"
            tp_df.to_parquet(cache_dir / tp_file, compression="zstd")
            sl_df.to_parquet(cache_dir / sl_file, compression="zstd")
            lbl.to_parquet(cache_dir / label_file, compression="zstd")
            ret.to_parquet(cache_dir / return_file, compression="zstd")
            entries.append(
                {
                    "key": key,
                    "tp_file": tp_file,
                    "sl_file": sl_file,
                    "label_file": label_file,
                    "return_file": return_file,
                    "bound_saturation": _safe_float(geom.get("bound_saturation"), 0.0)
                    if isinstance(geom, dict)
                    else 0.0,
                }
            )
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
def _read_symbol_parquet_dir(folder: Path) -> Dict[str, pd.DataFrame]:
    files = sorted(folder.glob("symbol=*.parquet"))
    if not files:
        files = sorted(folder.glob("symbol=*/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No symbol parquet files in {folder}")

    by_symbol_parts: Dict[str, List[pd.DataFrame]] = {}
    for f in files:
        raw_sym = None
        for part in f.parts:
            if part.startswith("symbol="):
                raw_sym = part.replace("symbol=", "")
                break
        if raw_sym is None:
            raw_sym = f.stem.replace("symbol=", "")
        df = pd.read_parquet(f)
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
                raise ValueError(f"Cannot infer timestamp index for {f}")

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

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
    # Use entry-time ATR only: shift by 1 bar so the barrier is set using only
    # data available at the moment the signal fires, not the current bar's ATR
    # (which would be look-ahead on 15m bars where ATR is computed on the closing price).
    atr = artifacts.features["atr_pct"].shift(1).clip(lower=1e-6)
    # Fill the first row (NaN after shift) with the second row to avoid all-NaN columns.
    atr = atr.bfill(limit=1)

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
        # median is already causal (rolling on shifted atr); shift(1) already applied above.
        med = atr.rolling(int(cfg.get("base_atr_window", 168)), min_periods=24).median().fillna(atr.median())
        tp = cfg["k_tp"] * (atr / (med + EPS)) * float(cfg["tp_base_pct"])
    elif tp_method == "semi_atr_norm":
        med = atr.rolling(int(cfg.get("base_atr_window", 168)), min_periods=24).median().fillna(atr.median())
        atrn_tp = cfg["k_tp"] * (atr / (med + EPS)) * float(cfg.get("tp_base_pct", 0.02))
        abs_tp = pd.DataFrame(float(cfg.get("tp_abs_pct", cfg.get("tp_base_pct", 0.02))), index=atr.index, columns=atr.columns)
        tp = 0.5 * atrn_tp + 0.5 * abs_tp
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
    candidate_filter = pd.DataFrame(True, index=c.index, columns=c.columns)
    if cfg_runtime is not None:
        candidate_defaults = get_candidate_filter_defaults(cfg_runtime)
        try:
            cf = select_trade_candidates_vectorized(
                artifacts.panel,
                artifacts.features,
                pct=float(candidate_defaults["train_extreme_pct_hourly"]),
                metric=metric,
                min_range_pct=float(candidate_defaults["train_min_range_pct"]),
                min_vol_zscore=float(candidate_defaults["train_min_vol_zscore"]),
                sign_consistency_min=float(candidate_defaults.get("min_feat_sign_consistency", 0.80)),
            )
            if cf is not None:
                candidate_filter = cf.astype(bool)
        except Exception:
            pass

    # Split candidates by move direction using the deviation metric.
    # "up" movers = top pct% (best performers): used by TF_long + MR_short
    # "down" movers = bottom pct% (worst performers): used by MR_long + TF_short
    if metric in feats:
        df_metric = feats[metric]
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
            lbl_s = lbl.stack(future_stack=True)
            stacked_idx = lbl_s.index
            label_arr = lbl_s.to_numpy(dtype=np.float32, copy=False)
            payoff_arr = ret.stack(future_stack=True).to_numpy(dtype=np.float32, copy=False)
            tp_arr = tp_df.stack(future_stack=True).to_numpy(dtype=np.float32, copy=False)
            sl_arr = sl_df.stack(future_stack=True).to_numpy(dtype=np.float32, copy=False)
            
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
            del label_arr, payoff_arr, tp_arr, sl_arr, stacked_idx, lbl_s
            gc.collect()

    events = pd.concat(events_rows, ignore_index=True)
    events["label"] = events["label"].astype(np.float32, copy=False)
    events["payoff"] = events["payoff"].astype(np.float32, copy=False)
    events["tp"] = events["tp"].astype(np.float32, copy=False)
    events["sl"] = events["sl"].astype(np.float32, copy=False)
    events["horizon"] = events["horizon"].astype(np.int16, copy=False)
    events["side"] = events["side"].astype("category")
    tprint(
        f"[eval:{cfg_id}] raw_events={len(events):,} "
        f"mem_peak_mb={_memory_snapshot_mb():.1f}"
    )
    # Per-(side,horizon) funnel: raw → prefilter → rr → final counts.
    # Printed before any further filtering so the user can see where events are lost.
    _funnel_rows = []
    for (_side, _h), _g in events.groupby(["side", "horizon"], observed=True):
        _n_raw = len(_g)
        _fee = float(cfg.get("fee_pct", 0.5)) / 100.0
        _slip = float(cfg.get("slip_buffer", 0.1)) / 100.0
        _tp_net = _g["tp"] - _fee - _slip
        _sl_net = _g["sl"] + _fee + _slip
        _rr = _tp_net / np.maximum(_sl_net, EPS)
        _min_rr = float(cfg.get("min_net_rr", 0.7))
        _n_rr = int((_rr >= _min_rr).sum())
        _funnel_rows.append({"side": _side, "h": int(_h), "n_raw": _n_raw, "n_rr": _n_rr})
    if _funnel_rows:
        _fn = pd.DataFrame(_funnel_rows).sort_values(["side", "h"])
        _parts = [f"{r.side}_H{r.h}: {r.n_raw:,}→{r.n_rr:,}" for _, r in _fn.iterrows()]
        _rr_fracs = [r.n_rr / max(r.n_raw, 1) for _, r in _fn.iterrows()]
        tprint(
            f"[eval:{cfg_id}] funnel(rr≥{float(cfg.get('min_net_rr',0.7)):.2f}) "
            + "  ".join(_parts)
            + f"  |  rr_kept min={min(_rr_fracs)*100:.1f}% median={float(np.median(_rr_fracs))*100:.1f}%"
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

    # Hard candidate pre-filter: keep only rows in the candidate mask.
    candidate_mask = bucket_map["Candidate"]
    pre_candidate_n = len(events)
    if np.any(candidate_mask):
        events = events.loc[candidate_mask].copy().reset_index(drop=True)
        stacked_index = pd.MultiIndex.from_arrays([events["ts"], events["symbol"]])
        stack_key = _index_cache_key(stacked_index)
        cache_bucket_stack = eval_cache.setdefault("bucket_stack", {})
        if stack_key in cache_bucket_stack:
            bucket_map = cache_bucket_stack[stack_key]
        else:
            bucket_map = {}
            for bname, bmask in bucket_masks.items():
                bucket_map[bname] = bmask.stack().reindex(stacked_index).fillna(False).to_numpy(dtype=bool)
            cache_bucket_stack[stack_key] = bucket_map
    else:
        events = events.iloc[0:0].copy()
    tprint(f"[eval:{cfg_id}] candidate_prefilter_kept={len(events):,}/{pre_candidate_n:,}")

    # Assign bucket from side × move_direction, matching _strategy_bucket_context:
    #   (long,  up)   → TF_long   (buy_momentum)
    #   (short, up)   → MR_short  (sell_rips)
    #   (long,  down) → MR_long   (buy_dips)
    #   (short, down) → TF_short  (sell_weakness)
    up_mask   = bucket_map["up"]
    down_mask = bucket_map["down"]
    side_arr  = events["side"].astype(str).to_numpy()
    bucket = np.full(len(events), "Global", dtype=object)
    bucket[(side_arr == "long")  & up_mask]   = "TF_long"
    bucket[(side_arr == "short") & up_mask]   = "MR_short"
    bucket[(side_arr == "long")  & down_mask] = "MR_long"
    bucket[(side_arr == "short") & down_mask] = "TF_short"
    events["bucket"] = pd.Categorical(bucket)

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
    events["vol_quintile"] = (q + 1).astype(np.int8, copy=False)

    regime = np.where(ratio < 0.85, "low_vol", np.where(ratio > 1.15, "high_vol", "medium_vol"))
    events["regime"] = pd.Categorical(regime)

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
    gc.collect()

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
    tprint(
        f"[eval:{cfg_id}] rr_filter_kept={len(events):,}/{pre_rr_n:,} min_net_rr={min_rr:.3f}"
    )
    if events.empty:
        summary = {
            "config_id": cfg_id,
            "mode": cfg.get("mode", "unknown"),
            "k_tp": cfg.get("k_tp"),
            "sl_method": cfg.get("sl_method"),
            "sl_as_tp_pct": cfg.get("sl_as_tp_pct"),
            "regime_model": cfg.get("tp_regime_model"),
            "horizon_scaling": cfg.get("horizon_scaling"),
            "ic_label": 0.0,
            "ic_label_bucket_mean": 0.0,
            "ic_payoff": 0.0,
            "ic_payoff_bucket_mean": 0.0,
            "ic_snr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "tp_hit_rate": 0.0,
            "sl_hit_rate": 0.0,
            "timeout_rate": 1.0,
            "ess": 0.0,
            "ess_full": float(full_n),
            "coverage": 0.0,
            "ic_std_time": 0.0,
            "ic_std_asset": 0.0,
            "worst_bucket_IC": 0.0,
            "stage1_score": -10.0,
            "stage2_score": -10.0,
            "brier": 0.0,
            "ece": 0.0,
            "monotonicity": 0.0,
            "oof_payoff_decile_spread": 0.0,
            "hard_gate": False,
            "pass_cells": 0,
            "total_cells": 0,
            "worst_bucket_coverage": 0.0,
        }
        detail = {
            "config": serialize_key(cfg),
            "feature_columns": [],
            "bucket_metrics": {},
            "bucket_horizon_metrics": {},
            "calibration": {"brier": 0.0, "ece": 0.0, "monotonicity": 0.0, "oof_payoff_decile_spread": 0.0},
        }
        tprint(f"[eval:done] {cfg_id} empty_after_filters hard_gate=False")
        return summary, detail, None

    pass_cells = 0
    # Pre-initialize all 12 canonical cells with n=0 defaults so cells with zero
    # surviving events are always materialized (not absent) in bucket_horizon_metrics.
    _DEFAULT_CELL_METRICS = {
        "n": 0, "tp_hit": 0.0, "sl_hit": 0.0, "timeout": 1.0,
        "bind": 0.0, "balance": 0.0, "sl_to_tp": float("nan"),
        "tp_mean": float("nan"), "sl_mean": float("nan"), "payoff_mean": float("nan"),
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
    _CANONICAL_BUCKETS = ["MR_long", "MR_short", "TF_long", "TF_short"]
    _CANONICAL_HORIZONS = [2, 4, 8]
    bucket_h_metrics = {
        (b, h): dict(_DEFAULT_CELL_METRICS)
        for b in _CANONICAL_BUCKETS
        for h in _CANONICAL_HORIZONS
    }
    total_cells = len(bucket_h_metrics)
    for (b, h), g in events.groupby(["bucket", "horizon"]):
        tp_hit = float((g["label"] == 1).mean())
        timeout = float((g["label"] == 0).mean())
        # Horizon-dependent timeout threshold: shorter horizons tolerate more timeouts.
        # If cfg explicitly sets max_timeout_rate (non-zero), use that; otherwise adaptive.
        if _max_timeout_base > 0.0:
            max_timeout_h = _max_timeout_base
        else:
            max_timeout_h = 0.90 + 0.025 * (int(h) / 8.0)
        ok = (len(g) >= min_raw) and (tp_hit >= min_tp_hit) and (timeout <= max_timeout_h)
        pass_cells += int(ok)
        sl_hit_cell = float((g["label"] == -1).mean())
        bind_cell = tp_hit + sl_hit_cell
        balance_cell = (min(tp_hit, sl_hit_cell) / max(max(tp_hit, sl_hit_cell), EPS))
        # tp_mean/sl_mean: actual barrier sizes in % for this bucket-horizon cell.
        # These are the values that will be passed to the production labeler.
        bucket_h_metrics[(b, h)] = {
            "n": int(len(g)),
            "tp_hit": tp_hit,
            "sl_hit": sl_hit_cell,
            "timeout": timeout,
            "bind": round(bind_cell, 4),
            "balance": round(balance_cell, 4),
            "sl_to_tp": round(sl_hit_cell / max(tp_hit, EPS), 4),
            "tp_mean": float(g["tp"].mean()),
            "sl_mean": float(g["sl"].mean()),
            "payoff_mean": float(g["payoff"].mean()),
            "max_timeout_threshold": round(max_timeout_h, 4),
            "ok": ok,
            # ic_payoff per cell filled in after OOF scoring below
            "ic_payoff": float("nan"),
        }

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
    decile_spread = oof_payoff_decile_spread(pred, payoff)
    tprint(
        f"[eval:{cfg_id}] model_scored n={len(pred):,} ic_payoff={_safe_spearman(pred, payoff):.4f} "
        f"ic_label={_safe_spearman(pred, y_signed):.4f} decile_spread={decile_spread:.6f}"
    )

    ic_label = _safe_spearman(pred, y_signed)
    ic_payoff = _safe_spearman(pred, payoff)

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
    for _bname, _bg in events.groupby("bucket"):
        _bp = pred[_bg.index.values]
        _bucket_top10_thresh[str(_bname)] = float(np.quantile(_bp, 0.90)) if len(_bp) >= 10 else float("inf")

    for (b, h), g in events.groupby(["bucket", "horizon"]):
        if (b, h) not in bucket_h_metrics:
            continue
        idx = g.index.values
        p_cell = pred[idx]
        y_tp_cell = (g["label"].values == 1).astype(np.float32)
        # IC on payoff
        cell_ic = _safe_spearman(p_cell, g["payoff"].values)
        # IC on label (signed: TP=1, timeout=0, SL=-1)
        cell_ic_label = _safe_spearman(p_cell, g["label"].values)
        # Fast AUC on TP classification
        n_pos = int(y_tp_cell.sum()); n_neg = len(y_tp_cell) - n_pos
        if n_pos > 0 and n_neg > 0:
            ranks_cell = pd.Series(p_cell).rank(method="average").to_numpy(np.float64)
            u_cell = ranks_cell[y_tp_cell == 1].sum() - n_pos * (n_pos + 1) / 2.0
            auc_cell = float(u_cell / (n_pos * n_neg))
        else:
            auc_cell = float("nan")
        # TP separation: P(TP|top10 score) - P(TP|rest) — threshold is per-bucket, not global.
        _b10_thresh = _bucket_top10_thresh.get(str(b), float("inf"))
        top_mask_cell = p_cell >= _b10_thresh
        tp_top_cell = float(y_tp_cell[top_mask_cell].mean()) if top_mask_cell.any() else float("nan")
        tp_rest_cell = float(y_tp_cell[~top_mask_cell].mean()) if (~top_mask_cell).any() else float("nan")
        tp_sep_cell = (tp_top_cell - tp_rest_cell) if not (math.isnan(tp_top_cell) or math.isnan(tp_rest_cell)) else float("nan")
        bucket_h_metrics[(b, h)]["ic_payoff"] = round(float(cell_ic), 5)
        bucket_h_metrics[(b, h)]["ic_label"] = round(float(cell_ic_label), 5)
        bucket_h_metrics[(b, h)]["auc_label"] = round(auc_cell, 4) if not math.isnan(auc_cell) else float("nan")
        bucket_h_metrics[(b, h)]["tp_sep_top10"] = round(tp_sep_cell, 5) if not math.isnan(tp_sep_cell) else float("nan")

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
        _payoff_cell = g["payoff"].values.astype(np.float64)
        _tp_mask_cell = g["label"].values == 1
        _sl_mask_cell = g["label"].values == -1
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
    for (b, h), g in events.groupby(["bucket", "horizon"]):
        if (b, h) not in bucket_h_metrics:
            continue
        idx = g.index.values
        p_cell = pred[idx]
        y_tp_cell = (g["label"].values == 1).astype(np.float32)
        # auc_bound: only bound events
        bound_mask = g["label"].values != 0
        if bound_mask.sum() >= 10:
            p_bound = p_cell[bound_mask]
            y_bound = y_tp_cell[bound_mask]
            n_pos_b = int(y_bound.sum()); n_neg_b = len(y_bound) - n_pos_b
            if n_pos_b > 0 and n_neg_b > 0:
                r_b = pd.Series(p_bound).rank(method="average").to_numpy(np.float64)
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
                    r_v = pd.Series(p_v).rank(method="average").to_numpy(np.float64)
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
        if len(p_cell) >= 10:
            _brier_cell = float(np.mean((p_cell - y_bin_cell) ** 2))
            _ece_cell = expected_calibration_error(y_bin_cell, p_cell)
            _dec_cell = pd.qcut(pd.Series(p_cell), min(10, len(p_cell) // 2), labels=False, duplicates="drop") if len(p_cell) >= 20 else pd.Series(np.zeros(len(p_cell), dtype=int))
            _pay_cell = g["payoff"].values.astype(np.float64)
            _pay_by_dec = pd.DataFrame({"d": _dec_cell, "p": _pay_cell}).groupby("d")["p"].mean()
            _mono_cell = float((np.diff(_pay_by_dec.values) >= 0).mean()) if len(_pay_by_dec) > 1 else float("nan")
        else:
            _brier_cell = float("nan")
            _ece_cell = float("nan")
            _mono_cell = float("nan")
        bucket_h_metrics[(b, h)]["brier"] = round(_brier_cell, 5) if not math.isnan(_brier_cell) else float("nan")
        bucket_h_metrics[(b, h)]["ece"] = round(_ece_cell, 5) if not math.isnan(_ece_cell) else float("nan")
        bucket_h_metrics[(b, h)]["monotonicity"] = round(_mono_cell, 4) if not math.isnan(_mono_cell) else float("nan")

        # Per-cell IC stability: std of monthly IC(payoff) and per-asset IC(payoff).
        _ts_cell = g["ts"].values
        _pay_cell_f = g["payoff"].values.astype(np.float64)
        _p_cell_f = p_cell.astype(np.float64)
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
    cell_ap_lifts = [v["ap_lift"] for v in bucket_h_metrics.values() if not math.isnan(v.get("ap_lift", float("nan")))]
    cell_tp_over_sls = [v["tp_over_sl"] for v in bucket_h_metrics.values() if not math.isnan(v.get("tp_over_sl", float("nan")))]
    cell_dir_sups = [v["dir_superiority_top_decile"] for v in bucket_h_metrics.values() if not math.isnan(v.get("dir_superiority_top_decile", float("nan")))]
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

    # Config-level bind/balance (aggregate across all events).
    tp_hit_agg = float((events["label"] == 1).mean()) if len(events) else 0.0
    sl_hit_agg = float((events["label"] == -1).mean()) if len(events) else 0.0
    bind_agg = tp_hit_agg + sl_hit_agg
    balance_agg = min(tp_hit_agg, sl_hit_agg) / max(max(tp_hit_agg, sl_hit_agg), EPS)
    sl_to_tp_agg = sl_hit_agg / max(tp_hit_agg, EPS)

    # Degeneracy flags.
    timeout_agg = float((events["label"] == 0).mean()) if len(events) else 1.0
    flag_degenerate_timeout = timeout_agg > 0.85
    flag_degenerate_sl = sl_hit_agg > 0.60
    flag_degenerate_tp = tp_hit_agg < 0.05

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

    # brier/ece/mono/ic_std_time/ic_std_asset: derived from per-cell values (already in bucket_h_metrics)
    # rather than computed globally (which would mix all 4 buckets × 3 horizons).
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
    min_cell_mono = float(np.min(_cell_monos)) if _cell_monos else float("nan")

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
        - 0.2 * float((events["label"] == 0).mean() if len(events) else 1.0)
    )

    # Learnability-focused stage2_score.
    # Primary: AUC on TP classification (median across cells) + TP separation in top-10% score.
    # Secondary: sortino as payoff quality guard, ic_snr as stability.
    # Degeneracy penalties: low coverage/bind (multiplicative), top10_vs_rest<0, min_cell_ic_label<-0.02.
    _auc_term = (median_cell_auc - 0.5) * 4.0 if not math.isnan(median_cell_auc) else 0.0  # centre at 0.5, scale to ~[-2,+2]
    _sep_term = median_cell_tp_sep * 10.0                  # TP sep typically 0-0.05 → scale to 0-0.5
    # Secondary objective: directional superiority in top decile (E[r|TP,top10] / abs(E[r|SL,top10])).
    # Bonus for dir_sup > 1.0 (TP payoff exceeds SL loss in top decile), capped at +0.2.
    _dir_sup_term = 0.0
    if not math.isnan(median_cell_dir_superiority):
        _dir_sup_term = float(np.clip((median_cell_dir_superiority - 1.0) * 0.2, -0.2, 0.2))
    # Secondary objective: payoff edge (E[r|TP] / abs(E[r|SL])).
    # Bonus for tp_over_sl > 1.05, capped at +0.1.
    _tp_over_sl_term = 0.0
    if not math.isnan(median_cell_tp_over_sl):
        _tp_over_sl_term = float(np.clip((median_cell_tp_over_sl - 1.0) * 0.1, -0.1, 0.1))
    stage2_score = (
        0.35 * _auc_term
        + 0.25 * _sep_term
        + 0.15 * sortino
        + 0.10 * ic_snr
        + 0.10 * _dir_sup_term
        + 0.05 * _tp_over_sl_term
        - 0.10 * float(ic_time.std() if ic_time.size else 0.0)
    )
    # Multiplicative penalties: prevent degenerate low-n or pure-timeout configs from winning.
    stage2_score *= math.sqrt(max(coverage, 0.0))          # coverage ∈ [0,1] → sqrt shrinks low-n
    stage2_score *= max(bind_agg, 0.0)                     # bind ∈ [0,1] → zero-out pure-timeout configs
    # Additive penalties for learnability failures.
    if top10_vs_rest_spread < 0.0:
        stage2_score -= 0.05                               # top10 by score not better than rest → weak ranking
    if min_cell_ic_label < -0.02:
        stage2_score -= 0.05                               # at least one cell has inverted label IC
    if worst_bucket_ic < -0.1:
        stage2_score -= 0.5
    # Additive penalty for failing payoff edge: tp_over_sl < 1.0 globally means SL losses exceed TP gains.
    if not math.isnan(min_cell_tp_over_sl) and min_cell_tp_over_sl < 1.0:
        stage2_score -= 0.10

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
        "tp_hit_rate": tp_hit_agg,
        "sl_hit_rate": sl_hit_agg,
        "timeout_rate": timeout_agg,
        "bind": round(bind_agg, 4),
        "balance": round(balance_agg, 4),
        "sl_to_tp": round(sl_to_tp_agg, 4),
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
        # brier/ece/mono: median across cells (per-cell values in bucket_horizon_metrics)
        "brier": brier,
        "ece": ece,
        "monotonicity": mono,
        "min_cell_brier": round(min_cell_brier, 5) if not math.isnan(min_cell_brier) else float("nan"),
        "min_cell_mono": round(min_cell_mono, 4) if not math.isnan(min_cell_mono) else float("nan"),
        "oof_payoff_decile_spread": decile_spread,
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
    }

    detail = {
        "config": serialize_key(cfg),
        "feature_columns": feat_cols,
        "bucket_metrics": per_bucket,
        "bucket_horizon_metrics": {f"{k[0]}_H{k[1]}": v for k, v in bucket_h_metrics.items()},
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
    gc.collect()

    return summary, detail, weights_df


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
        "min_tp_hit_rate": 0.01,
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
    }


def stage1_grid(cfg_runtime: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    cfgs = []

    # atr_norm only: atr_mult_rr is consistently degenerate (timeout>0.89, tp_hit<0.06)
    # and wastes compute. atr_norm normalises by rolling median ATR so the barrier scales
    # with the event's volatility spike rather than the absolute ATR level.
    # tp_base_pct = base TP when atr/median_atr == 1.0.
    # base_atr_window: [336, 504, 672] (14d–28d). Prior run confirmed 672>504>336.
    # 84/168 excluded: too short, produces high timeout + low bind + low coverage.
    # 840 reserved for Stage 2A upward probe if winner lands at 672.
    for k_tp, tp_base, sl_as_tp, atr_win in product(
        [0.7, 1.0, 1.25, 1.6],
        [0.006, 0.010, 0.015],  # base TP at median-vol: 0.6%, 1.0%, 1.5%
        [0.4, 0.6],
        [336, 504, 672],   # 14d / 21d / 28d slow median reference (prior run: 672>504>336)
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

    # Dedup
    uniq = {config_id(c): c for c in cfgs}
    return list(uniq.values())


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
    return list(uniq.values())


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
_PROMOTE_MIN_AUC: float = 0.56        # min_cell_auc floor (lowered from 0.57 for crypto noise)
_PROMOTE_MIN_AP_LIFT: float = 1.25    # AP / base_rate — model must lift precision 25% above random
_PROMOTE_MIN_TP_OVER_SL: float = 1.05 # E[r|TP] / abs(E[r|SL]) — 5% payoff edge required
_PROMOTE_MAX_SL_TO_TP: float = 3      # SL-hit / TP-hit ratio cap — prevents SL-dominated labelers

# Tier definitions for the feasible-set builder.
# Each tier is a tuple of (cov_min, bind_min, timeout_max, tp_sep_min, auc_min, ap_lift_min, tp_over_sl_min).
# bind_min here is the GLOBAL aggregate gate; per-cell uses _PROMOTE_MIN_BIND_CELL.
# Tier 0: all gates active.
# Tiers 1-2: relax learnability gates (tp_sep, auc, ap_lift) but keep structural + payoff edge.
# Tier 3: structural-only — only reached when nothing passes any learnability gate.
_FEASIBLE_TIERS: List[Tuple[float, float, float, float, float, float, float]] = [
    # (cov_min, bind_min, timeout_max, tp_sep_min, auc_min, ap_lift_min, tp_over_sl_min)
    (_PROMOTE_MIN_COVERAGE, _PROMOTE_MIN_BIND, _PROMOTE_MAX_TIMEOUT, _PROMOTE_MIN_TP_SEP, _PROMOTE_MIN_AUC, _PROMOTE_MIN_AP_LIFT, _PROMOTE_MIN_TP_OVER_SL),  # Tier 0: full
    (_PROMOTE_MIN_COVERAGE, _PROMOTE_MIN_BIND, _PROMOTE_MAX_TIMEOUT, 0.02,                 0.54,              1.10,                 _PROMOTE_MIN_TP_OVER_SL),  # Tier 1: relax auc/sep/ap_lift
    (_PROMOTE_MIN_COVERAGE, _PROMOTE_MIN_BIND, _PROMOTE_MAX_TIMEOUT, 0.0,                  0.52,              0.0,                  _PROMOTE_MIN_TP_OVER_SL),  # Tier 2: relax further, keep payoff edge
    # Tier 3: structural-only — learnability + payoff gates fully dropped.
    (_PROMOTE_MIN_COVERAGE, _PROMOTE_MIN_BIND, _PROMOTE_MAX_TIMEOUT, 0.0,                  0.0,               0.0,                  0.0),
]


def _build_feasible_set(
    df: pd.DataFrame,
    per_cell: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """Build the feasible set using explicit tiers — no silent progressive relaxation.

    Returns (feasible_df, tier_used) where tier_used is 0 (strictest) to 3 (structural only).
    Structural gates (coverage/bind/timeout) are NEVER relaxed.
    Learnability gates (tp_sep, auc, ap_lift) relax across tiers.
    Payoff edge gate (tp_over_sl) is kept through Tier 2, dropped only at Tier 3.
    If no config passes even Tier 3, returns (df, -1).

    per_cell=True: use _PROMOTE_MIN_BIND_CELL (0.38) instead of _PROMOTE_MIN_BIND (0.50)
    for the bind gate, since H2 cells structurally have lower bind than the aggregate.
    Also applies sl_to_tp cap if the column is present.
    """
    _bind_floor = _PROMOTE_MIN_BIND_CELL if per_cell else _PROMOTE_MIN_BIND
    for tier, (cov_min, bind_min, to_max, sep_min, auc_min, ap_lift_min, tp_over_sl_min) in enumerate(_FEASIBLE_TIERS):
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
        if auc_min > 0.0 and "min_cell_auc" in df.columns:
            mask = mask & (df["min_cell_auc"].fillna(0.0) >= auc_min)
        if ap_lift_min > 0.0 and "min_cell_ap_lift" in df.columns:
            mask = mask & (df["min_cell_ap_lift"].fillna(0.0) >= ap_lift_min)
        if tp_over_sl_min > 0.0 and "min_cell_tp_over_sl" in df.columns:
            mask = mask & (df["min_cell_tp_over_sl"].fillna(0.0) >= tp_over_sl_min)
        # sl_to_tp cap: reject SL-dominated labelers (sl_hit / tp_hit > 2.04).
        if "sl_to_tp" in df.columns:
            mask = mask & (df["sl_to_tp"].fillna(999.0) <= _PROMOTE_MAX_SL_TO_TP)
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


def _diverse_subset(
    feasible_df: pd.DataFrame,
    details: Dict[str, Any],
    min_distance: float = 1.0,
    max_configs: int = 20,
) -> pd.DataFrame:
    """Greedy diversity selection: pick configs ensuring param distance >= min_distance
    from all previously selected configs.  feasible_df must already be sorted by
    learnability (best first).
    """
    selected_indices: List[int] = []
    selected_cfgs: List[Dict[str, Any]] = []

    for idx, row in feasible_df.iterrows():
        cid = row["config_id"]
        cfg_i = details.get(cid, {}).get("config", {})
        if not isinstance(cfg_i, dict):
            continue
        too_close = any(
            _param_distance(cfg_i, sel_cfg) < min_distance
            for sel_cfg in selected_cfgs
        )
        if not too_close:
            selected_indices.append(idx)
            selected_cfgs.append(cfg_i)
        if len(selected_indices) >= max_configs:
            break

    return feasible_df.loc[selected_indices] if selected_indices else feasible_df.head(1)


def _rank_by_learnability(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a feasible-set DataFrame by the canonical learnability ranking:
    min_auc (3dp bucketed) → min_tp_sep → cell_dispersion asc → timeout_range asc → min_auc_bound.
    Returns a new sorted DataFrame."""
    _min_auc = (df["min_cell_auc"].fillna(0.0) * 1000).round() / 1000
    _min_sep = df["min_cell_tp_sep"].fillna(0.0)
    _disp = df["cell_dispersion"].fillna(999.0)
    _to_r = df["timeout_range"].fillna(999.0) if "timeout_range" in df.columns else pd.Series(0.0, index=df.index)
    _min_auc_b = df["min_cell_auc_bound"].fillna(0.0) if "min_cell_auc_bound" in df.columns else _min_auc
    return df.assign(_ma=_min_auc, _ms=_min_sep, _d=_disp, _tr=_to_r, _mab=_min_auc_b).sort_values(
        ["_ma", "_ms", "_d", "_tr", "_mab"], ascending=[False, False, True, True, False]
    ).drop(columns=["_ma", "_ms", "_d", "_tr", "_mab"])


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
    for h in [2, 4, 8]
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
    _to_r_col = out_df["timeout_range"].fillna(999.0) if "timeout_range" in out_df.columns else pd.Series(0.0, index=out_df.index)
    struct_mask = (out_df["hard_gate"] == True) & (out_df["pass_cells"] == out_df["total_cells"]) & (_to_r_col <= _MAX_TIMEOUT_RANGE)
    base_df = out_df[struct_mask].copy()
    if base_df.empty:
        base_df = out_df[(out_df["hard_gate"] == True) & (out_df["pass_cells"] == out_df["total_cells"])].copy()
    if base_df.empty:
        base_df = out_df[out_df["hard_gate"] == True].copy()

    result: Dict[str, pd.DataFrame] = {}

    for cell_key in _CELL_KEYS:
        # Extract per-cell metrics for each config.
        cell_rows = []
        for _, row in base_df.iterrows():
            cid = row["config_id"]
            bh = details.get(cid, {}).get("bucket_horizon_metrics", {})
            cell_m = bh.get(cell_key, {})
            if not cell_m:
                continue
            auc_cell = cell_m.get("auc_label", float("nan"))
            sep_cell = cell_m.get("tp_sep_top10", float("nan"))
            timeout_cell = cell_m.get("timeout", 1.0)
            bind_cell = cell_m.get("bind", 0.0)
            coverage_cell = row.get("coverage", 0.0)  # config-level coverage as proxy
            ok_cell = cell_m.get("ok", False)
            ap_lift_cell = cell_m.get("ap_lift", float("nan"))
            tp_over_sl_cell = cell_m.get("tp_over_sl", float("nan"))
            dir_sup_cell = cell_m.get("dir_superiority_top_decile", float("nan"))
            cell_rows.append({
                "config_id": cid,
                "min_cell_auc": auc_cell if not math.isnan(auc_cell) else 0.0,
                "min_cell_tp_sep": sep_cell if not math.isnan(sep_cell) else 0.0,
                "cell_dispersion": row.get("cell_dispersion", 999.0),
                "timeout_range": row.get("timeout_range", 999.0),
                "min_cell_auc_bound": cell_m.get("auc_bound", float("nan")),
                "coverage": coverage_cell,
                "bind": bind_cell,
                "timeout_rate": timeout_cell,
                "ok": ok_cell,
                "min_cell_ap_lift": ap_lift_cell if not math.isnan(ap_lift_cell) else 0.0,
                "min_cell_tp_over_sl": tp_over_sl_cell if not math.isnan(tp_over_sl_cell) else 0.0,
                "min_cell_dir_superiority": dir_sup_cell if not math.isnan(dir_sup_cell) else 0.0,
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
        diverse = _diverse_subset(feasible, details, min_distance=min_distance, max_configs=max_configs_per_cell)
        # If diversity filter left only 1 config, relax min_distance to get a second.
        if len(diverse) < 2 and len(feasible) >= 2:
            diverse = _diverse_subset(feasible, details, min_distance=min_distance * 0.5, max_configs=max_configs_per_cell)
        if len(diverse) < 2 and len(feasible) >= 2:
            diverse = feasible.head(2)  # last resort: top-2 by learnability rank

        # Attach full out_df row data for downstream use.
        full_rows = out_df[out_df["config_id"].isin(diverse["config_id"])].copy()
        result[cell_key] = full_rows

        tprint(
            f"[per_cell] {cell_key}: {len(diverse)} diverse configs "
            f"(tier={tier}, feasible_pool={len(feasible)})"
        )

    return result


# ---------------------------
# Per-bucket feasible-set builder
# ---------------------------
_BUCKET_NAMES = ["TF_long", "TF_short", "MR_long", "MR_short"]
_HORIZONS = [2, 4, 8]


def _per_bucket_metrics_from_details(
    cid: str,
    details: Dict[str, Any],
    bucket_name: str,
    global_row: pd.Series,
) -> Dict[str, Any]:
    """Extract per-bucket aggregate metrics from bucket_horizon_metrics for one bucket.

    Uses only the 3 horizon cells belonging to `bucket_name` (e.g. "MR_long_H2/H4/H8").
    Falls back to global_row values for coverage/ess (which are config-level, not per-bucket).
    """
    bh = details.get(cid, {}).get("bucket_horizon_metrics", {})
    cell_keys = [f"{bucket_name}_H{h}" for h in _HORIZONS]
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
    payoffs   = [c.get("payoff_mean",  float("nan")) for c in cells]
    disps     = [c.get("payoff_mean",  float("nan")) for c in cells]

    timeout_vals = [t for t in timeouts if not math.isnan(t)]
    timeout_range = float(np.max(timeout_vals) - np.min(timeout_vals)) if len(timeout_vals) > 1 else 0.0
    bind_vals = [b for b in binds if not math.isnan(b)]
    bind_min = float(np.min(bind_vals)) if bind_vals else float("nan")
    sl_to_tp_max = float(np.max([s for s in sl_to_tps if not math.isnan(s)])) if any(not math.isnan(s) for s in sl_to_tps) else float("nan")

    payoff_vals = [p for p in payoffs if not math.isnan(p)]
    cell_dispersion = float(np.std(payoff_vals)) if len(payoff_vals) > 1 else 0.0

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
        "bind":               round(bind_min, 4)             if not math.isnan(bind_min)           else 0.0,
        "timeout_rate":       round(_safe_min(timeouts), 4)  if not math.isnan(_safe_min(timeouts)) else 1.0,
        "sl_to_tp":           round(sl_to_tp_max, 4)         if not math.isnan(sl_to_tp_max)       else 999.0,
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

    for bucket_name in _BUCKET_NAMES:
        bucket_rows = []
        for _, row in base_df.iterrows():
            cid = row["config_id"]
            bm = _per_bucket_metrics_from_details(cid, details, bucket_name, row)
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
        diverse = _diverse_subset(feasible, details, min_distance=min_distance, max_configs=max_configs_per_bucket)
        if len(diverse) < 2 and len(feasible) >= 2:
            diverse = _diverse_subset(feasible, details, min_distance=min_distance * 0.5, max_configs=max_configs_per_bucket)
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
                   f"{'tp_hit':>7}  {'sl_hit':>7}  {'timeout':>8}  {'bind':>6}  "
                   f"{'auc':>6}  {'auc_bnd':>7}  {'ic_lbl':>7}  {'tp_sep%':>8}  "
                   f"{'vol_lo':>6}  {'vol_hi':>6}  ok")
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
                    f"{m.get('tp_hit',0):>7.3f}  {m.get('sl_hit',0):>7.3f}  "
                    f"{m.get('timeout',0):>8.3f}  {m.get('bind',0):>6.3f}  "
                    f"{_fmt(m.get('auc_label', float('nan'))):>6}  "
                    f"{_fmt(m.get('auc_bound', float('nan'))):>7}  "
                    f"{_fmt(m.get('ic_label', float('nan'))):>7}  "
                    f"{(format(m.get('tp_sep_top10',0)*100,'+.2f') if not (isinstance(m.get('tp_sep_top10',float('nan')),float) and math.isnan(m.get('tp_sep_top10',float('nan')))) else '  n/a'):>8}pp  "
                    f"{_fmt(m.get('auc_vol_low', float('nan'))):>6}  "
                    f"{_fmt(m.get('auc_vol_high', float('nan'))):>6}  "
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
            _cfgs = [details.get(c, {}).get("config", {}) for c in _cids]
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


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    
    # Clear caches at the start of each run
    _clear_caches()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    # Inject candidate-threshold best params (min_feat_sign_consistency, train_extreme_pct_hourly,
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

    horizons = [2, 4, 8] if not args.horizons else [int(x) for x in args.horizons.split(",")]

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
        # Per-cell promotion: find best configs independently per (bucket, horizon) cell
        # (12 cells = 4 buckets × 3 horizons), then union as stage2 candidates.
        # This ensures stage2 explores geometries optimal for each specific cell,
        # since H2/H4/H8 have different optimal params and MR/TF have different dynamics.
        per_cell_winners = promote_stage1_per_cell(stage1_df, details, top_k_per_cell=max(2, args.top_k // 4))
        winners_set: set = set()
        for cell_key, cwinners in per_cell_winners.items():
            if cwinners:
                tprint(f"[promote] {cell_key}: {len(cwinners)} winners → {cwinners}")
            winners_set.update(cwinners)
        # Also include global top-k as fallback (covers configs good across all cells).
        global_winners = promote_stage1(stage1_df, top_k=max(3, args.top_k // 2))
        winners_set.update(global_winners)
        winners = list(winners_set)
        tprint(f"[promote] Total stage2 candidates: {len(winners)} (per-cell union across 12 cells + global fallback)")

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
    # Two-level learnability sort:
    # Level 1 (hard flags, binary): hard_gate / timeout_range_ok / all_cells_pass /
    #   geometry-health gates (coverage, bind, timeout, min_tp_sep, min_auc floors).
    # Level 2 (continuous ranking among survivors):
    #   min_auc desc → min_tp_sep desc → cell_dispersion asc → timeout_range asc
    #   → min_auc_bound desc (tie-breaker only; fragile without healthy bind/coverage).
    out_df["_all_cells_pass"] = (out_df["pass_cells"] == out_df["total_cells"]).astype(int)
    out_df["_to_range_ok"] = (out_df["timeout_range"].fillna(999.0) <= _MAX_TIMEOUT_RANGE).astype(int) if "timeout_range" in out_df.columns else 1
    # Geometry-health gate flags (soft in sort — hard in promote/best-params)
    _cov_ok = (out_df["coverage"] >= _PROMOTE_MIN_COVERAGE).astype(int) if "coverage" in out_df.columns else pd.Series(1, index=out_df.index)
    _bind_ok = (out_df["bind"] >= _PROMOTE_MIN_BIND).astype(int) if "bind" in out_df.columns else pd.Series(1, index=out_df.index)
    _to_ok = (out_df["timeout_rate"] <= _PROMOTE_MAX_TIMEOUT).astype(int) if "timeout_rate" in out_df.columns else pd.Series(1, index=out_df.index)
    _sep_ok = (out_df["min_cell_tp_sep"].fillna(0.0) >= _PROMOTE_MIN_TP_SEP).astype(int) if "min_cell_tp_sep" in out_df.columns else pd.Series(1, index=out_df.index)
    _auc_ok = (out_df["min_cell_auc"].fillna(0.0) >= _PROMOTE_MIN_AUC).astype(int) if "min_cell_auc" in out_df.columns else pd.Series(1, index=out_df.index)
    out_df["_health_flags"] = _cov_ok + _bind_ok + _to_ok + _sep_ok + _auc_ok  # 0-5; sort desc
    # Bucket min_auc to 3dp so configs that differ by <0.001 are treated as tied
    # and min_tp_sep (the more discriminating learnability signal) breaks the tie.
    _min_auc_col = (out_df["min_cell_auc"].fillna(0.0) * 1000).round() / 1000
    _min_sep_col = out_df["min_cell_tp_sep"].fillna(0.0)
    _disp_col = out_df["cell_dispersion"].fillna(999.0)
    _to_range_col = out_df["timeout_range"].fillna(999.0) if "timeout_range" in out_df.columns else pd.Series(0.0, index=out_df.index)
    _min_auc_b_col = out_df["min_cell_auc_bound"].fillna(0.0) if "min_cell_auc_bound" in out_df.columns else _min_auc_col
    out_df = out_df.assign(
        _min_auc_sort=_min_auc_col,
        _min_sep_sort=_min_sep_col,
        _disp_sort=_disp_col,
        _to_range_sort=_to_range_col,
        _min_auc_b_sort=_min_auc_b_col,
    ).sort_values(
        ["hard_gate", "_to_range_ok", "_all_cells_pass", "_health_flags",
         "_min_auc_sort", "_min_sep_sort", "_disp_sort", "_to_range_sort", "_min_auc_b_sort"],
        ascending=[False, False, False, False, False, False, True, True, False],
    ).drop(columns=["_all_cells_pass", "_to_range_ok", "_health_flags",
                    "_min_auc_sort", "_min_sep_sort", "_disp_sort", "_to_range_sort", "_min_auc_b_sort"])
    tprint(
        f"Scoring complete: total_rows={len(out_df)} stage1_rows={len(stage1_df)} stage2_rows={len(stage2_df)} "
        f"mem_peak_mb={_memory_snapshot_mb():.1f}"
    )

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

        # ── Step 2: tiered feasible-set + learnability ranking ────────────────
        _best_pool, _tier = _build_feasible_set(_struct_pool)
        _best_pool = _rank_by_learnability(_best_pool)
        tprint(f"Global feasible pool: {len(_best_pool)} configs (tier={_tier})")

        # ── Step 3: diversity control on global pool ──────────────────────────
        _diverse_pool = _diverse_subset(_best_pool, details, min_distance=1.0, max_configs=20)
        tprint(f"Global diverse pool: {len(_diverse_pool)} configs after diversity filter")

        # ── Step 4: global best (fallback for downstream consumers) ──────────
        best_cid = _best_pool.iloc[0]["config_id"]
        best_params = details.get(best_cid, {}).get("config", {})
        if isinstance(best_params, dict):
            best_params = dict(best_params)
            raw_mode = best_params.get("mode", "")
            canonical_mode = raw_mode.split("_2A")[0].split("_refine")[0]
            best_params["mode"] = canonical_mode
            save_best_params_csv(TBM_BEST_PARAMS_CSV, best_params, metadata={"source": "compare_tbm_parameters", "config_id": best_cid})
            tprint(f"Saved global best params CSV: {TBM_BEST_PARAMS_CSV} (best={best_cid})")

        # ── Step 5: per-(bucket, horizon) feasible sets — canonical output ───
        # 12 cells = 4 buckets × 3 horizons; up to max_configs_per_cell diverse configs per cell.
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
            for rank_i, (_, crow) in enumerate(cell_df.iterrows(), 1):
                cid = crow["config_id"]
                cfg_i = details.get(cid, {}).get("config", {})
                if not isinstance(cfg_i, dict):
                    continue
                bh_i = details.get(cid, {}).get("bucket_horizon_metrics", {})
                cell_m_i = bh_i.get(cell_key, {})
                _grid_rows.append({
                    "cell_key": cell_key,
                    "bucket": _cell_bucket,
                    "config_id": cid,
                    "rank": rank_i,
                    "k_tp": float(cfg_i.get("k_tp", float("nan"))),
                    "sl_as_tp_pct": float(cfg_i.get("sl_as_tp_pct", float("nan"))),
                    "base_atr_window": int(cfg_i.get("base_atr_window", _fallback_window)),
                    "tp_base_pct": float(cfg_i.get("tp_base_pct", float("nan"))),
                    "tp_abs_lo_pct": float(cfg_i.get("tp_abs_lo_pct", float("nan"))),
                    "sl_abs_lo_pct": float(cfg_i.get("sl_abs_lo_pct", float("nan"))),
                    "mode": str(cfg_i.get("mode", "")).split("_2A")[0].split("_refine")[0],
                    "cell_auc": float(cell_m_i.get("auc_label", float("nan"))),
                    "cell_auc_bound": float(cell_m_i.get("auc_bound", float("nan"))),
                    "cell_tp_sep": float(cell_m_i.get("tp_sep_top10", float("nan"))),
                    "cell_timeout": float(cell_m_i.get("timeout", float("nan"))),
                    "cell_bind": float(cell_m_i.get("bind", float("nan"))),
                    "cell_ap_lift": float(cell_m_i.get("ap_lift", float("nan"))),
                    "cell_tp_over_sl": float(cell_m_i.get("tp_over_sl", float("nan"))),
                    "cell_brier": float(cell_m_i.get("brier", float("nan"))),
                    "cell_ece": float(cell_m_i.get("ece", float("nan"))),
                    "cell_monotonicity": float(cell_m_i.get("monotonicity", float("nan"))),
                    "cell_ic_std_time": float(cell_m_i.get("ic_std_time", float("nan"))),
                    "cell_ic_std_asset": float(cell_m_i.get("ic_std_asset", float("nan"))),
                    "cell_ic_payoff": float(cell_m_i.get("ic_payoff", float("nan"))),
                    "cell_ic_label": float(cell_m_i.get("ic_label", float("nan"))),
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
                "sl_as_tp_pct": float(_fb_params.get("sl_as_tp_pct", float("nan"))),
                "base_atr_window": int(_fb_params.get("base_atr_window", _fallback_window)),
                "tp_base_pct": float(_fb_params.get("tp_base_pct", float("nan"))),
                "tp_abs_lo_pct": float(_fb_params.get("tp_abs_lo_pct", float("nan"))),
                "sl_abs_lo_pct": float(_fb_params.get("sl_abs_lo_pct", float("nan"))),
                "mode": str(_fb_params.get("mode", "")).split("_2A")[0].split("_refine")[0],
                "cell_auc": float("nan"), "cell_auc_bound": float("nan"),
                "cell_tp_sep": float("nan"), "cell_timeout": float("nan"),
                "cell_bind": float("nan"), "cell_ap_lift": float("nan"),
                "cell_tp_over_sl": float("nan"), "cell_brier": float("nan"),
                "cell_ece": float("nan"), "cell_monotonicity": float("nan"),
                "cell_ic_std_time": float("nan"), "cell_ic_std_asset": float("nan"),
                "cell_ic_payoff": float("nan"), "cell_ic_label": float("nan"),
                "cell_n": 0,
            })

        if _grid_rows:
            _grid_df = pd.DataFrame(_grid_rows).dropna(subset=["k_tp", "sl_as_tp_pct"])
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

    tprint(f"Saved CSV: {output_path}")
    tprint(f"Saved JSON: {detail_path}")
    if not out_df.empty:
        _print_winning_geometry_summary(out_df, details, per_cell_grids=per_cell_grids, top_k=5)

    # Per-cell geometry quality report
    try:
        from extreme_price_movements.reports.bucket_report import report_compare_tbm
        import re as _re
        _run_id = _re.sub(r"[^0-9_]", "", str(Path(output_path).stem)) or "tbm_run"
        _rp = report_compare_tbm(_run_id, str(TBM_GEOMETRY_GRID_CSV))
        tprint(f"TBM geometry bucket report: {_rp}")
    except Exception as _rpe:
        tprint(f"WARNING: TBM geometry bucket report failed: {_rpe}")

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
    p.add_argument("--stage2", dest="stage2", action="store_true", help="Run stage2 validation")
    p.add_argument("--no-stage2", dest="stage2", action="store_false", help="Disable stage2 validation")
    p.set_defaults(stage2=True)
    p.add_argument("--top-k", type=int, default=10, help="Stage1 promotion top-k")
    p.add_argument("--winners", nargs="*", default=[], help="Explicit stage1 config IDs")
    p.add_argument("--horizons", default="2,4,8", help="Comma-separated horizons in hours")
    p.add_argument("--max-configs", type=int, default=20, help="Max configs when --quick")
    p.add_argument("--max-stage2-configs", type=int, default=24, help="Max configs per Stage2 substage (hierarchical cap)")
    p.add_argument("--lookback-years", type=int, default=2, help="Years of history to keep")
    p.add_argument("--weights-output", default="", help="Optional sample-weights parquet output path")
    p.add_argument("--tbm-cache-max-mb", type=int, default=1536, help="Max on-disk persisted TBM cache size in MB")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
