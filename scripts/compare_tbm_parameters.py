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
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extreme_price_movements.data_store import to_panel
from extreme_price_movements.labeling import compute_triple_barrier_labels
from extreme_price_movements.config import CFG, TEST_FEATURE_KEYS


EPS = 1e-12

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


@dataclass
class RunArtifacts:
    panel: Dict[str, pd.DataFrame]
    features: Dict[str, pd.DataFrame]


def _subsample_symbols(symbols: Sequence[str]) -> List[str]:
    """Deterministic symbol subsample: alphabetical, keep every 2nd token."""
    syms_sorted = sorted(set(map(str, symbols)))
    return syms_sorted[::2] if syms_sorted else []


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

    return RunArtifacts(panel=p_out, features=f_out)


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
def build_bucket_masks(artifacts: RunArtifacts) -> Dict[str, pd.DataFrame]:
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
    eval_cache: Dict[str, Any],
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
    eval_cache: Dict[str, Any],
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
    n_folds: int = 5,
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
    n_folds: int = 5,
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
    eval_cache: Dict[str, Any],
    detailed_slices: bool = False,
    collect_weights: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[pd.DataFrame]]:
    events_rows: List[pd.DataFrame] = []

    for h in horizons:
        for side in ["long", "short"]:
            # Key1: Barriers depends only on geometric params.
            barrier_cfg = _get_barrier_params(cfg)
            key1 = json.dumps(serialize_key({"h": h, "side": side, "cfg": barrier_cfg}), sort_keys=True)

            if key1 not in layer1_cache:
                tp_df, sl_df, geom_stats = build_barriers(artifacts, cfg, h, side)
                layer1_cache[key1] = (tp_df, sl_df, geom_stats)
            tp_df, sl_df, geom_stats = layer1_cache[key1]

            # Key2: Labels depends fully on barriers (key1) + horizon/side (in key1).
            # Note: compute_triple_barrier_labels uses JIT logic that may interpret TP
            # as trailing activation. It does NOT use sl_activation_minutes.
            # So key2 is effectively just key1.
            key2 = key1

            if key2 not in layer2_cache:
                lbl, ret = compute_triple_barrier_labels(artifacts.panel, tp_df, sl_df, h, side=side)
                layer2_cache[key2] = (lbl, ret)
            lbl, ret = layer2_cache[key2]

            df = pd.DataFrame(
                {
                    "label": lbl.stack(),
                    "payoff": ret.stack(),
                    "tp": tp_df.stack(),
                    "sl": sl_df.stack(),
                }
            )
            df.index.names = ["ts", "symbol"]
            df = df.dropna(subset=["label", "payoff", "tp", "sl"]).reset_index()
            df["side"] = side
            df["horizon"] = h
            df["bound_saturation"] = geom_stats["bound_saturation"]
            events_rows.append(df)

    events = pd.concat(events_rows, ignore_index=True)

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

    events = events[events["net_rr"] >= min_rr].reset_index(drop=True)

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

    cfg_id = config_id(cfg)

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

    del events, pred, y_signed, y_bin, payoff, weights
    gc.collect()

    return summary, detail, weights_df


# ---------------------------
# Grids
# ---------------------------
def base_param_template() -> Dict[str, Any]:
    return {
        "tp_abs_lo_pct": 0.005,
        "tp_abs_hi_pct": 0.08,
        "sl_abs_lo_pct": 0.005,
        "sl_abs_hi_pct": 0.08,
        "tp_mult_lo": 0.5,
        "tp_mult_hi": 3.0,
        "sl_mult_lo": 0.3,
        "sl_mult_hi": 2.0,
        "mix_weight": 0.5,
        "horizon_alpha": 0.5,
        "horizon_base": 4,
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
        "fee_pct": 0.5,
        "slip_buffer": 0.1,
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
        "tp_abs_pct": 0.02,
        "tp_base_pct": 0.02,
        "base_atr_window": 168,
    }


def stage1_grid() -> List[Dict[str, Any]]:
    cfgs = []
    for k_tp, sl_as_tp, regime_model, h_scaling in product(
        [0.8, 1.0, 1.25, 1.6, 2.0],
        [0.4, 0.5, 0.6, 0.7],
        ["none", "mix"],
        ["none", "sqrt"],
    ):
        c = base_param_template()
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
def run(args: argparse.Namespace) -> None:
    features = load_features(Path(args.features))
    panel = load_panel(Path(args.panel)) if args.panel else None

    if panel is None:
        raise ValueError("--panel is required for TBM optimization")

    artifacts = align_artifacts(panel, features, lookback_years=args.lookback_years)
    bucket_masks = build_bucket_masks(artifacts)

    # Use LRUCache to prevent OOM on large grids
    layer1_cache: Dict[str, Any] = LRUCache(max_size=40)
    layer2_cache: Dict[str, Any] = LRUCache(max_size=40)
    eval_cache: Dict[str, Any] = {} # eval cache is small (bucket stack), no need for LRU

    stage1_cfgs = stage1_grid()
    if args.quick:
        stage1_cfgs = stage1_cfgs[: max(1, args.max_configs)]

    stage1_rows = []
    details: Dict[str, Any] = {}
    collected_weights: List[pd.DataFrame] = []
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
        if weights_df is not None and not weights_df.empty:
            collected_weights.append(weights_df)
        if i % 5 == 0:
            print(f"[stage1] {i}/{len(stage1_cfgs)} done")
            gc.collect()

    stage1_df = pd.DataFrame(stage1_rows)

    winners = []
    if args.winners:
        winners = [x.strip() for x in args.winners if x.strip()]
    elif args.stage2:
        winners = promote_stage1(stage1_df, top_k=args.top_k)

    stage2_df = pd.DataFrame()
    if args.stage2 and winners:
        id_to_cfg = {config_id(c): c for c in stage1_cfgs}
        base_cfgs = [id_to_cfg[w] for w in winners if w in id_to_cfg]
        stage2_cfgs = stage2_grids_from_stage1(base_cfgs, max_per_substage=args.max_stage2_configs)

        rows = []
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
            if weights_df is not None and not weights_df.empty:
                collected_weights.append(weights_df)
            if i % 5 == 0:
                print(f"[stage2] {i}/{len(stage2_cfgs)} done")
                gc.collect()

        stage2_df = pd.DataFrame(rows)

    out_df = stage1_df if stage2_df.empty else pd.concat([stage1_df, stage2_df], ignore_index=True)
    out_df = out_df.sort_values(["stage2_score", "stage1_score", "ic_payoff"], ascending=False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    detail_path = output_path.with_suffix(".json")
    with detail_path.open("w") as f:
        json.dump(details, f, indent=2)

    if collected_weights:
        weights_path = Path(args.weights_output) if args.weights_output else output_path.with_suffix(".weights.parquet")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        all_weights = pd.concat(collected_weights, ignore_index=True)
        all_weights.to_parquet(weights_path, index=False)
        print(f"Saved sample weights: {weights_path}")

    print(f"Saved CSV: {output_path}")
    print(f"Saved JSON: {detail_path}")



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize/compare TBM parameter sets")
    p.add_argument("--features", required=True, help="Path to features directory (symbol=*.parquet)")
    p.add_argument("--panel", required=True, help="Path to panel parquet or symbol parquet directory")
    p.add_argument("--output", required=True, help="Output CSV path")
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
