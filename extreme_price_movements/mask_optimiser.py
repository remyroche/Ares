from __future__ import annotations

import argparse
import glob
import logging
import multiprocessing as mp
import os
import random
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

from extreme_price_movements.config import CFG, enable_perp_feature_keys
from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    load_features_selected,
    to_panel,
)
from extreme_price_movements.purged_cv import PurgedKFold
from extreme_price_movements.utils import tprint


LOGGER = logging.getLogger(__name__)
_LOGGED_FAILURE_COUNTS: Dict[str, int] = {}


def _log_bounded_warning(key: str, msg: str, limit: int = 3) -> None:
    c = _LOGGED_FAILURE_COUNTS.get(key, 0)
    if c < limit:
        LOGGER.warning(msg)
    _LOGGED_FAILURE_COUNTS[key] = c + 1


# =============================================================================
# CONSTANTS
# =============================================================================

MODE_PRICE_UP_TF = "price_up_tf"
MODE_PRICE_UP_MR = "price_up_mr"
MODE_PRICE_DOWN_TF = "price_down_tf"
MODE_PRICE_DOWN_MR = "price_down_mr"

ALL_MODES = [
    MODE_PRICE_UP_MR,
    MODE_PRICE_UP_TF,
    MODE_PRICE_DOWN_MR,
    MODE_PRICE_DOWN_TF,
]


# Priority order for quote-currency deduplication.
_QUOTE_PRIORITY: list[str] = ["USDT", "USDC", "BUSD", "EUR"]


def _dedup_universe_by_base(symbols: list[str]) -> list[str]:
    """Return at most one symbol per base asset, preferring the highest-priority quote."""
    _KNOWN_QUOTES = set(_QUOTE_PRIORITY)

    def _parse(sym: str) -> tuple[str, str]:
        """Return (base, quote) parsed from any separator format."""
        clean = sym.replace("/", "").replace("_", "").upper()
        for q in sorted(_KNOWN_QUOTES, key=len, reverse=True):
            if clean.endswith(q) and len(clean) > len(q):
                return clean[: -len(q)], q
        return clean, ""  # unknown quote — treat as unique

    best: dict[str, tuple[int, str]] = {}  # base -> (priority_rank, original_sym)
    for sym in symbols:
        base, quote = _parse(sym)
        rank = _QUOTE_PRIORITY.index(quote) if quote in _QUOTE_PRIORITY else len(_QUOTE_PRIORITY)
        if base not in best or rank < best[base][0]:
            best[base] = (rank, sym)

    deduped = sorted(v for _, v in best.values())
    return deduped


# =============================================================================
# NUMBA KERNELS
# =============================================================================

@njit(cache=True)
def rolling_max_index_nb(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    out_val = np.full(n, np.nan, dtype=np.float32)
    out_idx = np.zeros(n, dtype=np.int32)
    if n == 0 or window <= 0:
        return out_val, out_idx

    deque_idx = np.zeros(n, dtype=np.int32)
    head = 0
    tail = 0

    for i in range(n):
        left = i - window + 1
        while head < tail and deque_idx[head] < left:
            head += 1

        v = x[i]
        if not np.isnan(v):
            while head < tail:
                j = deque_idx[tail - 1]
                vj = x[j]
                if np.isnan(vj) or vj <= v:
                    tail -= 1
                else:
                    break
            deque_idx[tail] = i
            tail += 1

        if head < tail:
            idx = deque_idx[head]
            out_idx[i] = idx
            out_val[i] = x[idx]

    return out_val, out_idx


@njit(cache=True)
def rolling_min_index_nb(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    out_val = np.full(n, np.nan, dtype=np.float32)
    out_idx = np.zeros(n, dtype=np.int32)
    if n == 0 or window <= 0:
        return out_val, out_idx

    deque_idx = np.zeros(n, dtype=np.int32)
    head = 0
    tail = 0

    for i in range(n):
        left = i - window + 1
        while head < tail and deque_idx[head] < left:
            head += 1

        v = x[i]
        if not np.isnan(v):
            while head < tail:
                j = deque_idx[tail - 1]
                vj = x[j]
                if np.isnan(vj) or vj >= v:
                    tail -= 1
                else:
                    break
            deque_idx[tail] = i
            tail += 1

        if head < tail:
            idx = deque_idx[head]
            out_idx[i] = idx
            out_val[i] = x[idx]

    return out_val, out_idx


@njit(cache=True)
def rolling_std_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(x.shape[0], np.nan, dtype=np.float32)
    n = x.shape[0]
    if n == 0 or window <= 0:
        return out

    sum_x = 0.0
    sum_sq = 0.0
    valid_count = 0

    for i in range(n):
        val = x[i]
        if not np.isnan(val):
            sum_x += val
            sum_sq += val * val
            valid_count += 1

        if i >= window:
            old_val = x[i - window]
            if not np.isnan(old_val):
                sum_x -= old_val
                sum_sq -= old_val * old_val
                valid_count -= 1

        if valid_count > 1:
            var = (sum_sq - (sum_x * sum_x) / valid_count) / (valid_count - 1)
            out[i] = np.sqrt(var) if var > 0 else 0.0
        elif valid_count == 1:
            out[i] = 0.0
    return out


@njit(cache=True)
def dilate_mask_by_groups_nb(mask: np.ndarray, group_indices: np.ndarray, duration_bars: int) -> np.ndarray:
    out = mask.copy()
    if duration_bars <= 1:
        return out

    n_local = group_indices.shape[0]
    for local_i in range(n_local):
        gidx = group_indices[local_i]
        if mask[gidx]:
            end_local = min(n_local, local_i + duration_bars)
            for local_j in range(local_i + 1, end_local):
                out[group_indices[local_j]] = True
    return out


def dilate_mask_by_asset(mask: np.ndarray, asset_groups: Dict[int, np.ndarray], duration_bars: int) -> np.ndarray:
    if duration_bars <= 1:
        return mask.copy()
    out = mask.copy()
    for _, idxs in asset_groups.items():
        if idxs.shape[0] == 0:
            continue
        out = dilate_mask_by_groups_nb(out, idxs.astype(np.int32), duration_bars)
    return out


@njit(cache=True)
def compute_impulse_coherence_nb(
    returns: np.ndarray,
    volatility: np.ndarray,
    high_val: np.ndarray,
    low_val: np.ndarray,
    start_px: np.ndarray,
    high_idx_local: np.ndarray,
    low_idx_local: np.ndarray,
    start_idx_local: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = returns.shape[0]
    bars_to_peak_up = np.full(n, np.nan, dtype=np.float32)
    bars_to_peak_dn = np.full(n, np.nan, dtype=np.float32)
    speed_up = np.full(n, np.nan, dtype=np.float32)
    speed_dn = np.full(n, np.nan, dtype=np.float32)
    mono_up = np.full(n, np.nan, dtype=np.float32)
    mono_dn = np.full(n, np.nan, dtype=np.float32)
    vol_exp = np.full(n, np.nan, dtype=np.float32)

    pref_ret = np.zeros(n + 1, dtype=np.float32)
    pref_abs = np.zeros(n + 1, dtype=np.float32)
    for i in range(n):
        r = returns[i]
        if np.isnan(r):
            pref_ret[i + 1] = pref_ret[i]
            pref_abs[i + 1] = pref_abs[i]
        else:
            pref_ret[i + 1] = pref_ret[i] + r
            pref_abs[i + 1] = pref_abs[i] + abs(r)

    for i in range(window, n):
        st = start_idx_local[i]
        st_px = start_px[i]

        peak_h = high_idx_local[i]
        peak_l = low_idx_local[i]

        b_up = peak_h - st
        b_dn = peak_l - st

        bars_to_peak_up[i] = b_up
        bars_to_peak_dn[i] = b_dn

        imp_up = (high_val[i] - st_px) / st_px if st_px > 1e-9 else 0.0
        imp_dn = (st_px - low_val[i]) / st_px if st_px > 1e-9 else 0.0

        speed_up[i] = imp_up / max(1.0, b_up)
        speed_dn[i] = imp_dn / max(1.0, b_dn)

        up_left = min(max(st + 1, 0), n)
        up_right = min(max(peak_h + 1, up_left), n)
        dir_sum_up = pref_ret[up_right] - pref_ret[up_left]
        abs_sum_up = pref_abs[up_right] - pref_abs[up_left]
        mono_up[i] = dir_sum_up / abs_sum_up if abs_sum_up > 1e-9 else 0.0

        dn_left = min(max(st + 1, 0), n)
        dn_right = min(max(peak_l + 1, dn_left), n)
        dir_sum_dn = -(pref_ret[dn_right] - pref_ret[dn_left])
        abs_sum_dn = pref_abs[dn_right] - pref_abs[dn_left]
        mono_dn[i] = dir_sum_dn / abs_sum_dn if abs_sum_dn > 1e-9 else 0.0

        pre_vol = volatility[st]
        post_vol = volatility[i]
        vol_exp[i] = post_vol / pre_vol if pre_vol > 1e-9 else 1.0

    return bars_to_peak_up, bars_to_peak_dn, speed_up, speed_dn, mono_up, mono_dn, vol_exp


def rolling_max_index_safe(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    out_val = np.full(n, np.nan, dtype=np.float32)
    out_idx = np.zeros(n, dtype=np.int32)
    if n == 0 or window <= 0:
        return out_val, out_idx
    for i in range(n):
        left = max(0, i - window + 1)
        sl = x[left : i + 1]
        valid_local = np.where(~np.isnan(sl))[0]
        if valid_local.shape[0] == 0:
            continue
        best_local = valid_local[int(np.argmax(sl[valid_local]))]
        best_idx = left + best_local
        out_idx[i] = best_idx
        out_val[i] = x[best_idx]
    return out_val, out_idx


def rolling_min_index_safe(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    out_val = np.full(n, np.nan, dtype=np.float32)
    out_idx = np.zeros(n, dtype=np.int32)
    if n == 0 or window <= 0:
        return out_val, out_idx
    for i in range(n):
        left = max(0, i - window + 1)
        sl = x[left : i + 1]
        valid_local = np.where(~np.isnan(sl))[0]
        if valid_local.shape[0] == 0:
            continue
        best_local = valid_local[int(np.argmin(sl[valid_local]))]
        best_idx = left + best_local
        out_idx[i] = best_idx
        out_val[i] = x[best_idx]
    return out_val, out_idx


def rolling_std_safe(x: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float32)
    if n == 0 or window <= 0:
        return out
    for i in range(n):
        left = max(0, i - window + 1)
        sl = x[left : i + 1]
        sl = sl[np.isfinite(sl)]
        if sl.shape[0] > 1:
            out[i] = np.float32(np.std(sl, ddof=1))
        elif sl.shape[0] == 1:
            out[i] = 0.0
    return out


def compute_impulse_coherence_safe(
    returns: np.ndarray,
    volatility: np.ndarray,
    high_val: np.ndarray,
    low_val: np.ndarray,
    start_px: np.ndarray,
    high_idx_local: np.ndarray,
    low_idx_local: np.ndarray,
    start_idx_local: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = returns.shape[0]
    bars_to_peak_up = np.full(n, np.nan, dtype=np.float32)
    bars_to_peak_dn = np.full(n, np.nan, dtype=np.float32)
    speed_up = np.full(n, np.nan, dtype=np.float32)
    speed_dn = np.full(n, np.nan, dtype=np.float32)
    mono_up = np.full(n, np.nan, dtype=np.float32)
    mono_dn = np.full(n, np.nan, dtype=np.float32)
    vol_exp = np.full(n, np.nan, dtype=np.float32)

    pref_ret = np.zeros(n + 1, dtype=np.float32)
    pref_abs = np.zeros(n + 1, dtype=np.float32)
    for i in range(n):
        r = returns[i]
        if np.isnan(r):
            pref_ret[i + 1] = pref_ret[i]
            pref_abs[i + 1] = pref_abs[i]
        else:
            pref_ret[i + 1] = pref_ret[i] + r
            pref_abs[i + 1] = pref_abs[i] + abs(r)

    for i in range(window, n):
        st = int(start_idx_local[i])
        st_px = float(start_px[i])
        peak_h = int(high_idx_local[i])
        peak_l = int(low_idx_local[i])
        b_up = peak_h - st
        b_dn = peak_l - st
        bars_to_peak_up[i] = b_up
        bars_to_peak_dn[i] = b_dn

        imp_up = (high_val[i] - st_px) / st_px if st_px > 1e-9 else 0.0
        imp_dn = (st_px - low_val[i]) / st_px if st_px > 1e-9 else 0.0
        speed_up[i] = imp_up / max(1.0, float(b_up))
        speed_dn[i] = imp_dn / max(1.0, float(b_dn))

        up_left = min(max(st + 1, 0), n)
        up_right = min(max(peak_h + 1, up_left), n)
        dir_sum_up = pref_ret[up_right] - pref_ret[up_left]
        abs_sum_up = pref_abs[up_right] - pref_abs[up_left]
        mono_up[i] = dir_sum_up / abs_sum_up if abs_sum_up > 1e-9 else 0.0

        dn_left = min(max(st + 1, 0), n)
        dn_right = min(max(peak_l + 1, dn_left), n)
        dir_sum_dn = -(pref_ret[dn_right] - pref_ret[dn_left])
        abs_sum_dn = pref_abs[dn_right] - pref_abs[dn_left]
        mono_dn[i] = dir_sum_dn / abs_sum_dn if abs_sum_dn > 1e-9 else 0.0

        pre_vol = volatility[st]
        post_vol = volatility[i]
        vol_exp[i] = post_vol / pre_vol if pre_vol > 1e-9 else 1.0

    return bars_to_peak_up, bars_to_peak_dn, speed_up, speed_dn, mono_up, mono_dn, vol_exp


def dilate_mask_by_asset_safe(mask: np.ndarray, asset_groups: Dict[int, np.ndarray], duration_bars: int) -> np.ndarray:
    if duration_bars <= 1:
        return mask.copy()
    out = mask.copy()
    for idxs in asset_groups.values():
        if idxs.shape[0] == 0:
            continue
        local_hits = np.where(mask[idxs])[0]
        for local_i in local_hits:
            end_local = min(idxs.shape[0], local_i + duration_bars)
            out[idxs[local_i + 1 : end_local]] = True
    return out


@njit(cache=True)
def active_days_fraction_nb(mask: np.ndarray, day_ids: np.ndarray, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    seen = np.zeros(n_days, dtype=np.uint8)
    n = mask.shape[0]
    for i in range(n):
        if mask[i]:
            seen[day_ids[i]] = 1
    return float(np.sum(seen)) / float(n_days)


@njit(cache=True)
def daily_event_stats_nb(mask: np.ndarray, day_ids: np.ndarray, n_days: int) -> Tuple[float, float]:
    counts = np.zeros(n_days, dtype=np.int32)
    n = mask.shape[0]
    for i in range(n):
        if mask[i]:
            counts[day_ids[i]] += 1

    active_days = 0
    total = 0.0
    for d in range(n_days):
        if counts[d] > 0:
            active_days += 1
        total += counts[d]

    mean = total / max(1, n_days)

    var = 0.0
    for d in range(n_days):
        diff = counts[d] - mean
        var += diff * diff
    std = np.sqrt(var / max(1, n_days))
    return float(mean), float(std)


@njit(cache=True)
def fold_base_rate_nb(mask: np.ndarray, target: np.ndarray, val_idx: np.ndarray) -> float:
    total = 0
    pos = 0
    for k in range(val_idx.shape[0]):
        i = val_idx[k]
        if mask[i] and not np.isnan(target[i]):
            total += 1
            pos += target[i]
    if total == 0:
        return 0.0
    return float(pos) / float(total)


@njit(cache=True)
def simple_mask_count_nb(mask: np.ndarray) -> int:
    return int(np.sum(mask))


def active_days_fraction_safe(mask: np.ndarray, day_ids: np.ndarray, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    if mask.shape[0] == 0:
        return 0.0
    active_days = np.unique(day_ids[mask])
    return float(active_days.shape[0]) / float(n_days)


def daily_event_stats_safe(mask: np.ndarray, day_ids: np.ndarray, n_days: int) -> Tuple[float, float]:
    if n_days <= 0:
        return 0.0, 0.0
    counts = np.zeros(n_days, dtype=np.int32)
    if np.any(mask):
        vals, freqs = np.unique(day_ids[mask], return_counts=True)
        counts[vals.astype(np.int32)] = freqs.astype(np.int32)
    return float(np.mean(counts)), float(np.std(counts))


def fold_base_rate_safe(mask: np.ndarray, target: np.ndarray, val_idx: np.ndarray) -> float:
    if val_idx.shape[0] == 0:
        return 0.0
    valid = mask[val_idx] & np.isfinite(target[val_idx])
    if not np.any(valid):
        return 0.0
    return float(np.mean(target[val_idx][valid]))


def simple_mask_count_safe(mask: np.ndarray) -> int:
    return int(np.sum(mask))


@njit(cache=True)
def safe_mean_nb(x: np.ndarray) -> float:
    if x.shape[0] == 0:
        return 0.0
    s = 0.0
    n = 0
    for i in range(x.shape[0]):
        v = x[i]
        if not np.isnan(v):
            s += v
            n += 1
    return s / max(1, n)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CandidateKey:
    family: str
    z_hours: int
    param: str
    duration_hours: int

    def as_str(self) -> str:
        return f"{self.family}|z={self.z_hours}|p={self.param}|d={self.duration_hours}"


# =============================================================================
# HELPERS
# =============================================================================

def _mode_is_up(mode: str) -> bool:
    return mode in (MODE_PRICE_UP_TF, MODE_PRICE_UP_MR)


def _mode_is_tf(mode: str) -> bool:
    return mode in (MODE_PRICE_UP_TF, MODE_PRICE_DOWN_TF)


def _get_side_mask(mode: str, m_high: np.ndarray, m_low: np.ndarray) -> np.ndarray:
    return m_high if _mode_is_up(mode) else m_low


def _mode_primary_target(mode: str, forward_returns: np.ndarray, ret_threshold: float) -> np.ndarray:
    # 1 = desired outcome for that mode
    valid = np.isfinite(forward_returns)
    if mode == MODE_PRICE_UP_TF:
        out = (forward_returns > ret_threshold).astype(np.float32)
    elif mode == MODE_PRICE_UP_MR:
        out = (forward_returns < -ret_threshold).astype(np.float32)
    elif mode == MODE_PRICE_DOWN_TF:
        out = (forward_returns < -ret_threshold).astype(np.float32)
    elif mode == MODE_PRICE_DOWN_MR:
        out = (forward_returns > ret_threshold).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out[~valid] = np.nan
    return out


def _signed_mode_return(mode: str, forward_returns: np.ndarray) -> np.ndarray:
    # Positive = good for the mode
    valid = np.isfinite(forward_returns)
    if mode == MODE_PRICE_UP_TF:
        out = forward_returns.astype(np.float32)
    elif mode == MODE_PRICE_UP_MR:
        out = (-forward_returns).astype(np.float32)
    elif mode == MODE_PRICE_DOWN_TF:
        out = (-forward_returns).astype(np.float32)
    elif mode == MODE_PRICE_DOWN_MR:
        out = forward_returns.astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out[~valid] = np.nan
    return out


def _resolve_path(path: str) -> str:
    if not path:
        return path
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(pkg_root, path))


def _find_latest_feature_dir(data_root: str) -> Optional[str]:
    feat_dir = os.path.join(data_root, "features")
    if not os.path.isdir(feat_dir):
        return None
    dirs = sorted(glob.glob(os.path.join(feat_dir, "20*")))
    return dirs[-1] if dirs else None


def _rng_sample_half(items: List[Any], seed: int = 42) -> List[Any]:
    if len(items) <= 1:
        return items[:]
    rng = random.Random(seed)
    k = max(1, len(items) // 2)
    return rng.sample(items, k)


def _rng_sample_fraction(items: List[Any], frac: float, seed: int = 42) -> List[Any]:
    if len(items) <= 1:
        return items[:]
    frac = min(max(float(frac), 0.0), 1.0)
    if frac >= 0.999:
        return items[:]
    rng = random.Random(seed)
    k = max(1, int(round(len(items) * frac)))
    k = min(k, len(items))
    return rng.sample(items, k)


def _zscore_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - m) / s).astype(np.float32)


def _metric_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _coherence_metrics_single_side(
    mask: np.ndarray,
    bars_to_peak: np.ndarray,
    speed: np.ndarray,
    mono: np.ndarray,
) -> Dict[str, float]:
    valid = mask & np.isfinite(bars_to_peak) & np.isfinite(speed) & np.isfinite(mono)
    if not np.any(valid):
        return {
            "bars_to_peak_dispersion": 1e9,
            "speed_dispersion": 1e9,
            "monotonicity_dispersion": 1e9,
            "impulse_shape_dispersion": 1e9,
        }
    bp = float(np.std(bars_to_peak[valid])) if np.sum(valid) > 1 else 0.0
    sp = float(np.std(speed[valid])) if np.sum(valid) > 1 else 0.0
    mo = float(np.std(mono[valid])) if np.sum(valid) > 1 else 0.0
    return {
        "bars_to_peak_dispersion": bp,
        "speed_dispersion": sp,
        "monotonicity_dispersion": mo,
        "impulse_shape_dispersion": bp + sp + mo,
    }


def _compute_regime_distinctness_single_side(
    side_mask: np.ndarray,
    mode: str,
    forward_returns: np.ndarray,
    mae_high: np.ndarray,
    mfe_high: np.ndarray,
    mae_low: np.ndarray,
    mfe_low: np.ndarray,
) -> float:
    if not np.any(side_mask):
        return 0.0

    valid = np.isfinite(forward_returns)
    ret_g = _signed_mode_return(mode, forward_returns[valid])
    ret_e = _signed_mode_return(mode, forward_returns[valid & side_mask])

    if ret_g.shape[0] < 10 or ret_e.shape[0] < 10:
        return 0.0

    std_g = np.std(ret_g)
    std_e = np.std(ret_e)
    std_ratio = std_e / std_g if std_g > 1e-9 else 1.0

    t_upper = np.percentile(ret_g, 95)
    t_lower = np.percentile(ret_g, 5)
    tail_g = np.mean((ret_g >= t_upper) | (ret_g <= t_lower))
    tail_e = np.mean((ret_e >= t_upper) | (ret_e <= t_lower))
    tail_ratio = tail_e / tail_g if tail_g > 1e-9 else 1.0

    if _mode_is_up(mode):
        mae_arr = mae_high
        mfe_arr = mfe_high
    else:
        mae_arr = mae_low
        mfe_arr = mfe_low

    mae_g = float(np.nanmean(mae_arr[valid])) if np.any(valid) else 1.0
    mae_e = float(np.nanmean(mae_arr[valid & side_mask])) if np.any(valid & side_mask) else mae_g
    mae_ratio = mae_e / mae_g if mae_g > 1e-9 else 1.0

    mfe_g = float(np.nanmean(mfe_arr[valid])) if np.any(valid) else 1.0
    mfe_e = float(np.nanmean(mfe_arr[valid & side_mask])) if np.any(valid & side_mask) else mfe_g
    mfe_ratio = mfe_e / mfe_g if mfe_g > 1e-9 else 1.0

    return float(np.mean(np.clip([std_ratio, tail_ratio, mae_ratio, mfe_ratio], 0.0, 5.0)))


def _build_temporal_folds(timestamps: np.ndarray, n_samples: int, n_splits: int = 2) -> List[Tuple[np.ndarray, np.ndarray]]:
    try:
        cv = PurgedKFold(n_splits=n_splits, purge=43200, embargo=43200, times=timestamps)
        dummy = np.empty((n_samples, 1), dtype=np.float32)
        folds = list(cv.split(dummy))
        if folds:
            return [(tr.astype(np.int32), va.astype(np.int32)) for tr, va in folds]
    except Exception as e:
        LOGGER.warning("PurgedKFold unavailable; using timestamp-group fallback: %s", e)

    if n_samples < 10:
        return []

    uniq_ts = np.unique(timestamps)
    n_groups = uniq_ts.shape[0]
    if n_groups < 2:
        return []

    folds = []
    chunk_size = n_groups // (n_splits + 1)
    if chunk_size == 0:
        chunk_size = max(1, n_groups // 2)
        n_splits = 1

    for i in range(1, n_splits + 1):
        tr_group_end = i * chunk_size
        va_group_end = min(n_groups, (i + 1) * chunk_size)
        if i == n_splits:
            va_group_end = n_groups
        if tr_group_end >= va_group_end:
            break

        tr_ts = uniq_ts[:tr_group_end]
        va_ts = uniq_ts[tr_group_end:va_group_end]
        tr = np.where(np.isin(timestamps, tr_ts))[0].astype(np.int32)
        va = np.where(np.isin(timestamps, va_ts))[0].astype(np.int32)
        if tr.shape[0] == 0 or va.shape[0] == 0:
            continue
        folds.append((tr, va))

    return folds


def _impute_and_scale_train_valid(X_train: np.ndarray, X_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_tr = X_train.astype(np.float32, copy=True)
    X_va = X_valid.astype(np.float32, copy=True)

    X_tr[~np.isfinite(X_tr)] = np.nan
    X_va[~np.isfinite(X_va)] = np.nan

    n_features = X_tr.shape[1]
    med = np.zeros(n_features, dtype=np.float32)
    mean = np.zeros(n_features, dtype=np.float32)
    std = np.ones(n_features, dtype=np.float32)

    for j in range(n_features):
        col = X_tr[:, j]
        valid = ~np.isnan(col)
        if np.any(valid):
            m = np.median(col[valid]).astype(np.float32)
            med[j] = m
            X_tr[~valid, j] = m
            X_va[np.isnan(X_va[:, j]), j] = m
        else:
            med[j] = 0.0
            X_tr[:, j] = 0.0
            X_va[:, j] = 0.0

        mean[j] = np.mean(X_tr[:, j]).astype(np.float32)
        s = np.std(X_tr[:, j]).astype(np.float32)
        std[j] = s if s > 1e-6 else 1.0

    X_tr = ((X_tr - mean) / std).astype(np.float32)
    X_va = ((X_va - mean) / std).astype(np.float32)
    return X_tr, X_va


def _classifier_oof_auc(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    n_splits: int = 2,
) -> float:
    if X.shape[0] < 20 or np.unique(y[np.isfinite(y)]).shape[0] < 2:
        return 0.5
    folds = _build_temporal_folds(timestamps, X.shape[0], n_splits=n_splits)
    if not folds:
        return 0.5

    preds = np.full(X.shape[0], np.nan, dtype=np.float32)
    for tr, va in folds:
        if tr.shape[0] == 0 or va.shape[0] == 0:
            continue
        if np.unique(y[tr][np.isfinite(y[tr])]).shape[0] < 2:
            continue
        X_tr, X_va = _impute_and_scale_train_valid(X[tr], X[va])
        clf = LogisticRegression(solver="liblinear", max_iter=100)
        try:
            clf.fit(X_tr, y[tr])
            preds[va] = clf.predict_proba(X_va)[:, 1].astype(np.float32)
        except Exception as e:
            _log_bounded_warning("classifier_fit", f"Classifier fold fit failed: {e}")

    valid_mask = np.isfinite(preds) & np.isfinite(y)
    if np.sum(valid_mask) == 0:
        return 0.5
    if np.unique(y[valid_mask]).shape[0] < 2 or np.unique(preds[valid_mask]).shape[0] < 2:
        return 0.5
    try:
        return float(roc_auc_score(y[valid_mask], preds[valid_mask]))
    except Exception as e:
        _log_bounded_warning("roc_auc", f"AUC scoring failed: {e}")
        return 0.5


def _classifier_oof_auc_from_folds(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> float:
    if X.shape[0] < 20 or np.unique(y[np.isfinite(y)]).shape[0] < 2:
        return 0.5
    if not folds:
        return 0.5

    preds = np.full(X.shape[0], np.nan, dtype=np.float32)
    for tr, va in folds:
        if tr.shape[0] == 0 or va.shape[0] == 0:
            continue
        if np.unique(y[tr][np.isfinite(y[tr])]).shape[0] < 2:
            continue
        X_tr, X_va = _impute_and_scale_train_valid(X[tr], X[va])
        clf = LogisticRegression(solver="liblinear", max_iter=100)
        try:
            clf.fit(X_tr, y[tr])
            preds[va] = clf.predict_proba(X_va)[:, 1].astype(np.float32)
        except Exception as e:
            _log_bounded_warning("classifier_fit", f"Classifier fold fit failed: {e}")

    valid_mask = np.isfinite(preds) & np.isfinite(y)
    if np.sum(valid_mask) == 0:
        return 0.5
    if np.unique(y[valid_mask]).shape[0] < 2 or np.unique(preds[valid_mask]).shape[0] < 2:
        return 0.5
    try:
        return float(roc_auc_score(y[valid_mask], preds[valid_mask]))
    except Exception as e:
        _log_bounded_warning("roc_auc", f"AUC scoring failed: {e}")
        return 0.5


def _incremental_information_metrics(
    learn_X: np.ndarray,
    side_mask: np.ndarray,
    y_primary: np.ndarray,
    timestamps: np.ndarray,
    idx_e: np.ndarray,
    idx_ne: np.ndarray,
    n_splits: int = 3,
) -> Dict[str, float]:
    metrics = {
        "incremental_information_delta_auc": float("nan"),
        "incremental_information_delta_auc_fold_mean": float("nan"),
        "incremental_information_delta_auc_fold_std": float("nan"),
        "incremental_information_positive_fold_fraction": float("nan"),
        "incremental_information_positive_fold_count": float("nan"),
        "incremental_information_fold_count": float("nan"),
    }

    idx_all = np.sort(np.concatenate([idx_e, idx_ne]).astype(np.int32))
    if idx_all.shape[0] < 100:
        return metrics

    y_all = y_primary[idx_all]
    ts_all = timestamps[idx_all]
    event_feature = side_mask[idx_all].astype(np.float32).reshape(-1, 1)
    X_base = learn_X[idx_all]
    X_aug = np.concatenate([X_base, event_feature], axis=1).astype(np.float32, copy=False)
    folds = _build_temporal_folds(ts_all, idx_all.shape[0], n_splits=n_splits)
    if not folds:
        return metrics

    auc_base = _classifier_oof_auc_from_folds(X_base, y_all, folds)
    auc_aug = _classifier_oof_auc_from_folds(X_aug, y_all, folds)
    metrics["incremental_information_delta_auc"] = float(auc_aug - auc_base)

    positive_fold_count = 0
    evaluated_fold_count = 0
    fold_deltas: List[float] = []
    for tr, va in folds:
        if tr.shape[0] == 0 or va.shape[0] == 0:
            continue
        y_tr = y_all[tr]
        y_va = y_all[va]
        if np.unique(y_tr[np.isfinite(y_tr)]).shape[0] < 2 or np.unique(y_va[np.isfinite(y_va)]).shape[0] < 2:
            continue
        auc_base_fold = _classifier_oof_auc_from_folds(X_base, y_all, [(tr, va)])
        auc_aug_fold = _classifier_oof_auc_from_folds(X_aug, y_all, [(tr, va)])
        delta_fold = float(auc_aug_fold - auc_base_fold)
        fold_deltas.append(delta_fold)
        evaluated_fold_count += 1
        if delta_fold > 0:
            positive_fold_count += 1

    if evaluated_fold_count > 0:
        metrics["incremental_information_delta_auc_fold_mean"] = float(np.mean(np.asarray(fold_deltas, dtype=np.float32)))
        metrics["incremental_information_delta_auc_fold_std"] = float(np.std(np.asarray(fold_deltas, dtype=np.float32)))
        metrics["incremental_information_positive_fold_fraction"] = float(
            positive_fold_count / float(evaluated_fold_count)
        )
        metrics["incremental_information_positive_fold_count"] = float(positive_fold_count)
        metrics["incremental_information_fold_count"] = float(evaluated_fold_count)

    return metrics


def _primary_gain_fold_deltas(
    learn_X: np.ndarray,
    y_primary: np.ndarray,
    timestamps: np.ndarray,
    idx_e: np.ndarray,
    idx_ne: np.ndarray,
    n_splits: int = 3,
) -> np.ndarray:
    if idx_e.shape[0] < 50 or idx_ne.shape[0] < 50:
        return np.asarray([], dtype=np.float32)

    folds_e = _build_temporal_folds(timestamps[idx_e], idx_e.shape[0], n_splits=n_splits)
    folds_ne = _build_temporal_folds(timestamps[idx_ne], idx_ne.shape[0], n_splits=n_splits)
    if not folds_e or not folds_ne:
        return np.asarray([], dtype=np.float32)

    out: List[float] = []
    for (tr_e, va_e), (tr_ne, va_ne) in zip(folds_e, folds_ne):
        auc_e = _classifier_oof_auc_from_folds(learn_X[idx_e], y_primary[idx_e], [(tr_e, va_e)])
        auc_ne = _classifier_oof_auc_from_folds(learn_X[idx_ne], y_primary[idx_ne], [(tr_ne, va_ne)])
        out.append(float(auc_e - auc_ne))
    return np.asarray(out, dtype=np.float32)


def _stability_from_fold_deltas(delta_folds: np.ndarray) -> Dict[str, float]:
    delta_folds = np.asarray(delta_folds, dtype=np.float32)
    delta_folds = delta_folds[np.isfinite(delta_folds)]
    if delta_folds.size == 0:
        return {
            "delta_fold_mean": float("nan"),
            "delta_fold_std": float("nan"),
            "positive_fold_fraction": float("nan"),
            "stability_score": float("nan"),
            "fold_count": 0.0,
        }

    mean_delta = float(np.mean(delta_folds))
    std_delta = float(np.std(delta_folds))
    positive_fold_fraction = float(np.mean(delta_folds > 0))
    stability_score = 0.5 * max(0.0, 1.0 - std_delta / (abs(mean_delta) + 1e-9)) + 0.5 * positive_fold_fraction
    return {
        "delta_fold_mean": mean_delta,
        "delta_fold_std": std_delta,
        "positive_fold_fraction": positive_fold_fraction,
        "stability_score": float(stability_score),
        "fold_count": float(delta_folds.size),
    }


def _build_regime_rationale(row: pd.Series) -> str:
    reasons: List[str] = []
    delta_r = _metric_or_nan(row.get("delta_r"))
    delta_r_shrunk = _metric_or_nan(row.get("delta_r_shrunk"))
    s_r = _metric_or_nan(row.get("S_r"))
    d_r = _metric_or_nan(row.get("D_r"))
    positive_fraction = _metric_or_nan(row.get("positive_fold_fraction_r"))
    metric_name = str(row.get("selected_delta_metric", ""))
    incr_delta = _metric_or_nan(row.get("incremental_information_delta_auc"))
    incr_positive = _metric_or_nan(row.get("incremental_information_positive_fold_fraction"))
    disp_edge = _metric_or_nan(row.get("dispersion_to_edge_ratio"))
    primary_nan = _metric_or_nan(row.get("primary_predictability_gain_is_nan"))

    if np.isfinite(delta_r) and delta_r > 0:
        reasons.append(f"positive bucket OOS delta_r={delta_r:.4f}")
    if np.isfinite(delta_r_shrunk):
        reasons.append(f"shrunk delta={delta_r_shrunk:.4f}")
    if np.isfinite(s_r):
        reasons.append(f"stability={s_r:.3f}")
    if np.isfinite(positive_fraction):
        reasons.append(f"positive-fold fraction={positive_fraction:.3f}")
    if np.isfinite(incr_delta):
        reasons.append(f"delta-auc={incr_delta:.4f}")
    if np.isfinite(incr_positive):
        reasons.append(f"delta-auc positive folds={incr_positive:.3f}")
    if np.isfinite(disp_edge):
        reasons.append(f"dispersion/edge={disp_edge:.3f}")
    if np.isfinite(primary_nan) and primary_nan > 0.5:
        reasons.append("primary directional gain unavailable")
    if np.isfinite(d_r):
        reasons.append(f"dispersion={d_r:.3f}")
    return "; ".join(reasons)


def _predictability_gain_from_metrics(metrics: Dict[str, Any]) -> float:
    vals = [
        _metric_or_nan(metrics.get("continuation_predictability_gain")),
        _metric_or_nan(metrics.get("reversal_predictability_gain")),
        _metric_or_nan(metrics.get("MAE_predictability_gain")),
        _metric_or_nan(metrics.get("MFE_predictability_gain")),
    ]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(max(vals))


def _mode_primary_predictability_col(mode: str) -> str:
    return "continuation_predictability_gain" if _mode_is_tf(mode) else "reversal_predictability_gain"


def _mode_predictability_gain_from_metrics(mode: str, metrics: Dict[str, Any]) -> float:
    vals = [
        _metric_or_nan(metrics.get(_mode_primary_predictability_col(mode))),
        _metric_or_nan(metrics.get("MAE_predictability_gain")),
        _metric_or_nan(metrics.get("MFE_predictability_gain")),
    ]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(max(vals))


def _compute_legacy_conditional_learnability(
    mode: str,
    side_mask: np.ndarray,
    shared: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    ret_threshold = float(cfg.get("phase1_ret_threshold", 0.0))
    learn_X = shared["learn_X"]
    forward_returns = shared["forward_returns"]
    timestamps = shared["timestamps"]
    valid = np.isfinite(forward_returns)

    idx_g = np.where(valid)[0].astype(np.int32)
    idx_e = np.where(valid & side_mask)[0].astype(np.int32)

    out = {
        "continuation_predictability_gain": float("nan"),
        "reversal_predictability_gain": float("nan"),
        "MAE_predictability_gain": float("nan"),
        "MFE_predictability_gain": float("nan"),
        "predictability_gain": float("nan"),
    }
    if idx_e.shape[0] < 50 or idx_g.shape[0] < 100:
        return out

    max_global = int(cfg.get("phase2_metric_max_samples_per_class", 25_000))
    if max_global > 0 and idx_g.shape[0] > max_global:
        rng = np.random.RandomState(123)
        idx_g = np.sort(rng.choice(idx_g, max_global, replace=False).astype(np.int32))
    max_event = int(cfg.get("legacy_stage2_event_max_samples", cfg.get("phase2_metric_max_samples_per_class", 25_000)))
    if max_event > 0 and idx_e.shape[0] > max_event:
        rng = np.random.RandomState(456)
        idx_e = np.sort(rng.choice(idx_e, max_event, replace=False).astype(np.int32))

    n_splits = int(cfg.get("phase2_classifier_n_splits", 3))
    y_cont = _mode_primary_target(mode, forward_returns, ret_threshold)
    y_rev = np.full(y_cont.shape[0], np.nan, dtype=np.float32)
    valid_y = np.isfinite(y_cont)
    y_rev[valid_y] = 1.0 - y_cont[valid_y]

    auc_cont_g = _classifier_oof_auc(learn_X[idx_g], y_cont[idx_g], timestamps[idx_g], n_splits=n_splits)
    auc_cont_e = _classifier_oof_auc(learn_X[idx_e], y_cont[idx_e], timestamps[idx_e], n_splits=n_splits)
    out["continuation_predictability_gain"] = float(auc_cont_e - auc_cont_g)

    auc_rev_g = _classifier_oof_auc(learn_X[idx_g], y_rev[idx_g], timestamps[idx_g], n_splits=n_splits)
    auc_rev_e = _classifier_oof_auc(learn_X[idx_e], y_rev[idx_e], timestamps[idx_e], n_splits=n_splits)
    out["reversal_predictability_gain"] = float(auc_rev_e - auc_rev_g)

    if mode in (MODE_PRICE_UP_TF, MODE_PRICE_UP_MR):
        mae_arr = shared["mae_high"]
        mfe_arr = shared["mfe_high"]
    else:
        mae_arr = shared["mae_low"]
        mfe_arr = shared["mfe_low"]

    r2_mae_g = _ridge_regression_oof_r2(learn_X[idx_g], mae_arr[idx_g], timestamps[idx_g], clip_q=0.98, n_splits=n_splits)
    r2_mae_e = _ridge_regression_oof_r2(learn_X[idx_e], mae_arr[idx_e], timestamps[idx_e], clip_q=0.98, n_splits=n_splits)
    out["MAE_predictability_gain"] = float(r2_mae_e - r2_mae_g)

    r2_mfe_g = _ridge_regression_oof_r2(learn_X[idx_g], mfe_arr[idx_g], timestamps[idx_g], clip_q=0.98, n_splits=n_splits)
    r2_mfe_e = _ridge_regression_oof_r2(learn_X[idx_e], mfe_arr[idx_e], timestamps[idx_e], clip_q=0.98, n_splits=n_splits)
    out["MFE_predictability_gain"] = float(r2_mfe_e - r2_mfe_g)

    out["predictability_gain"] = _mode_predictability_gain_from_metrics(mode, out)
    return out


def _apply_secondary_conditioner(
    mask_h: np.ndarray,
    mask_l: np.ndarray,
    conditioner: str,
    mono_up: np.ndarray,
    mono_dn: np.ndarray,
    vol_exp: np.ndarray,
    spread_to_atr: np.ndarray,
    alternation_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    new_h = mask_h.copy()
    new_l = mask_l.copy()

    if conditioner == "none":
        return new_h, new_l
    if conditioner == "liquidity_veto":
        safe_liq = spread_to_atr < 0.25
        return new_h & safe_liq, new_l & safe_liq
    if conditioner == "monotonicity_adjust":
        return new_h & (mono_up > 0.25), new_l & (mono_dn > 0.25)
    if conditioner == "volatility_adjust":
        return new_h & (vol_exp < 5.0), new_l & (vol_exp < 5.0)
    if conditioner == "alternation_adjust":
        return new_h & (alternation_array < 0.70), new_l & (alternation_array < 0.70)
    return new_h, new_l


def _ridge_regression_oof_r2(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    clip_q: float = 0.98,
    n_splits: int = 2,
) -> float:
    valid = np.isfinite(y)
    if np.sum(valid) < 20:
        return 0.0

    y = y.astype(np.float32, copy=True)
    folds = _build_temporal_folds(timestamps, X.shape[0], n_splits=n_splits)
    if not folds:
        return 0.0

    preds = np.full(X.shape[0], np.nan, dtype=np.float32)

    for tr, va in folds:
        if tr.shape[0] == 0 or va.shape[0] == 0:
            continue

        tr_valid = tr[np.isfinite(y[tr])]
        if tr_valid.shape[0] < 10:
            continue

        y_tr = y[tr_valid]
        hi = np.quantile(y_tr, clip_q).astype(np.float32)
        lo = np.quantile(y_tr, 1.0 - clip_q).astype(np.float32) if np.any(y_tr < 0) else 0.0
        y_tr_clip = np.clip(y_tr, lo, hi).astype(np.float32)

        X_tr, X_va = _impute_and_scale_train_valid(X[tr_valid], X[va])
        reg = Ridge(alpha=1.0)
        try:
            reg.fit(X_tr, y_tr_clip)
            preds[va] = reg.predict(X_va).astype(np.float32)
        except Exception as e:
            _log_bounded_warning("ridge_fit", f"Ridge fold fit failed: {e}")

    valid2 = np.isfinite(preds) & np.isfinite(y)
    if np.sum(valid2) < 10:
        return 0.0
    ssr = float(np.sum((y[valid2] - preds[valid2]) ** 2))
    sst = float(np.sum((y[valid2] - np.mean(y[valid2])) ** 2))
    if sst < 1e-9:
        return 0.0
    return float(1.0 - ssr / sst)


def _ridge_regression_fold_r2s(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    clip_q: float = 0.98,
    n_splits: int = 3,
) -> np.ndarray:
    valid = np.isfinite(y)
    if np.sum(valid) < 20:
        return np.asarray([], dtype=np.float32)

    y = y.astype(np.float32, copy=True)
    folds = _build_temporal_folds(timestamps, X.shape[0], n_splits=n_splits)
    if not folds:
        return np.asarray([], dtype=np.float32)

    scores: List[float] = []
    for tr, va in folds:
        if tr.shape[0] == 0 or va.shape[0] == 0:
            continue

        tr_valid = tr[np.isfinite(y[tr])]
        va_valid = va[np.isfinite(y[va])]
        if tr_valid.shape[0] < 10 or va_valid.shape[0] < 10:
            continue

        y_tr = y[tr_valid]
        hi = np.quantile(y_tr, clip_q).astype(np.float32)
        lo = np.quantile(y_tr, 1.0 - clip_q).astype(np.float32) if np.any(y_tr < 0) else 0.0
        y_tr_clip = np.clip(y_tr, lo, hi).astype(np.float32)

        X_tr, X_va = _impute_and_scale_train_valid(X[tr_valid], X[va_valid])
        reg = Ridge(alpha=1.0)
        try:
            reg.fit(X_tr, y_tr_clip)
            preds = reg.predict(X_va).astype(np.float32)
        except Exception as e:
            _log_bounded_warning("ridge_fit", f"Ridge fold fit failed: {e}")
            continue

        y_va = y[va_valid]
        sst = float(np.sum((y_va - np.mean(y_va)) ** 2))
        if sst < 1e-9:
            continue
        ssr = float(np.sum((y_va - preds) ** 2))
        scores.append(float(1.0 - ssr / sst))

    return np.asarray(scores, dtype=np.float32)


def _extract_learnability_features(feature_dict: Dict[str, np.ndarray], n_samples: int) -> np.ndarray:
    keys = list(_required_feature_keys())
    learnability_keys = [k for k in keys if k not in {"atr", "vol_regime_z"}]
    X = np.full((n_samples, len(learnability_keys)), np.nan, dtype=np.float32)
    for i, k in enumerate(learnability_keys):
        if k not in feature_dict:
            X[:, i] = 0.0
            continue
        arr = np.asarray(feature_dict[k], dtype=np.float32)
        arr = arr.copy()
        arr[np.isinf(arr)] = np.nan
        X[:, i] = arr
    return X


def _required_feature_keys() -> Tuple[str, ...]:
    return (
        "atr",
        "vol_regime_z",
        "range_1_atr",
        "close_location_in_bar",
        "rv_ratio_6_24",
        "impulse_vol_ratio",
        "vol_compression_ratio",
        "range_decay",
        "momentum_last_3bars_impulse_return",
        "reversal_bar_strength",
        "climax_volume_ratio",
        "rejection_volume_ratio",
        "vol_regime_shift",
        "bar_direction_entropy",
    )


def _flatten_wide_frame(df: pd.DataFrame, index: pd.Index, columns: pd.Index) -> np.ndarray:
    return df.reindex(index=index, columns=columns).to_numpy(dtype=np.float32, copy=False).reshape(-1)


def _build_day_ids(timestamps: np.ndarray) -> Tuple[np.ndarray, int]:
    days = timestamps.astype("datetime64[D]")
    uniq, inv = np.unique(days, return_inverse=True)
    return inv.astype(np.int32), int(uniq.shape[0])


def _build_timestamp_ids(timestamps: np.ndarray) -> Tuple[np.ndarray, int]:
    uniq, inv = np.unique(timestamps, return_inverse=True)
    return inv.astype(np.int32), int(uniq.shape[0])


def _build_vol_regime_ids(vol_feature: np.ndarray) -> np.ndarray:
    x = np.asarray(vol_feature, dtype=np.float32)
    valid = np.isfinite(x)
    out = np.ones(x.shape[0], dtype=np.int8)
    if np.sum(valid) < 10:
        return out
    q1 = np.quantile(x[valid], 1 / 3).astype(np.float32)
    q2 = np.quantile(x[valid], 2 / 3).astype(np.float32)
    out[x <= q1] = 0
    out[(x > q1) & (x <= q2)] = 1
    out[x > q2] = 2
    return out


def _sample_half_history_mask(day_ids: np.ndarray, seed: int = 42) -> np.ndarray:
    uniq_days = np.unique(day_ids)
    selected_days = list(set(_rng_sample_half(uniq_days.tolist(), seed=seed)))
    return np.isin(day_ids, selected_days)


def _sample_history_fraction_mask(day_ids: np.ndarray, frac: float, seed: int = 42) -> np.ndarray:
    uniq_days = np.unique(day_ids)
    selected_days = list(set(_rng_sample_fraction(uniq_days.tolist(), frac=frac, seed=seed)))
    return np.isin(day_ids, selected_days)


def _validate_long_panel_shape(
    timestamps: np.ndarray,
    symbols: np.ndarray,
    require_rectangular: bool = False,
) -> None:
    if timestamps.shape[0] == 0:
        return
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("Long panel must be sorted by timestamp ascending.")

    uniq_ts, first_idx, counts = np.unique(timestamps, return_index=True, return_counts=True)
    if require_rectangular and np.unique(counts).shape[0] > 1:
        raise ValueError("Rectangular panel required, but per-timestamp row counts differ.")

    ref_order = None
    for pos, st in enumerate(first_idx):
        c = counts[pos]
        curr = symbols[st : st + c]
        if ref_order is None:
            ref_order = curr
        elif require_rectangular and (c != ref_order.shape[0] or np.any(curr != ref_order)):
            raise ValueError("Rectangular panel required, but symbol ordering differs by timestamp.")


def _safe_param_to_string(param: Any) -> str:
    if isinstance(param, tuple):
        return str(tuple(float(x) for x in param))
    return str(float(param)) if isinstance(param, (int, float, np.integer, np.floating)) else str(param)


def _build_candidate_grid(cfg: Dict[str, Any]) -> List[Tuple[int, str, Any, int]]:
    grid: List[Tuple[int, str, Any, int]] = []
    duration_grid = cfg.get("duration_grid", [1, 2, 4])
    for z_hr in cfg.get("z_hours_grid", [6, 10, 16]):
        for fam in cfg.get("families", ["std_threshold", "abs_move_threshold", "std_plus_abs"]):
            for d_hr in duration_grid:
                if fam == "std_threshold":
                    for p in cfg.get("x_std_grid", [1.4, 1.5, 1.6]):
                        grid.append((z_hr, fam, float(p), d_hr))
                elif fam == "abs_move_threshold":
                    for p in cfg.get("y_move_pct_grid", [4.0, 5.0, 6.0]):
                        grid.append((z_hr, fam, float(p), d_hr))
                elif fam == "std_plus_abs":
                    for s in cfg.get("std_plus_abs_std_grid", [1.4, 1.5, 1.6]):
                        for a in cfg.get("std_plus_abs_abs_grid", [4.0, 5.0, 6.0]):
                            grid.append((z_hr, fam, (float(s), float(a)), d_hr))
    return grid


def _build_asset_groups_from_codes(
    symbol_codes: np.ndarray,
    n_symbols: int,
) -> Dict[int, np.ndarray]:
    asset_groups: Dict[int, np.ndarray] = {}
    for aid in range(n_symbols):
        idxs = np.where(symbol_codes == aid)[0].astype(np.int32)
        if idxs.shape[0] > 0:
            asset_groups[int(aid)] = idxs
    return asset_groups


def _compute_z_cache(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ret_1: np.ndarray,
    vol_g: np.ndarray,
    asset_groups: Dict[int, np.ndarray],
    z: int,
    bph: int,
) -> Dict[str, np.ndarray]:
    n = close.shape[0]
    cache = {
        "up": np.zeros(n, dtype=np.float32),
        "dn": np.zeros(n, dtype=np.float32),
        "rng": np.zeros(n, dtype=np.float32),
        "std_up": np.zeros(n, dtype=np.float32),
        "std_dn": np.zeros(n, dtype=np.float32),
        "b_up": np.zeros(n, dtype=np.float32),
        "b_dn": np.zeros(n, dtype=np.float32),
        "s_up": np.zeros(n, dtype=np.float32),
        "s_dn": np.zeros(n, dtype=np.float32),
        "m_up": np.zeros(n, dtype=np.float32),
        "m_dn": np.zeros(n, dtype=np.float32),
        "v_exp": np.zeros(n, dtype=np.float32),
    }

    for _, idxs in asset_groups.items():
        ast_high = high[idxs]
        ast_low = low[idxs]
        ast_close = close[idxs]
        ast_ret = ret_1[idxs]
        ast_vol = vol_g[idxs]

        hv, hi = rolling_max_index_safe(ast_high, z)
        lv, li = rolling_min_index_safe(ast_low, z)
        st_idx = np.maximum(0, np.arange(ast_close.shape[0], dtype=np.int32) - z + 1)
        st_px = ast_close[st_idx]

        um = np.where(st_px > 1e-9, (hv - st_px) / st_px, 0.0).astype(np.float32)
        dm = np.where(st_px > 1e-9, (st_px - lv) / st_px, 0.0).astype(np.float32)
        rm = np.where(st_px > 1e-9, (hv - lv) / st_px, 0.0).astype(np.float32)

        b_u, b_d, s_u, s_d, m_u, m_d, v_e = compute_impulse_coherence_safe(
            ast_ret, ast_vol, hv, lv, st_px, hi, li, st_idx, z
        )

        cache["up"][idxs] = um
        cache["dn"][idxs] = dm
        cache["rng"][idxs] = rm
        cache["std_up"][idxs] = rolling_std_safe(um, 30 * 24 * bph).astype(np.float32)
        cache["std_dn"][idxs] = rolling_std_safe(dm, 30 * 24 * bph).astype(np.float32)
        cache["b_up"][idxs] = b_u
        cache["b_dn"][idxs] = b_d
        cache["s_up"][idxs] = s_u
        cache["s_dn"][idxs] = s_d
        cache["m_up"][idxs] = m_u
        cache["m_dn"][idxs] = m_d
        cache["v_exp"][idxs] = v_e

    return cache


def _balanced_sample_indices(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    max_each: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_each <= 0:
        return idx_a, idx_b

    rng = np.random.RandomState(seed)

    def _sample(idx: np.ndarray) -> np.ndarray:
        if idx.shape[0] <= max_each:
            return idx
        sampled = rng.choice(idx, max_each, replace=False)
        sampled.sort()
        return sampled.astype(np.int32)

    return _sample(idx_a), _sample(idx_b)


def _cap_rows_for_optimization(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    cfg: Dict[str, Any],
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]:
    max_rows = int(cfg.get("mask_opt_max_rows", 10_000))
    if max_rows <= 0 or data.shape[0] <= max_rows:
        return data, feature_dict, forward_returns

    tprint(f"Capping at {max_rows} rows (random sample)...")
    rng = np.random.RandomState(seed)
    indices = rng.choice(data.shape[0], max_rows, replace=False)
    indices.sort()

    data_capped = data.iloc[indices].reset_index(drop=True)
    forward_capped = forward_returns[indices]
    feature_dict_capped = {k: v[indices] for k, v in feature_dict.items()}
    return data_capped, feature_dict_capped, forward_capped


def _materialize_layer_runtime_cfg(cfg: Dict[str, Any], layer_name: str) -> Dict[str, Any]:
    runtime_cfg = dict(cfg)
    if layer_name == "layer1":
        runtime_cfg["mask_opt_max_rows"] = int(cfg.get("layer1_mask_opt_max_rows", cfg.get("mask_opt_max_rows", 10_000)))
        runtime_cfg["phase1_classifier_max_samples_per_class"] = int(
            cfg.get("layer1_phase1_classifier_max_samples_per_class", cfg.get("phase1_classifier_max_samples_per_class", 15_000))
        )
        runtime_cfg["phase2_metric_max_samples_per_class"] = int(
            cfg.get("layer1_phase2_metric_max_samples_per_class", cfg.get("phase2_metric_max_samples_per_class", 25_000))
        )
        runtime_cfg["phase1_classifier_n_splits"] = int(cfg.get("layer1_phase1_classifier_n_splits", 2))
        runtime_cfg["phase2_classifier_n_splits"] = int(cfg.get("layer1_phase2_classifier_n_splits", 2))
        runtime_cfg["phase2_metric_fold_splits"] = int(cfg.get("layer1_phase2_metric_fold_splits", 2))
        runtime_cfg["incremental_information_n_splits"] = int(cfg.get("layer1_incremental_information_n_splits", 2))
    else:
        runtime_cfg["phase1_classifier_n_splits"] = int(cfg.get("phase1_classifier_n_splits", 2))
        runtime_cfg["phase2_classifier_n_splits"] = int(cfg.get("phase2_classifier_n_splits", 2))
        runtime_cfg["phase2_metric_fold_splits"] = int(cfg.get("phase2_metric_fold_splits", 3))
        runtime_cfg["incremental_information_n_splits"] = int(cfg.get("incremental_information_n_splits", 3))
    runtime_cfg["regime_score_layer"] = layer_name
    return runtime_cfg


def _rescale_mode_gates_for_sample_size(cfg: Dict[str, Any], n_rows: int) -> Dict[str, Any]:
    runtime_cfg = dict(cfg)
    if n_rows <= 0:
        return runtime_cfg

    # Per-bucket masks are sparse on capped samples; fixed 5k-event gates are too high.
    target_event_density = float(runtime_cfg.get("mask_opt_target_event_density", 0.025))
    min_events_floor = int(runtime_cfg.get("mask_opt_min_events_floor", 150))
    scaled_min_events = max(min_events_floor, int(round(n_rows * target_event_density)))

    base_active = float(runtime_cfg.get("phase2_min_active_days_fraction", 0.80))
    active_days_floor = float(runtime_cfg.get("mask_opt_min_active_days_floor", 0.25))
    scaled_active_days = min(base_active, max(active_days_floor, base_active * np.sqrt(n_rows / 300_000.0)))

    runtime_cfg["phase1_min_total_events"] = scaled_min_events
    runtime_cfg["phase2_min_total_events"] = scaled_min_events
    runtime_cfg["phase1_min_active_days_fraction"] = scaled_active_days
    runtime_cfg["phase2_min_active_days_fraction"] = scaled_active_days
    tprint(
        "Rescaled per-bucket gates for capped sample: "
        f"rows={n_rows}, min_events={scaled_min_events}, min_active_days_fraction={scaled_active_days:.3f}"
    )
    return runtime_cfg


def _generate_event_masks_fast(
    family: str,
    param_val: Any,
    up_move: np.ndarray,
    dn_move: np.ndarray,
    rolling_std_up: np.ndarray,
    rolling_std_dn: np.ndarray,
    asset_groups: Optional[Dict[int, np.ndarray]],
    duration_bars: int,
) -> Tuple[np.ndarray, np.ndarray]:
    mask_h = np.zeros(up_move.shape[0], dtype=bool)
    mask_l = np.zeros(dn_move.shape[0], dtype=bool)

    std_up_floored = np.maximum(rolling_std_up, 1e-6)
    std_dn_floored = np.maximum(rolling_std_dn, 1e-6)

    if family == "std_threshold":
        x_std = float(param_val)
        mask_h = up_move >= (x_std * std_up_floored)
        mask_l = dn_move >= (x_std * std_dn_floored)

    elif family == "abs_move_threshold":
        y_move = float(param_val) / 100.0
        mask_h = up_move >= y_move
        mask_l = dn_move >= y_move

    elif family == "std_plus_abs":
        std_val, abs_val_pct = param_val
        y_move = float(abs_val_pct) / 100.0
        mask_h = (up_move >= float(std_val) * std_up_floored) & (up_move >= y_move)
        mask_l = (dn_move >= float(std_val) * std_dn_floored) & (dn_move >= y_move)

    else:
        raise ValueError(f"Unknown family: {family}")

    if duration_bars > 1 and asset_groups is not None:
        mask_h = dilate_mask_by_asset_safe(mask_h, asset_groups, duration_bars)
        mask_l = dilate_mask_by_asset_safe(mask_l, asset_groups, duration_bars)

    return mask_h, mask_l


def _generate_event_masks(
    family: str,
    param_val: Any,
    up_move: np.ndarray,
    dn_move: np.ndarray,
    rolling_std_up: np.ndarray,
    rolling_std_dn: np.ndarray,
    asset_groups: Optional[Dict[int, np.ndarray]] = None,
    duration_bars: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for event mask generation.

    Several inference modules import ``_generate_event_masks`` directly.
    Keep this stable alias so those modules continue to import successfully
    after the fast path refactor.
    """
    return _generate_event_masks_fast(
        family=family,
        param_val=param_val,
        up_move=up_move,
        dn_move=dn_move,
        rolling_std_up=rolling_std_up,
        rolling_std_dn=rolling_std_dn,
        asset_groups=asset_groups,
        duration_bars=duration_bars,
    )


def _simple_score_for_mode(
    mode: str,
    feature_dict: Dict[str, np.ndarray],
    side_mask: np.ndarray,
) -> np.ndarray:
    n = side_mask.shape[0]
    score = np.zeros(n, dtype=np.float32)

    def get(name: str) -> np.ndarray:
        if name not in feature_dict:
            return np.zeros(n, dtype=np.float32)
        return np.nan_to_num(np.asarray(feature_dict[name], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # very simple fixed interpretable score
    impulse = get("momentum_last_3bars_impulse_return")
    vol = get("climax_volume_ratio")
    rev = get("reversal_bar_strength")
    rng = get("range_1_atr")
    entropy = get("bar_direction_entropy")

    if mode == MODE_PRICE_UP_TF:
        score = 0.35 * impulse + 0.20 * vol + 0.20 * rng - 0.15 * rev - 0.10 * entropy
    elif mode == MODE_PRICE_UP_MR:
        score = 0.35 * rev + 0.25 * rng + 0.20 * vol - 0.20 * impulse
    elif mode == MODE_PRICE_DOWN_TF:
        score = 0.35 * (-impulse) + 0.20 * vol + 0.20 * rng - 0.15 * rev - 0.10 * entropy
    elif mode == MODE_PRICE_DOWN_MR:
        score = 0.35 * rev + 0.25 * rng + 0.20 * vol + 0.20 * impulse
    else:
        score = np.zeros(n, dtype=np.float32)

    score[~side_mask] = np.nan
    return score.astype(np.float32)


# =============================================================================
# SHARED CACHE BUILD
# =============================================================================

def _build_shared_cache(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    tprint("Building shared cache...")
    bph = int(cfg.get("bars_per_hour", 1))
    horizon = int(cfg.get("phase1_forward_horizon_bars", 12))

    high = np.asarray(data["high"].values, dtype=np.float32)
    low = np.asarray(data["low"].values, dtype=np.float32)
    close = np.asarray(data["close"].values, dtype=np.float32)
    timestamps = pd.to_datetime(data["timestamp"]).values
    symbols = np.asarray(data["symbol"].astype(str).values)
    _validate_long_panel_shape(timestamps, symbols, require_rectangular=False)

    forward_returns = np.asarray(forward_returns, dtype=np.float32)
    atr = np.asarray(feature_dict.get("atr", np.ones_like(close, dtype=np.float32)), dtype=np.float32)

    # ids
    symbol_uniques, symbol_codes = np.unique(symbols, return_inverse=True)
    symbol_codes = symbol_codes.astype(np.int32)
    day_ids, n_days = _build_day_ids(timestamps)
    timestamp_ids, n_timestamps = _build_timestamp_ids(timestamps)
    regime_source = np.asarray(feature_dict.get("vol_regime_z", np.zeros_like(close, dtype=np.float32)), dtype=np.float32)
    if regime_source.shape[0] != close.shape[0]:
        regime_source = np.zeros_like(close, dtype=np.float32)
    regime_ids = _build_vol_regime_ids(regime_source)

    # per-asset groups
    asset_groups = _build_asset_groups_from_codes(symbol_codes, symbol_uniques.shape[0])

    # returns / vol / alternation
    ret_1 = np.zeros(close.shape[0], dtype=np.float32)
    vol_g = np.zeros(close.shape[0], dtype=np.float32)
    alternation = np.zeros(close.shape[0], dtype=np.float32)

    for aid, idxs in asset_groups.items():
        c = close[idxs]
        r = np.zeros(idxs.shape[0], dtype=np.float32)
        if idxs.shape[0] > 1:
            prev = np.where(c[:-1] > 1e-9, c[:-1], 1.0)
            r[1:] = ((c[1:] - c[:-1]) / prev).astype(np.float32)
        ret_1[idxs] = r
        vol_g[idxs] = rolling_std_safe(r, 30 * 24 * bph).astype(np.float32)

        sign = np.sign(r).astype(np.float32)
        prev_sign = np.zeros_like(sign)
        if sign.shape[0] > 1:
            prev_sign[1:] = sign[:-1]
        alt = (sign != prev_sign).astype(np.float32)
        # lightweight rolling mean
        window = 6
        s = 0.0
        for i in range(alt.shape[0]):
            s += alt[i]
            if i >= window:
                s -= alt[i - window]
            alternation[idxs[i]] = s / float(min(i + 1, window))

    # MAE / MFE
    n = close.shape[0]
    mae_high = np.zeros(n, dtype=np.float32)
    mfe_high = np.zeros(n, dtype=np.float32)
    mae_low = np.zeros(n, dtype=np.float32)
    mfe_low = np.zeros(n, dtype=np.float32)

    # compute by asset to avoid cross-asset leakage
    for aid, idxs in asset_groups.items():
        h = high[idxs]
        l = low[idxs]
        c = close[idxs]
        a = atr[idxs]
        m1 = np.zeros(idxs.shape[0], dtype=np.float32)
        m2 = np.zeros(idxs.shape[0], dtype=np.float32)
        m3 = np.zeros(idxs.shape[0], dtype=np.float32)
        m4 = np.zeros(idxs.shape[0], dtype=np.float32)
        for i in range(max(0, idxs.shape[0] - horizon)):
            h_sl = h[i + 1 : i + horizon + 1]
            l_sl = l[i + 1 : i + horizon + 1]
            if h_sl.shape[0] == 0:
                continue
            atr_i = max(a[i], 1e-9)
            c_i = c[i]
            m1[i] = (c_i - np.min(l_sl)) / atr_i
            m2[i] = (np.max(h_sl) - c_i) / atr_i
            m3[i] = (np.max(h_sl) - c_i) / atr_i
            m4[i] = (c_i - np.min(l_sl)) / atr_i
        mae_high[idxs] = m1
        mfe_high[idxs] = m2
        mae_low[idxs] = m3
        mfe_low[idxs] = m4

    # learnability features
    learn_X = _extract_learnability_features(feature_dict, n)

    # full folds
    folds = _build_temporal_folds(timestamps, n, n_splits=2)

    return {
        "bph": bph,
        "high": high,
        "low": low,
        "close": close,
        "timestamps": timestamps,
        "symbols": symbols,
        "symbol_uniques": symbol_uniques,
        "symbol_codes": symbol_codes,
        "asset_groups": asset_groups,
        "forward_returns": forward_returns,
        "atr": atr,
        "ret_1": ret_1,
        "vol_g": vol_g,
        "alternation": alternation,
        "mae_high": mae_high,
        "mfe_high": mfe_high,
        "mae_low": mae_low,
        "mfe_low": mfe_low,
        "learn_X": learn_X,
        "day_ids": day_ids,
        "n_days": n_days,
        "timestamp_ids": timestamp_ids,
        "n_timestamps": n_timestamps,
        "regime_ids": regime_ids,
        "folds": folds,
        "z_grid": sorted(set(int(z * bph) for z in cfg.get("z_hours_grid", [6, 10, 16]))),
        "candidate_grid": _build_candidate_grid(cfg),
    }


# =============================================================================
# PHASE 1 + PHASE 2
# =============================================================================

def _phase1_subsample_indices(shared: Dict[str, Any], cfg: Dict[str, Any], seed: int = 42) -> np.ndarray:
    symbol_uniques = shared["symbol_uniques"]
    symbol_codes = shared["symbol_codes"]
    day_ids = shared["day_ids"]

    symbol_frac = float(cfg.get("stage1_symbol_fraction", 0.35))
    history_frac = float(cfg.get("stage1_history_fraction", 0.35))
    selected_symbols = _rng_sample_fraction(list(range(symbol_uniques.shape[0])), frac=symbol_frac, seed=seed)
    symbol_mask = np.isin(symbol_codes, np.asarray(selected_symbols, dtype=np.int32))

    history_mask = _sample_history_fraction_mask(day_ids, frac=history_frac, seed=seed)
    return symbol_mask & history_mask


def _build_phase_local_shared(
    shared: Dict[str, Any],
    subset_mask: np.ndarray,
) -> Dict[str, Any]:
    symbol_codes_local = shared["symbol_codes"][subset_mask]
    day_ids_local_raw = shared["day_ids"][subset_mask]
    _, day_ids_local = np.unique(day_ids_local_raw, return_inverse=True)
    phase_local = {
        "high": shared["high"][subset_mask],
        "low": shared["low"][subset_mask],
        "close": shared["close"][subset_mask],
        "ret_1": shared["ret_1"][subset_mask],
        "vol_g": shared["vol_g"][subset_mask],
        "timestamps": shared["timestamps"][subset_mask],
        "forward_returns": shared["forward_returns"][subset_mask],
        "mae_high": shared["mae_high"][subset_mask],
        "mfe_high": shared["mfe_high"][subset_mask],
        "mae_low": shared["mae_low"][subset_mask],
        "mfe_low": shared["mfe_low"][subset_mask],
        "learn_X": shared["learn_X"][subset_mask],
        "day_ids": day_ids_local.astype(np.int32),
        "symbol_codes": symbol_codes_local,
        "asset_groups": _build_asset_groups_from_codes(symbol_codes_local, shared["symbol_uniques"].shape[0]),
    }
    phase_local["n_days"] = int(np.max(phase_local["day_ids"]) + 1) if phase_local["day_ids"].shape[0] > 0 else 0
    return phase_local


def _compute_primary_phase1_classifier_gain(
    mode: str,
    side_mask: np.ndarray,
    learn_X: np.ndarray,
    forward_returns: np.ndarray,
    timestamps: np.ndarray,
    ret_threshold: float,
    max_samples_per_class: int = 0,
    n_splits: int = 2,
) -> float:
    y_global = _mode_primary_target(mode, forward_returns, ret_threshold)
    valid = np.isfinite(forward_returns)
    idx_ne = np.where(valid & ~side_mask)[0].astype(np.int32)
    idx_e = np.where(valid & side_mask)[0].astype(np.int32)

    if idx_e.shape[0] < 50 or idx_ne.shape[0] < 50:
        return float("nan")

    idx_e, idx_ne = _balanced_sample_indices(idx_e, idx_ne, max_samples_per_class, seed=42)
    auc_ne = _classifier_oof_auc(learn_X[idx_ne], y_global[idx_ne], timestamps[idx_ne], n_splits=n_splits)
    auc_e = _classifier_oof_auc(learn_X[idx_e], y_global[idx_e], timestamps[idx_e], n_splits=n_splits)
    return float(auc_e - auc_ne)


def _compute_full_metrics_for_candidate(
    mode: str,
    side_mask: np.ndarray,
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    impulse_shape_dispersion: float,
    basic_directionality_edge: float,
) -> Dict[str, float]:
    ret_threshold = float(cfg.get("phase1_ret_threshold", 0.0))
    learn_X = shared["learn_X"]
    forward_returns = shared["forward_returns"]
    timestamps = shared["timestamps"]

    y_primary = _mode_primary_target(mode, forward_returns, ret_threshold)
    valid = np.isfinite(forward_returns)
    idx_ne = np.where(valid & ~side_mask)[0].astype(np.int32)
    idx_e = np.where(valid & side_mask)[0].astype(np.int32)

    metrics: Dict[str, float] = {
        "primary_predictability_gain": float("nan"),
        "continuation_predictability_gain": float("nan"),
        "reversal_predictability_gain": float("nan"),
        "bucket_primary_delta_fold_mean": float("nan"),
        "bucket_primary_delta_fold_std": float("nan"),
        "bucket_primary_delta_fold_count": float("nan"),
        "MAE_predictability_gain": float("nan"),
        "MFE_predictability_gain": float("nan"),
        "reversal_utility_gain": float("nan"),
        "mae_event_oos_r2": float("nan"),
        "mfe_event_oos_r2": float("nan"),
        "magnitude_delta_r": float("nan"),
        "magnitude_positive_fold_fraction": float("nan"),
        "magnitude_stability_score": float("nan"),
        "magnitude_fold_count": float("nan"),
        "magnitude_delta_fold_mean": float("nan"),
        "magnitude_delta_fold_std": float("nan"),
        "selected_delta_metric": "",
        "incremental_information_delta_auc": float("nan"),
        "incremental_information_positive_fold_fraction": float("nan"),
        "incremental_information_positive_fold_count": float("nan"),
        "incremental_information_fold_count": float("nan"),
        "dispersion_to_edge_ratio": float("nan"),
        "edge_to_dispersion_ratio": float("nan"),
        "primary_predictability_gain_is_nan": 1.0,
    }

    if idx_e.shape[0] < 50 or idx_ne.shape[0] < 50:
        if np.isfinite(impulse_shape_dispersion) and np.isfinite(basic_directionality_edge):
            metrics["dispersion_to_edge_ratio"] = float(
                impulse_shape_dispersion / max(abs(basic_directionality_edge), 1e-6)
            )
            metrics["edge_to_dispersion_ratio"] = float(
                abs(basic_directionality_edge) / max(impulse_shape_dispersion, 1e-6)
            )
        return metrics

    max_samples_per_class = int(cfg.get("phase2_metric_max_samples_per_class", 25_000))
    classifier_n_splits = int(cfg.get("phase2_classifier_n_splits", 2))
    metric_fold_splits = int(cfg.get("phase2_metric_fold_splits", 3))
    incremental_info_n_splits = int(cfg.get("incremental_information_n_splits", 3))
    idx_e, idx_ne = _balanced_sample_indices(idx_e, idx_ne, max_samples_per_class, seed=123)

    # primary classifier
    auc_ne = _classifier_oof_auc(learn_X[idx_ne], y_primary[idx_ne], timestamps[idx_ne], n_splits=classifier_n_splits)
    auc_e = _classifier_oof_auc(learn_X[idx_e], y_primary[idx_e], timestamps[idx_e], n_splits=classifier_n_splits)
    primary_gain = float(auc_e - auc_ne)
    metrics["primary_predictability_gain"] = primary_gain
    metrics["primary_predictability_gain_is_nan"] = 0.0
    primary_delta_folds = _primary_gain_fold_deltas(
        learn_X=learn_X,
        y_primary=y_primary,
        timestamps=timestamps,
        idx_e=idx_e,
        idx_ne=idx_ne,
        n_splits=classifier_n_splits,
    )
    if primary_delta_folds.size > 0:
        metrics["bucket_primary_delta_fold_mean"] = float(np.mean(primary_delta_folds))
        metrics["bucket_primary_delta_fold_std"] = float(np.std(primary_delta_folds))
        metrics["bucket_primary_delta_fold_count"] = float(primary_delta_folds.size)

    # classify it into continuation/reversal labels for reporting
    if _mode_is_tf(mode):
        metrics["continuation_predictability_gain"] = primary_gain
    else:
        metrics["reversal_predictability_gain"] = primary_gain

    metrics.update(
        _incremental_information_metrics(
            learn_X=learn_X,
            side_mask=side_mask,
            y_primary=y_primary,
            timestamps=timestamps,
            idx_e=idx_e,
            idx_ne=idx_ne,
            n_splits=incremental_info_n_splits,
        )
    )
    metrics["dispersion_to_edge_ratio"] = float(
        impulse_shape_dispersion / max(abs(basic_directionality_edge), 1e-6)
    )
    metrics["edge_to_dispersion_ratio"] = float(
        abs(basic_directionality_edge) / max(impulse_shape_dispersion, 1e-6)
    )

    # regression targets
    if mode == MODE_PRICE_UP_TF:
        mae_arr = shared["mae_high"]
        mfe_arr = shared["mfe_high"]
        reversal_utility = (-_signed_mode_return(MODE_PRICE_UP_TF, forward_returns)).astype(np.float32)
    elif mode == MODE_PRICE_UP_MR:
        mae_arr = shared["mae_high"]
        mfe_arr = shared["mfe_high"]
        reversal_utility = _signed_mode_return(mode, forward_returns)
    elif mode == MODE_PRICE_DOWN_TF:
        mae_arr = shared["mae_low"]
        mfe_arr = shared["mfe_low"]
        reversal_utility = (-_signed_mode_return(MODE_PRICE_DOWN_TF, forward_returns)).astype(np.float32)
    else:
        mae_arr = shared["mae_low"]
        mfe_arr = shared["mfe_low"]
        reversal_utility = _signed_mode_return(mode, forward_returns)

    mae_ne = _ridge_regression_oof_r2(
        learn_X[idx_ne], mae_arr[idx_ne], timestamps[idx_ne], clip_q=0.98, n_splits=classifier_n_splits
    )
    mae_e = _ridge_regression_oof_r2(
        learn_X[idx_e], mae_arr[idx_e], timestamps[idx_e], clip_q=0.98, n_splits=classifier_n_splits
    )
    metrics["MAE_predictability_gain"] = float(mae_e - mae_ne)
    metrics["mae_event_oos_r2"] = float(mae_e)

    mfe_ne = _ridge_regression_oof_r2(
        learn_X[idx_ne], mfe_arr[idx_ne], timestamps[idx_ne], clip_q=0.98, n_splits=classifier_n_splits
    )
    mfe_e = _ridge_regression_oof_r2(
        learn_X[idx_e], mfe_arr[idx_e], timestamps[idx_e], clip_q=0.98, n_splits=classifier_n_splits
    )
    metrics["MFE_predictability_gain"] = float(mfe_e - mfe_ne)
    metrics["mfe_event_oos_r2"] = float(mfe_e)

    rev_ne = _ridge_regression_oof_r2(
        learn_X[idx_ne], reversal_utility[idx_ne], timestamps[idx_ne], clip_q=0.98, n_splits=classifier_n_splits
    )
    rev_e = _ridge_regression_oof_r2(
        learn_X[idx_e], reversal_utility[idx_e], timestamps[idx_e], clip_q=0.98, n_splits=classifier_n_splits
    )
    metrics["reversal_utility_gain"] = float(rev_e - rev_ne)

    mae_folds = _ridge_regression_fold_r2s(
        learn_X[idx_e], mae_arr[idx_e], timestamps[idx_e], clip_q=0.98, n_splits=metric_fold_splits
    )
    mfe_folds = _ridge_regression_fold_r2s(
        learn_X[idx_e], mfe_arr[idx_e], timestamps[idx_e], clip_q=0.98, n_splits=metric_fold_splits
    )

    if np.isfinite(metrics["mfe_event_oos_r2"]) and (
        not np.isfinite(metrics["mae_event_oos_r2"]) or metrics["mfe_event_oos_r2"] >= metrics["mae_event_oos_r2"]
    ):
        selected_folds = mfe_folds
        metrics["magnitude_delta_r"] = float(metrics["mfe_event_oos_r2"])
        metrics["selected_delta_metric"] = "mfe_event_oos_r2"
    else:
        selected_folds = mae_folds
        metrics["magnitude_delta_r"] = float(metrics["mae_event_oos_r2"])
        metrics["selected_delta_metric"] = "mae_event_oos_r2"

    stability = _stability_from_fold_deltas(selected_folds)
    metrics["magnitude_positive_fold_fraction"] = float(stability["positive_fold_fraction"])
    metrics["magnitude_stability_score"] = float(stability["stability_score"])
    metrics["magnitude_fold_count"] = float(stability["fold_count"])
    metrics["magnitude_delta_fold_mean"] = float(stability["delta_fold_mean"])
    metrics["magnitude_delta_fold_std"] = float(stability["delta_fold_std"])

    return metrics


def _final_topk_diagnostics(
    mode: str,
    contenders: pd.DataFrame,
    candidate_masks: Dict[str, Dict[str, np.ndarray]],
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    forward_returns = shared["forward_returns"]
    timestamps = shared["timestamps"]
    symbol_codes = shared["symbol_codes"]
    symbol_uniques = shared["symbol_uniques"]
    regime_ids = shared["regime_ids"]

    ret_threshold = float(cfg.get("phase1_ret_threshold", 0.0))
    min_asset_events = int(cfg.get("diag_min_asset_events", 30))
    min_regime_events = int(cfg.get("diag_min_regime_events", 50))

    for _, row in contenders.iterrows():
        name = row["name"]
        masks = candidate_masks[name]
        side_mask = _get_side_mask(mode, masks["m_high"], masks["m_low"])

        # A. cross-asset generalization
        asset_scores: List[float] = []
        y = _mode_primary_target(mode, forward_returns, ret_threshold)
        valid_fwd = np.isfinite(forward_returns)
        for aid in range(symbol_uniques.shape[0]):
            idx = np.where((symbol_codes == aid) & valid_fwd)[0]
            sub_mask = side_mask[idx]
            if np.sum(sub_mask) < min_asset_events:
                continue
            score = float(np.mean(y[idx][sub_mask]) - np.mean(y[idx]))
            asset_scores.append(score)

        if asset_scores:
            asset_scores_arr = np.asarray(asset_scores, dtype=np.float32)
            median_asset_pred = float(np.median(asset_scores_arr))
            mean_asset_pred = float(np.mean(asset_scores_arr))
            p25_asset_pred = float(np.quantile(asset_scores_arr, 0.25))
            p75_asset_pred = float(np.quantile(asset_scores_arr, 0.75))
            share_assets_pos = float(np.mean(asset_scores_arr > 0))
            n_assets_eval = int(asset_scores_arr.shape[0])
        else:
            median_asset_pred = mean_asset_pred = p25_asset_pred = p75_asset_pred = 0.0
            share_assets_pos = 0.0
            n_assets_eval = 0

        # B. regime stability
        regime_preds = {}
        y_signed = _signed_mode_return(mode, forward_returns)
        valid_fwd = np.isfinite(forward_returns)
        for rid, lbl in [(0, "low"), (1, "normal"), (2, "high")]:
            m = side_mask & (regime_ids == rid) & valid_fwd
            if np.sum(m) < min_regime_events:
                regime_preds[lbl] = np.nan
            else:
                regime_preds[lbl] = float(np.mean(y_signed[m]))
        regime_vals = np.array([regime_preds["low"], regime_preds["normal"], regime_preds["high"]], dtype=np.float32)
        valid_reg = regime_vals[np.isfinite(regime_vals)]
        if valid_reg.shape[0] > 0:
            regime_std = float(np.std(valid_reg))
            regime_min = float(np.min(valid_reg))
            regime_max = float(np.max(valid_reg))
        else:
            regime_std = regime_min = regime_max = 0.0

        # C. feature predictability ceiling
        simple_score = _simple_score_for_mode(mode, feature_dict, side_mask)
        valid_idx = np.where(np.isfinite(simple_score) & np.isfinite(forward_returns))[0]
        if valid_idx.shape[0] >= 20:
            s = simple_score[valid_idx]
            y_s = y_signed[valid_idx]
            q80 = np.nanquantile(s, 0.80)
            q20 = np.nanquantile(s, 0.20)
            top_mask = s >= q80
            bot_mask = s <= q20
            top_ret = float(np.nanmean(y_s[top_mask])) if np.any(top_mask) else 0.0
            bot_ret = float(np.nanmean(y_s[bot_mask])) if np.any(bot_mask) else 0.0
            spread = top_ret - bot_ret
        else:
            top_ret = 0.0
            bot_ret = 0.0
            spread = 0.0

        rows.append({
            "mode": mode,
            "contender_name": name,
            "final_shortlist_score": float(row.get("shortlist_score", 0.0)),
            "n_assets_evaluated": n_assets_eval,
            "median_asset_predictability": median_asset_pred,
            "mean_asset_predictability": mean_asset_pred,
            "p25_asset_predictability": p25_asset_pred,
            "p75_asset_predictability": p75_asset_pred,
            "share_assets_positive_predictability": share_assets_pos,
            "predictability_low_vol": regime_preds["low"],
            "predictability_normal_vol": regime_preds["normal"],
            "predictability_high_vol": regime_preds["high"],
            "regime_predictability_std": regime_std,
            "min_regime_predictability": regime_min,
            "max_regime_predictability": regime_max,
            "simple_score_top20_mean_return": top_ret,
            "simple_score_bottom20_mean_return": bot_ret,
            "simple_score_spread": spread,
        })

    return pd.DataFrame(rows)


def _run_mode_search(
    mode: str,
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    tprint("=" * 80)
    tprint(f"LAYER 0 MODE SEARCH: {mode}")
    tprint("=" * 80)

    bph = shared["bph"]
    timestamps = shared["timestamps"]
    day_ids = shared["day_ids"]
    n_days = shared["n_days"]
    folds = shared["folds"]
    forward_returns = shared["forward_returns"]

    ret_threshold = float(cfg.get("phase1_ret_threshold", 0.0))
    phase1_mask = _phase1_subsample_indices(shared, cfg, seed=42)
    phase1_shared = _build_phase_local_shared(shared, phase1_mask)
    candidate_grid = shared["candidate_grid"]
    candidate_registry: Dict[str, Dict[str, Any]] = {}

    # cache by geometry key
    geom_cache_phase1: Dict[str, Dict[str, Any]] = {}
    geom_cache_phase2: Dict[str, Dict[str, Any]] = {}
    phase1_z_cache: Dict[int, Dict[str, np.ndarray]] = {}
    phase2_z_cache: Dict[int, Dict[str, np.ndarray]] = {}

    phase1_rows: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Phase 1: 50% symbols + 50% history + cheap metrics + primary classifier only
    # -------------------------------------------------------------------------
    tprint(f"Phase 1 ({mode}): evaluating {len(candidate_grid)} candidates on 50% symbols + 50% history...")

    global_target = _mode_primary_target(mode, forward_returns, ret_threshold)
    phase1_global_idx = np.where(phase1_mask)[0]
    phase1_global_target = global_target[phase1_mask]
    phase1_valid_fwd = np.isfinite(phase1_shared["forward_returns"])
    phase1_n_assets = max(1, len(phase1_shared["asset_groups"]))

    phase1_ratio = float(np.sum(phase1_mask)) / float(phase1_mask.shape[0]) if phase1_mask.shape[0] > 0 else 1.0
    phase1_min_total_events = max(10, int(cfg.get("phase1_min_total_events", 5000) * phase1_ratio))

    global_to_phase1_local = np.full(forward_returns.shape[0], -1, dtype=np.int32)
    global_to_phase1_local[phase1_global_idx] = np.arange(phase1_global_idx.shape[0], dtype=np.int32)
    phase1_fold_val_locals: List[np.ndarray] = []
    for _, va in folds:
        loc = global_to_phase1_local[va]
        loc = loc[loc >= 0].astype(np.int32)
        phase1_fold_val_locals.append(loc)

    for z_hr, fam, param, d_hr in candidate_grid:
        z = int(z_hr * bph)
        duration_bars = int(d_hr * bph)
        key = CandidateKey(fam, int(z_hr), _safe_param_to_string(param), int(d_hr)).as_str()
        candidate_registry[key] = {"family": fam, "z_hours": int(z_hr), "duration_hours": int(d_hr), "param": param}

        if key not in geom_cache_phase1:
            if z not in phase1_z_cache:
                tprint(f"Precomputing Phase 1 rolling tensors for z={z} bars...")
                phase1_z_cache[z] = _compute_z_cache(
                    high=phase1_shared["high"],
                    low=phase1_shared["low"],
                    close=phase1_shared["close"],
                    ret_1=phase1_shared["ret_1"],
                    vol_g=phase1_shared["vol_g"],
                    asset_groups=phase1_shared["asset_groups"],
                    z=z,
                    bph=bph,
                )
            zc = phase1_z_cache[z]
            m_high, m_low = _generate_event_masks_fast(
                family=fam,
                param_val=param,
                up_move=zc["up"],
                dn_move=zc["dn"],
                rolling_std_up=zc["std_up"],
                rolling_std_dn=zc["std_dn"],
                asset_groups=phase1_shared["asset_groups"],
                duration_bars=duration_bars,
            )
            side_mask = _get_side_mask(mode, m_high, m_low)

            total_events = simple_mask_count_safe(side_mask)
            if total_events < phase1_min_total_events:
                geom_cache_phase1[key] = {"rejected": True}
                continue

            active_days_frac = active_days_fraction_safe(side_mask, phase1_shared["day_ids"], phase1_shared["n_days"])
            if active_days_frac < float(cfg.get("phase1_min_active_days_fraction", 0.80)):
                geom_cache_phase1[key] = {"rejected": True}
                continue

            if _mode_is_up(mode):
                coh = _coherence_metrics_single_side(side_mask, zc["b_up"], zc["s_up"], zc["m_up"])
            else:
                coh = _coherence_metrics_single_side(side_mask, zc["b_dn"], zc["s_dn"], zc["m_dn"])

            distinct = _compute_regime_distinctness_single_side(
                side_mask=side_mask,
                mode=mode,
                forward_returns=phase1_shared["forward_returns"],
                mae_high=phase1_shared["mae_high"],
                mfe_high=phase1_shared["mfe_high"],
                mae_low=phase1_shared["mae_low"],
                mfe_low=phase1_shared["mfe_low"],
            )

            ev_day_mean, ev_day_std = daily_event_stats_safe(side_mask, phase1_shared["day_ids"], phase1_shared["n_days"])
            ev_day_per_asset = float(total_events) / float(max(1, phase1_shared["n_days"] * phase1_n_assets))

            fold_rates = []
            fold_event_counts = []
            for val_idx_local in phase1_fold_val_locals:
                if val_idx_local.shape[0] == 0:
                    continue
                fold_event_counts.append(float(np.sum(side_mask[val_idx_local])))
                fold_rates.append(fold_base_rate_safe(side_mask, phase1_global_target, val_idx_local))
            fold_rate_std = float(np.std(np.asarray(fold_rates, dtype=np.float32))) if fold_rates else 1.0
            fold_event_count_std = float(np.std(np.asarray(fold_event_counts, dtype=np.float32))) if fold_event_counts else 1.0

            non_event = (~side_mask) & phase1_valid_fwd
            if np.any(side_mask & phase1_valid_fwd) and np.any(non_event):
                basic_edge = float(
                    np.nanmean(phase1_global_target[side_mask & phase1_valid_fwd]) - np.nanmean(phase1_global_target[non_event])
                )
            else:
                basic_edge = 0.0

            stats = {
                "rejected": False,
                "total_events": int(total_events),
                "active_days_fraction": float(active_days_frac),
                "events_per_day_mean": float(ev_day_mean),
                "events_per_day_std": float(ev_day_std),
                "events_per_day_per_asset": float(ev_day_per_asset),
                "bars_to_peak_dispersion": float(coh["bars_to_peak_dispersion"]),
                "speed_dispersion": float(coh["speed_dispersion"]),
                "monotonicity_dispersion": float(coh["monotonicity_dispersion"]),
                "impulse_shape_dispersion": float(coh["impulse_shape_dispersion"]),
                "regime_distinctness_score": float(distinct),
                "fold_base_rate_stability": float(fold_rate_std),
                "fold_continuation_rate_std": float(fold_rate_std),
                "fold_event_count_std": float(fold_event_count_std),
                "basic_directionality_edge_event_vs_non_event": float(basic_edge),
            }
            geom_cache_phase1[key] = stats

        stats = geom_cache_phase1[key]
        if stats.get("rejected", False):
            continue

        phase1_rows.append({
            "name": key,
            "family": fam,
            "z_hours": z_hr,
            "param": _safe_param_to_string(param),
            "duration_hours": d_hr,
            **{k: v for k, v in stats.items() if k not in {"rejected"}},
        })

    if not phase1_rows:
        return {"status": "failed", "reason": f"no_phase1_candidates_{mode}"}

    df1 = pd.DataFrame(phase1_rows)
    df1["phase1_proxy_score"] = (
        _zscore_np(df1["active_days_fraction"].values)
        + 0.5 * _zscore_np(df1["regime_distinctness_score"].values)
        + 0.5 * _zscore_np(np.log1p(df1["total_events"].values.astype(np.float32)))
        - _zscore_np(df1["impulse_shape_dispersion"].values)
        - 0.5 * _zscore_np(df1["fold_continuation_rate_std"].values)
        - 0.5 * _zscore_np(df1["fold_event_count_std"].values)
    )

    df1 = df1.sort_values("phase1_proxy_score", ascending=False).head(int(cfg.get("top_k_for_learnability", 8))).copy()

    # -------------------------------------------------------------------------
    # Phase 2: full symbols & history + full metrics, only top phase1 candidates
    # -------------------------------------------------------------------------
    tprint(f"Phase 2 ({mode}): full symbols/history for top {len(df1)} candidates...")

    phase2_rows: List[Dict[str, Any]] = []
    phase2_n_assets = max(1, len(shared["asset_groups"]))

    for _, row in df1.iterrows():
        name = row["name"]
        reg = candidate_registry[name]
        fam = reg["family"]
        z_hr = int(reg["z_hours"])
        d_hr = int(reg["duration_hours"])
        param = reg["param"]

        z = int(z_hr * bph)
        duration_bars = int(d_hr * bph)
        key = name

        if key not in geom_cache_phase2:
            if z not in phase2_z_cache:
                tprint(f"Precomputing Phase 2 rolling tensors for z={z} bars...")
                phase2_z_cache[z] = _compute_z_cache(
                    high=shared["high"],
                    low=shared["low"],
                    close=shared["close"],
                    ret_1=shared["ret_1"],
                    vol_g=shared["vol_g"],
                    asset_groups=shared["asset_groups"],
                    z=z,
                    bph=bph,
                )
            zc = phase2_z_cache[z]
            m_high, m_low = _generate_event_masks_fast(
                family=fam,
                param_val=param,
                up_move=zc["up"],
                dn_move=zc["dn"],
                rolling_std_up=zc["std_up"],
                rolling_std_dn=zc["std_dn"],
                asset_groups=shared["asset_groups"],
                duration_bars=duration_bars,
            )
            side_mask = _get_side_mask(mode, m_high, m_low)
            total_events = int(np.sum(side_mask))
            if total_events < int(cfg.get("phase2_min_total_events", 5000)):
                geom_cache_phase2[key] = {"rejected": True}
                continue

            active_days_frac = active_days_fraction_safe(side_mask, day_ids, n_days)
            if active_days_frac < float(cfg.get("phase2_min_active_days_fraction", 0.80)):
                geom_cache_phase2[key] = {"rejected": True}
                continue

            if _mode_is_up(mode):
                coh = _coherence_metrics_single_side(side_mask, zc["b_up"], zc["s_up"], zc["m_up"])
            else:
                coh = _coherence_metrics_single_side(side_mask, zc["b_dn"], zc["s_dn"], zc["m_dn"])

            distinct = _compute_regime_distinctness_single_side(
                side_mask=side_mask,
                mode=mode,
                forward_returns=forward_returns,
                mae_high=shared["mae_high"],
                mfe_high=shared["mfe_high"],
                mae_low=shared["mae_low"],
                mfe_low=shared["mfe_low"],
            )

            ev_day_mean, ev_day_std = daily_event_stats_safe(side_mask, day_ids, n_days)
            ev_day_per_asset = float(total_events) / float(max(1, n_days * phase2_n_assets))
            side_vol_exp = zc["v_exp"][side_mask & np.isfinite(zc["v_exp"])]
            post_event_vol_dispersion = float(np.std(side_vol_exp)) if side_vol_exp.shape[0] > 1 else 0.0

            fold_rates = [fold_base_rate_safe(side_mask, global_target, va) for _, va in folds]
            fold_event_counts = [float(np.sum(side_mask[va])) for _, va in folds]
            fold_rate_std = float(np.std(np.asarray(fold_rates, dtype=np.float32))) if fold_rates else 1.0
            fold_continuation_rate_std = fold_rate_std
            fold_event_count_std = float(np.std(np.asarray(fold_event_counts, dtype=np.float32))) if fold_event_counts else 1.0

            valid_fwd_p2 = np.isfinite(forward_returns)
            non_event = (~side_mask) & valid_fwd_p2
            if np.any(side_mask & valid_fwd_p2) and np.any(non_event):
                basic_edge = float(np.nanmean(global_target[side_mask & valid_fwd_p2]) - np.nanmean(global_target[non_event]))
            else:
                basic_edge = 0.0

            full_metrics = _compute_full_metrics_for_candidate(
                mode,
                side_mask,
                shared,
                feature_dict,
                cfg,
                float(coh["impulse_shape_dispersion"]),
                float(basic_edge),
            )
            legacy_metrics = _compute_legacy_conditional_learnability(mode, side_mask, shared, cfg)

            geom_cache_phase2[key] = {
                "rejected": False,
                "total_events": total_events,
                "active_days_fraction": float(active_days_frac),
                "events_per_day_mean": float(ev_day_mean),
                "events_per_day_std": float(ev_day_std),
                "events_per_day_per_asset": float(ev_day_per_asset),
                "bars_to_peak_dispersion": float(coh["bars_to_peak_dispersion"]),
                "speed_dispersion": float(coh["speed_dispersion"]),
                "monotonicity_dispersion": float(coh["monotonicity_dispersion"]),
                "impulse_shape_dispersion": float(coh["impulse_shape_dispersion"]),
                "post_event_vol_dispersion": float(post_event_vol_dispersion),
                "regime_distinctness_score": float(distinct),
                "fold_base_rate_stability": float(fold_rate_std),
                "fold_continuation_rate_std": float(fold_continuation_rate_std),
                "fold_event_count_std": float(fold_event_count_std),
                "basic_directionality_edge_event_vs_non_event": float(basic_edge),
                **full_metrics,
                **legacy_metrics,
            }

        stats = geom_cache_phase2[key]
        if stats.get("rejected", False):
            continue

        phase2_rows.append({
            "name": key,
            "family": fam,
            "z_hours": z_hr,
            "param": _safe_param_to_string(param),
            "duration_hours": d_hr,
            "conditioner_mode": "none",
            **{k: v for k, v in stats.items() if k not in {"rejected"}},
        })

    if not phase2_rows:
        return {"status": "failed", "reason": f"no_phase2_candidates_{mode}"}

    df2 = pd.DataFrame(phase2_rows)
    df2["D_r"] = (
        0.35 * _zscore_np(df2["impulse_shape_dispersion"].values)
        + 0.35 * _zscore_np(df2["post_event_vol_dispersion"].values)
        + 0.15 * _zscore_np(df2["fold_continuation_rate_std"].values)
        + 0.15 * _zscore_np(df2["fold_event_count_std"].values)
    )
    df2["N_r"] = df2["total_events"].astype(np.float32)
    df2["selected_delta_metric"] = df2["selected_delta_metric"].astype(str)
    primary_col = _mode_primary_predictability_col(mode)
    df2["bucket_primary_predictability_gain"] = df2[primary_col].astype(np.float32)
    df2["predictability_gain"] = [
        _mode_predictability_gain_from_metrics(mode, row.to_dict()) for _, row in df2.iterrows()
    ]
    df2["delta_r_raw"] = df2["bucket_primary_predictability_gain"].astype(np.float32)
    df2["delta_r_fallback"] = (0.5 * df2["incremental_information_delta_auc"].astype(np.float32)).astype(np.float32)
    df2["delta_r"] = np.maximum(
        np.nan_to_num(df2["delta_r_raw"].values, nan=-np.inf),
        np.nan_to_num(df2["delta_r_fallback"].values, nan=-np.inf),
    ).astype(np.float32)
    use_primary = np.nan_to_num(df2["delta_r_raw"].values, nan=-np.inf) >= np.nan_to_num(df2["delta_r_fallback"].values, nan=-np.inf)
    df2["selected_delta_metric"] = np.where(use_primary, primary_col, "0.5*incremental_information_delta_auc")
    selected_fold_mean = np.where(
        use_primary,
        df2["bucket_primary_delta_fold_mean"].astype(np.float32).values,
        (0.5 * df2["incremental_information_delta_auc_fold_mean"].astype(np.float32).values),
    )
    selected_fold_std = np.where(
        use_primary,
        df2["bucket_primary_delta_fold_std"].astype(np.float32).values,
        (0.5 * df2["incremental_information_delta_auc_fold_std"].astype(np.float32).values),
    )
    df2["delta_r_fold_mean"] = selected_fold_mean.astype(np.float32)
    df2["delta_r_fold_std"] = selected_fold_std.astype(np.float32)
    df2["positive_fold_fraction_r"] = df2["incremental_information_positive_fold_fraction"].astype(np.float32)
    df2["S_r"] = (
        0.5
        * np.maximum(
            0.0,
            1.0
            - (
                np.nan_to_num(df2["delta_r_fold_std"].values, nan=np.inf)
                / (np.abs(np.nan_to_num(df2["delta_r_fold_mean"].values, nan=0.0)) + 1e-9)
            ),
        )
        + 0.5 * np.nan_to_num(df2["incremental_information_positive_fold_fraction"].values, nan=0.0)
    ).astype(np.float32)
    df2["delta_r_shrunk"] = (
        df2["delta_r"].astype(np.float32).values
        * (df2["N_r"].astype(np.float32).values / (df2["N_r"].astype(np.float32).values + 500.0))
    ).astype(np.float32)
    df2["acceptance_pass"] = True
    min_predictability_gain = float(cfg.get("min_predictability_gain", 0.0))
    min_positive_fold_fraction = float(cfg.get("min_positive_fold_fraction", 0.0))
    if bool(cfg.get("enable_learnability_check", True)):
        primary_gain = df2["bucket_primary_predictability_gain"].values
        incr_gain = df2["incremental_information_delta_auc"].values
        incr_pos_frac = df2["incremental_information_positive_fold_fraction"].values
        primary_ok = np.isfinite(primary_gain) & (primary_gain > min_predictability_gain)
        incr_ok = np.isfinite(incr_gain) & (incr_gain > 0.0)
        if min_positive_fold_fraction > 0.0:
            incr_ok = incr_ok & np.isfinite(incr_pos_frac) & (incr_pos_frac >= min_positive_fold_fraction)
        df2.loc[~(primary_ok | incr_ok), "acceptance_pass"] = False
    df2 = df2[df2["acceptance_pass"] == True].copy()
    if df2.empty:
        return {"status": "failed", "reason": f"learnability_check_failed_{mode}", "layer0_candidate_table_": df2}

    df2["score_r"] = (
        df2["delta_r_shrunk"].astype(np.float32).values
        * np.sqrt(np.maximum(df2["N_r"].astype(np.float32).values, 0.0))
        * np.maximum(df2["S_r"].astype(np.float32).values, 0.0)
        / (1.0 + df2["D_r"].astype(np.float32).values)
    ).astype(np.float32)

    lambda_feature = 5.0
    feature_learnability_gain = np.nan_to_num(df2["predictability_gain"].values, nan=0.0).astype(np.float32)
    df2["feature_learnability_gain"] = feature_learnability_gain

    df2["score_ml"] = (
        df2["score_r"].astype(np.float32).values
        * (1.0 + lambda_feature * feature_learnability_gain)
    ).astype(np.float32)

    df2["shortlist_score"] = df2["score_ml"].astype(np.float32)
    df2["decision"] = "ranked"
    df2["regime_id"] = df2["name"].astype(str)
    df2["regime_definition"] = df2["name"].astype(str)
    df2["rationale"] = df2.apply(_build_regime_rationale, axis=1)

    df2 = df2.sort_values(
        ["score_ml", "feature_learnability_gain", "delta_r", "total_events"],
        ascending=[False, False, False, False]
    )
    shortlist_max = int(cfg.get("shortlist_max_candidates", 4))
    df_short = df2.head(shortlist_max).copy()
    if df_short.empty:
        return {"status": "failed", "reason": f"no_shortlist_candidates_{mode}", "layer0_candidate_table_": df2}

    candidate_masks: Dict[str, Dict[str, np.ndarray]] = {}
    for _, row in df_short.iterrows():
        name = row["name"]
        reg = candidate_registry[name]
        z = int(int(reg["z_hours"]) * bph)
        duration_bars = int(int(reg["duration_hours"]) * bph)
        if z not in phase2_z_cache:
            tprint(f"Precomputing Phase 2 rolling tensors for z={z} bars...")
            phase2_z_cache[z] = _compute_z_cache(
                high=shared["high"],
                low=shared["low"],
                close=shared["close"],
                ret_1=shared["ret_1"],
                vol_g=shared["vol_g"],
                asset_groups=shared["asset_groups"],
                z=z,
                bph=bph,
            )
        zc = phase2_z_cache[z]
        m_high, m_low = _generate_event_masks_fast(
            family=reg["family"],
            param_val=reg["param"],
            up_move=zc["up"],
            dn_move=zc["dn"],
            rolling_std_up=zc["std_up"],
            rolling_std_dn=zc["std_dn"],
            asset_groups=shared["asset_groups"],
            duration_bars=duration_bars,
        )
        candidate_masks[name] = {"m_high": m_high, "m_low": m_low}

    cond_rows: List[pd.Series] = []
    if bool(cfg.get("enable_secondary_conditioners", True)):
        spread_to_atr = np.nan_to_num(
            np.asarray(feature_dict.get("spread_to_atr", np.zeros_like(shared["close"], dtype=np.float32)), dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        alternation_array = np.asarray(shared.get("alternation", np.zeros_like(shared["close"], dtype=np.float32)), dtype=np.float32)
        for _, row in df_short.iterrows():
            cand_name = str(row["name"])
            reg = candidate_registry[cand_name]
            z = int(int(reg["z_hours"]) * bph)
            zc = phase2_z_cache[z]
            base_masks = candidate_masks[cand_name]
            for conditioner in cfg.get("conditioner_modes", ["none"]):
                if conditioner == "none":
                    continue
                new_h, new_l = _apply_secondary_conditioner(
                    base_masks["m_high"],
                    base_masks["m_low"],
                    conditioner,
                    zc["m_up"],
                    zc["m_dn"],
                    zc["v_exp"],
                    spread_to_atr,
                    alternation_array,
                )
                new_side_mask = _get_side_mask(mode, new_h, new_l)
                tot_events = int(np.sum(new_side_mask))
                if tot_events < int(cfg.get("phase2_min_total_events", 5000)):
                    continue
                coh = (
                    _coherence_metrics_single_side(new_side_mask, zc["b_up"], zc["s_up"], zc["m_up"])
                    if _mode_is_up(mode)
                    else _coherence_metrics_single_side(new_side_mask, zc["b_dn"], zc["s_dn"], zc["m_dn"])
                )
                new_fold_rates = [fold_base_rate_safe(new_side_mask, global_target, va) for _, va in folds]
                new_fold_cont_std = float(np.std(np.asarray(new_fold_rates, dtype=np.float32))) if new_fold_rates else 1.0
                base_disp = float(row.get("impulse_shape_dispersion", 1e9))
                base_f_cont = float(row.get("fold_continuation_rate_std", 1e9))
                improved = (
                    coh["impulse_shape_dispersion"] < base_disp * 0.95
                    or new_fold_cont_std < base_f_cont * 0.95
                )
                if not improved:
                    continue
                valid_new = np.isfinite(forward_returns)
                non_event_new = (~new_side_mask) & valid_new
                if np.any(new_side_mask & valid_new) and np.any(non_event_new):
                    basic_edge_new = float(
                        np.nanmean(global_target[new_side_mask & valid_new]) - np.nanmean(global_target[non_event_new])
                    )
                else:
                    basic_edge_new = 0.0
                new_metrics = _compute_full_metrics_for_candidate(
                    mode,
                    new_side_mask,
                    shared,
                    feature_dict,
                    cfg,
                    float(coh["impulse_shape_dispersion"]),
                    float(basic_edge_new),
                )
                gain_max = _mode_predictability_gain_from_metrics(mode, new_metrics)
                new_row = row.copy()
                new_name = cand_name + f"_{conditioner}"
                new_row["name"] = new_name
                new_row["conditioner_mode"] = conditioner
                new_row["total_events"] = tot_events
                new_row["impulse_shape_dispersion"] = float(coh["impulse_shape_dispersion"])
                new_row["fold_continuation_rate_std"] = float(new_fold_cont_std)
                new_row["predictability_gain"] = float(gain_max)
                new_row["bucket_primary_predictability_gain"] = float(new_metrics.get(primary_col, np.nan))
                new_row["continuation_predictability_gain"] = new_metrics.get("continuation_predictability_gain", np.nan)
                new_row["reversal_predictability_gain"] = new_metrics.get("reversal_predictability_gain", np.nan)
                new_row["MAE_predictability_gain"] = new_metrics.get("MAE_predictability_gain", np.nan)
                new_row["MFE_predictability_gain"] = new_metrics.get("MFE_predictability_gain", np.nan)
                new_row["incremental_information_delta_auc"] = new_metrics.get("incremental_information_delta_auc", np.nan)
                new_row["incremental_information_positive_fold_fraction"] = new_metrics.get("incremental_information_positive_fold_fraction", np.nan)
                new_row["incremental_information_positive_fold_count"] = new_metrics.get("incremental_information_positive_fold_count", np.nan)
                new_row["incremental_information_fold_count"] = new_metrics.get("incremental_information_fold_count", np.nan)
                new_row["incremental_information_delta_auc_fold_mean"] = new_metrics.get("incremental_information_delta_auc_fold_mean", np.nan)
                new_row["incremental_information_delta_auc_fold_std"] = new_metrics.get("incremental_information_delta_auc_fold_std", np.nan)
                new_row["bucket_primary_delta_fold_mean"] = new_metrics.get("bucket_primary_delta_fold_mean", np.nan)
                new_row["bucket_primary_delta_fold_std"] = new_metrics.get("bucket_primary_delta_fold_std", np.nan)
                new_row["dispersion_to_edge_ratio"] = new_metrics.get("dispersion_to_edge_ratio", np.nan)
                new_row["edge_to_dispersion_ratio"] = new_metrics.get("edge_to_dispersion_ratio", np.nan)
                new_row["primary_predictability_gain_is_nan"] = new_metrics.get("primary_predictability_gain_is_nan", 1.0)
                new_row["post_event_vol_dispersion"] = float(np.std(zc["v_exp"][new_side_mask & np.isfinite(zc["v_exp"])])) if np.any(new_side_mask & np.isfinite(zc["v_exp"])) else 0.0
                new_row["fold_event_count_std"] = float(np.std(np.asarray([float(np.sum(new_side_mask[va])) for _, va in folds], dtype=np.float32)))
                new_row["delta_r_raw"] = float(new_row["bucket_primary_predictability_gain"])
                new_row["delta_r_fallback"] = float(0.5 * new_row["incremental_information_delta_auc"]) if np.isfinite(new_row["incremental_information_delta_auc"]) else float("nan")
                raw_val = _metric_or_nan(new_row["delta_r_raw"])
                fb_val = _metric_or_nan(new_row["delta_r_fallback"])
                use_primary_new = raw_val >= fb_val
                new_row["delta_r"] = float(max(raw_val, fb_val))
                new_row["selected_delta_metric"] = primary_col if use_primary_new else "0.5*incremental_information_delta_auc"
                new_row["delta_r_fold_mean"] = float(
                    new_row["bucket_primary_delta_fold_mean"] if use_primary_new else 0.5 * _metric_or_nan(new_row["incremental_information_delta_auc_fold_mean"])
                )
                new_row["delta_r_fold_std"] = float(
                    new_row["bucket_primary_delta_fold_std"] if use_primary_new else 0.5 * _metric_or_nan(new_row["incremental_information_delta_auc_fold_std"])
                )
                new_row["positive_fold_fraction_r"] = float(_metric_or_nan(new_row["incremental_information_positive_fold_fraction"]))
                new_row["S_r"] = float(
                    0.5
                    * max(
                        0.0,
                        1.0 - _metric_or_nan(new_row["delta_r_fold_std"]) / (abs(_metric_or_nan(new_row["delta_r_fold_mean"])) + 1e-9),
                    )
                    + 0.5 * max(0.0, _metric_or_nan(new_row["incremental_information_positive_fold_fraction"]))
                )
                new_row["D_r"] = float("nan")
                new_row["N_r"] = float(tot_events)
                new_row["delta_r_shrunk"] = float(new_row["delta_r"] * (new_row["N_r"] / (new_row["N_r"] + 500.0)))
                new_row["score_r"] = float("nan")
                new_row["feature_learnability_gain"] = float("nan")
                new_row["score_ml"] = float("nan")
                cond_rows.append(new_row)
                candidate_masks[new_name] = {"m_high": new_h, "m_low": new_l}

    if cond_rows:
        df_short = pd.concat([df_short, pd.DataFrame(cond_rows)], ignore_index=True)
        df_short["D_r"] = (
            0.35 * _zscore_np(df_short["impulse_shape_dispersion"].values)
            + 0.35 * _zscore_np(df_short["post_event_vol_dispersion"].values)
            + 0.15 * _zscore_np(df_short["fold_continuation_rate_std"].values)
            + 0.15 * _zscore_np(df_short["fold_event_count_std"].values)
        )
        df_short["score_r"] = (
            df_short["delta_r_shrunk"].astype(np.float32).values
            * np.sqrt(np.maximum(df_short["N_r"].astype(np.float32).values, 0.0))
            * np.maximum(df_short["S_r"].astype(np.float32).values, 0.0)
            / (1.0 + df_short["D_r"].astype(np.float32).values)
        ).astype(np.float32)

        df_short["feature_learnability_gain"] = np.nan_to_num(df_short["predictability_gain"].values, nan=0.0).astype(np.float32)
        df_short["score_ml"] = (
            df_short["score_r"].astype(np.float32).values
            * (1.0 + lambda_feature * df_short["feature_learnability_gain"].values)
        ).astype(np.float32)

        df_short["shortlist_score"] = df_short["score_ml"].astype(np.float32)

        base_rows = df_short[df_short["conditioner_mode"] == "none"].set_index("name")
        keep_idx: List[int] = []
        for idx, row in df_short.iterrows():
            cond_mode = str(row.get("conditioner_mode", "none"))
            if cond_mode == "none":
                keep_idx.append(idx)
                continue
            base_name = str(row["name"]).rsplit("_", 1)[0]
            if base_name not in base_rows.index:
                continue
            base_row = base_rows.loc[base_name]
            if (
                _metric_or_nan(row.get("score_ml")) > _metric_or_nan(base_row.get("score_ml"))
                and _metric_or_nan(row.get("feature_learnability_gain")) >= _metric_or_nan(base_row.get("feature_learnability_gain"))
            ):
                keep_idx.append(idx)
        df_short = df_short.loc[keep_idx].sort_values(
            ["score_ml", "feature_learnability_gain", "delta_r", "total_events"],
            ascending=[False, False, False, False]
        ).copy()

    final_diag_k = int(cfg.get("final_top_k_for_diagnostics", 3))
    df_diag_input = df_short.sort_values("shortlist_score", ascending=False).head(final_diag_k).copy()
    df_diag = _final_topk_diagnostics(mode, df_diag_input, candidate_masks, shared, feature_dict, cfg)

    best = df_short.iloc[0].to_dict()
    best_masks = candidate_masks[best["name"]]
    best_active_mask = _get_side_mask(mode, best_masks["m_high"], best_masks["m_low"])

    return {
        "status": "ok",
        "mode": mode,
        "phase1_candidate_table_": df1,
        "layer0_candidate_table_": df2,
        "layer0_shortlist_": df_short,
        "layer0_best_config_": best,
        "layer0_best_active_mask_": best_active_mask,
        "layer0_candidate_masks_": candidate_masks,
        "final_topk_diagnostics_table_": df_diag,
    }


def _mode_worker(
    conn: Any,
    mode: str,
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    cfg: Dict[str, Any],
) -> None:
    try:
        shared = _build_shared_cache(data, feature_dict, forward_returns, cfg)
        res = _run_mode_search(mode, shared, feature_dict, cfg)
        conn.send(("ok", res))
    except Exception:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


def _run_mode_search_isolated(
    mode: str,
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_mode_worker,
        args=(child_conn, mode, data, feature_dict, forward_returns, cfg),
    )
    proc.start()
    child_conn.close()
    payload: Optional[Tuple[str, Any]] = None
    timeout_seconds = float(cfg.get("mask_opt_mode_timeout_seconds", 0.0))
    if timeout_seconds > 0:
        if parent_conn.poll(timeout_seconds):
            payload = parent_conn.recv()
    else:
        payload = parent_conn.recv()
    if payload is None:
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return {"status": "failed", "reason": f"mode_timeout_{mode}"}
        return {"status": "failed", "reason": f"mode_crashed_{mode}_exit_{proc.exitcode}"}
    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join()
    status, body = payload
    if status == "ok":
        return body
    return {"status": "failed", "reason": f"mode_exception_{mode}", "traceback": body}


# =============================================================================
# PUBLIC ORCHESTRATOR
# =============================================================================

def optimize_layer0_masks_by_mode(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    cfg: Dict[str, Any],
    modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return optimize_layer_masks_by_mode(data, feature_dict, forward_returns, cfg, modes=modes, layer_name="layer0")


def optimize_layer_masks_by_mode(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    cfg: Dict[str, Any],
    modes: Optional[List[str]] = None,
    layer_name: str = "layer0",
) -> Dict[str, Any]:
    if modes is None:
        modes = ALL_MODES[:]

    runtime_cfg = _materialize_layer_runtime_cfg(cfg, layer_name)
    data, feature_dict, forward_returns = _cap_rows_for_optimization(
        data=data,
        feature_dict=feature_dict,
        forward_returns=np.asarray(forward_returns, dtype=np.float32),
        cfg=runtime_cfg,
        seed=42,
    )
    runtime_cfg = _rescale_mode_gates_for_sample_size(runtime_cfg, int(data.shape[0]))

    mode_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []
    isolate_modes = bool(runtime_cfg.get("mask_opt_isolate_modes", True)) and len(modes) > 1
    shared: Optional[Dict[str, Any]] = None
    if not isolate_modes:
        shared = _build_shared_cache(data, feature_dict, forward_returns, runtime_cfg)

    for mode in modes:
        if isolate_modes:
            tprint(f"Running mode {mode} in isolated subprocess...")
            res = _run_mode_search_isolated(mode, data, feature_dict, forward_returns, runtime_cfg)
        else:
            assert shared is not None
            res = _run_mode_search(mode, shared, feature_dict, runtime_cfg)
        mode_results[mode] = res

        if res.get("status") == "ok":
            best = res["layer0_best_config_"]
            primary_col = _mode_primary_predictability_col(mode)
            summary_rows.append({
                "mode": mode,
                "status": "ok",
                "best_name": str(best.get("name", "")),
                "candidate_count": len(res["layer0_candidate_table_"]),
                "shortlist_count": len(res["layer0_shortlist_"]),
                "best_shortlist_score": float(best.get("shortlist_score", 0.0)),
                "score_r": float(best.get("score_r", best.get("shortlist_score", 0.0))),
                "delta_r": _metric_or_nan(best.get("delta_r")),
                "N_r": float(best.get("N_r", best.get("total_events", 0.0))),
                "S_r": _metric_or_nan(best.get("S_r")),
                "D_r": _metric_or_nan(best.get("D_r")),
                "event_count": int(best.get("total_events", 0)),
                "active_days_fraction": float(best.get("active_days_fraction", 0.0)),
                "events_per_day_mean": _metric_or_nan(best.get("events_per_day_mean")),
                "events_per_day_std": _metric_or_nan(best.get("events_per_day_std")),
                "events_per_day_per_asset": _metric_or_nan(best.get("events_per_day_per_asset")),
                "primary_gain": _metric_or_nan(best.get(primary_col)),
                "primary_gain_is_nan": _metric_or_nan(best.get("primary_predictability_gain_is_nan")),
                "incremental_information_delta_auc": _metric_or_nan(best.get("incremental_information_delta_auc")),
                "incremental_information_positive_fold_fraction": _metric_or_nan(best.get("incremental_information_positive_fold_fraction")),
                "dispersion_to_edge_ratio": _metric_or_nan(best.get("dispersion_to_edge_ratio")),
                "selected_delta_metric": str(best.get("selected_delta_metric", "")),
                "decision": str(best.get("decision", "ranked")),
                "rationale": str(best.get("rationale", "")),
            })
        else:
            summary_rows.append({
                "mode": mode,
                "status": res.get("reason", "failed"),
                "best_name": "",
                "candidate_count": 0,
                "shortlist_count": 0,
                "best_shortlist_score": 0.0,
                "score_r": float("nan"),
                "delta_r": float("nan"),
                "N_r": 0.0,
                "S_r": float("nan"),
                "D_r": float("nan"),
                "event_count": 0,
                "active_days_fraction": 0.0,
                "events_per_day_mean": float("nan"),
                "events_per_day_std": float("nan"),
                "events_per_day_per_asset": float("nan"),
                "primary_gain": 0.0,
                "primary_gain_is_nan": float("nan"),
                "incremental_information_delta_auc": float("nan"),
                "incremental_information_positive_fold_fraction": float("nan"),
                "dispersion_to_edge_ratio": float("nan"),
                "selected_delta_metric": "",
                "decision": "failed",
                "rationale": "",
            })

    return {
        "status": "ok",
        "mode_results": mode_results,
        "mode_summary_table_": pd.DataFrame(summary_rows),
    }


# =============================================================================
# CLI
# =============================================================================

def run_mask_optimization_4modes(args: argparse.Namespace) -> None:
    from copy import deepcopy

    cfg = deepcopy(CFG)

    # defaults aligned with requested optimization spec
    cfg["z_hours_grid"] = [4, 6, 8, 10]
    cfg["duration_grid"] = [1, 2, 3]
    cfg["x_std_grid"] = [1.4, 1.5, 1.6]
    cfg["y_move_pct_grid"] = [4.0, 5.0, 6.0, 7.0]
    cfg["std_plus_abs_std_grid"] = [1.4, 1.5, 1.6]
    cfg["std_plus_abs_abs_grid"] = [4.0, 5.0, 6.0]
    cfg["phase1_min_total_events"] = 5000
    cfg["phase2_min_total_events"] = 5000
    cfg["phase1_min_active_days_fraction"] = 0.80
    cfg["phase2_min_active_days_fraction"] = 0.80
    cfg["mask_opt_target_event_density"] = 0.025
    cfg["mask_opt_min_events_floor"] = 150
    cfg["mask_opt_min_active_days_floor"] = 0.25
    cfg["min_positive_fold_fraction"] = 0.60
    cfg["shortlist_max_candidates"] = 4
    cfg["final_top_k_for_diagnostics"] = 3
    cfg["mask_opt_max_rows"] = 30_000
    cfg["phase1_classifier_max_samples_per_class"] = int(cfg.get("phase1_classifier_max_samples_per_class", 15_000))
    cfg["phase2_metric_max_samples_per_class"] = int(cfg.get("phase2_metric_max_samples_per_class", 25_000))

    if args.data_root:
        cfg["data_root"] = _resolve_path(args.data_root)
    else:
        cfg["data_root"] = _resolve_path(cfg.get("data_root", "data"))

    if args.perps:
        cfg["use_perps"] = True
        if not cfg["data_root"].endswith("_perp"):
            cfg["data_root"] += "_perp"
        cfg = enable_perp_feature_keys(cfg)

    if args.features:
        feature_path = _resolve_path(args.features)
    else:
        feature_path = _find_latest_feature_dir(cfg["data_root"])

    if not feature_path:
        tprint(f"ERROR: no features found in {cfg['data_root']}/features")
        return

    tprint(f"Loading data: data_root={cfg['data_root']} | features={feature_path}")

    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg.get("timeframe", "1h"))

    ohlcv_dir = os.path.join(cfg["data_root"], "ohlcv")
    all_symbols = []
    for path in glob.glob(os.path.join(ohlcv_dir, "symbol=*")):
        base = os.path.basename(path)
        if base.startswith("symbol="):
            raw = base.replace("symbol=", "")
            all_symbols.append(raw.replace("_", "/", 1))
    all_symbols.sort()
    
    # Apply deduplication (only use symbols passed on by universe)
    all_symbols = _dedup_universe_by_base(all_symbols)
    tprint(f"Symbols after deduplication: {len(all_symbols)}")

    # Then subsample to 50%
    rng = random.Random(42)
    k = max(1, len(all_symbols) // 2)
    symbols = rng.sample(all_symbols, k)
    tprint(f"Selected {len(symbols)} symbols (50%% subsample)")

    start_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(365.25 * args.lookback_years))

    dfs_by_symbol: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = store.load(s, start_ts=start_ts)
        if not df.empty:
            dfs_by_symbol[s] = df

    if not dfs_by_symbol:
        tprint("ERROR: no symbol data loaded")
        return

    panel = to_panel(dfs_by_symbol)
    if not panel or "close" not in panel or panel["close"].empty:
        tprint("ERROR: panel empty or missing close")
        return

    ts_str = os.path.basename(feature_path)
    try:
        ts = pd.Timestamp(ts_str.replace("_", " "))
    except Exception:
        ts = pd.Timestamp.now(tz="UTC")
    data_root_dir = os.path.dirname(os.path.dirname(feature_path))

    feat_dict_raw = load_features_selected(
        ts=ts,
        root_dir=data_root_dir,
        feature_keys=list(_required_feature_keys()),
        symbols=symbols,
        start_ts=start_ts,
    )
    if not feat_dict_raw:
        tprint("ERROR: empty feature dictionary")
        return

    common_idx = panel["close"].index
    common_syms = panel["close"].columns

    fwd_hours = int(cfg.get("mask_opt_forward_hours", 12))
    fwd_ret_wide = panel["close"].pct_change(fwd_hours, fill_method=None).shift(-fwd_hours)

    n_timestamps = len(common_idx)
    n_symbols = len(common_syms)
    data_stacked = pd.DataFrame({
        "timestamp": np.repeat(common_idx.to_numpy(), n_symbols),
        "symbol": np.tile(common_syms.to_numpy(dtype=object), n_timestamps),
        "close": _flatten_wide_frame(panel["close"], common_idx, common_syms),
        "high": _flatten_wide_frame(panel["high"], common_idx, common_syms),
        "low": _flatten_wide_frame(panel["low"], common_idx, common_syms),
    })

    feature_dict: Dict[str, np.ndarray] = {}
    for k, df in feat_dict_raw.items():
        if isinstance(df, pd.DataFrame):
            arr = _flatten_wide_frame(df, common_idx, common_syms)
            arr[np.isinf(arr)] = np.nan
            feature_dict[k] = arr.astype(np.float32)

    fwd_ret_stacked = _flatten_wide_frame(fwd_ret_wide, common_idx, common_syms)

    tprint(f"Total rows available before capping: {data_stacked.shape[0]}")
    data_stacked, feature_dict, fwd_ret_stacked = _cap_rows_for_optimization(
        data=data_stacked,
        feature_dict=feature_dict,
        forward_returns=fwd_ret_stacked,
        cfg=cfg,
        seed=42,
    )

    if args.mode == "all":
        modes = ALL_MODES[:]
    else:
        modes = [args.mode]

    tprint(f"Starting 4-mode Layer 0 optimization on {data_stacked.shape[0]} rows...")
    result = optimize_layer0_masks_by_mode(data_stacked, feature_dict, fwd_ret_stacked, cfg, modes=modes)

    if result.get("status") != "ok":
        tprint("Optimization failed.")
        return

    tprint("=" * 80)
    tprint("MODE SUMMARY")
    tprint("=" * 80)
    tprint(result["mode_summary_table_"].to_string(index=False))

    # optional save
    try:
        from extreme_price_movements.offline_optimisers.params_store import (
            save_best_params_csv,
            INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV,
            REPORTS_DIR,
        )
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = REPORTS_DIR / "inference_candidate_mask_mode_summary.csv"
        result["mode_summary_table_"].to_csv(summary_path, index=False)
        for mode in modes:
            mode_res = result["mode_results"].get(mode, {})
            candidate_table = mode_res.get("layer0_candidate_table_")
            shortlist_table = mode_res.get("layer0_shortlist_")
            if isinstance(candidate_table, pd.DataFrame):
                candidate_path = REPORTS_DIR / f"layer0_candidate_table_{mode}.csv"
                candidate_table.to_csv(candidate_path, index=False)
            if isinstance(shortlist_table, pd.DataFrame):
                shortlist_path = REPORTS_DIR / f"layer0_shortlist_{mode}.csv"
                shortlist_table.to_csv(shortlist_path, index=False)
            if mode_res.get("status") == "ok":
                best = mode_res["layer0_best_config_"]
                out = dict(best)
                out["mode"] = mode
                out_path = INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV.with_name(
                    f"{INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV.stem}_{mode}{INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV.suffix}"
                )
                save_best_params_csv(
                    out_path,
                    out,
                    metadata={"source": "mask_optimiser_4mode"},
                )
    except Exception as e:
        tprint(f"Warning: failed to save best params: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Layer 0 masks in 4 explicit modes")
    parser.add_argument("--data-root", help="Override data root")
    parser.add_argument("--features", help="Path to features directory")
    parser.add_argument("--perps", action="store_true", help="Use perpetual mode data")
    parser.add_argument("--max-symbols", type=int, help="Cap symbols for speed")
    parser.add_argument("--lookback-years", type=float, default=2.0, help="Years of data to load")
    parser.add_argument(
        "--mode",
        choices=["all"] + ALL_MODES,
        default="all",
        help="Run all modes or one mode only",
    )
    args = parser.parse_args()
    run_mask_optimization_4modes(args)
